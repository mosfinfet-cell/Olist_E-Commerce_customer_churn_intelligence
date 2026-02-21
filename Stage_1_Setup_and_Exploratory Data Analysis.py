# ── CELL 1: Imports ──────────────────────────────────────────────────────────
# Standard data science stack — these are your daily tools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Make charts look clean
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
sns.set_palette('muted')

print('✅ Imports successful')

# ── CELL 2: Load All Tables ──────────────────────────────────────────────────

PATH = '/Users/SanthoshVislavath/Desktop/Product_analytics/Olist_data_analysis/'  # Adjust if your folder is elsewhere

orders    = pd.read_csv(PATH + 'olist_orders_dataset.csv')
items     = pd.read_csv(PATH + 'olist_order_items_dataset.csv')
customers = pd.read_csv(PATH + 'olist_customers_dataset.csv')
payments  = pd.read_csv(PATH + 'olist_order_payments_dataset.csv')
reviews   = pd.read_csv(PATH + 'olist_order_reviews_dataset.csv')
products  = pd.read_csv(PATH + 'olist_products_dataset.csv')
sellers   = pd.read_csv(PATH + 'olist_sellers_dataset.csv')
cat_names = pd.read_csv(PATH + 'product_category_name_translation.csv')
geolocation = pd.read_csv(PATH + 'olist_geolocation_dataset.csv')

# Print a summary of what we loaded
tables = {
    'orders': orders, 'items': items, 'customers': customers,
    'payments': payments, 'reviews': reviews, 'products': products,
    'sellers': sellers, 'cat_names': cat_names, 'geolocation':geolocation
}

print(f"{'Table':<15} {'Rows':>9} {'Columns':>10}")
print('-' * 36)
for name, df in tables.items():
    print(f"{name:<15} {len(df):>9,} {len(df.columns):>10}")

# ── CELL 3: Demonstrate the customer_id vs customer_unique_id gotcha ─────────

print('customer_id (per-order) unique values:', orders['customer_id'].nunique())

merged_check = orders.merge(customers[['customer_id','customer_unique_id']], on='customer_id')
print('customer_unique_id (true customer) unique values:', merged_check['customer_unique_id'].nunique())

print()
print('📌 These should differ — the gap shows repeat customers using different order-level IDs')
print(f'   Repeat customer rate: {1 - merged_check["customer_unique_id"].nunique()/orders["customer_id"].nunique():.1%}')
print("\n")

# ── CELL 4: Convert date columns ─────────────────────────────────────────────


date_cols = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]

for col in date_cols:
    orders[col] = pd.to_datetime(orders[col])

print('Date range of orders:')
print(f"  Earliest: {orders['order_purchase_timestamp'].min().date()}")
print(f"  Latest:   {orders['order_purchase_timestamp'].max().date()}")
print(f"  Span:     {(orders['order_purchase_timestamp'].max() - orders['order_purchase_timestamp'].min()).days} days")


# ── CELL 5: Data Quality Check ───────────────────────────────────────────────
# Before any analysis: check for nulls, duplicates, and anomalies

print('=== NULL CHECK: orders table ===')
null_pct = (orders.isnull().sum() / len(orders) * 100).round(1)
print(null_pct[null_pct > 0].to_string())

print()
print('=== ORDER STATUS BREAKDOWN ===')
print(orders['order_status'].value_counts())

print()
print('=== DECISION: Filter to delivered orders only ===')
print('Rationale: canceled/unavailable orders have no revenue and distort RFM scores')

# ── CELL 6: Filter to delivered orders ──────────────────────────────────────

orders_clean = orders[orders['order_status'] == 'delivered'].copy()
print(f'Orders before filter: {len(orders):,}')
print(f'Orders after filter:  {len(orders_clean):,}')
print(f'Dropped:              {len(orders) - len(orders_clean):,} ({(1 - len(orders_clean)/len(orders)):.1%})')

# ── CELL 7: Build the Master Dataframe ───────────────────────────────────────
# Join the tables we need for RFM analysis
# 💡 Always join step-by-step and check row counts — silent join errors are common

# Step 1: Orders + Customer unique ID
df = orders_clean.merge(
    customers[['customer_id', 'customer_unique_id', 'customer_city', 'customer_state']],
    on='customer_id', how='left'
)
print(f'After customer join: {len(df):,} rows')

# Step 2: Add payment value (aggregate payments per order)
payment_agg = payments.groupby('order_id')['payment_value'].sum().reset_index()
df = df.merge(payment_agg, on='order_id', how='left')
print(f'After payment join:  {len(df):,} rows')

# Step 3: Add average review score per order
review_agg = reviews.groupby('order_id')['review_score'].mean().reset_index()
df = df.merge(review_agg, on='order_id', how='left')
print(f'After review join:   {len(df):,} rows')

print()
print('Master dataframe columns:')
print(df.columns.tolist())


# ── CELL 8: Monthly Order Volume ─────────────────────────────────────────────
# Business question: Is this a growing business with seasonal patterns?

df['order_month'] = df['order_purchase_timestamp'].dt.to_period('M')
monthly = df.groupby('order_month').agg(
    orders=('order_id', 'count'),
    revenue=('payment_value', 'sum')
).reset_index()
monthly['order_month_dt'] = monthly['order_month'].dt.to_timestamp()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

ax1.bar(monthly['order_month_dt'], monthly['orders'], color='steelblue', alpha=0.8, width=20)
ax1.set_title('Monthly Order Volume', fontweight='bold', pad=12)
ax1.set_ylabel('Orders')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

ax2.bar(monthly['order_month_dt'], monthly['revenue']/1000, color='coral', alpha=0.8, width=20)
ax2.set_title('Monthly Revenue (R$ thousands)', fontweight='bold', pad=12)
ax2.set_ylabel('Revenue (R$ 000s)')

plt.tight_layout()
plt.savefig('/Users/SanthoshVislavath/Desktop/Product_analytics/Olist_data_analysis/output/01_monthly_volume.png', dpi=150, bbox_inches='tight')
plt.show()

print('📌 INSIGHT: Look for the growth trend and any sudden drops — those drops are where churn analysis starts')

# ── CELL 9: Order Value Distribution ─────────────────────────────────────────
# Business question: What's the typical order value? Are there outliers?


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# Cap at 500 for visibility (there are outliers above)
vals = df['payment_value'].clip(0, 500)
ax1.hist(vals, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax1.set_title('Order Value Distribution (capped at R$500)', fontweight='bold')
ax1.set_xlabel('Order Value (R$)')
ax1.set_ylabel('Count')

# Log scale version to see the full picture
ax2.hist(df['payment_value'].clip(1, None), bins=80, color='coral', edgecolor='white', alpha=0.8)
ax2.set_xscale('log')
ax2.set_title('Order Value Distribution (log scale)', fontweight='bold')
ax2.set_xlabel('Order Value (R$, log scale)')

plt.tight_layout()
plt.savefig('/Users/SanthoshVislavath/Desktop/Product_analytics/Olist_data_analysis/output/02_order_value_dist.png', dpi=150, bbox_inches='tight')
plt.show()

# Descriptive stats
print('Order Value Statistics:')
print(df['payment_value'].describe().apply(lambda x: f'R$ {x:,.2f}'))

# ── CELL 10: Repeat Purchase Rate ────────────────────────────────────────────
# Business question: What % of customers ever come back for a 2nd order?
# This is THE most important metric for an e-commerce marketplace

customer_order_counts = df.groupby('customer_unique_id')['order_id'].count().reset_index()
customer_order_counts.columns = ['customer_unique_id', 'order_count']

repeat_rate = (customer_order_counts['order_count'] > 1).mean()
print(f'Customers with exactly 1 order:  {(customer_order_counts["order_count"]==1).sum():,} ({(customer_order_counts["order_count"]==1).mean():.1%})')
print(f'Customers with 2+ orders:        {(customer_order_counts["order_count"]>1).sum():,} ({repeat_rate:.1%})')
print(f'Customers with 3+ orders:        {(customer_order_counts["order_count"]>2).sum():,} ({(customer_order_counts["order_count"]>2).mean():.1%})')

fig, ax = plt.subplots(figsize=(10, 4))
dist = customer_order_counts['order_count'].clip(1,8).value_counts().sort_index()
ax.bar(dist.index, dist.values, color='steelblue', edgecolor='white', alpha=0.85)
ax.set_title('Distribution of Orders per Customer', fontweight='bold', pad=12)
ax.set_xlabel('Number of Orders')
ax.set_ylabel('Customer Count')
ax.set_xticks(range(1,9))
ax.set_xticklabels([str(i) if i<8 else '8+' for i in range(1,9)])
plt.tight_layout()
plt.savefig('/Users/SanthoshVislavath/Desktop/Product_analytics/Olist_data_analysis/output/03_repeat_purchase.png', dpi=150, bbox_inches='tight')
plt.show()

print()
print('💡 PM INSIGHT: If repeat rate is <10%, the business has a retention crisis.')
print('   This number is your NORTH STAR metric for the entire churn analysis.')


# ── CELL 11: Review Score Distribution ───────────────────────────────────────
# Business question: Is poor product quality driving churn?
# (Hypothesis H5 from our product spec)

fig, ax = plt.subplots(figsize=(8, 4))
score_counts = reviews['review_score'].value_counts().sort_index()
colors = ['#c84b31', '#e07b39', '#c9922a', '#2d5a8e', '#2a7d4f']
ax.bar(score_counts.index, score_counts.values, color=colors, edgecolor='white', width=0.6)
ax.set_title('Review Score Distribution', fontweight='bold', pad=12)
ax.set_xlabel('Review Score (1=Poor, 5=Excellent)')
ax.set_ylabel('Number of Reviews')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

for i, (score, count) in enumerate(score_counts.items()):
    ax.text(score, count + 200, f'{count/len(reviews):.1%}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/SanthoshVislavath/Desktop/Product_analytics/Olist_data_analysis/output/04_review_scores.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'Average review score: {reviews["review_score"].mean():.2f}')
print(f'% with score 1-2 (detractors): {(reviews["review_score"]<=2).mean():.1%}')

# ── CELL 12: Delivery Time Analysis ──────────────────────────────────────────
# Business question: Are late deliveries correlated with bad reviews / churn?
# This is causal hypothesis testing — does X predict Y?

df_delivery = df.dropna(subset=['order_delivered_customer_date', 'order_estimated_delivery_date'])
df_delivery = df_delivery.copy()
df_delivery['delivery_days'] = (
    df_delivery['order_delivered_customer_date'] - df_delivery['order_purchase_timestamp']
).dt.days
df_delivery['days_vs_estimate'] = (
    df_delivery['order_delivered_customer_date'] - df_delivery['order_estimated_delivery_date']
).dt.days  # Positive = late, Negative = early

df_delivery = df_delivery.merge(review_agg, on='order_id', how='left')

# Group by lateness bucket
df_delivery['lateness'] = pd.cut(
    df_delivery['days_vs_estimate'],
    bins=[-999, -7, 0, 3, 7, 999],
    labels=['Very Early (7d+)', 'On Time / Early', 'Slightly Late (1-3d)', 'Late (4-7d)', 'Very Late (7d+)']
)

#print(df_delivery.columns)

delivery_review = df_delivery.groupby('lateness', observed=True)['review_score_x'].agg(['mean','count'])
print('Average review score by delivery performance:')
print(delivery_review.round(2))

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(delivery_review.index, delivery_review['mean'], 
       color=['#2a7d4f','#2a7d4f','#c9922a','#e07b39','#c84b31'],
       edgecolor='white', alpha=0.85)
ax.axhline(y=reviews['review_score'].mean(), color='gray', linestyle='--', alpha=0.6, label='Overall average')
ax.set_title('Avg Review Score by Delivery Performance', fontweight='bold', pad=12)
ax.set_ylabel('Avg Review Score')
ax.set_ylim(1, 5.2)
ax.legend()
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('/Users/SanthoshVislavath/Desktop/Product_analytics/Olist_data_analysis/output/05_delivery_vs_reviews.png', dpi=150, bbox_inches='tight')
plt.show()

print()
print('💡 PM INSIGHT: If late delivery → bad reviews, and bad reviews → no repeat purchase,')
print('   then delivery SLA is a critical lever for retention. We will test this in Stage 4.')


# ── CELL 13: Save clean master dataframe ─────────────────────────────────────
import os
os.makedirs('output', exist_ok=True)

# Save for use in Stage 2
df.to_csv('/Users/SanthoshVislavath/Desktop/Product_analytics/Olist_data_analysis/output/master_df.csv', index=False)
customer_order_counts.to_csv('/Users/SanthoshVislavath/Desktop/Product_analytics/Olist_data_analysis/output/customer_order_counts.csv', index=False)

print('Files saved to output/:')
print('  master_df.csv              → use in Stage 2 (RFM)')
print('  customer_order_counts.csv  → reference for repeat rate')
