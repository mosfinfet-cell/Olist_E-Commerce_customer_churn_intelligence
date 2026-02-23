# ── CELL 1: Imports & Setup ──────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

import os
os.makedirs('output', exist_ok=True)

print('Ready')

# ── CELL 2: Load the master dataframe from Stage 1 ───────────────────────────
# If you haven't run Stage 1 yet, go do that first!
# master_df.csv was saved in output/ at the end of stage1_eda.ipynb

df = pd.read_csv('/Users/SanthoshVislavath/Desktop/Product_analytics/Olist_data_analysis/output/master_df.csv', parse_dates=['order_purchase_timestamp'])

print(f'Rows loaded:      {len(df):,}')
print(f'Unique customers: {df["customer_unique_id"].nunique():,}')
print(f'Date range:       {df["order_purchase_timestamp"].min().date()} → {df["order_purchase_timestamp"].max().date()}')
print()
df.head(3)
# ── CELL 3: Set the snapshot date ────────────────────────────────────────────
# Convention: 1 day after the last recorded order

SNAPSHOT_DATE = df['order_purchase_timestamp'].max() + pd.Timedelta(days =1)

print(f'Snapshot date: {SNAPSHOT_DATE.date()}')
print()
print(' All recency values are measured FROM this date backwards.')
print('   A customer who last ordered 30 days before this = Recency 30.')
# ── CELL 4: Calculate Raw R, F, M per customer ───────────────────────────────
# This is the core aggregation — one row per unique customer
#
# Recency  = days since their MOST RECENT order (lower = better)
# Frequency = total number of orders placed
# Monetary  = total payment value across all orders

rfm = df.groupby('customer_unique_id').agg(
    recency   = ('order_purchase_timestamp', lambda x: (SNAPSHOT_DATE - x.max()).days),
    frequency = ('order_id', 'nunique'),
    monetary  = ('payment_value', 'sum')
).reset_index()

print(f'RFM table shape: {rfm.shape} (one row per customer)')
print()
print('Raw RFM Statistics')
print(rfm[['recency', 'frequency', 'monetary']].describe().round(1).to_string())
# ── CELL 5: Visualise raw distributions BEFORE scoring ───────────────────────
# ALWAYS look at raw distributions before building scores.
# Heavily skewed data means the middle buckets will behave unexpectedly.

fig, axes = plt.subplots(1,3, figsize=(15,4))

#Recency
axes[0].hist(rfm['recency'], bins=50, color='#c84b31', edgecolor='white', alpha=0.85)
axes[0].set_title('Recency Distribution\n(Days since last order)', fontweight ='bold')
axes[0].set_xlabel('Days')
axes[0].set_ylabel('Customers')
axes[0].axvline(rfm['recency'].median(),color='black', linestyle='--', alpha=0.6,
                label=f'Median: {rfm["recency"].median():.0f}d')
axes[0].legend(fontsize =9)

#Frequency
axes[1].hist(rfm['frequency'], bins= range(1, rfm['frequency'].max() + 2),
            color='#2d5a8e', edgecolor='white', alpha=0.85, align='left')
axes[1].set_title('Frequency Distribution \n (Number of orders)', fontweight ='bold')
axes[1].set_xlabel('Orders')
axes[1].set_xticks(range(1, min(rfm['frequency'].max()+1, 12)))

# Monetary
axes[2].hist(rfm['monetary'].clip(0, 1000), bins=50, color='#2a7d4f', edgecolor='white', alpha=0.85)
axes[2].set_title('Monetary Distribution\n(total spend, capped R$1000)', fontweight='bold')
axes[2].set_xlabel('R$')
axes[2].axvline(rfm['monetary'].median(), color='black', linestyle='--', alpha=0.6,
                label=f'Median: R${rfm["monetary"].median():.0f}')
axes[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig('/Users/SanthoshVislavath/Desktop/Product_analytics/Olist_data_analysis/output/06_raw_rfm_distributions.png', dpi=150, bbox_inches='tight')
plt.show()


# ── CELL 6: Assign R, F, M scores (1–5) ─────────────────────────────────────
# qcut = quantile cut — divides into equal-sized buckets by rank
# Note the label reversal for Recency!

# RECENCY: low days = recent = good → highest score (5)
rfm['R'] = pd.qcut(rfm['recency'], q =5, labels = [5,4,3,2,1]).astype(int)

# FREQUENCY: high orders = good → highest score (5)
# rank(method='first') handles ties so qcut doesn't fail
rfm['F'] = pd.qcut(rfm['frequency'].rank(method ='first'), q = 5, labels =[1,2,3,4,5]).astype(int)

# MONETARY: high spend = good → highest score (5)
rfm['M'] = pd.qcut(rfm['monetary'], q = 5, labels = [1,2,3,4,5]).astype(int)

# Combined RFM score (3–15 range)
rfm['RFM_score'] = rfm['R'] + rfm['F'] + rfm['M']

print('Score distributions — each should have ~equal counts:')
for col in ['R', 'F', 'M']:
    print(f'\n  {col} score value counts:')
    print(f'  {rfm[col].value_counts().sort_index().to_dict()}')

print(f'\nRFM combined score range: {rfm["RFM_score"].min()}–{rfm["RFM_score"].max()}')

# ── CELL 7: Visualise score distributions ────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
score_colors = ['#c84b31','#2d5a8e','#2a7d4f','#6b6257']

for ax, col, title, color in zip(axes, ['R','F','M','RFM_score'],
                                  ['Recency Score','Frequency Score','Monetary Score','Combined Score'],
                                  score_colors):
    counts = rfm[col].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=color, edgecolor='white', alpha=0.85)
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlabel('Score')
    ax.set_ylabel('Customers')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.suptitle('RFM Score Distributions', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/SanthoshVislavath/Desktop/Product_analytics/Olist_data_analysis/output/07_rfm_scores.png', dpi=150, bbox_inches='tight')
plt.show()

# ── CELL 8: Assign segment labels ────────────────────────────────────────────

def assign_segment(row):
    r, f = row['R'], row['F']
    if r >= 4 and f >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3:
        return 'Loyal'
    elif r >= 4 and f == 1:
        return 'New Customers'
    elif r >= 3 and f <= 2:
        return 'Potential Loyalists'
    elif r <= 2 and f >= 3:
        return 'At Risk'
    else:
        return 'Churned'

rfm['segment'] = rfm.apply(assign_segment, axis=1)

seg_counts = rfm['segment'].value_counts()
print('Customer count by segment:')
print(seg_counts.to_string())
print()
total = len(rfm)
print('As percentage of total customers:')
print((seg_counts / total * 100).round(1).astype(str).apply(lambda x: x + '%').to_string())
# ── CELL 9: Segment profile summary ─────────────────────────────────────────
# This table is the heart of the RFM analysis — what does each segment look like?

seg_profile = rfm.groupby('segment').agg(
    customers     = ('customer_unique_id', 'count'),
    avg_recency   = ('recency', 'mean'),
    avg_frequency = ('frequency', 'mean'),
    avg_monetary  = ('monetary', 'mean'),
    total_revenue = ('monetary', 'sum'),
    avg_rfm_score = ('RFM_score', 'mean')
).round(1).reset_index()

seg_profile['pct_customers'] = (seg_profile['customers'] / total * 100).round(1)
seg_profile['pct_revenue']   = (seg_profile['total_revenue'] / rfm['monetary'].sum() * 100).round(1)

# Sort by average RFM score descending
seg_profile = seg_profile.sort_values('avg_rfm_score', ascending=False).reset_index(drop=True)

print('=== SEGMENT PROFILE TABLE ===')
display_cols = ['segment','customers','pct_customers','avg_recency',
                'avg_frequency','avg_monetary','total_revenue','pct_revenue']
print(seg_profile[display_cols].to_string(index=False))
print()

# ── CELL 10: RFM Segment Visualisation — 4-panel dashboard ───────────────────

SEG_COLORS = {
    'Champions':          '#2a7d4f',
    'Loyal':              '#2d5a8e',
    'New Customers':      '#c9922a',
    'Potential Loyalists':'#6b8faf',
    'At Risk':            '#e07b39',
    'Churned':            '#c84b31',
}

seg_profile['color'] = seg_profile['segment'].map(SEG_COLORS)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Customer count
ax = axes[0, 0]
bars = ax.barh(seg_profile['segment'], seg_profile['customers'],
               color=seg_profile['color'], edgecolor='white', alpha=0.9)
ax.set_title('Customers per Segment', fontweight='bold', pad=10)
ax.set_xlabel('Customers')
for bar, val in zip(bars, seg_profile['customers']):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', fontsize=9)
ax.set_xlim(0, seg_profile['customers'].max() * 1.2)

# Panel 2: Revenue share
ax = axes[0, 1]
wedges, texts, autotexts = ax.pie(
    seg_profile['total_revenue'],
    labels=seg_profile['segment'],
    colors=seg_profile['color'],
    autopct='%1.1f%%',
    startangle=140,
    pctdistance=0.75,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
for t in autotexts: t.set_fontsize(9)
ax.set_title('Revenue Share by Segment', fontweight='bold', pad=10)

# Panel 3: Average LTV (monetary)
ax = axes[1, 0]
bars = ax.barh(seg_profile['segment'], seg_profile['avg_monetary'],
               color=seg_profile['color'], edgecolor='white', alpha=0.9)
ax.set_title('Average Lifetime Value (R$) per Segment', fontweight='bold', pad=10)
ax.set_xlabel('Avg LTV (R$)')
for bar, val in zip(bars, seg_profile['avg_monetary']):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f'R${val:,.0f}', va='center', fontsize=9)
ax.set_xlim(0, seg_profile['avg_monetary'].max() * 1.25)

# Panel 4: RFM Scatter — Recency vs Frequency, sized by Monetary
ax = axes[1, 1]
for seg, grp in rfm.groupby('segment'):
    # Sample max 300 points per segment for clarity
    sample = grp.sample(min(300, len(grp)), random_state=42)
    ax.scatter(
        sample['recency'],
        sample['frequency'],
        s=sample['monetary'].clip(0, 500) / 5,
        c=SEG_COLORS.get(seg, '#999'),
        alpha=0.4,
        label=seg
    )
ax.set_title('Recency vs Frequency\n(bubble size = monetary value)', fontweight='bold', pad=10)
ax.set_xlabel('Recency (days)')
ax.set_ylabel('Frequency (orders)')
ax.legend(fontsize=8, loc='upper right')

plt.suptitle('RFM Segment Analysis Dashboard', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/Users/SanthoshVislavath/Desktop/Product_analytics/Olist_data_analysis/output/08_rfm_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

# ── CELL 11: The Revenue-at-Stake Analysis ────────────────────────────────────
# This is the PRODUCT twist — translating segments into business decisions
#
# For each segment, we ask: "What happens if we move X% of these customers
# up to the next tier? How much revenue does that unlock?"

champions_ltv = rfm[rfm['segment'] == 'Champions']['monetary'].mean()

print('=== REVENUE AT STAKE BY SEGMENT ===')
print(f'{"Segment":<22} {"Customers":>10} {"Avg LTV":>10} {"Total Rev":>12} {"Gap to Champions":>18} {"Revenue Opportunity (30% conv)":>30}')
print('-' * 105)

opportunities = []
for _, row in seg_profile.iterrows():
    gap = champions_ltv - row['avg_monetary']
    opportunity_30pct = row['customers'] * 0.30 * gap if gap > 0 else 0
    opportunities.append({
        'segment': row['segment'],
        'customers': row['customers'],
        'avg_monetary': row['avg_monetary'],
        'total_rev': row['total_revenue'],
        'ltv_gap': gap,
        'opportunity_30pct': opportunity_30pct
    })
    print(f"{row['segment']:<22} {row['customers']:>10,} {row['avg_monetary']:>10.0f} {row['total_revenue']:>12,.0f} "
          f"{gap:>18.0f} {opportunity_30pct:>30,.0f}")

total_opportunity = sum(o['opportunity_30pct'] for o in opportunities)
print(f'\nTotal revenue opportunity (30% conversion to Champions LTV): R${total_opportunity:,.0f}')
print()

# ── CELL 12: The Segment Action Matrix ───────────────────────────────────────
# This is what makes RFM a PRODUCT tool, not just a data tool.
# Each segment should have a different intervention strategy.

action_matrix = {
    'Champions': {
        'goal':    'Keep & amplify',
        'actions': [
            'Reward program — exclusive early access to new products',
            'Referral incentive — they trust the platform, let them advocate',
            'No discounts needed — they pay full price already',
        ],
        'metric':  'Referral rate, NPS score, LTV maintenance'
    },
    'Loyal': {
        'goal':    'Upgrade to Champions',
        'actions': [
            'Category expansion nudge — show complementary products',
            'Loyalty tier system — make reaching Champions visible',
            'Personalised recommendations based on purchase history',
        ],
        'metric':  'Order frequency increase, category breadth'
    },
    'New Customers': {
        'goal':    'Trigger second purchase within 30 days',
        'actions': [
            'Day-7 follow-up email: "Your next order — based on what you bought"',
            'Onboarding category rail on homepage',
            'NPS at delivery — catch dissatisfied customers before they leave',
        ],
        'metric':  'Day-30 repeat purchase rate'
    },
    'Potential Loyalists': {
        'goal':    'Build habit before they go cold',
        'actions': [
            'Replenishment reminder for consumable categories',
            'Wishlist or save-for-later feature to reduce friction',
            'Gentle re-engagement at day-45 of inactivity (not yet At Risk)',
        ],
        'metric':  'Day-60 retention rate'
    },
    'At Risk': {
        'goal':    'Win back before fully churned',
        'actions': [
            'Win-back email sequence: day 60, 90, 120 of inactivity',
            'Personalised voucher (10%) — they know the platform, just need a nudge',
            'Churn survey — understand WHY they stopped',
        ],
        'metric':  'Reactivation rate, win-back ROI'
    },
    'Churned': {
        'goal':    'Learn from, selectively re-engage',
        'actions': [
            'Exit survey — mine for product improvement signals',
            'Suppress from paid retargeting (they\'ve already decided)',
            'If high original LTV: personal outreach + significant incentive',
        ],
        'metric':  'Survey response rate, cost-per-reactivation'
    },
}

print('=== SEGMENT ACTION MATRIX ===')
for seg, info in action_matrix.items():
    count = rfm[rfm['segment'] == seg].shape[0]
    print(f'\n🎯 {seg} ({count:,} customers)')
    print(f'   Goal: {info["goal"]}')
    for action in info['actions']:
        print(f'   → {action}')
    print(f'   Measure: {info["metric"]}')
# ── CELL 13: RFM Heatmap — R vs F scores coloured by avg Monetary ────────────
# A classic way to visualise the RFM space in 2D

pivot = rfm.pivot_table(values='monetary', index='F', columns='R', aggfunc='mean')
# Flip F axis — high frequency at top
pivot = pivot.iloc[::-1]

fig, ax = plt.subplots(figsize=(9, 6))
heatmap = sns.heatmap(
    pivot,
    annot=True, fmt='.0f', cmap='RdYlGn',
    linewidths=0.5, linecolor='white',
    cbar_kws={'label': 'Avg Monetary Value (R$)'},
    ax=ax
)
ax.set_title('RFM Heatmap: Avg Monetary Value by Recency & Frequency Score',
             fontweight='bold', pad=14)
ax.set_xlabel('Recency Score (5=most recent)', fontweight='bold')
ax.set_ylabel('Frequency Score (5=most orders)', fontweight='bold')

# Annotate corners
ax.text(4.5, 0.5, '← Champions', va='center', ha='center', fontsize=8,
        color='white', fontweight='bold')
ax.text(0.5, 4.5, 'Churned →', va='center', ha='center', fontsize=8,
        color='black', fontweight='bold')

plt.tight_layout()
plt.savefig('output/09_rfm_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()


# ── CELL 14: Save outputs ────────────────────────────────────────────────────

# Full RFM table with segments
rfm.to_csv('output/rfm_segments.csv', index=False)

# Segment summary for use in Stage 5 (product spec)
seg_profile.to_csv('output/segment_profile.csv', index=False)

print('Files saved:')
print('  output/rfm_segments.csv    → full customer-level RFM table (use in Stage 3+)')
print('  output/segment_profile.csv → segment summary (use in Stage 5 product spec)')
print()
print('Charts saved:')
print('  output/06_raw_rfm_distributions.png')
print('  output/07_rfm_scores.png')
print('  output/08_rfm_dashboard.png')
print('  output/09_rfm_heatmap.png')
