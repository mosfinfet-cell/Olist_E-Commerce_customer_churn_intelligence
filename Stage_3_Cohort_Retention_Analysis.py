# ── CELL 1: Imports & Setup ──────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

import os
os.makedirs('output', exist_ok=True)

print('Ready')

# ── CELL 2: Load data ────────────────────────────────────────────────────────
# master_df.csv from Stage 1 | rfm_segments.csv from Stage 2

df  = pd.read_csv('/Users/SanthoshVislavath/Desktop/Product_analytics/Olist_data_analysis/output/master_df.csv', parse_dates=['order_purchase_timestamp'])
rfm = pd.read_csv('/Users/SanthoshVislavath/Desktop/Product_analytics/Olist_data_analysis/output/rfm_segments.csv')

print(f'Orders:    {len(df):,}')
print(f'Customers: {df["customer_unique_id"].nunique():,}')
print(f'RFM rows:  {len(rfm):,}')
print(f'Date range: {df["order_purchase_timestamp"].min().date()} → {df["order_purchase_timestamp"].max().date()}')
# ── CELL 3: Assign cohort month to every customer ────────────────────────────

# Step 1: each customer's FIRST order date → cohort
first_order = df.groupby('customer_unique_id')['order_purchase_timestamp'].min().reset_index()
first_order.columns = ['customer_unique_id', 'first_order_date']
first_order['cohort_month'] = first_order['first_order_date'].dt.to_period('M')

# Step 2: merge cohort back onto every order row
df = df.merge(first_order[['customer_unique_id','cohort_month']], on='customer_unique_id', how='left')

# Step 3: compute each order's month as a Period
df['order_period'] = df['order_purchase_timestamp'].dt.to_period('M')

# Step 4: months since acquisition (0 = first month, 1 = one month later, etc.)
df['months_since_acq'] = (df['order_period'] - df['cohort_month']).apply(lambda x: x.n)

print('Months since acquisition value counts (first 12 months):')
print(df[df['months_since_acq'] <= 11]['months_since_acq'].value_counts().sort_index().to_dict())
print()
print(f'Unique cohort months: {df["cohort_month"].nunique()}')
print(f'Cohort range: {df["cohort_month"].min()} → {df["cohort_month"].max()}')
# ── CELL 4: Build the cohort size table ──────────────────────────────────────
# Cohort size = unique customers whose FIRST purchase was in that month
# This is the denominator for all retention rate calculations

cohort_size = (
    df[df['months_since_acq'] == 0]
    .groupby('cohort_month')['customer_unique_id']
    .nunique()
    .reset_index()
)
cohort_size.columns = ['cohort_month', 'cohort_size']

print('Largest cohorts (by acquisition count):')
print(cohort_size.sort_values('cohort_size', ascending=False).head(10).to_string(index=False))
# ── CELL 5: Build the cohort activity table ──────────────────────────────────
# Count UNIQUE active customers per cohort per month-offset

cohort_activity = (
    df[df['months_since_acq'] <= 11]
    .groupby(['cohort_month', 'months_since_acq'])['customer_unique_id']
    .nunique()
    .reset_index()
)
cohort_activity.columns = ['cohort_month', 'months_since_acq', 'active_customers']

# Merge in cohort size and calculate retention rate
cohort_activity = cohort_activity.merge(cohort_size, on='cohort_month', how='left')
cohort_activity['retention_rate'] = (
    cohort_activity['active_customers'] / cohort_activity['cohort_size'] * 100
).round(1)

print(f'Rows: {len(cohort_activity):,}')
print()
print('Sample — first cohort, all months:')
sample_cohort = cohort_activity['cohort_month'].min()
print(
    cohort_activity[cohort_activity['cohort_month'] == sample_cohort]
    [['months_since_acq','active_customers','cohort_size','retention_rate']]
    .to_string(index=False)
)
# ── CELL 6: Create the retention pivot matrix ────────────────────────────────

retention_matrix = cohort_activity.pivot(
    index='cohort_month',
    columns='months_since_acq',
    values='retention_rate'
)

# Filter out tiny cohorts (noisy percentages) and cohorts without 6+ months of follow-up
min_size = 30
valid_cohorts = cohort_size[cohort_size['cohort_size'] >= min_size]['cohort_month']
retention_matrix = retention_matrix[retention_matrix.index.isin(valid_cohorts)]
retention_matrix = retention_matrix.dropna(subset=[5])   # must have month-5 data

print(f'Cohorts in matrix: {len(retention_matrix)}')
print(f'Month columns: {list(retention_matrix.columns)}')
print()
print('Retention matrix preview (first 5 cohorts, first 7 months):')
print(retention_matrix.iloc[:5, :7].to_string())
# ── CELL 7: The Cohort Retention Heatmap ─────────────────────────────────────

month_cols = [c for c in range(12) if c in retention_matrix.columns]
plot_matrix = retention_matrix[month_cols].copy()
plot_matrix.index = plot_matrix.index.astype(str)

fig, ax = plt.subplots(figsize=(16, max(8, len(plot_matrix) * 0.55)))

sns.heatmap(
    plot_matrix,
    annot=True, fmt='.0f',
    cmap='YlGn',
    vmin=0, vmax=100,
    linewidths=0.8, linecolor='white',
    cbar_kws={'label': 'Retention Rate (%)', 'shrink': 0.6},
    mask=plot_matrix.isna(),
    ax=ax,
    annot_kws={'size': 9, 'weight': 'bold'}
)

ax.set_title(
    'Monthly Cohort Retention Heatmap\n'
    '(% of original cohort still active, by months since first purchase)',
    fontsize=13, fontweight='bold', pad=16
)
ax.set_xlabel('Months Since First Purchase', fontsize=11, fontweight='bold')
ax.set_ylabel('Acquisition Cohort', fontsize=11, fontweight='bold')
ax.set_xticklabels([f'M+{int(c.get_text())}' for c in ax.get_xticklabels()], rotation=0)

plt.tight_layout()
plt.savefig('output/10_cohort_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
# ── CELL 8: Retention curves ─────────────────────────────────────────────────
# Line chart: each cohort's retention curve over time
# Colour gradient: blue = earlier cohorts, red = later cohorts

fig, ax = plt.subplots(figsize=(13, 6))

n_cohorts = len(plot_matrix)
colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, n_cohorts))

for i, (cohort, row) in enumerate(plot_matrix.iterrows()):
    valid = row.dropna()
    if len(valid) < 3:
        continue
    ax.plot(
        valid.index.astype(int),
        valid.values,
        marker='o', markersize=4, linewidth=1.8,
        color=colors[i], alpha=0.8, label=cohort
    )

ax.set_title('Cohort Retention Curves\n(blue = earlier cohorts, red = later cohorts)',
             fontsize=13, fontweight='bold', pad=14)
ax.set_xlabel('Months Since First Purchase', fontsize=11)
ax.set_ylabel('Retention Rate (%)', fontsize=11)
ax.set_ylim(0, 105)
ax.set_xticks(month_cols)
ax.set_xticklabels([f'M+{m}' for m in month_cols])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.axhline(y=10, color='red', linestyle=':', alpha=0.3, linewidth=1.5, label='10% threshold')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8, framealpha=0.9)

plt.tight_layout()
plt.savefig('output/11_retention_curves.png', dpi=150, bbox_inches='tight')
plt.show()
# ── CELL 9: Cohort performance statistics ────────────────────────────────────

plot_matrix_sorted = plot_matrix.sort_index()
n = len(plot_matrix_sorted)
early_cohorts = plot_matrix_sorted.iloc[:n//2]
late_cohorts  = plot_matrix_sorted.iloc[n//2:]

print('=== COHORT PERFORMANCE COMPARISON ===')
print()
for month in [1, 3, 6]:
    if month not in plot_matrix_sorted.columns:
        continue
    early_avg = early_cohorts[month].mean()
    late_avg  = late_cohorts[month].mean()
    gap = early_avg - late_avg
    print(f'M+{month} retention:')
    print(f'  Early cohorts avg: {early_avg:.1f}%')
    print(f'  Late cohorts avg:  {late_avg:.1f}%')
    print(f'  Gap:               {gap:.1f} percentage points')
    print()

print('=== MONTH-OVER-MONTH DROP-OFF (avg across all cohorts) ===')
avg_retention = plot_matrix_sorted.mean()
for i in range(1, len(avg_retention)):
    prev = avg_retention.iloc[i-1]
    curr = avg_retention.iloc[i]
    if pd.isna(prev) or pd.isna(curr):
        continue
    drop = prev - curr
    m_prev = int(avg_retention.index[i-1])
    m_curr = int(avg_retention.index[i])
    print(f'  M+{m_prev} → M+{m_curr}: -{drop:.1f}pp  (retention: {curr:.1f}%)')
# ── CELL 10: Find the retention cliff ────────────────────────────────────────
# The cliff = the single month with the largest average drop.
# This is the MOST actionable finding because it defines your intervention deadline.

avg_ret = plot_matrix_sorted.mean().dropna()
drops = avg_ret.diff().dropna()

cliff_month = int(drops.idxmin())
cliff_prev  = int(drops.index[list(drops.index).index(cliff_month) - 1])
cliff_drop  = drops.min()

print(' RETENTION CLIFF IDENTIFIED')
print(f'   Largest single-month drop: M+{cliff_prev} → M+{cliff_month}')
print(f'   Average drop: {cliff_drop:.1f} percentage points')
print()
print(f'   Avg retention BEFORE cliff (M+{cliff_prev}): {avg_ret[cliff_prev]:.1f}%')
print(f'   Avg retention AT cliff     (M+{cliff_month}): {avg_ret[cliff_month]:.1f}%')
print()
print('PRODUCT INTERPRETATION:')
print(f'   Customers who have not re-purchased by M+{cliff_month} are very unlikely to return.')
print(f'   Your re-engagement trigger should fire at M+{cliff_month - 1} — BEFORE the cliff.')
print()
print('   Ask: What experience happens around this time?')
print('   → Delivery quality? No reason to return? Competitor promotion?')
# ── CELL 11: Early vs late cohort comparison chart ───────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# ── Left: average retention curves, early vs late ──
early_avg_curve = early_cohorts[month_cols].mean()
late_avg_curve  = late_cohorts[month_cols].mean()

ax1.plot(month_cols, early_avg_curve.values, marker='o', markersize=5,
         linewidth=2.2, color='#2d5a8e', label=f'Earlier cohorts (n={len(early_cohorts)})')
ax1.plot(month_cols, late_avg_curve.values, marker='s', markersize=5,
         linewidth=2.2, color='#c84b31', linestyle='--', label=f'Later cohorts (n={len(late_cohorts)})')

valid_m = [m for m in month_cols
           if not pd.isna(early_avg_curve.get(m)) and not pd.isna(late_avg_curve.get(m))]
ax1.fill_between(
    valid_m,
    [early_avg_curve[m] for m in valid_m],
    [late_avg_curve[m] for m in valid_m],
    alpha=0.12, color='gray', label='Retention gap'
)

ax1.set_title('Earlier vs Later Cohorts\n(average retention curves)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Months Since First Purchase')
ax1.set_ylabel('Retention Rate (%)')
ax1.set_xticks(month_cols)
ax1.set_xticklabels([f'M+{m}' for m in month_cols])
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax1.set_ylim(0, 105)
ax1.legend(fontsize=9)

# ── Right: M+3 retention bar by cohort ──
if 3 in plot_matrix_sorted.columns:
    m3_data = plot_matrix_sorted[3].dropna().sort_index()
    bar_colors = ['#c84b31' if str(c) >= '2018' else '#2d5a8e' for c in m3_data.index]
    ax2.bar(range(len(m3_data)), m3_data.values, color=bar_colors, edgecolor='white', alpha=0.85)
    avg_line = m3_data.mean()
    ax2.axhline(avg_line, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
    ax2.text(len(m3_data) - 0.5, avg_line + 1, f'Avg {avg_line:.1f}%', fontsize=9)
    ax2.set_title('M+3 Retention Rate by Cohort', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Cohort')
    ax2.set_ylabel('Retention at M+3 (%)')
    ax2.set_xticks(range(len(m3_data)))
    ax2.set_xticklabels([str(c) for c in m3_data.index], rotation=45, ha='right', fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax2.legend(handles=[
        mpatches.Patch(color='#2d5a8e', label='Earlier cohorts'),
        mpatches.Patch(color='#c84b31', label='Later cohorts')
    ], fontsize=9)

plt.tight_layout()
plt.savefig('output/12_cohort_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
# ── CELL 12: RFM segment mix by acquisition cohort ───────────────────────────

rfm_cohort = rfm.merge(
    first_order[['customer_unique_id', 'cohort_month']],
    on='customer_unique_id', how='left'
)
rfm_cohort['cohort_month'] = rfm_cohort['cohort_month'].astype(str)

seg_mix = (
    rfm_cohort.groupby(['cohort_month', 'segment'])
    .size().reset_index(name='count')
)
cohort_totals = rfm_cohort.groupby('cohort_month').size().reset_index(name='total')
seg_mix = seg_mix.merge(cohort_totals, on='cohort_month')
seg_mix['pct'] = (seg_mix['count'] / seg_mix['total'] * 100).round(1)

churn_by_cohort = (
    seg_mix[seg_mix['segment'].isin(['At Risk', 'Churned'])]
    .groupby('cohort_month')['pct'].sum()
    .reset_index()
)
churn_by_cohort.columns = ['cohort_month', 'pct_at_risk_or_churned']
churn_by_cohort = churn_by_cohort.sort_values('cohort_month')

print('% customers who are At Risk or Churned, by acquisition cohort:')
print(churn_by_cohort.to_string(index=False))
# ── CELL 13: Visualise churn rate trend + segment stack ──────────────────────

SEG_COLORS = {
    'Champions': '#2a7d4f', 'Loyal': '#2d5a8e',
    'New Customers': '#c9922a', 'Potential Loyalists': '#6b8faf',
    'At Risk': '#e07b39', 'Churned': '#c84b31'
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# ── Left: churn rate trend line ──
x = range(len(churn_by_cohort))
avg_churn = churn_by_cohort['pct_at_risk_or_churned'].mean()
bar_colors = ['#c84b31' if p > avg_churn else '#2d5a8e'
              for p in churn_by_cohort['pct_at_risk_or_churned']]
ax1.bar(x, churn_by_cohort['pct_at_risk_or_churned'], color=bar_colors, edgecolor='white', alpha=0.85)
ax1.axhline(avg_churn, color='black', linestyle='--', alpha=0.5, linewidth=1.5,
            label=f'Avg: {avg_churn:.1f}%')
ax1.set_title('% At Risk + Churned by Acquisition Cohort', fontsize=12, fontweight='bold')
ax1.set_xlabel('Acquisition Cohort')
ax1.set_ylabel('% At Risk or Churned')
ax1.set_xticks(x)
ax1.set_xticklabels(churn_by_cohort['cohort_month'], rotation=45, ha='right', fontsize=8)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax1.legend(fontsize=9)

# ── Right: stacked segment mix (every 3rd cohort for clarity) ──
selected = sorted(rfm_cohort['cohort_month'].unique())[::3]
seg_pivot = seg_mix[seg_mix['cohort_month'].isin(selected)].pivot(
    index='cohort_month', columns='segment', values='pct'
).fillna(0)

seg_order = ['Champions','Loyal','Potential Loyalists','New Customers','At Risk','Churned']
seg_order = [s for s in seg_order if s in seg_pivot.columns]
bottom = np.zeros(len(seg_pivot))
for seg in seg_order:
    ax2.bar(range(len(seg_pivot)), seg_pivot[seg], bottom=bottom,
            label=seg, color=SEG_COLORS.get(seg, '#999'), edgecolor='white', linewidth=0.5)
    bottom += seg_pivot[seg].values

ax2.set_title('RFM Segment Mix by Acquisition Cohort', fontsize=12, fontweight='bold')
ax2.set_xlabel('Acquisition Cohort')
ax2.set_ylabel('% of Cohort')
ax2.set_xticks(range(len(seg_pivot)))
ax2.set_xticklabels(seg_pivot.index, rotation=45, ha='right', fontsize=8)
ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))

plt.tight_layout()
plt.savefig('output/13_segment_mix_by_cohort.png', dpi=150, bbox_inches='tight')
plt.show()

# ── CELL 14: Hypothesis Formation ────────────────────────────────────────────


hypotheses = [
    {
        'id': 'H1',
        'observation': f'Retention cliff at M+{cliff_month} — largest single drop-off point',
        'hypothesis':  'Customers with no re-purchase trigger by this month have no active reason to return',
        'test':        'Compare review scores of cliff-churners vs retained customers',
        'intervention': f'Re-engagement email triggered at M+{cliff_month - 1} — before the cliff'
    },
    {
        'id': 'H2',
        'observation': 'Later cohorts retain significantly worse than earlier cohorts',
        'hypothesis':  'Product quality declined: more sellers → higher variance → lower avg review scores',
        'test':        'Compare average review scores across cohort vintages',
        'intervention': 'Seller quality gating: flag sellers below 3.5-star rolling average'
    },
    {
        'id': 'H3',
        'observation': 'Later cohorts have higher % At Risk and Churned in RFM segments',
        'hypothesis':  'Later cohorts were acquired from lower-intent channels — lower fit customers',
        'test':        'Compare first-order category mix and payment method across cohorts',
        'intervention': 'Tighten acquisition channel mix; prioritise high-LTV category entry points'
    },
    {
        'id': 'H4',
        'observation': 'High single-order rate across all cohorts',
        'hypothesis':  'There is no habit-formation loop — customers complete first order, find no reason to return',
        'test':        'Map the post-purchase journey: what does the customer see after delivery?',
        'intervention': 'Post-delivery personalised recommendation rail + day-7 NPS trigger'
    },
]

print('=== HYPOTHESIS MATRIX (for Stage 4 testing) ===\n')
for h in hypotheses:
    print(f'[{h["id"]}] {h["observation"]}')
    print(f'  Hypothesis:   {h["hypothesis"]}')
    print(f'  Test:         {h["test"]}')
    print(f'  Intervention: {h["intervention"]}')
    print()
# ── CELL 15: Save outputs ─────────────────────────────────────────────────────

retention_matrix.to_csv('output/retention_matrix.csv')
cohort_activity.to_csv('output/cohort_activity.csv', index=False)
churn_by_cohort.to_csv('output/churn_by_cohort.csv', index=False)
rfm_cohort.to_csv('output/rfm_with_cohort.csv', index=False)

print('Files saved:')
print('  output/retention_matrix.csv    → pivot table')
print('  output/cohort_activity.csv     → raw cohort data')
print('  output/churn_by_cohort.csv     → churn % per cohort')
print('  output/rfm_with_cohort.csv     → RFM + cohort joined (for Stage 4)')
print()
print('Charts:')
print('  output/10_cohort_heatmap.png         ← PORTFOLIO SHOWPIECE')
print('  output/11_retention_curves.png')
print('  output/12_cohort_comparison.png')
print('  output/13_segment_mix_by_cohort.png')
