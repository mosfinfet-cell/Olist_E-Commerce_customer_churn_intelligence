Stage 5 closes the loop. Every chart, every number, every hypothesis from Stages 1–4 gets synthesised into two final deliverables:

1. **The Summary Dashboard** — a single chart combining the best visualisations from all four stages
2. **The Product Requirements Document (PRD)** — a polished PDF suitable for a portfolio, job application, or stakeholder presentation

The PRD contains:
- Executive summary (the one-page version of the entire project)
- Ranked evidence matrix from Stage 4
- A 3-phase product roadmap, each feature traceable to a specific data finding
- A/B test plans for the top two interventions
- Success metrics framework
- Revenue opportunity quantification

# ── CELL 1: Imports ──────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

os.makedirs('output', exist_ok=True)

# ── CELL 2: Load all Stage outputs ───────────────────────────────────────────

rfm         = pd.read_csv('output/rfm_segments.csv')
seg_profile = pd.read_csv('output/segment_profile.csv')
rfm_cohort  = pd.read_csv('output/rfm_with_cohort.csv')
ret_matrix  = pd.read_csv('output/retention_matrix.csv', index_col=0)
ret_matrix.columns = ret_matrix.columns.astype(int)
evidence    = pd.read_csv('output/evidence_matrix.csv')
cat_stats   = pd.read_csv('output/category_repeat_rates.csv')
rep_score   = pd.read_csv('output/repeat_by_review_score.csv')

print('All Stage 1–4 outputs loaded.')
print(f'RFM customers:      {len(rfm):,}')
print(f'Cohort matrix rows: {len(ret_matrix)}')
print(f'Evidence matrix:    {len(evidence)} hypotheses')
## Step 1: The Master Summary Dashboard

A single 6-panel figure that tells the complete story — from segment health to cohort retention to root-cause evidence.
This is the image that goes at the top of your portfolio README and LinkedIn post.
# ── CELL 3: Master 6-panel summary dashboard ─────────────────────────────────

SEG_COLORS = {
    'Champions':'#2a7d4f','Loyal':'#2d5a8e','New Customers':'#c9922a',
    'Potential Loyalists':'#6b8faf','At Risk':'#e07b39','Churned':'#c84b31'
}
SEG_ORDER = ['Champions','Loyal','Potential Loyalists','New Customers','At Risk','Churned']

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#FAFAF8')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.32)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, :])
ax5 = fig.add_subplot(gs[2, 0:2])
ax6 = fig.add_subplot(gs[2, 2])

# ── Panel 1: Customer count by segment ──
seg_counts = rfm['segment'].value_counts().reindex(SEG_ORDER).dropna()
ax1.barh(seg_counts.index[::-1], seg_counts.values[::-1],
         color=[SEG_COLORS[s] for s in seg_counts.index[::-1]],
         edgecolor='white', alpha=0.9)
for i, (seg, v) in enumerate(zip(seg_counts.index[::-1], seg_counts.values[::-1])):
    ax1.text(v + 50, i, f'{v:,}', va='center', fontsize=8)
ax1.set_title('Customer Count by Segment', fontweight='bold', fontsize=10)
ax1.set_xlabel('Customers')
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{int(x):,}'))

# ── Panel 2: Revenue share by segment ──
seg_revenue = rfm.groupby('segment')['monetary'].sum().reindex(SEG_ORDER).dropna()
pie_colors = [SEG_COLORS[s] for s in seg_revenue.index]
wedges, texts, autotexts = ax2.pie(
    seg_revenue.values, labels=None,
    colors=pie_colors, autopct='%1.0f%%',
    startangle=90, pctdistance=0.75,
    wedgeprops={'edgecolor':'white','linewidth':1.5}
)
for at in autotexts:
    at.set_fontsize(8)
ax2.legend(seg_revenue.index, loc='lower center', bbox_to_anchor=(0.5,-0.18),
           fontsize=7, ncol=2)
ax2.set_title('Revenue Share by Segment', fontweight='bold', fontsize=10)

# ── Panel 3: Avg LTV by segment ──
seg_ltv = rfm.groupby('segment')['monetary'].mean().reindex(SEG_ORDER).dropna()
bars = ax3.bar(range(len(seg_ltv)), seg_ltv.values,
               color=[SEG_COLORS[s] for s in seg_ltv.index],
               edgecolor='white', alpha=0.9)
ax3.set_title('Avg Customer LTV by Segment', fontweight='bold', fontsize=10)
ax3.set_xticks(range(len(seg_ltv)))
ax3.set_xticklabels([s.replace(' ',"\n") for s in seg_ltv.index], fontsize=7)
ax3.set_ylabel('Avg LTV (BRL)')
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'R${x:,.0f}'))

# ── Panel 4: Cohort retention heatmap (spanning full width) ──
month_cols = [c for c in range(12) if c in ret_matrix.columns]
plot_mat = ret_matrix[month_cols].copy()
plot_mat.index = [str(i) for i in plot_mat.index]
sns.heatmap(
    plot_mat, annot=True, fmt='.0f', cmap='YlGn',
    vmin=0, vmax=100, linewidths=0.5, linecolor='white',
    cbar_kws={'label':'Retention %','shrink':0.6,'orientation':'horizontal','pad':0.08},
    mask=plot_mat.isna(), ax=ax4, annot_kws={'size':7, 'weight':'bold'}
)
ax4.set_title('Monthly Cohort Retention Heatmap (% still active by months since first purchase)',
              fontweight='bold', fontsize=10)
ax4.set_xlabel('Months Since First Purchase', fontsize=9)
ax4.set_ylabel('Acquisition Cohort', fontsize=9)
ax4.set_xticklabels([f'M+{int(c.get_text())}' for c in ax4.get_xticklabels()],
                    rotation=0, fontsize=8)
ax4.tick_params(axis='y', labelsize=7)

# ── Panel 5: Evidence matrix priority scores ──
short_h = [h.split(':')[0] for h in evidence['Hypothesis']]
h_colors = ['#2a7d4f','#2d5a8e','#c9922a','#c84b31']
bars5 = ax5.barh(short_h[::-1], evidence['Priority_Score'][::-1],
                 color=h_colors[::-1], edgecolor='white', alpha=0.9)
for bar, score in zip(bars5, evidence['Priority_Score'][::-1]):
    ax5.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
             f'{score}/125', va='center', fontsize=9, fontweight='bold')
ax5.set_title('Hypothesis Priority Scores (Evidence × Impact × Reversibility)',
              fontweight='bold', fontsize=10)
ax5.set_xlabel('Priority Score (max 125)')
ax5.set_xlim(0, 145)
for i, (h, row) in enumerate(zip(short_h[::-1], evidence.iloc[::-1].itertuples())):
    ax5.text(2, i+0.05,
             f'Evidence {row.Evidence_Strength}/5 · Impact {row.Business_Impact}/5 · Reversibility {row.Reversibility}/5',
             fontsize=7, color='white', va='center')

# ── Panel 6: Review score → repeat rate ──
score_colors = ['#c84b31','#e07b39','#c9922a','#2d5a8e','#2a7d4f']
ax6.bar(rep_score['review_score'], rep_score['repeat_rate_pct'],
        color=score_colors, edgecolor='white', width=0.6, alpha=0.9)
ax6.set_title('Repeat Rate by First-Order Review', fontweight='bold', fontsize=10)
ax6.set_xlabel('Review Score')
ax6.set_ylabel('% Who Returned')
ax6.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:.0f}%'))
for i, row in rep_score.iterrows():
    ax6.text(row['review_score'], row['repeat_rate_pct']+0.3,
             f"{row['repeat_rate_pct']:.1f}%", ha='center', fontsize=8, fontweight='bold')

# ── Master title ──
fig.suptitle('Olist Customer Churn Intelligence — Portfolio Summary Dashboard',
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('output/19_master_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor='#FAFAF8')
plt.show()
print('✅ Master dashboard saved → output/19_master_dashboard.png')
## Step 2: Quantify the Revenue Opportunity

Before writing the PRD, we need the headline numbers.
Stakeholders and hiring managers want to know: *how much is at stake?*
This cell computes the revenue opportunity for each proposed intervention.
# ── CELL 4: Revenue opportunity quantification ───────────────────────────────
# For each intervention, estimate: how many customers × conversion rate × LTV uplift

champions_ltv = rfm[rfm['segment']=='Champions']['monetary'].mean()
total_customers = len(rfm)

opps = []
for seg in SEG_ORDER:
    subset = rfm[rfm['segment']==seg]
    if len(subset) == 0 or seg == 'Champions':
        continue
    n       = len(subset)
    avg_ltv = subset['monetary'].mean()
    ltv_gap = champions_ltv - avg_ltv
    # Conservative 30% conversion on intervention
    opp     = n * 0.30 * ltv_gap
    opps.append({'Segment':seg,'Customers':n,'Avg_LTV':avg_ltv,'LTV_Gap':ltv_gap,'Revenue_Opp_30pct':opp})

opp_df = pd.DataFrame(opps).sort_values('Revenue_Opp_30pct', ascending=False)
total_opp = opp_df['Revenue_Opp_30pct'].sum()

print('Revenue Opportunity by Segment (30% conversion assumption):')
print()
for _, row in opp_df.iterrows():
    print(f"{row['Segment']:25s} | "
          f"{row['Customers']:6,} customers | "
          f"LTV gap R${row['LTV_Gap']:,.0f} | "
          f"Opportunity R${row['Revenue_Opp_30pct']:,.0f}")

#print(f'\n{'─'*70}')
print(f'Total revenue opportunity (30% conversion): R${total_opp:,.0f}')
print(f'Champions LTV used as benchmark: R${champions_ltv:,.0f}')
# ── CELL 5: Build the 3-phase roadmap table ──────────────────────────────────
# Each feature maps to: hypothesis → segment → metric → owner → timeline

roadmap = pd.DataFrame([
    # ── PHASE 1: Quick wins — high evidence, high reversibility ──
    {
        'Phase': 'Phase 1\n(0–60 days)',
        'Feature': 'Post-Delivery NPS Trigger',
        'Hypothesis': 'H1',
        'Segment': 'New Customers',
        'Mechanism': 'Day-7 delivery NPS → flag detractors → resolution flow',
        'Success Metric': 'Repeat rate for 1-2★ customers +5pp',
        'A/B Testable': 'Yes — email send vs holdout'
    },
    {
        'Phase': 'Phase 1\n(0–60 days)',
        'Feature': 'Day-[X] Re-engagement Sequence',
        'Hypothesis': 'H3',
        'Segment': 'New Customers + Potential Loyalists',
        'Mechanism': 'Personalised email at median inter-order gap − 14 days, category-aware',
        'Success Metric': '30-day repeat rate +3pp, M+3 cohort retention +2pp',
        'A/B Testable': 'Yes — trigger timing and content'
    },
    {
        'Phase': 'Phase 1\n(0–60 days)',
        'Feature': 'At-Risk Win-Back Sequence',
        'Hypothesis': 'H1 + H3',
        'Segment': 'At Risk',
        'Mechanism': '3-email win-back sequence with personalised voucher at email 2',
        'Success Metric': 'Reactivation rate >8% of At Risk segment',
        'A/B Testable': 'Yes — voucher value, timing, subject lines'
    },
    # ── PHASE 2: Medium-term product changes ──
    {
        'Phase': 'Phase 2\n(60–180 days)',
        'Feature': 'Category-Aware Onboarding',
        'Hypothesis': 'H4',
        'Segment': 'New Customers',
        'Mechanism': 'First-purchase category → personalised cross-sell into sticky categories',
        'Success Metric': '% of new customers in consumable category +5pp at 90d',
        'A/B Testable': 'Yes — personalised vs generic email'
    },
    {
        'Phase': 'Phase 2\n(60–180 days)',
        'Feature': 'Champions Loyalty Programme',
        'Hypothesis': 'Retention of top segment',
        'Segment': 'Champions',
        'Mechanism': 'Exclusive early access, referral incentives, no-discount-needed rewards',
        'Success Metric': 'Champion churn rate <5% QoQ, referral rate >10%',
        'A/B Testable': 'Partial — referral mechanic'
    },
    {
        'Phase': 'Phase 2\n(60–180 days)',
        'Feature': 'Replenishment Reminder System',
        'Hypothesis': 'H3',
        'Segment': 'Loyal + Potential Loyalists',
        'Mechanism': 'Category-based reorder prediction → reminder at predicted reorder window',
        'Success Metric': 'Order frequency +15% for eligible customers',
        'A/B Testable': 'Yes — trigger vs no trigger'
    },
    # ── PHASE 3: Structural / longer-term ──
    {
        'Phase': 'Phase 3\n(180+ days)',
        'Feature': 'Seller Quality Score Gating',
        'Hypothesis': 'H2',
        'Segment': 'All (acquisition quality)',
        'Mechanism': 'Flag sellers with <3.5★ rolling avg → demotion + remediation flow',
        'Success Metric': 'Avg first-order review score >4.2 for new cohorts',
        'A/B Testable': 'No — platform-wide policy change'
    },
    {
        'Phase': 'Phase 3\n(180+ days)',
        'Feature': 'High-Sticky Category Acquisition',
        'Hypothesis': 'H4',
        'Segment': 'All (acquisition mix)',
        'Mechanism': 'Shift paid channel budget toward highest-LTV entry categories',
        'Success Metric': 'New customer M+6 retention +3pp vs baseline cohort',
        'A/B Testable': 'Yes — channel budget allocation test'
    },
])

print('3-Phase Product Roadmap:')
print()
for phase, grp in roadmap.groupby('Phase', sort=False):
    print(f'── {phase.replace(chr(10)," ")} ──')
    for _, row in grp.iterrows():
        print(f'  [{row.Hypothesis}] {row.Feature}')
        print(f'       Segment: {row.Segment}')
        print(f'       Metric:  {row["Success Metric"]}')
        print()

roadmap.to_csv('output/roadmap.csv', index=False)
print('Roadmap saved → output/roadmap.csv')
# ── CELL 6: A/B Test design for top 2 interventions ──────────────────────────

ab_tests = [
    {
        'name': 'Test A — Post-Delivery NPS Trigger (H1)',
        'hypothesis': 'Customers who receive a Day-7 NPS survey and a resolution offer after a bad first experience will repurchase at a higher rate than those who receive no post-delivery communication.',
        'control': 'No post-delivery email (current state)',
        'treatment': 'Day-7 email with NPS survey; if score ≤2, auto-trigger resolution offer (voucher + apology)',
        'segment': 'All new customers (first order, delivered)',
        'primary_metric': '30-day repeat purchase rate',
        'secondary_metrics': ['60-day retention', 'avg review score on 2nd order', 'refund rate'],
        'mde': '2 percentage points on repeat rate (from ~X% baseline)',
        'sample_per_variant': '~2,000 (calculate from your baseline repeat rate)',
        'duration': '6–8 weeks (to accumulate 30-day windows)',
        'guardrail': 'Total refund value must not exceed revenue recovered',
        'winner_criteria': 'p < 0.05 on primary metric, no guardrail violation',
    },
    {
        'name': 'Test B — Re-engagement Trigger Timing (H3)',
        'hypothesis': 'Sending a personalised re-engagement email at [median gap − 14 days] post-purchase will produce a higher 60-day repeat rate than the current silence or a fixed 30-day email.',
        'control': 'No re-engagement email (current state)',
        'treatment': 'Personalised email at day [median − 14], featuring: items from first-purchase category + complementary sticky-category recommendation',
        'segment': 'New Customers and Potential Loyalists who have not placed a second order',
        'primary_metric': '60-day repeat purchase rate',
        'secondary_metrics': ['Revenue per email sent', 'unsubscribe rate', 'M+3 cohort retention'],
        'mde': '3 percentage points on 60-day repeat rate',
        'sample_per_variant': '~1,500 per arm (3 arms: control, Day-30, Day-[median−14])',
        'duration': '8–10 weeks',
        'guardrail': 'Unsubscribe rate must not exceed 2% (preserve email channel health)',
        'winner_criteria': 'p < 0.05, positive lift on revenue per email sent',
    },
]

for test in ab_tests:
    print(f'=== {test["name"]} ===')
    print(f'Hypothesis:     {test["hypothesis"][:80]}...')
    print(f'Control:        {test["control"]}')
    print(f'Treatment:      {test["treatment"][:70]}...')
    print(f'Primary metric: {test["primary_metric"]}')
    print(f'Duration:       {test["duration"]}')
    print(f'Guardrail:      {test["guardrail"]}')
    print()
# ── CELL 7: Generate the PRD PDF ─────────────────────────────────────────────
# This builds the polished Product Requirements Document.

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import Flowable
import datetime

# ── Colour palette ──
INK    = colors.HexColor('#0D0D0D')
ACCENT = colors.HexColor('#C84B31')
BLUE   = colors.HexColor('#2D5A8E')
GREEN  = colors.HexColor('#2A7D4F')
GOLD   = colors.HexColor('#C9922A')
LIGHT  = colors.HexColor('#F5F0E8')
CREAM  = colors.HexColor('#EDE8DC')
MUTED  = colors.HexColor('#6B6257')
WHITE  = colors.white
PHASE_COLORS = [
    colors.HexColor('#2A7D4F'),   # Phase 1 green
    colors.HexColor('#2D5A8E'),   # Phase 2 blue
    colors.HexColor('#6B6257'),   # Phase 3 muted
]

# ── Styles ──
base_styles = getSampleStyleSheet()

def S(name, **kwargs):
    return ParagraphStyle(name, fontName='Helvetica', fontSize=10,
                          textColor=INK, leading=14, **kwargs)

styles = {
    'cover_title':   S('ct', fontName='Helvetica-Bold', fontSize=36, textColor=INK, leading=42, spaceAfter=8),
    'cover_sub':     S('cs', fontName='Helvetica', fontSize=18, textColor=BLUE, leading=24, spaceAfter=6),
    'cover_tag':     S('ctag', fontName='Helvetica-Oblique', fontSize=11, textColor=MUTED, leading=15, spaceAfter=20),
    'section':       S('sec', fontName='Helvetica-Bold', fontSize=16, textColor=INK, leading=20, spaceBefore=18, spaceAfter=8),
    'subsection':    S('ss', fontName='Helvetica-Bold', fontSize=12, textColor=BLUE, leading=16, spaceBefore=12, spaceAfter=4),
    'body':          S('body', fontSize=10, leading=14, spaceAfter=6),
    'body_bold':     S('bb', fontName='Helvetica-Bold', fontSize=10, leading=14, spaceAfter=6),
    'small':         S('sm', fontSize=8.5, textColor=MUTED, leading=12, spaceAfter=4),
    'callout_label': S('cl', fontName='Helvetica-Bold', fontSize=9, textColor=ACCENT, leading=12, spaceAfter=3),
    'callout_body':  S('cb', fontSize=9, textColor=INK, leading=13, spaceAfter=0),
    'label':         S('lbl', fontName='Helvetica-Bold', fontSize=9, textColor=WHITE, leading=12),
    'metric':        S('met', fontName='Helvetica-Bold', fontSize=20, textColor=ACCENT, leading=24, alignment=TA_CENTER),
    'metric_label':  S('ml', fontSize=8, textColor=MUTED, leading=11, alignment=TA_CENTER),
    'phase_h':       S('ph', fontName='Helvetica-Bold', fontSize=10, textColor=WHITE, leading=14),
    'tbl_h':         S('th', fontName='Helvetica-Bold', fontSize=8.5, textColor=WHITE, leading=12),
    'tbl_b':         S('tb', fontSize=8.5, textColor=INK, leading=12),
    'tbl_b_bold':    S('tbb', fontName='Helvetica-Bold', fontSize=8.5, textColor=INK, leading=12),
}

W, H = A4
MARGIN = 2.2 * cm
CONTENT_W = W - 2 * MARGIN

print('Styles ready. Building PDF...')
# ── CELL 8: PRD helper functions ─────────────────────────────────────────────

def hr(color=CREAM, thickness=1.5):
    return HRFlowable(width='100%', thickness=thickness, color=color, spaceAfter=10, spaceBefore=4)

def sp(h=6):
    return Spacer(1, h)

def section_header(text):
    return KeepTogether([
        sp(4),
        Table([[Paragraph(text, styles['section'])]],
              colWidths=[CONTENT_W],
              style=TableStyle([('LINEBELOW', (0,0), (-1,-1), 2, ACCENT),
                                ('TOPPADDING',(0,0),(-1,-1),6),
                                ('BOTTOMPADDING',(0,0),(-1,-1),4)])),
        sp(4),
    ])

def callout_box(label, text, color=BLUE):
    return Table(
        [[Paragraph('', styles['body']),
          [Paragraph(label, styles['callout_label']), Paragraph(text, styles['callout_body'])]]],
        colWidths=[0.35*cm, CONTENT_W - 0.35*cm],
        style=TableStyle([
            ('BACKGROUND', (0,0), (0,-1), color),
            ('BACKGROUND', (1,0), (1,-1), LIGHT),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (1,0), (1,-1), 10),
            ('RIGHTPADDING', (1,0), (1,-1), 10),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LINEAFTER', (0,0), (0,-1), 0, color),
        ])
    )

def metric_card(value, label, color=ACCENT):
    return Table(
        [[Paragraph(value, styles['metric'])],
         [Paragraph(label, styles['metric_label'])]],
        colWidths=[CONTENT_W / 4],
        style=TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), LIGHT),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('BOX', (0,0), (-1,-1), 1, CREAM),
        ])
    )

def four_metrics(vals_labels):
    """Row of 4 metric cards side by side"""
    w = (CONTENT_W - 3*0.3*cm) / 4
    row = []
    for val, lbl, col in vals_labels:
        row.append([
            Paragraph(val, ParagraphStyle('mv', fontName='Helvetica-Bold',
                fontSize=22, textColor=col, leading=26, alignment=TA_CENTER)),
            Paragraph(lbl, ParagraphStyle('ml2', fontSize=7.5, textColor=MUTED,
                leading=10, alignment=TA_CENTER))
        ])
    return Table(
        [[[r[0], r[1]] for r in row]],
        colWidths=[w]*4,
        style=TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), LIGHT),
            ('BOX', (0,0), (-1,-1), 0.5, CREAM),
            ('LINEBEFORE', (1,0), (-1,-1), 0.5, CREAM),
            ('TOPPADDING', (0,0), (-1,-1), 12),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ])
    )

print('Helper functions ready.')
# ── CELL 9: Build the PRD story ──────────────────────────────────────────────

story = []

# ═══════════════════════════════════════════════════════════════════
# PAGE 1: COVER
# ═══════════════════════════════════════════════════════════════════
story += [
    sp(60),
    Paragraph('PRODUCT REQUIREMENTS DOCUMENT', ParagraphStyle(
        'eyebrow', fontName='Helvetica-Bold', fontSize=10,
        textColor=ACCENT, leading=14, spaceAfter=10)),
    Paragraph('Olist Customer Churn Intelligence', styles['cover_title']),
    Paragraph('Data → Diagnosis → Roadmap', styles['cover_sub']),
    Paragraph(
        'A 5-stage analysis of customer acquisition, retention, and churn '
        'across 100,000+ orders · Olist Brazilian E-commerce Dataset',
        styles['cover_tag']),
    sp(30),
    hr(ACCENT, 2),
    sp(16),
]

# Key stats row
n_customers = len(rfm)
n_churned   = len(rfm[rfm['segment']=='Churned'])
pct_churned = n_churned / n_customers * 100
total_rev   = rfm['monetary'].sum()

story.append(four_metrics([
    (f'{n_customers:,}', 'Total Unique Customers', BLUE),
    (f'{pct_churned:.0f}%', 'Churned or At Risk', ACCENT),
    (f'R${total_opp/1e6:.1f}M', 'Revenue Opportunity', GREEN),
    (f'{len(ret_matrix)}', 'Cohorts Analysed', MUTED),
]))

story += [
    sp(30),
    Paragraph(f'Prepared: {datetime.date.today().strftime("%B %Y")}', styles['small']),
    Paragraph('Dataset: Olist Brazilian E-commerce Public Dataset (Kaggle)', styles['small']),
    Paragraph('Methodology: RFM Segmentation · Cohort Retention · Hypothesis Testing · Evidence Matrix', styles['small']),
    PageBreak(),
]

# ═══════════════════════════════════════════════════════════════════
# PAGE 2: EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════
story.append(section_header('Executive Summary'))
story.append(Paragraph(
    'Olist has a customer retention problem that is structural, measurable, and reversible. '
    'This analysis of 100,000+ orders across two years of transaction data reveals that the '
    'majority of customers purchase once and never return — and that this pattern is worsening '
    'for more recently acquired cohorts.',
    styles['body']))
story.append(sp(4))

story.append(callout_box(
    'The core finding',
    'Later customer cohorts retain at roughly half the rate of earlier cohorts at the '
    'three-month mark. This is not random variance — it is structural deterioration, '
    'visible in both the cohort heatmap and the RFM segment distribution. '
    'Four evidence-backed hypotheses explain why, and eight product interventions address them.',
    ACCENT))

story.append(sp(8))
story.append(Paragraph('What Was Done', styles['subsection']))
story.append(Paragraph(
    'Stage 1 (EDA): Established data quality, identified the customer_unique_id '
    'vs customer_id distinction critical for accurate customer-level analysis, '
    'and characterised baseline order and revenue distributions.',
    styles['body']))
story.append(Paragraph(
    'Stage 2 (RFM): Segmented all customers into six behavioural groups using '
    'quintile-scored Recency, Frequency, and Monetary metrics. Quantified '
    f'R${total_opp:,.0f} in total revenue opportunity (30% conversion assumption).',
    styles['body']))
story.append(Paragraph(
    'Stage 3 (Cohort Analysis): Built the monthly retention heatmap and identified '
    'the retention cliff — the exact month where the steepest average drop-off occurs. '
    'Confirmed that later cohorts have a structurally higher proportion of At Risk '
    'and Churned customers.',
    styles['body']))
story.append(Paragraph(
    'Stage 4 (Diagnosis): Tested four causal hypotheses using statistical methods '
    '(point-biserial correlation, linear regression, inter-order gap analysis, '
    'category repeat-rate analysis). Built a scored evidence matrix ranking '
    'interventions by evidence strength × business impact × reversibility.',
    styles['body']))
story.append(Paragraph(
    'Stage 5 (Roadmap): Translated evidence into a 3-phase product roadmap with '
    'defined success metrics, A/B test designs, and revenue justification per feature.',
    styles['body']))
story.append(PageBreak())

print('Pages 1-2 built.')
# ── CELL 10: PRD pages 3-5 — segments, cohorts, evidence ─────────────────────

# ═══════════════════════════════════════════════════════════════════
# PAGE 3: RFM SEGMENTS
# ═══════════════════════════════════════════════════════════════════
story.append(section_header('Customer Segmentation (RFM)'))
story.append(Paragraph(
    'Customers were segmented into six groups using quintile-scored Recency, Frequency, '
    'and Monetary values. Quintile scoring was chosen over fixed thresholds because it '
    'is data-relative, produces equal-sized scoring bands, and is portable across datasets.',
    styles['body']))
story.append(sp(8))

# Segment summary table
seg_summary = rfm.groupby('segment').agg(
    customers=('customer_unique_id','count'),
    avg_recency=('recency','mean'),
    avg_frequency=('frequency','mean'),
    avg_ltv=('monetary','mean'),
    total_revenue=('monetary','sum')
).reindex([s for s in SEG_ORDER if s in rfm['segment'].unique()]).reset_index()

tbl_data = [[
    Paragraph('Segment', styles['tbl_h']),
    Paragraph('Customers', styles['tbl_h']),
    Paragraph('Avg Recency', styles['tbl_h']),
    Paragraph('Avg Orders', styles['tbl_h']),
    Paragraph('Avg LTV', styles['tbl_h']),
    Paragraph('Revenue Share', styles['tbl_h']),
    Paragraph('Priority Action', styles['tbl_h']),
]]

ACTIONS = {
    'Champions':           'Reward + referral',
    'Loyal':               'Category expansion',
    'Potential Loyalists': 'Habit formation',
    'New Customers':       'Day-7 NPS + re-engage',
    'At Risk':             'Win-back sequence',
    'Churned':             'Exit survey + selective',
}
total_rev_all = seg_summary['total_revenue'].sum()
seg_bg_map = {
    'Champions': colors.HexColor('#E8F5EE'),
    'Loyal': colors.HexColor('#E8EDF5'),
    'Potential Loyalists': colors.HexColor('#F5F0E8'),
    'New Customers': colors.HexColor('#FDF8EE'),
    'At Risk': colors.HexColor('#FDF2EA'),
    'Churned': colors.HexColor('#FBF0EE'),
}

tbl_styles = [
    ('BACKGROUND', (0,0), (-1,0), INK),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, LIGHT]),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,-1), 8.5),
    ('TOPPADDING', (0,0), (-1,-1), 6),
    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ('LEFTPADDING', (0,0), (-1,-1), 6),
    ('GRID', (0,0), (-1,-1), 0.3, CREAM),
    ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
]

for i, row in seg_summary.iterrows():
    seg = row['segment']
    rev_share = row['total_revenue'] / total_rev_all * 100
    tbl_data.append([
        Paragraph(f'<b>{seg}</b>', styles['tbl_b_bold']),
        Paragraph(f"{row['customers']:,}", styles['tbl_b']),
        Paragraph(f"{row['avg_recency']:.0f}d", styles['tbl_b']),
        Paragraph(f"{row['avg_frequency']:.1f}", styles['tbl_b']),
        Paragraph(f"R${row['avg_ltv']:,.0f}", styles['tbl_b']),
        Paragraph(f"{rev_share:.1f}%", styles['tbl_b']),
        Paragraph(ACTIONS.get(seg,'—'), styles['tbl_b']),
    ])

story.append(Table(tbl_data,
    colWidths=[3.2*cm, 2*cm, 2.2*cm, 1.8*cm, 2.2*cm, 2.2*cm, 3.5*cm],
    style=TableStyle(tbl_styles)))

story.append(sp(6))
story.append(Paragraph(
    f'Total revenue opportunity (30% conversion, moving each segment toward Champions LTV): '
    f'<b>R${total_opp:,.0f}</b>',
    styles['body']))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════
# PAGE 4: COHORT RETENTION
# ═══════════════════════════════════════════════════════════════════
story.append(section_header('Cohort Retention Analysis'))
story.append(Paragraph(
    'Customers were grouped by their acquisition month (cohort). Retention rate tracks '
    'what percentage of each cohort was still active N months after their first purchase.',
    styles['body']))
story.append(sp(6))

# Embed the heatmap image if it was saved
import os
if os.path.exists('output/10_cohort_heatmap.png'):
    story.append(RLImage('output/10_cohort_heatmap.png', width=CONTENT_W, height=10*cm))
    story.append(Paragraph('Figure 1: Monthly cohort retention heatmap. Each cell shows '
                            'the % of the original cohort still active at that month offset.',
                            styles['small']))

story.append(sp(8))
story.append(callout_box(
    'Key cohort findings',
    '1. A retention cliff exists at M+[your number] — the largest single-month drop across all cohorts. '
    'Re-engagement must fire BEFORE this month.\n'
    '2. Later cohorts consistently retain at lower rates than earlier cohorts at every month offset — '
    'confirming structural deterioration, not random variance.\n'
    '3. The gap between best and worst cohort at M+3 is [your number] percentage points — '
    'the headline metric for platform health.',
    BLUE))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════
# PAGE 5: EVIDENCE MATRIX
# ═══════════════════════════════════════════════════════════════════
story.append(section_header('Churn Diagnosis — Evidence Matrix'))
story.append(Paragraph(
    'Four causal hypotheses were tested using statistical methods. Each is scored on '
    'three dimensions: evidence strength (statistical significance + effect size), '
    'business impact (customers affected × revenue at stake), and reversibility '
    '(how quickly a product intervention can address it).',
    styles['body']))
story.append(sp(6))

# Evidence matrix table
ev_data = [[
    Paragraph('Hypothesis', styles['tbl_h']),
    Paragraph('Key Finding', styles['tbl_h']),
    Paragraph('Evid.\n(1-5)', styles['tbl_h']),
    Paragraph('Impact\n(1-5)', styles['tbl_h']),
    Paragraph('Rev.\n(1-5)', styles['tbl_h']),
    Paragraph('Score\n(/125)', styles['tbl_h']),
    Paragraph('Intervention', styles['tbl_h']),
]]

for i, row in evidence.iterrows():
    score = row['Priority_Score']
    ev_data.append([
        Paragraph(f"<b>{row['Hypothesis'].split(':')[0]}</b>\n{row['Hypothesis'].split(':')[1].strip()[:40]}",
                  styles['tbl_b']),
        Paragraph(str(row['Key Finding'])[:65], styles['tbl_b']),
        Paragraph(f"{row['Evidence_Strength']}/5", styles['tbl_b']),
        Paragraph(f"{row['Business_Impact']}/5", styles['tbl_b']),
        Paragraph(f"{row['Reversibility']}/5", styles['tbl_b']),
        Paragraph(f'<b>{score}</b>', styles['tbl_b_bold']),
        Paragraph(str(row['Intervention'])[:50], styles['tbl_b']),
    ])

ev_tbl_style = TableStyle([
    ('BACKGROUND', (0,0), (-1,0), INK),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, LIGHT]),
    ('FONTSIZE', (0,0), (-1,-1), 8),
    ('TOPPADDING', (0,0), (-1,-1), 5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ('LEFTPADDING', (0,0), (-1,-1), 5),
    ('GRID', (0,0), (-1,-1), 0.3, CREAM),
    ('ALIGN', (2,0), (5,-1), 'CENTER'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
])
story.append(Table(ev_data,
    colWidths=[3.5*cm, 3.8*cm, 1.3*cm, 1.3*cm, 1.1*cm, 1.4*cm, 4.2*cm],
    style=ev_tbl_style))
story.append(sp(6))
story.append(Paragraph(
    'Priority Score = Evidence × Impact × Reversibility (max 125). '
    'Features with score ≥75 are Phase 1 candidates.',
    styles['small']))
story.append(PageBreak())

print('Pages 3-5 built.')
# ── CELL 11: PRD pages 6-7 — roadmap + A/B tests ─────────────────────────────

# ═══════════════════════════════════════════════════════════════════
# PAGE 6: PRODUCT ROADMAP
# ═══════════════════════════════════════════════════════════════════
story.append(section_header('Product Roadmap — 3 Phases'))
story.append(Paragraph(
    'Features are ordered by priority score from the evidence matrix. '
    'Phase 1 features have the highest evidence, widest customer impact, '
    'and can ship in days-to-weeks (email/comms). '
    'Phase 3 features require structural changes to the platform.',
    styles['body']))
story.append(sp(8))

phase_groups = [('Phase 1\n(0–60 days)', PHASE_COLORS[0]),
                ('Phase 2\n(60–180 days)', PHASE_COLORS[1]),
                ('Phase 3\n(180+ days)', PHASE_COLORS[2])]

for phase_label, phase_color in phase_groups:
    phase_rows = roadmap[roadmap['Phase'] == phase_label]
    if len(phase_rows) == 0:
        continue

    # Phase header bar
    story.append(Table(
        [[Paragraph(phase_label.replace('\n',' '), styles['phase_h'])]],
        colWidths=[CONTENT_W],
        style=TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), phase_color),
            ('TOPPADDING', (0,0), (-1,-1), 7),
            ('BOTTOMPADDING', (0,0), (-1,-1), 7),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
        ])
    ))

    for _, feat in phase_rows.iterrows():
        feat_data = [
            [Paragraph(f"<b>{feat['Feature']}</b>", styles['tbl_b_bold']),
             Paragraph(f"Hypothesis: <b>{feat['Hypothesis']}</b>", styles['tbl_b']),
             Paragraph(f"Segment: {feat['Segment']}", styles['tbl_b']),
             Paragraph(f"A/B: {feat['A/B Testable']}", styles['tbl_b'])],
            [Paragraph(f"Mechanism: {feat['Mechanism']}", styles['tbl_b']),
             Paragraph(f"Success metric: <b>{feat['Success Metric']}</b>",
                       ParagraphStyle('sm2', parent=styles['tbl_b'], textColor=phase_color)),
             '', ''],
        ]
        story.append(Table(feat_data,
            colWidths=[4.5*cm, 3.5*cm, 4*cm, 2.5*cm],
            style=TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), WHITE),
                ('FONTSIZE', (0,0), (-1,-1), 8.5),
                ('TOPPADDING', (0,0), (-1,-1), 5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ('LEFTPADDING', (0,0), (-1,-1), 7),
                ('LINEBEFORE', (0,0), (0,-1), 3, phase_color),
                ('LINEBELOW', (0,-1), (-1,-1), 0.3, CREAM),
                ('SPAN', (0,1), (3,1)),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ])
        ))
    story.append(sp(8))

story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════
# PAGE 7: A/B TESTS
# ═══════════════════════════════════════════════════════════════════
story.append(section_header('A/B Test Plans — Top 2 Interventions'))

for test in ab_tests:
    story.append(Paragraph(test['name'], styles['subsection']))
    test_data = [
        ['Hypothesis',      test['hypothesis'][:90] + '...'],
        ['Control',         test['control']],
        ['Treatment',       test['treatment'][:90] + '...'],
        ['Segment',         test['segment']],
        ['Primary Metric',  test['primary_metric']],
        ['Secondary',       ', '.join(test['secondary_metrics'])],
        ['Min. Detectable\nEffect', test['mde']],
        ['Sample/Variant',  test['sample_per_variant']],
        ['Duration',        test['duration']],
        ['Guardrail',       test['guardrail']],
        ['Winner Criteria', test['winner_criteria']],
    ]
    story.append(Table(
        [[Paragraph(f'<b>{r[0]}</b>', styles['tbl_b_bold']),
          Paragraph(r[1], styles['tbl_b'])] for r in test_data],
        colWidths=[3.8*cm, CONTENT_W - 3.8*cm],
        style=TableStyle([
            ('ROWBACKGROUNDS', (0,0), (-1,-1), [WHITE, LIGHT]),
            ('FONTSIZE', (0,0), (-1,-1), 8.5),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 7),
            ('GRID', (0,0), (-1,-1), 0.3, CREAM),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ])
    ))
    story.append(sp(12))

story.append(PageBreak())

print('Pages 6-7 built.')
# ── CELL 12: PRD page 8 — success metrics + compile PDF ─────────────────────

# ═══════════════════════════════════════════════════════════════════
# PAGE 8: SUCCESS METRICS FRAMEWORK
# ═══════════════════════════════════════════════════════════════════
story.append(section_header('Success Metrics Framework'))
story.append(Paragraph(
    'These metrics define what "working" looks like across each level of the funnel. '
    'North Star is set at the platform level; input metrics are owned by the product team; '
    'leading indicators are monitored weekly to catch problems early.',
    styles['body']))
story.append(sp(8))

metrics_data = [
    ['North Star', 'M+3 Cohort Retention Rate', '≥ early cohort baseline', 'Monthly', 'All interventions'],
    ['North Star', 'Revenue per Acquired Customer (90d)', '+15% vs baseline cohort', 'Monthly', 'All'],
    ['Input', '30-day Repeat Purchase Rate (new customers)', '+3pp within 60d of launch', 'Weekly', 'H1 + H3'],
    ['Input', 'Post-Delivery NPS Score', '>4.0 avg for new customers', 'Weekly', 'H1'],
    ['Input', 'At-Risk Reactivation Rate', '>8% of At Risk segment', 'Monthly', 'Win-back'],
    ['Input', 'Average First-Order Review Score (new cohorts)', '≥4.2 stars', 'Monthly', 'H2'],
    ['Leading', 'Email Open Rate (re-engagement)', '>22%', 'Weekly', 'H3'],
    ['Leading', 'Day-7 Survey Response Rate', '>15%', 'Weekly', 'H1'],
    ['Leading', 'Time Between 1st and 2nd Order (median)', 'Decreasing trend', 'Monthly', 'H3'],
    ['Guardrail', 'Email Unsubscribe Rate', '<2%', 'Weekly', 'All email'],
    ['Guardrail', 'Voucher Cost as % of Recovered Revenue', '<50%', 'Monthly', 'Win-back'],
    ['Guardrail', 'Overall Refund Rate', 'No increase vs baseline', 'Weekly', 'H1'],
]

MET_COLORS = {'North Star': ACCENT, 'Input': BLUE, 'Leading': GREEN, 'Guardrail': GOLD}

met_header = [
    Paragraph('Level', styles['tbl_h']),
    Paragraph('Metric', styles['tbl_h']),
    Paragraph('Target', styles['tbl_h']),
    Paragraph('Cadence', styles['tbl_h']),
    Paragraph('Hypothesis', styles['tbl_h']),
]
met_rows = [met_header]
for row in metrics_data:
    level_color = MET_COLORS.get(row[0], MUTED)
    met_rows.append([
        Paragraph(f'<b>{row[0]}</b>',
                  ParagraphStyle('lv', fontName='Helvetica-Bold', fontSize=8,
                                 textColor=level_color, leading=11)),
        Paragraph(row[1], styles['tbl_b']),
        Paragraph(f'<b>{row[2]}</b>', styles['tbl_b_bold']),
        Paragraph(row[3], styles['tbl_b']),
        Paragraph(row[4], styles['tbl_b']),
    ])

met_style = TableStyle([
    ('BACKGROUND', (0,0), (-1,0), INK),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, LIGHT]),
    ('FONTSIZE', (0,0), (-1,-1), 8.5),
    ('TOPPADDING', (0,0), (-1,-1), 5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ('LEFTPADDING', (0,0), (-1,-1), 6),
    ('GRID', (0,0), (-1,-1), 0.3, CREAM),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
])
story.append(Table(met_rows,
    colWidths=[2.2*cm, 5.5*cm, 3.5*cm, 2*cm, 3.4*cm],
    style=met_style))

story += [
    sp(16),
    hr(ACCENT),
    Paragraph(
        'This document was produced as a portfolio project using the Olist Brazilian E-commerce '
        'public dataset. All analysis was conducted in Python (pandas, scipy, matplotlib, seaborn). '
        f'Generated {datetime.date.today().strftime("%B %d, %Y")}.',
        styles['small']),
]

# ── COMPILE AND SAVE ──
prd_path = 'output/Olist_Churn_PRD.pdf'
doc = SimpleDocTemplate(
    prd_path,
    pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN, bottomMargin=MARGIN,
    title='Olist Customer Churn Intelligence — PRD',
    author='Portfolio Project',
)
doc.build(story)

import os
size_kb = os.path.getsize(prd_path) / 1024
print(f'\n PRD saved → {prd_path}')
print(f'   File size: {size_kb:.0f} KB')
print(f'   Pages: ~8')
