# -*- coding: utf-8 -*-
"""
Plot dist_from_wall grouped by edge/centre zone.
Aesthetics match the paired-boxplot reference style (purple palette, jittered
scatter, stats annotation inside plot area).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

# ── USER INPUT ────────────────────────────────────────────────────────────────
excel_file = r'C:/Runita/NMR/analysis/TAC6/Data/Open/Open_BoundaryAnalysis_v2.xlsx'
output_png = r'C:/Runita/NMR/analysis/TAC6/Data/Open/dist_from_wall_EdgeCentre.png'
# ─────────────────────────────────────────────────────────────────────────────

mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'DejaVu Sans'],
    'font.size':         11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})

# ── Colors (match reference purple palette) ───────────────────────────────────
COLOR_EDGE   = '#9B87BE'   # light purple  (left group)
COLOR_CENTRE = '#6B7AAD'   # blue-slate    (right group)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_excel(excel_file, sheet_name=0, header=0)

col_dist = df.columns[21]   # dist_from_wall_cm  (column V)
col_zone = df.columns[22]   # zone               (column W)

dist = df[col_dist].dropna()
zone = df[col_zone].loc[dist.index]

# Filter to place-cells only if a 'place_cell' column exists
if 'place_cell' in df.columns:
    pc_mask = df['place_cell'].loc[dist.index].eq(True)
    dist    = dist[pc_mask]
    zone    = zone[pc_mask]

edge_vals = np.asarray(dist[zone.str.lower().str.contains('edge', na=False)], dtype=float)
cent_vals = np.asarray(dist[zone.str.lower().str.contains('cent', na=False)], dtype=float)

# ── Statistics (Mann-Whitney U, independent groups) ───────────────────────────
u_stat, pval = stats.mannwhitneyu(edge_vals, cent_vals, alternative='two-sided')
n_edge = len(edge_vals)
n_cent = len(cent_vals)

if pval < 0.001:
    sig_tag = '***'
    p_label = 'p<0.001'
elif pval < 0.01:
    sig_tag = '**'
    p_label = f'p={pval:.3f}'
elif pval < 0.05:
    sig_tag = '*'
    p_label = f'p={pval:.3f}'
else:
    sig_tag = 'ns'
    p_label = f'p={pval:.3f}'

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 6))

groups      = [edge_vals, cent_vals]
group_names = ['Edge', 'Centre']
colors      = [COLOR_EDGE, COLOR_CENTRE]

bp = ax.boxplot(
    groups,
    positions=[1, 2],
    patch_artist=True,
    widths=0.45,
    showfliers=False,
    medianprops=dict(color='black', linewidth=2.5),
    whiskerprops=dict(linewidth=1.5, color='black'),
    capprops=dict(linewidth=1.5, color='black'),
    boxprops=dict(linewidth=1.5),
)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

# Jittered individual data points (same style as reference)
rng = np.random.default_rng(42)
for i, (vals, color) in enumerate(zip(groups, colors), start=1):
    jitter = rng.uniform(-0.13, 0.13, size=len(vals))
    ax.scatter(i + jitter, vals,
               color=color, alpha=0.55, s=20, zorder=3, edgecolors='none')

# Stats annotation inside plot (top-left, matching reference text style)
stat_text = f'Mann-Whitney {sig_tag}: {p_label}'
n_text    = f'n = {n_edge} edge, {n_cent} centre'
ax.text(0.05, 0.97, stat_text,
        transform=ax.transAxes, ha='left', va='top',
        fontsize=8.5, color='#444444')
ax.text(0.05, 0.91, n_text,
        transform=ax.transAxes, ha='left', va='top',
        fontsize=8.5, color='#444444')

ax.set_xticks([1, 2])
ax.set_xticklabels(group_names, fontsize=12)
ax.set_ylabel('Distance from wall (cm)', fontsize=12)
ax.set_title('Place cell distance from wall\nby zone', fontsize=13, pad=10)
ax.set_xlim(0.4, 2.6)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.show()

print(f'\nMann-Whitney U = {u_stat:.1f},  {p_label}  ({sig_tag})')
print(f'n edge = {n_edge},  n centre = {n_cent}')
