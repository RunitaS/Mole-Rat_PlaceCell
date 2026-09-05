# -*- coding: utf-8 -*-
"""
Plot dist_from_wall grouped by edge/centre zone.
Aesthetics match the paired-boxplot reference style (purple palette, jittered
scatter, stats annotation inside plot area).
Also plots the proportion of edge vs centre cells for Open Field and Linear Track.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

# ── USER INPUT ────────────────────────────────────────────────────────────────
excel_file_of = r'C:/Runita/NMR/analysis/TAC6/Data/Open/Open_BoundaryAnalysis_v2.xlsx'
excel_file_lt = r'C:/Runita/NMR/analysis/TAC6/Data/Linear/Linear_BoundaryAnalysis.xlsx'
output_png        = r'C:/Runita/NMR/analysis/TAC6/Data/Open/dist_from_wall_EdgeCentre.png'
output_png_prop   = r'C:/Runita/NMR/analysis/TAC6/Data/Open/proportion_EdgeCentre_arenas.png'
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

# ── Colors ────────────────────────────────────────────────────────────────────
COLOR_EDGE   = '#9B87BE'   # light purple
COLOR_CENTRE = '#6B7AAD'   # blue-slate


def load_zone_data(excel_file):
    """Load dist_from_wall and zone columns from an Excel file, filtered to place cells."""
    df = pd.read_excel(excel_file, sheet_name=0, header=0)
    col_dist = df.columns[21]   # dist_from_wall_cm  (column V)
    col_zone = df.columns[22]   # zone               (column W)

    dist = df[col_dist].dropna()
    zone = df[col_zone].loc[dist.index]

    if 'place_cell' in df.columns:
        pc_mask = df['place_cell'].loc[dist.index].eq(True)
        dist    = dist[pc_mask]
        zone    = zone[pc_mask]

    edge_vals = np.asarray(dist[zone.str.lower().str.contains('edge', na=False)], dtype=float)
    cent_vals = np.asarray(dist[zone.str.lower().str.contains('cent', na=False)], dtype=float)
    return edge_vals, cent_vals


# ── Load both datasets ────────────────────────────────────────────────────────
edge_of, cent_of = load_zone_data(excel_file_of)
edge_lt, cent_lt = load_zone_data(excel_file_lt)

# ── Statistics for Open Field (existing plot) ─────────────────────────────────
u_stat, pval = stats.mannwhitneyu(edge_of, cent_of, alternative='two-sided')
n_edge = len(edge_of)
n_cent = len(cent_of)

if pval < 0.001:
    sig_tag = '***'; p_label = 'p<0.001'
elif pval < 0.01:
    sig_tag = '**';  p_label = f'p={pval:.3f}'
elif pval < 0.05:
    sig_tag = '*';   p_label = f'p={pval:.3f}'
else:
    sig_tag = 'ns';  p_label = f'p={pval:.3f}'

# ── Figure 1: dist_from_wall boxplot (Open Field) ─────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 6))

groups      = [edge_of, cent_of]
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

rng = np.random.default_rng(42)
for i, (vals, color) in enumerate(zip(groups, colors), start=1):
    jitter = rng.uniform(-0.13, 0.13, size=len(vals))
    ax.scatter(i + jitter, vals,
               color=color, alpha=0.55, s=20, zorder=3, edgecolors='none')

stat_text = f'Mann-Whitney {sig_tag}: {p_label}'
n_text    = f'n = {n_edge} edge, {n_cent} centre'
ax.text(0.05, 0.97, stat_text, transform=ax.transAxes, ha='left', va='top',
        fontsize=8.5, color='#444444')
ax.text(0.05, 0.91, n_text,    transform=ax.transAxes, ha='left', va='top',
        fontsize=8.5, color='#444444')

ax.set_xticks([1, 2])
ax.set_xticklabels(group_names, fontsize=12)
ax.set_ylabel('Distance from wall (cm)', fontsize=12)
ax.set_title('Place cell distance from wall\nby zone (Open Field)', fontsize=13, pad=10)
ax.set_xlim(0.4, 2.6)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.show()

print(f'\nOpen Field — Mann-Whitney U = {u_stat:.1f},  {p_label}  ({sig_tag})')
print(f'n edge = {n_edge},  n centre = {n_cent}')

# ── Figure 2: Proportion of edge / centre cells across arenas ─────────────────
def pct(edge, cent):
    total = len(edge) + len(cent)
    if total == 0:
        return 0.0, 0.0
    return 100 * len(edge) / total, 100 * len(cent) / total

pct_edge_of, pct_cent_of = pct(edge_of, cent_of)
pct_edge_lt, pct_cent_lt = pct(edge_lt, cent_lt)

arena_labels  = ['Open Field', 'Linear Track']
pct_edge_vals = [pct_edge_of, pct_edge_lt]
pct_cent_vals = [pct_cent_of, pct_cent_lt]

x      = np.arange(len(arena_labels))
width  = 0.5

fig2, ax2 = plt.subplots(figsize=(5.5, 5))

# Stacked: Edge on bottom, Centre on top
bars_edge = ax2.bar(x, pct_edge_vals, width,
                    label='Edge', color=COLOR_EDGE, alpha=0.85, linewidth=1.2,
                    edgecolor='black')
bars_cent = ax2.bar(x, pct_cent_vals, width,
                    label='Centre', color=COLOR_CENTRE, alpha=0.85, linewidth=1.2,
                    edgecolor='black', bottom=pct_edge_vals)

# Labels centred within each segment
for bar, h_edge, h_cent in zip(bars_edge, pct_edge_vals, pct_cent_vals):
    mid_x = bar.get_x() + bar.get_width() / 2
    if h_edge > 4:
        ax2.text(mid_x, h_edge / 2, f'{h_edge:.1f}%',
                 ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    if h_cent > 4:
        ax2.text(mid_x, h_edge + h_cent / 2, f'{h_cent:.1f}%',
                 ha='center', va='center', fontsize=9, color='white', fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels(arena_labels, fontsize=12)
ax2.set_ylabel('Proportion of cells (%)', fontsize=12)
ax2.set_title('Edge vs Centre cell proportion\nacross arenas', fontsize=13, pad=10)
ax2.set_ylim(0, 100)
ax2.legend(frameon=False, fontsize=10)

n_text_of = f'OF: n={len(edge_of)+len(cent_of)}'
n_text_lt = f'LT: n={len(edge_lt)+len(cent_lt)}'
ax2.text(0.98, 0.97, f'{n_text_of}   {n_text_lt}',
         transform=ax2.transAxes, ha='right', va='top',
         fontsize=8.5, color='#444444')

plt.tight_layout()
plt.savefig(output_png_prop, dpi=300, bbox_inches='tight')
plt.show()

print(f'\nOpen Field    — Edge: {pct_edge_of:.1f}%,  Centre: {pct_cent_of:.1f}%'
      f'  (n={len(edge_of)+len(cent_of)})')
print(f'Linear Track  — Edge: {pct_edge_lt:.1f}%,  Centre: {pct_cent_lt:.1f}%'
      f'  (n={len(edge_lt)+len(cent_lt)})')
