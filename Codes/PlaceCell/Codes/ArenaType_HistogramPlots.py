# -*- coding: utf-8 -*-
"""
Histogram plots of place-cell characterization metrics, split by arena type
(Open / Linear / Circle). Reuses the loading, place-cell filtering, and arena
color scheme from ArenaType_StatsComparison.py, so it reads the SAME workbook
(INPUT_EXCEL there) and analyzes the SAME place-cell population.

For every metric below (all restricted to place_cell == True), one histogram
figure is produced with the arena types overlaid on shared bins:

  Sheet 'Full':
    - peak_fr, sir, sparsity, coherence, stability_score

  Sheet 'PlaceFields' (aggregated per cell):
    - n_fields, total_area_cm2, pct_area

The two speed-score metrics (speed_score, speed_score_td) are plotted
separately, one figure each: only place cells that passed BOTH shuffle-
testing procedures (speed_modulated_final == True, i.e. speed_modulated_shuffle
AND speed_modulated_td) are included. Each figure has one panel per arena
type, and within a panel cells are split into p-type (green) and n-type
(red) speed cells using their p_speed / n_speed classification.

Output: PNGs under <PLOTS_DIR>/Histograms/, where PLOTS_DIR is the one
defined in ArenaType_StatsComparison.py.
"""

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ArenaType_StatsComparison import (
    INPUT_EXCEL, PLOTS_DIR, ARENA_TYPES, ARENA_COLORS, _DEFAULT_COLOR,
    FULL_METRICS, FIELD_METRICS, _to_tri_bool,
    load_sheets, build_full_metrics_df, build_field_metrics_df,
)

# ── Parameters ──────────────────────────────────────────────────────────────

HIST_DIR = os.path.join(PLOTS_DIR, 'Histograms')
N_BINS   = 15

SPEED_METRICS = {
    'speed_score':    'Speed score (binning procedure)',
    'speed_score_td': 'Speed score (instantaneous procedure)',
}
HIST_FULL_METRICS = {k: v for k, v in FULL_METRICS.items() if k not in SPEED_METRICS}

# Explicit p-type/n-type speed-cell colors (green/red), per request -- these
# are independent of the per-arena ARENA_COLORS palette used elsewhere.
P_COLOR = {'face': '#4CAF50', 'edge': '#1B5E20'}  # p-type (positive) speed cells
N_COLOR = {'face': '#E53935', 'edge': '#7A0C0C'}  # n-type (negative) speed cells


# ── Helpers ───────────────────────────────────────────────────────────────

def _style_axis(ax):
    ax.set_facecolor('#FCFCFB')
    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, color='#E1E0D9', zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', colors='#52514E', labelsize=9)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color('#C3C2B7')


def plot_histogram_by_arena(df: pd.DataFrame, metric_col: str, metric_label: str,
                             out_path: str):
    """One figure per metric: histograms overlaid for each arena type present,
    on shared bins so the three distributions are directly comparable."""
    values = {}
    for arena in ARENA_TYPES:
        vals = pd.to_numeric(df.loc[df['arena_type'] == arena, metric_col],
                              errors='coerce').dropna().to_numpy(dtype=float)
        if len(vals):
            values[arena] = vals
    if not values:
        return

    all_vals = np.concatenate(list(values.values()))
    bins = np.histogram_bin_edges(all_vals, bins=N_BINS)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor('white')
    _style_axis(ax)

    for arena, vals in values.items():
        c = ARENA_COLORS.get(arena, _DEFAULT_COLOR)
        ax.hist(vals, bins=bins, histtype='stepfilled', facecolor=c['face'],
                edgecolor='none', alpha=0.55, label=f'{arena} (n={len(vals)})',
                zorder=2)
        ax.hist(vals, bins=bins, histtype='step', edgecolor=c['edge'],
                linewidth=1.5, zorder=3)

    ax.set_xlabel(metric_label, fontsize=10)
    ax.set_ylabel('Number of cells', fontsize=10)
    ax.set_title(metric_label, fontsize=11, fontweight='bold', pad=10)
    ax.legend(loc='best', fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def _speed_type_masks(sub: pd.DataFrame):
    """Boolean (p_mask, n_mask) numpy arrays for a sub-dataframe, from its
    p_speed / n_speed columns."""
    p_mask = sub['p_speed'].apply(_to_tri_bool) == True   # noqa: E712
    n_mask = sub['n_speed'].apply(_to_tri_bool) == True   # noqa: E712
    return p_mask.to_numpy(), n_mask.to_numpy()


def plot_speed_histograms_by_arena(df: pd.DataFrame, metric_col: str,
                                    metric_label: str, out_path: str):
    """Speed-score histogram restricted to cells that passed BOTH shuffle-
    testing procedures, with one panel per arena type and p-type (green) /
    n-type (red) speed cells stacked within each panel."""
    if 'speed_modulated_final' in df.columns:
        passed = df['speed_modulated_final'].apply(_to_tri_bool) == True  # noqa: E712
    else:
        passed = ((df['speed_modulated_shuffle'].apply(_to_tri_bool) == True) &   # noqa: E712
                  (df['speed_modulated_td'].apply(_to_tri_bool) == True))          # noqa: E712

    sub_all = df.loc[passed].copy()
    sub_all[metric_col] = pd.to_numeric(sub_all[metric_col], errors='coerce')
    sub_all = sub_all.dropna(subset=[metric_col])

    present = [a for a in ARENA_TYPES if (sub_all['arena_type'] == a).any()]
    if not present:
        print(f'  No cells passed both shuffle tests for {metric_label}; skipping plot.')
        return

    bins = np.histogram_bin_edges(sub_all[metric_col].to_numpy(dtype=float), bins=N_BINS)

    fig, axes = plt.subplots(1, len(present), figsize=(4.3 * len(present), 4.5),
                              sharey=True)
    fig.patch.set_facecolor('white')
    axes = np.atleast_1d(axes)

    for ax, arena in zip(axes, present):
        sub = sub_all.loc[sub_all['arena_type'] == arena]
        p_mask, n_mask = _speed_type_masks(sub)
        p_vals = sub.loc[p_mask, metric_col].to_numpy(dtype=float)
        n_vals = sub.loc[n_mask, metric_col].to_numpy(dtype=float)

        _style_axis(ax)
        ax.hist([p_vals, n_vals], bins=bins, stacked=True,
                color=[P_COLOR['face'], N_COLOR['face']],
                edgecolor=[P_COLOR['edge'], N_COLOR['edge']],
                linewidth=1.2, zorder=2)
        ax.set_title(f'{arena}\n(n={len(sub)}: {len(p_vals)} p-type, {len(n_vals)} n-type)',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel(metric_label, fontsize=9)

    axes[0].set_ylabel('Number of cells', fontsize=10)
    handles = [Patch(facecolor=P_COLOR['face'], edgecolor=P_COLOR['edge'], label='p-type speed cell'),
               Patch(facecolor=N_COLOR['face'], edgecolor=N_COLOR['edge'], label='n-type speed cell')]
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(f'{metric_label}\n(cells passed both shuffle-testing procedures)',
                 fontsize=11, fontweight='bold', y=1.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    print(f'Loading {INPUT_EXCEL} ...')
    full, fields = load_sheets()

    full_df  = build_full_metrics_df(full)
    field_df = build_field_metrics_df(full, fields)

    os.makedirs(HIST_DIR, exist_ok=True)

    print('\nPlotting Full-sheet metric histograms by arena type...')
    for col, label in HIST_FULL_METRICS.items():
        plot_histogram_by_arena(full_df, col, label,
                                 os.path.join(HIST_DIR, f'Hist_Full_{col}.png'))

    print('Plotting PlaceFields-sheet metric histograms by arena type...')
    for col, label in FIELD_METRICS.items():
        plot_histogram_by_arena(field_df, col, label,
                                 os.path.join(HIST_DIR, f'Hist_Fields_{col}.png'))

    print('Plotting speed-score histograms (p-type/n-type, both shuffle tests passed)...')
    for col, label in SPEED_METRICS.items():
        plot_speed_histograms_by_arena(full_df, col, label,
                                        os.path.join(HIST_DIR, f'Hist_Speed_{col}.png'))

    print(f'\nDone. Histogram plots saved to {HIST_DIR}')
