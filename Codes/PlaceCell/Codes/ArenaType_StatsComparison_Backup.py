# -*- coding: utf-8 -*-
"""
Statistical comparison of place-cell characterization metrics across arena
types (Open / Linear / Circle), using the workbook produced by
PlaceCellChar_FieldDetect_Main_v3.py (the 'output_excel' file). Results are
appended back into that SAME workbook as additional sheets (Descriptives,
OmnibusTests, PostHoc, ...) rather than written to a separate file.

Arena type is extracted from the folder path stored in the 'session' column
(column A) of both the 'Full' and 'PlaceFields' sheets, e.g. a session of
'Fa1059\\Linear\\Day10\\1_0' is assigned arena type 'Linear'. Only rows with
place_cell == True are analyzed.

Metrics compared (all restricted to place_cell == True):

  Sheet 'Full':
    - peak_fr          Peak firing rate
    - sir              Spatial information score
    - sparsity         Sparsity
    - coherence        Coherence score
    - stability_score  Stability score
    - speed_score      Speed score (binning procedure)
    - speed_score_td   Speed score (instantaneous procedure)

  Sheet 'PlaceFields' (aggregated per cell):
    - n_fields         Number of place fields per cell
    - total_area_cm2   Area occupied by fields per cell (sum of field areas)
    - pct_area         Percentage area occupied by fields per cell (sum of
                        each field's pct_of_occupied_area)

For each metric, per-group descriptive statistics are computed, followed by
an omnibus 3-group test (one-way ANOVA if all groups pass a Shapiro-Wilk
normality check and a Levene equal-variance check, otherwise Kruskal-Wallis)
and pairwise post-hoc tests (t-test or Mann-Whitney U, matching the omnibus
choice) with Holm-Bonferroni correction across the 3 pairwise comparisons.

Parameters to edit (below):
    INPUT_EXCEL  = path to the output_excel file from
                   PlaceCellChar_FieldDetect_Main_v3.py -- also where these
                   statistics sheets get appended (in place)
    PLOTS_DIR    = folder to save comparison box plots to
"""

import os
import re
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Parameters ──────────────────────────────────────────────────────────────

INPUT_EXCEL = r'C:/Runita/NMR/analysis/ThesisStuff/All_TT_PlaceChar.xlsx'
PLOTS_DIR   = r'C:/Runita/NMR/analysis/ThesisStuff/ArenaType_StatsPlots'

ARENA_TYPES = ['Open', 'Linear', 'Circle']
ALPHA       = 0.05

FULL_METRICS = {
    'peak_fr':         'Peak firing rate',
    'sir':             'Spatial information score',
    'sparsity':        'Sparsity',
    'coherence':       'Coherence score',
    'stability_score': 'Stability score',
    'speed_score':     'Speed score (binning procedure)',
    'speed_score_td':  'Speed score (instantaneous procedure)',
}

FIELD_METRICS = {
    'n_fields':       'Number of place fields per cell',
    'total_area_cm2': 'Area occupied by fields per cell',
    'pct_area':       'Percentage area occupied by fields per cell',
}

# ── Helpers ───────────────────────────────────────────────────────────────

def _to_bool(x) -> bool:
    """Normalize place_cell values that may come back from Excel as bool,
    numpy bool, or string ('True'/'False')."""
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() == 'true'
    if pd.isna(x):
        return False
    return bool(x)


def _to_tri_bool(x):
    """Normalize a boolean-ish Excel cell (bootstrap_sig / speed_modulated_*
    / p_speed / n_speed) to True, False, or None (unknown/blank/NaN), since
    these columns may come back from Excel as bool, string, or a blank cell
    that pandas reads as NaN."""
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s == 'true':
            return True
        if s == 'false':
            return False
        return None
    if pd.isna(x):
        return None
    return bool(x)


def extract_arena_type(session_path: str):
    """Return 'Open', 'Linear', or 'Circle' if one of those appears as a
    path component of `session_path`, else None."""
    parts = re.split(r'[\\/]+', str(session_path))
    for part in parts:
        for arena in ARENA_TYPES:
            if part.strip().lower() == arena.lower():
                return arena
    return None


def load_sheets():
    full   = pd.read_excel(INPUT_EXCEL, sheet_name='Full')
    fields = pd.read_excel(INPUT_EXCEL, sheet_name='PlaceFields')
    return full, fields


def build_full_metrics_df(full: pd.DataFrame) -> pd.DataFrame:
    df = full[full['place_cell'].apply(_to_bool)].copy()
    df['arena_type'] = df['session'].apply(extract_arena_type)

    unknown = df[df['arena_type'].isna()]
    if len(unknown):
        print(f'WARNING: {len(unknown)} place cell(s) had a session path with no '
              f'recognizable arena type ({ARENA_TYPES}); excluded from analysis:')
        for s in sorted(unknown['session'].unique()):
            print(f'    {s}')
    return df.dropna(subset=['arena_type'])


def build_field_metrics_df(full: pd.DataFrame, fields: pd.DataFrame) -> pd.DataFrame:
    place_cells = full[full['place_cell'].apply(_to_bool)][
        ['session', 'unit', 'n_fields_detected']].drop_duplicates()

    agg = (fields.groupby(['session', 'unit'])
                 .agg(n_fields=('field_number', 'count'),
                      total_area_cm2=('area_cm2', 'sum'),
                      pct_area=('pct_of_occupied_area', 'sum'))
                 .reset_index())

    merged = place_cells.merge(agg, on=['session', 'unit'], how='left')
    # Place cells with no rows in 'PlaceFields' (n_fields_detected == 0) get 0s.
    for col in ('n_fields', 'total_area_cm2', 'pct_area'):
        merged[col] = merged[col].fillna(0.0)

    merged['arena_type'] = merged['session'].apply(extract_arena_type)

    unknown = merged[merged['arena_type'].isna()]
    if len(unknown):
        print(f'WARNING: {len(unknown)} place cell(s) had a session path with no '
              f'recognizable arena type ({ARENA_TYPES}); excluded from field analysis:')
        for s in sorted(unknown['session'].unique()):
            print(f'    {s}')
    return merged.dropna(subset=['arena_type'])


def compare_groups(df: pd.DataFrame, metric_col: str, metric_label: str):
    """Descriptive stats, omnibus test, and post-hoc pairwise tests for one
    metric across arena types. Returns (desc_df, omnibus_row, posthoc_df)."""
    groups = {}
    for arena in ARENA_TYPES:
        vals = pd.to_numeric(df.loc[df['arena_type'] == arena, metric_col],
                              errors='coerce').dropna().to_numpy(dtype=float)
        if len(vals):
            groups[arena] = vals

    desc_rows = []
    for arena, vals in groups.items():
        desc_rows.append({
            'metric': metric_label, 'arena_type': arena, 'n': len(vals),
            'mean': np.mean(vals), 'median': np.median(vals),
            'std': np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
            'sem': stats.sem(vals) if len(vals) > 1 else np.nan,
            'min': np.min(vals), 'max': np.max(vals),
        })
    desc_df = pd.DataFrame(desc_rows)

    omnibus = {'metric': metric_label, 'n_groups': len(groups),
               'groups': ', '.join(f'{a} (n={len(v)})' for a, v in groups.items())}

    if len(groups) < 2 or any(len(v) < 2 for v in groups.values()):
        omnibus.update({'test': 'insufficient data', 'statistic': np.nan,
                         'p_value': np.nan, 'significant': False,
                         'normal_all_groups': np.nan, 'equal_variance_p': np.nan})
        return desc_df, omnibus, pd.DataFrame()

    normal = True
    for vals in groups.values():
        if len(vals) < 3:
            normal = False
            continue
        try:
            if stats.shapiro(vals).pvalue < ALPHA:
                normal = False
        except Exception:
            normal = False

    try:
        p_levene = stats.levene(*groups.values()).pvalue
    except Exception:
        p_levene = np.nan
    equal_var = bool(p_levene >= ALPHA) if not np.isnan(p_levene) else False

    use_parametric = normal and equal_var and len(groups) >= 2
    if use_parametric:
        stat, p = stats.f_oneway(*groups.values())
        test_name = 'one-way ANOVA'
    else:
        stat, p = stats.kruskal(*groups.values())
        test_name = 'Kruskal-Wallis'

    omnibus.update({'test': test_name, 'statistic': stat, 'p_value': p,
                     'significant': bool(p < ALPHA), 'normal_all_groups': normal,
                     'equal_variance_p': p_levene})

    posthoc_rows = []
    pairs = list(combinations(groups.keys(), 2))
    raw_ps = []
    for g1, g2 in pairs:
        v1, v2 = groups[g1], groups[g2]
        if use_parametric:
            stat_p, p_p = stats.ttest_ind(v1, v2, equal_var=True)
            ph_test = 't-test'
        else:
            stat_p, p_p = stats.mannwhitneyu(v1, v2, alternative='two-sided')
            ph_test = 'Mann-Whitney U'
        raw_ps.append(p_p)
        posthoc_rows.append({'metric': metric_label, 'comparison': f'{g1} vs {g2}',
                              'test': ph_test, 'statistic': stat_p, 'p_raw': p_p})

    # Holm-Bonferroni step-down correction across the (up to 3) pairwise tests.
    if raw_ps:
        m = len(raw_ps)
        order = np.argsort(raw_ps)
        corrected = [None] * m
        prev = 0.0
        for rank, idx in enumerate(order):
            adj = min(max((m - rank) * raw_ps[idx], prev), 1.0)
            corrected[idx] = adj
            prev = adj
        for row, p_adj in zip(posthoc_rows, corrected):
            row['p_holm'] = p_adj
            row['significant'] = bool(p_adj < ALPHA)

    posthoc_df = pd.DataFrame(posthoc_rows)
    return desc_df, omnibus, posthoc_df



# Pastel palette (face = pastel fill, used for the box; edge = a deeper tone
# of the same hue, used for outlines/medians; dark/light = two further tones
# of the same hue used to shade individual points by their significance
# status -- see _point_colors).
ARENA_COLORS = {
    'Open':   {'face': '#AFCBFF', 'edge': '#5B8DEF', 'dark': '#3567C7', 'light': '#E4EFFF'},
    'Linear': {'face': '#FFD8A8', 'edge': '#F0993D', 'dark': '#C97418', 'light': '#FFF1DC'},
    'Circle': {'face': '#B8F2CB', 'edge': '#3FA96A', 'dark': '#237A48', 'light': '#E3FBEA'},
}
_DEFAULT_COLOR = {'face': '#D9D9D9', 'edge': '#8C8C8C', 'dark': '#8C8C8C', 'light': '#E8E8E8'}

# Points that failed their significance/shuffle test are always drawn in this
# neutral gray, regardless of arena.
SIG_GRAY = {'face': '#C9C9C9', 'edge': '#8C8C8C'}

# Metrics whose individual points are shaded by p-value/shuffle-test status
# rather than plain arena color -- see _point_colors. Also drives which plots
# get a significance legend.
SIGNIFICANCE_METRICS = {'coherence', 'stability_score', 'speed_score', 'speed_score_td'}


def _point_colors(metric_col: str, sub: pd.DataFrame, arena: str) -> list:
    """Per-point (face, edge) colors for one arena's jittered scatter dots,
    aligned row-for-row with `sub` (which must still be indexed against the
    original columns, e.g. 'coherence_bootstrap_sig').

    - coherence: gray unless the cell's coherence passed its shuffle test
      (coherence_bootstrap_sig is True).
    - stability_score: gray unless its split-half p-value is <= ALPHA.
    - speed_score (binned): gray unless speed_modulated_shuffle is True;
      significant positive scores get the darker tone, significant negative
      scores the lighter tone.
    - speed_score_td (instantaneous): gray unless speed_modulated_td is True;
      p-Speed cells get the darker tone, n-Speed cells the lighter tone.
    - everything else: plain arena face/edge for every point.
    """
    c = ARENA_COLORS.get(arena, _DEFAULT_COLOR)
    base = (c['face'], c['edge'])
    gray = (SIG_GRAY['face'], SIG_GRAY['edge'])

    if metric_col == 'coherence':
        return [base if _to_tri_bool(s) is True else gray
                for s in sub['coherence_bootstrap_sig']]

    if metric_col == 'stability_score':
        p = pd.to_numeric(sub['stability_p_value'], errors='coerce')
        return [base if (pd.notna(v) and v <= ALPHA) else gray for v in p]

    if metric_col == 'speed_score':
        sig  = sub['speed_modulated_shuffle']
        vals = pd.to_numeric(sub['speed_score'], errors='coerce')
        out = []
        for s, v in zip(sig, vals):
            if _to_tri_bool(s) is True and pd.notna(v):
                out.append((c['dark'], c['dark']) if v > 0 else (c['light'], c['edge']))
            else:
                out.append(gray)
        return out

    if metric_col == 'speed_score_td':
        sig     = sub['speed_modulated_td']
        p_speed = sub['p_speed']
        n_speed = sub['n_speed']
        out = []
        for s, p_, n_ in zip(sig, p_speed, n_speed):
            if _to_tri_bool(s) is True and _to_tri_bool(p_) is True:
                out.append((c['dark'], c['dark']))
            elif _to_tri_bool(s) is True and _to_tri_bool(n_) is True:
                out.append((c['light'], c['edge']))
            else:
                out.append(gray)
        return out

    return [base] * len(sub)


_SIGNIFICANCE_LEGENDS = {
    'coherence': [
        ('colored', 'passed shuffle test (p < 0.05)'),
        ('gray',    'failed shuffle test'),
    ],
    'stability_score': [
        ('colored', f'p ≤ {ALPHA:g}'),
        ('gray',    f'p > {ALPHA:g}'),
    ],
    'speed_score': [
        ('dark',  'significant, positive'),
        ('light', 'significant, negative'),
        ('gray',  'not significant'),
    ],
    'speed_score_td': [
        ('dark',  'p-Speed (significant, positive)'),
        ('light', 'n-Speed (significant, negative)'),
        ('gray',  'not significant'),
    ],
}


def _add_significance_legend(ax, metric_col: str):
    entries = _SIGNIFICANCE_LEGENDS.get(metric_col)
    if not entries:
        return
    ref = ARENA_COLORS['Open']
    tone_face = {'colored': ref['face'], 'dark': ref['dark'],
                 'light': ref['light'], 'gray': SIG_GRAY['face']}
    tone_edge = {'colored': ref['edge'], 'dark': ref['dark'],
                 'light': ref['edge'], 'gray': SIG_GRAY['edge']}
    handles = [Line2D([0], [0], marker='o', linestyle='None', markersize=6,
                       markerfacecolor=tone_face[tone], markeredgecolor=tone_edge[tone],
                       label=label)
               for tone, label in entries]
    ax.legend(handles=handles, loc='best', fontsize=7, frameon=False,
              handletextpad=0.4, labelspacing=0.3)


def plot_metric(df: pd.DataFrame, metric_col: str, metric_label: str, out_path: str):
    present = []
    subsets = {}
    for a in ARENA_TYPES:
        if a not in df['arena_type'].unique():
            continue
        sub = df.loc[df['arena_type'] == a].copy()
        sub[metric_col] = pd.to_numeric(sub[metric_col], errors='coerce')
        sub = sub.dropna(subset=[metric_col])
        if len(sub):
            present.append(a)
            subsets[a] = sub
    if not present:
        return

    data = [subsets[a][metric_col].to_numpy(dtype=float) for a in present]
    positions = np.arange(1, len(present) + 1)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FCFCFB')

    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, color='#E1E0D9',
                   zorder=0)
    ax.set_axisbelow(True)

    bp = ax.boxplot(data, positions=positions, showfliers=False,
                     patch_artist=True, widths=0.55,
                     boxprops=dict(linewidth=1.3),
                     whiskerprops=dict(linewidth=1.1, color='#8C8C8C'),
                     capprops=dict(linewidth=1.1, color='#8C8C8C'),
                     medianprops=dict(linewidth=1.6),
                     zorder=2)
    for patch, arena in zip(bp['boxes'], present):
        c = ARENA_COLORS.get(arena, _DEFAULT_COLOR)
        patch.set_facecolor(c['face'])
        patch.set_edgecolor(c['edge'])
    for median, arena in zip(bp['medians'], present):
        median.set_color(ARENA_COLORS.get(arena, _DEFAULT_COLOR)['edge'])

    rng = np.random.default_rng(0)
    for pos, arena, vals in zip(positions, present, data):
        sub = subsets[arena]
        colors = _point_colors(metric_col, sub, arena)
        faces  = [fc for fc, _ec in colors]
        edges  = [ec for _fc, ec in colors]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=16,
                   color=faces, alpha=0.85,
                   edgecolors=edges, linewidths=0.5, zorder=3)

    if metric_col in SIGNIFICANCE_METRICS:
        _add_significance_legend(ax, metric_col)

    tick_labels = [f'{a}\n(n={len(subsets[a])})' for a in present]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylabel(metric_label, fontsize=10)
    ax.set_title(metric_label, fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(axis='both', colors='#52514E', labelsize=9)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color('#C3C2B7')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def run_all_comparisons(df: pd.DataFrame, metrics: dict, plot_prefix: str):
    all_desc, all_omnibus, all_posthoc = [], [], []
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for col, label in metrics.items():
        desc_df, omnibus, posthoc_df = compare_groups(df, col, label)
        all_desc.append(desc_df)
        all_omnibus.append(omnibus)
        if len(posthoc_df):
            all_posthoc.append(posthoc_df)
        plot_metric(df, col, label,
                    os.path.join(PLOTS_DIR, f'{plot_prefix}_{col}.png'))

    desc_df    = pd.concat(all_desc, ignore_index=True) if all_desc else pd.DataFrame()
    omnibus_df = pd.DataFrame(all_omnibus)
    posthoc_df = pd.concat(all_posthoc, ignore_index=True) if all_posthoc else pd.DataFrame()
    return desc_df, omnibus_df, posthoc_df


# ── Categorical (proportion) comparisons ────────────────────────────────────
# Boolean/tri-state outcomes generated by the characterization pipeline
# (theta modulation, shuffle/stability significance, speed modulation) that
# weren't covered above because they're proportions, not continuous metrics.
# Compared with a chi-square test of independence (omnibus, arena_type x
# outcome) and pairwise Fisher's exact tests (post-hoc, Holm-Bonferroni
# corrected), mirroring the ANOVA/Kruskal + t-test/Mann-Whitney structure
# used for the continuous metrics above.

def build_all_units_df(full: pd.DataFrame) -> pd.DataFrame:
    """Like build_full_metrics_df, but keeps every recorded unit (not just
    confirmed place cells) -- needed to compute place-cell yield per arena."""
    df = full.copy()
    df['arena_type'] = df['session'].apply(extract_arena_type)
    return df.dropna(subset=['arena_type'])


def compare_categorical(df: pd.DataFrame, col: str, label: str):
    """Descriptive counts, omnibus chi-square test, and pairwise Fisher's
    exact post-hoc tests for one True/False outcome across arena types."""
    rows, table, present = [], [], []
    for arena in ARENA_TYPES:
        sub = df.loc[df['arena_type'] == arena, col].apply(_to_tri_bool).dropna()
        n_total = len(sub)
        if n_total == 0:
            continue
        n_true = int((sub == True).sum())  # noqa: E712
        present.append(arena)
        table.append([n_true, n_total - n_true])
        rows.append({'metric': label, 'arena_type': arena, 'n': n_total,
                      'n_true': n_true, 'n_false': n_total - n_true,
                      'pct_true': 100.0 * n_true / n_total})
    desc_df = pd.DataFrame(rows)

    omnibus = {'metric': label, 'n_groups': len(present),
               'groups': ', '.join(f'{a} (n={r["n"]})' for a, r in zip(present, rows))}

    if len(present) < 2:
        omnibus.update({'test': 'insufficient data', 'statistic': np.nan,
                         'p_value': np.nan, 'dof': np.nan, 'significant': False})
        return desc_df, omnibus, pd.DataFrame()

    arr = np.array(table)
    try:
        chi2, p, dof, _ = stats.chi2_contingency(arr)
        test_name = 'Chi-square test of independence'
    except Exception:
        chi2, p, dof, test_name = np.nan, np.nan, np.nan, 'chi-square failed'

    omnibus.update({'test': test_name, 'statistic': chi2, 'p_value': p, 'dof': dof,
                     'significant': bool(p < ALPHA) if pd.notna(p) else False})

    posthoc_rows = []
    pairs = list(combinations(range(len(present)), 2))
    raw_ps = []
    for i, j in pairs:
        sub_table = np.array([table[i], table[j]])
        try:
            odds_ratio, p_p = stats.fisher_exact(sub_table)
        except Exception:
            odds_ratio, p_p = np.nan, np.nan
        raw_ps.append(p_p)
        posthoc_rows.append({'metric': label, 'comparison': f'{present[i]} vs {present[j]}',
                              'test': "Fisher's exact", 'odds_ratio': odds_ratio, 'p_raw': p_p})

    if raw_ps:
        m = len(raw_ps)
        order = np.argsort(raw_ps)
        corrected = [None] * m
        prev = 0.0
        for rank, idx in enumerate(order):
            adj = min(max((m - rank) * raw_ps[idx], prev), 1.0)
            corrected[idx] = adj
            prev = adj
        for row, p_adj in zip(posthoc_rows, corrected):
            row['p_holm'] = p_adj
            row['significant'] = bool(p_adj < ALPHA)

    posthoc_df = pd.DataFrame(posthoc_rows)
    return desc_df, omnibus, posthoc_df


def plot_categorical(df: pd.DataFrame, col: str, label: str, out_path: str):
    present, pct_true, ns = [], [], []
    for a in ARENA_TYPES:
        sub = df.loc[df['arena_type'] == a, col].apply(_to_tri_bool).dropna()
        if len(sub) == 0:
            continue
        present.append(a)
        ns.append(len(sub))
        pct_true.append(100.0 * (sub == True).sum() / len(sub))  # noqa: E712
    if not present:
        return

    fig, ax = plt.subplots(figsize=(4.0, 4.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FCFCFB')
    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, color='#E1E0D9', zorder=0)
    ax.set_axisbelow(True)

    positions = np.arange(len(present))
    faces = [ARENA_COLORS.get(a, _DEFAULT_COLOR)['face'] for a in present]
    edges = [ARENA_COLORS.get(a, _DEFAULT_COLOR)['edge'] for a in present]
    ax.bar(positions, pct_true, color=faces, edgecolor=edges, linewidth=1.3,
           width=0.6, zorder=2)

    tick_labels = [f'{a}\n(n={n})' for a, n in zip(present, ns)]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylabel(f'% {label}', fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title(label, fontsize=10, fontweight='bold', pad=10, wrap=True)
    ax.tick_params(axis='both', colors='#52514E', labelsize=9)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color('#C3C2B7')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def compare_speed_direction(df: pd.DataFrame):
    """3-way (p-Speed / n-Speed / non-speed) classification of the
    instantaneous speed score across arena types -- a chi-square test of
    independence on the arena_type x {p-Speed, n-Speed, non-speed} table,
    since 'speed direction' isn't a simple True/False outcome."""
    label = 'Speed-direction classification (p-Speed / n-Speed / non-speed, % of place cells)'
    rows, table, present = [], [], []
    for arena in ARENA_TYPES:
        sub = df.loc[df['arena_type'] == arena]
        known = sub['speed_modulated_td'].apply(_to_tri_bool).notna()
        n_total = int(known.sum())
        if n_total == 0:
            continue
        n_p = int((sub['p_speed'].apply(_to_tri_bool) == True).sum())  # noqa: E712
        n_n = int((sub['n_speed'].apply(_to_tri_bool) == True).sum())  # noqa: E712
        n_non = n_total - n_p - n_n
        present.append(arena)
        table.append([n_p, n_n, n_non])
        rows.append({'metric': label, 'arena_type': arena, 'n': n_total,
                      'n_p_speed': n_p, 'n_n_speed': n_n, 'n_non_speed': n_non,
                      'pct_p_speed': 100.0 * n_p / n_total,
                      'pct_n_speed': 100.0 * n_n / n_total,
                      'pct_non_speed': 100.0 * n_non / n_total})
    desc_df = pd.DataFrame(rows)

    omnibus = {'metric': label, 'n_groups': len(present),
               'groups': ', '.join(f'{a} (n={r["n"]})' for a, r in zip(present, rows))}

    if len(present) < 2:
        omnibus.update({'test': 'insufficient data', 'statistic': np.nan,
                         'p_value': np.nan, 'dof': np.nan, 'significant': False})
        return desc_df, omnibus, pd.DataFrame()

    arr = np.array(table)
    try:
        chi2, p, dof, _ = stats.chi2_contingency(arr)
        test_name = 'Chi-square test of independence'
    except Exception:
        chi2, p, dof, test_name = np.nan, np.nan, np.nan, 'chi-square failed'

    omnibus.update({'test': test_name, 'statistic': chi2, 'p_value': p, 'dof': dof,
                     'significant': bool(p < ALPHA) if pd.notna(p) else False})

    posthoc_rows = []
    pairs = list(combinations(range(len(present)), 2))
    raw_ps = []
    for i, j in pairs:
        sub_table = np.array([table[i], table[j]])
        try:
            chi2_p, p_p, _dof_p, _ = stats.chi2_contingency(sub_table)
        except Exception:
            chi2_p, p_p = np.nan, np.nan
        raw_ps.append(p_p)
        posthoc_rows.append({'metric': label, 'comparison': f'{present[i]} vs {present[j]}',
                              'test': 'Chi-square test of independence',
                              'statistic': chi2_p, 'p_raw': p_p})

    if raw_ps:
        m = len(raw_ps)
        order = np.argsort(raw_ps)
        corrected = [None] * m
        prev = 0.0
        for rank, idx in enumerate(order):
            adj = min(max((m - rank) * raw_ps[idx], prev), 1.0)
            corrected[idx] = adj
            prev = adj
        for row, p_adj in zip(posthoc_rows, corrected):
            row['p_holm'] = p_adj
            row['significant'] = bool(p_adj < ALPHA)

    posthoc_df = pd.DataFrame(posthoc_rows)
    return desc_df, omnibus, posthoc_df


def plot_speed_direction(df: pd.DataFrame, out_path: str):
    present, pct_p, pct_n, pct_non, ns = [], [], [], [], []
    for a in ARENA_TYPES:
        sub = df.loc[df['arena_type'] == a]
        n_total = int(sub['speed_modulated_td'].apply(_to_tri_bool).notna().sum())
        if n_total == 0:
            continue
        n_p = int((sub['p_speed'].apply(_to_tri_bool) == True).sum())  # noqa: E712
        n_n = int((sub['n_speed'].apply(_to_tri_bool) == True).sum())  # noqa: E712
        present.append(a)
        ns.append(n_total)
        pct_p.append(100.0 * n_p / n_total)
        pct_n.append(100.0 * n_n / n_total)
        pct_non.append(100.0 * (n_total - n_p - n_n) / n_total)
    if not present:
        return

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FCFCFB')
    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, color='#E1E0D9', zorder=0)
    ax.set_axisbelow(True)

    positions = np.arange(len(present))
    dark   = [ARENA_COLORS.get(a, _DEFAULT_COLOR)['dark']  for a in present]
    light  = [ARENA_COLORS.get(a, _DEFAULT_COLOR)['light'] for a in present]
    edges  = [ARENA_COLORS.get(a, _DEFAULT_COLOR)['edge']  for a in present]
    bottoms2 = [p + n for p, n in zip(pct_p, pct_n)]
    ax.bar(positions, pct_p, color=dark, edgecolor=edges, linewidth=1.1,
           width=0.6, zorder=2, label='p-Speed')
    ax.bar(positions, pct_n, bottom=pct_p, color=light, edgecolor=edges, linewidth=1.1,
           width=0.6, zorder=2, label='n-Speed')
    ax.bar(positions, pct_non, bottom=bottoms2, color=SIG_GRAY['face'],
           edgecolor=SIG_GRAY['edge'], linewidth=1.1, width=0.6, zorder=2, label='non-speed')

    tick_labels = [f'{a}\n(n={n})' for a, n in zip(present, ns)]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylabel('% of place cells', fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title('Speed-direction classification (instantaneous)', fontsize=11,
                 fontweight='bold', pad=10)
    ax.tick_params(axis='both', colors='#52514E', labelsize=9)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color('#C3C2B7')
    ax.legend(loc='upper right', fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == '__main__':
    print(f'Loading {INPUT_EXCEL} ...')
    full, fields = load_sheets()

    full_df  = build_full_metrics_df(full)
    field_df = build_field_metrics_df(full, fields)

    print(f'\nPlace cells by arena type (Full sheet):')
    print(full_df['arena_type'].value_counts().reindex(ARENA_TYPES).fillna(0).astype(int))

    print(f'\nPlace cells by arena type (PlaceFields sheet, aggregated per cell):')
    print(field_df['arena_type'].value_counts().reindex(ARENA_TYPES).fillna(0).astype(int))

    full_desc, full_omnibus, full_posthoc = run_all_comparisons(
        full_df, FULL_METRICS, plot_prefix='Full')
    field_desc, field_omnibus, field_posthoc = run_all_comparisons(
        field_df, FIELD_METRICS, plot_prefix='Fields')

    desc_df    = pd.concat([full_desc, field_desc], ignore_index=True)
    omnibus_df = pd.concat([full_omnibus, field_omnibus], ignore_index=True)
    posthoc_df = pd.concat([full_posthoc, field_posthoc], ignore_index=True)

    # mode='a' + if_sheet_exists='replace' appends these as new sheets in the
    # SAME workbook that was just read from INPUT_EXCEL, overwriting only
    # sheets with a matching name (e.g. on a re-run) and leaving the
    # 'Full' / 'First_Half' / 'Second_Half' / 'PlaceFields' sheets untouched.
    with pd.ExcelWriter(INPUT_EXCEL, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        desc_df.to_excel(writer, sheet_name='Descriptives', index=False)
        omnibus_df.to_excel(writer, sheet_name='OmnibusTests', index=False)
        posthoc_df.to_excel(writer, sheet_name='PostHoc', index=False)

    print(f'\nDone. Statistics appended to {INPUT_EXCEL}')
    print(f'Box plots saved to {PLOTS_DIR}')

    sig = omnibus_df[omnibus_df['significant'] == True]  # noqa: E712
    if len(sig):
        print('\nSignificant omnibus differences (p < 0.05):')
        for _, row in sig.iterrows():
            print(f"  {row['metric']:45s} {row['test']:16s} "
                  f"stat={row['statistic']:.3f}  p={row['p_value']:.4g}")
    else:
        print('\nNo significant omnibus differences found.')

    # ── Extra continuous metric: mean firing rate ───────────────────────────
    # Generated by the characterization pipeline but not covered by
    # FULL_METRICS above; added here rather than editing that dict so the
    # original comparison set stays untouched.
    EXTRA_FULL_METRICS = {'mean_fr': 'Mean firing rate'}
    extra_desc, extra_omnibus, extra_posthoc = run_all_comparisons(
        full_df, EXTRA_FULL_METRICS, plot_prefix='Full')

    # ── Categorical (proportion) comparisons ────────────────────────────────
    # Place-cell yield uses every recorded unit (place_cell True/False/None),
    # not just confirmed place cells.
    all_units_df = build_all_units_df(full)

    def _stability_sig(v):
        v = pd.to_numeric(v, errors='coerce')
        return bool(v <= ALPHA) if pd.notna(v) else None

    full_df = full_df.copy()
    full_df['stability_significant'] = full_df['stability_p_value'].apply(_stability_sig)

    CATEGORICAL_METRICS_ALL_UNITS = {
        'place_cell': 'Place-cell yield (% of recorded units classified as place cells)',
    }
    CATEGORICAL_METRICS_PLACE_CELLS = {
        'theta_modulated':          'Theta modulation (% of place cells)',
        'coherence_bootstrap_sig':  'Coherence shuffle significance (% of place cells)',
        'stability_significant':    'Split-half stability significance, p <= 0.05 (% of place cells)',
        'speed_modulated_shuffle':  'Speed modulation, binned (% of place cells)',
        'speed_modulated_td':       'Speed modulation, instantaneous (% of place cells)',
    }

    cat_desc_all, cat_omni_all, cat_post_all = [], [], []
    for col, label in CATEGORICAL_METRICS_ALL_UNITS.items():
        d, o, p = compare_categorical(all_units_df, col, label)
        cat_desc_all.append(d)
        cat_omni_all.append(o)
        if len(p):
            cat_post_all.append(p)
        plot_categorical(all_units_df, col, label,
                          os.path.join(PLOTS_DIR, f'Categorical_{col}.png'))

    for col, label in CATEGORICAL_METRICS_PLACE_CELLS.items():
        d, o, p = compare_categorical(full_df, col, label)
        cat_desc_all.append(d)
        cat_omni_all.append(o)
        if len(p):
            cat_post_all.append(p)
        plot_categorical(full_df, col, label,
                          os.path.join(PLOTS_DIR, f'Categorical_{col}.png'))

    cat_desc_df    = pd.concat(cat_desc_all, ignore_index=True) if cat_desc_all else pd.DataFrame()
    cat_omnibus_df = pd.DataFrame(cat_omni_all)
    cat_posthoc_df = pd.concat(cat_post_all, ignore_index=True) if cat_post_all else pd.DataFrame()

    speed_dir_desc, speed_dir_omnibus, speed_dir_posthoc = compare_speed_direction(full_df)
    plot_speed_direction(full_df, os.path.join(PLOTS_DIR, 'Categorical_speed_direction.png'))
    speed_dir_omnibus_df = pd.DataFrame([speed_dir_omnibus])

    with pd.ExcelWriter(INPUT_EXCEL, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        extra_desc.to_excel(writer,        sheet_name='Descriptives_Extra',     index=False)
        extra_omnibus.to_excel(writer,     sheet_name='OmnibusTests_Extra',     index=False)
        extra_posthoc.to_excel(writer,     sheet_name='PostHoc_Extra',          index=False)
        cat_desc_df.to_excel(writer,       sheet_name='CategoricalDescriptives', index=False)
        cat_omnibus_df.to_excel(writer,    sheet_name='CategoricalOmnibus',      index=False)
        cat_posthoc_df.to_excel(writer,    sheet_name='CategoricalPostHoc',      index=False)
        speed_dir_desc.to_excel(writer,       sheet_name='SpeedDirection_Desc',    index=False)
        speed_dir_omnibus_df.to_excel(writer, sheet_name='SpeedDirection_Omnibus', index=False)
        speed_dir_posthoc.to_excel(writer,    sheet_name='SpeedDirection_PostHoc', index=False)

    print(f'\nExtra continuous + categorical statistics appended to {INPUT_EXCEL}')

    all_omnibus_final = pd.concat(
        [extra_omnibus, cat_omnibus_df, speed_dir_omnibus_df], ignore_index=True)
    sig_extra = all_omnibus_final[all_omnibus_final['significant'] == True]  # noqa: E712
    if len(sig_extra):
        print('\nSignificant differences among the extra/categorical comparisons (p < 0.05):')
        for _, row in sig_extra.iterrows():
            stat = row['statistic']
            stat_str = f'{stat:.3f}' if pd.notna(stat) else 'nan'
            print(f"  {row['metric']:65s} {row['test']:30s} "
                  f"stat={stat_str}  p={row['p_value']:.4g}")
    else:
        print('\nNo significant differences found among the extra/categorical comparisons.')
