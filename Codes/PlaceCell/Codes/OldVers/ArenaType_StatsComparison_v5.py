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
    - final_speed_score Final speed score (mean of the binned and instantaneous
                        scores). Speed cells are those with speed_cell == True:
                        p-type if final_speed_score > 0, n-type if < 0. Cells
                        with speed_cell False / 'not tested' are non-speed.

  Sheet 'PlaceFields' (aggregated per cell):
    - n_fields         Number of place fields per cell
    - total_area_cm2   Area occupied by fields per cell (sum of field areas)
    - pct_area         Percentage area occupied by fields per cell (sum of
                        each field's pct_of_occupied_area)

For each metric, per-group descriptive statistics are computed, then a
Shapiro-Wilk normality test is run on every arena group:
  - all groups normal  -> one-way ANOVA, then pairwise t-tests (Student's if
                          Levene p >= 0.05, else Welch's)
  - any group non-normal -> Kruskal-Wallis, then pairwise Mann-Whitney U
With only two groups present, the pairwise test is used directly. Post-hoc
p-values are Holm-Bonferroni corrected across the pairwise comparisons.

Percentage (categorical) outcomes use a chi-square test of independence,
followed by pairwise Fisher's exact tests (2x2 yes/no outcomes) or pairwise
2x3 chi-square tests (speed direction), Holm-Bonferroni corrected.

Parameters to edit (below):
    INPUT_EXCEL  = path to the output_excel file from
                   PlaceCellChar_FieldDetect_Main_v3.py -- also where these
                   statistics sheets get appended (in place)
    PLOTS_DIR    = folder to save comparison box plots to
"""

import os
import re
import textwrap
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── Parameters ──────────────────────────────────────────────────────────────

INPUT_EXCEL = r'X:\NMR_group_data\Runita\Analysis\Thesis\Data_v2_Accepted/All_TT_PlaceChar_VisitCrit.xlsx'
PLOTS_DIR   = r'X:\NMR_group_data\Runita\Analysis\Thesis\Data_v2_Accepted/ArenaType_StatsPlots'

ARENA_TYPES = ['Open', 'Linear', 'Circle']
ALPHA       = 0.05

FULL_METRICS = {
    'peak_fr':         'Peak firing rate',
    'sir':             'Spatial information score',
    'sparsity':        'Sparsity',
    'coherence':       'Coherence score',
    'stability_score':   'Stability score',
    'final_speed_score': 'Final speed score',
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
    """Normalize a boolean-ish Excel cell (bootstrap_sig / speed_cell / ...)
    to True, False, or None (unknown/blank/NaN/'not tested'), since
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


def _speed_class(score, cell):
    """'p' (p-type speed cell), 'n' (n-type), 'non' (non-speed), or None if
    final_speed_score is missing. A cell is a speed cell only when speed_cell
    is True; its type follows the sign of final_speed_score. speed_cell False
    or 'not tested' (the shuffle was skipped because the initial fit wasn't
    significant) both count as non-speed."""
    s = pd.to_numeric(score, errors='coerce')
    if pd.isna(s):
        return None
    if _to_tri_bool(cell) is True and s != 0:
        return 'p' if s > 0 else 'n'
    return 'non'


def add_speed_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'speed_class' ('p'/'n'/'non'/None) and 'speed_cell_final'
    (True/False/None, for the % speed cells comparison) columns."""
    df = df.copy()
    df['speed_class'] = [_speed_class(s, c)
                         for s, c in zip(df['final_speed_score'], df['speed_cell'])]
    df['speed_cell_final'] = df['speed_class'].map(
        lambda k: None if k is None else k in ('p', 'n'))
    return df


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
    return add_speed_class(df.dropna(subset=['arena_type']))


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
            'q1': np.percentile(vals, 25), 'q3': np.percentile(vals, 75),
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

    # Shapiro-Wilk normality per group gates the test choice: if every group
    # is normal (p >= ALPHA), parametric tests are used (one-way ANOVA +
    # pairwise t-tests); otherwise non-parametric (Kruskal-Wallis + pairwise
    # Mann-Whitney U). Levene's test picks Student's vs Welch's t-test.
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
    equal_var = bool(pd.notna(p_levene) and p_levene >= ALPHA)

    def _pairwise(v1, v2):
        if normal:
            name = "Student's t-test" if equal_var else "Welch's t-test"
            s, pp = stats.ttest_ind(v1, v2, equal_var=equal_var)
        else:
            name = 'Mann-Whitney U'
            s, pp = stats.mannwhitneyu(v1, v2, alternative='two-sided')
        return name, s, pp

    if len(groups) == 2:
        # Only two groups: the pairwise test itself is the omnibus test.
        test_name, stat, p = _pairwise(*groups.values())
    elif normal:
        stat, p = stats.f_oneway(*groups.values())
        test_name = 'One-way ANOVA'
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
        ph_test, stat_p, p_p = _pairwise(groups[g1], groups[g2])
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



# ── Figure style (matches the Fig1def spike-quality figures) ─────────────────
PAL_MAGENTA = '#FA7FFA'
PAL_CYAN    = '#7FFAFA'
PAL_BLUE    = '#7FB3E5'
PAL_DKBLUE  = '#0066CC'
PAL_GRAY    = '#7F7F7F'
PAL_GREEN   = '#00C000'
PAL_ORANGE  = '#FF8000'
PAL_RED     = '#FF0000'
PAL_BLACK   = '#000000'
FIG_FONT    = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']

FS_LABEL, FS_TITLE, FS_TICK, FS_LEGEND, FS_STATS = 15, 17, 13, 12, 10
STATS_LINE_IN = 0.2    # panel height per line of stats text (inches)
LEGEND_ROW_IN = 0.3    # panel height per row of legend entries (inches)
FIG_DPI = 500

plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    FIG_FONT,
    'text.color':         PAL_BLACK,
    'axes.labelcolor':    PAL_BLACK,
    'axes.edgecolor':     PAL_BLACK,
    'axes.linewidth':     1.5,
    'xtick.color':        PAL_BLACK,
    'ytick.color':        PAL_BLACK,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'xtick.major.width':  1.5,
    'ytick.major.width':  1.5,
    'axes.facecolor':     'white',
    'figure.facecolor':   'white',
    'savefig.facecolor':  'white',
    'axes.grid':          False,
    'legend.frameon':     False,
})

# Per-arena colors (face = 50 % tint used for boxes/bars/points; dark =
# full-strength tone and light = 80 % tint, used to shade individual points by
# their significance status -- see _point_colors; edge = black, as in the
# reference figures). Colors are assigned by an arena's position in
# ARENA_TYPES (cyan, magenta, green).
_HUES = [
    {'face': PAL_CYAN,    'edge': PAL_BLACK, 'dark': '#00B3B3',  'light': '#CCFDFD'},  # cyan
    {'face': PAL_MAGENTA, 'edge': PAL_BLACK, 'dark': '#C000C0',  'light': '#FDCCFD'},  # magenta
    {'face': PAL_BLUE,    'edge': PAL_BLACK, 'dark': PAL_DKBLUE, 'light': '#CCE1F5'},  # blue
]
ARENA_COLORS = {a: _HUES[i] for i, a in enumerate(ARENA_TYPES)}
_DEFAULT_COLOR = {'face': PAL_GRAY, 'edge': PAL_BLACK, 'dark': PAL_GRAY, 'light': '#D4D4D4'}

# Points that failed their significance/shuffle test are always drawn in this
# neutral gray (the reference figures' "discarded" gray), regardless of arena.
SIG_GRAY = {'face': PAL_GRAY, 'edge': PAL_BLACK}


def _new_fig(figsize, legend_rows=0, stats_lines=None):
    """Figure + main axes. If there is a legend and/or stats text, a second
    borderless axes sits below the plot to hold them (legend on top, stats
    underneath), so neither overlaps the data. The panel is sized to fit."""
    w, h = figsize
    panel_h = legend_rows * LEGEND_ROW_IN + len(stats_lines or []) * STATS_LINE_IN
    if panel_h > 0:
        panel_h += 0.2
        fig, (ax, ax_info) = plt.subplots(2, 1, figsize=(w, h + panel_h),
                                          gridspec_kw={'height_ratios': [h, panel_h]})
        ax_info.axis('off')
        return fig, ax, ax_info
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, None


def _draw_stats(ax_info, stats_lines):
    """Write the stats text at the bottom of the panel (below any legend)."""
    if ax_info is None or not stats_lines:
        return
    ax_info.text(0.5, 0.0, '\n'.join(stats_lines), transform=ax_info.transAxes,
                 fontsize=FS_STATS, va='bottom', ha='center', linespacing=1.3)


# ── Stats text shown under each plot ─────────────────────────────────────────

def _fmt_p(p) -> str:
    if p is None or pd.isna(p):
        return 'n/a'
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def _stars(p) -> str:
    if p is None or pd.isna(p):
        return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < ALPHA else 'n.s.'


def _desc_str(vals) -> str:
    """'mean ± SD | median [Q1, Q3]' for an array of values."""
    vals = np.asarray(vals, dtype=float)
    if not len(vals):
        return 'n/a'
    sd = f'{np.std(vals, ddof=1):.3g}' if len(vals) > 1 else 'n/a'
    return (f'mean {np.mean(vals):.3g} ± {sd} SD | median {np.median(vals):.3g} '
            f'[Q1 {np.percentile(vals, 25):.3g}, Q3 {np.percentile(vals, 75):.3g}]')


def _test_lines(omnibus: dict, posthoc_df: pd.DataFrame) -> list:
    """Omnibus test (statistic, dof, p) and post-hoc pairwise tests (statistic,
    raw p, Holm-corrected p) as text lines."""
    test = omnibus.get('test', '')
    if test == 'insufficient data':
        return ['Insufficient data for a statistical test']
    sym = ('H' if 'Kruskal' in test else 'F' if 'ANOVA' in test else
           'U' if 'Mann-Whitney' in test else 't' if 't-test' in test else 'χ²')
    dof = omnibus.get('dof')
    dof_str = f', dof = {int(dof)}' if dof is not None and pd.notna(dof) else ''
    p = omnibus.get('p_value')
    lines = [f'{test}: {sym} = {omnibus["statistic"]:.3f}{dof_str}, '
             f'p = {_fmt_p(p)} {_stars(p)}']
    if posthoc_df is not None and len(posthoc_df):
        lines.append(f'Post-hoc ({posthoc_df["test"].iloc[0]}, Holm-Bonferroni corrected):')
        for _, r in posthoc_df.iterrows():
            if 'Fisher' in r['test']:
                stat_str = f'OR = {r["odds_ratio"]:.3g}'
            elif 'Mann-Whitney' in r['test']:
                stat_str = f'U = {r["statistic"]:.1f}'
            elif 't-test' in r['test']:
                stat_str = f't = {r["statistic"]:.3f}'
            else:
                stat_str = f'χ² = {r["statistic"]:.3f}'
            lines.append(f'{r["comparison"]}: {stat_str}, p = {_fmt_p(r["p_raw"])}, '
                         f'p(Holm) = {_fmt_p(r["p_holm"])} {_stars(r["p_holm"])}')
    return lines


def _continuous_stats_lines(desc_df, omnibus, posthoc_df) -> list:
    lines = []
    for _, r in desc_df.iterrows():
        sd = f'{r["std"]:.3g}' if pd.notna(r['std']) else 'n/a'
        lines.append(f'{r["arena_type"]} (n={int(r["n"])}): mean {r["mean"]:.3g} ± {sd} SD | '
                     f'median {r["median"]:.3g} [Q1 {r["q1"]:.3g}, Q3 {r["q3"]:.3g}]')
    lines += _test_lines(omnibus, posthoc_df)
    if omnibus.get('test') != 'insufficient data':
        lines.append(f'Shapiro-Wilk, all groups normal: {"yes" if omnibus.get("normal_all_groups") else "no"}; '
                     f'Levene equal-variance p = {_fmt_p(omnibus.get("equal_variance_p"))}')
    return lines


def _categorical_stats_lines(desc_df, omnibus, posthoc_df) -> list:
    lines = [f'{r["arena_type"]} (n={int(r["n"])}): {int(r["n_true"])} yes / '
             f'{int(r["n_false"])} no = {r["pct_true"]:.1f} %'
             for _, r in desc_df.iterrows()]
    return lines + _test_lines(omnibus, posthoc_df)


def _speed_dir_stats_lines(desc_df, omnibus, posthoc_df) -> list:
    lines = [f'{r["arena_type"]} (n={int(r["n"])}): '
             f'p-Speed {int(r["n_p_speed"])} ({r["pct_p_speed"]:.1f} %), '
             f'n-Speed {int(r["n_n_speed"])} ({r["pct_n_speed"]:.1f} %), '
             f'non-speed {int(r["n_non_speed"])} ({r["pct_non_speed"]:.1f} %)'
             for _, r in desc_df.iterrows()]
    return lines + _test_lines(omnibus, posthoc_df)


def _style_axes(ax):
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.spines[['top', 'right']].set_visible(False)


def _set_title(ax, text):
    ax.set_title(textwrap.fill(text, 34), fontsize=FS_TITLE - 2, pad=12)


def _save(fig, out_path):
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def _draw_sig_brackets(ax, pos_by_group: dict, posthoc_df, y_base: float,
                       y_step: float, y_text: float):
    """For every pairwise comparison with Holm-corrected p < ALPHA, draw a
    single horizontal line above the two groups with the number of stars
    (* p<0.05, ** p<0.01, *** p<0.001) centered on it. Narrower spans are
    stacked lowest so lines don't cross. Returns the y of the highest line
    drawn, or None if no comparison was significant."""
    if posthoc_df is None or not len(posthoc_df):
        return None
    sig = []
    for _, r in posthoc_df.iterrows():
        p = r.get('p_holm')
        if p is None or pd.isna(p) or p >= ALPHA:
            continue
        g1, g2 = [g.strip() for g in str(r['comparison']).split(' vs ')]
        if g1 not in pos_by_group or g2 not in pos_by_group:
            continue
        x1, x2 = sorted((pos_by_group[g1], pos_by_group[g2]))
        sig.append((x2 - x1, x1, x2, _stars(p)))
    y = y_base
    for _span, x1, x2, stars in sorted(sig):
        ax.plot([x1, x2], [y, y], color=PAL_BLACK, linewidth=1.5,
                solid_capstyle='butt', clip_on=False, zorder=5)
        ax.text((x1 + x2) / 2, y + y_text, stars, ha='center', va='center',
                fontsize=FS_LABEL + 3, fontweight='bold', clip_on=False, zorder=5)
        y += y_step
    return y - y_step if sig else None


# Metrics whose individual points are shaded by p-value/shuffle-test status
# rather than plain arena color -- see _point_colors. Also drives which plots
# get a significance legend.
SIGNIFICANCE_METRICS = {'coherence', 'stability_score', 'final_speed_score'}


def _point_colors(metric_col: str, sub: pd.DataFrame, arena: str) -> list:
    """Per-point (face, edge) colors for one arena's jittered scatter dots,
    aligned row-for-row with `sub` (which must still be indexed against the
    original columns, e.g. 'coherence_bootstrap_sig').

    - coherence: gray unless the cell's coherence passed its shuffle test
      (coherence_bootstrap_sig is True).
    - stability_score: gray unless its split-half p-value is <= ALPHA.
    - final_speed_score: gray unless speed_cell is True (False / 'not
      tested' are gray); p-type cells (score > 0) get the darker tone, n-type
      cells (score < 0) the lighter tone.
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

    if metric_col == 'final_speed_score':
        tone = {'p': (c['dark'], c['edge']), 'n': (c['light'], c['edge'])}
        return [tone.get(k, gray) for k in sub['speed_class']]

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
    'final_speed_score': [
        ('dark',  'p-type speed cell (positive)'),
        ('light', 'n-type speed cell (negative)'),
        ('gray',  'not a speed cell / not tested'),
    ],
}


def _add_significance_legend(ax_info, metric_col: str):
    """Draw the significance legend into the borderless panel below the plot."""
    entries = _SIGNIFICANCE_LEGENDS.get(metric_col)
    if not entries:
        return
    ref = _HUES[0]
    tone_face = {'colored': ref['face'], 'dark': ref['dark'],
                 'light': ref['light'], 'gray': SIG_GRAY['face']}
    handles = [Line2D([0], [0], marker='o', linestyle='None', markersize=9,
                       markerfacecolor=tone_face[tone], markeredgecolor=PAL_BLACK,
                       markeredgewidth=1.0, label=label)
               for tone, label in entries]
    ax_info.legend(handles=handles, loc='upper center', fontsize=FS_LEGEND, ncol=1)


def plot_metric(df: pd.DataFrame, metric_col: str, metric_label: str, out_path: str,
                stats_lines=None, posthoc_df=None):
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

    has_legend = metric_col in SIGNIFICANCE_METRICS
    fig, ax, ax_info = _new_fig(
        (7, 5.5), stats_lines=stats_lines,
        legend_rows=len(_SIGNIFICANCE_LEGENDS[metric_col]) if has_legend else 0)

    bp = ax.boxplot(data, positions=positions, showfliers=False,
                     patch_artist=True, widths=0.55,
                     boxprops=dict(linewidth=1.5, color=PAL_BLACK),
                     whiskerprops=dict(linewidth=1.5, color=PAL_BLACK),
                     capprops=dict(linewidth=1.5, color=PAL_BLACK),
                     medianprops=dict(linewidth=2.0, color=PAL_BLACK),
                     zorder=2)
    for patch, arena in zip(bp['boxes'], present):
        c = ARENA_COLORS.get(arena, _DEFAULT_COLOR)
        patch.set_facecolor(c['face'])
        patch.set_edgecolor(c['edge'])

    rng = np.random.default_rng(0)
    for pos, arena, vals in zip(positions, present, data):
        sub = subsets[arena]
        colors = _point_colors(metric_col, sub, arena)
        faces  = [fc for fc, _ec in colors]
        edges  = [ec for _fc, ec in colors]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=30,
                   facecolors=faces, edgecolors=edges, linewidths=0.8, zorder=3)

    # Significance lines above significantly different pairs of groups.
    y_lo, y_hi = ax.get_ylim()
    span = y_hi - y_lo
    y_data_max = max(np.max(v) for v in data)
    top = _draw_sig_brackets(ax, dict(zip(present, positions)), posthoc_df,
                             y_base=y_data_max + 0.06 * span,
                             y_step=0.10 * span, y_text=0.03 * span)
    if top is not None:
        ax.set_ylim(y_lo, max(y_hi, top + 0.08 * span))

    if has_legend:
        _add_significance_legend(ax_info, metric_col)
    _draw_stats(ax_info, stats_lines)

    tick_labels = [f'{a}\n(n={len(subsets[a])})' for a in present]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel(textwrap.fill(metric_label, 30), fontsize=FS_LABEL, labelpad=8)
    _set_title(ax, metric_label)
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, out_path)


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
                    os.path.join(PLOTS_DIR, f'{plot_prefix}_{col}.png'),
                    stats_lines=_continuous_stats_lines(desc_df, omnibus, posthoc_df),
                    posthoc_df=posthoc_df)

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


def _set_percent_ylim(ax, pos_by_group: dict, posthoc_df, bar_top: float):
    """Draw significance lines above 100 %-scale bars and extend the y-axis to
    fit them, keeping the ticks at 0-100."""
    top = _draw_sig_brackets(ax, pos_by_group, posthoc_df, y_base=bar_top + 4,
                             y_step=8, y_text=3)
    ax.set_ylim(0, 100 if top is None else max(100, top + 8))
    ax.set_yticks(np.arange(0, 101, 20))


def plot_categorical(df: pd.DataFrame, col: str, label: str, out_path: str,
                     stats_lines=None, posthoc_df=None):
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

    fig, ax, ax_info = _new_fig((7, 5.5), stats_lines=stats_lines)
    _draw_stats(ax_info, stats_lines)

    positions = np.arange(len(present))
    faces = [ARENA_COLORS.get(a, _DEFAULT_COLOR)['face'] for a in present]
    edges = [ARENA_COLORS.get(a, _DEFAULT_COLOR)['edge'] for a in present]
    ax.bar(positions, pct_true, color=faces, edgecolor=edges, linewidth=1.2,
           width=0.6, zorder=3)

    tick_labels = [f'{a}\n(n={n})' for a, n in zip(present, ns)]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel('% of cells', fontsize=FS_LABEL, labelpad=8)
    _set_percent_ylim(ax, dict(zip(present, positions)), posthoc_df, max(pct_true))
    _set_title(ax, label)
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, out_path)


def compare_speed_direction(df: pd.DataFrame):
    """3-way (p-Speed / n-Speed / non-speed) classification from the final
    speed score / speed_cell across arena types -- a chi-square test of
    independence on the arena_type x {p-Speed, n-Speed, non-speed} table,
    since 'speed direction' isn't a simple True/False outcome."""
    label = 'Speed-direction classification (p-Speed / n-Speed / non-speed, % of place cells)'
    rows, table, present = [], [], []
    for arena in ARENA_TYPES:
        cls = df.loc[df['arena_type'] == arena, 'speed_class'].dropna()
        n_total = len(cls)
        if n_total == 0:
            continue
        n_p = int((cls == 'p').sum())
        n_n = int((cls == 'n').sum())
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


def plot_speed_direction(df: pd.DataFrame, out_path: str, stats_lines=None,
                         posthoc_df=None):
    present, pct_p, pct_n, pct_non, ns = [], [], [], [], []
    for a in ARENA_TYPES:
        cls = df.loc[df['arena_type'] == a, 'speed_class'].dropna()
        n_total = len(cls)
        if n_total == 0:
            continue
        n_p = int((cls == 'p').sum())
        n_n = int((cls == 'n').sum())
        present.append(a)
        ns.append(n_total)
        pct_p.append(100.0 * n_p / n_total)
        pct_n.append(100.0 * n_n / n_total)
        pct_non.append(100.0 * (n_total - n_p - n_n) / n_total)
    if not present:
        return

    fig, ax, ax_info = _new_fig((7, 5.5), legend_rows=1, stats_lines=stats_lines)

    positions = np.arange(len(present))
    dark   = [ARENA_COLORS.get(a, _DEFAULT_COLOR)['dark']  for a in present]
    light  = [ARENA_COLORS.get(a, _DEFAULT_COLOR)['light'] for a in present]
    edges  = [ARENA_COLORS.get(a, _DEFAULT_COLOR)['edge']  for a in present]
    bottoms2 = [p + n for p, n in zip(pct_p, pct_n)]
    ax.bar(positions, pct_p, color=dark, edgecolor=edges, linewidth=1.2,
           width=0.6, zorder=3, label='p-Speed')
    ax.bar(positions, pct_n, bottom=pct_p, color=light, edgecolor=edges, linewidth=1.2,
           width=0.6, zorder=3, label='n-Speed')
    ax.bar(positions, pct_non, bottom=bottoms2, color=SIG_GRAY['face'],
           edgecolor=SIG_GRAY['edge'], linewidth=1.2, width=0.6, zorder=3, label='non-speed')

    tick_labels = [f'{a}\n(n={n})' for a, n in zip(present, ns)]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel('% of place cells', fontsize=FS_LABEL, labelpad=8)
    _set_percent_ylim(ax, dict(zip(present, positions)), posthoc_df, 100)
    _set_title(ax, 'Speed-direction classification (final speed score)')
    _style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    ax_info.legend(handles, labels, loc='upper center', fontsize=FS_LEGEND, ncol=3)
    _draw_stats(ax_info, stats_lines)
    fig.tight_layout()
    _save(fig, out_path)


# ── Histograms by arena type ────────────────────────────────────────────────
# Histogram plots of the place-cell characterization metrics, split by arena
# type. Run at the end of __main__ on the SAME place-cell populations
# (full_df / field_df) analyzed above.

HIST_DIR = os.path.join(PLOTS_DIR, 'Histograms')
N_BINS   = 15

SPEED_METRICS = {
    'final_speed_score': 'Final speed score',
}
HIST_FULL_METRICS = {k: v for k, v in FULL_METRICS.items() if k not in SPEED_METRICS}

# Explicit p-type/n-type speed-cell colors (green/red) -- independent of the
# per-arena ARENA_COLORS palette used elsewhere.
P_COLOR = {'face': PAL_GREEN, 'edge': PAL_BLACK}  # p-type (positive) speed cells
N_COLOR = {'face': PAL_RED,   'edge': PAL_BLACK}  # n-type (negative) speed cells


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

    desc_df, omnibus, posthoc_df = compare_groups(df, metric_col, metric_label)
    stats_lines = _continuous_stats_lines(desc_df, omnibus, posthoc_df)
    fig, ax, ax_info = _new_fig((7, 5), legend_rows=(len(values) + 1) // 2,
                                stats_lines=stats_lines)

    # Overlaid groups: translucent tinted fill plus a full-strength outline in
    # the group's dark tone (black outlines would be unreadable where the
    # distributions overlap).
    for arena, vals in values.items():
        c = ARENA_COLORS.get(arena, _DEFAULT_COLOR)
        ax.hist(vals, bins=bins, histtype='stepfilled', facecolor=c['face'],
                edgecolor='none', alpha=0.6, zorder=3)
        ax.hist(vals, bins=bins, histtype='step', edgecolor=c['dark'],
                linewidth=1.5, zorder=4)

    handles = [Patch(facecolor=ARENA_COLORS.get(a, _DEFAULT_COLOR)['face'],
                     edgecolor=ARENA_COLORS.get(a, _DEFAULT_COLOR)['dark'],
                     linewidth=1.5, label=f'{a} (n={len(v)})')
               for a, v in values.items()]
    ax_info.legend(handles=handles, loc='upper center', fontsize=FS_LEGEND,
                   ncol=min(2, len(handles)))
    _draw_stats(ax_info, stats_lines)

    ax.set_xlabel(metric_label, fontsize=FS_LABEL, labelpad=8)
    ax.set_ylabel('Number of cells', fontsize=FS_LABEL, labelpad=8)
    _set_title(ax, metric_label)
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, out_path)


def _speed_type_masks(sub: pd.DataFrame):
    """Boolean (p_mask, n_mask) numpy arrays for a sub-dataframe, from its
    speed_class column."""
    return (sub['speed_class'] == 'p').to_numpy(), (sub['speed_class'] == 'n').to_numpy()


def plot_speed_histograms_by_arena(df: pd.DataFrame, metric_col: str,
                                    metric_label: str, out_path: str):
    """Final-speed-score histogram restricted to speed cells (speed_cell ==
    True), with one panel per arena type and p-type (green) / n-type (red)
    speed cells stacked within each panel."""
    passed = df['speed_class'].isin(['p', 'n'])

    sub_all = df.loc[passed].copy()
    sub_all[metric_col] = pd.to_numeric(sub_all[metric_col], errors='coerce')
    sub_all = sub_all.dropna(subset=[metric_col])

    present = [a for a in ARENA_TYPES if (sub_all['arena_type'] == a).any()]
    if not present:
        print(f'  No speed cells for {metric_label}; skipping plot.')
        return

    bins = np.histogram_bin_edges(sub_all[metric_col].to_numpy(dtype=float), bins=N_BINS)

    # Stats text: descriptives per arena type and p/n type, then a
    # normality-gated (ANOVA/t-test or Kruskal-Wallis/Mann-Whitney U)
    # comparison of the pooled speed-modulated cells across arena types.
    split = {}
    stats_lines = []
    for arena in present:
        sub = sub_all.loc[sub_all['arena_type'] == arena]
        p_mask, n_mask = _speed_type_masks(sub)
        split[arena] = (sub.loc[p_mask, metric_col].to_numpy(dtype=float),
                        sub.loc[n_mask, metric_col].to_numpy(dtype=float))
        for name, v in zip(('p-type', 'n-type'), split[arena]):
            stats_lines.append(f'{arena} {name} (n={len(v)}): {_desc_str(v)}')
    _, omnibus, posthoc_df = compare_groups(sub_all, metric_col, metric_label)
    stats_lines.append('Across arena types (all speed-modulated cells pooled):')
    stats_lines += _test_lines(omnibus, posthoc_df)

    n_pan   = len(present)
    panel_h = len(stats_lines) * STATS_LINE_IN + 0.4
    fig = plt.figure(figsize=(6 * n_pan, 5.5 + panel_h))
    gs = fig.add_gridspec(2, n_pan, height_ratios=[5.5, panel_h])
    axes = []
    for i in range(n_pan):
        axes.append(fig.add_subplot(gs[0, i], sharey=axes[0] if axes else None))
        if i > 0:
            axes[i].tick_params(labelleft=False)
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.axis('off')
    _draw_stats(ax_info, stats_lines)

    for ax, arena in zip(axes, present):
        sub = sub_all.loc[sub_all['arena_type'] == arena]
        p_vals, n_vals = split[arena]

        ax.hist([p_vals, n_vals], bins=bins, stacked=True,
                color=[P_COLOR['face'], N_COLOR['face']],
                edgecolor=PAL_BLACK, linewidth=1.2, zorder=3)
        ax.set_title(f'{arena}\n(n={len(sub)}: {len(p_vals)} p-type, {len(n_vals)} n-type)',
                     fontsize=FS_LABEL, pad=12)
        ax.set_xlabel(textwrap.fill(metric_label, 28), fontsize=FS_LABEL, labelpad=8)
        _style_axes(ax)

    axes[0].set_ylabel('Number of cells', fontsize=FS_LABEL, labelpad=8)
    handles = [Patch(facecolor=P_COLOR['face'], edgecolor=P_COLOR['edge'],
                     linewidth=1.2, label='p-type speed cell'),
               Patch(facecolor=N_COLOR['face'], edgecolor=N_COLOR['edge'],
                     linewidth=1.2, label='n-type speed cell')]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=FS_LEGEND,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f'{metric_label}\n(speed cells only)',
                 fontsize=FS_TITLE, y=1.2)
    fig.tight_layout()
    _save(fig, out_path)


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
        'speed_cell_final':         'Speed cells (% of place cells)',
    }

    cat_desc_all, cat_omni_all, cat_post_all = [], [], []
    for col, label in CATEGORICAL_METRICS_ALL_UNITS.items():
        d, o, p = compare_categorical(all_units_df, col, label)
        cat_desc_all.append(d)
        cat_omni_all.append(o)
        if len(p):
            cat_post_all.append(p)
        plot_categorical(all_units_df, col, label,
                          os.path.join(PLOTS_DIR, f'Categorical_{col}.png'),
                          stats_lines=_categorical_stats_lines(d, o, p),
                          posthoc_df=p)

    for col, label in CATEGORICAL_METRICS_PLACE_CELLS.items():
        d, o, p = compare_categorical(full_df, col, label)
        cat_desc_all.append(d)
        cat_omni_all.append(o)
        if len(p):
            cat_post_all.append(p)
        plot_categorical(full_df, col, label,
                          os.path.join(PLOTS_DIR, f'Categorical_{col}.png'),
                          stats_lines=_categorical_stats_lines(d, o, p),
                          posthoc_df=p)

    cat_desc_df   = pd.concat(cat_desc_all, ignore_index=True) if cat_desc_all else pd.DataFrame()
    cat_omnibus_df = pd.DataFrame(cat_omni_all)
    cat_posthoc_df = pd.concat(cat_post_all, ignore_index=True) if cat_post_all else pd.DataFrame()

    speed_dir_desc, speed_dir_omnibus, speed_dir_posthoc = compare_speed_direction(full_df)
    plot_speed_direction(full_df, os.path.join(PLOTS_DIR, 'Categorical_speed_direction.png'),
                         stats_lines=_speed_dir_stats_lines(speed_dir_desc, speed_dir_omnibus,
                                                            speed_dir_posthoc),
                         posthoc_df=speed_dir_posthoc)
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

    # ── All statistics in one CSV ───────────────────────────────────────────
    # Stacks every statistics table written to the workbook above into one
    # long-format CSV. 'table' names the source sheet; the tables have
    # different columns, so cells that don't apply to a table are left blank.
    all_stats_tables = {
        'Descriptives':             desc_df,
        'OmnibusTests':             omnibus_df,
        'PostHoc':                  posthoc_df,
        'Descriptives_Extra':       extra_desc,
        'OmnibusTests_Extra':       extra_omnibus,
        'PostHoc_Extra':            extra_posthoc,
        'CategoricalDescriptives':  cat_desc_df,
        'CategoricalOmnibus':       cat_omnibus_df,
        'CategoricalPostHoc':       cat_posthoc_df,
        'SpeedDirection_Desc':      speed_dir_desc,
        'SpeedDirection_Omnibus':   speed_dir_omnibus_df,
        'SpeedDirection_PostHoc':   speed_dir_posthoc,
    }
    stats_csv_parts = []
    for sheet_name, tbl in all_stats_tables.items():
        if tbl is None or not len(tbl):
            continue
        tbl = tbl.copy()
        tbl.insert(0, 'table', sheet_name)
        stats_csv_parts.append(tbl)

    STATS_CSV = os.path.splitext(INPUT_EXCEL)[0] + '_AllStats.csv'
    pd.concat(stats_csv_parts, ignore_index=True).to_csv(
        STATS_CSV, index=False, encoding='utf-8-sig')
    print(f'\nAll statistics saved to {STATS_CSV}')

    # ── Histograms by arena type ────────────────────────────────────────────
    # Same place-cell populations (full_df / field_df) as the statistics above.
    os.makedirs(HIST_DIR, exist_ok=True)

    print('\nPlotting Full-sheet metric histograms by arena type...')
    for col, label in HIST_FULL_METRICS.items():
        plot_histogram_by_arena(full_df, col, label,
                                 os.path.join(HIST_DIR, f'Hist_Full_{col}.png'))

    print('Plotting PlaceFields-sheet metric histograms by arena type...')
    for col, label in FIELD_METRICS.items():
        plot_histogram_by_arena(field_df, col, label,
                                 os.path.join(HIST_DIR, f'Hist_Fields_{col}.png'))

    print('Plotting final speed-score histogram (p-type/n-type speed cells)...')
    for col, label in SPEED_METRICS.items():
        plot_speed_histograms_by_arena(full_df, col, label,
                                        os.path.join(HIST_DIR, f'Hist_Speed_{col}.png'))

    print(f'\nDone. Histogram plots saved to {HIST_DIR}')
