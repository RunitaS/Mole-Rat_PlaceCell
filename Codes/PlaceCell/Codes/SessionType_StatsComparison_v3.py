# -*- coding: utf-8 -*-
"""
Statistical comparison of place-cell characterization metrics across SESSION
types within each arena type, using the workbook produced by
PlaceCellChar_FieldDetect_Main_v3.py (the 'output_excel' file). Results are
appended back into that SAME workbook as additional sheets rather than
written to a separate file.

Arena type (Open / Linear / Circle) and session type are both extracted from
the folder path stored in the 'session' column (column A) of the 'Full' and
'PlaceFields' sheets, e.g. 'Fa1059\\Linear\\Day10\\1_0' is arena 'Linear',
session type '0'. The session type comes from the label of the last folder:

    Open   : Cntrl, Rotate, Zero          (e.g. '1_Cntrl', '2Rotate')
    Linear : 0, 90, 180, 270              (digits at the end, e.g. '3_180')
    Circle : Stnd, NoRot, Rot             (e.g. '2NoRot', '4Stnd')

The full comparison (descriptives, omnibus + post-hoc tests, categorical and
speed-direction comparisons, box/bar plots and histograms) is run separately
for each arena type, comparing the session types WITHIN that arena. Sheets
are suffixed with the arena name (e.g. 'Descriptives_Linear'), and plots go
in one sub-folder of PLOTS_DIR per arena. Only rows with place_cell == True
are analyzed (except place-cell yield, which uses all recorded units).

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

For each metric, per-group descriptive statistics are computed. A Shapiro-Wilk
normality check and a Levene equal-variance check are run per group and
reported for reference, but the tests themselves are always non-parametric:
an omnibus Kruskal-Wallis test across the session-type groups of an arena,
followed by pairwise Wilcoxon rank-sum post-hoc tests, with Holm-Bonferroni
correction across that arena's pairwise comparisons.

Parameters to edit (below):
    INPUT_EXCEL  = path to the output_excel file from
                   PlaceCellChar_FieldDetect_Main_v3.py -- also where these
                   statistics sheets get appended (in place)
    PLOTS_DIR    = folder to save comparison plots to (one sub-folder per arena)
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

INPUT_EXCEL = r'X:\NMR_group_data\Runita\Analysis\Thesis\Data_v2_Accepted\All_TT_PlaceChar_VisitCrit.xlsx'
PLOTS_DIR   = r'X:\NMR_group_data\Runita\Analysis\Thesis\Data_v2_Accepted\SessionType_StatsPlots'

ARENA_TYPES = ['Open', 'Linear', 'Circle']

# Session types within each arena, in plotting order.
ARENA_SESSION_TYPES = {
    'Open':   ['Cntrl', 'Rotate', 'Zero'],
    'Linear': ['0', '90', '180', '270'],
    'Circle': ['Stnd', 'NoRot', 'Rot'],
}
# Flat order over all arenas; session-type labels are unique across arenas, and
# the plotting/stat helpers skip groups that are absent from the data they get.
SESSION_TYPE_ORDER = [t for ts in ARENA_SESSION_TYPES.values() for t in ts]
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


def extract_session_type(session_path: str, arena: str):
    """Return the session type named in the last folder of `session_path`
    (e.g. '1_Cntrl' -> 'Cntrl' for Open, '3_180' -> '180' for Linear,
    '2NoRot' -> 'NoRot' for Circle), or None if it can't be recognized."""
    if arena is None:
        return None
    label = re.split(r'[\\/]+', str(session_path).strip())[-1].strip().lower()

    if arena == 'Open':
        for t in ARENA_SESSION_TYPES['Open']:
            if t.lower() in label:
                return t
    elif arena == 'Linear':
        m = re.search(r'_(270|180|90|0)$', label)
        if m:
            return m.group(1)
    elif arena == 'Circle':
        # 'norot' must be tested before 'rot', which it contains.
        if 'norot' in label:
            return 'NoRot'
        if 'stnd' in label:
            return 'Stnd'
        if 'rot' in label:
            return 'Rot'
    return None


def _tag_session_type(df: pd.DataFrame, what: str) -> pd.DataFrame:
    """Add 'arena_type' and 'session_type' columns to a dataframe with a
    'session' column, warn about (and drop) rows where either isn't
    recognized."""
    df = df.copy()
    df['arena_type'] = df['session'].apply(extract_arena_type)
    df['session_type'] = [extract_session_type(s, a)
                          for s, a in zip(df['session'], df['arena_type'])]

    unknown = df[df['arena_type'].isna() | df['session_type'].isna()]
    if len(unknown):
        print(f'WARNING: {len(unknown)} {what} had a session path with no '
              f'recognizable arena/session type; excluded from analysis:')
        for s in sorted(unknown['session'].astype(str).unique()):
            print(f'    {s}')
    return df.dropna(subset=['arena_type', 'session_type'])


def _label(session_type: str) -> str:
    """Display label for plots: Linear session types are angles."""
    return f'{session_type}°' if session_type.isdigit() else session_type


def load_sheets():
    full   = pd.read_excel(INPUT_EXCEL, sheet_name='Full')
    fields = pd.read_excel(INPUT_EXCEL, sheet_name='PlaceFields')
    return full, fields


def build_full_metrics_df(full: pd.DataFrame) -> pd.DataFrame:
    df = full[full['place_cell'].apply(_to_bool)]
    return _tag_session_type(df, 'place cell(s)')


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

    return _tag_session_type(merged, 'place cell(s) (field analysis)')


def compare_groups(df: pd.DataFrame, metric_col: str, metric_label: str):
    """Descriptive stats, omnibus test, and post-hoc pairwise tests for one
    metric across arena types. Returns (desc_df, omnibus_row, posthoc_df)."""
    groups = {}
    for arena in SESSION_TYPE_ORDER:
        vals = pd.to_numeric(df.loc[df['session_type'] == arena, metric_col],
                              errors='coerce').dropna().to_numpy(dtype=float)
        if len(vals):
            groups[arena] = vals

    desc_rows = []
    for arena, vals in groups.items():
        desc_rows.append({
            'metric': metric_label, 'session_type': arena, 'n': len(vals),
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

    # Shapiro-Wilk normality (per group) and Levene equal-variance are still
    # run and reported for reference, but no longer gate the test choice:
    # the omnibus/post-hoc tests below are always non-parametric.
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
        stat_p, p_p = stats.ranksums(v1, v2)
        ph_test = 'Wilcoxon rank-sum'
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

# Per-session-type colors (face = 50 % tint used for boxes/bars/points; dark =
# full-strength tone and light = 80 % tint, used to shade individual points by
# their significance status -- see _point_colors; edge = black, as in the
# reference figures). Colors are assigned by a session type's position within
# its arena, so the groups compared within one arena are always distinct
# (cyan, magenta, green, red).
_HUES = [
    {'face': PAL_CYAN,    'edge': PAL_BLACK, 'dark': '#00B3B3',  'light': '#CCFDFD'},  # cyan
    {'face': PAL_MAGENTA, 'edge': PAL_BLACK, 'dark': '#C000C0',  'light': '#FDCCFD'},  # magenta
    {'face': '#7FE07F',   'edge': PAL_BLACK, 'dark': PAL_GREEN,  'light': '#CCF2CC'},  # green
    {'face': '#FF7F7F',   'edge': PAL_BLACK, 'dark': PAL_RED,    'light': '#FFCCCC'},  # red
]
SESSION_COLORS = {t: _HUES[i]
                  for ts in ARENA_SESSION_TYPES.values() for i, t in enumerate(ts)}
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


def _sig_pairs(posthoc_df, x_of: dict) -> list:
    """(x1, x2, stars) for each post-hoc pair with a significant Holm-corrected
    p-value (x1 < x2). `x_of` maps a session type to its x position."""
    pairs = []
    if posthoc_df is None or not len(posthoc_df):
        return pairs
    for _, r in posthoc_df.iterrows():
        stars = _stars(r.get('p_holm'))
        if stars in ('', 'n.s.'):
            continue
        g1, g2 = str(r['comparison']).split(' vs ')
        if g1 in x_of and g2 in x_of:
            x1, x2 = sorted((x_of[g1], x_of[g2]))
            pairs.append((x1, x2, stars))
    return pairs


def _assign_levels(pairs: list) -> list:
    """Stack brackets so overlapping ones sit on different levels (narrowest
    lowest). Returns (x1, x2, stars, level) tuples."""
    placed = []
    for x1, x2, stars in sorted(pairs, key=lambda t: t[1] - t[0]):
        level = 0
        while any(l == level and x1 <= b2 and x2 >= b1 for b1, b2, _s, l in placed):
            level += 1
        placed.append((x1, x2, stars, level))
    return placed


def _draw_sig_brackets(ax, placed: list, y_base: float, step: float, inset: float = 0.0):
    """One horizontal line per significant pair, with its stars above it."""
    for x1, x2, stars, level in placed:
        y = y_base + level * step
        ax.plot([x1 + inset, x2 - inset], [y, y], color=PAL_BLACK, linewidth=1.5,
                solid_capstyle='butt', clip_on=False, zorder=5)
        ax.annotate(stars, ((x1 + x2) / 2, y), xytext=(0, -2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=FS_TICK + 3, fontweight='bold',
                    annotation_clip=False)


def _add_sig_brackets(ax, posthoc_df, x_of: dict, inset: float = 0.05):
    """Draw significance brackets above the data of `ax` and extend the y-axis
    to make room for them. Call after everything else is plotted."""
    placed = _assign_levels(_sig_pairs(posthoc_df, x_of))
    if not placed:
        return
    lo, hi = ax.get_ylim()
    ticks = [t for t in ax.get_yticks() if lo <= t <= hi]
    rng = hi - lo
    step = 0.10 * rng
    y_base = hi + 0.02 * rng
    _draw_sig_brackets(ax, placed, y_base, step, inset)
    ax.set_yticks(ticks)
    ax.set_ylim(lo, y_base + (max(p[3] for p in placed) + 1) * step)


def _pair_label(cmp: str) -> str:
    return ' vs '.join(_label(g) for g in str(cmp).split(' vs '))


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
    sym = 'H' if 'Kruskal' in test else 'χ²'
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
            elif 'Wilcoxon' in r['test']:
                stat_str = f'z = {r["statistic"]:.3f}'
            else:
                stat_str = f'χ² = {r["statistic"]:.3f}'
            lines.append(f'{_pair_label(r["comparison"])}: {stat_str}, p = {_fmt_p(r["p_raw"])}, '
                         f'p(Holm) = {_fmt_p(r["p_holm"])} {_stars(r["p_holm"])}')
    return lines


def _continuous_stats_lines(desc_df, omnibus, posthoc_df) -> list:
    lines = []
    for _, r in desc_df.iterrows():
        sd = f'{r["std"]:.3g}' if pd.notna(r['std']) else 'n/a'
        lines.append(f'{_label(r["session_type"])} (n={int(r["n"])}): mean {r["mean"]:.3g} ± {sd} SD | '
                     f'median {r["median"]:.3g} [Q1 {r["q1"]:.3g}, Q3 {r["q3"]:.3g}]')
    lines += _test_lines(omnibus, posthoc_df)
    if omnibus.get('test') != 'insufficient data':
        lines.append(f'Shapiro-Wilk, all groups normal: {"yes" if omnibus.get("normal_all_groups") else "no"}; '
                     f'Levene equal-variance p = {_fmt_p(omnibus.get("equal_variance_p"))} '
                     f'(reference only; tests are non-parametric)')
    return lines


def _categorical_stats_lines(desc_df, omnibus, posthoc_df) -> list:
    lines = [f'{_label(r["session_type"])} (n={int(r["n"])}): {int(r["n_true"])} yes / '
             f'{int(r["n_false"])} no = {r["pct_true"]:.1f} %'
             for _, r in desc_df.iterrows()]
    return lines + _test_lines(omnibus, posthoc_df)


def _speed_dir_stats_lines(desc_df, omnibus, posthoc_df) -> list:
    lines = [f'{_label(r["session_type"])} (n={int(r["n"])}): '
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
    c = SESSION_COLORS.get(arena, _DEFAULT_COLOR)
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
                out.append((c['dark'], c['edge']) if v > 0 else (c['light'], c['edge']))
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
                out.append((c['dark'], c['edge']))
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
    for a in SESSION_TYPE_ORDER:
        if a not in df['session_type'].unique():
            continue
        sub = df.loc[df['session_type'] == a].copy()
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
        c = SESSION_COLORS.get(arena, _DEFAULT_COLOR)
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

    if has_legend:
        _add_significance_legend(ax_info, metric_col)
    _draw_stats(ax_info, stats_lines)

    tick_labels = [f'{_label(a)}\n(n={len(subsets[a])})' for a in present]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel(textwrap.fill(metric_label, 30), fontsize=FS_LABEL, labelpad=8)
    _set_title(ax, metric_label)
    _style_axes(ax)
    _add_sig_brackets(ax, posthoc_df, dict(zip(present, positions)))
    fig.tight_layout()
    _save(fig, out_path)


def run_all_comparisons(df: pd.DataFrame, metrics: dict, plot_prefix: str,
                        plots_dir: str):
    all_desc, all_omnibus, all_posthoc = [], [], []
    os.makedirs(plots_dir, exist_ok=True)
    for col, label in metrics.items():
        desc_df, omnibus, posthoc_df = compare_groups(df, col, label)
        all_desc.append(desc_df)
        all_omnibus.append(omnibus)
        if len(posthoc_df):
            all_posthoc.append(posthoc_df)
        plot_metric(df, col, label,
                    os.path.join(plots_dir, f'{plot_prefix}_{col}.png'),
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
    return _tag_session_type(full, 'unit(s)')


def compare_categorical(df: pd.DataFrame, col: str, label: str):
    """Descriptive counts, omnibus chi-square test, and pairwise Fisher's
    exact post-hoc tests for one True/False outcome across arena types."""
    rows, table, present = [], [], []
    for arena in SESSION_TYPE_ORDER:
        sub = df.loc[df['session_type'] == arena, col].apply(_to_tri_bool).dropna()
        n_total = len(sub)
        if n_total == 0:
            continue
        n_true = int((sub == True).sum())  # noqa: E712
        present.append(arena)
        table.append([n_true, n_total - n_true])
        rows.append({'metric': label, 'session_type': arena, 'n': n_total,
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


def plot_categorical(df: pd.DataFrame, col: str, label: str, out_path: str,
                     stats_lines=None, posthoc_df=None):
    present, pct_true, ns = [], [], []
    for a in SESSION_TYPE_ORDER:
        sub = df.loc[df['session_type'] == a, col].apply(_to_tri_bool).dropna()
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
    faces = [SESSION_COLORS.get(a, _DEFAULT_COLOR)['face'] for a in present]
    edges = [SESSION_COLORS.get(a, _DEFAULT_COLOR)['edge'] for a in present]
    ax.bar(positions, pct_true, color=faces, edgecolor=edges, linewidth=1.2,
           width=0.6, zorder=3)

    tick_labels = [f'{_label(a)}\n(n={n})' for a, n in zip(present, ns)]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel('% of cells', fontsize=FS_LABEL, labelpad=8)
    ax.set_ylim(0, 100)
    _set_title(ax, label)
    _style_axes(ax)
    _add_sig_brackets(ax, posthoc_df, dict(zip(present, positions)))
    fig.tight_layout()
    _save(fig, out_path)


def compare_speed_direction(df: pd.DataFrame):
    """3-way (p-Speed / n-Speed / non-speed) classification of the
    instantaneous speed score across arena types -- a chi-square test of
    independence on the arena_type x {p-Speed, n-Speed, non-speed} table,
    since 'speed direction' isn't a simple True/False outcome."""
    label = 'Speed-direction classification (p-Speed / n-Speed / non-speed, % of place cells)'
    rows, table, present = [], [], []
    for arena in SESSION_TYPE_ORDER:
        sub = df.loc[df['session_type'] == arena]
        known = sub['speed_modulated_td'].apply(_to_tri_bool).notna()
        n_total = int(known.sum())
        if n_total == 0:
            continue
        n_p = int((sub['p_speed'].apply(_to_tri_bool) == True).sum())  # noqa: E712
        n_n = int((sub['n_speed'].apply(_to_tri_bool) == True).sum())  # noqa: E712
        n_non = n_total - n_p - n_n
        present.append(arena)
        table.append([n_p, n_n, n_non])
        rows.append({'metric': label, 'session_type': arena, 'n': n_total,
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
    for a in SESSION_TYPE_ORDER:
        sub = df.loc[df['session_type'] == a]
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

    fig, ax, ax_info = _new_fig((7, 5.5), legend_rows=1, stats_lines=stats_lines)

    positions = np.arange(len(present))
    dark   = [SESSION_COLORS.get(a, _DEFAULT_COLOR)['dark']  for a in present]
    light  = [SESSION_COLORS.get(a, _DEFAULT_COLOR)['light'] for a in present]
    edges  = [SESSION_COLORS.get(a, _DEFAULT_COLOR)['edge']  for a in present]
    bottoms2 = [p + n for p, n in zip(pct_p, pct_n)]
    ax.bar(positions, pct_p, color=dark, edgecolor=edges, linewidth=1.2,
           width=0.6, zorder=3, label='p-Speed')
    ax.bar(positions, pct_n, bottom=pct_p, color=light, edgecolor=edges, linewidth=1.2,
           width=0.6, zorder=3, label='n-Speed')
    ax.bar(positions, pct_non, bottom=bottoms2, color=SIG_GRAY['face'],
           edgecolor=SIG_GRAY['edge'], linewidth=1.2, width=0.6, zorder=3, label='non-speed')

    tick_labels = [f'{_label(a)}\n(n={n})' for a, n in zip(present, ns)]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel('% of place cells', fontsize=FS_LABEL, labelpad=8)
    ax.set_ylim(0, 100)
    _set_title(ax, 'Speed-direction classification (instantaneous)')
    _style_axes(ax)
    _add_sig_brackets(ax, posthoc_df, dict(zip(present, positions)))
    handles, labels = ax.get_legend_handles_labels()
    ax_info.legend(handles, labels, loc='upper center', fontsize=FS_LEGEND, ncol=3)
    _draw_stats(ax_info, stats_lines)
    fig.tight_layout()
    _save(fig, out_path)


# ── Histograms by arena type ────────────────────────────────────────────────
# Histogram plots of the place-cell characterization metrics, split by arena
# type. Run at the end of __main__ on the SAME place-cell populations
# (full_df / field_df) analyzed above.

N_BINS   = 15

SPEED_METRICS = {
    'speed_score':    'Speed score (binning procedure)',
    'speed_score_td': 'Speed score (instantaneous procedure)',
}
HIST_FULL_METRICS = {k: v for k, v in FULL_METRICS.items() if k not in SPEED_METRICS}

# Explicit p-type/n-type speed-cell colors (green/red) -- independent of the
# per-arena SESSION_COLORS palette used elsewhere.
P_COLOR = {'face': PAL_GREEN, 'edge': PAL_BLACK}  # p-type (positive) speed cells
N_COLOR = {'face': PAL_RED,   'edge': PAL_BLACK}  # n-type (negative) speed cells


def plot_histogram_by_session_type(df: pd.DataFrame, metric_col: str, metric_label: str,
                             out_path: str):
    """One figure per metric: histograms overlaid for each arena type present,
    on shared bins so the three distributions are directly comparable."""
    values = {}
    for arena in SESSION_TYPE_ORDER:
        vals = pd.to_numeric(df.loc[df['session_type'] == arena, metric_col],
                              errors='coerce').dropna().to_numpy(dtype=float)
        if len(vals):
            values[arena] = vals
    if not values:
        return

    all_vals = np.concatenate(list(values.values()))
    bins = np.histogram_bin_edges(all_vals, bins=N_BINS)

    desc_df, omnibus, posthoc_df = compare_groups(df, metric_col, metric_label)
    stats_lines = _continuous_stats_lines(desc_df, omnibus, posthoc_df)
    fig, ax, ax_info = _new_fig((7, 5), legend_rows=(len(values) + 2) // 2,
                                stats_lines=stats_lines)

    # Overlaid groups: translucent tinted fill plus a full-strength outline in
    # the group's dark tone (black outlines would be unreadable where 3-4
    # distributions overlap).
    for arena, vals in values.items():
        c = SESSION_COLORS.get(arena, _DEFAULT_COLOR)
        ax.hist(vals, bins=bins, histtype='stepfilled', facecolor=c['face'],
                edgecolor='none', alpha=0.6, zorder=3)
        ax.hist(vals, bins=bins, histtype='step', edgecolor=c['dark'],
                linewidth=1.5, zorder=4)

    # Overlaid distributions have no separate plots to bracket, so significance
    # brackets connect the group medians (dashed lines).
    y_data_top = ax.get_ylim()[1]
    for arena, vals in values.items():
        ax.vlines(np.median(vals), 0, y_data_top,
                  colors=SESSION_COLORS.get(arena, _DEFAULT_COLOR)['dark'],
                  linestyles='--', linewidth=1.5, zorder=5)
    _add_sig_brackets(ax, posthoc_df, {a: np.median(v) for a, v in values.items()},
                      inset=0.0)

    handles = [Patch(facecolor=SESSION_COLORS.get(a, _DEFAULT_COLOR)['face'],
                     edgecolor=SESSION_COLORS.get(a, _DEFAULT_COLOR)['dark'],
                     linewidth=1.5, label=f'{_label(a)} (n={len(v)})')
               for a, v in values.items()]
    handles.append(Line2D([0], [0], color=PAL_BLACK, linestyle='--', linewidth=1.5,
                          label='group median'))
    ax_info.legend(handles=handles, loc='upper center', fontsize=FS_LEGEND,
                   ncol=2)
    _draw_stats(ax_info, stats_lines)

    ax.set_xlabel(metric_label, fontsize=FS_LABEL, labelpad=8)
    ax.set_ylabel('Number of cells', fontsize=FS_LABEL, labelpad=8)
    _set_title(ax, metric_label)
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, out_path)


def _speed_type_masks(sub: pd.DataFrame):
    """Boolean (p_mask, n_mask) numpy arrays for a sub-dataframe, from its
    p_speed / n_speed columns."""
    p_mask = sub['p_speed'].apply(_to_tri_bool) == True   # noqa: E712
    n_mask = sub['n_speed'].apply(_to_tri_bool) == True   # noqa: E712
    return p_mask.to_numpy(), n_mask.to_numpy()


def plot_speed_histograms_by_session_type(df: pd.DataFrame, metric_col: str,
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

    present = [a for a in SESSION_TYPE_ORDER if (sub_all['session_type'] == a).any()]
    if not present:
        print(f'  No cells passed both shuffle tests for {metric_label}; skipping plot.')
        return

    bins = np.histogram_bin_edges(sub_all[metric_col].to_numpy(dtype=float), bins=N_BINS)

    # Stats text: descriptives per session type and p/n type, then a
    # Kruskal-Wallis + Holm-corrected Wilcoxon rank-sum comparison of the
    # pooled speed-modulated cells across session types.
    split = {}
    stats_lines = []
    for arena in present:
        sub = sub_all.loc[sub_all['session_type'] == arena]
        p_mask, n_mask = _speed_type_masks(sub)
        split[arena] = (sub.loc[p_mask, metric_col].to_numpy(dtype=float),
                        sub.loc[n_mask, metric_col].to_numpy(dtype=float))
        for name, v in zip(('p-type', 'n-type'), split[arena]):
            stats_lines.append(f'{_label(arena)} {name} (n={len(v)}): {_desc_str(v)}')
    _, omnibus, posthoc_df = compare_groups(sub_all, metric_col, metric_label)
    stats_lines.append('Across session types (all speed-modulated cells pooled):')
    stats_lines += _test_lines(omnibus, posthoc_df)

    # Significance brackets above the panel titles (a thin axes row on top);
    # bracket levels are worked out from panel indices now, and drawn at the
    # panels' real x positions once the layout is final.
    idx_of = {a: i for i, a in enumerate(present)}
    levels = _assign_levels(_sig_pairs(posthoc_df, idx_of))
    n_levels = max((l for *_r, l in levels), default=-1) + 1
    sig_h = n_levels * 0.4 + 0.05

    n_pan   = len(present)
    panel_h = len(stats_lines) * STATS_LINE_IN + 0.4
    fig = plt.figure(figsize=(6 * n_pan, sig_h + 5.5 + panel_h))
    gs = fig.add_gridspec(3, n_pan, height_ratios=[sig_h, 5.5, panel_h])
    ax_sig = fig.add_subplot(gs[0, :])
    ax_sig.axis('off')
    axes = []
    for i in range(n_pan):
        axes.append(fig.add_subplot(gs[1, i], sharey=axes[0] if axes else None))
        if i > 0:
            axes[i].tick_params(labelleft=False)
    ax_info = fig.add_subplot(gs[2, :])
    ax_info.axis('off')
    _draw_stats(ax_info, stats_lines)

    for ax, arena in zip(axes, present):
        sub = sub_all.loc[sub_all['session_type'] == arena]
        p_vals, n_vals = split[arena]

        ax.hist([p_vals, n_vals], bins=bins, stacked=True,
                color=[P_COLOR['face'], N_COLOR['face']],
                edgecolor=PAL_BLACK, linewidth=1.2, zorder=3)
        ax.set_title(f'{_label(arena)}\n(n={len(sub)}: {len(p_vals)} p-type, {len(n_vals)} n-type)',
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
    fig.suptitle(f'{metric_label}\n(cells passed both shuffle-testing procedures)',
                 fontsize=FS_TITLE, y=1.2)
    fig.tight_layout()

    if levels:
        # Use figure coordinates as the x data coordinates of the bracket row.
        pos = ax_sig.get_position()
        ax_sig.set_xlim(pos.x0, pos.x1)
        ax_sig.set_ylim(0, n_levels)
        centers = [(a.get_position().x0 + a.get_position().x1) / 2 for a in axes]
        placed = [(centers[int(x1)], centers[int(x2)], s, l) for x1, x2, s, l in levels]
        _draw_sig_brackets(ax_sig, placed, y_base=0.1, step=1.0,
                           inset=0.01 * (pos.x1 - pos.x0))
    _save(fig, out_path)


# ── Per-arena analysis ──────────────────────────────────────────────────────

# Extra continuous metric generated by the characterization pipeline but not
# covered by FULL_METRICS above.
EXTRA_FULL_METRICS = {'mean_fr': 'Mean firing rate'}

# Place-cell yield uses every recorded unit (place_cell True/False/None), not
# just confirmed place cells.
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


def _stability_sig(v):
    v = pd.to_numeric(v, errors='coerce')
    return bool(v <= ALPHA) if pd.notna(v) else None


def _print_significant(tbl: pd.DataFrame, title: str, metric_width: int, test_width: int):
    sig = tbl[tbl['significant'] == True]  # noqa: E712
    if not len(sig):
        print(f'\nNo significant differences found ({title}).')
        return
    print(f'\nSignificant differences, {title} (p < 0.05):')
    for _, row in sig.iterrows():
        stat = row['statistic']
        stat_str = f'{stat:.3f}' if pd.notna(stat) else 'nan'
        print(f"  {row['metric']:{metric_width}s} {row['test']:{test_width}s} "
              f"stat={stat_str}  p={row['p_value']:.4g}")


def run_arena(arena: str, full_df: pd.DataFrame, field_df: pd.DataFrame,
              all_units_df: pd.DataFrame) -> dict:
    """Run the whole session-type comparison for one arena type: statistics
    (appended to INPUT_EXCEL with an '_<arena>' sheet suffix), box/bar plots
    in PLOTS_DIR/<arena>, and histograms in PLOTS_DIR/<arena>/Histograms.
    Returns {sheet_name: table} for the combined CSV."""
    types = ARENA_SESSION_TYPES[arena]
    plots_dir = os.path.join(PLOTS_DIR, arena)
    hist_dir  = os.path.join(plots_dir, 'Histograms')

    # Restrict every population to this arena, so only its session types are
    # compared with each other.
    full_df      = full_df[full_df['arena_type'] == arena].copy()
    field_df     = field_df[field_df['arena_type'] == arena].copy()
    all_units_df = all_units_df[all_units_df['arena_type'] == arena].copy()

    print(f'\n{"=" * 70}\n{arena} arena -- session types: {types}\n{"=" * 70}')
    print('Place cells by session type (Full sheet):')
    print(full_df['session_type'].value_counts().reindex(types).fillna(0).astype(int))
    print('Place cells by session type (PlaceFields sheet, aggregated per cell):')
    print(field_df['session_type'].value_counts().reindex(types).fillna(0).astype(int))

    if not len(full_df):
        print(f'No place cells in the {arena} arena; skipping.')
        return {}

    os.makedirs(plots_dir, exist_ok=True)

    # ── Continuous metrics ──────────────────────────────────────────────────
    full_desc, full_omnibus, full_posthoc = run_all_comparisons(
        full_df, FULL_METRICS, plot_prefix='Full', plots_dir=plots_dir)
    field_desc, field_omnibus, field_posthoc = run_all_comparisons(
        field_df, FIELD_METRICS, plot_prefix='Fields', plots_dir=plots_dir)

    desc_df    = pd.concat([full_desc, field_desc], ignore_index=True)
    omnibus_df = pd.concat([full_omnibus, field_omnibus], ignore_index=True)
    posthoc_df = pd.concat([full_posthoc, field_posthoc], ignore_index=True)

    extra_desc, extra_omnibus, extra_posthoc = run_all_comparisons(
        full_df, EXTRA_FULL_METRICS, plot_prefix='Full', plots_dir=plots_dir)

    # ── Categorical (proportion) comparisons ────────────────────────────────
    full_df['stability_significant'] = full_df['stability_p_value'].apply(_stability_sig)

    cat_desc_all, cat_omni_all, cat_post_all = [], [], []
    for pop, metrics in ((all_units_df, CATEGORICAL_METRICS_ALL_UNITS),
                         (full_df, CATEGORICAL_METRICS_PLACE_CELLS)):
        for col, label in metrics.items():
            d, o, p = compare_categorical(pop, col, label)
            cat_desc_all.append(d)
            cat_omni_all.append(o)
            if len(p):
                cat_post_all.append(p)
            plot_categorical(pop, col, label,
                              os.path.join(plots_dir, f'Categorical_{col}.png'),
                              stats_lines=_categorical_stats_lines(d, o, p),
                              posthoc_df=p)

    cat_desc_df    = pd.concat(cat_desc_all, ignore_index=True) if cat_desc_all else pd.DataFrame()
    cat_omnibus_df = pd.DataFrame(cat_omni_all)
    cat_posthoc_df = pd.concat(cat_post_all, ignore_index=True) if cat_post_all else pd.DataFrame()

    speed_dir_desc, speed_dir_omnibus, speed_dir_posthoc = compare_speed_direction(full_df)
    plot_speed_direction(full_df, os.path.join(plots_dir, 'Categorical_speed_direction.png'),
                         stats_lines=_speed_dir_stats_lines(speed_dir_desc, speed_dir_omnibus,
                                                            speed_dir_posthoc),
                         posthoc_df=speed_dir_posthoc)
    speed_dir_omnibus_df = pd.DataFrame([speed_dir_omnibus])

    # ── Write to the workbook ───────────────────────────────────────────────
    # mode='a' + if_sheet_exists='replace' appends these as new sheets in the
    # SAME workbook that was read from INPUT_EXCEL, overwriting only sheets
    # with a matching name (e.g. on a re-run) and leaving every other sheet
    # untouched. Sheet names are capped at 31 characters by Excel.
    tables = {
        f'Descriptives_{arena}':             desc_df,
        f'OmnibusTests_{arena}':             omnibus_df,
        f'PostHoc_{arena}':                  posthoc_df,
        f'Descriptives_Extra_{arena}':       extra_desc,
        f'OmnibusTests_Extra_{arena}':       extra_omnibus,
        f'PostHoc_Extra_{arena}':            extra_posthoc,
        f'CategoricalDescriptives_{arena}':  cat_desc_df,
        f'CategoricalOmnibus_{arena}':       cat_omnibus_df,
        f'CategoricalPostHoc_{arena}':       cat_posthoc_df,
        f'SpeedDirection_Desc_{arena}':      speed_dir_desc,
        f'SpeedDirection_Omnibus_{arena}':   speed_dir_omnibus_df,
        f'SpeedDirection_PostHoc_{arena}':   speed_dir_posthoc,
    }
    with pd.ExcelWriter(INPUT_EXCEL, engine='openpyxl', mode='a',
                        if_sheet_exists='replace') as writer:
        for sheet_name, tbl in tables.items():
            tbl.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f'\nStatistics for {arena} appended to {INPUT_EXCEL}')
    print(f'Box/bar plots saved to {plots_dir}')

    _print_significant(omnibus_df, f'{arena}, continuous metrics', 45, 16)
    _print_significant(
        pd.concat([extra_omnibus, cat_omnibus_df, speed_dir_omnibus_df], ignore_index=True),
        f'{arena}, extra/categorical metrics', 65, 30)

    # ── Histograms by session type ──────────────────────────────────────────
    os.makedirs(hist_dir, exist_ok=True)

    print(f'\nPlotting {arena} Full-sheet metric histograms by session type...')
    for col, label in HIST_FULL_METRICS.items():
        plot_histogram_by_session_type(full_df, col, label,
                                        os.path.join(hist_dir, f'Hist_Full_{col}.png'))

    print(f'Plotting {arena} PlaceFields-sheet metric histograms by session type...')
    for col, label in FIELD_METRICS.items():
        plot_histogram_by_session_type(field_df, col, label,
                                        os.path.join(hist_dir, f'Hist_Fields_{col}.png'))

    print(f'Plotting {arena} speed-score histograms (p-type/n-type, both shuffle tests passed)...')
    for col, label in SPEED_METRICS.items():
        plot_speed_histograms_by_session_type(full_df, col, label,
                                               os.path.join(hist_dir, f'Hist_Speed_{col}.png'))
    print(f'Histogram plots for {arena} saved to {hist_dir}')

    return tables


if __name__ == '__main__':
    print(f'Loading {INPUT_EXCEL} ...')
    full, fields = load_sheets()

    full_df      = build_full_metrics_df(full)
    field_df     = build_field_metrics_df(full, fields)
    all_units_df = build_all_units_df(full)

    print('\nPlace cells by arena type (Full sheet):')
    print(full_df['arena_type'].value_counts().reindex(ARENA_TYPES).fillna(0).astype(int))

    all_tables = {}
    for arena in ARENA_TYPES:
        for sheet_name, tbl in run_arena(arena, full_df, field_df, all_units_df).items():
            all_tables[sheet_name] = tbl

    # ── All statistics in one CSV ───────────────────────────────────────────
    # Stacks every statistics table written to the workbook into one
    # long-format CSV. 'table' names the source sheet; the tables have
    # different columns, so cells that don't apply to a table are left blank.
    stats_csv_parts = []
    for sheet_name, tbl in all_tables.items():
        if tbl is None or not len(tbl):
            continue
        tbl = tbl.copy()
        tbl.insert(0, 'table', sheet_name)
        stats_csv_parts.append(tbl)

    if stats_csv_parts:
        STATS_CSV = os.path.splitext(INPUT_EXCEL)[0] + '_SessionType_AllStats.csv'
        pd.concat(stats_csv_parts, ignore_index=True).to_csv(
            STATS_CSV, index=False, encoding='utf-8-sig')
        print(f'\nAll statistics saved to {STATS_CSV}')

    print(f'\nDone. Plots saved under {PLOTS_DIR}')