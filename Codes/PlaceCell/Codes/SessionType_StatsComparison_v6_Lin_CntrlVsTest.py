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
    - final_speed_score Speed score (mean of the binned and instantaneous
                        scores). Speed cells are those with speed_cell == True:
                        p-type if final_speed_score > 0, n-type if < 0. Cells
                        with speed_cell False / 'not tested' are non-speed.

  Sheet 'PlaceFields' (aggregated per cell):
    - n_fields         Number of place fields per cell
    - total_area_cm2   Area occupied by fields per cell (sum of field areas)
    - pct_area         Percentage area occupied by fields per cell (sum of
                        each field's pct_of_occupied_area)

Repeated measures: data are nested as animal > arena > recording day >
session (folders: animal ID \\ arena type \\ recording day \\ session type,
with the neural and tracking files inside the session folder). The animal,
arena, day and session type are each recognized by name wherever they sit in
the path, so the folder order itself doesn't matter. Sessions of one day are sorted together, so the same unit label
(e.g. TT1_SS_12_SS_01.ntt) within animal + arena + day is the same cell
recorded in several sessions (cell_id, see CellClusteredStats_Utils.add_cell_ids);
the same label on another day is a different cell. Those rows are not
independent, and not every cell is a place cell in every session (partially
paired data), so the tests account for the nesting (CellClusteredStats_Utils.py):

  - continuous metrics: linear mixed model (on ranks, METRIC_TRANSFORM)
        metric ~ session_type + (1|animal) + (1|animal:day) + (1|animal:day:cell)
    (levels the data can't support, e.g. animal with one animal, are left
    out); omnibus likelihood-ratio test, pairwise Wald contrasts
    (compare_continuous_nested).
    Alternatively (CONTINUOUS_TEST = 'KW'): Kruskal-Wallis + pairwise
    Mann-Whitney U, Holm-Bonferroni corrected, assuming independent rows.
  - % outcomes: logistic GEE clustered by cell (cluster-robust SEs); omnibus
    Wald test, pairwise Wald contrasts (odds ratios).
  - speed direction (p/n/non): multinomial GEE clustered by cell.

Pairwise p-values are Holm-Bonferroni corrected within each metric. For the
% outcomes, if no cell is repeated (e.g. sessions sorted separately, as in
Circle) or a group is at 0 % / 100 %, chi-square + Fisher's exact are used
instead; the 'model_note' column records which test was used and why.

Only the arenas in ANALYZE_ARENAS are analyzed (Linear track by default).

First vs second half of the session (sheets 'First_Half' / 'Second_Half',
same columns as 'Full', one row per session + unit):

  - Between session types: the same session-type comparison as above is run
    on each half separately, for the metrics the half sheets hold (peak/mean
    firing rate, SIR, sparsity, coherence; theta modulation, coherence shuffle
    significance, place-cell yield). Stability, speed and field metrics are
    not computed per half. Place-cell population = whole-session place cells
    (their half-session values); place-cell yield uses each half's own
    place_cell classification of all units.
  - Within session: for each session type, the same cells' 1st half vs 2nd
    half values are compared with a paired test: Wilcoxon signed-rank
    (continuous metrics) / exact McNemar (True/False outcomes). The whole
    session is left out (both halves are subsets of it, so comparing against
    it is not meaningful); with two conditions there is no omnibus test or
    multiple-comparison correction. Cells missing a value in either half are
    excluded from that test (complete cases). Plots: 1st half = session
    color, 2nd half = light cyan; significance bracket only where the test is
    significant.
  - Pooled over session types: the same paired tests with every session
    type's cells clubbed into one group ('All'): all 1st halves vs all 2nd
    halves. Sheets 'Pooled_*_<arena>', plots in HalfSession/Pooled. A cell
    recorded in several sessions contributes one row per session (rows
    treated as independent units).

Parameters to edit (below):
    INPUT_EXCEL  = path to the output_excel file from
                   PlaceCellChar_FieldDetect_Main_v3.py -- also where these
                   statistics sheets get appended (in place)
    PLOTS_DIR    = folder to save comparison plots to (one sub-folder per arena)
    ANALYZE_ARENAS = arena types to analyze
"""

import os
import re
import shutil
import textwrap
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

import CellClusteredStats_Utils as ccs

# ── Parameters ──────────────────────────────────────────────────────────────

INPUT_EXCEL = r'X:\NMR_group_data\Runita\Analysis\Mean_KDE_Open_PascalOldenburg\LinearTrack_GeoMagVsZero\Data_Control\All_TT_PlaceChar_AdptBin.xlsx'
PLOTS_DIR   = r'X:\NMR_group_data\Runita\Analysis\Mean_KDE_Open_PascalOldenburg\LinearTrack_GeoMagVsZero\Data_Control\SessionType_StatsPlots_Lin_FldVsZero'


ARENA_TYPES = ['Open', 'Linear', 'Circle']
# Arenas actually analyzed (the others are recognized but skipped).
ANALYZE_ARENAS = ['Linear']

# Half-session sheets of INPUT_EXCEL, keyed by condition name.
HALF_SHEETS = {'H1': 'First_Half', 'H2': 'Second_Half'}

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

# Animal ID for rows whose session path names no animal (a single-animal
# workbook whose paths start at the arena folder, e.g. 'Open\\Day10\\1_Cntrl').
# Needed so the same unit label within one day is recognized as the same cell.
# Set to None if such a workbook holds several animals.
ANIMAL_ID_IF_MISSING = 'Animal_1'

# Test for the continuous metrics:
#   'LMM' : nested linear mixed model (accounts for repeated cells, days and
#           animals; see compare_continuous_nested)
#   'KW'  : Kruskal-Wallis omnibus + pairwise Mann-Whitney U post-hoc, Holm-
#           Bonferroni corrected -- ASSUMES every row is independent (ignores
#           cells recorded in several sessions of one day)
CONTINUOUS_TEST = 'KW'

# With CONTINUOUS_TEST = 'LMM', the model is fitted to 'rank' (the ranks of
# the metric; robust, the mixed-model analogue of Kruskal-Wallis) or 'none'
# (the raw values).
METRIC_TRANSFORM = 'rank'

FULL_METRICS = {
    'peak_fr':         'Peak firing rate',
    'sir':             'Spatial information score',
    'sparsity':        'Sparsity',
    'coherence':       'Coherence score',
    'stability_score':   'Stability score',
    'final_speed_score': 'Speed score',
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
    """Add 'arena_type', 'session_type' and 'cell_id' (+ 'animal', 'rec_day')
    columns to a dataframe with 'session' and 'unit' columns, warn about (and
    drop) rows where the arena/session type isn't recognized."""
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
    return ccs.add_cell_ids(df.dropna(subset=['arena_type', 'session_type']),
                            include_arena=True, default_animal=ANIMAL_ID_IF_MISSING)


def _label(session_type: str) -> str:
    """Display label for plots: Linear session types are angles."""
    return f'{session_type}°' if session_type.isdigit() else session_type


def load_sheets():
    full   = pd.read_excel(INPUT_EXCEL, sheet_name='Full')
    fields = pd.read_excel(INPUT_EXCEL, sheet_name='PlaceFields')
    return full, fields


def build_full_metrics_df(full: pd.DataFrame) -> pd.DataFrame:
    df = full[full['place_cell'].apply(_to_bool)]
    return add_speed_class(_tag_session_type(df, 'place cell(s)'))


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
    metric across session types, with a mixed model accounting for cells
    recorded in several sessions and for cells sharing a recording day and
    animal (see CellClusteredStats_Utils.compare_continuous_nested). Returns
    (desc_df, omnibus_row, posthoc_df)."""
    groups, n_units = {}, {}
    for arena in SESSION_TYPE_ORDER:
        sub = df.loc[df['session_type'] == arena,
                     [metric_col, 'cell_id', 'animal', 'rec_day']].copy()
        sub[metric_col] = pd.to_numeric(sub[metric_col], errors='coerce')
        sub = sub.dropna(subset=[metric_col])
        if len(sub):
            groups[arena] = sub[metric_col].to_numpy(dtype=float)
            n_units[arena] = (sub['cell_id'].nunique(), sub['animal'].nunique(),
                              sub[['animal', 'rec_day']].drop_duplicates().shape[0])

    desc_rows = []
    for arena, vals in groups.items():
        desc_rows.append({
            'metric': metric_label, 'session_type': arena, 'n': len(vals),
            'n_cells': n_units[arena][0], 'n_animals': n_units[arena][1],
            'n_days': n_units[arena][2],
            'mean': np.mean(vals), 'median': np.median(vals),
            'q1': np.percentile(vals, 25), 'q3': np.percentile(vals, 75),
            'std': np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
            'sem': stats.sem(vals) if len(vals) > 1 else np.nan,
            'min': np.min(vals), 'max': np.max(vals),
        })
    desc_df = pd.DataFrame(desc_rows)

    omnibus = {'metric': metric_label, 'n_groups': len(groups),
               'groups': ', '.join(f'{a} (n={len(v)})' for a, v in groups.items())}

    if CONTINUOUS_TEST == 'KW':
        omni, posthoc_rows = None, []
        if len(groups) >= 2 and all(len(v) >= 2 for v in groups.values()):
            sub = df[df['session_type'].isin(list(groups))].copy()
            sub[metric_col] = pd.to_numeric(sub[metric_col], errors='coerce')
            sub = sub.dropna(subset=[metric_col])
            omni, posthoc_rows = ccs._classic_continuous(
                groups, f'{ccs.repetition_summary(sub, "session_type")}; '
                        f'Kruskal-Wallis chosen, independence ASSUMED')
    elif CONTINUOUS_TEST == 'LMM':
        omni, posthoc_rows = ccs.compare_continuous_nested(df, 'session_type', SESSION_TYPE_ORDER,
                                                           metric_col, transform=METRIC_TRANSFORM)
    else:
        raise ValueError(f"CONTINUOUS_TEST must be 'LMM' or 'KW', not {CONTINUOUS_TEST!r}")
    if omni is None:
        omnibus.update({'test': 'insufficient data', 'statistic': np.nan,
                         'p_value': np.nan, 'significant': False})
        return desc_df, omnibus, pd.DataFrame()

    omnibus.update(omni)
    omnibus['significant'] = bool(omni['p_value'] < ALPHA)
    posthoc_df = pd.DataFrame([{'metric': metric_label, **r} for r in posthoc_rows])
    return desc_df, omnibus, posthoc_df



# ── Figure style (matches the Figure 3 place-cell histograms) ────────────────
# Arial, bold axis labels, small plain tick labels, thin black left/bottom
# axes with short outward ticks, solid-filled bars separated by thin white
# gaps, and a dashed median line labelled 'med=...'.
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

FS_LABEL, FS_TITLE, FS_TICK, FS_LEGEND, FS_STATS, FS_MED = 14, 15, 11, 11, 9, 10
STATS_LINE_IN = 0.2    # panel height per line of stats text (inches)
LEGEND_ROW_IN = 0.3    # panel height per row of legend entries (inches)
FIG_DPI = 500

AXIS_LW     = 0.8      # spines, ticks and the dashed median line
BOX_LW      = 1.0      # box-plot outlines, whiskers, caps; significance lines
MEDIAN_LW   = 1.5      # box-plot median
BAR_EDGE    = 'white'  # bar/histogram outlines: a thin white gap between bars
BAR_EDGE_LW = 0.6
BAR_WIDTH   = 0.6      # categorical bars, as a fraction of the bar spacing
GROUP_W     = 1.1      # box/bar plots: x-axis width per condition (inches)
AXIS_PAD_W  = 1.4      # box/bar plots: extra width for the y-axis label/ticks
STATS_CHARS_PER_IN = 15  # approx. characters of stats text per inch of width

plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    FIG_FONT,
    'pdf.fonttype':       42,       # keep text editable in Illustrator
    'ps.fonttype':        42,
    'svg.fonttype':       'none',
    'text.color':         PAL_BLACK,
    'axes.labelcolor':    PAL_BLACK,
    'axes.labelweight':   'bold',
    'axes.titleweight':   'bold',
    'axes.edgecolor':     PAL_BLACK,
    'axes.linewidth':     AXIS_LW,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'xtick.color':        PAL_BLACK,
    'ytick.color':        PAL_BLACK,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'xtick.major.width':  AXIS_LW,
    'ytick.major.width':  AXIS_LW,
    'xtick.major.size':   3.5,
    'ytick.major.size':   3.5,
    'axes.facecolor':     'white',
    'figure.facecolor':   'white',
    'savefig.facecolor':  'white',
    'axes.grid':          False,
    'legend.frameon':     False,
})

# Per-session-type colors: shades of the arena's hue (blue = open arena,
# cyan = linear track, magenta = circular track). face = fill for
# boxes/bars/histograms/points; dark / light = deeper and paler tones of that
# shade, used for p-type / n-type speed cells (see _point_colors,
# plot_speed_direction); edge = point outline.
def _shade(face, dark, light):
    return {'face': face, 'edge': PAL_BLACK, 'dark': dark, 'light': light}

SESSION_COLORS = {
    # Open: Cntrl light blue, Rotate dark blue, Zero gray
    'Cntrl':  _shade(PAL_BLUE,   PAL_DKBLUE, '#CCE1F5'),
    'Rotate': _shade(PAL_DKBLUE, '#003D7A',  '#99C2EB'),
    'Zero':   _shade('#BFBFBF',  '#404040',  '#E6E6E6'),
    # Linear: 0 darkest cyan -> 270 lightest
    '0':      _shade('#008080',  '#004D4D',  '#99CCCC'),
    '90':     _shade('#00B3B3',  '#006B6B',  '#99E1E1'),
    '180':    _shade('#33D6D6',  '#1F8080',  '#ADEFEF'),
    '270':    _shade(PAL_CYAN,   '#00B3B3',  '#CCFDFD'),
    # Circle: Stnd lightest magenta, NoRot lighter, Rot dark
    'Stnd':   _shade('#FDC4FD',  '#D94FD9',  '#FEE7FE'),
    'NoRot':  _shade(PAL_MAGENTA, '#C000C0', '#FDCCFD'),
    'Rot':    _shade('#C000C0',  '#730073',  '#E699E6'),
}
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


def _bar_fig_width(n_groups: int) -> float:
    """Figure width for a box/bar plot with `n_groups` conditions: a fixed
    width per condition, so boxes/bars and the gaps between them keep their
    proportions whatever the number of conditions."""
    return n_groups * GROUP_W + AXIS_PAD_W


def _wrap_stats(stats_lines, fig_w: float) -> list:
    """Re-wrap the stats text lines to fit a figure `fig_w` inches wide."""
    width = max(40, int(fig_w * STATS_CHARS_PER_IN))
    return [w for line in (stats_lines or []) for w in (textwrap.wrap(line, width) or [''])]


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
        ax.plot([x1 + inset, x2 - inset], [y, y], color=PAL_BLACK, linewidth=BOX_LW,
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
    dof = omnibus.get('dof')
    dof_str = f', dof = {int(dof)}' if dof is not None and pd.notna(dof) else ''
    p = omnibus.get('p_value')
    lines = [f'{test}: {omnibus.get("stat_name", "stat")} = {omnibus["statistic"]:.3f}{dof_str}, '
             f'p = {_fmt_p(p)} {_stars(p)}']
    if posthoc_df is not None and len(posthoc_df):
        lines.append(f'Post-hoc ({posthoc_df["test"].iloc[0]}, Holm-Bonferroni corrected):')
        for _, r in posthoc_df.iterrows():
            fmt = '.3g' if r['stat_name'] == 'OR' else '.1f' if r['stat_name'] == 'U' else '.3f'
            stat_str = f'{r["stat_name"]} = {r["statistic"]:{fmt}}'
            lines.append(f'{_pair_label(r["comparison"])}: {stat_str}, p = {_fmt_p(r["p_raw"])}, '
                         f'p(Holm) = {_fmt_p(r["p_holm"])} {_stars(r["p_holm"])}')
    return lines


def _continuous_stats_lines(desc_df, omnibus, posthoc_df) -> list:
    lines = []
    for _, r in desc_df.iterrows():
        sd = f'{r["std"]:.3g}' if pd.notna(r['std']) else 'n/a'
        lines.append(f'{_label(r["session_type"])} (n={int(r["n"])}): mean {r["mean"]:.3g} ± {sd} SD | '
                     f'median {r["median"]:.3g} [Q1 {r["q1"]:.3g}, Q3 {r["q3"]:.3g}]')
    return lines + _test_lines(omnibus, posthoc_df) + _model_note_lines(omnibus)


def _model_note_lines(omnibus: dict) -> list:
    """Which model was used, and on how many cells (one line; wrapped to the
    figure width by _wrap_stats)."""
    note = omnibus.get('model_note')
    return [note] if isinstance(note, str) and note else []


def _categorical_stats_lines(desc_df, omnibus, posthoc_df) -> list:
    lines = [f'{_label(r["session_type"])} (n={int(r["n"])}): {int(r["n_true"])} yes / '
             f'{int(r["n_false"])} no = {r["pct_true"]:.1f} %'
             for _, r in desc_df.iterrows()]
    return lines + _test_lines(omnibus, posthoc_df) + _model_note_lines(omnibus)


def _speed_dir_stats_lines(desc_df, omnibus, posthoc_df) -> list:
    lines = [f'{_label(r["session_type"])} (n={int(r["n"])}): '
             f'p-Speed {int(r["n_p_speed"])} ({r["pct_p_speed"]:.1f} %), '
             f'n-Speed {int(r["n_n_speed"])} ({r["pct_n_speed"]:.1f} %), '
             f'non-speed {int(r["n_non_speed"])} ({r["pct_non_speed"]:.1f} %)'
             for _, r in desc_df.iterrows()]
    return lines + _test_lines(omnibus, posthoc_df) + _model_note_lines(omnibus)


def _style_axes(ax):
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.spines[['top', 'right']].set_visible(False)


def _set_title(ax, text):
    # ~7 bold title characters per inch of figure width
    ax.set_title(textwrap.fill(text, max(20, int(ax.figure.get_figwidth() * 7))),
                 fontsize=FS_TITLE - 2, pad=12)


def _save(fig, out_path):
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)

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
    c = SESSION_COLORS.get(arena, _DEFAULT_COLOR)
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


def _add_significance_legend(ax_info, metric_col: str, ref: dict):
    """Draw the significance legend into the borderless panel below the plot,
    using the tones of the color dict `ref` as examples."""
    entries = _SIGNIFICANCE_LEGENDS.get(metric_col)
    if not entries:
        return
    tone_face = {'colored': ref['face'], 'dark': ref['dark'],
                 'light': ref['light'], 'gray': SIG_GRAY['face']}
    handles = [Line2D([0], [0], marker='o', linestyle='None', markersize=8,
                       markerfacecolor=tone_face[tone], markeredgecolor=PAL_BLACK,
                       markeredgewidth=0.5, label=label)
               for tone, label in entries]
    ax_info.legend(handles=handles, loc='upper center', fontsize=FS_LEGEND, ncol=1)


def plot_metric(df: pd.DataFrame, metric_col: str, metric_label: str, out_path: str,
                stats_lines=None, posthoc_df=None, box_face=None):
    """Box plot + jittered points per session type. `box_face` (a color)
    overrides the per-session box fill (e.g. gray for the 2nd half); the
    points keep their session colors."""
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
    fig_w = _bar_fig_width(len(present))
    stats_lines = _wrap_stats(stats_lines, fig_w)
    fig, ax, ax_info = _new_fig(
        (fig_w, 5.5), stats_lines=stats_lines,
        legend_rows=len(_SIGNIFICANCE_LEGENDS[metric_col]) if has_legend else 0)

    bp = ax.boxplot(data, positions=positions, showfliers=False,
                     patch_artist=True, widths=0.55,
                     boxprops=dict(linewidth=BOX_LW, color=PAL_BLACK),
                     whiskerprops=dict(linewidth=BOX_LW, color=PAL_BLACK),
                     capprops=dict(linewidth=BOX_LW, color=PAL_BLACK),
                     medianprops=dict(linewidth=MEDIAN_LW, color=PAL_BLACK),
                     zorder=2)
    for patch, arena in zip(bp['boxes'], present):
        patch.set_facecolor(box_face or SESSION_COLORS.get(arena, _DEFAULT_COLOR)['face'])
        patch.set_edgecolor(PAL_BLACK)

    rng = np.random.default_rng(0)
    for pos, arena, vals in zip(positions, present, data):
        sub = subsets[arena]
        colors = _point_colors(metric_col, sub, arena)
        faces  = [fc for fc, _ec in colors]
        edges  = [ec for _fc, ec in colors]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=22,
                   facecolors=faces, edgecolors=edges, linewidths=0.5, zorder=3)

    if has_legend:
        _add_significance_legend(ax_info, metric_col,
                                 SESSION_COLORS.get(present[0], _DEFAULT_COLOR))
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
                        plots_dir: str, box_face=None):
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
                    posthoc_df=posthoc_df, box_face=box_face)

    desc_df    = pd.concat(all_desc, ignore_index=True) if all_desc else pd.DataFrame()
    omnibus_df = pd.DataFrame(all_omnibus)
    posthoc_df = pd.concat(all_posthoc, ignore_index=True) if all_posthoc else pd.DataFrame()
    return desc_df, omnibus_df, posthoc_df


# ── Categorical (proportion) comparisons ────────────────────────────────────
# Boolean/tri-state outcomes generated by the characterization pipeline
# (theta modulation, shuffle/stability significance, speed modulation) that
# weren't covered above because they're proportions, not continuous metrics.
# Compared with a logistic GEE clustered by cell (omnibus Wald test, pairwise
# Wald contrasts, Holm-Bonferroni corrected) -- see
# CellClusteredStats_Utils.compare_binary, which falls back to chi-square + Fisher's
# exact when no cell is repeated or a group is at 0 % / 100 %.

def build_all_units_df(full: pd.DataFrame) -> pd.DataFrame:
    """Like build_full_metrics_df, but keeps every recorded unit (not just
    confirmed place cells) -- needed to compute place-cell yield per arena."""
    return _tag_session_type(full, 'unit(s)')


def compare_categorical(df: pd.DataFrame, col: str, label: str):
    """Descriptive counts, omnibus test and Holm-corrected pairwise tests for
    one True/False outcome across session types, accounting for cells
    recorded in several sessions (CellClusteredStats_Utils.compare_binary)."""
    omni, posthoc_rows, table = ccs.compare_binary(df, 'session_type', SESSION_TYPE_ORDER,
                                                   col, _to_tri_bool)
    present = [a for a in SESSION_TYPE_ORDER
               if (df.loc[df['session_type'] == a, col].apply(_to_tri_bool).notna()).any()]
    rows = [{'metric': label, 'session_type': a, 'n': t + f, 'n_true': t, 'n_false': f,
             'pct_true': 100.0 * t / (t + f)}
            for a, (t, f) in zip(present, table)]
    desc_df = pd.DataFrame(rows)

    omnibus = {'metric': label, 'n_groups': len(present),
               'groups': ', '.join(f'{a} (n={r["n"]})' for a, r in zip(present, rows))}

    if omni is None:
        omnibus.update({'test': 'insufficient data', 'statistic': np.nan,
                         'p_value': np.nan, 'dof': np.nan, 'significant': False})
        return desc_df, omnibus, pd.DataFrame()

    omnibus.update(omni)
    omnibus['significant'] = bool(pd.notna(omni['p_value']) and omni['p_value'] < ALPHA)
    posthoc_df = pd.DataFrame([{'metric': label, **r} for r in posthoc_rows])
    return desc_df, omnibus, posthoc_df


def plot_categorical(df: pd.DataFrame, col: str, label: str, out_path: str,
                     stats_lines=None, posthoc_df=None, box_face=None):
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

    fig_w = _bar_fig_width(len(present))
    stats_lines = _wrap_stats(stats_lines, fig_w)
    fig, ax, ax_info = _new_fig((fig_w, 5.5), stats_lines=stats_lines)
    _draw_stats(ax_info, stats_lines)

    positions = np.arange(len(present))
    faces = [box_face or SESSION_COLORS.get(a, _DEFAULT_COLOR)['face'] for a in present]
    ax.bar(positions, pct_true, color=faces, edgecolor=BAR_EDGE, linewidth=BAR_EDGE_LW,
           width=BAR_WIDTH, zorder=2)

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
    """3-way (p-Speed / n-Speed / non-speed) classification from the final
    speed score / speed_cell across session types -- a multinomial GEE
    clustered by cell (CellClusteredStats_Utils.compare_nominal), since 'speed
    direction' isn't a simple True/False outcome."""
    label = 'Speed-direction classification (p-Speed / n-Speed / non-speed, % of place cells)'
    omni, posthoc_rows, table = ccs.compare_nominal(df, 'session_type', SESSION_TYPE_ORDER,
                                                    'speed_class', ['p', 'n', 'non'])
    present = [a for a in SESSION_TYPE_ORDER
               if df.loc[df['session_type'] == a, 'speed_class'].notna().any()]
    rows = []
    for arena, (n_p, n_n, n_non) in zip(present, table):
        n_total = n_p + n_n + n_non
        rows.append({'metric': label, 'session_type': arena, 'n': n_total,
                      'n_p_speed': n_p, 'n_n_speed': n_n, 'n_non_speed': n_non,
                      'pct_p_speed': 100.0 * n_p / n_total,
                      'pct_n_speed': 100.0 * n_n / n_total,
                      'pct_non_speed': 100.0 * n_non / n_total})
    desc_df = pd.DataFrame(rows)

    omnibus = {'metric': label, 'n_groups': len(present),
               'groups': ', '.join(f'{a} (n={r["n"]})' for a, r in zip(present, rows))}

    if omni is None:
        omnibus.update({'test': 'insufficient data', 'statistic': np.nan,
                         'p_value': np.nan, 'dof': np.nan, 'significant': False})
        return desc_df, omnibus, pd.DataFrame()

    omnibus.update(omni)
    omnibus['significant'] = bool(pd.notna(omni['p_value']) and omni['p_value'] < ALPHA)
    posthoc_rows = [{'metric': label, **r} for r in posthoc_rows]

    posthoc_df = pd.DataFrame(posthoc_rows)
    return desc_df, omnibus, posthoc_df


def plot_speed_direction(df: pd.DataFrame, out_path: str, stats_lines=None,
                         posthoc_df=None):
    present, pct_p, pct_n, pct_non, ns = [], [], [], [], []
    for a in SESSION_TYPE_ORDER:
        cls = df.loc[df['session_type'] == a, 'speed_class'].dropna()
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

    fig_w = _bar_fig_width(len(present))
    stats_lines = _wrap_stats(stats_lines, fig_w)
    fig, ax, ax_info = _new_fig((fig_w, 5.5), legend_rows=1, stats_lines=stats_lines)

    positions = np.arange(len(present))
    dark   = [SESSION_COLORS.get(a, _DEFAULT_COLOR)['dark']  for a in present]
    light  = [SESSION_COLORS.get(a, _DEFAULT_COLOR)['light'] for a in present]
    bottoms2 = [p + n for p, n in zip(pct_p, pct_n)]
    bar_kw = dict(edgecolor=BAR_EDGE, linewidth=BAR_EDGE_LW, width=BAR_WIDTH, zorder=2)
    ax.bar(positions, pct_p, color=dark, label='p-Speed', **bar_kw)
    ax.bar(positions, pct_n, bottom=pct_p, color=light, label='n-Speed', **bar_kw)
    ax.bar(positions, pct_non, bottom=bottoms2, color=SIG_GRAY['face'],
           label='non-speed', **bar_kw)

    tick_labels = [f'{_label(a)}\n(n={n})' for a, n in zip(present, ns)]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel('% of place cells', fontsize=FS_LABEL, labelpad=8)
    ax.set_ylim(0, 100)
    _set_title(ax, 'Speed-direction classification (speed score)')
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
    'final_speed_score': 'Speed score',
}
HIST_FULL_METRICS = {k: v for k, v in FULL_METRICS.items() if k not in SPEED_METRICS}

# Explicit p-type/n-type speed-cell colors (green/red) -- independent of the
# per-arena SESSION_COLORS palette used elsewhere.
P_COLOR = {'face': PAL_GREEN, 'edge': PAL_BLACK}  # p-type (positive) speed cells
N_COLOR = {'face': PAL_RED,   'edge': PAL_BLACK}  # n-type (negative) speed cells

HIST_PANEL_W, HIST_PANEL_H = 3.6, 3.4   # size of one histogram panel (inches)


def _panel_bracket_levels(posthoc_df, present: list):
    """Significance-bracket levels between histogram panels (worked out from
    panel indices) and the height (inches) of the bracket row they need."""
    levels = _assign_levels(_sig_pairs(posthoc_df, {a: i for i, a in enumerate(present)}))
    n_levels = max((l for *_r, l in levels), default=-1) + 1
    return levels, (n_levels * 0.4 + 0.05 if levels else 0.0)


def _new_panel_fig(n_pan: int, stats_lines=None, sig_h: float = 0.0):
    """Figure with a row of `n_pan` histogram panels (one per group, as in the
    Figure 3 histograms) on a shared x-axis, a borderless panel underneath for
    the stats text and, if `sig_h` > 0, a borderless row of that height
    (inches) above the panels for significance brackets. Returns (fig, axes,
    ax_info, ax_sig); ax_sig is None without a bracket row."""
    info_h = len(stats_lines or []) * STATS_LINE_IN + 0.4
    heights = ([sig_h] if sig_h else []) + [HIST_PANEL_H, info_h]
    fig = plt.figure(figsize=(max(HIST_PANEL_W * n_pan, 7), sum(heights)))
    gs = fig.add_gridspec(len(heights), n_pan, height_ratios=heights)
    row, ax_sig = 0, None
    if sig_h:
        ax_sig = fig.add_subplot(gs[0, :])
        ax_sig.axis('off')
        row = 1
    axes = []
    for i in range(n_pan):
        axes.append(fig.add_subplot(gs[row, i], sharex=axes[0] if axes else None))
    ax_info = fig.add_subplot(gs[row + 1, :])
    ax_info.axis('off')
    return fig, axes, ax_info, ax_sig


def _draw_panel_brackets(ax_sig, axes: list, levels: list):
    """Draw significance brackets between histogram panels in the bracket row.
    Call after fig.tight_layout(), once the panel positions are final."""
    if not levels:
        return
    # Use figure coordinates as the x data coordinates of the bracket row.
    pos = ax_sig.get_position()
    ax_sig.set_xlim(pos.x0, pos.x1)
    ax_sig.set_ylim(0, max(l for *_r, l in levels) + 1)
    centers = [(a.get_position().x0 + a.get_position().x1) / 2 for a in axes]
    placed = [(centers[int(x1)], centers[int(x2)], s, l) for x1, x2, s, l in levels]
    _draw_sig_brackets(ax_sig, placed, y_base=0.1, step=1.0,
                       inset=0.01 * (pos.x1 - pos.x0))


def _hist_panel(ax, vals, bins, face: str):
    """Solid histogram with thin white gaps between the bars, plus a dashed
    line at the median labelled 'med=...' in the top-right corner."""
    counts, _, _ = ax.hist(vals, bins=bins, color=face, edgecolor=BAR_EDGE,
                           linewidth=BAR_EDGE_LW, zorder=2)
    ax.set_ylim(0, max(counts.max(), 1) * 1.15)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    med = np.median(vals)
    ax.axvline(med, color=PAL_BLACK, linestyle=(0, (3, 2)), linewidth=AXIS_LW, zorder=4)
    ax.text(0.98, 0.99, f'med={med:.3g}', transform=ax.transAxes,
            ha='right', va='top', fontsize=FS_MED)


def plot_histogram_by_session_type(df: pd.DataFrame, metric_col: str, metric_label: str,
                             out_path: str):
    """One figure per metric: one histogram panel per session type present, on
    shared bins so the distributions are directly comparable. Significance
    brackets connect the panels of significantly different session types."""
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
    stats_lines = _wrap_stats(_continuous_stats_lines(desc_df, omnibus, posthoc_df),
                              max(HIST_PANEL_W * len(values), 7))
    levels, sig_h = _panel_bracket_levels(posthoc_df, list(values))
    fig, axes, ax_info, ax_sig = _new_panel_fig(len(values), stats_lines, sig_h)

    for ax, (arena, vals) in zip(axes, values.items()):
        _hist_panel(ax, vals, bins, SESSION_COLORS.get(arena, _DEFAULT_COLOR)['face'])
        ax.set_title(f'{_label(arena)} (n={len(vals)})', fontsize=FS_LABEL, pad=10)
        ax.set_xlabel(textwrap.fill(metric_label, 22), fontsize=FS_LABEL, labelpad=6)
        _style_axes(ax)
    axes[0].set_ylabel('Count', fontsize=FS_LABEL, labelpad=6)
    _draw_stats(ax_info, stats_lines)

    fig.tight_layout()
    _draw_panel_brackets(ax_sig, axes, levels)
    _save(fig, out_path)


def _speed_type_masks(sub: pd.DataFrame):
    """Boolean (p_mask, n_mask) numpy arrays for a sub-dataframe, from its
    speed_class column."""
    return (sub['speed_class'] == 'p').to_numpy(), (sub['speed_class'] == 'n').to_numpy()


def plot_speed_histograms_by_session_type(df: pd.DataFrame, metric_col: str,
                                    metric_label: str, out_path: str):
    """Speed-score histogram restricted to speed cells (speed_cell ==
    True), with one panel per session type and p-type (green) / n-type (red)
    speed cells stacked within each panel."""
    passed = df['speed_class'].isin(['p', 'n'])

    sub_all = df.loc[passed].copy()
    sub_all[metric_col] = pd.to_numeric(sub_all[metric_col], errors='coerce')
    sub_all = sub_all.dropna(subset=[metric_col])

    present = [a for a in SESSION_TYPE_ORDER if (sub_all['session_type'] == a).any()]
    if not present:
        print(f'  No speed cells for {metric_label}; skipping plot.')
        return

    bins = np.histogram_bin_edges(sub_all[metric_col].to_numpy(dtype=float), bins=N_BINS)

    # Stats text: descriptives per session type and p/n type, then a
    # cell-clustered comparison (compare_groups) of the pooled
    # speed-modulated cells across session types.
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

    # Significance brackets above the panel titles (a thin axes row on top).
    levels, sig_h = _panel_bracket_levels(posthoc_df, present)
    fig, axes, ax_info, ax_sig = _new_panel_fig(len(present), stats_lines, sig_h)
    _draw_stats(ax_info, stats_lines)

    for ax, arena in zip(axes, present):
        sub = sub_all.loc[sub_all['session_type'] == arena]
        p_vals, n_vals = split[arena]

        ax.hist([p_vals, n_vals], bins=bins, stacked=True,
                color=[P_COLOR['face'], N_COLOR['face']],
                edgecolor=BAR_EDGE, linewidth=BAR_EDGE_LW, zorder=2)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f'{_label(arena)}\n(n={len(sub)}: {len(p_vals)} p-type, {len(n_vals)} n-type)',
                     fontsize=FS_LABEL, pad=10)
        ax.set_xlabel(textwrap.fill(metric_label, 22), fontsize=FS_LABEL, labelpad=6)
        _style_axes(ax)

    axes[0].set_ylabel('Count', fontsize=FS_LABEL, labelpad=6)
    handles = [Patch(facecolor=P_COLOR['face'], edgecolor='none', label='p-type speed cell'),
               Patch(facecolor=N_COLOR['face'], edgecolor='none', label='n-type speed cell')]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=FS_LEGEND,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f'{metric_label}\n(speed cells only)',
                 fontsize=FS_TITLE, y=1.2)
    fig.tight_layout()
    _draw_panel_brackets(ax_sig, axes, levels)
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
    'speed_cell_final':         'Speed cells (% of place cells)',
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


def _write_tables(tables: dict):
    """Append `tables` ({sheet_name: DataFrame}) to INPUT_EXCEL as sheets.
    mode='a' + if_sheet_exists='replace' overwrites only sheets with a
    matching name (e.g. on a re-run) and leaves every other sheet untouched.
    Sheet names are capped at 31 characters by Excel.

    The workbook is written to a temporary copy next to it and only swapped
    in (os.replace) once the save has completed, so a crash, an interrupted
    run or another program touching the file mid-save can't leave
    INPUT_EXCEL half-written/corrupted. Close the workbook in Excel first."""
    tmp = os.path.splitext(INPUT_EXCEL)[0] + '_writing.tmp.xlsx'
    shutil.copy2(INPUT_EXCEL, tmp)
    try:
        with pd.ExcelWriter(tmp, engine='openpyxl', mode='a',
                            if_sheet_exists='replace') as writer:
            for sheet_name, tbl in tables.items():
                tbl.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        os.replace(tmp, INPUT_EXCEL)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


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
    _write_tables(tables)
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

    print(f'Plotting {arena} speed-score histogram (p-type/n-type speed cells)...')
    for col, label in SPEED_METRICS.items():
        plot_speed_histograms_by_session_type(full_df, col, label,
                                               os.path.join(hist_dir, f'Hist_Speed_{col}.png'))
    print(f'Histogram plots for {arena} saved to {hist_dir}')

    return tables


# ── First vs second half of the session ─────────────────────────────────────
# The 'First_Half' / 'Second_Half' sheets hold the same characterization run
# on each half of every session (one row per session + unit, like 'Full').
# Stability, speed and field metrics aren't computed per half, so only the
# metrics below are compared.

CONDITIONS  = ['Whole', 'H1', 'H2']
# Pooled analysis: halves only (both are subsets of the whole session).
HALF_CONDITIONS = ['H1', 'H2']
COND_LABELS = {'Whole': 'Full session', 'H1': 'Geomagnetic Field 1', 'H2': 'Geomagnetic Field 2'}
HALF_DIRS   = {'H1': 'FirstHalf', 'H2': 'SecondHalf'}

HALF_CONTINUOUS_METRICS = {
    'peak_fr':   'Peak firing rate',
    'mean_fr':   'Mean firing rate',
    'sir':       'Spatial information score',
    'sparsity':  'Sparsity',
    'coherence': 'Coherence score',
}
# True/False outcomes of the whole-session place cells.
HALF_BINARY_METRICS_PLACE_CELLS = {
    'theta_modulated':         'Theta modulation (% of place cells)',
    'coherence_bootstrap_sig': 'Coherence shuffle significance (% of place cells)',
}
# Place-cell classification of every recorded unit (each condition's own flag).
HALF_BINARY_METRICS_ALL_UNITS = {
    'place_cell': 'Place-cell yield (% of recorded units classified as place cells)',
}

# 2nd half is drawn in light cyan; whole session and 1st half keep the session's
# shade of cyan (whole session hatched to tell it from the 1st half).
HALF2_CYAN  = '#C2F7F7'
WHOLE_HATCH = '////'
COND_STEP   = 0.28
COND_WIDTH  = 0.24
PAIR_LINE_COLOR = '#C8C8C8'

# Group label of the pooled analysis (all session types clubbed together),
# and the group order of the within-session tests/plots.
POOLED = 'All'
WITHIN_GROUP_ORDER = SESSION_TYPE_ORDER + [POOLED]
SESSION_COLORS[POOLED] = _shade('#4DA6A6', '#1F5C5C', '#B8DBDB')


def _cond_offsets(conds) -> dict:
    """x offset of each condition's box/bar, centred on the group tick."""
    mid = (len(conds) - 1) / 2
    return {c: COND_STEP * (i - mid) for i, c in enumerate(conds)}


def _omnibus_sheet_names(conds) -> tuple:
    """(continuous, binary) test-sheet name parts for a condition set."""
    return ('Friedman', 'CochranQ') if len(conds) > 2 else ('Wilcoxon', 'McNemar')


def _cond_style(session_type: str, cond: str):
    """(face color, hatch) of one condition's box/bar within a session type."""
    if cond == 'H2':
        return HALF2_CYAN, None
    face = SESSION_COLORS.get(session_type, _DEFAULT_COLOR)['face']
    return face, (WHOLE_HATCH if cond == 'Whole' else None)


def load_half_sheets() -> dict:
    return {cond: pd.read_excel(INPUT_EXCEL, sheet_name=sheet)
            for cond, sheet in HALF_SHEETS.items()}


def build_condition_df(full: pd.DataFrame, halves: dict, arena: str,
                       place_cells_only: bool) -> pd.DataFrame:
    """Long table, one row per unit x condition (Whole / H1 / H2), for the
    units of `arena`: the whole-session place cells (place_cells_only) or all
    recorded units. Half-session rows are matched to the whole session by
    session + unit, so every unit has one row per condition (NaN where a
    half has no value)."""
    base = full[full['place_cell'].apply(_to_bool)] if place_cells_only else full
    base = _tag_session_type(base, 'unit(s) (half-session analysis)')
    base = base[base['arena_type'] == arena]
    tag_cols = ['session', 'unit', 'arena_type', 'session_type', 'animal', 'rec_day', 'cell_id']
    keys = base[tag_cols]

    frames = [base.assign(condition='Whole')]
    for cond, half in halves.items():
        half = half.drop(columns=[c for c in tag_cols if c not in ('session', 'unit')],
                         errors='ignore')
        frames.append(keys.merge(half, on=['session', 'unit'], how='left')
                          .assign(condition=cond))
    return pd.concat(frames, ignore_index=True)


def _to_binary_float(v) -> float:
    b = _to_tri_bool(v)
    return np.nan if b is None else float(b)


def _paired_wide(df: pd.DataFrame, col: str, session_type: str, binary: bool,
                 conds=CONDITIONS):
    """(complete-case wide table: one row per unit, one column per condition
    in `conds`; number of units before dropping incomplete ones)."""
    sub = df.loc[(df['session_type'] == session_type) & df['condition'].isin(conds),
                 ['session', 'unit', 'condition', col]].copy()
    sub[col] = (sub[col].map(_to_binary_float) if binary
                else pd.to_numeric(sub[col], errors='coerce'))
    wide = (sub.set_index(['session', 'unit', 'condition'])[col]
               .unstack('condition').reindex(columns=conds))
    return wide.dropna(), len(wide)


def _wilcoxon(x, y):
    """Two-sided Wilcoxon signed-rank test (T, p); p = 1 if every pair ties."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if not np.any(d != 0):
        return np.nan, 1.0
    res = stats.wilcoxon(x, y, alternative='two-sided')
    return res.statistic, res.pvalue


def _mcnemar(x, y):
    """Exact McNemar test (smaller discordant count, p); p = 1 without
    discordant pairs."""
    x, y = np.asarray(x, dtype=int), np.asarray(y, dtype=int)
    b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    if b + c == 0:
        return np.nan, 1.0
    res = mcnemar([[int(((x == 1) & (y == 1)).sum()), b],
                   [c, int(((x == 0) & (y == 0)).sum())]], exact=True)
    return res.statistic, res.pvalue


def compare_within_session(df: pd.DataFrame, col: str, label: str, binary: bool,
                           conds=CONDITIONS):
    """The `conds` conditions (default whole session vs 1st half vs 2nd half)
    of the same units, separately for each session type. Three conditions:
    Friedman + Wilcoxon signed-rank (continuous) / Cochran's Q + exact McNemar
    (binary), post-hoc p-values Holm-corrected per session type. Two
    conditions: the Wilcoxon / McNemar test itself is the omnibus row and the
    single post-hoc row (p_holm = p_raw). Returns (desc_df, omnibus_df,
    posthoc_df)."""
    paired_only = len(conds) == 2
    desc_rows, omni_rows, post_rows = [], [], []
    for st in WITHIN_GROUP_ORDER:
        if not (df['session_type'] == st).any():
            continue
        wide, n_all = _paired_wide(df, col, st, binary, conds)
        n = len(wide)

        for cond in conds:
            v = wide[cond].to_numpy(dtype=float)
            row = {'metric': label, 'session_type': st, 'condition': COND_LABELS[cond], 'n': n}
            if binary:
                row.update({'n_true': int(v.sum()), 'n_false': int(n - v.sum()),
                            'pct_true': 100.0 * v.mean() if n else np.nan})
            elif n:
                row.update({'mean': np.mean(v), 'median': np.median(v),
                            'q1': np.percentile(v, 25), 'q3': np.percentile(v, 75),
                            'std': np.std(v, ddof=1) if n > 1 else np.nan,
                            'sem': stats.sem(v) if n > 1 else np.nan,
                            'min': np.min(v), 'max': np.max(v)})
            desc_rows.append(row)

        if paired_only:
            omni = {'metric': label, 'session_type': st, 'n_cells': n,
                    'n_excluded_incomplete': n_all - n,
                    'test': 'McNemar (exact)' if binary else 'Wilcoxon signed-rank',
                    'stat_name': 'b/c min' if binary else 'T', 'dof': np.nan}
            stat, p = np.nan, np.nan
            if n >= 3:
                c1, c2 = conds
                stat, p = (_mcnemar if binary else _wilcoxon)(wide[c1], wide[c2])
                if binary:
                    omni['pct_diff'] = 100.0 * (wide[c1].mean() - wide[c2].mean())
                else:
                    omni['median_diff'] = float(np.median(wide[c1] - wide[c2]))
            else:
                omni['test'] = 'insufficient data'
            omni.update({'statistic': stat, 'p_value': p,
                         'significant': bool(pd.notna(p) and p < ALPHA)})
            omni_rows.append(omni)
        else:
            omni = {'metric': label, 'session_type': st, 'n_cells': n,
                    'n_excluded_incomplete': n_all - n,
                    'test': "Cochran's Q" if binary else 'Friedman',
                    'stat_name': 'Q' if binary else 'chi2', 'dof': len(conds) - 1}
            stat, p, kendall_w = np.nan, np.nan, np.nan
            if n >= 3:
                if binary:
                    if (wide.nunique(axis=1) == 1).all():   # no unit changes class
                        stat, p = 0.0, 1.0
                    else:
                        res = cochrans_q(wide.to_numpy(dtype=int))
                        stat, p = float(res.statistic), float(res.pvalue)
                else:
                    stat, p = stats.friedmanchisquare(*[wide[c].to_numpy(dtype=float)
                                                        for c in conds])
                    kendall_w = stat / (n * (len(conds) - 1))
            else:
                omni['test'] = 'insufficient data'
            omni.update({'statistic': stat, 'p_value': p,
                         'significant': bool(pd.notna(p) and p < ALPHA)})
            if not binary:
                omni['kendall_w'] = kendall_w
            omni_rows.append(omni)

        if n < 3:
            continue
        rows = []
        for c1, c2 in combinations(conds, 2):
            s, pp = (_mcnemar if binary else _wilcoxon)(wide[c1], wide[c2])
            r = {'metric': label, 'session_type': st, 'comparison': f'{c1} vs {c2}',
                 'test': 'McNemar (exact)' if binary else 'Wilcoxon signed-rank',
                 'stat_name': 'b/c min' if binary else 'T', 'statistic': s,
                 'n': n, 'p_raw': pp}
            if binary:
                r['pct_diff'] = 100.0 * (wide[c1].mean() - wide[c2].mean())
            else:
                r['median_diff'] = float(np.median(wide[c1] - wide[c2]))
            rows.append(r)
        post_rows += ccs._finish_posthoc(rows)
    return pd.DataFrame(desc_rows), pd.DataFrame(omni_rows), pd.DataFrame(post_rows)


def _within_stats_lines(omni_df, post_df, desc_df, binary: bool, conds=CONDITIONS) -> list:
    paired_only = len(conds) == 2
    lines = []
    for _, o in omni_df.iterrows():
        st = o['session_type']
        excl = (f', {int(o["n_excluded_incomplete"])} incomplete excluded'
                if o['n_excluded_incomplete'] else '')
        d = desc_df[desc_df['session_type'] == st]
        if binary:
            parts = [f'{COND_LABELS[c]} {r["pct_true"]:.1f} %'
                     for c, (_, r) in zip(conds, d.iterrows()) if pd.notna(r.get('pct_true'))]
        else:
            parts = [f'{COND_LABELS[c]} med {r["median"]:.3g}'
                     for c, (_, r) in zip(conds, d.iterrows()) if pd.notna(r.get('median'))]
        lines.append(f'{_label(st)} (n={int(o["n_cells"])}{excl}): ' + ', '.join(parts))
        if o['test'] == 'insufficient data':
            lines.append('    insufficient data for a statistical test')
            continue
        w = f', Kendall W = {o["kendall_w"]:.3f}' if pd.notna(o.get('kendall_w', np.nan)) else ''
        dof = f'dof = {int(o["dof"])}, ' if pd.notna(o['dof']) else ''
        stat = f'{o["stat_name"]} = {o["statistic"]:.3f}, ' if pd.notna(o['statistic']) else ''
        lines.append(f'    {o["test"]}: {stat}{dof}'
                     f'p = {_fmt_p(o["p_value"])} {_stars(o["p_value"])}{w}')
        if paired_only:
            continue
        ph = post_df[post_df['session_type'] == st] if len(post_df) else post_df
        for _, r in ph.iterrows():
            c1, c2 = r['comparison'].split(' vs ')
            lines.append(f'    {COND_LABELS[c1]} vs {COND_LABELS[c2]}: p = {_fmt_p(r["p_raw"])}, '
                         f'p(Holm) = {_fmt_p(r["p_holm"])} {_stars(r["p_holm"])}')
    if paired_only:
        test = 'exact McNemar' if binary else 'Wilcoxon signed-rank'
        lines.append(f'Paired {COND_LABELS[conds[0]]} vs {COND_LABELS[conds[1]]} ({test}); '
                     f'bracket shown only where significant.')
    else:
        test = "Cochran's Q + exact McNemar" if binary else 'Friedman + Wilcoxon signed-rank'
        lines.append(f'Within-session repeated measures ({test}, Holm-Bonferroni corrected per '
                     f'session type); brackets shown only where the omnibus test is significant.')
    return lines


def _within_title(label: str, conds) -> str:
    names = ['whole session' if c == 'Whole' else COND_LABELS[c] for c in conds]
    return f'{label}: ' + ' vs '.join(names)


def _within_brackets(ax, omni_df, post_df, x0: dict, conds=CONDITIONS):
    """Significance brackets between the conditions of each session type
    whose omnibus test is significant."""
    if not len(post_df):
        return
    sig_st = set(omni_df.loc[omni_df['significant'] == True, 'session_type'])  # noqa: E712
    ph = post_df[post_df['session_type'].isin(sig_st)].copy()
    if not len(ph):
        return
    ph['comparison'] = [' vs '.join(f'{st}:{c}' for c in cmp.split(' vs '))
                        for st, cmp in zip(ph['session_type'], ph['comparison'])]
    off = _cond_offsets(conds)
    x_of = {f'{st}:{c}': x0[st] + off[c] for st in x0 for c in conds}
    _add_sig_brackets(ax, ph, x_of, inset=0.02)


def _condition_legend(ax_info, session_type: str, conds=CONDITIONS):
    handles = []
    for cond in conds:
        face, hatch = _cond_style(session_type, cond)
        handles.append(Patch(facecolor=face, edgecolor=PAL_BLACK, hatch=hatch,
                             linewidth=BOX_LW, label=COND_LABELS[cond]))
    ax_info.legend(handles=handles, loc='upper center', fontsize=FS_LEGEND, ncol=len(conds))


def _within_fig(n_groups: int, stats_lines):
    fig_w = n_groups * GROUP_W * 2 + AXIS_PAD_W
    stats_lines = _wrap_stats(stats_lines, fig_w)
    fig, ax, ax_info = _new_fig((fig_w, 5.5), legend_rows=1, stats_lines=stats_lines)
    _draw_stats(ax_info, stats_lines)
    return fig, ax, ax_info


def plot_within_continuous(df: pd.DataFrame, col: str, label: str, out_path: str,
                           desc_df, omni_df, post_df, conds=CONDITIONS):
    """Per session type: one box per condition in `conds` (default whole
    session / 1st half / 2nd half) of the same units, with each unit's values
    joined by a thin line."""
    present = [st for st in WITHIN_GROUP_ORDER
               if st in set(omni_df['session_type']) and
               int(omni_df.loc[omni_df['session_type'] == st, 'n_cells'].iloc[0]) > 0]
    if not present:
        return
    fig, ax, ax_info = _within_fig(len(present),
                                   _within_stats_lines(omni_df, post_df, desc_df, False, conds))
    x0 = {st: 2.0 * i for i, st in enumerate(present)}
    off = _cond_offsets(conds)
    rng = np.random.default_rng(0)
    box_kw = dict(showfliers=False, patch_artist=True, widths=COND_WIDTH,
                  boxprops=dict(linewidth=BOX_LW, color=PAL_BLACK),
                  whiskerprops=dict(linewidth=BOX_LW, color=PAL_BLACK),
                  capprops=dict(linewidth=BOX_LW, color=PAL_BLACK),
                  medianprops=dict(linewidth=MEDIAN_LW, color=PAL_BLACK), zorder=2)

    for st in present:
        wide, _ = _paired_wide(df, col, st, binary=False, conds=conds)
        xs = np.array([x0[st] + off[c] for c in conds])
        bp = ax.boxplot([wide[c].to_numpy(dtype=float) for c in conds],
                        positions=xs, **box_kw)
        for patch, cond in zip(bp['boxes'], conds):
            face, hatch = _cond_style(st, cond)
            patch.set_facecolor(face)
            patch.set_edgecolor(PAL_BLACK)
            if hatch:
                patch.set_hatch(hatch)

        jit = rng.uniform(-0.06, 0.06, size=len(wide))
        X = xs[:, None] + jit[None, :]
        Y = wide[conds].to_numpy(dtype=float).T
        ax.plot(X, Y, color=PAIR_LINE_COLOR, linewidth=0.4,
                alpha=0.7, zorder=2.5)
        face = SESSION_COLORS.get(st, _DEFAULT_COLOR)['face']
        ax.scatter(X.ravel(), Y.ravel(), s=12, facecolors=face, edgecolors=PAL_BLACK,
                   linewidths=0.4, zorder=3)

    _condition_legend(ax_info, present[0], conds)
    n_of = dict(zip(omni_df['session_type'], omni_df['n_cells']))
    ax.set_xticks([x0[st] for st in present])
    ax.set_xticklabels([f'{_label(st)}\n(n={int(n_of[st])})' for st in present])
    ax.set_xlim(-0.6, x0[present[-1]] + 0.6)
    ax.set_ylabel(textwrap.fill(label, 30), fontsize=FS_LABEL, labelpad=8)
    _set_title(ax, _within_title(label, conds))
    _style_axes(ax)
    _within_brackets(ax, omni_df, post_df, x0, conds)
    fig.tight_layout()
    _save(fig, out_path)


def plot_within_binary(df: pd.DataFrame, col: str, label: str, out_path: str,
                       desc_df, omni_df, post_df, conds=CONDITIONS):
    """Per session type: % True in each condition of `conds` (default whole
    session / 1st half / 2nd half)."""
    present = [st for st in WITHIN_GROUP_ORDER
               if st in set(omni_df['session_type']) and
               int(omni_df.loc[omni_df['session_type'] == st, 'n_cells'].iloc[0]) > 0]
    if not present:
        return
    fig, ax, ax_info = _within_fig(len(present),
                                   _within_stats_lines(omni_df, post_df, desc_df, True, conds))
    x0 = {st: 2.0 * i for i, st in enumerate(present)}
    off = _cond_offsets(conds)
    for st in present:
        d = desc_df[desc_df['session_type'] == st].reset_index(drop=True)
        for cond, (_, r) in zip(conds, d.iterrows()):
            face, hatch = _cond_style(st, cond)
            ax.bar(x0[st] + off[cond], r['pct_true'], width=COND_WIDTH,
                   color=face, hatch=hatch, edgecolor=PAL_BLACK, linewidth=BAR_EDGE_LW,
                   zorder=2)

    _condition_legend(ax_info, present[0], conds)
    n_of = dict(zip(omni_df['session_type'], omni_df['n_cells']))
    ax.set_xticks([x0[st] for st in present])
    ax.set_xticklabels([f'{_label(st)}\n(n={int(n_of[st])})' for st in present])
    ax.set_xlim(-0.6, x0[present[-1]] + 0.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel('% of cells', fontsize=FS_LABEL, labelpad=8)
    _set_title(ax, _within_title(label, conds))
    _style_axes(ax)
    _within_brackets(ax, omni_df, post_df, x0, conds)
    fig.tight_layout()
    _save(fig, out_path)


def _paired_data_table(pc_df: pd.DataFrame) -> pd.DataFrame:
    """Wide table of the extracted parameters: one row per whole-session place
    cell, one column per metric x condition (e.g. 'sir_H1')."""
    cols = list(HALF_CONTINUOUS_METRICS) + list(HALF_BINARY_METRICS_PLACE_CELLS) + ['place_cell']
    idx = ['session', 'unit', 'session_type', 'animal', 'rec_day', 'cell_id']
    wide = pc_df.set_index(idx + ['condition'])[cols].unstack('condition')
    wide = wide.reindex(columns=pd.MultiIndex.from_product([cols, CONDITIONS]))
    wide.columns = [f'{m}_{c}' for m, c in wide.columns]
    return wide.reset_index()


def run_halves(arena: str, full: pd.DataFrame, halves: dict) -> dict:
    """First vs second half analysis for one arena: session-type comparison on
    each half, and the within-session (and pooled) 1st vs 2nd half comparison.
    Writes sheets to INPUT_EXCEL and plots to PLOTS_DIR/<arena>/HalfSession.
    Returns {sheet_name: table}."""
    pc_df  = build_condition_df(full, halves, arena, place_cells_only=True)
    all_df = build_condition_df(full, halves, arena, place_cells_only=False)
    if not len(pc_df):
        print(f'No place cells in the {arena} arena; skipping the half-session analysis.')
        return {}
    half_root  = os.path.join(PLOTS_DIR, arena, 'HalfSession')
    within_dir = os.path.join(half_root, 'WithinSession')
    os.makedirs(within_dir, exist_ok=True)
    print(f'\n{"=" * 70}\n{arena} arena -- first vs second half of the session\n{"=" * 70}')

    tables = {f'HalfData_{arena}': _paired_data_table(pc_df)}

    # ── Between session types, each half separately ─────────────────────────
    for cond in HALF_SHEETS:
        tag, face = cond, (HALF2_CYAN if cond == 'H2' else None)
        plots_dir = os.path.join(half_root, HALF_DIRS[cond])
        pc_c, all_c = pc_df[pc_df['condition'] == cond], all_df[all_df['condition'] == cond]
        metrics = {k: f'{v} ({COND_LABELS[cond]})' for k, v in HALF_CONTINUOUS_METRICS.items()}
        desc, omni, post = run_all_comparisons(pc_c, metrics, plot_prefix=tag,
                                               plots_dir=plots_dir, box_face=face)

        cat_d, cat_o, cat_p = [], [], []
        for pop, mets in ((all_c, HALF_BINARY_METRICS_ALL_UNITS),
                          (pc_c, HALF_BINARY_METRICS_PLACE_CELLS)):
            for col, lab in mets.items():
                lab = f'{lab} ({COND_LABELS[cond]})'
                d, o, p = compare_categorical(pop, col, lab)
                cat_d.append(d)
                cat_o.append(o)
                if len(p):
                    cat_p.append(p)
                plot_categorical(pop, col, lab,
                                 os.path.join(plots_dir, f'{tag}_Categorical_{col}.png'),
                                 stats_lines=_categorical_stats_lines(d, o, p),
                                 posthoc_df=p, box_face=face)
        cat_omni = pd.DataFrame(cat_o)
        tables.update({
            f'Desc_{arena}_{tag}':       desc,
            f'Omnibus_{arena}_{tag}':    omni,
            f'PostHoc_{arena}_{tag}':    post,
            f'CatDesc_{arena}_{tag}':    pd.concat(cat_d, ignore_index=True) if cat_d else pd.DataFrame(),
            f'CatOmnibus_{arena}_{tag}': cat_omni,
            f'CatPostHoc_{arena}_{tag}': pd.concat(cat_p, ignore_index=True) if cat_p else pd.DataFrame(),
        })
        _print_significant(pd.concat([omni, cat_omni], ignore_index=True),
                           f'{arena}, {COND_LABELS[cond]}, between session types', 65, 16)

    # ── Within session: 1st half vs 2nd half ────────────────────────────────
    # Per session type, the same cells' two halves (paired test). The whole
    # session is not compared: both halves are subsets of it.
    tables.update(_run_within(pc_df, all_df, within_dir, 'Within', arena,
                              conds=HALF_CONDITIONS))

    # ── Pooled over session types: 1st half vs 2nd half ─────────────────────
    # Every session type's cells in one group ('All'): all 1st halves clubbed
    # together vs all 2nd halves. The whole session is not compared (both
    # halves are subsets of it), so this is a direct paired test.
    pooled_dir = os.path.join(half_root, 'Pooled')
    os.makedirs(pooled_dir, exist_ok=True)
    tables.update(_run_within(pc_df.assign(session_type=POOLED),
                              all_df.assign(session_type=POOLED),
                              pooled_dir, 'Pooled', arena, conds=HALF_CONDITIONS))

    _write_tables(tables)
    print(f'\nHalf-session statistics for {arena} appended to {INPUT_EXCEL}')
    print(f'Half-session plots saved to {half_root}')

    for prefix, what, conds in (('Within', 'Within-session', HALF_CONDITIONS),
                                ('Pooled', 'Pooled (all session types)', HALF_CONDITIONS)):
        cont_name, bin_name = _omnibus_sheet_names(conds)
        cont_test, bin_test = (('Friedman', "Cochran's Q") if len(conds) > 2
                               else ('Wilcoxon signed-rank', 'McNemar (exact)'))
        for tbl, test in ((tables[f'{prefix}_{cont_name}_{arena}'], cont_test),
                          (tables[f'{prefix}_{bin_name}_{arena}'], bin_test)):
            sig = tbl[tbl['significant'] == True] if len(tbl) else tbl  # noqa: E712
            print(f'\n{what} {test} tests, {arena}: {len(sig)} of {len(tbl)} significant')
            for _, r in sig.iterrows():
                print(f"  {r['metric']:45s} {_label(r['session_type']):5s} "
                      f"stat={r['statistic']:.3f}  p={r['p_value']:.4g}  (n={int(r['n_cells'])})")
    return tables


def _run_within(pc_df: pd.DataFrame, all_df: pd.DataFrame, out_dir: str,
                prefix: str, arena: str, conds=CONDITIONS) -> dict:
    """Repeated-measures tests between the `conds` conditions (default whole
    vs 1st half vs 2nd half: Friedman / Cochran's Q + post-hoc; two
    conditions: Wilcoxon / McNemar only) and plots for every metric, per
    session_type group of pc_df / all_df. Returns {sheet_name: table}, sheet
    names starting with `prefix`."""
    w_desc, w_omni, w_post = [], [], []
    b_desc, b_omni, b_post = [], [], []
    for col, lab in HALF_CONTINUOUS_METRICS.items():
        d, o, p = compare_within_session(pc_df, col, lab, binary=False, conds=conds)
        w_desc.append(d); w_omni.append(o); w_post.append(p)
        plot_within_continuous(pc_df, col, lab, os.path.join(out_dir, f'{prefix}_{col}.png'),
                               d, o, p, conds)
    for pop, mets in ((all_df, HALF_BINARY_METRICS_ALL_UNITS),
                      (pc_df, HALF_BINARY_METRICS_PLACE_CELLS)):
        for col, lab in mets.items():
            d, o, p = compare_within_session(pop, col, lab, binary=True, conds=conds)
            b_desc.append(d); b_omni.append(o); b_post.append(p)
            plot_within_binary(pop, col, lab,
                               os.path.join(out_dir, f'{prefix}_Categorical_{col}.png'),
                               d, o, p, conds)

    def _cat(parts):
        parts = [p for p in parts if len(p)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    cont_name, bin_name = _omnibus_sheet_names(conds)
    tables = {
        f'{prefix}_Desc_{arena}':          _cat(w_desc),
        f'{prefix}_{cont_name}_{arena}':   _cat(w_omni),
        f'{prefix}_CatDesc_{arena}':       _cat(b_desc),
        f'{prefix}_{bin_name}_{arena}':    _cat(b_omni),
    }
    # With two conditions the single pairwise test is already the test sheet.
    if len(conds) > 2:
        tables[f'{prefix}_PostHoc_{arena}']    = _cat(w_post)
        tables[f'{prefix}_CatPostHoc_{arena}'] = _cat(b_post)
    return tables


def print_repeated_cells(df: pd.DataFrame, what: str):
    """Per arena: the number (and IDs) of animals in the test, and every cell
    recorded in more than one session of the same recording day, listed by
    animal / day with the session types it appears in."""
    print(f'\n{"=" * 70}\nRepeated cells and animals -- {what}\n{"=" * 70}')
    for arena in ARENA_TYPES:
        sub = df[df['arena_type'] == arena]
        if not len(sub):
            continue
        animals = sorted(sub['animal'].dropna().astype(str).unique())
        print(f'\n{arena}: {len(animals)} animal(s) used in the test '
              f'({", ".join(animals)}); {sub["cell_id"].nunique()} cells, '
              f'{len(sub)} observations')

        counts = sub.groupby('cell_id').size()
        repeated = sub[sub['cell_id'].isin(counts[counts > 1].index)]
        if not len(repeated):
            print('  No cell repeated within a recording day.')
            continue
        print(f'  {repeated["cell_id"].nunique()} cell(s) repeated within a recording day:')
        order = ARENA_SESSION_TYPES[arena]
        for (animal, day), day_df in repeated.groupby(['animal', 'rec_day'], sort=True):
            print(f'  {animal} / {day}:')
            for unit, cell_df in day_df.groupby('unit', sort=True):
                types = sorted(cell_df['session_type'].unique(),
                               key=lambda t: order.index(t) if t in order else len(order))
                print(f'      {unit}  ({len(cell_df)} sessions: '
                      f'{", ".join(_label(t) for t in types)})')


if __name__ == '__main__':
    print(f'Loading {INPUT_EXCEL} ...')
    full, fields = load_sheets()

    full_df      = build_full_metrics_df(full)
    field_df     = build_field_metrics_df(full, fields)
    all_units_df = build_all_units_df(full)

    print('\nPlace cells by arena type (Full sheet):')
    print(full_df['arena_type'].value_counts().reindex(ARENA_TYPES).fillna(0).astype(int))

    print(f'Analyzing arena(s): {", ".join(ANALYZE_ARENAS)}')
    full_df      = full_df[full_df['arena_type'].isin(ANALYZE_ARENAS)]
    field_df     = field_df[field_df['arena_type'].isin(ANALYZE_ARENAS)]
    all_units_df = all_units_df[all_units_df['arena_type'].isin(ANALYZE_ARENAS)]

    all_tables = {}
    for arena in ANALYZE_ARENAS:
        for sheet_name, tbl in run_arena(arena, full_df, field_df, all_units_df).items():
            all_tables[sheet_name] = tbl

    # ── First vs second half of the session ─────────────────────────────────
    halves = load_half_sheets()
    for arena in ANALYZE_ARENAS:
        all_tables.update(run_halves(arena, full, halves))

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
        STATS_CSV = (os.path.splitext(INPUT_EXCEL)[0]
                     + f'_SessionType_{"_".join(ANALYZE_ARENAS)}_AllStats.csv')
        pd.concat(stats_csv_parts, ignore_index=True).to_csv(
            STATS_CSV, index=False, encoding='utf-8-sig')
        print(f'\nAll statistics saved to {STATS_CSV}')

    print(f'\nContinuous-metric test used: '
          f'{"Kruskal-Wallis + Mann-Whitney U (Holm), independence assumed" if CONTINUOUS_TEST == "KW" else "nested LMM"}')
    print_repeated_cells(full_df, 'place cells (Full sheet)')

    print(f'\nDone. Plots saved under {PLOTS_DIR}')