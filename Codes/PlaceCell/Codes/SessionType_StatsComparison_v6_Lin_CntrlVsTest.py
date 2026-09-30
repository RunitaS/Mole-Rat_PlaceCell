# -*- coding: utf-8 -*-
"""
Control vs Test comparison of linear-track place-cell properties, using two
workbooks produced by PlaceCellChar_FieldDetect_Main_v3.py (same format):

    CNTRL_EXCEL : cells recorded in the control condition (geomagnetic field
                  in both halves of the session; see
                  SessionType_StatsComparison_v6_Lin_FldVsFld.py)
    TEST_EXCEL  : cells recorded in the test condition (geomagnetic field in
                  the 1st half, zero field in the 2nd half; see
                  SessionType_StatsComparison_v6_Lin_FldVsZero.py)

Both conditions have the same four Linear session types (0, 90, 180, 270),
read from the last folder of the 'session' path (e.g. 'Fa1059\\Linear\\Day10\\1_0'
-> '0'). Only whole-session place cells (place_cell == True) are analyzed,
except place-cell yield, which uses all recorded units.

1. Whole session (sheets 'Full' / 'PlaceFields'), one figure per metric with
   Control and Test side by side at each session type:
     - Session types: Kruskal-Wallis over the 8 groups (Control 0/90/180/270,
       Test 0/90/180/270), post-hoc pairwise Mann-Whitney U (POSTHOC_PAIRS
       sets which pairs), Holm-Bonferroni corrected within each metric.
     - All sessions: Control vs Test with every session type pooled
       (Kruskal-Wallis with 2 groups + Mann-Whitney U).
   % outcomes (place-cell yield, theta modulation, coherence shuffle
   significance, stability significance, speed cells): chi-square test +
   pairwise Fisher's exact, Holm-Bonferroni corrected.

2. Half session (sheets 'First_Half' / 'Second_Half'): a 2 x 2 mixed design
   -- condition (Control / Test: different cells, independent) x half (1st /
   2nd: the same cells, repeated measures). Separately for each session type
   and for all sessions pooled, five planned tests, Holm-Bonferroni corrected
   together (compare_half):
     - 1st vs 2nd half within Control and within Test (paired): Wilcoxon
       signed-rank / exact McNemar
     - Control vs Test in the 1st and in the 2nd half (independent):
       Mann-Whitney U / Fisher's exact
     - interaction: each cell's change from the 1st to the 2nd half, Control
       vs Test (Mann-Whitney U) -- does the zero field (Test) change the cells
       more than a second geomagnetic-field half (Control) does?
   Cells missing either half are excluded (complete cases). Plots: each
   condition's two halves touching, a gap between Control and Test, each
   cell's halves joined by a line; the interaction bracket is marked 'Δ'.

Caveat: a cell recorded in several sessions of one day contributes one row
per session; those rows are treated as independent. Holm-Bonferroni controls
the family-wise error over the post-hoc pairs; the rank tests themselves
handle unequal group sizes (Control has fewer cells than Test).

Output: all statistics tables in OUTPUT_EXCEL (a new workbook; the two input
workbooks are not modified) plus one long-format CSV next to it, and plots in
PLOTS_DIR/FullSession and PLOTS_DIR/HalfSession.
"""

import os
import re
import shutil
import textwrap
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch

import CellClusteredStats_Utils as ccs

# ── Parameters ──────────────────────────────────────────────────────────────

DATA_ROOT   = r'X:\NMR_group_data\Runita\Analysis\Mean_KDE_Open_PascalOldenburg\LinearTrack_GeoMagVsZero'
CNTRL_EXCEL = os.path.join(DATA_ROOT, 'Data_Control', 'All_TT_PlaceChar_AdptBin.xlsx')
TEST_EXCEL  = os.path.join(DATA_ROOT, 'Data', 'All_TT_PlaceChar_AdptBin.xlsx')
OUTPUT_EXCEL = os.path.join(DATA_ROOT, 'CntrlVsTest_PlaceChar_AdptBin_Stats.xlsx')
PLOTS_DIR    = os.path.join(DATA_ROOT, 'SessionType_StatsPlots_Lin_CntrlVsTest')

ARENA = 'Linear'
SESSION_TYPES = ['0', '90', '180', '270']
POOLED = 'All'                       # all session types clubbed together
CLUSTERS = SESSION_TYPES + [POOLED]  # x-axis groups of every plot

CONDS = ['Cntrl', 'Test']
COND_NAMES = {'Cntrl': 'Control', 'Test': 'Test'}
INPUT_EXCELS = {'Cntrl': CNTRL_EXCEL, 'Test': TEST_EXCEL}

# Half-session sheets and the field each condition had in each half.
HALF_SHEETS = {'H1': 'First_Half', 'H2': 'Second_Half'}
HALF_SUBS = ['Cntrl_H1', 'Cntrl_H2', 'Test_H1', 'Test_H2']
HALF_SHORT = {'Cntrl_H1': 'Control GF1', 'Cntrl_H2': 'Control GF2',
              'Test_H1': 'Test GF', 'Test_H2': 'Test Zero',
              # each cell's change from the 1st to the 2nd half
              'Cntrl_D': 'Control Δ(GF2−GF1)', 'Test_D': 'Test Δ(Zero−GF)'}
HALF_LONG = {'Cntrl_H1': 'Control: Geomagnetic Field 1',
             'Cntrl_H2': 'Control: Geomagnetic Field 2',
             'Test_H1': 'Test: Geomagnetic Field',
             'Test_H2': 'Test: Zero Field'}

ALPHA = 0.05

# Post-hoc Mann-Whitney U pairs of the 8-group (condition x session type)
# Kruskal-Wallis test, Holm-Bonferroni corrected over the pairs tested:
#   'all'     : all 28 pairs
#   'planned' : Control vs Test at the same session type, and session types
#               within each condition (16 pairs; a less strict correction)
POSTHOC_PAIRS = 'all'

# Animal ID for rows whose session path names no animal (see
# CellClusteredStats_Utils.add_cell_ids).
ANIMAL_ID_IF_MISSING = 'Animal_1'

FULL_METRICS = {
    'peak_fr':           'Peak firing rate',
    'mean_fr':           'Mean firing rate',
    'sir':               'Spatial information score',
    'sparsity':          'Sparsity',
    'coherence':         'Coherence score',
    'stability_score':   'Stability score',
    'final_speed_score': 'Speed score',
}
FIELD_METRICS = {
    'n_fields':       'Number of place fields per cell',
    'total_area_cm2': 'Area occupied by fields per cell',
    'pct_area':       'Percentage area occupied by fields per cell',
}
BINARY_METRICS_ALL_UNITS = {
    'place_cell': 'Place-cell yield (% of recorded units classified as place cells)',
}
BINARY_METRICS_PLACE_CELLS = {
    'theta_modulated':         'Theta modulation (% of place cells)',
    'coherence_bootstrap_sig': 'Coherence shuffle significance (% of place cells)',
    'stability_significant':   'Split-half stability significance, p <= 0.05 (% of place cells)',
    'speed_cell_final':        'Speed cells (% of place cells)',
}

HALF_CONTINUOUS_METRICS = {
    'peak_fr':   'Peak firing rate',
    'mean_fr':   'Mean firing rate',
    'sir':       'Spatial information score',
    'sparsity':  'Sparsity',
    'coherence': 'Coherence score',
}
HALF_BINARY_METRICS_PLACE_CELLS = {
    'theta_modulated':         'Theta modulation (% of place cells)',
    'coherence_bootstrap_sig': 'Coherence shuffle significance (% of place cells)',
}
HALF_BINARY_METRICS_ALL_UNITS = {
    'place_cell': 'Place-cell yield (% of recorded units classified as place cells)',
}

# Plot stats text: list every post-hoc pair up to this many per test,
# otherwise only the significant ones (all pairs are in the workbook).
MAX_POSTHOC_LINES = 12

# ── Figure style (as in the FldVsFld / FldVsZero scripts) ───────────────────

PAL_BLACK = '#000000'
PAL_GRAY  = '#7F7F7F'
FIG_FONT  = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']

FS_LABEL, FS_TITLE, FS_TICK, FS_LEGEND, FS_STATS = 14, 15, 11, 11, 9
STATS_LINE_IN = 0.18   # panel height per line of stats text (inches)
LEGEND_ROW_IN = 0.3    # panel height per row of legend entries (inches)
FIG_DPI = 500

AXIS_LW     = 0.8
BOX_LW      = 1.0
MEDIAN_LW   = 1.5
BAR_EDGE_LW = 0.6
BOX_STEP    = 0.32     # x distance between neighbouring boxes of one cluster
BOX_WIDTH   = 0.26
CLUSTER_GAP = 0.5      # extra x gap between session-type clusters
BOX_W_IN    = 0.6      # figure width per box (inches)
AXIS_PAD_W  = 1.4      # extra figure width for the y-axis label/ticks
STATS_CHARS_PER_IN = 15

plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    FIG_FONT,
    'pdf.fonttype':       42,
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

# (face, light) shade of cyan per session type: Test boxes use the face,
# Control boxes the light tone (hatched).
SESSION_COLORS = {
    '0':    ('#008080', '#99CCCC'),
    '90':   ('#00B3B3', '#99E1E1'),
    '180':  ('#33D6D6', '#ADEFEF'),
    '270':  ('#7FFAFA', '#CCFDFD'),
    POOLED: ('#4DA6A6', '#B8DBDB'),
}
CNTRL_HATCH = '////'
CNTRL_H2_FACE = '#E0E0E0'   # Control 2nd half (geomagnetic field 2)
TEST_H2_FACE  = '#A6A6A6'   # Test 2nd half (zero field), gray as in FldVsZero

# ── Helpers ─────────────────────────────────────────────────────────────────

def _to_bool(x) -> bool:
    """place_cell values may come back from Excel as bool or string."""
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() == 'true'
    if pd.isna(x):
        return False
    return bool(x)


def _to_tri_bool(x):
    """True, False, or None (blank / NaN / 'not tested')."""
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        return True if s == 'true' else False if s == 'false' else None
    if pd.isna(x):
        return None
    return bool(x)


def _speed_class(score, cell):
    """'p' / 'n' speed cell (speed_cell True, sign of final_speed_score),
    'non', or None if final_speed_score is missing."""
    s = pd.to_numeric(score, errors='coerce')
    if pd.isna(s):
        return None
    if _to_tri_bool(cell) is True and s != 0:
        return 'p' if s > 0 else 'n'
    return 'non'


def _stability_sig(v):
    v = pd.to_numeric(v, errors='coerce')
    return bool(v <= ALPHA) if pd.notna(v) else None


def extract_arena_type(session_path: str):
    for part in re.split(r'[\\/]+', str(session_path)):
        if part.strip().lower() == ARENA.lower():
            return ARENA
    return None


def extract_session_type(session_path: str):
    """'3_180' -> '180' (last folder of the path), else None."""
    label = re.split(r'[\\/]+', str(session_path).strip())[-1].strip().lower()
    m = re.search(r'_(270|180|90|0)$', label)
    return m.group(1) if m else None


def _label(session_type: str) -> str:
    if session_type == POOLED:
        return 'All sessions'
    return f'{session_type}°' if str(session_type).isdigit() else str(session_type)


def _split_group(group: str):
    """'Cntrl_H1_90' -> ('Cntrl_H1', '90')."""
    sub, cluster = group.rsplit('_', 1)
    return sub, cluster


# ── Data ────────────────────────────────────────────────────────────────────

TAG_COLS = ['session', 'unit', 'arena_type', 'session_type', 'animal', 'rec_day',
            'cell_id', 'condition']


def _tag(df: pd.DataFrame, cond: str, what: str) -> pd.DataFrame:
    """Add arena/session type, animal/day/cell_id and condition columns;
    drop (and report) rows outside the Linear arena or with no session type."""
    df = df.copy()
    df['arena_type'] = df['session'].apply(extract_arena_type)
    df['session_type'] = [extract_session_type(s) if a == ARENA else None
                          for s, a in zip(df['session'], df['arena_type'])]
    bad = df[df['session_type'].isna()]
    if len(bad):
        print(f'WARNING [{cond}]: {len(bad)} {what} not in a recognizable {ARENA} '
              f'session; excluded:')
        for s in sorted(bad['session'].astype(str).unique()):
            print(f'    {s}')
    df = ccs.add_cell_ids(df.dropna(subset=['session_type']), include_arena=True,
                          default_animal=ANIMAL_ID_IF_MISSING)
    # The same unit label in the two workbooks is not the same cell.
    df['cell_id'] = cond + '|' + df['cell_id'].astype(str)
    df['condition'] = cond
    return df


def load_condition(cond: str, path: str) -> dict:
    """Place cells (Full), per-cell field metrics, all units, and the
    half-session rows of the whole-session place cells / all units."""
    print(f'Loading {COND_NAMES[cond]}: {path}')
    xl = pd.ExcelFile(path)
    full   = xl.parse('Full')
    fields = xl.parse('PlaceFields')
    halves = {h: xl.parse(sheet) for h, sheet in HALF_SHEETS.items()}
    is_pc = full['place_cell'].apply(_to_bool)

    pc = _tag(full[is_pc], cond, 'place cell(s)')
    pc['speed_class'] = [_speed_class(s, c)
                         for s, c in zip(pc['final_speed_score'], pc['speed_cell'])]
    pc['speed_cell_final'] = pc['speed_class'].map(
        lambda k: None if k is None else k in ('p', 'n'))
    pc['stability_significant'] = pc['stability_p_value'].apply(_stability_sig)

    agg = (fields.groupby(['session', 'unit'])
                 .agg(n_fields=('field_number', 'count'),
                      total_area_cm2=('area_cm2', 'sum'),
                      pct_area=('pct_of_occupied_area', 'sum'))
                 .reset_index())
    fld = full.loc[is_pc, ['session', 'unit']].drop_duplicates().merge(
        agg, on=['session', 'unit'], how='left')
    for col in FIELD_METRICS:   # place cells without fields
        fld[col] = fld[col].fillna(0.0)
    fld = _tag(fld, cond, 'place cell(s) (field analysis)')

    units = _tag(full, cond, 'unit(s)')

    def _halves(base):
        keys = base[TAG_COLS]
        frames = []
        for h, half in halves.items():
            half = half.drop(columns=[c for c in TAG_COLS if c not in ('session', 'unit')],
                             errors='ignore')
            frames.append(keys.merge(half, on=['session', 'unit'], how='left')
                              .assign(half=h, sub=f'{cond}_{h}'))
        return pd.concat(frames, ignore_index=True)

    for df in (pc, fld, units):
        df['sub'] = cond
    return {'pc': pc, 'fields': fld, 'units': units,
            'half_pc': _halves(pc), 'half_units': _halves(units)}


def _with_pooled(df: pd.DataFrame) -> pd.DataFrame:
    """Append a copy of every row with session_type = POOLED, and set the
    'group' column (sub + '_' + session type, e.g. 'Cntrl_90', 'Test_H2_All')."""
    out = pd.concat([df, df.assign(session_type=POOLED)], ignore_index=True)
    out['group'] = out['sub'] + '_' + out['session_type'].astype(str)
    return out


# ── Layouts: which boxes are drawn and which groups are tested together ─────

def _pairs(levels: list) -> list:
    pairs = list(combinations(levels, 2))
    if POSTHOC_PAIRS == 'all':
        return pairs
    if POSTHOC_PAIRS == 'planned':
        # same condition (session types within it) or same session type
        return [(a, b) for a, b in pairs
                if _split_group(a)[0] == _split_group(b)[0]
                or _split_group(a)[1] == _split_group(b)[1]]
    raise ValueError(f"POSTHOC_PAIRS must be 'all' or 'planned', not {POSTHOC_PAIRS!r}")


def _full_style(sub, cluster):
    face, light = SESSION_COLORS.get(cluster, (PAL_GRAY, '#D4D4D4'))
    return (light, CNTRL_HATCH) if sub == 'Cntrl' else (face, None)


def _half_style(sub, cluster):
    face, light = SESSION_COLORS.get(cluster, (PAL_GRAY, '#D4D4D4'))
    return {'Cntrl_H1': (light, CNTRL_HATCH), 'Cntrl_H2': (CNTRL_H2_FACE, CNTRL_HATCH),
            'Test_H1': (face, None), 'Test_H2': (TEST_H2_FACE, None)}[sub]


FULL_LAYOUT = {
    'clusters': CLUSTERS,
    'subs': CONDS,
    'sub_short': COND_NAMES,
    'sub_long': COND_NAMES,
    'style': _full_style,
    # One Kruskal-Wallis over the 8 condition x session-type groups, and one
    # Control vs Test test with all session types pooled.
    'test_sets': [('Session types', [f'{c}_{st}' for st in SESSION_TYPES for c in CONDS]),
                  (POOLED, [f'{c}_{POOLED}' for c in CONDS])],
    'title': 'Control vs Test',
}
HALF_LAYOUT = {
    'clusters': CLUSTERS,
    'subs': HALF_SUBS,
    'sub_short': HALF_SHORT,
    'sub_long': HALF_LONG,
    'style': _half_style,
    # Tests: planned mixed-design comparisons per session type (compare_half).
    'title': 'half sessions, Control vs Test',
}

# Pooled only: every Control place cell (all session types) vs every Test
# place cell -- the 'All sessions' group of the layouts above on its own.
FULL_POOLED_LAYOUT = {**FULL_LAYOUT, 'clusters': [POOLED],
                      'test_sets': [(POOLED, [f'{c}_{POOLED}' for c in CONDS])],
                      'title': 'all sessions pooled, Control vs Test'}
HALF_POOLED_LAYOUT = {**HALF_LAYOUT, 'clusters': [POOLED],
                      'title': 'half sessions, all sessions pooled, Control vs Test'}


def _display(group: str, layout: dict) -> str:
    sub, cluster = _split_group(group)
    return f'{layout["sub_short"][sub]} {_label(cluster)}'


# ── Statistics ──────────────────────────────────────────────────────────────

def _desc_row(label, group, layout):
    sub, cluster = _split_group(group)
    return {'metric': label, 'group': group, 'condition': layout['sub_long'][sub],
            'session_type': cluster}


def compare_continuous(df: pd.DataFrame, layout: dict, col: str, label: str):
    """Descriptives per group; per test set a Kruskal-Wallis test and
    pairwise Mann-Whitney U tests (Holm-Bonferroni corrected within the set)."""
    values, desc_rows = {}, []
    for c in layout['clusters']:
        for s in layout['subs']:
            g = f'{s}_{c}'
            v = pd.to_numeric(df.loc[df['group'] == g, col],
                              errors='coerce').dropna().to_numpy(dtype=float)
            if not len(v):
                continue
            values[g] = v
            desc_rows.append({**_desc_row(label, g, layout), 'n': len(v),
                              'mean': np.mean(v), 'median': np.median(v),
                              'q1': np.percentile(v, 25), 'q3': np.percentile(v, 75),
                              'std': np.std(v, ddof=1) if len(v) > 1 else np.nan,
                              'sem': stats.sem(v) if len(v) > 1 else np.nan,
                              'min': np.min(v), 'max': np.max(v)})

    omni_rows, post_rows = [], []
    for set_name, levels in layout['test_sets']:
        groups = {g: values[g] for g in levels if g in values}
        omni = {'metric': label, 'test_set': set_name, 'n_groups': len(groups),
                'groups': ', '.join(f'{g} (n={len(v)})' for g, v in groups.items()),
                'test': 'Kruskal-Wallis', 'stat_name': 'H'}
        if len(groups) < 2 or any(len(v) < 2 for v in groups.values()):
            omni.update({'test': 'insufficient data', 'statistic': np.nan, 'dof': np.nan,
                         'p_value': np.nan, 'epsilon_sq': np.nan, 'significant': False})
            omni_rows.append(omni)
            continue
        all_v = np.concatenate(list(groups.values()))
        if np.all(all_v == all_v[0]):
            h, p = 0.0, 1.0
        else:
            h, p = stats.kruskal(*groups.values())
        omni.update({'statistic': h, 'dof': len(groups) - 1, 'p_value': p,
                     'epsilon_sq': h / (len(all_v) - 1),
                     'significant': bool(p < ALPHA)})
        omni_rows.append(omni)

        rows = []
        for g1, g2 in _pairs(list(groups)):
            x, y = groups[g1], groups[g2]
            u, pp = stats.mannwhitneyu(x, y, alternative='two-sided')
            rows.append({'metric': label, 'test_set': set_name,
                         'comparison': f'{g1} vs {g2}', 'test': 'Mann-Whitney U',
                         'stat_name': 'U', 'statistic': u, 'n1': len(x), 'n2': len(y),
                         'median_1': np.median(x), 'median_2': np.median(y),
                         'rank_biserial': 2.0 * u / (len(x) * len(y)) - 1.0,
                         'p_raw': pp})
        post_rows += ccs._finish_posthoc(rows)
    return pd.DataFrame(desc_rows), pd.DataFrame(omni_rows), pd.DataFrame(post_rows)


def compare_binary(df: pd.DataFrame, layout: dict, col: str, label: str):
    """Descriptives per group; per test set a chi-square test and pairwise
    Fisher's exact tests (Holm-Bonferroni corrected within the set)."""
    counts, desc_rows = {}, []
    for c in layout['clusters']:
        for s in layout['subs']:
            g = f'{s}_{c}'
            v = df.loc[df['group'] == g, col].map(_to_tri_bool).dropna()
            if not len(v):
                continue
            t = int((v == True).sum())  # noqa: E712
            counts[g] = (t, len(v) - t)
            desc_rows.append({**_desc_row(label, g, layout), 'n': len(v), 'n_true': t,
                              'n_false': len(v) - t, 'pct_true': 100.0 * t / len(v)})

    omni_rows, post_rows = [], []
    for set_name, levels in layout['test_sets']:
        present = [g for g in levels if g in counts]
        omni = {'metric': label, 'test_set': set_name, 'n_groups': len(present),
                'groups': ', '.join(f'{g} (n={sum(counts[g])})' for g in present),
                'test': 'Chi-square test of independence', 'stat_name': 'chi2'}
        if len(present) < 2:
            omni.update({'test': 'insufficient data', 'statistic': np.nan, 'dof': np.nan,
                         'p_value': np.nan, 'significant': False})
            omni_rows.append(omni)
            continue
        table = np.array([counts[g] for g in present])
        if (table.sum(axis=0) == 0).any():       # every unit True (or False)
            chi2, p = 0.0, 1.0
        else:
            chi2, p, _dof, _exp = stats.chi2_contingency(table)
        omni.update({'statistic': chi2, 'dof': len(present) - 1, 'p_value': p,
                     'significant': bool(p < ALPHA)})
        omni_rows.append(omni)

        rows = []
        for g1, g2 in _pairs(present):
            odds, pp = stats.fisher_exact(np.array([counts[g1], counts[g2]]))
            rows.append({'metric': label, 'test_set': set_name,
                         'comparison': f'{g1} vs {g2}', 'test': "Fisher's exact",
                         'stat_name': 'OR', 'statistic': odds,
                         'pct_1': 100.0 * counts[g1][0] / sum(counts[g1]),
                         'pct_2': 100.0 * counts[g2][0] / sum(counts[g2]),
                         'p_raw': pp})
        post_rows += ccs._finish_posthoc(rows)
    return pd.DataFrame(desc_rows), pd.DataFrame(omni_rows), pd.DataFrame(post_rows)


# ── Stats text shown under each plot ────────────────────────────────────────

def _fmt_p(p) -> str:
    """'= 0.012' / '< 0.001' / '= n/a', to follow 'p' or 'p(Holm)'."""
    if p is None or pd.isna(p):
        return '= n/a'
    return '< 0.001' if p < 0.001 else f'= {p:.3f}'


def _stars(p) -> str:
    if p is None or pd.isna(p):
        return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < ALPHA else 'n.s.'


def _stats_lines(desc, omni, post, layout, binary: bool) -> list:
    lines = []
    for _, r in desc.iterrows():
        name = _display(r['group'], layout)
        if binary:
            lines.append(f'{name} (n={int(r["n"])}): {int(r["n_true"])} yes / '
                         f'{int(r["n_false"])} no = {r["pct_true"]:.1f} %')
        else:
            sd = f'{r["std"]:.3g}' if pd.notna(r['std']) else 'n/a'
            lines.append(f'{name} (n={int(r["n"])}): median {r["median"]:.3g} '
                         f'[Q1 {r["q1"]:.3g}, Q3 {r["q3"]:.3g}] | mean {r["mean"]:.3g} ± {sd} SD')
    for _, o in omni.iterrows():
        what = _label(o['test_set']) if o['test_set'] in CLUSTERS else o['test_set']
        if o['test'] == 'insufficient data':
            lines.append(f'{what}: insufficient data for a statistical test')
            continue
        eff = (f', ε² = {o["epsilon_sq"]:.3f}'
               if pd.notna(o.get('epsilon_sq', np.nan)) else '')
        lines.append(f'{what} -- {o["test"]}: {o["stat_name"]} = {o["statistic"]:.3f}, '
                     f'dof = {int(o["dof"])}, p {_fmt_p(o["p_value"])} '
                     f'{_stars(o["p_value"])}{eff}')
        ph = post[post['test_set'] == o['test_set']] if len(post) else post
        if not len(ph):
            continue
        shown = ph if len(ph) <= MAX_POSTHOC_LINES else ph[ph['significant'] == True]  # noqa: E712
        lines.append(f'    Post-hoc {ph["test"].iloc[0]} (Holm-Bonferroni, {len(ph)} pairs)'
                     + ('' if len(shown) == len(ph)
                        else f'; {len(shown)} significant pair(s) shown') + ':')
        for _, r in shown.iterrows():
            g1, g2 = r['comparison'].split(' vs ')
            fmt = '.3g' if r['stat_name'] == 'OR' else '.1f'
            lines.append(f'    {_display(g1, layout)} vs {_display(g2, layout)}: '
                         f'{r["stat_name"]} = {r["statistic"]:{fmt}}, p {_fmt_p(r["p_raw"])}, '
                         f'p(Holm) {_fmt_p(r["p_holm"])} {_stars(r["p_holm"])}')
    return lines


# ── Plots ───────────────────────────────────────────────────────────────────

def _wrap_stats(stats_lines, fig_w: float) -> list:
    width = max(40, int(fig_w * STATS_CHARS_PER_IN))
    return [w for line in (stats_lines or [])
            for w in (textwrap.wrap(line, width, subsequent_indent='    ') or [''])]


def _sig_pairs(posthoc_df, x_of: dict) -> list:
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
            if _split_group(g1)[0].endswith('_D'):   # half-session interaction
                stars = f'Δ {stars}'
            pairs.append((x1, x2, stars))
    return pairs


def _assign_levels(pairs: list) -> list:
    """Stack brackets so overlapping ones sit on different levels."""
    placed = []
    for x1, x2, stars in sorted(pairs, key=lambda t: t[1] - t[0]):
        level = 0
        while any(l == level and x1 <= b2 and x2 >= b1 for b1, b2, _s, l in placed):
            level += 1
        placed.append((x1, x2, stars, level))
    return placed


def _add_sig_brackets(ax, posthoc_df, x_of: dict, inset: float = 0.02):
    """Significance brackets above the data; call after everything is plotted."""
    placed = _assign_levels(_sig_pairs(posthoc_df, x_of))
    if not placed:
        return
    lo, hi = ax.get_ylim()
    ticks = [t for t in ax.get_yticks() if lo <= t <= hi]
    step = 0.08 * (hi - lo)
    y_base = hi + 0.02 * (hi - lo)
    for x1, x2, stars, level in placed:
        y = y_base + level * step
        ax.plot([x1 + inset, x2 - inset], [y, y], color=PAL_BLACK, linewidth=BOX_LW,
                solid_capstyle='butt', clip_on=False, zorder=5)
        ax.annotate(stars, ((x1 + x2) / 2, y), xytext=(0, -2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=FS_TICK + 1, fontweight='bold',
                    annotation_clip=False)
    ax.set_yticks(ticks)
    ax.set_ylim(lo, y_base + (max(p[3] for p in placed) + 1) * step)


def plot_groups(df: pd.DataFrame, layout: dict, col: str, label: str, out_path: str,
                stats_lines, posthoc_df, binary: bool):
    """Session-type clusters on the x axis, the layout's sub-groups (Control /
    Test, or the four halves) side by side within each cluster: box plot +
    points (continuous) or % True bars (binary)."""
    subs = layout['subs']
    data = {}
    for c in layout['clusters']:
        for s in subs:
            g = f'{s}_{c}'
            if binary:
                v = df.loc[df['group'] == g, col].map(_to_tri_bool).dropna()
                v = (v == True).to_numpy(dtype=float)  # noqa: E712
            else:
                v = pd.to_numeric(df.loc[df['group'] == g, col],
                                  errors='coerce').dropna().to_numpy(dtype=float)
            if len(v):
                data[g] = v
    present = [c for c in layout['clusters'] if any(f'{s}_{c}' in data for s in subs)]
    if not present:
        return

    cl_w = len(subs) * BOX_STEP + CLUSTER_GAP
    x0 = {c: i * cl_w for i, c in enumerate(present)}
    off = {s: BOX_STEP * (i - (len(subs) - 1) / 2) for i, s in enumerate(subs)}

    fig, ax, ax_info, stats_lines = _new_frame(len(present), len(subs), stats_lines)
    rng = np.random.default_rng(0)
    x_of, xs, ns = {}, [], []
    for c in present:
        for s in subs:
            g = f'{s}_{c}'
            if g not in data:
                continue
            v, x = data[g], x0[c] + off[s]
            face, hatch = layout['style'](s, c)
            if binary:
                _draw_bar(ax, v, x, face, hatch)
            else:
                _draw_box(ax, v, x, face, hatch)
                jit = rng.uniform(-0.07, 0.07, size=len(v))
                ax.scatter(x + jit, v, s=10, facecolors=face, edgecolors=PAL_BLACK,
                           linewidths=0.4, zorder=3)
            x_of[g] = x
            xs.append(x)
            ns.append(len(v))

    _finish_fig(fig, ax, ax_info, layout, label, binary, stats_lines, out_path,
                ticks=xs, tick_labels=[f'n={n}' for n in ns], x0=x0,
                xlim=(min(xs) - 0.35, max(xs) + 0.35), posthoc_df=posthoc_df, x_of=x_of)


def _new_frame(n_clusters: int, n_subs: int, stats_lines):
    """Figure with the plot axes on top and a borderless panel below for the
    legend and stats text. At least 1.2 in per box, so single-cluster
    (pooled) figures fit the legend and stats text."""
    fig_w = max(n_clusters * n_subs * BOX_W_IN, n_subs * 1.2) + AXIS_PAD_W
    stats_lines = _wrap_stats(stats_lines, fig_w)
    h = 5.5
    panel_h = (LEGEND_ROW_IN * (1 if n_subs <= 2 else 2)
               + len(stats_lines) * STATS_LINE_IN + 0.5)
    fig, (ax, ax_info) = plt.subplots(2, 1, figsize=(fig_w, h + panel_h),
                                      gridspec_kw={'height_ratios': [h, panel_h]})
    ax_info.axis('off')
    return fig, ax, ax_info, stats_lines


def _draw_box(ax, v, x, face, hatch, width=BOX_WIDTH):
    bp = ax.boxplot([v], positions=[x], widths=width, showfliers=False,
                    patch_artist=True,
                    boxprops=dict(linewidth=BOX_LW, color=PAL_BLACK),
                    whiskerprops=dict(linewidth=BOX_LW, color=PAL_BLACK),
                    capprops=dict(linewidth=BOX_LW, color=PAL_BLACK),
                    medianprops=dict(linewidth=MEDIAN_LW, color=PAL_BLACK),
                    zorder=2)
    bp['boxes'][0].set_facecolor(face)
    bp['boxes'][0].set_edgecolor(PAL_BLACK)
    if hatch:
        bp['boxes'][0].set_hatch(hatch)


def _draw_bar(ax, v, x, face, hatch, width=BOX_WIDTH):
    """Bar of the % True of the 0/1 values `v`."""
    ax.bar(x, 100.0 * np.mean(v), width=width, color=face, hatch=hatch,
           edgecolor=PAL_BLACK, linewidth=BAR_EDGE_LW, zorder=2)


def _finish_fig(fig, ax, ax_info, layout, label, binary, stats_lines, out_path,
                ticks, tick_labels, x0, xlim, posthoc_df, x_of):
    """Ticks (n per box/pair), session-type names under them, axis labels,
    title, significance brackets, legend + stats text; then save."""
    fig_w = fig.get_figwidth()
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=FS_TICK - 3)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for c, x in x0.items():
        ax.text(x, -0.08, _label(c), transform=trans, ha='center', va='top',
                fontsize=FS_LABEL - 2, fontweight='bold')
    ax.set_xlim(*xlim)
    if binary:
        ax.set_ylim(0, 100)
        ax.set_ylabel('% of cells', fontsize=FS_LABEL, labelpad=8)
    else:
        ax.set_ylabel(textwrap.fill(label, 30), fontsize=FS_LABEL, labelpad=8)
    ax.set_title(textwrap.fill(f'{label} ({layout["title"]})', max(20, int(fig_w * 7))),
                 fontsize=FS_TITLE - 2, pad=12)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    _add_sig_brackets(ax, posthoc_df, x_of)

    subs = layout['subs']
    ref = next(iter(x0))
    handles = []
    for s in subs:
        face, hatch = layout['style'](s, ref)
        handles.append(Patch(facecolor=face, edgecolor=PAL_BLACK, hatch=hatch,
                             linewidth=BOX_LW, label=layout['sub_long'][s]))
    ax_info.legend(handles=handles, loc='upper center', fontsize=FS_LEGEND,
                   ncol=len(subs) if len(subs) <= 2 else 2)
    ax_info.text(0.5, 0.0, '\n'.join(stats_lines), transform=ax_info.transAxes,
                 fontsize=FS_STATS, va='bottom', ha='center', linespacing=1.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


# ── Half session: mixed design (condition between, half within cells) ───────

HALVES = list(HALF_SHEETS)   # ['H1', 'H2']
HALF_BOX_W = 0.26            # boxes of one condition's two halves touch
COND_GAP   = 0.22            # gap between the Control pair and the Test pair
PAIR_LINE_COLOR = '#C8C8C8'  # lines joining one cell's two halves
MIN_N = 3                    # fewest cells per group for a test


def _half_offsets() -> dict:
    """x offset of each half's box within a cluster: Control GF1|GF2 touching,
    a gap, then Test GF|Zero touching."""
    w, g = HALF_BOX_W, COND_GAP
    return {'Cntrl_H1': -(g / 2 + 1.5 * w), 'Cntrl_H2': -(g / 2 + 0.5 * w),
            'Test_H1':    g / 2 + 0.5 * w,  'Test_H2':    g / 2 + 1.5 * w}


def _to_binary_float(v) -> float:
    b = _to_tri_bool(v)
    return np.nan if b is None else float(b)


def _half_wide(df: pd.DataFrame, col: str, cond: str, cluster: str, binary: bool):
    """(complete-case table: one row per cell of `cond` / `cluster`, columns
    H1 and H2; number of cells before dropping those missing a half)."""
    sub = df[(df['condition'] == cond) & (df['session_type'] == cluster)]
    if not len(sub):
        return pd.DataFrame(columns=HALVES), 0
    v = (sub[col].map(_to_binary_float) if binary
         else pd.to_numeric(sub[col], errors='coerce'))
    wide = (sub.assign(_v=v.astype(float)).set_index(['session', 'unit', 'half'])['_v']
               .unstack('half').reindex(columns=HALVES))
    return wide.dropna(), len(wide)


def _wilcoxon(x, y):
    """Two-sided Wilcoxon signed-rank test (T, p); p = 1 if every pair ties."""
    if not np.any(np.asarray(y) - np.asarray(x) != 0):
        return np.nan, 1.0
    res = stats.wilcoxon(x, y, alternative='two-sided')
    return res.statistic, res.pvalue


def _mcnemar(x, y):
    """Exact McNemar test (smaller discordant count, p); p = 1 without
    discordant pairs."""
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    if b + c == 0:
        return np.nan, 1.0
    return min(b, c), stats.binomtest(min(b, c), b + c, 0.5).pvalue


def _mwu(x, y):
    """Two-sided Mann-Whitney U (U, p); p = 1 if every value ties."""
    both = np.concatenate([x, y])
    if np.all(both == both[0]):
        return len(x) * len(y) / 2.0, 1.0
    res = stats.mannwhitneyu(x, y, alternative='two-sided')
    return res.statistic, res.pvalue


def compare_half(df: pd.DataFrame, layout: dict, col: str, label: str, binary: bool):
    """2 x 2 mixed design per session type (and pooled): condition (Control /
    Test, independent cells) x half (H1 / H2, the same cells). Five planned
    tests, Holm-Bonferroni corrected per session type:
      - H1 vs H2 within Control and within Test: Wilcoxon signed-rank /
        exact McNemar (paired)
      - Control vs Test in H1 and in H2: Mann-Whitney U / Fisher's exact
        (independent)
      - interaction: each cell's change H2 - H1, Control vs Test, Mann-Whitney
        U (does the change between halves differ between the conditions?)
    Cells missing either half are excluded (complete cases), so every test of
    a session type uses the same cells. Returns (desc_df, tests_df)."""
    desc_rows, test_rows = [], []
    for cl in layout['clusters']:
        wide = {}
        for cond in CONDS:
            w, n_all = _half_wide(df, col, cond, cl, binary)
            if not n_all:
                continue
            wide[cond] = w
            for h in HALVES:
                v = w[h].to_numpy(dtype=float)
                row = {**_desc_row(label, f'{cond}_{h}_{cl}', layout), 'n': len(v),
                       'n_excluded_incomplete': n_all - len(v)}
                if binary:
                    row.update({'n_true': int(v.sum()), 'n_false': int(len(v) - v.sum()),
                                'pct_true': 100.0 * v.mean() if len(v) else np.nan})
                elif len(v):
                    row.update({'mean': np.mean(v), 'median': np.median(v),
                                'q1': np.percentile(v, 25), 'q3': np.percentile(v, 75),
                                'std': np.std(v, ddof=1) if len(v) > 1 else np.nan,
                                'sem': stats.sem(v) if len(v) > 1 else np.nan,
                                'min': np.min(v), 'max': np.max(v)})
                desc_rows.append(row)

        rows = []

        def _add(g1, g2, kind, test, stat_name, stat, x, y, p, **extra):
            rows.append({'metric': label, 'session_type': cl, 'comparison': f'{g1} vs {g2}',
                         'type': kind, 'test': test, 'stat_name': stat_name,
                         'statistic': stat, 'n1': len(x), 'n2': len(y), **extra, 'p_raw': p})

        # H1 vs H2 within each condition (same cells: paired)
        for cond, w in wide.items():
            if len(w) < MIN_N:
                continue
            x, y = w['H1'].to_numpy(dtype=float), w['H2'].to_numpy(dtype=float)
            if binary:
                s, p = _mcnemar(x, y)
                _add(f'{cond}_H1_{cl}', f'{cond}_H2_{cl}', 'within (paired)',
                     'McNemar (exact)', 'b/c min', s, x, y, p,
                     pct_1=100.0 * x.mean(), pct_2=100.0 * y.mean())
            else:
                s, p = _wilcoxon(x, y)
                _add(f'{cond}_H1_{cl}', f'{cond}_H2_{cl}', 'within (paired)',
                     'Wilcoxon signed-rank', 'T', s, x, y, p,
                     median_1=np.median(x), median_2=np.median(y),
                     median_diff=float(np.median(y - x)))

        # Control vs Test (different cells: independent), and the interaction
        if len(wide) == 2 and all(len(w) >= MIN_N for w in wide.values()):
            wc, wt = wide['Cntrl'], wide['Test']
            for h in HALVES:
                x, y = wc[h].to_numpy(dtype=float), wt[h].to_numpy(dtype=float)
                if binary:
                    odds, p = stats.fisher_exact([[int(x.sum()), int(len(x) - x.sum())],
                                                  [int(y.sum()), int(len(y) - y.sum())]])
                    _add(f'Cntrl_{h}_{cl}', f'Test_{h}_{cl}', 'between (independent)',
                         "Fisher's exact", 'OR', odds, x, y, p,
                         pct_1=100.0 * x.mean(), pct_2=100.0 * y.mean())
                else:
                    u, p = _mwu(x, y)
                    _add(f'Cntrl_{h}_{cl}', f'Test_{h}_{cl}', 'between (independent)',
                         'Mann-Whitney U', 'U', u, x, y, p,
                         median_1=np.median(x), median_2=np.median(y),
                         rank_biserial=2.0 * u / (len(x) * len(y)) - 1.0)
            dx = (wc['H2'] - wc['H1']).to_numpy(dtype=float)
            dy = (wt['H2'] - wt['H1']).to_numpy(dtype=float)
            u, p = _mwu(dx, dy)
            extra = ({'net_change_pct_1': 100.0 * dx.mean(), 'net_change_pct_2': 100.0 * dy.mean()}
                     if binary else {'median_1': np.median(dx), 'median_2': np.median(dy)})
            _add(f'Cntrl_D_{cl}', f'Test_D_{cl}', 'interaction (change H2 - H1)',
                 'Mann-Whitney U on H2 - H1', 'U', u, dx, dy, p,
                 rank_biserial=2.0 * u / (len(dx) * len(dy)) - 1.0, **extra)
        if rows:
            test_rows += ccs._finish_posthoc(rows)
    return pd.DataFrame(desc_rows), pd.DataFrame(test_rows)


def _half_stats_lines(desc, tests, layout, binary: bool) -> list:
    lines = []
    for _, r in desc.iterrows():
        name = _display(r['group'], layout)
        excl = (f', {int(r["n_excluded_incomplete"])} incomplete excluded'
                if r['n_excluded_incomplete'] else '')
        if binary:
            lines.append(f'{name} (n={int(r["n"])}{excl}): {int(r["n_true"])} yes = '
                         f'{r["pct_true"]:.1f} %')
        elif pd.notna(r.get('median', np.nan)):
            lines.append(f'{name} (n={int(r["n"])}{excl}): median {r["median"]:.3g} '
                         f'[Q1 {r["q1"]:.3g}, Q3 {r["q3"]:.3g}]')
    for cl in (tests['session_type'].unique() if len(tests) else []):
        lines.append(f'{_label(cl)} (Holm-Bonferroni over the {int((tests["session_type"] == cl).sum())} tests):')
        for _, r in tests[tests['session_type'] == cl].iterrows():
            g1, g2 = r['comparison'].split(' vs ')
            fmt = '.3g' if r['stat_name'] == 'OR' else '.1f'
            stat = (f'{r["stat_name"]} = {r["statistic"]:{fmt}}, '
                    if pd.notna(r['statistic']) else '')
            lines.append(f'    {_display(g1, layout)} vs {_display(g2, layout)} ({r["test"]}): '
                         f'{stat}p {_fmt_p(r["p_raw"])}, p(Holm) {_fmt_p(r["p_holm"])} '
                         f'{_stars(r["p_holm"])}')
    lines.append('Halves of one condition: paired (same cells); Control vs Test: independent; '
                 'Δ = each cell\'s change from 1st to 2nd half (Δ bracket: interaction).')
    return lines


def plot_half(df: pd.DataFrame, layout: dict, col: str, label: str, out_path: str,
              stats_lines, tests, binary: bool):
    """Per session type: Control GF1|GF2 (touching), a gap, Test GF|Zero
    (touching); the same complete-case cells as the tests. Continuous: boxes
    + points, each cell's two halves joined by a thin line. Binary: % bars."""
    off = _half_offsets()
    wides = {(cl, cond): _half_wide(df, col, cond, cl, binary)[0]
             for cl in layout['clusters'] for cond in CONDS}
    present = [cl for cl in layout['clusters'] if any(len(wides[(cl, c)]) for c in CONDS)]
    if not present:
        return
    cl_w = 4 * HALF_BOX_W + COND_GAP + CLUSTER_GAP
    x0 = {cl: i * cl_w for i, cl in enumerate(present)}

    fig, ax, ax_info, stats_lines = _new_frame(len(present), len(HALF_SUBS), stats_lines)
    rng = np.random.default_rng(0)
    x_of, ticks, tick_labels = {}, [], []
    for cl in present:
        for cond in CONDS:
            w = wides[(cl, cond)]
            if not len(w):
                continue
            xs = np.array([x0[cl] + off[f'{cond}_{h}'] for h in HALVES])
            faces = []
            for h, x in zip(HALVES, xs):
                face, hatch = layout['style'](f'{cond}_{h}', cl)
                faces.append(face)
                v = w[h].to_numpy(dtype=float)
                if binary:
                    _draw_bar(ax, v, x, face, hatch, width=HALF_BOX_W)
                else:
                    _draw_box(ax, v, x, face, hatch, width=HALF_BOX_W)
                x_of[f'{cond}_{h}_{cl}'] = x
            if not binary:
                jit = rng.uniform(-0.05, 0.05, size=len(w))
                X = xs[:, None] + jit[None, :]
                Y = w[HALVES].to_numpy(dtype=float).T
                ax.plot(X, Y, color=PAIR_LINE_COLOR, linewidth=0.4, alpha=0.7, zorder=2.5)
                for i, face in enumerate(faces):
                    ax.scatter(X[i], Y[i], s=8, facecolors=face, edgecolors=PAL_BLACK,
                               linewidths=0.3, zorder=3)
            x_of[f'{cond}_D_{cl}'] = xs.mean()
            ticks.append(xs.mean())
            tick_labels.append(f'n={len(w)}')

    edge = 2 * HALF_BOX_W + COND_GAP / 2 + 0.2
    _finish_fig(fig, ax, ax_info, layout, label, binary, stats_lines, out_path,
                ticks=ticks, tick_labels=tick_labels, x0=x0,
                xlim=(min(x0.values()) - edge, max(x0.values()) + edge),
                posthoc_df=tests, x_of=x_of)


# ── Analysis runners ────────────────────────────────────────────────────────

def _cat(parts):
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def run_metrics(jobs, layout: dict, out_dir: str, prefix: str) -> tuple:
    """jobs: (df, {col: label}, binary) tuples. Returns concatenated
    (desc, omnibus, posthoc) tables."""
    os.makedirs(out_dir, exist_ok=True)
    desc_all, omni_all, post_all = [], [], []
    for df, metrics, binary in jobs:
        for col, label in metrics.items():
            if col not in df.columns:
                print(f'  Column {col!r} missing; skipped.')
                continue
            d, o, p = (compare_binary if binary else compare_continuous)(df, layout, col, label)
            desc_all.append(d)
            omni_all.append(o)
            post_all.append(p)
            name = f'{prefix}_{"Categorical_" if binary else ""}{col}.png'
            plot_groups(df, layout, col, label, os.path.join(out_dir, name),
                        _stats_lines(d, o, p, layout, binary), p, binary)
    return _cat(desc_all), _cat(omni_all), _cat(post_all)


def run_half(jobs, layout: dict, out_dir: str, prefix: str) -> tuple:
    """Half-session mixed-design tests (compare_half) and plots (plot_half).
    jobs: (df, {col: label}, binary) tuples. Returns (desc, tests)."""
    os.makedirs(out_dir, exist_ok=True)
    desc_all, test_all = [], []
    for df, metrics, binary in jobs:
        for col, label in metrics.items():
            if col not in df.columns:
                print(f'  Column {col!r} missing; skipped.')
                continue
            d, t = compare_half(df, layout, col, label, binary)
            desc_all.append(d)
            test_all.append(t)
            name = f'{prefix}_{"Categorical_" if binary else ""}{col}.png'
            plot_half(df, layout, col, label, os.path.join(out_dir, name),
                      _half_stats_lines(d, t, layout, binary), t, binary)
    return _cat(desc_all), _cat(test_all)


def _print_significant_half(tests: pd.DataFrame, title: str):
    sig = tests[tests['significant'] == True] if len(tests) else tests  # noqa: E712
    print(f'\n{title}: {len(sig)} of {len(tests)} tests significant (p(Holm) < {ALPHA})')
    for _, r in sig.iterrows():
        g1, g2 = r['comparison'].split(' vs ')
        print(f"  {r['metric'][:45]:45s} {_display(g1, HALF_LAYOUT)} vs "
              f"{_display(g2, HALF_LAYOUT)}  p(Holm)={r['p_holm']:.4g}")


def _print_significant(omni: pd.DataFrame, title: str):
    sig = omni[omni['significant'] == True] if len(omni) else omni  # noqa: E712
    print(f'\n{title}: {len(sig)} of {len(omni)} omnibus tests significant (p < {ALPHA})')
    for _, r in sig.iterrows():
        what = _label(r['test_set']) if r['test_set'] in CLUSTERS else r['test_set']
        print(f"  {r['metric'][:60]:60s} {what:14s} {r['stat_name']}={r['statistic']:.3f}  "
              f"p={r['p_value']:.4g}")


def _write_workbook(tables: dict):
    """Write `tables` ({sheet: DataFrame}) to OUTPUT_EXCEL via a temporary file
    that replaces it only once the save has completed."""
    tmp = os.path.splitext(OUTPUT_EXCEL)[0] + '_writing.tmp.xlsx'
    try:
        with pd.ExcelWriter(tmp, engine='openpyxl') as writer:
            for sheet, tbl in tables.items():
                (tbl if len(tbl) else pd.DataFrame({'note': ['no data']})).to_excel(
                    writer, sheet_name=sheet[:31], index=False)
        os.replace(tmp, OUTPUT_EXCEL)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _print_counts(df: pd.DataFrame, what: str, cols):
    tbl = (df.groupby([cols, 'session_type']).size().unstack(fill_value=0)
             .reindex(columns=SESSION_TYPES, fill_value=0))
    print(f'\n{what}:')
    print(tbl.to_string())


if __name__ == '__main__':
    data = {cond: load_condition(cond, path) for cond, path in INPUT_EXCELS.items()}

    def _both(key):
        return pd.concat([data[c][key] for c in CONDS], ignore_index=True)

    pc, fields, units = _both('pc'), _both('fields'), _both('units')
    half_pc, half_units = _both('half_pc'), _both('half_units')

    _print_counts(pc, 'Place cells per condition and session type', 'condition')
    _print_counts(units, 'Recorded units per condition and session type', 'condition')

    full_dir = os.path.join(PLOTS_DIR, 'FullSession')
    half_dir = os.path.join(PLOTS_DIR, 'HalfSession')

    # ── 1. Whole session: Control vs Test per session type (+ pooled) ───────
    print(f'\n{"=" * 70}\nWhole session: Control vs Test\n{"=" * 70}')
    f_desc, f_omni, f_post = run_metrics(
        [(_with_pooled(pc), FULL_METRICS, False),
         (_with_pooled(fields), FIELD_METRICS, False)],
        FULL_LAYOUT, full_dir, 'Full')
    b_desc, b_omni, b_post = run_metrics(
        [(_with_pooled(units), BINARY_METRICS_ALL_UNITS, True),
         (_with_pooled(pc), BINARY_METRICS_PLACE_CELLS, True)],
        FULL_LAYOUT, full_dir, 'Full')

    # ── 2. Half session: condition (between) x half (within), mixed design ──
    print(f'\n{"=" * 70}\nHalf session: Control GF1/GF2 vs Test GF/Zero\n{"=" * 70}')
    h_desc, h_tests = run_half(
        [(_with_pooled(half_pc), HALF_CONTINUOUS_METRICS, False)],
        HALF_LAYOUT, half_dir, 'Half')
    hb_desc, hb_tests = run_half(
        [(_with_pooled(half_units), HALF_BINARY_METRICS_ALL_UNITS, True),
         (_with_pooled(half_pc), HALF_BINARY_METRICS_PLACE_CELLS, True)],
        HALF_LAYOUT, half_dir, 'Half')

    # ── 3. Pooled plots: all Control vs all Test place cells ────────────────
    # The same tests as the 'All sessions' rows of the sheets above (so no
    # extra sheets), drawn on their own in <FullSession|HalfSession>/Pooled.
    print(f'\n{"=" * 70}\nPooled over session types: Control vs Test\n{"=" * 70}')
    run_metrics([(_with_pooled(pc), FULL_METRICS, False),
                 (_with_pooled(fields), FIELD_METRICS, False),
                 (_with_pooled(units), BINARY_METRICS_ALL_UNITS, True),
                 (_with_pooled(pc), BINARY_METRICS_PLACE_CELLS, True)],
                FULL_POOLED_LAYOUT, os.path.join(full_dir, 'Pooled'), 'Pooled_Full')
    run_half([(_with_pooled(half_pc), HALF_CONTINUOUS_METRICS, False),
              (_with_pooled(half_units), HALF_BINARY_METRICS_ALL_UNITS, True),
              (_with_pooled(half_pc), HALF_BINARY_METRICS_PLACE_CELLS, True)],
             HALF_POOLED_LAYOUT, os.path.join(half_dir, 'Pooled'), 'Pooled_Half')

    data_cols = ['condition', 'session', 'unit', 'session_type', 'animal', 'rec_day', 'cell_id']
    tables = {
        'Full_Desc':            f_desc,
        'Full_KruskalWallis':   f_omni,
        'Full_MannWhitney':     f_post,
        'Full_CatDesc':         b_desc,
        'Full_ChiSquare':       b_omni,
        'Full_Fisher':          b_post,
        'Half_Desc':            h_desc,
        'Half_Tests':           h_tests,
        'Half_CatDesc':         hb_desc,
        'Half_CatTests':        hb_tests,
        'Data_Full':   pc[data_cols + [c for c in list(FULL_METRICS) + list(BINARY_METRICS_PLACE_CELLS)
                                       if c in pc.columns]],
        'Data_Fields': fields[data_cols + list(FIELD_METRICS)],
        'Data_Half':   half_pc[data_cols + ['half'] + [c for c in list(HALF_CONTINUOUS_METRICS)
                                                       + list(HALF_BINARY_METRICS_PLACE_CELLS)
                                                       if c in half_pc.columns]],
    }
    _write_workbook(tables)
    print(f'\nStatistics saved to {OUTPUT_EXCEL}')

    stats_csv = os.path.splitext(OUTPUT_EXCEL)[0] + '_AllStats.csv'
    _cat([t.assign(table=name)[['table'] + list(t.columns)]
          for name, t in tables.items() if not name.startswith('Data_') and len(t)]
         ).to_csv(stats_csv, index=False, encoding='utf-8-sig')
    print(f'All statistics saved to {stats_csv}')

    _print_significant(f_omni, 'Whole session, continuous (Kruskal-Wallis)')
    _print_significant(b_omni, 'Whole session, % outcomes (chi-square)')
    _print_significant_half(h_tests, 'Half session, continuous')
    _print_significant_half(hb_tests, 'Half session, % outcomes')

    print(f'\nPost-hoc pairs for the 8-group test: {POSTHOC_PAIRS!r}')
    print('NOTE: a cell recorded in several sessions of one day counts once per session '
          '(treated as independent).')
    print(f'\nDone. Plots saved under {PLOTS_DIR}')
