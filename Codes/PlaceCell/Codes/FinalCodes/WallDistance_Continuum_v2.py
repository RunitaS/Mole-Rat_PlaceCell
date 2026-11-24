# -*- coding: utf-8 -*-
"""
WallDistance_Continuum_v2.py -- ALL wall-distance (continuum) analyses in one script.

Replaces, for the wall-distance plots and tables:
    BoundaryAnalysis\\EdgeCentreZone_StatsComparison_v6_WallDistance_StatsContinuum_v1.py
    FinalCodes\\WallDistance_PerArenaPlots_v1.py
    FinalCodes\\WallDistance_StabilitySignSig_v1.py
    the WallDist_* plots of ArenaSessionType_StatsComparison_v21.py (INCLUDE_WALL_DISTANCE)
Every number comes from ONE set of inputs (the latest v21 run, PC_Char\\v4; listed with their
dates and MD5 hashes in the 'Inputs' sheet of the stats workbook), and every plot of one metric
has the SAME y-axis, so the Circle, Linear and Open plots can be put side by side.

WHAT IS COMPUTED (the computations are imported from ArenaSessionType_StatsComparison_v21.py,
so they are the same as in the v4 box plots; this script owns the inputs, outputs and plots):
  wall_dist of every place cell = wall distance (2 x 2 cm bins, nearest wall = 0) of the PEAK
  BIN OF ITS RAW RATE MAP, re-scanned from the .ntt + tracking files in v21.WD_PLACE_CELL_DIR
  at every run (v21.scan_place_cell_dir):
      Open   dia 60 cm -> 15 rings: 0 = outermost ring ... 14 = centre
      Linear 80 x 8 cm, 4 rows: outer rows 0, middle rows 1 (track ends 0, LINEAR_ENDS_ARE_WALLS)
      Circle 4 cm wide ring = 2 rows, both against a wall -> every cell at 0
  Fields inherit the distance of their cell's peak bin.
  Metrics: v21.wd_metric_specs() -- Full sheet metrics (incl. within-field rate), field
  metrics, field shape (Circle fields linearized), stability score per stability type, % metrics
  (theta, coherence shuffle, stability significance, SigPos, SigNeg, speed cells, p-/n-speed),
  temporal stability, multi-peak.
  Tests per group (arena / session type / animal), as in v21 (CONTINUOUS_TEST, ALPHA):
      across distance levels (Kruskal-Wallis + Mann-Whitney / chi-square, Holm)
      trend with distance (Spearman / logistic regression; Holm over the metrics of a group)
      pooled over groups: group-adjusted distance effect + distance x group interaction
      groups at matched distance
  Animal-controlled comparisons, additionally (from the v1 continuum script): the two arenas at
  every row both have (MatchedRows; Circle vs Open: row 0; Linear vs Open: row 0, row 1, rows
  0+1): Mann-Whitney U / Fisher (or rank LMM / GEE with CONTINUOUS_TEST = 'LMM'), Holm over the
  tests of a metric (p_holm_tests, the stars) and over the metrics of a test (p_holm_metrics).

LEVELS (all place cells with a wall distance):
  Arena            Circle / Linear / Open together, and each arena alone (Arena\\<arena>\\)
  SessionType      session types within each arena (Circle too: all at distance 0)
  AnimalControlled Circle_vs_Open (Fa8477, Fa5834) and Linear_vs_Open (Fa1059, Fa23BD)
  AnimalID         every animal

COMMON SCALE (user's choice, 20 Nov 2026):
  y   continuous metrics: one range per metric = lowest to highest value of ANY place cell with a
      wall distance (all arenas) + Y_PAD_FRAC, the same in every plot of that metric, so no
      point is cut off. Yes/no metrics: 0-100 % in every plot. Sheet 'YRanges' lists them.
  x   DIST_W_IN inches per wall-distance bin in every trend plot (Circle narrow, Linear wider,
      Open widest) and CAT_W_IN per column of the MatchedRows plots; the axes are AX_H_IN high
      in every plot. Legend and stats text sit under the axes, left-aligned.

OUTPUTS
  WDC = Results\\WallDistanceContinuum (all plots are drawn here):
      WallDistanceSchematic_<arena>.png, PeakDistance_vs_Dwell.png (panels share one y-axis)
      Arena\\WallDist_<metric>.png              + Arena\\<arena>\\WallDist_<metric>.png
      SessionType\\<arena>\\WallDist_<metric>.png
      AnimalControlledArenaType\\<pair>\\WallDist_<metric>.png + \\MatchedRows\\
      AnimalID\\WallDist_<metric>.png
      WallDistance_Stats.xlsx (+ WallDistance_AllStats.csv)
  PC_Char\\v4 (v21.OUTPUT_DIR), copies of the same files in v21's layout:
      ArenaType\\WallDistance\\ (+ <arena>\\), SessionType\\<arena>\\WallDistance\\,
      AnimalControlledArenaType\\WallDistance\\ (the two pairs stitched side by side, also
      MatchedRows\\), AnimalID\\WallDistance\\, WallDistance\\ (schematics,
      WallDist_PeakDistance_vs_Dwell.png, WallDistance_Stats.xlsx)
  Old outputs: the first time this script writes to a folder that holds files from other
  scripts, everything in it is MOVED to <folder>\\_Archive_<timestamp>\\ (nothing deleted).
  Folders it wrote before (marker file MARKER_NAME) are cleared of its own old outputs and
  rewritten; _Archive_* folders are never touched.
  NOTE: the WD sheets of v4\\SessionArena_Stats.xlsx are from the 12 Nov v21 run and are NOT
  updated; use WallDistance_Stats.xlsx. Re-running v21 with INCLUDE_WALL_DISTANCE = True writes
  its own WallDist_* plots into the v4 folders again (own scales).

INPUTS (only read): see INPUTS below / the 'Inputs' sheet.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ArenaSessionType_StatsComparison_v21 as v       # noqa: E402  (sets the Agg backend)
import PlotLayout_Utils as plu                          # noqa: E402
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.lines import Line2D                     # noqa: E402
from matplotlib.ticker import MaxNLocator               # noqa: E402

ccs = v.ccs

# ── Inputs: the latest v21 run (PC_Char\v4) ─────────────────────────────────

PLACE_CELL_DIR = v.WD_PLACE_CELL_DIR       # .ntt + tracking of the place cells (peak-bin scan)
V4_DIR         = v.OUTPUT_DIR              # ...\Results\PC_Char\v4
INPUTS = {
    # 'Full' + 'PlaceFields' (the same file as ...\CrossCheck_DoubleRun\, identical MD5)
    'INPUT_EXCEL':       v.INPUT_EXCEL,
    # field-shape columns incl. the linearized Circle fields (v21 run of 15 Nov)
    'FIELD_SHAPE_EXCEL': v.FIELD_SHAPE_EXCEL,
    # temporal stability, sheet PerCell (v21 run of 12 Nov)
    'TS_EXCEL':          os.path.join(V4_DIR, v.TS_RESULTS_XLSX),
    # within-field firing rate, sheet PerCell (17 Nov)
    'IFR_EXCEL':         os.path.join(V4_DIR, v.IFR_RESULTS_XLSX),
    # multi-peak KDE fields, sheet Fields (12 Nov)
    'MULTIPEAK_EXCEL':   v.MULTIPEAK_EXCEL,
}

# ── Outputs ─────────────────────────────────────────────────────────────────

WDC_DIR = os.path.join(PLACE_CELL_DIR, 'Results', 'WallDistanceContinuum')
STATS_NAME = 'WallDistance_Stats.xlsx'
CSV_NAME = 'WallDistance_AllStats.csv'
MARKER_NAME = '_written_by_WallDistance_Continuum_v2.txt'
ARCHIVE_PREFIX = '_Archive_'
# Stray files of the old helper scripts in V4_DIR, archived with V4_DIR\WallDistance\.
V4_STRAY_FILES = ['WallDist_StabilitySignSig_Stats.xlsx']
NOT_IN_CSV = ('WD_Sessions', 'WD_Cells', 'Inputs', 'YRanges', 'Run')

# ── Plot layout (inches) ────────────────────────────────────────────────────

DIST_W_IN  = 0.5     # x-axis width per wall-distance bin (trend plots)
CAT_W_IN   = 0.9     # x-axis width per column (MatchedRows plots)
AX_H_IN    = 3.6     # axes height, every plot
LEFT_IN    = 1.5     # room for the y label + ticks
TEXT_MIN_W = 5.0     # legend / stats text wrapped to max(axes width, this)
Y_PAD_FRAC = 0.04    # continuous y-range: data range padded by this fraction on each side
PCT_YLIM   = (0.0, 105.0)
XLABEL_H_IN = 1.05   # room under the axes for the tick labels + the 2-line x label
TREND_XLABEL = f'Distance from the nearest wall\n(bins of {v.WD_BIN_CM:g} cm)'
MATCHED_XLABEL = f'Wall distance (bins of {v.WD_BIN_CM:g} cm)\nboth arenas have'


# ── Small helpers ───────────────────────────────────────────────────────────

def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def input_manifest() -> pd.DataFrame:
    rows = []
    for key, path in INPUTS.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f'{key}: {path} not found')
        st = os.stat(path)
        rows.append(dict(input=key, path=path,
                         modified=datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds'),
                         size_bytes=st.st_size, md5=_md5(path)))
    rows.append(dict(input='PLACE_CELL_DIR (peak-bin scan, read at run time)', path=PLACE_CELL_DIR,
                     modified='', size_bytes=np.nan, md5=''))
    df = pd.DataFrame(rows)
    print('\nInputs:')
    for r in df.itertuples():
        print(f'  {r.input:18s} {r.modified:19s} {r.path}')
    return df


def run_info() -> pd.DataFrame:
    def _git(*args):
        try:
            return subprocess.run(['git', *args], cwd=os.path.dirname(os.path.abspath(__file__)),
                                  capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ''
    files = [os.path.abspath(__file__), v.__file__]
    return pd.DataFrame([
        ('run_timestamp', datetime.now().isoformat(timespec='seconds')),
        ('script', os.path.abspath(__file__)),
        ('computations_from', v.__file__),
        ('git_commit', _git('rev-parse', 'HEAD') or 'n/a'),
        ('git_uncommitted (script + v21)', _git('status', '--porcelain', '--', *files) or 'none'),
        ('CONTINUOUS_TEST', v.CONTINUOUS_TEST), ('METRIC_TRANSFORM', v.METRIC_TRANSFORM),
        ('ALPHA', v.ALPHA), ('LINEAR_ENDS_ARE_WALLS', v.LINEAR_ENDS_ARE_WALLS),
        ('LINK_CELLS_ACROSS_ARENAS', v.LINK_CELLS_ACROSS_ARENAS),
        ('python', sys.version.replace('\n', ' ')),
    ], columns=['item', 'value'])


def prepare_dir(d: str, stamp: str, extra_files=()) -> None:
    """Make `d` ready to receive this script's outputs. Holds files of other scripts (no
    MARKER_NAME): move everything (and `extra_files`, paths elsewhere) into
    d\\_Archive_<stamp>\\. Written by this script before: delete its old outputs. _Archive_*
    folders are kept as they are."""
    os.makedirs(d, exist_ok=True)
    entries = [e for e in os.listdir(d) if not e.startswith(ARCHIVE_PREFIX)]
    ours = os.path.isfile(os.path.join(d, MARKER_NAME))
    if ours:
        for e in entries:
            p = os.path.join(d, e)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    else:
        movers = [os.path.join(d, e) for e in entries] + [f for f in extra_files if os.path.exists(f)]
        if movers:
            arch = os.path.join(d, ARCHIVE_PREFIX + stamp)
            os.makedirs(arch, exist_ok=True)
            for p in movers:
                shutil.move(p, os.path.join(arch, os.path.basename(p)))
            print(f'  {len(movers)} old item(s) moved to {arch}')
    with open(os.path.join(d, MARKER_NAME), 'w', encoding='utf-8') as f:
        f.write(f'Outputs of {os.path.abspath(__file__)}, run {stamp}.\n'
                f'Files here are deleted and rewritten by the next run; _Archive_* folders are kept.\n')


# ── Data ────────────────────────────────────────────────────────────────────

def load_full_and_fields() -> tuple:
    """INPUT_EXCEL 'Full' / 'PlaceFields' with the field-shape columns: 2-D arenas from
    'bin_coords' (PlaceFieldShape_Utils), Circle fields (linearized 2-D map) from
    FIELD_SHAPE_EXCEL."""
    full = pd.read_excel(INPUTS['INPUT_EXCEL'], sheet_name='Full')
    fields = pd.read_excel(INPUTS['INPUT_EXCEL'], sheet_name='PlaceFields')
    keys = ['session', 'unit', 'field_number']
    cols = v.pfs.SHAPE_COLUMNS + v.CIRCLE_LIN_COLUMNS
    shp = pd.read_excel(INPUTS['FIELD_SHAPE_EXCEL'], sheet_name='PlaceFields')
    missing = [c for c in cols if c not in shp.columns]
    if missing:
        raise RuntimeError(f'FIELD_SHAPE_EXCEL has no {missing}; run v21 first.')
    shp = shp[keys + cols].drop_duplicates(keys)
    m = fields[keys].merge(shp, on=keys, how='left')
    m.index = fields.index
    fields = v.pfs.add_field_shape(fields)
    is_circle = fields['arena_type'].astype(str) == 'Circle'
    for c in v.pfs.SHAPE_COLUMNS:
        fields.loc[is_circle, c] = m.loc[is_circle, c]
    for c in v.CIRCLE_LIN_COLUMNS:
        fields[c] = m[c]
    return full, fields


def build_tables(full: pd.DataFrame, fields: pd.DataFrame) -> dict:
    """The analysis tables of v21 (before the wall distance is attached)."""
    full_df = v.build_full_metrics_df(full)
    if v.INCLUDE_INFIELD_RATE:
        full_df = v.merge_infield_rate(full_df, pd.read_excel(INPUTS['IFR_EXCEL'], sheet_name='PerCell'))
    tables = {'full': full_df,
              'field': v.build_field_metrics_df(full, fields),
              'shape': v.build_field_shape_df(full, fields),
              'ts': v._tag_groups(pd.read_excel(INPUTS['TS_EXCEL'], sheet_name='PerCell'),
                                  'place cell(s) (temporal stability)')}
    v.RUN_MULTIPEAK_KDE = False          # read MULTIPEAK_EXCEL, do not re-run the KDE
    v.MULTIPEAK_EXCEL = INPUTS['MULTIPEAK_EXCEL']
    tables.update({f'MP_{lvl}': d for lvl, d in v.build_multipeak_dfs(full, fields).items()})
    return tables


def attach_wall_distance(tables: dict, cells_df: pd.DataFrame) -> dict:
    wd = {name: v.attach_distance(d, cells_df, f'Wall distance, table {name!r}')
          for name, d in tables.items()}
    wd['full'] = v.add_speed_direction_flags(wd['full'])
    print('\nPlace cells by arena and wall distance (Full sheet):')
    print(pd.crosstab(wd['full']['arena_type'], wd['full']['wall_dist'], margins=True).to_string())
    return wd


def _select(wd: dict, mask_fn) -> dict:
    return {k: d[mask_fn(d)] for k, d in wd.items()}


# ── Tests of one level ──────────────────────────────────────────────────────

def analyse_level(grp, wd: dict, metrics: list) -> tuple:
    """v21's wall-distance tests of every metric over the groups of `grp` (cells shaded per this
    comparison, as in v21.run_comparison). Returns (grp with shades, {stem: res}, tables)."""
    name = grp.title or grp.noun.capitalize()
    print(f'\n{"-" * 70}\n{name} -- wall distance\n{"-" * 70}')
    full = wd['full']
    grp = v.replace(grp, shade=v.cell_shades(full))
    counts = []
    for g in grp.order:
        c = (full[full[grp.col] == g].groupby('wall_dist')
             .agg(n_rows=('cell_id', 'size'), n_cells=('cell_id', 'nunique')).reset_index())
        if len(c):
            print(f'  {v._label(g)}: ' + ', '.join(f'd{r.wall_dist}={r.n_rows}' for r in c.itertuples()))
            c.insert(0, grp.col, g)
            counts.append(c)
    if not counts:
        print('  No place cells with a wall distance; skipped.')
        return grp, {}, {}

    results = {}
    trend_rows, pooled_rows, omni_rows, match_rows = [], [], [], []
    desc_t, post_t, match_post_t = [], [], []
    for m in metrics:
        cells = v._wd_cells(wd, m, grp)
        subs = {g: cells[cells[grp.col] == g] for g in grp.order if (cells[grp.col] == g).any()}
        res = dict(subs=subs, trend={}, across={}, pooled=None, interaction=None, matched=[])
        for g, sub in subs.items():
            desc, omni, ph = v.across_distance(sub, m)
            if len(desc):
                desc.insert(0, grp.col, g)
                desc_t.append(desc)
            if omni is not None:
                res['across'][g] = omni
                omni_rows.append({grp.col: g, 'metric_stem': m.stem, **omni})
            if len(ph):
                ph.insert(0, grp.col, g)
                post_t.append(ph)
            if m.kind != 'speed_dir':
                res['trend'][g] = v.group_trend(sub, m)
                trend_rows.append({grp.col: g, 'metric_stem': m.stem, 'metric': m.label,
                                   'metric_col': m.col, 'stab_type': m.stab_type or '',
                                   'kind': m.kind, **res['trend'][g]})
        if m.kind != 'speed_dir':
            res['pooled'], res['interaction'] = v.pooled_trend(list(subs.values()), m)
            for what, r in (('distance (group-adjusted)', res['pooled']),
                            ('distance x group interaction', res['interaction'])):
                if r is not None:
                    pooled_rows.append(dict(metric_stem=m.stem, metric=m.label, metric_col=m.col,
                                            effect=what, **r))
        res['matched'], mph = v.matched_distance(list(subs.values()), m, grp)
        match_rows += [dict(metric_stem=m.stem, **o) for o in res['matched']]
        match_post_t += mph
        results[m.stem] = res

    trend_df = pd.DataFrame(trend_rows)
    if 'p_value' not in trend_df.columns:
        trend_df['p_value'] = np.nan
    trend_df['p_holm_metrics'] = trend_df.groupby(grp.col)['p_value'].transform(
        lambda p: ccs.holm(p.to_numpy(float)))
    trend_df['significant'] = trend_df['p_value'] < v.ALPHA
    sig = trend_df[trend_df['significant']]
    print(f'  Significant trends with wall distance (p < {v.ALPHA}): {len(sig)}')
    for _, r in sig.iterrows():
        print(f'    {v._label(r[grp.col]):12s} {r["metric"][:60]:60s} p = {v._fmt_p(r["p_value"])} '
              f'(Holm over metrics {v._fmt_p(r["p_holm_metrics"])})')
    tables = {'WD_Counts': pd.concat(counts, ignore_index=True),
              'WD_Trend': trend_df.drop(columns=['text'], errors='ignore'),
              'WD_Pooled': pd.DataFrame(pooled_rows).drop(columns=['text'], errors='ignore'),
              'WD_Desc': v._concat_tables(desc_t),
              'WD_Omnibus': pd.DataFrame(omni_rows),
              'WD_PostHoc': v._concat_tables(post_t),
              'WD_MatchOmnibus': pd.DataFrame(match_rows),
              'WD_MatchPostHoc': v._concat_tables(match_post_t)}
    return grp, results, tables


def common_rows(arenas: list, units: dict) -> list:
    """Wall distances every arena of `arenas` has (Circle + Open: [0]; Linear + Open: [0, 1])."""
    return sorted(set.intersection(*(set(np.unique(units[a][units[a] >= 0]).tolist())
                                     for a in arenas)))


def two_arena_test(sub: pd.DataFrame, m, labels: list) -> dict:
    """labels[0] vs labels[1] (column 'grp') on the rows of `sub`. Continuous: Mann-Whitney U
    ('KW') or rank LMM likelihood-ratio test ('LMM'). Yes/no: Fisher's exact test ('KW') or
    CellClusteredStats_Utils.compare_binary ('LMM')."""
    a, b = labels
    ya = sub.loc[sub['grp'] == a, '_y'].to_numpy(float)
    yb = sub.loc[sub['grp'] == b, '_y'].to_numpy(float)
    row = dict(group_a=a, group_b=b, n_a=len(ya), n_b=len(yb), n_cells=sub['cell_id'].nunique(),
               n_cells_in_both=int((sub.groupby('cell_id')['grp'].nunique() > 1).sum()))
    if m.kind == 'box':
        row.update(median_a=np.median(ya) if len(ya) else np.nan,
                   median_b=np.median(yb) if len(yb) else np.nan)
    else:
        row.update(pct_a=100.0 * ya.mean() if len(ya) else np.nan,
                   pct_b=100.0 * yb.mean() if len(yb) else np.nan)
    if len(ya) < 2 or len(yb) < 2:
        return dict(row, test='n/a', text='fewer than 2 values in an arena')
    if np.ptp(np.r_[ya, yb]) == 0:
        return dict(row, test='n/a', text='metric is constant')
    if m.kind == 'box':
        if v.CONTINUOUS_TEST == 'LMM':
            return dict(row, **v.rank_model_test(sub, 'y ~ C(grp)', 'y ~ 1'))
        u, p = stats.mannwhitneyu(ya, yb, alternative='two-sided')
        rbc = 2.0 * u / (len(ya) * len(yb)) - 1.0      # > 0: labels[0] higher
        return dict(row, test='Mann-Whitney U', statistic=float(u), stat_name='U',
                    p_value=float(p), rank_biserial=float(rbc),
                    model_note='Mann-Whitney U (rows treated as independent)',
                    text=f'U = {u:.1f}, r_rb = {rbc:+.2f}, {v._p_str(p)}')
    if v.CONTINUOUS_TEST == 'LMM':
        omni, ph, _t = ccs.compare_binary(sub, 'grp', labels, '_y',
                                          lambda x: None if pd.isna(x) else bool(x))
        odds, p, test, note = ph[0]['odds_ratio'], ph[0]['p_raw'], ph[0]['test'], omni['model_note']
    else:
        table = [[int(ya.sum()), int(len(ya) - ya.sum())], [int(yb.sum()), int(len(yb) - yb.sum())]]
        odds, p = stats.fisher_exact(table)
        test, note = "Fisher's exact", "Fisher's exact test (rows treated as independent)"
    return dict(row, test=test, statistic=float(odds), stat_name='OR', p_value=float(p),
                odds_ratio=float(odds), model_note=note, text=f'OR = {odds:.3g}, {v._p_str(p)}')


def analyse_matched_rows(grp, wd: dict, metrics: list, rows: list) -> tuple:
    """The two arenas of an animal-controlled comparison at each shared row and, with >= 2
    rows, the rows pooled. Returns (sets, {stem: (sub, res rows)}, table)."""
    labels = list(grp.order)
    sets = [(f'Row {d}', [d]) for d in rows]
    if len(rows) >= 2:
        sets.append((f'Rows {"+".join(map(str, rows))}', rows))
    out, subs = [], {}
    for m in metrics:
        if m.kind not in ('box', 'cat'):
            continue
        sub = v._wd_cells(wd, m, grp)
        subs[m.stem] = sub = sub[sub['wall_dist'].isin(rows)]
        for name, ds in sets:
            r = two_arena_test(sub[sub['wall_dist'].isin(ds)], m, labels)
            out.append(dict(metric_stem=m.stem, metric=m.label, metric_col=m.col,
                            stab_type=m.stab_type or '', kind=m.kind, rows=name, **r))
    df = pd.DataFrame(out)
    if 'p_value' not in df.columns:
        df['p_value'] = np.nan
    df['p_holm_tests'] = df.groupby('metric_stem')['p_value'].transform(
        lambda p: ccs.holm(p.to_numpy(float)))
    df['p_holm_metrics'] = df.groupby('rows')['p_value'].transform(
        lambda p: ccs.holm(p.to_numpy(float)))
    df['significant_holm_tests'] = df['p_holm_tests'] < v.ALPHA
    df['significant_holm_metrics'] = df['p_holm_metrics'] < v.ALPHA
    per_metric = {stem: (subs[stem], df[df['metric_stem'] == stem].to_dict('records'))
                  for stem in subs}
    return sets, per_metric, df.drop(columns=['text'], errors='ignore')


# ── Common y-range ──────────────────────────────────────────────────────────

def y_ranges(wd: dict, metrics: list) -> dict:
    """{stem: (lo, hi)}: continuous metrics = range of every place cell with a wall distance
    (all arenas; restricted to the stability type of the metric) padded by Y_PAD_FRAC; yes/no
    metrics = PCT_YLIM."""
    everyone = v.Grouping('arena_type', v.ARENA_TYPES, 'arena types')
    out = {}
    for m in metrics:
        if m.kind == 'cat':
            out[m.stem] = PCT_YLIM
        elif m.kind == 'box':
            y = v._wd_cells(wd, m, everyone)['_y'].to_numpy(float)
            if not len(y):
                continue
            lo, hi = float(np.min(y)), float(np.max(y))
            pad = Y_PAD_FRAC * (hi - lo) if hi > lo else max(abs(hi) * 0.1, 1.0)
            out[m.stem] = (lo - pad, hi + pad)
    return out


# ── Plots ───────────────────────────────────────────────────────────────────

def _fig(ax_w: float, title: str, handles: list, lines: list, top_extra: float = 0.0):
    """Figure with an axes of ax_w x AX_H_IN inches at a fixed place, the title above it and
    the legend + stats text under it (left-aligned at the axes)."""
    text_w = max(ax_w, TEXT_MIN_W)
    title_lines = textwrap.wrap(title, max(20, int(text_w * 7)))
    lines = v._wrap_stats(lines, text_w)
    ncol = max(1, min(len(handles), int(text_w // 2.4)))
    legend_rows = int(np.ceil(len(handles) / ncol)) if handles else 0
    top = 0.2 + 0.26 * len(title_lines) + top_extra
    xlabel_h = XLABEL_H_IN
    legend_h = legend_rows * v.LEGEND_ROW_IN
    stats_h = len(lines) * v.STATS_LINE_IN + 0.1
    fig_w = LEFT_IN + text_w + 0.3
    fig_h = top + AX_H_IN + xlabel_h + legend_h + stats_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([LEFT_IN / fig_w, 1 - (top + AX_H_IN) / fig_h, ax_w / fig_w, AX_H_IN / fig_h])
    x0 = LEFT_IN / fig_w
    fig.text(x0, 1 - 0.15 / fig_h, '\n'.join(title_lines), ha='left', va='top',
             fontsize=v.FS_TITLE - 2)
    y_leg = 1 - (top + AX_H_IN + xlabel_h) / fig_h
    if handles:
        fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(x0, y_leg), ncol=ncol,
                   fontsize=v.FS_LEGEND, frameon=False, borderaxespad=0, handletextpad=0.4)
    if lines:
        fig.text(x0, y_leg - (legend_h + 0.05) / fig_h, '\n'.join(lines), ha='left', va='top',
                 fontsize=v.FS_STATS, linespacing=1.3)
    return fig, ax, text_w


def _finish(fig, ax, ylim, ylabel: str, xlabel: str, text_w: float, out_path: str, pct: bool):
    ax.set_ylim(*ylim)
    if pct:
        ax.set_yticks(np.arange(0, 101, 20))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_ylabel(ylabel, fontsize=v.FS_LABEL, labelpad=8)
    ax.set_xlabel(xlabel, fontsize=v.FS_LABEL - 2, labelpad=4)   # fixed 2 lines (XLABEL_H_IN)
    v._style_axes(ax)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=v.FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_trend(m, res: dict, grp, out_path: str, ylim: tuple):
    """x = wall distance (DIST_W_IN per bin); continuous: every cell as a point + median and
    IQR of each group per distance; yes/no: % yes with Wilson 95 % CI. y = ylim."""
    subs = res['subs']
    if not subs or m.kind not in ('box', 'cat'):
        return
    max_d = max(int(s['wall_dist'].max()) for s in subs.values())
    lines = v._wd_stats_lines(res, grp)
    handles = [v._wd_group_handle(g, len(s), res['trend'].get(g, {})) for g, s in subs.items()]
    if m.kind == 'box':
        if grp.shade:
            lines.append(f"Point shade: the cell's {v.FULL_METRICS[v.SHADE_METRIC].lower()} "
                         f"({'rank' if v.SHADE_SCALE == 'rank' else 'value'} within this "
                         f"comparison), darker = higher")
        face = v._cell_color(next(iter(subs)), 0.5)
        handles += [Line2D([0], [0], marker=mk, linestyle='None', markersize=8,
                           markerfacecolor=face, markeredgecolor=v.PAL_BLACK, markeredgewidth=0.5,
                           label=lab) for mk, lab in v._SIGNIFICANCE_LEGENDS.get(m.col, [])]
    else:
        lines.append('Error bars: Wilson 95 % CI.')
    lines.append('y-axis: the same range in every wall-distance plot of this metric.')
    fig, ax, text_w = _fig((max_d + 1) * DIST_W_IN, v._titled(f'{m.label} vs wall distance', grp),
                           handles, lines)

    rng = np.random.default_rng(0)
    for (g, sub), off in zip(subs.items(), v._wd_offsets(len(subs))):
        c = v._color(g)
        d, y = sub['wall_dist'].to_numpy(float), sub['_y'].to_numpy(float)
        lv = np.unique(d)
        if m.kind == 'box':
            styles = v._point_styles(m.col, sub, g, grp.shade)
            faces = np.array([fc for fc, _mk in styles])
            markers = np.array([mk for _fc, mk in styles])
            x = d + off + rng.uniform(-0.05, 0.05, len(d))
            for mk in dict.fromkeys(markers):
                sel = markers == mk
                ax.scatter(x[sel], y[sel], s=v.MARKER_SIZE.get(mk, 30), marker=mk, alpha=0.75,
                           facecolors=faces[sel], edgecolors=v.PAL_BLACK, linewidths=0.4, zorder=2)
            med = np.array([np.median(y[d == l]) for l in lv])
            q1 = np.array([np.percentile(y[d == l], 25) for l in lv])
            q3 = np.array([np.percentile(y[d == l], 75) for l in lv])
            ax.errorbar(lv + off, med, yerr=[med - q1, q3 - med], fmt='-D', color=c['dark'],
                        markersize=6, markeredgecolor=v.PAL_BLACK, markeredgewidth=0.6,
                        linewidth=1.5, elinewidth=1.2, capsize=3, zorder=4)
        else:
            pct, lo, hi = [], [], []
            for l in lv:
                yy = y[d == l]
                a, b = v._wilson(yy.sum(), len(yy))
                pct.append(100.0 * yy.mean())
                lo.append(100.0 * a)
                hi.append(100.0 * b)
            pct = np.array(pct)
            yerr = np.clip([pct - np.array(lo), np.array(hi) - pct], 0, None)
            ax.errorbar(lv + off, pct, yerr=yerr, fmt='-D', color=c['dark'],
                        markerfacecolor=c['face'], markersize=6, markeredgecolor=v.PAL_BLACK,
                        markeredgewidth=0.6, linewidth=1.5, elinewidth=1.2, capsize=3, zorder=4)
    ax.set_xticks(range(max_d + 1))
    ax.set_xlim(-0.5, max_d + 0.5)
    _finish(fig, ax, ylim, textwrap.fill(m.label, 30) if m.kind == 'box' else '% of cells',
            TREND_XLABEL, text_w, out_path, pct=m.kind != 'box')


def plot_matched_rows(m, grp, sub: pd.DataFrame, res: list, sets: list, out_path: str,
                      ylim: tuple):
    """The two arenas side by side at each shared row (+ the pooled rows); stars above the axes
    = p Holm-corrected over the tests of this metric."""
    if not len(sub):
        return
    lines = []
    for (name, _ds), r in zip(sets, res):
        if m.kind == 'box':
            desc = (f'{r["group_a"]} n={r["n_a"]}, median {r.get("median_a", np.nan):.3g} vs '
                    f'{r["group_b"]} n={r["n_b"]}, median {r.get("median_b", np.nan):.3g}')
        else:
            desc = (f'{r["group_a"]} n={r["n_a"]}, {r.get("pct_a", np.nan):.1f} % vs '
                    f'{r["group_b"]} n={r["n_b"]}, {r.get("pct_b", np.nan):.1f} %')
        lines.append(f'{name}: {desc}: {r["text"]}; p(Holm, tests) = '
                     f'{v._fmt_p(r.get("p_holm_tests"))}, p(Holm, metrics) = '
                     f'{v._fmt_p(r.get("p_holm_metrics"))}')
    lines += sorted({r['model_note'] for r in res if isinstance(r.get('model_note'), str)})
    n_both = max(r['n_cells_in_both'] for r in res)
    if n_both:
        lines.append(f'{n_both} cell(s) in both arenas (same unit, same day).')
    lines.append('Stars: p Holm-corrected over the tests of this metric.'
                 + (' Error bars: Wilson 95 % CI.' if m.kind != 'box' else ''))
    lines.append('y-axis: the same range in every wall-distance plot of this metric.')
    handles = []
    for g in grp.order:
        c = v._color(g)
        handles.append(Line2D([0], [0], marker='D', linestyle='', color=c['dark'],
                              markerfacecolor=c['dark'], markeredgecolor=v.PAL_BLACK,
                              markeredgewidth=0.6, label=g))
    fig, ax, text_w = _fig(len(sets) * CAT_W_IN,
                           v._titled(f'{m.label}, arenas at the rows both have', grp),
                           handles, lines, top_extra=0.3)
    rng = np.random.default_rng(0)
    for g, off in zip(grp.order, (-0.2, 0.2)):
        c = v._color(g)
        for i, (_name, ds) in enumerate(sets):
            at = sub[(sub['grp'] == g) & sub['wall_dist'].isin(ds)]
            if not len(at):
                continue
            y = at['_y'].to_numpy(float)
            if m.kind == 'box':
                styles = v._point_styles(m.col, at, g, grp.shade)
                faces = np.array([fc for fc, _mk in styles])
                markers = np.array([mk for _fc, mk in styles])
                x = i + off + rng.uniform(-0.06, 0.06, len(y))
                for mk in dict.fromkeys(markers):
                    sel = markers == mk
                    ax.scatter(x[sel], y[sel], s=v.MARKER_SIZE.get(mk, 30), marker=mk, alpha=0.75,
                               facecolors=faces[sel], edgecolors=v.PAL_BLACK, linewidths=0.4,
                               zorder=2)
                med, q1, q3 = np.median(y), np.percentile(y, 25), np.percentile(y, 75)
                ax.errorbar(i + off, med, yerr=[[med - q1], [q3 - med]], fmt='D', color=c['dark'],
                            markersize=7, markeredgecolor=v.PAL_BLACK, markeredgewidth=0.6,
                            elinewidth=1.4, capsize=4, zorder=4)
            else:
                lo, hi = v._wilson(y.sum(), len(y))
                pct = 100.0 * y.mean()
                yerr = np.clip([[pct - 100.0 * lo], [100.0 * hi - pct]], 0, None)
                ax.errorbar(i + off, pct, yerr=yerr, fmt='D', color=c['dark'],
                            markerfacecolor=c['face'], markersize=7, markeredgecolor=v.PAL_BLACK,
                            markeredgewidth=0.6, elinewidth=1.4, capsize=4, zorder=4)
    for i, r in enumerate(res):
        p = r.get('p_holm_tests')
        ax.annotate(v._stars(p) if pd.notna(p) else 'n/a', xy=(i, 1.0),
                    xycoords=('data', 'axes fraction'), xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=v.FS_LABEL, fontweight='bold',
                    annotation_clip=False)
    if len(sets) > 2:
        ax.axvline(len(sets) - 1.5, color=v.PAL_GRAY, linestyle=(0, (3, 3)), linewidth=0.8)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([name for name, _ds in sets], fontsize=v.FS_TICK)
    ax.set_xlim(-0.5, len(sets) - 0.5)
    _finish(fig, ax, ylim, textwrap.fill(m.label, 30) if m.kind == 'box' else '% of cells',
            MATCHED_XLABEL, text_w, out_path, pct=m.kind != 'box')


def plot_level(grp, results: dict, metrics: list, yr: dict, out_dir: str, per_group: bool = False):
    """WallDist_<stem>.png of every metric (groups together) in out_dir; per_group: also each
    group alone in out_dir\\<group>\\ (its trend + across-distance test only)."""
    n = 0
    for m in metrics:
        res = results.get(m.stem)
        if res is None or m.stem not in yr:
            continue
        plot_trend(m, res, grp, os.path.join(out_dir, f'WallDist_{m.stem}.png'), yr[m.stem])
        n += 1
        if not per_group:
            continue
        for g in res['subs']:
            if g not in res['trend']:
                continue
            g_res = dict(subs={g: res['subs'][g]}, trend={g: res['trend'][g]},
                         across={g: res['across'][g]} if g in res['across'] else {},
                         pooled=None, interaction=None, matched=[])
            g_grp = v.replace(grp, order=[g], title=f'{v._label(g)} arena'
                              if grp.col == 'arena_type' else v._label(g))
            plot_trend(m, g_res, g_grp, os.path.join(out_dir, str(g), f'WallDist_{m.stem}.png'),
                       yr[m.stem])
    print(f'  {n} wall-distance plot(s){" (+ one per group)" if per_group else ""} saved to {out_dir}')


def plot_peak_distribution(full_df, sessions_df, units, out_path) -> pd.DataFrame:
    """v21's peak-distance vs dwell figure, its arena panels on one y-axis."""
    orig = v._save

    def _save_shared(fig, path):
        plu.match_ylims([a for a in fig.axes if plu._y_matchable(a)])
        orig(fig, path)
    v._save = _save_shared
    try:
        return v.plot_peak_distribution(full_df, sessions_df, units, out_path)
    finally:
        v._save = orig


# ── Copies to PC_Char\v4 ─────────────────────────────────────────────────────

def _copy_tree(src: str, dst: str) -> int:
    n = 0
    for root, _dirs, files in os.walk(src):
        for f in files:
            if f == MARKER_NAME or ARCHIVE_PREFIX in root:
                continue
            rel = os.path.relpath(os.path.join(root, f), src)
            os.makedirs(os.path.dirname(os.path.join(dst, rel)), exist_ok=True)
            shutil.copy2(os.path.join(root, f), os.path.join(dst, rel))
            n += 1
    return n


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    manifest = input_manifest()
    info = run_info()
    metrics = v.wd_metric_specs()

    # Output folders (old outputs archived / own old outputs cleared).
    v4_dirs = {'ArenaType': os.path.join(V4_DIR, 'ArenaType', v.WD_PLOTS_SUBDIR),
               'AnimalControlled': os.path.join(V4_DIR, v.ANIMAL_CONTROLLED_DIR, v.WD_PLOTS_SUBDIR),
               'AnimalID': os.path.join(V4_DIR, v.ANIMAL_ID_DIR, v.WD_PLOTS_SUBDIR),
               'Top': os.path.join(V4_DIR, v.WD_PLOTS_SUBDIR)}
    v4_dirs.update({f'Ses{a}': os.path.join(V4_DIR, 'SessionType', a, v.WD_PLOTS_SUBDIR)
                    for a in v.ARENA_TYPES})
    print('\nPreparing output folders ...')
    prepare_dir(WDC_DIR, stamp)
    for key, d in v4_dirs.items():
        prepare_dir(d, stamp, extra_files=[os.path.join(V4_DIR, f) for f in V4_STRAY_FILES]
                    if key == 'Top' else ())

    # Data.
    print(f'\nLoading {INPUTS["INPUT_EXCEL"]} ...')
    full, fields = load_full_and_fields()
    tables = build_tables(full, fields)
    all_units_df = v.build_all_units_df(full)

    geoms = v.make_geometries()
    units = {a: v.wall_dist_units(g, a) for a, g in geoms.items()}
    v.plot_distance_schematics(geoms, units, WDC_DIR)
    print(f'\nScanning {PLACE_CELL_DIR} for the raw-map peak bin of every unit ...')
    sessions_df, cells_df = v.scan_place_cell_dir(geoms, units)
    wd = attach_wall_distance(tables, cells_df)
    occ_df = plot_peak_distribution(wd['full'], sessions_df, units,
                                    os.path.join(WDC_DIR, 'PeakDistance_vs_Dwell.png'))

    yr = y_ranges(wd, metrics)
    yr_df = pd.DataFrame([dict(metric_stem=s, y_min=lo, y_max=hi) for s, (lo, hi) in yr.items()])
    all_tables = {}

    # 1. Arenas together (+ each arena alone).
    grp, res, tbl = analyse_level(v.ARENA_GROUPING, wd, metrics)
    plot_level(grp, res, metrics, yr, os.path.join(WDC_DIR, 'Arena'), per_group=True)
    all_tables.update({f'Arena_{k}': t for k, t in tbl.items()})

    # 2. Session types within each arena.
    for arena in v.ARENA_TYPES:
        grp = v.session_grouping(arena)
        sel = _select(wd, lambda d, a=arena, g=grp: (d['arena_type'] == a) & d['session_type'].isin(g.order))
        grp, res, tbl = analyse_level(grp, sel, metrics)
        plot_level(grp, res, metrics, yr, os.path.join(WDC_DIR, 'SessionType', arena))
        for k, t in tbl.items():
            if len(t) and 'arena_type' not in t.columns:
                t.insert(0, 'arena_type', arena)
            all_tables[f'Ses{arena}_{k}'] = t

    # 3. Animal-controlled arena comparisons (+ MatchedRows).
    ac_dirs = []
    for name, (animals, arenas) in v.ANIMAL_CONTROLLED_COMPARISONS.items():
        grp = v.animal_controlled_grouping(animals, arenas)
        wanted = {a.lower() for a in animals}
        sel = _select(wd, lambda d, w=wanted, g=grp: d['animal'].astype(str).str.lower().isin(w)
                      & d['arena_type'].isin(g.order))
        grp, res, tbl = analyse_level(grp, sel, metrics)
        out_dir = os.path.join(WDC_DIR, v.ANIMAL_CONTROLLED_DIR, name)
        ac_dirs.append(out_dir)
        plot_level(grp, res, metrics, yr, out_dir)
        rows = common_rows(list(grp.order), units)
        print(f'  MatchedRows ({", ".join(grp.order)} at rows {rows})')
        sets, per_metric, mr_df = analyse_matched_rows(grp, sel, metrics, rows)
        for m in metrics:
            if m.stem in per_metric and m.stem in yr:
                sub, r = per_metric[m.stem]
                plot_matched_rows(m, grp, sub, r, sets,
                                  os.path.join(out_dir, 'MatchedRows', f'WallDist_{m.stem}.png'),
                                  yr[m.stem])
        tag = 'AC' + ''.join(grp.order)
        tbl['WD_MatchedRows'] = mr_df
        for k, t in tbl.items():
            if len(t):
                t.insert(0, 'animals', ', '.join(animals))
            all_tables[f'{tag}_{k}'] = t

    # 4. Animal IDs.
    grp, res, tbl = analyse_level(v.animal_id_grouping(all_units_df), wd, metrics)
    plot_level(grp, res, metrics, yr, os.path.join(WDC_DIR, v.ANIMAL_ID_DIR))
    all_tables.update({f'Animal_{k}': t for k, t in tbl.items()})

    # Statistics.
    sheets = {'Inputs': manifest, 'Run': info, 'YRanges': yr_df,
              'WD_Sessions': sessions_df.drop(columns=['_skey']),
              'WD_Cells': cells_df.drop(columns=['_skey', '_ukey']),
              'WD_PeakOccupancy': occ_df, **all_tables}
    stats_xlsx = os.path.join(WDC_DIR, STATS_NAME)
    with pd.ExcelWriter(stats_xlsx, engine='openpyxl') as w:
        for s, t in sheets.items():
            t.to_excel(w, sheet_name=s[:31], index=False)
    parts = []
    for s, t in sheets.items():
        if s in NOT_IN_CSV or t is None or not len(t):
            continue
        t = t.copy()
        t.insert(0, 'table', s)
        parts.append(t)
    pd.concat(parts, ignore_index=True).to_csv(os.path.join(WDC_DIR, CSV_NAME), index=False,
                                               encoding='utf-8-sig')
    print(f'\nStatistics saved to {stats_xlsx} (+ {CSV_NAME})')

    # Copies in v21's layout under PC_Char\v4.
    print('\nCopying to', V4_DIR)
    n = _copy_tree(os.path.join(WDC_DIR, 'Arena'), v4_dirs['ArenaType'])
    for arena in v.ARENA_TYPES:
        n += _copy_tree(os.path.join(WDC_DIR, 'SessionType', arena), v4_dirs[f'Ses{arena}'])
    n += _copy_tree(os.path.join(WDC_DIR, v.ANIMAL_ID_DIR), v4_dirs['AnimalID'])
    for f in os.listdir(WDC_DIR):
        if f.startswith('WallDistanceSchematic_'):
            shutil.copy2(os.path.join(WDC_DIR, f), v4_dirs['Top'])
            n += 1
    shutil.copy2(os.path.join(WDC_DIR, 'PeakDistance_vs_Dwell.png'),
                 os.path.join(v4_dirs['Top'], 'WallDist_PeakDistance_vs_Dwell.png'))
    shutil.copy2(stats_xlsx, os.path.join(v4_dirs['Top'], STATS_NAME))
    print(f'  {n + 2} file(s) copied')
    v._combine_side_by_side(ac_dirs, v4_dirs['AnimalControlled'])

    print(f'\nDone. Plots in {WDC_DIR} and the WallDistance\\ folders under {V4_DIR}.')


if __name__ == '__main__':
    main()
