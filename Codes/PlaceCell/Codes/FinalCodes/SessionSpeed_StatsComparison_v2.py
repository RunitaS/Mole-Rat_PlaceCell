# -*- coding: utf-8 -*-
"""
Running speed of every recording session -- the mean speed and the distribution
of instantaneous speeds -- compared across the same conditions as
ArenaSessionType_StatsComparison_v19.py, with the same tests and the same figure
formatting (its functions are imported, not copied, so a formatting or
statistics change in v19 applies here too):

  1. Session types WITHIN each arena type
         Open   : Cntrl, Rotate, Zero
         Linear : 0, 90, 180, 270
         Circle : Stnd, NoRot, Rot
  2. Arena types as a whole (Circle / Linear / Open, all session types pooled)
  3. Animal-controlled arena types (ANIMAL_CONTROLLED_COMPARISONS in v19)
         Circle vs Open : Fa8477, Fa5834
         Linear vs Open : Fa1059, Fa23BD
     saved side by side, each panel with its own statistics.

One row = one recording session (no units, no spike data). Tests come from
v19's CONTINUOUS_TEST ('KW': Kruskal-Wallis + pairwise Mann-Whitney U,
Holm-Bonferroni corrected, sessions treated as independent).

V2: speed distribution (and the import moved from v18, now in OldVers\\, to v19).

Sessions: every folder <root>\\<animal>\\<arena>\\<day>\\<session> holding one
*_cm.csv file, from TRACKING_ROOTS. A session found in several roots is taken
from the first one. QualFiltData_AllCells comes first because its tracking is
what the place-cell analyses use (off-arena frames corrected, some sessions
trimmed); Corr_Data adds the sessions with no spike-sorted units, which
QualFiltData_AllCells does not have.

Speed, per session, using PlaceCell_Main_v21's own tracking functions:
    1. _load_tracking: cleaning, as in the place-cell analyses.
    2. _smooth_tracking_position: artifact-jump removal, Gaussian smoothing.
    3. _instantaneous_speed: speed of every step between consecutive frames
       (cm/s); n frames give n - 1 values (the copy of the last step that
       _instantaneous_speed appends for the last frame is dropped for the
       distribution).
    4. Time-weighted mean over the whole session (= distance / duration).
Mean-speed metrics (SPEED_METRICS):
    mean_speed_cms         every frame, including immobile ones
    mean_speed_moving_cms  frames with speed >= pc.IMMOBILITY_SPEED_CMS only
The per-session table also has pct_time_moving and the as-tracked speeds.

Speed distribution (V2). The n - 1 instantaneous speeds of a session, immobile
ones included, are binned in SPEED_DIST_BIN_CMS (10 cm/s) bins from 0 to
SPEED_DIST_MAX_CMS (90 cm/s): 9 bins, [0, 10), [10, 20), ..., [80, 90]. Each
bin's value is the % of the session's speed values in 0-90 cm/s that fall in
it (the 9 bins sum to 100 %); values above 90 cm/s are left out and counted
(n_speed_above_max). The per-session % are columns 'pct_speed_<lo>_<hi>cms'.
Per comparison (levels 1-3), the sessions of each group are pooled:
    mean % per bin (+ SEM, median, quartiles; sheet '*_SpdDist_Desc').
    Whole distribution: permutation test of the session labels. Statistic =
        between-group sum of squares of the 9-bin % vectors,
        sum_g n_g * sum_b (mean_gb - mean_b)^2; the labels are shuffled freely
        across sessions N_PERMUTATIONS times, p = (1 + #(T_perm >= T)) / (1 + N).
        R2 = between-group SS / total SS. Sheet '*_SpdDist_Perm'. Bins with a
        large spread of % across sessions (usually the slow ones) weigh most.
    Per bin: the same test as every other metric (compare_groups: 'KW' =
        Kruskal-Wallis + pairwise Mann-Whitney U, Holm-corrected over the pairs
        within the bin). The 9 omnibus p-values are also Holm-corrected across
        the bins of the comparison (p_holm_bins). Sheets '*_SpdDist_Omnibus',
        '*_SpdDist_PostHoc' (bin_p_holm_bins = p_holm_bins of the pair's bin).
        Bins in which every session has the same % (e.g. all 0) are not tested.
    Plots, in the SPEED_DIST_SUBDIR folder of every comparison folder:
        SpeedDist_MeanCurve.png  mean % +- SEM per bin and group; the tests
                                 underneath
        SpeedDist_BoxPlots.png   every session's % per bin and group (box + points)
        Stars above a bin: p_holm_bins (* < 0.05, ** < 0.01, *** < 0.001).
    Mobile speed distribution: the same values with the immobile ones
        (< MOB_DIST_MIN_SPEED_CMS, 0.5 cm/s; counted as n_speed_immobile)
        discarded; bins [0.5, 10), [10, 20), ..., [80, 90], each the % of the
        session's mobile values in 0.5-90 cm/s (the 9 bins sum to 100 %).
        Columns 'pct_mob_speed_<lo>_<hi>cms', the same tests, sheets
        '*_SpdDistMob_*', plots SpeedDist_Mob_MeanCurve.png and
        SpeedDist_Mob_BoxPlot.png in the same folder.
    All of these treat sessions as independent (several sessions per animal
    and day); level 3 controls for which animals were recorded in which arena.

Tracking scale: each *_cm.csv was scaled by stretching that session's pixel
range to the arena size, so a session whose pixel range was not the whole arena
gets the wrong cm/pixel, and speed is directly proportional to it (animal stayed
on part of the track: too large, speed inflated). The cm/pixel of every session
is measured from its pixel and cm columns; sessions more than SCALE_OUTLIER_FRAC
from the median of their animal + arena are flagged, and with
CORRECT_SCALE_OUTLIERS their positions are rescaled to that median before
smoothing (mean speed and distribution alike). The *_cm.csv files are not
modified.

Output (OUTPUT_DIR):
    SessionType\\<arena>\\                  box plots, session types of <arena>
    SessionType\\<arena>\\Histograms\\
    SessionType\\<arena>\\SpeedDistribution\\
    ArenaType\\                             box plots, arena types
    ArenaType\\Histograms\\
    ArenaType\\SpeedDistribution\\
    AnimalControlledArenaType\\             the comparisons side by side (same subfolders)
    AnimalControlledArenaType\\AnimalControlled_Stats.xlsx  level-3 tables only
    SessionSpeed_Stats.xlsx                 'PerSession', 'Ses<arena>_*', 'Arena_*', 'AC<A><B>_*'
    SessionSpeed_AllStats.csv               the statistics tables, long format
    RunSnapshots\\<YYYYmmdd_HHMMSS>\\        code + RunMetadata.xlsx, saved before the analysis
"""

import glob
import importlib.metadata
import os
import platform
import re
import shutil
import sys
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Also applies v19's figure style (rcParams).
import ArenaSessionType_StatsComparison_v19 as asc
import PlotLayout_Utils as plu    # shared y-range of the side-by-side plots

ccs = asc.ccs
pc  = asc.pc

# ── Parameters ──────────────────────────────────────────────────────────────

_THESIS = r'X:\NMR_group_data\Runita\Analysis\Thesis'

# In order of preference: a session in several roots is taken from the first.
TRACKING_ROOTS = [
    os.path.join(_THESIS, r'Corr_Data_SpkQltyFilt\SpikeQualityFilt\QualFiltData_AllCells'),
    os.path.join(_THESIS, 'Corr_Data'),
]

OUTPUT_DIR = os.path.join(_THESIS, r'Corr_Data_SpkQltyFilt\SpikeQualityFilt'
                                   r'\SpikeQualityFilt_v1_PlaceCellData'
                                   r'\PC_True_SiSkaggsAdptBin_Final\Results\SessionSpeed')

STATS_XLSX = os.path.join(OUTPUT_DIR, 'SessionSpeed_Stats.xlsx')
STATS_CSV  = os.path.join(OUTPUT_DIR, 'SessionSpeed_AllStats.csv')
RUN_SNAPSHOT_SUBDIR = 'RunSnapshots'

# Sessions whose cm/pixel is further than this fraction from the median of their
# animal + arena are flagged (scale_outlier). True: their positions are rescaled
# to that median. False: flagged only; every speed is as tracked.
SCALE_OUTLIER_FRAC     = 0.2
CORRECT_SCALE_OUTLIERS = True

SPEED_METRICS = {
    'mean_speed_cms':        'Mean running speed, whole session (cm/s)',
    'mean_speed_moving_cms': 'Mean running speed, moving frames only (cm/s)',
}

# Speed distribution: bin width and upper end (cm/s); values above the upper
# end are left out of the % (and counted).
SPEED_DIST_BIN_CMS = 10
SPEED_DIST_MAX_CMS = 90
N_PERMUTATIONS     = 10_000   # label shuffles for the whole-distribution test
PERMUTATION_SEED   = 0
SPEED_DIST_SUBDIR  = 'SpeedDistribution'
# Mobility-only distribution: speed values below this (cm/s) are discarded and
# the rest of 0 .. SPEED_DIST_MAX_CMS re-binned (first bin [0.5, 10)).
MOB_DIST_MIN_SPEED_CMS = 0.5


def _edge_tag(x: float) -> str:
    return f'{int(x):02d}' if float(x).is_integer() else f'{x:g}'


class DistSpec:
    """One speed distribution: its bins, per-session % columns, table names and plots."""

    def __init__(self, key, col_prefix, edges, desc, title, ylabel, curve_png, box_png):
        self.key, self.desc, self.title, self.ylabel = key, desc, title, ylabel
        self.curve_png, self.box_png = curve_png, box_png
        self.edges = np.asarray(edges, dtype=float)
        self.bins = list(zip(self.edges[:-1].tolist(), self.edges[1:].tolist()))
        self.metrics = {f'{col_prefix}_{_edge_tag(lo)}_{_edge_tag(hi)}cms':
                        f'Speed {lo:g}-{hi:g} cm/s ({ylabel})' for lo, hi in self.bins}
        self.cols = list(self.metrics)

    def __repr__(self):   # for RunMetadata.xlsx
        return f'DistSpec({self.key}, edges={self.edges.tolist()}, plots={self.curve_png}, {self.box_png})'


_SPEED_DIST_EDGES = np.arange(0, SPEED_DIST_MAX_CMS + SPEED_DIST_BIN_CMS / 2, SPEED_DIST_BIN_CMS)

# All speed values in 0 .. SPEED_DIST_MAX_CMS.
SPEED_DIST = DistSpec(
    key='SpdDist', col_prefix='pct_speed', edges=_SPEED_DIST_EDGES,
    desc=f'Speed distribution ({len(_SPEED_DIST_EDGES) - 1} bins of {SPEED_DIST_BIN_CMS} cm/s, '
         f'0-{SPEED_DIST_MAX_CMS} cm/s)',
    title='Distribution of instantaneous speed', ylabel='% of speed values',
    curve_png='SpeedDist_MeanCurve.png', box_png='SpeedDist_BoxPlots.png')

# The same values without the immobile ones (< MOB_DIST_MIN_SPEED_CMS); % of
# the mobile values in MOB_DIST_MIN_SPEED_CMS .. SPEED_DIST_MAX_CMS.
MOB_SPEED_DIST = DistSpec(
    key='SpdDistMob', col_prefix='pct_mob_speed',
    edges=np.concatenate([[MOB_DIST_MIN_SPEED_CMS], _SPEED_DIST_EDGES[1:]]),
    desc=f'Mobile speed distribution ({len(_SPEED_DIST_EDGES) - 1} bins, '
         f'{MOB_DIST_MIN_SPEED_CMS:g}-{SPEED_DIST_MAX_CMS} cm/s)',
    title=f'Mobile speed distribution (≥ {MOB_DIST_MIN_SPEED_CMS:g} cm/s)',
    ylabel='% of mobile speed values',
    curve_png='SpeedDist_Mob_MeanCurve.png', box_png='SpeedDist_Mob_BoxPlot.png')

DIST_SPECS = [SPEED_DIST, MOB_SPEED_DIST]

# ── Sessions and speed ──────────────────────────────────────────────────────


def find_sessions() -> list:
    """(session, csv_path, root) of every session folder in TRACKING_ROOTS, the
    first root winning. Folders with an arena in their path and no tracking file
    in any root are reported; so are folders with several."""
    sessions, no_tracking = {}, {}
    for root in TRACKING_ROOTS:
        for dirpath in glob.glob(os.path.join(glob.escape(root), '*', '*', '*', '*')):
            if not os.path.isdir(dirpath):
                continue
            rel = os.path.relpath(dirpath, root)
            key = asc._session_key(rel)
            if key in sessions:
                continue
            tracking = [f for f in os.listdir(dirpath) if f.lower().endswith('_cm.csv')]
            if len(tracking) == 1:
                sessions[key] = (rel, os.path.join(dirpath, tracking[0]), root)
            elif len(tracking) > 1:
                print(f'WARNING: {len(tracking)} *_cm.csv files in {dirpath}; skipped.')
            elif asc.extract_arena_type(rel) is not None:
                no_tracking[key] = dirpath
    for key, dirpath in sorted(no_tracking.items()):
        if key not in sessions:
            print(f'WARNING: no *_cm.csv file for session folder {dirpath}; skipped.')
    return sorted(sessions.values())


def measure_cm_per_px(csv_path: str) -> float:
    """cm/pixel of a tracking file: straight-line fit of its cm column on its
    pixel column, on the axis with the larger pixel range (the track's long axis).
    Frames marked was_interpolated have no matching pixel position."""
    data = pd.read_csv(csv_path)
    px = data.iloc[:, [1, 2]].to_numpy(dtype=float)
    cm = data.iloc[:, [3, 4]].to_numpy(dtype=float)
    ok = np.isfinite(px).all(axis=1) & np.isfinite(cm).all(axis=1) & ~np.isin(cm[:, 0], [1, -1])
    if 'was_interpolated' in data.columns:
        ok &= ~data['was_interpolated'].astype(bool).to_numpy()
    axis = int(np.argmax(np.ptp(px[ok], axis=0)))
    return float(np.polyfit(px[ok, axis], cm[ok, axis], 1)[0])


def speed_distribution(v: np.ndarray, spec: DistSpec) -> dict:
    """% of the (finite) speed values in spec.edges[0] .. spec.edges[-1] that
    fall in each bin of `spec` (the last bin includes its upper edge)."""
    in_range = (v >= spec.edges[0]) & (v <= spec.edges[-1])
    counts = np.histogram(v[in_range], bins=spec.edges)[0]
    n = int(in_range.sum())
    return {col: 100.0 * c / n if n else np.nan for col, c in zip(spec.cols, counts)}


def speed_distributions(step_speed: np.ndarray) -> dict:
    """Every distribution of DIST_SPECS, plus the number of values, of values
    above SPEED_DIST_MAX_CMS and of immobile values (< MOB_DIST_MIN_SPEED_CMS)."""
    v = step_speed[np.isfinite(step_speed)]
    out = {'n_speed_values':    int(len(v)),
           'n_speed_above_max': int((v > SPEED_DIST_MAX_CMS).sum()),
           'n_speed_immobile':  int((v < MOB_DIST_MIN_SPEED_CMS).sum())}
    for spec in DIST_SPECS:
        out.update(speed_distribution(v, spec))
    return out


def session_speed(csv_path: str, scale: float = 1.0) -> dict:
    """Whole-session speed of one tracking file, positions multiplied by `scale`
    before smoothing, so the artifact-jump and immobility thresholds (cm/s)
    apply to the corrected speeds."""
    x, y, t = pc._load_tracking(csv_path, pc.arena_width_cm)
    if len(t) < 2:
        return {'n_frames': len(t)}
    x, y = pc._smooth_tracking_position(x * scale, y * scale, t)
    speed = pc._instantaneous_speed(x, y, t)
    dt = np.diff(t) * 1e-6
    dt = np.append(dt, dt[-1])
    ok = np.isfinite(speed) & (dt > 0)
    moving = ok & (speed >= pc.IMMOBILITY_SPEED_CMS)
    return {
        'n_frames':              len(t),
        'duration_min':          (t[-1] - t[0]) * 1e-6 / 60,
        'mean_speed_cms':        np.average(speed[ok], weights=dt[ok]),
        'mean_speed_moving_cms': np.average(speed[moving], weights=dt[moving]) if moving.any() else np.nan,
        'pct_time_moving':       100.0 * dt[moving].sum() / dt[ok].sum(),
        # speed[:-1]: the n - 1 steps between frames (the last entry repeats the last step)
        **speed_distributions(speed[:-1]),
    }


def build_session_df() -> pd.DataFrame:
    """One row per session: its groups, tracking scale, speed and speed distribution."""
    sessions = find_sessions()
    print(f'Found {len(sessions)} session(s) with tracking.')
    rows = []
    for i, (rel, csv_path, root) in enumerate(sessions, start=1):
        print(f'  [{i}/{len(sessions)}] {rel}')
        rows.append({'session': rel, 'unit': rel, 'tracking_file': csv_path, 'tracking_root': root,
                     'cm_per_px': measure_cm_per_px(csv_path), **session_speed(csv_path)})
    # 'unit' only makes each row its own observation in v19's functions.
    df = asc._tag_groups(pd.DataFrame(rows), 'session(s)')

    df['mean_speed_as_tracked_cms']        = df['mean_speed_cms']
    df['mean_speed_moving_as_tracked_cms'] = df['mean_speed_moving_cms']
    df['cm_per_px_median'] = df.groupby(['animal', 'arena_type'])['cm_per_px'].transform('median')
    df['scale_outlier'] = (df['cm_per_px'] / df['cm_per_px_median'] - 1).abs() > SCALE_OUTLIER_FRAC
    df['scale_applied'] = 1.0
    if df['scale_outlier'].any():
        print(f'\n{int(df["scale_outlier"].sum())} session(s) with an outlying cm/pixel '
              f'({"corrected" if CORRECT_SCALE_OUTLIERS else "NOT corrected"}):')
    for idx, r in df[df['scale_outlier']].iterrows():
        scale = r['cm_per_px_median'] / r['cm_per_px']
        print(f'    {r["session"]}: {r["cm_per_px"]:.3f} cm/px, median {r["cm_per_px_median"]:.3f}, '
              f'as tracked {r["mean_speed_cms"]:.2f} cm/s')
        if CORRECT_SCALE_OUTLIERS:
            for col, val in session_speed(r['tracking_file'], scale).items():
                df.loc[idx, col] = val
            df.loc[idx, 'scale_applied'] = scale

    above = df[df['n_speed_above_max'] > 0]
    print(f'\nSpeed values above {SPEED_DIST_MAX_CMS} cm/s (left out of the distribution): '
          f'{int(df["n_speed_above_max"].sum())} of {int(df["n_speed_values"].sum())}, '
          f'in {len(above)} session(s)')
    for _, r in above.iterrows():
        print(f'    {r["session"]}: {int(r["n_speed_above_max"])} of {int(r["n_speed_values"])}')
    pct_immobile = 100.0 * df['n_speed_immobile'].sum() / df['n_speed_values'].sum()
    print(f'Speed values below {MOB_DIST_MIN_SPEED_CMS:g} cm/s (left out of the mobile distribution): '
          f'{int(df["n_speed_immobile"].sum())} of {int(df["n_speed_values"].sum())} ({pct_immobile:.1f} %)')
    return df


# ── Statistics ──────────────────────────────────────────────────────────────

_CELL_SUMMARY = re.compile(r'\d+ observations from \d+ cells \([^)]*\)(, \d+ recording days, \d+ animals)?')
_compare_groups_v19 = asc.compare_groups


def compare_sessions(df: pd.DataFrame, metric_col: str, metric_label: str, grp: asc.Grouping):
    """v19's compare_groups, with its tables and the note under each figure
    counting sessions, recording days and animals instead of cells."""
    desc_df, omnibus, posthoc_df = _compare_groups_v19(df, metric_col, metric_label, grp)
    sub = df[df[grp.col].isin(grp.order) & pd.to_numeric(df[metric_col], errors='coerce').notna()]
    summary = (f'{len(sub)} sessions, {len(sub[["animal", "rec_day"]].drop_duplicates())} recording days, '
               f'{sub["animal"].nunique()} animals')
    if isinstance(omnibus.get('model_note'), str):
        omnibus['model_note'] = _CELL_SUMMARY.sub(summary, omnibus['model_note'])
    if len(desc_df):
        desc_df = desc_df.rename(columns={'n_cells': 'n_animals'})
        desc_df['n_animals'] = [sub.loc[sub[grp.col] == g, 'animal'].nunique() for g in desc_df[grp.col]]
    return desc_df, omnibus, posthoc_df


# asc.run_all_comparisons and asc.plot_histogram_by_group call it by this name.
asc.compare_groups = compare_sessions


# ── Speed distribution: tests ───────────────────────────────────────────────

def _dist_matrices(df: pd.DataFrame, grp: asc.Grouping, spec: DistSpec) -> dict:
    """{group: (n_sessions, n_bins) array of the per-session %}, groups in plot
    order, sessions with no distribution left out."""
    sub = df[df[grp.col].isin(grp.order)].dropna(subset=spec.cols)
    return {g: sub.loc[sub[grp.col] == g, spec.cols].to_numpy(float)
            for g in grp.order if (sub[grp.col] == g).any()}


def permutation_test(df: pd.DataFrame, grp: asc.Grouping, spec: DistSpec) -> dict:
    """Whole-distribution test: between-group sum of squares of the per-session
    % vectors, against N_PERMUTATIONS free shuffles of the group labels across
    sessions. Groups with < 2 sessions are left out."""
    mats = {g: m for g, m in _dist_matrices(df, grp, spec).items() if len(m) >= 2}
    row = {'metric': spec.desc,
           'n_groups': len(mats), 'groups': ', '.join(f'{g} (n={len(m)})' for g, m in mats.items())}
    if len(mats) < 2:
        return {**row, 'test': 'insufficient data', 'statistic': np.nan, 'p_value': np.nan,
                'significant': False}

    P = np.vstack(list(mats.values()))
    codes = np.repeat(np.arange(len(mats)), [len(m) for m in mats.values()])
    n_g = np.bincount(codes).astype(float)
    grand = P.mean(axis=0)

    def between_ss(c):
        means = (np.eye(len(mats))[c].T @ P) / n_g[:, None]
        return float((n_g[:, None] * (means - grand) ** 2).sum())

    t_obs = between_ss(codes)
    rng = np.random.default_rng(PERMUTATION_SEED)
    t_perm = np.array([between_ss(rng.permutation(codes)) for _ in range(N_PERMUTATIONS)])
    p = (1 + int((t_perm >= t_obs * (1 - 1e-12)).sum())) / (1 + N_PERMUTATIONS)
    total_ss = float(((P - grand) ** 2).sum())
    n_days = len(df[df[grp.col].isin(mats)][['animal', 'rec_day']].drop_duplicates())
    return {**row, 'test': f'Permutation test ({N_PERMUTATIONS} shuffles of the session labels)',
            'stat_name': 'SS_between', 'statistic': t_obs, 'p_value': p,
            'R2': t_obs / total_ss if total_ss > 0 else np.nan,
            'n_permutations': N_PERMUTATIONS, 'seed': PERMUTATION_SEED,
            'significant': bool(p < asc.ALPHA),
            'model_note': f'{len(P)} sessions, {n_days} recording days, '
                          f'{df[df[grp.col].isin(mats)]["animal"].nunique()} animals; '
                          f'labels shuffled freely across sessions (sessions treated as independent)'}


def run_speed_distribution(df: pd.DataFrame, grp: asc.Grouping, out_dir: str, spec: DistSpec) -> dict:
    """Tests and plots of one speed distribution (`spec`) in one comparison.
    Returns {table_name: table}."""
    desc_all, omni_rows, post_all = [], [], []
    for (lo, hi), (col, label) in zip(spec.bins, spec.metrics.items()):
        d, o, p = compare_sessions(df, col, label, grp)
        tag = {'speed_bin': f'{lo:g}-{hi:g} cm/s', 'bin_lo_cms': lo, 'bin_hi_cms': hi}
        for tbl in (d, p):
            if len(tbl):
                for i, (k, v) in enumerate(tag.items()):
                    tbl.insert(1 + i, k, v)
                (desc_all if tbl is d else post_all).append(tbl)
        omni_rows.append({'metric': o['metric'], **tag, 'metric_col': col,
                          **{k: v for k, v in o.items() if k != 'metric'}})

    omni_df = pd.DataFrame(omni_rows)
    omni_df['p_holm_bins'] = ccs.holm(omni_df['p_value'].to_numpy(float))
    omni_df['significant_bins'] = omni_df['p_holm_bins'] < asc.ALPHA
    post_df = asc._concat_tables(post_all)
    if len(post_df):
        post_df['bin_p_holm_bins'] = post_df['speed_bin'].map(
            dict(zip(omni_df['speed_bin'], omni_df['p_holm_bins'])))
    perm = permutation_test(df, grp, spec)

    print(f'{spec.desc}: {perm["test"]}, p = {asc._fmt_p(perm["p_value"])}'
          + (f', R2 = {perm["R2"]:.3f}' if 'R2' in perm else ''))
    for _, r in omni_df[omni_df['significant_bins']].iterrows():
        print(f'    {r["speed_bin"]:12s} {r["test"]}: p = {asc._fmt_p(r["p_value"])}, '
              f'p(Holm, bins) = {asc._fmt_p(r["p_holm_bins"])}')

    os.makedirs(out_dir, exist_ok=True)
    p_bins = omni_df['p_holm_bins'].to_numpy(float)
    plot_distribution_curve(df, grp, omni_df, post_df, perm, p_bins,
                            os.path.join(out_dir, spec.curve_png), spec)
    plot_distribution_boxes(df, grp, p_bins, os.path.join(out_dir, spec.box_png), spec)
    print(f'{spec.desc}: plots saved to {out_dir}')
    return {f'{spec.key}_Desc':    asc._concat_tables(desc_all),
            f'{spec.key}_Omnibus': omni_df,
            f'{spec.key}_PostHoc': post_df,
            f'{spec.key}_Perm':    pd.DataFrame([perm])}


# ── Speed distribution: plots ───────────────────────────────────────────────

_DIST_XLABEL = 'Instantaneous speed (cm/s)'


def _bin_stars(ax, x, top, p_bins, pad):
    for xb, tb, p in zip(x, top, p_bins):
        if pd.notna(p) and p < asc.ALPHA:
            ax.text(xb, tb + pad, asc._stars(p), ha='center', va='bottom',
                    fontsize=asc.FS_LABEL + 1, fontweight='bold', zorder=5)


def _dist_stats_lines(omni_df: pd.DataFrame, post_df: pd.DataFrame, perm: dict) -> list:
    lines = []
    if pd.notna(perm.get('p_value')):
        lines.append(f'Whole distribution: {perm["test"]}, between-group SS = {perm["statistic"]:.1f}, '
                     f'R² = {perm["R2"]:.3f}, p = {asc._fmt_p(perm["p_value"])} '
                     f'{asc._stars(perm["p_value"])}')
    tested = omni_df[omni_df['p_value'].notna()]
    if len(tested):
        r0 = tested.iloc[0]
        ph = (f'; pairs: {post_df["test"].iloc[0]}, Holm within the bin' if len(post_df) else '')
        lines.append(f'Per bin: {r0["test"]}, p(Holm) across the {len(omni_df)} bins{ph}')
    for _, r in omni_df.iterrows():
        if pd.isna(r['p_value']):
            lines.append(f'{r["speed_bin"]}: not tested (insufficient data / identical values)')
            continue
        dof = f'({int(r["dof"])})' if pd.notna(r.get('dof')) else ''
        line = (f'{r["speed_bin"]}: {r["stat_name"]}{dof} = {r["statistic"]:.2f}, '
                f'p = {asc._fmt_p(r["p_value"])}, p(Holm, bins) = {asc._fmt_p(r["p_holm_bins"])} '
                f'{asc._stars(r["p_holm_bins"])}')
        if r['significant_bins'] and len(post_df):
            pairs = post_df[(post_df['speed_bin'] == r['speed_bin']) & (post_df['p_holm'] < asc.ALPHA)]
            if len(pairs):
                line += '; ' + ', '.join(f'{asc._pair_label(q["comparison"])} p(Holm) = '
                                         f'{asc._fmt_p(q["p_holm"])}' for _, q in pairs.iterrows())
        lines.append(line)
    if len(tested) and isinstance(tested.iloc[0].get('model_note'), str):
        lines.append(tested.iloc[0]['model_note'])
    return lines


def plot_distribution_curve(df, grp, omni_df, post_df, perm, p_bins, out_path, spec: DistSpec):
    """Mean % +- SEM of every group in every speed bin, with the tests underneath."""
    mats = _dist_matrices(df, grp, spec)
    if not mats:
        return
    x = 0.5 * (spec.edges[:-1] + spec.edges[1:])
    fig_w = 8.5
    lines = asc._wrap_stats(_dist_stats_lines(omni_df, post_df, perm), fig_w)
    fig, ax, ax_info = asc._new_fig((fig_w, 5.0), legend_rows=1, stats_lines=lines)

    k = len(mats)
    offs = (np.arange(k) - (k - 1) / 2) * min(0.8, 2.4 / k)   # cm/s, so error bars don't overlap
    top = np.zeros(len(x))
    handles = []
    for (g, P), off in zip(mats.items(), offs):
        c = asc._color(g)
        m = P.mean(axis=0)
        se = P.std(axis=0, ddof=1) / np.sqrt(len(P)) if len(P) > 1 else np.zeros(len(x))
        handles.append(ax.errorbar(x + off, m, yerr=se, fmt='-o', color=c['dark'],
                                   markerfacecolor=c['face'], markeredgecolor=asc.PAL_BLACK,
                                   markeredgewidth=0.6, markersize=6, linewidth=1.5,
                                   elinewidth=1.0, capsize=3, zorder=3,
                                   label=f'{asc._label(g)} (n={len(P)})'))
        top = np.maximum(top, m + se)
    y_hi = max(float(top.max()), 1.0)
    _bin_stars(ax, x, top, p_bins, 0.03 * y_hi)
    ax.set_ylim(0, y_hi * 1.15)
    ax.set_xticks(spec.edges)
    ax.set_xticklabels([f'{e:g}' for e in spec.edges])
    ax.set_xlim(spec.edges[0], spec.edges[-1])
    ax.set_xlabel(_DIST_XLABEL, fontsize=asc.FS_LABEL, labelpad=6)
    ax.set_ylabel(spec.ylabel, fontsize=asc.FS_LABEL, labelpad=8)
    asc._set_title(ax, asc._titled(f'{spec.title} (mean ± SEM)', grp))
    asc._style_axes(ax)
    ax_info.legend(handles=handles, loc='upper center', ncol=min(k, 4), fontsize=asc.FS_LEGEND)
    asc._draw_stats(ax_info, lines)
    fig.tight_layout()
    asc._save(fig, out_path)


def plot_distribution_boxes(df, grp, p_bins, out_path, spec: DistSpec):
    """Every session's % in every speed bin: one box (+ points) per group and bin."""
    mats = _dist_matrices(df, grp, spec)
    if not mats:
        return
    k, n_bins = len(mats), len(spec.bins)
    w = 0.8 / k
    fig_w = max(8.5, n_bins * k * 0.35 + asc.AXIS_PAD_W)
    lines = asc._wrap_stats([f'Stars: {"Kruskal-Wallis" if asc.CONTINUOUS_TEST == "KW" else "LMM"} '
                             f'per bin, Holm-corrected across the {n_bins} bins. '
                             f'Box: median and quartiles; whiskers 1.5 IQR; points: sessions.'], fig_w)
    fig, ax, ax_info = asc._new_fig((fig_w, 5.0), legend_rows=1, stats_lines=lines)

    rng = np.random.default_rng(0)
    top = np.zeros(n_bins)
    for gi, (g, P) in enumerate(mats.items()):
        c = asc._color(g)
        pos = np.arange(n_bins) + (gi - (k - 1) / 2) * w
        bp = ax.boxplot(P, positions=pos, widths=w * 0.8, showfliers=False, patch_artist=True,
                        boxprops=dict(linewidth=asc.BOX_LW * 0.8, color=asc.PAL_BLACK),
                        whiskerprops=dict(linewidth=asc.BOX_LW * 0.8, color=asc.PAL_BLACK),
                        capprops=dict(linewidth=asc.BOX_LW * 0.8, color=asc.PAL_BLACK),
                        medianprops=dict(linewidth=asc.MEDIAN_LW, color=asc.PAL_BLACK), zorder=2)
        for patch in bp['boxes']:
            patch.set_facecolor(c['face'])
        for b in range(n_bins):
            ax.scatter(pos[b] + rng.uniform(-0.25, 0.25, len(P)) * w, P[:, b], s=6,
                       facecolors=c['dark'], edgecolors=asc.PAL_BLACK, linewidths=0.2,
                       alpha=0.7, zorder=3)
        top = np.maximum(top, P.max(axis=0))
    y_hi = max(float(top.max()), 1.0)
    _bin_stars(ax, np.arange(n_bins), top, p_bins, 0.03 * y_hi)
    ax.set_ylim(0, y_hi * 1.12)
    ax.set_xticks(np.arange(n_bins))
    ax.set_xticklabels([f'{lo:g}-{hi:g}' for lo, hi in spec.bins])
    ax.set_xlim(-0.6, n_bins - 0.4)
    ax.set_xlabel(_DIST_XLABEL, fontsize=asc.FS_LABEL, labelpad=6)
    ax.set_ylabel(spec.ylabel, fontsize=asc.FS_LABEL, labelpad=8)
    asc._set_title(ax, asc._titled(f'{spec.title}, per session', grp))
    asc._style_axes(ax)
    handles = [Patch(facecolor=asc._color(g)['face'], edgecolor=asc.PAL_BLACK,
                     label=f'{asc._label(g)} (n={len(P)})') for g, P in mats.items()]
    ax_info.legend(handles=handles, loc='upper center', ncol=min(k, 4), fontsize=asc.FS_LEGEND)
    asc._draw_stats(ax_info, lines)
    fig.tight_layout()
    asc._save(fig, out_path)


# ── One comparison ──────────────────────────────────────────────────────────

def run_comparison(grp: asc.Grouping, df: pd.DataFrame, plots_dir: str) -> dict:
    """Compare every speed metric and the speed distribution across the groups
    of `grp`: statistics, box plots in `plots_dir`, histograms in
    `plots_dir`\\Histograms, distribution plots in `plots_dir`\\SPEED_DIST_SUBDIR.
    Returns {table_name: table}. No SIR exists per session, so points are not shaded."""
    name = grp.title or grp.noun.capitalize()
    print(f'\n{"=" * 70}\n{name} -- comparing {grp.noun}: '
          f'{[asc._label(g) for g in grp.order]}\n{"=" * 70}')
    print('Sessions per group:')
    print(df[grp.col].value_counts().reindex(grp.order).fillna(0).astype(int))
    if not len(df):
        print('No sessions; skipping.')
        return {}

    desc_df, omnibus_df, posthoc_df = asc.run_all_comparisons(df, SPEED_METRICS, 'Speed', grp, plots_dir)
    print(f'Box plots saved to {plots_dir}')
    asc._print_significant(omnibus_df, f'{name}, running speed', 45, 16)

    hist_dir = os.path.join(plots_dir, 'Histograms')
    os.makedirs(hist_dir, exist_ok=True)
    for col, label in SPEED_METRICS.items():
        asc.plot_histogram_by_group(df, col, label, os.path.join(hist_dir, f'Hist_Speed_{col}.png'), grp)
    print(f'Histogram plots saved to {hist_dir}')

    dist_tables = {}
    for spec in DIST_SPECS:
        dist_tables.update(run_speed_distribution(df, grp, os.path.join(plots_dir, SPEED_DIST_SUBDIR), spec))
    return {'Desc': desc_df, 'Omnibus': omnibus_df, 'PostHoc': posthoc_df, **dist_tables}


def run_animal_controlled(df: pd.DataFrame, out_dir: str) -> dict:
    """Level 3: each entry of ANIMAL_CONTROLLED_COMPARISONS compares its arenas
    using only the sessions of its animals; the plots are saved side by side,
    each with its own statistics. Returns {sheet_name: table}."""
    all_tables = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Same y-range for the two versions of every plot (plu.shared_y_axes).
        with plu.shared_y_axes([os.path.join(tmp_dir, n) for n in asc.ANIMAL_CONTROLLED_COMPARISONS]):
            for name, (animals, arenas) in asc.ANIMAL_CONTROLLED_COMPARISONS.items():
                grp = asc.animal_controlled_grouping(animals, arenas)
                wanted = {a.lower() for a in animals}
                sel = df[df['animal'].astype(str).str.lower().isin(wanted) & df['arena_type'].isin(grp.order)]
                print(f'\n{name}: sessions per animal and arena:')
                print(pd.crosstab(sel['animal'], sel['arena_type'], margins=True))
                tables = run_comparison(grp, sel, os.path.join(tmp_dir, name))
                tag = 'AC' + ''.join(grp.order)   # e.g. 'ACCircleOpen'
                for tname, tbl in tables.items():
                    if len(tbl):
                        tbl.insert(0, 'animals', ', '.join(animals))
                    all_tables[f'{tag}_{tname}'] = tbl
        asc._combine_side_by_side([os.path.join(tmp_dir, name) for name in asc.ANIMAL_CONTROLLED_COMPARISONS],
                                  out_dir)
    if all_tables:
        xlsx = os.path.join(out_dir, 'AnimalControlled_Stats.xlsx')
        with pd.ExcelWriter(xlsx, engine='openpyxl') as writer:
            for sheet_name, tbl in all_tables.items():
                tbl.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        print(f'\nAnimal-controlled statistics saved to {xlsx}')
    return all_tables


# ── Run snapshot (code + metadata, saved before the analysis) ───────────────

def save_run_snapshot() -> str:
    """Copy this script and the local modules it uses to
    OUTPUT_DIR\\RUN_SNAPSHOT_SUBDIR\\<timestamp>\\code, and write RunMetadata.xlsx
    there: 'Run' (time, git commit, versions) and 'Parameters' (every setting)."""
    now = datetime.now()
    snap_dir = os.path.join(OUTPUT_DIR, RUN_SNAPSHOT_SUBDIR, now.strftime('%Y%m%d_%H%M%S'))
    code_dir = os.path.join(snap_dir, 'code')
    os.makedirs(code_dir, exist_ok=True)

    script = os.path.abspath(__file__)
    modules = [asc, asc.ccs, asc.pfs, asc.pc]
    for path in [script] + [m.__file__ for m in modules]:
        shutil.copy2(path, code_dir)

    run_rows = [
        ('run_timestamp', now.isoformat(timespec='seconds')),
        ('script',        script),
        ('git_commit',    asc._git('rev-parse', 'HEAD') or 'n/a'),
        ('git_uncommitted_changes (script + local modules)',
         asc._git('status', '--porcelain', '--', script, *[m.__file__ for m in modules]) or 'none'),
        ('python',        sys.version.replace('\n', ' ')),
        ('platform',      platform.platform()),
        ('machine',       platform.node()),
    ]
    for pkg in ['numpy', 'pandas', 'scipy', 'matplotlib', 'statsmodels', 'openpyxl']:
        try:
            run_rows.append((f'{pkg}_version', importlib.metadata.version(pkg)))
        except importlib.metadata.PackageNotFoundError:
            run_rows.append((f'{pkg}_version', 'not installed'))

    params = [row for mod in [sys.modules[__name__]] + modules for row in asc._module_parameters(mod)]
    with pd.ExcelWriter(os.path.join(snap_dir, 'RunMetadata.xlsx'), engine='openpyxl') as writer:
        pd.DataFrame(run_rows, columns=['item', 'value']).to_excel(writer, sheet_name='Run', index=False)
        pd.DataFrame(params, columns=['module', 'parameter', 'value']).to_excel(
            writer, sheet_name='Parameters', index=False)
    print(f'Run snapshot (code + metadata) saved to {snap_dir}')
    return snap_dir


def main():
    save_run_snapshot()

    df = build_session_df()
    print('\nSessions by arena and session type:')
    print(pd.crosstab(df['arena_type'], df['session_type'].fillna('unrecognized'), margins=True))
    print('\nSessions by animal and arena:')
    print(pd.crosstab(df['animal'], df['arena_type'], margins=True))

    all_tables = {}

    # ── 1. Session types within each arena ──────────────────────────────────
    for arena in asc.ARENA_TYPES:
        grp = asc.session_grouping(arena)
        tables = run_comparison(grp, df[(df['arena_type'] == arena) & df['session_type'].isin(grp.order)],
                                os.path.join(OUTPUT_DIR, 'SessionType', arena))
        for name, tbl in tables.items():
            if len(tbl) and 'arena_type' not in tbl.columns:
                tbl.insert(0, 'arena_type', arena)
            all_tables[f'Ses{arena}_{name}'] = tbl

    # ── 2. Arena types as a whole (all session types pooled) ────────────────
    tables = run_comparison(asc.ARENA_GROUPING, df, os.path.join(OUTPUT_DIR, 'ArenaType'))
    for name, tbl in tables.items():
        all_tables[f'Arena_{name}'] = tbl

    # ── 3. Arena types, only animals recorded in all compared arenas ────────
    all_tables.update(run_animal_controlled(df, os.path.join(OUTPUT_DIR, asc.ANIMAL_CONTROLLED_DIR)))

    # ── Save everything (new workbook + one long-format CSV) ────────────────
    with pd.ExcelWriter(STATS_XLSX, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='PerSession', index=False)
        for sheet_name, tbl in all_tables.items():
            tbl.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    print(f'\nPer-session speeds and statistics saved to {STATS_XLSX}')

    parts = [tbl.assign(table=sheet_name) for sheet_name, tbl in all_tables.items() if len(tbl)]
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(STATS_CSV, index=False, encoding='utf-8-sig')
        print(f'All statistics also saved to {STATS_CSV}')

    print(f'\nDone. Plots saved under {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
