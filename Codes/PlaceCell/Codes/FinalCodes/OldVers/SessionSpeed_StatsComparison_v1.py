# -*- coding: utf-8 -*-
"""
Mean running speed of every recording session, compared across the same
conditions as ArenaSessionType_StatsComparison_v18.py, with the same tests and
the same figure formatting (its functions are imported, not copied, so a
formatting or statistics change in v18 applies here too):

  1. Session types WITHIN each arena type
         Open   : Cntrl, Rotate, Zero
         Linear : 0, 90, 180, 270
         Circle : Stnd, NoRot, Rot
  2. Arena types as a whole (Circle / Linear / Open, all session types pooled)
  3. Animal-controlled arena types (ANIMAL_CONTROLLED_COMPARISONS in v18)
         Circle vs Open : Fa8477, Fa5834
         Linear vs Open : Fa1059, Fa23BD
     saved side by side, each panel with its own statistics.

One row = one recording session (no units, no spike data). Tests come from
v18's CONTINUOUS_TEST ('KW': Kruskal-Wallis + pairwise Mann-Whitney U,
Holm-Bonferroni corrected, sessions treated as independent).

Sessions: every folder <root>\\<animal>\\<arena>\\<day>\\<session> holding one
*_cm.csv file, from TRACKING_ROOTS. A session found in several roots is taken
from the first one. QualFiltData_AllCells comes first because its tracking is
what the place-cell analyses use (off-arena frames corrected, some sessions
trimmed); Corr_Data adds the sessions with no spike-sorted units, which
QualFiltData_AllCells does not have.

Speed, per session, using PlaceCell_Main_v21's own tracking functions:
    1. _load_tracking: cleaning, as in the place-cell analyses.
    2. _smooth_tracking_position: artifact-jump removal, Gaussian smoothing.
    3. _instantaneous_speed: per-frame speed (cm/s).
    4. Time-weighted mean over the whole session (= distance / duration).
Metrics (SPEED_METRICS):
    mean_speed_cms         every frame, including immobile ones
    mean_speed_moving_cms  frames with speed >= pc.IMMOBILITY_SPEED_CMS only
The per-session table also has pct_time_moving and the as-tracked speeds.

Tracking scale: each *_cm.csv was scaled by stretching that session's pixel
range to the arena size, so a session whose pixel range was not the whole arena
gets the wrong cm/pixel, and speed is directly proportional to it (animal stayed
on part of the track: too large, speed inflated). The cm/pixel of every session
is measured from its pixel and cm columns; sessions more than SCALE_OUTLIER_FRAC
from the median of their animal + arena are flagged, and with
CORRECT_SCALE_OUTLIERS their positions are rescaled to that median before
smoothing. The *_cm.csv files are not modified.

Output (OUTPUT_DIR):
    SessionType\\<arena>\\                  box plots, session types of <arena>
    SessionType\\<arena>\\Histograms\\
    ArenaType\\                             box plots, arena types
    ArenaType\\Histograms\\
    AnimalControlledArenaType\\             box plots, the comparisons side by side
    AnimalControlledArenaType\\Histograms\\
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

# Also applies v18's figure style (rcParams).
import ArenaSessionType_StatsComparison_v18 as asc

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
    }


def build_session_df() -> pd.DataFrame:
    """One row per session: its groups, tracking scale and speed."""
    sessions = find_sessions()
    print(f'Found {len(sessions)} session(s) with tracking.')
    rows = []
    for i, (rel, csv_path, root) in enumerate(sessions, start=1):
        print(f'  [{i}/{len(sessions)}] {rel}')
        rows.append({'session': rel, 'unit': rel, 'tracking_file': csv_path, 'tracking_root': root,
                     'cm_per_px': measure_cm_per_px(csv_path), **session_speed(csv_path)})
    # 'unit' only makes each row its own observation in v18's functions.
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
    return df


# ── Statistics ──────────────────────────────────────────────────────────────

_CELL_SUMMARY = re.compile(r'\d+ observations from \d+ cells \([^)]*\)(, \d+ recording days, \d+ animals)?')
_compare_groups_v18 = asc.compare_groups


def compare_sessions(df: pd.DataFrame, metric_col: str, metric_label: str, grp: asc.Grouping):
    """v18's compare_groups, with its tables and the note under each figure
    counting sessions, recording days and animals instead of cells."""
    desc_df, omnibus, posthoc_df = _compare_groups_v18(df, metric_col, metric_label, grp)
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


def run_comparison(grp: asc.Grouping, df: pd.DataFrame, plots_dir: str) -> dict:
    """Compare every speed metric across the groups of `grp`: statistics, box
    plots in `plots_dir`, histograms in `plots_dir`\\Histograms. Returns
    {table_name: table}. No SIR exists per session, so points are not shaded."""
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
    return {'Desc': desc_df, 'Omnibus': omnibus_df, 'PostHoc': posthoc_df}


def run_animal_controlled(df: pd.DataFrame, out_dir: str) -> dict:
    """Level 3: each entry of ANIMAL_CONTROLLED_COMPARISONS compares its arenas
    using only the sessions of its animals; the plots are saved side by side,
    each with its own statistics. Returns {sheet_name: table}."""
    all_tables = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
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
