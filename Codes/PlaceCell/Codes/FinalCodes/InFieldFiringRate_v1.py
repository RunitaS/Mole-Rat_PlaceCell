# -*- coding: utf-8 -*-
"""
Within-field mean firing rate of every place cell, compared across the
conditions of ArenaSessionType_StatsComparison_v20.py -- standalone, so it can
run without re-running PlaceCell_Main_v22 or the whole v20 analysis.

Within-field mean firing rate (one value per place cell):
    spikes while the animal was inside any of the cell's place fields
    / time the animal spent inside them,
pooled over all of the cell's fields, moving frames only (speed >=
pc.IMMOBILITY_SPEED_CMS, and the spikes assigned to those frames: the data behind
mean_fr, the ratemaps and field_n_spikes). Equal to the occupancy-weighted mean
of the raw rate over all field bins. Same computation as PlaceCell_Main_v22
(pc._infield_firing_rate) and v20 (asc.run_infield_rate, used here):
  1. Place cells (place_cell True) and their fields (bin_coords) from INPUT_EXCEL
     ('Full', 'PlaceFields'), the workbook of the v20 statistics.
  2. Each cell's moving-only spike-count and occupancy maps are rebuilt with
     PlaceCell_Main_v22.compute_metrics from its .ntt + tracking file in
     PLACE_CELL_FOLDER (2-D grid; Circle track: the angular maps of
     pc._build_ratemap_circular, the maps its fields were detected on).
  3. Per field: field_n_spikes, field_dwell_s, field_mean_fr_hz; per cell:
     infield_n_spikes / infield_dwell_s = infield_mean_fr.
  Checks: ratemap_check is False if a field's rebuilt PLACE_FIELD_RATEMAP_TYPE
  peak differs from its INPUT_EXCEL peak_fr_hz by > asc.IFR_RATEMAP_CHECK_TOL_HZ
  (its bins may not line up with the rebuilt grid; INPUT_EXCEL predates the
  2 Nov tracking edits); spikes_match compares the rebuilt in-field spikes with
  INPUT_EXCEL field_n_spikes. EXCLUDE_RATEMAP_MISMATCH = True leaves cells that
  fail ratemap_check out of the statistics.

Statistics and plots (functions of ArenaSessionType_StatsComparison_v20,
imported as asc; same tests, colours, point shading by SIR, Holm correction),
one folder per level in OUTPUT_DIR:
  SessionType\\<arena>\\       level 1, session types within each arena
  ArenaType\\                  level 2, Circle / Linear / Open
  AnimalControlledArenaType\\  level 3, asc.ANIMAL_CONTROLLED_COMPARISONS side by side
                               (same y-range in both panels)
  AnimalID\\                   level 4, animals
  AllCells\\                   level 5, every place cell pooled, descriptives only
Each folder:
  Full_infield_mean_fr.png                 box plot + points, asc.CONTINUOUS_TEST
                                           (Kruskal-Wallis + pairwise Mann-Whitney U,
                                           Holm-Bonferroni; rows = unit x session)
  Histograms\\Hist_Full_infield_mean_fr.png one histogram per group
The wall-distance analysis of v20 is not repeated here.

Outputs:
  OUTPUT_DIR\\InFieldFiringRate.xlsx    PerCell, PerField (written by asc.run_infield_rate)
  OUTPUT_DIR\\InFieldRate_Stats.xlsx    sheets 'Ses<arena>_*', 'Arena_*', 'AC<A><B>_*',
                                       'Animal_*', 'All_*' (Desc / Omnibus / PostHoc)
  OUTPUT_DIR\\InFieldRate_AllStats.csv
  OUTPUT_DIR\\RunSnapshots\\<timestamp>\\ code + RunMetadata.xlsx
"""

import os
import sys
import shutil
import platform
import tempfile
import importlib.metadata
from dataclasses import replace
from datetime import datetime

import pandas as pd

import matplotlib
matplotlib.use('Agg')

# CellClusteredStats_Utils.py lives one folder up (appended, so this folder's
# PlaceCell_Main_v22.py and ArenaSessionType_StatsComparison_v20.py come first).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import PlaceCell_Main_v22 as pc
import ArenaSessionType_StatsComparison_v20 as asc   # rate rebuild, statistics, plot style
import CellClusteredStats_Utils as ccs

# ── Parameters ──────────────────────────────────────────────────────────────

# Place cells ('Full') and their fields ('PlaceFields', bin_coords): v20's INPUT_EXCEL.
INPUT_EXCEL = asc.INPUT_EXCEL

# .ntt + tracking files of those place cells (one folder per session).
PLACE_CELL_FOLDER = asc.TS_PLACE_CELL_FOLDER

OUTPUT_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final\Results\PC_Char\InFieldFiringRate_v1'

# True: rebuild the rate of every cell (RESULTS_XLSX), then the statistics.
# False: run the statistics on the PerCell sheet of an existing RESULTS_XLSX.
RUN_INFIELD_RATE = True

# True: cells with a field that failed ratemap_check are left out of the statistics.
EXCLUDE_RATEMAP_MISMATCH = False

RESULTS_XLSX = os.path.join(OUTPUT_DIR, 'InFieldFiringRate.xlsx')
STATS_XLSX   = os.path.join(OUTPUT_DIR, 'InFieldRate_Stats.xlsx')
STATS_CSV    = os.path.join(OUTPUT_DIR, 'InFieldRate_AllStats.csv')
RUN_SNAPSHOT_SUBDIR = 'RunSnapshots'

METRICS = {asc.INFIELD_RATE_COL: asc.INFIELD_RATE_LABEL}
PLOT_PREFIX = 'Full'

# Rate-rebuild settings of v20 / PlaceCell_Main_v22 (listed so the run snapshot records them).
RATEMAP_CHECK_TOL_HZ = asc.IFR_RATEMAP_CHECK_TOL_HZ
MAX_WORKERS          = asc.IFR_MAX_WORKERS
IMMOBILITY_SPEED_CMS = pc.IMMOBILITY_SPEED_CMS
TARGET_BIN_CM        = pc.target_bin_cm
CONTINUOUS_TEST      = asc.CONTINUOUS_TEST


# ── One comparison ──────────────────────────────────────────────────────────

def run_rate_comparison(grp, df: pd.DataFrame, plots_dir: str) -> dict:
    """Box plot + histogram + tests of the within-field rate across the groups of
    `grp`; df already restricted to the cells to compare. Returns {table: df}."""
    name = grp.title or grp.noun.capitalize()
    print(f'\n{"=" * 70}\n{name} -- {[asc._label(g) for g in grp.order]}\n{"=" * 70}')
    has = df[df[asc.INFIELD_RATE_COL].notna()]
    print('Place-cell rows with a within-field rate per group:')
    print(has[grp.col].value_counts().reindex(grp.order).fillna(0).astype(int).to_string())
    if not len(has):
        print('No values; skipping.')
        return {}
    os.makedirs(plots_dir, exist_ok=True)
    # Point shade: the cell's SIR rank among this comparison's place cells (as v20).
    grp = replace(grp, shade=asc.cell_shades(df))

    desc, omni, post = asc.run_all_comparisons(df, METRICS, PLOT_PREFIX, grp, plots_dir)
    hist_dir = os.path.join(plots_dir, 'Histograms')
    os.makedirs(hist_dir, exist_ok=True)
    for col, label in METRICS.items():
        asc.plot_histogram_by_group(df, col, label,
                                    os.path.join(hist_dir, f'Hist_{PLOT_PREFIX}_{col}.png'), grp)
    if grp.tests:
        asc._print_significant(omni, name, 45, 16)
    print(f'Plots saved to {plots_dir}')
    return {'Desc': desc, 'Omnibus': omni, 'PostHoc': post}


def run_animal_controlled(df: pd.DataFrame, out_dir: str) -> dict:
    """Level 3: each asc.ANIMAL_CONTROLLED_COMPARISONS entry, plots side by side."""
    all_tables = {}
    with tempfile.TemporaryDirectory() as tmp:
        # Same y-range for the two versions of every plot (asc.plu.shared_y_axes).
        with asc.plu.shared_y_axes([os.path.join(tmp, n) for n in asc.ANIMAL_CONTROLLED_COMPARISONS]):
            for name, (animals, arenas) in asc.ANIMAL_CONTROLLED_COMPARISONS.items():
                grp = asc.animal_controlled_grouping(animals, arenas)
                wanted = {a.lower() for a in animals}
                sel = df[df['animal'].astype(str).str.lower().isin(wanted)
                         & df['arena_type'].isin(grp.order)]
                print(f'\n{name}: place-cell rows per animal and arena:')
                print(pd.crosstab(sel['animal'], sel['arena_type'], margins=True).to_string())
                tables = run_rate_comparison(grp, sel, os.path.join(tmp, name))
                tag = 'AC' + ''.join(grp.order)   # e.g. 'ACCircleOpen'
                for tname, tbl in tables.items():
                    if len(tbl):
                        tbl.insert(0, 'animals', ', '.join(animals))
                    all_tables[f'{tag}_{tname}'] = tbl
        asc._combine_side_by_side([os.path.join(tmp, n) for n in asc.ANIMAL_CONTROLLED_COMPARISONS],
                                  out_dir)
    return all_tables


# ── Run snapshot ────────────────────────────────────────────────────────────

def _module_settings() -> dict:
    return {k: v for k, v in globals().items()
            if k.isupper() and not k.startswith('_')
            and isinstance(v, (str, int, float, bool, list, tuple, dict, type(None)))}


def save_run_snapshot() -> str:
    """This script + the local modules it uses, and RunMetadata.xlsx (run info +
    every setting of this script, v20 and PlaceCell_Main_v22), in
    OUTPUT_DIR\\RUN_SNAPSHOT_SUBDIR\\<timestamp>."""
    now = datetime.now()
    snap = os.path.join(OUTPUT_DIR, RUN_SNAPSHOT_SUBDIR, now.strftime('%Y%m%d_%H%M%S'))
    code = os.path.join(snap, 'code')
    os.makedirs(code, exist_ok=True)
    script = os.path.abspath(__file__)
    mods = [pc, asc, ccs, asc.plu]
    for p in [script] + [m.__file__ for m in mods]:
        shutil.copy2(p, code)
    run = [('run_timestamp', now.isoformat(timespec='seconds')), ('script', script),
           ('git_commit', asc._git('rev-parse', 'HEAD') or 'n/a'),
           ('git_uncommitted_changes', asc._git('status', '--porcelain', '--', script,
                                                *[m.__file__ for m in mods]) or 'none'),
           ('python', sys.version.replace('\n', ' ')), ('platform', platform.platform()),
           ('machine', platform.node())]
    for pkg in ('numpy', 'pandas', 'scipy', 'matplotlib', 'statsmodels'):
        try:
            run.append((f'{pkg}_version', importlib.metadata.version(pkg)))
        except importlib.metadata.PackageNotFoundError:
            pass
    params = ([{'module': __name__, 'parameter': k, 'value': str(v)}
               for k, v in _module_settings().items()]
              + asc._module_parameters(asc) + asc._module_parameters(pc))
    with pd.ExcelWriter(os.path.join(snap, 'RunMetadata.xlsx'), engine='openpyxl') as w:
        pd.DataFrame(run, columns=['item', 'value']).to_excel(w, sheet_name='Run', index=False)
        pd.DataFrame(params).to_excel(w, sheet_name='Parameters', index=False)
    print(f'Run snapshot saved to {snap}')
    return snap


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_run_snapshot()
    asc.IFR_EXCLUDE_RATEMAP_MISMATCH = EXCLUDE_RATEMAP_MISMATCH   # read by asc.merge_infield_rate

    print(f'Loading {INPUT_EXCEL} ...')
    full   = pd.read_excel(INPUT_EXCEL, sheet_name='Full')
    fields = pd.read_excel(INPUT_EXCEL, sheet_name='PlaceFields')

    if RUN_INFIELD_RATE:
        per_cell, _ = asc.run_infield_rate(full, fields, RESULTS_XLSX,
                                           place_cell_folder=PLACE_CELL_FOLDER)
    else:
        print(f'Loading PerCell of {RESULTS_XLSX} ...')
        per_cell = pd.read_excel(RESULTS_XLSX, sheet_name='PerCell')

    # Place-cell rows of 'Full' (arena / session type / animal / cell_id / SIR for
    # the point shade), with the within-field rate.
    df = asc.merge_infield_rate(asc.build_full_metrics_df(full), per_cell)
    print('\nPlace-cell rows with a within-field rate, by arena and session type:')
    has = df[df[asc.INFIELD_RATE_COL].notna()]
    print(pd.crosstab(has['arena_type'], has['session_type'].fillna('unrecognized'), margins=True))

    all_tables = {}
    # 1. Session types within each arena
    for arena in asc.ARENA_TYPES:
        grp = asc.session_grouping(arena)
        sub = df[(df['arena_type'] == arena) & df['session_type'].isin(grp.order)]
        for k, t in run_rate_comparison(grp, sub, os.path.join(OUTPUT_DIR, 'SessionType', arena)).items():
            if len(t) and 'arena_type' not in t.columns:
                t.insert(0, 'arena_type', arena)
            all_tables[f'Ses{arena}_{k}'] = t
    # 2. Arena types
    for k, t in run_rate_comparison(asc.ARENA_GROUPING, df,
                                    os.path.join(OUTPUT_DIR, 'ArenaType')).items():
        all_tables[f'Arena_{k}'] = t
    # 3. Animal-controlled arena types
    all_tables.update(run_animal_controlled(df, os.path.join(OUTPUT_DIR, asc.ANIMAL_CONTROLLED_DIR)))
    # 4. Animal IDs
    animal_grp = asc.animal_id_grouping(df)
    if asc.CONTINUOUS_TEST == 'LMM':
        print('NOTE: with the animal as the compared group, the LMM\'s animal random intercept '
              'is confounded with the group effect.')
    for k, t in run_rate_comparison(animal_grp, df, os.path.join(OUTPUT_DIR, asc.ANIMAL_ID_DIR)).items():
        all_tables[f'Animal_{k}'] = t
    # 5. All place cells pooled (no tests)
    for k, t in run_rate_comparison(asc.ALL_CELLS_GROUPING, df,
                                    os.path.join(OUTPUT_DIR, asc.ALL_CELLS_DIR)).items():
        if 'Omnibus' not in k and 'PostHoc' not in k:   # one group: nothing tested
            all_tables[f'All_{k}'] = t

    with pd.ExcelWriter(STATS_XLSX, engine='openpyxl') as w:
        for sheet, tbl in all_tables.items():
            tbl.to_excel(w, sheet_name=sheet[:31], index=False)
    parts = []
    for sheet, tbl in all_tables.items():
        if tbl is not None and len(tbl):
            t = tbl.copy()
            t.insert(0, 'table', sheet)
            parts.append(t)
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(STATS_CSV, index=False, encoding='utf-8-sig')
    print(f'\nStatistics saved to {STATS_XLSX} and {STATS_CSV}')
    print(f'Continuous-metric test: '
          f'{"Kruskal-Wallis + Mann-Whitney U (Holm), independence assumed" if asc.CONTINUOUS_TEST == "KW" else "nested LMM"}')
    print(f'Done. Plots under {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
