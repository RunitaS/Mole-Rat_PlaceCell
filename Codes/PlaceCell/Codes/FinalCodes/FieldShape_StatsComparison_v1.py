# -*- coding: utf-8 -*-
"""
Place-field SHAPE comparison only -- the field-shape part of
ArenaSessionType_StatsComparison_v21.py, without the rest of that script
(firing / field metrics, proportions, temporal stability, multi-peak, ...),
so that the shape results can be redone on their own.

Everything is taken from ArenaSessionType_StatsComparison_v21 (imported as
`asc`): its INPUT_EXCEL, groupings, tests (CONTINUOUS_TEST, Holm), plots and
settings. Only OUTPUT_DIR and the switches below are set here.

Steps
  1. Field shape of every field of INPUT_EXCEL 'PlaceFields' (asc.load_sheets):
     Linear / Open fields from their 2-D bins; Circle fields on the linearized
     2-D map of the ring (asc.add_all_field_shapes, CIRCLE_LIN_* settings of
     v21: columns = the angular bins the field was detected on, rows = 2 radial
     rows of 2 cm around the median radius; the ring is cut at theta = 0, and
     the cut is moved one bin at a time until it does not pass through a field;
     a field's 2-D bins are those in its angular columns that pass the field
     threshold on the linearized Gaussian-smoothed map). Saved, with the
     per-field linearization columns (fld_lin_*), to asc.FIELD_SHAPE_EXCEL
     (copy of INPUT_EXCEL; INPUT_EXCEL is not modified).
  2. RUN_FIELD_DIMENSION_MAPS: field-dimension figures (asc.run_field_dimension_maps)
     of the cells of FD_ARENAS (default Circle only: the new linearized figures),
     saved in <session>\\field_dimensions\\ of asc.FD_PLACE_CELL_FOLDER.
  3. The five FIELD_SHAPE_METRICS (max / min / mean width, width ratio, axis
     angle; one value per field of a place cell), compared at every level of
     v21, with the same tests and plots:
       SessionType\\<arena>\\          session types within each arena
       ArenaType\\                     Circle / Linear / Open
       AnimalControlledArenaType\\     Circle vs Open (Fa8477, Fa5834) |
                                      Linear vs Open (Fa1059, Fa23BD), side by side
       AnimalID\\                      animals
       AllCells\\                      all fields pooled, no tests
     In each folder: FieldShape_<metric>.png (fld_width_ratio also _PosControl /
     _NegControl: red rings = fields of cells whose raw-map peak bin is at wall
     distance 0, green = the others), Histograms\\Hist_FieldShape_<metric>.png and,
     with INCLUDE_WALL_DISTANCE, WallDistance\\WallDist_FieldShape_<metric>.png
     (shape metrics vs wall distance of the cell's peak bin; none when every
     group has a single distance, e.g. SessionType\\Circle). Existing plots of
     the same name are overwritten; nothing else in those folders is touched.
  4. Tables: OUTPUT_DIR\\FieldShape_Stats.xlsx (sheets 'Ses<arena>_*', 'Arena_*',
     'AC<A><B>_*', 'Animal_*', 'All_*': Desc / Omnibus / PostHoc and the WD_*
     wall-distance tables) + FieldShape_AllStats.csv (long format), and
     AnimalControlledArenaType\\FieldShape_AnimalControlled_Stats.xlsx.
     SessionArena_Stats.xlsx of the full run is NOT modified.
  Run snapshot (code + every parameter): OUTPUT_DIR\\RunSnapshots\\<time>_FieldShape\\.

Wall distance of each cell's raw-map peak bin: read from the 'WD_Cells' sheet of
OUTPUT_DIR\\SessionArena_Stats.xlsx (written by a full v21 run; WD_CELLS_SOURCE =
'auto' / 'excel'), or rescanned from asc.WD_PLACE_CELL_DIR ('rescan', slow).

Run:  python FieldShape_StatsComparison_v1.py
"""

import os
import sys
import shutil
import tempfile
import platform
import importlib.metadata
from dataclasses import replace
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ArenaSessionType_StatsComparison_v21 as asc   # noqa: E402

# ── Parameters ──────────────────────────────────────────────────────────────

OUTPUT_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final\Results\PC_Char\v4'

# Field-dimension figures (one per place cell, next to its .ntt) for these arenas
# (None = every arena; they are slow, the 2-D arenas are unchanged since v19).
RUN_FIELD_DIMENSION_MAPS = True
FD_ARENAS = ['Circle']

# Shape metrics vs the wall distance of the cell's peak bin, and the red / green
# rings on FieldShape_fld_width_ratio_PosControl / _NegControl.png.
INCLUDE_WALL_DISTANCE = True
#   'auto'  : 'WD_Cells' of WD_CELLS_EXCEL if it exists, else rescan
#   'excel' : 'WD_Cells' of WD_CELLS_EXCEL (error if missing)
#   'rescan': scan asc.WD_PLACE_CELL_DIR again (every session's tracking + every unit)
WD_CELLS_SOURCE = 'auto'
WD_CELLS_EXCEL  = os.path.join(OUTPUT_DIR, 'SessionArena_Stats.xlsx')

SHAPE_STATS_XLSX = os.path.join(OUTPUT_DIR, 'FieldShape_Stats.xlsx')
SHAPE_STATS_CSV  = os.path.join(OUTPUT_DIR, 'FieldShape_AllStats.csv')
AC_STATS_XLSX    = 'FieldShape_AnimalControlled_Stats.xlsx'   # in ANIMAL_CONTROLLED_DIR

# v21 functions read these module globals.
asc.OUTPUT_DIR = OUTPUT_DIR
asc.FD_ARENAS  = FD_ARENAS


# ── Run snapshot ─────────────────────────────────────────────────────────────

def save_run_snapshot() -> str:
    """Copy this script, v21 and their local modules to
    OUTPUT_DIR\\RunSnapshots\\<time>_FieldShape\\code and write RunMetadata.xlsx
    ('Run' and 'Parameters', as v21's snapshot)."""
    now = datetime.now()
    snap_dir = os.path.join(OUTPUT_DIR, asc.RUN_SNAPSHOT_SUBDIR,
                            now.strftime('%Y%m%d_%H%M%S') + '_FieldShape')
    code_dir = os.path.join(snap_dir, 'code')
    os.makedirs(code_dir, exist_ok=True)
    mods = [asc, asc.ccs, asc.pfs, asc.pc, asc.plu]
    paths = [os.path.abspath(__file__)] + [m.__file__ for m in mods]
    for p in paths:
        shutil.copy2(p, code_dir)
    run_rows = [('run_timestamp', now.isoformat(timespec='seconds')),
                ('script', os.path.abspath(__file__)),
                ('git_commit', asc._git('rev-parse', 'HEAD') or 'n/a'),
                ('git_uncommitted_changes', asc._git('status', '--porcelain', '--', *paths) or 'none'),
                ('python', sys.version.replace('\n', ' ')),
                ('platform', platform.platform()), ('machine', platform.node())]
    for pkg in asc._SNAPSHOT_PACKAGES:
        try:
            run_rows.append((f'{pkg}_version', importlib.metadata.version(pkg)))
        except importlib.metadata.PackageNotFoundError:
            run_rows.append((f'{pkg}_version', 'not installed'))
    params = [r for m in [sys.modules[__name__]] + mods for r in asc._module_parameters(m)]
    with pd.ExcelWriter(os.path.join(snap_dir, 'RunMetadata.xlsx'), engine='openpyxl') as w:
        pd.DataFrame(run_rows, columns=['item', 'value']).to_excel(w, sheet_name='Run', index=False)
        pd.DataFrame(params, columns=['module', 'parameter', 'value']).to_excel(
            w, sheet_name='Parameters', index=False)
    print(f'Run snapshot (code + metadata) saved to {snap_dir}')
    return snap_dir


# ── Wall distance ────────────────────────────────────────────────────────────

def load_wd_cells() -> pd.DataFrame:
    """Raw-map peak bin + wall distance of every unit (v21 'WD_Cells' table)."""
    use_excel = WD_CELLS_SOURCE == 'excel' or (WD_CELLS_SOURCE == 'auto'
                                               and os.path.isfile(WD_CELLS_EXCEL))
    if use_excel:
        cells = pd.read_excel(WD_CELLS_EXCEL, sheet_name='WD_Cells')
        cells['_skey'] = cells['session'].map(asc._session_key)
        cells['_ukey'] = cells['unit'].map(asc._unit_key)
        print(f'\nWall distance of {len(cells)} unit(s) read from {WD_CELLS_EXCEL} (WD_Cells)')
        return cells
    if WD_CELLS_SOURCE not in ('auto', 'rescan'):
        raise ValueError(f"WD_CELLS_SOURCE must be 'auto', 'excel' or 'rescan', got {WD_CELLS_SOURCE!r}")
    geoms = asc.make_geometries()
    units = {a: asc.wall_dist_units(g, a) for a, g in geoms.items()}
    print(f'\nScanning {asc.WD_PLACE_CELL_DIR} for the wall distance ...')
    _, cells = asc.scan_place_cell_dir(geoms, units)
    return cells


def build_wd_dfs(full_df: pd.DataFrame, shape_df: pd.DataFrame) -> dict:
    cells = load_wd_cells()
    return {'full':  asc.attach_distance(full_df, cells, "Wall distance, table 'full'"),
            'shape': asc.attach_distance(shape_df, cells, "Wall distance, table 'shape'")}


# ── One comparison ───────────────────────────────────────────────────────────

def run_shape_comparison(grp, full_df: pd.DataFrame, shape_df: pd.DataFrame, plots_dir: str,
                         wd_dfs: dict = None) -> dict:
    """The field-shape part of asc.run_comparison: box plots in `plots_dir`,
    histograms in `plots_dir`\\Histograms, wall-distance plots in
    `plots_dir`\\WallDistance. Returns {table_name: table}."""
    name = grp.title or grp.noun.capitalize()
    what = (f'comparing {grp.noun}: {[asc._label(g) for g in grp.order]}' if grp.tests
            else f'{", ".join(grp.order)} pooled, no tests')
    print(f'\n{"=" * 70}\n{name} -- field shape, {what}\n{"=" * 70}')
    print('Place fields per group:')
    print(shape_df[grp.col].value_counts().reindex(grp.order).fillna(0).astype(int))
    if not len(full_df) or not len(shape_df):
        print('No place fields; skipping.')
        return {}

    # Each cell's shade, from this comparison's place cells (as in run_comparison).
    grp = replace(grp, shade=asc.cell_shades(full_df))
    desc, omni, post = asc.run_all_comparisons(shape_df, asc.FIELD_SHAPE_METRICS, 'FieldShape',
                                               grp, plots_dir)
    hist_dir = os.path.join(plots_dir, 'Histograms')
    os.makedirs(hist_dir, exist_ok=True)
    for col, label in asc.FIELD_SHAPE_METRICS.items():
        asc.plot_histogram_by_group(shape_df, col, label,
                                    os.path.join(hist_dir, f'Hist_FieldShape_{col}.png'), grp)
    print(f'Box plots saved to {plots_dir}, histograms to {hist_dir}')
    if grp.tests:
        asc._print_significant(omni, f'{name}, field shape', 45, 16)

    tables = {'Desc': desc, 'Omnibus': omni, 'PostHoc': post}
    if wd_dfs is not None:
        tables.update(asc.run_wall_distance(grp, wd_dfs, os.path.join(plots_dir, asc.WD_PLOTS_SUBDIR),
                                            metric_tables=('shape',)))
    return tables


def run_animal_controlled(full_df, shape_df, wd_dfs) -> dict:
    """Level 3 (asc.run_animal_controlled, shape only): each comparison in a
    temporary folder, then combined side by side (same y-range) into
    OUTPUT_DIR\\ANIMAL_CONTROLLED_DIR."""
    out_dir = os.path.join(OUTPUT_DIR, asc.ANIMAL_CONTROLLED_DIR)
    all_tables = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        names = list(asc.ANIMAL_CONTROLLED_COMPARISONS)
        with asc.plu.shared_y_axes([os.path.join(tmp_dir, n) for n in names]):
            for name, (animals, arenas) in asc.ANIMAL_CONTROLLED_COMPARISONS.items():
                grp = asc.animal_controlled_grouping(animals, arenas)
                wanted = {a.lower() for a in animals}

                def _select(d, grp=grp, wanted=wanted):
                    return d[d['animal'].astype(str).str.lower().isin(wanted)
                             & d['arena_type'].isin(grp.order)]

                sel_shape = _select(shape_df)
                print(f'\n{name}: place fields per animal and arena:')
                print(pd.crosstab(sel_shape['animal'], sel_shape['arena_type'], margins=True))
                tables = run_shape_comparison(
                    grp, _select(full_df), sel_shape, os.path.join(tmp_dir, name),
                    wd_dfs=None if wd_dfs is None else {k: _select(d) for k, d in wd_dfs.items()})
                tag = 'AC' + ''.join(grp.order)
                for tname, tbl in tables.items():
                    if len(tbl):
                        tbl.insert(0, 'animals', ', '.join(animals))
                    all_tables[f'{tag}_{tname}'] = tbl
        asc._combine_side_by_side([os.path.join(tmp_dir, n) for n in names], out_dir)
    if all_tables:
        xlsx = os.path.join(out_dir, AC_STATS_XLSX)
        _write_xlsx(all_tables, xlsx)
        print(f'\nAnimal-controlled field-shape statistics saved to {xlsx}')
    return all_tables


def _write_xlsx(tables: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for sheet, tbl in tables.items():
            tbl.to_excel(writer, sheet_name=sheet[:31], index=False)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_run_snapshot()

    print(f'Loading {asc.INPUT_EXCEL} ...')
    full, fields = asc.load_sheets()          # shape columns, Circle fields linearized
    if RUN_FIELD_DIMENSION_MAPS:
        asc.run_field_dimension_maps()

    full_df      = asc.build_full_metrics_df(full)
    shape_df     = asc.build_field_shape_df(full, fields)
    all_units_df = asc.build_all_units_df(full)
    print('\nPlace fields with a shape, by arena and session type:')
    print(pd.crosstab(shape_df['arena_type'], shape_df['session_type'].fillna('unrecognized'),
                      margins=True))

    wd_dfs = build_wd_dfs(full_df, shape_df) if INCLUDE_WALL_DISTANCE else None
    shape_df = asc.add_wall_field_flag(shape_df, wd_dfs)   # red / green rings on fld_width_ratio

    def _sub(dfs, select):
        return None if dfs is None else {k: select(d) for k, d in dfs.items()}

    all_tables = {}
    # 1. Session types within each arena
    for arena in asc.ARENA_TYPES:
        grp = asc.session_grouping(arena)

        def _in_arena(d, arena=arena, grp=grp):
            return d[(d['arena_type'] == arena) & d['session_type'].isin(grp.order)]

        tables = run_shape_comparison(grp, _in_arena(full_df), _in_arena(shape_df),
                                      os.path.join(OUTPUT_DIR, 'SessionType', arena),
                                      wd_dfs=_sub(wd_dfs, _in_arena))
        for name, tbl in tables.items():
            if len(tbl) and 'arena_type' not in tbl.columns:
                tbl.insert(0, 'arena_type', arena)
            all_tables[f'Ses{arena}_{name}'] = tbl

    # 2. Arena types
    for name, tbl in run_shape_comparison(asc.ARENA_GROUPING, full_df, shape_df,
                                          os.path.join(OUTPUT_DIR, 'ArenaType'),
                                          wd_dfs=wd_dfs).items():
        all_tables[f'Arena_{name}'] = tbl

    # 3. Animal-controlled arena types
    all_tables.update(run_animal_controlled(full_df, shape_df, wd_dfs))

    # 4. Animal IDs
    animal_grp = asc.animal_id_grouping(all_units_df)
    for name, tbl in run_shape_comparison(animal_grp, full_df, shape_df,
                                          os.path.join(OUTPUT_DIR, asc.ANIMAL_ID_DIR),
                                          wd_dfs=wd_dfs).items():
        all_tables[f'Animal_{name}'] = tbl

    # 5. All cells pooled (no tests, no wall distance -- as in v21)
    for name, tbl in run_shape_comparison(asc.ALL_CELLS_GROUPING, full_df, shape_df,
                                          os.path.join(OUTPUT_DIR, asc.ALL_CELLS_DIR)).items():
        if 'Omnibus' not in name and 'PostHoc' not in name:
            all_tables[f'All_{name}'] = tbl

    _write_xlsx(all_tables, SHAPE_STATS_XLSX)
    print(f'\nField-shape statistics saved to {SHAPE_STATS_XLSX}')
    parts = []
    for sheet, tbl in all_tables.items():
        if tbl is not None and len(tbl):
            tbl = tbl.copy()
            tbl.insert(0, 'table', sheet)
            parts.append(tbl)
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(SHAPE_STATS_CSV, index=False, encoding='utf-8-sig')
        print(f'Also saved to {SHAPE_STATS_CSV}')
    print(f'\nDone. Field-shape plots saved under {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
