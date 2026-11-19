# -*- coding: utf-8 -*-
"""
Lap rasters (LapRaster_AllArenas_v1) for the Best / Worst / Mean SIR cells – V1

Redraws only the lap raster figures of the cells PlaceCell_BestWorstMedian_SIR_v1
picked, with LapRaster_AllArenas_v1 (its RATEMAP_TYPE, Skaggs alpha and minimum
field size, set on PlaceCell_Main_v21 when it is imported). The cells are read
from each SELECTION_DIRS folder's BestWorstMedian_SIR_selection.xlsx, not picked
again, and each figure overwrites
  <selection dir>\\<Arena>\\<Best|Worst|Mean>\\<rank>_<animal>_<session>_<unit>_lap_raster.png
(the same file name PlaceCell_BestWorstMedian_SIR_v1 writes). A cell picked in
more than one folder is computed once and its figure copied.
"""

import os
import re
import shutil

import pandas as pd

import LapRaster_AllArenas_v1 as lap   # sets its ratemap settings on PlaceCell_Main_v21

pc = lap.pc


# ── Directories (edit these per run) ──────────────────────────────────────────

# Session folders (root_folder of PlaceCell_BestWorstMedian_SIR_v1): the
# selection's 'session' column is relative to this folder.
root_folder = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final'

_BWM = os.path.join(root_folder, 'Results', 'BestWorstMedian_SIR')
# Folders holding a BestWorstMedian_SIR_selection.xlsx and <Arena>\<Group> figures.
SELECTION_DIRS = (_BWM, os.path.join(_BWM, 'Vals'))

SELECTION_FILE = 'BestWorstMedian_SIR_selection.xlsx'


def _safe_name(text: str) -> str:
    """As PlaceCell_BestWorstMedian_SIR_v1._safe_name."""
    return re.sub(r'[\\/:*?"<>|\s]+', '_', text).strip('_')


def main() -> None:
    # (session, unit) -> every lap raster path it is drawn to.
    targets: dict[tuple[str, str], list[str]] = {}
    for sel_dir in SELECTION_DIRS:
        sel = pd.read_excel(os.path.join(sel_dir, SELECTION_FILE))
        for _, cell in sel.iterrows():
            session, unit = str(cell['session']), str(cell['unit'])
            name = _safe_name(f'{cell["rank"]}_{session}_{os.path.splitext(unit)[0]}')
            path = os.path.join(sel_dir, str(cell['arena']), str(cell['group']), f'{name}_lap_raster.png')
            if not os.path.isfile(path):
                print(f'[NEW] no existing figure at {path}')
            targets.setdefault((session, unit), []).append(path)
    print(f'{sum(map(len, targets.values()))} lap rasters, {len(targets)} distinct cells '
          f'({lap.RATEMAP_TYPE}, alpha = {pc.SKAGGS_ALPHA:g}, '
          f'min field = {pc.MIN_FIELD_SIZE_BINS} bins)\n')

    n_done = 0
    for i, ((session, unit), paths) in enumerate(targets.items(), start=1):
        print(f'[{i}/{len(targets)}]  {session}  |  {unit}')
        dirpath = os.path.join(root_folder, session)
        try:
            csv_path = lap._find_tracking_csv(dirpath)
            if csv_path is None:
                raise FileNotFoundError(f'no unique tracking file in {dirpath}')
            metrics, ctx = pc.compute_metrics(csv_path, os.path.join(dirpath, unit),
                                              pc.arena_width_cm, pc.target_bin_cm)
            if not ctx or not ctx['valid_mask'].any():
                raise ValueError('no valid bins')
            rows = lap.lap_raster_cell(metrics, ctx, dirpath, session, unit, paths[0])
        except Exception as e:
            print(f'  ERROR: {e}')
            continue
        if rows is None:
            print('  [SKIP] no place field detected; existing figure(s) left as they were')
            continue
        for p in paths[1:]:
            shutil.copyfile(paths[0], p)
        for p in paths:
            print(f'  [SAVED] {p}')
        n_done += len(paths)

    print(f'\nDone. {n_done}/{sum(map(len, targets.values()))} lap rasters redrawn.')


if __name__ == '__main__':
    main()
