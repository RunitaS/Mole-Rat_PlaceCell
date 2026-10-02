# -*- coding: utf-8 -*-
"""
Inventory of the sessions and cells (units) that pass the coverage criteria of
MeanRateMap_GaussianFit_QuadrantAnalysis_KDE_v14_EdgeCenterCoverCrit.py, i.e. the ones that
actually enter the mean rate map / quadrant / KDE analysis.

Rather than re-implementing the criteria (which would silently drift the moment the pipeline's
thresholds or geometry change), this script imports the pipeline module and calls its own
functions -- `make_handler`, `_detect_arena_key`, `_session_positions`, `compute_cell_ratemap`,
`_zone_occupancy_pct` -- so every number here is produced by exactly the code that runs during
the analysis, reading the same ROOT_DIRECTORY and the same COVERAGE_FRACTION /
MIN_ZONE_COVERAGE_PCT / min_occ_s / target_bin_cm constants.

Tracking is read through the pipeline's `_session_positions`, which is the one entry point that
applies the open-field frame corrections (arena centring, then the 120 deg 'Rotate' fix). This
script previously called `_load_tracking` / `_smooth_tracking_position` itself and so skipped
them, which made it report open-field 'Rotate' sessions as included that the pipeline was in
fact dropping.

Both coverage criteria depend on tracking alone, never on spikes, so this runs without loading
any .ntt spike train and without the 1000-shuffle bootstrap -- minutes rather than hours:

  1. Zone coverage (MIN_ZONE_COVERAGE_PCT, session level, collect_arena_results): the animal's
     occupied time must be >= 20% in the edge zone AND >= 20% in the centre zone, else the
     whole session folder (every unit in it) is dropped before any spikes are loaded.
  2. Bin coverage (COVERAGE_FRACTION, checked per unit in process_unit but identical for every
     unit of a session because the occupancy map is tracking-only): >= min_occ_s occupancy in
     at least COVERAGE_FRACTION of the arena's bins.

A session also never reaches either criterion if its folder does not hold exactly one tracking
file plus at least one .ntt file (the `len(tracking_files) == 1 and len(ntt_files) > 0` gate in
collect_arena_results); such folders are listed as excluded with that reason.

Outputs one workbook, OUTPUT_DIR/AllArenas/IncludedSessionsCells.xlsx, with sheets:
  Sessions_Included  -- one row per session that passed both criteria (with its coverage numbers)
  Sessions_Excluded  -- one row per session that did not, with the criterion it failed
  Cells_Included     -- one row per unit (.ntt file) inside an included session
  Summary            -- per-arena counts and the parameter values the screen ran at
plus flat CSV copies of the two main sheets next to it.

Every unit of an included session is used by the analysis: the pipeline does not re-test cells
for place-cell qualification, because ROOT_DIRECTORY already holds cells qualified upstream (see
that script's CELL SELECTION note). So Cells_Included IS the list of cells the maps and
statistics are built from -- there is no second, narrower subset.

Where a previous AllArenas_Summary.xlsx is present, its per-cell diagnostics (n_spikes, peak_fr,
sir, sparsity, bootstrap_sig) are merged in for reference. They are read from that saved run, not
recomputed here, and the workbook records its timestamp so a stale merge is visible. They do not
affect inclusion.
"""

import os
import sys
import importlib.util

import numpy as np
import pandas as pd

PIPELINE_FILE = 'MeanRateMap_GaussianFit_QuadrantAnalysis_KDE_v14_EdgeCenterCoverCrit.py'


def _load_pipeline():
    """Imports the pipeline module from this script's own folder. Its heavy work all sits under
    `if __name__ == '__main__'`, so importing it only brings in the constants and functions."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), PIPELINE_FILE)
    spec = importlib.util.spec_from_file_location('mrm_pipeline_v14', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


P = _load_pipeline()


def _path_parts(session_name: str) -> tuple:
    """Splits a session's ROOT_DIRECTORY-relative path (e.g. 'Fa8477\\Open\\Day8\\1Cntrl') into
    (animal, arena_folder, day, session_folder), padding with '' when the tree is shallower."""
    parts = session_name.split(os.sep)
    parts = parts + [''] * (4 - len(parts)) if len(parts) < 4 else parts
    animal, arena_folder = parts[0], parts[1]
    day, session_folder = parts[-2], parts[-1]
    return animal, arena_folder, day, session_folder


def session_coverage(csv_path: str, handler) -> dict | None:
    """Both coverage criteria for one session, from tracking alone.

    Reuses compute_cell_ratemap with an empty spike train: the occupancy map and the `valid`
    bin mask it returns are exactly the ones process_unit tests against
    handler.coverage_threshold_bins, and feeding that same mask to the pipeline's own
    _zone_occupancy_pct reproduces session_zone_coverage's edge/centre split. Returns None when
    the tracking file yields too few usable samples (the pipeline drops such sessions too).

    Positions come from P._session_positions, the pipeline's single tracking entry point, so the
    open-field centring and 120 deg rotation are applied here exactly as they are in the run."""
    x_cm, y_cm, t = P._session_positions(csv_path, handler)
    if len(t) < 2:
        return None

    cell = P.compute_cell_ratemap(x_cm, y_cm, t, np.empty(0, dtype=np.float64), handler)
    if cell is None:
        return None

    edge_pct, centre_pct = P._zone_occupancy_pct(handler, cell['occ_map'], cell['valid'])
    covered_bins = int(cell['valid'].sum())
    return dict(
        edge_occupancy_pct=round(edge_pct, 2),
        centre_occupancy_pct=round(centre_pct, 2),
        covered_bins=covered_bins,
        total_arena_bins=int(handler.total_arena_bins),
        coverage_pct=round(100.0 * covered_bins / handler.total_arena_bins, 2),
        coverage_threshold_bins=int(handler.coverage_threshold_bins),
        total_occupancy_s=round(float(cell['occ_map'][cell['valid']].sum()), 1),
    )


def scan_arena(arena_key: str, cfg: dict) -> tuple:
    """Walks ROOT_DIRECTORY for one arena and screens every session it finds, mirroring
    collect_arena_results' folder walk and its tracking-file/.ntt gate. Returns
    (included_rows, excluded_rows, cell_rows)."""
    handler = P.make_handler(cfg)
    root = P.ROOT_DIRECTORY

    included, excluded, cells = [], [], []
    for dirpath, _, filenames in os.walk(root):
        if P._detect_arena_key(dirpath) != arena_key:
            continue
        tracking_files_all = [f for f in filenames if f.lower().endswith(('.csv', '.xlsx'))]
        if P.COORD_UNITS == 'cm':
            tracking_files = [f for f in tracking_files_all if f.lower().endswith('_cm.csv')]
        else:
            tracking_files = [f for f in tracking_files_all if not f.lower().endswith('_cm.csv')]
        ntt_files = sorted(f for f in filenames if f.lower().endswith('.ntt'))
        if not ntt_files:
            continue

        session_name = os.path.relpath(dirpath, root)
        animal, arena_folder, day, session_folder = _path_parts(session_name)
        base = dict(arena=arena_key, animal=animal, arena_folder=arena_folder, day=day,
                    session_folder=session_folder, session=session_name,
                    n_units_in_folder=len(ntt_files))

        if len(tracking_files) != 1:
            excluded.append(dict(base, excluded_reason='tracking file count',
                                 detail=f'{len(tracking_files)} tracking files match '
                                        f"COORD_UNITS='{P.COORD_UNITS}' (need exactly 1)"))
            continue

        csv_path = os.path.join(dirpath, tracking_files[0])
        try:
            cov = session_coverage(csv_path, handler)
        except Exception as e:
            excluded.append(dict(base, excluded_reason='tracking error', detail=str(e)))
            continue
        if cov is None:
            excluded.append(dict(base, excluded_reason='no usable tracking',
                                 detail='fewer than 2 valid tracking samples'))
            continue

        row = dict(base, tracking_file=tracking_files[0], **cov)

        if (cov['edge_occupancy_pct'] < P.MIN_ZONE_COVERAGE_PCT or
                cov['centre_occupancy_pct'] < P.MIN_ZONE_COVERAGE_PCT):
            excluded.append(dict(row, excluded_reason='zone coverage',
                                 detail=f"edge={cov['edge_occupancy_pct']:.1f}%  "
                                        f"centre={cov['centre_occupancy_pct']:.1f}%  "
                                        f'(need >= {P.MIN_ZONE_COVERAGE_PCT:.0f}% in each)'))
            continue

        if cov['covered_bins'] < handler.coverage_threshold_bins:
            excluded.append(dict(row, excluded_reason='bin coverage',
                                 detail=f"{cov['covered_bins']}/{cov['total_arena_bins']} bins "
                                        f"({cov['coverage_pct']:.0f}%), need >= "
                                        f'{handler.coverage_threshold_bins} '
                                        f'({P.COVERAGE_FRACTION:.0%})'))
            continue

        included.append(row)
        for ntt_file in ntt_files:
            cells.append(dict(arena=arena_key, animal=animal, arena_folder=arena_folder, day=day,
                              session_folder=session_folder, session=session_name, unit=ntt_file,
                              edge_occupancy_pct=cov['edge_occupancy_pct'],
                              centre_occupancy_pct=cov['centre_occupancy_pct'],
                              coverage_pct=cov['coverage_pct']))

    print(f'[{arena_key}] {len(included)} sessions included ({len(cells)} units), '
          f'{len(excluded)} sessions excluded.')
    return included, excluded, cells


def _merge_diagnostics(cells_df: pd.DataFrame, summary_path: str) -> tuple:
    """Adds the per-cell diagnostics (n_spikes, peak_fr, sir, sparsity, bootstrap_sig) from the
    last saved AllArenas_Summary.xlsx to the cell list, for reference only -- none of them
    affects inclusion. Returns (dataframe, provenance_note); units that run did not cover are
    left NaN, and a missing or older summary simply skips the merge."""
    if cells_df.empty or not os.path.isfile(summary_path):
        return cells_df, f'not merged -- {summary_path} not found'
    try:
        summary = pd.read_excel(summary_path)
    except Exception as e:
        return cells_df, f'not merged -- could not read {summary_path}: {e}'

    keys = ['arena', 'session', 'unit']
    diagnostics = [c for c in ('n_spikes', 'peak_fr', 'sir', 'sparsity', 'bootstrap_sig')
                   if c in summary.columns]
    if not set(keys).issubset(summary.columns) or not diagnostics:
        return cells_df, f'not merged -- {os.path.basename(summary_path)} lacks the expected columns'

    stamp = pd.Timestamp(os.path.getmtime(summary_path), unit='s').strftime('%Y-%m-%d %H:%M')
    merged = cells_df.merge(summary[keys + diagnostics], on=keys, how='left')
    n_matched = int(merged[diagnostics[0]].notna().sum())
    note = (f'{", ".join(diagnostics)} merged from {os.path.basename(summary_path)} '
            f'(saved {stamp}); {n_matched}/{len(merged)} units matched')
    return merged, note


def main() -> None:
    out_dir = os.path.join(P.OUTPUT_DIR, 'AllArenas')
    os.makedirs(out_dir, exist_ok=True)

    print(f'ROOT_DIRECTORY : {P.ROOT_DIRECTORY}')
    print(f'Criteria       : MIN_ZONE_COVERAGE_PCT={P.MIN_ZONE_COVERAGE_PCT:.0f}%, '
          f'COVERAGE_FRACTION={P.COVERAGE_FRACTION:.0%}, min_occ_s={P.min_occ_s}, '
          f'target_bin_cm={P.target_bin_cm}\n')

    all_included, all_excluded, all_cells = [], [], []
    for arena_key in P._ARENA_ORDER:
        inc, exc, cells = scan_arena(arena_key, P.ARENA_CONFIGS[arena_key])
        all_included += inc
        all_excluded += exc
        all_cells += cells

    inc_df = pd.DataFrame(all_included)
    exc_df = pd.DataFrame(all_excluded)
    cells_df = pd.DataFrame(all_cells)
    for df, keys in ((inc_df, ['arena', 'session']), (exc_df, ['arena', 'session']),
                     (cells_df, ['arena', 'session', 'unit'])):
        if not df.empty:
            df.sort_values(keys, inplace=True, ignore_index=True)

    summary_path = os.path.join(out_dir, 'AllArenas_Summary.xlsx')
    cells_df, merge_note = _merge_diagnostics(cells_df, summary_path)
    print(f'\ndiagnostics: {merge_note}')

    counts = []
    for arena_key in P._ARENA_ORDER:
        inc_n = 0 if inc_df.empty else int((inc_df['arena'] == arena_key).sum())
        exc_n = 0 if exc_df.empty else int((exc_df['arena'] == arena_key).sum())
        cell_n = 0 if cells_df.empty else int((cells_df['arena'] == arena_key).sum())
        counts.append(dict(arena=arena_key, sessions_included=inc_n, sessions_excluded=exc_n,
                           cells_used=cell_n))
    counts.append(dict(arena='ALL',
                       sessions_included=sum(c['sessions_included'] for c in counts),
                       sessions_excluded=sum(c['sessions_excluded'] for c in counts),
                       cells_used=sum(c['cells_used'] for c in counts)))
    counts_df = pd.DataFrame(counts)

    params_df = pd.DataFrame([
        dict(parameter='ROOT_DIRECTORY', value=P.ROOT_DIRECTORY),
        dict(parameter='COORD_UNITS', value=P.COORD_UNITS),
        dict(parameter='MIN_ZONE_COVERAGE_PCT', value=P.MIN_ZONE_COVERAGE_PCT),
        dict(parameter='COVERAGE_FRACTION', value=P.COVERAGE_FRACTION),
        dict(parameter='min_occ_s', value=P.min_occ_s),
        dict(parameter='target_bin_cm', value=P.target_bin_cm),
        dict(parameter='OPEN_EDGE_ZONE_THRESHOLD_CM', value=P.OPEN_EDGE_ZONE_THRESHOLD_CM),
        dict(parameter='LINEAR_EDGE_ZONE_THRESHOLD_CM', value=P.LINEAR_EDGE_ZONE_THRESHOLD_CM),
        dict(parameter='pipeline_script', value=PIPELINE_FILE),
        dict(parameter='place-cell re-test', value='none -- cells pre-qualified upstream, '
                                                   'every unit of an included session is used'),
        dict(parameter='diagnostics', value=merge_note),
        dict(parameter='generated', value=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')),
    ])

    xlsx_path = os.path.join(out_dir, 'IncludedSessionsCells.xlsx')
    with pd.ExcelWriter(xlsx_path) as writer:
        counts_df.to_excel(writer, sheet_name='Summary', index=False)
        params_df.to_excel(writer, sheet_name='Summary', index=False, startrow=len(counts_df) + 3)
        inc_df.to_excel(writer, sheet_name='Sessions_Included', index=False)
        exc_df.to_excel(writer, sheet_name='Sessions_Excluded', index=False)
        cells_df.to_excel(writer, sheet_name='Cells_Included', index=False)
    print(f'[SAVED] {xlsx_path}')

    for df, name in ((inc_df, 'IncludedSessions.csv'), (cells_df, 'IncludedCells.csv')):
        p = os.path.join(out_dir, name)
        df.to_csv(p, index=False)
        print(f'[SAVED] {p}')

    print('\n' + counts_df.to_string(index=False))


if __name__ == '__main__':
    main()
    print('\nDone.')
