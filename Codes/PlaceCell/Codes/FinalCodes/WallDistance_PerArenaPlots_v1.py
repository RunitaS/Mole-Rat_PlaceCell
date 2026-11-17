"""
WallDistance_PerArenaPlots_v1.py

Standalone re-plot of the ArenaType wall-distance figures, one arena at a time,
without re-running ArenaSessionType_StatsComparison_v20.py.

The ArenaType\\WallDistance\\WallDist_*.png plots put Circle, Linear and Open on
one y-axis; the Circle values stretch it so far that trends in the other arenas
are hard to see. This script saves every WallDist_<metric>.png again for each
arena alone, in

    OUTPUT_DIR\\ArenaType\\WallDistance\\Circle\\
    OUTPUT_DIR\\ArenaType\\WallDistance\\Linear\\
    OUTPUT_DIR\\ArenaType\\WallDistance\\Open\\

y-axis scaled to that arena's values only (yes/no metrics: to that arena's % and
Wilson CI range instead of 0-100 %), x-axis to its own distance levels. Stats
text: that arena's trend with wall distance and its across-distance test, the
same tests as in the combined plots (no pooled / interaction / matched-distance
lines, which compare arenas). The combined plots are left untouched.

Everything comes from v20 (settings, data building, tests, plot code, imported
from ArenaSessionType_StatsComparison_v20.py). To be fast it reuses the outputs
of the last v20 run instead of recomputing them:
  * wall distance of every unit: sheet 'WD_Cells' of v20's STATS_XLSX
    (re-scans WD_PLACE_CELL_DIR only if that sheet is missing),
  * temporal stability: sheet 'PerCell' of OUTPUT_DIR\\TS_RESULTS_XLSX
    (re-runs the maps only if missing),
  * multi-peak fields: MULTIPEAK_EXCEL (re-runs the KDE only if missing),
  * within-field rate: INPUT_EXCEL column if present, else sheet 'PerCell' of
    OUTPUT_DIR\\IFR_RESULTS_XLSX (rebuilt, and that file written, only if missing).
INPUT_EXCEL's 'Full' / 'PlaceFields' are read directly (FIELD_SHAPE_EXCEL is not
rewritten). Apart from the plots (and IFR_RESULTS_XLSX when it has to be
rebuilt), nothing in OUTPUT_DIR is written.

The same per-arena plots are made by v20 itself when WD_PER_ARENA_PLOTS = True.
"""

import os
import sys
from dataclasses import replace

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ArenaSessionType_StatsComparison_v20 as v  # noqa: E402


def load_inputs():
    """full, fields of INPUT_EXCEL (with the field-shape columns, nothing saved)."""
    print(f'Loading {v.INPUT_EXCEL} ...')
    full = pd.read_excel(v.INPUT_EXCEL, sheet_name='Full')
    fields = v.pfs.add_field_shape(pd.read_excel(v.INPUT_EXCEL, sheet_name='PlaceFields'))
    return full, fields


def load_ts_df():
    if not v.INCLUDE_TEMPORAL_STABILITY:
        return None
    path = os.path.join(v.OUTPUT_DIR, v.TS_RESULTS_XLSX)
    if os.path.isfile(path):
        print(f'Temporal stability: PerCell sheet of {path}')
        return v._tag_groups(pd.read_excel(path, sheet_name='PerCell'),
                             'place cell(s) (temporal stability)')
    print(f'{path} not found; running the temporal stability maps.')
    return v.build_temporal_stability_df()


def load_mp_dfs(full, fields):
    if not v.INCLUDE_MULTIPEAK:
        return None
    if os.path.isfile(v.MULTIPEAK_EXCEL):
        v.RUN_MULTIPEAK_KDE = False     # read MULTIPEAK_EXCEL instead of re-running the KDE
    else:
        print(f'{v.MULTIPEAK_EXCEL} not found; running the multi-peak KDE.')
    return v.build_multipeak_dfs(full, fields)


def load_wd_cells():
    """Units with their wall distance, from the last run's WD_Cells sheet (else a re-scan)."""
    try:
        cells_df = pd.read_excel(v.STATS_XLSX, sheet_name='WD_Cells')
        print(f'Wall distance per unit: WD_Cells sheet of {v.STATS_XLSX}')
    except (FileNotFoundError, ValueError):
        print(f'No WD_Cells sheet in {v.STATS_XLSX}; scanning {v.WD_PLACE_CELL_DIR} ...')
        geoms = v.make_geometries()
        units = {a: v.wall_dist_units(g, a) for a, g in geoms.items()}
        return v.scan_place_cell_dir(geoms, units)[1]
    cells_df['_skey'] = cells_df['session'].map(v._session_key)
    cells_df['_ukey'] = cells_df['unit'].map(v._unit_key)
    return cells_df


def build_wd_dfs(full_df, field_df, shape_df, ts_df, mp_dfs, cells_df) -> dict:
    """As build_wall_distance_dfs, without the scan, schematics and peak-distance plot."""
    wd_dfs = {'full': full_df, 'field': field_df, 'shape': shape_df}
    if ts_df is not None:
        wd_dfs['ts'] = ts_df
    if mp_dfs is not None:
        wd_dfs.update({f'MP_{level}': d for level, d in mp_dfs.items()})
    wd_dfs = {name: v.attach_distance(d, cells_df, f'Wall distance, table {name!r}')
              for name, d in wd_dfs.items()}
    wd_dfs['full'] = v.add_speed_direction_flags(wd_dfs['full'])
    print('\nPlace cells by arena and wall distance (Full sheet):')
    print(pd.crosstab(wd_dfs['full']['arena_type'], wd_dfs['full']['wall_dist'],
                      margins=True).to_string())
    return wd_dfs


def per_arena_results(grp, wd_dfs) -> tuple:
    """Per-arena trend and across-distance tests of every plotted metric, as in
    run_wall_distance (pooled / matched tests are not needed for one-arena plots)."""
    metrics = [m for m in v.wd_metric_specs() if m.table in wd_dfs and m.kind in ('box', 'cat')]
    results = {}
    for i, m in enumerate(metrics, start=1):
        print(f'[{i}/{len(metrics)}] {m.stem}')
        cells = v._wd_cells(wd_dfs, m, grp)
        subs = {g: cells[cells[grp.col] == g] for g in grp.order if (cells[grp.col] == g).any()}
        res = dict(subs=subs, trend={}, across={})
        for g, sub in subs.items():
            _, omni, _ = v.across_distance(sub, m)
            if omni is not None:
                res['across'][g] = omni
            res['trend'][g] = v.group_trend(sub, m)
        results[m.stem] = res
    return metrics, results


def main():
    full, fields = load_inputs()
    full_df = v.build_full_metrics_df(full)
    if v.INCLUDE_INFIELD_RATE:
        ifr_xlsx = os.path.join(v.OUTPUT_DIR, v.IFR_RESULTS_XLSX)
        has_col = (v.INFIELD_RATE_COL in full.columns
                   and pd.to_numeric(full[v.INFIELD_RATE_COL], errors='coerce').notna().any())
        if v.INFIELD_RATE_SOURCE != 'excel' and not has_col and os.path.isfile(ifr_xlsx):
            print(f'Within-field firing rate: PerCell sheet of {ifr_xlsx}')
            full_df = v.merge_infield_rate(full_df, pd.read_excel(ifr_xlsx, sheet_name='PerCell'))
        else:
            full_df = v.add_infield_rate(full_df, full, fields)
    field_df = v.build_field_metrics_df(full, fields)
    shape_df = v.build_field_shape_df(full, fields)
    ts_df = load_ts_df()
    full_df = v.add_peak_dwell_flag(full_df, ts_df)
    mp_dfs = load_mp_dfs(full, fields)

    wd_dfs = build_wd_dfs(full_df, field_df, shape_df, ts_df, mp_dfs, load_wd_cells())

    # The ArenaType comparison (all session types pooled), shaded as in run_comparison.
    grp = replace(v.ARENA_GROUPING, shade=v.cell_shades(full_df))
    metrics, results = per_arena_results(grp, wd_dfs)

    plots_dir = os.path.join(v.OUTPUT_DIR, 'ArenaType', v.WD_PLOTS_SUBDIR)
    v.plot_wd_per_group(metrics, results, grp, plots_dir)
    print(f'\nDone. Per-arena wall-distance plots in {plots_dir}\\<arena>\\')


if __name__ == '__main__':
    main()
