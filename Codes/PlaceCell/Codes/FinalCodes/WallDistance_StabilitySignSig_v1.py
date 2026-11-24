# -*- coding: utf-8 -*-
"""
WallDistance_StabilitySignSig_v1.py

SUPERSEDED (20 Nov 2026) by WallDistance_Continuum_v2.py (stability_sig_pos / _neg are among
its metrics). Do not re-run.

Wall-distance (continuum) analysis of SIGNIFICANT POSITIVE and SIGNIFICANT
NEGATIVE split-half stability only, without re-running
ArenaSessionType_StatsComparison_v21.py:

    stability_sig_pos  p <= 0.05 and stability_score > 0   (% of place cells)
    stability_sig_neg  p <= 0.05 and stability_score < 0   (% of place cells)

Denominator: every place cell with a stability type (SigPos / SigNeg / NonSig;
v21 _stability_type), as in Categorical_stability_sig_pos/neg.png and
StabilitySignSig_Comparison_v1.py.

Everything else is v21's wall-distance analysis (its functions are imported and
its run_wall_distance is called with only these two metrics):
  * wall_dist of each place cell = the wall distance of the peak bin of its raw
    rate map (v21 docstring, 'Wall distance'); taken from sheet 'WD_Cells' of
    v21's STATS_XLSX (last full v21 run), or re-scanned from WD_PLACE_CELL_DIR
    if that sheet is missing (WallDistance_PerArenaPlots_v1.load_wd_cells).
  * Tests within each group: across distance levels (chi-square + Holm
    post hoc), trend with distance (logistic regression on wall_dist, or GEE
    clustered by cell if CONTINUOUS_TEST == 'LMM'); pooled over the groups:
    group-adjusted distance effect and distance x group interaction; groups at
    matched distance (chi-square + Holm).
  * Plots: % yes with Wilson 95 % CI at each distance, every group side by side.
  * Levels 1-4 as in v21 (level 5, all cells pooled, has no wall-distance
    analysis in v21 either):
        OUTPUT_DIR\\SessionType\\<arena>\\WallDistance\\          (Circle: one distance level -> tests only, no plot)
        OUTPUT_DIR\\ArenaType\\WallDistance\\                     + <arena>\\ per-arena plots (WD_PER_ARENA_PLOTS)
        OUTPUT_DIR\\AnimalControlledArenaType\\WallDistance\\     comparisons side by side, shared y-axis
        OUTPUT_DIR\\AnimalID\\WallDistance\\
            WallDist_Categorical_stability_sig_pos.png
            WallDist_Categorical_stability_sig_neg.png

Statistics: OUTPUT_DIR\\WallDist_StabilitySignSig_Stats.xlsx, the sheets v21
writes for wall distance ('<level>_WD_Counts', '_WD_Trend', '_WD_Pooled',
'_WD_Desc', '_WD_Omnibus', '_WD_PostHoc', '_WD_MatchOmnibus',
'_WD_MatchPostHoc'; <level> = 'Ses<arena>', 'Arena', 'AC<arenas>', 'Animal').
NOTE: 'p_holm_metrics' in '*_WD_Trend' is Holm over these TWO metrics within a
group, not over v21's full metric list, so it is not the value a full v21 run
would give.

INPUT_EXCEL and the v21 workbooks are only read; only the plots above and
WallDist_StabilitySignSig_Stats.xlsx are written.
"""

import os
import sys
import tempfile
from dataclasses import replace

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ArenaSessionType_StatsComparison_v21 as sc      # noqa: E402
import WallDistance_PerArenaPlots_v1 as wpa             # noqa: E402  (load_wd_cells)

INPUT_EXCEL = sc.INPUT_EXCEL
OUTPUT_DIR  = sc.OUTPUT_DIR
STATS_XLSX  = os.path.join(OUTPUT_DIR, 'WallDist_StabilitySignSig_Stats.xlsx')

METRIC_COLS = ('stability_sig_pos', 'stability_sig_neg')

# run_wall_distance takes its metrics from sc.wd_metric_specs(): restrict it to the two above.
METRICS = [m for m in sc.wd_metric_specs() if m.kind == 'cat' and m.col in METRIC_COLS]
assert [m.col for m in METRICS] == list(METRIC_COLS), METRICS
sc.wd_metric_specs = lambda: list(METRICS)


def run_level(grp: sc.Grouping, wd_full: pd.DataFrame, plots_dir: str) -> dict:
    """v21's wall-distance analysis of METRICS over the groups of `grp` (`wd_full`: the place
    cells to compare, with wall_dist), cells shaded as in run_comparison."""
    if not len(wd_full):
        print(f'\n{grp.title or grp.noun}: no place cells with a wall distance; skipped.')
        return {}
    grp = replace(grp, shade=sc.cell_shades(wd_full))
    return sc.run_wall_distance(grp, {'full': wd_full}, plots_dir)


def main():
    print(f'Loading {INPUT_EXCEL} ...')
    full = pd.read_excel(INPUT_EXCEL, sheet_name='Full')
    full_df = sc.build_full_metrics_df(full)

    wd_full = sc.attach_distance(full_df, wpa.load_wd_cells(), "Wall distance, table 'full'")

    print('\nPlace cells with a wall distance, by arena and stability type:')
    print(pd.crosstab(wd_full['arena_type'], wd_full['stability_type'].fillna('none'),
                      margins=True).to_string())
    for col in METRIC_COLS:
        print(f'\n{col}: significant cells by arena and wall distance')
        sig = wd_full[wd_full[col].map(sc._to_tri_bool) == True]          # noqa: E712
        print(pd.crosstab(sig['arena_type'], sig['wall_dist'], margins=True).to_string()
              if len(sig) else '  none')

    all_tables = {}

    # ── 1. Session types within each arena ──────────────────────────────────
    for arena in sc.ARENA_TYPES:
        grp = sc.session_grouping(arena)
        sub = wd_full[(wd_full['arena_type'] == arena) & wd_full['session_type'].isin(grp.order)]
        tables = run_level(grp, sub, os.path.join(OUTPUT_DIR, 'SessionType', arena,
                                                  sc.WD_PLOTS_SUBDIR))
        for name, tbl in tables.items():
            if len(tbl) and 'arena_type' not in tbl.columns:
                tbl.insert(0, 'arena_type', arena)
            all_tables[f'Ses{arena}_{name}'] = tbl

    # ── 2. Arena types (all session types pooled; + per-arena plots) ────────
    tables = run_level(sc.ARENA_GROUPING, wd_full,
                       os.path.join(OUTPUT_DIR, 'ArenaType', sc.WD_PLOTS_SUBDIR))
    all_tables.update({f'Arena_{n}': t for n, t in tables.items()})

    # ── 3. Animal-controlled arena types, side by side ──────────────────────
    ac_dir = os.path.join(OUTPUT_DIR, sc.ANIMAL_CONTROLLED_DIR)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dirs = [os.path.join(tmp_dir, n) for n in sc.ANIMAL_CONTROLLED_COMPARISONS]
        with sc.plu.shared_y_axes(tmp_dirs):
            for name, (animals, arenas) in sc.ANIMAL_CONTROLLED_COMPARISONS.items():
                grp = sc.animal_controlled_grouping(animals, arenas)
                wanted = {a.lower() for a in animals}
                sub = wd_full[wd_full['animal'].astype(str).str.lower().isin(wanted)
                              & wd_full['arena_type'].isin(grp.order)]
                tables = run_level(grp, sub, os.path.join(tmp_dir, name, sc.WD_PLOTS_SUBDIR))
                tag = 'AC' + ''.join(grp.order)
                for tname, tbl in tables.items():
                    if len(tbl):
                        tbl.insert(0, 'animals', ', '.join(animals))
                    all_tables[f'{tag}_{tname}'] = tbl
        sc._combine_side_by_side(tmp_dirs, ac_dir)

    # ── 4. Animal IDs ───────────────────────────────────────────────────────
    grp = sc.animal_id_grouping(sc.build_all_units_df(full))
    tables = run_level(grp, wd_full, os.path.join(OUTPUT_DIR, sc.ANIMAL_ID_DIR, sc.WD_PLOTS_SUBDIR))
    all_tables.update({f'Animal_{n}': t for n, t in tables.items()})

    with pd.ExcelWriter(STATS_XLSX, engine='openpyxl') as writer:
        for sheet_name, tbl in all_tables.items():
            tbl.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    print(f'\nStatistics saved to {STATS_XLSX}')
    print(f'Plots: WallDist_Categorical_stability_sig_pos/neg.png in the {sc.WD_PLOTS_SUBDIR}\\ '
          f'folders under {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
