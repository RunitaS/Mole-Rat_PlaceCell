# -*- coding: utf-8 -*-
"""
% of place cells with SIGNIFICANT POSITIVE and SIGNIFICANT NEGATIVE split-half
stability, compared across every condition of ArenaSessionType_StatsComparison_v21.py.

In v21, 'Categorical_stability_significant.png' counts every cell with
split-half p <= ALPHA (0.05), whatever the sign of its stability score. This
script splits that by sign:

    stability_sig_pos  p <= 0.05 and stability_score > 0
    stability_sig_neg  p <= 0.05 and stability_score < 0

Denominator: ALL place cells with a stability type (SigPos / SigNeg / NonSig;
v21 _stability_type), the same base as the % significant plot, so
% pos + % neg = % significant. Test: chi-square omnibus + Holm-corrected
pairwise chi-square (v21 compare_categorical), as for every other proportion.

Only this analysis is run (no field shape, temporal stability, multi-peak,
wall distance ...). The cells, groups, tests and plot style are those of v21
(its functions are imported), and the plots are written to the same folders of
v21's OUTPUT_DIR, with the file names v21 now gives them:

    OUTPUT_DIR\\SessionType\\<arena>\\           level 1, session types of each arena
    OUTPUT_DIR\\ArenaType\\                      level 2, arena types
    OUTPUT_DIR\\AnimalControlledArenaType\\      level 3, comparisons side by side
    OUTPUT_DIR\\AnimalID\\                       level 4, animals
    OUTPUT_DIR\\AllCells\\                       level 5, all cells pooled (no tests)
        Categorical_stability_sig_pos.png
        Categorical_stability_sig_neg.png
    OUTPUT_DIR\\StabilitySignSig_Stats.xlsx      'Desc', 'Omnibus', 'PostHoc'
                                                (one row per level / group / metric)

INPUT_EXCEL and OUTPUT_DIR are v21's (sc.INPUT_EXCEL, sc.OUTPUT_DIR); INPUT_EXCEL
is only read.
"""

import os
import sys
import tempfile
from dataclasses import replace

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ArenaSessionType_StatsComparison_v21 as sc   # noqa: E402

INPUT_EXCEL = sc.INPUT_EXCEL
OUTPUT_DIR  = sc.OUTPUT_DIR
STATS_XLSX  = os.path.join(OUTPUT_DIR, 'StabilitySignSig_Stats.xlsx')

METRICS = {c: sc.CATEGORICAL_METRICS_PLACE_CELLS[c]
           for c in ('stability_sig_pos', 'stability_sig_neg')}


def compare_and_plot(full_df: pd.DataFrame, grp: sc.Grouping, plots_dir: str, level: str,
                     tables: dict):
    """Both metrics across the groups of `grp`: stats into `tables`, plots into `plots_dir`."""
    if not len(full_df):
        print(f'{level}: no place cells; skipped.')
        return
    grp = replace(grp, shade=sc.cell_shades(full_df))
    os.makedirs(plots_dir, exist_ok=True)
    for col, label in METRICS.items():
        d, o, p = sc.compare_categorical(full_df, col, label, grp)
        sc.plot_categorical(full_df, col, label, os.path.join(plots_dir, f'Categorical_{col}.png'),
                            grp, stats_lines=sc._categorical_stats_lines(d, o, p, grp),
                            posthoc_df=p)
        d = d.rename(columns={grp.col: 'group'})
        for name, tbl in (('Desc', d), ('Omnibus', pd.DataFrame([o])), ('PostHoc', p)):
            if len(tbl) and (grp.tests or name == 'Desc'):   # one group: nothing tested
                tbl = tbl.copy()
                tbl.insert(0, 'level', level)
                tables[name].append(tbl)
        print(f'\n{level} -- {label}')
        print(d[['group', 'n', 'n_true', 'pct_true']].to_string(index=False))
        if grp.tests:
            print(f"  {o.get('test')}: p = {o.get('p_value')}")


def main():
    print(f'Loading {INPUT_EXCEL} ...')
    full = pd.read_excel(INPUT_EXCEL, sheet_name='Full')
    full_df = sc.build_full_metrics_df(full)

    print('\nPlace cells by arena and stability type:')
    print(pd.crosstab(full_df['arena_type'], full_df['stability_type'].fillna('none'),
                      margins=True))

    tables = {'Desc': [], 'Omnibus': [], 'PostHoc': []}

    # ── 1. Session types within each arena ──────────────────────────────────
    for arena in sc.ARENA_TYPES:
        grp = sc.session_grouping(arena)
        sub = full_df[(full_df['arena_type'] == arena) & full_df['session_type'].isin(grp.order)]
        compare_and_plot(sub, grp, os.path.join(OUTPUT_DIR, 'SessionType', arena),
                         f'Session type, {arena}', tables)

    # ── 2. Arena types ──────────────────────────────────────────────────────
    compare_and_plot(full_df, sc.ARENA_GROUPING, os.path.join(OUTPUT_DIR, 'ArenaType'),
                     'Arena type', tables)

    # ── 3. Animal-controlled arena types, side by side ──────────────────────
    ac_dir = os.path.join(OUTPUT_DIR, sc.ANIMAL_CONTROLLED_DIR)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dirs = [os.path.join(tmp_dir, n) for n in sc.ANIMAL_CONTROLLED_COMPARISONS]
        with sc.plu.shared_y_axes(tmp_dirs):
            for name, (animals, arenas) in sc.ANIMAL_CONTROLLED_COMPARISONS.items():
                grp = sc.animal_controlled_grouping(animals, arenas)
                wanted = {a.lower() for a in animals}
                sub = full_df[full_df['animal'].astype(str).str.lower().isin(wanted)
                              & full_df['arena_type'].isin(grp.order)]
                compare_and_plot(sub, grp, os.path.join(tmp_dir, name),
                                 f'Animal-controlled, {name} ({", ".join(animals)})', tables)
        sc._combine_side_by_side(tmp_dirs, ac_dir)

    # ── 4. Animal IDs ───────────────────────────────────────────────────────
    compare_and_plot(full_df, sc.animal_id_grouping(sc.build_all_units_df(full)),
                     os.path.join(OUTPUT_DIR, sc.ANIMAL_ID_DIR), 'Animal ID', tables)

    # ── 5. All cells pooled (no tests) ──────────────────────────────────────
    compare_and_plot(full_df, sc.ALL_CELLS_GROUPING, os.path.join(OUTPUT_DIR, sc.ALL_CELLS_DIR),
                     'All cells', tables)

    with pd.ExcelWriter(STATS_XLSX, engine='openpyxl') as writer:
        for name, parts in tables.items():
            tbl = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            tbl.to_excel(writer, sheet_name=name, index=False)
    print(f'\nStatistics saved to {STATS_XLSX}')
    print(f'Plots saved under {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
