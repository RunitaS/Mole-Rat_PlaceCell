# -*- coding: utf-8 -*-
"""
Statistical comparison of the multi-peak (KDE) place-field metrics written by
PlaceCell_MultiPeak_KDE.py (its output_excel, 'Fields' sheet), at the same two
levels as ArenaSessionType_StatsComparison_v8.py:

  1. Session types (field conditions) WITHIN each arena type:
         Open   : Cntrl, Rotate, Zero
         Linear : 0, 90, 180, 270
         Circle : Stnd, NoRot, Rot
  2. Arena types as a whole (Circle / Linear / Open, all session types pooled).

The tests, plots and tables are those of the comparison script (imported as
STATS_MODULE below, so its CONTINUOUS_TEST, METRIC_TRANSFORM,
LINK_CELLS_ACROSS_ARENAS, colours and figure style all apply unchanged):
  - continuous metrics: Kruskal-Wallis + pairwise Mann-Whitney U ('KW') or
    the nested linear mixed model ('LMM'), Holm-Bonferroni corrected; box
    plots with jittered points and significance lines, plus histograms.
  - % outcomes: chi-square test of independence + pairwise chi-square tests,
    Holm-Bonferroni corrected; bar plots.

Metrics compared:
  Per cell (one row per place cell):
    - n_multi_peak_fields    Multi-peak fields per cell
    - pct_multi_peak_fields  % of the cell's fields that are multi-peak
    - max_n_peaks            Most peaks in any one of the cell's fields
    - has_multi_peak_field   % of place cells with >= 1 multi-peak field
  Per field (one row per place field; several fields of a cell are repeated
  measures -- with 'KW' they are treated as independent, the LMM's cell
  random effect accounts for them):
    - n_peaks                Number of peaks per field
    - multi_peak             % of place fields that are multi-peak
  Per multi-peak field only:
    - shallowest_trough_rel    trough / lower peak of the weakest separation
    - lowest_counted_peak_rel  lowest counted peak / highest peak
    - peak_separation_cm       distance between neighbouring peaks (cm)

Outputs (OUTPUT_DIR):
    SessionType\\<arena>\\              box/bar plots, session types of <arena>
    SessionType\\<arena>\\Histograms\\
    ArenaType\\                         box/bar plots, arena types
    ArenaType\\Histograms\\
    MultiPeak_Stats.xlsx                sheets 'Ses<arena>_*' and 'Arena_*'
    MultiPeak_AllStats.csv

Run on its own (edit MULTIPEAK_EXCEL / OUTPUT_DIR below), or from the end of
PlaceCell_MultiPeak_KDE.py (RUN_STATS = True), which calls run_stats().
"""

import importlib
import os

import pandas as pd

# ── Parameters ──────────────────────────────────────────────────────────────

MULTIPEAK_EXCEL = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Alpha200\MultiPeak_KDE_Skaggs_adaptive_binned_Rel75\MultiPeak_KDE_Fields.xlsx'
OUTPUT_DIR      = os.path.join(os.path.dirname(MULTIPEAK_EXCEL), 'Stats')

# Comparison script whose tests, plots and figure style are reused.
STATS_MODULE = 'ArenaSessionType_StatsComparison_v8'

# Fields left out of every comparison:
EXCLUDE_NO_KDE_FIELDS     = True   # no KDE fitted (too few non-zero bins, see kde_note)
EXCLUDE_RATEMAP_MISMATCH  = True   # rebuilt ratemap did not reproduce the Excel peak

CELL_METRICS = {
    'n_multi_peak_fields':   'Multi-peak fields per cell',
    'pct_multi_peak_fields': 'Percentage of fields per cell that are multi-peak',
    'max_n_peaks':           'Most peaks in a field of the cell',
}
FIELD_METRICS = {
    'n_peaks': 'Number of peaks per field',
}
MULTI_FIELD_METRICS = {
    'shallowest_trough_rel':   'Trough / lower peak (multi-peak fields)',
    'lowest_counted_peak_rel': 'Lowest peak / highest peak (multi-peak fields)',
    'peak_separation_cm':      'Distance between neighbouring peaks, cm (multi-peak fields)',
}
CELL_CATEGORICAL = {
    'has_multi_peak_field': 'Cells with a multi-peak field (% of place cells)',
}
FIELD_CATEGORICAL = {
    'multi_peak': 'Multi-peak fields (% of place fields)',
}


def load_tables(multipeak_excel: str, sv):
    """(cells_df, fields_df, multi_df) tagged with arena_type / session_type /
    cell_id by the comparison script's _tag_groups."""
    fields = pd.read_excel(multipeak_excel, sheet_name='Fields')
    n_all = len(fields)
    if EXCLUDE_NO_KDE_FIELDS:
        fields = fields[fields['kde_note'].fillna('').astype(str).str.strip() == '']
    if EXCLUDE_RATEMAP_MISMATCH:
        fields = fields[fields['ratemap_check'].apply(sv._to_bool)]
    print(f'{len(fields)} of {n_all} fields kept '
          f'(EXCLUDE_NO_KDE_FIELDS={EXCLUDE_NO_KDE_FIELDS}, '
          f'EXCLUDE_RATEMAP_MISMATCH={EXCLUDE_RATEMAP_MISMATCH}).')

    # 'arena_type' in the Fields sheet is the field geometry ('2D'/'Circle');
    # _tag_groups replaces it with the arena (Circle / Linear / Open).
    fields = fields.drop(columns=['arena_type'])
    fields['multi_peak'] = fields['multi_peak'].apply(sv._to_bool)

    cells = (fields.groupby(['session', 'unit'])
                   .agg(n_fields=('field_number', 'count'),
                        n_multi_peak_fields=('multi_peak', 'sum'),
                        max_n_peaks=('n_peaks', 'max'))
                   .reset_index())
    cells['pct_multi_peak_fields'] = 100.0 * cells['n_multi_peak_fields'] / cells['n_fields']
    cells['has_multi_peak_field']  = cells['n_multi_peak_fields'] > 0

    fields_df = sv._tag_groups(fields, 'field(s)')
    cells_df  = sv._tag_groups(cells,  'place cell(s)')
    multi_df  = fields_df[fields_df['multi_peak']]
    return cells_df, fields_df, multi_df


def run_comparison(sv, grp, cells_df, fields_df, multi_df, plots_dir: str) -> dict:
    """Every multi-peak metric across the groups of `grp`: statistics, box/bar
    plots in `plots_dir` and histograms in `plots_dir`/Histograms. Returns
    {table_name: table}."""
    name = grp.title or grp.noun.capitalize()
    print(f'\n{"=" * 70}\n{name} -- comparing {grp.noun}: '
          f'{[sv._label(g) for g in grp.order]}\n{"=" * 70}')
    print('Place cells / fields / multi-peak fields per group:')
    print(pd.DataFrame({
        'cells':  cells_df[grp.col].value_counts(),
        'fields': fields_df[grp.col].value_counts(),
        'multi-peak fields': multi_df[grp.col].value_counts(),
    }).reindex(grp.order).fillna(0).astype(int))
    if not len(cells_df):
        print('No place cells; skipping.')
        return {}

    # ── Continuous metrics ──────────────────────────────────────────────────
    continuous = ((cells_df,  CELL_METRICS,        'Cell'),
                  (fields_df, FIELD_METRICS,       'Field'),
                  (multi_df,  MULTI_FIELD_METRICS, 'MultiField'))
    desc, omni, post = [], [], []
    for df, metrics, prefix in continuous:
        d, o, p = sv.run_all_comparisons(df, metrics, prefix, grp, plots_dir)
        desc.append(d)
        omni.append(o)
        post.append(p)

    # ── Categorical (proportion) comparisons ────────────────────────────────
    cat_desc, cat_omni, cat_post = [], [], []
    for df, metrics in ((cells_df, CELL_CATEGORICAL), (fields_df, FIELD_CATEGORICAL)):
        for col, label in metrics.items():
            d, o, p = sv.compare_categorical(df, col, label, grp)
            cat_desc.append(d)
            cat_omni.append(o)
            if len(p):
                cat_post.append(p)
            sv.plot_categorical(df, col, label,
                                os.path.join(plots_dir, f'Categorical_{col}.png'), grp,
                                stats_lines=sv._categorical_stats_lines(d, o, p, grp),
                                posthoc_df=p)

    omnibus_df     = pd.concat(omni, ignore_index=True)
    cat_omnibus_df = pd.DataFrame(cat_omni)
    print(f'Box/bar plots saved to {plots_dir}')
    sv._print_significant(omnibus_df, f'{name}, continuous metrics', 60, 16)
    sv._print_significant(cat_omnibus_df, f'{name}, categorical metrics', 50, 30)

    # ── Histograms ──────────────────────────────────────────────────────────
    hist_dir = os.path.join(plots_dir, 'Histograms')
    os.makedirs(hist_dir, exist_ok=True)
    for df, metrics, prefix in continuous:
        for col, label in metrics.items():
            sv.plot_histogram_by_group(df, col, label,
                                       os.path.join(hist_dir, f'Hist_{prefix}_{col}.png'), grp)
    print(f'Histogram plots saved to {hist_dir}')

    def _cat(tables):
        tables = [t for t in tables if len(t)]
        return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

    return {
        'Desc':       _cat(desc),
        'Omnibus':    omnibus_df,
        'PostHoc':    _cat(post),
        'CatDesc':    _cat(cat_desc),
        'CatOmnibus': cat_omnibus_df,
        'CatPostHoc': _cat(cat_post),
    }


def run_stats(multipeak_excel: str = MULTIPEAK_EXCEL, output_dir: str = OUTPUT_DIR) -> None:
    sv = importlib.import_module(STATS_MODULE)
    print(f'\nMulti-peak statistics: {multipeak_excel}')
    print(f'Tests, plots and style from {STATS_MODULE} '
          f'(CONTINUOUS_TEST = {sv.CONTINUOUS_TEST!r})')
    cells_df, fields_df, multi_df = load_tables(multipeak_excel, sv)

    print('\nPlace cells by arena and session type:')
    print(pd.crosstab(cells_df['arena_type'], cells_df['session_type'].fillna('unrecognized'),
                      margins=True))

    os.makedirs(output_dir, exist_ok=True)
    all_tables = {}

    # ── 1. Session types (field conditions) within each arena ───────────────
    for arena in sv.ARENA_TYPES:
        grp = sv.session_grouping(arena)

        def _in_arena(d):
            return d[(d['arena_type'] == arena) & d['session_type'].isin(grp.order)]

        tables = run_comparison(sv, grp, _in_arena(cells_df), _in_arena(fields_df),
                                _in_arena(multi_df),
                                os.path.join(output_dir, 'SessionType', arena))
        for name, tbl in tables.items():
            if len(tbl) and 'arena_type' not in tbl.columns:
                tbl.insert(0, 'arena_type', arena)
            all_tables[f'Ses{arena}_{name}'] = tbl

    # ── 2. Arena types as a whole (all session types pooled) ────────────────
    tables = run_comparison(sv, sv.ARENA_GROUPING, cells_df, fields_df, multi_df,
                            os.path.join(output_dir, 'ArenaType'))
    for name, tbl in tables.items():
        all_tables[f'Arena_{name}'] = tbl

    # ── Save all statistics (workbook + one long-format CSV) ────────────────
    stats_xlsx = os.path.join(output_dir, 'MultiPeak_Stats.xlsx')
    stats_csv  = os.path.join(output_dir, 'MultiPeak_AllStats.csv')
    with pd.ExcelWriter(stats_xlsx, engine='openpyxl') as writer:
        for sheet_name, tbl in all_tables.items():
            tbl.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    print(f'\nAll statistics saved to {stats_xlsx}')

    parts = []
    for sheet_name, tbl in all_tables.items():
        if tbl is None or not len(tbl):
            continue
        tbl = tbl.copy()
        tbl.insert(0, 'table', sheet_name)
        parts.append(tbl)
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(stats_csv, index=False, encoding='utf-8-sig')
        print(f'All statistics also saved to {stats_csv}')
    print(f'\nDone. Plots saved under {output_dir}')


if __name__ == '__main__':
    run_stats()
