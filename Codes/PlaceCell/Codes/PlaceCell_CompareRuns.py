"""
PlaceCell_CompareRuns.py

Compare the output Excel files of PlaceCell_Main_v16.py that were run on the
SAME data set with different smoothing / place-cell criteria.

Only the 'Full' sheet of each file is used.

Cell identity = (session, unit).  Cells that are NOT present in every file are
removed from all analyses below and listed in a separate Excel file
(Cells_Not_In_All_Files.xlsx) with the files they are present in / absent from.

Cell types (Full sheet, same definitions as the summary printed by v16):
    Total units processed             : every row
    Place cells found                 : place_cell == True
    Speed cells (binned AND inst.)    : speed_cell == True
    Place cells also speed-modulated  : place_cell == True  AND speed_cell == True
    Non-place cells speed-modulated   : place_cell == False AND speed_cell == True
    Speed not tested                  : speed_cell == 'not tested'
    Place fields detected             : n_fields_detected > 0   (cells with >= 1 field;
                                        the total number of fields is also reported)

Percent accepted / rejected uses the total number of (common) units in the file
as the denominator: accepted = cells in the type, rejected = all other units.

Outputs (OUTPUT_DIR):
    Fig1_Percent_Accepted_Rejected.png/.svg
    Fig2_Count_Accepted.png/.svg
    Fig3_Common_<A>_vs_<B>.png/.svg      one per file pair  (n choose 2)
    Fig4_Common_All_Files.png/.svg       cells common to every file
    PlaceCell_CompareRuns_Summary.xlsx   counts, %, pairwise + all-file overlap,
                                         per-cell membership for each type
    Cells_Not_In_All_Files.xlsx          cells missing from at least one file
"""

import os
import glob
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

# ── User settings ─────────────────────────────────────────────────────────────

INPUT_DIR  = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\PC_SmoothCompare\PC_Res'
OUTPUT_DIR = os.path.join(INPUT_DIR, 'Compare_Output')
SHEET      = 'Full'
KEY_COLS   = ['session', 'unit']

# ── Plot style (fixed colour per file, in file order; never cycled) ───────────

# File colours are equidistant on the magma colormap: n files sit at
# k/(n+1), k = 1..n (e.g. 0.2, 0.4, 0.6, 0.8 for 4 files), so the gaps between
# colours and to both ends of the map are equal and the near-black /
# near-white ends are never used.
FILE_CMAP      = 'magma'
REJECT_COLOR   = '#e1e0d9'
COMMON_COLOR   = '#52514e'
INK_PRIMARY    = '#0b0b0b'
INK_SECONDARY  = '#52514e'
INK_MUTED      = '#898781'
GRID_COLOR     = '#e1e0d9'
SURFACE        = '#fcfcfb'

TYPES = [
    ('total',          'Total units processed',            'Total units\nprocessed'),
    ('place',          'Place cells found',                'Place\ncells'),
    ('speed',          'Speed cells (binned AND inst. tests)', 'Speed\ncells'),
    ('place_speed',    'Place cells also speed-modulated', 'Place +\nspeed-mod.'),
    ('nonplace_speed', 'Non-place cells speed-modulated',  'Non-place +\nspeed-mod.'),
    ('speed_not_tested', 'Speed not tested',               'Speed not\ntested'),
    ('place_fields',   'Place fields detected',            'Cells with\nplace fields'),
]
TYPE_NAME  = {k: name for k, name, _ in TYPES}
TYPE_SHORT = {k: short for k, _, short in TYPES}


# ── Loading ───────────────────────────────────────────────────────────────────

def _file_labels(paths):
    """Short label per file: file stem with the prefix shared by all stems removed
    (cut back to the last '_' so e.g. 'All_TT_PlaceChar_' is stripped)."""
    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if len(stems) > 1:
        prefix = os.path.commonprefix(stems)
        prefix = prefix[:prefix.rfind('_') + 1] if '_' in prefix else ''
        stems = [s[len(prefix):] or s for s in stems]
    # Disambiguate identical names coming from different folders
    if len(set(stems)) < len(stems):
        stems = [f'{os.path.basename(os.path.dirname(p))}/{s}' for p, s in zip(paths, stems)]
    return stems


def _load_full(path):
    df = pd.read_excel(path, sheet_name=SHEET,
                       usecols=KEY_COLS + ['place_cell', 'speed_cell', 'n_fields_detected'])
    df['session'] = df['session'].astype(str)
    df['unit']    = df['unit'].astype(str)
    dups = df.duplicated(KEY_COLS).sum()
    if dups:
        print(f'  WARNING: {dups} duplicated (session, unit) rows in {path}; keeping first.')
        df = df.drop_duplicates(KEY_COLS)
    return df.set_index(KEY_COLS)


def _type_masks(df):
    """Boolean Series per cell type, indexed by (session, unit)."""
    place    = pd.to_numeric(df['place_cell'], errors='coerce')
    speed    = df['speed_cell'].astype(str).str.strip().str.lower()
    n_fields = pd.to_numeric(df['n_fields_detected'], errors='coerce')

    is_place    = place == 1
    is_nonplace = place == 0
    is_speed    = speed == 'true'
    return {
        'total':            pd.Series(True, index=df.index),
        'place':            is_place,
        'speed':            is_speed,
        'place_speed':      is_place & is_speed,
        'nonplace_speed':   is_nonplace & is_speed,
        'speed_not_tested': speed == 'not tested',
        'place_fields':     n_fields > 0,
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, out_dir, name):
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'), dpi=300,
                    bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def _draw_grouped(ax, type_idx, type_keys, series_labels, series_colors, values):
    """Draw grouped bars for the types at positions type_idx onto ax."""
    n_series = len(series_labels)
    x = np.arange(len(type_idx))
    width = 0.8 / n_series
    ymax = max(max(vals[t] for t in type_idx) for vals in values) or 1
    for s, (label, color, vals) in enumerate(zip(series_labels, series_colors, values)):
        pos = x - 0.4 + width * (s + 0.5)
        v_sub = [vals[t] for t in type_idx]
        bars = ax.bar(pos, v_sub, width, color=color, label=label,
                      edgecolor=SURFACE, linewidth=1.5)
        for b, v in zip(bars, v_sub):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + ymax * 0.01, f'{v}',
                    ha='center', va='bottom', fontsize=7, color=INK_SECONDARY,
                    rotation=90 if n_series > 3 else 0)
    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_SHORT[type_keys[t]] for t in type_idx], fontsize=9, color=INK_PRIMARY)
    ax.set_ylim(0, ymax * 1.18)


def _grouped_counts(series_labels, series_colors, values, title, ylabel, out_dir, name):
    """Grouped bar chart: x = cell type, one bar per series. values[s][t] = count.
    'Total units processed' gets its own panel (own y-scale) on the left so it
    does not flatten the much smaller categories."""
    type_keys = [k for k, _, _ in TYPES]
    i_total = type_keys.index('total')
    i_rest  = [i for i in range(len(type_keys)) if i != i_total]
    n_series = len(series_labels)

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(max(10, 1.6 * len(type_keys) + 0.4 * n_series), 5),
        gridspec_kw={'width_ratios': [1, len(i_rest)], 'wspace': 0.12}, facecolor=SURFACE)
    for ax in (ax0, ax1):
        _style_axes(ax)
    _draw_grouped(ax0, [i_total], type_keys, series_labels, series_colors, values)
    _draw_grouped(ax1, i_rest,    type_keys, series_labels, series_colors, values)
    ax0.set_ylabel(ylabel, color=INK_PRIMARY)
    fig.suptitle(title, color=INK_PRIMARY, fontsize=11, x=0.06, ha='left')
    ax1.legend(frameon=False, fontsize=8, labelcolor=INK_PRIMARY,
               loc='upper left', bbox_to_anchor=(1.0, 1.0))
    _save(fig, out_dir, name)


def _plot_percent(labels, colors, pct_acc, out_dir):
    """Fig 1: stacked 100 % bars (accepted = file colour, rejected = gray)."""
    type_keys = [k for k, _, _ in TYPES]
    n_files = len(labels)
    x = np.arange(len(type_keys))
    width = 0.8 / n_files

    fig, ax = plt.subplots(figsize=(max(10, 1.7 * len(type_keys)), 5.5), facecolor=SURFACE)
    _style_axes(ax)
    for f, (label, color) in enumerate(zip(labels, colors)):
        pos = x - 0.4 + width * (f + 0.5)
        acc = np.array([pct_acc[f][k] for k in type_keys])
        ax.bar(pos, acc, width, color=color, label=f'{label} (accepted)',
               edgecolor=SURFACE, linewidth=1.5)
        ax.bar(pos, 100 - acc, width, bottom=acc, color=REJECT_COLOR,
               label='Rejected' if f == 0 else None, edgecolor=SURFACE, linewidth=1.5)
        for p, a in zip(pos, acc):
            ax.text(p, min(a, 100) + 1, f'{a:.1f}', ha='center', va='bottom',
                    fontsize=6.5, color=INK_PRIMARY, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_SHORT[k] for k in type_keys], fontsize=9, color=INK_PRIMARY)
    ax.set_ylabel('% of total units', color=INK_PRIMARY)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_title('Accepted vs rejected cells per type (% of total units, Full session)',
                 color=INK_PRIMARY, fontsize=11, loc='left')
    handles, leg_labels = ax.get_legend_handles_labels()
    i_rej = leg_labels.index('Rejected')     # move 'Rejected' to the end of the legend
    order = [i for i in range(len(leg_labels)) if i != i_rej] + [i_rej]
    ax.legend([handles[i] for i in order], [leg_labels[i] for i in order],
              frameon=False, fontsize=8, labelcolor=INK_PRIMARY,
              loc='upper left', bbox_to_anchor=(1.0, 1.0))
    _save(fig, out_dir, 'Fig1_Percent_Accepted_Rejected')


# ── Main ──────────────────────────────────────────────────────────────────────

def main(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    paths = sorted(p for p in glob.glob(os.path.join(input_dir, '*.xlsx'))
                   if not os.path.basename(p).startswith('~$'))
    if len(paths) < 2:
        raise SystemExit(f'Need at least 2 Excel files in {input_dir}, found {len(paths)}.')
    os.makedirs(output_dir, exist_ok=True)

    labels = _file_labels(paths)
    cmap = matplotlib.colormaps[FILE_CMAP]
    colors = [to_hex(cmap(v))
              for v in np.linspace(0, 1, len(paths) + 2)[1:-1]]

    print(f'Comparing {len(paths)} files in {input_dir}:')
    dfs = []
    for label, p in zip(labels, paths):
        df = _load_full(p)
        print(f'  {label:<20s} {len(df):5d} units   {os.path.basename(p)}')
        dfs.append(df)

    # ── Cells not present in every file ──────────────────────────────────────

    all_keys    = sorted(set().union(*[set(df.index) for df in dfs]))
    common_keys = set.intersection(*[set(df.index) for df in dfs])
    missing_rows = []
    for key in all_keys:
        if key in common_keys:
            continue
        present = [l for l, df in zip(labels, dfs) if key in df.index]
        absent  = [l for l, df in zip(labels, dfs) if key not in df.index]
        row: dict = {'session': key[0], 'unit': key[1],
               'n_files_present': len(present),
               'present_in': ', '.join(present),
               'absent_from': ', '.join(absent)}
        row.update({f'in_{l}': key in df.index for l, df in zip(labels, dfs)})
        missing_rows.append(row)

    df_missing = pd.DataFrame(missing_rows,
                              columns=['session', 'unit', 'n_files_present', 'present_in', 'absent_from']
                                      + [f'in_{l}' for l in labels])
    df_files = pd.DataFrame({'label': labels,
                             'file_name': [os.path.basename(p) for p in paths],
                             'directory': [os.path.dirname(p) for p in paths],
                             'full_path': paths,
                             'n_units_in_file': [len(df) for df in dfs],
                             'n_units_not_in_all_files': [len(set(df.index) - common_keys) for df in dfs]})
    missing_xlsx = os.path.join(output_dir, 'Cells_Not_In_All_Files.xlsx')
    with pd.ExcelWriter(missing_xlsx, engine='openpyxl') as writer:
        df_missing.to_excel(writer, sheet_name='Cells_Not_In_All', index=False)
        df_files.to_excel(writer, sheet_name='Files', index=False)
    print(f'\nUnits common to all files : {len(common_keys)}')
    print(f'Units NOT in all files    : {len(df_missing)}  (excluded; listed in {missing_xlsx})')

    # Restrict every file to the common cells, in the same order
    common_index = pd.MultiIndex.from_tuples(sorted(common_keys), names=KEY_COLS)
    dfs = [df.loc[common_index] for df in dfs]
    masks = [_type_masks(df) for df in dfs]
    sets  = [{k: set(m[k].index[m[k]]) for k in m} for m in masks]
    type_keys = [k for k, _, _ in TYPES]
    n_total = len(common_index)

    # ── Counts and % accepted / rejected ─────────────────────────────────────

    count_rows, pct_acc = [], []
    for label, s, df in zip(labels, sets, dfs):
        pct_f = {}
        for k in type_keys:
            n = len(s[k])
            pct_f[k] = 100.0 * n / n_total if n_total else 0.0
            count_rows.append({'file': label, 'type': TYPE_NAME[k],
                               'n_accepted': n, 'n_rejected': n_total - n,
                               'pct_accepted': pct_f[k], 'pct_rejected': 100.0 - pct_f[k],
                               'denominator_total_units': n_total})
        pct_acc.append(pct_f)
        n_fields = int(pd.to_numeric(df['n_fields_detected'], errors='coerce').fillna(0).sum())
        count_rows.append({'file': label, 'type': 'Place fields detected (total no. of fields)',
                           'n_accepted': n_fields, 'n_rejected': np.nan,
                           'pct_accepted': np.nan, 'pct_rejected': np.nan,
                           'denominator_total_units': n_total})
    df_counts = pd.DataFrame(count_rows)

    df_counts_wide = (df_counts.pivot(index='type', columns='file', values='n_accepted')
                      .reindex([TYPE_NAME[k] for k in type_keys]
                               + ['Place fields detected (total no. of fields)'])[labels])

    print('\nCounts per type (common units only):')
    print(df_counts_wide.to_string())

    _plot_percent(labels, colors, pct_acc, output_dir)
    _grouped_counts(labels, colors,
                    [[len(s[k]) for k in type_keys] for s in sets],
                    'Number of accepted cells per type (Full session)',
                    'Number of cells', output_dir, 'Fig2_Count_Accepted')

    # ── Pairwise overlap ─────────────────────────────────────────────────────

    pair_rows = []
    for (i, j) in itertools.combinations(range(len(labels)), 2):
        la, lb = labels[i], labels[j]
        for k in type_keys:
            a, b = sets[i][k], sets[j][k]
            union = len(a | b)
            pair_rows.append({'file_A': la, 'file_B': lb, 'type': TYPE_NAME[k],
                              'n_A': len(a), 'n_B': len(b), 'n_common': len(a & b),
                              'n_A_only': len(a - b), 'n_B_only': len(b - a),
                              'jaccard': len(a & b) / union if union else np.nan})
        _grouped_counts([la, lb, 'Common'], [colors[i], colors[j], COMMON_COLOR],
                        [[len(sets[i][k]) for k in type_keys],
                         [len(sets[j][k]) for k in type_keys],
                         [len(sets[i][k] & sets[j][k]) for k in type_keys]],
                        f'Common cells per type: {la} vs {lb}',
                        'Number of cells', output_dir,
                        f'Fig3_Common_{la}_vs_{lb}'.replace('/', '-'))
    df_pairs = pd.DataFrame(pair_rows)

    # ── Common to all files ──────────────────────────────────────────────────

    all_rows, common_all = [], {}
    for k in type_keys:
        inter = set.intersection(*[s[k] for s in sets])
        union = set.union(*[s[k] for s in sets])
        common_all[k] = inter
        row: dict = {'type': TYPE_NAME[k]}
        row.update({f'n_{l}': len(s[k]) for l, s in zip(labels, sets)})
        row.update({'n_common_all_files': len(inter), 'n_in_any_file': len(union),
                    'jaccard_all': len(inter) / len(union) if union else np.nan})
        all_rows.append(row)
    df_all = pd.DataFrame(all_rows)

    _grouped_counts(labels + ['Common to all'], colors + [COMMON_COLOR],
                    [[len(s[k]) for k in type_keys] for s in sets]
                    + [[len(common_all[k]) for k in type_keys]],
                    f'Cells common to all {len(labels)} files per type',
                    'Number of cells', output_dir, 'Fig4_Common_All_Files')

    # ── Per-cell membership (which file accepts each cell, per type) ─────────

    member_rows = []
    for k in type_keys:
        if k == 'total':
            continue
        union = set.union(*[s[k] for s in sets])
        for key in sorted(union):
            flags = {l: int(key in s[k]) for l, s in zip(labels, sets)}
            row: dict = {'type': TYPE_NAME[k], 'session': key[0], 'unit': key[1]}
            row.update(flags)
            row.update({'n_files': sum(flags.values()),
                        'in_all_files': sum(flags.values()) == len(labels)})
            member_rows.append(row)
    df_member = pd.DataFrame(member_rows)

    summary_xlsx = os.path.join(output_dir, 'PlaceCell_CompareRuns_Summary.xlsx')
    with pd.ExcelWriter(summary_xlsx, engine='openpyxl') as writer:
        df_counts_wide.reset_index().to_excel(writer, sheet_name='Counts', index=False)
        df_counts.to_excel(writer, sheet_name='Counts_Pct_Long', index=False)
        df_pairs.to_excel(writer, sheet_name='Pairwise_Common', index=False)
        df_all.to_excel(writer, sheet_name='Common_All_Files', index=False)
        df_member.to_excel(writer, sheet_name='Cell_Membership', index=False)
        df_files.to_excel(writer, sheet_name='Files', index=False)

    print('\nCommon to all files:')
    print(df_all.to_string(index=False))
    print(f'\nDone. Figures and tables saved to {output_dir}')


if __name__ == '__main__':
    main()
