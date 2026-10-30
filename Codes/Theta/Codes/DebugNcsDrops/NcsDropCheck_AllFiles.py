# -*- coding: utf-8 -*-
"""
Check every .ncs file under ROOT_DIR for incomplete records and missing samples.

For each file this measures
  - partial records : records with NumValidSamples < 512
  - invalid slots   : the stale samples that fill out those partial records
  - missing samples : slots on the timestamp-anchored 32 kHz grid with no real sample
                      (see ncs_repair.py); this is what the repair interpolates
  - raw trace error : how far `np.concatenate(data['samples'])` has run ahead of the
                      timestamps by the end of the file

Writes one row per file to ncs_drop_check.csv, and one plot per animal of the missing
sample rate of each file.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ncs_repair import (MAX_GAP_S, NATIVE_FS, SAMPLES_PER_RECORD, find_runs, grid_layout,
                        header_value, open_ncs, read_header)

# %% ==================== Parameters ==========================================

ROOT_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Alpha200'
OUT_DIR = os.path.join(ROOT_DIR, 'Results', 'NcsDropCheck')
N_WORKERS = 6

# Per-animal colors — first 4 colors of matplotlib's Pastel1 palette.
ANIMAL_COLORS = {
    'Fa8477': '#FBB4AE',  # Pastel1[0] red
    'Fa1059': '#B3CDE3',  # Pastel1[1] blue
    'Fa23BD': '#CCEBC5',  # Pastel1[2] green
    'Fa5834': '#DECBE4',  # Pastel1[3] purple
}

# %% ==================== Per-file check ======================================


def check_file(fpath):
    """Return one row of drop metrics for an .ncs file."""
    rel = os.path.relpath(fpath, ROOT_DIR)
    parts = rel.split(os.sep)
    row = {'animal': parts[0], 'session': os.sep.join(parts[1:-1]), 'file': parts[-1], 'rel_path': rel,
           'size_mod_record': (os.path.getsize(fpath) - 16 * 1024) % 1044}

    header = read_header(fpath)
    row['fs_header'] = float(header_value(header, 'SamplingFrequency'))
    row['ad_bit_volts'] = float(header_value(header, 'ADBitVolts'))
    row['ad_channel'] = header_value(header, 'ADChannel')

    data = open_ncs(fpath)
    ts = np.asarray(data['timestamp']).astype(np.int64)
    nv = np.asarray(data['n_valid']).astype(np.int64)
    row['fs_records'] = ','.join(str(v) for v in np.unique(data['fs']))
    row['n_ts_backward'] = int((np.diff(ts) < 0).sum())

    # Incomplete records, and whether their tails really are stale: each slot past
    # NumValidSamples should hold the previous record's value at that position.
    partial = np.flatnonzero(nv < SAMPLES_PER_RECORD)
    n_stale = sum(i > 0 and np.array_equal(data['samples'][i][nv[i]:], data['samples'][i - 1][nv[i]:])
                  for i in partial)
    n_invalid = int((SAMPLES_PER_RECORD - nv).sum())

    # Missing samples on the timestamp-anchored grid
    order = np.argsort(ts, kind='stable')
    sample_idx, n_grid = grid_layout(ts[order], nv[order], NATIVE_FS)
    covered = np.zeros(n_grid, dtype=bool)
    covered[sample_idx] = True
    n_placed = int(covered.sum())
    _, run_lengths = find_runs(~covered)
    max_gap_samples = int(round(MAX_GAP_S * NATIVE_FS))

    row.update({
        'n_records': len(data),
        'n_partial_records': len(partial),
        'pct_partial_records': 100 * len(partial) / len(data),
        'n_partial_tails_stale': int(n_stale),
        'n_invalid_slots': n_invalid,
        'pct_invalid_slots': 100 * n_invalid / (len(data) * SAMPLES_PER_RECORD),
        'n_grid': n_grid,
        'duration_s': n_grid / NATIVE_FS,
        'n_valid': int(nv.sum()),
        'n_missing': n_grid - n_placed,
        'missing_pct': 100 * (n_grid - n_placed) / n_grid,
        'n_overlap': int(nv.sum()) - n_placed,   # real samples sharing a slot with a later record
        'net_lost_pct': 100 * (n_grid - int(nv.sum())) / n_grid,
        'n_gaps': len(run_lengths),
        'median_gap_ms': float(np.median(run_lengths) / NATIVE_FS * 1e3) if len(run_lengths) else 0.0,
        'longest_gap_ms': float(run_lengths.max() / NATIVE_FS * 1e3) if len(run_lengths) else 0.0,
        'n_gaps_over_limit': int((run_lengths > max_gap_samples).sum()),
        'n_samples_over_limit': int(run_lengths[run_lengths > max_gap_samples].sum()),
        'raw_trace_error_s': (len(data) * SAMPLES_PER_RECORD - n_grid) / NATIVE_FS,
    })
    return row


# %% ==================== Per-animal plot =====================================


def plot_animal(animal, table, y_max, out_dir):
    """One bar per .ncs file, in the animal's color, on a y-axis shared across animals."""
    table = table.sort_values(['session', 'file']).reset_index(drop=True)
    labels = table['session'].str.replace(os.sep, ' / ', regex=False) + ' / ' + table['file'].str.replace('.ncs', '', regex=False)

    fig, ax = plt.subplots(figsize=(max(7.0, 0.15 * len(table) + 1.5), 6.0))
    ax.bar(np.arange(len(table)), table['missing_pct'], width=0.8,
           color=ANIMAL_COLORS[animal], edgecolor='#555555', linewidth=0.5)

    # Files of one session share the same timing, so mark where sessions change
    session_changes = np.flatnonzero(table['session'].values[1:] != table['session'].values[:-1]) + 0.5
    for x in session_changes:
        ax.axvline(x, color='#DDDDDD', linewidth=0.6, zorder=0)

    ax.set_xlim(-0.75, len(table) - 0.25)
    ax.set_ylim(0, y_max)
    ax.set_xticks(np.arange(len(table)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel('Missing samples (% of session)')
    ax.set_title(f'{animal}: missing sample rate per LFP file (n = {len(table)} files, '
                 f'{table["session"].nunique()} sessions)', loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#EEEEEE', linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()

    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(out_dir, f'missing_sample_rate_{animal}.{ext}'), dpi=300)
    plt.close(fig)


# %% ==================== Main ================================================

if __name__ == '__main__':
    ncs_files = sorted(
        os.path.join(root, fname)
        for root, _, fnames in os.walk(ROOT_DIR)
        for fname in fnames if fname.endswith('.ncs')
    )
    print(f'Found {len(ncs_files)} .ncs files')

    with ThreadPoolExecutor(N_WORKERS) as pool:
        rows = []
        for i, row in enumerate(pool.map(check_file, ncs_files), 1):
            rows.append(row)
            if i % 25 == 0:
                print(f'  checked {i}/{len(ncs_files)}', flush=True)

    summary = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    summary.to_csv(os.path.join(OUT_DIR, 'ncs_drop_check.csv'), index=False)

    y_max = summary['missing_pct'].max() * 1.05
    for animal, table in summary.groupby('animal'):
        plot_animal(animal, table, y_max, OUT_DIR)
    print(f'Saved results -> {OUT_DIR}')
