# -*- coding: utf-8 -*-
"""
Trim tracking .csv files to the time span of the LFP (.ncs) recording.

Walks ROOT_DIR (all subfolders). In every folder that has both .ncs and .csv
files:
    1. Takes the first .ncs file (alphabetical order).
    2. Reads its first and last record timestamps (Neuralynx, microseconds).
    3. In each .csv (timestamps in column A) finds
           start row = the closest timestamp BELOW the ncs start  (Ts csv < Ts ncs)
           end row   = the closest timestamp ABOVE the ncs end    (Ts csv > Ts ncs)
    4. Deletes every row above the start row and below the end row.

The original .csv is kept as <name>.csv.bak (skipped if it already exists, so
re-running never overwrites the true original).
"""

import os
import shutil
import numpy as np
import pandas as pd

# ── USER INPUT ──────────────────────────────────────────────────────────────
ROOT_DIR = r'X:/NMR_group_data/Runita/Analysis/Thesis/Data_v2_Accepted'
MAKE_BACKUP = False   # keep the untrimmed csv as <name>.csv.bak
DRY_RUN = False      # True: only report what would be cut, write nothing
# ─────────────────────────────────────────────────────────────────────────────

NCS_HEADER_BYTES = 16 * 1024
NCS_DTYPE = np.dtype([
    ('timestamp',   '<u8'),
    ('sc_number',   '<u4'),
    ('cell_number', '<u4'),
    ('params',      '<u4'),
    ('samples',     '<i2', (512,)),
])


def ncs_first_last_timestamps(ncs_path):
    """First and last record timestamps (us) of a Neuralynx .ncs file."""
    data = np.memmap(ncs_path, dtype=NCS_DTYPE, mode='r', offset=NCS_HEADER_BYTES)
    if len(data) == 0:
        raise ValueError('no records in file')
    return int(data['timestamp'][0]), int(data['timestamp'][-1])


def csv_scale_to_ncs(csv_t, ncs_ref):
    """Multiplier that puts csv timestamps on the ncs clock (us).

    Auto-detected from orders of magnitude, since tracking files may be in
    us, msec or sec while .ncs is always us.
    """
    ref = np.nanmedian(np.abs(csv_t))
    if ref == 0 or not np.isfinite(ref):
        return 1.0
    return 10.0 ** np.round(np.log10(ncs_ref / ref))


def trim_csv(csv_path, ncs_start, ncs_end):
    df = pd.read_csv(csv_path)
    t = pd.to_numeric(df.iloc[:, 0], errors='coerce').to_numpy(dtype=float)
    finite = np.isfinite(t)
    if not finite.any():
        return 'skipped: no numeric timestamps in column A'

    scale = csv_scale_to_ncs(t[finite], ncs_start)
    t_us = t * scale

    below = np.flatnonzero(finite & (t_us < ncs_start))
    above = np.flatnonzero(finite & (t_us > ncs_end))
    if below.size == 0:
        return 'skipped: no csv timestamp below the ncs start'
    if above.size == 0:
        return 'skipped: no csv timestamp above the ncs end'

    i_start = below[np.argmax(t_us[below])]   # closest lower
    i_end = above[np.argmin(t_us[above])]     # closest higher

    msg = (f'kept rows {i_start}..{i_end} of {len(df)} '
           f'(removed {i_start} above, {len(df) - 1 - i_end} below; csv x{scale:g} -> us)')
    if DRY_RUN:
        return 'dry run: ' + msg

    if MAKE_BACKUP:
        bak = csv_path + '.bak'
        if not os.path.exists(bak):
            shutil.copy2(csv_path, bak)
    df.iloc[i_start:i_end + 1].to_csv(csv_path, index=False)
    return msg


def main():
    n_folders = n_matched = n_no_ncs = n_csv = 0
    for folder, _dirs, files in os.walk(ROOT_DIR):   # every folder, at any depth
        n_folders += 1
        ncs_files = sorted(f for f in files if f.lower().endswith('.ncs'))
        csv_files = sorted(f for f in files if f.lower().endswith('.csv'))
        if not csv_files:
            continue
        if not ncs_files:
            n_no_ncs += 1
            print(f'[{folder}] has {len(csv_files)} .csv but no .ncs -- skipped', flush=True)
            continue

        try:
            ncs_start, ncs_end = ncs_first_last_timestamps(os.path.join(folder, ncs_files[0]))
        except Exception as e:
            print(f'[{folder}] cannot read {ncs_files[0]}: {e}', flush=True)
            continue

        n_matched += 1
        print(f'[{folder}] {ncs_files[0]}: start={ncs_start}, end={ncs_end}', flush=True)
        for c in csv_files:
            n_csv += 1
            try:
                print(f'    {c}: {trim_csv(os.path.join(folder, c), ncs_start, ncs_end)}', flush=True)
            except Exception as e:
                print(f'    {c}: ERROR {e}', flush=True)

    print(f'\nDone: {n_folders} folders scanned, {n_matched} with matching .csv + .ncs '
          f'({n_csv} .csv processed), {n_no_ncs} with .csv but no .ncs', flush=True)


if __name__ == '__main__':
    main()
