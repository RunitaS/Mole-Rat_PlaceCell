# -*- coding: utf-8 -*-
"""
Trim tracking .csv files to the time window covered by each session's .nev
event file.

For every subfolder of ROOT_DIR that contains both a .nev file and a .csv
file:
  1. Read the .nev file's event records (16 KB Neuralynx header + fixed
     binary records, same layout as cut_neuralynx_sessions.py). The first
     record's TimeStamp is treated as the session start, the last record's
     TimeStamp as the session stop (Unix epoch, microseconds).
  2. Read the .csv file's timestamps from column A -- also Unix epoch
     microseconds, so directly comparable to the .nev timestamps
     (CSV_TIMESTAMP_TO_US_FACTOR below stays 1; change it only if a csv
     ever shows up in a different unit, e.g. 1000 for milliseconds).
  3. Find the csv row whose timestamp is closest to the .nev start
     timestamp, and the row closest to the .nev stop timestamp. Each match
     must land within MATCH_TOLERANCE_MS milliseconds of its target -- if
     either match misses that tolerance, the folder is skipped and a
     warning is printed.
  4. Keep only the csv rows between the matched start and stop rows
     (inclusive), dropping every row above and below, and overwrite the
     original .csv file with the trimmed result.

Requires: numpy, pandas
"""

import os
import numpy as np
import pandas as pd

# ── USER INPUT ──────────────────────────────────────────────────────────────
ROOT_DIR = r'X:/NMR_group_data/Runita/Data/Ephys_Data/AllSortedData/Tetrode'  # root folder containing subfolders with .nev + .csv files

MATCH_TOLERANCE_MS = 500  # max allowed difference between a .nev timestamp and its matched csv row
# (camera start/stop typically lags the .nev event by up to a few hundred ms
# even on a synced clock, so 50 ms was too tight and skipped valid folders)

# Multiply csv column-A timestamps by this factor to put them in Unix epoch
# microseconds, matching the .nev file's TimeStamp unit. Both are already
# Unix epoch microseconds, so this stays 1.
CSV_TIMESTAMP_TO_US_FACTOR = 1
# ─────────────────────────────────────────────────────────────────────────────

HEADER_SIZE = 16384  # standard Neuralynx text header size, all file types


def nev_dtype() -> np.dtype:
    return np.dtype([
        ("nstx", "<i2"),
        ("npkt_id", "<i2"),
        ("npkt_data_size", "<i2"),
        ("TimeStamp", "<u8"),
        ("nevent_id", "<i2"),
        ("nttl", "<i2"),
        ("ncrc", "<i2"),
        ("ndummy1", "<i2"),
        ("ndummy2", "<i2"),
        ("dnExtra", "<i4", (8,)),
        ("EventString", "S128"),
    ])


def read_nev_start_stop(nev_path):
    """Return (start_ts, stop_ts) in microseconds: the first and last event
    TimeStamp in the .nev file, or None if the file has no event records."""
    records = np.fromfile(nev_path, dtype=nev_dtype(), offset=HEADER_SIZE)
    if len(records) == 0:
        return None
    timestamps = records["TimeStamp"]
    return int(timestamps[0]), int(timestamps[-1])


def find_closest_row(csv_timestamps_us, target_us, tolerance_us):
    """Return (row_label, diff_us) for the csv row closest to target_us, or
    (None, diff_us) if that closest row still falls outside tolerance_us."""
    diffs = (csv_timestamps_us - target_us).abs()
    idx = diffs.idxmin()
    diff = diffs.loc[idx]
    if diff > tolerance_us:
        return None, diff
    return idx, diff


def trim_csv_to_nev_window(csv_path, nev_path, tolerance_ms, csv_to_us_factor):
    start_stop = read_nev_start_stop(nev_path)
    if start_stop is None:
        print(f"    No event records in {os.path.basename(nev_path)}, skipping.")
        return
    start_ts, stop_ts = start_stop

    df = pd.read_csv(csv_path)
    csv_ts_col = df.columns[0]
    csv_timestamps_us = df[csv_ts_col].astype(float) * csv_to_us_factor

    tolerance_us = tolerance_ms * 1000

    start_idx, start_diff = find_closest_row(csv_timestamps_us, start_ts, tolerance_us)
    stop_idx, stop_diff = find_closest_row(csv_timestamps_us, stop_ts, tolerance_us)

    if start_idx is None or stop_idx is None:
        print(f"    No csv row within {tolerance_ms} ms of nev start/stop "
              f"(closest diffs: start={start_diff / 1000:.2f} ms, "
              f"stop={stop_diff / 1000:.2f} ms), skipping.")
        return

    lo, hi = sorted((start_idx, stop_idx))
    trimmed = df.loc[lo:hi].reset_index(drop=True)

    trimmed.to_csv(csv_path, index=False)

    print(f"    nev window: start={start_ts} stop={stop_ts}")
    print(f"    matched csv rows: start_idx={start_idx} (diff={start_diff / 1000:.2f} ms), "
          f"stop_idx={stop_idx} (diff={stop_diff / 1000:.2f} ms)")
    print(f"    kept {len(trimmed)}/{len(df)} rows -> overwrote {csv_path}")


def main():
    for folder, _dirnames, filenames in os.walk(ROOT_DIR):
        nev_files = sorted(f for f in filenames if f.lower().endswith('.nev'))
        csv_files = sorted(f for f in filenames if f.lower().endswith('.csv'))

        if not nev_files or not csv_files:
            continue

        if len(nev_files) > 1:
            print(f"[{folder}] Multiple .nev files found, using the first: {nev_files[0]}")
        if len(csv_files) > 1:
            print(f"[{folder}] Multiple .csv files found, using the first: {csv_files[0]}")

        print(f"[{folder}] Trimming {csv_files[0]} to {nev_files[0]} window")
        trim_csv_to_nev_window(
            os.path.join(folder, csv_files[0]),
            os.path.join(folder, nev_files[0]),
            MATCH_TOLERANCE_MS,
            CSV_TIMESTAMP_TO_US_FACTOR,
        )


if __name__ == '__main__':
    main()
