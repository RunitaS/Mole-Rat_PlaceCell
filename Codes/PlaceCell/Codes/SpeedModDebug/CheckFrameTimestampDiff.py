# -*- coding: utf-8 -*-
"""
Frame-to-frame timestamp difference check.

Walks ROOT_FOLDER recursively, finds every .csv file whose column A holds
per-frame timestamps, computes consecutive timestamp differences (n+1 - n),
and plots them against the expected inter-frame interval for a 30 fps
tracking video (1000/30 = 33.333 ms).

For each .csv file found, writes into the SAME folder as that .csv:
    <stem>_frameTimeDiff.png   - plot: black = measured dt, thick line = expected dt
    <stem>_frameTimeDiff.xlsx  - per-frame dt values + summary stats

Parameters to edit:
    ROOT_FOLDER  = r'F:/Check'
    CSV_PATTERN  = '*.csv'      # glob pattern used to pick csv files
    FPS          = 30           # tracking frame rate (Hz)
"""

import os
import glob

import numpy as np
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

ROOT_FOLDER = r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True/Fa1059/Day9'
CSV_PATTERN = '*.csv'
FPS = 30
EXPECTED_DT_MS = 1000.0 / FPS

# candidate raw-timestamp units and the factor to convert a raw diff to ms
_UNIT_TO_MS = {
    's':  1000.0,
    'ms': 1.0,
    'us': 1.0 / 1000.0,
    'ns': 1.0 / 1_000_000.0,
}


def _infer_unit_scale(raw_diffs: np.ndarray) -> tuple[str, float]:
    """Pick the timestamp unit whose converted median dt is closest to
    EXPECTED_DT_MS (in log-ratio terms), and return (unit_name, ms_scale)."""
    median_raw = np.median(raw_diffs)
    best_unit, best_scale, best_err = None, None, np.inf
    for unit, scale in _UNIT_TO_MS.items():
        converted = median_raw * scale
        if converted <= 0:
            continue
        err = abs(np.log(converted / EXPECTED_DT_MS))
        if err < best_err:
            best_unit, best_scale, best_err = unit, scale, err
    return best_unit, best_scale


def process_csv(csv_path: str) -> None:
    data = pd.read_csv(csv_path)
    t_raw = np.asarray(data.iloc[:, 0], dtype=float)
    t_raw = t_raw[~np.isnan(t_raw)]

    if t_raw.size < 2:
        print(f'  [skip] fewer than 2 valid timestamps: {csv_path}')
        return

    raw_diffs = np.diff(t_raw)
    valid = raw_diffs > 0
    if not np.any(valid):
        print(f'  [skip] no positive timestamp differences: {csv_path}')
        return

    unit, ms_scale = _infer_unit_scale(raw_diffs[valid])
    dt_ms = raw_diffs * ms_scale
    t_s = (t_raw[1:] - t_raw[0]) * ms_scale / 1000.0

    stem = os.path.splitext(csv_path)[0]
    png_path = stem + '_frameTimeDiff.png'
    xlsx_path = stem + '_frameTimeDiff.xlsx'

    # ---- plot ----
    fig = Figure(figsize=(10, 4), dpi=150)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    ax.axhline(EXPECTED_DT_MS, color='tab:red', linewidth=3.0,
               label=f'Expected ({EXPECTED_DT_MS:.3f} ms @ {FPS} fps)')
    ax.plot(t_s, dt_ms, color='black', linewidth=0.7,
            label='Measured frame dt')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frame-to-frame dt (ms)')
    ax.set_title(os.path.basename(csv_path) + f'  (inferred unit: {unit})')
    ax.legend(loc='upper right')
    fig.tight_layout()
    canvas.print_png(png_path)

    # ---- excel ----
    out_df = pd.DataFrame({
        'FrameIndex_n+1':      np.arange(1, t_raw.size),
        'Timestamp_raw_col_A': t_raw[1:],
        'Time_s':              t_s,
        'Diff_ms':             dt_ms,
        'Expected_diff_ms':    EXPECTED_DT_MS,
        'Deviation_ms':        dt_ms - EXPECTED_DT_MS,
    })

    summary_df = pd.DataFrame({
        'Metric': ['inferred_unit', 'n_frames', 'n_diffs',
                   'mean_diff_ms', 'median_diff_ms', 'std_diff_ms',
                   'min_diff_ms', 'max_diff_ms', 'expected_diff_ms'],
        'Value': [unit, t_raw.size, dt_ms.size,
                  np.mean(dt_ms), np.median(dt_ms), np.std(dt_ms),
                  np.min(dt_ms), np.max(dt_ms), EXPECTED_DT_MS],
    })

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        out_df.to_excel(writer, sheet_name='FrameDiffs', index=False)

    print(f'  [ok] {csv_path}  -> unit={unit}, '
          f'mean={np.mean(dt_ms):.3f} ms, std={np.std(dt_ms):.3f} ms')


def main(root_folder: str = ROOT_FOLDER, pattern: str = CSV_PATTERN) -> None:
    csv_paths = []
    for dirpath, _, _ in os.walk(root_folder):
        csv_paths.extend(glob.glob(os.path.join(dirpath, pattern)))

    print(f'Found {len(csv_paths)} csv file(s) under {root_folder}')
    for csv_path in csv_paths:
        try:
            process_csv(csv_path)
        except Exception as exc:
            print(f'  [error] {csv_path}: {exc}')


if __name__ == '__main__':
    main()
