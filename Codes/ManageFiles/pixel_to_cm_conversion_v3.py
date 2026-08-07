# -*- coding: utf-8 -*-
"""
Batch-convert pixel tracking coordinates (columns A/B/C = timestamp/x/y) to cm.

Before the pixel-to-cm conversion, tracked points are first cleaned of
jump artifacts (e.g. the tracker briefly latching onto a point outside the
arena) in two passes:
  1. Speed threshold: any frame reached via a frame-to-frame speed above
     MAX_SPEED_CM_PER_S is flagged.
  2. Hampel filter: run per-axis on the speed-cleaned trace to catch
     remaining jumps that don't show up as a single implausible-speed step
     (e.g. a drift to an outlier point spread over a couple of frames) —
     a point is flagged if it deviates from its local rolling median by
     more than HAMPEL_N_SIGMAS robust standard deviations (1.4826 * MAD).
Flagged frames from either pass are linearly interpolated from neighboring
valid frames.

For each subfolder of ROOT_DIR containing one .xlsx tracking file:
    x_dist_px = max(x) - min(x)
    y_dist_px = max(y) - min(y)
    scale_cm_per_px = arena_size_cm / max(x_dist_px, y_dist_px)
    x_cm = x_px * scale_cm_per_px
    y_cm = y_px * scale_cm_per_px

arena_size_cm is the real-world length of the longer axis of the tracked
area (e.g. arena diameter/side). It differs between sessions (60 cm vs
80 cm in some cases), so it is configurable per-folder below.
"""

import os
import numpy as np
import pandas as pd

# ── USER INPUT ──────────────────────────────────────────────────────────────
ROOT_DIR = r'X:/NMR_group_data/Runita/Data/Ephys_Data/AllSortedData/Tetrode/Fa1059/Open'  # root folder containing subfolders with .xlsx tracking files
# Real-world size (cm) of the longer tracked axis, used as the default for
# any subfolder not listed in ARENA_SIZE_CM_BY_FOLDER below.
DEFAULT_ARENA_SIZE_CM = 60

# Per-folder overrides: {subfolder_name: arena_size_cm}
# Add an entry here for any session whose arena size differs from the default
# (e.g. 80 cm arenas).
ARENA_SIZE_CM_BY_FOLDER = {
    # 'ExperimentDay10_NestBuild': 80,
}

FRAME_RATE_HZ = 30  # tracking camera frame rate
MAX_SPEED_CM_PER_S = 80  # frame-to-frame speed above this is treated as a jump artifact

HAMPEL_WINDOW = 5  # points on each side of the rolling window (window length = 2*HAMPEL_WINDOW+1)
HAMPEL_N_SIGMAS = 3  # flag points more than this many robust std devs from the local median
HAMPEL_MIN_THRESHOLD_PX = 2.0  # floor on the deviation threshold, guards against near-zero local MAD

OUTPUT_SUFFIX = '_cm'  # output saved as <original_name>_cm.csv in the same subfolder
# ─────────────────────────────────────────────────────────────────────────────


def get_arena_size_cm(folder_name):
    return ARENA_SIZE_CM_BY_FOLDER.get(folder_name, DEFAULT_ARENA_SIZE_CM)


def remove_speed_jumps(x_px, y_px, arena_size_cm, frame_rate_hz, max_speed_cm_per_s):
    """Flag frames reached at an implausible speed and linearly interpolate them.

    The speed threshold is defined in cm/s, so it needs a pixel scale to be
    expressed in pixels/frame. A rough scale from the raw (uncleaned) min/max
    range is good enough for this: real jumps are large enough that they are
    still caught even though the jump itself inflates that raw range.
    """
    rough_range_px = max(x_px.max() - x_px.min(), y_px.max() - y_px.min())
    rough_scale_cm_per_px = arena_size_cm / rough_range_px
    max_px_per_frame = (max_speed_cm_per_s / frame_rate_hz) / rough_scale_cm_per_px

    dx = x_px.diff()
    dy = y_px.diff()
    step_px = np.sqrt(dx**2 + dy**2)  # step_px[i] = distance from frame i-1 to i

    fast_step = step_px > max_px_per_frame
    is_jump = fast_step.fillna(False) | fast_step.shift(-1, fill_value=False)

    x_clean = x_px.where(~is_jump)
    y_clean = y_px.where(~is_jump)
    x_clean = x_clean.interpolate(method='linear', limit_direction='both')
    y_clean = y_clean.interpolate(method='linear', limit_direction='both')

    return x_clean, y_clean, int(is_jump.sum())


def hampel_mask(series, window=5, n_sigmas=3, min_threshold=2.0):
    """Flag points that deviate from their local rolling median by more than
    n_sigmas robust standard deviations (1.4826 * MAD), Hampel-identifier style.

    Unlike a step-to-step speed check, this compares each point against a
    window of neighbors, so it also catches jumps that drift to an outlier
    location over a couple of frames rather than a single implausible step.
    """
    win_len = 2 * window + 1
    rolling_median = series.rolling(window=win_len, center=True, min_periods=1).median()
    abs_dev = (series - rolling_median).abs()
    mad = abs_dev.rolling(window=win_len, center=True, min_periods=1).median()
    threshold = np.maximum(1.4826 * n_sigmas * mad, min_threshold)
    return abs_dev > threshold


def remove_jumps(x_px, y_px, arena_size_cm, frame_rate_hz, max_speed_cm_per_s,
                  hampel_window, hampel_n_sigmas, hampel_min_threshold_px):
    x_px, y_px, n_speed = remove_speed_jumps(
        x_px, y_px, arena_size_cm, frame_rate_hz, max_speed_cm_per_s
    )

    is_hampel_jump = (
        hampel_mask(x_px, hampel_window, hampel_n_sigmas, hampel_min_threshold_px)
        | hampel_mask(y_px, hampel_window, hampel_n_sigmas, hampel_min_threshold_px)
    )
    x_clean = x_px.where(~is_hampel_jump).interpolate(method='linear', limit_direction='both')
    y_clean = y_px.where(~is_hampel_jump).interpolate(method='linear', limit_direction='both')

    return x_clean, y_clean, n_speed, int(is_hampel_jump.sum())


def convert_file(xlsx_path, arena_size_cm):
    df = pd.read_excel(xlsx_path, header=0)

    t_col, x_col, y_col = df.columns[0], df.columns[1], df.columns[2]
    x_px = df[x_col].astype(float)
    y_px = df[y_col].astype(float)

    x_px, y_px, n_speed_jumps, n_hampel_jumps = remove_jumps(
        x_px, y_px, arena_size_cm, FRAME_RATE_HZ, MAX_SPEED_CM_PER_S,
        HAMPEL_WINDOW, HAMPEL_N_SIGMAS, HAMPEL_MIN_THRESHOLD_PX
    )

    x_dist_px = x_px.max() - x_px.min()
    y_dist_px = y_px.max() - y_px.min()
    longer_dist_px = max(x_dist_px, y_dist_px)

    scale_cm_per_px = arena_size_cm / longer_dist_px

    x_cm = x_px * scale_cm_per_px
    y_cm = y_px * scale_cm_per_px
    df['x_cm'] = x_cm - x_cm.min()
    df['y_cm'] = y_cm - y_cm.min()

    out_path = os.path.join(
        os.path.dirname(xlsx_path),
        os.path.splitext(os.path.basename(xlsx_path))[0] + OUTPUT_SUFFIX + '.csv'
    )
    df.to_csv(out_path, index=False)

    print(f"  speed_jumps={n_speed_jumps}, hampel_jumps={n_hampel_jumps}, "
          f"x_dist_px={x_dist_px:.2f}, y_dist_px={y_dist_px:.2f}, "
          f"arena_size_cm={arena_size_cm}, scale={scale_cm_per_px:.5f} cm/px")
    print(f"  Saved -> {out_path}")


def main():
    for folder, _dirnames, filenames in os.walk(ROOT_DIR):
        folder_name = os.path.basename(folder)
        xlsx_files = sorted(
            os.path.join(folder, f) for f in filenames
            if f.lower().endswith('.xlsx') and not f.endswith(OUTPUT_SUFFIX + '.xlsx')
        )

        if not xlsx_files:
            continue
        if len(xlsx_files) > 1:
            print(f"[{folder_name}] Multiple .xlsx files found, using the first: {xlsx_files[0]}")

        arena_size_cm = get_arena_size_cm(folder_name)
        print(f"[{folder_name}] Converting {os.path.basename(xlsx_files[0])} "
              f"(arena_size_cm={arena_size_cm})")
        convert_file(xlsx_files[0], arena_size_cm)


if __name__ == '__main__':
    main()
