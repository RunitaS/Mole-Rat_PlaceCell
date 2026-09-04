# -*- coding: utf-8 -*-
"""
Flag mislabeled tracking points that fall outside the circular arena
(e.g. the experimenter got tracked instead of the animal), and mark the
corresponding frames on the paired video with a red circle.

Expects, per .csv tracking file, a same-named video file in the same folder
(e.g. Session1.csv + Session1.mp4). Walks ROOT_DIR recursively.

CSV layout: raw DeepLabCut output, NOT the flat layout used by
batch_occupancy_maps.py. A 3-row header (scorer / bodyparts / coords) is
followed by one frame-index-labelled data row per frame, with an
(x, y, likelihood) triplet per tracked bodypart. This script reads the four
head-mounted LED markers (LED_BODYPART_SCHEMES below -- the first scheme
fully present in a given file's columns is used) and collapses them into one
head position per frame -- the mean (x, y) of whichever LEDs cleared
LIKELIHOOD_THRESHOLD that frame (a frame is dropped if none did). Frame
index is converted to a msec timestamp using the paired video's fps.

How a point is flagged:
    The arena is a circle, auto-detected directly in the video's pixel space
    (Hough transform on a representative frame); if that fails, the arena is
    assumed to fill ARENA_FILL_FRACTION of the frame's shorter side, centered
    in the frame. Since the tracking x/y are already in that same pixel
    space, no unit conversion or center estimate from the data itself is
    needed -- a point is simply flagged if it falls farther than the
    detected radius (+ MARGIN_PX) from the detected center.

Drawing flagged points on the video:
    Because the tracking data and the video share one pixel coordinate
    system, points are drawn directly at their (x, y) pixel location -- no
    mapping/scaling step.
    A *_calibration_check.png is saved for each file -- check that the green
    circle actually traces the arena edge, and that the small dots
    (blue = kept, red = flagged) land on the animal's real path, before
    trusting the annotated video / timestamp list.

Outputs, per input file, saved to OUTPUT_DIR:
    <name>_flagged_timestamps.csv  -- frame_idx, timestamp_sec, x_px, y_px,
                                       dist_from_center_px for every flagged point
    <name>_annotated.mp4           -- full video, red circle drawn on flagged frames
    <name>_calibration_check.png   -- one frame showing the detected arena
                                       circle + all points, for a sanity check
    <name>_occupancy_map.png       -- occupancy heatmap (as in batch_occupancy_maps.py)
                                       with the detected arena circle and every
                                       flagged out-of-bounds point overlaid, so
                                       flagged points can be checked in context
    <name>_corrected_tracking.csv  -- frame_idx, timestamp_sec, x_px, y_px,
                                       was_interpolated for every tracked frame;
                                       every flagged out-of-arena point is deleted
                                       and linearly interpolated from its nearest
                                       surrounding in-arena points
Plus ALL_flagged_timestamps.csv, all files' flagged points concatenated.

Rerun behavior: if a file's flagging outputs (flagged_timestamps.csv,
annotated.mp4, or calibration_check.png) already exist in OUTPUT_DIR, the
(slow, video-based) flagging step is skipped for that file on subsequent
runs. The occupancy map and the corrected-tracking csv, which only need the
.csv (+ video, to locate the arena circle), are always (re)generated.

Requires opencv-python (`pip install opencv-python`) in addition to the
usual pandas/numpy/matplotlib/scipy.
"""

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe for batch/headless runs
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter

# ── USER INPUT ──────────────────────────────────────────────────────────────
ROOT_DIR = r'X:/NMR_group_data/Runita/Temp/OccCOrrection/FlagIncorr'   # searched recursively for .csv files
OUTPUT_DIR = r'X:/NMR_group_data/Runita/Temp/OccCOrrection/FlagIncorr'                     # annotated videos / reports / calibration images go here

VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']  # tried, in order, next to each .csv

MARGIN_PX = 0.0             # extra tolerance (pixels) added to the detected radius before a point counts as "outside"
FLIP_Y = False               # set True if the tracking y-axis increases opposite to the video's pixel rows

# DLC bodypart names averaged into one head position per frame. Different DLC
# projects/models label the head LEDs differently -- each inner list is one
# known naming scheme; the first one fully present in a given CSV's columns
# is used for that file.
LED_BODYPART_SCHEMES = [
    ['LED_N', 'LED_S', 'LED_W', 'LED_E'],
    ['LEDbig', 'LED1', 'LED2', 'LED3'],
]
LIKELIHOOD_THRESHOLD = 0.6  # DLC bodyparts below this confidence are excluded from that frame's average

# Pixel-mapping assumptions -- verify against the *_calibration_check.png
ARENA_FILL_FRACTION = 0.95  # fallback only, if the arena circle can't be auto-detected in the video

MARK_RADIUS_PX = 18
MARK_THICKNESS = 4
MARK_COLOR_BGR = (0, 0, 255)  # red

MAKE_CALIBRATION_IMAGE = True

# Occupancy-map overlay (lets flagged points be checked against the animal's path)
OCC_N_BINS_ACROSS_DIAMETER = 30  # bin size is derived from the detected arena diameter, so this stays meaningful across videos of different resolution
OCC_SMOOTHING_SIGMA = 1.5
OCC_DPI = 200
# ─────────────────────────────────────────────────────────────────────────────


def find_csv_files(root_dir):
    csv_paths = []
    for folder, _dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith('.csv'):
                csv_paths.append(os.path.join(folder, f))
    return sorted(csv_paths)


def find_matching_video(csv_path):
    base = os.path.splitext(csv_path)[0]
    for ext in VIDEO_EXTENSIONS:
        for candidate in (base + ext, base + ext.upper()):
            if os.path.exists(candidate):
                return candidate
    return None


def pick_led_bodyparts(df_columns):
    """Returns the first LED_BODYPART_SCHEMES entry whose bodyparts are all
    present in df_columns (a DLC (bodypart, coord) column index), or raises
    ValueError if none match."""
    available = set(bp for bp, _coord in df_columns)
    for scheme in LED_BODYPART_SCHEMES:
        if all(bp in available for bp in scheme):
            return scheme
    raise ValueError(
        f"none of the known LED_BODYPART_SCHEMES {LED_BODYPART_SCHEMES} "
        f"are fully present in this file's bodyparts {sorted(available)}"
    )


def load_tracking(csv_path, fps):
    """Loads a raw DeepLabCut tracking csv (3-row header: scorer / bodyparts /
    coords) and collapses the LED bodyparts (whichever LED_BODYPART_SCHEMES
    entry matches this file) into one head position per frame -- the mean
    (x, y) of whichever LEDs cleared LIKELIHOOD_THRESHOLD that frame.
    fps converts the DLC frame index into a msec timestamp; pass None to keep
    frame index as a (unitless) pseudo-timestamp instead."""
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    df.columns = df.columns.droplevel(0)  # drop the repeated scorer-name level

    led_bodyparts = pick_led_bodyparts(df.columns)

    frame_idx = pd.to_numeric(pd.Series(df.index), errors='coerce').to_numpy()

    xs = np.stack([pd.to_numeric(df[(bp, 'x')], errors='coerce').to_numpy() for bp in led_bodyparts])
    ys = np.stack([pd.to_numeric(df[(bp, 'y')], errors='coerce').to_numpy() for bp in led_bodyparts])
    likelihoods = np.stack([pd.to_numeric(df[(bp, 'likelihood')], errors='coerce').to_numpy() for bp in led_bodyparts])

    good = likelihoods >= LIKELIHOOD_THRESHOLD
    with np.errstate(invalid='ignore'):
        x = np.nanmean(np.where(good, xs, np.nan), axis=0)
        y = np.nanmean(np.where(good, ys, np.nan), axis=0)

    t = frame_idx / fps * 1000.0 if fps else frame_idx.astype(float)  # msec

    valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    return t[valid], x[valid], y[valid]


def apply_flip(x, y, frame_height):
    if FLIP_Y and frame_height:
        return x, frame_height - y
    return x, y


def flag_outside_arena(x, y, center_x, center_y, radius):
    dist = np.hypot(x - center_x, y - center_y)
    return dist > radius, dist


def detect_arena_circle_px(video_path, frame_width, frame_height):
    """Best-effort detection of the circular arena boundary in pixel space
    via Hough circle transform on a representative (mid-video) frame.
    Returns (center_x_px, center_y_px, radius_px), or None if detection fails."""
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid = min(n_frames // 2, n_frames - 1) if n_frames > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    min_dim = min(frame_width, frame_height)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=min_dim,
        param1=100, param2=60,
        minRadius=int(min_dim * 0.3), maxRadius=int(min_dim * 0.55),
    )
    if circles is None or len(circles[0]) == 0:
        return None

    cx, cy, r = circles[0][0]
    return float(cx), float(cy), float(r)


def get_arena_circle(video_path):
    """Returns (center_x_px, center_y_px, radius_px, calib_info), locating the
    arena directly in the video's own pixel space -- the same space the
    tracking x/y already live in, so nothing needs to be rescaled."""
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    detected = detect_arena_circle_px(video_path, frame_width, frame_height)
    if detected is not None:
        center_x_px, center_y_px, radius_px = detected
        method = 'hough-detected'
    else:
        center_x_px, center_y_px = frame_width / 2.0, frame_height / 2.0
        radius_px = (min(frame_width, frame_height) / 2.0) * ARENA_FILL_FRACTION
        method = 'fallback-fill-fraction'

    calib_info = dict(
        frame_width=frame_width, frame_height=frame_height, fps=fps,
        center_x_px=center_x_px, center_y_px=center_y_px, radius_px=radius_px,
        method=method,
    )
    return center_x_px, center_y_px, radius_px, calib_info


def compute_occupancy_map(t, x, y, bin_size_px, smoothing_sigma):
    """Same binning as batch_occupancy_maps.py, but on raw (unfiltered) x/y
    so mislabeled points remain visible instead of being interpolated out."""
    x_edges = np.arange(x.min(), x.max() + bin_size_px, bin_size_px)
    y_edges = np.arange(y.min(), y.max() + bin_size_px, bin_size_px)

    frame_counts, x_edges, y_edges = np.histogram2d(x, y, bins=[x_edges, y_edges])
    unvisited = frame_counts == 0

    dt_sec = np.median(np.diff(t)) / 1000.0
    occ_map_sec = frame_counts * dt_sec
    if smoothing_sigma > 0:
        occ_map_sec = gaussian_filter(occ_map_sec, sigma=smoothing_sigma)

    return occ_map_sec, x_edges, y_edges, unvisited


def plot_occupancy_with_flags(t, x, y, bad_mask, circle, title, out_path):
    """circle: (center_x, center_y, radius) in the same px space as x/y, or
    None if no video was available to locate the arena."""
    if circle is not None:
        _cx, _cy, radius = circle
        bin_size_px = max(2.0 * radius / OCC_N_BINS_ACROSS_DIAMETER, 1e-6)
    else:
        bin_size_px = max((x.max() - x.min()) / OCC_N_BINS_ACROSS_DIAMETER, 1e-6)

    occ_map_sec, x_edges, y_edges, unvisited = compute_occupancy_map(
        t, x, y, bin_size_px, OCC_SMOOTHING_SIGMA
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = matplotlib.colormaps['viridis'].copy()
    cmap.set_bad('white')
    masked_map = np.ma.masked_where(unvisited, occ_map_sec)

    im = ax.imshow(
        masked_map.T, origin='lower',
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap=cmap, aspect='equal', interpolation='nearest',
    )
    plt.colorbar(im, ax=ax, label='Occupancy (s)')

    if circle is not None:
        center_x, center_y, radius = circle
        arena_circle = Circle(
            (center_x, center_y), radius, fill=False,
            edgecolor='lime', linewidth=1.5, linestyle='--',
            label=f'detected arena ({radius:.0f} px radius)',
        )
        ax.add_patch(arena_circle)

    n_bad = int(bad_mask.sum())
    if n_bad > 0:
        ax.scatter(
            x[bad_mask], y[bad_mask], s=16, facecolors='none',
            edgecolors='red', linewidths=1.3, label=f'flagged out-of-bounds (n={n_bad})',
        )

    ax.set_title(title)
    ax.set_xlabel('X position (px)')
    ax.set_ylabel('Y position (px)')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=OCC_DPI)
    plt.close(fig)


def make_occupancy_map(csv_path, output_dir, video_path):
    name = os.path.splitext(os.path.basename(csv_path))[0]

    circle = None
    calib_info = None
    if video_path is not None:
        try:
            center_x_px, center_y_px, radius_px, calib_info = get_arena_circle(video_path)
            circle = (center_x_px, center_y_px, radius_px)
        except Exception as exc:
            print(f"  WARNING: could not locate arena circle for occupancy map ({exc})")

    fps = calib_info['fps'] if calib_info else None
    t_ms, x, y = load_tracking(csv_path, fps)
    if len(t_ms) < 2:
        print("  SKIP occupancy map: fewer than 2 valid tracking samples")
        return

    bad_mask = np.zeros(len(x), dtype=bool)
    if circle is not None:
        xf, yf = apply_flip(x, y, calib_info['frame_height'])
        bad_mask, _dist = flag_outside_arena(xf, yf, circle[0], circle[1], circle[2] + MARGIN_PX)

    out_path = os.path.join(output_dir, f"{name}_occupancy_map.png")
    plot_occupancy_with_flags(t_ms, x, y, bad_mask, circle, name, out_path)
    print(f"  occupancy map -> {out_path}")


def save_calibration_image(video_path, calib_info, x_all, y_all, bad_mask, out_path):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid = min(n_frames // 2, n_frames - 1) if n_frames > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return

    cx, cy, r = int(calib_info['center_x_px']), int(calib_info['center_y_px']), int(calib_info['radius_px'])
    cv2.circle(frame, (cx, cy), r, (0, 255, 0), 3)
    cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

    xf, yf = apply_flip(x_all, y_all, calib_info['frame_height'])
    for xi, yi, bad in zip(xf, yf, bad_mask):
        color = (0, 0, 255) if bad else (255, 200, 0)
        cv2.circle(frame, (int(xi), int(yi)), 3, color, -1)

    cv2.putText(frame, f"calibration method: {calib_info['method']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(out_path, frame)


def annotate_video(video_path, out_path, bad_frame_points):
    """bad_frame_points: dict frame_idx -> list of (x_px, y_px)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx = 0
    n_written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in bad_frame_points:
            for (px, py) in bad_frame_points[frame_idx]:
                cv2.circle(frame, (int(px), int(py)), MARK_RADIUS_PX, MARK_COLOR_BGR, MARK_THICKNESS)
        writer.write(frame)
        frame_idx += 1
        n_written += 1

    cap.release()
    writer.release()
    return n_written


def process_file(csv_path, video_path, output_dir):
    name = os.path.splitext(os.path.basename(csv_path))[0]

    center_x_px, center_y_px, radius_px, calib_info = get_arena_circle(video_path)
    fps = calib_info['fps']
    if not fps or fps <= 0:
        print(f"  SKIP: could not read a valid fps from {video_path}")
        return None
    print(f"  pixel calibration: {calib_info['method']}, "
          f"center=({center_x_px:.0f}, {center_y_px:.0f}) px, radius={radius_px:.0f} px")

    t_ms, x, y = load_tracking(csv_path, fps)
    if len(t_ms) == 0:
        print(f"  SKIP: no valid tracking samples in {csv_path}")
        return None

    xf, yf = apply_flip(x, y, calib_info['frame_height'])
    bad_mask, dist_px = flag_outside_arena(xf, yf, center_x_px, center_y_px, radius_px + MARGIN_PX)
    n_bad = int(bad_mask.sum())
    print(f"  {n_bad}/{len(x)} points flagged outside the detected {radius_px:.1f} px-radius arena")

    if n_bad == 0:
        return []

    if MAKE_CALIBRATION_IMAGE:
        calib_path = os.path.join(output_dir, f"{name}_calibration_check.png")
        save_calibration_image(video_path, calib_info, x, y, bad_mask, calib_path)
        print(f"  calibration check image -> {calib_path}  "
              f"(verify the green circle traces the arena edge before trusting results)")

    bad_idx = np.where(bad_mask)[0]
    t_sec = t_ms[bad_idx] / 1000.0
    frame_idx_for_bad = np.clip(np.round(t_sec * fps).astype(int), 0, None)

    bad_frame_points = {}
    rows = []
    for i, fi, ts in zip(bad_idx, frame_idx_for_bad, t_sec):
        bad_frame_points.setdefault(int(fi), []).append((xf[i], yf[i]))
        rows.append(dict(
            frame_idx=int(fi), timestamp_sec=float(ts),
            x_px=float(x[i]), y_px=float(y[i]), dist_from_center_px=float(dist_px[i]),
        ))

    report_path = os.path.join(output_dir, f"{name}_flagged_timestamps.csv")
    pd.DataFrame(rows).sort_values('timestamp_sec').to_csv(report_path, index=False)
    print(f"  flagged-timestamp list -> {report_path}")

    annotated_path = os.path.join(output_dir, f"{name}_annotated.mp4")
    n_frames = annotate_video(video_path, annotated_path, bad_frame_points)
    print(f"  annotated video ({n_frames} frames) -> {annotated_path}")

    return rows


def interpolate_flagged_points(t, x, y, bad_mask):
    """Deletes every flagged (out-of-arena) point and linearly interpolates
    it from the nearest surrounding in-arena points, using timestamp t as
    the interpolation axis (so this still works across gaps left by frames
    load_tracking already dropped). A flagged point beyond the first/last
    good point (no good neighbor on that side) is held at the nearest good
    value, since there's nothing on that side to interpolate between."""
    x_corr = x.copy()
    y_corr = y.copy()
    good = ~bad_mask
    if not good.any() or not bad_mask.any():
        return x_corr, y_corr
    x_corr[bad_mask] = np.interp(t[bad_mask], t[good], x[good])
    y_corr[bad_mask] = np.interp(t[bad_mask], t[good], y[good])
    return x_corr, y_corr


def save_corrected_tracking(csv_path, output_dir, video_path):
    """Locates the arena circle the same way make_occupancy_map does, flags
    out-of-arena points, and writes <name>_corrected_tracking.csv with every
    flagged point deleted and linearly interpolated from its nearest good
    neighbors. Always (re)generated, like the occupancy map, since it only
    needs the .csv (+ video, to locate the arena circle)."""
    name = os.path.splitext(os.path.basename(csv_path))[0]

    if video_path is None:
        print("  SKIP correction: no matching video found next to this .csv (arena circle unknown)")
        return

    try:
        center_x_px, center_y_px, radius_px, calib_info = get_arena_circle(video_path)
    except Exception as exc:
        print(f"  WARNING: could not locate arena circle for correction ({exc})")
        return

    fps = calib_info['fps']
    t_ms, x, y = load_tracking(csv_path, fps)
    if len(t_ms) == 0:
        print("  SKIP correction: no valid tracking samples")
        return

    xf, yf = apply_flip(x, y, calib_info['frame_height'])
    bad_mask, _dist = flag_outside_arena(xf, yf, center_x_px, center_y_px, radius_px + MARGIN_PX)
    x_corr, y_corr = interpolate_flagged_points(t_ms, x, y, bad_mask)

    frame_idx = np.clip(np.round(t_ms / 1000.0 * fps).astype(int), 0, None)
    out_path = os.path.join(output_dir, f"{name}_corrected_tracking.csv")
    pd.DataFrame(dict(
        frame_idx=frame_idx, timestamp_sec=t_ms / 1000.0,
        x_px=x_corr, y_px=y_corr, was_interpolated=bad_mask,
    )).to_csv(out_path, index=False)
    print(f"  {int(bad_mask.sum())} point(s) interpolated -> corrected tracking -> {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_files = find_csv_files(ROOT_DIR)
    print(f"Found {len(csv_files)} .csv file(s) under {ROOT_DIR}")

    all_reports = []
    for i, csv_path in enumerate(csv_files, start=1):
        print(f"[{i}/{len(csv_files)}] {csv_path}")
        name = os.path.splitext(os.path.basename(csv_path))[0]
        video_path = find_matching_video(csv_path)

        report_path = os.path.join(OUTPUT_DIR, f"{name}_flagged_timestamps.csv")
        annotated_path = os.path.join(OUTPUT_DIR, f"{name}_annotated.mp4")
        calib_path = os.path.join(OUTPUT_DIR, f"{name}_calibration_check.png")
        already_flagged = (
            os.path.exists(report_path) or os.path.exists(annotated_path) or os.path.exists(calib_path)
        )

        if already_flagged:
            print(f"  flagging outputs already exist in {OUTPUT_DIR} -- skipping flagging/video step")
            if os.path.exists(report_path):
                rows = pd.read_csv(report_path).to_dict('records')
                for r in rows:
                    r['csv_file'] = os.path.basename(csv_path)
                all_reports.extend(rows)
        else:
            if video_path is None:
                print(f"  SKIP flagging: no matching video found next to this .csv")
            else:
                try:
                    rows = process_file(csv_path, video_path, OUTPUT_DIR)
                    if rows:
                        for r in rows:
                            r['csv_file'] = os.path.basename(csv_path)
                        all_reports.extend(rows)
                except Exception as exc:
                    print(f"  ERROR: {exc}")

        try:
            make_occupancy_map(csv_path, OUTPUT_DIR, video_path)
        except Exception as exc:
            print(f"  ERROR making occupancy map: {exc}")

        try:
            save_corrected_tracking(csv_path, OUTPUT_DIR, video_path)
        except Exception as exc:
            print(f"  ERROR saving corrected tracking: {exc}")

    if all_reports:
        summary_path = os.path.join(OUTPUT_DIR, "ALL_flagged_timestamps.csv")
        pd.DataFrame(all_reports).to_csv(summary_path, index=False)
        print(f"\nSummary of all flagged points across all files -> {summary_path}")
    else:
        print("\nNo flagged points found in any file.")


if __name__ == '__main__':
    main()
