# -*- coding: utf-8 -*-
"""
Detect mislabeled (out-of-arena) tracking points by point-density clustering,
correct them by interpolation, and recompute the pixel->cm conversion from
the cleaned pixel coordinates.

Background / why this exists
-----------------------------
pixel_to_cm_conversion_v4_batch.py derives its px->cm scale from the raw
bounding box of the tracked points (arena_size_cm / max(x_range_px, y_range_px)).
If some points are mislabeled (e.g. tracker latched onto something outside
the arena), the bounding box is inflated by those bad points and the entire
px->cm scale for that file comes out wrong -- every point in the file, not
just the bad ones, ends up on the wrong scale.

This script fixes that at the source: it identifies mislabeled points
directly from where they sit in the point cloud (density-based, not a
speed/jump heuristic), deletes and interpolates them in pixel space, and
*then* computes the px->cm scale from the cleaned points.

Input files
-----------
.csv files, one per session, with (positionally, regardless of header names):
    column A (index 0): timestamp
    column B (index 1): x, pixels
    column C (index 2): y, pixels
    column D (index 3): x, cm  (from a previous, unreliable conversion -- ignored/recomputed)
    column E (index 4): y, cm  (from a previous, unreliable conversion -- ignored/recomputed)

Two arena types, each with its own hardcoded input directory (searched
recursively) and its own known real-world geometry:
    CIRCLE_DIR -- 1D circular (annular) track, outer diameter CIRCLE_OUTER_DIAMETER_CM,
                  inner diameter CIRCLE_INNER_DIAMETER_CM
    OPEN_DIR   -- open-field circular arena (solid disc), diameter OPEN_ARENA_DIAMETER_CM

How mislabeled points are detected (density-based)
---------------------------------------------------
1. Bin the raw (x_px, y_px) point cloud into a 2D histogram (bin size chosen
   to be roughly DENSITY_BIN_CM on a side, using a rough/approximate scale
   just for sizing the grid -- it does not need to be accurate).
2. Threshold bins by occupancy: a bin only counts as part of the real
   arena/track footprint if it holds at least MIN_BIN_COUNT_FRACTION of the
   busiest bin's count.
3. Label connected components (8-connectivity) among the surviving bins.
   The true arena/track is the component holding the most total points --
   this works for both a solid disc (open field) and a ring (circular
   track) without assuming either shape, since real tracking naturally
   clusters into one dense, connected blob while mislabeled points land in
   small, disconnected, low-density bins elsewhere in the frame.
4. Any point whose bin isn't part of that largest component is flagged as
   mislabeled.

Flagged points are deleted and linearly interpolated (in pixel space, using
the real timestamp as the interpolation axis) from their nearest surrounding
good neighbors. Points beyond the first/last good point are held at the
nearest good value.

Output, per input file, saved next to the input (OUTPUT_SUFFIX appended):
    <name>_corrected.csv     -- timestamp, x_px, y_px, x_cm, y_cm, was_interpolated
    <name>_density_check.png -- density map + flagged points, for a sanity check
                                 (only if MAKE_DIAGNOSTIC_PLOT is True)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe for batch runs
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure

# ── USER INPUT ──────────────────────────────────────────────────────────────
# Hardcoded per-arena input directories (each searched recursively for .csv files)
CIRCLE_DIR = r'C:/Runita/NMR/analysis/TrackingCorrection/Circle'  # 1D circular (annular) track
OPEN_DIR = r'C:/Runita/NMR/analysis/TrackingCorrection/Open'      # open-field arena (solid disc)

# Real-world geometry per arena, used for the final px->cm scale
CIRCLE_OUTER_DIAMETER_CM = 80.0
CIRCLE_INNER_DIAMETER_CM = 72.0  # not used for scaling directly, kept for reference/QC
OPEN_ARENA_DIAMETER_CM = 60.0

# Density-based detection parameters
DENSITY_BIN_CM = 2.0             # approx. bin size (cm) for the 2D occupancy histogram
MIN_BIN_COUNT_FRACTION = 0.02    # a bin must hold >= this fraction of the busiest bin's count to count as "in arena"

MAKE_DIAGNOSTIC_PLOT = True
OUTPUT_SUFFIX = '_corrected'
# ─────────────────────────────────────────────────────────────────────────────


def find_csv_files(root_dir):
    csv_paths = []
    for folder, _dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith('.csv') and not f.endswith(OUTPUT_SUFFIX + '.csv'):
                csv_paths.append(os.path.join(folder, f))
    return sorted(csv_paths)


def load_raw_tracking(csv_path):
    """Reads columns positionally: A=timestamp, B=x_px, C=y_px. Columns D/E
    (previous, unreliable cm values), if present, are ignored -- they get
    recomputed from the cleaned pixel coordinates."""
    df = pd.read_csv(csv_path)
    t_col, x_col, y_col = df.columns[0], df.columns[1], df.columns[2]
    t = pd.to_numeric(df[t_col], errors='coerce').to_numpy(dtype=float)
    x_px = pd.to_numeric(df[x_col], errors='coerce').to_numpy(dtype=float)
    y_px = pd.to_numeric(df[y_col], errors='coerce').to_numpy(dtype=float)

    valid = np.isfinite(t) & np.isfinite(x_px) & np.isfinite(y_px)
    return t[valid], x_px[valid], y_px[valid]


def flag_low_density_points(x_px, y_px, arena_size_cm, bin_cm, min_bin_count_fraction):
    """Flags points that don't belong to the single largest connected,
    dense blob in the (x_px, y_px) point cloud. Returns (bad_mask, info_dict)."""
    x_range_px = max(x_px.max() - x_px.min(), 1e-6)
    y_range_px = max(y_px.max() - y_px.min(), 1e-6)

    # Rough scale just to size the histogram grid -- doesn't need to be accurate,
    # since it's inflated by the very outliers we're about to remove.
    rough_scale_cm_per_px = arena_size_cm / max(x_range_px, y_range_px)
    bin_size_px = max(bin_cm / rough_scale_cm_per_px, 1e-6)

    x_edges = np.arange(x_px.min(), x_px.max() + bin_size_px, bin_size_px)
    y_edges = np.arange(y_px.min(), y_px.max() + bin_size_px, bin_size_px)
    counts, x_edges, y_edges = np.histogram2d(x_px, y_px, bins=[x_edges, y_edges])

    occupied = counts >= max(min_bin_count_fraction * counts.max(), 1)

    structure = generate_binary_structure(2, 2)  # 8-connectivity, so a ring's bins connect
    labeled, n_components = label(occupied, structure=structure)

    if n_components == 0:
        # degenerate case (shouldn't happen with any real data): nothing flagged
        return np.zeros(len(x_px), dtype=bool), dict(
            x_edges=x_edges, y_edges=y_edges, counts=counts, labeled=labeled, main_label=0,
        )

    # the "real" arena/track is the component with the most total points, not
    # just the most bins -- a small stray cluster of mislabeled points could
    # otherwise span more (mostly-empty) bins than a tight, busy real cluster
    component_totals = np.array([
        counts[labeled == lbl].sum() for lbl in range(1, n_components + 1)
    ])
    main_label = 1 + int(np.argmax(component_totals))

    x_bin_idx = np.clip(np.digitize(x_px, x_edges) - 1, 0, counts.shape[0] - 1)
    y_bin_idx = np.clip(np.digitize(y_px, y_edges) - 1, 0, counts.shape[1] - 1)
    point_labels = labeled[x_bin_idx, y_bin_idx]

    bad_mask = point_labels != main_label
    info = dict(x_edges=x_edges, y_edges=y_edges, counts=counts, labeled=labeled, main_label=main_label)
    return bad_mask, info


def interpolate_flagged_points(t, x, y, bad_mask):
    """Deletes every flagged point and linearly interpolates it (using
    timestamp t as the interpolation axis) from the nearest surrounding
    good points. A flagged point beyond the first/last good point is held
    at the nearest good value."""
    x_corr, y_corr = x.copy(), y.copy()
    good = ~bad_mask
    if not good.any() or not bad_mask.any():
        return x_corr, y_corr
    x_corr[bad_mask] = np.interp(t[bad_mask], t[good], x[good])
    y_corr[bad_mask] = np.interp(t[bad_mask], t[good], y[good])
    return x_corr, y_corr


def plot_density_check(x_px, y_px, bad_mask, info, title, out_path):
    counts, x_edges, y_edges, labeled, main_label = (
        info['counts'], info['x_edges'], info['y_edges'], info['labeled'], info['main_label']
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    im = ax.imshow(
        counts.T, origin='lower',
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap='viridis', aspect='equal', interpolation='nearest',
    )
    main_component_edge = (labeled.T == main_label).astype(float)
    ax.contour(
        main_component_edge, levels=[0.5], colors='lime', linewidths=1.5,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    )
    plt.colorbar(im, ax=ax, label='points per bin')
    ax.set_title('occupancy density\n(green = kept "in-arena" component)')
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')

    ax = axes[1]
    ax.scatter(x_px[~bad_mask], y_px[~bad_mask], s=4, c='tab:blue', label=f'kept (n={int((~bad_mask).sum())})')
    n_bad = int(bad_mask.sum())
    if n_bad > 0:
        ax.scatter(x_px[bad_mask], y_px[bad_mask], s=16, facecolors='none',
                   edgecolors='red', linewidths=1.3, label=f'flagged (n={n_bad})')
    ax.set_aspect('equal')
    ax.invert_yaxis()  # match typical video pixel-row orientation
    ax.set_title('flagged points')
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def process_file(csv_path, arena_size_cm):
    name = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.dirname(csv_path)

    t, x_px, y_px = load_raw_tracking(csv_path)
    if len(t) < 2:
        print(f"  SKIP: fewer than 2 valid tracking samples in {csv_path}")
        return

    bad_mask, info = flag_low_density_points(
        x_px, y_px, arena_size_cm, DENSITY_BIN_CM, MIN_BIN_COUNT_FRACTION
    )
    n_bad = int(bad_mask.sum())
    print(f"  {n_bad}/{len(x_px)} points flagged as low-density outliers")

    if MAKE_DIAGNOSTIC_PLOT:
        plot_path = os.path.join(out_dir, f"{name}_density_check.png")
        plot_density_check(x_px, y_px, bad_mask, info, name, plot_path)
        print(f"  density check image -> {plot_path}")

    x_corr, y_corr = interpolate_flagged_points(t, x_px, y_px, bad_mask)

    # Recompute the px->cm scale from the *cleaned* points, same convention
    # as pixel_to_cm_conversion_v4_batch.py: arena_size_cm / longer bounding-box axis.
    x_dist_px = x_corr.max() - x_corr.min()
    y_dist_px = y_corr.max() - y_corr.min()
    scale_cm_per_px = arena_size_cm / max(x_dist_px, y_dist_px)

    x_cm = x_corr * scale_cm_per_px
    y_cm = y_corr * scale_cm_per_px
    x_cm -= x_cm.min()
    y_cm -= y_cm.min()

    print(f"  x_dist_px={x_dist_px:.2f}, y_dist_px={y_dist_px:.2f}, "
          f"arena_size_cm={arena_size_cm}, scale={scale_cm_per_px:.5f} cm/px")

    out_path = os.path.join(out_dir, f"{name}{OUTPUT_SUFFIX}.csv")
    pd.DataFrame(dict(
        timestamp=t, x_px=x_corr, y_px=y_corr, x_cm=x_cm, y_cm=y_cm, was_interpolated=bad_mask,
    )).to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")


def process_arena(root_dir, arena_size_cm, arena_label):
    csv_files = find_csv_files(root_dir)
    print(f"\n[{arena_label}] Found {len(csv_files)} .csv file(s) under {root_dir}")
    for i, csv_path in enumerate(csv_files, start=1):
        print(f"[{arena_label} {i}/{len(csv_files)}] {csv_path}")
        try:
            process_file(csv_path, arena_size_cm)
        except Exception as exc:
            print(f"  ERROR: {exc}")


def main():
    process_arena(CIRCLE_DIR, CIRCLE_OUTER_DIAMETER_CM, 'circle')
    process_arena(OPEN_DIR, OPEN_ARENA_DIAMETER_CM, 'open')


if __name__ == '__main__':
    main()
