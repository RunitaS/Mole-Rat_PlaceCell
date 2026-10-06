# -*- coding: utf-8 -*-
"""
Jump correction for open-field tracking in a circular arena (~60 cm diameter).

The .csv files listed in FILE_NAMES are searched for by exact name anywhere
under ROOT_DIR (recursively). Each one is processed as follows:

1. Find the arena wall (a circle in pixel space) from the tracking data
   itself -- no video needed:
     a. Keep only the dense part of the point cloud: bin the points into a
        2D occupancy histogram, keep bins with >= MIN_BIN_COUNT points, and
        keep the connected group of bins that holds the most points. Sparse
        stray points (and small isolated clusters outside the arena) are
        dropped here, so they can't pull the wall estimate outward.
     b. Around a provisional center, split the dense points into N_SECTORS
        angular sectors and take the WALL_PERCENTILE-th percentile of the
        radial distance in each sector as one point on the wall.
     c. Least-squares fit a circle to those wall points. The fit is
        one-sided: the animal can't go past the wall but may never reach it
        in some directions, so wall points lying well inside the fitted
        circle (> WALL_INNER_TOL of the radius) are dropped and the circle is
        refit. Repeat (b)-(c) with the updated center until it stops moving.
2. Flag every point farther than the fitted radius (+ WALL_MARGIN_CM) from
   the fitted center as outside the arena. Rows with a missing x/y are
   flagged too.
3. Delete the flagged points and linearly interpolate them (timestamp as the
   interpolation axis) from the previous and next reliable (in-arena) points.
   Flagged points before the first / after the last reliable point are held
   at that reliable point's value.
4. If the file has cm columns (D/E) and RECOMPUTE_CM is True, recompute them
   from the corrected pixels using the fitted circle:
       scale_cm_per_px = ARENA_DIAMETER_CM / (2 * fitted_radius_px)
   with the arena's bounding square starting at 0 cm on each axis. This
   replaces the bounding-box scale from pixel_to_cm_conversion_v4_batch.py,
   which is inflated by any out-of-arena points.

Input .csv layout (read positionally, regardless of header names):
    column A (index 0): timestamp
    column B (index 1): x, pixels
    column C (index 2): y, pixels
    column D (index 3): x, cm   (optional, recomputed if present)
    column E (index 4): y, cm   (optional, recomputed if present)
Any further columns are copied through unchanged.

Output, per input file, saved next to it (the input is not modified):
    <name>_jumpcorr.csv        -- all original columns with corrected x/y,
                                  plus a 'was_interpolated' column
    <name>_jumpcorr_check.png  -- fitted arena wall, flagged points and the
                                  corrected trajectory, for a sanity check
                                  (only if MAKE_DIAGNOSTIC_PLOT is True)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe for batch runs
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import label, generate_binary_structure
from scipy.optimize import least_squares

# ── USER INPUT ──────────────────────────────────────────────────────────────
ROOT_DIR = r'X:/NMR_group_data/Runita/Analysis/Thesis/Corr_Data_SpkQltyFilt/SpikeQualityFilt/QualFiltData_AllCells'  # searched recursively

# Exact names of the 4 tracking files to correct ('.csv' is added if missing)
FILE_NAMES = [
    'ExperimentDay10_NestBuild_CntrlZeroRotate_3Rotate_cm.csv',
    'ExperimentDay11_NestBuild_ZeroRotateCntrl_2Rotate_cm.csv',
    'ExperimentDay11_NestBuild_ZeroRotateCntrl_3Cntrl_cm.csv',
    'Fa5834_Day22_cm.csv',
]

ARENA_DIAMETER_CM = 60.0
WALL_MARGIN_CM = 2.0        # tolerance past the fitted wall before a point counts as outside (head LED can sit over the wall edge)

# Wall-estimation parameters
DENSITY_N_BINS_ACROSS = 30  # occupancy-histogram bins across the arena diameter (~2 cm bins for a 60 cm arena)
MIN_BIN_COUNT = 3           # a bin needs at least this many points to count as part of the dense arena footprint
N_SECTORS = 36              # angular sectors used to sample the wall (10 deg each)
MIN_POINTS_PER_SECTOR = 20  # sectors with fewer dense points than this are skipped (animal rarely went there)
WALL_PERCENTILE = 99.0      # radial-distance percentile in each sector taken as the wall position
WALL_INNER_TOL = 0.08       # sector wall points more than this fraction of the radius inside the fit are treated as "animal never reached the wall here" and dropped
MAX_FIT_ITER = 10

RECOMPUTE_CM = True         # rewrite columns D/E (if present) from the corrected pixels + fitted circle
MAKE_DIAGNOSTIC_PLOT = True
OUTPUT_SUFFIX = '_jumpcorr'
INTERPOLATED_COLUMN = 'was_interpolated'
# ─────────────────────────────────────────────────────────────────────────────


def normalize_file_names(file_names):
    return [n if n.lower().endswith('.csv') else n + '.csv' for n in file_names]


def find_named_files(root_dir, file_names):
    """Returns {file_name: [full paths]} for every exact-name match (case-
    insensitive, as on Windows) anywhere under root_dir."""
    wanted = {n.lower(): n for n in file_names}
    found = {n: [] for n in file_names}
    for folder, _dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower() in wanted:
                found[wanted[f.lower()]].append(os.path.join(folder, f))
    return found


def dense_point_mask(x_px, y_px):
    """Flags the points belonging to the dense arena footprint: the connected
    group of well-visited occupancy bins holding the most points. Bin size
    comes from the 1st-99th percentile spread of the points, so a handful of
    far-out points don't stretch the grid."""
    spread_px = max(np.percentile(x_px, 99) - np.percentile(x_px, 1),
                    np.percentile(y_px, 99) - np.percentile(y_px, 1), 1e-6)
    bin_size_px = spread_px / DENSITY_N_BINS_ACROSS

    x_edges = np.arange(x_px.min(), x_px.max() + bin_size_px, bin_size_px)
    y_edges = np.arange(y_px.min(), y_px.max() + bin_size_px, bin_size_px)
    counts, x_edges, y_edges = np.histogram2d(x_px, y_px, bins=[x_edges, y_edges])

    occupied = counts >= MIN_BIN_COUNT
    structure = generate_binary_structure(2, 2)  # 8-connectivity
    labeled, n_components = label(occupied, structure=structure)
    if n_components == 0:
        return np.ones(len(x_px), dtype=bool)

    component_totals = np.array([
        counts[labeled == lbl].sum() for lbl in range(1, n_components + 1)
    ])
    main_label = 1 + int(np.argmax(component_totals))

    x_bin_idx = np.clip(np.digitize(x_px, x_edges) - 1, 0, counts.shape[0] - 1)
    y_bin_idx = np.clip(np.digitize(y_px, y_edges) - 1, 0, counts.shape[1] - 1)
    return labeled[x_bin_idx, y_bin_idx] == main_label


def fit_circle_least_squares(x, y):
    """Algebraic (Kasa) circle fit for a starting guess, refined by a
    geometric fit (minimizing distance to the circle), since the algebraic
    fit underestimates the radius when the points cover only part of the
    circle. Returns (center_x, center_y, radius)."""
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = -D / 2.0, -E / 2.0
    r = np.sqrt(max(cx**2 + cy**2 - F, 1e-12))

    def residuals(p):
        return np.hypot(x - p[0], y - p[1]) - p[2]
    cx, cy, r = least_squares(residuals, [cx, cy, r]).x
    return cx, cy, abs(r)


def fit_wall_circle(wall_x, wall_y):
    """One-sided robust circle fit. The animal can't go past the wall but may
    never reach it in some directions, so a sector's wall point can only
    underestimate the true wall. Wall points sitting more than WALL_INNER_TOL
    (fraction of the radius) inside the current fit are dropped and the
    circle is refit, until the set of kept points stops changing.
    Returns (center_x, center_y, radius, keep_mask)."""
    keep = np.ones(len(wall_x), dtype=bool)
    cx, cy, r = fit_circle_least_squares(wall_x, wall_y)
    for _ in range(MAX_FIT_ITER):
        new_keep = np.hypot(wall_x - cx, wall_y - cy) >= r * (1 - WALL_INNER_TOL)
        if new_keep.sum() < 3 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
        cx, cy, r = fit_circle_least_squares(wall_x[keep], wall_y[keep])
    return cx, cy, r, keep


def sample_wall_points(x, y, cx, cy):
    """One wall point per angular sector around (cx, cy): the point direction
    at the WALL_PERCENTILE-th percentile of radial distance in that sector."""
    dx, dy = x - cx, y - cy
    dist = np.hypot(dx, dy)
    sector = ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS

    wall_x, wall_y = [], []
    for s in range(N_SECTORS):
        in_sector = sector == s
        if in_sector.sum() < MIN_POINTS_PER_SECTOR:
            continue
        r_wall = np.percentile(dist[in_sector], WALL_PERCENTILE)
        theta = -np.pi + (s + 0.5) * 2 * np.pi / N_SECTORS  # sector mid-angle
        wall_x.append(cx + r_wall * np.cos(theta))
        wall_y.append(cy + r_wall * np.sin(theta))
    return np.array(wall_x), np.array(wall_y)


def estimate_arena_circle(x_px, y_px):
    """Returns (center_x, center_y, radius, wall_x, wall_y, wall_keep), all
    in pixels; wall_keep marks the wall points used in the final fit."""
    dense = dense_point_mask(x_px, y_px)
    xd, yd = x_px[dense], y_px[dense]

    cx, cy = np.median(xd), np.median(yd)
    radius = np.nan
    wall_x = wall_y = wall_keep = np.array([])
    for _ in range(MAX_FIT_ITER):
        wall_x, wall_y = sample_wall_points(xd, yd, cx, cy)
        if len(wall_x) < 3:
            raise ValueError(f"only {len(wall_x)} angular sector(s) had enough points to locate the wall")
        new_cx, new_cy, radius, wall_keep = fit_wall_circle(wall_x, wall_y)
        shift = np.hypot(new_cx - cx, new_cy - cy)
        cx, cy = new_cx, new_cy
        if shift < 0.01 * radius:
            break
    return cx, cy, radius, wall_x, wall_y, wall_keep


def interpolate_flagged_points(t, x, y, bad_mask):
    """Deletes every flagged point and linearly interpolates it (timestamp t
    as the interpolation axis) from the previous and next reliable points.
    A flagged point before the first / after the last reliable point is held
    at that reliable point's value."""
    x_corr, y_corr = x.copy(), y.copy()
    good = ~bad_mask
    if not bad_mask.any():
        return x_corr, y_corr
    x_corr[bad_mask] = np.interp(t[bad_mask], t[good], x[good])
    y_corr[bad_mask] = np.interp(t[bad_mask], t[good], y[good])
    return x_corr, y_corr


def plot_check(x_raw, y_raw, x_corr, y_corr, bad_mask, circle, wall_xyk, margin_px, title, out_path):
    cx, cy, radius = circle
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    finite = np.isfinite(x_raw) & np.isfinite(y_raw)
    kept = finite & ~bad_mask
    flagged = finite & bad_mask
    ax.scatter(x_raw[kept], y_raw[kept], s=2, c='tab:blue', alpha=0.4, label=f'in arena (n={int(kept.sum())})')
    if flagged.any():
        ax.scatter(x_raw[flagged], y_raw[flagged], s=16, facecolors='none', edgecolors='red',
                   linewidths=1.2, label=f'outside arena, removed (n={int(flagged.sum())})')
    wall_x, wall_y, wall_keep = wall_xyk
    ax.scatter(wall_x[wall_keep], wall_y[wall_keep], s=25, c='orange', marker='x',
               label='wall points used in fit', zorder=3)
    if (~wall_keep).any():
        ax.scatter(wall_x[~wall_keep], wall_y[~wall_keep], s=25, c='0.5', marker='x',
                   label='wall points dropped (wall not reached)', zorder=3)
    ax.add_patch(Circle((cx, cy), radius, fill=False, edgecolor='lime', linewidth=1.8,
                        label=f'fitted wall (r={radius:.1f} px)'))
    ax.add_patch(Circle((cx, cy), radius + margin_px, fill=False, edgecolor='lime', linewidth=1,
                        linestyle='--', label=f'cutoff (+{WALL_MARGIN_CM:g} cm)'))
    ax.set_title('raw tracking')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.85)

    ax = axes[1]
    ax.plot(x_corr, y_corr, lw=0.4, c='0.3')
    ax.add_patch(Circle((cx, cy), radius, fill=False, edgecolor='lime', linewidth=1.8))
    ax.set_title('corrected trajectory')

    # same view on both panels, wide enough to show the raw outliers
    x_lo, x_hi = np.nanmin(x_raw), np.nanmax(x_raw)
    y_lo, y_hi = np.nanmin(y_raw), np.nanmax(y_raw)
    pad = 0.05 * radius
    for ax in axes:
        ax.set_xlim(min(x_lo, cx - radius) - pad, max(x_hi, cx + radius) + pad)
        ax.set_ylim(min(y_lo, cy - radius) - pad, max(y_hi, cy + radius) + pad)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # match video pixel-row orientation
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def process_file(csv_path):
    name = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.dirname(csv_path)

    df = pd.read_csv(csv_path)
    if df.shape[1] < 3:
        print(f"  SKIP: expected at least 3 columns (timestamp, x_px, y_px), found {df.shape[1]}")
        return

    t = pd.to_numeric(df.iloc[:, 0], errors='coerce').to_numpy(dtype=float)
    x_px = pd.to_numeric(df.iloc[:, 1], errors='coerce').to_numpy(dtype=float)
    y_px = pd.to_numeric(df.iloc[:, 2], errors='coerce').to_numpy(dtype=float)

    # Interpolation needs a strictly increasing axis; fall back to row order otherwise.
    if not (np.all(np.isfinite(t)) and np.all(np.diff(t) > 0)):
        print("  WARNING: timestamps missing or not strictly increasing -- interpolating by row order instead")
        t = np.arange(len(df), dtype=float)

    finite_xy = np.isfinite(x_px) & np.isfinite(y_px)
    if finite_xy.sum() < MIN_POINTS_PER_SECTOR * 3:
        print(f"  SKIP: only {int(finite_xy.sum())} valid tracking samples")
        return

    cx, cy, radius, wall_x, wall_y, wall_keep = estimate_arena_circle(x_px[finite_xy], y_px[finite_xy])
    scale_cm_per_px = ARENA_DIAMETER_CM / (2.0 * radius)
    margin_px = WALL_MARGIN_CM / scale_cm_per_px

    dist = np.hypot(x_px - cx, y_px - cy)
    outside = finite_xy & (dist > radius + margin_px)
    bad_mask = outside | ~finite_xy
    print(f"  fitted wall: center=({cx:.1f}, {cy:.1f}) px, radius={radius:.1f} px "
          f"(fit to {int(wall_keep.sum())} of {len(wall_x)}/{N_SECTORS} sampled sectors), "
          f"scale={scale_cm_per_px:.5f} cm/px")
    print(f"  {int(outside.sum())}/{len(df)} points outside the arena, "
          f"{int((~finite_xy).sum())} missing -> {int(bad_mask.sum())} interpolated")

    if bad_mask.all():
        print("  SKIP: no reliable points left to interpolate from")
        return

    x_corr, y_corr = interpolate_flagged_points(t, x_px, y_px, bad_mask)

    out_df = df.copy()
    out_df.iloc[:, 1] = x_corr
    out_df.iloc[:, 2] = y_corr
    if RECOMPUTE_CM and df.shape[1] >= 5:
        out_df.iloc[:, 3] = (x_corr - (cx - radius)) * scale_cm_per_px
        out_df.iloc[:, 4] = (y_corr - (cy - radius)) * scale_cm_per_px
    out_df[INTERPOLATED_COLUMN] = bad_mask

    out_path = os.path.join(out_dir, f"{name}{OUTPUT_SUFFIX}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")

    if MAKE_DIAGNOSTIC_PLOT:
        plot_path = os.path.join(out_dir, f"{name}{OUTPUT_SUFFIX}_check.png")
        plot_check(x_px, y_px, x_corr, y_corr, bad_mask, (cx, cy, radius),
                   (wall_x, wall_y, wall_keep), margin_px, name, plot_path)
        print(f"  check image -> {plot_path}")


def main():
    file_names = normalize_file_names(FILE_NAMES)
    found = find_named_files(ROOT_DIR, file_names)

    missing = [n for n, paths in found.items() if not paths]
    if missing:
        print(f"WARNING: not found under {ROOT_DIR}: {', '.join(missing)}")

    csv_paths = []
    for n, paths in found.items():
        if len(paths) > 1:
            print(f"WARNING: '{n}' found in {len(paths)} folders -- all copies will be processed")
        csv_paths.extend(sorted(paths))

    for i, csv_path in enumerate(csv_paths, start=1):
        print(f"[{i}/{len(csv_paths)}] {csv_path}")
        try:
            process_file(csv_path)
        except Exception as exc:
            print(f"  ERROR: {exc}")


if __name__ == '__main__':
    main()
