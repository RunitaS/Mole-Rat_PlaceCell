# -*- coding: utf-8 -*-
"""
Batch-generate occupancy maps from animal tracking .csv files.

Walks ROOT_DIR (including all subfolders and sub-subfolders), finds every
.csv tracking file, and plots an occupancy map from:
    Column A (index 0) -> timestamp, msec
    Column D (index 3) -> x position, cm
    Column E (index 4) -> y position, cm

Each occupancy map is saved as a .png in OUTPUT_DIR, named after the .csv
file it was generated from (e.g. Session1.csv -> Session1.png).
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe for batch/headless runs
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ── USER INPUT ──────────────────────────────────────────────────────────────
ROOT_DIR = r'X:/NMR_group_data/Runita/AllData_Backup/AllSortedData/Tetrode'  # root folder to search recursively
OUTPUT_DIR = r'C:/Runita/NMR/analysis/AllSort_Results/OccupancyMaps/CorrectionTest'  # all .png maps are saved here (flat, not mirrored)

BIN_SIZE_CM = 2       # spatial bin edge length, cm
SMOOTHING_SIGMA = 1.5   # Gaussian smoothing sigma, in bins; set to 0 to disable
DPI = 300
# ─────────────────────────────────────────────────────────────────────────────


def find_csv_files(root_dir):
    print(f"Scanning {root_dir} for .csv files ...", flush=True)
    csv_paths = []
    folders_scanned = 0
    t_start = time.time()
    t_last_print = t_start

    for folder, _dirnames, filenames in os.walk(root_dir):
        folders_scanned += 1
        for f in filenames:
            if f.lower().endswith('.csv'):
                csv_paths.append(os.path.join(folder, f))

        # Network/slow drives can take a while with no results yet -- print a
        # heartbeat every few seconds so this doesn't look hung.
        now = time.time()
        if now - t_last_print >= 5:
            print(f"  ... still scanning ({folders_scanned} folders checked, "
                  f"{len(csv_paths)} .csv found so far, current: {folder})", flush=True)
            t_last_print = now

    print(f"Scan complete in {time.time() - t_start:.1f}s: "
          f"{folders_scanned} folders checked, {len(csv_paths)} .csv file(s) found", flush=True)
    return sorted(csv_paths)


def load_tracking(csv_path):
    df = pd.read_csv(csv_path)

    t = pd.to_numeric(df.iloc[:, 0], errors='coerce').to_numpy()  # Column A: msec
    x = pd.to_numeric(df.iloc[:, 3], errors='coerce').to_numpy()  # Column D: cm
    y = pd.to_numeric(df.iloc[:, 4], errors='coerce').to_numpy()  # Column E: cm

    valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    return t[valid], x[valid], y[valid]


def compute_occupancy_map(t, x, y, bin_size_cm, smoothing_sigma):
    x_edges = np.arange(x.min(), x.max() + bin_size_cm, bin_size_cm)
    y_edges = np.arange(y.min(), y.max() + bin_size_cm, bin_size_cm)

    frame_counts, x_edges, y_edges = np.histogram2d(x, y, bins=[x_edges, y_edges])
    unvisited = frame_counts == 0  # raw occupancy, before smoothing spreads counts

    # Convert frame counts to seconds spent per bin, using the median sample
    # interval so the map does not depend on an assumed frame rate.
    dt_sec = np.median(np.diff(t)) / 1000.0
    occ_map_sec = frame_counts * dt_sec

    if smoothing_sigma > 0:
        occ_map_sec = gaussian_filter(occ_map_sec, sigma=smoothing_sigma)

    return occ_map_sec, x_edges, y_edges, unvisited


def plot_occupancy_map(occ_map_sec, x_edges, y_edges, unvisited, title, out_path, dpi):
    fig, ax = plt.subplots(figsize=(7, 6))

    cmap = matplotlib.colormaps['viridis'].copy()
    cmap.set_bad('white')  # unoccupied bins
    masked_map = np.ma.masked_where(unvisited, occ_map_sec)

    im = ax.imshow(
        masked_map.T, origin='lower',
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap=cmap, aspect='equal', interpolation='nearest',
    )
    plt.colorbar(im, ax=ax, label='Occupancy (s)')

    ax.set_title(title)
    ax.set_xlabel('X position (cm)')
    ax.set_ylabel('Y position (cm)')

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_files = find_csv_files(ROOT_DIR)

    if not csv_files:
        print(f"No .csv files found under {ROOT_DIR}")
        return

    print(f"Found {len(csv_files)} .csv file(s) under {ROOT_DIR}", flush=True)

    n_total = len(csv_files)
    n_ok, n_skip, n_err = 0, 0, 0
    run_start = time.time()

    for i, csv_path in enumerate(csv_files, start=1):
        csv_name = os.path.splitext(os.path.basename(csv_path))[0]
        out_path = os.path.join(OUTPUT_DIR, csv_name + '.png')
        t0 = time.time()

        print(f"[{i}/{n_total}] Processing {csv_path} ...", flush=True)

        try:
            t, x, y = load_tracking(csv_path)
            if len(t) < 2:
                print(f"[{i}/{n_total}] SKIP: fewer than 2 valid tracking samples", flush=True)
                n_skip += 1
                continue

            occ_map_sec, x_edges, y_edges, unvisited = compute_occupancy_map(
                t, x, y, BIN_SIZE_CM, SMOOTHING_SIGMA
            )

            if os.path.exists(out_path):
                print(f"[{i}/{n_total}] WARN: {out_path} already exists and will be "
                      f"overwritten (duplicate .csv filename across folders)", flush=True)

            plot_occupancy_map(occ_map_sec, x_edges, y_edges, unvisited, csv_name, out_path, DPI)
            n_ok += 1
            print(f"[{i}/{n_total}] OK ({time.time() - t0:.1f}s) -> {out_path}", flush=True)

        except Exception as exc:
            n_err += 1
            print(f"[{i}/{n_total}] ERROR: {csv_path}: {exc}", flush=True)

    print(f"Done in {time.time() - run_start:.1f}s: "
          f"{n_ok} saved, {n_skip} skipped, {n_err} errored (of {n_total} total)", flush=True)


if __name__ == '__main__':
    main()
