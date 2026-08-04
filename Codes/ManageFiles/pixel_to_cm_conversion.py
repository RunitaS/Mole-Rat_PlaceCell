# -*- coding: utf-8 -*-
"""
Batch-convert pixel tracking coordinates (columns A/B/C = timestamp/x/y) to cm.

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
import glob
import pandas as pd

# ── USER INPUT ──────────────────────────────────────────────────────────────
ROOT_DIR = r'X:\NMR_group_data\Runita\Analysis\RIN_Analysis\Characterization'

# Real-world size (cm) of the longer tracked axis, used as the default for
# any subfolder not listed in ARENA_SIZE_CM_BY_FOLDER below.
DEFAULT_ARENA_SIZE_CM = 60

# Per-folder overrides: {subfolder_name: arena_size_cm}
# Add an entry here for any session whose arena size differs from the default
# (e.g. 80 cm arenas).
ARENA_SIZE_CM_BY_FOLDER = {
    # 'ExperimentDay10_NestBuild': 80,
}

OUTPUT_SUFFIX = '_cm'  # output saved as <original_name>_cm.csv in the same subfolder
# ─────────────────────────────────────────────────────────────────────────────


def get_arena_size_cm(folder_name):
    return ARENA_SIZE_CM_BY_FOLDER.get(folder_name, DEFAULT_ARENA_SIZE_CM)


def convert_file(xlsx_path, arena_size_cm):
    df = pd.read_excel(xlsx_path, header=0)

    t_col, x_col, y_col = df.columns[0], df.columns[1], df.columns[2]
    x_px = df[x_col].astype(float)
    y_px = df[y_col].astype(float)

    x_dist_px = x_px.max() - x_px.min()
    y_dist_px = y_px.max() - y_px.min()
    longer_dist_px = max(x_dist_px, y_dist_px)

    scale_cm_per_px = arena_size_cm / longer_dist_px

    df['x_cm'] = x_px * scale_cm_per_px
    df['y_cm'] = y_px * scale_cm_per_px

    out_path = os.path.join(
        os.path.dirname(xlsx_path),
        os.path.splitext(os.path.basename(xlsx_path))[0] + OUTPUT_SUFFIX + '.csv'
    )
    df.to_csv(out_path, index=False)

    print(f"  x_dist_px={x_dist_px:.2f}, y_dist_px={y_dist_px:.2f}, "
          f"arena_size_cm={arena_size_cm}, scale={scale_cm_per_px:.5f} cm/px")
    print(f"  Saved -> {out_path}")


def main():
    subfolders = sorted(
        f for f in glob.glob(os.path.join(ROOT_DIR, '*'))
        if os.path.isdir(f)
    )

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        xlsx_files = glob.glob(os.path.join(folder, '*.xlsx'))
        xlsx_files = [f for f in xlsx_files if not os.path.basename(f).endswith(OUTPUT_SUFFIX + '.xlsx')]

        if not xlsx_files:
            print(f"[{folder_name}] No .xlsx file found, skipping.")
            continue
        if len(xlsx_files) > 1:
            print(f"[{folder_name}] Multiple .xlsx files found, using the first: {xlsx_files[0]}")

        arena_size_cm = get_arena_size_cm(folder_name)
        print(f"[{folder_name}] Converting {os.path.basename(xlsx_files[0])} "
              f"(arena_size_cm={arena_size_cm})")
        convert_file(xlsx_files[0], arena_size_cm)


if __name__ == '__main__':
    main()
