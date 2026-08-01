# -*- coding: utf-8 -*-
"""
Walks ROOT_DIR and all its subfolders, converts every .csv file found into
an .xlsx file (same name, same folder), then deletes the original .csv.

Data is read as raw text (dtype=str, no NA conversion) so every cell is
written to the .xlsx exactly as it appears in the .csv — no numeric
rounding, no date reformatting, no "007" -> 7 type coercion.
"""

import os
import sys
import time
import pandas as pd

#%% Settings

# Set this to your root folder
ROOT_DIR = r'X:/NMR_group_data/Runita/Data/Ephys_Data/AllSortedData'  

# Replaces .csv with .xlsx (deletes the .csv after conversion)
DELETE_ORIGINAL_CSV = False  

#%% Validate ROOT_DIR before doing anything

abs_root = os.path.abspath(ROOT_DIR)
print(f'ROOT_DIR resolves to: {abs_root}')

if not os.path.exists(ROOT_DIR):
    sys.exit(
        f"ERROR: ROOT_DIR does not exist or is not reachable:\n"
        f"  {ROOT_DIR}\n"
        f"Check that the path is spelled correctly and, if it's a mapped/network "
        f"drive (e.g. X:), that it is currently connected."
    )

if not os.path.isdir(ROOT_DIR):
    sys.exit(f"ERROR: ROOT_DIR exists but is not a directory:\n  {ROOT_DIR}")

#%% Find all .csv files under ROOT_DIR

csv_files = []
n_dirs_scanned = 0
t_start = time.time()

for root, dirs, files in os.walk(ROOT_DIR):
    n_dirs_scanned += 1
    for fname in files:
        if fname.lower().endswith('.csv'):
            csv_files.append(os.path.join(root, fname))

    if n_dirs_scanned % 100 == 0:
        print(f'  ...scanned {n_dirs_scanned} folders, {len(csv_files)} .csv found so far '
              f'({time.time() - t_start:.0f}s elapsed) - currently in: {root}')

print(f'Found {len(csv_files)} .csv file(s) under {ROOT_DIR} '
      f'(scanned {n_dirs_scanned} folders in {time.time() - t_start:.0f}s)')

if not csv_files:
    # The path is valid but empty of CSVs
    top_level = os.listdir(ROOT_DIR)
    print('No .csv files were found. Top-level contents of ROOT_DIR:')
    for item in top_level:
        print(f'  {item}')
    sys.exit(0)

#%% Convert each .csv -> .xlsx

n_ok, n_fail = 0, 0

for csv_path in csv_files:
    xlsx_path = os.path.splitext(csv_path)[0] + '.xlsx'
    rel = os.path.relpath(csv_path, ROOT_DIR)

    try:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding='latin1')

        df.to_excel(xlsx_path, index=False, engine='openpyxl')

        if DELETE_ORIGINAL_CSV:
            os.remove(csv_path)

        print(f'  OK: {rel} -> {os.path.basename(xlsx_path)}')
        n_ok += 1

    except Exception as e:
        print(f'  FAIL: {rel} - {e}')
        n_fail += 1

print(f'\nDone. {n_ok} converted, {n_fail} failed.')