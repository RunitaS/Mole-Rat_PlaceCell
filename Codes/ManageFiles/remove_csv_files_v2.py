"""
Removes all .csv files from ROOT_DIRECTORY, including all its subdirectories.

Runs in dry-run mode by default (lists files, deletes nothing).
Pass --delete to actually remove the files.
"""

import argparse
import os
import sys

ROOT_DIRECTORY = "X:/NMR_group_data/Runita/Data/Ephys_Data/AllSortedData/Tetrode"


def remove_csv_files(directory, dry_run=False):
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        sys.exit(1)

    removed = 0
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if name.lower().endswith(".csv"):
                path = os.path.join(root, name)
                if dry_run:
                    print(f"Would delete: {path}")
                else:
                    os.remove(path)
                    print(f"Deleted: {path}")
                removed += 1

    if removed == 0:
        print(f"No .csv files found in {directory}")
    elif dry_run:
        print(f"Found {removed} .csv file(s). Re-run with --delete to remove them.")
    else:
        print(f"Removed {removed} file(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete .csv files under a root directory.")
    parser.add_argument("directory", nargs="?", default=ROOT_DIRECTORY, help="Root directory to search")
    parser.add_argument("--delete", action="store_true", help="Actually delete files (default is dry-run)")
    args = parser.parse_args()

    remove_csv_files(args.directory, dry_run=False)
