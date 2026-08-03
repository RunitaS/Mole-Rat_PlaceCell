"""
Removes all files containing 'CSC7' in their name from ROOT_DIRECTORY,
including all its subdirectories.
"""

import os
import sys

ROOT_DIRECTORY = "X:/NMR_group_data/Runita/Data/Ephys_Data/AllSortedData/Tetrode"


def remove_csc7_files(directory):
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        sys.exit(1)

    removed = 0
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if "CSC7" in name:
                path = os.path.join(root, name)
                os.remove(path)
                print(f"Deleted: {path}")
                removed += 1

    if removed == 0:
        print(f"No files containing 'CSC7' found in {directory}")
    else:
        print(f"Removed {removed} file(s).")


if __name__ == "__main__":
    remove_csc7_files(ROOT_DIRECTORY)
