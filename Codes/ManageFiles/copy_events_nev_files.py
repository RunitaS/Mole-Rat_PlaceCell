"""
Copies all 'Events.nev' files from SOURCE_DIRECTORY to DEST_DIRECTORY,
placing each file in the same relative subdirectory it had under the source.

Both directories are expected to hold the same folder tree. A file is only
copied when its matching subfolder already exists under DEST_DIRECTORY;
missing subfolders are reported, not created.

Folder names are matched exactly (case-insensitive). Suffix-insensitive
matching (e.g. '_Geo' / '_Zero') is disabled: IGNORED_FOLDER_SUFFIXES is
empty. Add suffixes back to it to re-enable that behaviour.

Set DRY_RUN = True to only list what would be copied (copies nothing), or
DRY_RUN = False to actually copy the files. It can also be set from the
command line with --dry-run True / --dry-run False.
Files that already exist in the destination are overwritten.
"""

import argparse
import functools
import os
import shutil
import sys

SOURCE_DIRECTORY = "X:/NMR_group_data/Runita/Analysis/Thesis/Corr_Data_SpkQltyFilt/SpikeQualityFilt/QualFiltData_AllCells"
DEST_DIRECTORY = "X:/NMR_group_data/Runita/Analysis/Thesis/Corr_Data_SpkQltyFilt/SpikeQualityFilt_v2/QualFilt_Data"
FILE_NAME = "Events.nev"
IGNORED_FOLDER_SUFFIXES = ()  # Disabled. e.g. ("_Geo", "_Zero") to ignore these suffixes when matching.
DRY_RUN = False  # True: only list what would be copied. False: actually copy.


def base_folder_name(name):
    """Folder name with any ignored suffix removed, lowercased for comparison."""
    lower = name.lower()
    for suffix in IGNORED_FOLDER_SUFFIXES:
        if lower.endswith(suffix.lower()):
            return lower[: -len(suffix)]
    return lower


@functools.lru_cache(maxsize=None)
def list_subdirs(path):
    try:
        return tuple(e.name for e in os.scandir(path) if e.is_dir())
    except OSError:
        return ()


def find_dest_dirs(dest, rel_dir):
    """All folders under dest that match rel_dir, one path component at a time."""
    parts = [] if rel_dir == "." else rel_dir.split(os.sep)
    candidates = [dest]

    for part in parts:
        next_candidates = []
        for parent in candidates:
            exact = os.path.join(parent, part)
            if os.path.isdir(exact):
                next_candidates.append(exact)
                continue
            target = base_folder_name(part)
            next_candidates.extend(
                os.path.join(parent, child)
                for child in list_subdirs(parent)
                if base_folder_name(child) == target
            )
        candidates = next_candidates
        if not candidates:
            break

    return [os.path.normpath(c) for c in candidates]


def copy_events_nev_files(source, dest, dry_run=True):
    for directory in (source, dest):
        if not os.path.isdir(directory):
            print(f"Directory not found: {directory}")
            sys.exit(1)

    copied, overwritten, failed = 0, 0, []

    for root, _dirs, files in os.walk(source):
        for name in files:
            if name.lower() != FILE_NAME.lower():
                continue

            src_path = os.path.join(root, name)
            rel_dir = os.path.relpath(root, source)
            dest_dirs = find_dest_dirs(dest, rel_dir)

            if not dest_dirs:
                expected = os.path.normpath(os.path.join(dest, rel_dir))
                failed.append((src_path, f"no matching folder (expected {expected})"))
                print(f"No matching folder for: {src_path}\n    expected: {expected}")
                continue

            for dest_dir in dest_dirs:
                dest_path = os.path.join(dest_dir, name)
                exists = os.path.exists(dest_path)
                verb, done = ("overwrite", "Overwrote") if exists else ("copy", "Copied")

                if dry_run:
                    print(f"Would {verb}: {src_path}\n        -> {dest_path}")
                else:
                    try:
                        shutil.copy2(src_path, dest_path)
                    except OSError as e:
                        failed.append((src_path, f"copy error to {dest_path}: {e}"))
                        print(f"Failed to copy: {src_path}\n    -> {dest_path}\n    {e}")
                        continue
                    print(f"{done}: {src_path}\n    -> {dest_path}")
                copied += 1
                overwritten += exists

    print()
    action = "Would copy" if dry_run else "Copied"
    print(f"{action} {copied} file(s), of which {overwritten} replaced an existing file.")
    print(f"{len(failed)} file(s) skipped or failed.")

    if failed:
        print("\nSkipped / failed files:")
        for src_path, reason in failed:
            print(f"  {src_path}\n      reason: {reason}")

    if dry_run and copied:
        print("\nDry run only. Set DRY_RUN = False (or pass --dry-run False) to copy them.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Copy '{FILE_NAME}' files into the matching subfolders of another directory.")
    parser.add_argument("source", nargs="?", default=SOURCE_DIRECTORY, help=f"Directory 1 (has the {FILE_NAME} files)")
    parser.add_argument("dest", nargs="?", default=DEST_DIRECTORY, help=f"Directory 2 (missing the {FILE_NAME} files)")
    parser.add_argument("--dry-run", choices=["True", "False"], default=str(DRY_RUN),
                        help=f"True: only list what would be copied. False: actually copy. (default: {DRY_RUN})")
    args = parser.parse_args()

    copy_events_nev_files(args.source, args.dest, dry_run=(args.dry_run == "True"))
