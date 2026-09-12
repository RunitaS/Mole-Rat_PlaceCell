"""
Substitute noisy tetrode LFP channels with a copy of their nearest clean
neighbour's .ncs file, so that every tetrode's spike (.ntt) data can be
correlated with an LFP channel (either its own or a substituted neighbour's).

For every session folder found under ROOT_DIR, this script copies each
clean-channel source .ncs file and saves the copy under the noisy channel's
filename. The original clean-channel files are left untouched.
"""

import os
import shutil

# ==== USER CONFIGURATION =====================================================
# Hardcode the root folder to search. All subfolders and sub-subfolders will
# be scanned for session folders containing the clean-channel .ncs files below.
ROOT_DIR = r"X:/NMR_group_data/Runita/Analysis/Thesis/Data/Fa5834"  # <-- EDIT THIS PATH

# Mapping of {destination filename (noisy channel to fill in): source filename
# (clean channel to copy from)}, applied within every session folder.
COPY_MAP = {
     "CSC5ch2.ncs": "CSC8ch2.ncs",
     "CSC6ch2.ncs": "CSC8ch2.ncs",
     "CSC4ch2.ncs": "CSC1ch2.ncs",
     "CSC3ch2.ncs": "CSC2ch2.ncs",
     "CSC7ch2.ncs": "CSC2ch2.ncs",
 }

# COPY_MAP = {
#     "CSC5ch2.ncs": "CSC8ch2.ncs",
#     "CSC6ch2.ncs": "CSC8ch2.ncs",
#     "CSC4ch2.ncs": "CSC8ch2.ncs",
#     "CSC3ch2.ncs": "CSC8ch2.ncs",
#     "CSC7ch2.ncs": "CSC8ch2.ncs",
# }

DRY_RUN = False  # Set to False to actually copy files after checking the preview output
OVERWRITE_EXISTING = True  # Set to False to skip destinations that already exist
# ==============================================================================


def process_session(dirpath, filenames):
    """Apply COPY_MAP within a single folder. Returns number of files copied."""
    filenames_set = set(filenames)
    copied = 0

    for dest_name, src_name in COPY_MAP.items():
        if src_name not in filenames_set:
            continue  # this folder isn't a session with the needed clean source file

        src_path = os.path.join(dirpath, src_name)
        dest_path = os.path.join(dirpath, dest_name)

        if os.path.isfile(dest_path) and not OVERWRITE_EXISTING:
            print(f"  SKIP (already exists): {dest_path}")
            continue

        if DRY_RUN:
            print(f"  [DRY RUN] Would copy {src_path} -> {dest_path}")
        else:
            shutil.copy2(src_path, dest_path)
            print(f"  Copied: {src_path} -> {dest_path}")
        copied += 1

    return copied


def main():
    if not os.path.isdir(ROOT_DIR):
        raise FileNotFoundError(f"Root directory not found: {ROOT_DIR}")

    session_count = 0
    total_copied = 0

    for dirpath, _, filenames in os.walk(ROOT_DIR):
        ncs_files = [f for f in filenames if f.lower().endswith(".ncs")]
        if not ncs_files:
            continue

        copied = process_session(dirpath, filenames)
        if copied:
            print(f"Session folder: {dirpath} ({copied} file(s) {'to be ' if DRY_RUN else ''}copied)")
            session_count += 1
            total_copied += copied

    mode = "DRY RUN - " if DRY_RUN else ""
    print(f"\n{mode}Done. {total_copied} file(s) across {session_count} session folder(s).")
    if DRY_RUN:
        print("No files were modified. Set DRY_RUN = False to perform the copies.")


if __name__ == "__main__":
    main()
