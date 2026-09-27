import os
import sys

from openpyxl import Workbook

# ---- Settings ----
TARGET = r"X:\NMR_group_data\Runita\Data\Ephys_Data\AllSortedData\Tetrode"   # <-- a single .ntt file OR a root folder
MIN_SPIKES = 30         # files with fewer spikes than this are deleted
DRY_RUN = True          # True: only report what would be deleted. Set False to actually delete files.
SAVE_XLSX = True        # write a per-file report (only when TARGET is a folder)

# Neuralynx .ntt layout: 16384-byte (16 KB) text header, then 304-byte spike records.
HEADER_BYTES = 16384
RECORD_BYTES = 304


def count_spikes(fpath):
    """Total number of spike records in one .ntt file."""
    size = os.path.getsize(fpath)
    if size < HEADER_BYTES:
        raise ValueError(f"file smaller than header ({size} bytes)")
    if (size - HEADER_BYTES) % RECORD_BYTES != 0:
        raise ValueError(f"data section ({size - HEADER_BYTES} bytes) is not a multiple of "
                         f"{RECORD_BYTES}; file may be truncated/corrupt - left untouched")
    return (size - HEADER_BYTES) // RECORD_BYTES


def check_and_delete(fpath):
    """Count spikes in fpath and delete it if there are fewer than MIN_SPIKES.
    Returns (n_spikes, deleted)."""
    n = count_spikes(fpath)
    delete = n < MIN_SPIKES
    if delete and not DRY_RUN:
        os.remove(fpath)
    return n, delete


def process_root(root_dir):
    rows = []             # (folder, file, n_spikes, deleted)
    failed = []           # (path, error)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in sorted(f for f in filenames if f.lower().endswith(".ntt")):
            fpath = os.path.join(dirpath, fname)
            try:
                n, deleted = check_and_delete(fpath)
                rows.append((dirpath, fname, n, deleted))
                if deleted:
                    verb = "would delete" if DRY_RUN else "deleted"
                    print(f"{fpath}: {n} spikes -> {verb}")
            except Exception as e:
                print(f"FAILED: {fpath} -> {e}")
                failed.append((fpath, str(e)))

    return rows, failed


if __name__ == "__main__":
    print(f"{'DRY RUN - no files will be deleted' if DRY_RUN else 'LIVE RUN - files will be deleted'}")
    print(f"Target: {TARGET}   (delete if < {MIN_SPIKES} spikes)\n")

    if os.path.isfile(TARGET):
        if not TARGET.lower().endswith(".ntt"):
            sys.exit(f"Not an .ntt file: {TARGET}")
        n, deleted = check_and_delete(TARGET)
        verb = ("would be deleted" if DRY_RUN else "deleted") if deleted else "kept"
        print(f"{TARGET}: {n} spikes -> {verb}")
        sys.exit(0)

    if not os.path.isdir(TARGET):
        sys.exit(f"Not a valid file or directory: {TARGET}")

    rows, failed = process_root(TARGET)
    n_deleted = sum(1 for r in rows if r[3])

    print(f"\nChecked {len(rows)} .ntt files. "
          f"{'Would delete' if DRY_RUN else 'Deleted'} {n_deleted} files with < {MIN_SPIKES} spikes. "
          f"Failed: {len(failed)}.")

    if SAVE_XLSX:
        tag = "DRYRUN" if DRY_RUN else "APPLIED"
        out_xlsx = os.path.join(TARGET, f"LowSpike_ntt_deletion_report_{tag}.xlsx")
        wb = Workbook()

        ws = wb.active
        ws.title = "Per_file"
        ws.append(["Folder", "FileName", "N_spikes", f"Deleted (<{MIN_SPIKES})"])
        for r in rows:
            ws.append(list(r))

        ws_fail = wb.create_sheet("Failed")
        ws_fail.append(["FullPath", "Error"])
        for p, e in failed:
            ws_fail.append([p, e])

        wb.save(out_xlsx)
        print(f"Saved: {out_xlsx}")
