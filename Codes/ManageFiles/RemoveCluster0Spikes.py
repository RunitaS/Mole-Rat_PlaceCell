import os
import shutil
import sys

import numpy as np
from openpyxl import Workbook

# ---- Settings ----
ROOT_DIR = r"X:\NMR_group_data\Runita\Data\Ephys_Data\AllSortedData\Tetrode"   # <-- change this
DRY_RUN = True          # True: only report what would be removed. Set False to actually rewrite files.
BACKUP_ROOT = None      # e.g. r"X:\...\Tetrode_backup_before_cluster0_removal" -> originals copied here
                        # (mirroring the folder structure under ROOT_DIR) before being rewritten. None = no backup.
SAVE_XLSX = True        # write a per-file report into ROOT_DIR

# Neuralynx .ntt layout: 16384-byte (16 KB) text header, then 304-byte spike records.
HEADER_BYTES = 16384
RECORD_BYTES = 304

ntt_dtype = np.dtype([
    ('timestamp',   '<u8'),
    ('sc_number',   '<u4'),
    ('cell_number', '<u4'),
    ('params',      '<u4', (8,)),
    ('waveforms',   '<i2', (32, 4)),
])
assert ntt_dtype.itemsize == RECORD_BYTES


def remove_cluster0(fpath):
    """Strip every record with cell_number == 0 from one .ntt file.
    Returns (n_total, n_removed, n_kept)."""
    size = os.path.getsize(fpath)
    if size < HEADER_BYTES:
        raise ValueError(f"file smaller than header ({size} bytes)")
    if (size - HEADER_BYTES) % RECORD_BYTES != 0:
        raise ValueError(f"data section ({size - HEADER_BYTES} bytes) is not a multiple of "
                         f"{RECORD_BYTES}; file may be truncated/corrupt - left untouched")

    with open(fpath, "rb") as f:
        header = f.read(HEADER_BYTES)
        records = np.fromfile(f, dtype=ntt_dtype)

    keep = records['cell_number'] != 0
    n_total = records.size
    n_kept = int(keep.sum())
    n_removed = n_total - n_kept

    if n_removed == 0 or DRY_RUN:
        return n_total, n_removed, n_kept

    if BACKUP_ROOT is not None:
        backup_path = os.path.join(BACKUP_ROOT, os.path.relpath(fpath, ROOT_DIR))
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copy2(fpath, backup_path)

    # Write to a temp file next to the original, then swap it in, so a crash
    # mid-write never leaves a half-written .ntt behind.
    tmp_path = fpath + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(header)
        records[keep].tofile(f)
    os.replace(tmp_path, fpath)

    return n_total, n_removed, n_kept


def process_root(root_dir):
    if not os.path.isdir(root_dir):
        sys.exit(f"Not a valid directory: {root_dir}")

    rows = []             # (folder, file, n_total, n_removed, n_kept)
    failed = []           # (path, error)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if BACKUP_ROOT is not None and os.path.abspath(dirpath).startswith(os.path.abspath(BACKUP_ROOT)):
            dirnames[:] = []   # never touch the backup copies if they live under ROOT_DIR
            continue
        for fname in sorted(f for f in filenames if f.lower().endswith(".ntt")):
            fpath = os.path.join(dirpath, fname)
            try:
                n_total, n_removed, n_kept = remove_cluster0(fpath)
                rows.append((dirpath, fname, n_total, n_removed, n_kept))
                if n_removed:
                    verb = "would remove" if DRY_RUN else "removed"
                    print(f"{fpath}: {verb} {n_removed}/{n_total} cluster-0 spikes ({n_kept} left)")
            except Exception as e:
                print(f"FAILED: {fpath} -> {e}")
                failed.append((fpath, str(e)))

    return rows, failed


if __name__ == "__main__":
    print(f"{'DRY RUN - no files will be modified' if DRY_RUN else 'LIVE RUN - files will be rewritten'}")
    print(f"Root: {ROOT_DIR}\n")

    rows, failed = process_root(ROOT_DIR)

    n_files_changed = sum(1 for r in rows if r[3] > 0)
    n_spikes_removed = sum(r[3] for r in rows)
    now_empty = [r for r in rows if r[3] > 0 and r[4] == 0]

    print(f"\n=== Files left with ZERO spikes after removal ({len(now_empty)}) ===")
    for d, f, *_ in now_empty:
        print(os.path.join(d, f))

    print(f"\nChecked {len(rows)} .ntt files. "
          f"{'Would modify' if DRY_RUN else 'Modified'} {n_files_changed} files, "
          f"{n_spikes_removed} cluster-0 spikes in total. Failed: {len(failed)}.")

    if SAVE_XLSX:
        tag = "DRYRUN" if DRY_RUN else "APPLIED"
        out_xlsx = os.path.join(ROOT_DIR, f"Cluster0_removal_report_{tag}.xlsx")
        wb = Workbook()

        ws = wb.active
        ws.title = "Per_file"
        ws.append(["Folder", "FileName", "N_spikes_before", "N_cluster0_removed", "N_spikes_after"])
        for r in rows:
            ws.append(list(r))

        ws_fail = wb.create_sheet("Failed")
        ws_fail.append(["FullPath", "Error"])
        for p, e in failed:
            ws_fail.append([p, e])

        wb.save(out_xlsx)
        print(f"Saved: {out_xlsx}")
