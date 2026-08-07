#!/usr/bin/env python3
"""
Concatenate spike-sorted NTT files by tetrode, assigning fresh cluster IDs.

Within each folder, every .ntt file whose name starts with a tetrode
prefix (TT1, TT2, ..., TT10, ...) is grouped by that prefix alone —
regardless of what follows in the filename. Sorted resorts leave behind
filenames with repeated "_SS_NN" suffixes (e.g. TT6_SS_03_SS_01.ntt,
TT1_0001_SS_19_SS_19.ntt), so the old approach of parsing a cluster
number out of the filename's *last* _SS_NN suffix would split a single
tetrode's files into several bogus groups (e.g. "TT6_SS_03", "TT6_SS_07",
"TT6" all treated as different tetrodes). That's fixed here: grouping
is purely by leading "TTn" prefix, and every file in the group gets a
brand-new sequential cluster number (1, 2, 3, ...) based on a natural
sort of the filenames — the original embedded numbers are not reused.

Usage
-----
  python concat_ntt.py     # INPUT_ROOT / OUTPUT_ROOT below are used as-is

Only .ntt files starting with a "TT<number>" prefix are processed; other
extensions (.ncs, .nvt, …) are ignored. INPUT_ROOT is scanned recursively
and is never written to; the OUTPUT_ROOT subtree (if nested inside
INPUT_ROOT) is skipped during the scan so re-running never picks up
already-concatenated output as input.

Output files are written under OUTPUT_ROOT (set below), in a subfolder
named after the last 4 components of the source folder's path, joined
with underscores. For example, files found in
    INPUT_ROOT\...\Tetrode\Fa1059\Open\Day9\2Rotate
are written to
    OUTPUT_ROOT\Fa1059_Open_Day9_2Rotate\TT1.ntt
Running the script twice is safe: existing output files are skipped.
"""

import os
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

# ── configuration ──────────────────────────────────────X:\NMR_group_data\Runita\Data\Ephys_Data\AllSortedData\Tetrode───────────────────────
INPUT_ROOT  = r"X:/NMR_group_data/Runita/Data/Ephys_Data/AllSortedData/Tetrode"
OUTPUT_ROOT = r"X:/NMR_group_data/Runita/Data/Ephys_Data/AllSortedData/Tetrode/Concat"
# ─────────────────────────────────────────────────────────────────────────────

# ── Neuralynx NTT binary layout ──────────────────────────────────────────────
HEADER_BYTES   = 16384   # ASCII text header, fixed size
RECORD_BYTES   = 304     # bytes per spike record
#   offset  0 :  8 bytes  uint64   TimeStamp  (µs)
#   offset  8 :  4 bytes  uint32   ScNumber   (acquisition entity)
#   offset 12 :  4 bytes  uint32   CellNumber (cluster id)  ← we rewrite this
#   offset 16 : 32 bytes  uint32×8 Features
#   offset 48 :256 bytes  int16×128 Waveforms (4 ch × 32 samples)
CELLNUM_OFFSET = 12
# ─────────────────────────────────────────────────────────────────────────────

TETRODE_PATTERN = re.compile(r'^([Tt][Tt]\d+)')  # leading TT<number>, e.g. TT1, TT10
DIGIT_CHUNK = re.compile(r'(\d+)')


# ── helpers ───────────────────────────────────────────────────────────────────

def tetrode_prefix(filename: str) -> str | None:
    """
    'TT6_SS_03_SS_01.ntt'  →  'TT6'
    'TT1_0001_SS_19_SS_19.ntt'  →  'TT1'
    Returns None if the name does not start with a TT<number> prefix.
    """
    stem = Path(filename).stem
    m = TETRODE_PATTERN.match(stem)
    return m.group(1).upper() if m else None


def natural_sort_key(filepath: Path):
    """Split a filename into text/number chunks so 'SS_2' sorts before 'SS_10'."""
    parts = DIGIT_CHUNK.split(filepath.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def rewrite_cell_numbers(raw: bytes, cluster_id: int) -> bytes:
    """Return a copy of *raw* (N × RECORD_BYTES) with CellNumber set to cluster_id."""
    buf = bytearray(raw)
    n_records = len(buf) // RECORD_BYTES
    for i in range(n_records):
        struct.pack_into('<I', buf, i * RECORD_BYTES + CELLNUM_OFFSET, cluster_id)
    return bytes(buf)


# ── core logic ────────────────────────────────────────────────────────────────

def concatenate_group(entries: list, output_path: Path) -> None:
    """
    Merge a list of (cluster_id, Path) entries into output_path.

    Parameters
    ----------
    entries     : sorted list of (cluster_id, filepath)
    output_path : destination .ntt file
    """
    # Borrow the header from the first file
    first_path = entries[0][1]
    with open(first_path, 'rb') as fh:
        header = fh.read(HEADER_BYTES)
        if len(header) < HEADER_BYTES:
            raise ValueError(f"Truncated header in {first_path}")

    combined = bytearray()
    total_spikes = 0

    for cluster_id, filepath in entries:
        with open(filepath, 'rb') as fh:
            fh.seek(HEADER_BYTES)
            raw = fh.read()

        n_spikes = len(raw) // RECORD_BYTES
        if n_spikes == 0:
            print(f"    [WARN] {filepath.name} is empty, skipping.")
            continue

        # Trim any trailing incomplete record
        raw = raw[: n_spikes * RECORD_BYTES]
        combined.extend(rewrite_cell_numbers(raw, cluster_id))
        total_spikes += n_spikes
        print(f"    cluster {cluster_id:3d}  →  {n_spikes:7,d} spikes   ({filepath.name})")

    with open(output_path, 'wb') as fh:
        fh.write(header)
        fh.write(combined)

    print(f"  ✓  {output_path.name}  "
          f"[{len(entries)} clusters, {total_spikes:,} spikes total]\n")


def session_label(dirpath: Path) -> str:
    """
    Build a label from the last 4 components of *dirpath*.

    '...\\Tetrode\\Fa1059\\Open\\Day9\\2Rotate'  →  'Fa1059_Open_Day9_2Rotate'
    """
    parts = dirpath.parts[-4:]
    return "_".join(parts)


def process_tree(root: str | Path, output_root: str | Path = OUTPUT_ROOT) -> None:
    root = Path(root)
    output_root = Path(output_root)
    if not root.is_dir():
        sys.exit(f"ERROR: '{root}' is not a directory.")

    found_any = False

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)

        # Never descend into the output tree (avoids re-ingesting our own output).
        try:
            dirpath.relative_to(output_root)
            dirnames[:] = []
            continue
        except ValueError:
            pass

        # Group NTT files by leading tetrode prefix (TT1, TT2, ...), ignoring
        # everything that follows in the filename.
        groups: dict[str, list] = defaultdict(list)
        for fname in filenames:
            if not fname.lower().endswith('.ntt'):
                continue                         # ignore .ncs, .nvt, etc.
            prefix = tetrode_prefix(fname)
            if prefix is None:
                continue                         # not a TT-prefixed file
            groups[prefix].append(dirpath / fname)

        if not groups:
            continue

        label = session_label(dirpath)
        out_dir = output_root / label
        out_dir.mkdir(parents=True, exist_ok=True)

        for prefix, filepaths in sorted(groups.items()):
            found_any = True
            output_path = out_dir / f"{prefix}.ntt"

            print(f"[{label}]  {prefix}  ({len(filepaths)} file(s))")

            if output_path.exists():
                print(f"  [SKIP] {output_path.name} already exists.\n")
                continue

            # Order the source files deterministically, then hand out brand-new
            # sequential cluster numbers (1, 2, 3, ...) in that order.
            filepaths.sort(key=natural_sort_key)
            entries = [(i + 1, fp) for i, fp in enumerate(filepaths)]
            concatenate_group(entries, output_path)

    if not found_any:
        print("No TT-prefixed .ntt files were found.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"\nInput root:  {INPUT_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}\n{'─' * 60}\n")
    process_tree(INPUT_ROOT, OUTPUT_ROOT)
    print("Done.")
