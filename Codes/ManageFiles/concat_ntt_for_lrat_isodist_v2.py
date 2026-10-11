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

Before concatenation, spikes whose original CellNumber is 0 (unsorted /
discarded) are dropped from every file. Files left with no spikes after
this filter are skipped and do not consume a cluster number, so no
cluster 0 spikes reach the output and the new numbering has no gaps.

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
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── configuration ──────────────────────────────────────X:\NMR_group_data\Runita\Data\Ephys_Data\AllSortedData\Tetrode───────────────────────
INPUT_ROOT  = r"X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt"
OUTPUT_ROOT = r"X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFiltConcat"
# ─────────────────────────────────────────────────────────────────────────────

# ── Neuralynx NTT binary layout ──────────────────────────────────────────────
HEADER_BYTES   = 16384   # ASCII text header, fixed size
RECORD_BYTES   = 304     # bytes per spike record
#   offset  0 :  8 bytes  uint64   TimeStamp  (µs)
#   offset  8 :  4 bytes  uint32   ScNumber   (acquisition entity)
#   offset 12 :  4 bytes  uint32   CellNumber (cluster id)  ← we rewrite this
#   offset 16 : 32 bytes  uint32×8 Features
#   offset 48 :256 bytes  int16×128 Waveforms (4 ch × 32 samples)
NTT_DTYPE = np.dtype([
    ('timestamp',   '<u8'),
    ('sc_number',   '<u4'),
    ('cell_number', '<u4'),
    ('params',      '<u4', (8,)),
    ('waveforms',   '<i2', (32, 4)),
])
assert NTT_DTYPE.itemsize == RECORD_BYTES
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


def drop_cluster0(raw: bytes) -> np.ndarray:
    """
    Parse *raw* (N × RECORD_BYTES) and return a writable copy of the records
    whose original CellNumber is non-zero (cluster 0 = unsorted/discarded).
    """
    records = np.frombuffer(raw, dtype=NTT_DTYPE)
    return records[records['cell_number'] != 0]  # boolean indexing → copy


# ── core logic ────────────────────────────────────────────────────────────────

def concatenate_group(filepaths: list, output_path: Path) -> None:
    """
    Merge *filepaths* into output_path, dropping cluster-0 spikes from each
    file first. Files that still have spikes are given brand-new sequential
    cluster numbers (1, 2, 3, ...) in the order given, so numbering has no
    gaps from empty or all-cluster-0 files.

    Parameters
    ----------
    filepaths   : sorted list of source .ntt paths
    output_path : destination .ntt file
    """
    # Borrow the header from the first file
    first_path = filepaths[0]
    with open(first_path, 'rb') as fh:
        header = fh.read(HEADER_BYTES)
        if len(header) < HEADER_BYTES:
            raise ValueError(f"Truncated header in {first_path}")

    combined = bytearray()
    total_spikes = 0
    total_dropped = 0
    n_clusters = 0

    for filepath in filepaths:
        with open(filepath, 'rb') as fh:
            fh.seek(HEADER_BYTES)
            raw = fh.read()

        n_records = len(raw) // RECORD_BYTES
        if n_records == 0:
            print(f"    [WARN] {filepath.name} is empty, skipping.")
            continue

        # Trim any trailing incomplete record
        raw = raw[: n_records * RECORD_BYTES]
        kept = drop_cluster0(raw)
        n_spikes = len(kept)
        n_dropped = n_records - n_spikes
        total_dropped += n_dropped
        if n_spikes == 0:
            print(f"    [WARN] {filepath.name} has only cluster 0 spikes "
                  f"({n_dropped:,} dropped), skipping.")
            continue

        n_clusters += 1
        kept['cell_number'] = n_clusters
        combined.extend(kept.tobytes())
        total_spikes += n_spikes
        print(f"    cluster {n_clusters:3d}  →  {n_spikes:7,d} spikes   "
              f"({n_dropped:,} cluster-0 dropped)   ({filepath.name})")

    if n_clusters == 0:
        print(f"  [SKIP] no non-zero-cluster spikes; {output_path.name} not written.\n")
        return

    with open(output_path, 'wb') as fh:
        fh.write(header)
        fh.write(combined)

    print(f"  ✓  {output_path.name}  "
          f"[{n_clusters} clusters, {total_spikes:,} spikes total, "
          f"{total_dropped:,} cluster-0 spikes dropped]\n")


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

            # Order the source files deterministically; concatenate_group hands
            # out brand-new sequential cluster numbers (1, 2, 3, ...) in that order.
            filepaths.sort(key=natural_sort_key)
            concatenate_group(filepaths, output_path)

    if not found_any:
        print("No TT-prefixed .ntt files were found.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"\nInput root:  {INPUT_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}\n{'─' * 60}\n")
    process_tree(INPUT_ROOT, OUTPUT_ROOT)
    print("Done.")
