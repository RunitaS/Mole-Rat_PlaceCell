#!/usr/bin/env python3
"""
Concatenate spike-sorted NTT files by tetrode, preserving cluster identity.

Files named  TT1_SS_01.ntt, TT1_SS_02.ntt, ..., TT1_SS_N.ntt
are merged into  TT1.ntt  where every spike carries the cluster number
extracted from the source filename (_SS_01 → cluster 1, _SS_02 → cluster 2, …).
Spikes whose original CellNumber is 0 (unsorted/discarded) are dropped, and
_SS_00 files are skipped entirely, so no cluster 0 spikes reach the output.

Usage
-----
  python concat_ntt.py     # INPUT_ROOT / OUTPUT_ROOT below are used as-is

Only files matching the *_SS_NN.ntt pattern are processed; all other
extensions (.ncs, .nvt, …) and already-concatenated .ntt files are ignored.
INPUT_ROOT is scanned recursively and is never written to.

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

SS_PATTERN = re.compile(r'^(.+)_[Ss][Ss]_(\d+)$')  # matches TT1_SS_01, etc.


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_ss_name(filename: str):
    """
    'TT1_SS_03.ntt'  →  ('TT1', 3)
    Returns (None, None) if the name does not match the SS pattern.
    """
    stem = Path(filename).stem
    m = SS_PATTERN.match(stem)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def drop_cluster0_and_relabel(raw: bytes, cluster_id: int) -> tuple[bytes, int]:
    """
    Drop records of *raw* (N × RECORD_BYTES) whose original CellNumber is 0,
    then set CellNumber to cluster_id on the rest.
    Returns (kept records as bytes, number of records dropped).
    """
    records = np.frombuffer(raw, dtype=NTT_DTYPE)
    kept = records[records['cell_number'] != 0]  # drop unsorted/discarded cluster 0
    kept['cell_number'] = cluster_id
    return kept.tobytes(), len(records) - len(kept)


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
    total_dropped = 0
    n_clusters = 0

    for cluster_id, filepath in entries:
        if cluster_id == 0:
            print(f"    [SKIP] {filepath.name} is cluster 0 (unsorted), skipping.")
            continue

        with open(filepath, 'rb') as fh:
            fh.seek(HEADER_BYTES)
            raw = fh.read()

        n_records = len(raw) // RECORD_BYTES
        if n_records == 0:
            print(f"    [WARN] {filepath.name} is empty, skipping.")
            continue

        # Trim any trailing incomplete record
        raw = raw[: n_records * RECORD_BYTES]
        kept, n_dropped = drop_cluster0_and_relabel(raw, cluster_id)
        n_spikes = n_records - n_dropped
        total_dropped += n_dropped
        if n_spikes == 0:
            print(f"    [WARN] {filepath.name} has only cluster 0 spikes "
                  f"({n_dropped:,} dropped), skipping.")
            continue

        combined.extend(kept)
        total_spikes += n_spikes
        n_clusters += 1
        print(f"    cluster {cluster_id:3d}  →  {n_spikes:7,d} spikes   "
              f"({n_dropped:,} cluster-0 dropped)   ({filepath.name})")

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

    for dirpath, _, filenames in os.walk(root):
        dirpath = Path(dirpath)

        # Group matching NTT files by tetrode base name
        groups: dict[str, list] = defaultdict(list)
        for fname in filenames:
            if not fname.lower().endswith('.ntt'):
                continue                         # ignore .ncs, .nvt, etc.
            base, cluster_id = parse_ss_name(fname)
            if base is None:
                continue                         # not an SS-patterned file
            groups[base].append((cluster_id, dirpath / fname))

        if not groups:
            continue

        label = session_label(dirpath)
        out_dir = output_root / label
        out_dir.mkdir(parents=True, exist_ok=True)

        for base_name, entries in sorted(groups.items()):
            found_any = True
            output_path = out_dir / f"{base_name}.ntt"

            print(f"[{label}]  {base_name}  ({len(entries)} file(s))")

            if output_path.exists():
                print(f"  [SKIP] {output_path.name} already exists.\n")
                continue

            # Sort by cluster number before merging
            entries.sort(key=lambda x: x[0])
            concatenate_group(entries, output_path)

    if not found_any:
        print("No files matching the *_SS_NN.ntt pattern were found.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"\nInput root:  {INPUT_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}\n{'─' * 60}\n")
    process_tree(INPUT_ROOT, OUTPUT_ROOT)
    print("Done.")
