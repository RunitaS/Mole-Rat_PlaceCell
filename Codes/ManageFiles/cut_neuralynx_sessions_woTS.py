"""
Walk a root directory tree, and for every folder that contains Neuralynx
recording files (.nev events, .ncs LFP, .ntt spikes), split every .nev, .ncs
and .ntt file in that folder into two halves at the midpoint of the session.
Each half's .nev also gets an extra event "Cut at midpoint TimeStamp <ts>"
(timestamp = mid_ts): the last event of the first half and the first event
of the second half. Any .csv files in the folder are copied unchanged into
both halves.

The midpoint is taken from the folder's .nev events file:
    mid_ts = (first event timestamp + last event timestamp) / 2

Records with TimeStamp < mid_ts go to the first half, records with
TimeStamp >= mid_ts go to the second half (so no record is duplicated).
The halves are written in native Neuralynx format (16 KB header copied
as-is + only the records in that half) into two new subfolders created
inside the source folder:
    <folder name>_Geo1   (first half)
    <folder name>_Geo2   (second half)

e.g. files in folder "1_90" are written to "1_90/1_90_Geo1" and
"1_90/1_90_Geo2".

Requires: numpy

Usage:
    1. Edit ROOT_DIR below.
    2. Run: python cut_neuralynx_sessions_woTS.py

Folders whose name already ends in _Geo1/_Geo2 (output of a previous run)
are skipped automatically.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration -- EDIT THESE
# ---------------------------------------------------------------------------

ROOT_DIR = Path(r"X:\NMR_group_data\Runita\Analysis\Mean_KDE_Open_PascalOldenburg\LinearTrack_GeoMagVsZero\Data_Control")
# Cut files are saved into subfolders created inside each source folder, so
# there is no separate output root to configure.

HALF_SUFFIXES = ("_Geo1", "_Geo2")  # first half, second half
CUT_EXTENSIONS = (".nev", ".ncs", ".ntt")  # file types that get split

HEADER_SIZE = 16384  # standard Neuralynx text header size, all file types
NCS_SAMPLES_PER_RECORD = 512  # fixed for standard Neuralynx CSC records


# ---------------------------------------------------------------------------
# Neuralynx binary record layouts
#
# Every Neuralynx file (.nev, .ncs, .ntt, ...) is a 16 KB ASCII header
# followed by fixed-size binary records. Cutting a file to a time window is
# just: copy the header verbatim, keep only the records whose TimeStamp
# falls inside the window, and write header + kept records back out.
# ---------------------------------------------------------------------------

def read_header_text(path: Path) -> str:
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE)
    return raw.decode("latin-1", errors="ignore")


def header_field(header_text: str, name: str, default=None, cast=int):
    """Pull a '-FieldName value' entry out of a Neuralynx header, if present."""
    for line in header_text.splitlines():
        line = line.strip()
        if line.startswith(f"-{name}"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return cast(parts[1])
                except ValueError:
                    return default
    return default


def ncs_dtype() -> np.dtype:
    return np.dtype([
        ("TimeStamp", "<u8"),
        ("ChannelNumber", "<u4"),
        ("SampleFreq", "<u4"),
        ("NumValidSamples", "<u4"),
        ("Samples", "<i2", (NCS_SAMPLES_PER_RECORD,)),
    ])


def nev_dtype() -> np.dtype:
    return np.dtype([
        ("nstx", "<i2"),
        ("npkt_id", "<i2"),
        ("npkt_data_size", "<i2"),
        ("TimeStamp", "<u8"),
        ("nevent_id", "<i2"),
        ("nttl", "<i2"),
        ("ncrc", "<i2"),
        ("ndummy1", "<i2"),
        ("ndummy2", "<i2"),
        ("dnExtra", "<i4", (8,)),
        ("EventString", "S128"),
    ])


def spike_dtype(header_text: str, num_channels: int) -> np.dtype:
    samples_per_spike = header_field(header_text, "SamplesPerSpike", default=32)
    return np.dtype([
        ("TimeStamp", "<u8"),
        ("ScNumber", "<u4"),
        ("CellNumber", "<u4"),
        ("Params", "<u4", (8,)),
        ("Samples", "<i2", (samples_per_spike, num_channels)),
    ])


def dtype_for_file(path: Path, header_text: str) -> np.dtype:
    ext = path.suffix.lower()
    if ext == ".ncs":
        return ncs_dtype()
    if ext == ".nev":
        return nev_dtype()
    if ext == ".ntt":
        return spike_dtype(header_text, num_channels=4)
    if ext == ".nst":
        return spike_dtype(header_text, num_channels=2)
    if ext == ".nse":
        return spike_dtype(header_text, num_channels=1)
    raise ValueError(f"Unsupported Neuralynx file type: {path}")


def load_records(path: Path):
    header_text = read_header_text(path)
    dtype = dtype_for_file(path, header_text)
    with open(path, "rb") as f:
        header = f.read(HEADER_SIZE)
    records = np.fromfile(path, dtype=dtype, offset=HEADER_SIZE)
    return header, records


def write_cut_file(header: bytes, records: np.ndarray, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header)
        records.tofile(f)
    return len(records)


def nev_midpoint(folder: Path):
    """Return (first_ts, last_ts, mid_ts) from the folder's first .nev file,
    or None if there is no .nev file / fewer than 2 events."""
    nev_files = sorted(folder.glob("*.nev"))
    if not nev_files:
        return None
    _, records = load_records(nev_files[0])
    if len(records) < 2:
        return None
    ts = records["TimeStamp"].astype(np.int64)
    first_ts, last_ts = int(ts.min()), int(ts.max())
    mid_ts = (first_ts + last_ts) // 2
    return first_ts, last_ts, mid_ts


# ---------------------------------------------------------------------------
# Folder discovery
# ---------------------------------------------------------------------------

def find_session_folders(root: Path):
    """Yield every folder under root that directly contains at least one
    .nev/.ncs/.ntt file, skipping _Geo1/_Geo2 output folders."""
    for dirpath, dirnames, filenames in os.walk(root):
        # don't descend into output folders from a previous run
        dirnames[:] = [d for d in dirnames if not d.endswith(HALF_SUFFIXES)]
        if Path(dirpath).name.endswith(HALF_SUFFIXES):
            continue
        exts = {Path(fn).suffix.lower() for fn in filenames}
        if exts & {".nev", ".ncs", ".ntt"}:
            yield Path(dirpath)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def cut_marker_event(template: np.void, mid_ts: int) -> np.ndarray:
    """Build a one-record .nev array marking the cut at mid_ts. Packet
    fields (nstx, npkt_id, ...) are copied from an existing event in the
    same file so the record matches what the acquisition system wrote."""
    marker = np.array([template], dtype=template.dtype)
    marker["TimeStamp"] = mid_ts
    marker["nevent_id"] = 0
    marker["nttl"] = 0
    marker["ncrc"] = 0
    marker["dnExtra"] = 0
    marker["EventString"] = f"Cut at midpoint TimeStamp {mid_ts}".encode("latin-1")
    return marker


def split_folder_at(folder: Path, mid_ts: int):
    first_dir = folder / f"{folder.name}{HALF_SUFFIXES[0]}"
    second_dir = folder / f"{folder.name}{HALF_SUFFIXES[1]}"

    files = []
    for ext in CUT_EXTENSIONS:
        files += sorted(folder.glob(f"*{ext}"))

    for path in files:
        header, records = load_records(path)
        in_first = records["TimeStamp"] < mid_ts
        first, second = records[in_first], records[~in_first]
        if path.suffix.lower() == ".nev" and len(records):
            # mark the cut: last event of the first half, first event of the second
            marker = cut_marker_event(records[0], mid_ts)
            first = np.concatenate([first, marker])
            second = np.concatenate([marker, second])
        n1 = write_cut_file(header, first, first_dir / path.name)
        n2 = write_cut_file(header, second, second_dir / path.name)
        print(f"    {path.name}: {n1} + {n2} / {len(records)} records")

    # .csv files are copied unchanged into both halves
    for csv_path in sorted(folder.glob("*.csv")):
        for out_dir in (first_dir, second_dir):
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv_path, out_dir / csv_path.name)
        print(f"    {csv_path.name}: copied to both halves")

    print(f"  -> {first_dir}")
    print(f"  -> {second_dir}")


def main():
    if not ROOT_DIR.exists():
        raise SystemExit(f"ROOT_DIR does not exist: {ROOT_DIR}")

    session_folders = list(find_session_folders(ROOT_DIR))
    print(f"Found {len(session_folders)} session folder(s) under {ROOT_DIR}\n")

    for folder in session_folders:
        print(f"=== {folder} ===")
        result = nev_midpoint(folder)
        if result is None:
            print("  No .nev file / fewer than 2 events found, skipping.")
            continue

        first_ts, last_ts, mid_ts = result
        print(f"  .nev first={first_ts}  last={last_ts}  midpoint={mid_ts}  "
              f"(half = {(last_ts - first_ts) / 2e6:.1f} s)")
        split_folder_at(folder, mid_ts)

    print("Done.")


if __name__ == "__main__":
    main()
