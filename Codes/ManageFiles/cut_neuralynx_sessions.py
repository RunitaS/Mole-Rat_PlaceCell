"""
Walk a root directory tree, and for every folder that contains Neuralynx
recording files (.nev events, .ncs LFP, .ntt spikes), show a form with up to
MAX_CUTS rows. Each row has a start-event dropdown and an end-event dropdown
(both populated from that folder's .nev event labels) and a manually-typed
output folder name. For each filled-in row, every .nev/.ncs/.ntt file in that
source folder gets cut to the window between the two chosen events and
written into a new subfolder (the typed name) created inside that same
source folder, in native Neuralynx format (16 KB header copied as-is + only
the records whose timestamp falls inside the window).

Requires: numpy  (tkinter ships with standard CPython on Windows)

Usage:
    1. Edit ROOT_DIR below.
    2. Run: python cut_neuralynx_sessions.py
    3. For each session folder, a window opens with up to 5 rows:
         - Start event (dropdown of "timestamp | event label")
         - End event (dropdown of "timestamp | event label")
         - Output folder name (type freely)
       Fill in as many rows as you need (leave a row fully blank to skip
       it), then click "Run cuts". Click "Skip folder" to move on without
       cutting anything, or "Quit" to stop processing entirely.

Note: if you re-run the script over the same ROOT_DIR later, the cut
subfolders created by a previous run will themselves show up as session
folders (they contain .nev/.ncs/.ntt too) -- just click "Skip folder" for
those.
"""

from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np

# ---------------------------------------------------------------------------
# Configuration -- EDIT THESE
# ---------------------------------------------------------------------------

ROOT_DIR = Path(r"X:/NMR_group_data/Runita/AllData_Backup/AllSortedData/Tetrode/Fa5834/ExpDay1_Cntrl_23Oct25")
# Cut files are saved into a subfolder created next to each source folder's
# own files (i.e. inside the same folder the .nev/.ncs/.ntt came from), so
# there is no separate output root to configure.

MAX_CUTS = 5  # number of start/end/name rows shown per folder

HEADER_SIZE = 16384  # standard Neuralynx text header size, all file types
NCS_SAMPLES_PER_RECORD = 512  # fixed for standard Neuralynx CSC records


# ---------------------------------------------------------------------------
# Neuralynx binary record layouts
#
# Every Neuralynx file (.nev, .ncs, .ntt, ...) is a 16 KB ASCII header
# followed by fixed-size binary records. Cutting a file to a time window is
# just: copy the header verbatim, keep only the records whose TimeStamp
# falls inside [start, stop], and write header + kept records back out.
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


def load_nev_events(folder: Path):
    """Return a list of (timestamp, label) for the folder's first .nev file,
    or None if there is no .nev file to read events from."""
    nev_files = sorted(folder.glob("*.nev"))
    if not nev_files:
        return None
    _, records = load_records(nev_files[0])
    events = []
    for rec in records:
        ts = int(rec["TimeStamp"])
        label = rec["EventString"].split(b"\x00", 1)[0].decode("latin-1", errors="ignore").strip()
        events.append((ts, label))
    return events


# ---------------------------------------------------------------------------
# Folder discovery
# ---------------------------------------------------------------------------

def find_session_folders(root: Path):
    """Yield every folder under root that directly contains at least one
    .nev/.ncs/.ntt file."""
    for dirpath, _dirnames, filenames in os.walk(root):
        exts = {Path(fn).suffix.lower() for fn in filenames}
        if exts & {".nev", ".ncs", ".ntt"}:
            yield Path(dirpath)


# ---------------------------------------------------------------------------
# Cut-selection form (Tkinter)
# ---------------------------------------------------------------------------

def run_folder_form(folder: Path, events: list[tuple[int, str]]):
    """Show a form with up to MAX_CUTS rows: start-event dropdown,
    end-event dropdown, output folder name entry (typed freely). Returns
    (cuts, quit_all) where cuts is a list of (start_ts, stop_ts, out_name)
    for the rows that were filled in, and quit_all is True if the user
    asked to stop processing entirely."""

    display_to_ts = {}
    display_values = []
    for ts, label in events:
        disp = f"{ts}  |  {label}" if label else f"{ts}"
        display_to_ts[disp] = ts
        display_values.append(disp)

    result = {"cuts": [], "quit": False}

    root = tk.Tk()
    root.title(f"Select cuts -- {folder.name}")

    header_font = ("Segoe UI", 9, "bold")
    tk.Label(root, text=str(folder), font=("Segoe UI", 10, "bold")).grid(
        row=0, column=0, columnspan=3, padx=8, pady=(8, 2), sticky="w")

    tk.Label(root, text="Start event", font=header_font).grid(row=1, column=0, padx=8, pady=2)
    tk.Label(root, text="End event", font=header_font).grid(row=1, column=1, padx=8, pady=2)
    tk.Label(root, text="Output folder name", font=header_font).grid(row=1, column=2, padx=8, pady=2)

    start_boxes = []
    end_boxes = []
    name_entries = []

    for i in range(MAX_CUTS):
        start_cb = ttk.Combobox(root, values=display_values, state="readonly", width=40)
        start_cb.grid(row=2 + i, column=0, padx=8, pady=3)
        end_cb = ttk.Combobox(root, values=display_values, state="readonly", width=40)
        end_cb.grid(row=2 + i, column=1, padx=8, pady=3)
        name_entry = tk.Entry(root, width=25)
        name_entry.insert(0, f"cut{i + 1}")
        name_entry.grid(row=2 + i, column=2, padx=8, pady=3)

        start_boxes.append(start_cb)
        end_boxes.append(end_cb)
        name_entries.append(name_entry)

    def on_run():
        cuts = []
        for row, (start_cb, end_cb, name_entry) in enumerate(
                zip(start_boxes, end_boxes, name_entries), start=1):
            start_disp = start_cb.get()
            end_disp = end_cb.get()
            name = name_entry.get().strip()
            if not start_disp and not end_disp and not name:
                continue  # unused row
            if not start_disp or not end_disp or not name:
                messagebox.showerror(
                    "Incomplete row",
                    f"Row {row}: fill in start event, end event, and folder name "
                    "(or leave all three blank to skip that row).")
                return
            start_ts = display_to_ts[start_disp]
            stop_ts = display_to_ts[end_disp]
            if stop_ts < start_ts:
                start_ts, stop_ts = stop_ts, start_ts
            cuts.append((start_ts, stop_ts, name))

        names = [c[2] for c in cuts]
        if len(names) != len(set(names)):
            messagebox.showerror("Duplicate folder name", "Each row needs a unique output folder name.")
            return

        result["cuts"] = cuts
        root.destroy()

    def on_skip():
        root.destroy()

    def on_quit():
        result["quit"] = True
        root.destroy()

    btn_frame = tk.Frame(root)
    btn_frame.grid(row=2 + MAX_CUTS, column=0, columnspan=3, pady=10)
    tk.Button(btn_frame, text="Run cuts", command=on_run, width=14).pack(side="left", padx=5)
    tk.Button(btn_frame, text="Skip folder", command=on_skip, width=14).pack(side="left", padx=5)
    tk.Button(btn_frame, text="Quit", command=on_quit, width=14).pack(side="left", padx=5)

    root.protocol("WM_DELETE_WINDOW", on_skip)
    root.mainloop()

    return result["cuts"], result["quit"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def cut_folder_to(folder: Path, start_ts: float, stop_ts: float, out_dir: Path):
    files = sorted(folder.glob("*.nev")) + sorted(folder.glob("*.ncs")) + sorted(folder.glob("*.ntt"))
    for path in files:
        header, records = load_records(path)
        mask = (records["TimeStamp"] >= start_ts) & (records["TimeStamp"] <= stop_ts)
        cut_records = records[mask]
        out_path = out_dir / path.name
        n_written = write_cut_file(header, cut_records, out_path)
        print(f"    {path.name}: wrote {n_written}/{len(records)} records -> {out_path}")


def main():
    if not ROOT_DIR.exists():
        raise SystemExit(f"ROOT_DIR does not exist: {ROOT_DIR}")

    session_folders = list(find_session_folders(ROOT_DIR))
    print(f"Found {len(session_folders)} session folder(s) under {ROOT_DIR}\n")

    for folder in session_folders:
        print(f"=== {folder} ===")
        events = load_nev_events(folder)
        if not events:
            print("  No .nev file / events found, skipping.")
            continue

        cuts, quit_all = run_folder_form(folder, events)
        if not cuts:
            print("  No cuts selected, skipping.")
        else:
            for start_ts, stop_ts, name in cuts:
                out_dir = folder / name
                print(f"  Cutting {start_ts} -> {stop_ts} into {out_dir}")
                cut_folder_to(folder, start_ts, stop_ts, out_dir)

        if quit_all:
            print("Stopping.")
            break

    print("Done.")


if __name__ == "__main__":
    main()
