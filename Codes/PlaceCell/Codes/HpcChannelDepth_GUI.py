"""
HpcChannelDepth_GUI.py

Walk through every Open Ephys Neuropixels 2.0 recording found under a set of
directories, show a GUI for each one (animal ID, recording day, session...),
let the user type the hippocampus TOP and BOTTOM channel, and write the
channel depths plus session details to an Excel sheet.

Depth convention
----------------
Depth is taken from the probe geometry that SpikeInterface/probeinterface
attaches to the recording (channel location y, in µm). y = 0 is the most distal
electrode site (electrode 0, at the tip end of the shank) and increases towards
the part of the probe that sits outside the brain. So the hippocampus TOP
channel should have a LARGER depth value than the BOTTOM channel.
If you also want the distance from the physical tip (not just from the first
site), set TIP_OFFSET_UM below; it is added to every depth.

Usage
-----
1. Edit DATA_DIRS (and OUTPUT_XLSX if needed) below.
2. python HpcChannelDepth_GUI.py
3. For each recording: enter top/bottom channel -> "Save & Next".
   The Excel file is rewritten after every save, and already-entered
   recordings are pre-filled when the script is re-run (resume support).
"""

import os
import re
import datetime as dt
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import spikeinterface.full as si

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =============================================================================
# CONFIG
# =============================================================================
# key = animal ID, value = one directory or a LIST of directories.
# Each animal ID must appear only ONCE (a repeated dict key silently overwrites
# the earlier one) - put all of that animal's directories in its list.
# Each directory is searched recursively for Open Ephys recording folders
# (named YYYY-MM-DD_HH-MM-SS), e.g.
#   .../Fa0156/Day9_CntrlCntrlCntrl_Shank3BankA/1Cntrl/2024-08-20_23-53-22
DATA_DIRS = {
    'Fa0156': [
        r"X:\NMR_group_data\_Scratch_folder\AdrianLabelledData\SingleSessions\0156\Day2_CntrlCntrl_Shank1BankA\1Cntrl",
        r"X:\NMR_group_data\_Scratch_folder\AdrianLabelledData\SingleSessions\0156\Day3_CntrlNoRotRotCntrlZero_Shank1BankA\1Cntrl",
        r"X:\NMR_group_data\_Scratch_folder\AdrianLabelledData\SingleSessions\0156\Day6_CntrlRotNORot_Shank4BankA\1Cntrl",
        r"X:\NMR_group_data\_Scratch_folder\AdrianLabelledData\SingleSessions\0156\Day7_CntrlCntrl_Shank4BankA\1Cntrl",
        r"X:\NMR_group_data\Runita\Analysis\ephys_sorted\AdrianLabelledData\SingleSessions\0156\Day8_CntrlNoRotRot_Shank3BankA\IndividualSessions\1Cntrl",
        r"X:\NMR_group_data\Runita\Analysis\ephys_sorted\AdrianLabelledData\SingleSessions\0156\Day9_CntrlCntrlCntrl_Shank3BankA\1Cntrl",
    ],
    'Fa1680378B': [
        r"X:\NMR_group_data\_Scratch_folder\AdrianLabelledData\SingleSessions\1680378B\Day1_CntrlCntrlCntrlZero_Shank1BankA\Cntrl1",
        r"X:\NMR_group_data\_Scratch_folder\AdrianLabelledData\SingleSessions\1680378B\Day2_CntrlNoRotRotCntrlZero_Shank1BankA\Cntrl1",
        r"X:\NMR_group_data\_Scratch_folder\AdrianLabelledData\SingleSessions\1680378B\Day3_CntrlRotNoRotCntrl_Shank2BankA\1Cntrl",
        r"X:\NMR_group_data\Runita\Analysis\ephys_sorted\AdrianLabelledData\SingleSessions\1680378B\Day5_CntrlNoRotRot_Shank4BankA\1Cntrl",
        r"X:\NMR_group_data\Runita\Analysis\ephys_sorted\AdrianLabelledData\SingleSessions\1680378B\Day6_CntrlCntrlCntrl_Shank4BankA\1Cntrl",
        r"X:\NMR_group_data\Runita\Analysis\ephys_sorted\AdrianLabelledData\SingleSessions\1680378B\Day9_CntrlCntrlCntrl_Shank3BankA\1Cntrl",
        r"X:\NMR_group_data\Runita\Analysis\ephys_sorted\AdrianLabelledData\SingleSessions\1680378B\Day10_CntrlRotNoRot_Shank3BankA\1Cntrl",
    ],
}

OUTPUT_XLSX = r"X:\NMR_group_data\Runita\Data\Ephys_Data\AllSortedData\Neuropixel\HpcChannelDepths.xlsx"
STREAM_ID = '1'          # same stream as in ExtractNpxSpikes.ipynb
TIP_OFFSET_UM = 0.0      # distance from physical tip to electrode 0 (µm); 0 = depth from first site

REC_FOLDER_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')


# =============================================================================
# Helpers
# =============================================================================
def get_rec_folders(directory):
    """Same logic as NpxUtils.get_rec_folders: all Open Ephys timestamp folders."""
    subfolders = []
    for root, dirs, _ in os.walk(directory):
        subfolders.extend([os.path.join(root, d) for d in dirs if REC_FOLDER_PATTERN.match(d)])
    return subfolders


def parse_session_details(rec_path, animal_key):
    """Derive animal / day / session / sub-session from the directory structure."""
    parts = os.path.normpath(rec_path).split(os.sep)

    animal = animal_key
    for p in parts:
        m = re.fullmatch(r'[A-Za-z]{2}\d{4}', p)
        if m:
            animal = p
            break

    day, session_folder, session_idx = None, '', None
    for i in range(len(parts) - 1, -1, -1):
        m = re.search(r'Day[_\-\s]?(\d+)', parts[i], flags=re.IGNORECASE)
        if m:
            day = int(m.group(1))
            session_folder = parts[i]
            session_idx = i
            break

    # folders between the session folder and the timestamp folder, e.g. '1Cntrl'
    if session_idx is not None:
        sub_session = os.sep.join(parts[session_idx + 1:-1])
    else:
        sub_session = parts[-2] if len(parts) > 1 else ''

    shank, bank = None, None
    m = re.search(r'Shank(\d+)', session_folder, flags=re.IGNORECASE)
    if m:
        shank = int(m.group(1))
    m = re.search(r'Bank([A-D])', session_folder, flags=re.IGNORECASE)
    if m:
        bank = m.group(1).upper()

    return dict(Animal=animal, Day=day, Session_Folder=session_folder,
                SubSession=sub_session, Rec_Folder=parts[-1],
                Session_Shank=shank, Session_Bank=bank)


def load_recording_info(rec_path):
    """Read the raw recording with SpikeInterface and pull out probe geometry."""
    rec = si.read_openephys(rec_path, stream_id=STREAM_ID)
    channel_ids = np.array([str(c) for c in rec.get_channel_ids()])
    locations = rec.get_channel_locations()          # (n_ch, 2): x, y in µm
    try:
        groups = rec.get_channel_groups()
    except Exception:
        groups = np.zeros(len(channel_ids), dtype=int)

    probe_model, probe_serial = '', ''
    try:
        probe = rec.get_probe()
        probe_model = probe.model_name or probe.annotations.get('model_name', '') or probe.name or ''
        probe_serial = probe.serial_number or probe.annotations.get('serial_number', '') or ''
    except Exception:
        pass

    return dict(channel_ids=channel_ids, locations=locations, groups=groups,
                n_channels=rec.get_num_channels(), fs=rec.get_sampling_frequency(),
                duration=rec.get_total_duration(),
                probe_model=probe_model, probe_serial=probe_serial)


def resolve_channel(text, channel_ids):
    """Accept '214', 'CH214' or an exact channel id; return the index into channel_ids."""
    s = text.strip()
    if not s:
        raise ValueError("empty")
    ids = list(channel_ids)
    if s in ids:
        return ids.index(s)
    if s.upper() in ids:
        return ids.index(s.upper())
    if s.isdigit():
        for cand in (f"CH{int(s)}", str(int(s))):
            if cand in ids:
                return ids.index(cand)
    raise ValueError(f"channel '{s}' not found (ids run {ids[0]} ... {ids[-1]})")


def channel_number(ch_id):
    m = re.search(r'(\d+)$', ch_id)
    return int(m.group(1)) if m else ch_id


# =============================================================================
# GUI
# =============================================================================
class ChannelDepthApp:
    COLUMNS = [
        'Animal', 'Day', 'Session_Folder', 'SubSession', 'Rec_Folder',
        'Session_Shank', 'Session_Bank', 'Probe_Model', 'Probe_Serial',
        'Stream_ID', 'N_Channels', 'Sampling_Rate_Hz', 'Duration_s',
        'Top_Channel', 'Top_Channel_ID', 'Top_Shank_Group', 'Top_x_um', 'Top_Depth_um',
        'Bottom_Channel', 'Bottom_Channel_ID', 'Bottom_Shank_Group', 'Bottom_x_um', 'Bottom_Depth_um',
        'Hpc_Span_um', 'Tip_Offset_um', 'Notes', 'Entered_At', 'Recording_Path',
    ]

    def __init__(self, root, recordings):
        self.root = root
        self.recordings = recordings          # list of (animal_key, rec_path)
        self.idx = 0
        self.info_cache = {}
        self.results = self._load_existing()  # Recording_Path -> row dict
        self.last_entry = None                # (animal, day, top, bottom) for "copy previous"

        root.title("Hippocampus channel depth")
        root.geometry("1100x720")
        self._build_widgets()

        # start at the first recording that has not been entered yet
        done = [i for i, (_, p) in enumerate(recordings) if p in self.results]
        self.idx = next((i for i in range(len(recordings)) if i not in done), 0)
        self.show_recording()

    # ------------------------------------------------------------------ layout
    def _build_widgets(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill='both', expand=True)

        left = ttk.Frame(main)
        left.pack(side='left', fill='both', expand=True)
        right = ttk.Frame(main)
        right.pack(side='right', fill='y')

        self.progress_var = tk.StringVar()
        ttk.Label(left, textvariable=self.progress_var, font=('Segoe UI', 12, 'bold')).pack(anchor='w')

        info = ttk.LabelFrame(left, text="Session details", padding=8)
        info.pack(fill='x', pady=6)
        self.info_vars = {}
        for r, key in enumerate(['Animal', 'Day', 'Session_Folder', 'SubSession', 'Rec_Folder',
                                 'Probe', 'Channels', 'Channel_IDs', 'Depth_Range', 'Status', 'Path']):
            ttk.Label(info, text=key.replace('_', ' ') + ':', width=14).grid(row=r, column=0, sticky='nw')
            v = tk.StringVar()
            font = ('Segoe UI', 11, 'bold') if key in ('Animal', 'Day') else ('Segoe UI', 9)
            ttk.Label(info, textvariable=v, font=font, wraplength=620, justify='left').grid(
                row=r, column=1, sticky='w')
            self.info_vars[key] = v

        entry = ttk.LabelFrame(left, text="Hippocampus channels", padding=8)
        entry.pack(fill='x', pady=6)
        self.top_var, self.bottom_var, self.notes_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.top_prev, self.bottom_prev = tk.StringVar(), tk.StringVar()

        ttk.Label(entry, text="Top channel:").grid(row=0, column=0, sticky='w')
        self.top_entry = ttk.Entry(entry, textvariable=self.top_var, width=12)
        self.top_entry.grid(row=0, column=1, sticky='w', padx=4)
        ttk.Label(entry, textvariable=self.top_prev, foreground='firebrick').grid(row=0, column=2, sticky='w')

        ttk.Label(entry, text="Bottom channel:").grid(row=1, column=0, sticky='w')
        ttk.Entry(entry, textvariable=self.bottom_var, width=12).grid(row=1, column=1, sticky='w', padx=4)
        ttk.Label(entry, textvariable=self.bottom_prev, foreground='navy').grid(row=1, column=2, sticky='w')

        ttk.Label(entry, text="Notes:").grid(row=2, column=0, sticky='w')
        ttk.Entry(entry, textvariable=self.notes_var, width=60).grid(row=2, column=1, columnspan=2, sticky='w', padx=4)

        for v in (self.top_var, self.bottom_var):
            v.trace_add('write', lambda *_: self.update_preview())

        btns = ttk.Frame(left)
        btns.pack(fill='x', pady=8)
        ttk.Button(btns, text="◀ Back", command=self.go_back).pack(side='left', padx=3)
        ttk.Button(btns, text="Skip ▶", command=self.skip).pack(side='left', padx=3)
        ttk.Button(btns, text="Copy previous", command=self.copy_previous).pack(side='left', padx=3)
        ttk.Button(btns, text="Save & Next ▶", command=self.save_and_next).pack(side='left', padx=3)
        ttk.Button(btns, text="Finish", command=self.finish).pack(side='right', padx=3)
        self.root.bind('<Return>', lambda e: self.save_and_next())

        self.status_var = tk.StringVar()
        ttk.Label(left, textvariable=self.status_var, foreground='gray30').pack(anchor='w', pady=4)

        # probe map
        self.fig = Figure(figsize=(3.2, 6.5), dpi=90)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    # ---------------------------------------------------------------- display
    def _get_info(self, rec_path):
        if rec_path not in self.info_cache:
            self.info_cache[rec_path] = load_recording_info(rec_path)
        return self.info_cache[rec_path]

    def show_recording(self):
        animal_key, rec_path = self.recordings[self.idx]
        self.details = parse_session_details(rec_path, animal_key)
        self.progress_var.set(f"Recording {self.idx + 1} / {len(self.recordings)}   "
                              f"({len(self.results)} entered)")
        for k in ('Animal', 'Day', 'Session_Folder', 'SubSession', 'Rec_Folder'):
            self.info_vars[k].set(str(self.details[k]))
        self.info_vars['Path'].set(rec_path)
        self.info_vars['Status'].set("Loading recording ...")
        self.root.update_idletasks()

        try:
            self.info = self._get_info(rec_path)
            ids, locs = self.info['channel_ids'], self.info['locations']
            self.info_vars['Probe'].set(f"{self.info['probe_model']}  SN {self.info['probe_serial']}".strip())
            self.info_vars['Channels'].set(f"{self.info['n_channels']} ch, {self.info['fs']:.0f} Hz, "
                                           f"{self.info['duration']:.1f} s")
            self.info_vars['Channel_IDs'].set(f"{ids[0]} ... {ids[-1]}")
            self.info_vars['Depth_Range'].set(f"{locs[:, 1].min():.0f} - {locs[:, 1].max():.0f} µm from tip-most site")
            self.info_vars['Status'].set("Already entered - edit and save to overwrite"
                                         if rec_path in self.results else "Not entered yet")
        except Exception as e:
            self.info = None
            for k in ('Probe', 'Channels', 'Channel_IDs', 'Depth_Range'):
                self.info_vars[k].set('')
            self.info_vars['Status'].set(f"ERROR loading: {e}")

        prev = self.results.get(rec_path)
        self.top_var.set('' if prev is None else str(prev['Top_Channel']))
        self.bottom_var.set('' if prev is None else str(prev['Bottom_Channel']))
        notes = None if prev is None else prev.get('Notes')
        self.notes_var.set('' if notes is None or pd.isna(notes) else str(notes))
        self.update_preview()
        self.top_entry.focus_set()

    def _channel_summary(self, text):
        """Return (idx, description) for an entered channel, or (None, message)."""
        if self.info is None or not text.strip():
            return None, ''
        try:
            i = resolve_channel(text, self.info['channel_ids'])
        except ValueError as e:
            return None, f"✗ {e}"
        x, y = self.info['locations'][i]
        return i, (f"{self.info['channel_ids'][i]}  |  shank/group {self.info['groups'][i]}  |  "
                   f"x = {x:.0f} µm  |  depth = {y + TIP_OFFSET_UM:.0f} µm")

    def update_preview(self):
        ti, tdesc = self._channel_summary(self.top_var.get())
        bi, bdesc = self._channel_summary(self.bottom_var.get())
        self.top_prev.set(tdesc)
        self.bottom_prev.set(bdesc)
        if ti is not None and bi is not None:
            ty, by = self.info['locations'][ti, 1], self.info['locations'][bi, 1]
            msg = f"Hpc span = {ty - by:.0f} µm"
            if ty < by:
                msg += "   ⚠ top channel is DEEPER than bottom channel - check the order"
            self.status_var.set(msg)
        else:
            self.status_var.set('')
        self.draw_probe(ti, bi)

    def draw_probe(self, ti, bi):
        self.ax.clear()
        if self.info is not None:
            locs = self.info['locations']
            self.ax.scatter(locs[:, 0], locs[:, 1], s=6, c='lightgray', marker='s')
            if ti is not None:
                self.ax.scatter(*locs[ti], s=60, c='firebrick', marker='s', label='top')
                self.ax.axhline(locs[ti, 1], color='firebrick', lw=0.8, ls='--')
            if bi is not None:
                self.ax.scatter(*locs[bi], s=60, c='navy', marker='s', label='bottom')
                self.ax.axhline(locs[bi, 1], color='navy', lw=0.8, ls='--')
            if ti is not None and bi is not None:
                lo, hi = sorted([locs[ti, 1], locs[bi, 1]])
                self.ax.axhspan(lo, hi, color='gold', alpha=0.25)
            if ti is not None or bi is not None:
                self.ax.legend(loc='upper right', fontsize=8)
        self.ax.set_xlabel('x (µm)')
        self.ax.set_ylabel('depth from tip-most site (µm)')
        self.ax.set_title('Probe map', fontsize=10)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ---------------------------------------------------------------- actions
    def save_and_next(self):
        if self.info is None:
            messagebox.showwarning("Cannot save", "This recording could not be loaded. Use Skip.")
            return
        ti, tmsg = self._channel_summary(self.top_var.get())
        bi, bmsg = self._channel_summary(self.bottom_var.get())
        if ti is None or bi is None:
            messagebox.showwarning("Invalid channel", f"Top: {tmsg or 'missing'}\nBottom: {bmsg or 'missing'}")
            return

        ids, locs, groups = self.info['channel_ids'], self.info['locations'], self.info['groups']
        top_y, bot_y = locs[ti, 1] + TIP_OFFSET_UM, locs[bi, 1] + TIP_OFFSET_UM
        if top_y < bot_y and not messagebox.askyesno(
                "Check order", "The top channel is deeper than the bottom channel.\nSave anyway?"):
            return

        _, rec_path = self.recordings[self.idx]
        row = dict(self.details)
        row.update(
            Probe_Model=self.info['probe_model'], Probe_Serial=self.info['probe_serial'],
            Stream_ID=STREAM_ID, N_Channels=self.info['n_channels'],
            Sampling_Rate_Hz=self.info['fs'], Duration_s=round(self.info['duration'], 3),
            Top_Channel=channel_number(ids[ti]), Top_Channel_ID=ids[ti],
            Top_Shank_Group=groups[ti], Top_x_um=float(locs[ti, 0]), Top_Depth_um=float(top_y),
            Bottom_Channel=channel_number(ids[bi]), Bottom_Channel_ID=ids[bi],
            Bottom_Shank_Group=groups[bi], Bottom_x_um=float(locs[bi, 0]), Bottom_Depth_um=float(bot_y),
            Hpc_Span_um=float(top_y - bot_y), Tip_Offset_um=TIP_OFFSET_UM,
            Notes=self.notes_var.get(), Entered_At=dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            Recording_Path=rec_path,
        )
        self.results[rec_path] = row
        self.last_entry = (row['Animal'], row['Day'], self.top_var.get(), self.bottom_var.get())
        if self.write_excel():
            self.next_recording()

    def copy_previous(self):
        if self.last_entry is None:
            self.status_var.set("Nothing to copy yet.")
            return
        animal, day, top, bottom = self.last_entry
        self.top_var.set(top)
        self.bottom_var.set(bottom)
        if (animal, day) != (self.details['Animal'], self.details['Day']):
            self.status_var.set(self.status_var.get() +
                                f"   (copied from {animal} Day {day} - different session!)")

    def skip(self):
        self.next_recording()

    def go_back(self):
        if self.idx > 0:
            self.idx -= 1
            self.show_recording()

    def next_recording(self):
        if self.idx < len(self.recordings) - 1:
            self.idx += 1
            self.show_recording()
        else:
            missing = len(self.recordings) - len(self.results)
            messagebox.showinfo("Done", f"Reached the last recording.\n"
                                        f"{len(self.results)} entered, {missing} not entered.\n"
                                        f"Saved to:\n{os.path.abspath(OUTPUT_XLSX)}")

    def finish(self):
        if self.write_excel():
            self.root.destroy()

    # --------------------------------------------------------------- file I/O
    def _load_existing(self):
        if not os.path.isfile(OUTPUT_XLSX):
            return {}
        try:
            df = pd.read_excel(OUTPUT_XLSX, sheet_name='HpcChannels')
            print(f"Resuming: {len(df)} entries loaded from {OUTPUT_XLSX}")
            return {r['Recording_Path']: r for r in df.to_dict('records')}
        except Exception as e:
            print(f"Could not read existing {OUTPUT_XLSX} ({e}); starting fresh.")
            return {}

    def write_excel(self):
        df = pd.DataFrame(list(self.results.values()))
        df = df.reindex(columns=self.COLUMNS)
        df = df.sort_values(['Animal', 'Day', 'Session_Folder', 'SubSession', 'Rec_Folder'],
                            na_position='last')
        try:
            with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='HpcChannels', index=False)
                ws = writer.sheets['HpcChannels']
                for col_cells in ws.columns:
                    width = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
                    ws.column_dimensions[col_cells[0].column_letter].width = min(width + 2, 60)
                ws.freeze_panes = 'B2'
        except PermissionError:
            messagebox.showerror("Excel file locked",
                                 f"Could not write {OUTPUT_XLSX}.\nClose it in Excel and press Save again.")
            return False
        self.status_var.set(f"Saved {len(df)} rows to {os.path.abspath(OUTPUT_XLSX)}")
        return True


# =============================================================================
# Main
# =============================================================================
def collect_recordings(data_dirs):
    recordings, seen = [], set()
    for animal_key, dirs in data_dirs.items():
        if isinstance(dirs, str):
            dirs = [dirs]
        for d in dirs:
            if not os.path.isdir(d):
                print(f"WARNING: directory not found, skipped: {d}")
                continue
            if REC_FOLDER_PATTERN.match(os.path.basename(os.path.normpath(d))):
                found = [d]                   # the directory is itself a recording folder
            else:
                found = sorted(get_rec_folders(d))
            print(f"{animal_key}: {len(found)} recording folders in {d}")
            if not found:
                print(f"WARNING: no YYYY-MM-DD_HH-MM-SS recording folder found in {d}")
            for rec_path in found:
                rec_path = os.path.normpath(rec_path)
                if rec_path not in seen:
                    seen.add(rec_path)
                    recordings.append((animal_key, rec_path))
    return recordings


if __name__ == '__main__':
    print(f"SpikeInterface version: {si.__version__}")
    recordings = collect_recordings(DATA_DIRS)
    if not recordings:
        raise SystemExit("No Open Ephys recording folders found - check DATA_DIRS.")
    print(f"Total recordings to annotate: {len(recordings)}")

    root = tk.Tk()
    ChannelDepthApp(root, recordings)
    root.mainloop()
