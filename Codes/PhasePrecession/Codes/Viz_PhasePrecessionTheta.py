# -*- coding: utf-8 -*-
"""
Interactive viewer: scroll through a unit's raw LFP + theta-band trace with
its spikes overlaid, and a second panel showing the instantaneous theta
phase at which each spike fired. A time slider and a window-length slider
let you scrub through the whole recording at an adjustable zoom (default
~5 s); "Save snapshot" writes the currently visible window to PNG.

Two ways to pick which cell(s) to look at:

  1. EXCEL_SUMMARY_PATH set to a theta_phase.xlsx produced by
     ThetaMod_PhasePrecession_v2.py -- the viewer builds a list of every
     unit flagged is_precessing / is_recessing (per PRECESSION_FILTER) and
     you page through them with the Prev/Next cell buttons ('p' / 'n').
  2. EXCEL_SUMMARY_PATH = None -- the viewer opens exactly one cell, given
     by NTT_FILE / CELL_NUMBER (same convention as
     Understanding_PassIndex/Viz_PassIndex_AllSteps.py).

All Neuralynx I/O and theta-phase math (load_ncs, load_ntt_spike_times,
bandpass_filter, assign_spike_phase) are imported unchanged from
ThetaMod_PhasePrecession_v2.py so the phase values shown here match the
batch pipeline exactly.

By default (IN_FIELD_ONLY = True) only spikes fired inside the place field
are shown/counted -- "in field" defined the same way
ThetaMod_PhasePrecession_v2.auto_filter_band defines it for the 'place'
method: a rate-map bin with occupancy-normalized rate > FIELD_PEAK_FRACTION
(10%) of the cell's peak rate. Set IN_FIELD_ONLY = False to see all spikes.

Keyboard shortcuts (figure must have focus): Left/Right = prev/next window,
Up/Down = widen/narrow window, ','/'.' = prev/next spike, 's' = save
snapshot, 'p'/'n' = prev/next cell.

Requires: numpy, pandas, scipy, matplotlib. On Windows, Tk (bundled with
standard Python) or PyQt5 for an interactive backend.
"""

from pathlib import Path

import numpy as np
import matplotlib

_INTERACTIVE_BACKEND = None
for _candidate in ('TkAgg', 'Qt5Agg', 'QtAgg'):
    try:
        matplotlib.use(_candidate, force=True)
        _INTERACTIVE_BACKEND = _candidate
        break
    except Exception:
        continue
if _INTERACTIVE_BACKEND is None:
    raise RuntimeError(
        'No interactive matplotlib backend available. Install Tk (usually '
        'bundled with Python) or `pip install pyqt5`.'
    )

import pandas as pd
from scipy import signal
from matplotlib.widgets import Slider, Button, TextBox

# Reused unchanged from the batch pipeline so numbers match exactly.
from ThetaMod_PhasePrecession_v2 import (
    load_ncs, load_ntt_spike_times, bandpass_filter, assign_spike_phase,
    _natural_key, LFP_FILTER_BAND as _PIPELINE_LFP_BAND,
    load_tracking, spk_pos, rate_map, _find_tracking_file,
    BINSIDE as _PIPELINE_BINSIDE, SMTH_WIDTH as _PIPELINE_SMTH_WIDTH,
    TRACKING_TIME_UNIT as _PIPELINE_TRACKING_TIME_UNIT,
)

# Importing the pipeline module force-switches matplotlib to 'Agg' (it's
# built for headless batch PNG export). Re-apply the interactive backend
# now -- safe because no figure/canvas has been created yet.
matplotlib.use(_INTERACTIVE_BACKEND, force=True)
import matplotlib.pyplot as plt

# ============================================================================
# Configuration -- EDIT THESE
# ============================================================================

# --- Option 1: page through every flagged unit from a batch run ---
EXCEL_SUMMARY_PATH = Path(r"C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True/Fa8477/theta_phase.xlsx")
PRECESSION_FILTER = 'precessing_or_recessing'  # 'precessing' | 'recessing' | 'precessing_or_recessing'

# --- Option 2: view exactly one cell (used when EXCEL_SUMMARY_PATH is None
# or doesn't exist) ---
NTT_FILE = Path(r"C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True/Fa8477/Day5/1Stnd/TT1_SS_01.ntt")
CELL_NUMBER = None  # None = auto-pick the sorted cluster with the most spikes

LFP_FILTER_BAND = _PIPELINE_LFP_BAND  # Hz, theta band for the phase panel

# Restrict displayed spikes to the place field, defined exactly as in
# ThetaMod_PhasePrecession_v2.auto_filter_band: rate-map bins with firing
# rate > FIELD_PEAK_FRACTION * peak rate (10% of peak, "place" method).
IN_FIELD_ONLY = True
FIELD_PEAK_FRACTION = 0.1

DEFAULT_WINDOW_SEC = 5.0
MIN_WINDOW_SEC = 0.5
MAX_WINDOW_SEC = 60.0

OUTPUT_SUBDIR = 'ThetaSpikeViewer_Snapshots'  # under each session folder


# ============================================================================
# Cell list
# ============================================================================

def build_cell_list_from_excel(excel_path: Path, filter_mode: str) -> list[dict]:
    df = pd.read_excel(excel_path, sheet_name='ThetaPhase')
    df = df[df['PrecessionTested'] == True]  # noqa: E712

    if filter_mode == 'precessing':
        df = df[df['is_precessing'] == True]  # noqa: E712
    elif filter_mode == 'recessing':
        df = df[df['is_recessing'] == True]  # noqa: E712
    else:
        df = df[(df['is_precessing'] == True) | (df['is_recessing'] == True)]  # noqa: E712

    cells = []
    for _, row in df.iterrows():
        cells.append(dict(
            folder=Path(row['FolderPath']), ntt_file=row['ntt_file'],
            cell_number=int(row['cell_number']), unit_label=row['Unit'],
            is_precessing=bool(row['is_precessing']), is_recessing=bool(row['is_recessing']),
            rho=row.get('rho', np.nan), p=row.get('precession_p', np.nan),
            slope=row.get('slope_deg_per_pass', np.nan), tmi=row.get('TMI', np.nan),
        ))
    if not cells:
        raise ValueError(f'No units match PRECESSION_FILTER={filter_mode!r} in {excel_path}')
    return cells


def build_cell_list_manual(ntt_file: Path, cell_number) -> list[dict]:
    units = load_ntt_spike_times(ntt_file)
    if not units:
        raise ValueError(f'No sorted units found in {ntt_file}')
    if cell_number is None:
        cell_number = max(units, key=lambda c: len(units[c]))
        print(f'CELL_NUMBER=None -> auto-picked cluster {cell_number} ({len(units[cell_number])} spikes)')
    elif cell_number not in units:
        raise ValueError(f'Cluster {cell_number} not found in {ntt_file} (available: {sorted(units)})')

    unit_label = f'{ntt_file.stem}_cell{cell_number}' if len(units) > 1 else ntt_file.stem
    return [dict(
        folder=ntt_file.parent, ntt_file=ntt_file.name, cell_number=cell_number,
        unit_label=unit_label, is_precessing=None, is_recessing=None,
        rho=np.nan, p=np.nan, slope=np.nan, tmi=np.nan,
    )]


# ============================================================================
# Viewer
# ============================================================================

class ThetaSpikeViewer:
    def __init__(self, cells: list[dict]):
        self.cells = cells
        self.cell_idx = 0
        self._lfp_cache = {}  # folder (str) -> (lfp_sig, lfp_ts, lfp_fs, filtered_lfp, lfp_phase_unwrapped)
        self._tracking_cache = {}  # folder (str) -> (pos_ts, pos_xy) or None if no tracking file

        self.window_sec = DEFAULT_WINDOW_SEC
        self.t_start = 0.0

        self.fig = plt.figure(figsize=(14, 7.5))
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1.2, 1],
                                    left=0.08, right=0.97, top=0.90, bottom=0.30, hspace=0.15)
        self.ax_lfp = self.fig.add_subplot(gs[0])
        self.ax_phase = self.fig.add_subplot(gs[1], sharex=self.ax_lfp)
        self.ax_phase.set_ylim(0, 360)
        self.ax_phase.set_yticks([0, 90, 180, 270, 360])
        self.ax_phase.set_xlabel('Time (s)')
        self.ax_phase.set_ylabel('Theta phase (deg)')
        self.ax_lfp.set_ylabel('LFP (uV)')

        self._build_widgets()
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self._load_cell(self.cell_idx)
        plt.show()

    # -- widgets --------------------------------------------------------

    def _build_widgets(self):
        ax_t = self.fig.add_axes([0.10, 0.20, 0.65, 0.03])
        ax_w = self.fig.add_axes([0.10, 0.15, 0.65, 0.03])
        self.slider_t = Slider(ax_t, 'Start time (s)', 0.0, 1.0, valinit=0.0)
        self.slider_w = Slider(ax_w, 'Window (s)', MIN_WINDOW_SEC, MAX_WINDOW_SEC, valinit=DEFAULT_WINDOW_SEC)
        self.slider_t.on_changed(self._on_slider_t)
        self.slider_w.on_changed(self._on_slider_w)

        ax_goto = self.fig.add_axes([0.10, 0.115, 0.20, 0.03])
        self.textbox_goto = TextBox(ax_goto, 'Go to t=', initial='0.0')
        self.textbox_goto.on_submit(self._on_goto)

        specs = [
            ('prev_win', '<< Win', 0.10, 0.05),
            ('next_win', 'Win >>', 0.19, 0.05),
            ('prev_spk', '< Spike', 0.29, 0.09),
            ('next_spk', 'Spike >', 0.39, 0.09),
            ('save', 'Save PNG', 0.49, 0.10),
            ('prev_cell', '<< Cell', 0.64, 0.10),
            ('next_cell', 'Cell >>', 0.75, 0.10),
        ]
        self.buttons = {}
        for name, label, x, w in specs:
            ax_b = self.fig.add_axes([x, 0.05, w, 0.05])
            btn = Button(ax_b, label)
            self.buttons[name] = btn
        self.buttons['prev_win'].on_clicked(lambda evt: self.step_window(-1))
        self.buttons['next_win'].on_clicked(lambda evt: self.step_window(+1))
        self.buttons['prev_spk'].on_clicked(lambda evt: self.jump_to_spike(-1))
        self.buttons['next_spk'].on_clicked(lambda evt: self.jump_to_spike(+1))
        self.buttons['save'].on_clicked(lambda evt: self.save_snapshot())
        self.buttons['prev_cell'].on_clicked(lambda evt: self.load_relative_cell(-1))
        self.buttons['next_cell'].on_clicked(lambda evt: self.load_relative_cell(+1))
        if len(self.cells) <= 1:
            self.buttons['prev_cell'].ax.set_visible(False)
            self.buttons['next_cell'].ax.set_visible(False)

    def _on_key(self, event):
        if event.key == 'right':
            self.step_window(+1)
        elif event.key == 'left':
            self.step_window(-1)
        elif event.key == 'up':
            self.slider_w.set_val(min(MAX_WINDOW_SEC, self.window_sec * 1.5))
        elif event.key == 'down':
            self.slider_w.set_val(max(MIN_WINDOW_SEC, self.window_sec / 1.5))
        elif event.key == '.':
            self.jump_to_spike(+1)
        elif event.key == ',':
            self.jump_to_spike(-1)
        elif event.key == 's':
            self.save_snapshot()
        elif event.key == 'n':
            self.load_relative_cell(+1)
        elif event.key == 'p':
            self.load_relative_cell(-1)

    # -- data loading -----------------------------------------------------

    def _load_cell(self, idx):
        self.cell_idx = idx % len(self.cells)
        cell = self.cells[self.cell_idx]

        folder_key = str(cell['folder'])
        if folder_key not in self._lfp_cache:
            ncs_files = sorted(cell['folder'].glob('*.ncs'), key=_natural_key)
            if not ncs_files:
                raise FileNotFoundError(f'No .ncs file found in {cell["folder"]}')
            lfp_sig, lfp_ts, lfp_fs = load_ncs(ncs_files[0])
            filtered_lfp = bandpass_filter(lfp_sig, LFP_FILTER_BAND[0], LFP_FILTER_BAND[1], lfp_fs)
            lfp_phase_unwrapped = np.unwrap(np.angle(signal.hilbert(filtered_lfp)))
            self._lfp_cache[folder_key] = dict(
                lfp_sig=lfp_sig, lfp_ts=lfp_ts, lfp_fs=lfp_fs,
                filtered_lfp=filtered_lfp, lfp_phase_unwrapped=lfp_phase_unwrapped,
            )
        lfp = self._lfp_cache[folder_key]

        units = load_ntt_spike_times(cell['folder'] / cell['ntt_file'])
        spk_ts_all = units.get(cell['cell_number'], np.array([]))

        if IN_FIELD_ONLY and len(spk_ts_all):
            spk_ts = self._filter_infield(cell['folder'], spk_ts_all)
            n_dropped = len(spk_ts_all) - len(spk_ts)
            if n_dropped:
                print(f"{cell['unit_label']}: {n_dropped}/{len(spk_ts_all)} spike(s) outside the "
                      f"place field (rate < {FIELD_PEAK_FRACTION:.0%} of peak) excluded")
        else:
            spk_ts = spk_ts_all

        spk_phase_deg = np.degrees(assign_spike_phase(spk_ts, lfp['lfp_ts'], lfp['lfp_phase_unwrapped'])) \
            if len(spk_ts) else np.array([])

        self.cur = dict(cell=cell, spk_ts=spk_ts, spk_phase_deg=spk_phase_deg, **lfp)

        t_min, t_max = float(lfp['lfp_ts'][0]), float(lfp['lfp_ts'][-1])
        self.t_min, self.t_max = t_min, t_max
        self.window_sec = min(self.window_sec, max(MIN_WINDOW_SEC, t_max - t_min))
        self._set_slider_range(self.slider_w, MIN_WINDOW_SEC, min(MAX_WINDOW_SEC, t_max - t_min))
        self.slider_w.set_val(self.window_sec)
        self.set_window_start(t_min, redraw=False)
        self._redraw()

    def load_relative_cell(self, step):
        self._load_cell(self.cell_idx + step)

    def _filter_infield(self, folder, spk_ts):
        """Keep only spikes fired while the animal was inside the place
        field, i.e. in a rate-map bin with rate > FIELD_PEAK_FRACTION * peak
        rate -- the same definition ThetaMod_PhasePrecession_v2.auto_filter_band
        uses to estimate field diameter for the 'place' method. Falls back to
        returning all spikes unchanged if no tracking file is found.
        """
        folder_key = str(folder)
        if folder_key not in self._tracking_cache:
            try:
                tracking_path = _find_tracking_file(folder)
                self._tracking_cache[folder_key] = load_tracking(tracking_path, _PIPELINE_TRACKING_TIME_UNIT)
            except FileNotFoundError:
                print(f'No tracking file found in {folder} -- showing all spikes unfiltered.')
                self._tracking_cache[folder_key] = None
        tracking = self._tracking_cache[folder_key]
        if tracking is None:
            return spk_ts

        pos_ts, pos_xy = tracking
        n_dims = pos_xy.shape[1]
        binside = 2.0 * n_dims if _PIPELINE_BINSIDE == 'auto' else _PIPELINE_BINSIDE
        smth_width = 3.0 * binside if _PIPELINE_SMTH_WIDTH == 'auto' else _PIPELINE_SMTH_WIDTH

        spk_xy, _ = spk_pos(pos_ts, pos_xy, spk_ts)
        rmap, _, x_edges, y_edges = rate_map(pos_ts, pos_xy, spk_xy, binside, smth_width)
        peak = np.nanmax(rmap)
        field_mask = rmap > FIELD_PEAK_FRACTION * peak

        xi = np.clip(np.digitize(spk_xy[:, 0], x_edges) - 1, 0, len(x_edges) - 2)
        yi = np.clip(np.digitize(spk_xy[:, 1], y_edges) - 1, 0, len(y_edges) - 2)
        return spk_ts[field_mask[xi, yi]]

    # -- window navigation --------------------------------------------------

    def _set_slider_range(self, slider, vmin, vmax):
        vmax = max(vmax, vmin)
        slider.valmin, slider.valmax = vmin, vmax
        slider.ax.set_xlim(vmin, vmax)

    def set_window_start(self, t_start, redraw=True):
        t_start = float(np.clip(t_start, self.t_min, max(self.t_min, self.t_max - self.window_sec)))
        self.t_start = t_start
        self._set_slider_range(self.slider_t, self.t_min, max(self.t_min, self.t_max - self.window_sec))
        # avoid re-entrant on_changed recursion
        self.slider_t.eventson = False
        self.slider_t.set_val(t_start)
        self.slider_t.eventson = True
        if redraw:
            self._redraw()

    def step_window(self, direction):
        self.set_window_start(self.t_start + direction * self.window_sec)

    def jump_to_spike(self, direction):
        spk_ts = self.cur['spk_ts']
        if len(spk_ts) == 0:
            return
        center = self.t_start + self.window_sec / 2.0
        if direction > 0:
            idx = np.searchsorted(spk_ts, center, side='right')
            if idx >= len(spk_ts):
                return
            target = spk_ts[idx]
        else:
            idx = np.searchsorted(spk_ts, center, side='left') - 1
            if idx < 0:
                return
            target = spk_ts[idx]
        self.set_window_start(target - self.window_sec / 2.0)

    def _on_slider_t(self, val):
        self.t_start = val
        self._redraw()

    def _on_slider_w(self, val):
        self.window_sec = val
        self.set_window_start(self.t_start)  # re-clamps range + redraws

    def _on_goto(self, text):
        try:
            self.set_window_start(float(text))
        except ValueError:
            pass

    # -- rendering --------------------------------------------------------

    def _redraw(self):
        t0, t1 = self.t_start, self.t_start + self.window_sec
        lfp_ts, lfp_sig = self.cur['lfp_ts'], self.cur['lfp_sig']
        filtered_lfp = self.cur['filtered_lfp']
        lfp_phase_unwrapped = self.cur['lfp_phase_unwrapped']
        spk_ts, spk_phase_deg = self.cur['spk_ts'], self.cur['spk_phase_deg']

        i0, i1 = np.searchsorted(lfp_ts, [t0, t1])
        ts_win = lfp_ts[i0:i1]

        self.ax_lfp.cla()
        self.ax_phase.cla()

        if len(ts_win) > 1:
            self.ax_lfp.plot(ts_win, lfp_sig[i0:i1], color='0.7', linewidth=0.7, label='raw LFP')
            self.ax_lfp.plot(ts_win, filtered_lfp[i0:i1], color='tab:blue', linewidth=1.2,
                              label=f'theta {LFP_FILTER_BAND[0]:g}-{LFP_FILTER_BAND[1]:g} Hz')
            self.ax_lfp.legend(loc='upper right', fontsize=8)

            phase_deg_win = np.degrees(np.mod(lfp_phase_unwrapped[i0:i1], 2 * np.pi))
            plot_phase = phase_deg_win.copy()
            jump = np.where(np.abs(np.diff(plot_phase)) > 180)[0]
            plot_phase[jump] = np.nan
            self.ax_phase.plot(ts_win, plot_phase, color='tab:blue', linewidth=1.0)

        for ref in (0, 90, 180, 270, 360):
            self.ax_phase.axhline(ref, color='0.85', linewidth=0.5, zorder=0)

        spk_mask = (spk_ts >= t0) & (spk_ts <= t1)
        spk_win, spk_phase_win = spk_ts[spk_mask], spk_phase_deg[spk_mask]
        if len(spk_win):
            ylo, yhi = self.ax_lfp.get_ylim() if len(ts_win) > 1 else (-1, 1)
            self.ax_lfp.vlines(spk_win, ylo, yhi, colors='crimson', linewidth=1.0, alpha=0.7, zorder=5)
            self.ax_lfp.set_ylim(ylo, yhi)
            self.ax_phase.scatter(spk_win, spk_phase_win, s=28, color='crimson',
                                   edgecolors='k', linewidths=0.3, zorder=5)

        self.ax_lfp.set_xlim(t0, t1)
        self.ax_phase.set_ylim(0, 360)
        self.ax_phase.set_yticks([0, 90, 180, 270, 360])
        self.ax_lfp.set_ylabel('LFP (uV)')
        self.ax_phase.set_ylabel('Theta phase (deg)')
        self.ax_phase.set_xlabel('Time (s)')

        cell = self.cur['cell']
        flag = ''
        if cell['is_precessing'] is not None:
            tag = 'PRECESSING' if cell['is_precessing'] else ('RECESSING' if cell['is_recessing'] else '')
            flag = (f" | {tag}  rho={cell['rho']:.2f} p={cell['p']:.3g} "
                    f"slope={cell['slope']:.1f} deg/pass  TMI={cell['tmi']:.2f}")
        self.ax_lfp.set_title(
            f"[{self.cell_idx + 1}/{len(self.cells)}] {cell['unit_label']}{flag}\n"
            f"t = {t0:.2f}-{t1:.2f} s ({self.window_sec:.2f} s window) | "
            f"{len(spk_win)} spike(s) shown", fontsize=10)

        self.fig.canvas.draw_idle()

    # -- saving --------------------------------------------------------

    def save_snapshot(self):
        cell = self.cur['cell']
        out_dir = cell['folder'] / OUTPUT_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        t0, t1 = self.t_start, self.t_start + self.window_sec
        out_path = out_dir / f"{cell['unit_label']}_t{t0:.1f}-{t1:.1f}s.png"
        self.fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved snapshot: {out_path}')
        self.ax_lfp.set_title(self.ax_lfp.get_title() + '\n[saved]', fontsize=10)
        self.fig.canvas.draw_idle()


# ============================================================================
# Main
# ============================================================================

def main():
    if EXCEL_SUMMARY_PATH is not None and Path(EXCEL_SUMMARY_PATH).exists():
        print(f'Loading flagged units from {EXCEL_SUMMARY_PATH} (filter={PRECESSION_FILTER!r})')
        cells = build_cell_list_from_excel(Path(EXCEL_SUMMARY_PATH), PRECESSION_FILTER)
        print(f'{len(cells)} unit(s) match. Use the Cell buttons / n / p keys to page through them.')
    else:
        print('EXCEL_SUMMARY_PATH not set/found -- viewing single cell from NTT_FILE/CELL_NUMBER.')
        cells = build_cell_list_manual(NTT_FILE, CELL_NUMBER)

    ThetaSpikeViewer(cells)


if __name__ == '__main__':
    main()
