# -*- coding: utf-8 -*-
"""
Interactive viewer: scroll through a unit's cleaned LFP + theta-band trace
with its spikes overlaid, and a second panel showing the instantaneous theta
phase at which each spike fired. A time slider and a window-length slider
let you scrub through the whole recording at an adjustable zoom (default
~5 s); "Save snapshot" writes the currently visible window to PNG.

Two ways to pick which cell(s) to look at:

  1. EXCEL_SUMMARY_PATH set to the theta_phase_Hilbert.xlsx produced by
     ThetaMod_PhasePrec_v22_fitR_shuffle.py (default: that script's
     RESULTS_FOLDER / OUTPUT_EXCEL_NAME) -- the viewer builds a list of every
     precession-tested unit whose PrecessionClass is in CLASS_FILTER and you
     page through them with the Prev/Next cell buttons ('p' / 'n').
  2. EXCEL_SUMMARY_PATH = None -- the viewer opens exactly one cell, given
     by NTT_FILE / CELL_NUMBER (same convention as
     Understanding_PassIndex/Viz_PassIndex_AllSteps.py).

All Neuralynx I/O and theta-phase math are imported unchanged from
ThetaMod_PhasePrec_v22_fitR_shuffle.py so the phase values shown here match
the batch pipeline exactly:
  - LFP: load_ncs -> clean_lfp (resample to 1 kHz, 50/100/150/200 Hz notch,
    detrend) -> bandpass_filter (LFP_FILTER_BAND) -> hilbert_phase (0 deg at
    a theta peak), unwrapped; spike phase via assign_spike_phase (NaN for
    spikes in an LFP gap).
  - LFP channel: the 'lfp_file' recorded in the Excel row; otherwise
    match_ncs_to_ntt if the pipeline's MATCH_SPIKE_LFP_FILE is True, else the
    session's first .ncs file (natural sort), as in process_session.
  - ACG theta-positive epochs (if the pipeline's USE_ACG_THETA_EPOCHS):
    acg_theta_epoch_mask on the cleaned LFP. Theta-positive epochs are shaded
    green; spikes inside them (the ones the pipeline uses for TMI / phase
    precession) are drawn in crimson, spikes outside them in open grey.

By default (IN_FIELD_ONLY = True) only spikes fired inside the place field
are shown/counted -- "in field" defined the same way
ThetaMod_PhasePrec_v22_fitR_shuffle.auto_filter_band defines it for the
'place' method: a rate-map bin (pipeline BINSIDE / SMTH_WIDTH, built from all
spikes in the tracking/LFP overlap window) with rate > FIELD_PEAK_FRACTION
(20%) of the cell's peak rate. Set IN_FIELD_ONLY = False to see all spikes.

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
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Reused unchanged from the batch pipeline so numbers match exactly.
from ThetaMod_PhasePrec_v22_fitR_shuffle import (
    load_ncs, load_ntt_spike_times, clean_lfp, bandpass_filter, hilbert_phase,
    assign_spike_phase, acg_theta_epoch_mask, times_in_ok_epochs, match_ncs_to_ntt,
    _natural_key, load_tracking, spk_pos, rate_map, _find_tracking_file,
    LFP_FILTER_BAND as _PIPELINE_LFP_BAND,
    BINSIDE as _PIPELINE_BINSIDE, SMTH_WIDTH as _PIPELINE_SMTH_WIDTH,
    TRACKING_TIME_UNIT as _PIPELINE_TRACKING_TIME_UNIT,
    MATCH_SPIKE_LFP_FILE as _PIPELINE_MATCH_SPIKE_LFP_FILE,
    USE_ACG_THETA_EPOCHS as _PIPELINE_USE_ACG_THETA_EPOCHS,
    ACG_ED_MIN_THRESH as _PIPELINE_ACG_ED_MIN_THRESH,
    ACG_EPOCH_SEC as _PIPELINE_ACG_EPOCH_SEC,
    RESULTS_FOLDER as _PIPELINE_RESULTS_FOLDER,
    OUTPUT_EXCEL_NAME as _PIPELINE_OUTPUT_EXCEL_NAME,
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
EXCEL_SUMMARY_PATH = _PIPELINE_RESULTS_FOLDER / _PIPELINE_OUTPUT_EXCEL_NAME
# PrecessionClass values to page through (v22 classes: 'phase_precessing',
# 'phase_succeeding', 'phase_locked', 'no_phase_relation', 'non_precessing').
# None = every precession-tested unit.
CLASS_FILTER = ('phase_precessing', 'phase_succeeding')

# --- Option 2: view exactly one cell (used when EXCEL_SUMMARY_PATH is None
# or doesn't exist) ---
#NTT_FILE = Path(r"C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True/Fa8477/Day5/1Stnd/TT1_SS_01.ntt")
#CELL_NUMBER = None  # None = auto-pick the sorted cluster with the most spikes

LFP_FILTER_BAND = _PIPELINE_LFP_BAND  # Hz, theta band for the phase panel

# Restrict displayed spikes to the place field, defined exactly as in
# ThetaMod_PhasePrec_v22_fitR_shuffle.auto_filter_band: rate-map bins with
# firing rate > FIELD_PEAK_FRACTION * peak rate (20% of peak, "place" method).
IN_FIELD_ONLY = True
FIELD_PEAK_FRACTION = 0.2

# Shade ACG theta-positive epochs and split spikes into used (inside) vs
# excluded (outside) -- follows the pipeline's USE_ACG_THETA_EPOCHS.
USE_ACG_THETA_EPOCHS = _PIPELINE_USE_ACG_THETA_EPOCHS

DEFAULT_WINDOW_SEC = 5.0
MIN_WINDOW_SEC = 0.5
MAX_WINDOW_SEC = 60.0

OUTPUT_SUBDIR = 'ThetaSpikeViewer_Snapshots'  # under each session folder

_CLASS_TAGS = {
    'phase_precessing': 'PRECESSING', 'phase_succeeding': 'SUCCEEDING',
    'phase_locked': 'PHASE-LOCKED', 'no_phase_relation': 'NO PHASE RELATION',
    'non_precessing': 'NON-PRECESSING (multi-line fit)',
}


# ============================================================================
# Cell list
# ============================================================================

def _str_or_none(val):
    return val if isinstance(val, str) and val.strip() else None


def build_cell_list_from_excel(excel_path: Path, class_filter) -> list[dict]:
    df = pd.read_excel(excel_path, sheet_name='ThetaPhase')
    df = df[df['PrecessionTested'] == True]  # noqa: E712
    if class_filter is not None:
        df = df[df['PrecessionClass'].isin(class_filter)]

    cells = []
    for _, row in df.iterrows():
        cells.append(dict(
            folder=Path(row['FolderPath']), ntt_file=row['ntt_file'],
            cell_number=int(row['cell_number']), unit_label=row['Unit'],
            lfp_file=_str_or_none(row.get('lfp_file')),
            precession_class=_str_or_none(row.get('PrecessionClass')),
            rho=row.get('rho', np.nan), p=row.get('precession_p', np.nan),
            slope=row.get('slope_deg_per_pass', np.nan),
            fit_R=row.get('fit_R', np.nan), fit_R_p=row.get('fit_R_shuffle_p', np.nan),
            tmi=row.get('TMI', np.nan),
        ))
    if not cells:
        raise ValueError(f'No units match CLASS_FILTER={class_filter!r} in {excel_path}')
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
        unit_label=unit_label, lfp_file=None, precession_class=None,
        rho=np.nan, p=np.nan, slope=np.nan, fit_R=np.nan, fit_R_p=np.nan, tmi=np.nan,
    )]


# ============================================================================
# Viewer
# ============================================================================

class ThetaSpikeViewer:
    def __init__(self, cells: list[dict]):
        self.cells = cells
        self.cell_idx = 0
        self._lfp_cache = {}  # .ncs path (str) -> dict of cleaned LFP, theta, phase, ACG epochs
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

    def _select_ncs(self, cell) -> Path:
        """The .ncs file the pipeline referenced this unit to: the Excel row's
        'lfp_file' if available, else v22 process_session's own choice."""
        folder = cell['folder']
        if cell['lfp_file'] and (folder / cell['lfp_file']).exists():
            return folder / cell['lfp_file']
        ncs_files = sorted(folder.glob('*.ncs'), key=_natural_key)
        if not ncs_files:
            raise FileNotFoundError(f'No .ncs file found in {folder}')
        if _PIPELINE_MATCH_SPIKE_LFP_FILE:
            return match_ncs_to_ntt(folder / cell['ntt_file'], ncs_files)
        return ncs_files[0]

    def _load_lfp(self, ncs_path: Path) -> dict:
        """Cleaned LFP, theta-band trace, unwrapped Hilbert phase and ACG
        theta-positive epochs, computed exactly as in v22 process_session."""
        key = str(ncs_path)
        if key not in self._lfp_cache:
            print(f'Loading LFP: {ncs_path}')
            lfp_sig, lfp_ts, lfp_fs = load_ncs(ncs_path)
            lfp_sig, lfp_ts, lfp_fs = clean_lfp(lfp_sig, lfp_ts, lfp_fs)
            filtered_lfp = bandpass_filter(lfp_sig, LFP_FILTER_BAND[0], LFP_FILTER_BAND[1], lfp_fs)
            lfp_phase_unwrapped = np.unwrap(hilbert_phase(filtered_lfp, lfp_fs)[0])

            acg_epoch_ok = None
            if USE_ACG_THETA_EPOCHS:
                acg_epoch_ok, _ = acg_theta_epoch_mask(lfp_sig)
                print(f'  ACG theta-positive epochs (ED_min < {_PIPELINE_ACG_ED_MIN_THRESH:g}): '
                      f'{int(acg_epoch_ok.sum())}/{len(acg_epoch_ok)}')

            self._lfp_cache[key] = dict(
                lfp_name=ncs_path.name, lfp_sig=lfp_sig, lfp_ts=lfp_ts, lfp_fs=lfp_fs,
                filtered_lfp=filtered_lfp, lfp_phase_unwrapped=lfp_phase_unwrapped,
                acg_epoch_ok=acg_epoch_ok,
            )
        return self._lfp_cache[key]

    def _load_tracking(self, folder):
        folder_key = str(folder)
        if folder_key not in self._tracking_cache:
            try:
                tracking_path = _find_tracking_file(folder)
                self._tracking_cache[folder_key] = load_tracking(tracking_path, _PIPELINE_TRACKING_TIME_UNIT)
            except FileNotFoundError:
                print(f'No tracking file found in {folder} -- showing all spikes unfiltered.')
                self._tracking_cache[folder_key] = None
        return self._tracking_cache[folder_key]

    def _load_cell(self, idx):
        self.cell_idx = idx % len(self.cells)
        cell = self.cells[self.cell_idx]

        lfp = self._load_lfp(self._select_ncs(cell))

        units = load_ntt_spike_times(cell['folder'] / cell['ntt_file'])
        spk_ts_all = units.get(cell['cell_number'], np.array([]))

        if IN_FIELD_ONLY and len(spk_ts_all):
            spk_ts = self._filter_infield(cell['folder'], spk_ts_all, lfp['lfp_ts'])
            n_dropped = len(spk_ts_all) - len(spk_ts)
            if n_dropped:
                print(f"{cell['unit_label']}: {n_dropped}/{len(spk_ts_all)} spike(s) outside the "
                      f"place field (rate <= {FIELD_PEAK_FRACTION:.0%} of peak) excluded")
        else:
            spk_ts = spk_ts_all

        # Spikes inside ACG theta-positive epochs = the spikes the pipeline uses.
        if USE_ACG_THETA_EPOCHS and len(spk_ts):
            spk_in_theta = times_in_ok_epochs(spk_ts, lfp['lfp_ts'][0], lfp['acg_epoch_ok'])
        else:
            spk_in_theta = np.ones(len(spk_ts), dtype=bool)

        spk_phase_deg = (np.degrees(assign_spike_phase(spk_ts, lfp['lfp_ts'], lfp['lfp_phase_unwrapped'],
                                                       lfp['lfp_fs']))
                         if len(spk_ts) else np.array([]))

        self.cur = dict(cell=cell, spk_ts=spk_ts, spk_phase_deg=spk_phase_deg,
                        spk_in_theta=spk_in_theta, **lfp)

        t_min, t_max = float(lfp['lfp_ts'][0]), float(lfp['lfp_ts'][-1])
        self.t_min, self.t_max = t_min, t_max
        self.window_sec = min(self.window_sec, max(MIN_WINDOW_SEC, t_max - t_min))
        self._set_slider_range(self.slider_w, MIN_WINDOW_SEC, min(MAX_WINDOW_SEC, t_max - t_min))
        self.slider_w.set_val(self.window_sec)
        self.set_window_start(t_min, redraw=False)
        self._redraw()

    def load_relative_cell(self, step):
        self._load_cell(self.cell_idx + step)

    def _filter_infield(self, folder, spk_ts, lfp_ts):
        """Keep only spikes fired while the animal was inside the place
        field, i.e. in a rate-map bin with rate > FIELD_PEAK_FRACTION * peak
        rate -- the same definition ThetaMod_PhasePrec_v22_fitR_shuffle
        .auto_filter_band uses to estimate field diameter for the 'place'
        method. As in v22, the rate map is built from all spikes in the
        tracking/LFP overlap window. Falls back to returning all spikes
        unchanged if no tracking file is found.
        """
        tracking = self._load_tracking(folder)
        if tracking is None:
            return spk_ts

        pos_ts, pos_xy = tracking
        t_start, t_stop = max(pos_ts.min(), lfp_ts.min()), min(pos_ts.max(), lfp_ts.max())
        spk_ts = spk_ts[(spk_ts >= t_start) & (spk_ts <= t_stop)]
        if not len(spk_ts):
            return spk_ts

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

    def _shade_theta_epochs(self, t0, t1):
        """Shade the ACG theta-positive epochs overlapping [t0, t1] (contiguous
        epochs merged into one span) on both panels."""
        epoch_ok = self.cur['acg_epoch_ok']
        if epoch_ok is None or not len(epoch_ok):
            return
        lfp_t0 = self.cur['lfp_ts'][0]
        e0 = max(0, int(np.floor((t0 - lfp_t0) / _PIPELINE_ACG_EPOCH_SEC)))
        e1 = min(len(epoch_ok), int(np.ceil((t1 - lfp_t0) / _PIPELINE_ACG_EPOCH_SEC)))
        if e1 <= e0:
            return
        ok = np.concatenate(([False], epoch_ok[e0:e1], [False])).astype(np.int8)
        starts = np.flatnonzero(np.diff(ok) == 1) + e0
        stops = np.flatnonzero(np.diff(ok) == -1) + e0
        for s, e in zip(starts, stops):
            for ax in (self.ax_lfp, self.ax_phase):
                ax.axvspan(lfp_t0 + s * _PIPELINE_ACG_EPOCH_SEC, lfp_t0 + e * _PIPELINE_ACG_EPOCH_SEC,
                           color='tab:green', alpha=0.10, lw=0, zorder=0)

    def _redraw(self):
        t0, t1 = self.t_start, self.t_start + self.window_sec
        lfp_ts, lfp_sig = self.cur['lfp_ts'], self.cur['lfp_sig']
        filtered_lfp = self.cur['filtered_lfp']
        lfp_phase_unwrapped = self.cur['lfp_phase_unwrapped']
        spk_ts, spk_phase_deg = self.cur['spk_ts'], self.cur['spk_phase_deg']
        spk_in_theta = self.cur['spk_in_theta']

        i0, i1 = np.searchsorted(lfp_ts, [t0, t1])
        ts_win = lfp_ts[i0:i1]

        self.ax_lfp.cla()
        self.ax_phase.cla()

        if len(ts_win) > 1:
            self.ax_lfp.plot(ts_win, lfp_sig[i0:i1], color='0.7', linewidth=0.7,
                             label='cleaned LFP (1 kHz, notched, detrended)')
            self.ax_lfp.plot(ts_win, filtered_lfp[i0:i1], color='tab:blue', linewidth=1.2,
                              label=f'theta {LFP_FILTER_BAND[0]:g}-{LFP_FILTER_BAND[1]:g} Hz')

            phase_deg_win = np.degrees(np.mod(lfp_phase_unwrapped[i0:i1], 2 * np.pi))
            plot_phase = phase_deg_win.copy()
            jump = np.where(np.abs(np.diff(plot_phase)) > 180)[0]
            plot_phase[jump] = np.nan
            self.ax_phase.plot(ts_win, plot_phase, color='tab:blue', linewidth=1.0)

        if USE_ACG_THETA_EPOCHS:
            self._shade_theta_epochs(t0, t1)

        for ref in (0, 90, 180, 270, 360):
            self.ax_phase.axhline(ref, color='0.85', linewidth=0.5, zorder=0)

        spk_mask = (spk_ts >= t0) & (spk_ts <= t1)
        spk_win, spk_phase_win = spk_ts[spk_mask], spk_phase_deg[spk_mask]
        used_win = spk_in_theta[spk_mask]
        if len(spk_win):
            ylo, yhi = self.ax_lfp.get_ylim() if len(ts_win) > 1 else (-1, 1)
            self.ax_lfp.vlines(spk_win[used_win], ylo, yhi, colors='crimson', linewidth=1.0,
                               alpha=0.7, zorder=5)
            self.ax_lfp.vlines(spk_win[~used_win], ylo, yhi, colors='0.5', linewidth=1.0,
                               linestyles='--', alpha=0.7, zorder=5)
            self.ax_lfp.set_ylim(ylo, yhi)
            self.ax_phase.scatter(spk_win[used_win], spk_phase_win[used_win], s=28, color='crimson',
                                   edgecolors='k', linewidths=0.3, zorder=5)
            self.ax_phase.scatter(spk_win[~used_win], spk_phase_win[~used_win], s=28,
                                   facecolors='none', edgecolors='0.4', linewidths=0.8, zorder=5)
        if len(ts_win) > 1:
            handles, labels = self.ax_lfp.get_legend_handles_labels()
            if USE_ACG_THETA_EPOCHS:
                handles += [Patch(color='tab:green', alpha=0.25),
                            Line2D([0], [0], color='crimson'),
                            Line2D([0], [0], color='0.5', linestyle='--')]
                labels += ['ACG theta-positive epoch', 'spike (theta epoch, used)',
                           'spike (outside theta epoch, excluded)']
            self.ax_lfp.legend(handles, labels, loc='upper right', fontsize=8)

        self.ax_lfp.set_xlim(t0, t1)
        self.ax_phase.set_ylim(0, 360)
        self.ax_phase.set_yticks([0, 90, 180, 270, 360])
        self.ax_lfp.set_ylabel('LFP (uV)')
        self.ax_phase.set_ylabel('Theta phase (deg, 0 = peak)')
        self.ax_phase.set_xlabel('Time (s)')

        cell = self.cur['cell']
        flag = ''
        if cell['precession_class'] is not None:
            tag = _CLASS_TAGS.get(cell['precession_class'], cell['precession_class'])
            flag = (f" | {tag}  rho={cell['rho']:.2f} p={cell['p']:.3g} "
                    f"slope={cell['slope']:.1f} deg/pass  fitR={cell['fit_R']:.3f} "
                    f"(shuffle p={cell['fit_R_p']:.3g})  TMI={cell['tmi']:.2f}")
        n_used = int(used_win.sum())
        self.ax_lfp.set_title(
            f"[{self.cell_idx + 1}/{len(self.cells)}] {cell['unit_label']} (LFP: {self.cur['lfp_name']}){flag}\n"
            f"t = {t0:.2f}-{t1:.2f} s ({self.window_sec:.2f} s window) | "
            f"{len(spk_win)} spike(s) shown"
            + (f', {n_used} in theta epochs' if USE_ACG_THETA_EPOCHS else ''), fontsize=10)

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
        print(f'Loading flagged units from {EXCEL_SUMMARY_PATH} (CLASS_FILTER={CLASS_FILTER!r})')
        cells = build_cell_list_from_excel(Path(EXCEL_SUMMARY_PATH), CLASS_FILTER)
        print(f'{len(cells)} unit(s) match. Use the Cell buttons / n / p keys to page through them.')
    else:
        print('EXCEL_SUMMARY_PATH not set/found -- viewing single cell from NTT_FILE/CELL_NUMBER.')
        cells = build_cell_list_manual(NTT_FILE, CELL_NUMBER)

    ThetaSpikeViewer(cells)


if __name__ == '__main__':
    main()
