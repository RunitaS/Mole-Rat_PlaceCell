# -*- coding: utf-8 -*-
"""
Interactive viewer: scroll through a unit's cleaned LFP + theta-band trace
with its spikes overlaid, and a second panel showing the instantaneous theta
phase at which each spike fired. A time slider and a window-length slider
let you scrub through the whole recording at an adjustable zoom (default
~5 s); "Save snapshot" writes the currently visible window to PNG.

V3: in-field epochs come from the PASS INDEX of ThetaMod_PhasePrec_v22_fitR_shuffle
(Climer et al. 2013), computed exactly as that pipeline's compute_pass_index does:
rate map from all spikes -> field index map -> field index along the
trajectory -> resampled along arc length -> spatially bandpass-filtered
(auto_filter_band) -> Hilbert phase. Pass index = that phase / pi, in [-1, 1):
-1 field entry, 0 field-index peak (field center), +1 field exit. Each cycle of
pass index from -1 to +1 is one pass; with INFIELD_EPOCH_MODE = 'pass' a whole
pass is IN THE PLACE FIELD when the (unfiltered) field index reaches
FIELD_PEAK_FRACTION of the cell's peak rate during it (0.2 = auto_filter_band's
own field definition); with 'field_index' only the part of the pass where the
field index is above that fraction counts. In-field epochs are shaded light red
in both panels, and a spike is "in field" if it fires during one of them. The in-field spike timestamps (with their pass index and theta
phase) and the in-field epochs are written per cell to
<session>/ThetaSpikeViewer_Snapshots/<unit>_InFieldSpikes.csv / _InFieldEpochs.csv.

Two ways to pick which cell(s) to look at:

  1. EXCEL_SUMMARY_PATH set to the theta_phase xlsx produced by
     ThetaMod_PhasePrec_v22_fitR_shuffle.py -- the viewer builds a list of every
     unit flagged is_precessing / is_recessing (per PRECESSION_FILTER) and
     you page through them with the Prev/Next cell buttons ('p' / 'n').
  2. EXCEL_SUMMARY_PATH = None (or missing) -- the viewer opens exactly one
     cell, given by NTT_FILE / CELL_NUMBER.

All Neuralynx I/O, LFP cleaning, theta-phase and pass-index math are imported
unchanged from ThetaMod_PhasePrec_v22_fitR_shuffle.py so the values shown
here match the batch pipeline exactly.

IN_FIELD_ONLY = True shows/counts only in-field spikes; False shows all spikes
(out-of-field ones in grey).

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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.widgets import Slider, Button, TextBox

# Reused unchanged from the batch pipeline so numbers match exactly.
from ThetaMod_PhasePrec_v22_fitR_shuffle import (
    load_ncs, load_ntt_spike_times, clean_lfp, bandpass_filter, hilbert_phase,
    assign_spike_phase, phase_at, _natural_key, match_ncs_to_ntt,
    load_tracking, spk_pos, rate_map, field_index_map, field_index_per_position,
    sample_along_arc, auto_filter_band, _find_tracking_file,
    LFP_FILTER_BAND as _PIPELINE_LFP_BAND, METHOD as _PIPELINE_METHOD,
    BINSIDE as _PIPELINE_BINSIDE, SMTH_WIDTH as _PIPELINE_SMTH_WIDTH,
    FILTER_BAND as _PIPELINE_FILTER_BAND,
    TRACKING_TIME_UNIT as _PIPELINE_TRACKING_TIME_UNIT,
    MATCH_SPIKE_LFP_FILE as _PIPELINE_MATCH_SPIKE_LFP_FILE,
    RESULTS_FOLDER as _PIPELINE_RESULTS_FOLDER, OUTPUT_EXCEL_NAME as _PIPELINE_EXCEL_NAME,
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
EXCEL_SUMMARY_PATH = _PIPELINE_RESULTS_FOLDER / _PIPELINE_EXCEL_NAME
PRECESSION_FILTER = 'precessing_or_recessing'  # 'precessing' | 'recessing' | 'precessing_or_recessing'

# --- Option 2: view exactly one cell (used when EXCEL_SUMMARY_PATH is None
# or doesn't exist) ---
NTT_FILE = Path(r"C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True/Fa8477/Day5/1Stnd/TT1_SS_01.ntt")
CELL_NUMBER = None  # None = auto-pick the sorted cluster with the most spikes

LFP_FILTER_BAND = _PIPELINE_LFP_BAND  # Hz, theta band for the phase panel

# A pass (one -1 -> +1 cycle of pass index) is in the place field if the field
# index (rate map normalized to the cell's peak) reaches this fraction during it.
# 0.2 = the field definition of ThetaMod_PhasePrec_v22's auto_filter_band.
FIELD_PEAK_FRACTION = 0.2
# 'pass': the whole pass (pass index -1 -> +1, trough to trough of the filtered field
#         index) is in field -- wider than the field itself, as the troughs lie outside it.
# 'field_index': only the part of each pass where the field index >= FIELD_PEAK_FRACTION.
INFIELD_EPOCH_MODE = 'pass'
IN_FIELD_ONLY = True        # True: show only in-field spikes; False: all (out-of-field in grey)
INFIELD_COLOR = '#FF9999'   # background of in-field epochs
INFIELD_ALPHA = 0.25
SAVE_INFIELD_CSV = True     # write <unit>_InFieldSpikes.csv / _InFieldEpochs.csv on loading a cell
PASS_INDEX_CMAP = 'Spectral'  # diverging colormap of the pass-index strip (-1 entry, 0 center, +1 exit)
PASS_INDEX_PLOT_SAMPLES = 2000  # pass-index samples per visible window

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
# In-field epochs from the pass index
# ============================================================================

def _wrap_pm_pi(u):
    return np.mod(u + np.pi, 2 * np.pi) - np.pi


def compute_infield_epochs(pos_ts, pos_xy, spk_ts, field_peak_fraction=FIELD_PEAK_FRACTION,
                           mode=INFIELD_EPOCH_MODE):
    """Pass-index trace and in-field passes for one unit, computed as in
    ThetaMod_PhasePrec_v22's compute_pass_index (same rate map, field index,
    arc-length resampling, spatial filter and Hilbert phase). `spk_ts` (all of
    the unit's spikes in the tracking/LFP overlap) defines the rate map.

    Passes are the runs of constant pass number floor((phase + pi) / 2pi) of the
    unwrapped field phase (its running max, so backward phase jitter can't split
    a pass), i.e. one pass index cycle from -1 (entry) through 0
    (field-index peak) to +1 (exit); together they tile the tracked time.
    mode='pass': a whole pass is in field if the unfiltered field index reaches
    field_peak_fraction in it. mode='field_index': only the stretches of each pass
    where the field index is >= field_peak_fraction are in field.

    Returns (epochs DataFrame [pass_id, t_start, t_end, peak_field_index, in_field],
    pass_index_at(t) and field_index_at(t), functions giving the pass index and
    the unfiltered field index at arbitrary times; NaN outside the tracking).
    """
    n_dims = pos_xy.shape[1]
    binside = 2.0 * n_dims if _PIPELINE_BINSIDE == 'auto' else _PIPELINE_BINSIDE
    smth_width = 3.0 * binside if _PIPELINE_SMTH_WIDTH == 'auto' else _PIPELINE_SMTH_WIDTH

    spk_xy, _ = spk_pos(pos_ts, pos_xy, spk_ts)
    rmap, occupancy, x_edges, y_edges = rate_map(pos_ts, pos_xy, spk_xy, binside, smth_width)
    fi_map = field_index_map(rmap, occupancy, _PIPELINE_METHOD)
    field_index = field_index_per_position(pos_xy, fi_map, x_edges, y_edges)

    cc, ts2, resampled = sample_along_arc(pos_ts, pos_xy, field_index)
    filter_band = (auto_filter_band(_PIPELINE_METHOD, rmap, occupancy, binside, n_dims)
                   if _PIPELINE_FILTER_BAND == 'auto' else _PIPELINE_FILTER_BAND)
    fs_arc = 1.0 / np.mean(np.diff(cc))
    filtered_field_index = bandpass_filter(resampled, filter_band[0], filter_band[1], fs_arc)
    field_phase_unwrapped = np.unwrap(hilbert_phase(filtered_field_index, fs_arc)[0])

    # running max: a brief backward step of the Hilbert phase across a pass
    # boundary must not split one pass into several
    pass_id = np.floor((np.maximum.accumulate(field_phase_unwrapped) + np.pi) / (2 * np.pi)).astype(np.int64)
    above = resampled >= field_peak_fraction
    if mode == 'pass':
        seg_key = pass_id
    elif mode == 'field_index':
        seg_key = 2 * pass_id + above  # split each pass into its above/below-threshold stretches
    else:
        raise ValueError(f"INFIELD_EPOCH_MODE must be 'pass' or 'field_index', not {mode!r}")
    change = np.flatnonzero(np.diff(seg_key) != 0) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(seg_key)]))  # exclusive
    peak_fi = np.maximum.reduceat(resampled, starts)
    epochs = pd.DataFrame(dict(
        pass_id=pass_id[starts],
        t_start=ts2[starts],
        t_end=ts2[np.minimum(stops, len(ts2) - 1)],
        peak_field_index=peak_fi,
        in_field=peak_fi >= field_peak_fraction if mode == 'pass' else above[starts],
    ))

    def pass_index_at(t):
        return _wrap_pm_pi(phase_at(ts2, field_phase_unwrapped, t)) / np.pi

    def field_index_at(t):
        return np.interp(np.asarray(t, dtype=np.float64), ts2, resampled, left=np.nan, right=np.nan)

    return epochs, pass_index_at, field_index_at


def times_in_epochs(t, epochs: pd.DataFrame):
    """Boolean mask over times `t` falling inside any of `epochs` (t_start <= t < t_end)."""
    t = np.asarray(t, dtype=np.float64)
    starts, ends = epochs['t_start'].to_numpy(), epochs['t_end'].to_numpy()
    idx = np.searchsorted(starts, t, side='right') - 1
    ok = idx >= 0
    out = np.zeros(len(t), dtype=bool)
    out[ok] = t[ok] < ends[idx[ok]]
    return out


# ============================================================================
# Viewer
# ============================================================================

class ThetaSpikeViewer:
    def __init__(self, cells: list[dict]):
        self.cells = cells
        self.cell_idx = 0
        self._lfp_cache = {}  # .ncs path (str) -> dict(lfp_sig, lfp_ts, lfp_fs, filtered_lfp, lfp_phase_unwrapped)
        self._tracking_cache = {}  # folder (str) -> (pos_ts, pos_xy) or None if no tracking file

        self.window_sec = DEFAULT_WINDOW_SEC
        self.t_start = 0.0

        self.fig = plt.figure(figsize=(14, 8.5))
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1.2, 1, 0.3],
                                    left=0.08, right=0.92, top=0.90, bottom=0.30, hspace=0.15)
        self.ax_lfp = self.fig.add_subplot(gs[0])
        self.ax_phase = self.fig.add_subplot(gs[1], sharex=self.ax_lfp)
        self.ax_pi = self.fig.add_subplot(gs[2], sharex=self.ax_lfp)
        pos = self.ax_pi.get_position()
        cax = self.fig.add_axes([0.93, pos.y0, 0.01, pos.height])
        cbar = self.fig.colorbar(ScalarMappable(norm=Normalize(-1, 1), cmap=PASS_INDEX_CMAP), cax=cax)
        cbar.set_ticks([-1, 0, 1])
        cbar.ax.tick_params(labelsize=8)
        self.ax_phase.set_ylim(0, 360)
        self.ax_phase.set_yticks([0, 90, 180, 270, 360])
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

    def _load_lfp(self, cell):
        """Cleaned LFP + unwrapped Hilbert theta phase, as in v22's process_session."""
        ncs_files = sorted(cell['folder'].glob('*.ncs'), key=_natural_key)
        if not ncs_files:
            raise FileNotFoundError(f'No .ncs file found in {cell["folder"]}')
        ncs_path = (match_ncs_to_ntt(cell['folder'] / cell['ntt_file'], ncs_files)
                    if _PIPELINE_MATCH_SPIKE_LFP_FILE else ncs_files[0])
        key = str(ncs_path)
        if key not in self._lfp_cache:
            print(f'Loading LFP: {ncs_path}')
            lfp_sig, lfp_ts, lfp_fs = clean_lfp(*load_ncs(ncs_path))
            filtered_lfp = bandpass_filter(lfp_sig, LFP_FILTER_BAND[0], LFP_FILTER_BAND[1], lfp_fs)
            lfp_phase_unwrapped = np.unwrap(hilbert_phase(filtered_lfp, lfp_fs)[0])
            self._lfp_cache[key] = dict(
                lfp_sig=lfp_sig, lfp_ts=lfp_ts, lfp_fs=lfp_fs, lfp_file=ncs_path.name,
                filtered_lfp=filtered_lfp, lfp_phase_unwrapped=lfp_phase_unwrapped,
            )
        return self._lfp_cache[key]

    def _load_tracking(self, folder):
        folder_key = str(folder)
        if folder_key not in self._tracking_cache:
            try:
                tracking_path = _find_tracking_file(folder)
                self._tracking_cache[folder_key] = load_tracking(tracking_path, _PIPELINE_TRACKING_TIME_UNIT)
            except FileNotFoundError:
                print(f'No tracking file found in {folder} -- no pass index / place-field epochs.')
                self._tracking_cache[folder_key] = None
        return self._tracking_cache[folder_key]

    def _load_cell(self, idx):
        self.cell_idx = idx % len(self.cells)
        cell = self.cells[self.cell_idx]
        lfp = self._load_lfp(cell)
        lfp_ts = lfp['lfp_ts']

        units = load_ntt_spike_times(cell['folder'] / cell['ntt_file'])
        spk_ts_all = units.get(cell['cell_number'], np.array([]))

        # ---- in-field epochs (pass index) and in-field spikes ----
        epochs = pass_index_at = field_index_at = None
        spk_in_field = np.ones(len(spk_ts_all), dtype=bool)
        spk_pass_index = np.full(len(spk_ts_all), np.nan)
        tracking = self._load_tracking(cell['folder'])
        if tracking is not None and len(spk_ts_all):
            pos_ts, pos_xy = tracking
            t_lo, t_hi = max(pos_ts.min(), lfp_ts.min()), min(pos_ts.max(), lfp_ts.max())
            overlap = (spk_ts_all >= t_lo) & (spk_ts_all <= t_hi)
            epochs, pass_index_at, field_index_at = compute_infield_epochs(pos_ts, pos_xy, spk_ts_all[overlap])
            spk_in_field = times_in_epochs(spk_ts_all, epochs[epochs['in_field']]) & overlap
            spk_pass_index = pass_index_at(spk_ts_all)
            n_pass = epochs.loc[epochs['in_field'], 'pass_id'].nunique()
            print(f"{cell['unit_label']}: {n_pass} in-field pass(es) "
                  f"(field index >= {FIELD_PEAK_FRACTION:g}, mode={INFIELD_EPOCH_MODE!r}), "
                  f"{int(spk_in_field.sum())}/{len(spk_ts_all)} spike(s) in field")

        spk_phase_deg = (np.degrees(assign_spike_phase(spk_ts_all, lfp_ts, lfp['lfp_phase_unwrapped'],
                                                       lfp['lfp_fs']))
                         if len(spk_ts_all) else np.array([]))

        if epochs is not None and SAVE_INFIELD_CSV:
            self._save_infield_csv(cell, epochs, spk_ts_all, spk_in_field, spk_pass_index, spk_phase_deg)

        keep = spk_in_field if IN_FIELD_ONLY else np.ones(len(spk_ts_all), dtype=bool)
        self.cur = dict(cell=cell, spk_ts=spk_ts_all[keep], spk_phase_deg=spk_phase_deg[keep],
                        spk_in_field=spk_in_field[keep],
                        infield_epochs=None if epochs is None else epochs[epochs['in_field']],
                        pass_index_at=pass_index_at, field_index_at=field_index_at, **lfp)

        t_min, t_max = float(lfp_ts[0]), float(lfp_ts[-1])
        self.t_min, self.t_max = t_min, t_max
        self.window_sec = min(self.window_sec, max(MIN_WINDOW_SEC, t_max - t_min))
        self._set_slider_range(self.slider_w, MIN_WINDOW_SEC, min(MAX_WINDOW_SEC, t_max - t_min))
        self.slider_w.set_val(self.window_sec)
        self.set_window_start(t_min, redraw=False)
        self._redraw()

    def _save_infield_csv(self, cell, epochs, spk_ts, spk_in_field, spk_pass_index, spk_phase_deg):
        """In-field spike timestamps (+ pass index, theta phase, pass number) and
        the in-field epochs, one CSV each, in the session's OUTPUT_SUBDIR."""
        out_dir = cell['folder'] / OUTPUT_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        infield = epochs[epochs['in_field']].reset_index(drop=True)
        spk = spk_ts[spk_in_field]
        epoch_idx = np.searchsorted(infield['t_start'].to_numpy(), spk, side='right') - 1
        pd.DataFrame(dict(
            spike_ts_s=spk,
            pass_index=spk_pass_index[spk_in_field],
            theta_phase_deg=spk_phase_deg[spk_in_field],
            pass_id=infield['pass_id'].to_numpy()[epoch_idx] if len(spk) else np.array([], dtype=int),
        )).to_csv(out_dir / f"{cell['unit_label']}_InFieldSpikes.csv", index=False)
        infield.drop(columns='in_field').to_csv(out_dir / f"{cell['unit_label']}_InFieldEpochs.csv",
                                                index=False)
        print(f'  In-field spikes / epochs saved to {out_dir}')

    def load_relative_cell(self, step):
        self._load_cell(self.cell_idx + step)

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

    def _draw_infield_epochs(self, t0, t1):
        epochs = self.cur['infield_epochs']
        if epochs is None:
            return
        vis = epochs[(epochs['t_end'] > t0) & (epochs['t_start'] < t1)]
        for ax in (self.ax_lfp, self.ax_phase):
            for start, end in zip(vis['t_start'], vis['t_end']):
                ax.axvspan(max(start, t0), min(end, t1), color=INFIELD_COLOR, alpha=INFIELD_ALPHA,
                           linewidth=0, zorder=0)

    def _draw_pass_index(self, t0, t1):
        """Place-field pass index over time as a PASS_INDEX_CMAP color strip
        (-1 entry, 0 center, +1 exit) with the pass-index trace on top. Only
        pass-index values while the animal is inside the place field (field
        index >= FIELD_PEAK_FRACTION) are shown, whatever INFIELD_EPOCH_MODE."""
        ax = self.ax_pi
        pass_index_at, field_index_at = self.cur['pass_index_at'], self.cur['field_index_at']
        if pass_index_at is None:
            ax.text(0.5, 0.5, 'no tracking file -- no pass index', transform=ax.transAxes,
                    ha='center', va='center', fontsize=8, color='0.4')
        else:
            tt = np.linspace(t0, t1, PASS_INDEX_PLOT_SAMPLES)
            pi = pass_index_at(tt)
            pi[~(field_index_at(tt) >= FIELD_PEAK_FRACTION)] = np.nan
            ax.imshow(pi[None, :], aspect='auto', cmap=PASS_INDEX_CMAP, vmin=-1, vmax=1,
                      extent=(t0, t1, -1, 1), origin='lower', interpolation='nearest', zorder=0)
            pi_line = pi.copy()
            pi_line[1:][np.abs(np.diff(pi)) > 1] = np.nan  # don't draw the +1 -> -1 wrap
            ax.plot(tt, pi_line, color='k', linewidth=0.8, zorder=2)
        ax.axhline(0, color='0.5', linewidth=0.5, linestyle=':', zorder=1)
        ax.set_ylim(-1, 1)
        ax.set_yticks([-1, 0, 1])
        ax.set_ylabel('Pass\nindex')

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
        self.ax_pi.cla()
        self._draw_infield_epochs(t0, t1)
        self._draw_pass_index(t0, t1)

        if len(ts_win) > 1:
            self.ax_lfp.plot(ts_win, lfp_sig[i0:i1], color='0.7', linewidth=0.7, label='cleaned LFP')
            self.ax_lfp.plot(ts_win, filtered_lfp[i0:i1], color='tab:blue', linewidth=1.2,
                              label=f'theta {LFP_FILTER_BAND[0]:g}-{LFP_FILTER_BAND[1]:g} Hz')
            handles, _ = self.ax_lfp.get_legend_handles_labels()
            if self.cur['infield_epochs'] is not None:
                handles.append(Patch(color=INFIELD_COLOR, alpha=INFIELD_ALPHA,
                                     label='in place field (pass index)'))
            self.ax_lfp.legend(handles=handles, loc='upper right', fontsize=8)

            phase_deg_win = np.degrees(np.mod(lfp_phase_unwrapped[i0:i1], 2 * np.pi))
            plot_phase = phase_deg_win.copy()
            jump = np.where(np.abs(np.diff(plot_phase)) > 180)[0]
            plot_phase[jump] = np.nan
            self.ax_phase.plot(ts_win, plot_phase, color='tab:blue', linewidth=1.0)

        for ref in (0, 90, 180, 270, 360):
            self.ax_phase.axhline(ref, color='0.85', linewidth=0.5, zorder=0)

        spk_mask = (spk_ts >= t0) & (spk_ts <= t1)
        spk_win, spk_phase_win = spk_ts[spk_mask], spk_phase_deg[spk_mask]
        in_win = self.cur['spk_in_field'][spk_mask]
        if len(spk_win):
            ylo, yhi = self.ax_lfp.get_ylim() if len(ts_win) > 1 else (-1, 1)
            colors = np.where(in_win, 'crimson', '0.5')
            self.ax_lfp.vlines(spk_win, ylo, yhi, colors=colors, linewidth=1.0, alpha=0.7, zorder=5)
            self.ax_lfp.set_ylim(ylo, yhi)
            self.ax_phase.scatter(spk_win, spk_phase_win, s=28, c=colors,
                                   edgecolors='k', linewidths=0.3, zorder=5)

        self.ax_lfp.set_xlim(t0, t1)
        self.ax_phase.set_ylim(0, 360)
        self.ax_phase.set_yticks([0, 90, 180, 270, 360])
        self.ax_lfp.set_ylabel('LFP (uV)')
        self.ax_phase.set_ylabel('Theta phase (deg)')
        self.ax_phase.tick_params(labelbottom=False)
        self.ax_lfp.tick_params(labelbottom=False)
        self.ax_pi.set_xlabel('Time (s)')
        self.ax_pi.set_xlim(t0, t1)

        cell = self.cur['cell']
        flag = ''
        if cell['is_precessing'] is not None:
            tag = 'PRECESSING' if cell['is_precessing'] else ('RECESSING' if cell['is_recessing'] else '')
            flag = (f" | {tag}  rho={cell['rho']:.2f} p={cell['p']:.3g} "
                    f"slope={cell['slope']:.1f} deg/pass  TMI={cell['tmi']:.2f}")
        self.ax_lfp.set_title(
            f"[{self.cell_idx + 1}/{len(self.cells)}] {cell['unit_label']}{flag}\n"
            f"t = {t0:.2f}-{t1:.2f} s ({self.window_sec:.2f} s window) | "
            f"{int(in_win.sum())} in-field / {len(spk_win)} spike(s) shown", fontsize=10)

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
