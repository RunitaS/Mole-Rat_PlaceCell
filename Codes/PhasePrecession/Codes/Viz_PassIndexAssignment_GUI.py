# -*- coding: utf-8 -*-
"""
Interactive viewer for the pass-index assignment of
ThetaMod_PhasePrec_v22_fitR_shuffle.py (Climer et al. 2013 Pass Index).

1. Reads the pipeline's summary Excel (sheet 'ThetaPhase') and lists the
   precession-tested cells with their phase-precession quantifications
   (class, rho, p, slope, fit R and its shuffle p, r^2, phase range, phase at
   pass index 0). The list can be filtered by PrecessionClass (default:
   phase_precessing + phase_succeeding) and sorted by any column (click its
   header), so a cell can be picked by the quality of its phase relationship.
2. For the chosen cell, the pass index is recomputed exactly as in
   compute_pass_index (same functions, imported from the pipeline script;
   same rate map from all spikes in the tracking/LFP overlap window, same
   field-index map, arc-length resampling, spatial bandpass and Hilbert
   phase), and assigned to every tracked position sample and spike.
3. Playback: the trajectory develops in light gray with the current position
   as a red dot; the place field (rate map >= FIELD_THRESH_FRAC * peak, the
   same 20 % threshold auto_filter_band uses to estimate the field size) is
   outlined in green. Every time the animal enters the field, the pass index
   at the entry sample is shown in bold at the bottom. Play at a chosen speed
   (x real time) or scrub with the seek bar; buttons jump to the previous/next
   field entry.

Right-hand panels: pass index vs time (with the raw field index, the
bandpass-filtered field index and in-field periods shaded), and the
distribution of pass index at all field entries/exits -- if the assignment
works, entries should sit at negative pass index and exits at positive.

Spikes are drawn coloured by their pass index (hsv, as in the pipeline's
PassIndex figure). Note: all spikes in the overlap window are shown; the
pipeline's phase analysis uses only the ACG theta-positive subset of them
(the rate map / pass index itself is built from all spikes, as here).

Usage:  python Viz_PassIndexAssignment_GUI.py [path\\to\\theta_phase_Hilbert.xlsx]
Without an argument the pipeline's RESULTS_FOLDER / OUTPUT_EXCEL_NAME is used.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ============================================================================
# Configuration
# ============================================================================
PIPELINE_PATH = Path(__file__).with_name('ThetaMod_PhasePrec_v22_fitR_shuffle.py')
SHEET_NAME = 'ThetaPhase'
DEFAULT_CLASSES = ('phase_precessing', 'phase_succeeding')
ALL_CLASSES = ('phase_precessing', 'phase_succeeding', 'phase_locked', 'no_phase_relation',
               'non_precessing')

FIELD_THRESH_FRAC = 0.2     # field outline / in-field test: rate map >= this * peak rate
ENTRY_DEBOUNCE_SEC = 0.25   # out-of-field gaps shorter than this are not counted as leaving the field
TIME_WINDOW_SEC = 20.0      # half-width of the pass-index-vs-time panel window
FRAME_MS = 40               # playback refresh interval
DEFAULT_SPEED = '5'         # x real time
SPEED_CHOICES = ('0.25', '0.5', '1', '2', '5', '10', '20', '50', '100')

# Treeview columns: (header, Excel column or derived key, width, format)
LIST_COLUMNS = [
    ('Class', 'PrecessionClass', 130, None),
    ('Animal', 'Animal', 65, None),
    ('Arena', 'Arena', 55, None),
    ('SesType', 'SessionType', 60, None),
    ('Session', 'SessionFolder', 150, None),
    ('Unit', 'Unit', 100, None),
    ('n spk', 'PassIndex_n_spikes', 55, '{:.0f}'),
    ('rho', 'rho', 60, '{:+.3f}'),
    ('p', 'precession_p', 65, '{:.2g}'),
    ('slope (deg/pass)', 'slope_deg_per_pass', 100, '{:+.1f}'),
    ('fit R', 'fit_R', 55, '{:.3f}'),
    ('fitR shuf p', 'fit_R_shuffle_p', 75, '{:.3g}'),
    ('r²', 'r_squared', 55, '{:.3f}'),
    ('range (deg)', 'phase_range_deg', 75, '{:.0f}'),
    ('phase@PI0', 'phase_at_pi0_deg', 70, '{:.0f}'),
    ('TMI', 'TMI', 50, '{:.2f}'),
]


# ============================================================================
# Pipeline import and per-cell computation
# ============================================================================

def load_pipeline(path: Path):
    """Import the pipeline script as a module (its main() is not run). It sets
    matplotlib's pyplot backend to Agg, which does not affect the Tk canvas
    used here (FigureCanvasTkAgg is used directly, without pyplot)."""
    spec = importlib.util.spec_from_file_location('phaseprec_pipeline', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ncs_time_range(path: Path, mod):
    """(first, last) sample time (s) of a Neuralynx .ncs file, read from its
    first and last records only (no full load)."""
    dtype = np.dtype([('TimeStamp', '<u8'), ('ChannelNumber', '<u4'), ('SampleFreq', '<u4'),
                      ('NumValidSamples', '<u4'), ('Samples', '<i2', (mod.NCS_SAMPLES_PER_RECORD,))])
    n_rec = (path.stat().st_size - mod.HEADER_BYTES) // dtype.itemsize
    if n_rec <= 0:
        return None
    with open(path, 'rb') as fh:
        fh.seek(mod.HEADER_BYTES)
        first = np.frombuffer(fh.read(dtype.itemsize), dtype)[0]
        fh.seek(mod.HEADER_BYTES + (n_rec - 1) * dtype.itemsize)
        last = np.frombuffer(fh.read(dtype.itemsize), dtype)[0]
    fs = float(first['SampleFreq']) or 32000.0
    return first['TimeStamp'] / 1e6, last['TimeStamp'] / 1e6 + (mod.NCS_SAMPLES_PER_RECORD - 1) / fs


def _wrap_pm_pi(u):
    return np.mod(u + np.pi, 2 * np.pi) - np.pi


def _fill_short_gaps(mask, ts, max_gap):
    """Set interior False runs of `mask` lasting < max_gap s to True (tracking
    jitter at the field border should not count as leaving and re-entering)."""
    m = mask.copy()
    d = np.diff(np.concatenate(([0], (~m).astype(np.int8), [0])))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if s > 0 and e < len(m) and ts[e] - ts[s - 1] < max_gap:
            m[s:e] = True
    return m


def compute_cell(mod, row: pd.Series) -> dict:
    """Recompute the pass index of one Excel row's cell, as compute_pass_index does."""
    folder = Path(row['FolderPath'])
    tracking_path = mod._find_tracking_file(folder)
    pos_ts, pos_xy = mod.load_tracking(tracking_path, mod.TRACKING_TIME_UNIT)

    units = mod.load_ntt_spike_times(folder / row['ntt_file'])
    cell_number = int(row['cell_number'])
    if cell_number not in units:
        raise KeyError(f'cell {cell_number} not found in {row["ntt_file"]} (has {list(units)})')
    spk_all = units[cell_number]

    # Same spike window as process_session: overlap of tracking and LFP
    t_start, t_stop = pos_ts.min(), pos_ts.max()
    lfp_path = folder / str(row.get('lfp_file', ''))
    if lfp_path.is_file():
        rng_lfp = ncs_time_range(lfp_path, mod)
        if rng_lfp is not None:
            t_start, t_stop = max(t_start, rng_lfp[0]), min(t_stop, rng_lfp[1])
    spk_ts = spk_all[(spk_all >= t_start) & (spk_all <= t_stop)]

    n_dims = pos_xy.shape[1]
    binside = 2.0 * n_dims if mod.BINSIDE == 'auto' else mod.BINSIDE
    smth_width = 3.0 * binside if mod.SMTH_WIDTH == 'auto' else mod.SMTH_WIDTH

    # ---- identical to compute_pass_index ----
    spk_xy, _ = mod.spk_pos(pos_ts, pos_xy, spk_ts)
    rmap, occupancy, x_edges, y_edges = mod.rate_map(pos_ts, pos_xy, spk_xy, binside, smth_width)
    fi_map = mod.field_index_map(rmap, occupancy, mod.METHOD)
    field_index = mod.field_index_per_position(pos_xy, fi_map, x_edges, y_edges)
    cc, ts2, resampled = mod.sample_along_arc(pos_ts, pos_xy, field_index)
    if mod.FILTER_BAND == 'auto':
        filter_band = mod.auto_filter_band(mod.METHOD, rmap, occupancy, binside, n_dims)
    else:
        filter_band = mod.FILTER_BAND
    fs_arc = 1.0 / np.mean(np.diff(cc))
    filtered = mod.bandpass_filter(resampled, filter_band[0], filter_band[1], fs_arc)
    field_phase_unwrapped = np.unwrap(mod.hilbert_phase(filtered, fs_arc)[0])
    spk_pi = _wrap_pm_pi(mod.phase_at(ts2, field_phase_unwrapped, spk_ts)) / np.pi
    # ---- the same assignment at every tracked position sample ----
    pos_pi = _wrap_pm_pi(mod.phase_at(ts2, field_phase_unwrapped, pos_ts)) / np.pi
    pos_filt = np.interp(pos_ts, ts2, filtered, left=np.nan, right=np.nan)
    max_abs = np.nanmax(np.abs(filtered))
    pos_filt_norm = pos_filt / max_abs if max_abs > 0 else pos_filt

    # In-field test per position sample (same binning as field_index_per_position)
    peak = np.nanmax(rmap)
    field_mask = (rmap >= FIELD_THRESH_FRAC * peak) & (occupancy > 0)
    xi = np.clip(np.digitize(pos_xy[:, 0], x_edges) - 1, 0, len(x_edges) - 2)
    yi = np.clip(np.digitize(pos_xy[:, 1], y_edges) - 1, 0, len(y_edges) - 2)
    in_field = _fill_short_gaps(field_mask[xi, yi], pos_ts, ENTRY_DEBOUNCE_SEC)
    entries = np.flatnonzero(~in_field[:-1] & in_field[1:]) + 1
    exits = np.flatnonzero(in_field[:-1] & ~in_field[1:]) + 1

    t0 = pos_ts[0]
    return dict(
        row=row, tracking_path=tracking_path,
        pos_t=pos_ts - t0, pos_xy=pos_xy, pos_pi=pos_pi, pos_fi=field_index,
        pos_filt_norm=pos_filt_norm, in_field=in_field,
        spk_t=spk_ts - t0, spk_xy=spk_xy, spk_pi=spk_pi,
        rmap=rmap, occupancy=occupancy, x_edges=x_edges, y_edges=y_edges,
        field_thresh=FIELD_THRESH_FRAC * peak, peak_rate=peak,
        entries=entries, exits=exits, filter_band=filter_band,
        n_spikes_window=len(spk_ts), n_spikes_total=len(spk_all),
    )


# ============================================================================
# GUI
# ============================================================================

def _fmt(val, fmt):
    if val is None or (isinstance(val, float) and not np.isfinite(val)) or pd.isna(val):
        return ''
    if fmt is None:
        return str(val)
    try:
        return fmt.format(float(val))
    except (TypeError, ValueError):
        return str(val)


class PassIndexViewer:
    def __init__(self, root: tk.Tk, mod, excel_path: Path | None):
        self.root, self.mod = root, mod
        self.df = pd.DataFrame()
        self.cell = None
        self.cur_t = 0.0
        self.playing = False
        self._after_id = None
        self._seek_guard = False
        self._sort_state = {}

        root.title('Pass index assignment viewer')
        root.geometry('1500x980')
        self._build_ui()
        root.protocol('WM_DELETE_WINDOW', self._on_close)
        if excel_path is not None and excel_path.is_file():
            self.load_excel(excel_path)

    # ---------------------------------------------------------------- layout
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=4)
        top.pack(fill='x')
        ttk.Button(top, text='Open Excel...', command=self._browse_excel).pack(side='left')
        self.excel_label = ttk.Label(top, text='(no Excel loaded)', foreground='gray30')
        self.excel_label.pack(side='left', padx=6)

        filt = ttk.Frame(self.root, padding=(4, 0))
        filt.pack(fill='x')
        ttk.Label(filt, text='Show classes:').pack(side='left')
        self.class_vars = {}
        for cls in ALL_CLASSES:
            var = tk.BooleanVar(value=cls in DEFAULT_CLASSES)
            ttk.Checkbutton(filt, text=cls, variable=var, command=self._refresh_list).pack(side='left', padx=3)
            self.class_vars[cls] = var
        self.fitr_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(filt, text='fit R shuffle pass only', variable=self.fitr_only,
                        command=self._refresh_list).pack(side='left', padx=12)
        self.count_label = ttk.Label(filt, text='')
        self.count_label.pack(side='left', padx=12)

        paned = ttk.PanedWindow(self.root, orient='vertical')
        paned.pack(fill='both', expand=True)

        # ---- cell list ----
        list_frame = ttk.Frame(paned)
        cols = [c[0] for c in LIST_COLUMNS]
        self.tree = ttk.Treeview(list_frame, columns=cols, show='headings', height=8, selectmode='browse')
        for header, _key, width, _fmt_ in LIST_COLUMNS:
            self.tree.heading(header, text=header, command=lambda h=header: self._sort_by(h))
            self.tree.column(header, width=width, anchor='center', stretch=False)
        vsb = ttk.Scrollbar(list_frame, orient='vertical', command=self.tree.yview)
        hsb = ttk.Scrollbar(list_frame, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        btns = ttk.Frame(list_frame)
        btns.grid(row=2, column=0, sticky='w', pady=2)
        ttk.Button(btns, text='Load selected cell', command=self._load_selected).pack(side='left')
        ttk.Button(btns, text='Open saved PassIndex plot', command=self._open_saved_plot).pack(side='left', padx=6)
        self.status = ttk.Label(btns, text='Double-click a cell to load it.', foreground='gray30')
        self.status.pack(side='left', padx=10)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        self.tree.bind('<Double-1>', lambda e: self._load_selected())
        paned.add(list_frame, weight=1)

        # ---- viewer ----
        view = ttk.Frame(paned)
        self.fig = Figure(figsize=(14, 7), dpi=90)
        self.canvas = FigureCanvasTkAgg(self.fig, master=view)
        toolbar = NavigationToolbar2Tk(self.canvas, view, pack_toolbar=False)
        toolbar.update()

        ctrl = ttk.Frame(view, padding=4)
        ttk.Button(ctrl, text='⏮ prev entry', command=lambda: self._jump_entry(-1)).pack(side='left')
        ttk.Button(ctrl, text='−1 s', width=5, command=lambda: self._step(-1.0)).pack(side='left', padx=2)
        self.play_btn = ttk.Button(ctrl, text='▶ Play', width=9, command=self.toggle_play)
        self.play_btn.pack(side='left', padx=2)
        ttk.Button(ctrl, text='+1 s', width=5, command=lambda: self._step(1.0)).pack(side='left', padx=2)
        ttk.Button(ctrl, text='next entry ⏭', command=lambda: self._jump_entry(1)).pack(side='left')
        ttk.Button(ctrl, text='⟲ restart', command=lambda: self.set_time(0.0)).pack(side='left', padx=6)

        ttk.Label(ctrl, text='Speed (× real time):').pack(side='left', padx=(14, 2))
        self.speed_var = tk.StringVar(value=DEFAULT_SPEED)
        ttk.Combobox(ctrl, textvariable=self.speed_var, values=SPEED_CHOICES, width=6).pack(side='left')
        ttk.Label(ctrl, text='Trail (s, 0 = full):').pack(side='left', padx=(14, 2))
        self.trail_var = tk.StringVar(value='0')
        ttk.Spinbox(ctrl, textvariable=self.trail_var, from_=0, to=3600, increment=5, width=6,
                    command=self._redraw_current).pack(side='left')
        self.show_rmap = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text='rate map', variable=self.show_rmap,
                        command=self._toggle_rmap).pack(side='left', padx=12)
        self.show_spikes = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text='spikes', variable=self.show_spikes,
                        command=self._redraw_current).pack(side='left')
        self.time_label = ttk.Label(ctrl, text='t = 0.00 s', width=24)
        self.time_label.pack(side='right')

        seek_row = ttk.Frame(view, padding=(4, 0))
        ttk.Label(seek_row, text='Seek:').pack(side='left')
        self.seek = ttk.Scale(seek_row, from_=0.0, to=1.0, orient='horizontal', command=self._on_seek)
        self.seek.pack(side='left', fill='x', expand=True, padx=4)

        toolbar.pack(side='bottom', fill='x')
        seek_row.pack(side='bottom', fill='x')
        ctrl.pack(side='bottom', fill='x')
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        paned.add(view, weight=4)

    # ---------------------------------------------------------------- list
    def _browse_excel(self):
        path = filedialog.askopenfilename(title='Phase-precession summary Excel',
                                          filetypes=[('Excel', '*.xlsx'), ('All files', '*.*')])
        if path:
            self.load_excel(Path(path))

    def load_excel(self, path: Path):
        try:
            df = pd.read_excel(path, sheet_name=SHEET_NAME)
        except Exception as exc:
            messagebox.showerror('Excel', f'Could not read {path}:\n{exc}')
            return
        df = df.loc[df['PrecessionTested'] == True].copy()  # noqa: E712
        df['Animal'] = [self.mod.detect_animal_id(Path(p)) for p in df['FolderPath']]
        df['SessionFolder'] = [Path(p).name for p in df['FolderPath']]
        for col in ('Arena', 'SessionType'):
            if col not in df:
                df[col] = ''
        self.df = df
        self.excel_label.config(text=str(path))
        self._refresh_list()

    def _refresh_list(self):
        self.tree.delete(*self.tree.get_children())
        if self.df.empty:
            return
        classes = [c for c, v in self.class_vars.items() if v.get()]
        sub = self.df[self.df['PrecessionClass'].isin(classes)]
        if self.fitr_only.get() and 'fit_R_Significant' in sub:
            sub = sub[sub['fit_R_Significant'] == True]  # noqa: E712
        sub = sub.sort_values(['PrecessionClass', 'rho'], key=lambda s: s.abs() if s.name == 'rho' else s,
                              ascending=[True, False])
        for idx, r in sub.iterrows():
            vals = [_fmt(r.get(key), fmt) for _h, key, _w, fmt in LIST_COLUMNS]
            self.tree.insert('', 'end', iid=str(idx), values=vals)
        counts = self.df['PrecessionClass'].value_counts()
        self.count_label.config(text=f'{len(sub)} shown | ' + ', '.join(
            f'{c}: {counts.get(c, 0)}' for c in ALL_CLASSES if counts.get(c, 0)))

    def _sort_by(self, header):
        items = [(self.tree.set(iid, header), iid) for iid in self.tree.get_children('')]

        def key(v):
            try:
                return (0, float(v[0]))
            except ValueError:
                return (1, v[0])
        desc = not self._sort_state.get(header, False)
        self._sort_state[header] = desc
        for i, (_v, iid) in enumerate(sorted(items, key=key, reverse=desc)):
            self.tree.move(iid, '', i)

    def _selected_row(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo('Select a cell', 'Select a cell in the list first.')
            return None
        return self.df.loc[int(sel[0])]

    def _open_saved_plot(self):
        row = self._selected_row()
        if row is None:
            return
        folder = Path(row['FolderPath'])
        png = folder / 'ThetaMod_PhasePrecession_LFPshuffle' / f'{row["Animal"]}_{row["Unit"]}_PassIndex.png'
        if png.is_file():
            os.startfile(png)
        else:
            messagebox.showinfo('Not found', f'No saved plot at\n{png}')

    def _load_selected(self):
        row = self._selected_row()
        if row is None:
            return
        self.pause()
        self.status.config(text=f'Loading {row["Animal"]} | {row["SessionFolder"]} | {row["Unit"]} ...')
        self.root.config(cursor='watch')
        self.root.update_idletasks()
        try:
            self.cell = compute_cell(self.mod, row)
        except Exception as exc:
            messagebox.showerror('Load failed', f'{type(exc).__name__}: {exc}')
            self.status.config(text='Load failed.')
            return
        finally:
            self.root.config(cursor='')
        c = self.cell
        self.status.config(text=f'Loaded: {c["tracking_path"].name} | {c["n_spikes_window"]} spikes in window '
                                f'({c["n_spikes_total"]} total) | {len(c["entries"])} field entries')
        self._draw_static()
        self._seek_guard = True
        self.seek.config(to=float(c['pos_t'][-1]))
        self._seek_guard = False
        self.set_time(0.0)

    # ---------------------------------------------------------------- figure
    def _draw_static(self):
        c, row, fig = self.cell, self.cell['row'], self.fig
        fig.clear()
        gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], left=0.05, right=0.97, bottom=0.2,
                              top=0.86, wspace=0.22, hspace=0.45)
        ax = self.ax_map = fig.add_subplot(gs[:, 0])
        self.ax_time = fig.add_subplot(gs[0, 1])
        self.ax_hist = fig.add_subplot(gs[1, 1])

        xe, ye = c['x_edges'], c['y_edges']
        rmap_shown = np.ma.masked_where(c['occupancy'] == 0, c['rmap'])   # unvisited bins blank
        self.rmap_im = ax.imshow(rmap_shown.T, origin='lower', cmap='jet', alpha=0.3, zorder=0,
                                 extent=[xe[0], xe[-1], ye[0], ye[-1]], visible=self.show_rmap.get())
        xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
        ax.contour(xc, yc, c['rmap'].T, levels=[c['field_thresh']], colors='limegreen',
                   linewidths=2.5, zorder=3)
        (self.trail_line,) = ax.plot([], [], color='0.72', lw=0.8, zorder=1)
        self.spk_sc = ax.scatter(np.empty(0), np.empty(0), c=np.empty(0), cmap='hsv', vmin=-1, vmax=1,
                                 s=16, edgecolors='k', linewidths=0.2, zorder=4)
        fig.colorbar(self.spk_sc, ax=ax, fraction=0.04, pad=0.02, label='Spike pass index')
        (self.entry_mark,) = ax.plot([], [], marker='*', ms=16, color='limegreen', mec='k', ls='', zorder=5)
        self.entry_annot = ax.annotate('', xy=(0, 0), xytext=(8, 8), textcoords='offset points',
                                       fontsize=10, fontweight='bold', color='darkgreen', zorder=6,
                                       bbox=dict(boxstyle='round', fc='white', alpha=0.8, ec='darkgreen'))
        (self.dot,) = ax.plot([], [], 'o', color='red', ms=10, mec='k', zorder=7)
        pad = 2.0
        ax.set_xlim(c['pos_xy'][:, 0].min() - pad, c['pos_xy'][:, 0].max() + pad)
        ax.set_ylim(c['pos_xy'][:, 1].min() - pad, c['pos_xy'][:, 1].max() + pad)
        ax.set_aspect('equal')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_title(f'Trajectory (gray), position (red), field outline (green: rate ≥ '
                     f'{FIELD_THRESH_FRAC:.0%} of peak {c["peak_rate"]:.1f} Hz)', fontsize=9)

        # ---- pass index vs time ----
        at = self.ax_time
        t = c['pos_t']
        pi_plot = c['pos_pi'].copy()
        pi_plot[1:][np.abs(np.diff(pi_plot)) > 1.0] = np.nan   # break the line at the +-1 wrap
        at.fill_between(t, 0, 1, where=c['in_field'], transform=at.get_xaxis_transform(),
                        color='limegreen', alpha=0.15, lw=0, label='in field')
        at.plot(t, c['pos_fi'], color='green', lw=0.8, alpha=0.8, label='field index (raw)')
        at.plot(t, c['pos_filt_norm'], color='tab:blue', lw=0.8, ls='--', label='field index (filtered, norm.)')
        at.plot(t, pi_plot, color='k', lw=1.3, label='pass index')
        at.plot(t[c['entries']], c['pos_pi'][c['entries']], '^', color='limegreen', mec='k', ms=8,
                label='entry')
        at.plot(t[c['exits']], c['pos_pi'][c['exits']], 'v', color='orange', mec='k', ms=8, label='exit')
        self.time_cursor = at.axvline(0, color='red', lw=1.5)
        at.axhline(0, color='0.6', lw=0.5)
        at.set_ylim(-1.15, 1.7)   # headroom for the legend
        at.set_yticks([-1, -0.5, 0, 0.5, 1])
        at.set_xlabel('Time (s)')
        at.set_ylabel('Pass index / field index')
        at.set_title('Pass index vs time (red line = now)', fontsize=9)
        at.legend(fontsize=7, ncol=3, loc='upper center', framealpha=0.85)

        # ---- pass index at entries / exits ----
        ah = self.ax_hist
        bins = np.linspace(-1, 1, 21)
        pi_entry = c['pos_pi'][c['entries']]
        pi_exit = c['pos_pi'][c['exits']]
        ah.hist(pi_entry[np.isfinite(pi_entry)], bins=bins, color='limegreen', alpha=0.6, ec='k',
                label=f'entries (n={len(pi_entry)}, median {np.nanmedian(pi_entry) if len(pi_entry) else np.nan:+.2f})')
        ah.hist(pi_exit[np.isfinite(pi_exit)], bins=bins, color='orange', alpha=0.5, ec='k',
                label=f'exits (n={len(pi_exit)}, median {np.nanmedian(pi_exit) if len(pi_exit) else np.nan:+.2f})')
        self.hist_cursor = ah.axvline(np.nan, color='red', ls='--', lw=2)
        ah.set_xlim(-1, 1)
        ah.set_xlabel('Pass index')
        ah.set_ylabel('Count')
        ah.set_title('Pass index at field entries / exits (expected: entries < 0, exits > 0)', fontsize=9)
        ah.legend(fontsize=8)

        lo, hi = c['filter_band']
        fig.suptitle(
            f'{row["Animal"]} | {row.get("Arena", "")} {row.get("SessionType", "")} | {row["SessionFolder"]} | '
            f'{row["Unit"]}  —  {row["PrecessionClass"]}\n'
            f'rho={_fmt(row.get("rho"), "{:+.3f}")}  p={_fmt(row.get("precession_p"), "{:.2g}")}  '
            f'slope={_fmt(row.get("slope_deg_per_pass"), "{:+.1f}")} deg/pass  '
            f'fit R={_fmt(row.get("fit_R"), "{:.3f}")} (shuffle p={_fmt(row.get("fit_R_shuffle_p"), "{:.3g}")})  '
            f'r²={_fmt(row.get("r_squared"), "{:.3f}")}  |  spatial filter {lo:.4f}–{hi:.4f} cyc/cm '
            f'(periods {1 / hi:.1f}–{1 / lo:.1f} cm)', fontsize=10)
        self.live_text = fig.text(0.5, 0.075, '', ha='center', fontsize=11, family='monospace')
        self.entry_text = fig.text(0.5, 0.02, '', ha='center', fontsize=16, fontweight='bold')

    def _toggle_rmap(self):
        if self.cell is not None:
            self.rmap_im.set_visible(self.show_rmap.get())
            self.canvas.draw_idle()

    def _trail_sec(self):
        try:
            return max(0.0, float(self.trail_var.get()))
        except ValueError:
            return 0.0

    def _render(self):
        c, t = self.cell, self.cur_t
        pos_t, xy = c['pos_t'], c['pos_xy']
        i = int(np.clip(np.searchsorted(pos_t, t, 'right') - 1, 0, len(pos_t) - 1))
        trail = self._trail_sec()
        j = int(np.searchsorted(pos_t, t - trail)) if trail > 0 else 0
        self.trail_line.set_data(xy[j:i + 1, 0], xy[j:i + 1, 1])
        self.dot.set_data([xy[i, 0]], [xy[i, 1]])

        if self.show_spikes.get():
            k = int(np.searchsorted(c['spk_t'], t, 'right'))
            kj = int(np.searchsorted(c['spk_t'], t - trail)) if trail > 0 else 0
            self.spk_sc.set_offsets(c['spk_xy'][kj:k])
            self.spk_sc.set_array(c['spk_pi'][kj:k])
        else:
            self.spk_sc.set_offsets(np.empty((0, 2)))
            self.spk_sc.set_array(np.empty(0))

        self.ax_time.set_xlim(t - TIME_WINDOW_SEC, t + TIME_WINDOW_SEC)
        self.time_cursor.set_xdata([t, t])

        in_f = bool(c['in_field'][i])
        self.live_text.set_text(
            f't = {t:8.2f} s | current pass index = {c["pos_pi"][i]:+.3f} | '
            f'field index = {c["pos_fi"][i]:.2f} | {"IN FIELD" if in_f else "out of field"}')

        entry_t = pos_t[c['entries']]
        e = int(np.searchsorted(entry_t, t, 'right')) - 1
        if e >= 0:
            ei = c['entries'][e]
            pi_e = c['pos_pi'][ei]
            self.entry_text.set_text(f'FIELD ENTRY #{e + 1}/{len(entry_t)} at t = {entry_t[e]:.2f} s   →   '
                                     f'PASS INDEX AT ENTRY = {pi_e:+.3f}')
            self.entry_text.set_color('darkgreen' if in_f else 'black')
            self.entry_mark.set_data([xy[ei, 0]], [xy[ei, 1]])
            self.entry_annot.xy = (xy[ei, 0], xy[ei, 1])
            self.entry_annot.set_text(f'{pi_e:+.2f}')
            self.entry_annot.set_visible(True)
            self.hist_cursor.set_xdata([pi_e, pi_e])
        else:
            self.entry_text.set_text('No field entry yet')
            self.entry_text.set_color('black')
            self.entry_mark.set_data([], [])
            self.entry_annot.set_visible(False)
            self.hist_cursor.set_xdata([np.nan, np.nan])

        self.time_label.config(text=f't = {t:.2f} / {pos_t[-1]:.1f} s')
        self.canvas.draw_idle()

    # ---------------------------------------------------------------- playback
    def set_time(self, t):
        if self.cell is None:
            return
        self.cur_t = float(np.clip(t, 0.0, self.cell['pos_t'][-1]))
        self._seek_guard = True
        self.seek.set(self.cur_t)
        self._seek_guard = False
        self._render()

    def _redraw_current(self):
        if self.cell is not None:
            self._render()

    def _on_seek(self, value):
        if self._seek_guard or self.cell is None:
            return
        self.cur_t = float(value)
        self._render()

    def _step(self, dt):
        self.set_time(self.cur_t + dt)

    def _jump_entry(self, direction):
        if self.cell is None or not len(self.cell['entries']):
            return
        self.pause()
        entry_t = self.cell['pos_t'][self.cell['entries']]
        if direction > 0:
            k = np.searchsorted(entry_t, self.cur_t + 1e-6, 'right')
            if k < len(entry_t):
                self.set_time(entry_t[k])
        else:
            k = np.searchsorted(entry_t, self.cur_t - 1e-6, 'left') - 1
            if k >= 0:
                self.set_time(entry_t[k])

    def toggle_play(self):
        if self.playing:
            self.pause()
        elif self.cell is not None:
            if self.cur_t >= self.cell['pos_t'][-1]:
                self.cur_t = 0.0
            self.playing = True
            self.play_btn.config(text='⏸ Pause')
            self._last_wall = time.perf_counter()
            self._tick()

    def pause(self):
        self.playing = False
        self.play_btn.config(text='▶ Play')
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _tick(self):
        if not self.playing:
            return
        now = time.perf_counter()
        try:
            speed = max(0.0, float(self.speed_var.get()))
        except ValueError:
            speed = 1.0
        # advance by the real elapsed wall time, so the speed holds even if drawing is slow
        t = self.cur_t + (now - self._last_wall) * speed
        self._last_wall = now
        end = self.cell['pos_t'][-1]
        self.set_time(t)
        if t >= end:
            self.pause()
            return
        self._after_id = self.root.after(FRAME_MS, self._tick)

    def _on_close(self):
        self.pause()
        self.root.destroy()


def main():
    mod = load_pipeline(PIPELINE_PATH)
    excel_path = Path(sys.argv[1]) if len(sys.argv) > 1 else mod.RESULTS_FOLDER / mod.OUTPUT_EXCEL_NAME
    root = tk.Tk()
    PassIndexViewer(root, mod, excel_path)
    root.mainloop()


if __name__ == '__main__':
    main()
