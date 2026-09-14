# -*- coding: utf-8 -*-
"""
Interactive manual-review tool for Generalized Phase (GP) phase-reversal
corrections (Davis, Muller et al. 2020, Nature 587:432-436) on a single LFP
channel.

Purpose: the plain Hilbert-transform analytic-signal phase
(np.angle(signal.hilbert(x))) assigns phase via a four-quadrant arctangent
every sample, with no safeguard against brief epochs where the filtered
signal's instantaneous frequency collapses or goes negative (e.g. near a
local double-peak/notch riding on the otherwise-clean theta oscillation).
Those epochs show up as backward ("reversed") jumps in phase. GP detects
every such epoch (instantaneous frequency below the bandpass filter's own
low edge), blanks it (extended by a safety window, `NWIN`), and reconstructs
it by shape-preserving (pchip) interpolation of the unwrapped phase trend
from the surrounding reliable samples.

This script walks through EVERY phase-reversal/phase-slip epoch GP found in
one LFP file and opens an interactive matplotlib window (arrow keys or
Prev/Next buttons, or type an epoch number and hit Enter) so each correction
can be checked by eye: the raw filtered trace, raw vs. GP-corrected wrapped
phase, raw vs. GP-corrected unwrapped phase (aligned to the same start so
the raw trace's backward dip and GP's smoothed replacement are directly
comparable), and raw vs. GP instantaneous frequency.

The GP algorithm here (generalized_phase_vector_debug) is copied verbatim
from Debug_GPA_PhaseValues_ThetaMod_PhasePrecession_withGenPhsAp_v8.py's
generalized_phase_vector (not imported -- that module calls
`matplotlib.use('Agg')` at import time, which would silently kill this
script's interactive window), with extra diagnostic arrays returned:
idx_raw (the pre-extension low-frequency detection, before the NWIN safety
margin is applied) and raw_unwrap (a naive full-trace np.unwrap of the raw
Hilbert phase, INCLUDING through the reversal epochs -- i.e. the actual
uncorrected artifact, kept here only for visualization; the main pipeline
never uses this because it is the thing being fixed).

Requires: numpy, scipy, matplotlib (with an interactive backend -- TkAgg is
tried first since it ships with most Windows Python installs).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import label
from scipy.interpolate import PchipInterpolator

import matplotlib
for _backend in ('TkAgg', 'Qt5Agg', 'QtAgg'):
    try:
        matplotlib.use(_backend)
        break
    except Exception:
        continue
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

# ============================================================================
# Configuration -- EDIT THESE
# ============================================================================

# Session folder to auto-pick an LFP file from (first .ncs, natural sort --
# same convention as the main pipeline's `theta_ncs = ncs_files[0]`). Must be
# a leaf session folder that directly contains the .ncs file(s) (not
# searched recursively). OR set LFP_FILE_OVERRIDE to a specific .ncs path and
# leave DATA_FOLDER unused.
DATA_FOLDER = Path(r"C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/Debug/Fa23BD/Open/Day5/1Cntrl")
LFP_FILE_OVERRIDE = None   # e.g. Path(r"C:/.../CSC1.ncs"); None = auto-pick from DATA_FOLDER

LFP_FILTER_BAND = (3.0, 7.0)   # Hz, theta band -- must match the main pipeline's LFP_FILTER_BAND
NWIN = 3                        # GP's safety-window multiplier -- must match the main pipeline

CONTEXT_SEC = 0.5                # seconds of context shown before/after each flagged epoch
MIN_CONTEXT_CYCLES = 2.0         # context is also widened to cover at least this many theta cycles

START_SEC = None   # None = from file start; set to restrict analysis to a sub-window
END_SEC = None      # None = to file end

NCS_SAMPLES_PER_RECORD = 512
HEADER_BYTES = 16 * 1024
DEFAULT_ADBITVOLTS = 0.000000195


# ============================================================================
# Neuralynx .ncs I/O (ported verbatim from the main pipeline)
# ============================================================================

def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', path.name)]


def _read_header_text(path: Path) -> str:
    with open(path, 'rb') as fh:
        raw = fh.read(HEADER_BYTES)
    return raw.decode('latin-1', errors='replace')


def _header_field(header_text: str, name: str, default=None, cast=float):
    for line in header_text.splitlines():
        line = line.strip()
        if line.startswith(f'-{name}'):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return cast(parts[1])
                except ValueError:
                    return default
    return default


def load_ncs(path: Path):
    """Load a Neuralynx .ncs file. Returns (samples_uV, timestamps_s, fs_hz)."""
    header_text = _read_header_text(path)
    adbitvolts = _header_field(header_text, 'ADBitVolts', default=DEFAULT_ADBITVOLTS)

    ncs_dtype = np.dtype([
        ('TimeStamp', '<u8'),
        ('ChannelNumber', '<u4'),
        ('SampleFreq', '<u4'),
        ('NumValidSamples', '<u4'),
        ('Samples', '<i2', (NCS_SAMPLES_PER_RECORD,)),
    ])
    records = np.fromfile(path, dtype=ncs_dtype, offset=HEADER_BYTES)
    if len(records) == 0:
        raise ValueError(f'No records found in {path}')

    fs = float(records['SampleFreq'][0])
    if fs <= 0:
        fs = _header_field(header_text, 'SamplingFrequency', default=32000.0)

    sample_period_us = 1e6 / fs
    offsets_us = np.arange(NCS_SAMPLES_PER_RECORD) * sample_period_us
    timestamps_us = (records['TimeStamp'][:, None].astype(np.float64) + offsets_us[None, :]).ravel()
    samples_uV = (records['Samples'].astype(np.float64) * adbitvolts * 1e6).ravel()

    return samples_uV, timestamps_us / 1e6, fs


def bandpass_filter(data, low, high, fs, order=4):
    """Zero-phase Butterworth bandpass (SOS form), ported verbatim from the
    main pipeline."""
    nyq = fs / 2.0
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 1 - 1e-6)
    if low_n >= high_n:
        raise ValueError(
            f'invalid filter band after clamping to Nyquist: '
            f'requested ({low:.4g}, {high:.4g}), fs={fs:.4g} -> '
            f'normalized ({low_n:.4g}, {high_n:.4g})'
        )
    sos = signal.butter(order, [low_n, high_n], btype='band', output='sos')
    filtered = signal.sosfiltfilt(sos, data)
    if np.any(np.isnan(filtered)):
        filtered = signal.sosfilt(sos, data)
    return filtered


# ============================================================================
# Generalized Phase -- debug variant of generalized_phase_vector (main
# pipeline), returning extra pre-correction diagnostics for visualization.
# Algorithm itself is unchanged; see the main pipeline's own docstring for
# the full rationale.
# ============================================================================

def _gp_rewrap(xp):
    return xp - 2 * np.pi * np.floor((xp - np.pi) / (2 * np.pi)) - 2 * np.pi


def generalized_phase_vector_debug(x, fs, lp, nwin=NWIN):
    """Same computation as the main pipeline's generalized_phase_vector, with
    extra returns for reviewing exactly what was detected/corrected:

    xgp        : GP-corrected analytic signal (identical to the main
                 pipeline's own output for the same x/fs/lp/nwin).
    wt_raw     : instantaneous frequency of the raw (direction-rectified)
                 Hilbert phase -- same array GP itself uses to detect
                 phase-slip epochs.
    idx        : final (post-nwin-extension) phase-slip mask -- the samples
                 GP actually reconstructed. Identical to the main pipeline's
                 own `idx`.
    idx_raw    : PRE-extension mask (wt_raw < lp only) -- the literal
                 low-frequency/reversal detection before the nwin safety
                 margin is added, so the review can distinguish "actually
                 flagged" from "extended buffer".
    raw_unwrap : naive np.unwrap of the raw (direction-rectified) Hilbert
                 phase across the WHOLE trace, including through phase-slip
                 epochs. This is the uncorrected artifact itself -- never
                 used by the real pipeline -- kept here only so it can be
                 plotted next to the GP-corrected unwrap for comparison.
    """
    x = np.asarray(x, dtype=np.float64)
    npts = x.shape[0]
    dt = 1.0 / fs

    def _inst_freq(xo):
        wt = np.zeros(npts)
        wt[:-1] = np.angle(xo[1:] * np.conj(xo[:-1])) / (2 * np.pi * dt)
        return wt

    xo = signal.hilbert(x)
    ph = np.angle(xo)
    md = np.abs(xo)
    wt_raw = _inst_freq(xo)

    finite_wt = wt_raw[np.isfinite(wt_raw)]
    sign_if = np.sign(np.mean(finite_wt)) if finite_wt.size else 1.0
    if sign_if == -1:
        xo = md * np.exp(1j * (sign_if * ph))
        ph = np.angle(xo)
        md = np.abs(xo)
        wt_raw = _inst_freq(xo)

    if np.all(np.isnan(ph)):
        nanarr = np.full(npts, np.nan, dtype=np.complex128)
        return nanarr, wt_raw, np.ones(npts, dtype=bool), np.ones(npts, dtype=bool), np.full(npts, np.nan)

    raw_unwrap = np.unwrap(ph)

    idx_raw = wt_raw < lp
    idx_raw[0] = False

    idx = idx_raw.copy()
    labeled, n_groups = label(idx)
    for kk in range(1, n_groups + 1):
        idxs = np.flatnonzero(labeled == kk)
        start, stop = idxs[0], idxs[-1]
        extended_stop = min(start + (stop - start) * nwin, npts - 1)
        idx[start:extended_stop + 1] = True

    valid = ~idx
    if np.count_nonzero(valid) < 2:
        nanarr = np.full(npts, np.nan, dtype=np.complex128)
        return nanarr, wt_raw, np.ones(npts, dtype=bool), idx_raw, raw_unwrap

    valid_positions = np.flatnonzero(valid)
    p_valid_unwrapped = np.unwrap(ph[valid])

    p = np.empty(npts, dtype=np.float64)
    p[valid] = p_valid_unwrapped
    invalid_positions = np.flatnonzero(idx)
    if invalid_positions.size:
        filler = PchipInterpolator(valid_positions, p_valid_unwrapped, extrapolate=True)
        p[invalid_positions] = filler(invalid_positions)

    p = _gp_rewrap(p)
    xgp = md * np.exp(1j * p)
    return xgp, wt_raw, idx, idx_raw, raw_unwrap


def gp_instantaneous_frequency(xgp, idx, fs, lowcut, highcut):
    """Ported verbatim from the main pipeline."""
    dt = 1.0 / fs
    freq = np.full(len(xgp), np.nan)
    freq[:-1] = np.angle(xgp[1:] * np.conj(xgp[:-1])) / (2 * np.pi * dt)
    reconstructed = idx[:-1] | idx[1:]
    freq[:-1][reconstructed] = np.nan
    with np.errstate(invalid='ignore'):
        out_of_band = (freq < lowcut) | (freq > highcut)
    freq[out_of_band] = np.nan
    return freq


# ============================================================================
# Epoch bookkeeping
# ============================================================================

class ReversalEpoch:
    __slots__ = ('start', 'stop', 'raw_start', 'raw_stop', 'is_true_reversal', 'min_freq')

    def __init__(self, start, stop, raw_start, raw_stop, is_true_reversal, min_freq):
        self.start = start          # first sample index of the FINAL (extended) flagged run
        self.stop = stop            # last sample index (inclusive)
        self.raw_start = raw_start  # first sample index of the literal low-freq detection (pre-nwin)
        self.raw_stop = raw_stop
        self.is_true_reversal = is_true_reversal  # wt_raw actually went negative inside the raw span
        self.min_freq = min_freq    # minimum wt_raw (Hz) inside the raw span


def find_reversal_epochs(idx, idx_raw, wt_raw) -> list:
    """Group the final (post-nwin) idx mask into contiguous epochs, and
    locate the literal (pre-extension) low-frequency span and worst-case
    instantaneous frequency inside each one, for the review window's title.
    """
    labeled, n_groups = label(idx)
    epochs = []
    for kk in range(1, n_groups + 1):
        positions = np.flatnonzero(labeled == kk)
        start, stop = int(positions[0]), int(positions[-1])

        raw_positions = np.flatnonzero(idx_raw[start:stop + 1]) + start
        if raw_positions.size:
            raw_start, raw_stop = int(raw_positions[0]), int(raw_positions[-1])
            span_freq = wt_raw[raw_start:raw_stop + 1]
        else:
            raw_start, raw_stop = start, stop
            span_freq = wt_raw[start:stop + 1]

        min_freq = float(np.nanmin(span_freq)) if span_freq.size else np.nan
        is_true_reversal = bool(np.isfinite(min_freq) and min_freq < 0)

        epochs.append(ReversalEpoch(start, stop, raw_start, raw_stop, is_true_reversal, min_freq))
    return epochs


# ============================================================================
# Interactive reviewer
# ============================================================================

class EpochReviewer:
    def __init__(self, t, x, wt_raw, wt_post, ph_raw_wrapped, ph_gp_wrapped,
                 raw_unwrap, gp_unwrap, idx, idx_raw, epochs, fs, filter_band, title):
        self.t = t
        self.x = x
        self.wt_raw = wt_raw
        self.wt_post = wt_post
        self.ph_raw_wrapped = ph_raw_wrapped
        self.ph_gp_wrapped = ph_gp_wrapped
        self.raw_unwrap = raw_unwrap
        self.gp_unwrap = gp_unwrap
        self.idx = idx
        self.idx_raw = idx_raw
        self.epochs = epochs
        self.fs = fs
        self.filter_band = filter_band
        self.title = title
        self.n = len(epochs)
        self.i = 0

        self.fig, self.axes = plt.subplots(4, 1, figsize=(11, 10), sharex=False)
        self.fig.subplots_adjust(bottom=0.14, hspace=0.55, top=0.90)

        ax_prev = self.fig.add_axes([0.30, 0.02, 0.10, 0.045])
        ax_next = self.fig.add_axes([0.42, 0.02, 0.10, 0.045])
        ax_box = self.fig.add_axes([0.60, 0.02, 0.08, 0.045])
        self.btn_prev = Button(ax_prev, '< Prev')
        self.btn_next = Button(ax_next, 'Next >')
        self.box = TextBox(ax_box, 'Go to #', initial='1')
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.box.on_submit(self._on_submit)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self._draw()

    def _on_prev(self, _event=None):
        if self.n == 0:
            return
        self.i = (self.i - 1) % self.n
        self._draw()

    def _on_next(self, _event=None):
        if self.n == 0:
            return
        self.i = (self.i + 1) % self.n
        self._draw()

    def _on_submit(self, text):
        if self.n == 0:
            return
        try:
            k = int(text) - 1
        except ValueError:
            return
        self.i = min(max(k, 0), self.n - 1)
        self._draw()

    def _on_key(self, event):
        if event.key == 'right':
            self._on_next()
        elif event.key == 'left':
            self._on_prev()

    def _draw(self):
        for ax in self.axes:
            ax.clear()

        if self.n == 0:
            self.axes[0].text(0.5, 0.5, 'No phase-reversal / phase-slip epochs detected.',
                               ha='center', va='center', fontsize=13, transform=self.axes[0].transAxes)
            for ax in self.axes[1:]:
                ax.axis('off')
            self.fig.canvas.draw_idle()
            return

        ep = self.epochs[self.i]
        fs = self.fs
        ctx = max(int(CONTEXT_SEC * fs), int(MIN_CONTEXT_CYCLES / self.filter_band[0] * fs))
        lo = max(0, ep.start - ctx)
        hi = min(len(self.t) - 1, ep.stop + ctx)
        sl = slice(lo, hi + 1)

        t_win = self.t[sl]
        raw_span_t = (self.t[ep.raw_start], self.t[ep.raw_stop])
        ext_span_t = (self.t[ep.start], self.t[ep.stop])

        # Panel 1: filtered LFP trace
        ax = self.axes[0]
        ax.plot(t_win, self.x[sl], color='0.2', linewidth=0.9)
        ax.axvspan(*ext_span_t, color='orange', alpha=0.25, label='nwin-extended (blanked+reconstructed)')
        ax.axvspan(*raw_span_t, color='red', alpha=0.35, label='literal detection (freq < low band edge)')
        ax.set_ylabel('LFP (uV)')
        kind = 'TRUE REVERSAL (freq < 0 Hz)' if ep.is_true_reversal else 'low-frequency dip (not negative)'
        ax.set_title(f'Epoch {self.i + 1}/{self.n}  |  {kind}  |  min raw freq = {ep.min_freq:.2f} Hz\n'
                      f'{self.title}', fontsize=10)
        ax.legend(loc='upper right', fontsize=7)

        # Panel 2: wrapped phase, raw vs GP
        ax = self.axes[1]
        ax.plot(t_win, np.degrees(self.ph_raw_wrapped[sl]), '.-', color='#4C72B0', markersize=2,
                linewidth=0.8, label='raw Hilbert phase (wrapped)')
        ax.plot(t_win, np.degrees(self.ph_gp_wrapped[sl]), '.-', color='#DD8452', markersize=2,
                linewidth=0.8, label='GP-corrected phase (wrapped)')
        ax.axvspan(*ext_span_t, color='orange', alpha=0.15)
        ax.axvspan(*raw_span_t, color='red', alpha=0.2)
        ax.set_ylabel('Phase (deg)')
        ax.set_ylim(-185, 185)
        ax.legend(loc='upper right', fontsize=7)

        # Panel 3: unwrapped phase, raw (naive, uncorrected) vs GP, aligned at window start
        ax = self.axes[2]
        raw_u = np.degrees(self.raw_unwrap[sl]); raw_u = raw_u - raw_u[0]
        gp_u = np.degrees(self.gp_unwrap[sl]); gp_u = gp_u - gp_u[0]
        ax.plot(t_win, raw_u, color='#4C72B0', linewidth=1.2, label='raw unwrap (uncorrected artifact)')
        ax.plot(t_win, gp_u, color='#DD8452', linewidth=1.2, label='GP-corrected unwrap')
        ax.axvspan(*ext_span_t, color='orange', alpha=0.15)
        ax.axvspan(*raw_span_t, color='red', alpha=0.2)
        ax.axhline(0, color='0.7', linewidth=0.7)
        ax.set_ylabel('Cumulative phase\n(deg, aligned)')
        ax.legend(loc='upper right', fontsize=7)

        # Panel 4: instantaneous frequency, raw vs GP-post
        ax = self.axes[3]
        ax.plot(t_win, self.wt_raw[sl], color='#4C72B0', linewidth=1.0, label='raw inst. freq (pre-GP)')
        ax.plot(t_win, self.wt_post[sl], color='#DD8452', linewidth=1.4, label='GP inst. freq (post, masked)')
        ax.axhline(0, color='k', linewidth=0.7)
        ax.axhline(self.filter_band[0], color='red', linestyle='--', linewidth=0.8, label='filter low edge')
        ax.axhline(self.filter_band[1], color='red', linestyle='--', linewidth=0.8)
        ax.axvspan(*ext_span_t, color='orange', alpha=0.15)
        ax.axvspan(*raw_span_t, color='red', alpha=0.2)
        y_lo = min(-2.0, np.nanmin(self.wt_raw[sl]) if np.any(np.isfinite(self.wt_raw[sl])) else -2.0)
        y_hi = self.filter_band[1] + 5.0
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel('Inst. freq (Hz)')
        ax.set_xlabel('Time (s)')
        ax.legend(loc='upper right', fontsize=7)

        self.box.set_val(str(self.i + 1))
        self.fig.canvas.draw_idle()


# ============================================================================
# Main
# ============================================================================

def _pick_lfp_file() -> Path:
    if LFP_FILE_OVERRIDE is not None:
        return Path(LFP_FILE_OVERRIDE)
    ncs_files = sorted(DATA_FOLDER.glob('*.ncs'), key=_natural_key)
    if not ncs_files:
        raise FileNotFoundError(f'No .ncs files found in {DATA_FOLDER}')
    return ncs_files[0]


def main():
    lfp_path = _pick_lfp_file()
    print(f'Loading LFP: {lfp_path}')
    lfp_sig, lfp_ts, fs = load_ncs(lfp_path)

    if START_SEC is not None or END_SEC is not None:
        t0 = START_SEC if START_SEC is not None else lfp_ts[0]
        t1 = END_SEC if END_SEC is not None else lfp_ts[-1]
        mask = (lfp_ts >= t0) & (lfp_ts <= t1)
        lfp_sig, lfp_ts = lfp_sig[mask], lfp_ts[mask]
        print(f'Restricted to [{t0:.2f}, {t1:.2f}] s -> {len(lfp_sig)} samples')

    print(f'fs = {fs:.3f} Hz, duration = {(lfp_ts[-1] - lfp_ts[0]):.1f} s, filter band = {LFP_FILTER_BAND}')

    filtered = bandpass_filter(lfp_sig, LFP_FILTER_BAND[0], LFP_FILTER_BAND[1], fs)
    xgp, wt_raw, idx, idx_raw, raw_unwrap = generalized_phase_vector_debug(
        filtered, fs, LFP_FILTER_BAND[0], nwin=NWIN)
    wt_post = gp_instantaneous_frequency(xgp, idx, fs, LFP_FILTER_BAND[0], LFP_FILTER_BAND[1])

    ph_raw_wrapped = _gp_rewrap(raw_unwrap)
    ph_gp_wrapped = np.angle(xgp)
    gp_unwrap = np.unwrap(ph_gp_wrapped)

    epochs = find_reversal_epochs(idx, idx_raw, wt_raw)

    n_true_reversal = sum(1 for e in epochs if e.is_true_reversal)
    pct_flagged = 100.0 * np.count_nonzero(idx) / len(idx)
    print(f'\nFound {len(epochs)} phase-slip epoch(s) '
          f'({n_true_reversal} with a literal negative-frequency reversal, '
          f'{len(epochs) - n_true_reversal} low-frequency-only).')
    print(f'{pct_flagged:.3f}% of samples were inside a blanked+reconstructed (nwin-extended) span.')
    if epochs:
        durations_ms = [(e.stop - e.start + 1) / fs * 1000.0 for e in epochs]
        print(f'Reconstructed-span duration: median={np.median(durations_ms):.1f} ms, '
              f'max={np.max(durations_ms):.1f} ms')

    EpochReviewer(lfp_ts, filtered, wt_raw, wt_post, ph_raw_wrapped, ph_gp_wrapped,
                  raw_unwrap, gp_unwrap, idx, idx_raw, epochs, fs, LFP_FILTER_BAND,
                  title=lfp_path.name)
    plt.show()


if __name__ == '__main__':
    main()
