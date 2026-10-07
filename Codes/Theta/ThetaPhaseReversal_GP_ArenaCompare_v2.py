# -*- coding: utf-8 -*-
"""
Theta phase-reversal detection (Generalized-Phase criterion) + manual review GUI.

For every .ncs LFP file found (recursively) under ROOT_FOLDER, one file after
another:

  1. Load + clean the LFP (same cleaning as
     ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean_v3.py / the ACG step of
     ThetaMod_PhasePrec_v18.py): resample to FS (1 kHz), notch the mains
     harmonics (50/100/150/200 Hz), linear detrend, zero-phase 100 Hz low-pass.
  2. Theta-positive epochs via the ACG method (Dunn et al. 2022, Nat Commun
     13:6997, Supp. Fig. 5): the cleaned trace is split into 1 s epochs;
     epochs whose delta (1-3 Hz) power exceeds theta (3-7 Hz) power, or whose
     peak-to-peak amplitude is a robust MAD outlier, are rejected; each
     remaining epoch's autocorrelogram is matched against a bank of 3-7 Hz
     reference-sinusoid autocorrelograms and the epoch is theta-positive if
     the normalised Euclidean distance to the best match, ED_min, is below
     ACG_ED_MIN_THRESH.
  3. Theta phase by Hilbert transform of the 3-7 Hz band-passed cleaned LFP.
  4. Phase reversals detected with the Generalized Phase (GP) criterion
     (Davis, Muller et al. 2020, Nature 587:432-436; generalized_phase_vector.m,
     github.com/mullerlab/generalized-phase -- see generalized_phase_vector in
     ThetaPhaseBreak_RefCode.py). GP computes the instantaneous frequency of
     the analytic signal sample by sample,
         IF[i] = angle(x[i+1] * conj(x[i])) / (2*pi*dt),
     and flags the epochs where IF drops below a threshold. GP's own pipeline
     then *corrects* (pchip-interpolates) the phase across those epochs; here
     the phase is left untouched and the flagged epochs are instead kept as
     candidate phase-reversal events. Default threshold = GP's own: IF below
     the passband's low edge (THETA_BAND[0]). In a band-passed signal a
     backward phase jump smaller than 180 deg appears as an IF dip, not as
     negative IF, so this is the criterion that catches them; set
     REVERSAL_IF_THRESH_HZ = 0 to keep only strict negative-frequency
     epochs (phase literally running backwards). Flagged runs closer than
     REVERSAL_MERGE_GAP_MS are merged into one event, and only events lying
     entirely inside ACG theta-positive epochs are kept. GP's nwin-extended
     epoch (the window GP would have overwritten) is reported for reference.
  5. A plot of when the reversals occur across the whole recording (raster,
     rate per minute of theta-positive LFP, theta-positive fraction). Events in
     the session's .nev file whose text contains 'zero' are drawn as red
     vertical lines.
  6. Once every file has been processed, MANUAL_REVIEW decides what happens.
     True opens a Tkinter GUI that shows each event (cleaned LFP + theta +
     envelope, Hilbert phase, instantaneous frequency, and the event's position
     in the whole recording) so it can be accepted (biological) or rejected
     (noise). Files with events are offered one at a time and any file can be
     skipped. False marks every unreviewed event as accepted (decision = 'auto')
     without showing the GUI.
  7. Gaussian KDE of when the accepted reversals occur (scipy.stats.gaussian_kde,
     fixed kernel SD KDE_BANDWIDTH_S), divided by the KDE of theta-positive time
     made with the same kernel. The result is a smooth rate in reversals per min of
     theta-positive LFP, flat at the session mean if reversals occur at a constant
     rate whenever theta is present. Peaks are tested against a Monte-Carlo null
     that spreads the same events over theta-positive time only (KDE_NULL):
     'isi' shuffles the inter-event intervals measured in theta time, which keeps
     bursts of reversals; 'uniform' places events at random (Poisson). The rate
     is z-scored against the null's mean and SD at each time point, and a peak is
     significant if its z exceeds the (1 - KDE_ALPHA) quantile of the null's
     *maximum* z over the whole recording: a studentized global envelope that
     corrects for testing every time point. Its p_global is the fraction of
     shuffles whose maximum z reaches the peak's z.

Outputs, per .ncs file, in <session folder>/ThetaPhaseReversals/:
    <stem>_PhaseReversals.csv              one row per event, incl. review status
                                           (saved after every decision)
    <stem>_PhaseReversals_info.json        file-level info (durations, zero events)
    <stem>_ReversalOccurrence.png          occurrence plot of all detected events
    <stem>_ReversalOccurrence_Reviewed.png same plot coloured by review decision
    <stem>_ReversalKDE.csv                 KDE rate, theta occupancy, null band on the time grid
    <stem>_ReversalKDE_Peaks.csv           significant KDE peaks (one row per peak)
    <stem>_ReversalKDE.png                 KDE rate vs null + significant peaks
And ROOT_FOLDER/ThetaPhaseReversal_Summary.xlsx: one row per .ncs file
(counts, rates, review status) + the analysis parameters.

Re-running the script restores earlier manual accept/reject decisions from an
existing <stem>_PhaseReversals.csv (events matched on start time to the ms),
so a review can be interrupted and resumed. Auto-accepted events (MANUAL_REVIEW = False)
are not restored -- they come back as unreviewed. Changing detection
parameters changes the events; decisions for events that no longer exist are
dropped.

All files are detected before the review starts and their results are kept in
memory until reviewed (files without events are not kept).

Assumes Neuralynx timestamps (.ncs and .nev) share one clock (they do when
both come from the same acquisition system).

Requires: numpy, scipy, pandas, matplotlib, openpyxl, tkinter.
"""

from __future__ import annotations

import json
import re
import traceback
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import next_fast_len, rfft, irfft

import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.figure import Figure
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ============================================================================
# Configuration -- EDIT THESE
# ============================================================================

ROOT_FOLDER = Path(r"X:/NMR_group_data/Runita/Analysis/Mean_KDE_Open_PascalOldenburg/LinearTrack_GeoMagVsZero/Data")
OUTPUT_SUBFOLDER = 'ThetaPhaseReversals'           # created inside each session folder
SUMMARY_EXCEL_NAME = 'ThetaPhaseReversal_Summary.xlsx'   # written to ROOT_FOLDER

MANUAL_REVIEW = False    # True: after detecting all files, open the GUI to accept/reject each reversal
                        # False: no GUI -- accept every unreviewed reversal (decision = 'auto')

# --- LFP cleaning (ACG_theta_continuity_..._LFPclean_v3.py) ---
FS = 1000.0                                  # Hz, every file is resampled to this rate
LINE_HARMONICS = [50.0, 100.0, 150.0, 200.0]  # mains notch frequencies
NOTCH_Q = 30.0
LOWPASS_CUTOFF_HZ = 100.0
LOWPASS_ORDER = 4

# --- ACG theta-positive epochs (Dunn et al. 2022; see acg_theta_epoch_mask) ---
USE_ACG_THETA_EPOCHS = True       # False: keep reversals from the whole recording
ACG_ED_MIN_THRESH = 0.95           # theta-positive if ED_min is strictly below this
ACG_EPOCH_SEC = 1.0
ACG_FREQ_RANGE = (3.0, 7.0)       # Hz, reference sinusoid bank
ACG_FREQ_RES = 0.1                # Hz, bank step
ACG_REJECT_DELTA_OVER_THETA = True    # drop epochs whose 1-3 Hz power exceeds 3-7 Hz power
ACG_DELTA_BAND = (1.0, 3.0)
ACG_REJECT_MAD_ARTIFACTS = True       # drop epochs with outlier peak-to-peak amplitude
ACG_MAD_THRESH = 5.0

# --- Theta phase (Hilbert transform) ---
THETA_BAND = (3.0, 7.0)           # Hz
THETA_FILTER_ORDER = 4            # Butterworth, zero-phase (sosfiltfilt)

# --- Generalized-Phase reversal detection ---
REVERSAL_IF_THRESH_HZ = THETA_BAND[0]   # flag samples whose instantaneous frequency is below this.
                                  # THETA_BAND[0] = GP's own criterion (IF below the passband's
                                  # low edge). After the 3-7 Hz band-pass a backward phase jump
                                  # < 180 deg only slows the IF (e.g. 120 deg -> ~2.6 Hz, never
                                  # negative), so 0 Hz would catch only near-180 deg jumps at
                                  # envelope collapses (mostly noise-like).
                                  # 0 = strict negative frequency (phase literally runs backwards).
GP_NWIN = 3                       # GP safety-window multiplier (reported only)
REVERSAL_MERGE_GAP_MS = 10.0      # flagged runs closer than this are one event
REVERSAL_MIN_DUR_MS = 0.0         # drop events shorter than this (0 keeps all)
EDGE_EXCLUDE_S = 2.0              # ignore the first/last seconds (filter edge effects)

# --- .nev events ---
ZERO_EVENT_TEXT = 'zero'          # case-insensitive substring marking the red lines

# --- Gaussian KDE of accepted reversal times + peak significance ---
RUN_KDE = True
KDE_BANDWIDTH_S = 30.0            # Gaussian kernel SD (s). Fixed in seconds (not Scott's rule) so
                                  # the observed and every null KDE are smoothed identically
KDE_GRID_S = 1.0                  # evaluation step (s)
KDE_MIN_THETA_FRAC = 0.2          # rate evaluated only where the kernel-weighted theta-positive
                                  # fraction is >= this (rate is unstable where theta is rare)
KDE_NULL = 'isi'                  # 'isi': shuffle inter-event intervals in theta time (keeps bursts)
                                  # 'uniform': events uniformly at random in theta time (Poisson)
KDE_N_SHUFFLES = 1000
KDE_ALPHA = 0.05                  # family-wise (whole-recording) significance level
KDE_MIN_EVENTS = 5                # skip the KDE below this many accepted events
KDE_SEED = 0

# --- Plots / GUI ---
RATE_BIN_S = 60.0                 # bin width of the reversal-rate panel
REVIEW_HALF_WINDOW_S = 1.0        # GUI shows +/- this around the event (zoomable)
IF_PLOT_LIMS = (-10.0, 15.0)      # Hz, y-range of the GUI instantaneous-frequency panel
STATUS_COLORS = {'unreviewed': '#333333', 'accepted': '#1b9e77', 'rejected': '#c0c0c0'}
THETA_SHADE = '#7fc97f'
ZERO_COLOR = 'red'

NCS_SAMPLES_PER_RECORD = 512
HEADER_BYTES = 16 * 1024
DEFAULT_ADBITVOLTS = 0.000000195

PARAM_NAMES = [
    'FS', 'LINE_HARMONICS', 'NOTCH_Q', 'LOWPASS_CUTOFF_HZ', 'LOWPASS_ORDER',
    'USE_ACG_THETA_EPOCHS', 'ACG_ED_MIN_THRESH', 'ACG_EPOCH_SEC', 'ACG_FREQ_RANGE', 'ACG_FREQ_RES',
    'ACG_REJECT_DELTA_OVER_THETA', 'ACG_DELTA_BAND', 'ACG_REJECT_MAD_ARTIFACTS', 'ACG_MAD_THRESH',
    'THETA_BAND', 'THETA_FILTER_ORDER', 'REVERSAL_IF_THRESH_HZ', 'GP_NWIN', 'REVERSAL_MERGE_GAP_MS',
    'REVERSAL_MIN_DUR_MS', 'EDGE_EXCLUDE_S', 'ZERO_EVENT_TEXT', 'RATE_BIN_S',
    'RUN_KDE', 'KDE_BANDWIDTH_S', 'KDE_GRID_S', 'KDE_MIN_THETA_FRAC', 'KDE_NULL', 'KDE_N_SHUFFLES',
    'KDE_ALPHA', 'KDE_MIN_EVENTS', 'KDE_SEED',
]


def analysis_params() -> dict:
    g = globals()
    return {n: g[n] for n in PARAM_NAMES}


# ============================================================================
# Neuralynx file I/O
# ============================================================================

NCS_DTYPE = np.dtype([
    ('TimeStamp', '<u8'),
    ('ChannelNumber', '<u4'),
    ('SampleFreq', '<u4'),
    ('NumValidSamples', '<u4'),
    ('Samples', '<i2', (NCS_SAMPLES_PER_RECORD,)),
])

NEV_DTYPE = np.dtype([
    ('nstx', '<i2'),
    ('npkt_id', '<i2'),
    ('npkt_data_size', '<i2'),
    ('TimeStamp', '<u8'),
    ('nevent_id', '<i2'),
    ('nttl', '<i2'),
    ('ncrc', '<i2'),
    ('ndummy1', '<i2'),
    ('ndummy2', '<i2'),
    ('dnExtra', '<i4', (8,)),
    ('EventString', 'S128'),
])


def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(path))]


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
    """Load a Neuralynx .ncs file.

    Returns (samples_uV, fs_hz, anchor_idx, anchor_t_s): the concatenated valid
    samples, the sampling rate, and (sample index -> timestamp) anchor points at
    the first and last valid sample of every record, so per-sample times can be
    interpolated without building a full-rate timestamp array (and stay correct
    across gaps between records).
    """
    header_text = _read_header_text(path)
    adbitvolts = _header_field(header_text, 'ADBitVolts', default=DEFAULT_ADBITVOLTS)

    records = np.memmap(path, dtype=NCS_DTYPE, mode='r', offset=HEADER_BYTES)
    if len(records) == 0:
        raise ValueError(f'No records found in {path}')

    fs = float(records['SampleFreq'][0])
    if fs <= 0:
        fs = _header_field(header_text, 'SamplingFrequency', default=32000.0)

    nvalid = np.clip(records['NumValidSamples'].astype(np.int64), 0, NCS_SAMPLES_PER_RECORD)
    rec_ts_s = records['TimeStamp'].astype(np.float64) / 1e6
    samples = np.asarray(records['Samples'])
    if np.all(nvalid == NCS_SAMPLES_PER_RECORD):
        raw = samples.reshape(-1)
    else:
        raw = samples[np.arange(NCS_SAMPLES_PER_RECORD)[None, :] < nvalid[:, None]]
    samples_uV = raw.astype(np.float64) * adbitvolts * 1e6

    starts = np.concatenate(([0], np.cumsum(nvalid)[:-1]))
    has = nvalid > 0
    anchor_idx = np.column_stack([starts[has], starts[has] + nvalid[has] - 1]).ravel().astype(np.float64)
    anchor_t = np.column_stack([rec_ts_s[has], rec_ts_s[has] + (nvalid[has] - 1) / fs]).ravel()
    return samples_uV, fs, anchor_idx, anchor_t


def find_nev_file(folder: Path):
    candidates = sorted(folder.glob('*.nev'), key=_natural_key)
    if len(candidates) > 1:
        print(f'    Multiple .nev files in {folder}, using {candidates[0].name}')
    return candidates[0] if candidates else None


def load_nev_events(nev_path: Path) -> pd.DataFrame:
    """Event records of a .nev file: time_s (Neuralynx clock), ttl, event string."""
    records = np.fromfile(nev_path, dtype=NEV_DTYPE, offset=HEADER_BYTES)
    strings = [s.split(b'\x00')[0].decode('latin-1', errors='replace').strip()
               for s in records['EventString']]
    return pd.DataFrame({'time_s': records['TimeStamp'].astype(np.float64) / 1e6,
                         'ttl': records['nttl'].astype(int),
                         'event_string': strings})


# ============================================================================
# LFP cleaning
# ============================================================================

def _notch(x, fs_hz, freqs, q):
    y = np.asarray(x, dtype=np.float64)
    for f0 in freqs:
        if 0 < f0 < fs_hz / 2.0:
            b, a = signal.iirnotch(f0, q, fs_hz)
            y = signal.filtfilt(b, a, y)
    return y


def bandpass_filter(x, low, high, fs, order=4):
    sos = signal.butter(order, [low, high], btype='band', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, x)


def preprocess_lfp(samples_uV, fs_native, anchor_idx, anchor_t):
    """Resample to FS, notch mains harmonics, linear detrend, low-pass.
    Returns (clean_uV, t_s) with t_s the Neuralynx time of every output sample."""
    fs_in = int(round(fs_native))
    g = gcd(int(FS), fs_in)
    x = signal.resample_poly(samples_uV, int(FS) // g, fs_in // g)
    x = _notch(x, FS, LINE_HARMONICS, NOTCH_Q)
    x = signal.detrend(x, type='linear')
    sos = signal.butter(LOWPASS_ORDER, LOWPASS_CUTOFF_HZ, btype='low', fs=FS, output='sos')
    x = signal.sosfiltfilt(sos, x)
    t = np.interp(np.arange(len(x)) * (fs_native / FS), anchor_idx, anchor_t)
    return x, t


# ============================================================================
# ACG theta-positive epochs (Dunn et al. 2022; same criteria as
# ThetaMod_PhasePrec_v18.acg_theta_epoch_mask, vectorised, run on the already
# cleaned trace)
# ============================================================================

def _autocorr_full(x2d):
    """Full-lag autocorrelation of every row (== signal.correlate(x, x, 'full'))."""
    n = x2d.shape[-1]
    nfft = next_fast_len(2 * n - 1)
    spec = rfft(x2d, nfft, axis=-1)
    r = irfft(np.abs(spec) ** 2, nfft, axis=-1)
    return np.concatenate([r[..., nfft - (n - 1):], r[..., :n]], axis=-1)


def acg_theta_epoch_mask(clean):
    """Per-epoch theta-positive mask. Epoch i covers samples
    [i*nperseg, (i+1)*nperseg). Returns (ok, ed_min); ed_min is NaN for epochs
    rejected by the delta/theta or MAD screens."""
    nperseg = int(round(FS * ACG_EPOCH_SEC))
    n_total = len(clean) // nperseg
    if n_total == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    epochs = clean[:n_total * nperseg].reshape(n_total, nperseg)

    keep = np.ones(n_total, dtype=bool)
    if ACG_REJECT_DELTA_OVER_THETA:
        f, pxx = signal.welch(epochs, fs=FS, window='hann', nperseg=nperseg, noverlap=0,
                              detrend='constant', axis=-1)
        low = pxx[:, (f >= ACG_DELTA_BAND[0]) & (f <= ACG_DELTA_BAND[1])].sum(axis=1)
        theta = pxx[:, (f >= ACG_FREQ_RANGE[0]) & (f <= ACG_FREQ_RANGE[1])].sum(axis=1)
        keep &= low <= theta
    if ACG_REJECT_MAD_ARTIFACTS and keep.any():
        p2p = epochs.max(axis=1) - epochs.min(axis=1)
        med = np.median(p2p[keep])
        mad = np.median(np.abs(p2p[keep] - med)) or 1e-20
        keep &= ~(0.6745 * (p2p - med) / mad > ACG_MAD_THRESH)

    ref_freqs = np.arange(ACG_FREQ_RANGE[0], ACG_FREQ_RANGE[1] + ACG_FREQ_RES / 2, ACG_FREQ_RES)
    reft = np.arange(nperseg) / FS
    ref_xc = _autocorr_full(np.sin(2 * np.pi * ref_freqs[:, None] * reft[None, :]))
    ref_xc /= ref_xc.max(axis=1, keepdims=True)
    ref_sq = np.sum(ref_xc ** 2, axis=1)
    ref_norm = np.sqrt(ref_sq)

    ed_min = np.full(n_total, np.nan)
    kept = np.flatnonzero(keep)
    for c in range(0, len(kept), 1000):
        sel = kept[c:c + 1000]
        xc = _autocorr_full(epochs[sel])
        peak = xc.max(axis=1)
        good = np.isfinite(peak) & (peak > 0)
        xc = xc[good] / peak[good, None]
        ed2 = ref_sq[None, :] - 2 * xc @ ref_xc.T + np.sum(xc ** 2, axis=1)[:, None]
        ed = np.sqrt(np.clip(ed2, 0, None)) / ref_norm[None, :]
        ed_min[sel[good]] = ed.min(axis=1)

    ok = np.isfinite(ed_min) & (ed_min < ACG_ED_MIN_THRESH)
    return ok, ed_min


def _true_runs(mask):
    """(starts, stops) of the runs of True in a 1-D bool array (stops exclusive)."""
    d = np.diff(np.asarray(mask, dtype=np.int8), prepend=0, append=0)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


# ============================================================================
# Hilbert phase + Generalized-Phase reversal detection
# ============================================================================

def analytic_signal(x):
    n = len(x)
    return signal.hilbert(x, N=next_fast_len(n))[:n]


def gp_instantaneous_frequency(xa):
    """Instantaneous frequency (Hz) of an analytic signal as in
    generalized_phase_vector.m: IF[i] = angle(x[i+1] conj(x[i])) / (2 pi dt),
    i.e. the phase step from sample i to i+1. If the mean IF is negative the
    rotation direction is flipped first (GP's sign rectification).
    Returns (inst_freq, xa) with xa possibly conjugated."""
    wt = np.empty(len(xa))
    wt[:-1] = np.angle(xa[1:] * np.conj(xa[:-1])) * FS / (2 * np.pi)
    wt[-1] = wt[-2]
    if np.nanmean(wt) < 0:
        xa, wt = np.conj(xa), -wt
    return wt, xa


EVENT_COLUMNS = ['event_id', 'start_idx', 'stop_idx', 'gp_ext_stop_idx', 'start_s', 'stop_s',
                 'start_rel_s', 'center_rel_s', 'duration_ms', 'gp_ext_duration_ms', 'min_IF_Hz',
                 'phase_lost_deg', 'backward_phase_deg', 'envelope_uV', 'envelope_ratio', 'acg_epoch', 'acg_ED_min',
                 'status', 'decision']   # decision: 'none' | 'manual' (GUI) | 'auto' (accept all)


def detect_phase_reversals(inst_freq, env, theta_ok_sample, epoch_ed_min, t):
    """Runs of samples with IF < REVERSAL_IF_THRESH_HZ (the GP criterion),
    merged, duration-filtered and restricted to theta-positive epochs.

    An event spans IF steps start..stop-1, i.e. LFP samples start..stop.
    phase_lost_deg = phase the event fell behind a cycle running at the median
    theta-positive IF (size of the slip/reversal);
    backward_phase_deg = phase actually travelled backwards (negative IF only);
    envelope_ratio = mean Hilbert envelope in the event / median envelope over
    theta-positive samples (reversals at very low amplitude are often noise)."""
    n = len(inst_freq)
    flagged = inst_freq < REVERSAL_IF_THRESH_HZ
    edge = int(round(EDGE_EXCLUDE_S * FS))
    flagged[:max(edge, 1)] = False
    flagged[n - max(edge, 1):] = False
    starts, stops = _true_runs(flagged)

    merge_gap = int(round(REVERSAL_MERGE_GAP_MS * FS / 1000))
    if len(starts) > 1 and merge_gap > 0:
        brk = (starts[1:] - stops[:-1]) >= merge_gap
        starts = np.concatenate(([starts[0]], starts[1:][brk]))
        stops = np.concatenate((stops[:-1][brk], [stops[-1]]))

    dur_ms = (stops - starts) / FS * 1000
    keep = dur_ms >= REVERSAL_MIN_DUR_MS
    if USE_ACG_THETA_EPOCHS:
        bad_cum = np.concatenate(([0], np.cumsum(~theta_ok_sample)))
        keep &= (bad_cum[stops + 1] - bad_cum[starts]) == 0
    starts, stops = starts[keep], stops[keep]
    if len(starts) == 0:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    pairs = np.column_stack([starts, stops]).ravel()
    n_steps = stops - starts
    min_if = np.minimum.reduceat(inst_freq, pairs)[::2]
    backward = 0.0 - np.add.reduceat(np.minimum(inst_freq, 0.0), pairs)[::2] / FS * 360.0
    env_mean = np.add.reduceat(env, pairs)[::2] / n_steps
    ref = theta_ok_sample if theta_ok_sample.any() else np.ones(n, dtype=bool)
    ref_env, ref_if = np.median(env[ref]), np.median(inst_freq[ref])
    phase_lost = (ref_if * n_steps - np.add.reduceat(inst_freq, pairs)[::2]) / FS * 360.0

    nperseg = int(round(FS * ACG_EPOCH_SEC))
    epoch = starts // nperseg
    ed = np.where(epoch < len(epoch_ed_min), epoch_ed_min[np.minimum(epoch, len(epoch_ed_min) - 1)],
                  np.nan) if len(epoch_ed_min) else np.full(len(starts), np.nan)
    gp_ext = np.minimum(starts + n_steps * GP_NWIN, n - 1)
    centers = (starts + stops) // 2

    return pd.DataFrame({
        'event_id': np.arange(1, len(starts) + 1),
        'start_idx': starts, 'stop_idx': stops, 'gp_ext_stop_idx': gp_ext,
        'start_s': t[starts], 'stop_s': t[stops],
        'start_rel_s': t[starts] - t[0], 'center_rel_s': t[centers] - t[0],
        'duration_ms': n_steps / FS * 1000, 'gp_ext_duration_ms': (gp_ext - starts) / FS * 1000,
        'min_IF_Hz': min_if, 'phase_lost_deg': phase_lost, 'backward_phase_deg': backward,
        'envelope_uV': env_mean, 'envelope_ratio': env_mean / ref_env,
        'acg_epoch': epoch, 'acg_ED_min': ed,
        'status': 'unreviewed', 'decision': 'none',
    }, columns=EVENT_COLUMNS)


# ============================================================================
# Per-file analysis + outputs
# ============================================================================

@dataclass
class FileResult:
    ncs_path: Path
    nev_path: Path | None
    t: np.ndarray               # Neuralynx time (s) of every FS sample
    clean: np.ndarray           # cleaned LFP (uV)
    theta: np.ndarray           # theta band-passed LFP (uV)
    phase: np.ndarray           # Hilbert phase (rad, wrapped)
    env: np.ndarray             # Hilbert envelope (uV)
    inst_freq: np.ndarray       # GP instantaneous frequency (Hz)
    theta_ok_sample: np.ndarray
    epoch_ok: np.ndarray
    epoch_ed_min: np.ndarray
    epoch_edges_rel_s: np.ndarray   # n_epochs + 1 epoch boundaries, s from recording start
    events: pd.DataFrame
    zero_rel_s: np.ndarray      # 'zero' .nev events, s from recording start
    n_restored: int = 0

    @property
    def t0(self):
        return float(self.t[0])

    @property
    def recording_dur_s(self):
        return float(self.t[-1] - self.t[0])

    @property
    def theta_positive_s(self):
        return float(self.epoch_ok.sum() * ACG_EPOCH_SEC)

    def theta_spans_rel_s(self):
        s, e = _true_runs(self.epoch_ok)
        return np.column_stack([self.epoch_edges_rel_s[s], self.epoch_edges_rel_s[e]])


def output_paths(ncs_path: Path) -> dict:
    out_dir = ncs_path.parent / OUTPUT_SUBFOLDER
    stem = ncs_path.stem
    return dict(dir=out_dir,
                csv=out_dir / f'{stem}_PhaseReversals.csv',
                info=out_dir / f'{stem}_PhaseReversals_info.json',
                plot=out_dir / f'{stem}_ReversalOccurrence.png',
                plot_reviewed=out_dir / f'{stem}_ReversalOccurrence_Reviewed.png',
                kde_csv=out_dir / f'{stem}_ReversalKDE.csv',
                kde_peaks=out_dir / f'{stem}_ReversalKDE_Peaks.csv',
                kde_plot=out_dir / f'{stem}_ReversalKDE.png')


def _restore_previous_statuses(events: pd.DataFrame, csv_path: Path) -> int:
    """Copy manual accept/reject decisions from an earlier run's CSV onto events
    with the same start time (to the ms). Auto-accepted events are not restored.
    Returns the number restored."""
    if events.empty or not csv_path.exists():
        return 0
    try:
        prev = pd.read_csv(csv_path)
    except Exception:
        return 0
    if not {'start_s', 'status'} <= set(prev.columns):
        return 0
    if 'decision' in prev.columns:
        prev = prev[prev['decision'] == 'manual']
    prev = prev[prev['status'].isin(['accepted', 'rejected'])]
    lut = dict(zip(np.round(prev['start_s'].to_numpy() * 1e3).astype(np.int64), prev['status']))
    keys = np.round(events['start_s'].to_numpy() * 1e3).astype(np.int64)
    events['status'] = [lut.get(k, 'unreviewed') for k in keys]
    events['decision'] = np.where(events['status'] != 'unreviewed', 'manual', 'none')
    return int((events['status'] != 'unreviewed').sum())


def accept_all_unreviewed(res: FileResult) -> int:
    """Mark every unreviewed event as accepted without review. Returns the count."""
    unrev = res.events['status'] == 'unreviewed'
    res.events.loc[unrev, 'status'] = 'accepted'
    res.events.loc[unrev, 'decision'] = 'auto'
    return int(unrev.sum())


def analyze_ncs_file(ncs_path: Path) -> FileResult:
    print(f'\n=== {ncs_path} ===')
    samples_uV, fs_native, anchor_idx, anchor_t = load_ncs(ncs_path)
    clean, t = preprocess_lfp(samples_uV, fs_native, anchor_idx, anchor_t)
    del samples_uV
    n = len(clean)
    nperseg = int(round(FS * ACG_EPOCH_SEC))
    n_epochs = n // nperseg

    if USE_ACG_THETA_EPOCHS:
        epoch_ok, epoch_ed_min = acg_theta_epoch_mask(clean)
        theta_ok_sample = np.zeros(n, dtype=bool)
        theta_ok_sample[:n_epochs * nperseg] = np.repeat(epoch_ok, nperseg)
        print(f'  ACG theta-positive epochs (ED_min < {ACG_ED_MIN_THRESH:g}): '
              f'{int(epoch_ok.sum())}/{len(epoch_ok)}')
    else:
        epoch_ok = np.ones(n_epochs, dtype=bool)
        epoch_ed_min = np.full(n_epochs, np.nan)
        theta_ok_sample = np.ones(n, dtype=bool)
    epoch_edges_rel_s = t[np.minimum(np.arange(n_epochs + 1) * nperseg, n - 1)] - t[0]

    theta = bandpass_filter(clean, THETA_BAND[0], THETA_BAND[1], FS, THETA_FILTER_ORDER)
    inst_freq, xa = gp_instantaneous_frequency(analytic_signal(theta))
    phase, env = np.angle(xa), np.abs(xa)

    events = detect_phase_reversals(inst_freq, env, theta_ok_sample, epoch_ed_min, t)
    print(f'  Phase reversals (IF < {REVERSAL_IF_THRESH_HZ:g} Hz, theta-positive): {len(events)}')

    nev_path = find_nev_file(ncs_path.parent)
    zero_rel_s = np.zeros(0)
    if nev_path is not None:
        nev = load_nev_events(nev_path)
        is_zero = nev['event_string'].str.lower().str.contains(ZERO_EVENT_TEXT.lower(), regex=False)
        zero_rel_s = nev.loc[is_zero, 'time_s'].to_numpy() - t[0]
        print(f'  {nev_path.name}: {len(nev)} events, {len(zero_rel_s)} containing "{ZERO_EVENT_TEXT}"')
    else:
        print('  No .nev file in this folder -- no zero-event markers.')

    res = FileResult(ncs_path=ncs_path, nev_path=nev_path, t=t, clean=clean, theta=theta,
                     phase=phase, env=env, inst_freq=inst_freq, theta_ok_sample=theta_ok_sample,
                     epoch_ok=epoch_ok, epoch_ed_min=epoch_ed_min,
                     epoch_edges_rel_s=epoch_edges_rel_s, events=events, zero_rel_s=zero_rel_s)
    res.n_restored = _restore_previous_statuses(events, output_paths(ncs_path)['csv'])
    if res.n_restored:
        print(f'  Restored {res.n_restored} earlier review decisions.')
    return res


def save_events_csv(res: FileResult):
    p = output_paths(res.ncs_path)
    p['dir'].mkdir(parents=True, exist_ok=True)
    res.events.to_csv(p['csv'], index=False)


def save_file_outputs(res: FileResult, reviewed: bool):
    """Events CSV, file-info JSON, and the occurrence plot (reviewed or not).
    Once reviewed, also the KDE of the accepted reversals and its peak test."""
    p = output_paths(res.ncs_path)
    save_events_csv(res)
    kde = reversal_kde_analysis(res) if (reviewed and RUN_KDE) else None
    info = dict(
        ncs_path=str(res.ncs_path),
        nev_file=res.nev_path.name if res.nev_path else None,
        lfp_start_s=res.t0,
        recording_dur_min=res.recording_dur_s / 60,
        theta_positive_min=res.theta_positive_s / 60,
        theta_positive_frac=res.theta_positive_s / res.recording_dur_s if res.recording_dur_s > 0 else np.nan,
        n_acg_epochs=int(len(res.epoch_ok)),
        n_theta_epochs=int(res.epoch_ok.sum()),
        n_zero_events=int(len(res.zero_rel_s)),
        zero_event_rel_min=[round(float(z) / 60, 4) for z in res.zero_rel_s],
        analysis_params=analysis_params(),
    )
    if kde is not None:
        pk = kde['peaks']
        info.update(kde_n_events=kde['n_events'],
                    kde_baseline_per_theta_min=kde['baseline'],
                    kde_global_thresh_z=kde['z_thresh'],
                    kde_n_sig_peaks=len(pk),
                    kde_sig_peak_min=[round(float(m), 3) for m in pk['peak_min']],
                    kde_sig_peak_p=[float(v) for v in pk['p_global']])
        pd.DataFrame({'time_s': kde['grid_s'],
                      'event_intensity_per_s': kde['intensity'],
                      'theta_occupancy_frac': kde['occupancy'],
                      'rate_per_theta_min': kde['rate'],
                      'null_lo_per_theta_min': kde['null_lo'],
                      'null_hi_per_theta_min': kde['null_hi'],
                      'null_mean_per_theta_min': kde['null_mean'],
                      'null_sd_per_theta_min': kde['null_sd'],
                      'z': (kde['rate'] - kde['null_mean']) / kde['null_sd'],
                      'global_thresh_per_theta_min': kde['thresh_curve'],
                      'valid': kde['valid']}).to_csv(p['kde_csv'], index=False)
        pk.to_csv(p['kde_peaks'], index=False)
        plot_reversal_kde(res, kde, p['kde_plot'])
    with open(p['info'], 'w') as fh:
        json.dump(info, fh, indent=2, default=str)
    plot_reversal_occurrence(res, p['plot_reviewed'] if reviewed else p['plot'], reviewed)


# ============================================================================
# Occurrence plot
# ============================================================================

def _shade_spans(ax, spans, **kw):
    """Full-height shaded x-spans as one collection (fast for thousands of spans)."""
    if len(spans) == 0:
        return
    verts = [[(a, 0), (a, 1), (b, 1), (b, 0)] for a, b in spans]
    ax.add_collection(PolyCollection(verts, transform=ax.get_xaxis_transform(), **kw), autolim=False)


def plot_reversal_occurrence(res: FileResult, out_path: Path, reviewed: bool):
    ev = res.events
    dur_s = res.recording_dur_s
    t_ev = ev['center_rel_s'].to_numpy(dtype=float)
    status = ev['status'].to_numpy()

    fig = Figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.6, 0.9], hspace=0.12)
    ax_r = fig.add_subplot(gs[0])
    ax_rate = fig.add_subplot(gs[1], sharex=ax_r)
    ax_th = fig.add_subplot(gs[2], sharex=ax_r)

    spans_min = res.theta_spans_rel_s() / 60
    for ax in (ax_r, ax_rate, ax_th):
        _shade_spans(ax, spans_min, facecolor=THETA_SHADE, alpha=0.18, linewidth=0)

    handles = []
    if reviewed:
        for st, col in STATUS_COLORS.items():
            sel = status == st
            ax_r.vlines(t_ev[sel] / 60, 0, 1, colors=col, linewidth=0.7)
            handles.append(Line2D([0], [0], color=col, lw=2, label=f'{st} (n={int(sel.sum())})'))
    else:
        ax_r.vlines(t_ev / 60, 0, 1, colors='k', linewidth=0.6)
        handles.append(Line2D([0], [0], color='k', lw=2, label=f'detected (n={len(ev)})'))
    ax_r.set_ylim(0, 1)
    ax_r.set_yticks([])
    ax_r.set_ylabel('Phase\nreversals')

    edges = np.arange(0, dur_s + RATE_BIN_S, RATE_BIN_S)
    starts_rel = res.epoch_edges_rel_s[:-1]
    theta_s = np.histogram(starts_rel[res.epoch_ok], bins=edges)[0] * ACG_EPOCH_SEC

    def rate(sel):
        cnt = np.histogram(t_ev[sel], bins=edges)[0]
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(theta_s > 0, cnt / theta_s * 60, np.nan)

    ax_rate.stairs(rate(np.ones(len(ev), dtype=bool)), edges / 60,
                   color='0.45' if reviewed else 'k', linewidth=1.2, label='all detected')
    if reviewed and (status == 'accepted').any():
        ax_rate.stairs(rate(status == 'accepted'), edges / 60, color=STATUS_COLORS['accepted'],
                       linewidth=1.6, label='accepted')
    ax_rate.set_ylabel('Reversals per min\nof theta-positive LFP')
    ax_rate.set_ylim(bottom=0)
    ax_rate.legend(loc='upper left', bbox_to_anchor=(1.005, 1), fontsize=8, frameon=False)

    bin_w = np.minimum(edges[1:], dur_s) - edges[:-1]
    with np.errstate(invalid='ignore', divide='ignore'):
        ax_th.stairs(np.clip(theta_s / bin_w, 0, 1), edges / 60, color='#2e7d32', fill=True, alpha=0.5)
    ax_th.set_ylim(0, 1.05)
    ax_th.set_ylabel('Theta-positive\nfraction')
    ax_th.set_xlabel('Recording time (min)')

    for ax in (ax_r, ax_rate, ax_th):
        for z in res.zero_rel_s / 60:
            ax.axvline(z, color=ZERO_COLOR, linewidth=1.3, zorder=5)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in (ax_r, ax_rate):
        ax.tick_params(labelbottom=False)
    ax_r.set_xlim(0, dur_s / 60)

    handles.append(Patch(facecolor=THETA_SHADE, alpha=0.4, label='ACG theta-positive epochs'))
    handles.append(Line2D([0], [0], color=ZERO_COLOR, lw=1.5,
                          label=f'"{ZERO_EVENT_TEXT}" event (n={len(res.zero_rel_s)})'))
    ax_r.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.005, 1), fontsize=8, frameon=False)

    n_acc = int((status == 'accepted').sum())
    title = (f'{res.ncs_path.parent.name} / {res.ncs_path.name} -- {len(ev)} phase reversals '
             f'(IF < {REVERSAL_IF_THRESH_HZ:g} Hz, theta-positive)')
    if reviewed:
        title += f', {n_acc} accepted'
    title += (f'\nTheta-positive: {res.theta_positive_s / 60:.1f} of {dur_s / 60:.1f} min'
              + (f' | detected rate {len(ev) / (res.theta_positive_s / 60):.2f} /min theta'
                 if res.theta_positive_s > 0 else ''))
    ax_r.set_title(title, fontsize=10)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')


# ============================================================================
# Gaussian KDE of reversal times + Monte-Carlo peak significance
# ============================================================================

def kde_intensity(times, grid, bw_s, weight=1.0):
    """scipy.stats.gaussian_kde of `times` with kernel SD bw_s (s), scaled from a
    probability density to an intensity: sum_i weight * N(grid; times_i, bw_s)."""
    times = np.asarray(times, dtype=float)
    if len(times) == 0:
        return np.zeros(len(grid))
    sd = np.std(times, ddof=1) if len(times) > 1 else 0.0   # gaussian_kde's own (ddof=1) SD
    if sd <= 0:   # gaussian_kde needs spread; all points at one time -> one kernel
        dens = np.exp(-0.5 * ((grid - times[0]) / bw_s) ** 2) / (bw_s * np.sqrt(2 * np.pi))
    else:
        dens = stats.gaussian_kde(times, bw_method=bw_s / sd)(grid)
    return weight * len(times) * dens


class ThetaClock:
    """Maps recording time (s from start) <-> cumulative theta-positive time, over
    the ACG theta-positive epochs [lo, hi), so a null can move events within theta
    time only."""

    def __init__(self, lo, hi):
        self.lo, self.len = lo, hi - lo
        self.cum = np.concatenate(([0.0], np.cumsum(self.len)))
        self.total = float(self.cum[-1])

    def to_theta(self, t):
        k = np.clip(np.searchsorted(self.lo, t, 'right') - 1, 0, len(self.lo) - 1)
        return self.cum[k] + np.clip(t - self.lo[k], 0, self.len[k])

    def from_theta(self, tau):
        k = np.clip(np.searchsorted(self.cum, tau, 'right') - 1, 0, len(self.lo) - 1)
        return self.lo[k] + (tau - self.cum[k])


def _null_theta_times(tau_sorted, total, rng):
    """One null surrogate of the event times, in theta time."""
    if KDE_NULL == 'uniform':
        return rng.uniform(0.0, total, len(tau_sorted))
    gaps = np.diff(np.concatenate(([0.0], tau_sorted, [total])))   # N + 1 intervals
    return np.cumsum(rng.permutation(gaps))[:-1]


def reversal_kde_analysis(res: FileResult):
    """KDE rate of the accepted reversals and its significant peaks (see step 7 of
    the module docstring). Returns None if there are fewer than KDE_MIN_EVENTS
    accepted events or no usable theta time.

    rate(t) = 60 * event_KDE(t) / theta_occupancy_KDE(t)   [reversals / min theta]
    Studentized global envelope (Myllymaki et al. 2017, J R Stat Soc B 79:381):
    z(t) = (rate(t) - null mean(t)) / null SD(t), so theta-poor stretches (wide
    null) do not inflate the threshold for the rest of the recording.
    z_thresh = (1 - KDE_ALPHA) quantile of max_t z_null(t) over the shuffles.
    Significant peak = maximum z of each run where z > z_thresh; p_global =
    (1 + #{null max z >= peak z}) / (1 + KDE_N_SHUFFLES), i.e. already corrected
    for all time points."""
    if KDE_NULL not in ('isi', 'uniform'):
        raise ValueError(f"KDE_NULL must be 'isi' or 'uniform', got {KDE_NULL!r}")
    ev = res.events
    t_ev = np.sort(ev.loc[ev['status'] == 'accepted', 'center_rel_s'].to_numpy(dtype=float))
    if len(t_ev) < KDE_MIN_EVENTS or not res.epoch_ok.any():
        print(f'  KDE skipped: {len(t_ev)} accepted events (need >= {KDE_MIN_EVENTS}).')
        return None

    ok = np.flatnonzero(res.epoch_ok)
    lo, hi = res.epoch_edges_rel_s[ok], res.epoch_edges_rel_s[ok + 1]
    clock = ThetaClock(lo, hi)
    grid = np.arange(0.0, res.recording_dur_s + KDE_GRID_S / 2, KDE_GRID_S)
    occ = kde_intensity((lo + hi) / 2, grid, KDE_BANDWIDTH_S, weight=ACG_EPOCH_SEC)
    valid = occ >= KDE_MIN_THETA_FRAC
    if not valid.any():
        print(f'  KDE skipped: theta-positive fraction never reaches {KDE_MIN_THETA_FRAC:g}.')
        return None

    def rate(times):
        return 60.0 * kde_intensity(times, grid[valid], KDE_BANDWIDTH_S) / occ[valid]

    intensity = kde_intensity(t_ev, grid, KDE_BANDWIDTH_S)
    obs = np.where(valid, 60.0 * intensity / np.maximum(occ, 1e-12), np.nan)

    rng = np.random.default_rng(KDE_SEED)
    tau = clock.to_theta(t_ev)
    null = np.empty((KDE_N_SHUFFLES, int(valid.sum())), dtype=np.float32)
    for i in range(KDE_N_SHUFFLES):
        null[i] = rate(clock.from_theta(_null_theta_times(tau, clock.total, rng)))
    null_lo, null_hi = np.full(len(grid), np.nan), np.full(len(grid), np.nan)
    null_lo[valid], null_hi[valid] = np.quantile(null, [KDE_ALPHA / 2, 1 - KDE_ALPHA / 2], axis=0)
    mu = np.full(len(grid), np.nan)
    sd = np.full(len(grid), np.nan)
    mu[valid] = null.mean(axis=0)
    sd[valid] = np.maximum(null.std(axis=0), 1e-9)
    null -= mu[valid]
    null /= sd[valid]
    null_max_z = null.max(axis=1)
    del null
    z_thresh = float(np.quantile(null_max_z, 1 - KDE_ALPHA))
    z_obs = (obs - mu) / sd
    thresh_curve = mu + z_thresh * sd
    baseline = len(t_ev) / clock.total * 60.0

    rows = []
    starts, stops = _true_runs(valid & (z_obs > z_thresh))
    for a, b in zip(starts, stops):
        k = a + int(np.argmax(z_obs[a:b]))
        h, z = float(obs[k]), float(z_obs[k])
        if len(res.zero_rel_s):
            d = grid[k] - res.zero_rel_s
            zero_off = float(d[np.argmin(np.abs(d))])
        else:
            zero_off = np.nan
        rows.append(dict(
            peak_s=grid[k], peak_min=grid[k] / 60,
            segment_start_s=grid[a], segment_stop_s=grid[b - 1],
            peak_rate_per_theta_min=h, fold_over_baseline=h / baseline,
            peak_z=z, p_global=(1 + int(np.sum(null_max_z >= z))) / (1 + KDE_N_SHUFFLES),
            n_events_in_segment=int(np.sum((t_ev >= grid[a]) & (t_ev <= grid[b - 1]))),
            theta_occupancy_at_peak=float(occ[k]),
            peak_minus_nearest_zero_s=zero_off))
    peaks = pd.DataFrame(rows, columns=[
        'peak_s', 'peak_min', 'segment_start_s', 'segment_stop_s', 'peak_rate_per_theta_min',
        'fold_over_baseline', 'peak_z', 'p_global', 'n_events_in_segment', 'theta_occupancy_at_peak',
        'peak_minus_nearest_zero_s'])
    peaks.insert(0, 'peak_id', np.arange(1, len(peaks) + 1))
    print(f'  KDE: {len(t_ev)} accepted events, baseline {baseline:.2f}/min theta, global '
          f'{100 * (1 - KDE_ALPHA):g}% threshold z = {z_thresh:.2f} '
          f'({KDE_NULL} null x {KDE_N_SHUFFLES}) -> {len(peaks)} significant peak(s)')
    return dict(event_s=t_ev, n_events=len(t_ev), grid_s=grid, intensity=intensity,
                occupancy=occ, rate=obs, valid=valid, null_lo=null_lo, null_hi=null_hi,
                null_mean=mu, null_sd=sd, thresh_curve=thresh_curve,
                baseline=baseline, z_thresh=z_thresh, peaks=peaks)


def plot_reversal_kde(res: FileResult, kde: dict, out_path: Path):
    gm = kde['grid_s'] / 60
    peaks = kde['peaks']
    pk_col = '#d95f02'

    fig = Figure(figsize=(14, 7))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.7, 2.2, 0.9], hspace=0.12)
    ax_r = fig.add_subplot(gs[0])
    ax_k = fig.add_subplot(gs[1], sharex=ax_r)
    ax_o = fig.add_subplot(gs[2], sharex=ax_r)

    spans_min = res.theta_spans_rel_s() / 60
    for ax in (ax_r, ax_k, ax_o):
        _shade_spans(ax, spans_min, facecolor=THETA_SHADE, alpha=0.18, linewidth=0)
        for _, p in peaks.iterrows():
            ax.axvspan(p['segment_start_s'] / 60, p['segment_stop_s'] / 60, color=pk_col, alpha=0.15, lw=0)

    ax_r.vlines(kde['event_s'] / 60, 0, 1, colors=STATUS_COLORS['accepted'], linewidth=0.7)
    ax_r.set_ylim(0, 1)
    ax_r.set_yticks([])
    ax_r.set_ylabel('Accepted\nreversals')

    ax_k.fill_between(gm, kde['null_lo'], kde['null_hi'], color='0.6', alpha=0.35, lw=0,
                      label=f'null {100 * (1 - KDE_ALPHA):g}% pointwise band')
    ax_k.axhline(kde['baseline'], color='0.3', ls='--', lw=1,
                 label=f'baseline {kde["baseline"]:.2f}/min theta')
    ax_k.plot(gm, kde['thresh_curve'], color=pk_col, lw=1.2,
              label=f'global threshold, null mean + {kde["z_thresh"]:.2f} SD (FWER {KDE_ALPHA:g})')
    ax_k.plot(gm, kde['rate'], color='k', lw=1.4, label=f'KDE rate (kernel SD {KDE_BANDWIDTH_S:g} s)')
    for _, p in peaks.iterrows():
        ax_k.plot(p['peak_min'], p['peak_rate_per_theta_min'], 'v', color=pk_col, ms=8)
        ax_k.annotate(f'p={p["p_global"]:.3g}', (p['peak_min'], p['peak_rate_per_theta_min']),
                      xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8, color=pk_col)
    ax_k.set_ylim(bottom=0)
    ax_k.set_ylabel('Reversals per min\nof theta-positive LFP')
    ax_k.legend(loc='upper left', bbox_to_anchor=(1.005, 1), fontsize=8, frameon=False)

    ax_o.plot(gm, kde['occupancy'], color='#2e7d32', lw=1.2)
    ax_o.axhline(KDE_MIN_THETA_FRAC, color='0.4', ls=':', lw=1)
    ax_o.set_ylim(0, 1.05)
    ax_o.set_ylabel('Theta-positive\nfraction (KDE)')
    ax_o.set_xlabel('Recording time (min)')

    for ax in (ax_r, ax_k, ax_o):
        for z in res.zero_rel_s / 60:
            ax.axvline(z, color=ZERO_COLOR, linewidth=1.3, zorder=5)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in (ax_r, ax_k):
        ax.tick_params(labelbottom=False)
    ax_r.set_xlim(0, res.recording_dur_s / 60)
    ax_r.set_title(f'{res.ncs_path.parent.name} / {res.ncs_path.name} -- Gaussian KDE of '
                   f'{kde["n_events"]} accepted reversals, null: {KDE_NULL} x {KDE_N_SHUFFLES}\n'
                   f'{len(peaks)} peak(s) above the global {100 * (1 - KDE_ALPHA):g}% studentized null maximum '
                   f'(shaded orange); red = "{ZERO_EVENT_TEXT}" events', fontsize=10)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')


# ============================================================================
# Summary workbook
# ============================================================================

def find_ncs_files(root: Path) -> list[Path]:
    return sorted(root.rglob('*.ncs'), key=_natural_key)


def write_summary(ncs_files: list[Path]):
    rows = []
    for ncs in ncs_files:
        p = output_paths(ncs)
        row = dict(Folder=str(ncs.parent), ncs_file=ncs.name)
        if p['info'].exists():
            with open(p['info']) as fh:
                info = json.load(fh)
            row.update({k: info.get(k) for k in ('nev_file', 'recording_dur_min', 'theta_positive_min',
                                                 'theta_positive_frac', 'n_zero_events', 'kde_n_events',
                                                 'kde_baseline_per_theta_min',
                                                 'kde_global_thresh_z', 'kde_n_sig_peaks')})
            row['kde_sig_peak_min'] = '; '.join(f'{m:g}' for m in info.get('kde_sig_peak_min', []))
            row['kde_sig_peak_p'] = '; '.join(f'{v:.3g}' for v in info.get('kde_sig_peak_p', []))
        if p['csv'].exists():
            ev = pd.read_csv(p['csv'])
            st = ev['status'] if 'status' in ev else pd.Series(dtype=str)
            dec = ev['decision'] if 'decision' in ev else pd.Series(dtype=str)
            n_acc, n_rej = int((st == 'accepted').sum()), int((st == 'rejected').sum())
            n_un = int((st == 'unreviewed').sum())
            n_auto = int((dec == 'auto').sum())
            theta_min = row.get('theta_positive_min') or np.nan
            if not len(ev):
                review_status = 'no events'
            elif n_un == 0 and n_auto == 0:
                review_status = 'fully reviewed'
            elif n_un == 0:
                review_status = ('accepted without review' if n_auto == len(ev)
                                 else 'partially reviewed, rest accepted without review')
            elif n_acc + n_rej:
                review_status = 'partially reviewed'
            else:
                review_status = 'not reviewed'
            row.update(n_detected=len(ev), n_accepted=n_acc, n_auto_accepted=n_auto,
                       n_rejected=n_rej, n_unreviewed=n_un,
                       detected_per_theta_min=len(ev) / theta_min if theta_min else np.nan,
                       accepted_per_theta_min=n_acc / theta_min if theta_min else np.nan,
                       review_status=review_status)
        else:
            row['review_status'] = 'not processed (skipped)'
        rows.append(row)

    cols = ['Folder', 'ncs_file', 'nev_file', 'recording_dur_min', 'theta_positive_min',
            'theta_positive_frac', 'n_zero_events', 'n_detected', 'n_accepted', 'n_auto_accepted', 'n_rejected',
            'n_unreviewed', 'detected_per_theta_min', 'accepted_per_theta_min', 'review_status',
            'kde_n_events', 'kde_baseline_per_theta_min', 'kde_global_thresh_z',
            'kde_n_sig_peaks', 'kde_sig_peak_min', 'kde_sig_peak_p']
    df = pd.DataFrame(rows, columns=cols)
    params = pd.DataFrame([(k, repr(v)) for k, v in analysis_params().items()],
                          columns=['Parameter', 'Value'])
    params = pd.concat([pd.DataFrame([('CodeName', Path(__file__).name),
                                      ('RunTimestamp', pd.Timestamp.now().isoformat(timespec='seconds'))],
                                     columns=['Parameter', 'Value']), params], ignore_index=True)
    out = ROOT_FOLDER / SUMMARY_EXCEL_NAME
    try:
        with pd.ExcelWriter(out, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Summary', index=False)
            params.to_excel(writer, sheet_name='Parameters', index=False)
    except PermissionError:
        out = out.with_name(f'{out.stem}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.xlsx')
        with pd.ExcelWriter(out, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Summary', index=False)
            params.to_excel(writer, sheet_name='Parameters', index=False)
    print(f'\nSummary written to {out}')
    return out


# ============================================================================
# Review GUI
# ============================================================================

class ReversalReviewGUI:
    """Reviews already-detected files one at a time: a prompt screen
    (review / skip / quit), then an event-by-event review screen (accept /
    reject). Decisions are written to the file's CSV after every click.

    ncs_files: every .ncs file (for the summary workbook).
    results: detection results of the files that have events, in review order;
             each is dropped from memory once its file is finished or skipped."""

    def __init__(self, ncs_files: list[Path], results: dict[Path, FileResult]):
        self.ncs_files = list(ncs_files)
        self.results = dict(results)
        self.review_files = list(results)
        self.file_idx = -1
        self.res: FileResult | None = None
        self.ev_idx = 0
        self.half_win = REVIEW_HALF_WINDOW_S
        self.mode = 'prompt'

        self.root = tk.Tk()
        self.root.title('Theta phase-reversal review')
        self.root.geometry('1450x980')
        self.root.protocol('WM_DELETE_WINDOW', self.quit_app)
        self.root.bind('<KeyPress>', self._on_key)
        self._build_prompt()
        self._build_review()
        self.next_file()

    def run(self):
        self.root.mainloop()

    # ---------------- layout ----------------
    def _build_prompt(self):
        f = self.prompt_frame = ttk.Frame(self.root, padding=40)
        self.prompt_title = ttk.Label(f, font=('Segoe UI', 15, 'bold'))
        self.prompt_title.pack(anchor='w')
        self.prompt_info = ttk.Label(f, font=('Segoe UI', 11), justify='left')
        self.prompt_info.pack(anchor='w', pady=(12, 20))
        bf = ttk.Frame(f)
        bf.pack(anchor='w')
        tk.Button(bf, text='Review this file', width=26, bg='#cfe8ff',
                  command=self.review_current).pack(side='left', padx=4)
        tk.Button(bf, text='Skip this file', width=16, command=self.skip_file).pack(side='left', padx=4)
        tk.Button(bf, text='Quit', width=10, command=self.quit_app).pack(side='left', padx=4)
        self.prompt_status = ttk.Label(f, font=('Segoe UI', 11, 'italic'), foreground='#a33')
        self.prompt_status.pack(anchor='w', pady=(20, 0))

    def _build_review(self):
        f = self.review_frame = ttk.Frame(self.root)
        self.info_var = tk.StringVar()
        ttk.Label(f, textvariable=self.info_var, font=('Consolas', 10), justify='left',
                  padding=(8, 6)).pack(side='top', fill='x')

        bar = ttk.Frame(f, padding=(6, 4))
        bar.pack(side='bottom', fill='x')

        def btn(text, cmd, bg=None, width=None, side='left'):
            b = tk.Button(bar, text=text, command=cmd, takefocus=0)
            if bg:
                b.configure(bg=bg)
            if width:
                b.configure(width=width)
            b.pack(side=side, padx=3)

        btn('◀ Prev (←)', self.prev_event)
        btn('Accept (A)', self.accept, bg='#b9e6c9', width=12)
        btn('Reject (R)', self.reject, bg='#f5bcbc', width=12)
        btn('Next ▶ (→)', self.next_event)
        btn('Next unreviewed (U)', self.next_unreviewed)
        btn('Zoom in (+)', lambda: self.zoom(0.5))
        btn('Zoom out (−)', lambda: self.zoom(2.0))
        self.status_var = tk.StringVar()
        ttk.Label(bar, textvariable=self.status_var, foreground='#a33').pack(side='left', padx=12)
        btn('Quit (saves)', self.quit_app, side='right')
        btn('Finish file → next (F)', self.finish_file, bg='#cfe8ff', side='right')

        tb_frame = ttk.Frame(f)
        tb_frame.pack(side='bottom', fill='x')

        self.fig = Figure(figsize=(13, 8), dpi=100)
        outer = self.fig.add_gridspec(2, 1, height_ratios=[7, 1.1], hspace=0.32,
                                      left=0.07, right=0.98, top=0.95, bottom=0.07)
        gs = outer[0].subgridspec(3, 1, height_ratios=[3, 2, 2], hspace=0.12)
        self.ax_lfp = self.fig.add_subplot(gs[0])
        self.ax_ph = self.fig.add_subplot(gs[1], sharex=self.ax_lfp)
        self.ax_if = self.fig.add_subplot(gs[2], sharex=self.ax_lfp)
        self.ax_ov = self.fig.add_subplot(outer[1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=f)
        self.toolbar = NavigationToolbar2Tk(self.canvas, tb_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side='left')
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        self.canvas.mpl_connect('button_press_event', self._on_click)

    # ---------------- file flow ----------------
    def next_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.review_files):
            self.finish_all()
            return
        self._show_prompt()

    def _show_prompt(self):
        self.mode = 'prompt'
        self.review_frame.pack_forget()
        self.prompt_frame.pack(fill='both', expand=True)
        ncs = self.review_files[self.file_idx]
        res = self.results[ncs]
        n_files = len(self.review_files)
        self.root.title(f'Theta phase-reversal review -- file {self.file_idx + 1}/{n_files}')
        self.prompt_title.config(text=f'File {self.file_idx + 1} of {n_files}:  {ncs.name}')
        st = res.events['status']
        lines = [f'Folder:  {ncs.parent}',
                 f'.nev file:  {res.nev_path.name if res.nev_path else "none found (no zero-event markers)"}',
                 f'Detected phase reversals:  {len(st)}']
        if res.n_restored:
            lines.append(f'Restored from an earlier review: {int((st == "accepted").sum())} accepted, '
                         f'{int((st == "rejected").sum())} rejected, '
                         f'{int((st == "unreviewed").sum())} unreviewed')
        self.prompt_info.config(text='\n'.join(lines))
        self.prompt_status.config(text='')

    def review_current(self):
        res = self.results[self.review_files[self.file_idx]]
        self.res = res
        unrev = np.flatnonzero(res.events['status'].to_numpy() == 'unreviewed')
        self.ev_idx = int(unrev[0]) if len(unrev) else 0
        self.mode = 'review'
        self.prompt_frame.pack_forget()
        self.review_frame.pack(fill='both', expand=True)
        self._draw_event()

    def skip_file(self):
        ncs = self.review_files[self.file_idx]
        print(f'Skipped review of {ncs} (events stay unreviewed)')
        self.results.pop(ncs, None)
        self.next_file()

    def finish_file(self):
        if self.res is None:
            return
        n_un = int((self.res.events['status'] == 'unreviewed').sum())
        if n_un and not messagebox.askyesno(
                'Unreviewed events', f'{n_un} events are still unreviewed (they stay "unreviewed").\n'
                                     f'Finish this file and move on?'):
            return
        save_file_outputs(self.res, reviewed=True)
        self.results.pop(self.res.ncs_path, None)
        self.res = None
        self.next_file()

    def finish_all(self):
        self.mode = 'done'
        out = write_summary(self.ncs_files)
        messagebox.showinfo('Done', f'All files handled.\nSummary: {out}')
        self.root.destroy()

    def quit_app(self):
        try:
            if self.res is not None:
                save_file_outputs(self.res, reviewed=True)
            write_summary(self.ncs_files)
        except Exception:
            traceback.print_exc()
        self.root.destroy()

    # ---------------- event navigation / decisions ----------------
    def _set_status(self, status):
        self.res.events.loc[self.ev_idx, 'status'] = status
        self.res.events.loc[self.ev_idx, 'decision'] = 'manual'
        save_events_csv(self.res)
        self.next_unreviewed(after_decision=True)

    def accept(self):
        if self.res is not None:
            self._set_status('accepted')

    def reject(self):
        if self.res is not None:
            self._set_status('rejected')

    def prev_event(self):
        if self.res is not None and self.ev_idx > 0:
            self.ev_idx -= 1
            self._draw_event()

    def next_event(self):
        if self.res is not None and self.ev_idx < len(self.res.events) - 1:
            self.ev_idx += 1
            self._draw_event()

    def next_unreviewed(self, after_decision=False):
        if self.res is None:
            return
        st = self.res.events['status'].to_numpy()
        n = len(st)
        order = np.r_[self.ev_idx + 1:n, 0:self.ev_idx + 1]
        cand = order[st[order] == 'unreviewed']
        if len(cand):
            self.ev_idx = int(cand[0])
            self.status_var.set('')
        else:
            if after_decision and self.ev_idx < n - 1:
                self.ev_idx += 1
            self.status_var.set('All events reviewed -- press "Finish file" to continue.')
        self._draw_event()

    def zoom(self, factor):
        if self.res is not None:
            self.half_win = float(np.clip(self.half_win * factor, 0.125, 30.0))
            self._draw_event()

    def _on_key(self, event):
        if self.mode != 'review':
            return
        k = event.keysym
        if k in ('a', 'A'):
            self.accept()
        elif k in ('r', 'R'):
            self.reject()
        elif k == 'Left':
            self.prev_event()
        elif k == 'Right':
            self.next_event()
        elif k in ('u', 'U'):
            self.next_unreviewed()
        elif k in ('plus', 'equal', 'KP_Add'):
            self.zoom(0.5)
        elif k in ('minus', 'underscore', 'KP_Subtract'):
            self.zoom(2.0)
        elif k in ('f', 'F'):
            self.finish_file()

    def _on_click(self, event):
        """Click in the overview strip jumps to the nearest event."""
        if (self.res is None or event.inaxes is not self.ax_ov or event.xdata is None
                or str(self.toolbar.mode) != ''):
            return
        centers = self.res.events['center_rel_s'].to_numpy() / 60
        self.ev_idx = int(np.argmin(np.abs(centers - event.xdata)))
        self._draw_event()

    # ---------------- drawing ----------------
    def _draw_event(self):
        res, ev_df = self.res, self.res.events
        ev = ev_df.iloc[self.ev_idx]
        n = len(res.clean)
        s, e = int(ev['start_idx']), int(ev['stop_idx'])
        c = (s + e) // 2
        hw = int(round(self.half_win * FS))
        i0, i1 = max(0, c - hw), min(n, c + hw + 1)
        tt = (np.arange(i0, i1) - c) / FS

        for ax in (self.ax_lfp, self.ax_ph, self.ax_if):
            ax.clear()

        # non-theta epochs and other events inside the window
        nperseg = int(round(FS * ACG_EPOCH_SEC))
        for ep in range(i0 // nperseg, (i1 - 1) // nperseg + 1):
            if ep >= len(res.epoch_ok) or not res.epoch_ok[ep]:
                a, b = (ep * nperseg - c) / FS, ((ep + 1) * nperseg - c) / FS
                for ax in (self.ax_lfp, self.ax_ph, self.ax_if):
                    ax.axvspan(a, b, color='0.85', alpha=0.5, lw=0, zorder=0)
        near = np.flatnonzero((ev_df['stop_idx'].to_numpy() >= i0) & (ev_df['start_idx'].to_numpy() <= i1))
        for j in near:
            if j == self.ev_idx:
                continue
            a = (ev_df.at[j, 'start_idx'] - c) / FS
            b = (ev_df.at[j, 'stop_idx'] - c) / FS
            col = STATUS_COLORS.get(ev_df.at[j, 'status'], '#333333')
            for ax in (self.ax_lfp, self.ax_ph, self.ax_if):
                ax.axvspan(a, b, color=col, alpha=0.25, lw=0, zorder=1)
        ev_a, ev_b = (s - c) / FS, (e - c) / FS
        gp_b = (int(ev['gp_ext_stop_idx']) - c) / FS
        for ax in (self.ax_lfp, self.ax_ph, self.ax_if):
            ax.axvspan(ev_a, max(ev_b, ev_a + 1 / FS), color='red', alpha=0.25, lw=0, zorder=1)
            ax.axvspan(ev_b, gp_b, color='orange', alpha=0.12, lw=0, zorder=1)
            for z in res.zero_rel_s:
                zt = (z - (res.t[c] - res.t0))
                if tt[0] <= zt <= tt[-1]:
                    ax.axvline(zt, color=ZERO_COLOR, lw=1.5, zorder=4)

        ax = self.ax_lfp
        ax.plot(tt, res.clean[i0:i1], color='0.55', lw=0.7, label=f'cleaned LFP (≤{LOWPASS_CUTOFF_HZ:g} Hz)')
        ax.plot(tt, res.theta[i0:i1], color='C0', lw=1.4, label=f'theta {THETA_BAND[0]:g}–{THETA_BAND[1]:g} Hz')
        ax.plot(tt, res.env[i0:i1], color='C0', lw=0.7, ls='--', label='Hilbert envelope')
        ax.plot(tt, -res.env[i0:i1], color='C0', lw=0.7, ls='--')
        ylim = 1.1 * max(np.abs(res.clean[i0:i1]).max(), res.env[i0:i1].max(), 1e-6)
        ax.set_ylim(-ylim, ylim * 1.25)   # headroom for the legend
        ax.set_ylabel('µV')
        ax.legend(loc='upper right', fontsize=8, ncol=3, frameon=False)
        status = ev['status']
        ax.set_title(f'Event {self.ev_idx + 1}/{len(ev_df)}  —  {status.upper()}',
                     color={'accepted': STATUS_COLORS['accepted'], 'rejected': '#b22222'}.get(status, 'k'),
                     fontsize=11, fontweight='bold')

        ax = self.ax_ph
        ph = np.rad2deg(res.phase[i0:i1])
        neg = res.inst_freq[i0:i1] < REVERSAL_IF_THRESH_HZ
        ax.plot(tt[~neg], ph[~neg], '.', color='k', ms=1.8)
        ax.plot(tt[neg], ph[neg], '.', color='red', ms=3.5)
        ax.set_ylim(-190, 190)
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.set_ylabel('Phase (°)')

        ax = self.ax_if
        ax.plot(tt, np.clip(res.inst_freq[i0:i1], *IF_PLOT_LIMS), color='k', lw=0.8)
        ax.axhline(0, color='k', lw=0.6)
        for fb in THETA_BAND:
            ax.axhline(fb, color='0.5', lw=0.7, ls='--')
        if REVERSAL_IF_THRESH_HZ != 0:
            ax.axhline(REVERSAL_IF_THRESH_HZ, color='red', lw=0.7, ls=':')
        ax.set_ylim(*IF_PLOT_LIMS)
        ax.set_ylabel('IF (Hz)')
        self.fig.align_ylabels([self.ax_lfp, self.ax_ph, self.ax_if])
        ax.set_xlabel('Time from event centre (s)')
        self.ax_lfp.set_xlim(tt[0], tt[-1])
        for ax in (self.ax_lfp, self.ax_ph):
            ax.tick_params(labelbottom=False)

        self._draw_overview()

        n_acc = int((ev_df['status'] == 'accepted').sum())
        n_rej = int((ev_df['status'] == 'rejected').sum())
        rel = res.ncs_path.relative_to(ROOT_FOLDER) if res.ncs_path.is_relative_to(ROOT_FOLDER) else res.ncs_path
        self.info_var.set(
            f'File {self.file_idx + 1}/{len(self.review_files)}: {rel}\n'
            f'Event {self.ev_idx + 1}/{len(ev_df)} at {ev["start_rel_s"]:.3f} s '
            f'({ev["start_rel_s"] / 60:.2f} min) | duration {ev["duration_ms"]:.0f} ms '
            f'(GP-extended {ev["gp_ext_duration_ms"]:.0f} ms, orange) | min IF {ev["min_IF_Hz"]:.1f} Hz | '
            f'phase lost {ev["phase_lost_deg"]:.0f}° (run backwards {ev["backward_phase_deg"]:.0f}°)\n'
            f'Envelope {ev["envelope_uV"]:.1f} µV ({ev["envelope_ratio"]:.2f}× median theta envelope) | '
            f'ACG ED_min of epoch {ev["acg_ED_min"]:.2f} | '
            f'Accepted {n_acc}  Rejected {n_rej}  Unreviewed {len(ev_df) - n_acc - n_rej}')
        self.toolbar.update()
        self.canvas.draw_idle()

    def _draw_overview(self):
        res, ev_df = self.res, self.res.events
        ax = self.ax_ov
        ax.clear()
        dur_min = res.recording_dur_s / 60
        _shade_spans(ax, res.theta_spans_rel_s() / 60, facecolor=THETA_SHADE, alpha=0.25, linewidth=0)
        centers = ev_df['center_rel_s'].to_numpy() / 60
        cols = [STATUS_COLORS.get(st, '#333333') for st in ev_df['status']]
        ax.vlines(centers, 0, 1, colors=cols, linewidth=0.8)
        for z in res.zero_rel_s / 60:
            ax.axvline(z, color=ZERO_COLOR, lw=1.3)
        ax.axvline(centers[self.ev_idx], color='blue', lw=2.5)
        ax.set_xlim(0, dur_min)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel('Recording time (min) — click to jump to the nearest event  '
                      '(green band = theta-positive, red = "zero" event, blue = current)', fontsize=8)


# ============================================================================
# Main
# ============================================================================

def main():
    ncs_files = find_ncs_files(ROOT_FOLDER)
    if not ncs_files:
        raise FileNotFoundError(f'No .ncs files found under {ROOT_FOLDER}')
    print(f'Found {len(ncs_files)} .ncs file(s) under {ROOT_FOLDER}')

    # 1) detect reversals in every file; keep the results of files with events
    results: dict[Path, FileResult] = {}
    for ncs in ncs_files:
        try:
            res = analyze_ncs_file(ncs)
            save_file_outputs(res, reviewed=False)
        except Exception as exc:
            traceback.print_exc()
            print(f'ERROR processing {ncs}: {exc}')
            continue
        if not res.events.empty:
            results[ncs] = res

    n_events = sum(len(r.events) for r in results.values())
    n_unrev = sum(int((r.events['status'] == 'unreviewed').sum()) for r in results.values())
    print(f'\nDetection finished: {n_events} phase reversals in {len(results)} of '
          f'{len(ncs_files)} file(s), {n_unrev} unreviewed.')
    if not results:
        write_summary(ncs_files)
        return

    # 2) manual review (GUI) or accept everything unreviewed
    if MANUAL_REVIEW:
        ReversalReviewGUI(ncs_files, results).run()
        return

    for res in results.values():
        n_acc = accept_all_unreviewed(res)
        save_file_outputs(res, reviewed=True)
        print(f'  {res.ncs_path.name}: {n_acc} events accepted without review')
    write_summary(ncs_files)


if __name__ == '__main__':
    main()


"""
Yes, there's a standard method: compare the KDE against a Monte-Carlo null and use a global (maximum-statistic) envelope. I've added it to ThetaPhaseReversal_GP_Review_v1.py. It runs as step 7 on the accepted reversals of each recording, after review or auto-accept.

How it works

KDE rate. scipy.stats.gaussian_kde is applied to the reversal times with a fixed kernel SD, KDE_BANDWIDTH_S = 30 s. I divided it by a KDE of theta-positive time made with the same kernel, which gives reversals per minute of theta. Without this step, a "peak" can simply mean there was more theta at that point, since reversals can only be detected inside theta-positive epochs.
Background (null). The same events are moved around within theta-positive time 1000 times, and the KDE rate is recomputed each time. The default null, KDE_NULL = 'isi', shuffles the gaps between events. That keeps any short bursts of reversals but removes slow build-ups. 'uniform' places events at random instead, which is more lenient.
Significance.
Threshold: the 95th percentile of each shuffle's maximum over the whole recording. Checking a 95% band point by point would give false peaks somewhere in almost every recording; using the maximum corrects for testing every time point.
Peaks: each stretch above the threshold counts as one peak.
p-value: p_global is the fraction of shuffles whose maximum reached the peak height, so it is already corrected for multiple comparisons.
Outputs

<stem>_ReversalKDE.png (the plot), _ReversalKDE.csv (the curves) and _ReversalKDE_Peaks.csv (one row per peak).
Each peak row includes its time, how many times above the session mean it is, p_global, the number of events in the peak, and its offset from the nearest 'zero' event.
New kde_* columns in the summary Excel.
Test results so far (synthetic data)

False peaks: with no real peak present, 0/40 recordings had a false peak with the ISI null and 1/40 with the uniform null, against a 5% target. So the error rate is controlled.
Speed: about 5 s per recording for 1000 shuffles.
First burst test: a weak burst (only about 4 extra events) was not detected. That's expected at that size; the stronger rerun above checks a realistic one.
Things to know

Kernel width sets the timescale you can find. 30 s finds bursts lasting tens of seconds to minutes; set KDE_BANDWIDTH_S to the timescale you care about. I fixed it in seconds rather than using scipy's automatic choice, which would smooth over several minutes here and change from shuffle to shuffle.
Theta-poor stretches limit sensitivity. The threshold is set by the noisiest parts of the curve, which are where theta is rare. Raising KDE_MIN_THETA_FRAC (default 0.2) excludes those stretches and makes it easier to detect peaks elsewhere.
Each recording is tested separately. If you want to know whether reversals cluster around the 'zero' events across recordings, that's a different test: line the events up on the zero events and use a circular-shift null. I can add that if you want it.
"""