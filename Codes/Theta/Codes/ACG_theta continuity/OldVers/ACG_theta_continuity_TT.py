"""
Autocorrelogram (ACG) "theta continuity" analysis for Neuralynx .ncs LFP data.

Implements the peak-range oscillation-quantification method of:
    Dunn et al. 2022, Nature Communications 13:6997
    "A common neural code for social and solitary foraging behaviour..." (ferret theta paper)
    https://www.nature.com/articles/s41467-022-33507-2
    Method described in Supplementary Methods / Supplementary Figure 5 (peak-range
    measurement), used to produce Figure 4 (theta persists during immobility).
    Original MATLAB: quantify_xcorr_epochs.m / create_sine_ref_xcorrs.m
    https://github.com/slsdunn/theta-paper-code

Algorithm summary (Supplementary Fig. 5):
  1. Take a short (default 1 s) LFP epoch and compute its autocorrelogram,
     normalised so the zero-lag value is 1.
  2. Build a bank of reference sinusoids spanning a frequency range, and
     compute the (identically normalised) autocorrelogram of each.
  3. Find the reference sinusoid whose autocorrelogram has the smallest
     (normalised) Euclidean distance to the data epoch's autocorrelogram --
     its frequency is the epoch's frequency estimate.
  4. Using windows defined by the *matched reference sinusoid's* first side
     peak and its flanking troughs, find the corresponding peak/troughs in
     the *data* autocorrelogram. "Peak range" = peak height - mean trough
     height, normalised by the matched sinusoid's own peak range (peak range
     shrinks with frequency for a pure sinusoid -- Supp. Fig. 5h -- so this
     normalisation removes that frequency dependence).
  Peak range close to 1 => a clean, sinusoidal (theta-like) oscillation;
  close to 0 => a flat/non-oscillatory autocorrelogram.

This script is fully self-contained: it reads Neuralynx .ncs files directly
(no dependency on FOOOF.py) and recursively processes every .ncs file found
under a single ROOT_DIR (see Configuration below).
"""

import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# %% ==================== Configuration ===========================================

# Root directory to search recursively for .ncs files. Every .ncs found
# anywhere under this tree (in any subfolder) is processed.
ROOT_DIR = r'C:/Runita/NMR/analysis/AllSort_Results/LFP'

# Where results/figures are written (default: alongside ROOT_DIR).
OUTPUT_DIR = os.path.join(ROOT_DIR, 'ACG_theta_continuity_results_v2_imob1cms')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')

# ---- Acquisition ----
NATIVE_FS = 32000  # native Neuralynx .ncs sampling rate (Hz)
ADBitVolts = 0.000003051757812500000169  # V per ADC count (Neuralynx header)

# Neuralynx .ncs record format (512 int16 samples per record, 16 kB header skipped)
ncs_dtype = np.dtype([
    ('timestamp'  , '<u8'),
    ('sc_number'  , '<u4'),
    ('cell_number', '<u4'),
    ('params'     , '<u4'),
    ('samples'    , '<i2', (512,)),
])

# ---- 50 Hz line-noise cleaning (European mains) ----
LINE_HARMONICS = [50.0, 100.0, 150.0, 200.0]  # harmonics below Nyquist (FS_ACG/2 = 500 Hz)
NOTCH_Q        = 30.0

# ---- Artifact rejection (epoch-wise peak-to-peak, robust MAD outlier) ----
MAD_THRESH = 5.0

# ---- Velocity / running-speed epoch gating ----
# Every .ncs file's session folder is expected to hold one tracking .csv with
# a UNIX timestamp (us, column A) on the same clock as the .ncs timestamps,
# plus x/y position in cm (columns D, E).
#
# Position is cleaned before speed is computed from it (see
# _smooth_tracking_position): frame-to-frame jumps implying a speed above
# POS_JUMP_THRESH_CMS are treated as tracking artifacts and linearly
# interpolated over, then the x/y traces are Gaussian-smoothed with sigma
# POS_SMOOTH_SIGMA_SMP (in samples). Matches the protocol in
# PlaceCellCharacterization_SpeedModv3.py.
POS_JUMP_THRESH_CMS  = 80.0   # frame-to-frame jumps implying a speed above this (cm/s) are tracking artifacts
POS_SMOOTH_SIGMA_SMP = 1.0    # Gaussian smoothing sigma (in samples) applied to x/y tracking position

# The original MATLAB (create_sine_ref_xcorrs.m) hardcodes its reference-sine
# time base at 1 kHz ( reft = 0:1/1000:(datsize-1)/1000 ), so data epochs must
# be resampled to this rate before autocorrelation for the frequency axis of
# the reference bank to correspond correctly to Hz.
FS_ACG = 1000.0                 # Hz -- resample rate for ACG analysis
EPOCH_SEC = 1.0                 # s  -- autocorrelogram epoch length (Supp. Fig. 5a)

# Reference sinusoid bank (Supp. Fig. 5b-h uses 4-14 Hz for rat/ferret theta;
# set to this dataset's theta range, 3-7 Hz).
ACG_FREQ_RANGE = (3.0, 7.0)     # Hz
ACG_FREQ_RES = 0.1              # Hz -- step of the reference bank. Not given a specific
                                 # numeric value in the supplement; 0.1 Hz gives a fine
                                 # frequency estimate at a modest compute cost -- tighten
                                 # (e.g. 0.05) or loosen (e.g. 0.25) as needed.
ACG_PEAK_PROMINENCE = 0.05      # matches findMinMax(refxc, 0.05, 'fixed') in the original

# Fit-quality gate: ED_min is the normalised Euclidean distance from an epoch's
# data autocorrelogram to its closest-matching reference sinusoid (0 = perfect
# match). Epochs whose ED_min exceeds this threshold are marked 'skipped' (same
# flag used for too-short epochs), so every downstream aggregation/plot that
# already filters on ~skipped excludes them automatically. None disables this
# filtering. There's no universal cutoff -- inspect the ED_min column's
# distribution in your own data (e.g. results_df['ED_min'].hist()) before
# picking a value.
ACG_ED_MIN_THRESH = None

# Movement classification -- single-threshold binary split: epoch median
# speed above SPEED_THRESH_CMS -> 'moving', at/below it -> 'immobile'.
SPEED_THRESH_CMS = 4.0


# %% ==================== .ncs / tracking data import ==============================

def load_ncs(fpath):
    """Read a Neuralynx .ncs file.

    Returns
    -------
    lfp             : ndarray  raw trace in microvolts (uV)
    start_timestamp : int      UNIX timestamp (us) of the first sample -- same
                                clock as the tracking .csv, used to align
                                epochs to running speed.
    """
    data = np.memmap(fpath, dtype=ncs_dtype, mode='r', offset=16 * 1024)
    lfp = np.concatenate(data['samples']).astype(np.float64) * ADBitVolts * 1e6
    start_timestamp = int(data['timestamp'][0])
    return lfp, start_timestamp


def notch_filter(x, fs_hz, freqs, Q=30.0):
    """Zero-phase IIR notch at each frequency in `freqs` (skips freqs >= Nyquist)."""
    y = np.asarray(x, dtype=np.float64)
    nyq = fs_hz / 2.0
    for f0 in freqs:
        if f0 <= 0 or f0 >= nyq:
            continue
        b, a = signal.iirnotch(f0, Q, fs_hz)
        y = signal.filtfilt(b, a, y)
    return y


def detrend_signal(x, dtype='linear'):
    """Remove a polynomial trend (default: linear) from the full LFP trace,
    to clear slow drift that a notch filter alone doesn't address."""
    return signal.detrend(np.asarray(x, dtype=np.float64), type=dtype) # type: ignore


def _robust_high_outliers(x, thresh, ref_mask=None):
    """Boolean mask of samples that are high outliers by robust (MAD) z-score.

    If `ref_mask` is given, the median/MAD reference statistics are estimated
    from `x[ref_mask]` only, so epochs excluded upstream can't skew the
    robust threshold -- z-scores are still returned for every element of `x`.
    """
    x = np.asarray(x, dtype=np.float64)
    ref = x if ref_mask is None else x[np.asarray(ref_mask, dtype=bool)]
    if ref.size == 0:
        ref = x
    med = np.median(ref)
    mad = np.median(np.abs(ref - med))
    if mad == 0:
        mad = 1e-20
    robust_z = 0.6745 * (x - med) / mad
    return robust_z > thresh


def find_position_file(ncs_path):
    """Return the tracking .csv that lives alongside `ncs_path` (same folder).

    Every session folder holds exactly one tracking .csv shared by all its
    .ncs files. Returns None if none is found; more than one is present,
    the first (alphabetically) is used.
    """
    folder = os.path.dirname(ncs_path)
    candidates = sorted(
        fn for fn in os.listdir(folder)
        if fn.lower().endswith('.csv'))
    if not candidates:
        return None
    if len(candidates) > 1:
        print(f'    Multiple .csv tracking files in {folder}, '
              f'using: {candidates[0]}')
    return os.path.join(folder, candidates[0])


def _smooth_tracking_position(
    x_cm:            np.ndarray,
    y_cm:            np.ndarray,
    t_us:            np.ndarray,
    jump_thresh_cms: float = POS_JUMP_THRESH_CMS,
    sigma_samples:   float = POS_SMOOTH_SIGMA_SMP,
) -> tuple:
    """Clean and smooth the x/y tracking position before any downstream use.

    Iterative jump removal + Gaussian smoothing:
      1. Frame-to-frame steps between surviving ("good") samples that imply
         a speed above `jump_thresh_cms` are treated as tracking artifacts.
         This is re-tested against the shrinking good-sample set until no
         new artifacts are found, so runs of two or more consecutive bad
         frames (e.g. a tracker glitch that lingers for a few frames) are
         caught -- not just single-frame jumps relative to the immediately
         preceding raw sample.
      2. Artifact frames are filled by linear interpolation (in time) between
         the nearest surviving good samples, rather than held at the last
         good position -- holding would freeze position then "snap" back at
         the far edge of a dropout, itself implying an extra artificial
         speed spike right at the resumption point.
      3. The cleaned x/y traces are Gaussian-smoothed (`imgaussfilt` equivalent
         via `gaussian_filter1d`), sigma expressed in samples.
    """
    n = len(x_cm)
    if n < 2:
        return x_cm.copy(), y_cm.copy()

    bad = np.zeros(n, dtype=bool)
    for _ in range(n):
        good_idx = np.where(~bad)[0]
        if len(good_idx) < 2:
            break
        dt_good = np.diff(t_us[good_idx]) * 1e-6
        with np.errstate(invalid='ignore', divide='ignore'):
            step_speed = np.hypot(np.diff(x_cm[good_idx]), np.diff(y_cm[good_idx])) / dt_good
        step_speed[dt_good <= 0] = 0.0

        newly_bad = step_speed > jump_thresh_cms
        if not newly_bad.any():
            break
        # Mark the later sample of each offending pair as bad and re-test
        # against the remaining good set next round.
        bad[good_idx[1:][newly_bad]] = True

    good_idx = np.where(~bad)[0]
    if len(good_idx) == 0 or len(good_idx) == n:
        x_clean, y_clean = x_cm.copy(), y_cm.copy()
    else:
        x_clean = np.interp(t_us, t_us[good_idx], x_cm[good_idx])
        y_clean = np.interp(t_us, t_us[good_idx], y_cm[good_idx])

    x_smooth = gaussian_filter1d(x_clean, sigma=sigma_samples, mode='nearest')
    y_smooth = gaussian_filter1d(y_clean, sigma=sigma_samples, mode='nearest')
    return x_smooth, y_smooth


def _instantaneous_speed(x_cm: np.ndarray, y_cm: np.ndarray, t_us: np.ndarray) -> np.ndarray:
    """Per-frame running speed (cm/s), padded to len(t_us) (repeats last value)."""
    dt_s = np.diff(t_us) * 1e-6
    with np.errstate(invalid='ignore', divide='ignore'):
        speed = np.hypot(np.diff(x_cm), np.diff(y_cm)) / dt_s
    speed[dt_s <= 0] = np.nan
    return np.append(speed, speed[-1])


def compute_velocity_from_position(csv_path):
    """Compute running speed (cm/s) from a tracking .csv.

    Column layout (positional): UNIX timestamp in us (col A) on the same
    clock as the .ncs LFP timestamps, x in cm (col D), y in cm (col E).

    Position is first cleaned with `_smooth_tracking_position` (iterative
    jump removal + Gaussian smoothing -- see that function's docstring),
    then speed is the frame-to-frame displacement of the cleaned trace
    divided by the actual elapsed time between samples (`_instantaneous_speed`).

    Returns
    -------
    time_us : ndarray   absolute UNIX timestamp of each tracking sample (us)
                        -- same clock as the .ncs LFP start timestamp, so
                        epochs can be aligned to speed without assuming a
                        shared t=0
    speed   : ndarray   running speed (cm/s), same length as time_us
    """
    df = pd.read_csv(csv_path, usecols=[0, 3, 4])
    df.columns = ['time_us', 'x', 'y']
    df = df.sort_values('time_us').reset_index(drop=True)

    time_us = df['time_us'].to_numpy(dtype=np.float64)
    x_cm = df['x'].to_numpy(dtype=np.float64)
    y_cm = df['y'].to_numpy(dtype=np.float64)

    x_smooth, y_smooth = _smooth_tracking_position(x_cm, y_cm, time_us)
    speed = _instantaneous_speed(x_smooth, y_smooth, time_us)

    return time_us, speed


def parse_metadata_from_path(rel_path):
    """Map a .ncs path (relative to ROOT_DIR) to (animal, date, session,
    tetrode, channel) metadata.

    EDIT THIS to match your folder/filename layout if it doesn't fit. Default
    heuristic:
      * tetrode/channel parsed from a Neuralynx-style 'CSC<t>ch<c>' token if present;
      * animal/date/session taken from the first three parent folders of the
        path (relative to ROOT_DIR);
      * anything unknown falls back to the filename stem or 'NA'.
    """
    import re
    fname = os.path.basename(rel_path)
    parts = os.path.normpath(rel_path).split(os.sep)

    m = re.search(r'CSC(\d+)(?:ch(\d+))?', fname, re.IGNORECASE)
    tetrode = m.group(1) if m else 'NA'
    channel = m.group(2) if (m and m.group(2)) else os.path.splitext(fname)[0]

    animal  = parts[0] if len(parts) >= 2 else 'NA'
    date    = parts[1] if len(parts) >= 3 else 'NA'
    session = parts[2] if len(parts) >= 4 else (parts[1] if len(parts) >= 3 else 'NA')
    return {'animal': animal, 'date': date, 'session': session,
            'tetrode': tetrode, 'channel': channel}


# %% ==================== Core ACG algorithm (translated from quantify_xcorr_epochs.m) =

def zero_crossings(x):
    """Fractional (linearly-interpolated) sample indices where `x` crosses zero.

    Equivalent to the MATLAB helper `zero_crossings` used by
    create_sine_ref_xcorrs.m to bound the peak/trough search windows.
    """
    x = np.asarray(x, dtype=np.float64)
    s = np.sign(x)
    s[s == 0] = 1.0
    idx = np.where(np.diff(s) != 0)[0]
    x1, x2 = x[idx], x[idx + 1]
    frac = -x1 / (x2 - x1)
    return idx + frac


def _find_extrema(x, prominence, kind='max'):
    """Local maxima/minima of `x` above a fixed prominence threshold.

    Equivalent to MATLAB's `findMinMax(x, thresh, 'fixed')`.
    """
    if kind == 'max':
        idx, _ = find_peaks(x, prominence=prominence)
    else:
        idx, _ = find_peaks(-np.asarray(x), prominence=prominence)
    return idx, np.asarray(x)[idx]


def create_sine_ref_xcorrs(ref_freqs, datsize, fs=FS_ACG, peak_prominence=ACG_PEAK_PROMINENCE):
    """Build the reference bank of sinusoid autocorrelograms and their
    peak/trough search windows.

    Python translation of create_sine_ref_xcorrs.m.

    Parameters
    ----------
    ref_freqs : 1D array of candidate frequencies (Hz)
    datsize   : int, number of samples in one data epoch
    fs        : sampling rate (Hz) used to synthesise the reference sinusoids
                -- must match the rate the real data epochs are sampled at.

    Returns
    -------
    refXC      : (2*datsize-1, n_freqs) normalised autocorrelogram of each reference sine
    refED      : (n_freqs,) L2 norm of each reference autocorrelogram (for normalising ED)
    refRange   : (n_freqs,) peak range of each reference sine's own autocorrelogram
    refP1range : (2, n_freqs) int, [lo, hi] sample-index window bracketing the first
                 side peak (used to search the *data* autocorrelogram's peak)
    refT1range : (2, n_freqs) int, window bracketing the trough before that peak
    refT2range : (2, n_freqs) int, window bracketing the trough after that peak
    """
    ref_freqs = np.asarray(ref_freqs, dtype=np.float64)
    n_freqs = ref_freqs.size
    reft = np.arange(datsize) / fs
    n_lags = 2 * datsize - 1
    center = datsize - 1   # zero-lag index (0-based) in the full autocorrelogram

    refXC = np.empty((n_lags, n_freqs))
    refRange = np.empty(n_freqs)
    refP1range = np.empty((2, n_freqs), dtype=int)
    refT1range = np.empty((2, n_freqs), dtype=int)
    refT2range = np.empty((2, n_freqs), dtype=int)

    for n, freq in enumerate(ref_freqs):
        refsig = np.sin(2 * np.pi * freq * reft)
        refxc_raw = signal.correlate(refsig, refsig, mode='full', method='auto')
        refxc = refxc_raw / np.max(refxc_raw)
        refXC[:, n] = refxc

        max_idx, max_val = _find_extrema(refxc, peak_prominence, kind='max')
        min_idx, min_val = _find_extrema(refxc, peak_prominence, kind='min')

        center_matches = np.where(max_idx == center)[0]
        center_row = int(center_matches[0]) if center_matches.size else \
            int(np.argmin(np.abs(max_idx - center)))
        if center_row + 1 >= len(max_idx):
            raise ValueError(
                f'{freq:.3f} Hz reference sine: no side peak found after the '
                f'zero-lag peak -- epoch too short for this frequency / '
                f'freq_range minimum too low.')
        peak1_idx = max_idx[center_row + 1]
        peak1_val = max_val[center_row + 1]

        before = min_idx < peak1_idx
        after = min_idx > peak1_idx
        if not before.any() or not after.any():
            raise ValueError(f'{freq:.3f} Hz reference sine: could not bracket '
                              f'troughs around the first side peak.')
        trough1_idx, trough1_val = min_idx[before][-1], min_val[before][-1]
        trough2_idx, trough2_val = min_idx[after][0], min_val[after][0]

        refRange[n] = peak1_val - np.mean([trough1_val, trough2_val])

        crossings = zero_crossings(refxc)
        belowt1 = crossings[crossings < trough1_idx]
        abovet2 = crossings[crossings > trough2_idx]
        abovep1 = crossings[crossings > peak1_idx]
        belowp1 = crossings[crossings < peak1_idx]
        if not (belowp1.size and abovep1.size and belowt1.size and abovet2.size):
            raise ValueError(f'{freq:.3f} Hz reference sine: could not find '
                              f'bounding zero-crossings for the search windows.')

        refP1range[:, n] = [int(round(belowp1[-1])), int(round(abovep1[0]))]
        refT1range[:, n] = [int(round(belowt1[-1])), int(round(belowp1[-1]))]
        refT2range[:, n] = [int(round(abovep1[0])), int(round(abovet2[0]))]

    refED = np.linalg.norm(refXC, axis=0)
    return refXC, refED, refRange, refP1range, refT1range, refT2range


def quantify_xcorr_epochs(data_epochs, freq_range=ACG_FREQ_RANGE, freq_resolution=ACG_FREQ_RES,
                          fs=FS_ACG, peak_prominence=ACG_PEAK_PROMINENCE,
                          ed_min_thresh=ACG_ED_MIN_THRESH):
    """Autocorrelogram peak-range quantification of oscillatory activity.

    Python translation of quantify_xcorr_epochs.m (Dunn et al. 2022).

    Parameters
    ----------
    data_epochs : (winlength, nepochs) array -- each column one LFP epoch,
                  sampled at `fs` Hz. NaNs are allowed (an epoch is skipped
                  if fewer than 2 finite samples remain).
    freq_range      : (fmin, fmax) Hz, reference sinusoid bank range
    freq_resolution : Hz step of the reference bank
    fs              : sampling rate (Hz) of `data_epochs`
    ed_min_thresh   : if not None, epochs whose ED_min (normalised distance to
                      the closest-matching reference sinusoid) exceeds this
                      value are marked 'skipped' -- i.e. no reference sinusoid
                      fit them well enough to trust the peak-range measurement.

    Returns
    -------
    XC     : (2*winlength-1, nepochs) normalised data autocorrelograms
    result : DataFrame, one row per epoch, columns:
             ED_min, freq, peak1, peak1_idx, trough1, trough1_idx,
             trough2, trough2_idx, peakrange, peakrangenorm, skipped
             ('skipped' covers both too-short epochs and, if `ed_min_thresh`
             is set, epochs with a poor sinusoid fit)
    """
    data_epochs = np.asarray(data_epochs, dtype=np.float64)
    winlength, nepochs = data_epochs.shape
    n_lags = 2 * winlength - 1

    ref_freqs = np.arange(freq_range[0], freq_range[1] + freq_resolution / 2, freq_resolution)
    refXC, refED, refRange, refP1range, refT1range, refT2range = create_sine_ref_xcorrs(
        ref_freqs, winlength, fs=fs, peak_prominence=peak_prominence)

    XC = np.full((n_lags, nepochs), np.nan)
    ED_min = np.full(nepochs, np.nan)
    freq_est = np.full(nepochs, np.nan)
    peak1 = np.full(nepochs, np.nan)
    peak1_idx = np.full(nepochs, np.nan)
    trough1 = np.full(nepochs, np.nan)
    trough1_idx = np.full(nepochs, np.nan)
    trough2 = np.full(nepochs, np.nan)
    trough2_idx = np.full(nepochs, np.nan)
    peakrange = np.full(nepochs, np.nan)
    peakrangenorm = np.full(nepochs, np.nan)
    skipped = np.zeros(nepochs, dtype=bool)

    for n in range(nepochs):
        col = data_epochs[:, n]
        dataepoch = col[~np.isnan(col)]

        if dataepoch.size <= 1:
            skipped[n] = True
            continue

        xc_raw = signal.correlate(dataepoch, dataepoch, mode='full', method='auto')
        xc = xc_raw / np.max(xc_raw)
        XC[:xc.size, n] = xc
        xc_full = XC[:, n]   # NaN-padded to n_lags if the epoch was short

        # Euclidean distance (ignoring any NaN padding) between the data
        # epoch's autocorrelogram and every reference sinusoid autocorrelogram,
        # normalised by each reference's own norm.
        sq_diff = (refXC - xc_full[:, None]) ** 2
        ED = np.sqrt(np.nansum(sq_diff, axis=0))
        normED = ED / refED
        mi = int(np.nanargmin(normED))
        ED_min[n] = normED[mi]
        freq_est[n] = ref_freqs[mi]

        if ed_min_thresh is not None and ED_min[n] > ed_min_thresh:
            skipped[n] = True

        p1_lo, p1_hi = refP1range[:, mi]
        seg = xc_full[p1_lo:p1_hi + 1]
        pk_local = np.nanargmax(seg)
        pk_idx = p1_lo + pk_local
        pk_val = xc_full[pk_idx]

        t1_lo, t1_hi = refT1range[:, mi]
        seg = xc_full[t1_lo:t1_hi + 1]
        t1_local = np.nanargmin(seg)
        t1_idx = t1_lo + t1_local
        t1_val = xc_full[t1_idx]

        t2_lo, t2_hi = refT2range[:, mi]
        seg = xc_full[t2_lo:t2_hi + 1]
        t2_local = np.nanargmin(seg)
        t2_idx = t2_lo + t2_local
        t2_val = xc_full[t2_idx]

        peak1[n], peak1_idx[n] = pk_val, pk_idx
        trough1[n], trough1_idx[n] = t1_val, t1_idx
        trough2[n], trough2_idx[n] = t2_val, t2_idx

        pr = pk_val - np.mean([t1_val, t2_val])
        peakrange[n] = pr
        peakrangenorm[n] = pr / refRange[mi]

    result = pd.DataFrame({
        'ED_min': ED_min,
        'freq': freq_est,
        'peak1': peak1, 'peak1_idx': peak1_idx,
        'trough1': trough1, 'trough1_idx': trough1_idx,
        'trough2': trough2, 'trough2_idx': trough2_idx,
        'peakrange': peakrange,
        'peakrangenorm': peakrangenorm,
        'skipped': skipped,
    })
    return XC, result


# %% ==================== Epoching + movement labelling ============================

def build_epochs(lfp, nperseg):
    """Reshape a continuous trace into non-overlapping (n_total, nperseg) epochs."""
    n_total = len(lfp) // nperseg
    if n_total == 0:
        raise ValueError('Trace is shorter than one epoch.')
    epochs = lfp[:n_total * nperseg].reshape(n_total, nperseg)
    return epochs, n_total


def reject_artifact_epochs(epochs, mad_thresh=MAD_THRESH):
    """Boolean keep-mask: epoch-wise peak-to-peak amplitude, robust (MAD)
    outlier rejection."""
    p2p = epochs.max(axis=1) - epochs.min(axis=1)
    return ~_robust_high_outliers(p2p, mad_thresh)


def classify_epoch_movement(time_us, speed, n_total, nperseg, fs_hz, lfp_start_us,
                            speed_thresh=SPEED_THRESH_CMS):
    """Per-epoch median running speed and movement label.

    Epoch i spans absolute UNIX time [lfp_start_us + i*epoch_dur_us,
    lfp_start_us + (i+1)*epoch_dur_us) -- lfp_start_us is the .ncs file's
    first-sample timestamp, on the same UNIX clock as the tracking .csv, so
    epochs are aligned to actual recording time rather than an assumed
    shared t=0. Single-threshold binary label: median speed > `speed_thresh`
    -> 'moving', <= `speed_thresh` -> 'immobile'. An epoch whose median speed
    can't be determined (e.g. NaN from a zero/negative dt in the tracking
    data) is labelled 'other' and dropped downstream.
    """
    epoch_dur_us = 1e6 * nperseg / fs_hz
    med_speed = np.full(n_total, np.nan)
    for i in range(n_total):
        t0 = lfp_start_us + i * epoch_dur_us
        t1 = lfp_start_us + (i + 1) * epoch_dur_us
        in_epoch = (time_us >= t0) & (time_us < t1)
        if in_epoch.any():
            med_speed[i] = np.nanmedian(speed[in_epoch])
        else:
            nearest = np.argmin(np.abs(time_us - (t0 + t1) / 2))
            med_speed[i] = speed[nearest]

    label = np.full(n_total, 'other', dtype=object)
    valid = ~np.isnan(med_speed)
    label[valid & (med_speed > speed_thresh)] = 'moving'
    label[valid & (med_speed <= speed_thresh)] = 'immobile'
    return med_speed, label


# %% ==================== Per-file / root-directory processing =====================

def process_ncs_for_acg(fpath, freq_range=ACG_FREQ_RANGE, freq_resolution=ACG_FREQ_RES,
                        epoch_sec=EPOCH_SEC, fs_acg=FS_ACG, ed_min_thresh=ACG_ED_MIN_THRESH):
    """Load one .ncs file, resample/clean it, split it into epochs labelled by
    movement state, and compute the ACG peak-range quantification for every
    clean moving/immobile epoch.

    `ed_min_thresh` is forwarded to quantify_xcorr_epochs -- epochs with a
    poor sinusoid fit (ED_min above threshold) are marked 'skipped' rather
    than dropped, so the resulting DataFrame's row count stays aligned with
    'epoch_index'/'file'; filter on '~skipped' downstream as usual.

    Returns a DataFrame (one row per epoch) with the quantify_xcorr_epochs
    columns plus 'movement', 'median_speed_cms', and 'epoch_index'.
    """
    lfp, lfp_start_us = load_ncs(fpath)
    lfp = signal.resample_poly(lfp, int(fs_acg), int(NATIVE_FS))
    lfp = notch_filter(lfp, fs_acg, LINE_HARMONICS, NOTCH_Q)
    lfp = detrend_signal(lfp, dtype='linear')

    nperseg = int(round(fs_acg * epoch_sec))
    epochs, n_total = build_epochs(lfp, nperseg)
    keep_artifact = reject_artifact_epochs(epochs)

    pos_path = find_position_file(fpath)
    if pos_path is None:
        raise ValueError('No tracking .csv found next to this file -- '
                          'movement labelling (moving/immobile) requires it.')
    time_us, speed = compute_velocity_from_position(pos_path)
    med_speed, label = classify_epoch_movement(time_us, speed, n_total, nperseg,
                                               fs_acg, lfp_start_us)

    keep = keep_artifact & np.isin(label, ['moving', 'immobile'])
    if not keep.any():
        raise ValueError('No clean moving/immobile epochs after artifact rejection.')

    data_epochs = epochs[keep].T   # (nperseg, n_kept), matches quantify_xcorr_epochs layout
    _XC, result = quantify_xcorr_epochs(data_epochs, freq_range, freq_resolution,
                                        fs=fs_acg, peak_prominence=ACG_PEAK_PROMINENCE,
                                        ed_min_thresh=ed_min_thresh)
    result['movement'] = label[keep]
    result['median_speed_cms'] = med_speed[keep]
    result['epoch_index'] = np.where(keep)[0]
    # absolute path -- lets get_epoch_waveform() re-fetch this exact epoch's
    # raw waveform later (e.g. for plot_example_epoch), regardless of cwd
    result['file'] = os.path.abspath(fpath)
    return result


def get_epoch_waveform(fpath, epoch_index, epoch_sec=EPOCH_SEC, fs_acg=FS_ACG):
    """Re-derive the cleaned (resampled / notch-filtered / detrended) waveform
    of one specific epoch from its source .ncs file, for illustrative
    plotting (plot_example_epoch). `fpath` + `epoch_index` should come from
    the 'file' / 'epoch_index' columns of a process_ncs_for_acg results row.
    """
    lfp, _ = load_ncs(fpath)
    lfp = signal.resample_poly(lfp, int(fs_acg), int(NATIVE_FS))
    lfp = notch_filter(lfp, fs_acg, LINE_HARMONICS, NOTCH_Q)
    lfp = detrend_signal(lfp, dtype='linear')
    nperseg = int(round(fs_acg * epoch_sec))
    epochs, _ = build_epochs(lfp, nperseg)
    return epochs[int(epoch_index)]


def process_root_directory(root_dir):
    """Recursively find every .ncs under `root_dir`, run process_ncs_for_acg on
    each, and concatenate the per-epoch results, tagging each row with
    path-derived metadata (parse_metadata_from_path)."""
    ncs_files = []
    for root, _dirs, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith('.ncs'):
                ncs_files.append(os.path.join(root, fname))
    print(f'Found {len(ncs_files)} .ncs files under {root_dir}')

    all_results = []
    for fpath in ncs_files:
        rel = os.path.relpath(fpath, root_dir)
        try:
            res = process_ncs_for_acg(fpath)
            meta = parse_metadata_from_path(rel)
            for k, v in meta.items():
                res[k] = v
            n_moving = int((res['movement'] == 'moving').sum())
            n_immobile = int((res['movement'] == 'immobile').sum())
            print(f'  OK: {rel}  [{len(res)} epochs kept: '
                  f'{n_moving} moving, {n_immobile} immobile]')
            all_results.append(res)
        except Exception as e:
            print(f'  SKIP: {rel} -- {e}')

    print(f'-> {len(all_results)} files processed\n')
    if not all_results:
        raise ValueError(f'No files processed successfully under {root_dir}')
    return pd.concat(all_results, ignore_index=True)


# %% ==================== Aggregation / plotting ====================================

def summarize_peak_range(df, group_cols=('animal', 'channel', 'movement')):
    """Median, IQR and n per group -- for depth-profile / summary plots
    (style of Fig. 4 and Supp. Fig. 6a-b)."""
    group_cols = list(group_cols)
    return (df[~df['skipped']]
            .groupby(group_cols)['peakrangenorm']
            .agg(median='median',
                 q25=lambda x: x.quantile(0.25),
                 q75=lambda x: x.quantile(0.75),
                 n='count')
            .reset_index())


def plot_moving_vs_immobile(df, animal, ax=None, channel_order=None):
    """Split violin of normalised peak range, moving vs immobile, per channel
    (style of Dunn et al. Fig. 4 / Supp. Fig. 6a-b)."""
    sub = df[(df['animal'] == animal) & (~df['skipped'])]
    ax = ax or plt.gca()
    sns.violinplot(data=sub, x='channel', y='peakrangenorm', hue='movement',
                   order=channel_order, split=True, ax=ax, cut=0,
                   palette={'moving': '#E67E22', 'immobile': '#7F8C8D'})
    ax.set_ylabel('Autocorr. peak range (norm.)')
    ax.set_xlabel('Channel')
    ax.set_title(animal)
    return ax


# %% ==================== Figure 4 -style plots (Dunn et al. 2022) =================
#
# Reproduces the panel layout of Fig. 4c-h: for each movement condition, an
# example epoch (raw trace + autocorrelogram with the matched reference
# sinusoid overlaid and the peak-range measurement marked in green), plus a
# frequency-vs-peak-range scatter (coloured by running speed, with marginal
# count histograms) marking the moving/immobile group centroids as green
# diamonds.

def compute_group_centroids(df, group_col='movement', freq_col='freq',
                            peakrange_col='peakrangenorm'):
    """Mean frequency and mean normalised peak range per movement group --
    the green diamond markers in Dunn et al. Fig. 4e/h."""
    clean = df[~df['skipped']]
    out = (clean.groupby(group_col)[[freq_col, peakrange_col]]
                .mean()
                .rename(columns={freq_col: 'freq_centroid',
                                 peakrange_col: 'peakrangenorm_centroid'}))
    out['n'] = clean.groupby(group_col).size()
    return out.reset_index()


def pick_example_epoch(df, movement, animal=None):
    """Pick the epoch (row) within one movement group whose (freq,
    peakrangenorm) sits closest to that group's own centroid -- a
    representative example epoch for plot_example_epoch / plot_figure4_style.
    """
    sub = df[~df['skipped']]
    if animal is not None:
        sub = sub[sub['animal'] == animal]
    sub = sub[sub['movement'] == movement]
    if sub.empty:
        raise ValueError(f"No '{movement}' epochs available"
                         f"{f' for animal {animal}' if animal else ''}.")
    c_freq, c_pr = sub['freq'].mean(), sub['peakrangenorm'].mean()
    d2 = (sub['freq'] - c_freq) ** 2 + (sub['peakrangenorm'] - c_pr) ** 2
    return sub.loc[d2.idxmin()]


def plot_example_epoch(row, epoch_waveform, ax_trace=None, ax_acg=None,
                       fs=FS_ACG, color='tab:blue'):
    """One example-epoch panel pair: raw trace (top) + autocorrelogram with
    the matched reference sinusoid overlaid and the peak-range measurement
    marked in green (bottom) -- style of Dunn et al. Fig. 4c/d/f/g.

    `row` : Series from a quantify_xcorr_epochs / process_ncs_for_acg results
            DataFrame (needs 'freq', 'peak1', 'peak1_idx', 'trough1',
            'trough2', and optionally 'median_speed_cms').
    `epoch_waveform` : the raw (fs-Hz) LFP epoch `row` was computed from,
            e.g. from get_epoch_waveform(row['file'], row['epoch_index']).
    """
    if ax_trace is None:
        _fig, (ax_trace, ax_acg) = plt.subplots(2, 1, figsize=(3, 3))

    winlength = len(epoch_waveform)
    t = np.arange(winlength) / fs
    ax_trace.plot(t, epoch_waveform, color=color, linewidth=1)
    ax_trace.set_xlim(0, winlength / fs)
    ax_trace.axis('off')
    if 'median_speed_cms' in row and np.isfinite(row['median_speed_cms']):
        ax_trace.text(0.02, 0.95, f"{row['median_speed_cms']:.1f} cm s$^{{-1}}$",
                      transform=ax_trace.transAxes, color=color, fontsize=9,
                      va='top')

    if ax_acg is None:
        return ax_trace, None

    xc_raw = signal.correlate(epoch_waveform, epoch_waveform, mode='full', method='auto')
    xc = xc_raw / np.max(xc_raw)
    lags_s = (np.arange(len(xc)) - (winlength - 1)) / fs

    refXC, *_ = create_sine_ref_xcorrs(np.array([row['freq']]), winlength, fs=fs)
    ax_acg.plot(lags_s, refXC[:, 0], color='0.7', linewidth=1.5, label='matched sinusoid')
    ax_acg.plot(lags_s, xc, color='black', linewidth=1, label='data')

    peak_lag = lags_s[int(row['peak1_idx'])]
    trough_mean = np.mean([row['trough1'], row['trough2']])
    ax_acg.plot([peak_lag, peak_lag], [trough_mean, row['peak1']],
               color='limegreen', linewidth=2, zorder=5)
    ax_acg.scatter([peak_lag], [row['peak1']], color='limegreen', s=15, zorder=6)

    ax_acg.axhline(0, color='0.85', linewidth=0.5, zorder=0)
    ax_acg.set_xlim(0, winlength / fs)
    ax_acg.set_ylim(-1, 1)
    ax_acg.set_xlabel('Time (s)')
    ax_acg.set_ylabel('r')
    return ax_trace, ax_acg


def _draw_freq_peakrange_scatter(df, ax_main, ax_top, ax_right, animal=None,
                                 mode='both', speed_vmax=50.0, moving_cmap='viridis',
                                 moving_alpha=0.5, immobile_color='black',
                                 immobile_edgecolor='#FFB3BA', immobile_edgewidth=0.8,
                                 n_bins=25, log_counts=True):
    """Shared drawing routine for the Fig. 4e/h-style scatter: frequency vs.
    normalised autocorrelogram peak range. Immobile epochs (speed doesn't
    meaningfully vary) are drawn solid jet-black with a pastel-red outline;
    moving epochs are drawn in `moving_cmap` (coloured by per-epoch running
    speed) at `moving_alpha` opacity, layered on top -- this two-tone scheme
    keeps the two heavily-overlapping point clouds visually distinguishable.
    Marginal count histograms follow the same colouring, and moving/immobile
    centroids are marked as green diamonds.

    `mode` selects which group(s) are drawn: 'both' (default), 'moving', or
    'immobile'. Axes/bins are always scaled to the full (both-group)
    frequency/peak-range range regardless of `mode`, so single-group panels
    stay directly comparable to the merged one. Used by
    plot_freq_peakrange_summary(), plot_figure4_style(), and
    plot_freq_peakrange_by_movement().
    """
    sub = df[~df['skipped']].copy()
    if animal is not None:
        sub = sub[sub['animal'] == animal]
    sub = sub[sub['movement'].isin(['moving', 'immobile'])]

    show_immobile = mode in ('both', 'immobile')
    show_moving = mode in ('both', 'moving')
    immobile = sub[sub['movement'] == 'immobile'] if show_immobile else sub.iloc[0:0]
    moving = sub[sub['movement'] == 'moving'] if show_moving else sub.iloc[0:0]

    # immobile first (solid, opaque) so the semi-transparent moving layer
    # sits visibly on top rather than being hidden underneath it
    if show_immobile:
        ax_main.scatter(immobile['freq'], immobile['peakrangenorm'], color=immobile_color,
                        s=16, edgecolor=immobile_edgecolor, linewidth=immobile_edgewidth,
                        alpha=1.0, label='immobile', zorder=2)
    sc = None
    if show_moving:
        speed = np.clip(moving['median_speed_cms'].to_numpy(), 0, speed_vmax)
        sc = ax_main.scatter(moving['freq'], moving['peakrangenorm'], c=speed,
                             cmap=moving_cmap, vmin=0, vmax=speed_vmax, s=14,
                             edgecolor='none', alpha=moving_alpha, label='moving', zorder=3)

    centroids_all = compute_group_centroids(sub)
    shown = {g for g, show in (('moving', show_moving), ('immobile', show_immobile)) if show}
    centroids = centroids_all[centroids_all['movement'].isin(shown)]
    marker_labels = {'moving': 'moving centroid', 'immobile': 'immobile centroid'}
    for _, crow in centroids.iterrows():
        ax_main.scatter(crow['freq_centroid'], crow['peakrangenorm_centroid'],
                        marker='D', s=90, facecolor='limegreen',
                        edgecolor='black', linewidth=1, zorder=5,
                        label=marker_labels.get(crow['movement'], crow['movement']))
    ax_main.set_xlabel('Frequency (Hz)')
    ax_main.set_ylabel('Autocorr. peak range')
    ax_main.set_ylim(0, 1)
    ax_main.legend(loc='upper left', frameon=False, fontsize=7)

    cmap_obj = plt.get_cmap(moving_cmap)

    def _movement_hist(value_col, bin_edges):
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        widths = np.diff(bin_edges)

        imm_idx = np.clip(np.digitize(immobile[value_col], bin_edges) - 1, 0, len(bin_edges) - 2)
        imm_counts = np.array([(imm_idx == b).sum() for b in range(len(bin_edges) - 1)])

        mov_idx = np.clip(np.digitize(moving[value_col], bin_edges) - 1, 0, len(bin_edges) - 2)
        mov_counts = np.zeros(len(bin_edges) - 1)
        mov_colors = np.zeros((len(bin_edges) - 1, 4))
        for b in range(len(bin_edges) - 1):
            m = mov_idx == b
            mov_counts[b] = m.sum()
            if m.any():
                mean_speed = np.clip(moving.loc[m, 'median_speed_cms'].mean(), 0, speed_vmax)
                mov_colors[b] = cmap_obj(mean_speed / speed_vmax)
        return centers, widths, imm_counts, mov_counts, mov_colors

    # bins always span the FULL (both-group) range, so mode='moving' /
    # 'immobile' single-group panels stay axis-comparable with the merged one
    freq_bins = np.linspace(sub['freq'].min(), sub['freq'].max(), n_bins)
    centers_f, widths_f, imm_f, mov_f, colors_f = _movement_hist('freq', freq_bins)
    ax_top.bar(centers_f, imm_f, width=widths_f, color=immobile_color,
              edgecolor=immobile_edgecolor, linewidth=immobile_edgewidth, alpha=1.0, zorder=2)
    ax_top.bar(centers_f, mov_f, width=widths_f, color=colors_f,
              edgecolor='none', alpha=moving_alpha, zorder=3)
    ax_top.set_ylabel('Count')
    ax_top.tick_params(labelbottom=False)
    if log_counts:
        ax_top.set_yscale('log')

    pr_bins = np.linspace(0, 1, n_bins)
    centers_p, widths_p, imm_p, mov_p, colors_p = _movement_hist('peakrangenorm', pr_bins)
    ax_right.barh(centers_p, imm_p, height=widths_p, color=immobile_color,
                 edgecolor=immobile_edgecolor, linewidth=immobile_edgewidth, alpha=1.0, zorder=2)
    ax_right.barh(centers_p, mov_p, height=widths_p, color=colors_p,
                 edgecolor='none', alpha=moving_alpha, zorder=3)
    ax_right.set_xlabel('Count')
    ax_right.tick_params(labelleft=False)
    if log_counts:
        ax_right.set_xscale('log')

    if animal:
        ax_top.set_title(animal)

    return sc, centroids


def plot_freq_peakrange_summary(df, animal=None, speed_vmax=50.0, moving_cmap='viridis',
                                moving_alpha=0.5, immobile_color='black',
                                immobile_edgecolor='#FFB3BA', figsize=(5.5, 5)):
    """Frequency vs. normalised autocorrelogram peak range. Immobile epochs
    are solid jet-black with a pastel-red outline; moving epochs are
    coloured by per-epoch running speed (`moving_cmap`) at `moving_alpha`
    opacity, so the two heavily-overlapping point clouds stay
    distinguishable. Includes marginal count histograms and moving/immobile
    centroid markers -- reproduces Dunn et al. Fig. 4e/h.

    Returns (fig, axes_dict, centroids_df).
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          wspace=0.05, hspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    sc, centroids = _draw_freq_peakrange_scatter(
        df, ax_main, ax_top, ax_right, animal=animal, speed_vmax=speed_vmax,
        moving_cmap=moving_cmap, moving_alpha=moving_alpha,
        immobile_color=immobile_color, immobile_edgecolor=immobile_edgecolor)

    cbar = fig.colorbar(sc, ax=ax_right, fraction=0.2, pad=0.35) # type: ignore
    cbar.set_label('moving speed (cm s$^{-1}$)')

    return fig, {'main': ax_main, 'top': ax_top, 'right': ax_right}, centroids


def plot_freq_peakrange_by_movement(df, animal=None, speed_vmax=50.0, moving_cmap='viridis',
                                    moving_alpha=0.5, immobile_color='black',
                                    immobile_edgecolor='#FFB3BA', figsize=(15, 5)):
    """Three side-by-side Fig. 4e/h-style panels -- mobility only, immobility
    only, and both merged -- sharing the same frequency/peak-range axes and
    bins so the two point clouds can first be inspected on their own, then
    compared for overlap in the merged panel. Immobile epochs are drawn with
    a pastel-red outline in every panel.

    Returns (fig, centroids_df) -- centroids_df is from the merged panel.
    """
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(1, 3, wspace=0.6)

    centroids = None
    for col, (mode, title) in enumerate((('moving', 'Mobility'),
                                         ('immobile', 'Immobility'),
                                         ('both', 'Merged'))):
        sub_gs = outer[0, col].subgridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                                           wspace=0.05, hspace=0.05)
        ax_main = fig.add_subplot(sub_gs[1, 0])
        ax_top = fig.add_subplot(sub_gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(sub_gs[1, 1], sharey=ax_main)

        sc, c = _draw_freq_peakrange_scatter(
            df, ax_main, ax_top, ax_right, animal=animal, mode=mode,
            speed_vmax=speed_vmax, moving_cmap=moving_cmap, moving_alpha=moving_alpha,
            immobile_color=immobile_color, immobile_edgecolor=immobile_edgecolor)
        ax_top.set_title(title)
        if sc is not None:
            cbar = fig.colorbar(sc, ax=ax_right, fraction=0.25, pad=0.45)
            cbar.set_label('moving speed (cm s$^{-1}$)', fontsize=7)
        if mode == 'both':
            centroids = c

    if animal:
        fig.suptitle(animal)

    return fig, centroids


def plot_figure4_style(df, animal=None, trace_color='tab:blue',
                       speed_vmax=50.0, moving_cmap='viridis',
                       moving_alpha=0.5, immobile_color='black',
                       immobile_edgecolor='#FFB3BA',
                       moving_row=None, immobile_row=None, figsize=(9, 4)):
    """Full Dunn et al. Fig. 4-style panel group: an example moving epoch
    (trace + autocorrelogram), an example immobile epoch (trace +
    autocorrelogram), and the frequency-vs-peak-range summary scatter with
    moving/immobile centroids -- i.e. the c/d/e (or f/g/h) panel set.

    If `moving_row` / `immobile_row` aren't given, the epoch within each
    movement group whose (freq, peakrangenorm) is closest to that group's
    own centroid is used (pick_example_epoch).

    Returns (fig, centroids_df).
    """
    clean = df[~df['skipped']]
    if moving_row is None:
        moving_row = pick_example_epoch(clean, 'moving', animal=animal)
    if immobile_row is None:
        immobile_row = pick_example_epoch(clean, 'immobile', animal=animal)

    moving_wave = get_epoch_waveform(moving_row['file'], moving_row['epoch_index'])
    immobile_wave = get_epoch_waveform(immobile_row['file'], immobile_row['epoch_index'])

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, width_ratios=(1, 1, 2.4), height_ratios=(1, 2.2),
                          wspace=0.45, hspace=0.15)

    ax_trace_m = fig.add_subplot(gs[0, 0])
    ax_acg_m = fig.add_subplot(gs[1, 0])
    plot_example_epoch(moving_row, moving_wave, ax_trace=ax_trace_m,
                       ax_acg=ax_acg_m, color=trace_color)

    ax_trace_i = fig.add_subplot(gs[0, 1])
    ax_acg_i = fig.add_subplot(gs[1, 1])
    plot_example_epoch(immobile_row, immobile_wave, ax_trace=ax_trace_i,
                       ax_acg=ax_acg_i, color=trace_color)

    sub_gs = gs[:, 2].subgridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                                  wspace=0.05, hspace=0.05)
    ax_main = fig.add_subplot(sub_gs[1, 0])
    ax_top = fig.add_subplot(sub_gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(sub_gs[1, 1], sharey=ax_main)
    sc, centroids = _draw_freq_peakrange_scatter(
        df, ax_main, ax_top, ax_right, animal=animal, speed_vmax=speed_vmax,
        moving_cmap=moving_cmap, moving_alpha=moving_alpha,
        immobile_color=immobile_color, immobile_edgecolor=immobile_edgecolor)

    cbar = fig.colorbar(sc, ax=ax_right, fraction=0.2, pad=0.35) # type: ignore
    cbar.set_label('moving speed (cm s$^{-1}$)')

    if animal:
        fig.suptitle(animal)

    return fig, centroids


# %% ==================== Driver =====================================================

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    results_df = process_root_directory(ROOT_DIR)

    results_path = os.path.join(OUTPUT_DIR, 'acg_theta_continuity_epochs.csv')
    results_df.to_csv(results_path, index=False)
    print(f'Saved {len(results_df)} epoch-level rows -> {results_path}')

    summary_df = summarize_peak_range(results_df)
    summary_path = os.path.join(OUTPUT_DIR, 'acg_theta_continuity_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f'Saved summary -> {summary_path}')

    all_centroids = []
    for animal_label in results_df['animal'].unique(): # type: ignore # type: ignore
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_moving_vs_immobile(results_df, animal_label, ax=ax)
        fig.tight_layout()
        fig_path = os.path.join(FIGURE_DIR, f'{animal_label}_acg_moving_vs_immobile.png')
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        print(f'Saved figure -> {fig_path}')

        # Fig. 4-style: example moving/immobile epochs + freq-vs-peak-range
        # scatter with movement centroids
        try:
            fig4, centroids = plot_figure4_style(results_df, animal=animal_label)
            centroids['animal'] = animal_label
            all_centroids.append(centroids)
            fig4_path = os.path.join(FIGURE_DIR, f'{animal_label}_figure4_style.png')
            fig4.savefig(fig4_path, dpi=200, bbox_inches='tight')
            plt.close(fig4)
            print(f'Saved figure -> {fig4_path}')
        except ValueError as e:
            print(f'  SKIP Fig.4-style plot for {animal_label}: {e}')

        # mobility-only / immobility-only / merged, side by side
        try:
            fig_split, _ = plot_freq_peakrange_by_movement(results_df, animal=animal_label)
            split_path = os.path.join(FIGURE_DIR, f'{animal_label}_freq_peakrange_by_movement.png')
            fig_split.savefig(split_path, dpi=200, bbox_inches='tight')
            plt.close(fig_split)
            print(f'Saved figure -> {split_path}')
        except ValueError as e:
            print(f'  SKIP mobility/immobility split plot for {animal_label}: {e}')

    if all_centroids:
        centroids_df = pd.concat(all_centroids, ignore_index=True)
        centroids_path = os.path.join(OUTPUT_DIR, 'acg_theta_continuity_centroids.csv')
        centroids_df.to_csv(centroids_path, index=False)
        print(f'Saved movement centroids -> {centroids_path}')
