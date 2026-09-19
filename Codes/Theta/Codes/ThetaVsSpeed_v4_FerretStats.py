"""
Theta frequency/power vs. running speed.

Recursively finds every Neuralynx .ncs LFP file under ROOT_DIR, pairs it with
the tracking .csv in its own folder, cleans the LFP the same way as
ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean_v3.py (resample -> notch ->
detrend -> low-pass), then band-pass filters to theta and estimates
instantaneous frequency/power using the peak-trough interpolation method of
Dunn et al. 2022 (Nature Communications 13:5905, the ferret theta paper) --
a port of the paper's own implementation
(calculate_peak_trough_signal_parameters.m / findMinMax.m /
clean_peaks_and_troughs.m, Toolbox/Signal-processing/,
https://github.com/slsdunn/theta-paper-code), binned against running speed
computed from the cleaned tracking position.

Before the mixed-model statistics, the per-time-bin (BINSIZE) Frequency/Power
values are further aggregated into 1 cm/s-wide speed bins per session
(representative Frequency/Power/Speed per bin -- see aggregate_by_speed_bin),
so the random-intercept LMM is fit on one value per distinct speed actually
visited per session rather than on thousands of highly autocorrelated time
samples. The representative value used for every bin (both the per-time-bin
aggregation and the per-speed-bin aggregation) is controlled by the
BIN_AGG_METHOD config option -- 'median' or 'mean'.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm, linregress
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import openpyxl


# %% ==================== Configuration ===========================================

# Root directory to search recursively for .ncs files. Every .ncs found
# anywhere under this tree (in any subfolder) is processed, paired with the
# single tracking .csv that lives in the same folder.
ROOT_DIR = r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True'

OUTPUT_PARQUET = os.path.join(ROOT_DIR, 'ThetaVsSpeed_out.parquet')

# Speed vs. theta stats (mixed-model plots + summary) are written here.
OUTPUT_DIR = Path(ROOT_DIR) / 'Output_ThetaVsSpeed'

# ---- Acquisition (Neuralynx .ncs) ----
NATIVE_FS = 32000  # native .ncs sampling rate (Hz)
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
LINE_HARMONICS = [50.0, 100.0, 150.0, 200.0]
NOTCH_Q = 30.0

# ---- Low-pass filtering (restrict the broadband trace to <100 Hz before
# the theta-band filter below, so high-frequency content can't distort peak
# finding) ----
LOWPASS_CUTOFF_HZ = 100.0
LOWPASS_ORDER = 4

# ---- Working sample rate for the cleaned LFP / theta filtering / peak
# finding (the raw .ncs is resampled down to this rate) ----
FS_LFP = 250

# ---- Theta band-pass (applied after the general LFP cleanup, on top of it) ----
BANDPASS_CUTOFF = (3.0, 7.0)
FILTER_ORDER = 5

# ---- Peak/trough detection for the instantaneous frequency/power estimate
# (see estimate_instantaneous_frequency_power) -- findMinMax.m's hysteresis
# threshold, as a fraction of the band-passed signal's median absolute
# amplitude. Dunn et al.'s own value ("chosen empirically"). ----
PEAK_THRESH_PC = 0.25

# ---- Edge trimming (seconds) -- drops filtfilt edge artifacts from the
# start/end of each cleaned trace ----
TRIM_DUR = 0.25

BINSIZE = 0.25

# Representative value computed for every bin -- both the per-time-bin
# (BINSIZE) aggregation in bin_data and the per-speed-bin aggregation in
# aggregate_by_speed_bin (the values the mixed model is fit on). 'median' or
# 'mean'.
BIN_AGG_METHOD = 'median'

# Speed-binning applied before the mixed model (see aggregate_by_speed_bin):
# with BINSIZE=0.25 s time bins, a session contributes thousands of
# Frequency/Power/Speed points, which are not independent samples of the
# speed relationship and inflate the mixed model's apparent significance.
# Grouping by SPEED_BIN_WIDTH_CMS-wide speed bins and taking the median
# Frequency/Power within each (session, bin) group gives one value per
# actually-distinct speed visited per session.
SPEED_BIN_WIDTH_CMS = 1.0

# ---- Tracking position cleaning (matches
# ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean_v3.py's
# _smooth_tracking_position): frame-to-frame jumps implying a speed above
# POS_JUMP_THRESH_CMS are tracking artifacts, linearly interpolated over,
# then the x/y traces are Gaussian-smoothed with sigma POS_SMOOTH_SIGMA_SMP
# (in samples). Tracking .csv column layout: UNIX timestamp in us (col A),
# x in cm (col D), y in cm (col E).
POS_JUMP_THRESH_CMS = 80.0
POS_SMOOTH_SIGMA_SMP = 1.0


# %% ==================== .ncs / tracking data import ==============================

def load_ncs(fpath):
    """Read a Neuralynx .ncs file.

    Returns
    -------
    lfp             : ndarray  raw trace in microvolts (uV)
    start_timestamp : int      UNIX timestamp (us) of the first sample -- same
                                clock as the tracking .csv.
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
    """Remove a polynomial trend (default: linear) from the full LFP trace."""
    return signal.detrend(np.asarray(x, dtype=np.float64), type=dtype)  # type: ignore


def lowpass_filter(x, fs_hz, cutoff_hz, order=4):
    """Zero-phase Butterworth low-pass filter (filtfilt)."""
    nyq = fs_hz / 2.0
    b, a = signal.butter(order, cutoff_hz / nyq, btype='low')
    return signal.filtfilt(b, a, np.asarray(x, dtype=np.float64))


def find_position_file(ncs_path):
    """Return the tracking .csv that lives alongside `ncs_path` (same folder).

    Every session folder holds exactly one tracking .csv shared by all its
    .ncs files. Returns None if none is found; if more than one is present,
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


def _smooth_tracking_position(x_cm, y_cm, t_us,
                              jump_thresh_cms=POS_JUMP_THRESH_CMS,
                              sigma_samples=POS_SMOOTH_SIGMA_SMP):
    """Clean and smooth the x/y tracking position before any downstream use
    (iterative jump removal + Gaussian smoothing)."""
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


def _instantaneous_speed(x_cm, y_cm, t_us):
    """Per-frame running speed (cm/s), padded to len(t_us) (repeats last value)."""
    dt_s = np.diff(t_us) * 1e-6
    with np.errstate(invalid='ignore', divide='ignore'):
        speed = np.hypot(np.diff(x_cm), np.diff(y_cm)) / dt_s
    speed[dt_s <= 0] = np.nan
    return np.append(speed, speed[-1])


def compute_velocity_from_position(csv_path):
    """Compute running speed (cm/s) from a tracking .csv.

    Column layout (positional): UNIX timestamp in us (col A), x in cm
    (col D), y in cm (col E). Position is cleaned first (jump removal +
    Gaussian smoothing, see `_smooth_tracking_position`), then speed is the
    frame-to-frame displacement of the cleaned trace divided by the actual
    elapsed time between samples.

    Returns
    -------
    time_us : ndarray   absolute UNIX timestamp of each tracking sample (us)
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


def clean_lfp(raw_lfp, fs_native=NATIVE_FS, fs_target=FS_LFP):
    """Resample -> notch -> detrend -> low-pass, same pipeline/order as
    ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean_v3.py's
    process_ncs_for_acg."""
    lfp = signal.resample_poly(raw_lfp, int(fs_target), int(fs_native))
    lfp = notch_filter(lfp, fs_target, LINE_HARMONICS, NOTCH_Q)
    lfp = detrend_signal(lfp, dtype='linear')
    lfp = lowpass_filter(lfp, fs_target, LOWPASS_CUTOFF_HZ, LOWPASS_ORDER)
    return lfp


# %% ==================== Theta filtering / peak-based freq & power =================

def filter_signal(lfp, fs, bandpass_cutoff, filter_order):
    b, a = signal.butter(filter_order, bandpass_cutoff, btype="band", fs=fs)
    filtered = signal.filtfilt(b, a, lfp)
    return filtered


def trim_signal(sig, fs, trim_dur):
    trim_samples = int(trim_dur * fs)
    return sig[trim_samples:-trim_samples]


# ============================================================================
# Peak-trough interpolation (Dunn et al. 2022, Nature Communications
# 13:5905, the ferret theta paper) -- instantaneous frequency/power from
# linear interpolation of peak-to-peak and trough-to-trough intervals of the
# band-passed signal, rather than an analytic-signal (Hilbert/Generalized
# Phase) estimate. Ported verbatim from
# calculate_peak_trough_signal_parameters.m / findMinMax.m /
# clean_peaks_and_troughs.m (Toolbox/Signal-processing/,
# https://github.com/slsdunn/theta-paper-code).
# ============================================================================

def _find_min_max(sig, thresh_pc):
    """findMinMax.m: hysteresis-based local peak/trough detector -- a new
    peak is only confirmed once the signal has since fallen at least
    `thresh_pc` * median(|sig|) below it (and symmetrically for troughs), so
    small ripples on the way to the real extremum don't register as spurious
    extrema of their own.

    Returns
    -------
    max_point, min_point : (n, 2) arrays of [sample_index, value], in
        detection order.
    min_change : the amplitude range (thresh_pc * median(|sig|)) used.
    """
    n = len(sig)
    valid = ~np.isnan(sig)
    if not valid.any():
        return np.empty((0, 2)), np.empty((0, 2)), np.nan

    min_change = np.nanmedian(np.abs(sig)) * thresh_pc

    first_valid = np.flatnonzero(valid)[0]
    next_max_y = sig[first_valid] + min_change
    next_min_y = sig[first_valid] - min_change

    looking_for = 1  # 1 = next confirmation will be a trough (tracking a running local max) -- 0 = next confirmation will be a peak
    local_max_y = local_max_yx = None
    local_min_y = local_min_yx = None
    max_point, min_point = [], []

    for x in range(n):
        y = sig[x]
        if np.isnan(y):
            continue

        if y > next_max_y:
            if looking_for == 1:
                if local_min_yx is None:
                    looking_for = 0
                    continue
                min_point.append((local_min_yx, local_min_y))
            next_min_y = y - min_change
            next_max_y = y + min_change
            looking_for = 0
            local_max_y, local_max_yx = y, x

        if local_max_yx is not None and local_max_y <= y:
            local_max_y, local_max_yx = y, x

        if y < next_min_y:
            if looking_for == 0:
                if local_max_yx is None:
                    looking_for = 1
                    continue
                max_point.append((local_max_yx, local_max_y))
            next_max_y = y + min_change
            next_min_y = y - min_change
            looking_for = 1
            local_min_y, local_min_yx = y, x

        if local_min_yx is not None and local_min_y >= y:
            local_min_y, local_min_yx = y, x

    max_arr = np.array(max_point, dtype=np.float64).reshape(-1, 2)
    min_arr = np.array(min_point, dtype=np.float64).reshape(-1, 2)
    return max_arr, min_arr, min_change


def _clean_peaks_troughs(sig, peaks, troughs):
    """clean_peaks_and_troughs.m: merge peaks+troughs into one time-ordered
    extrema list, drop peaks below zero and troughs above zero (detection
    artifacts), then resolve any two consecutive same-type extrema -- either
    a spurious double-detection (the weaker of the pair is dropped) or a
    genuine missed opposite-type extremum in between (inserted, at the
    sample where the signal peaks/troughs within that span).

    Returns
    -------
    extrema : (n, 3) array of [sample_index, value, kind], sorted by sample
        index; kind is 1 for a peak, -1 for a trough.
    """
    kind = np.concatenate([np.ones(len(peaks)), -np.ones(len(troughs))])
    extrema = np.column_stack([
        np.concatenate([peaks[:, 0], troughs[:, 0]]),
        np.concatenate([peaks[:, 1], troughs[:, 1]]),
        kind,
    ])

    remove_negative_peaks = (extrema[:, 1] < 0) & (extrema[:, 2] == 1)
    remove_positive_troughs = (extrema[:, 1] > 0) & (extrema[:, 2] == -1)
    extrema = extrema[~(remove_negative_peaks | remove_positive_troughs)]
    extrema = extrema[np.argsort(extrema[:, 0], kind='stable')]

    while True:
        same_kind = np.flatnonzero(np.diff(extrema[:, 2]) == 0)
        if same_kind.size == 0:
            break
        i = int(same_kind[0])
        i0, i1 = int(extrema[i, 0]), int(extrema[i + 1, 0])
        seg = sig[i0:i1 + 1]

        if extrema[i, 2] == 1:  # two consecutive peaks
            missed_val = np.min(seg)
            missed_idx = i0 + int(np.argmin(seg))
            missed_kind = -1
            if missed_val > 0:  # no real trough between them -- spurious double peak
                drop = i if extrema[i, 1] < extrema[i + 1, 1] else i + 1
                extrema = np.delete(extrema, drop, axis=0)
                continue
        else:  # two consecutive troughs
            missed_val = np.max(seg)
            missed_idx = i0 + int(np.argmax(seg))
            missed_kind = 1
            if missed_val < 0:  # no real peak between them -- spurious double trough
                drop = i if extrema[i, 1] > extrema[i + 1, 1] else i + 1
                extrema = np.delete(extrema, drop, axis=0)
                continue

        extrema = np.insert(extrema, i + 1, [missed_idx, missed_val, missed_kind], axis=0)

    return extrema


def _interp1_extrap_nan(xp, fp, x):
    """MATLAB interp1 default (linear, no extrapolation): NaN outside
    [xp[0], xp[-1]]."""
    if len(xp) < 2:
        return np.full(len(x), np.nan)
    return np.interp(x, xp, fp, left=np.nan, right=np.nan)


def _calc_peak_trough_params(extrema, t, nan_mask):
    """calc_peak_trough_params (nested in
    calculate_peak_trough_signal_parameters.m): instantaneous frequency and
    power from peak-to-peak / trough-to-trough interpolation.

    Each peak-to-peak (and, independently, trough-to-trough) interval gives
    one frequency estimate (1 / interval duration), assigned at the sample
    midway between the two peaks (avoiding the lag a peak-locked estimate
    would otherwise have); linearly interpolating that across every sample
    of `t`, doing the same for troughs, and averaging the two gives the
    instantaneous frequency trace. Power is the squared amplitude envelope,
    where the envelope at each sample is the mean of the (linearly
    interpolated) peak and trough amplitude traces.
    """
    peaks = extrema[extrema[:, 2] == 1]
    troughs = extrema[extrema[:, 2] == -1]

    peak_idx, trough_idx = peaks[:, 0].astype(int), troughs[:, 0].astype(int)
    peak_t, trough_t = t[peak_idx], t[trough_idx]
    peak_f = 1.0 / np.diff(peak_t)
    trough_f = 1.0 / np.diff(trough_t)

    # midpoint sample between consecutive peaks/troughs, used as the time
    # coordinate for that interval's frequency estimate
    peak_mid_idx = np.round(peak_idx[:-1] + 0.5 * np.diff(peak_idx)).astype(int)
    trough_mid_idx = np.round(trough_idx[:-1] + 0.5 * np.diff(trough_idx)).astype(int)

    peak_f_t = _interp1_extrap_nan(t[peak_mid_idx], peak_f, t)
    trough_f_t = _interp1_extrap_nan(t[trough_mid_idx], trough_f, t)
    with np.errstate(invalid='ignore'):
        instantaneous_freq = np.nanmean(np.column_stack([peak_f_t, trough_f_t]), axis=1)

    peak_amp_t = _interp1_extrap_nan(peak_t, peaks[:, 1], t)
    trough_amp_t = _interp1_extrap_nan(trough_t, troughs[:, 1], t)
    peak_amp_t[nan_mask] = np.nan
    trough_amp_t[nan_mask] = np.nan
    with np.errstate(invalid='ignore'):
        envelope = np.nanmean(np.column_stack([np.abs(peak_amp_t), np.abs(trough_amp_t)]), axis=1)
    instantaneous_power = envelope ** 2

    instantaneous_freq[nan_mask] = np.nan
    return instantaneous_freq, instantaneous_power


def estimate_instantaneous_frequency_power(theta, t, thresh_pc=PEAK_THRESH_PC):
    """Per-sample instantaneous theta frequency (Hz) and power (uV^2) of the
    band-passed signal `theta`, via the peak-trough interpolation method of
    Dunn et al. 2022 (see module header): local peaks/troughs are detected
    with a hysteresis threshold (`thresh_pc` * median(|theta|)), cleaned so
    they strictly alternate, and the frequency/amplitude of each peak-to-peak
    and trough-to-trough interval is linearly interpolated across every
    sample -- rather than read off an analytic-signal phase.
    """
    nan_mask = np.isnan(theta)
    max_point, min_point, _min_change = _find_min_max(theta, thresh_pc)
    extrema = _clean_peaks_troughs(theta, max_point, min_point)
    return _calc_peak_trough_params(extrema, t, nan_mask)


_BIN_AGG_FUNCS = {'median': np.nanmedian, 'mean': np.nanmean}


def bin_data(data, ts, bins, agg_method=BIN_AGG_METHOD):
    agg_func = _BIN_AGG_FUNCS[agg_method]
    bin_centers = (bins[:-1] + bins[1:])/2
    data_binned = []
    for i in range(len(bins)-1):
        bin_start = bins[i]
        bin_stop = bins[i+1]
        # speed values within the bin
        idx = np.where((ts>=bin_start)&(ts<=bin_stop))[0]
        # nan-aware: instantaneous_freq masks out biologically-impossible
        # (non-positive) samples as NaN -- a bin should fall back to
        # whichever of its samples are still valid rather than being
        # discarded outright over a few masked ones.
        with np.errstate(invalid='ignore'):
            data_binned.append(agg_func(data[idx]) if len(idx) else np.nan)
    return data_binned, bin_centers


def drop_nan(inst_freq_binned, inst_power_binned, inst_speed_binned, bin_centers):
    idx_nan = np.where(np.isnan(inst_speed_binned))
    inst_freq_binned = np.delete(inst_freq_binned, idx_nan)
    inst_power_binned = np.delete(inst_power_binned, idx_nan)
    inst_speed_binned = np.delete(inst_speed_binned, idx_nan)
    bin_centers = np.delete(bin_centers, idx_nan)

    idx_nan = np.where(np.isnan(inst_power_binned))
    inst_freq_binned = np.delete(inst_freq_binned, idx_nan)
    inst_power_binned = np.delete(inst_power_binned, idx_nan)
    inst_speed_binned = np.delete(inst_speed_binned, idx_nan)
    bin_centers = np.delete(bin_centers, idx_nan)
    assert len(inst_freq_binned) == len(inst_power_binned) == len(inst_speed_binned) == len(bin_centers)
    return inst_freq_binned, inst_power_binned, inst_speed_binned, bin_centers


# %% ==================== Per-file / root-directory processing =====================

def process_ncs_session(fpath):
    """Load one .ncs file + its folder's tracking .csv, clean/preprocess the
    LFP (resample -> notch -> detrend -> low-pass -> theta band-pass), and
    return a DataFrame of binned instantaneous theta frequency/power vs.
    running speed for that session.
    """
    raw_lfp, lfp_start_us = load_ncs(fpath)
    lfp = clean_lfp(raw_lfp)
    theta = filter_signal(lfp, FS_LFP, BANDPASS_CUTOFF, FILTER_ORDER)

    lfp_ts = lfp_start_us / 1e6 + np.arange(len(theta)) / FS_LFP  # absolute UNIX time (s)

    theta = trim_signal(theta, FS_LFP, TRIM_DUR)
    lfp_ts = trim_signal(lfp_ts, FS_LFP, TRIM_DUR)

    pos_path = find_position_file(fpath)
    if pos_path is None:
        raise ValueError('No tracking .csv found next to this file.')
    time_us, speed = compute_velocity_from_position(pos_path)
    speed_ts = time_us / 1e6  # absolute UNIX time (s), same clock as lfp_ts

    # restrict speed samples to the (trimmed) LFP epoch's own time window,
    # rather than assuming a fixed tracking frame rate
    in_window = (speed_ts >= lfp_ts.min()) & (speed_ts <= lfp_ts.max())
    inst_speed = speed[in_window]
    speed_ts = speed_ts[in_window]

    inst_freq, inst_power = estimate_instantaneous_frequency_power(theta, lfp_ts, PEAK_THRESH_PC)

    bin_start = lfp_ts.min().round()
    bin_stop = lfp_ts.max().round()
    bins = np.arange(bin_start, bin_stop + BINSIZE, step=BINSIZE)

    inst_speed_binned, _ = bin_data(inst_speed, speed_ts, bins)
    inst_freq_binned, _ = bin_data(inst_freq, lfp_ts, bins)
    inst_power_binned, bin_centers = bin_data(inst_power, lfp_ts, bins)

    inst_freq_binned, inst_power_binned, inst_speed_binned, bin_centers = drop_nan(
        np.asarray(inst_freq_binned), np.asarray(inst_power_binned),
        np.asarray(inst_speed_binned), np.asarray(bin_centers))

    return pd.DataFrame({
        "Power": inst_power_binned,
        "Frequency": inst_freq_binned,
        "Speed": inst_speed_binned,
        "Time": bin_centers,
    })


def process_root_directory(root_dir):
    """Recursively find every .ncs under `root_dir`, run process_ncs_session
    on each, and concatenate the per-bin results, tagging each row with a
    session id derived from the file's path (relative to `root_dir`)."""
    ncs_files = []
    for root, _dirs, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith('.ncs'):
                ncs_files.append(os.path.join(root, fname))
    print(f'Found {len(ncs_files)} .ncs files under {root_dir}')

    all_results = []
    for fpath in ncs_files:
        rel = os.path.relpath(fpath, root_dir)
        ses_id = os.path.splitext(rel)[0].replace(os.sep, '_')
        try:
            df_session = process_ncs_session(fpath)
            df_session['Session'] = ses_id
            print(f'  OK: {rel}  [{len(df_session)} bins]')
            all_results.append(df_session)
        except Exception as e:
            print(f'  SKIP: {rel} -- {e}')

    print(f'-> {len(all_results)} files processed\n')
    if not all_results:
        raise ValueError(f'No files processed successfully under {root_dir}')
    return pd.concat(all_results, ignore_index=True)


# %% ==================== Speed vs. theta statistics (mixed linear model) ==========

def aggregate_by_speed_bin(df, speed_bin_width=SPEED_BIN_WIDTH_CMS, group="Session",
                            agg_method=BIN_AGG_METHOD):
    """Representative (median or mean, per agg_method) Frequency/Power/Speed
    per (session, speed bin).

    Reduces the raw per-time-bin rows (one every BINSIZE seconds -- thousands
    per session) to one row per `speed_bin_width` cm/s speed bin actually
    visited within each session, so the mixed model is fit on a value that
    summarises a distinct speed rather than on many highly autocorrelated
    time samples. `Speed` in the output is the agg_method'd speed of the
    bin's contributing samples (not the bin edge/center), so it stays a
    faithful x-value for the fit.
    """
    d = df.copy()
    d["SpeedBin"] = np.floor(d["Speed"] / speed_bin_width)
    agg = (d.groupby([group, "SpeedBin"])
            .agg(Frequency=("Frequency", agg_method),
                 Power=("Power", agg_method),
                 Speed=("Speed", agg_method),
                 n=("Speed", "size"))
            .reset_index(drop=False))
    return agg.drop(columns="SpeedBin")


def fit_mixed_model(df, response, predictor="Speed", group="Session"):
    """
    Random-intercept linear mixed model: response ~ predictor, grouped by
    session, so repeated bins from the same session don't count as
    independent observations.
    """
    model = smf.mixedlm(f"{response} ~ {predictor}", df, groups=df[group])
    return model.fit()


def plot_mixed_model_fit(df, result, response, predictor, ylabel, title, color, out_path):
    """Scatter the binned data plus the mixed model's fixed-effect fit and its 95% CI band."""
    intercept = result.params["Intercept"]
    slope = result.params[predictor]
    pvalue = result.pvalues[predictor]
    cov = result.cov_params()

    x_range = np.linspace(df[predictor].min(), df[predictor].max(), 100)
    y_pred = intercept + slope * x_range
    se = np.sqrt(
        cov.loc["Intercept", "Intercept"]
        + x_range**2 * cov.loc[predictor, predictor]
        + 2 * x_range * cov.loc["Intercept", predictor]
    )
    z = norm.ppf(0.975)
    y_ci_low = y_pred - z * se
    y_ci_high = y_pred + z * se

    plt.figure(figsize=(7, 5))
    plt.scatter(df[predictor], df[response], color=color, edgecolor='k', alpha=0.6, s=25)
    plt.plot(x_range, y_pred, color="black", linewidth=2, label="Mixed Linear Fit")
    plt.fill_between(x_range, y_ci_low, y_ci_high, color="grey", alpha=0.5, label="95% CI")
    plt.xlabel(f"{predictor} (cm/s)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.gca().text(
        0.02, 0.98, f"β = {slope:.4g}\np = {pvalue:.3g}",
        transform=plt.gca().transAxes, ha='left', va='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='grey'))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_speed_vs_theta_stats(df):
    """Fit and plot mixed linear models of Frequency~Speed and Power~Speed
    (random intercept per Session), on speed-binned representative values
    (BIN_AGG_METHOD) rather than the raw per-time-bin rows -- see
    aggregate_by_speed_bin."""
    df_stats = df.dropna(subset=["Frequency", "Power", "Speed"])
    df_stats = aggregate_by_speed_bin(df_stats)
    df_stats.to_excel(OUTPUT_DIR / "ThetaVsSpeed_binned_medians.xlsx", index=False)

    freq_result = fit_mixed_model(df_stats, "Frequency")
    print(freq_result.summary())
    plot_mixed_model_fit(
        df_stats, freq_result, "Frequency", "Speed",
        ylabel="Theta Frequency [Hz]",
        title="Theta Frequency vs. Speed (Mixed Model)",
        color="red", out_path=OUTPUT_DIR / "ThetaFrequency_vs_Speed_MixedModel.png")

    power_result = fit_mixed_model(df_stats, "Power")
    print(power_result.summary())
    plot_mixed_model_fit(
        df_stats, power_result, "Power", "Speed",
        ylabel="Theta Power [µV²]",
        title="Theta Power vs. Speed (Mixed Model)",
        color="royalblue", out_path=OUTPUT_DIR / "ThetaPower_vs_Speed_MixedModel.png")

    with open(OUTPUT_DIR / "MixedModel_Stats_Summary.txt", "w") as f:
        f.write("=== Theta Frequency vs Speed ===\n")
        f.write(str(freq_result.summary()))
        f.write("\n\n=== Theta Power vs Speed ===\n")
        f.write(str(power_result.summary()))

    return freq_result, power_result


# %% ==================== Speed vs. theta statistics: per-session OLS (Dunn et al. Fig. 3 method) ==========
#
# Dunn et al. 2022 (Nat Commun 13:5905) did NOT use a mixed-effects model
# (lme4/stargazer) for the theta-frequency/power vs. speed relationship shown in
# their Fig. 3c,d,i,j and the "Speed vs freq./pow." columns of Fig. 3f,l. That LMM
# machinery (Stats/figure4_LMM.R, figure5_LMM.R, figure6_LMM.R in
# https://github.com/slsdunn/theta-paper-code) was used for a *different* analysis:
# their autocorrelation-based "peak range" theta-regularity metric vs. movement
# state / atropine / trial epoch (their Figs. 4-6). For the continuous
# speed-vs-frequency/power relationship, the Methods/legend text describes a plain
# per-session (per-channel) ordinary least-squares regression fit to speed-binned
# median frequency/power values ("Regression line fitted to median values ... in
# each speed bin"), summarised across sessions/channels as mean +/- SD of the fitted
# slope (beta_1) -- e.g. "rat: 0.026 +/- 0.007, ferret: 0.010 +/- 0.003" (main text)
# -- with individual channels/sessions called significant via a Bonferroni-corrected
# p-value (their Fig. 3 legend: p < 0.0016 = 0.05/32 channels).
#
# The MATLAB helper that actually performs this fit
# (linear_fit_of_table_or_struct_vars.m) is referenced by their plotting code
# (Paper-figure-plotting/supfigure2_depth_profiles.m) but is not included in the
# public repository -- only the calling/plotting code is. The implementation below
# follows the Methods text and figure legends directly, as a paper-faithful
# alternative to run_speed_vs_theta_stats's mixed model above (that mixed model
# pools all sessions into one fit with session as a random intercept and a single
# shared slope -- a different, not wrong, question from the paper's per-session
# slope distribution).

FIG3_SPEED_BIN_WIDTH_CMS = 5.0  # matches the paper's Fig. 3 speed binning ("5 cms-1 bins")
FIG3_SPEED_UPPER_PCTILE = 90.0  # paper excluded speeds above the 90th percentile per session


def fit_per_session_ols(df, response, predictor="Speed", group="Session"):
    """Per-session ordinary least-squares regression of `response` on `predictor`,
    one independent fit per session (no pooling), matching Dunn et al.'s Fig. 3
    approach. Returns one row per session with slope (beta1), intercept, R^2,
    p-value and the number of speed bins the fit used.
    """
    rows = []
    for ses_id, g in df.groupby(group):
        g = g.dropna(subset=[predictor, response])
        if len(g) < 3:
            continue
        slope, intercept, r_value, p_value, _std_err = linregress(g[predictor], g[response])
        rows.append({
            group: ses_id,
            "beta1": slope,
            "intercept": intercept,
            "r2": r_value ** 2,
            "pvalue": p_value,
            "n_bins": len(g),
        })
    return pd.DataFrame(rows)


def run_speed_vs_theta_persession_stats(df, n_channels_for_bonferroni=1):
    """Reproduce Dunn et al.'s Fig. 3 speed-theta statistics: bin each session's
    data into FIG3_SPEED_BIN_WIDTH_CMS-wide speed bins (median Frequency/Power per
    bin, after dropping speeds above the FIG3_SPEED_UPPER_PCTILE percentile within
    that session, as in the paper), fit one OLS regression per session (not a mixed
    model), and summarise the fitted slopes as mean +/- SD across sessions -- the
    number the paper actually reports in text -- plus the count/proportion of
    sessions individually significant at a Bonferroni-corrected threshold
    (0.05 / n_channels_for_bonferroni; the paper used 32, its probe's channel
    count -- leave this at 1 if you're not correcting across multiple
    simultaneously-recorded channels per session).
    """
    df_clean = df.dropna(subset=["Frequency", "Power", "Speed"]).copy()

    def _trim_top_pctile(g):
        thresh = np.percentile(g["Speed"], FIG3_SPEED_UPPER_PCTILE)
        return g[g["Speed"] <= thresh]
    df_clean = df_clean.groupby("Session", group_keys=False).apply(_trim_top_pctile)

    df_binned = aggregate_by_speed_bin(df_clean, speed_bin_width=FIG3_SPEED_BIN_WIDTH_CMS)
    df_binned.to_excel(OUTPUT_DIR / "Fig3_speed_binned_medians.xlsx", index=False)

    alpha_bonf = 0.05 / n_channels_for_bonferroni

    summary_lines = []
    for response in ("Frequency", "Power"):
        fits = fit_per_session_ols(df_binned, response)
        fits.to_excel(OUTPUT_DIR / f"Fig3_persession_OLS_{response}.xlsx", index=False)

        n_sig = int((fits["pvalue"] < alpha_bonf).sum())
        line = (f"{response} vs Speed (per-session OLS, n={len(fits)} sessions): "
                f"beta1 = {fits['beta1'].mean():.4g} +/- {fits['beta1'].std():.4g} (mean +/- SD), "
                f"{n_sig}/{len(fits)} sessions significant at Bonferroni p < {alpha_bonf:.4g}")
        print(line)
        summary_lines.append(line)

    with open(OUTPUT_DIR / "Fig3_persession_OLS_summary.txt", "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    return df_binned


# %% ==================== Driver =====================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = process_root_directory(ROOT_DIR)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f'Saved {len(df)} rows -> {OUTPUT_PARQUET}')

    if df["Session"].nunique() > 0:
        run_speed_vs_theta_stats(df)                    # mixed model (session as random intercept)
        run_speed_vs_theta_persession_stats(df)          # Dunn et al. Fig. 3's per-session OLS
    else:
        print('No sessions processed successfully -- skipping speed vs. theta statistics.')
