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
values are further aggregated into 4 cm/s-wide speed bins per session
(representative Frequency/Power per bin -- see aggregate_by_speed_bin), and
the random-intercept LMM is fit with each bin's center speed as the predictor,
i.e. one value per speed bin actually visited per session rather than
thousands of highly autocorrelated time samples. The representative value used for every bin (both the per-time-bin
aggregation and the per-speed-bin aggregation) is controlled by the
BIN_AGG_METHOD config option -- 'median' or 'mean'.

Theta power is analysed three ways (see add_normalized_power), each with its
own mixed-model / per-session-OLS figures:
  * Power    -- no normalization, squared envelope in uV^2;
  * PowerdB  -- decibel baseline normalization, 10*log10(P / B), B = the
                session's representative power while running at
                DB_BASELINE_SPEED_CMS (Cohen 2014, Analyzing Neural Time Series
                Data, ch. 18);
  * PowerRel -- relative power, theta power as % of the session's total
                REL_POWER_BAND power -- the time-domain (Parseval) equivalent
                of the PSD normalization in Fig4a_PSD_v2_Simul.py.

MOBILITY VARIANT (copy of ThetaVsSpeed_v8_Ferret_Final.py). Same pipeline,
except:
  * Immobility excluded: per-time-bin (BINSIZE) rows with running speed below
    MIN_SPEED_CMS (4 cm/s) are left out of every fit. They are still computed
    and kept in the parquet (column Mobile = False), and the mixed-model
    figures draw their speed bins in gray (EXCLUDED_COLOR), not in the fit.
  * dB baseline moved from 2-5 cm/s to DB_BASELINE_SPEED_CMS = 4-7 cm/s (the
    old window lies mostly below the 4 cm/s cut).
  * Dunn Fig. 3 per-session OLS: speeds < 4 cm/s are dropped first, then the
    per-session 90th-percentile cut is taken on the remaining (mobile) speeds.
  * Speed bins are SPEED_BIN_WIDTH_CMS = 4 cm/s wide in every analysis (mixed
    model, Dunn Fig. 3 per-session OLS, individual-file plots), anchored at 0:
    [0,4) is immobility and the first mobile bin is a full [4,8) cm/s.
    (Before: 2 cm/s for the mixed model and 5 cm/s for Fig. 3, whose first
    mobile bin held only 4-5 cm/s.)
  * Relative power denominator unchanged: whole-session 1-100 Hz power.
  * Copied .ncs files dropped (DROP_DUPLICATE_LFP): within a recording folder,
    files with identical record timestamps / valid-sample counts / samples are
    copies of one signal; only the first in natural name order is analysed.
  * Results are written to Results/ThetaVsSpeed/Mobility_v3 (v2 = 2 / 5 cm/s
    speed bins, PowerdB in the individual-file plots).
  * Individual-file correlations (run_individual_correlations): the Dunn
    Fig. 3 per-session OLS is drawn for every analysed .ncs file on its own
    (Frequency and PowerRel -- relative power -- vs speed)
    -> IndivdualCorrel/<session>.png. Each
    file's slope is classed as significantly positive / n.s. positive / n.s.
    negative / significantly negative (Benjamini-Hochberg FDR across files,
    per measure) or not fitted, and the % of all files in each class is
    plotted overall and per animal (IndividualCorrel_PercentPositive.png).
    The individual fits drop no upper tail of speeds (the group per-session
    OLS keeps the 90th-percentile cut); instead a speed bin is kept only with
    >= INDIV_MIN_TIME_PER_BIN_S (5 s) of data.
  * ADC saturation removed (REJECT_SATURATED_BINS), as in
    Figc_ACG_ScalingCorrected_v11_SaturationReject.py: the amplifier clips at
    +/-InputRange, read from each file's own .ncs header (1500 or 1000 uV).
    Clipped samples are found on the raw, repaired native-rate trace, before
    any filtering, and every clipped run is widened by SATURATION_MARGIN_S
    (0.2 s) on both sides, because filtering spreads the clip edges. Every
    0.25 s time bin (BINSIZE) that overlaps a widened run is removed (its
    Frequency/Power are set to NaN and the row is dropped with the other NaN
    bins), and the widened-run samples are left out of the relative-power
    denominator (TotalPower). Per-file counts -> Saturation_summary.xlsx.
"""

import os
import re
import hashlib
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm, linregress, ttest_1samp, wilcoxon
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import openpyxl

# ---- Timestamp-anchored .ncs loading ----
# ncs_repair.py corrects the partial-record and sample-loss timing error in the
# raw .ncs files (see its header). It sits in this script's folder; the absolute
# path also finds it when running cell-by-cell from another working directory.
import sys
NCS_REPAIR_DIR = r'C:\Runita\NMR\Mole-Rat_PlaceCell\Codes\Theta\Codes\ncsDropCorr'
if NCS_REPAIR_DIR not in sys.path:
    sys.path.insert(0, NCS_REPAIR_DIR)
from ncs_repair import load_repaired_uv, open_ncs, find_runs


# %% ==================== Configuration ===========================================

# Root directory to search recursively for .ncs files. Every .ncs found
# anywhere under this tree (in any subfolder) is processed, paired with the
# single tracking .csv that lives in the same folder.
ROOT_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final'

# Speed vs. theta stats (mixed-model plots + summary) are written here.
OUTPUT_DIR = Path(ROOT_DIR) / 'Results' / 'ThetaVsSpeed' / 'Mobility_v3'

# Per-time-bin rows of every analysed file (all speeds; Mobile flags the rows
# used in the fits).
OUTPUT_PARQUET = str(OUTPUT_DIR / 'ThetaVsSpeed_Mobility_out.parquet')

# ---- Immobility exclusion ----
# Time bins with running speed below MIN_SPEED_CMS are immobility: kept in the
# output and drawn in EXCLUDED_COLOR on the speed-axis figures, but not used
# in any fit (mixed model or per-session OLS).
MIN_SPEED_CMS = 4.0
EXCLUDED_COLOR = '#B0B0B0'

# ---- Copied LFP files ----
# Some recording folders hold bit-identical copies of one .ncs signal (placed
# so every tetrode had a matched LFP file). True: analyse only the first file
# of each set of copies (natural name order, e.g. CSC1 of CSC1 = CSC4).
DROP_DUPLICATE_LFP = True
# Run metadata (all settings, data/script paths) + a copy of this script
# are written here -- see save_run_metadata.
META_DIR = OUTPUT_DIR / 'meta'

# ---- Acquisition (Neuralynx .ncs) ----
NATIVE_FS = 32000  # native .ncs sampling rate (Hz)
# V per ADC count is read from each file's own header (-ADBitVolts, see
# read_ncs_adbitvolts): it follows the recording's InputRange, which differs
# between animals (1500 uV -> 4.58e-8 V, 1000 uV -> 3.05e-8 V), so a single
# hard-coded constant would scale animals' power differently.

# Neuralynx .ncs record format (512 int16 samples per record, 16 kB header skipped)
ncs_dtype = np.dtype([
    ('timestamp'  , '<u8'),
    ('sc_number'  , '<u4'),
    ('cell_number', '<u4'),
    ('params'     , '<u4'),
    ('samples'    , '<i2', (512,)),
])

# ---- ADC saturation (as in Figc_ACG_ScalingCorrected_v11_SaturationReject.py) ----
# The clipping level is each file's own -InputRange header value (uV), never a
# fixed number: 1500 uV for some recordings, 1000 uV for others. Clipped samples
# are stored at ADMaxValue = 32767 counts, so a sample counts as clipped if
# |value| >= InputRange - SATURATION_TOL_STEPS ADC steps (one step = ADBitVolts);
# 1.5 steps flags exactly the +/-32767 rail and is safe against rounding.
REJECT_SATURATED_BINS = True     # True: drop every time bin within SATURATION_MARGIN_S of clipping
SATURATION_TOL_STEPS  = 1.5      # ADC steps below +/-InputRange that still count as clipped
# Each clipped run is widened by this much on both sides before time bins are
# rejected, since filtering spreads the sharp clip edges into neighbouring samples.
SATURATION_MARGIN_S   = 0.2      # s, either side of each clipped run

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
FS_LFP = 1000

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
# Grouping by SPEED_BIN_WIDTH_CMS-wide speed bins ([0,4), [4,8), ... cm/s) and
# taking the median Frequency/Power within each (session, bin) group gives one
# value per speed bin visited per session; the mixed model's predictor is the
# bin center (2, 6, 10, ... cm/s). The same width is used by the Fig. 3
# per-session OLS and the individual-file plots (FIG3_SPEED_BIN_WIDTH_CMS).
SPEED_BIN_WIDTH_CMS = 4.0

# ---- Theta-power normalization (see add_normalized_power) ----
# dB baseline: each session's baseline B is the BIN_AGG_METHOD power of its
# time bins with speed in [lo, hi) cm/s. A fixed speed window every session
# covers keeps the baseline from depending on how much fast running a
# session happens to contain (as a session-wide baseline would). The v8
# window (2-5 cm/s) lies mostly below MIN_SPEED_CMS, so it is moved to the
# slowest mobile speeds, keeping its 3 cm/s width. Per-session coverage is
# written to dB_baseline_coverage.xlsx.
DB_BASELINE_SPEED_CMS = (4.0, 7.0)
# Relative power: band whose total power (mean square of the band-passed
# LFP over the whole session) is the denominator -- matches NORM_BAND in
# Fig4a_PSD_v2_Simul.py.
REL_POWER_BAND = (1.0, 100.0)
REL_POWER_FILTER_ORDER = 4

# (column, file-name suffix, y-axis label, title, slope unit) per power measure
POWER_MEASURES = [
    ("Power", "", "Theta Power [µV²]",
     "Theta Power vs. Speed (no normalization)", "µV²"),
    ("PowerdB", "_dB",
     f"Theta Power [dB re {DB_BASELINE_SPEED_CMS[0]:g}–{DB_BASELINE_SPEED_CMS[1]:g} cm/s]",
     "Theta Power vs. Speed (dB baseline)", "dB"),
    ("PowerRel", "_Relative",
     f"Relative Theta Power [% of {REL_POWER_BAND[0]:g}–{REL_POWER_BAND[1]:g} Hz power]",
     "Theta Power vs. Speed (relative power)", "%"),
]

# ---- Tracking position cleaning (matches
# ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean_v3.py's
# _smooth_tracking_position): frame-to-frame jumps implying a speed above
# POS_JUMP_THRESH_CMS are tracking artifacts, linearly interpolated over,
# then the x/y traces are Gaussian-smoothed with sigma POS_SMOOTH_SIGMA_SMP
# (in samples). Tracking .csv column layout: UNIX timestamp in us (col A),
# x in cm (col D), y in cm (col E).
POS_JUMP_THRESH_CMS = 90.0
POS_SMOOTH_SIGMA_SMP = 1.0

# Per-animal colors — first 4 colors of matplotlib's Pastel1 palette.
ANIMAL_COLORS = {
    'Fa8477': '#FBB4AE',  # Pastel1[0] red
    'Fa1059': '#B3CDE3',  # Pastel1[1] blue
    'Fa23BD': '#CCEBC5',  # Pastel1[2] green
    'Fa5834': '#DECBE4',  # Pastel1[3] purple
}


def animal_color(label, default='#888888'):
    """ANIMAL_COLORS entry for an animal label, matched on the ID so that e.g.
    'Fa1059', '1059Test' and 'FA1059' all resolve; `default` if no ID matches."""
    label = str(label).upper()
    for animal_id, color in ANIMAL_COLORS.items():
        if animal_id[2:].upper() in label:
            return color
    return default


def ordered_animals(labels):
    """Unique animal labels in ANIMAL_COLORS order (unknown labels last, sorted)."""
    labels = list(pd.unique(pd.Series(labels).dropna()))
    known = [a for a in ANIMAL_COLORS if a in labels]
    return known + sorted(a for a in labels if a not in known)


# %% ==================== .ncs / tracking data import ==============================

def read_ncs_adbitvolts(fpath):
    """V per ADC count, from the '-ADBitVolts' line of the 16 kB .ncs header."""
    with open(fpath, 'rb') as f:
        header = f.read(16 * 1024).decode('latin-1')
    m = re.search(r'-ADBitVolts\s+(\S+)', header)
    if m is None:
        raise ValueError('No -ADBitVolts entry in the .ncs header.')
    return float(m.group(1))


def read_ncs_input_range(fpath):
    """Amplifier input range (uV) from the '-InputRange' line of the .ncs header,
    i.e. the level at which the recorded signal clips. None if the header has no
    InputRange (simulated files), in which case no saturation check is possible."""
    with open(fpath, 'rb') as f:
        header = f.read(16 * 1024).decode('latin-1')
    m = re.search(r'^-InputRange[ \t]+(\S+)', header, flags=re.MULTILINE)
    return float(m.group(1)) if m else None


def saturated_samples(lfp_uv, input_range_uv, ad_bit_volts, tol_steps=SATURATION_TOL_STEPS):
    """Boolean mask of clipped samples of the raw, native-rate trace (uV): those
    within `tol_steps` ADC steps of +/-`input_range_uv`. All False if the file
    has no InputRange."""
    if input_range_uv is None:
        return np.zeros(len(lfp_uv), dtype=bool)
    rail_uv = input_range_uv - tol_steps * ad_bit_volts * 1e6
    return np.abs(lfp_uv) >= rail_uv


def near_clip_mask(sat_native, n_out, fs_native=NATIVE_FS, fs_out=FS_LFP,
                   margin_s=SATURATION_MARGIN_S):
    """`sat_native` (native-rate clipped-sample mask) mapped onto the first
    `n_out` samples of the fs_out trace that clean_lfp returns (an fs_out sample
    is clipped if any of the native samples it covers is), then every clipped
    run widened by `margin_s` on both sides."""
    step = int(round(fs_native / fs_out))
    n_full = -(-len(sat_native) // step) * step
    padded = np.zeros(n_full, dtype=bool)
    padded[:len(sat_native)] = sat_native
    sat = padded.reshape(-1, step).any(axis=1)
    sat = np.pad(sat, (0, max(0, n_out - len(sat))))[:n_out]

    starts, lengths = find_runs(sat)
    margin = int(round(margin_s * fs_out))
    edges = np.zeros(n_out + 1, dtype=np.int64)
    np.add.at(edges, np.clip(starts - margin, 0, n_out), 1)
    np.add.at(edges, np.clip(starts + lengths + margin, 0, n_out), -1)
    return np.cumsum(edges[:-1]) > 0


def load_ncs(fpath):
    """Read a Neuralynx .ncs file.

    Returns
    -------
    lfp             : ndarray  repaired trace in microvolts (uV), on a uniform
                                NATIVE_FS grid (see ncs_repair.py)
    start_timestamp : int      UNIX timestamp (us) of the first sample -- same
                                clock as the tracking .csv.
    ad_bit_volts    : float    V per ADC count (this file's -ADBitVolts)
    """
    ad_bit_volts = read_ncs_adbitvolts(fpath)
    lfp, start_us = load_repaired_uv(fpath, ad_bit_volts)
    return lfp, start_us, ad_bit_volts


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
    # second-order sections, not (b, a): a narrow (3-7 Hz) order-5 band-pass at
    # 1 kHz has poles so close to the unit circle that the (b, a) polynomial
    # form rounds them outside it, and filtfilt then blows up to an all-NaN trace
    sos = signal.butter(filter_order, bandpass_cutoff, btype="band", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, lfp)
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
    # samples outside the first/last interval midpoint are NaN in both traces;
    # nanmean correctly returns NaN there but emits a RuntimeWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        instantaneous_freq = np.nanmean(np.column_stack([peak_f_t, trough_f_t]), axis=1)

    peak_amp_t = _interp1_extrap_nan(peak_t, peaks[:, 1], t)
    trough_amp_t = _interp1_extrap_nan(trough_t, troughs[:, 1], t)
    peak_amp_t[nan_mask] = np.nan
    trough_amp_t[nan_mask] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
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
        # an all-NaN bin correctly yields NaN, but nanmedian/nanmean warn about it
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            data_binned.append(agg_func(data[idx]) if len(idx) else np.nan)
    return data_binned, bin_centers


def bin_any(mask, ts, bins):
    """Per bin of `bins` (same [start, stop] inclusive edges as bin_data):
    True if any sample of `mask` with time `ts` (sorted) falls in it."""
    cum = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
    lo = np.searchsorted(ts, bins[:-1], side='left')
    hi = np.searchsorted(ts, bins[1:], side='right')
    return (cum[hi] - cum[lo]) > 0


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

    With REJECT_SATURATED_BINS, time bins within SATURATION_MARGIN_S of an ADC
    clipped sample are removed (see module header).

    Returns (per-time-bin DataFrame, saturation summary dict).
    """
    raw_lfp, lfp_start_us, ad_bit_volts = load_ncs(fpath)
    # clipped samples are found on the raw native-rate trace, before any filtering
    input_range_uv = read_ncs_input_range(fpath)
    sat_native = saturated_samples(raw_lfp, input_range_uv, ad_bit_volts)
    lfp = clean_lfp(raw_lfp)
    near_clip = near_clip_mask(sat_native, len(lfp))
    if not REJECT_SATURATED_BINS:
        near_clip = np.zeros(len(lfp), dtype=bool)
    theta = filter_signal(lfp, FS_LFP, BANDPASS_CUTOFF, FILTER_ORDER)
    broadband = filter_signal(lfp, FS_LFP, REL_POWER_BAND, REL_POWER_FILTER_ORDER)

    lfp_ts = lfp_start_us / 1e6 + np.arange(len(theta)) / FS_LFP  # absolute UNIX time (s)

    theta = trim_signal(theta, FS_LFP, TRIM_DUR)
    lfp_ts = trim_signal(lfp_ts, FS_LFP, TRIM_DUR)
    near_clip = trim_signal(near_clip, FS_LFP, TRIM_DUR)
    # session's total REL_POWER_BAND power (uV^2): by Parseval, the mean square
    # of the band-passed trace equals the integral of its PSD over that band.
    # Samples within SATURATION_MARGIN_S of clipping are left out.
    broadband = trim_signal(broadband, FS_LFP, TRIM_DUR)
    total_power = np.mean(broadband[~near_clip] ** 2) if (~near_clip).any() else np.nan

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
    inst_freq_binned = np.asarray(inst_freq_binned)
    inst_power_binned = np.asarray(inst_power_binned)
    inst_speed_binned = np.asarray(inst_speed_binned)

    # remove time bins within SATURATION_MARGIN_S of clipping: NaN here, then
    # dropped by drop_nan with the other NaN bins
    bin_sat = bin_any(near_clip, lfp_ts, bins)
    n_valid_before = int((np.isfinite(inst_power_binned) & np.isfinite(inst_speed_binned)).sum())
    inst_freq_binned[bin_sat] = np.nan
    inst_power_binned[bin_sat] = np.nan

    inst_freq_binned, inst_power_binned, inst_speed_binned, bin_centers = drop_nan(
        inst_freq_binned, inst_power_binned, inst_speed_binned, np.asarray(bin_centers))

    n_clipped = int(sat_native.sum())
    sat_info = {
        "InputRange_uV": input_range_uv,
        "n_clipped_samples": n_clipped,
        "pct_samples_clipped": 100 * n_clipped / len(sat_native) if len(sat_native) else np.nan,
        "n_clipped_runs": len(find_runs(sat_native)[0]),
        "pct_time_near_clip": 100 * near_clip.mean() if len(near_clip) else np.nan,
        "n_bins_before": n_valid_before,
        "n_bins_removed_saturation": n_valid_before - len(bin_centers),
        "n_bins_kept": len(bin_centers),
    }

    return pd.DataFrame({
        "Power": inst_power_binned,
        "Frequency": inst_freq_binned,
        "Speed": inst_speed_binned,
        "Time": bin_centers,
        "TotalPower": total_power,
    }), sat_info


def _natural_key(path):
    """'CSC10.ncs' after 'CSC2.ncs' (numbers compared as numbers)."""
    return tuple((0, int(t), '') if t.isdigit() else (1, 0, t.lower())
                 for t in re.split(r'(\d+)', os.path.basename(str(path))) if t)


def ncs_signal_signature(fpath):
    """SHA-1 of the file's ADBitVolts and every record's timestamp, valid-sample
    count and samples -- the header text and the channel-number field are left
    out, so a copy of a signal saved under another channel name still matches."""
    rec = open_ncs(fpath)
    h = hashlib.sha1()
    h.update(repr(read_ncs_adbitvolts(fpath)).encode())
    for field in ('timestamp', 'n_valid', 'samples'):
        h.update(np.ascontiguousarray(rec[field]).tobytes())
    del rec
    return h.hexdigest()


def find_duplicate_ncs(ncs_files, root_dir):
    """Within each recording folder, group .ncs files with identical
    ncs_signal_signature; the first of each group (natural name order) is kept.
    Returns (files to analyse, table of the dropped copies)."""
    rows = []
    for fpath in ncs_files:
        rows.append({'folder': os.path.dirname(fpath), 'file': fpath,
                     'signature': ncs_signal_signature(fpath)})
    files = pd.DataFrame(rows)
    files['kept_file'] = files.groupby(['folder', 'signature'])['file'].transform('first')
    dups = files[files['file'] != files['kept_file']].copy()
    n_sets = files.groupby(['folder', 'signature']).ngroups
    print(f'Duplicate LFP files: {len(files)} files, {n_sets} distinct signals; '
          f'{len(dups)} copies in {dups["folder"].nunique()} folders dropped '
          f'(first file of each set kept).')
    dups = pd.DataFrame({
        'Recording': dups['folder'].map(lambda p: os.path.relpath(p, root_dir).replace(os.sep, '/')),
        'file_name': dups['file'].map(os.path.basename),
        'copy_of':   dups['kept_file'].map(os.path.basename),
    }).reset_index(drop=True)
    kept = files.loc[files['file'] == files['kept_file'], 'file'].tolist()
    return kept, dups


def process_root_directory(root_dir):
    """Recursively find every .ncs under `root_dir`, run process_ncs_session
    on each, and concatenate the per-bin results, tagging each row with a
    session id derived from the file's path (relative to `root_dir`), plus the
    recording folder it came from (channels in the same folder were recorded
    simultaneously) and the animal (first path component).

    With DROP_DUPLICATE_LFP, copied files are found first (find_duplicate_ncs)
    and skipped; the dropped copies are written to OUTPUT_DIR."""
    ncs_files = []
    for root, _dirs, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith('.ncs'):
                ncs_files.append(os.path.join(root, fname))
    ncs_files.sort(key=lambda p: (os.path.dirname(p), _natural_key(p)))
    print(f'Found {len(ncs_files)} .ncs files under {root_dir}')

    if DROP_DUPLICATE_LFP:
        ncs_files, dups = find_duplicate_ncs(ncs_files, root_dir)
        dups.to_excel(OUTPUT_DIR / 'Dropped_duplicate_ncs.xlsx', index=False)

    all_results, sat_rows = [], []
    for fpath in ncs_files:
        rel = os.path.relpath(fpath, root_dir)
        ses_id = os.path.splitext(rel)[0].replace(os.sep, '_')
        rel_dir = os.path.dirname(rel)
        try:
            df_session, sat = process_ncs_session(fpath)
            df_session['Session'] = ses_id
            df_session['Recording'] = rel_dir.replace(os.sep, '/')
            df_session['Animal'] = rel_dir.split(os.sep)[0]
            if sat['InputRange_uV'] is None:
                sat_msg = 'no -InputRange in header, saturation check skipped'
            else:
                sat_msg = (f'InputRange {sat["InputRange_uV"]:g} uV, '
                           f'{sat["pct_samples_clipped"]:.3f}% samples clipped, '
                           f'{sat["n_bins_removed_saturation"]}/{sat["n_bins_before"]} bins removed '
                           f'within {SATURATION_MARGIN_S:g} s of clipping')
            print(f'  OK: {rel}  [{len(df_session)} bins; {sat_msg}]')
            all_results.append(df_session)
            sat_rows.append({'Session': ses_id, 'Animal': rel_dir.split(os.sep)[0],
                             'Recording': rel_dir.replace(os.sep, '/'),
                             'file_name': os.path.basename(fpath), **sat})
        except Exception as e:
            print(f'  SKIP: {rel} -- {e}')

    print(f'-> {len(all_results)} files processed\n')
    if not all_results:
        raise ValueError(f'No files processed successfully under {root_dir}')

    sat_df = pd.DataFrame(sat_rows)
    sat_df.to_excel(OUTPUT_DIR / 'Saturation_summary.xlsx', index=False)
    n_before, n_removed = sat_df['n_bins_before'].sum(), sat_df['n_bins_removed_saturation'].sum()
    print(f'ADC saturation: {int((sat_df["n_clipped_samples"] > 0).sum())}/{len(sat_df)} files contain '
          f'clipped samples; {n_removed}/{n_before} time bins ({100 * n_removed / max(n_before, 1):.2f}%) '
          f'removed within {SATURATION_MARGIN_S:g} s of clipping'
          + ('' if REJECT_SATURATED_BINS else ' [REJECT_SATURATED_BINS = False: nothing removed]'))
    return pd.concat(all_results, ignore_index=True)


def add_normalized_power(df, group="Session"):
    """Add the two normalized theta-power columns (see POWER_MEASURES) to the
    per-time-bin rows; `Power` itself is left in uV^2.

    PowerdB  = 10*log10(Power / B), B = the session's BIN_AGG_METHOD Power over
               time bins with speed in DB_BASELINE_SPEED_CMS. Normalizing per
               session turns each recording's multiplicative gain (electrode
               placement/impedance) into 0 dB at the baseline speed, so
               sessions are compared on change rather than absolute power.
               The median of dB values equals the dB of the median, so the
               later speed-bin aggregation is unaffected by the order.
    PowerRel = 100 * (Power / 2) / TotalPower. Power is the squared peak/trough
               amplitude; a sinusoid of amplitude A has mean square A^2/2,
               so Power/2 is the theta mean square that TotalPower (mean
               square of the REL_POWER_BAND trace) is directly comparable to --
               i.e. the theta-band integral of Fig4a_PSD's normalized PSD.

    Sessions without any baseline-speed bins get NaN PowerdB (and are
    dropped from the dB fits only).
    """
    d = df.copy()
    lo, hi = DB_BASELINE_SPEED_CMS
    in_baseline = (d["Speed"] >= lo) & (d["Speed"] < hi)
    baseline = d[in_baseline].groupby(group)["Power"].agg(BIN_AGG_METHOD)
    d["PowerBaseline"] = d[group].map(baseline)
    with np.errstate(divide='ignore', invalid='ignore'):
        d["PowerdB"] = 10 * np.log10(d["Power"] / d["PowerBaseline"])
    d["PowerdB"] = d["PowerdB"].replace([np.inf, -np.inf], np.nan)
    d["PowerRel"] = 100 * (d["Power"] / 2) / d["TotalPower"]

    n_missing = d.loc[d["PowerBaseline"].isna(), group].nunique()
    if n_missing:
        print(f'  {n_missing} sessions have no {lo:g}-{hi:g} cm/s bins -- no dB baseline, '
              f'excluded from the dB fits')
    return d


def save_baseline_coverage(df, group="Session"):
    """Per session: seconds of running in the dB baseline window
    (DB_BASELINE_SPEED_CMS), in the v8 window (2-5 cm/s) for comparison, below
    MIN_SPEED_CMS and in total, plus the baseline power -> dB_baseline_coverage.xlsx."""
    def secs(mask):
        return df[mask].groupby(group).size() * BINSIZE
    lo, hi = DB_BASELINE_SPEED_CMS
    s = df["Speed"]
    cov = pd.DataFrame({
        f"s_{lo:g}_{hi:g}cms (baseline)": secs((s >= lo) & (s < hi)),
        "s_2_5cms (v8 baseline)":          secs((s >= 2) & (s < 5)),
        f"s_below_{MIN_SPEED_CMS:g}cms":   secs(s < MIN_SPEED_CMS),
        "s_total":                         secs(s.notna()),
    }).fillna(0)
    cov["PowerBaseline_uV2"] = df.groupby(group)["PowerBaseline"].first()
    meta = df.drop_duplicates(group).set_index(group)[["Animal", "Recording"]]
    cov = meta.join(cov).reset_index()
    cov.to_excel(OUTPUT_DIR / "dB_baseline_coverage.xlsx", index=False)
    b = cov[f"s_{lo:g}_{hi:g}cms (baseline)"]
    print(f'  dB baseline {lo:g}-{hi:g} cm/s: median {b.median():.0f} s per session '
          f'(min {b.min():.0f} s); v8 2-5 cm/s: median {cov["s_2_5cms (v8 baseline)"].median():.0f} s')
    return cov


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
    bin's contributing samples; `SpeedBinCenter` is the bin's center
    ((k + 0.5) * speed_bin_width for bin [k*w, (k+1)*w)).
    """
    d = df.copy()
    d["SpeedBin"] = np.floor(d["Speed"] / speed_bin_width)
    # Recording/Animal are constant within a session -- carried through for plotting
    carry = [c for c in ("Recording", "Animal") if c in d.columns]
    values = [c for c in ("Frequency", *(m[0] for m in POWER_MEASURES)) if c in d.columns]
    agg = (d.groupby([group, *carry, "SpeedBin"])
            .agg(**{c: (c, agg_method) for c in values},
                 Speed=("Speed", agg_method),
                 n=("Speed", "size"))
            .reset_index(drop=False))
    agg["SpeedBinCenter"] = (agg["SpeedBin"] + 0.5) * speed_bin_width
    return agg.drop(columns="SpeedBin")


def fit_mixed_model(df, response, predictor="Speed", group="Session"):
    """
    Random-intercept linear mixed model: response ~ predictor, grouped by
    session, so repeated bins from the same session don't count as
    independent observations.
    """
    model = smf.mixedlm(f"{response} ~ {predictor}", df, groups=df[group])
    return model.fit()


def _axes_y_below(fig, ax, artist=None, pad=0.03):
    """y (in `ax` axes coords) just below `artist` -- or, if None, below
    everything already drawn on `ax` (tick labels, xlabel) -- so the next
    element placed there can't overlap it."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = artist.get_window_extent(renderer) if artist is not None else ax.get_tightbbox(renderer)
    return ax.transAxes.inverted().transform((0, bb.y0))[1] - pad


def add_details_below(fig, ax, text, legend_handles=None, legend_ncol=3, fontsize=9):
    """Put the legend (optional) and then the stats `text` underneath the
    axes, stacked so neither overlaps the data, the axis labels or each
    other. Save with bbox_inches='tight' so they're kept in the image."""
    anchor = None
    if legend_handles:
        anchor = ax.legend(handles=legend_handles, loc='upper center',
                           bbox_to_anchor=(0.5, _axes_y_below(fig, ax)),
                           ncol=legend_ncol, fontsize=fontsize, frameon=False)
    ax.text(0.5, _axes_y_below(fig, ax, anchor), text, transform=ax.transAxes,
            ha='center', va='top', fontsize=fontsize, linespacing=1.4)


def plot_mixed_model_fit(df, result, response, predictor, ylabel, title, color, out_path,
                         xlabel="Speed bin center (cm/s)", group="Session", df_excluded=None):
    """Scatter the binned data plus the mixed model's fixed-effect fit and its 95% CI band.
    Points are coloured by animal (ANIMAL_COLORS) when `df` has an Animal
    column, otherwise all drawn in `color`. `df_excluded` (immobility speed
    bins, not in the fit) is drawn in EXCLUDED_COLOR. The fit line covers only
    the fitted speed range. Legend and all model statistics are written below
    the plot."""
    intercept = result.params["Intercept"]
    slope = result.params[predictor]
    pvalue = result.pvalues[predictor]
    slope_se = result.bse[predictor]
    slope_z = result.tvalues[predictor]
    slope_ci_low, slope_ci_high = result.conf_int().loc[predictor]
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

    fig, ax = plt.subplots(figsize=(7, 5))
    handles = []
    n_excluded = 0 if df_excluded is None else len(df_excluded)
    if n_excluded:
        handles.append(ax.scatter(df_excluded[predictor], df_excluded[response], color=EXCLUDED_COLOR,
                                  edgecolor='#707070', linewidth=0.5, alpha=0.8, s=25,
                                  label=f"< {MIN_SPEED_CMS:g} cm/s (not in fit)"))
        ax.axvline(MIN_SPEED_CMS, color='#707070', linestyle=':', linewidth=1)
    if "Animal" in df.columns:
        for a in ordered_animals(df["Animal"]):
            sub = df[df["Animal"] == a]
            handles.append(ax.scatter(sub[predictor], sub[response], color=animal_color(a, default=color),
                                      edgecolor='k', linewidth=0.5, alpha=0.8, s=25, label=a))
    else:
        handles.append(ax.scatter(df[predictor], df[response], color=color, edgecolor='k',
                                  alpha=0.6, s=25, label="Speed bins"))
    handles += ax.plot(x_range, y_pred, color="black", linewidth=2, label="Mixed Linear Fit")
    handles.append(ax.fill_between(x_range, y_ci_low, y_ci_high, color="grey", alpha=0.5, label="95% CI"))
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

    n_animals = df["Animal"].nunique() if "Animal" in df.columns else np.nan
    details = "\n".join([
        f"Model: {response} ~ {predictor} + (1 | {group})   [linear mixed model, REML]",
        f"Slope β = {slope:.4g} ± {slope_se:.3g} SE,  95% CI [{slope_ci_low:.4g}, {slope_ci_high:.4g}],  "
        f"z = {slope_z:.2f},  " + (f"p = {pvalue:.3g}" if pvalue > 0 else "p < 1e-300"),
        f"Intercept = {intercept:.4g};  random-intercept var = {result.cov_re.iloc[0, 0]:.3g};  "
        f"residual var = {result.scale:.3g};  converged = {result.converged}",
        f"n = {result.nobs} speed bins from {df[group].nunique()} sessions, {n_animals} animals;  "
        f"bin width = {SPEED_BIN_WIDTH_CMS:g} cm/s,  value per bin = {BIN_AGG_METHOD}",
        f"Gray: {n_excluded} immobility speed bins (time bins < {MIN_SPEED_CMS:g} cm/s), "
        f"shown but excluded from the fit",
    ])
    add_details_below(fig, ax, details, legend_handles=handles, legend_ncol=4 if n_excluded else 3)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


def run_speed_vs_theta_stats(df):
    """Fit and plot mixed linear models of Frequency~Speed and, for each
    POWER_MEASURES column, Power~Speed (random intercept per Session), on
    SPEED_BIN_WIDTH_CMS-wide speed bins: per (session, bin) the representative
    (BIN_AGG_METHOD) Frequency/Power, against the bin's center speed -- rather
    than the raw per-time-bin rows (see aggregate_by_speed_bin)."""
    df_clean = df.dropna(subset=["Frequency", "Power", "Speed"])
    # mobile and immobile time bins are aggregated separately, so a speed bin
    # that straddles MIN_SPEED_CMS can't mix the two (with 4 cm/s bins and a
    # 4 cm/s cut none does); only the mobile rows enter the fits
    parts = []
    for mobile, d in df_clean.groupby("Mobile"):
        parts.append(aggregate_by_speed_bin(d).assign(InAnalysis=bool(mobile)))
    df_all = pd.concat(parts, ignore_index=True)
    # the model's Speed predictor is the bin center; the within-bin
    # representative speed is kept (as SpeedWithinBin) for reference only
    df_all = df_all.rename(columns={"Speed": "SpeedWithinBin", "SpeedBinCenter": "Speed"})
    df_all = df_all.sort_values(["Session", "Speed"]).reset_index(drop=True)
    df_all.to_excel(OUTPUT_DIR / "ThetaVsSpeed_binned_medians.xlsx", index=False)
    df_stats = df_all[df_all["InAnalysis"]]
    df_excl = df_all[~df_all["InAnalysis"]]

    freq_result = fit_mixed_model(df_stats, "Frequency")
    print(freq_result.summary())
    plot_mixed_model_fit(
        df_stats, freq_result, "Frequency", "Speed",
        ylabel="Theta Frequency [Hz]",
        title="Theta Frequency vs. Speed (Mixed Model)",
        color="red", out_path=OUTPUT_DIR / "ThetaFrequency_vs_Speed_MixedModel.png",
        df_excluded=df_excl)

    power_results = {}
    for col, suffix, ylabel, title, _unit in POWER_MEASURES:
        d = df_stats.dropna(subset=[col])
        power_results[col] = fit_mixed_model(d, col)
        print(power_results[col].summary())
        plot_mixed_model_fit(
            d, power_results[col], col, "Speed",
            ylabel=ylabel, title=f"{title}, Mixed Model",
            color="royalblue", out_path=OUTPUT_DIR / f"ThetaPower{suffix}_vs_Speed_MixedModel.png",
            df_excluded=df_excl.dropna(subset=[col]))

    with open(OUTPUT_DIR / "MixedModel_Stats_Summary.txt", "w", encoding="utf-8") as f:
        f.write("=== Theta Frequency vs Speed ===\n")
        f.write(str(freq_result.summary()))
        for col, _suffix, _ylabel, title, _unit in POWER_MEASURES:
            f.write(f"\n\n=== {title} [{col}] ===\n")
            f.write(str(power_results[col].summary()))

    return freq_result, power_results


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

# The paper's Fig. 3 used 5 cm/s bins ("5 cms-1 bins"); here the same 4 cm/s
# width as the mixed model is used, so the first mobile bin is a full [4,8).
FIG3_SPEED_BIN_WIDTH_CMS = SPEED_BIN_WIDTH_CMS
FIG3_SPEED_UPPER_PCTILE = 90.0  # paper excluded speeds above the 90th percentile per session


def fig3_trim_and_bin(df, upper_pctile=FIG3_SPEED_UPPER_PCTILE, min_bin_s=0.0):
    """The per-session OLS input: mobile time bins (>= MIN_SPEED_CMS) of each
    session, and their FIG3_SPEED_BIN_WIDTH_CMS speed-bin medians.

    upper_pctile : speeds above this per-session percentile (of the mobile
                   speeds) are dropped first; None = no percentile cut.
    min_bin_s    : speed bins (per session) with less than this much data
                   (n time bins * BINSIZE seconds) are dropped, together
                   with their time bins; 0 = keep every visited bin.
    Returns (kept time-bin rows, speed-bin rows)."""
    df_clean = df.dropna(subset=["Frequency", "Power", "Speed"]).copy()
    # immobility (< MIN_SPEED_CMS) is dropped first, so the percentile cut
    # below is taken on each session's mobile speeds; bins stay anchored at
    # 0, so with 4 cm/s bins the first mobile bin is [4, 8) cm/s
    df_clean = df_clean[df_clean["Mobile"]]

    if upper_pctile is not None:
        def _trim_top_pctile(g):
            thresh = np.percentile(g["Speed"], upper_pctile)
            return g[g["Speed"] <= thresh]
        df_clean = df_clean.groupby("Session", group_keys=False).apply(_trim_top_pctile)

    if min_bin_s > 0:
        speed_bin = np.floor(df_clean["Speed"] / FIG3_SPEED_BIN_WIDTH_CMS)
        n_in_bin = df_clean.groupby([df_clean["Session"], speed_bin])["Speed"].transform("size")
        df_clean = df_clean[n_in_bin * BINSIZE >= min_bin_s]

    df_binned = aggregate_by_speed_bin(df_clean, speed_bin_width=FIG3_SPEED_BIN_WIDTH_CMS)
    return df_clean, df_binned


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
        slope, intercept, r_value, p_value, std_err = linregress(g[predictor], g[response])
        rows.append({
            group: ses_id,
            "beta1": slope,
            "beta1_se": std_err,
            "intercept": intercept,
            "r": r_value,
            "r2": r_value ** 2,
            "pvalue": p_value,
            "n_bins": len(g),
        })
    return pd.DataFrame(rows)


def summarise_slopes_across_recordings(fits, meta, group="Session"):
    """Group-level test of the per-session OLS slopes (two-stage /
    summary-statistics approach): each recording's fitted beta1 is one data
    point, tested against 0 across recordings.

    Channels in the same recording folder share the same running-speed trace
    and aren't independent, so their slopes are averaged first to one beta1
    per recording. The recording-level slopes are then tested with a
    one-sample t-test and a Wilcoxon signed-rank test (H0: mean/median slope
    = 0). Per-animal mean slopes are returned too, since recordings from the
    same animal aren't fully independent either.
    """
    fits = fits.merge(meta, on=group, how="left")
    per_rec = (fits.groupby(["Animal", "Recording"], as_index=False)
                   .agg(beta1=("beta1", "mean"), n_channels=("beta1", "size")))
    per_animal = (per_rec.groupby("Animal", as_index=False)
                     .agg(beta1_mean=("beta1", "mean"), beta1_sd=("beta1", "std"),
                          n_recordings=("beta1", "size")))
    per_animal = (per_animal.set_index("Animal")
                     .loc[ordered_animals(per_animal["Animal"])].reset_index())

    slopes = per_rec["beta1"].to_numpy()
    n = len(slopes)
    t_res = ttest_1samp(slopes, 0.0) if n >= 2 else None
    try:
        w_res = wilcoxon(slopes)
    except ValueError:  # n too small or all slopes exactly 0
        w_res = None

    stats = {
        "n_recordings": n,
        "n_animals": per_rec["Animal"].nunique(),
        "beta1_mean": slopes.mean() if n else np.nan,
        "beta1_sem": slopes.std(ddof=1) / np.sqrt(n) if n >= 2 else np.nan,
        "t": t_res.statistic if t_res else np.nan,
        "t_pvalue": t_res.pvalue if t_res else np.nan,
        "wilcoxon_W": w_res.statistic if w_res else np.nan,
        "wilcoxon_pvalue": w_res.pvalue if w_res else np.nan,
        "n_positive": int((slopes > 0).sum()),
    }
    return stats, per_rec, per_animal


def plot_recording_slopes(per_rec, grp, ylabel, title, out_path):
    """One dot per recording (its OLS slope, channels averaged), grouped and
    coloured by animal (ANIMAL_COLORS), with each animal's mean as a black bar
    and the group-level test result annotated."""
    rng = np.random.default_rng(0)
    animals_here = ordered_animals(per_rec["Animal"])

    fig, ax = plt.subplots(figsize=(1.4 * len(animals_here) + 2, 4.5))
    for i, a in enumerate(animals_here):
        vals = per_rec.loc[per_rec["Animal"] == a, "beta1"].to_numpy()
        x = i + rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(x, vals, s=45, color=animal_color(a), edgecolor='k', linewidth=0.6, zorder=3)
        ax.hlines(vals.mean(), i - 0.25, i + 0.25, color='black', lw=2, zorder=4)
    ax.axhline(0, color='grey', ls='--', lw=1)
    ax.set_xticks(range(len(animals_here)))
    ax.set_xticklabels(animals_here)
    ax.set_xlim(-0.6, len(animals_here) - 0.4)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.set_title(title, fontsize=13, weight='bold')
    # all stats go below the plot, so they can't overlap the title or hide points
    details = "\n".join([
        f"Mean β₁ = {grp['beta1_mean']:.3g} ± {grp['beta1_sem']:.2g} SEM",
        f"One-sample t({grp['n_recordings'] - 1}) = {grp['t']:.2f},  p = {grp['t_pvalue']:.3g}",
        f"Wilcoxon W = {grp['wilcoxon_W']:.4g},  p = {grp['wilcoxon_pvalue']:.3g}",
        f"{grp['n_positive']}/{grp['n_recordings']} recordings with positive slope",
        f"n = {grp['n_recordings']} recordings, {grp['n_animals']} animals",
        "Black bars: per-animal mean",
    ])
    add_details_below(fig, ax, details)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


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
    _df_clean, df_binned = fig3_trim_and_bin(df)
    df_binned.to_excel(OUTPUT_DIR / "Fig3_speed_binned_medians.xlsx", index=False)

    alpha_bonf = 0.05 / n_channels_for_bonferroni

    meta = df[["Session", "Recording", "Animal"]].drop_duplicates("Session")

    units = {"Frequency": "Hz", **{m[0]: m[4] for m in POWER_MEASURES}}
    summary_lines = []
    for response in units:
        fits = fit_per_session_ols(df_binned, response)
        fits.to_excel(OUTPUT_DIR / f"Fig3_persession_OLS_{response}.xlsx", index=False)

        n_sig = int((fits["pvalue"] < alpha_bonf).sum())
        lines = [(f"{response} vs Speed (per-session OLS, n={len(fits)} sessions): "
                  f"beta1 = {fits['beta1'].mean():.4g} +/- {fits['beta1'].std():.4g} (mean +/- SD), "
                  f"{n_sig}/{len(fits)} sessions significant at Bonferroni p < {alpha_bonf:.4g}")]

        # group-level result across all recordings
        grp, per_rec, per_animal = summarise_slopes_across_recordings(fits, meta)
        with pd.ExcelWriter(OUTPUT_DIR / f"Fig3_group_slopes_{response}.xlsx") as xw:
            per_rec.to_excel(xw, sheet_name="per_recording", index=False)
            per_animal.to_excel(xw, sheet_name="per_animal", index=False)
            pd.DataFrame([grp]).to_excel(xw, sheet_name="group_test", index=False)
        plot_recording_slopes(
            per_rec, grp, ylabel=f"β₁ ({units[response]} per cm/s)",
            title=f"Theta {response} vs. Speed slope",
            out_path=OUTPUT_DIR / f"Fig3_recording_slopes_{response}.png")

        lines.append(
            f"  Group level (n={grp['n_recordings']} recordings from {grp['n_animals']} animals, "
            f"channels averaged within recording): beta1 = {grp['beta1_mean']:.4g} +/- "
            f"{grp['beta1_sem']:.4g} (mean +/- SEM); one-sample t({grp['n_recordings'] - 1}) = "
            f"{grp['t']:.3f}, p = {grp['t_pvalue']:.3g}; Wilcoxon W = {grp['wilcoxon_W']:.4g}, "
            f"p = {grp['wilcoxon_pvalue']:.3g}; {grp['n_positive']}/{grp['n_recordings']} "
            f"recordings with positive slope")
        for _, r in per_animal.iterrows():
            lines.append(f"    {r['Animal']}: beta1 = {r['beta1_mean']:.4g} +/- {r['beta1_sd']:.4g} "
                         f"(mean +/- SD, n={r['n_recordings']} recordings)")

        for line in lines:
            print(line)
        summary_lines.extend(lines)

    with open(OUTPUT_DIR / "Fig3_persession_OLS_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    return df_binned


# %% ==================== Individual-file correlations (per-file OLS plots + % positive) ====
#
# Every analysed .ncs file (one Session) is fit on its own with the per-session
# OLS above (fig3_trim_and_bin -> fit_per_session_ols), but with its own
# speed-bin inclusion rule (INDIV_SPEED_UPPER_PCTILE / INDIV_MIN_TIME_PER_BIN_S),
# so slopes differ from Fig3_persession_OLS_<measure>.xlsx. Per measure, the p-values
# of all fitted files are Benjamini-Hochberg FDR-corrected, and each file is
# put in one CORREL_CATEGORIES class. The % summary uses every analysed file
# as the denominator: files without a fit (< 3 speed bins, or no finite
# value of the measure) are counted as "Not fitted". Power is the relative
# (normalized) power PowerRel, not PowerdB.

INDIV_DIR = OUTPUT_DIR / 'IndivdualCorrel'
# (column, panel title, y-axis label)
INDIV_MEASURES = [
    ("Frequency", "Theta frequency", "Theta Frequency [Hz]"),
    ("PowerRel", "Theta power (relative)", POWER_MEASURES[2][2]),
]
INDIV_FDR_Q = 0.05
# Speed-bin inclusion for the individual fits only (the group per-session OLS
# above keeps the Dunn 90th-percentile cut): no per-file percentile cut, and a
# speed bin is kept only with >= INDIV_MIN_TIME_PER_BIN_S of data (5 s = 20
# time bins of BINSIZE), which drops thinly sampled bins (e.g. rare fast
# running or tracking artifacts) instead of the whole upper 10% of speeds.
INDIV_SPEED_UPPER_PCTILE = None
INDIV_MIN_TIME_PER_BIN_S = 5.0
# (class, colour), in bar-stacking order from the baseline up
CORREL_CATEGORIES = [
    ("Positive, significant", "#c43c3b"),
    ("Positive, n.s.",        "#f2b0ae"),
    ("Negative, n.s.",        "#9ec5f4"),
    ("Negative, significant", "#1c5cab"),
    ("Not fitted",            "#d9d8d4"),
]
_DARK_CATEGORIES = {"Positive, significant", "Negative, significant"}
# Epoch cloud behind the speed-bin medians (individual plots only): every
# 0.25 s time bin that went into the fit, drawn as a small borderless gray dot
# with low opacity, so the gray gets darker where epochs overlap (denser).
INDIV_EPOCH_COLOR = '0.35'      # darkest gray reached where many dots stack
# opacity of a single epoch dot = INDIV_EPOCH_ALPHA_K / sqrt(n epochs), clipped
# to INDIV_EPOCH_ALPHA_RANGE: files with few epochs still show each dot, files
# with thousands don't turn into a solid block
INDIV_EPOCH_ALPHA_K = 4.0
INDIV_EPOCH_ALPHA_RANGE = (0.04, 0.35)
INDIV_EPOCH_SIZE = 9            # marker area (pt^2)
# y-axis limited to this percentile range of the epochs (plus the medians,
# whiskers and fit), so a few extreme epochs can't flatten the plot; the
# number of epochs outside the axis is given in the panel text.
INDIV_EPOCH_YLIM_PCTILE = (1.0, 99.0)


def classify_individual_fits(fits, sessions, q_thresh=INDIV_FDR_Q):
    """`fits` (fit_per_session_ols output) re-indexed onto every session in
    `sessions`, with the BH-FDR q-value (over the fitted files) and the class."""
    out = pd.DataFrame({"Session": list(sessions)}).merge(fits, on="Session", how="left")
    out["qvalue"] = np.nan
    fitted = out["pvalue"].notna()
    if fitted.any():
        out.loc[fitted, "qvalue"] = multipletests(out.loc[fitted, "pvalue"], method="fdr_bh")[1]

    def _cls(r):
        if np.isnan(r["beta1"]):
            return "Not fitted"
        sig = r["qvalue"] < q_thresh
        if r["beta1"] > 0:
            return "Positive, significant" if sig else "Positive, n.s."
        return "Negative, significant" if sig else "Negative, n.s."
    out["Category"] = out.apply(_cls, axis=1)
    return out


def plot_individual_correlation(ses_id, raw, binned, cls_rows, out_path):
    """One file: per measure, its FIG3_SPEED_BIN_WIDTH_CMS speed-bin medians (whiskers = IQR of
    the 0.25 s time bins in that speed bin) against the bin's median speed,
    with the file's own OLS line (solid if FDR-significant, dashed if not).
    Behind them, every 0.25 s time bin (epoch) used in the fit is drawn at its
    own speed as a faint gray dot; overlapping dots add up, so denser regions
    are darker (INDIV_EPOCH_*)."""
    fig, axes = plt.subplots(1, len(INDIV_MEASURES), figsize=(5.5 * len(INDIV_MEASURES), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (col, panel_title, ylabel) in zip(axes, INDIV_MEASURES):
        b = binned.dropna(subset=[col]).sort_values("Speed")
        r = raw.dropna(subset=[col])
        fit = cls_rows[col]
        animal = fit["Animal"]
        y_keep = []  # values the y-axis must show

        if len(r):
            alpha = float(np.clip(INDIV_EPOCH_ALPHA_K / np.sqrt(len(r)), *INDIV_EPOCH_ALPHA_RANGE))
            ax.scatter(r["Speed"], r[col], s=INDIV_EPOCH_SIZE, color=INDIV_EPOCH_COLOR,
                       alpha=alpha, linewidths=0, zorder=1, rasterized=True)
            y_keep += list(np.percentile(r[col], INDIV_EPOCH_YLIM_PCTILE))

        if len(b):
            q = (r.assign(SpeedBin=np.floor(r["Speed"] / FIG3_SPEED_BIN_WIDTH_CMS))
                  .groupby("SpeedBin")[col].quantile([0.25, 0.75]).unstack())
            sb = np.floor(b["Speed"] / FIG3_SPEED_BIN_WIDTH_CMS)
            lo = q.reindex(sb)[0.25].to_numpy()
            hi = q.reindex(sb)[0.75].to_numpy()
            y = b[col].to_numpy()
            ax.errorbar(b["Speed"], y, yerr=[np.maximum(y - lo, 0), np.maximum(hi - y, 0)],
                        fmt='none', ecolor='0.2', elinewidth=1, capsize=3, zorder=2)
            ax.scatter(b["Speed"], y, s=55, color=animal_color(animal), edgecolor='k',
                       linewidth=0.7, zorder=3)
            y_keep += [np.nanmin(lo), np.nanmax(hi), y.min(), y.max()]

        if fit["Category"] != "Not fitted":
            x = np.linspace(b["Speed"].min(), b["Speed"].max(), 50)
            y_fit = fit["intercept"] + fit["beta1"] * x
            ax.plot(x, y_fit, color='black', lw=2,
                    ls='-' if fit["qvalue"] < INDIV_FDR_Q else '--', zorder=4)
            y_keep += [y_fit.min(), y_fit.max()]
            details = [
                f"β₁ = {fit['beta1']:.4g} ± {fit['beta1_se']:.2g} SE per cm/s",
                f"r = {fit['r']:.3f},  R² = {fit['r2']:.3f}",
                f"p = {fit['pvalue']:.3g},  q (BH-FDR) = {fit['qvalue']:.3g}",
                f"n = {int(fit['n_bins'])} speed bins ({len(r)} time bins)",
                f"Class: {fit['Category']}",
            ]
        else:
            why = (f"no finite {col} values" if r.empty and not raw.empty
                   else f"{len(b)} speed bins (< 3)")
            details = [f"Not fitted: {why}"]

        y_keep = np.asarray(y_keep, dtype=float)
        y_keep = y_keep[np.isfinite(y_keep)]
        if len(y_keep):
            y0, y1 = y_keep.min(), y_keep.max()
            pad = 0.05 * (y1 - y0) if y1 > y0 else 1.0
            ax.set_ylim(y0 - pad, y1 + pad)
            n_out = int(((r[col] < y0 - pad) | (r[col] > y1 + pad)).sum())
            details.append(f"Gray: {len(r)} epochs ({BINSIZE:g} s), darker = denser"
                           + (f"; {n_out} outside y-axis" if n_out else ""))
        details = "\n".join(details)
        ax.set_title(panel_title, fontsize=12, weight='bold')
        ax.set_xlabel(f"Speed (cm/s); points = median per {FIG3_SPEED_BIN_WIDTH_CMS:g} cm/s bin",
                      fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        add_details_below(fig, ax, details)
    fig.suptitle(ses_id, fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


def _stacked_percent_bars(ax, groups):
    """`groups`: list of (x label, Series of class counts). One 100% bar each."""
    cats = [c for c, _ in CORREL_CATEGORIES]
    for i, (_label, counts) in enumerate(groups):
        counts = counts.reindex(cats, fill_value=0)
        total = counts.sum()
        bottom = 0.0
        for cat, color in CORREL_CATEGORIES:
            pct = 100 * counts[cat] / total if total else 0
            if pct > 0:
                ax.bar(i, pct, bottom=bottom, width=0.7, color=color,
                       edgecolor='white', linewidth=1.5)
                if pct >= 4:  # count on a second line only where the segment has room
                    label = f"{pct:.0f}%\n({counts[cat]})" if pct >= 9 else f"{pct:.0f}% ({counts[cat]})"
                    ax.text(i, bottom + pct / 2, label, ha='center',
                            va='center', fontsize=8,
                            color='white' if cat in _DARK_CATEGORIES else '0.15')
                else:     # too thin for text inside: the % just right of the segment
                    ax.text(i + 0.37, bottom + pct / 2, f"{pct:.1f}%", ha='left',
                            va='center', fontsize=6.5, color='0.15')
            bottom += pct
        ax.text(i, 101.5, f"n = {total}", ha='center', va='bottom', fontsize=8, color='0.3')
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, 108)
    ax.spines[['top', 'right']].set_visible(False)


def plot_percent_positive(cls_long, out_path):
    """% of all analysed files in each class: one bar per measure (all files),
    then one bar per animal for each measure."""
    animals = ordered_animals(cls_long["Animal"])
    n_panels = 1 + len(INDIV_MEASURES)
    fig = plt.figure(figsize=(3 + 2.2 * len(animals) * len(INDIV_MEASURES), 7.2))
    gs = fig.add_gridspec(2, n_panels, height_ratios=[4, 1.25],
                          width_ratios=[len(INDIV_MEASURES)] + [len(animals)] * len(INDIV_MEASURES))
    ax0 = fig.add_subplot(gs[0, 0])
    _stacked_percent_bars(ax0, [(title, cls_long.loc[cls_long["Measure"] == col, "Category"].value_counts())
                                for col, title, _ in INDIV_MEASURES])
    ax0.set_ylabel("% of LFP files", fontsize=11)
    ax0.set_title("All files", fontsize=12, weight='bold')
    for k, (col, title, _) in enumerate(INDIV_MEASURES, start=1):
        ax = fig.add_subplot(gs[0, k], sharey=ax0)
        sub = cls_long[cls_long["Measure"] == col]
        _stacked_percent_bars(ax, [(a, sub.loc[sub["Animal"] == a, "Category"].value_counts())
                                   for a in animals])
        ax.set_title(f"{title}, by animal", fontsize=12, weight='bold')
        ax.tick_params(labelleft=False)

    lines = []
    for col, title, _ in INDIV_MEASURES:
        c = cls_long.loc[cls_long["Measure"] == col, "Category"].value_counts()
        n = int(c.sum())
        n_pos = int(c.get("Positive, significant", 0) + c.get("Positive, n.s.", 0))
        n_fit = n - int(c.get("Not fitted", 0))
        lines.append(f"{title}: slope > 0 in {n_pos}/{n} files ({100 * n_pos / n:.1f}%); "
                     f"significantly positive in {c.get('Positive, significant', 0)}/{n} "
                     f"({100 * c.get('Positive, significant', 0) / n:.1f}%); "
                     f"significantly negative in {c.get('Negative, significant', 0)}/{n}; "
                     f"not fitted {c.get('Not fitted', 0)}/{n}  (BH-FDR over {n_fit} fitted files)")
    trim_txt = ("no upper speed cut" if INDIV_SPEED_UPPER_PCTILE is None
                else f"top {100 - INDIV_SPEED_UPPER_PCTILE:g}% of each file's speeds dropped")
    lines.append(f"Per-file OLS of {FIG3_SPEED_BIN_WIDTH_CMS:g} cm/s speed-bin medians (speeds ≥ "
                 f"{MIN_SPEED_CMS:g} cm/s, {trim_txt}, speed bins with ≥ {INDIV_MIN_TIME_PER_BIN_S:g} s "
                 f"of data); significant = Benjamini-Hochberg q < {INDIV_FDR_Q:g}, per measure.")
    axl = fig.add_subplot(gs[1, :])
    axl.axis('off')
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, color in CORREL_CATEGORIES]
    axl.legend(handles, [c for c, _ in CORREL_CATEGORIES], loc='upper center',
               ncol=len(CORREL_CATEGORIES), frameon=False, fontsize=9)
    axl.text(0.5, 0.55, "\n".join(lines), transform=axl.transAxes, ha='center', va='top',
             fontsize=9, linespacing=1.5)
    fig.suptitle("Theta vs. running speed: individual-file correlations", fontsize=13, weight='bold')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


def run_individual_correlations(df):
    """Per-file OLS fits, FDR classes, one PNG per file in INDIV_DIR, the
    % summary figure and IndividualCorrel_per_file.xlsx."""
    INDIV_DIR.mkdir(parents=True, exist_ok=True)
    df_clean, df_binned = fig3_trim_and_bin(df, upper_pctile=INDIV_SPEED_UPPER_PCTILE,
                                            min_bin_s=INDIV_MIN_TIME_PER_BIN_S)
    meta = (df[["Session", "Recording", "Animal"]].drop_duplicates("Session")
              .sort_values("Session").reset_index(drop=True))
    sessions = meta["Session"].tolist()

    per_measure = {}
    for col, _title, _ylabel in INDIV_MEASURES:
        cls = classify_individual_fits(fit_per_session_ols(df_binned, col), sessions)
        per_measure[col] = cls.merge(meta, on="Session", how="left").assign(Measure=col)
    cls_long = pd.concat(per_measure.values(), ignore_index=True)

    raw_by_ses = dict(tuple(df_clean.groupby("Session")))
    bin_by_ses = dict(tuple(df_binned.groupby("Session")))
    empty_raw, empty_bin = df_clean.iloc[:0], df_binned.iloc[:0]
    for i, ses_id in enumerate(sessions, start=1):
        rows = {col: per_measure[col].set_index("Session").loc[ses_id] for col in per_measure}
        fname = re.sub(r'[^\w.-]', '_', ses_id) + '.png'
        plot_individual_correlation(ses_id, raw_by_ses.get(ses_id, empty_raw),
                                    bin_by_ses.get(ses_id, empty_bin), rows, INDIV_DIR / fname)
        if i % 25 == 0 or i == len(sessions):
            print(f'  individual correlation plots: {i}/{len(sessions)}')

    plot_percent_positive(cls_long, OUTPUT_DIR / "IndividualCorrel_PercentPositive.png")

    cats = [c for c, _ in CORREL_CATEGORIES]
    overall = (cls_long.groupby("Measure")["Category"].value_counts().unstack(fill_value=0)
                       .reindex(columns=cats, fill_value=0))
    by_animal = (cls_long.groupby(["Measure", "Animal"])["Category"].value_counts()
                         .unstack(fill_value=0).reindex(columns=cats, fill_value=0))
    for t in (overall, by_animal):
        t["n_files"] = t[cats].sum(axis=1)
        t["pct_slope_positive"] = 100 * (t["Positive, significant"] + t["Positive, n.s."]) / t["n_files"]
        t["pct_sig_positive"] = 100 * t["Positive, significant"] / t["n_files"]
        t["pct_sig_negative"] = 100 * t["Negative, significant"] / t["n_files"]
    cols = ["Measure", "Session", "Animal", "Recording", "beta1", "beta1_se", "intercept",
            "r", "r2", "pvalue", "qvalue", "n_bins", "Category"]
    with pd.ExcelWriter(OUTPUT_DIR / "IndividualCorrel_per_file.xlsx") as xw:
        cls_long[cols].to_excel(xw, sheet_name="per_file", index=False)
        overall.reset_index().to_excel(xw, sheet_name="counts_all_files", index=False)
        by_animal.reset_index().to_excel(xw, sheet_name="counts_by_animal", index=False)
    print(overall.to_string())
    return cls_long


# %% ==================== Run metadata ===============================================

def _script_path():
    """Absolute path of this script (falls back to the working directory when
    run cell-by-cell/interactively, where __file__ isn't defined)."""
    try:
        return Path(__file__).resolve()
    except NameError:
        return Path(NCS_REPAIR_DIR) / 'ThetaVsSpeed_v8_Ferret_Mobility.py'


def save_run_metadata(meta_dir=META_DIR):
    """Write every setting that determines this script's output (config block
    + hard-coded values used later in the file), the data directory it was
    run on, and the script's name/directory and run time to a 2-column
    (parameter, value) CSV in `meta_dir`, and save a copy of the script
    itself there, so the results folder records exactly how it was made."""
    import datetime
    import shutil
    script_path = _script_path()
    meta = {
        # ---- provenance ----
        'script_name':              script_path.name,
        'script_dir':               script_path.parent,
        'run_datetime':             datetime.datetime.now().isoformat(timespec='seconds'),
        'data_dir (ROOT_DIR)':      ROOT_DIR,
        'OUTPUT_DIR':               OUTPUT_DIR,
        'OUTPUT_PARQUET':           OUTPUT_PARQUET,
        # ---- mobility variant ----
        'MIN_SPEED_CMS':            f'{MIN_SPEED_CMS} (time bins below are excluded from all fits, '
                                    'drawn gray on the mixed-model figures)',
        'DROP_DUPLICATE_LFP':       DROP_DUPLICATE_LFP,
        'duplicate_signature':      'SHA-1 of ADBitVolts + record timestamp/n_valid/samples, '
                                    'per recording folder; first in natural name order kept',
        'fig3_order':               'drop < MIN_SPEED_CMS, then per-session 90th percentile of '
                                    'mobile speeds; bins anchored at 0',
        'rel_power_denominator':    'whole session (all speeds), samples near clipping excluded',
        # ---- acquisition ----
        'NATIVE_FS':                NATIVE_FS,
        'ADBitVolts':               'read per file from the .ncs header (-ADBitVolts)',
        'ncs_dtype':                str(ncs_dtype),
        'ncs_samples_used':         'valid samples on a timestamp-anchored grid, gaps up to '
                                    '1 s interpolated (ncs_repair.py)',
        # ---- ADC saturation ----
        'InputRange':               'per-file, from .ncs header (-InputRange, uV)',
        'REJECT_SATURATED_BINS':    REJECT_SATURATED_BINS,
        'SATURATION_TOL_STEPS':     SATURATION_TOL_STEPS,
        'SATURATION_MARGIN_S':      SATURATION_MARGIN_S,
        'saturation_detection':     'raw repaired native-rate trace, before filtering; '
                                    '|uV| >= InputRange - SATURATION_TOL_STEPS * ADBitVolts; '
                                    'mapped to FS_LFP, each clipped run widened by '
                                    'SATURATION_MARGIN_S both sides; time bins overlapping a '
                                    'widened run removed; widened-run samples excluded from TotalPower',
        # ---- LFP cleaning (resample -> notch -> detrend -> low-pass) ----
        'FS_LFP':                   FS_LFP,
        'LINE_HARMONICS':           LINE_HARMONICS,
        'NOTCH_Q':                  NOTCH_Q,
        'detrend_type':             'linear',
        'LOWPASS_CUTOFF_HZ':        LOWPASS_CUTOFF_HZ,
        'LOWPASS_ORDER':            LOWPASS_ORDER,
        # ---- theta band-pass / peak-trough frequency & power ----
        'BANDPASS_CUTOFF':          BANDPASS_CUTOFF,
        'FILTER_ORDER':             FILTER_ORDER,
        'PEAK_THRESH_PC':           PEAK_THRESH_PC,
        'TRIM_DUR':                 TRIM_DUR,
        # ---- binning ----
        'BINSIZE':                  BINSIZE,
        'BIN_AGG_METHOD':           BIN_AGG_METHOD,
        'SPEED_BIN_WIDTH_CMS':      SPEED_BIN_WIDTH_CMS,
        # ---- theta-power normalization ----
        'DB_BASELINE_SPEED_CMS':    DB_BASELINE_SPEED_CMS,
        'REL_POWER_BAND':           REL_POWER_BAND,
        'REL_POWER_FILTER_ORDER':   REL_POWER_FILTER_ORDER,
        'POWER_MEASURES':           [m[0] for m in POWER_MEASURES],
        # ---- tracking ----
        'tracking_csv_columns':     'A=time_us, D=x_cm, E=y_cm',
        'POS_JUMP_THRESH_CMS':      POS_JUMP_THRESH_CMS,
        'POS_SMOOTH_SIGMA_SMP':     POS_SMOOTH_SIGMA_SMP,
        # ---- statistics ----
        'mixed_model':              'response ~ Speed + (1 | Session), REML; Speed = speed-bin center',
        'FIG3_SPEED_BIN_WIDTH_CMS': FIG3_SPEED_BIN_WIDTH_CMS,
        'FIG3_SPEED_UPPER_PCTILE':  FIG3_SPEED_UPPER_PCTILE,
        'n_channels_for_bonferroni': 1,
        'ols_min_speed_bins':       3,
        'INDIV_SPEED_UPPER_PCTILE': INDIV_SPEED_UPPER_PCTILE,
        'INDIV_MIN_TIME_PER_BIN_S': f'{INDIV_MIN_TIME_PER_BIN_S} (speed bins with less data '
                                    'dropped from the individual fits)',
        'individual_correl':        f'per-file OLS of {[m[0] for m in INDIV_MEASURES]}, '
                                    f'Benjamini-Hochberg FDR q < {INDIV_FDR_Q} across fitted files '
                                    'per measure; % denominator = all analysed files',
        # ---- plots ----
        'ANIMAL_COLORS':            ANIMAL_COLORS,
        'figure_dpi':               200,
    }
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / 'run_metadata.csv'
    pd.DataFrame({'parameter': list(meta.keys()),
                  'value':     [str(v) for v in meta.values()]}).to_csv(out_path, index=False)
    print(f'Saved run metadata -> {out_path}')

    if script_path.is_file():
        shutil.copy2(script_path, meta_dir)
        print(f'Saved script copy -> {meta_dir / script_path.name}')
    else:
        print(f'  Script not found at {script_path} -- no copy saved to {meta_dir}')


# %% ==================== Driver =====================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_run_metadata(META_DIR)

    df = process_root_directory(ROOT_DIR)
    df["Mobile"] = df["Speed"] >= MIN_SPEED_CMS
    df = add_normalized_power(df)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f'Saved {len(df)} rows -> {OUTPUT_PARQUET}')
    print(f'  {(~df["Mobile"]).mean():.1%} of time bins are < {MIN_SPEED_CMS:g} cm/s '
          f'(excluded from the fits)')
    save_baseline_coverage(df)

    if df["Session"].nunique() > 0:
        run_speed_vs_theta_stats(df)                    # mixed model (session as random intercept)
        run_speed_vs_theta_persession_stats(df)          # Dunn et al. Fig. 3's per-session OLS
        run_individual_correlations(df)                  # one OLS plot per file + % positive
    else:
        print('No sessions processed successfully -- skipping speed vs. theta statistics.')
