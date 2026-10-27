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

Theta-epoch acceptance (broadband fitting): the reference bank spans a broad
1-20 Hz (ACG_REF_BANK, 0.1 Hz steps), every epoch is matched against all of
it, and an epoch is accepted as a theta epoch only if the frequency of its
closest-matching (ED_min) reference sinusoid lies in 3-7 Hz (ACG_FREQ_RANGE).
There is no threshold on the ED_min value itself.

This script is fully self-contained: it reads Neuralynx .ncs files directly
(no dependency on FOOOF.py) and recursively processes every .ncs file found
under a single ROOT_DIR (see Configuration below).

Arena / session-type segregation: every .ncs file is tagged with the arena
type (Circle / Linear / Open) and session type (e.g. Linear 0/90/180/270) of
the session folder it sits in -- the same folder as that session's .ntt files,
parsed exactly as ArenaSessionType_StatsComparison_v8.py parses the .ntt
session paths. After the whole-dataset run, the full pipeline is re-run on
each arena's epochs (FIGURE_DIR\\ArenaType\\<arena>\\) and each session type's
epochs (FIGURE_DIR\\SessionType\\<arena>\\<session type>\\), and the non-theta
rejection prevalence (overall / immobility / mobility epochs, and the
immobility-vs-mobility split of the rejected epochs) is compared statistically
across arena types and across each arena's session types (see
compare_rejection_prevalence).
"""

import os
import re
import textwrap
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.stats import (pearsonr, spearmanr, chi2_contingency, fisher_exact,
                         kruskal, mannwhitneyu)

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colorbar import make_axes
from matplotlib.colors import to_rgba

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# %% ==================== Per-animal colours ======================================

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


# Thin dark outline for markers, so the light Pastel1 colours stay visible on white.
MARKER_EDGE = dict(markeredgecolor='0.25', markeredgewidth=0.5)


# %% ==================== Configuration ===========================================

# Root directory to search recursively for .ncs files. Every .ncs found
# anywhere under this tree (in any subfolder) is processed.
ROOT_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Alpha200'

# Where results/figures are written (default: alongside ROOT_DIR).
OUTPUT_DIR = os.path.join(ROOT_DIR, 'ACG_theta_continuity_BroadBandFitting')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
# Run metadata (all settings, data/script paths) + a copy of this script
# are written here -- see save_run_metadata.
META_DIR = os.path.join(OUTPUT_DIR, 'meta')

# ---- Acquisition ----
NATIVE_FS = 32000  # native Neuralynx .ncs sampling rate (Hz)
# V per ADC count is read from each file's own header (-ADBitVolts, see
# read_ncs_adbitvolts): it follows the recording's InputRange, which differs
# between animals (1500 uV -> 4.58e-8 V, 1000 uV -> 3.05e-8 V), so a single
# hard-coded constant would scale animals' power differently.
# Simulated .ncs files (Simulate_LFP_FaSimul.py) have an empty header and were
# written with this scale, so it is used only when a header has no -ADBitVolts.
ADBITVOLTS_NO_HEADER = 0.000003051757812500000169

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

# ---- Low-pass filtering ----
# The raw .ncs trace is broadband (acquired 1-500 Hz). Restrict the signal
# used for ACG/reference-sinusoid matching to <100 Hz so high-frequency
# content (well above the 3-7 Hz theta band of interest) can't distort the
# autocorrelogram shape.
LOWPASS_CUTOFF_HZ = 100.0  # Hz
LOWPASS_ORDER = 4          # Butterworth order (zero-phase via filtfilt)

# ---- Artifact rejection (epoch-wise peak-to-peak, robust MAD outlier) ----
MAD_THRESH = 5.0

# ---- Delta/theta epoch filter (epoch-wise Welch PSD band-power comparison) ----
# An epoch is rejected if its delta-band power exceeds its theta-band power --
# noisy epochs are often dominated by low-frequency artifact that would
# otherwise pass through amplitude-based (MAD) rejection as slow theta.
DELTA_BAND = (1.0, 3.0)  # Hz
THETA_BAND = (3.0, 7.0)  # Hz -- matches ACG_FREQ_RANGE

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

# Reference sinusoid bank (Supp. Fig. 5b-h uses 4-14 Hz for rat/ferret theta).
# Theta-epoch acceptance: every epoch is matched against the whole broadband
# bank ACG_REF_BANK, and accepted as a theta epoch only if the frequency of
# its closest-matching (ED_min) reference sinusoid lies within ACG_FREQ_RANGE.
# Epochs matched to any other bank frequency are marked 'skipped' (and
# 'skipped_outofband') -- see quantify_xcorr_epochs.
ACG_FREQ_RANGE = (3.0, 7.0)     # Hz -- theta band: epochs accepted only if freq_est falls here
ACG_REF_BANK = (1.0, 20.0)      # Hz -- full span of the reference bank matched against. With
                                 # 1 s epochs, references below ~1.75 Hz have no side peak/trough
                                 # inside the epoch, so they have no peak-range windows: they still
                                 # take part in the ED_min match, but epochs matched to them get a
                                 # NaN peak range (they are rejected as non-theta anyway).
ACG_FREQ_RES = 0.1             # Hz -- step of the reference bank. Not given a specific
                                 # numeric value in the supplement; 0.1 Hz gives a fine
                                 # frequency estimate at a modest compute cost -- tighten
                                 # (e.g. 0.05) or loosen (e.g. 0.25) as needed.
ACG_PEAK_PROMINENCE = 0.05      # matches findMinMax(refxc, 0.05, 'fixed') in the original

# Movement classification -- single-threshold binary split: epoch median
# speed above SPEED_THRESH_CMS -> 'moving', at/below it -> 'immobile'.
SPEED_THRESH_CMS = 4.0

# ---- Arena / session-type segregation ----
# Arena and session type of each .ncs file come from its folder path (the
# session folder it shares with that session's .ntt files), as in
# ArenaSessionType_StatsComparison_v8.py: the arena is whichever ARENA_TYPES
# name appears as a folder in the path, the session type is read from the
# name of the .ncs file's own folder, e.g.
#   Fa1059\Linear\Day10\3_180\CSC1.ncs  ->  arena 'Linear', session type '180'
ARENA_TYPES = ['Circle', 'Linear', 'Open']
# Session types within each arena, in plotting order.
ARENA_SESSION_TYPES = {
    'Open':   ['Cntrl', 'Rotate', 'Zero'],
    'Linear': ['0', '90', '180', '270'],
    'Circle': ['Stnd', 'NoRot', 'Rot'],
}
# The full pipeline re-run on each arena / session-type subset, plus the
# non-theta-prevalence comparison plots, are written under these folders.
ARENA_FIGURE_DIR   = os.path.join(FIGURE_DIR, 'ArenaType')
SESSION_FIGURE_DIR = os.path.join(FIGURE_DIR, 'SessionType')
STATS_ALPHA = 0.05   # significance level for the non-theta-prevalence tests


# %% ==================== .ncs / tracking data import ==============================

def read_ncs_adbitvolts(fpath):
    """V per ADC count, from the '-ADBitVolts' line of the 16 kB .ncs header;
    ADBITVOLTS_NO_HEADER if the header has none (simulated files)."""
    with open(fpath, 'rb') as f:
        header = f.read(16 * 1024).decode('latin-1')
    m = re.search(r'-ADBitVolts\s+(\S+)', header)
    if m is None:
        print(f'    No -ADBitVolts in {os.path.basename(fpath)} header -- '
              f'using {ADBITVOLTS_NO_HEADER:g} V/count')
        return ADBITVOLTS_NO_HEADER
    return float(m.group(1))


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
    lfp = np.concatenate(data['samples']).astype(np.float64) * read_ncs_adbitvolts(fpath) * 1e6
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


def lowpass_filter(x, fs_hz, cutoff_hz, order=4):
    """Zero-phase Butterworth low-pass filter (filtfilt, so no phase distortion
    of the theta-band content the ACG matching cares about)."""
    nyq = fs_hz / 2.0
    b, a = signal.butter(order, cutoff_hz / nyq, btype='low')
    return signal.filtfilt(b, a, np.asarray(x, dtype=np.float64))


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


def extract_arena_type(session_path):
    """Return 'Open', 'Linear', or 'Circle' if one of those appears as a
    path component of `session_path`, else None.
    (Same as ArenaSessionType_StatsComparison_v8.extract_arena_type.)"""
    parts = re.split(r'[\\/]+', str(session_path))
    for part in parts:
        for arena in ARENA_TYPES:
            if part.strip().lower() == arena.lower():
                return arena
    return None


def extract_session_type(session_path, arena):
    """Return the session type named in the last folder of `session_path`
    (e.g. '1_Cntrl' -> 'Cntrl' for Open, '3_180' -> '180' for Linear,
    '2NoRot' -> 'NoRot' for Circle), or None if it can't be recognized.
    (Same as ArenaSessionType_StatsComparison_v8.extract_session_type.)"""
    if arena is None:
        return None
    label = re.split(r'[\\/]+', str(session_path).strip())[-1].strip().lower()

    if arena == 'Open':
        for t in ARENA_SESSION_TYPES['Open']:
            if t.lower() in label:
                return t
    elif arena == 'Linear':
        m = re.search(r'_(270|180|90|0)$', label)
        if m:
            return m.group(1)
    elif arena == 'Circle':
        # 'norot' must be tested before 'rot', which it contains.
        if 'norot' in label:
            return 'NoRot'
        if 'stnd' in label:
            return 'Stnd'
        if 'rot' in label:
            return 'Rot'
    return None


def _label(group):
    """Display label for plots: Linear session types are angles."""
    return f'{group}°' if str(group).isdigit() else str(group)


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

    References too slow to show a side peak and its flanking troughs within
    one epoch (below ~1.75 Hz for 1 s epochs) keep their refXC/refED, so they
    still take part in ED_min matching, but get refRange = NaN and windows of
    -1 -- quantify_xcorr_epochs leaves the peak range of epochs matched to
    them as NaN.
    """
    ref_freqs = np.asarray(ref_freqs, dtype=np.float64)
    n_freqs = ref_freqs.size
    reft = np.arange(datsize) / fs
    n_lags = 2 * datsize - 1
    center = datsize - 1   # zero-lag index (0-based) in the full autocorrelogram

    refXC = np.empty((n_lags, n_freqs))
    refRange = np.full(n_freqs, np.nan)
    refP1range = np.full((2, n_freqs), -1, dtype=int)
    refT1range = np.full((2, n_freqs), -1, dtype=int)
    refT2range = np.full((2, n_freqs), -1, dtype=int)

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
            continue   # no side peak after the zero-lag peak within the epoch
        peak1_idx = max_idx[center_row + 1]
        peak1_val = max_val[center_row + 1]

        before = min_idx < peak1_idx
        after = min_idx > peak1_idx
        if not before.any() or not after.any():
            continue   # can't bracket troughs around the first side peak
        trough1_idx, trough1_val = min_idx[before][-1], min_val[before][-1]
        trough2_idx, trough2_val = min_idx[after][0], min_val[after][0]

        crossings = zero_crossings(refxc)
        belowt1 = crossings[crossings < trough1_idx]
        abovet2 = crossings[crossings > trough2_idx]
        abovep1 = crossings[crossings > peak1_idx]
        belowp1 = crossings[crossings < peak1_idx]
        if not (belowp1.size and abovep1.size and belowt1.size and abovet2.size):
            continue   # no bounding zero-crossings for the search windows

        refRange[n] = peak1_val - np.mean([trough1_val, trough2_val])
        refP1range[:, n] = [int(round(belowp1[-1])), int(round(abovep1[0]))]
        refT1range[:, n] = [int(round(belowt1[-1])), int(round(belowp1[-1]))]
        refT2range[:, n] = [int(round(abovep1[0])), int(round(abovet2[0]))]

    refED = np.linalg.norm(refXC, axis=0)
    return refXC, refED, refRange, refP1range, refT1range, refT2range


def quantify_xcorr_epochs(data_epochs, freq_range=ACG_FREQ_RANGE, freq_resolution=ACG_FREQ_RES,
                          fs=FS_ACG, peak_prominence=ACG_PEAK_PROMINENCE,
                          ref_bank=ACG_REF_BANK):
    """Autocorrelogram peak-range quantification of oscillatory activity.

    Python translation of quantify_xcorr_epochs.m (Dunn et al. 2022).

    Parameters
    ----------
    data_epochs : (winlength, nepochs) array -- each column one LFP epoch,
                  sampled at `fs` Hz. NaNs are allowed (an epoch is skipped
                  if fewer than 2 finite samples remain).
    freq_range      : (fmin, fmax) Hz, theta band -- an epoch is accepted as a
                      theta epoch only if its freq_est (frequency of the
                      ED_min reference sinusoid) falls inside it; all other
                      epochs are marked 'skipped'
    freq_resolution : Hz step of the reference bank
    ref_bank        : (fmin, fmax) Hz span of the reference bank every epoch
                      is matched against -- broader than `freq_range`, so
                      epochs whose rhythm lies outside the theta band match
                      their own frequency instead of being clamped onto the
                      theta band edges
    fs              : sampling rate (Hz) of `data_epochs`

    Returns
    -------
    XC     : (2*winlength-1, nepochs) normalised data autocorrelograms
    result : DataFrame, one row per epoch, columns:
             ED_min, freq, peak1, peak1_idx, trough1, trough1_idx,
             trough2, trough2_idx, peakrange, peakrangenorm, skipped,
             skipped_outofband
             ('skipped' covers too-short epochs and epochs whose freq_est
             falls outside `freq_range`; 'skipped_outofband' flags only the
             latter, so the non-theta rejection rate can be reported
             separately from too-short-epoch skips)
    """
    data_epochs = np.asarray(data_epochs, dtype=np.float64)
    winlength, nepochs = data_epochs.shape
    n_lags = 2 * winlength - 1

    ref_freqs = np.round(np.arange(ref_bank[0], ref_bank[1] + freq_resolution / 2,
                                   freq_resolution), 6)   # round so 0.1 Hz steps compare exactly
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
    skipped_outofband = np.zeros(nepochs, dtype=bool)

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

        # theta-epoch acceptance: the ED_min sinusoid must lie in the theta band
        if not (freq_range[0] <= freq_est[n] <= freq_range[1]):
            skipped[n] = True
            skipped_outofband[n] = True

        # matched reference has no peak/trough windows (too slow to show a
        # side peak within one epoch) -> peak range stays NaN
        if not np.isfinite(refRange[mi]):
            continue

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
        'skipped_outofband': skipped_outofband,
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


def reject_deltatheta_epochs(epochs, fs_hz, low_band=DELTA_BAND, theta_band=THETA_BAND):
    """Boolean keep-mask: per-epoch Welch PSD is used to reject epochs whose
    delta-band (`low_band`) power exceeds their theta-band (`theta_band`)
    power. Meant to run BEFORE reject_artifact_epochs -- see
    process_ncs_for_acg, which passes this mask on to reject_artifact_epochs
    as `ref_mask` so the MAD reference statistics aren't skewed by epochs
    already excluded here."""
    n_total, nperseg = epochs.shape
    win = signal.get_window('hann', nperseg)
    keep = np.ones(n_total, dtype=bool)
    for i in range(n_total):
        f, Pxx = signal.welch(epochs[i], fs=fs_hz, window=win, # type: ignore
                              nperseg=nperseg, noverlap=0, detrend='constant')
        df = f[1] - f[0]
        idx_low   = (f >= low_band[0])   & (f <= low_band[1])
        idx_theta = (f >= theta_band[0]) & (f <= theta_band[1])
        low_pow   = np.sum(Pxx[idx_low]) * df
        theta_pow = np.sum(Pxx[idx_theta]) * df
        keep[i] = low_pow <= theta_pow
    return keep


def reject_artifact_epochs(epochs, mad_thresh=MAD_THRESH, ref_mask=None):
    """Boolean keep-mask: epoch-wise peak-to-peak amplitude, robust (MAD)
    outlier rejection. If `ref_mask` is given (e.g. the mask returned by
    reject_deltatheta_epochs), the median/MAD reference statistics are
    computed only from epochs[ref_mask], so epochs already excluded upstream
    can't skew the robust threshold -- z-scores are still returned for every
    epoch."""
    p2p = epochs.max(axis=1) - epochs.min(axis=1)
    return ~_robust_high_outliers(p2p, mad_thresh, ref_mask=ref_mask)


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
                        epoch_sec=EPOCH_SEC, fs_acg=FS_ACG, ref_bank=ACG_REF_BANK):
    """Load one .ncs file, resample/clean it, split it into epochs labelled by
    movement state, and compute the ACG peak-range quantification for every
    clean moving/immobile epoch.

    Non-theta epochs (ED_min sinusoid outside `freq_range`) are marked
    'skipped' by quantify_xcorr_epochs rather than dropped, so the resulting
    DataFrame's row count stays aligned with 'epoch_index'/'file'; filter on
    '~skipped' downstream as usual.

    Returns a DataFrame (one row per epoch) with the quantify_xcorr_epochs
    columns plus 'movement', 'median_speed_cms', and 'epoch_index'.
    """
    lfp, lfp_start_us = load_ncs(fpath)
    lfp = signal.resample_poly(lfp, int(fs_acg), int(NATIVE_FS))
    lfp = notch_filter(lfp, fs_acg, LINE_HARMONICS, NOTCH_Q)
    lfp = detrend_signal(lfp, dtype='linear')
    lfp = lowpass_filter(lfp, fs_acg, LOWPASS_CUTOFF_HZ, LOWPASS_ORDER)

    nperseg = int(round(fs_acg * epoch_sec))
    epochs, n_total = build_epochs(lfp, nperseg)
    keep_deltatheta = reject_deltatheta_epochs(epochs, fs_acg)
    keep_artifact = keep_deltatheta & reject_artifact_epochs(
        epochs, ref_mask=keep_deltatheta)

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
                                        ref_bank=ref_bank)
    result['movement'] = label[keep]
    result['median_speed_cms'] = med_speed[keep]
    result['epoch_index'] = np.where(keep)[0]
    # absolute path -- lets get_epoch_waveform() re-fetch this exact epoch's
    # raw waveform later (e.g. for plot_example_epoch), regardless of cwd
    result['file'] = os.path.abspath(fpath)
    return result


def get_epoch_waveform(fpath, epoch_index, epoch_sec=EPOCH_SEC, fs_acg=FS_ACG):
    """Re-derive the cleaned (resampled / notch-filtered / detrended / low-pass
    filtered) waveform of one specific epoch from its source .ncs file, for
    illustrative plotting (plot_example_epoch). `fpath` + `epoch_index` should
    come from the 'file' / 'epoch_index' columns of a process_ncs_for_acg
    results row.
    """
    lfp, _ = load_ncs(fpath)
    lfp = signal.resample_poly(lfp, int(fs_acg), int(NATIVE_FS))
    lfp = notch_filter(lfp, fs_acg, LINE_HARMONICS, NOTCH_Q)
    lfp = detrend_signal(lfp, dtype='linear')
    lfp = lowpass_filter(lfp, fs_acg, LOWPASS_CUTOFF_HZ, LOWPASS_ORDER)
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
            # arena / session type of the session folder this .ncs shares
            # with that session's .ntt files
            session_folder = os.path.dirname(rel)
            arena = extract_arena_type(session_folder)
            res['session_folder'] = session_folder
            res['arena_type'] = arena
            res['session_type'] = extract_session_type(session_folder, arena)
            n_moving = int((res['movement'] == 'moving').sum())
            n_immobile = int((res['movement'] == 'immobile').sum())
            n_outofband = int(res['skipped_outofband'].sum())
            pct_outofband = 100 * n_outofband / len(res) if len(res) else 0.0
            print(f'  OK: {rel}  [{arena or "?"} / {res["session_type"].iloc[0] or "?"}; '
                  f'{len(res)} epochs kept: '
                  f'{n_moving} moving, {n_immobile} immobile; '
                  f'freq_est outside {ACG_FREQ_RANGE} (non-theta) rejected {n_outofband} '
                  f'({pct_outofband:.1f}%)]')
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


# Ranges the matched-sinusoid frequency estimate ('freq' column, i.e.
# freq_est in quantify_xcorr_epochs) is bucketed into for
# compute_freq_range_counts -- inclusive [lo, hi] on both ends, matching the
# 0.1 Hz step of the reference bank (ACG_FREQ_RES) so no freq_est value can
# fall between two ranges.
FREQ_HIST_BIN_EDGES = [(3.0, 3.9), (4.0, 4.9), (5.0, 5.9), (6.0, 7.0)]


def compute_freq_range_counts(df, bin_edges=FREQ_HIST_BIN_EDGES):
    """Count of (non-skipped) epochs whose matched-sinusoid freq_est falls in
    each [lo, hi] range of `bin_edges`, clubbing all epochs together
    (all animals/channels/movement states)."""
    freq = df.loc[~df['skipped'], 'freq']
    labels = [f'{lo:g}-{hi:g}' for lo, hi in bin_edges]
    counts = [int(((freq >= lo) & (freq <= hi)).sum()) for lo, hi in bin_edges]
    return pd.DataFrame({'freq_range': labels, 'count': counts})


def _mean_count_per_recording(sub, bin_edges):
    """Per-bin epoch count histogrammed separately for each recording
    ('file' column) in `sub`, then averaged across those recordings.

    Normalises for animals/groups having different numbers of recordings
    (files) contributing epochs -- a plain pooled histogram would otherwise
    just track how many recordings/epochs went into each group rather than
    the underlying freq_est distribution. Returns (mean_counts, n_recordings).
    """
    files = sub['file'].unique()
    if len(files) == 0:
        return np.zeros(len(bin_edges) - 1), 0
    per_file_counts = np.array([
        np.histogram(sub.loc[sub['file'] == f, 'freq'], bins=bin_edges)[0]
        for f in files
    ])
    return per_file_counts.mean(axis=0), len(files)


def _band_label():
    return f'{ACG_FREQ_RANGE[0]:g}-{ACG_FREQ_RANGE[1]:g} Hz'


def _add_rejection_footer(ax, scope):
    """Footer below `ax`'s x-axis label, over every epoch in `scope`: the
    percentage rejected as non-theta, i.e. whose ED_min sinusoid (matched over
    the full ACG_REF_BANK) fell outside ACG_FREQ_RANGE ('skipped_outofband').
    Drawn in axes coordinates, so it is kept by savefig(bbox_inches='tight')
    whether or not `ax` was passed in."""
    n_total = len(scope)
    n_outofband = int(scope['skipped_outofband'].sum())
    pct_outofband = 100 * n_outofband / n_total if n_total else 0.0
    ax.text(0.5, -0.18,
            f'Rejected as non-theta (ED_min sinusoid outside {_band_label()}): '
            f'{pct_outofband:.1f}% ({n_outofband}/{n_total} epochs)',
            transform=ax.transAxes, ha='center', va='top', fontsize=8, color='0.2')


def plot_freq_est_histogram(df, animal=None, bins=30, ax=None, figsize=(5, 4)):
    """Line-histogram of matched-sinusoid freq_est ('freq' column) for
    accepted (non-skipped) epochs, plotted as the mean epoch count per
    recording (see _mean_count_per_recording) rather than the pooled
    absolute count, so groups with more recordings don't just read higher.

    If `animal` is given, restricted to that animal's epochs; otherwise
    clubbed across all animals/channels/movement states.

    Returns (fig, ax, freq series).
    """
    scope = df if animal is None else df[df['animal'] == animal]
    accepted = scope.loc[~scope['skipped'], ['file', 'freq']]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if accepted.empty:
        ax.set_title('No accepted epochs')
        return fig, ax, accepted['freq']

    bin_edges = np.linspace(accepted['freq'].min(), accepted['freq'].max(), bins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_counts, n_files = _mean_count_per_recording(accepted, bin_edges)
    color = animal_color(animal, default='#4C72B0') if animal else '#4C72B0'
    ax.plot(centers, mean_counts, marker='o', markersize=3, linewidth=1.2, color=color,
            **MARKER_EDGE)
    ax.set_xlabel('Matched sinusoid frequency estimate, freq_est (Hz)')
    ax.set_ylabel('Mean epoch count per recording')
    title = f'freq_est of accepted epochs (n={len(accepted)} epochs, {n_files} recordings)'
    ax.set_title(f'{animal}: {title}' if animal else title)
    _add_rejection_footer(ax, scope)
    fig.tight_layout()
    return fig, ax, accepted['freq']


def plot_freq_est_histogram_overlay(df, bins=30, ax=None, figsize=(6, 4)):
    """Line-overlay version of plot_freq_est_histogram: one line per animal
    (mean epoch count per recording, see _mean_count_per_recording), all
    drawn on shared bins/axes so animals' freq_est distributions can be
    compared directly regardless of how many recordings each contributed.
    Restricted to accepted (non-skipped) epochs.

    Returns (fig, ax).
    """
    accepted = df.loc[~df['skipped'], ['animal', 'file', 'freq']]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if accepted.empty:
        ax.set_title('No accepted epochs')
        return fig, ax

    bin_edges = np.linspace(accepted['freq'].min(), accepted['freq'].max(), bins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for animal_label in ordered_animals(accepted['animal']):
        sub = accepted[accepted['animal'] == animal_label]
        mean_counts, n_files = _mean_count_per_recording(sub, bin_edges)
        ax.plot(centers, mean_counts, marker='o', markersize=3, linewidth=1.2,
               color=animal_color(animal_label), **MARKER_EDGE,
               label=f'{animal_label} (n={len(sub)} epochs, {n_files} rec)')

    ax.set_xlabel('Matched sinusoid frequency estimate, freq_est (Hz)')
    ax.set_ylabel('Mean epoch count per recording')
    ax.set_title('freq_est of accepted epochs, by animal')
    ax.legend(fontsize=7, frameon=False)
    _add_rejection_footer(ax, df)
    fig.tight_layout()
    return fig, ax


def plot_rejected_freq_histogram(df, animal=None, bins=30, ax=None, figsize=(5, 4)):
    """Line-histogram of matched-sinusoid freq_est ('freq' column) for
    epochs rejected as non-theta (ED_min sinusoid outside ACG_FREQ_RANGE,
    'skipped_outofband'), plotted as the mean epoch count per recording (see
    _mean_count_per_recording) rather than the pooled absolute count, so
    groups with more recordings don't just read higher.

    If `animal` is given, restricted to that animal's epochs; otherwise
    clubbed across all animals/channels/movement states.

    Returns (fig, ax, rejected_freq series).
    """
    scope = df if animal is None else df[df['animal'] == animal]
    rejected = scope.loc[scope['skipped_outofband'], ['file', 'freq']]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if rejected.empty:
        ax.set_title(f'No epochs matched outside {_band_label()}')
        return fig, ax, rejected['freq']

    bin_edges = np.linspace(rejected['freq'].min(), rejected['freq'].max(), bins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_counts, n_files = _mean_count_per_recording(rejected, bin_edges)
    color = animal_color(animal, default='#C44E52') if animal else '#C44E52'
    ax.plot(centers, mean_counts, marker='o', markersize=3, linewidth=1.2, color=color,
            **MARKER_EDGE)
    ax.set_xlabel('Matched sinusoid frequency estimate, freq_est (Hz)')
    ax.set_ylabel('Mean epoch count per recording')
    title = (f'freq_est of non-theta rejected epochs '
             f'(n={len(rejected)} epochs, {n_files} recordings)')
    ax.set_title(f'{animal}: {title}' if animal else title)
    fig.tight_layout()
    return fig, ax, rejected['freq']


def plot_rejected_freq_histogram_overlay(df, bins=30, ax=None, figsize=(6, 4)):
    """Line-overlay version of plot_rejected_freq_histogram: one line per
    animal (mean epoch count per recording, see _mean_count_per_recording),
    all drawn on shared bins/axes so animals' freq_est distributions can be
    compared directly regardless of how many recordings each contributed.
    Restricted to epochs rejected as non-theta ('skipped_outofband').

    Returns (fig, ax).
    """
    rejected = df.loc[df['skipped_outofband'], ['animal', 'file', 'freq']]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if rejected.empty:
        ax.set_title(f'No epochs matched outside {_band_label()}')
        return fig, ax

    bin_edges = np.linspace(rejected['freq'].min(), rejected['freq'].max(), bins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for animal_label in ordered_animals(rejected['animal']):
        sub = rejected[rejected['animal'] == animal_label]
        mean_counts, n_files = _mean_count_per_recording(sub, bin_edges)
        ax.plot(centers, mean_counts, marker='o', markersize=3, linewidth=1.2,
               color=animal_color(animal_label), **MARKER_EDGE,
               label=f'{animal_label} (n={len(sub)} epochs, {n_files} rec)')

    ax.set_xlabel('Matched sinusoid frequency estimate, freq_est (Hz)')
    ax.set_ylabel('Mean epoch count per recording')
    ax.set_title('freq_est of non-theta rejected epochs, by animal')
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_edmin_vs_peakrangenorm(df, animal=None, ax=None, figsize=(6, 5),
                                color=None, point_size=10, alpha=0.3):
    """Scatter of ED_min (fit-quality distance to the closest-matching
    reference sinusoid) vs. peakrangenorm (normalised autocorrelogram peak
    range), to test whether a poor sinusoid fit (high ED_min) predicts a low
    peak range.

    Deliberately NOT filtered on 'skipped', so every matched epoch (theta and
    non-theta) is shown. Only epochs with no ED_min or peak range (too-short
    epochs, or epochs matched to a reference too slow to have peak-range
    windows) are excluded.

    Reports Pearson r and Spearman rho (each with a p-value) and overlays a
    linear least-squares fit line.

    `color` defaults to the animal's ANIMAL_COLORS entry (or '#4C72B0' when
    pooled across animals / the animal ID is unknown).

    Returns (fig, ax, stats_dict) where stats_dict has keys 'pearson_r',
    'pearson_p', 'spearman_r', 'spearman_p', 'n'.
    """
    if color is None:
        color = animal_color(animal, default='#4C72B0') if animal else '#4C72B0'
    scope = df if animal is None else df[df['animal'] == animal]
    sub = scope.dropna(subset=['ED_min', 'peakrangenorm'])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if len(sub) < 2:
        ax.set_title('Not enough epochs for correlation')
        return fig, ax, {'pearson_r': np.nan, 'pearson_p': np.nan,
                         'spearman_r': np.nan, 'spearman_p': np.nan, 'n': len(sub)}

    x = sub['ED_min'].to_numpy()
    y = sub['peakrangenorm'].to_numpy()

    r_p, p_p = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)

    # alpha on the fill only, so the thin dark outline stays visible
    ax.scatter(x, y, s=point_size, facecolor=to_rgba(color, alpha),
               edgecolor=MARKER_EDGE['markeredgecolor'], linewidth=0.3, zorder=2)

    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.array([x.min(), x.max()])
    ax.plot(x_line, slope * x_line + intercept, color='black', linewidth=1.5, zorder=4,
            label='linear fit')

    ax.set_xlabel('ED_min (distance to closest-matching reference sinusoid)')
    ax.set_ylabel('Autocorr. peak range (norm.)')
    title = 'ED_min vs. normalised peak range'
    ax.set_title(f'{animal}: {title}' if animal else title)
    ax.legend(loc='upper right', fontsize=7, frameon=False)

    stats_text = (f'Pearson r = {r_p:.3f} (p = {p_p:.2e})\n'
                 f'Spearman rho = {r_s:.3f} (p = {p_s:.2e})\n'
                 f'n = {len(sub)}')
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, ha='left', va='bottom',
           fontsize=8, color='0.1',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='0.7'))

    fig.tight_layout()
    return fig, ax, {'pearson_r': r_p, 'pearson_p': p_p,
                     'spearman_r': r_s, 'spearman_p': p_s, 'n': len(sub)}


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
    the centroid diamond markers in Dunn et al. Fig. 4e/h."""
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
                                 moving_alpha=0.5, immobile_speed_vmax=SPEED_THRESH_CMS,
                                 immobile_cmap='magma', immobile_alpha=1.0,
                                 immobile_edgecolor='#FFB3BA', immobile_edgewidth=0.8,
                                 moving_centroid_cmap_pos=0.8, immobile_centroid_color='#FF1493',
                                 n_bins=25, log_counts=True, show_rejection_annotation=True):
    """Shared drawing routine for the Fig. 4e/h-style scatter: frequency vs.
    normalised autocorrelogram peak range. Immobile epochs are drawn in
    `immobile_cmap` (coloured by per-epoch running speed over
    0-`immobile_speed_vmax`, i.e. the immobile speed range) with a pastel-red
    outline; moving epochs are drawn in `moving_cmap` (coloured by per-epoch
    running speed over 0-`speed_vmax`) at `moving_alpha` opacity, layered on
    top -- the two distinct colormaps keep the heavily-overlapping point
    clouds visually distinguishable. Marginal count histograms follow the
    same colouring (each bar coloured by its epochs' mean speed), and
    moving/immobile centroids are marked as diamonds in a bright colour
    (green from viridis for moving, at position `moving_centroid_cmap_pos`
    along the colormap; bright pink `immobile_centroid_color` for immobile).
    An annotation in `ax_main`
    reports the percentage of epochs (within `animal`'s scope, if given)
    rejected as non-theta (ED_min sinusoid outside ACG_FREQ_RANGE) -- these
    never reach the scatter/centroids since they're filtered out via
    'skipped'.

    `mode` selects which group(s) are drawn: 'both' (default), 'moving', or
    'immobile'. Axes/bins are always scaled to the full (both-group)
    frequency/peak-range range regardless of `mode`, so single-group panels
    stay directly comparable to the merged one. Used by
    plot_freq_peakrange_summary(), plot_figure4_style(), and
    plot_freq_peakrange_by_movement().

    Returns (sc_moving, sc_immobile, centroids) -- either scatter handle is
    None when that group isn't drawn (see `mode`); pass them to
    _add_speed_colorbars.
    """
    scope = df if animal is None else df[df['animal'] == animal]
    n_rejected = int(scope['skipped_outofband'].sum())
    pct_rejected = 100 * n_rejected / len(scope) if len(scope) else 0.0

    sub = df[~df['skipped']].copy()
    if animal is not None:
        sub = sub[sub['animal'] == animal]
    sub = sub[sub['movement'].isin(['moving', 'immobile'])]

    show_immobile = mode in ('both', 'immobile')
    show_moving = mode in ('both', 'moving')
    immobile = sub[sub['movement'] == 'immobile'] if show_immobile else sub.iloc[0:0]
    moving = sub[sub['movement'] == 'moving'] if show_moving else sub.iloc[0:0]

    # immobile first (opaque by default) so the semi-transparent moving layer
    # sits visibly on top rather than being hidden underneath it
    sc_imm = None
    if show_immobile:
        imm_speed = np.clip(immobile['median_speed_cms'].to_numpy(), 0, immobile_speed_vmax)
        sc_imm = ax_main.scatter(immobile['freq'], immobile['peakrangenorm'], c=imm_speed,
                                 cmap=immobile_cmap, vmin=0, vmax=immobile_speed_vmax, s=16,
                                 edgecolor=immobile_edgecolor, linewidth=immobile_edgewidth,
                                 alpha=immobile_alpha, label='immobile', zorder=2)
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
    # centroid diamonds: bright green from `moving_cmap` (viridis) for moving,
    # bright pink (`immobile_centroid_color`) for immobile
    marker_colors = {'moving': plt.get_cmap(moving_cmap)(moving_centroid_cmap_pos),
                     'immobile': immobile_centroid_color}
    for _, crow in centroids.iterrows():
        ax_main.scatter(crow['freq_centroid'], crow['peakrangenorm_centroid'],
                        marker='D', s=90, facecolor=marker_colors.get(crow['movement'], 'limegreen'),
                        edgecolor='black', linewidth=1, zorder=5,
                        label=marker_labels.get(crow['movement'], crow['movement']))
    ax_main.set_xlabel('Frequency (Hz)')
    ax_main.set_ylabel('Autocorr. peak range')
    ax_main.set_ylim(0, 1)
    ax_main.legend(loc='upper left', frameon=False, fontsize=7)
    if show_rejection_annotation:
        ax_main.text(0.98, 0.02,
                     f'Non-theta (outside {_band_label()}): {pct_rejected:.1f}% '
                     f'rejected ({n_rejected}/{len(scope)})',
                     transform=ax_main.transAxes, ha='right', va='bottom', fontsize=7,
                     color='0.3')

    moving_cmap_obj = plt.get_cmap(moving_cmap)
    immobile_cmap_obj = plt.get_cmap(immobile_cmap)

    def _speed_colored_counts(group, value_col, bin_edges, cmap_obj, vmax):
        """Per-bin epoch count of `group`, plus each bin's colour from the
        mean running speed of the epochs that fall in it."""
        n = len(bin_edges) - 1
        idx = np.clip(np.digitize(group[value_col], bin_edges) - 1, 0, n - 1)
        counts = np.zeros(n)
        colors = np.zeros((n, 4))
        for b in range(n):
            m = idx == b
            counts[b] = m.sum()
            if m.any():
                mean_speed = np.clip(group.loc[m, 'median_speed_cms'].mean(), 0, vmax)
                colors[b] = cmap_obj(mean_speed / vmax)
        return counts, colors

    def _movement_hist(value_col, bin_edges):
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        widths = np.diff(bin_edges)
        imm_counts, imm_colors = _speed_colored_counts(
            immobile, value_col, bin_edges, immobile_cmap_obj, immobile_speed_vmax)
        mov_counts, mov_colors = _speed_colored_counts(
            moving, value_col, bin_edges, moving_cmap_obj, speed_vmax)
        return centers, widths, imm_counts, imm_colors, mov_counts, mov_colors

    # bins always span the FULL (both-group) range, so mode='moving' /
    # 'immobile' single-group panels stay axis-comparable with the merged one
    freq_bins = np.linspace(sub['freq'].min(), sub['freq'].max(), n_bins)
    centers_f, widths_f, imm_f, imm_colors_f, mov_f, colors_f = _movement_hist('freq', freq_bins)
    ax_top.bar(centers_f, imm_f, width=widths_f, color=imm_colors_f,
              edgecolor=immobile_edgecolor, linewidth=immobile_edgewidth,
              alpha=immobile_alpha, zorder=2)
    ax_top.bar(centers_f, mov_f, width=widths_f, color=colors_f,
              edgecolor='none', alpha=moving_alpha, zorder=3)
    ax_top.set_ylabel('Count')
    ax_top.tick_params(labelbottom=False)
    if log_counts:
        ax_top.set_yscale('log')

    pr_bins = np.linspace(0, 1, n_bins)
    centers_p, widths_p, imm_p, imm_colors_p, mov_p, colors_p = _movement_hist('peakrangenorm', pr_bins)
    ax_right.barh(centers_p, imm_p, height=widths_p, color=imm_colors_p,
                 edgecolor=immobile_edgecolor, linewidth=immobile_edgewidth,
                 alpha=immobile_alpha, zorder=2)
    ax_right.barh(centers_p, mov_p, height=widths_p, color=colors_p,
                 edgecolor='none', alpha=moving_alpha, zorder=3)
    ax_right.set_xlabel('Count')
    ax_right.tick_params(labelleft=False)
    if log_counts:
        ax_right.set_xscale('log')

    if animal:
        ax_top.set_title(animal)

    return sc, sc_imm, centroids


def _add_speed_colorbars(fig, ax_right, sc_moving, sc_immobile, fraction=0.2, pad=0.35,
                         label_fontsize=None):
    """Speed colorbar(s) to the right of `ax_right` for whichever of the
    moving (`sc_moving`) / immobile (`sc_immobile`) scatters were drawn.
    Space is taken from `ax_right` once (same as a single fig.colorbar call
    with `fraction`/`pad`) and split vertically when both are present --
    moving on top, immobile below -- so the panel doesn't shrink twice.
    """
    mappables = [(sc, label) for sc, label in (
        (sc_moving, 'moving speed (cm s$^{-1}$)'),
        (sc_immobile, 'immobile speed (cm s$^{-1}$)')) if sc is not None]
    if not mappables:
        return []

    cax, _kw = make_axes(ax_right, fraction=fraction, pad=pad)
    pos = cax.get_position()
    cax.remove()

    n = len(mappables)
    gap = 0.1 * pos.height if n > 1 else 0.0
    height = (pos.height - gap * (n - 1)) / n
    cbars = []
    for i, (sc, label) in enumerate(mappables):
        y0 = pos.y1 - (i + 1) * height - i * gap
        cbar = fig.colorbar(sc, cax=fig.add_axes([pos.x0, y0, pos.width, height]))
        cbar.set_label(label, fontsize=label_fontsize)
        cbars.append(cbar)
    return cbars


def plot_freq_peakrange_summary(df, animal=None, speed_vmax=50.0, moving_cmap='viridis',
                                moving_alpha=0.5, immobile_speed_vmax=SPEED_THRESH_CMS,
                                immobile_cmap='magma', immobile_alpha=1.0,
                                immobile_edgecolor='#FFB3BA', figsize=(5.5, 5)):
    """Frequency vs. normalised autocorrelogram peak range. Immobile epochs
    are coloured by per-epoch running speed in `immobile_cmap` with a
    pastel-red outline; moving epochs are coloured by per-epoch running
    speed in `moving_cmap` at `moving_alpha` opacity, so the two
    heavily-overlapping point clouds stay distinguishable. Includes marginal
    count histograms, one speed colorbar per group, and moving/immobile
    centroid markers -- reproduces Dunn et al. Fig. 4e/h.

    Returns (fig, axes_dict, centroids_df).
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          wspace=0.05, hspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    sc, sc_imm, centroids = _draw_freq_peakrange_scatter(
        df, ax_main, ax_top, ax_right, animal=animal, speed_vmax=speed_vmax,
        moving_cmap=moving_cmap, moving_alpha=moving_alpha,
        immobile_speed_vmax=immobile_speed_vmax, immobile_cmap=immobile_cmap,
        immobile_alpha=immobile_alpha, immobile_edgecolor=immobile_edgecolor)

    _add_speed_colorbars(fig, ax_right, sc, sc_imm, fraction=0.2, pad=0.35)

    return fig, {'main': ax_main, 'top': ax_top, 'right': ax_right}, centroids


def _draw_freq_peakrange_pooled(df, ax_main, ax_top, ax_right, animals, point_size=14,
                                alpha=1.0, n_bins=25, log_counts=True,
                                show_rejection_annotation=True):
    """Movement-pooled counterpart of _draw_freq_peakrange_scatter: frequency
    vs. normalised autocorrelogram peak range for every accepted epoch, with
    moving and immobile epochs combined and no speed colouring. Each animal in
    `animals` is drawn in its own ANIMAL_COLORS (Pastel1) colour, with a thin
    dark outline so the light colours stay visible on white, plus marginal
    count histograms and one centroid diamond per animal.

    Same epoch selection as _draw_freq_peakrange_scatter (non-skipped,
    movement in moving/immobile), so the pooled cloud is exactly the union of
    the two movement-segregated clouds. With several animals, the one with the
    most epochs is drawn first so smaller animals' points stay visible on top.

    Returns centroids DataFrame (animal, freq_centroid, peakrangenorm_centroid, n).
    """
    scope = df[df['animal'].isin(animals)]
    n_rejected = int(scope['skipped_outofband'].sum())
    pct_rejected = 100 * n_rejected / len(scope) if len(scope) else 0.0

    sub = scope[~scope['skipped'] & scope['movement'].isin(['moving', 'immobile'])]
    if sub.empty:
        raise ValueError(f'No accepted moving/immobile epochs for {", ".join(animals)}.')

    freq_bins = np.linspace(sub['freq'].min(), sub['freq'].max(), n_bins)
    pr_bins = np.linspace(0, 1, n_bins)
    edge = MARKER_EDGE['markeredgecolor']

    draw_order = sorted(animals, key=lambda a: -int((sub['animal'] == a).sum()))
    for animal_label in draw_order:
        a_sub = sub[sub['animal'] == animal_label]
        if a_sub.empty:
            continue
        color = to_rgba(animal_color(animal_label), alpha)
        ax_main.scatter(a_sub['freq'], a_sub['peakrangenorm'], s=point_size,
                        facecolor=color, edgecolor=edge, linewidth=0.3, zorder=2,
                        label=f'{animal_label} (n={len(a_sub)})')
        ax_top.hist(a_sub['freq'], bins=freq_bins, histtype='stepfilled', log=log_counts,
                    facecolor=color, edgecolor=edge, linewidth=0.5)
        ax_right.hist(a_sub['peakrangenorm'], bins=pr_bins, histtype='stepfilled',
                      orientation='horizontal', log=log_counts,
                      facecolor=color, edgecolor=edge, linewidth=0.5)

    centroids = compute_group_centroids(sub, group_col='animal')
    for _, crow in centroids.iterrows():
        ax_main.scatter(crow['freq_centroid'], crow['peakrangenorm_centroid'],
                        marker='D', s=90, facecolor=animal_color(crow['animal']),
                        edgecolor='black', linewidth=1, zorder=5)
    # legend proxy for the per-animal centroid diamonds
    ax_main.scatter([], [], marker='D', s=90, facecolor='white', edgecolor='black',
                    linewidth=1, label='centroid')

    ax_main.set_xlabel('Frequency (Hz)')
    ax_main.set_ylabel('Autocorr. peak range')
    ax_main.set_ylim(0, 1)
    ax_main.legend(loc='upper left', frameon=False, fontsize=7)
    if show_rejection_annotation:
        ax_main.text(0.98, 0.02,
                     f'Non-theta (outside {_band_label()}): {pct_rejected:.1f}% '
                     f'rejected ({n_rejected}/{len(scope)})',
                     transform=ax_main.transAxes, ha='right', va='bottom', fontsize=7,
                     color='0.3')

    ax_top.set_ylabel('Count')
    ax_top.tick_params(labelbottom=False)
    ax_right.set_xlabel('Count')
    ax_right.tick_params(labelleft=False)
    return centroids


def plot_freq_peakrange_pooled(df, animals=None, alpha=1.0, figsize=(5.5, 5)):
    """Frequency vs. normalised autocorrelogram peak range with moving and
    immobile epochs combined (no movement segregation, no speed colormap) --
    each animal in its ANIMAL_COLORS colour, with marginal count histograms
    and per-animal centroid diamonds (see _draw_freq_peakrange_pooled).

    `animals` : one animal label (per-animal figure), a list of labels, or
                None for every animal in `df` on one cumulative figure.

    Returns (fig, axes_dict, centroids_df).
    """
    if animals is None:
        animals = ordered_animals(df['animal'])
    elif isinstance(animals, str):
        animals = [animals]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          wspace=0.05, hspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    centroids = _draw_freq_peakrange_pooled(df, ax_main, ax_top, ax_right, animals,
                                            alpha=alpha)
    ax_top.set_title(f'{animals[0]}: all epochs' if len(animals) == 1
                     else 'All animals: all epochs')

    return fig, {'main': ax_main, 'top': ax_top, 'right': ax_right}, centroids


def plot_freq_peakrange_by_movement(df, animal=None, speed_vmax=50.0, moving_cmap='viridis',
                                    moving_alpha=0.5, immobile_speed_vmax=SPEED_THRESH_CMS,
                                    immobile_cmap='magma', immobile_alpha=1.0,
                                    immobile_edgecolor='#FFB3BA', figsize=(15, 5)):
    """Three side-by-side Fig. 4e/h-style panels -- mobility only, immobility
    only, and both merged -- sharing the same frequency/peak-range axes and
    bins so the two point clouds can first be inspected on their own, then
    compared for overlap in the merged panel. Moving epochs are coloured by
    speed in `moving_cmap`, immobile epochs by speed in `immobile_cmap` (with
    a pastel-red outline) in every panel.

    A footer below the panels reports the non-theta rejection rate (within
    `animal`'s scope, if given): the overall percentage of epochs whose
    ED_min sinusoid fell outside ACG_FREQ_RANGE, how that rejected set splits
    between immobility and mobility epochs, and -- correcting for imbalanced
    moving/immobile epoch counts -- the percentage of immobility epochs
    (resp. mobility epochs) that were themselves rejected as non-theta.

    Returns (fig, centroids_df) -- centroids_df is from the merged panel.
    """
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(1, 3, wspace=0.6, bottom=0.22, top=0.88)

    centroids = None
    for col, (mode, title) in enumerate((('moving', 'Mobility'),
                                         ('immobile', 'Immobility'),
                                         ('both', 'Merged'))):
        sub_gs = outer[0, col].subgridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                                           wspace=0.05, hspace=0.05)
        ax_main = fig.add_subplot(sub_gs[1, 0])
        ax_top = fig.add_subplot(sub_gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(sub_gs[1, 1], sharey=ax_main)

        sc, sc_imm, c = _draw_freq_peakrange_scatter(
            df, ax_main, ax_top, ax_right, animal=animal, mode=mode,
            speed_vmax=speed_vmax, moving_cmap=moving_cmap, moving_alpha=moving_alpha,
            immobile_speed_vmax=immobile_speed_vmax, immobile_cmap=immobile_cmap,
            immobile_alpha=immobile_alpha, immobile_edgecolor=immobile_edgecolor,
            show_rejection_annotation=False)
        ax_top.set_title(title)
        _add_speed_colorbars(fig, ax_right, sc, sc_imm, fraction=0.25, pad=0.45,
                             label_fontsize=7)
        if mode == 'both':
            centroids = c

    if animal:
        fig.suptitle(animal)

    scope = df if animal is None else df[df['animal'] == animal]
    n_rejected = int(scope['skipped_outofband'].sum())
    pct_rejected = 100 * n_rejected / len(scope) if len(scope) else 0.0
    rej = scope[scope['skipped_outofband']]
    n_rej_immobile = int((rej['movement'] == 'immobile').sum())
    n_rej_moving = int((rej['movement'] == 'moving').sum())
    pct_rej_immobile = 100 * n_rej_immobile / n_rejected if n_rejected else 0.0
    pct_rej_moving = 100 * n_rej_moving / n_rejected if n_rejected else 0.0

    # Same rejected counts, but normalised within each movement state's own
    # total (not the pooled rejected-epoch total above) -- e.g. 1/100
    # immobility epochs rejected as non-theta -> 1%, independent of how many
    # moving epochs there are. This corrects for imbalanced moving/immobile
    # epoch counts (e.g. an animal that was active for most of the session).
    n_immobile_total = int((scope['movement'] == 'immobile').sum())
    n_moving_total = int((scope['movement'] == 'moving').sum())
    pct_immobile_rej_of_immobile = 100 * n_rej_immobile / n_immobile_total if n_immobile_total else 0.0
    pct_moving_rej_of_moving = 100 * n_rej_moving / n_moving_total if n_moving_total else 0.0

    band = _band_label()
    fig.text(0.5, 0.02,
             f'Non-theta (ED_min sinusoid outside {band}) rejected {n_rejected}/{len(scope)} '
             f'epochs overall ({pct_rejected:.1f}%)\n'
             f'Non-theta rejected epochs in immobility ({pct_rej_immobile:.1f}%) | '
             f'Non-theta rejected epochs in mobility ({pct_rej_moving:.1f}%)\n'
             f'% of total immobility epochs rejected as non-theta: '
             f'{pct_immobile_rej_of_immobile:.1f}% ({n_rej_immobile}/{n_immobile_total}) | '
             f'% of total mobility epochs rejected as non-theta: '
             f'{pct_moving_rej_of_moving:.1f}% ({n_rej_moving}/{n_moving_total})',
             ha='center', va='bottom', fontsize=8, color='0.2')

    return fig, centroids


def plot_figure4_style(df, animal=None, trace_color='tab:blue',
                       speed_vmax=50.0, moving_cmap='viridis',
                       moving_alpha=0.5, immobile_speed_vmax=SPEED_THRESH_CMS,
                       immobile_cmap='magma', immobile_alpha=1.0,
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
    sc, sc_imm, centroids = _draw_freq_peakrange_scatter(
        df, ax_main, ax_top, ax_right, animal=animal, speed_vmax=speed_vmax,
        moving_cmap=moving_cmap, moving_alpha=moving_alpha,
        immobile_speed_vmax=immobile_speed_vmax, immobile_cmap=immobile_cmap,
        immobile_alpha=immobile_alpha, immobile_edgecolor=immobile_edgecolor)

    _add_speed_colorbars(fig, ax_right, sc, sc_imm, fraction=0.2, pad=0.35)

    if animal:
        fig.suptitle(animal)

    return fig, centroids


# %% ==================== Run metadata ===============================================

def _script_path():
    """Absolute path of this script (falls back to the working directory when
    run cell-by-cell/interactively, where __file__ isn't defined)."""
    try:
        return os.path.abspath(__file__)
    except NameError:
        return os.path.join(os.getcwd(), 'Figc_ACG_ScalingCorrected_v4_SessionSegregation.py')


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
        'script_name':              os.path.basename(script_path),
        'script_dir':               os.path.dirname(script_path),
        'run_datetime':             datetime.datetime.now().isoformat(timespec='seconds'),
        'data_dir (ROOT_DIR)':      ROOT_DIR,
        'OUTPUT_DIR':               OUTPUT_DIR,
        'FIGURE_DIR':               FIGURE_DIR,
        # ---- acquisition ----
        'NATIVE_FS':                NATIVE_FS,
        'ADBitVolts':               'per-file, from .ncs header (-ADBitVolts)',
        'ADBITVOLTS_NO_HEADER':     ADBITVOLTS_NO_HEADER,
        'ncs_dtype':                str(ncs_dtype),
        # ---- LFP cleaning (resample -> notch -> detrend -> low-pass) ----
        'FS_ACG':                   FS_ACG,
        'LINE_HARMONICS':           LINE_HARMONICS,
        'NOTCH_Q':                  NOTCH_Q,
        'detrend_type':             'linear',
        'LOWPASS_CUTOFF_HZ':        LOWPASS_CUTOFF_HZ,
        'LOWPASS_ORDER':            LOWPASS_ORDER,
        # ---- epoching / epoch rejection ----
        'EPOCH_SEC':                EPOCH_SEC,
        'DELTA_BAND':               DELTA_BAND,
        'THETA_BAND':               THETA_BAND,
        'welch_window':             'hann',
        'welch_noverlap':           0,
        'MAD_THRESH':               MAD_THRESH,
        'MAD_robust_z_constant':    0.6745,
        # ---- tracking / movement labelling ----
        'tracking_csv_columns':     'A=time_us, D=x_cm, E=y_cm',
        'POS_JUMP_THRESH_CMS':      POS_JUMP_THRESH_CMS,
        'POS_SMOOTH_SIGMA_SMP':     POS_SMOOTH_SIGMA_SMP,
        'SPEED_THRESH_CMS':         SPEED_THRESH_CMS,
        # ---- arena / session-type segregation ----
        'arena_session_source':     '.ncs folder path (session folder shared with the .ntt files)',
        'ARENA_TYPES':              ARENA_TYPES,
        'ARENA_SESSION_TYPES':      ARENA_SESSION_TYPES,
        'ARENA_FIGURE_DIR':         ARENA_FIGURE_DIR,
        'SESSION_FIGURE_DIR':       SESSION_FIGURE_DIR,
        'STATS_ALPHA':              STATS_ALPHA,
        'prevalence_tests':         'epoch: chi-square / pairwise Yates or Fisher (Holm); '
                                    'session: Kruskal-Wallis / pairwise Mann-Whitney U (Holm)',
        # ---- ACG reference-sinusoid matching ----
        'ACG_FREQ_RANGE':           ACG_FREQ_RANGE,
        'ACG_REF_BANK':             ACG_REF_BANK,
        'ACG_FREQ_RES':             ACG_FREQ_RES,
        'ACG_PEAK_PROMINENCE':      ACG_PEAK_PROMINENCE,
        'theta_epoch_criterion':    'ED_min reference sinusoid frequency within ACG_FREQ_RANGE '
                                    '(no ED_min value threshold)',
        # ---- summaries / plots ----
        'FREQ_HIST_BIN_EDGES':      FREQ_HIST_BIN_EDGES,
        'freq_est_histogram_bins':  30,
        'moving_speed_vmax_cms':    50.0,
        'immobile_speed_vmax_cms':  SPEED_THRESH_CMS,
        'moving_cmap':              'viridis',
        'immobile_cmap':            'magma',
        'marginal_hist_n_bins':     25,
        'pooled_plot_colors':       ANIMAL_COLORS,
        'pooled_scatter_alpha':     1.0,
        'figure_dpi':               200,
    }
    os.makedirs(meta_dir, exist_ok=True)
    out_path = os.path.join(meta_dir, 'run_metadata.csv')
    pd.DataFrame({'parameter': list(meta.keys()),
                  'value':     [str(v) for v in meta.values()]}).to_csv(out_path, index=False)
    print(f'Saved run metadata -> {out_path}')

    if os.path.isfile(script_path):
        shutil.copy2(script_path, meta_dir)
        print(f'Saved script copy -> {os.path.join(meta_dir, os.path.basename(script_path))}')
    else:
        print(f'  Script not found at {script_path} -- no copy saved to {meta_dir}')


# %% ==================== Arena / session-type non-theta prevalence stats ===========
#
# Compares how often epochs are rejected as non-theta -- ED_min sinusoid
# outside ACG_FREQ_RANGE ('skipped_outofband') -- across arena types, and
# across the session types within each arena -- the same values as the
# plot_freq_peakrange_by_movement footer:
#   * % of all epochs rejected as non-theta
#   * % of immobility epochs / % of mobility epochs rejected (each
#     normalised within its own movement state)
#   * split of the rejected epochs between immobility and mobility
# Two levels of test:
#   * epoch level -- every epoch counted as independent: chi-square test of
#     independence on the groups x (rejected, kept) table, then pairwise tests
#     between groups (Yates chi-square, or Fisher's exact when an expected
#     count is < 5), Holm-Bonferroni corrected within each measure. Also,
#     within each group, immobility vs. mobility rejection rate (2 x 2),
#     Holm-corrected across groups.
#   * session level -- one % per recording session (all its .ncs channels
#     pooled), Kruskal-Wallis across groups + pairwise Mann-Whitney U (Holm).
#     Epochs of one session/channel aren't independent, so this is the more
#     conservative of the two.

# (key, description, rejected-count column, denominator column)
PREVALENCE_MEASURES = [
    ('overall',  '% of all epochs rejected as non-theta',
     'n_rejected', 'n_epochs'),
    ('immobile', '% of immobility epochs rejected as non-theta',
     'n_immobile_rejected', 'n_immobile'),
    ('moving',   '% of mobility epochs rejected as non-theta',
     'n_moving_rejected', 'n_moving'),
    ('rejected_in_immobility', '% of non-theta rejected epochs that are immobility epochs',
     'n_immobile_rejected', 'n_rejected'),
]

# Arena / session-type bar colours, as in ArenaSessionType_StatsComparison_v8.py
# (magenta = circular track, cyan = linear track, blue = open arena; session
# types are shades of their arena's hue).
GROUP_FACE = {
    'Open': '#7FB3E5', 'Linear': '#7FFAFA', 'Circle': '#FA7FFA',
    'Cntrl': '#7FB3E5', 'Rotate': '#0066CC', 'Zero': '#BFBFBF',
    '0': '#008080', '90': '#00B3B3', '180': '#33D6D6', '270': '#7FFAFA',
    'Stnd': '#FDC4FD', 'NoRot': '#FA7FFA', 'Rot': '#C000C0',
}
IMMOBILE_COLOR, MOVING_COLOR = '#7F8C8D', '#E67E22'   # as in plot_moving_vs_immobile


def rejection_counts(df, by):
    """Epoch and non-theta-rejected-epoch counts (all / immobility / mobility)
    per group of the `by` columns, plus the PREVALENCE_MEASURES percentages
    ('pct_<key>' columns, NaN where the denominator is 0) and
    'pct_rejected_in_mobility'."""
    rej = df['skipped_outofband'].astype(bool)
    imm = df['movement'].eq('immobile')
    mov = df['movement'].eq('moving')
    flags = pd.DataFrame({
        'n_epochs': 1, 'n_rejected': rej,
        'n_immobile': imm, 'n_immobile_rejected': rej & imm,
        'n_moving': mov, 'n_moving_rejected': rej & mov,
    }, index=df.index).astype(int)
    counts = flags.groupby([df[c] for c in by]).sum().reset_index()

    def _pct(k_col, n_col):
        return 100 * counts[k_col] / counts[n_col].where(counts[n_col] > 0)

    for key, _desc, k_col, n_col in PREVALENCE_MEASURES:
        counts[f'pct_{key}'] = _pct(k_col, n_col)
    counts['pct_rejected_in_mobility'] = _pct('n_moving_rejected', 'n_rejected')
    return counts


def _holm(pvals):
    """Holm-Bonferroni adjusted p-values (NaNs stay NaN, not counted)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.where(np.isfinite(p))[0]
    running = 0.0
    for rank, i in enumerate(ok[np.argsort(p[ok])]):
        running = max(running, (len(ok) - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def _holm_rows(rows):
    """Fill 'p_holm' of a family of pairwise test rows."""
    for row, p_adj in zip(rows, _holm([r['p_raw'] for r in rows])):
        row['p_holm'] = p_adj
    return rows


def _test_row(level, measure, comparison, res):
    row = {'level': level, 'measure': measure, 'comparison': comparison,
           'test': 'insufficient data', 'stat_name': '', 'statistic': np.nan,
           'dof': np.nan, 'p_raw': np.nan, 'p_holm': np.nan, 'note': ''}
    if res:
        row.update(res)
    return row


def _contingency_test(table):
    """Test of independence on a groups x categories count table: Fisher's
    exact for a 2 x 2 table with an expected count < 5, otherwise chi-square
    (Yates-corrected for 2 x 2). All-zero rows/columns are dropped first.
    Returns a result dict for _test_row, or None if fewer than 2 rows or
    columns remain (e.g. no rejected epochs in any group)."""
    arr = np.asarray(table, dtype=float)
    if arr.ndim != 2:
        return None
    arr = arr[arr.sum(axis=1) > 0]
    arr = arr[:, arr.sum(axis=0) > 0]
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    chi2, p, dof, expected = chi2_contingency(arr)
    n_low = int((expected < 5).sum())
    if arr.shape == (2, 2) and n_low:
        odds, p = fisher_exact(arr)
        return {'test': "Fisher's exact", 'stat_name': 'OR', 'statistic': odds,
                'p_raw': p, 'note': f'{n_low}/4 expected counts < 5'}
    note = (f'{n_low}/{expected.size} expected counts < 5 -- chi-square '
            f'approximation may be unreliable') if n_low else ''
    return {'test': 'Chi-square' + (' (Yates)' if arr.shape == (2, 2) else ''),
            'stat_name': 'chi2', 'statistic': chi2, 'dof': dof, 'p_raw': p, 'note': note}


def _kruskal_test(vals):
    """Kruskal-Wallis on {group: values}; None if < 2 groups or all values identical."""
    if len(vals) < 2:
        return None
    try:
        h, p = kruskal(*vals.values())
    except ValueError:
        return None
    if not np.isfinite(p):
        return None
    return {'test': 'Kruskal-Wallis', 'stat_name': 'H', 'statistic': h,
            'dof': len(vals) - 1, 'p_raw': p}


def _mannwhitney_test(a, b):
    try:
        u, p = mannwhitneyu(a, b, alternative='two-sided')
    except ValueError:
        return None
    return {'test': 'Mann-Whitney U', 'stat_name': 'U', 'statistic': u, 'p_raw': p}


def compare_rejection_prevalence(df, group_col, order, grouping):
    """Non-theta rejection prevalence (PREVALENCE_MEASURES) compared across the
    groups `order` of `group_col` ('arena_type' or 'session_type') -- tests
    described in the section header. `grouping` names the comparison in the
    output tables (e.g. 'Arena types', 'Linear session types').

    Returns (desc_df, sessions_df, tests_df): per-group counts/percentages,
    per-session counts/percentages, and one row per test ('comparison' is
    'omnibus', 'A vs B', or -- for the within-group immobility vs. mobility
    test -- the group name). 'significant' uses p_holm where it applies,
    p_raw for omnibus tests.
    """
    df = df[df[group_col].isin(order)]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    present = [g for g in order if (df[group_col] == g).any()]

    desc = (rejection_counts(df, [group_col]).rename(columns={group_col: 'group'})
            .set_index('group').loc[present].reset_index())
    sessions = (rejection_counts(df, [group_col, 'session_folder'])
                .rename(columns={group_col: 'group'}))
    desc.insert(1, 'n_sessions', desc['group'].map(sessions.groupby('group').size()).to_numpy())
    desc.insert(2, 'n_files', desc['group'].map(df.groupby(group_col)['file'].nunique()).to_numpy())

    rows = []
    for key, _desc, k_col, n_col in PREVALENCE_MEASURES:
        # epoch level
        d = desc[desc[n_col] > 0]
        table = {g: [k, n - k] for g, k, n in zip(d['group'], d[k_col], d[n_col])}
        rows.append(_test_row('epoch', key, 'omnibus', _contingency_test(list(table.values()))))
        rows += _holm_rows([_test_row('epoch', key, f'{a} vs {b}',
                                      _contingency_test([table[a], table[b]]))
                            for a, b in combinations(table, 2)])
        # session level
        vals = {g: sessions.loc[sessions['group'] == g, f'pct_{key}'].dropna().to_numpy(dtype=float)
                for g in present}
        vals = {g: v for g, v in vals.items() if len(v)}
        rows.append(_test_row('session', key, 'omnibus', _kruskal_test(vals)))
        rows += _holm_rows([_test_row('session', key, f'{a} vs {b}',
                                      _mannwhitney_test(vals[a], vals[b]))
                            for a, b in combinations(vals, 2)])

    # within each group: is the rejection rate different between immobility
    # and mobility epochs?
    rows += _holm_rows([
        _test_row('epoch (within group)', 'immobile vs moving', r['group'], _contingency_test([
            [r['n_immobile_rejected'], r['n_immobile'] - r['n_immobile_rejected']],
            [r['n_moving_rejected'], r['n_moving'] - r['n_moving_rejected']]]))
        for _, r in desc.iterrows()])

    tests = pd.DataFrame(rows)
    tests['significant'] = tests['p_holm'].fillna(tests['p_raw']) < STATS_ALPHA
    for tbl in (desc, sessions, tests):
        tbl.insert(0, 'grouping', grouping)
    return desc, sessions, tests


def _stars(p):
    if p is None or not np.isfinite(p):
        return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < STATS_ALPHA else 'n.s.'


def _fmt_p(p):
    """'p < 0.001', 'p = 0.123' or 'p = n/a'."""
    if p is None or not np.isfinite(p):
        return 'p = n/a'
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


def _draw_sig_brackets(ax, groups, pair_tests, y_base, y_step):
    """Line + stars over each pair of `groups` whose row in `pair_tests`
    ('comparison' = 'A vs B') has p_holm < STATS_ALPHA, narrower spans
    lowest so lines don't cross. Returns the y of the highest line, or None."""
    pos = {g: i for i, g in enumerate(groups)}
    sig = []
    for _, r in pair_tests.iterrows():
        if not (np.isfinite(r['p_holm']) and r['p_holm'] < STATS_ALPHA):
            continue
        a, b = r['comparison'].split(' vs ')
        if a in pos and b in pos:
            x1, x2 = sorted((pos[a], pos[b]))
            sig.append((x2 - x1, x1, x2, _stars(r['p_holm'])))
    y = y_base
    for _span, x1, x2, stars in sorted(sig):
        ax.plot([x1, x2], [y, y], color='black', linewidth=1, zorder=5)
        ax.text((x1 + x2) / 2, y, stars, ha='center', va='bottom', fontsize=11,
                fontweight='bold', zorder=5)
        y += y_step
    return y - y_step if sig else None


def plot_rejection_prevalence(desc, sessions, tests, title=None):
    """Four panels comparing the groups of one compare_rejection_prevalence
    result: % of all / immobility / mobility epochs rejected as non-theta
    (bars = all epochs of the group pooled, labelled with
    rejected/total; dots = individual recording sessions; brackets =
    epoch-level pairwise tests with Holm p < STATS_ALPHA), and the split of
    the rejected epochs between immobility and mobility (stacked bars).
    Omnibus results at both test levels are listed under the panels.

    Returns fig.
    """
    groups = list(desc['group'])
    x = np.arange(len(groups))
    faces = [GROUP_FACE.get(g, '#BFBFBF') for g in groups]
    fig, axes = plt.subplots(1, 4, figsize=(4 * (1.6 + 0.7 * len(groups)), 4.6))
    rng = np.random.default_rng(0)

    def _pair_tests(key):
        return tests[(tests['level'] == 'epoch') & (tests['measure'] == key)
                     & (tests['comparison'] != 'omnibus')]

    for ax, (key, measure_desc, k_col, n_col) in zip(axes[:3], PREVALENCE_MEASURES[:3]):
        pct = desc[f'pct_{key}'].to_numpy(dtype=float)
        ax.bar(x, np.nan_to_num(pct), width=0.6, color=faces, edgecolor='0.25',
               linewidth=0.6, zorder=2)
        top = np.nanmax(pct) if np.isfinite(pct).any() else 0.0
        for xi, g in zip(x, groups):
            v = sessions.loc[sessions['group'] == g, f'pct_{key}'].dropna().to_numpy(dtype=float)
            ax.scatter(xi + rng.uniform(-0.15, 0.15, len(v)), v, s=12, facecolor='white',
                       edgecolor='0.25', linewidth=0.6, zorder=3)
            if len(v):
                top = max(top, v.max())
        top = top if top > 0 else 1.0
        y_hi = _draw_sig_brackets(ax, groups, _pair_tests(key), top * 1.08, top * 0.12)
        ax.set_ylim(0, (y_hi if y_hi is not None else top) * 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{_label(g)}\n{p:.1f}%\n({int(k)}/{int(n)})' if np.isfinite(p)
                            else f'{_label(g)}\nn/a'
                            for g, p, k, n in zip(groups, pct, desc[k_col], desc[n_col])],
                           fontsize=8)
        ax.set_ylabel('% of epochs')
        ax.set_title(textwrap.fill(measure_desc, 30), fontsize=9)
    axes[0].scatter([], [], s=12, facecolor='white', edgecolor='0.25', linewidth=0.6,
                    label='one session')
    axes[0].legend(loc='upper right', fontsize=7, frameon=False)

    ax = axes[3]
    imm = np.nan_to_num(desc['pct_rejected_in_immobility'].to_numpy(dtype=float))
    mov = np.nan_to_num(desc['pct_rejected_in_mobility'].to_numpy(dtype=float))
    ax.bar(x, imm, width=0.6, color=IMMOBILE_COLOR, edgecolor='white', linewidth=0.6,
           label='immobility', zorder=2)
    ax.bar(x, mov, bottom=imm, width=0.6, color=MOVING_COLOR, edgecolor='white',
           linewidth=0.6, label='mobility', zorder=2)
    y_hi = _draw_sig_brackets(ax, groups, _pair_tests('rejected_in_immobility'), 104, 8)
    ax.set_ylim(0, max(100, y_hi or 100) + 8)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xticks(x)
    ax.set_xticklabels([f'{_label(g)}\n(n={int(n)}\nrejected)'
                        for g, n in zip(groups, desc['n_rejected'])], fontsize=8)
    ax.set_ylabel('% of non-theta rejected epochs')
    ax.set_title(textwrap.fill(PREVALENCE_MEASURES[3][1], 30), fontsize=9)
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=7, frameon=False)

    for a in axes:
        a.spines[['top', 'right']].set_visible(False)

    lines = []
    for key, measure_desc, *_ in PREVALENCE_MEASURES:
        parts = []
        for level, name in (('epoch', 'epochs'), ('session', 'sessions')):
            r = tests[(tests['level'] == level) & (tests['measure'] == key)
                      & (tests['comparison'] == 'omnibus')]
            if r.empty or not np.isfinite(r['p_raw'].iloc[0]):
                parts.append(f'{name}: insufficient data')
                continue
            r = r.iloc[0]
            dof = f', dof={int(r["dof"])}' if np.isfinite(r['dof']) else ''
            parts.append(f'{name}: {r["test"]} {r["stat_name"]}={r["statistic"]:.2f}{dof}, '
                         f'{_fmt_p(r["p_raw"])} {_stars(r["p_raw"])}')
        lines.append(f'{measure_desc} -- ' + ' | '.join(parts))
    within = tests[tests['level'] == 'epoch (within group)']
    if len(within):
        lines.append('Immobility vs mobility rejection rate within each group (Holm): ' +
                     ', '.join(f'{_label(g)} {_fmt_p(p)} {_stars(p)}'
                               for g, p in zip(within['comparison'], within['p_holm'])))
    lines.append('Bars: all epochs of the group pooled; dots: one recording session each; '
                 f'brackets: pairwise epoch-level tests, Holm p < {STATS_ALPHA:g}')
    fig.text(0.5, -0.02, '\n'.join(lines), ha='center', va='top', fontsize=7.5, color='0.2')

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


# %% ==================== Pipeline (whole dataset or one arena/session subset) ======

def _plot_and_save(path, plot_fn, *args, suptitle=None, **kwargs):
    """Call plot_fn(*args, **kwargs) -- whose (first) return value is the
    figure -- optionally replace its suptitle, save it to `path`, and return
    plot_fn's remaining return values as a tuple. If plotting fails (e.g. an
    arena/session subset with no epochs of one movement state), prints a SKIP
    line and returns None so the rest of the pipeline still runs."""
    try:
        out = plot_fn(*args, **kwargs)
    except Exception as e:
        print(f'  SKIP {os.path.basename(path)}: {type(e).__name__}: {e}')
        plt.close('all')
        return None
    fig, rest = (out[0], out[1:]) if isinstance(out, tuple) else (out, ())
    if suptitle:
        fig.suptitle(suptitle)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure -> {path}')
    return rest


def _moving_vs_immobile_fig(df, animal):
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_moving_vs_immobile(df, animal, ax=ax)
    fig.tight_layout()
    return fig


def run_pipeline(results_df, output_dir, figure_dir, title=None):
    """Every summary table and figure of this script for the epochs in
    `results_df` -- the whole dataset, or one arena / session-type subset of
    it (run_arena_session_pipelines). Tables are written to `output_dir`,
    figures to `figure_dir`. `title` (e.g. 'Linear arena') prefixes the
    figure suptitles.

    Subsetting the epoch table is the same as re-running the pipeline on that
    subset's .ncs files: every cleaning / rejection step (incl. the MAD
    artifact threshold) is computed within one .ncs file.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    def _fig(name):
        return os.path.join(figure_dir, name)

    def _title(text):
        return f'{title}: {text}' if title else text

    summary_df = summarize_peak_range(results_df)
    summary_path = os.path.join(output_dir, 'acg_theta_continuity_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f'Saved summary -> {summary_path}')

    # Matched-sinusoid freq_est histogram, clubbed across ALL epochs (every
    # animal/channel/movement state pooled together)
    _plot_and_save(_fig('freq_est_histogram.png'), plot_freq_est_histogram, results_df)

    freq_range_counts_df = compute_freq_range_counts(results_df)
    freq_range_counts_path = os.path.join(output_dir, 'freq_est_range_counts.csv')
    freq_range_counts_df.to_csv(freq_range_counts_path, index=False)
    print(f'Saved freq_est range counts -> {freq_range_counts_path}')

    # freq_est histogram restricted to epochs rejected as non-theta (ED_min
    # sinusoid outside ACG_FREQ_RANGE), clubbed across ALL epochs
    _plot_and_save(_fig('nontheta_rejected_freq_est_histogram.png'),
                   plot_rejected_freq_histogram, results_df)

    # ED_min vs. normalised peak range -- test whether a poor sinusoid fit
    # (high ED_min) predicts a low peak range, clubbed across ALL epochs
    edmin_pr_stats_rows = []
    out = _plot_and_save(_fig('edmin_vs_peakrangenorm.png'),
                         plot_edmin_vs_peakrangenorm, results_df)
    if out is not None:
        edmin_pr_stats = out[1]
        print(f"  (Pearson r={edmin_pr_stats['pearson_r']:.3f}, "
              f"Spearman rho={edmin_pr_stats['spearman_r']:.3f}, n={edmin_pr_stats['n']})")
        edmin_pr_stats_rows.append({'animal': 'ALL', **edmin_pr_stats})

    all_centroids = []
    for animal_label in results_df['animal'].unique():  # type: ignore
        _plot_and_save(_fig(f'{animal_label}_acg_moving_vs_immobile.png'),
                       _moving_vs_immobile_fig, results_df, animal_label)

        # freq_est histograms of this animal's accepted / non-theta rejected epochs
        _plot_and_save(_fig(f'{animal_label}_freq_est_histogram.png'),
                       plot_freq_est_histogram, results_df, animal=animal_label)
        _plot_and_save(_fig(f'{animal_label}_nontheta_rejected_freq_est_histogram.png'),
                       plot_rejected_freq_histogram, results_df, animal=animal_label)

        # ED_min vs. normalised peak range, this animal only
        out = _plot_and_save(_fig(f'{animal_label}_edmin_vs_peakrangenorm.png'),
                             plot_edmin_vs_peakrangenorm, results_df, animal=animal_label)
        if out is not None:
            edmin_pr_stats_rows.append({'animal': animal_label, **out[1]})

        # Fig. 4-style: example moving/immobile epochs + freq-vs-peak-range
        # scatter with movement centroids
        out = _plot_and_save(_fig(f'{animal_label}_figure4_style.png'),
                             plot_figure4_style, results_df, animal=animal_label,
                             suptitle=_title(animal_label))
        if out is not None:
            centroids = out[0]
            centroids['animal'] = animal_label
            all_centroids.append(centroids)

        # mobility-only / immobility-only / merged, side by side
        _plot_and_save(_fig(f'{animal_label}_freq_peakrange_by_movement.png'),
                       plot_freq_peakrange_by_movement, results_df, animal=animal_label,
                       suptitle=_title(animal_label))

        # moving + immobile epochs pooled (no movement segregation / speed
        # colouring), in this animal's colour
        _plot_and_save(_fig(f'{animal_label}_freq_peakrange_pooled.png'),
                       plot_freq_peakrange_pooled, results_df, animals=animal_label)

    # moving + immobile epochs pooled, every animal on one figure in its own
    # colour, with per-animal centroids
    out = _plot_and_save(_fig('ALL_freq_peakrange_pooled.png'),
                         plot_freq_peakrange_pooled, results_df, suptitle=title)
    if out is not None:
        pooled_centroids_path = os.path.join(output_dir, 'acg_theta_continuity_centroids_pooled.csv')
        out[1].to_csv(pooled_centroids_path, index=False)
        print(f'Saved pooled (moving + immobile) centroids -> {pooled_centroids_path}')

    # Fig. 4-style (moving/immobile, speed-coloured) with every animal's
    # epochs on one figure -- example epochs picked from the all-animal
    # moving/immobile centroids
    out = _plot_and_save(_fig('ALL_figure4_style.png'), plot_figure4_style, results_df,
                         animal=None, suptitle=_title('All animals'))
    if out is not None:
        centroids_all = out[0]
        centroids_all['animal'] = 'ALL'
        all_centroids.append(centroids_all)

    # mobility-only / immobility-only / merged, every animal's epochs together
    _plot_and_save(_fig('ALL_freq_peakrange_by_movement.png'),
                   plot_freq_peakrange_by_movement, results_df, animal=None,
                   suptitle=_title('All animals'))

    # freq_est line-overlays (accepted / non-theta rejected epochs), one line
    # per animal on shared axes for direct comparison
    _plot_and_save(_fig('freq_est_histogram_by_animal_overlay.png'),
                   plot_freq_est_histogram_overlay, results_df)
    _plot_and_save(_fig('nontheta_rejected_freq_est_histogram_by_animal_overlay.png'),
                   plot_rejected_freq_histogram_overlay, results_df)

    if all_centroids:
        centroids_df = pd.concat(all_centroids, ignore_index=True)
        centroids_path = os.path.join(output_dir, 'acg_theta_continuity_centroids.csv')
        centroids_df.to_csv(centroids_path, index=False)
        print(f'Saved movement centroids -> {centroids_path}')

    edmin_pr_stats_path = os.path.join(output_dir, 'edmin_vs_peakrangenorm_stats.csv')
    pd.DataFrame(edmin_pr_stats_rows).to_csv(edmin_pr_stats_path, index=False)
    print(f'Saved ED_min vs. peakrangenorm correlation stats -> {edmin_pr_stats_path}')


def run_arena_session_pipelines(results_df):
    """Re-run the full pipeline (run_pipeline) on each arena type's epochs
    (ARENA_FIGURE_DIR\\<arena>) and on each session type's epochs within an
    arena (SESSION_FIGURE_DIR\\<arena>\\<session type>), then compare
    non-theta rejection prevalence across arena types (ARENA_FIGURE_DIR) and
    across the session types of each arena (SESSION_FIGURE_DIR\\<arena>). The
    prevalence tables from every comparison are written to OUTPUT_DIR:
        nontheta_prevalence_by_group.csv     counts / % per arena or session type
        nontheta_prevalence_by_session.csv   counts / % per recording session
        nontheta_prevalence_tests.csv        every omnibus / pairwise test
    """
    files = results_df.drop_duplicates('file')
    no_arena = files.loc[files['arena_type'].isna(), 'session_folder'].unique()
    if len(no_arena):
        print(f'WARNING: {len(no_arena)} session folder(s) with no recognizable arena '
              f'type ({ARENA_TYPES}); excluded from the arena/session-type analysis:')
        for s in sorted(no_arena):
            print(f'    {s}')
    no_type = files.loc[files['arena_type'].notna() & files['session_type'].isna(),
                        'session_folder'].unique()
    if len(no_type):
        print(f'WARNING: {len(no_type)} session folder(s) with no recognizable session '
              f'type; kept in the arena comparison, excluded from the session-type comparison:')
        for s in sorted(no_type):
            print(f'    {s}')
    if not files['arena_type'].notna().any():
        print('No .ncs file has a recognizable arena type -- skipping the '
              'arena/session-type analysis.')
        return
    print('\n.ncs files by arena and session type:')
    print(pd.crosstab(files['arena_type'], files['session_type'].fillna('unrecognized'),
                      margins=True))

    desc_all, sessions_all, tests_all = [], [], []

    def _compare(df, group_col, order, grouping, fig_dir):
        desc, sessions, tests = compare_rejection_prevalence(df, group_col, order, grouping)
        if desc.empty:
            return
        desc_all.append(desc)
        sessions_all.append(sessions)
        tests_all.append(tests)
        if len(desc) < 2:
            print(f'  Only one group for {grouping} -- no comparison plot.')
            return
        os.makedirs(fig_dir, exist_ok=True)
        _plot_and_save(os.path.join(fig_dir, 'nontheta_prevalence_comparison.png'),
                       plot_rejection_prevalence, desc, sessions, tests,
                       title=f'{grouping}: non-theta (outside {_band_label()}) prevalence')

    for arena in ARENA_TYPES:
        arena_df = results_df[results_df['arena_type'] == arena]
        if arena_df.empty:
            print(f'\nNo epochs for arena {arena}; skipping.')
            continue
        print(f'\n{"=" * 70}\nArena: {arena} ({len(arena_df)} epochs, '
              f'{arena_df["file"].nunique()} .ncs files)\n{"=" * 70}')
        arena_dir = os.path.join(ARENA_FIGURE_DIR, arena)
        run_pipeline(arena_df, arena_dir, arena_dir, title=f'{arena} arena')

        for stype in ARENA_SESSION_TYPES[arena]:
            ses_df = arena_df[arena_df['session_type'] == stype]
            if ses_df.empty:
                continue
            print(f'\n--- {arena} arena, session type {stype} ({len(ses_df)} epochs, '
                  f'{ses_df["file"].nunique()} .ncs files) ---')
            ses_dir = os.path.join(SESSION_FIGURE_DIR, arena, stype)
            run_pipeline(ses_df, ses_dir, ses_dir, title=f'{arena} arena, {_label(stype)} session')

        _compare(arena_df, 'session_type', ARENA_SESSION_TYPES[arena],
                 f'{arena} session types', os.path.join(SESSION_FIGURE_DIR, arena))

    _compare(results_df, 'arena_type', ARENA_TYPES, 'Arena types', ARENA_FIGURE_DIR)

    if not tests_all:
        print('No arena/session-type groups to compare.')
        return
    for name, parts in (('nontheta_prevalence_by_group.csv', desc_all),
                        ('nontheta_prevalence_by_session.csv', sessions_all),
                        ('nontheta_prevalence_tests.csv', tests_all)):
        path = os.path.join(OUTPUT_DIR, name)
        pd.concat(parts, ignore_index=True).to_csv(path, index=False, encoding='utf-8-sig')
        print(f'Saved non-theta prevalence table -> {path}')

    tests = pd.concat(tests_all, ignore_index=True)
    sig = tests[tests['significant']]
    if sig.empty:
        print('\nNo significant non-theta prevalence differences.')
    else:
        print(f'\nSignificant non-theta prevalence differences (p < {STATS_ALPHA:g}):')
        for _, r in sig.iterrows():
            p = r['p_holm'] if np.isfinite(r['p_holm']) else r['p_raw']
            print(f"  {r['grouping']:22s} {r['level']:20s} {r['measure']:24s} "
                  f"{r['comparison']:24s} {r['test']:16s} p={p:.4g}")


# %% ==================== Driver =====================================================

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    save_run_metadata(META_DIR)

    results_df = process_root_directory(ROOT_DIR)

    n_outofband = int(results_df['skipped_outofband'].sum())
    pct_outofband = 100 * n_outofband / len(results_df) if len(results_df) else 0.0
    print(f'freq_est outside {ACG_FREQ_RANGE} (non-theta, bank {ACG_REF_BANK} Hz) rejected '
          f'{n_outofband}/{len(results_df)} epochs overall ({pct_outofband:.1f}%)')

    results_path = os.path.join(OUTPUT_DIR, 'acg_theta_continuity_epochs.csv')
    results_df.to_csv(results_path, index=False)
    print(f'Saved {len(results_df)} epoch-level rows -> {results_path}')

    # whole dataset
    run_pipeline(results_df, OUTPUT_DIR, FIGURE_DIR)

    # segregated by arena type and by session type (+ non-theta prevalence stats)
    run_arena_session_pipelines(results_df)
