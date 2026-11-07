# Fig4a_PSD_v3_PeriodicOnly.py
# Same PSD pipeline as Fig4a_PSD_v2_Simul_ScalingCorrected.py (ncsDropCorr),
# plus aperiodic removal: each file's own FOOOF aperiodic fit is subtracted
# from its PSD in log10 space, and every analysis after that (mean spectra,
# FOOOF peaks / theta range, band-by-band, group differences) runs on the
# periodic spectrum only. See 'Aperiodic removal' in the config block.
# Output goes to a separate folder (OUTPUT_DIR ..._PeriodicOnly), so v2
# outputs are not overwritten.
import os
import re
import sys
import math
import pickle
import shutil
import getpass
import hashlib
import datetime
import platform
import itertools
import subprocess
import importlib.metadata

import numpy as np
import pandas as pd
from scipy import signal, stats, interpolate, ndimage
from scipy.signal import savgol_filter

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from fooof import FOOOF, FOOOFGroup
from fooof.utils import interpolate_spectrum, trim_spectrum
from fooof.analysis.periodic import get_band_peak
from fooof.analysis import get_band_peak_fm
from fooof.bands import Bands
from fooof.plts.spectra import plot_spectra_shading
from fooof.sim.gen import gen_aperiodic

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# seaborn 0.13.2 (latest) still passes `vert=` to ax.bxp internally; matplotlib 3.11 warns
warnings.filterwarnings("ignore", message=".*vert: bool was.*deprecated", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*vert: bool was.*deprecated", module="seaborn")

# ---- Timestamp-anchored .ncs loading ----
# ncs_repair.py corrects the partial-record and sample-loss timing error in the
# raw .ncs files (see its header). It sits in this script's folder; the absolute
# path also finds it when running cell-by-cell from another working directory.
NCS_REPAIR_DIR = r'C:\Runita\NMR\Mole-Rat_PlaceCell\Codes\Theta\Codes\ncsDropCorr'
if NCS_REPAIR_DIR not in sys.path:
    sys.path.insert(0, NCS_REPAIR_DIR)
from ncs_repair import load_repaired_uv


# %% ==================== Configuration (from reference code) ====================

# Root folder holding one subfolder per animal, named exactly with that
# animal's ID (e.g. ROOT_DIR/1059Test/...).
ROOT_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_CI99_2'

# Common output folder: every CSV/XLSX/PKL/TXT result and all figures
# (band-by-band, theta range, aperiodic properties) are saved under here.
OUTPUT_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_CI99_2\PSD_FOOOF_NcsDropCorr_PeriodicOnly'
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')            # all figures

# Which animals (by ID / subfolder name under ROOT_DIR) to include. Comment/
# remove an ID here to exclude it. ANIMALS, STYLES, and everything downstream
# are derived automatically from this list.
ANIMALS_TO_INCLUDE = [
    'Fa1059',
    'Fa23BD',
    'Fa5834',
    'Fa8477',
    #'FaSimul',   # 10 simulated random-frequency + Poisson-noise LFPs (Simulate_LFP_FaSimul.py)
]

# {animal_id: folder_path} -- auto-built from ANIMALS_TO_INCLUDE + ROOT_DIR.
ANIMALS = {
    animal_id: os.path.join(ROOT_DIR, animal_id)
    for animal_id in ANIMALS_TO_INCLUDE
}

# ---- Acquisition / PSD ----
fs      = 32000  # original sampling rate (Hz)
fs_down = 1000    # target sampling rate after downsampling
nperseg = int(2 * fs_down)  # 2 s epochs (1000 samples at 1000 Hz) -> df = 0.5 Hz
# V per ADC count is read from each file's own header (-ADBitVolts, see
# read_ncs_adbitvolts): it follows the recording's InputRange, which differs
# between animals (1500 uV -> 4.58e-8 V, 1000 uV -> 3.05e-8 V), so a single
# hard-coded constant would scale animals' power differently.
# Simulated .ncs files (Simulate_LFP_FaSimul.py) have an empty header and were
# written with this scale, so it is used only when a header has no -ADBitVolts.
ADBITVOLTS_NO_HEADER = 0.000003051757812500000169
MAD_THRESH = 5.0          # dual-criteria epoch-rejection threshold (robust z)
LOW_BAND   = (1.0, 3.0)   # delta band: 1-3 Hz rejection criterion + delta/theta filter
NORM_BAND  = (1.0, 100.0) # band used for relative-power normalization

# Delta/theta epoch filter -- applied to velocity-passed epochs BEFORE MAD
# filtering: an epoch is rejected if its LOW_BAND (delta, 1-3 Hz) power
# exceeds its THETA_BAND (3-7 Hz, defined below) power. THETA_BAND is
# resolved at call time, so its definition later in this file still applies.
APPLY_DELTA_THETA_FILTER = False

# ---- Velocity / running-speed epoch gating ----
# Every .ncs file's session folder holds one tracking .csv with a UNIX
# timestamp (us, column A) on the same clock as the .ncs timestamps, plus x/y
# position in cm (columns D, E). Epochs are additionally gated on running
# speed: an epoch is only kept if the animal's median smoothed speed during it
# falls within [SPEED_MIN_CMS, SPEED_MAX_CMS].
SPEED_MIN_CMS       = 0    # lower running-speed bound for epoch acceptance (cm/s)
SPEED_MAX_CMS       = 90.0   # upper running-speed bound for epoch acceptance (cm/s)
SPEED_SMOOTH_WINDOW = 5     # Savitzky-Golay window (samples, odd; ~0.37 s at 30 Hz)
SPEED_SMOOTH_POLY   = 3

# ---- 50 Hz line-noise cleaning (European mains) ----
LINE_FREQ             = 50.0
LINE_HARMONICS        = [50.0, 100.0, 150.0, 200.0]  # harmonics below Nyquist (250 Hz)
APPLY_TIME_NOTCH      = True   # scipy IIR notch on the time series
NOTCH_Q               = 30.0
APPLY_SPECTRAL_INTERP = True   # FOOOF interpolate_spectrum on the PSD
INTERP_HALFWIDTH      = 2.0

# ---- Detrending (after notch filter) ----
APPLY_TIME_DETREND    = True   # remove slow drift from the full trace
DETREND_TYPE          = 'linear'  # 'linear' or 'constant' (see scipy.signal.detrend)

# ---- FOOOF / specparam ----
FOOOF_RANGE    = [1.0, 100.0]   # fit range (Hz)
FOOOF_SETTINGS = dict(
    peak_width_limits=[1.0, 8.0],
    max_n_peaks=6,
    min_peak_height=0.1,
    peak_threshold=2.0,
    aperiodic_mode='knee',    # use 'knee' if fitting a wide range with a spectral knee
)

# ---- Aperiodic removal -> periodic-only spectrum (v3) ----
# Step 1 (removal fit): FOOOF with FOOOF_SETTINGS is fit to each file's own
#   normalized PSD over FOOOF_RANGE. Its aperiodic component (offset, knee,
#   exponent) is evaluated on the fit frequencies and subtracted in log10 space:
#       periodic(f) = log10(PSD(f)) - aperiodic_fit(f)
#   This is FOOOF's flattened spectrum: log10 power above the 1/f, 0 = no
#   power above the aperiodic fit, negative between peaks is possible. The
#   per-file NORM_BAND normalization only shifts log10(PSD) by a constant,
#   which the fitted offset absorbs, so it does not change the periodic
#   spectrum.
# Step 2 (periodic re-fit): every later analysis -- mean spectra, FOOOF
#   peaks, theta CF/PW/BW and range, band power, group differences -- runs on
#   the periodic spectrum. Peaks are re-fit with FOOOF_SETTINGS_PERIODIC
#   ('fixed' aperiodic mode; its offset/exponent should come out near 0 and
#   are reported as periodic_fit_offset / periodic_fit_exponent as a check).
#   FOOOF takes linear power, so 10**periodic is passed to it.
# The removed aperiodic parameters (from step 1) are what the offset / knee /
# exponent columns and the aperiodic-property histograms report.
FOOOF_SETTINGS_PERIODIC = dict(FOOOF_SETTINGS, aperiodic_mode='fixed')

# Which fit's R^2 / error decide the fit-quality gates (theta-range estimate,
# low-quality-fit list): 'aperiodic' = the removal fit (step 1, same gate as
# v2 -- the removal is only valid when this fit is good); 'periodic' = the
# re-fit (step 2). R^2 of the re-fit is lower by construction (the 1/f trend
# it would explain has been removed), so the 0.9 / 0.98 thresholds below
# were set for the removal fit.
QUALITY_GATE_FIT = 'aperiodic'

# ---- Theta extraction / property plotting (from notebook) ----
THETA_BAND = (3.0, 7.0)        # Hz window used to pull the theta peak from FOOOF

# ---- Band-by-band analysis (canonical bands) ----
# A naive "band-by-band" power comparison (mean power in a fixed frequency
# window) can't tell whether a difference is a genuine oscillatory (periodic)
# change or just a shift in the aperiodic (1/f) component -- see
# https://fooof-tools.github.io/fooof/auto_motivations/measurements/plot_BandByBand.html
# In v3 both band measures are taken on the periodic spectrum (aperiodic
# removed, see below): mean periodic band power AND the FOOOF-parameterized
# peak power from the periodic re-fit. Bounded by FOOOF_RANGE (1-100 Hz above).
BANDS = Bands({
    'delta': [1, 3],
    'theta': [3, 7],
    'alpha': [8, 13],
    'beta':  [13, 30],
    'slow gamma': [30, 60],
    'fast gamma': [60, 90],
})

# Neuralynx .ncs record format (512 int16 samples per record, 16 kB header skipped)
ncs_dtype = np.dtype([
    ('timestamp'  , '<u8'),
    ('sc_number'  , '<u4'),
    ('cell_number', '<u4'),
    ('params'     , '<u4'),
    ('samples'    , '<i2', (512,)),
])

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


def _lighten(hex_color, amount=0.55):
    """Blend a hex color toward white, for use as a shaded fill color."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


# {label: (line_color, fill_color)} -- auto-built from ANIMALS, one entry per animal.
STYLES = {
    label: (animal_color(label), _lighten(animal_color(label)))
    for label in ANIMALS
}

# convenience: list of animal labels in ANIMALS order (used by notebook plots)
animals = list(ANIMALS.keys())


# %% ==================== Signal-processing helpers (from reference) ============

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
    lfp             : ndarray  repaired trace in microvolts (uV), on a uniform
                                fs grid (see ncs_repair.py)
    start_timestamp : int      UNIX timestamp (us) of the first sample -- same
                                clock as the tracking .csv, used to align
                                epochs to running speed.
    adbitvolts      : float    V per ADC count used to scale this file.
    """
    adbitvolts = read_ncs_adbitvolts(fpath)
    lfp, start_timestamp = load_repaired_uv(fpath, adbitvolts)
    return lfp, start_timestamp, adbitvolts


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
    return signal.detrend(np.asarray(x, dtype=np.float64), type=dtype)


def _robust_high_outliers(x, thresh, ref_mask=None):
    """Boolean mask of samples that are high outliers by robust (MAD) z-score.

    If `ref_mask` is given, the median/MAD reference statistics are estimated
    from `x[ref_mask]` only (e.g. epochs that already passed the velocity and
    delta/theta filters), so epochs excluded upstream can't skew the robust
    threshold -- z-scores are still returned for every element of `x`.
    """
    x   = np.asarray(x, dtype=np.float64)
    ref = x if ref_mask is None else x[np.asarray(ref_mask, dtype=bool)]
    if ref.size == 0:
        ref = x
    med = np.median(ref)
    mad = np.median(np.abs(ref - med))
    if mad == 0:
        mad = 1e-20
    robust_z = 0.6745 * (x - med) / mad
    return robust_z > thresh


def compute_psd_clean_epochs(lfp, fs_hz, nperseg, mad_thresh=5.0,
                             low_band=(1.0, 3.0), theta_band=None,
                             apply_delta_theta_filter=True, speed_keep=None):
    """Welch PSD averaged over 4 s epochs, with epoch rejection applied in
    three stages, in order:

      1. Running-speed gating -- `speed_keep` (one bool per epoch), computed
         upstream from tracking data.
      2. Delta/theta filter -- within the epochs kept by (1), an epoch is
         rejected if its `low_band` (delta, e.g. 1-3 Hz) power exceeds its
         `theta_band` (e.g. 3-7 Hz) power.
      3. Dual-criteria MAD outlier rejection -- an epoch is rejected if it is
         a high outlier (robust MAD z > mad_thresh) on EITHER broadband
         peak-to-peak amplitude OR `low_band` power. The median/MAD reference
         statistics are computed ONLY from the epochs that survive (1) and
         (2), so epochs already excluded can't skew the robust threshold.

    `theta_band` defaults to the module-level THETA_BAND (3-7 Hz).
    """
    theta_band = theta_band or THETA_BAND
    lfp = np.asarray(lfp, dtype=np.float64)
    n_total = len(lfp) // nperseg
    if n_total == 0:
        raise ValueError('Trace is shorter than one epoch.')

    epochs = lfp[:n_total * nperseg].reshape(n_total, nperseg)
    p2p = epochs.max(axis=1) - epochs.min(axis=1)

    win = signal.get_window('hann', nperseg)
    band_pow  = np.empty(n_total)  # delta (low_band) power, per epoch
    theta_pow = np.empty(n_total)  # theta_band power, per epoch
    psd_stack = None
    f = None
    for i in range(n_total):
        f, Pi = signal.welch(epochs[i], fs=fs_hz, window=win,
                             nperseg=nperseg, noverlap=0, detrend='constant')
        if psd_stack is None:
            psd_stack = np.empty((n_total, Pi.size))
        psd_stack[i] = Pi
        df  = f[1] - f[0]
        idx_low   = (f >= low_band[0])   & (f <= low_band[1])
        idx_theta = (f >= theta_band[0]) & (f <= theta_band[1])
        band_pow[i]  = np.sum(Pi[idx_low]) * df
        theta_pow[i] = np.sum(Pi[idx_theta]) * df

    # 1) velocity gating
    keep = np.asarray(speed_keep, dtype=bool) if speed_keep is not None \
        else np.ones(n_total, dtype=bool)

    # 2) delta/theta filter, restricted to the epochs already kept above
    if apply_delta_theta_filter:
        keep = keep & (band_pow <= theta_pow)

    # 3) MAD outlier rejection, referenced to the epochs kept by (1)-(2)
    reject = _robust_high_outliers(p2p, mad_thresh, ref_mask=keep) | \
             _robust_high_outliers(band_pow, mad_thresh, ref_mask=keep)
    keep = keep & ~reject

    if speed_keep is not None:
        n_clean = int(keep.sum())
        if n_clean == 0:
            raise ValueError(
                'No epochs pass the running-speed '
                f'({SPEED_MIN_CMS}-{SPEED_MAX_CMS} cm/s), delta/theta, and '
                'MAD artifact-rejection criteria.')
    else:
        n_clean = int(keep.sum())
        if n_clean == 0:
            keep = np.ones(n_total, dtype=bool)
            n_clean = n_total

    Pxx = psd_stack[keep].mean(axis=0)
    return f, Pxx, n_clean, n_total


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


def compute_velocity_from_position(csv_path):
    """Compute smoothed running speed (cm/s) from a tracking .csv.

    Column layout (positional): UNIX timestamp in us (col A) on the same
    clock as the .ncs LFP timestamps, x in cm (col D), y in cm (col E).
    Speed at sample n is the frame-to-frame displacement divided by the
    actual elapsed time between consecutive rows (from the column-A
    timestamps, so irregular/dropped tracking frames don't bias it),
    converted to cm/s, then smoothed with a Savitzky-Golay filter (preserves
    genuine speed transients better than a moving average while removing
    frame-to-frame tracking jitter).

    Returns
    -------
    time_us : ndarray   absolute UNIX timestamp of each tracking sample (us)
                        -- same clock as the .ncs LFP start timestamp, so
                        epochs can be aligned to speed without assuming a
                        shared t=0
    speed   : ndarray   smoothed running speed (cm/s), same length as time_us
    """
    df = pd.read_csv(csv_path, usecols=[0, 3, 4])
    df.columns = ['time_us', 'x', 'y']
    df = df.sort_values('time_us').reset_index(drop=True)

    time_us = df['time_us'].to_numpy(dtype=np.float64)

    dt_us = np.diff(time_us)
    dx = np.diff(df['x'].to_numpy(dtype=np.float64))
    dy = np.diff(df['y'].to_numpy(dtype=np.float64))
    dist = np.sqrt(dx**2 + dy**2)
    speed_step = np.divide(dist, dt_us, out=np.zeros_like(dist), where=dt_us > 0) * 1e6  # cm/s
    speed_raw = np.concatenate(([0.0], speed_step))

    n = len(speed_raw)
    window = min(SPEED_SMOOTH_WINDOW, n if n % 2 == 1 else n - 1)
    if window < SPEED_SMOOTH_POLY + 2:
        speed_smooth = speed_raw.copy()
    else:
        speed_smooth = savgol_filter(speed_raw, window, SPEED_SMOOTH_POLY)
    speed_smooth = np.clip(speed_smooth, 0, None)

    return time_us, speed_smooth


def compute_epoch_speed_keep(time_us, speed, n_total, nperseg, fs_hz, lfp_start_us,
                             speed_min=SPEED_MIN_CMS, speed_max=SPEED_MAX_CMS):
    """Boolean mask (len n_total): True where an epoch's median running speed
    falls within [speed_min, speed_max] cm/s.

    Epoch i spans absolute UNIX time [lfp_start_us + i*epoch_dur_us,
    lfp_start_us + (i+1)*epoch_dur_us) -- lfp_start_us is the .ncs file's
    first-sample timestamp, on the same UNIX clock as the tracking .csv, so
    epochs are aligned to actual recording time rather than an assumed
    shared t=0.
    """
    epoch_dur_us = 1e6 * nperseg / fs_hz
    keep = np.zeros(n_total, dtype=bool)
    for i in range(n_total):
        t0 = lfp_start_us + i * epoch_dur_us
        t1 = lfp_start_us + (i + 1) * epoch_dur_us
        in_epoch = (time_us >= t0) & (time_us < t1)
        if in_epoch.any():
            med_speed = np.median(speed[in_epoch])
        else:
            nearest = np.argmin(np.abs(time_us - (t0 + t1) / 2))
            med_speed = speed[nearest]
        keep[i] = (med_speed >= speed_min) & (med_speed <= speed_max)
    return keep


def clean_line_noise_psd(f, Pxx, harmonics, halfwidth=2.0):
    """Interpolate the PSD across each mains harmonic (FOOOF interpolate_spectrum)."""
    nyq    = f[-1]
    ranges = [[h - halfwidth, h + halfwidth]
              for h in harmonics if (h + halfwidth) < nyq]
    if not ranges:
        return f, Pxx
    f_i, P_i = interpolate_spectrum(f, Pxx, ranges)
    return f_i, P_i


# %% ==================== Per-animal processing (from reference) ================

# One dict per .ncs file seen by process_animal() (processed or skipped);
# written to the 'processed_files' sheet of the run-metadata workbook.
RUN_FILE_LOG = []


def process_animal(label, folder):
    """Process every .ncs under `folder`.

    Returns
    -------
    freq_vec, mean_psd, sem_psd, n_files, psds_norm, file_names
    """
    ncs_files = []
    for root, _dirs, files in os.walk(folder):
        for fname in files:
            if fname.endswith('.ncs'):
                ncs_files.append(os.path.join(root, fname))
    print(f'Found {len(ncs_files)} .ncs files in {folder}')

    psds       = []
    file_names = []
    freq_vec   = None
    velocity_cache = {}   # {csv_path: (time_us, speed)} -- shared across .ncs in a session

    for fpath in ncs_files:
        rel = os.path.relpath(fpath, folder)
        try:
            lfp, lfp_start_us, adbitvolts = load_ncs(fpath)
            lfp = signal.resample_poly(lfp, fs_down, fs)
            if APPLY_TIME_NOTCH:
                lfp = notch_filter(lfp, fs_down, LINE_HARMONICS, NOTCH_Q)
            if APPLY_TIME_DETREND:
                lfp = detrend_signal(lfp, dtype=DETREND_TYPE)

            pos_path = find_position_file(fpath)
            speed_keep = None
            if pos_path is None:
                print(f'    No tracking .csv found next to {rel} '
                      f'-- speed filter skipped for this file')
            else:
                if pos_path not in velocity_cache:
                    velocity_cache[pos_path] = compute_velocity_from_position(pos_path)
                time_us, speed = velocity_cache[pos_path]
                n_total_est = len(lfp) // nperseg
                speed_keep = compute_epoch_speed_keep(
                    time_us, speed, n_total_est, nperseg, fs_down, lfp_start_us)

            f, Pxx, n_clean, n_total = compute_psd_clean_epochs(
                lfp, fs_down, nperseg, mad_thresh=MAD_THRESH, low_band=LOW_BAND,
                apply_delta_theta_filter=APPLY_DELTA_THETA_FILTER,
                speed_keep=speed_keep)

            if APPLY_SPECTRAL_INTERP:
                f, Pxx = clean_line_noise_psd(
                    f, Pxx, LINE_HARMONICS, INTERP_HALFWIDTH)

            if freq_vec is None:
                freq_vec = f

            df        = f[1] - f[0]
            valid_idx = (f >= NORM_BAND[0]) & (f <= NORM_BAND[1])
            total_power = np.sum(Pxx[valid_idx]) * df
            Pxx_norm    = Pxx / total_power

            psds.append(Pxx_norm)
            file_names.append(rel)
            if speed_keep is not None:
                print(f'  OK: {rel}  [{n_clean}/{n_total} epochs kept; '
                      f'{int(speed_keep.sum())}/{n_total} met '
                      f'{SPEED_MIN_CMS}-{SPEED_MAX_CMS} cm/s speed criterion]')
            else:
                print(f'  OK: {rel}  [{n_clean}/{n_total} epochs kept]')
            RUN_FILE_LOG.append({
                'animal':                 label,
                'status':                 'OK',
                'file':                   rel,
                'full_path':              fpath,
                'tracking_csv':           pos_path or 'none (speed filter skipped)',
                'ADBitVolts_V_per_count': adbitvolts,
                'n_epochs_total':         n_total,
                'n_epochs_speed_ok':      int(speed_keep.sum()) if speed_keep is not None else np.nan,
                'n_epochs_kept':          n_clean,
                'skip_reason':            '',
            })

        except Exception as e:
            print(f'  SKIP: {rel} -- {e}')
            RUN_FILE_LOG.append({'animal': label, 'status': 'SKIP', 'file': rel,
                                 'full_path': fpath, 'skip_reason': str(e)})

    print(f'  -> {len(psds)} files processed\n')
    if not psds:
        raise ValueError(f'No files processed successfully in {folder}')

    psds     = np.array(psds)
    mean_psd = np.mean(psds, axis=0)
    sem_psd  = np.std(psds, axis=0) / np.sqrt(psds.shape[0])
    return freq_vec, mean_psd, sem_psd, psds.shape[0], psds, file_names


# %% ==================== Flat-structure metadata hook ==========================

def parse_metadata_from_path(rel_path):
    """Map a relative .ncs path to (date, session, tetrode, channel) metadata.

    The reference pipeline is flat (one PSD per file), so there is no built-in
    session/tetrode hierarchy. EDIT THIS to match your folder/filename layout if
    you want session- or arena-level grouping downstream (e.g. plot_arena_comparison).

    Default heuristic:
      * tetrode/channel parsed from a Neuralynx-style 'CSC<t>ch<c>' token if present;
      * date/session taken from the first two parent folders of the relative path;
      * anything unknown falls back to the filename stem or 'NA'.
    """
    fname = os.path.basename(rel_path)
    parts = os.path.normpath(rel_path).split(os.sep)

    m = re.search(r'CSC(\d+)(?:ch(\d+))?', fname, re.IGNORECASE)
    tetrode = m.group(1) if m else 'NA'
    channel = m.group(2) if (m and m.group(2)) else os.path.splitext(fname)[0]

    date    = parts[0] if len(parts) >= 2 else 'NA'
    session = parts[1] if len(parts) >= 3 else (parts[0] if len(parts) >= 2 else 'NA')
    return {'date': date, 'session': session, 'tetrode': tetrode, 'channel': channel}


# %% ==================== Aperiodic removal (v3) ================================

# One dict per file whose removal fit failed (NaN aperiodic parameters);
# those files are left out of every periodic analysis and listed in the
# 'periodic_dropped' sheet of the run-metadata workbook.
REMOVAL_DROPPED_LOG = []


def remove_aperiodic(results, fooof_settings=None, fooof_range=None,
                     save_fits=False, save_dir=None, fit_xlim=(1, 20)):
    """Fit FOOOF to each file's normalized PSD and subtract that file's own
    aperiodic fit in log10 space (see 'Aperiodic removal' in the config).

    Returns
    -------
    results_periodic : {animal: (freqs, mean, sem, n, spectra, file_names)}
        Same layout as `results`, but `freqs` is trimmed to `fooof_range` and
        `spectra` holds the periodic spectra, log10(PSD) - aperiodic_fit
        (one row per file); mean/sem are taken across files in this log10
        space.
    removal_fits : {(animal, file): dict}
        The removal fit per file: 'aperiodic_params' (offset[, knee],
        exponent), 'ap_fit_log10' (the curve removed), 'log10_psd' (the
        spectrum it was removed from), 'r_squared', 'error'.

    If `save_fits`, saves each removal fit (spectrum, full model, aperiodic
    fit) to `save_dir/<animal>/<file stem>_aperiodic_fit.png`.
    """
    fooof_settings = fooof_settings or FOOOF_SETTINGS
    fooof_range    = fooof_range or FOOOF_RANGE
    if save_fits:
        save_dir = save_dir or os.path.join(FIGURE_DIR, 'individual_fits_aperiodic_removal')

    results_periodic = {}
    removal_fits     = {}
    for animal, (freqs, _mean, _sem, _n, psds_norm, file_names) in results.items():
        print(f"  Aperiodic removal fit: {animal}  ({psds_norm.shape[0]} PSDs)")
        fg = FOOOFGroup(**fooof_settings)
        fg.fit(freqs, psds_norm, fooof_range)

        f_fit     = fg.freqs                     # trimmed to fooof_range
        log_psds  = fg.power_spectra             # log10, trimmed to fooof_range
        ap_params = fg.get_params('aperiodic_params')
        r2        = fg.get_params('r_squared')
        err       = fg.get_params('error')

        if save_fits:
            animal_dir = os.path.join(save_dir, animal)
            os.makedirs(animal_dir, exist_ok=True)

        flats, kept = [], []
        for i, fname in enumerate(file_names):
            ap = np.asarray(ap_params[i], dtype=np.float64)
            if np.any(~np.isfinite(ap)):
                print(f'    DROP (periodic): {fname} -- aperiodic fit failed')
                REMOVAL_DROPPED_LOG.append({'animal': animal, 'file': fname,
                                            'reason': 'aperiodic removal fit failed (NaN params)'})
                continue
            ap_fit = gen_aperiodic(f_fit, ap)
            flats.append(log_psds[i] - ap_fit)
            kept.append(fname)
            removal_fits[(animal, fname)] = {
                'aperiodic_params': ap,
                'ap_fit_log10':     ap_fit,
                'log10_psd':        log_psds[i].copy(),
                'r_squared':        float(r2[i]),
                'error':            float(err[i]),
            }

            if save_fits:
                fm = fg.get_fooof(i, regenerate=True)
                fig, ax = plt.subplots(figsize=(5, 3))
                _style_fooof_fit_ax(ax, fm, xlim=fit_xlim,
                                    title=f"{animal}: {os.path.basename(fname)} (removal fit)")
                stem = re.sub(r'[\\/]+', '_', os.path.splitext(fname)[0])
                fig.savefig(os.path.join(animal_dir, f'{stem}_aperiodic_fit.png'),
                            dpi=200, bbox_inches='tight')
                plt.close(fig)

        if not flats:
            print(f'    {animal}: no file has a usable aperiodic fit -- animal dropped')
            continue
        flats = np.array(flats)
        results_periodic[animal] = (f_fit, flats.mean(axis=0),
                                    flats.std(axis=0) / np.sqrt(flats.shape[0]),
                                    flats.shape[0], flats, kept)
        print(f"    {animal}: {flats.shape[0]}/{psds_norm.shape[0]} files -> periodic spectrum")
        if save_fits:
            print(f"    Saved {flats.shape[0]} removal-fit figures -> {animal_dir}")

    return results_periodic, removal_fits


def removal_fits_to_df(removal_fits):
    """One row per file: the removed aperiodic parameters and removal-fit quality."""
    rows = []
    for (animal, fname), r in removal_fits.items():
        ap = r['aperiodic_params']
        offset, exponent = ap[0], ap[-1]
        knee = ap[1] if len(ap) == 3 else np.nan
        rows.append({'animal': animal, 'file': fname, 'offset': offset,
                     'knee': knee, 'exponent': exponent,
                     'r_squared': r['r_squared'], 'error': r['error']})
    return pd.DataFrame(rows)


def plot_aperiodic_removal_check(results_periodic, removal_fits, animals_list,
                                 xlim=(1, 40), save=True, save_dir=None):
    """Per animal (one column): top = mean log10 PSD and mean removed
    aperiodic fit (across files); bottom = mean periodic spectrum +/- SEM,
    which is the top-row difference, with 0 = the aperiodic fit."""
    n = len(animals_list)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 6), squeeze=False, sharex=True)
    for col, animal in enumerate(animals_list):
        freqs, mean_p, sem_p, n_files, _spectra, files = results_periodic[animal]
        log_psd = np.array([removal_fits[(animal, f)]['log10_psd'] for f in files])
        ap_fit  = np.array([removal_fits[(animal, f)]['ap_fit_log10'] for f in files])
        color = animal_color(animal)

        ax = axes[0, col]
        ax.plot(freqs, log_psd.mean(axis=0), color='#333333', lw=1.4, label='log10 PSD')
        ax.plot(freqs, ap_fit.mean(axis=0), color='#EA080C', ls='--', lw=1.3,
                label='Aperiodic fit (removed)')
        ax.set_title(f'{animal} (n = {n_files} files)', fontsize=9)
        if col == 0:
            ax.set_ylabel('log10 normalized power', fontsize=AX_LABEL_FONTSIZE)
            ax.legend(fontsize=7, frameon=False)

        ax = axes[1, col]
        ax.plot(freqs, mean_p, color=color, lw=1.7)
        ax.fill_between(freqs, mean_p - sem_p, mean_p + sem_p, color=color, alpha=0.3)
        ax.axhline(0, color='gray', lw=0.8, ls=':')
        ax.set_xlabel('Frequency (Hz)', fontsize=AX_LABEL_FONTSIZE)
        if col == 0:
            ax.set_ylabel('Periodic power\n(log10 above aperiodic fit)', fontsize=AX_LABEL_FONTSIZE)

        for a in axes[:, col]:
            a.set_xlim(xlim)
            a.spines[['top', 'right']].set_visible(False)
            _fmt(a)

    fig.suptitle('Aperiodic removal: mean PSD, removed 1/f fit, and periodic remainder',
                 fontsize=11)
    plt.tight_layout()
    if save:
        out_dir = save_dir or FIGURE_DIR
        os.makedirs(out_dir, exist_ok=True)
        for ext in ('png', 'svg'):
            fig.savefig(os.path.join(out_dir, f'aperiodic_removal_check.{ext}'),
                        dpi=300, bbox_inches='tight')
        print(f"Saved aperiodic-removal check figure -> {out_dir}")
        plt.close(fig)
    else:
        plt.show()
    return fig


# %% ==================== FOOOF over per-file periodic spectra ===================

def build_fooof_results(results_periodic, removal_fits, fooof_settings=None,
                        fooof_range=None, save_fits=False, save_dir=None,
                        fit_xlim=(1, 20)):
    """Re-fit FOOOF (FOOOF_SETTINGS_PERIODIC, 'fixed' aperiodic) to every
    per-file PERIODIC spectrum (via FOOOFGroup) and return a results list.

    `results_periodic` holds log10 periodic spectra (see remove_aperiodic);
    FOOOF takes linear power, so 10**spectrum is passed to it.

    Each entry mirrors the notebook's fooof_results dicts so that
    fooof_results_to_df() works unchanged:
        animal, date, session, tetrode, channel,
        aperiodic_params  -- the REMOVED aperiodic params (removal fit)
        peak_params       -- peaks from the periodic re-fit
        r_squared, error  -- quality of the periodic re-fit
    plus 'periodic_fit_aperiodic_params' (re-fit's residual aperiodic, should
    be ~[0, 0]) and 'aperiodic_r_squared' / 'aperiodic_error' (removal fit).

    If `save_fits` is True, also saves one re-fit figure per file under
    `save_dir/<animal>/<sanitized file stem>_periodic_fit.png` (default
    save_dir: FIGURE_DIR/individual_fits_periodic).
    """
    fooof_settings = fooof_settings or FOOOF_SETTINGS_PERIODIC
    fooof_range    = fooof_range or FOOOF_RANGE

    if save_fits:
        save_dir = save_dir or os.path.join(FIGURE_DIR, 'individual_fits_periodic')

    fooof_results = []
    for animal, (freqs, _mean, _sem, _n, spectra, file_names) in results_periodic.items():
        print(f"  FOOOFGroup (periodic re-fit): {animal}  ({spectra.shape[0]} spectra)")
        fg = FOOOFGroup(**fooof_settings)
        fg.fit(freqs, np.power(10.0, spectra), fooof_range)

        if save_fits:
            animal_dir = os.path.join(save_dir, animal)
            os.makedirs(animal_dir, exist_ok=True)

        for i in range(spectra.shape[0]):
            # regenerate=True so the modeled spectrum/aperiodic fit are
            # available for plotting (regenerate=False only keeps params).
            fm = fg.get_fooof(i, regenerate=save_fits)
            meta = parse_metadata_from_path(file_names[i])
            removal = removal_fits[(animal, file_names[i])]
            fooof_results.append({
                'animal':           animal,
                'file':             file_names[i],
                **meta,
                'aperiodic_params': removal['aperiodic_params'],
                'peak_params':      fm.peak_params_,
                'r_squared':        fm.r_squared_,
                'error':            fm.error_,
                'periodic_fit_aperiodic_params': fm.aperiodic_params_,
                'aperiodic_r_squared': removal['r_squared'],
                'aperiodic_error':     removal['error'],
            })

            if save_fits:
                fig, ax = plt.subplots(figsize=(5, 3))
                _style_fooof_fit_ax(ax, fm, xlim=fit_xlim,
                                    title=f"{animal}: {os.path.basename(file_names[i])} (periodic)")
                stem = re.sub(r'[\\/]+', '_', os.path.splitext(file_names[i])[0])
                fig.savefig(os.path.join(animal_dir, f'{stem}_periodic_fit.png'),
                           dpi=200, bbox_inches='tight')
                plt.close(fig)

        if save_fits:
            print(f"    Saved {spectra.shape[0]} fit figures -> {animal_dir}")

    return fooof_results


def extract_theta_peak(peak_params, theta_band=None):
    """Return (cf, pw, bw) of the strongest FOOOF peak whose centre frequency
    falls within theta_band, or (nan, nan, nan) if none does."""
    theta_band = theta_band or THETA_BAND
    if len(peak_params) > 0:
        cfs = peak_params[:, 0]
        in_theta = (cfs >= theta_band[0]) & (cfs <= theta_band[1])
        theta_peaks = peak_params[in_theta]
        if len(theta_peaks) > 0:
            strongest = theta_peaks[np.argmax(theta_peaks[:, 1])]
            return tuple(strongest)
    return np.nan, np.nan, np.nan


def theta_range_from_peak(cf, bw):
    """Upper/lower theta bound from a FOOOF peak's centre freq + bandwidth
    (FOOOF's peak_params bandwidth is the full width, so +/- bw/2 around cf)."""
    if np.isnan(cf) or np.isnan(bw):
        return np.nan, np.nan
    return cf - bw / 2, cf + bw / 2


def fooof_results_to_df(fooof_results, theta_band):
    """Convert fooof_results list of dicts to a flat dataframe.

    Extracts theta peak (CF, PW, BW) from peak_params within theta_band, plus
    the resulting theta_low/theta_high frequency range. One row per LFP file.

    offset / knee / exponent are the REMOVED aperiodic parameters (removal
    fit); theta_* come from the periodic re-fit; r_squared / error are the
    periodic re-fit's, aperiodic_r_squared / aperiodic_error the removal
    fit's; periodic_fit_offset / periodic_fit_exponent are the re-fit's
    residual aperiodic (should be ~0 if the removal worked).
    """
    rows = []
    for r in fooof_results:
        ap = r['aperiodic_params']
        if len(ap) == 2:
            offset, exponent = ap
            knee = np.nan
        else:
            offset, knee, exponent = ap
        p_ap = r.get('periodic_fit_aperiodic_params', [np.nan, np.nan])

        theta_cf, theta_pw, theta_bw = extract_theta_peak(r['peak_params'], theta_band)
        theta_low, theta_high = theta_range_from_peak(theta_cf, theta_bw)

        rows.append({
            'animal':     r['animal'],
            'file':       r.get('file', ''),
            'date':       r['date'],
            'session':    r['session'],
            'tetrode':    r['tetrode'],
            'channel':    r['channel'],
            'offset':     offset,
            'knee':       knee,
            'exponent':   exponent,
            'theta_cf':   theta_cf,
            'theta_pw':   theta_pw,
            'theta_bw':   theta_bw,
            'theta_low':  theta_low,
            'theta_high': theta_high,
            'has_theta':  not np.isnan(theta_cf),
            'r_squared':  r['r_squared'],
            'error':      r['error'],
            'aperiodic_r_squared':   r.get('aperiodic_r_squared', np.nan),
            'aperiodic_error':       r.get('aperiodic_error', np.nan),
            'periodic_fit_offset':   p_ap[0],
            'periodic_fit_exponent': p_ap[-1],
        })

    df = pd.DataFrame(rows)
    print(f"Total units: {len(df)}")
    print(f"Units with theta peak: {df['has_theta'].sum()} ({100 * df['has_theta'].mean():.1f}%)")
    print(f"Animals: {df['animal'].unique()}")
    return df


# ---- FOOOF fit-quality thresholds for flagging poor fits ----
R_SQUARED_MIN = 0.98   # flag files with r_squared below this
ERROR_MAX     = 0.05    # flag files with error above this


def _gate_cols():
    """(r_squared column, error column) of the fit chosen by QUALITY_GATE_FIT."""
    if QUALITY_GATE_FIT == 'aperiodic':
        return 'aperiodic_r_squared', 'aperiodic_error'
    if QUALITY_GATE_FIT == 'periodic':
        return 'r_squared', 'error'
    raise ValueError("QUALITY_GATE_FIT must be 'aperiodic' or 'periodic'")


def export_low_quality_fits(df, out_path, r2_min=R_SQUARED_MIN, error_max=ERROR_MAX):
    """Write the list of files whose gating FOOOF fit (QUALITY_GATE_FIT) has
    r_squared < r2_min OR error > error_max to a .txt file (one file path per
    line)."""
    r2_col, err_col = _gate_cols()
    flagged = df[(df[r2_col] < r2_min) | (df[err_col] > error_max)]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as fh:
        for _, row in flagged.iterrows():
            fh.write(f"{row['file']}\n")
    print(f"Flagged {len(flagged)}/{len(df)} files "
          f"({r2_col} < {r2_min} or {err_col} > {error_max}) -> {out_path}")
    return flagged


# %% ==================== Theta-range summary + box plot ========================

# Only fits meeting these criteria contribute to the theta low/high estimate
THETA_FIT_R2_MIN  = 0.9
THETA_FIT_ERR_MAX = 0.05


def _good_theta_fits(df, r2_min=THETA_FIT_R2_MIN, error_max=THETA_FIT_ERR_MAX):
    """Fits with a detected theta peak (periodic re-fit) AND the gating fit's
    (QUALITY_GATE_FIT) r_squared > r2_min AND error < error_max."""
    r2_col, err_col = _gate_cols()
    return df[df['has_theta'] & (df[r2_col] > r2_min) & (df[err_col] < error_max)]


def theta_range_summary(df):
    """Per-animal and grand-average theta [low, high] bounds.

    Per animal: mean/std of theta_low and theta_high across that animal's
    individual FOOOF fits that had a detected theta peak (has_theta).
    Grand average ('ALL'): mean of the per-animal means (each included
    animal weighted equally, regardless of its number of fits).
    """
    valid = _good_theta_fits(df)

    per_animal = (valid.groupby('animal')[['theta_low', 'theta_high']]
                  .agg(['mean', 'std', 'count']))
    per_animal.columns = ['_'.join(c) for c in per_animal.columns]
    per_animal = per_animal.rename(columns={'theta_low_count': 'n_fits'}) \
                            .drop(columns=['theta_high_count']).reset_index()

    # preserve ANIMALS order
    order = [a for a in animals if a in per_animal['animal'].values]
    per_animal = per_animal.set_index('animal').loc[order].reset_index()

    grand = pd.DataFrame([{
        'animal':          'ALL',
        'theta_low_mean':  per_animal['theta_low_mean'].mean(),
        'theta_low_std':   per_animal['theta_low_mean'].std(),
        'theta_high_mean': per_animal['theta_high_mean'].mean(),
        'theta_high_std':  per_animal['theta_high_mean'].std(),
        'n_fits':          per_animal['n_fits'].sum(),
    }])

    summary = pd.concat([per_animal, grand], ignore_index=True)
    print("\nTheta range summary (mean +/- std across individual fits):")
    for _, row in summary.iterrows():
        print(f"  {row['animal']:>8s}: low = {row['theta_low_mean']:.2f} +/- {row['theta_low_std']:.2f} Hz, "
              f"high = {row['theta_high_mean']:.2f} +/- {row['theta_high_std']:.2f} Hz  (n={int(row['n_fits'])})")
    return summary


THETA_LOW_COLOR  = '#1A56DB'  # blue
THETA_HIGH_COLOR = '#E41A1C'  # red


def plot_theta_range_boxplot(df, summary=None, save=True, save_dir=None):
    """Box plot of each animal's individual theta low/high bounds (one pair
    of boxes -- low, high -- per animal), with each animal's mean low/high
    marked, plus a dashed line for the grand mean low/high across all
    included animals.
    """
    valid = _good_theta_fits(df)
    if summary is None:
        summary = theta_range_summary(df)

    animals_here = [a for a in animals if a in valid['animal'].unique()]
    n = len(animals_here)

    fig, ax = plt.subplots(figsize=(1.6 * n + 2, 4.5))

    positions_low  = [3 * i for i in range(n)]
    positions_high = [3 * i + 1 for i in range(n)]

    data_low  = [valid.loc[valid['animal'] == a, 'theta_low'].dropna().values  for a in animals_here]
    data_high = [valid.loc[valid['animal'] == a, 'theta_high'].dropna().values for a in animals_here]

    box_kwargs = dict(widths=0.7, patch_artist=True, showfliers=False,
                       medianprops=dict(color='black', lw=1.2))
    bp_low  = ax.boxplot(data_low,  positions=positions_low,  **box_kwargs)
    bp_high = ax.boxplot(data_high, positions=positions_high, **box_kwargs)
    for box in bp_low['boxes']:
        box.set(facecolor=_lighten(THETA_LOW_COLOR, 0.6), edgecolor=THETA_LOW_COLOR)
    for box in bp_high['boxes']:
        box.set(facecolor=_lighten(THETA_HIGH_COLOR, 0.6), edgecolor=THETA_HIGH_COLOR)

    # per-animal mean markers
    row_by_animal = summary.set_index('animal')
    mean_low  = [row_by_animal.loc[a, 'theta_low_mean']  for a in animals_here]
    mean_high = [row_by_animal.loc[a, 'theta_high_mean'] for a in animals_here]
    ax.scatter(positions_low,  mean_low,  marker='D', s=36, color=THETA_LOW_COLOR,
               edgecolor='black', linewidth=0.6, zorder=5, label='Mean low bound')
    ax.scatter(positions_high, mean_high, marker='D', s=36, color=THETA_HIGH_COLOR,
               edgecolor='black', linewidth=0.6, zorder=5, label='Mean high bound')

    # grand mean across included animals
    grand = summary.set_index('animal').loc['ALL']
    ax.axhline(grand['theta_low_mean'],  color=THETA_LOW_COLOR,  ls='--', lw=1.2, alpha=0.8,
              label=f"Grand mean low = {grand['theta_low_mean']:.2f} Hz")
    ax.axhline(grand['theta_high_mean'], color=THETA_HIGH_COLOR, ls='--', lw=1.2, alpha=0.8,
              label=f"Grand mean high = {grand['theta_high_mean']:.2f} Hz")

    ax.set_xticks([3 * i + 0.5 for i in range(n)])
    ax.set_xticklabels(animals_here)
    ax.set_ylabel('Theta frequency bound (Hz)')
    ax.set_title('Individual FOOOF theta-peak range by animal')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc='upper right')
    plt.tight_layout()

    if save:
        out_dir = save_dir or FIGURE_DIR
        os.makedirs(out_dir, exist_ok=True)
        for ext in ('png', 'svg'):
            fig.savefig(os.path.join(out_dir, f'theta_range_boxplot.{ext}'),
                       bbox_inches='tight', dpi=300)
        print(f"Saved theta-range box plot -> {out_dir}")
        plt.close(fig)
    else:
        plt.show()

    return fig


# %% ==================== Aperiodic-parameter histograms ========================

# xlim=None means the histogram range is taken from the data itself (min/max)
# rather than a fixed window -- used for 'knee', whose scale depends on
# FOOOF_RANGE and isn't comparable across setups the way offset/exponent are.
APERIODIC_PROPS = {
    'offset':   {'xlabel': 'Aperiodic Offset',   'xlim': (-3, 3.0)},
    'knee':     {'xlabel': 'Aperiodic Knee',     'xlim': None},
    'exponent': {'xlabel': 'Aperiodic Exponent', 'xlim': (0, 5.0)},
}


def plot_aperiodic_properties(df, props=None, save=True, save_dir=None):
    """Histograms of aperiodic-fit properties (offset, knee, exponent), one
    subplot per property, coloured by animal, pooled across every recording
    file analyzed (one value per file, from `fooof_results_to_df`'s output).

    Saves PNG + SVG to `save_dir` (default FIGURE_DIR) when `save=True`,
    otherwise shows the figure interactively.
    """
    if props is None:
        selected = list(APERIODIC_PROPS.keys())
    elif isinstance(props, str):
        selected = [props]
    else:
        selected = list(props)

    unknown = [p for p in selected if p not in APERIODIC_PROPS]
    if unknown:
        raise ValueError(f"Unknown property/ies: {unknown}. "
                         f"Choose from {list(APERIODIC_PROPS.keys())}")

    n_plots = len(selected)
    n_cols  = min(n_plots, 3)
    n_rows  = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.5 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax, key in zip(axes_flat, selected):
        meta     = APERIODIC_PROPS[key]
        vals_all = df[key].dropna()
        xlim     = meta['xlim'] or (vals_all.min(), vals_all.max())

        animals_here = sorted(df['animal'].unique())
        for i, animal in enumerate(animals_here):
            vals = df.loc[df['animal'] == animal, key].dropna()
            ax.hist(vals, bins=20, range=xlim,
                    alpha=0.6, color=animal_color(animal),
                    edgecolor='white', lw=0.5, label=str(animal))
        ax.legend(fontsize=7)

        ax.set_title(f'n = {vals_all.size}', fontsize=8)
        ax.set_xlabel(meta['xlabel'])
        ax.set_xlim(xlim)
        ax.set_ylabel('No. of recordings')
        ax.spines[['top', 'right']].set_visible(False)

    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    fig.suptitle('Removed aperiodic component (removal fit, all recordings)', fontsize=11)
    plt.tight_layout()

    if save:
        out_dir = save_dir or FIGURE_DIR
        os.makedirs(out_dir, exist_ok=True)
        tag = '_'.join(selected)
        for ext in ('png', 'svg'):
            fig.savefig(os.path.join(out_dir, f'aperiodic_properties_{tag}.{ext}'),
                        bbox_inches='tight', dpi=300)
        print(f"Saved aperiodic-property histograms -> {out_dir}")
        plt.close(fig)
    else:
        plt.show()

    return fig


# %% ==================== Band-by-band analysis ===================================
# Periodic band power vs FOOOF-parameterized peak power, per canonical band --
# see BANDS above and
# https://fooof-tools.github.io/fooof/auto_motivations/measurements/plot_BandByBand.html
# In v3 both are measured on the periodic spectrum (aperiodic removed), so
# band power can no longer track a pure 1/f exponent shift; the removed
# exponent is still reported alongside for reference.

def compute_band_power_periodic(freqs, periodic_spectrum, band_def):
    """Mean periodic power within `band_def`: mean of the log10 periodic
    spectrum (log10 power above the file's aperiodic fit) over the band, so
    0 = no power above the 1/f. Same log10 space as the peak-power (PW)
    values FOOOF reports."""
    _, band_power = trim_spectrum(freqs, periodic_spectrum, list(band_def))
    return np.mean(band_power)


def build_band_by_band_df(results_periodic, fooof_results, bands=None):
    """Per-file, per-band periodic band power vs FOOOF-parameterized peak
    power (periodic re-fit). Long-format: one row per (file, band).

    `exponent` is the aperiodic exponent removed from that file (removal
    fit), reported side by side as in the FOOOF docs' band-by-band example.
    """
    bands = bands or BANDS

    file_lookup = {}  # {(animal, file): (freqs, periodic_spectrum)}
    for animal, (freqs, _mean, _sem, _n, spectra, file_names) in results_periodic.items():
        for i, fname in enumerate(file_names):
            file_lookup[(animal, fname)] = (freqs, spectra[i])

    rows = []
    for r in fooof_results:
        freqs, periodic_spectrum = file_lookup[(r['animal'], r['file'])]
        exponent = r['aperiodic_params'][-1]  # removed exponent (last aperiodic param)

        for band_name, band_def in bands:
            band_power_periodic = compute_band_power_periodic(freqs, periodic_spectrum, band_def)
            peak_cf, peak_pw, peak_bw = get_band_peak(r['peak_params'], band_def)
            rows.append({
                'animal':          r['animal'],
                'file':            r['file'],
                'band':            band_name,
                'band_low':        band_def[0],
                'band_high':       band_def[1],
                'band_power_periodic': band_power_periodic,
                'band_power_peak': peak_pw,
                'peak_cf':         peak_cf,
                'peak_bw':         peak_bw,
                'has_peak':        not np.isnan(peak_pw),
                'exponent':        exponent,
                'r_squared':       r['r_squared'],
                'aperiodic_r_squared': r['aperiodic_r_squared'],
            })

    df = pd.DataFrame(rows)
    print(f"Band-by-band: {len(df)} (file, band) rows across "
          f"{len(bands.definitions)} bands and {df['animal'].nunique()} animals")
    return df


def print_peak_frequency_summary(df_bands, mean_fms, bands=None,
                                 highlight=('theta', 'fast gamma')):
    """Print the FOOOF peak frequency (centre freq) for every band in which a
    peak was detected -- per animal, as a per-file summary (n detected, mean,
    SD, median, range) and for the animal's mean-PSD fit. Bands in `highlight`
    (theta, fast gamma) additionally list the peak frequency of every file."""
    bands = bands or BANDS
    print("\n" + "=" * 78)
    print("PEAK FREQUENCY SUMMARY (FOOOF centre frequency, Hz)")
    print("=" * 78)

    for animal in sorted(df_bands['animal'].unique()):
        print(f"\n[{animal}]")
        fm = mean_fms.get(animal)
        for band_name, band_def in bands:
            sub = df_bands[(df_bands['animal'] == animal) & (df_bands['band'] == band_name)]
            det = sub[sub['has_peak']]
            star = '*' if band_name in highlight else ' '

            if fm is not None:
                mean_cf = get_band_peak_fm(fm, band_def)[0]
                mean_txt = 'no peak' if np.isnan(mean_cf) else f'{mean_cf:.2f} Hz'
            else:
                mean_txt = 'n/a'

            if len(det) == 0:
                print(f" {star}{band_name:<11} ({band_def[0]:g}-{band_def[1]:g} Hz): "
                      f"0/{len(sub)} files with a peak | mean-PSD peak: {mean_txt}")
                continue
            cf = det['peak_cf']
            print(f" {star}{band_name:<11} ({band_def[0]:g}-{band_def[1]:g} Hz): "
                  f"{len(det)}/{len(sub)} files with a peak | "
                  f"CF mean={cf.mean():.2f} +/- {cf.std():.2f} (SD), "
                  f"median={cf.median():.2f}, range={cf.min():.2f}-{cf.max():.2f} | "
                  f"mean-PSD peak: {mean_txt}")

    for band_name in highlight:
        det = df_bands[(df_bands['band'] == band_name) & df_bands['has_peak']]
        print(f"\n--- {band_name.upper()}: peak frequency per file ---")
        if det.empty:
            print("  no peaks detected")
            continue
        for _, row in det.sort_values(['animal', 'file']).iterrows():
            print(f"  {row['animal']:<8} {row['file']}: {row['peak_cf']:.2f} Hz "
                  f"(PW={row['band_power_peak']:.3f}, BW={row['peak_bw']:.2f})")
    print("=" * 78)


def plot_band_shaded_spectra(freqs, master_psds_dict, animals_list, bands=None,
                             freq_range=(1, 90), save=False, save_path=None):
    """Mean periodic spectrum per animal (log10 power above the aperiodic
    fit, averaged across files) with each canonical band shaded (fooof.plts'
    plot_spectra_shading, as used in the FOOOF docs' band-by-band example).

    Trimmed to `freq_range` (default 1-90 Hz) before plotting -- not just an
    xlim -- so the y-axis autoscale reflects only the plotted range."""
    bands = bands or BANDS
    freqs_trim, spectra = None, []
    for a in animals_list:
        freqs_trim, spec_trim = trim_spectrum(freqs, master_psds_dict[a][0], list(freq_range))
        spectra.append(spec_trim)
    colors  = [animal_color(a) for a in animals_list]
    shade_colors = sns.color_palette('Pastel2', n_colors=len(bands.definitions))

    fig, ax = plt.subplots(figsize=(7, 5))
    # spectra are already log10 -> log_powers=False (no second log)
    plot_spectra_shading(freqs_trim, spectra, log_powers=False, linewidth=2,
                         colors=colors, labels=animals_list,
                         shades=bands.definitions, shade_colors=shade_colors, ax=ax)
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.set_ylabel('Periodic power (log10 above aperiodic fit)', fontsize=AX_LABEL_FONTSIZE)
    ax.set_title('Mean periodic spectra (aperiodic removed) with canonical bands shaded',
                 fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    _fmt(ax)

    if save:
        out = save_path or os.path.join(FIGURE_DIR, 'band_shaded_periodic_spectra.svg')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(os.path.splitext(out)[0] + '.png', dpi=300, bbox_inches='tight')
        fig.savefig(out, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def compare_exp(fm1, fm2):
    """Difference in aperiodic exponent between two FOOOF fits (docs' compare_exp).
    In v3 this is applied to the mean-PSD removal fits (mean_fms_aperiodic),
    i.e. the difference in the 1/f that was removed."""
    return fm1.get_params('aperiodic_params', 'exponent') - fm2.get_params('aperiodic_params', 'exponent')


def compare_peak_pw(fm1, fm2, band_def):
    """Difference in FOOOF-parameterized peak power for `band_def` between two
    fits (docs' compare_peak_pw) -- nan if either fit has no peak in the band."""
    pw1 = get_band_peak_fm(fm1, band_def)[1]
    pw2 = get_band_peak_fm(fm2, band_def)[1]
    return pw1 - pw2


def compare_band_pw(fm1, fm2, band_def):
    """Difference in mean log10 band power for `band_def` between two fits
    (docs' compare_band_pw). Applied to the periodic re-fits (mean_fms), whose
    power_spectrum is the mean periodic spectrum, so this is the periodic
    band-power difference."""
    pw1 = np.mean(trim_spectrum(fm1.freqs, fm1.power_spectrum, list(band_def))[1])
    pw2 = np.mean(trim_spectrum(fm2.freqs, fm2.power_spectrum, list(band_def))[1])
    return pw1 - pw2


def summarize_band_by_band_group_diffs(mean_fms, mean_fms_aperiodic, bands=None):
    """Pairwise, per-band comparison of periodic band power, FOOOF peak
    power, and removed aperiodic exponent between each pair of animals --
    mirrors the FOOOF docs' compare_exp / compare_peak_pw / compare_band_pw:
    https://fooof-tools.github.io/fooof/auto_motivations/measurements/plot_BandByBand.html

    `mean_fms`           : {animal: FOOOF re-fit to that animal's mean
                           periodic spectrum} -> band power + peak power diffs
    `mean_fms_aperiodic` : {animal: FOOOF fit to that animal's mean PSD}
                           -> exponent diff (the 1/f that was removed)
    """
    bands = bands or BANDS
    rows = []
    for a1, a2 in itertools.combinations(mean_fms.keys(), 2):
        fm1, fm2 = mean_fms[a1], mean_fms[a2]
        exp_diff = compare_exp(mean_fms_aperiodic[a1], mean_fms_aperiodic[a2])
        for band_name, band_def in bands:
            rows.append({
                'animal_1':             a1,
                'animal_2':             a2,
                'band':                 band_name,
                'band_low':             band_def[0],
                'band_high':            band_def[1],
                'exponent_diff':        exp_diff,
                'band_power_periodic_diff': compare_band_pw(fm1, fm2, band_def),
                'band_power_peak_diff': compare_peak_pw(fm1, fm2, band_def),
            })
    df = pd.DataFrame(rows)

    for _, row in df.iterrows():
        print(f"  [{row['animal_1']} vs {row['animal_2']}] {row['band']} "
              f"({row['band_low']:g}-{row['band_high']:g} Hz): "
              f"periodic band power diff={row['band_power_periodic_diff']:+.3f}, "
              f"peak power diff={row['band_power_peak_diff']:+.3f}, "
              f"exponent diff={row['exponent_diff']:+.3f}")
    return df


def export_band_by_band_excel(df_bands, df_group_diffs, out_path, bands=None):
    """Write the band-by-band results to a single .xlsx workbook:
      - 'band_by_band' -- the full long-format per-file, per-band table
      - 'group_diffs'  -- pairwise animal comparison (see
                          summarize_band_by_band_group_diffs)
      - one sheet per band (e.g. 'theta_peaks') listing only the files with a
        significant FOOOF-detected peak in that band (has_peak == True, from
        the periodic re-fit).

    Requires the `openpyxl` package.
    """
    bands = bands or BANDS
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    peak_cols = ['animal', 'file', 'peak_cf', 'band_power_peak', 'peak_bw', 'r_squared']

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        df_bands.to_excel(writer, sheet_name='band_by_band', index=False)
        if df_group_diffs is not None:
            df_group_diffs.to_excel(writer, sheet_name='group_diffs', index=False)

        for band_name, _ in bands:
            sub = df_bands.loc[(df_bands['band'] == band_name) & df_bands['has_peak'],
                               peak_cols]
            sub = sub.sort_values(['animal', 'file']).reset_index(drop=True)
            sheet_name = f'{band_name}_peaks'[:31]  # Excel sheet-name length limit
            sub.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  {band_name}: {len(sub)} files with a significant peak "
                  f"-> sheet '{sheet_name}'")

    print(f"Band-by-band Excel workbook -> {out_path}")


def plot_band_by_band_comparison(df_bands, save=False, save_path=None):
    """For each canonical band: periodic band power (top row) vs FOOOF peak
    power from the periodic re-fit (middle row) vs the removed aperiodic
    exponent (bottom row, repeated per column for alignment), grouped by
    animal."""
    band_order   = list(dict.fromkeys(df_bands['band']))  # preserve first-seen order
    animal_order = sorted(df_bands['animal'].unique())
    palette = {a: animal_color(a) for a in animal_order}

    n_bands = len(band_order)
    fig, axes = plt.subplots(3, n_bands, figsize=(3.2 * n_bands, 8), squeeze=False)

    for col, band in enumerate(band_order):
        sub = df_bands[df_bands['band'] == band]

        ax_raw = axes[0, col]
        sns.boxplot(data=sub, x='animal', y='band_power_periodic', order=animal_order,
                    palette=palette, ax=ax_raw, fliersize=0)
        sns.stripplot(data=sub, x='animal', y='band_power_periodic', order=animal_order,
                      ax=ax_raw, color='k', size=2.5, alpha=0.3, jitter=True)
        ax_raw.axhline(0, color='gray', lw=0.8, ls=':')
        ax_raw.set_title(f"{band} ({sub['band_low'].iloc[0]:g}-{sub['band_high'].iloc[0]:g} Hz)",
                         fontsize=9)
        ax_raw.set_ylabel('Periodic band power\n(log10 above 1/f)' if col == 0 else '')
        ax_raw.set_xlabel('')

        ax_pk = axes[1, col]
        sub_pk = sub[sub['has_peak']]
        sns.boxplot(data=sub_pk, x='animal', y='band_power_peak', order=animal_order,
                    palette=palette, ax=ax_pk, fliersize=0)
        sns.stripplot(data=sub_pk, x='animal', y='band_power_peak', order=animal_order,
                      ax=ax_pk, color='k', size=2.5, alpha=0.3, jitter=True)
        ax_pk.set_title(f'n with peak = {len(sub_pk)}/{len(sub)}', fontsize=8)
        ax_pk.set_ylabel('FOOOF peak power\n(PW)' if col == 0 else '')
        ax_pk.set_xlabel('')

        ax_exp = axes[2, col]
        sns.boxplot(data=sub, x='animal', y='exponent', order=animal_order,
                    palette=palette, ax=ax_exp, fliersize=0)
        sns.stripplot(data=sub, x='animal', y='exponent', order=animal_order,
                      ax=ax_exp, color='k', size=2.5, alpha=0.3, jitter=True)
        ax_exp.set_ylabel('Removed aperiodic\nexponent' if col == 0 else '')
        ax_exp.set_xlabel('')

        for ax in (ax_raw, ax_pk, ax_exp):
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(axis='x', rotation=45, labelsize=TICK_LABEL_FONTSIZE)
            for t in ax.get_xticklabels():
                t.set_horizontalalignment('right')

    fig.suptitle('Band-by-band on the periodic spectrum: band power vs FOOOF peak power\n'
                 '(bottom row: removed aperiodic exponent, repeated per animal across bands)',
                 fontsize=10)
    plt.tight_layout()

    if save:
        out = save_path or os.path.join(FIGURE_DIR, 'band_by_band_comparison.svg')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches='tight')
        fig.savefig(out.replace('.svg', '.png'), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


# %% ==================== Property-plotting config + functions ===================

THETA_PROPS = {
    'theta_cf': {'xlabel': 'Centre Frequency (Hz)',   'xlim': (3.5, 6.5)},
    'theta_pw': {'xlabel': 'Power (a.u.)',            'xlim': (0,   1.0)},
    'theta_bw': {'xlabel': 'Peak Bandwidth (Hz)',     'xlim': (0,   3.5)},
    'exponent': {'xlabel': 'Aperiodic Exponent',      'xlim': (0,   5.0)},
    'offset':   {'xlabel': 'Aperiodic Offset',        'xlim': (-3,  3.0)},
    'r_squared':{'xlabel': 'R\u00b2',                 'xlim': (0,   1.0)},
    'error':    {'xlabel': 'FOOOF Error',             'xlim': (0,   0.5)},
}
THETA_SPECIFIC = {'theta_cf', 'theta_pw', 'theta_bw'}


def plot_theta_properties(df, props=None, save=False, save_dir=None):
    """Histograms of theta / aperiodic properties, coloured by animal."""
    if props is None:
        selected = list(THETA_PROPS.keys())
    elif isinstance(props, str):
        selected = [props]
    else:
        selected = list(props)

    unknown = [p for p in selected if p not in THETA_PROPS]
    if unknown:
        raise ValueError(f"Unknown property/ies: {unknown}. Choose from {list(THETA_PROPS.keys())}")

    df_theta = df[df['has_theta']]

    n_plots = len(selected)
    n_cols  = min(n_plots, 3)
    n_rows  = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.5 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax, key in zip(axes_flat, selected):
        meta    = THETA_PROPS[key]
        data_df = df_theta if key in THETA_SPECIFIC else df

        if 'animal' in data_df.columns:
            animals_here = sorted(data_df['animal'].unique())
            for i, animal in enumerate(animals_here):
                vals = data_df.loc[data_df['animal'] == animal, key].dropna()
                ax.hist(vals, bins=20, range=meta['xlim'],
                        alpha=0.6, color=animal_color(animal),
                        edgecolor='white', lw=0.5, label=str(animal))
            ax.legend(fontsize=7)
        else:
            vals = data_df[key].dropna()
            ax.hist(vals, bins=20, range=meta['xlim'],
                    color='#AAAAAA', edgecolor='#555555', lw=0.6)
            ax.axvline(vals.median(), color='steelblue', lw=1.5, ls='--',
                       label=f'median = {vals.median():.2f}')
            ax.legend(fontsize=7)

        n = data_df[key].notna().sum()
        ax.set_title(f'n = {n}', fontsize=8)
        ax.set_xlabel(meta['xlabel'])
        ax.set_xlim(meta['xlim'])
        ax.set_ylabel('No. of units')
        ax.spines[['top', 'right']].set_visible(False)

    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    fig.suptitle('Theta & aperiodic properties', fontsize=11)
    plt.tight_layout()

    if save:
        out_dir = save_dir or FIGURE_DIR
        os.makedirs(out_dir, exist_ok=True)
        tag = '_'.join(selected)
        for ext in ('png', 'svg'):
            fig.savefig(os.path.join(out_dir, f'theta_properties_{tag}.{ext}'),
                        bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close(fig)


def plot_fit_quality(df, group_col="animal", metrics="both", mode="box",
                     jitter=0.08, figsize=None, save=False, save_path=None,
                     show_points=True, point_color="k", point_size=3,
                     point_alpha=0.35, box_kwargs=None):
    """Plot FOOOF fit quality (R^2 / error) grouped by `group_col`."""
    if group_col not in df.columns:
        raise ValueError(f"group_col='{group_col}' not in df.columns")

    if isinstance(metrics, str):
        key = metrics.lower()
        if key in ("both", "all"):
            metrics_list = ["r_squared", "error"]
        elif key in ("r2", "r_squared", "rsquared"):
            metrics_list = ["r_squared"]
        elif key in ("err", "error"):
            metrics_list = ["error"]
        else:
            metrics_list = [metrics]
    else:
        metrics_list = list(metrics)

    for m in metrics_list:
        if m not in df.columns:
            raise ValueError(f"metric='{m}' not in df.columns")

    plot_df = df[[group_col, *metrics_list]].copy()
    order = sorted(plot_df[group_col].dropna().unique())

    palette_list = sns.color_palette("colorblind", n_colors=len(order))

    color_map = {g: animal_color(g, default=palette_list[i % len(palette_list)])

                 for i, g in enumerate(order)}

    n_metrics = len(metrics_list)
    if figsize is None:
        figsize = (5 * n_metrics, 4)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, squeeze=False)
    axes = axes[0]
    box_kwargs = box_kwargs or {}

    for ax, metric in zip(axes, metrics_list):
        sub = plot_df[[group_col, metric]].dropna()
        if mode == "box":
            sns.boxplot(data=sub, x=group_col, y=metric, order=order,
                        palette=color_map, ax=ax, fliersize=0, **box_kwargs)
            if show_points:
                sns.stripplot(data=sub, x=group_col, y=metric, order=order,
                              ax=ax, color=point_color, size=point_size,
                              alpha=point_alpha, jitter=jitter)
        elif mode == "square":
            for xi, g in enumerate(order):
                vals = sub.loc[sub[group_col] == g, metric].to_numpy()
                if vals.size == 0:
                    continue
                if show_points:
                    xs = xi + np.random.uniform(-jitter, jitter, size=vals.size)
                    ax.scatter(xs, vals, s=12, alpha=0.35, color=color_map[g])
                mean = np.nanmean(vals)
                sem = np.nanstd(vals) / np.sqrt(np.sum(~np.isnan(vals)))
                ax.errorbar([xi], [mean], yerr=[sem], fmt="s",
                            color="black", mfc="white", mec="black",
                            ms=8, capsize=4, lw=1.5, zorder=5)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order)
        else:
            raise ValueError("mode must be 'box' or 'square'")

        for t in ax.get_xticklabels():
            t.set_rotation(45)
            t.set_horizontalalignment('right')
        ax.set_title(metric)
        ax.set_xlabel(group_col)
        ax.spines[["top", "right"]].set_visible(False)
        # R^2 axes start at 0.5 unless a value is lower; error axes autoscale
        # (the v2 code set (0.5, None) on every metric, which hid error values)
        if 'r_squared' in metric and not sub.empty:
            ax.set_ylim(min(0.5, sub[metric].min() - 0.02), None)
    plt.tight_layout()

    if save:
        if save_path is None:
            save_path = os.path.join(FIGURE_DIR, 'fit_quality.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# %% ==================== Sample-fit + summary-figure helpers ====================

def get_sample_psd(results, animal=None, index=0):
    """Return (freqs, spectrum) for one file. Defaults to the first file of the
    first animal. Replaces the notebook's get_psd_from_store(). With
    results_periodic this is the file's log10 periodic spectrum."""
    if animal is None:
        animal = next(iter(results))
    freqs, _mean, _sem, _n, psds_norm, _files = results[animal]
    return freqs, psds_norm[index]


# (low, high) Hz from theta_range_summary()'s grand average; set in the main pipeline
THETA_RANGE_EST = None

AX_LABEL_FONTSIZE   = 10
TICK_LABEL_FONTSIZE = 10


def _fmt(ax):
    ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)


def plot_mean_psds_all_animals_on_ax(ax, freqs, master_psds_dict, animals_list,
                                     file_counts=None, xlim=(2, 20)):
    for i, animal in enumerate(animals_list):
        mean, sem = master_psds_dict[animal]
        color = animal_color(animal)
        if file_counts is not None:
            label = f"{animal} ({file_counts.get(animal, '?')} files)"
        else:
            label = str(animal)
        ax.plot(freqs, mean, lw=1.7, color=color, label=label)
        ax.fill_between(freqs, mean - sem, mean + sem, color=color, alpha=0.25)
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlim(xlim)
    ax.set_xlabel("Frequency (Hz)", fontsize=AX_LABEL_FONTSIZE)
    ax.set_ylabel("Periodic power (log10 above aperiodic fit)", fontsize=AX_LABEL_FONTSIZE)
    ax.set_title("A  Mean Periodic Spectra (aperiodic removed)", y=1.03, fontsize=10, pad=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    # theta highlight: bounds from the theta-range estimation (grand mean of
    # per-animal [low, high]); falls back to THETA_BAND if not yet estimated
    lo, hi = THETA_RANGE_EST if THETA_RANGE_EST is not None else THETA_BAND
    ax.axvspan(lo, hi, color='gray', alpha=0.12, zorder=0)
    _fmt(ax)


def _style_fooof_fit_ax(ax, fm, xlim=(1, 20), title="Sample FOOOF fit",
                        theta_band=None):
    """Plot an already-fit FOOOF model (original spectrum, full model, aperiodic
    fit) onto `ax` with the shared color/label styling used across the script.

    Also extracts the theta peak (strongest peak within theta_band) from `fm`
    and shades/labels its [cf - bw/2, cf + bw/2] range on the axis.
    """
    fm.plot(ax=ax, add_legend=False)

    line_styles = [
        ("Spectrum",     "#333333", "-",  1.6),
        ("Full Model",   "#1263E6", "--", 1.4),
        ("Aperiodic",    "#EA080C", "--", 1.4),
    ]
    for line, (label, color, ls, lw) in zip(ax.lines, line_styles):
        line.set_color(color)
        line.set_label(label)
        line.set_linestyle(ls)
        line.set_linewidth(lw)
        line.set_alpha(0.9)

    ax.set_xlim(xlim)
    ax.text(0.5, 1.11, title, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=10)
    ax.text(0.5, 1.01, f"R\u00b2={fm.r_squared_:.3f}, error={fm.error_:.3f}",
            transform=ax.transAxes, ha='center', va='bottom', fontsize=8)

    theta_cf, _theta_pw, theta_bw = extract_theta_peak(fm.peak_params_, theta_band)
    theta_low, theta_high = theta_range_from_peak(theta_cf, theta_bw)
    if not np.isnan(theta_low):
        ax.axvspan(theta_low, theta_high, color='green', alpha=0.15, zorder=0)
        ax.text(0.5, 0.99, f"Theta range: {theta_low:.2f}-{theta_high:.2f} Hz",
                transform=ax.transAxes, ha='center', va='top', fontsize=7.5,
                color='#1a7a1a')

    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel("Frequency (Hz)", fontsize=AX_LABEL_FONTSIZE)
    ax.set_ylabel("log10 power", fontsize=AX_LABEL_FONTSIZE)
    ax.grid(False)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    _fmt(ax)


def plot_sample_psd_and_fooof_on_ax(ax, freqs, periodic_spectrum, fooof_kwargs=None,
                                    freq_range=(1, 20), xlim=(1, 20)):
    """FOOOF re-fit (FOOOF_SETTINGS_PERIODIC) of one log10 periodic spectrum;
    FOOOF takes linear power, so 10**spectrum is passed."""
    if fooof_kwargs is None:
        fooof_kwargs = dict(**FOOOF_SETTINGS_PERIODIC, verbose=False)
    fm = FOOOF(**fooof_kwargs)
    fm.fit(freqs, np.power(10.0, periodic_spectrum), list(freq_range))
    _style_fooof_fit_ax(ax, fm, xlim=xlim, title="B  Sample periodic FOOOF fit")
    return fm


def plot_theta_prop_hist_on_ax(ax, df, prop, xlim=None, xlabel=None, title=None):
    df_theta = df[df["has_theta"]].copy()
    animals_here = sorted(df_theta["animal"].unique())
    palette = {a: animal_color(a) for a in animals_here}

    all_vals = df_theta[prop].dropna()
    if xlim is not None:
        all_vals = all_vals.clip(*xlim)
    if all_vals.empty:
        return
    bins = np.linspace(all_vals.min(), all_vals.max(), 21)
    bin_width = bins[1] - bins[0]

    for a in animals_here:
        vals = df_theta.loc[df_theta["animal"] == a, prop].dropna()
        counts, _ = np.histogram(vals, bins=bins)
        if counts.sum() == 0:
            continue
        proportion = counts / counts.sum()
        ax.bar(bins[:-1], proportion, width=bin_width,
               alpha=0.55, color=palette[a], edgecolor="white",
               linewidth=0.4, align="edge", label=a)

    ax.set_ylabel("Proportion", fontsize=AX_LABEL_FONTSIZE)
    ax.set_title(title or prop, fontsize=10, pad=8)
    ax.set_xlabel(xlabel or prop, fontsize=AX_LABEL_FONTSIZE)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.spines[['top', 'right']].set_visible(False)
    _fmt(ax)


def plot_master_summary(results, master_psds_dict, expanded_fooof_df,
                        file_counts, freqs, save=True):
    """Composite summary figure (adapted from notebook cell 61)."""
    fig = plt.figure(figsize=(12, 5.5))
    gs = gridspec.GridSpec(2, 4, figure=fig,
                           width_ratios=[2.8, 1.4, 0.72, 0.72],
                           height_ratios=[1, 1], hspace=0.65, wspace=0.55,
                           top=0.87, bottom=0.11, left=0.07, right=0.97)
    ax_A  = fig.add_subplot(gs[0:2, 0])
    ax_B  = fig.add_subplot(gs[0, 1])
    ax_C1 = fig.add_subplot(gs[0, 2])
    ax_C2 = fig.add_subplot(gs[0, 3])
    ax_D  = fig.add_subplot(gs[1, 1])
    ax_E  = fig.add_subplot(gs[1, 2])
    ax_F  = fig.add_subplot(gs[1, 3])
    axes = dict(A=ax_A, B=ax_B, C1=ax_C1, C2=ax_C2, D=ax_D, E=ax_E, F=ax_F)

    fig.suptitle("Characterising LFP Periodic Spectra using FOOOF (aperiodic removed)",
                 fontsize=12, y=0.97)

    # A: mean PSDs
    plot_mean_psds_all_animals_on_ax(axes["A"], freqs, master_psds_dict,
                                     list(master_psds_dict.keys()),
                                     file_counts=file_counts)

    # B: sample periodic FOOOF fit (first file of first animal)
    s_freqs, s_spec = get_sample_psd(results)
    plot_sample_psd_and_fooof_on_ax(axes["B"], freqs=s_freqs, periodic_spectrum=s_spec,
                                    freq_range=tuple(FOOOF_RANGE), xlim=(1, 20))

    # C1 / C2: fit quality
    def _quality_ax(ax, col, ylim, title):
        order_animals = sorted(expanded_fooof_df["animal"].unique())
        palette = {a: animal_color(a) for a in order_animals}
        long_df = expanded_fooof_df[["animal", col]].dropna()
        sns.boxplot(data=long_df, x="animal", y=col, ax=ax,
                    palette=palette, fliersize=0, linewidth=0.8)
        sns.stripplot(data=long_df, x="animal", y=col, ax=ax,
                      palette=palette, size=2.5, alpha=0.35, jitter=True)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=AX_LABEL_FONTSIZE, pad=4)
        ax.set_xlabel("")
        ax.set_ylabel(title, fontsize=AX_LABEL_FONTSIZE)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', rotation=30, labelsize=TICK_LABEL_FONTSIZE)
        ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)

    # quality of the fit that gates the analysis (QUALITY_GATE_FIT)
    r2_col, err_col = _gate_cols()
    gate_txt = 'removal fit' if QUALITY_GATE_FIT == 'aperiodic' else 'periodic re-fit'
    _quality_ax(axes["C1"], r2_col, None, f"R\u00b2 ({gate_txt})")
    _quality_ax(axes["C2"], err_col, None, f"Error ({gate_txt})")

    # D / E / F: theta histograms
    plot_theta_prop_hist_on_ax(axes["D"], expanded_fooof_df, prop="theta_cf",
                               xlim=THETA_PROPS["theta_cf"]["xlim"],
                               xlabel=THETA_PROPS["theta_cf"]["xlabel"],
                               title="D  Centre frequency")
    plot_theta_prop_hist_on_ax(axes["F"], expanded_fooof_df, prop="theta_pw",
                               xlim=THETA_PROPS["theta_pw"]["xlim"],
                               xlabel=THETA_PROPS["theta_pw"]["xlabel"],
                               title="E  Power")
    plot_theta_prop_hist_on_ax(axes["E"], expanded_fooof_df, prop="theta_bw",
                               xlim=THETA_PROPS["theta_bw"]["xlim"],
                               xlabel=THETA_PROPS["theta_bw"]["xlabel"],
                               title="F  Bandwidth")
    axes["D"].legend(frameon=False, fontsize=7, title="animal",
                     title_fontsize=7, loc="upper right")

    if save:
        os.makedirs(FIGURE_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURE_DIR, "lfp_psd_fooof_summary.png"),
                    dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(FIGURE_DIR, "lfp_psd_fooof_summary.svg"),
                    bbox_inches="tight")
    plt.show()
    return fig, axes


# %% ==================== Run metadata ===========================================

def _script_path():
    """Absolute path of this script. When run cell-by-cell in the VS Code
    interactive window __file__ is undefined, so fall back to the path VS Code
    injects (__vsc_ipynb_file__); None if neither is available."""
    try:
        return os.path.abspath(__file__)
    except NameError:
        p = globals().get('__vsc_ipynb_file__')
        return os.path.abspath(p) if p else None


def _file_sha256(path):
    """SHA-256 of a file -- identical hash <=> byte-identical code version."""
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git_info(path):
    """Repo / branch / commit of `path`, and whether the file itself is
    committed, modified, or untracked (in which case the commit alone does
    NOT identify the code -- use the SHA-256 / snapshot copy instead)."""
    def _git(*args):
        out = subprocess.run(['git', '-C', os.path.dirname(path), *args],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    try:
        commit = _git('rev-parse', 'HEAD')
    except (OSError, subprocess.SubprocessError):
        commit = None
    if commit is None:
        return {'git_repo': 'n/a (git unavailable or not a repository)'}

    status = _git('status', '--porcelain', '--', os.path.basename(path))
    if status is None:
        file_status = 'unknown'
    elif status == '':
        file_status = 'clean (identical to git_commit)'
    elif status.startswith('??'):
        file_status = 'untracked (not in git -- identify via script_sha256 / snapshot)'
    else:
        file_status = 'modified since git_commit (identify via script_sha256 / snapshot)'
    return {
        'git_repo':        _git('rev-parse', '--show-toplevel'),
        'git_branch':      _git('rev-parse', '--abbrev-ref', 'HEAD'),
        'git_commit':      commit,
        'git_file_status': file_status,
    }


def _package_versions(pkgs=('numpy', 'scipy', 'pandas', 'matplotlib',
                            'seaborn', 'fooof', 'openpyxl')):
    versions = {}
    for p in pkgs:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = 'not installed'
    return versions


def capture_code_provenance(run_start, snapshot_dir=None):
    """Identify the code version and environment for this run: script name,
    directory, last-modified time, SHA-256, git state, Python + package
    versions. If `snapshot_dir` is given, a copy of the script is saved there
    so the exact code can be recovered later.

    Call this at the START of the run, so it describes the code that is
    actually executing even if the file is edited while the pipeline runs.
    """
    script_path = _script_path()
    prov = {
        'script_name': os.path.basename(script_path) if script_path else 'unknown (run interactively)',
        'script_dir':  os.path.dirname(script_path) if script_path else 'unknown',
        'script_path': script_path or 'unknown',
    }
    if script_path and os.path.isfile(script_path):
        prov['script_last_modified'] = datetime.datetime.fromtimestamp(
            os.path.getmtime(script_path)).isoformat(timespec='seconds')
        prov['script_sha256'] = _file_sha256(script_path)
        prov.update(_git_info(script_path))
        if snapshot_dir:
            os.makedirs(snapshot_dir, exist_ok=True)
            snapshot = os.path.join(snapshot_dir, os.path.basename(script_path))
            shutil.copy2(script_path, snapshot)
            prov['script_snapshot'] = snapshot
    prov.update({
        'run_started':       run_start.isoformat(timespec='seconds'),
        'computer':          platform.node(),
        'user':              getpass.getuser(),
        'working_directory': os.getcwd(),
        'python_version':    sys.version.split()[0],
        'python_executable': sys.executable,
        **{f'version_{k}': v for k, v in _package_versions().items()},
    })
    return prov


def _analysis_settings():
    """{section: {parameter: value}} -- every analysis setting (config block +
    hard-coded values used later in the file)."""
    return {
        'paths / animals': {
            'ROOT_DIR':               ROOT_DIR,
            'OUTPUT_DIR':             OUTPUT_DIR,
            'FIGURE_DIR':             FIGURE_DIR,
            'ANIMALS_TO_INCLUDE':     ANIMALS_TO_INCLUDE,
        },
        'acquisition / PSD': {
            'fs':                     fs,
            'fs_down':                fs_down,
            'nperseg':                nperseg,
            'epoch_duration_s':       nperseg / fs_down,
            'processing_order':       'resample_poly -> notch (if APPLY_TIME_NOTCH) -> '
                                      'detrend (if APPLY_TIME_DETREND) -> per-epoch Welch -> '
                                      'speed gate -> delta/theta filter -> MAD rejection -> '
                                      'spectral line-noise interp -> NORM_BAND normalization',
            'ADBitVolts':             'read per file from .ncs header (-ADBitVolts); see processed_files sheet',
            'ADBITVOLTS_NO_HEADER':   ADBITVOLTS_NO_HEADER,
            'MAD_THRESH':             MAD_THRESH,
            'LOW_BAND':               LOW_BAND,
            'NORM_BAND':              NORM_BAND,
            'APPLY_DELTA_THETA_FILTER': APPLY_DELTA_THETA_FILTER,
            'welch_window':           'hann',
            'welch_noverlap':         0,
            'welch_detrend':          'constant',
            'MAD_robust_z_constant':  0.6745,
            'ncs_dtype':              str(ncs_dtype),
            'ncs_samples_used':       'valid samples on a timestamp-anchored grid, gaps up to '
                                      '1 s interpolated (ncs_repair.py)',
        },
        'velocity gating': {
            'SPEED_MIN_CMS':          SPEED_MIN_CMS,
            'SPEED_MAX_CMS':          SPEED_MAX_CMS,
            'SPEED_SMOOTH_WINDOW':    SPEED_SMOOTH_WINDOW,
            'SPEED_SMOOTH_POLY':      SPEED_SMOOTH_POLY,
            'epoch_speed_statistic':  'median speed within epoch (nearest sample if none)',
        },
        'line noise': {
            'LINE_FREQ':              LINE_FREQ,
            'LINE_HARMONICS':         LINE_HARMONICS,
            'APPLY_TIME_NOTCH':       APPLY_TIME_NOTCH,
            'NOTCH_Q':                NOTCH_Q,
            'APPLY_SPECTRAL_INTERP':  APPLY_SPECTRAL_INTERP,
            'INTERP_HALFWIDTH':       INTERP_HALFWIDTH,
        },
        'detrending': {
            'APPLY_TIME_DETREND':     APPLY_TIME_DETREND,
            'DETREND_TYPE':           DETREND_TYPE,
        },
        'FOOOF': {
            'FOOOF_RANGE':            FOOOF_RANGE,
            'FOOOF_SETTINGS (removal fit)':    FOOOF_SETTINGS,
            'FOOOF_SETTINGS_PERIODIC (re-fit)': FOOOF_SETTINGS_PERIODIC,
            'aperiodic_removal':      "per file: periodic = log10(normalized PSD) - that file's own "
                                      "aperiodic fit (FOOOF_SETTINGS, FOOOF_RANGE); all later "
                                      "analyses use the periodic spectrum; FOOOF re-fit gets "
                                      "10**periodic",
            'QUALITY_GATE_FIT':       QUALITY_GATE_FIT,
            'THETA_BAND':             THETA_BAND,
            'BANDS':                  {name: tuple(rng) for name, rng in BANDS},
            'R_SQUARED_MIN':          R_SQUARED_MIN,
            'ERROR_MAX':              ERROR_MAX,
            'THETA_FIT_R2_MIN':       THETA_FIT_R2_MIN,
            'THETA_FIT_ERR_MAX':      THETA_FIT_ERR_MAX,
            'THETA_RANGE_EST (computed)': THETA_RANGE_EST if THETA_RANGE_EST is not None
                                          else 'not yet computed',
        },
        'plotting': {
            'THETA_LOW_COLOR':        THETA_LOW_COLOR,
            'THETA_HIGH_COLOR':       THETA_HIGH_COLOR,
            'APERIODIC_PROPS':        APERIODIC_PROPS,
            'THETA_PROPS':            THETA_PROPS,
            'THETA_SPECIFIC':         sorted(THETA_SPECIFIC),
            'AX_LABEL_FONTSIZE':      AX_LABEL_FONTSIZE,
            'TICK_LABEL_FONTSIZE':    TICK_LABEL_FONTSIZE,
            'histogram_bins':         20,
            'individual_fit_plot_xlim': (1, 20),
            'sample_fit_xlim':        (1, 20),
            'mean_psd_plot_xlim':     (2, 20),
            'band_shaded_spectra_freq_range': (1, 90),
            'fit_quality_ylim_summary':       'autoscaled (gate-fit R2 / error)',
            'aperiodic_removal_check_xlim':   (1, 40),
            'plot_palette':           ANIMAL_COLORS,
        },
    }


def save_run_metadata(out_path, provenance, run_status='complete'):
    """Write the run metadata to a multi-sheet .xlsx workbook:
      - 'code_provenance' -- script name/dir/path, SHA-256, git state, snapshot
                             copy, run time, computer, Python/package versions
      - 'directories'     -- ROOT_DIR / OUTPUT_DIR / FIGURE_DIR and every
                             animal folder analyzed, with processed/skipped counts
      - 'settings'        -- every analysis setting, grouped by section
      - 'processed_files' -- every .ncs found: status, epochs kept, ADBitVolts,
                             tracking .csv used, skip reason
      - 'periodic_dropped'-- files left out of the periodic analysis because
                             their aperiodic removal fit failed
      - 'output_files'    -- every file written under OUTPUT_DIR during this run

    Requires the `openpyxl` package.
    """
    prov = {'run_status': run_status, **provenance,
            'metadata_written': datetime.datetime.now().isoformat(timespec='seconds')}
    df_prov = pd.DataFrame({'item': list(prov.keys()),
                            'value': [str(v) for v in prov.values()]})

    df_log = pd.DataFrame(RUN_FILE_LOG)
    dir_rows: list[dict] = [{'item': name, 'path': path}
                for name, path in (('ROOT_DIR', ROOT_DIR), ('OUTPUT_DIR', OUTPUT_DIR),
                                   ('FIGURE_DIR', FIGURE_DIR))]
    for animal_id, folder in ANIMALS.items():
        sub = df_log[df_log['animal'] == animal_id] if not df_log.empty else df_log
        dir_rows.append({
            'item':            f'animal folder: {animal_id}',
            'path':            folder,
            'exists':          os.path.isdir(folder),
            'n_ncs_processed': int((sub['status'] == 'OK').sum()) if not sub.empty else 0,
            'n_ncs_skipped':   int((sub['status'] == 'SKIP').sum()) if not sub.empty else 0,
        })
    df_dirs = pd.DataFrame(dir_rows)

    df_settings = pd.DataFrame(
        [{'section': section, 'parameter': k, 'value': str(v)}
         for section, params in _analysis_settings().items()
         for k, v in params.items()])

    # files (re)written under OUTPUT_DIR since the run started
    run_start_ts = datetime.datetime.fromisoformat(provenance['run_started']).timestamp() - 1
    outputs = []
    for root, _dirs, files in os.walk(OUTPUT_DIR):
        for fname in files:
            fpath = os.path.join(root, fname)
            if os.path.getmtime(fpath) >= run_start_ts:
                outputs.append({'file': os.path.relpath(fpath, OUTPUT_DIR),
                                'full_path': fpath})
    df_outputs = pd.DataFrame(outputs, columns=['file', 'full_path'])

    sheets = {
        'code_provenance': df_prov,
        'directories':     df_dirs,
        'settings':        df_settings,
        'processed_files': df_log,
        'periodic_dropped': pd.DataFrame(REMOVAL_DROPPED_LOG,
                                         columns=['animal', 'file', 'reason']),
        'output_files':    df_outputs,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                ws = writer.sheets[sheet_name]
                for col_cells in ws.columns:   # fit column width to content
                    width = max(len(str(c.value)) if c.value is not None else 0
                                for c in col_cells)
                    ws.column_dimensions[col_cells[0].column_letter].width = min(width + 2, 100)
    except PermissionError:
        print(f"WARNING: could not write {out_path} (is it open in Excel?) -- "
              f"run metadata NOT saved")
        return
    print(f"Run metadata -> {out_path}")


# %% ==================== MAIN PIPELINE (PSD -> remove 1/f -> FOOOF -> plots) ===

# 0) Record which code version is running (name, directory, SHA-256, git
#    state) and save a copy of the script next to the figures -- captured
#    before processing so it matches the code actually executed. The settings
#    / directories / files workbook is written to FIGURE_DIR after step 2
#    (status 'incomplete') and rewritten at the end (status 'complete').
RUN_START = datetime.datetime.now()
RUN_FILE_LOG.clear()
REMOVAL_DROPPED_LOG.clear()
RUN_METADATA_PATH = os.path.join(FIGURE_DIR, 'run_metadata.xlsx')
PROVENANCE = capture_code_provenance(
    RUN_START, snapshot_dir=os.path.join(FIGURE_DIR, 'code_snapshot'))

# 1) Generate PSDs for every animal via the reference folder-walk pipeline.
results = {}
for label, folder in ANIMALS.items():
    print(f"=== Processing {label} ===")
    results[label] = process_animal(label, folder)

# 2) Aperiodic removal: fit FOOOF to each file's own PSD and subtract its
#    aperiodic fit in log10 space -> periodic spectrum per file. Saves each
#    removal fit under FIGURE_DIR/individual_fits_aperiodic_removal/<animal>/.
#    EVERY step below runs on results_periodic, not on the raw PSDs.
print("\n=== Aperiodic removal ===")
results_periodic, removal_fits = remove_aperiodic(
    results, save_fits=True,
    save_dir=os.path.join(FIGURE_DIR, 'individual_fits_aperiodic_removal'))
animals_periodic = [a for a in animals if a in results_periodic]

# Cache the frequency vector and per-animal summaries the notebook code expects
# -- all PERIODIC (log10 power above the aperiodic fit, FOOOF_RANGE freqs).
freqs_store      = results_periodic[animals_periodic[0]][0]
master_psds_dict = {a: (r[1], r[2]) for a, r in results_periodic.items()}  # {animal: (mean, sem)}
file_counts      = {a: r[3] for a, r in results_periodic.items()}          # {animal: n_files}

# Persist the raw PSDs, the removed aperiodic fits and the periodic spectra.
os.makedirs(OUTPUT_DIR, exist_ok=True)
# first write: if a later step crashes, the workbook still records this run
save_run_metadata(RUN_METADATA_PATH, PROVENANCE, run_status='incomplete')
with open(os.path.join(OUTPUT_DIR, 'processed_psds.pkl'), 'wb') as fh:
    pickle.dump({
        'raw': {a: {'freqs': r[0], 'mean': r[1], 'sem': r[2],
                    'psds': r[4], 'files': r[5]} for a, r in results.items()},
        'periodic': {a: {'freqs': r[0], 'mean': r[1], 'sem': r[2],
                         'spectra_log10': r[4], 'files': r[5]}
                     for a, r in results_periodic.items()},
        'removal_fits': removal_fits,
        'note': 'periodic = log10(normalized PSD) - per-file aperiodic fit',
    }, fh)
df_removal = removal_fits_to_df(removal_fits)
df_removal.to_csv(os.path.join(OUTPUT_DIR, 'aperiodic_removal_fits.csv'), index=False)

plot_aperiodic_removal_check(results_periodic, removal_fits, animals_periodic, save=True)

# 3) FOOOF on each animal's averaged spectrum.
#    mean_fms_aperiodic: fit to the mean raw PSD -> exponent difference of the
#      removed 1/f in the group comparison (step 7).
#    mean_fms: periodic re-fit of the mean periodic spectrum -> peak/band
#      power differences (step 7) and mean-spectrum peaks (step 8).
mean_fms_aperiodic = {}
mean_fms = {}
for animal in animals_periodic:
    freqs_raw, mean_raw = results[animal][0], results[animal][1]
    fm_ap = FOOOF(**FOOOF_SETTINGS, verbose=False)
    fm_ap.fit(freqs_raw, mean_raw, FOOOF_RANGE)
    mean_fms_aperiodic[animal] = fm_ap

    freqs_p, mean_p = results_periodic[animal][0], results_periodic[animal][1]
    fm = FOOOF(**FOOOF_SETTINGS_PERIODIC, verbose=False)
    fm.fit(freqs_p, np.power(10.0, mean_p), FOOOF_RANGE)
    mean_fms[animal] = fm
    print(f"[{animal}] mean-PSD removal fit: R\u00b2={fm_ap.r_squared_:.3f}, "
          f"aperiodic={fm_ap.aperiodic_params_} | mean periodic re-fit: "
          f"R\u00b2={fm.r_squared_:.3f}, error={fm.error_:.3f}, "
          f"residual aperiodic={fm.aperiodic_params_}")

# 4) FOOOF re-fit on every individual periodic spectrum -> df_fooof (one row
#    per file). Also saves a re-fit figure for every file under
#    FIGURE_DIR/individual_fits_periodic/<animal>/.
fooof_results = build_fooof_results(
    results_periodic, removal_fits, save_fits=True,
    save_dir=os.path.join(FIGURE_DIR, 'individual_fits_periodic'))
df_fooof = pd.DataFrame(fooof_results)

# 5) Expand into per-property dataframe and make the property/quality plots.
expanded_fooof_df = fooof_results_to_df(fooof_results, theta_band=THETA_BAND)

# Check that the removal left no 1/f: the re-fit's residual aperiodic
# offset/exponent should be close to 0.
print("Residual aperiodic in periodic re-fit (median [IQR]): "
      f"offset {expanded_fooof_df['periodic_fit_offset'].median():+.3f} "
      f"[{expanded_fooof_df['periodic_fit_offset'].quantile(0.25):+.3f}, "
      f"{expanded_fooof_df['periodic_fit_offset'].quantile(0.75):+.3f}], "
      f"exponent {expanded_fooof_df['periodic_fit_exponent'].median():+.3f} "
      f"[{expanded_fooof_df['periodic_fit_exponent'].quantile(0.25):+.3f}, "
      f"{expanded_fooof_df['periodic_fit_exponent'].quantile(0.75):+.3f}]")

# Flag/export files with a poor gating fit (r_squared < R_SQUARED_MIN or
# error > ERROR_MAX; gating fit chosen by QUALITY_GATE_FIT).
export_low_quality_fits(
    expanded_fooof_df,
    os.path.join(OUTPUT_DIR, 'low_quality_fooof_fits.txt'))

expanded_fooof_df.to_csv(os.path.join(OUTPUT_DIR, 'fooof_periodic_results.csv'),
                         index=False)

plot_theta_properties(expanded_fooof_df, props=list(THETA_SPECIFIC), save=True)
plot_fit_quality(expanded_fooof_df, group_col="animal",
                 metrics=["aperiodic_r_squared", "aperiodic_error", "r_squared", "error"],
                 mode="box", save=True)

# 5b) Removed aperiodic-parameter histograms (offset, knee, exponent), by animal.
plot_aperiodic_properties(expanded_fooof_df, save=True)

# 5c) Theta-range quantification (per animal + grand average) and box plot.
theta_summary_df = theta_range_summary(expanded_fooof_df)
theta_summary_df.to_csv(os.path.join(OUTPUT_DIR, 'theta_range_summary.csv'),
                        index=False)
plot_theta_range_boxplot(expanded_fooof_df, summary=theta_summary_df, save=True)

# Grand-average estimated theta range -> grey highlight on the mean-PSD plots
_grand = theta_summary_df[theta_summary_df['animal'] == 'ALL'].iloc[0]
THETA_RANGE_EST = (_grand['theta_low_mean'], _grand['theta_high_mean'])
print(f"Theta highlight range: {THETA_RANGE_EST[0]:.2f}-{THETA_RANGE_EST[1]:.2f} Hz")

# Multi-animal mean +/- SEM periodic-spectrum plot.
fig, ax = plt.subplots(figsize=(6, 5))
plot_mean_psds_all_animals_on_ax(ax, freqs_store, master_psds_dict, animals_periodic,
                                 file_counts=file_counts)
os.makedirs(FIGURE_DIR, exist_ok=True)
fig.savefig(os.path.join(FIGURE_DIR, "mean_periodic_spectra_all_animals.png"),
            dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIGURE_DIR, "mean_periodic_spectra_all_animals.svg"),
            dpi=300, bbox_inches="tight")
plt.show()

# 6) Sample periodic fit + composite summary figure.
fig, ax = plt.subplots(figsize=(5, 3))
s_freqs, s_spec = get_sample_psd(results_periodic)
plot_sample_psd_and_fooof_on_ax(ax, freqs=s_freqs, periodic_spectrum=s_spec,
                                freq_range=tuple(FOOOF_RANGE), xlim=(1, 20))
fig.savefig(os.path.join(FIGURE_DIR, "sample_periodic_fooof.png"),
            dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIGURE_DIR, "sample_periodic_fooof.svg"),
            dpi=300, bbox_inches="tight")
plt.show()

plot_master_summary(results_periodic, master_psds_dict, expanded_fooof_df,
                    file_counts, freqs_store, save=True)

# 7) Band-by-band analysis on the periodic spectrum: periodic band power vs
#    FOOOF peak power (re-fit), per canonical band, with the removed exponent
#    alongside -- layout of the FOOOF docs' band-by-band page:
#    https://fooof-tools.github.io/fooof/auto_motivations/measurements/plot_BandByBand.html
#      a) mean periodic spectra, band-shaded
#      b) per-file periodic band power vs FOOOF peak power vs removed exponent
#      c) pairwise exponent / peak-power / band-power group comparison table
df_band_by_band = build_band_by_band_df(results_periodic, fooof_results, bands=BANDS)
df_band_by_band.to_csv(os.path.join(OUTPUT_DIR, 'band_by_band.csv'), index=False)

plot_band_shaded_spectra(freqs_store, master_psds_dict, animals_periodic, bands=BANDS, save=True)
plot_band_by_band_comparison(df_band_by_band, save=True)

print("\nBand-by-band group comparison (pairwise; band/peak power from each animal's "
      "mean periodic re-fit, exponent from its mean-PSD removal fit):")
df_group_diffs = summarize_band_by_band_group_diffs(mean_fms, mean_fms_aperiodic, bands=BANDS)
df_group_diffs.to_csv(os.path.join(OUTPUT_DIR, 'band_by_band_group_diffs.csv'), index=False)

export_band_by_band_excel(
    df_band_by_band, df_group_diffs,
    os.path.join(OUTPUT_DIR, 'band_by_band.xlsx'), bands=BANDS)

# 8) Peak frequency summary: for every band, the FOOOF peak centre frequency
#    wherever a peak was detected. Theta and fast gamma are printed in detail.
print_peak_frequency_summary(df_band_by_band, mean_fms, bands=BANDS,
                            highlight=('theta', 'fast gamma'))

# 9) Final run-metadata workbook (next to the figures): now includes the
#    computed THETA_RANGE_EST and every output file written during the run.
save_run_metadata(RUN_METADATA_PATH, PROVENANCE, run_status='complete')