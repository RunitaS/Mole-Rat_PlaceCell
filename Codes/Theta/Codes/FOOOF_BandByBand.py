import os
import re
import math
import pickle
import itertools

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

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# %% ==================== Configuration (from reference code) ====================

# Root directories -- one per animal.
# Add/remove/rename animals ONLY here -- plot colors, filenames, and legends
# below are all derived automatically from this dict's keys.
ANIMALS = {
    #'Fa8477':  r'X:/NMR_group_data/Runita/Data/Ephys_Data/AllSortedData/Tetrode/Fa8477',
    # 'FaDDE42': r'C:/Runita/NMR/analysis/SurgeryPaperSpikeLFP/LFP/Main/DDE42',
    'Fa23BD': r'C:/Runita/NMR/analysis/AllSort_Results/LFP/23BDTest',
    'Fa1059': r'C:/Runita/NMR/analysis/AllSort_Results/LFP/1059Test',
}


OUTPUT_DIR = r'C:/Runita/NMR/analysis/AllSort_Results/LFP/BandByBand/v1'  # saved plots go here
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')            # summary figures

# ---- Acquisition / PSD ----
fs      = 32000  # original sampling rate (Hz)
fs_down = 500    # target sampling rate after downsampling
nperseg = int(2 * fs_down)  # 2 s epochs (1000 samples at 1000 Hz) -> df = 0.5 Hz
ADBitVolts = 0.000003051757812500000169  # V per ADC count (Neuralynx header)
MAD_THRESH = 5.0          # dual-criteria epoch-rejection threshold (robust z)
LOW_BAND   = (1.0, 3.0)   # delta band: 1-3 Hz rejection criterion + delta/theta filter
NORM_BAND  = (1.0, 100.0) # band used for relative-power normalization

# Delta/theta epoch filter -- applied to velocity-passed epochs BEFORE MAD
# filtering: an epoch is rejected if its LOW_BAND (delta, 1-3 Hz) power
# exceeds its THETA_BAND (3-7 Hz, defined below) power. THETA_BAND is
# resolved at call time, so its definition later in this file still applies.
APPLY_DELTA_THETA_FILTER = True

# ---- Velocity / running-speed epoch gating ----
# Every .ncs file's session folder holds one tracking .csv with a UNIX
# timestamp (us, column A) on the same clock as the .ncs timestamps, plus x/y
# position in cm (columns D, E). Epochs are additionally gated on running
# speed: an epoch is only kept if the animal's median smoothed speed during it
# falls within [SPEED_MIN_CMS, SPEED_MAX_CMS].
SPEED_MIN_CMS       = 1    # lower running-speed bound for epoch acceptance (cm/s)
SPEED_MAX_CMS       = 90.0   # upper running-speed bound for epoch acceptance (cm/s)
SPEED_SMOOTH_WINDOW = 11     # Savitzky-Golay window (samples, odd; ~0.37 s at 30 Hz)
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

# ---- Theta extraction / property plotting (from notebook) ----
THETA_BAND = (3.0, 7.0)        # Hz window used to pull the theta peak from FOOOF

# ---- Band-by-band analysis (canonical bands) ----
# A naive "band-by-band" power comparison (mean power in a fixed frequency
# window) can't tell whether a difference is a genuine oscillatory (periodic)
# change or just a shift in the aperiodic (1/f) component -- see
# https://fooof-tools.github.io/fooof/auto_motivations/measurements/plot_BandByBand.html
# For each band below we therefore compute BOTH the naive band power AND the
# FOOOF-parameterized peak power (periodic component only) so the two can be
# compared directly. Bounded by FOOOF_RANGE (1-100 Hz above).
BANDS = Bands({
    'delta': [1, 3],
    'theta': [3, 7],
    'alpha': [8, 13],
    'beta':  [13, 30],
    'slow gamma': [30, 50],
    'fast gamma': [50, 90],
})

# Neuralynx .ncs record format (512 int16 samples per record, 16 kB header skipped)
ncs_dtype = np.dtype([
    ('timestamp'  , '<u8'),
    ('sc_number'  , '<u4'),
    ('cell_number', '<u4'),
    ('params'     , '<u4'),
    ('samples'    , '<i2', (512,)),
])

# ---- Plot palette (auto-cycled across however many animals are in ANIMALS) ----
_PALETTE = [
    '#1A56DB',  # blue
    '#4DAF4A',  # green
    '#E41A1C',  # red
    '#FF7F0E',  # orange
    '#9467BD',  # purple
    '#17BECF',  # cyan
    '#BCBD22',  # olive
    '#E377C2',  # pink
]


def _lighten(hex_color, amount=0.55):
    """Blend a hex color toward white, for use as a shaded fill color."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


# {label: (line_color, fill_color)} -- auto-built from ANIMALS, one entry per animal.
STYLES = {
    label: (_PALETTE[i % len(_PALETTE)], _lighten(_PALETTE[i % len(_PALETTE)]))
    for i, label in enumerate(ANIMALS)
}

# convenience: a simple list of line colors in ANIMALS order (used by notebook plots)
animals = list(ANIMALS.keys())
ANIMAL_COLORS = [STYLES[a][0] for a in animals]


# %% ==================== Signal-processing helpers (from reference) ============

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
            lfp, lfp_start_us = load_ncs(fpath)
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

        except Exception as e:
            print(f'  SKIP: {rel} -- {e}')

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


# %% ==================== FOOOF over per-file PSD matrices =======================

def build_fooof_results(results, fooof_settings=None, fooof_range=None,
                        save_fits=False, save_dir=None, fit_xlim=(1, 20)):
    """Fit FOOOF to every per-file PSD (via FOOOFGroup) and return a results list.

    Each entry mirrors the notebook's fooof_results dicts so that the original
    fooof_results_to_df() works unchanged:
        animal, date, session, tetrode, channel,
        aperiodic_params, peak_params, r_squared, error

    If `save_fits` is True, also saves one model-fit figure (original spectrum,
    full model, aperiodic fit) per file under
    `save_dir/<animal>/<sanitized file stem>_fooof_fit.png` (default save_dir:
    FIGURE_DIR/individual_fits, i.e. under OUTPUT_DIR).
    """
    fooof_settings = fooof_settings or FOOOF_SETTINGS
    fooof_range    = fooof_range or FOOOF_RANGE

    if save_fits:
        save_dir = save_dir or os.path.join(FIGURE_DIR, 'individual_fits')

    fooof_results = []
    for animal, (freqs, _mean, _sem, _n, psds_norm, file_names) in results.items():
        print(f"  FOOOFGroup: {animal}  ({psds_norm.shape[0]} PSDs)")
        fg = FOOOFGroup(**fooof_settings)
        fg.fit(freqs, psds_norm, fooof_range)

        if save_fits:
            animal_dir = os.path.join(save_dir, animal)
            os.makedirs(animal_dir, exist_ok=True)

        for i in range(psds_norm.shape[0]):
            # regenerate=True so the modeled spectrum/aperiodic fit are
            # available for plotting (regenerate=False only keeps params).
            fm = fg.get_fooof(i, regenerate=save_fits)
            meta = parse_metadata_from_path(file_names[i])
            fooof_results.append({
                'animal':           animal,
                'file':             file_names[i],
                **meta,
                'aperiodic_params': fm.aperiodic_params_,
                'peak_params':      fm.peak_params_,
                'r_squared':        fm.r_squared_,
                'error':            fm.error_,
            })

            if save_fits:
                fig, ax = plt.subplots(figsize=(5, 3))
                _style_fooof_fit_ax(ax, fm, xlim=fit_xlim,
                                    title=f"{animal}: {os.path.basename(file_names[i])}")
                stem = re.sub(r'[\\/]+', '_', os.path.splitext(file_names[i])[0])
                fig.savefig(os.path.join(animal_dir, f'{stem}_fooof_fit.png'),
                           dpi=200, bbox_inches='tight')
                plt.close(fig)

        if save_fits:
            print(f"    Saved {psds_norm.shape[0]} fit figures -> {animal_dir}")

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
    """
    rows = []
    for r in fooof_results:
        ap = r['aperiodic_params']
        if len(ap) == 2:
            offset, exponent = ap
            knee = np.nan
        else:
            offset, knee, exponent = ap

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
        })

    df = pd.DataFrame(rows)
    print(f"Total units: {len(df)}")
    print(f"Units with theta peak: {df['has_theta'].sum()} ({100 * df['has_theta'].mean():.1f}%)")
    print(f"Animals: {df['animal'].unique()}")
    return df


# ---- FOOOF fit-quality thresholds for flagging poor fits ----
R_SQUARED_MIN = 0.98   # flag files with r_squared below this
ERROR_MAX     = 0.4    # flag files with error above this


def export_low_quality_fits(df, out_path, r2_min=R_SQUARED_MIN, error_max=ERROR_MAX):
    """Write the list of files whose FOOOF fit has r_squared < r2_min OR
    error > error_max to a .txt file (one file path per line)."""
    flagged = df[(df['r_squared'] < r2_min) | (df['error'] > error_max)]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as fh:
        for _, row in flagged.iterrows():
            fh.write(f"{row['file']}\n")
    print(f"Flagged {len(flagged)}/{len(df)} files "
          f"(r_squared < {r2_min} or error > {error_max}) -> {out_path}")
    return flagged


# %% ==================== Band-by-band analysis ===================================
# Raw band power vs FOOOF-parameterized (periodic-only) peak power, per
# canonical band -- see BANDS above and
# https://fooof-tools.github.io/fooof/auto_motivations/measurements/plot_BandByBand.html

def compute_band_power_raw(freqs, power_spectrum, band_def):
    """Mean log10 power within `band_def`, with no aperiodic correction --
    the naive 'band-by-band' measure. log10 to match FOOOF's own working
    space, so this is directly comparable to the peak-power (PW) values
    FOOOF reports for the periodic component."""
    _, band_power = trim_spectrum(freqs, np.log10(power_spectrum), list(band_def))
    return np.mean(band_power)


def build_band_by_band_df(results, fooof_results, bands=None):
    """Per-file, per-band comparison of naive band power vs FOOOF-parameterized
    peak power. Long-format: one row per (file, band).

    Mirrors the FOOOF docs' band-by-band example -- apparent band-power
    differences can arise purely from a shift in the aperiodic exponent
    rather than a genuine oscillatory change, so raw power, FOOOF peak
    power, and the aperiodic exponent are reported side by side for the
    same file/band.
    """
    bands = bands or BANDS

    file_lookup = {}  # {(animal, file): (freqs, power_spectrum)}
    for animal, (freqs, _mean, _sem, _n, psds_norm, file_names) in results.items():
        for i, fname in enumerate(file_names):
            file_lookup[(animal, fname)] = (freqs, psds_norm[i])

    rows = []
    for r in fooof_results:
        freqs, power_spectrum = file_lookup[(r['animal'], r['file'])]
        exponent = r['aperiodic_params'][-1]  # last aperiodic param is always the exponent

        for band_name, band_def in bands:
            band_power_raw = compute_band_power_raw(freqs, power_spectrum, band_def)
            peak_cf, peak_pw, peak_bw = get_band_peak(r['peak_params'], band_def)
            rows.append({
                'animal':          r['animal'],
                'file':            r['file'],
                'band':            band_name,
                'band_low':        band_def[0],
                'band_high':       band_def[1],
                'band_power_raw':  band_power_raw,
                'band_power_peak': peak_pw,
                'peak_cf':         peak_cf,
                'peak_bw':         peak_bw,
                'has_peak':        not np.isnan(peak_pw),
                'exponent':        exponent,
                'r_squared':       r['r_squared'],
            })

    df = pd.DataFrame(rows)
    print(f"Band-by-band: {len(df)} (file, band) rows across "
          f"{len(bands.definitions)} bands and {df['animal'].nunique()} animals")
    return df


def plot_band_shaded_spectra(freqs, master_psds_dict, animals_list, bands=None,
                             freq_range=(1, 90), save=False, save_path=None):
    """Mean PSD per animal with each canonical band shaded (fooof.plts'
    plot_spectra_shading, as used in the FOOOF docs' band-by-band example).

    Trimmed to `freq_range` (default 1-90 Hz) before plotting -- not just an
    xlim -- so the y-axis autoscale reflects only the plotted range instead
    of being skewed by the high-frequency tail up near Nyquist."""
    bands = bands or BANDS
    freqs_trim, spectra = None, []
    for a in animals_list:
        freqs_trim, spec_trim = trim_spectrum(freqs, master_psds_dict[a][0], list(freq_range))
        spectra.append(spec_trim)
    colors  = [ANIMAL_COLORS[i % len(ANIMAL_COLORS)] for i in range(len(animals_list))]
    shade_colors = sns.color_palette('Pastel2', n_colors=len(bands.definitions))

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_spectra_shading(freqs_trim, spectra, log_powers=True, linewidth=2,
                         colors=colors, labels=animals_list,
                         shades=bands.definitions, shade_colors=shade_colors, ax=ax)
    ax.set_title('Mean power spectra with canonical bands shaded', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    _fmt(ax)

    if save:
        out = save_path or os.path.join(FIGURE_DIR, 'band_shaded_spectra.svg')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def plot_flattened_spectra_shaded(mean_fms, animals_list, bands=None,
                                  save=False, save_path=None):
    """Flattened (aperiodic-removed) spectra per animal, band-shaded -- the
    FOOOF docs' second figure. `fm.get_data('peak')` isolates the periodic
    component (data - aperiodic fit, in log10 space) so it can be compared
    directly across animals without the aperiodic component confounding it.

    `mean_fms` : {animal: fitted FOOOF} -- one FOOOF model per animal, fit to
    that animal's mean PSD (see main pipeline step 3).
    """
    bands = bands or BANDS
    freqs = next(iter(mean_fms.values())).freqs
    flat_spectra = [mean_fms[a].get_data('peak', space='log') for a in animals_list]
    colors = [ANIMAL_COLORS[i % len(ANIMAL_COLORS)] for i in range(len(animals_list))]
    shade_colors = sns.color_palette('Pastel2', n_colors=len(bands.definitions))

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_spectra_shading(freqs, flat_spectra, linewidth=2,
                         colors=colors, labels=animals_list,
                         shades=bands.definitions, shade_colors=shade_colors, ax=ax)
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.set_ylabel('Flattened power (data - aperiodic fit, log10)', fontsize=AX_LABEL_FONTSIZE)
    ax.set_title('Flattened spectra (periodic component only) with canonical bands shaded',
                 fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    _fmt(ax)

    if save:
        out = save_path or os.path.join(FIGURE_DIR, 'flattened_spectra_shaded.svg')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def compare_exp(fm1, fm2):
    """Difference in aperiodic exponent between two FOOOF fits (docs' compare_exp)."""
    return fm1.get_params('aperiodic_params', 'exponent') - fm2.get_params('aperiodic_params', 'exponent')


def compare_peak_pw(fm1, fm2, band_def):
    """Difference in FOOOF-parameterized peak power for `band_def` between two
    fits (docs' compare_peak_pw) -- nan if either fit has no peak in the band."""
    pw1 = get_band_peak_fm(fm1, band_def)[1]
    pw2 = get_band_peak_fm(fm2, band_def)[1]
    return pw1 - pw2


def compare_band_pw(fm1, fm2, band_def):
    """Difference in naive (raw, log10) band power for `band_def` between two
    fits (docs' compare_band_pw) -- the measure that can be confounded by a
    pure aperiodic-exponent shift."""
    pw1 = np.mean(trim_spectrum(fm1.freqs, fm1.power_spectrum, list(band_def))[1])
    pw2 = np.mean(trim_spectrum(fm2.freqs, fm2.power_spectrum, list(band_def))[1])
    return pw1 - pw2


def summarize_band_by_band_group_diffs(mean_fms, bands=None):
    """Pairwise, per-band comparison of raw band power, FOOOF peak power, and
    aperiodic exponent between each pair of animals' mean-PSD FOOOF fits --
    directly mirrors the FOOOF docs' compare_exp / compare_peak_pw /
    compare_band_pw functions:
    https://fooof-tools.github.io/fooof/auto_motivations/measurements/plot_BandByBand.html

    `mean_fms` : {animal: fitted FOOOF} (see main pipeline step 3).

    A large `band_power_raw_diff` with a small (or opposite-signed)
    `band_power_peak_diff` for the same band/pair flags exactly the
    aperiodic-confound scenario the docs page demonstrates: the raw
    band-power difference isn't backed by a genuine periodic difference, and
    likely just tracks `exponent_diff` instead.
    """
    bands = bands or BANDS
    rows = []
    for a1, a2 in itertools.combinations(mean_fms.keys(), 2):
        fm1, fm2 = mean_fms[a1], mean_fms[a2]
        exp_diff = compare_exp(fm1, fm2)
        for band_name, band_def in bands:
            rows.append({
                'animal_1':             a1,
                'animal_2':             a2,
                'band':                 band_name,
                'band_low':             band_def[0],
                'band_high':            band_def[1],
                'exponent_diff':        exp_diff,
                'band_power_raw_diff':  compare_band_pw(fm1, fm2, band_def),
                'band_power_peak_diff': compare_peak_pw(fm1, fm2, band_def),
            })
    df = pd.DataFrame(rows)

    for _, row in df.iterrows():
        print(f"  [{row['animal_1']} vs {row['animal_2']}] {row['band']} "
              f"({row['band_low']:g}-{row['band_high']:g} Hz): "
              f"raw power diff={row['band_power_raw_diff']:+.3f}, "
              f"peak power diff={row['band_power_peak_diff']:+.3f}, "
              f"exponent diff={row['exponent_diff']:+.3f}")
    return df


def export_band_by_band_excel(df_bands, df_group_diffs, out_path, bands=None):
    """Write the band-by-band results to a single .xlsx workbook:
      - 'band_by_band' -- the full long-format per-file, per-band table
      - 'group_diffs'  -- pairwise animal comparison (see
                          summarize_band_by_band_group_diffs)
      - one sheet per band (e.g. 'theta_peaks') listing only the files with a
        significant FOOOF-detected peak in that band (has_peak == True) --
        i.e. files where the periodic component is actually present, not
        just files with high raw power in that frequency range.

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
    """For each canonical band: naive raw band power (top row) vs FOOOF
    peak power (middle row) vs aperiodic exponent (bottom row), grouped by
    animal. Reproduces the FOOOF docs' point that raw band-power differences
    don't necessarily reflect a genuine periodic change -- they can instead
    track the aperiodic exponent (repeated per column for visual alignment
    with the band above it)."""
    band_order   = list(dict.fromkeys(df_bands['band']))  # preserve first-seen order
    animal_order = sorted(df_bands['animal'].unique())
    palette = {a: ANIMAL_COLORS[i % len(ANIMAL_COLORS)] for i, a in enumerate(animal_order)}

    n_bands = len(band_order)
    fig, axes = plt.subplots(3, n_bands, figsize=(3.2 * n_bands, 8), squeeze=False)

    for col, band in enumerate(band_order):
        sub = df_bands[df_bands['band'] == band]

        ax_raw = axes[0, col]
        sns.boxplot(data=sub, x='animal', y='band_power_raw', order=animal_order,
                    palette=palette, ax=ax_raw, fliersize=0)
        sns.stripplot(data=sub, x='animal', y='band_power_raw', order=animal_order,
                      ax=ax_raw, color='k', size=2.5, alpha=0.3, jitter=True)
        ax_raw.set_title(f"{band} ({sub['band_low'].iloc[0]:g}-{sub['band_high'].iloc[0]:g} Hz)",
                         fontsize=9)
        ax_raw.set_ylabel('Raw band power\n(log10)' if col == 0 else '')
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
        ax_exp.set_ylabel('Aperiodic exponent' if col == 0 else '')
        ax_exp.set_xlabel('')

        for ax in (ax_raw, ax_pk, ax_exp):
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(axis='x', rotation=45, labelsize=TICK_LABEL_FONTSIZE)
            for t in ax.get_xticklabels():
                t.set_horizontalalignment('right')

    fig.suptitle('Band-by-band: raw power vs FOOOF-parameterized peak power\n'
                 '(bottom row: aperiodic exponent, repeated per animal across bands)',
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
                        alpha=0.6, color=ANIMAL_COLORS[i % len(ANIMAL_COLORS)],
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

    try:
        palette_list = ANIMAL_COLORS
    except NameError:
        palette_list = sns.color_palette("colorblind", n_colors=len(order))
    color_map = {g: palette_list[i % len(palette_list)] for i, g in enumerate(order)}

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
        ax.set_ylim(0.5, None)
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
    """Return (freqs, psd) for one file. Defaults to the first file of the
    first animal. Replaces the notebook's get_psd_from_store()."""
    if animal is None:
        animal = next(iter(results))
    freqs, _mean, _sem, _n, psds_norm, _files = results[animal]
    return freqs, psds_norm[index]


AX_LABEL_FONTSIZE   = 10
TICK_LABEL_FONTSIZE = 10


def _fmt(ax):
    ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)


def plot_mean_psds_all_animals_on_ax(ax, freqs, master_psds_dict, animals_list,
                                     file_counts=None, xlim=(2, 20)):
    for i, animal in enumerate(animals_list):
        mean, sem = master_psds_dict[animal]
        color = ANIMAL_COLORS[i % len(ANIMAL_COLORS)]
        if file_counts is not None:
            label = f"{animal} ({file_counts.get(animal, '?')} files)"
        else:
            label = str(animal)
        ax.plot(freqs, mean, lw=1.7, color=color, label=label)
        ax.fill_between(freqs, mean - sem, mean + sem, color=color, alpha=0.25)
    ax.set_xlim(xlim)
    ax.set_xlabel("Frequency (Hz)", fontsize=AX_LABEL_FONTSIZE)
    ax.set_ylabel("Normalized PSD", fontsize=AX_LABEL_FONTSIZE)
    ax.set_title("A  Mean Power Spectra", y=1.03, fontsize=10, pad=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.axvspan(3, 7, color='gray', alpha=0.12, zorder=0)  # theta band highlight
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
        ("Original PSD", "#333333", "-",  1.6),
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
    ax.set_ylabel("Power", fontsize=AX_LABEL_FONTSIZE)
    ax.grid(False)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    _fmt(ax)


def plot_sample_psd_and_fooof_on_ax(ax, freqs, psd, fooof_kwargs=None,
                                    freq_range=(1, 20), xlim=(1, 20)):
    if fooof_kwargs is None:
        fooof_kwargs = dict(**FOOOF_SETTINGS, verbose=False)
    fm = FOOOF(**fooof_kwargs)
    fm.fit(freqs, psd, list(freq_range))
    _style_fooof_fit_ax(ax, fm, xlim=xlim, title="B  Sample FOOOF fit")
    return fm


def plot_theta_prop_hist_on_ax(ax, df, prop, xlim=None, xlabel=None, title=None):
    df_theta = df[df["has_theta"]].copy()
    animals_here = sorted(df_theta["animal"].unique())
    palette = {a: ANIMAL_COLORS[i % len(ANIMAL_COLORS)]
               for i, a in enumerate(animals_here)}

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

    fig.suptitle("Characterising LFP Power Spectra using FOOOF", fontsize=12, y=0.97)

    # A: mean PSDs
    plot_mean_psds_all_animals_on_ax(axes["A"], freqs, master_psds_dict,
                                     list(master_psds_dict.keys()),
                                     file_counts=file_counts)

    # B: sample FOOOF fit (first file of first animal)
    s_freqs, s_psd = get_sample_psd(results)
    plot_sample_psd_and_fooof_on_ax(axes["B"], freqs=s_freqs, psd=s_psd,
                                    freq_range=tuple(FOOOF_RANGE), xlim=(1, 20))

    # C1 / C2: fit quality
    def _quality_ax(ax, col, ylim, title):
        order_animals = sorted(expanded_fooof_df["animal"].unique())
        palette = {a: ANIMAL_COLORS[i % len(ANIMAL_COLORS)]
                   for i, a in enumerate(order_animals)}
        long_df = expanded_fooof_df[["animal", col]].dropna()
        sns.boxplot(data=long_df, x="animal", y=col, ax=ax,
                    palette=palette, fliersize=0, linewidth=0.8)
        sns.stripplot(data=long_df, x="animal", y=col, ax=ax,
                      palette=palette, size=2.5, alpha=0.35, jitter=True)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=AX_LABEL_FONTSIZE, pad=4)
        ax.set_xlabel("")
        ax.set_ylabel(title, fontsize=AX_LABEL_FONTSIZE)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', rotation=30, labelsize=TICK_LABEL_FONTSIZE)
        ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)

    _quality_ax(axes["C1"], "r_squared", (0.9, 1.0), "R\u00b2")
    _quality_ax(axes["C2"], "error", (0, 0.1), "Error")

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


# %% ==================== MAIN PIPELINE (PSD -> FOOOF -> plots) ==================

# 1) Generate PSDs for every animal via the reference folder-walk pipeline.
results = {}
for label, folder in ANIMALS.items():
    print(f"=== Processing {label} ===")
    results[label] = process_animal(label, folder)

# Cache the frequency vector and per-animal summaries the notebook code expects.
freqs_store      = results[next(iter(results))][0]
master_psds_dict = {a: (r[1], r[2]) for a, r in results.items()}   # {animal: (mean, sem)}
file_counts      = {a: r[3] for a, r in results.items()}           # {animal: n_files}

# Optionally persist the processed PSDs.
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, 'processed_psds.pkl'), 'wb') as fh:
    pickle.dump({a: {'freqs': r[0], 'mean': r[1], 'sem': r[2],
                     'psds': r[4], 'files': r[5]} for a, r in results.items()}, fh)

# 2) Multi-animal mean +/- SEM PSD plot.
fig, ax = plt.subplots(figsize=(6, 5))
plot_mean_psds_all_animals_on_ax(ax, freqs_store, master_psds_dict, animals,
                                 file_counts=file_counts)
os.makedirs(FIGURE_DIR, exist_ok=True)
fig.savefig(os.path.join(FIGURE_DIR, "mean_psds_all_animals.svg"),
            dpi=300, bbox_inches="tight")
plt.show()

# 3) FOOOF summary on each animal's averaged PSD (quick report).
#    mean_fms is reused below by the band-by-band group comparison (step 7).
mean_fms = {}
for animal, (freqs, mean, sem, n, psds_norm, files) in results.items():
    fm = FOOOF(**FOOOF_SETTINGS)
    fm.fit(freqs, mean, FOOOF_RANGE)
    mean_fms[animal] = fm
    print(f"[{animal}] mean-PSD FOOOF: R\u00b2={fm.r_squared_:.3f}, "
          f"error={fm.error_:.3f}, aperiodic={fm.aperiodic_params_}")

# 4) FOOOF on every individual PSD -> df_fooof (flat, one row per file).
#    Also saves a model-fit figure (original spectrum, full model, aperiodic
#    fit) for every file under FIGURE_DIR/individual_fits/<animal>/ (under OUTPUT_DIR).
fooof_results = build_fooof_results(
    results, save_fits=True,
    save_dir=os.path.join(FIGURE_DIR, 'individual_fits'))
df_fooof = pd.DataFrame(fooof_results)

# 5) Expand into per-property dataframe and make the property/quality plots.
expanded_fooof_df = fooof_results_to_df(fooof_results, theta_band=THETA_BAND)

# Flag/export files with a poor FOOOF fit (r_squared < 0.98 or error > 0.4).
export_low_quality_fits(
    expanded_fooof_df,
    os.path.join(OUTPUT_DIR, 'low_quality_fooof_fits.txt'))

plot_theta_properties(expanded_fooof_df, props=list(THETA_SPECIFIC), save=False)
plot_fit_quality(expanded_fooof_df, group_col="animal", metrics="both",
                 mode="box", save=False)

# 6) Sample fit + composite summary figure.
fig, ax = plt.subplots(figsize=(5, 3))
s_freqs, s_psd = get_sample_psd(results)
plot_sample_psd_and_fooof_on_ax(ax, freqs=s_freqs, psd=s_psd,
                                freq_range=tuple(FOOOF_RANGE), xlim=(1, 20))
fig.savefig(os.path.join(FIGURE_DIR, "sample_fooof.svg"),
            dpi=300, bbox_inches="tight")
plt.show()

plot_master_summary(results, master_psds_dict, expanded_fooof_df,
                    file_counts, freqs_store, save=True)

# 7) Band-by-band analysis: naive raw band power vs FOOOF-parameterized peak
#    power, per canonical band (delta/theta/alpha/beta) -- reproduces every
#    result on the FOOOF docs' band-by-band page using the real animals as
#    groups: https://fooof-tools.github.io/fooof/auto_motivations/measurements/plot_BandByBand.html
#      a) original spectra, band-shaded
#      b) flattened (aperiodic-removed) spectra, band-shaded
#      c) per-file raw power vs FOOOF peak power vs exponent (boxplots)
#      d) pairwise exponent / peak-power / band-power group comparison table
df_band_by_band = build_band_by_band_df(results, fooof_results, bands=BANDS)
df_band_by_band.to_csv(os.path.join(OUTPUT_DIR, 'band_by_band.csv'), index=False)

plot_band_shaded_spectra(freqs_store, master_psds_dict, animals, bands=BANDS, save=True)
plot_flattened_spectra_shaded(mean_fms, animals, bands=BANDS, save=True)
plot_band_by_band_comparison(df_band_by_band, save=True)

print("\nBand-by-band group comparison (pairwise, from each animal's mean-PSD FOOOF fit):")
df_group_diffs = summarize_band_by_band_group_diffs(mean_fms, bands=BANDS)
df_group_diffs.to_csv(os.path.join(OUTPUT_DIR, 'band_by_band_group_diffs.csv'), index=False)

export_band_by_band_excel(
    df_band_by_band, df_group_diffs,
    os.path.join(OUTPUT_DIR, 'band_by_band.xlsx'), bands=BANDS)