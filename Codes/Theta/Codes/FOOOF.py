import os
import re
import math
import pickle

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
from fooof.analysis import get_band_peak
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


OUTPUT_DIR = r'C:/Runita/NMR/analysis/AllSort_Results/LFP/thetadeltafilt/v1'  # saved plots go here
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
FOOOF_RANGE    = [1.0, 40.0]   # fit range (Hz)
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
# compared directly. Bounded by FOOOF_RANGE (1-40 Hz above).
BANDS = Bands({
    'delta': [1, 4],
    'theta': [4, 8],
    'alpha': [8, 13],
    'beta':  [13, 30],
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

