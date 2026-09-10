import os
import re
import math
import pickle

import numpy as np
import pandas as pd
from scipy import signal, stats, interpolate, ndimage

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from fooof import FOOOF, FOOOFGroup
from fooof.utils import interpolate_spectrum, trim_spectrum
from fooof.analysis import get_band_peak_fm, get_band_peak_fg
from fooof.bands import Bands
from fooof.plts.spectra import plot_spectra_shading

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# %% ==================== Configuration (from reference code) ====================

# Root directories -- one per animal.
# Add/remove/rename animals ONLY here -- plot colors, filenames, and legends
# below are all derived automatically from this dict's keys.
ANIMALS = {
    'Fa8477':  r'X:/NMR_group_data/Runita/Data/Ephys_Data/AllSortedData/Tetrode/Fa8477',
    # 'FaDDE42': r'C:/Runita/NMR/analysis/SurgeryPaperSpikeLFP/LFP/Main/DDE42',
    #'Fa23BD': r'C:/Runita/NMR/analysis/AllSort_Results/LFP/23BDTest',
    #'Fa1059': r'C:/Runita/NMR/analysis/AllSort_Results/LFP/1059Test',
}


OUTPUT_DIR = r'C:/Runita/NMR/analysis/AllSort_Results/LFP/thetadeltafilt/v2_aperiodicChar_clean'  # saved plots go here
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')            # summary figures

# ---- Acquisition / PSD ----
fs      = 32000  # original sampling rate (Hz)
fs_down = 1000    # target sampling rate after downsampling
nperseg = int(2 * fs_down)  # 2 s epochs (1000 samples at 1000 Hz) -> df = 0.5 Hz
ADBitVolts = 0.000003051757812500000169  # V per ADC count (Neuralynx header)
MAD_THRESH = 5.0          # dual-criteria epoch-rejection threshold (robust z)
LOW_BAND   = (1.0, 3.0)   # delta band: 1-3 Hz rejection criterion + delta/theta filter
NORM_BAND  = (1.0, 100.0) # band used for relative-power normalization

# Delta/theta epoch filter -- applied BEFORE MAD filtering: an epoch is
# rejected if its LOW_BAND (delta, 1-3 Hz) power exceeds its THETA_BAND
# (3-7 Hz, defined below) power. THETA_BAND is resolved at call time, so its
# definition later in this file still applies.
APPLY_DELTA_THETA_FILTER = True

# ---- Tracking-position artifact correction ----
# Every .ncs file's session folder holds one tracking .csv with a UNIX
# timestamp (us, column A) on the same clock as the .ncs timestamps, plus x/y
# position in cm (columns D, E). LFP epochs are NOT gated on running speed --
# all epochs are analyzed regardless of the animal's speed. Velocity is used
# only to clean the raw tracking positions: a frame-to-frame step implying a
# speed above POS_JUMP_THRESH_CMS is physically impossible for the animal and
# is treated as a tracking artifact, corrected by linear interpolation from
# the nearest valid samples. Low speeds (e.g. resting/grooming) are genuine
# and are left untouched.
POS_JUMP_THRESH_CMS = 90.0   # speed (cm/s) above which a tracking step is treated as an artifact

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
    """Remove a polynomial trend (default: linear) from the full LFP trace,
    to clear slow drift that a notch filter alone doesn't address."""
    return signal.detrend(np.asarray(x, dtype=np.float64), type=dtype) # type: ignore # type: ignore


def _robust_high_outliers(x, thresh, ref_mask=None):
    """Boolean mask of samples that are high outliers by robust (MAD) z-score.

    If `ref_mask` is given, the median/MAD reference statistics are estimated
    from `x[ref_mask]` only (e.g. epochs that already passed the delta/theta
    filter), so epochs excluded upstream can't skew the robust threshold --
    z-scores are still returned for every element of `x`.
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
                             apply_delta_theta_filter=True):
    """Welch PSD averaged over 4 s epochs, with epoch rejection applied in
    two stages, in order (all epochs are analyzed regardless of running
    speed -- no velocity gating):

      1. Delta/theta filter -- an epoch is rejected if its `low_band` (delta,
         e.g. 1-3 Hz) power exceeds its `theta_band` (e.g. 3-7 Hz) power.
      2. Dual-criteria MAD outlier rejection -- an epoch is rejected if it is
         a high outlier (robust MAD z > mad_thresh) on EITHER broadband
         peak-to-peak amplitude OR `low_band` power. The median/MAD reference
         statistics are computed ONLY from the epochs that survive (1), so
         epochs already excluded can't skew the robust threshold.

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
        f, Pi = signal.welch(epochs[i], fs=fs_hz, window=win, # type: ignore
                             nperseg=nperseg, noverlap=0, detrend='constant')
        if psd_stack is None:
            psd_stack = np.empty((n_total, Pi.size))
        psd_stack[i] = Pi
        df  = f[1] - f[0]
        idx_low   = (f >= low_band[0])   & (f <= low_band[1])
        idx_theta = (f >= theta_band[0]) & (f <= theta_band[1])
        band_pow[i]  = np.sum(Pi[idx_low]) * df
        theta_pow[i] = np.sum(Pi[idx_theta]) * df

    # 1) delta/theta filter
    keep = np.ones(n_total, dtype=bool)
    if apply_delta_theta_filter:
        keep = keep & (band_pow <= theta_pow)

    # 2) MAD outlier rejection, referenced to the epochs kept by (1)
    reject = _robust_high_outliers(p2p, mad_thresh, ref_mask=keep) | \
             _robust_high_outliers(band_pow, mad_thresh, ref_mask=keep)
    keep = keep & ~reject

    n_clean = int(keep.sum())
    if n_clean == 0:
        keep = np.ones(n_total, dtype=bool)
        n_clean = n_total

    Pxx = psd_stack[keep].mean(axis=0) # type: ignore
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


def clean_tracking_position(csv_path, jump_thresh_cms=POS_JUMP_THRESH_CMS):
    """Correct tracking-artifact jumps in a tracking .csv by interpolation.

    Column layout (positional): UNIX timestamp in us (col A) on the same
    clock as the .ncs LFP timestamps, x in cm (col D), y in cm (col E).

    A frame-to-frame step implying a speed above `jump_thresh_cms` (e.g.
    90 cm/s) is physically impossible for the animal and is treated as a
    tracking artifact rather than genuine movement. Detection is iterative:
    after marking the offending frames "bad", the remaining good frames are
    re-tested against each other, so runs of two or more consecutive bad
    frames (a tracker glitch that lingers) are caught too, not just
    single-frame jumps relative to the immediately preceding raw sample. Bad
    frames are filled by linear interpolation (in time) between the nearest
    surviving good samples, rather than held at the last good position --
    holding would freeze position then "snap" back at the far edge of a
    dropout, itself implying an extra artificial speed spike right at the
    resumption point.

    Genuinely low/near-zero speeds (resting, grooming) are NOT touched --
    only steps above `jump_thresh_cms` are treated as artifacts.

    Returns
    -------
    time_us : ndarray   absolute UNIX timestamp of each tracking sample (us)
                        -- same clock as the .ncs LFP start timestamp
    x_clean : ndarray   x position (cm), jump artifacts interpolated
    y_clean : ndarray   y position (cm), jump artifacts interpolated
    """
    df = pd.read_csv(csv_path, usecols=[0, 3, 4])
    df.columns = ['time_us', 'x', 'y']
    df = df.sort_values('time_us').reset_index(drop=True)

    time_us = df['time_us'].to_numpy(dtype=np.float64)
    x = df['x'].to_numpy(dtype=np.float64)
    y = df['y'].to_numpy(dtype=np.float64)

    n = len(x)
    if n < 2:
        return time_us, x.copy(), y.copy()

    bad = np.zeros(n, dtype=bool)
    for _ in range(n):
        good_idx = np.where(~bad)[0]
        if len(good_idx) < 2:
            break
        dt_good = np.diff(time_us[good_idx]) * 1e-6
        with np.errstate(invalid='ignore', divide='ignore'):
            step_speed = np.hypot(np.diff(x[good_idx]), np.diff(y[good_idx])) / dt_good
        step_speed[dt_good <= 0] = 0.0

        newly_bad = step_speed > jump_thresh_cms
        if not newly_bad.any():
            break
        # Mark the later sample of each offending pair as bad and re-test
        # against the remaining good set next round.
        bad[good_idx[1:][newly_bad]] = True

    good_idx = np.where(~bad)[0]
    if len(good_idx) == 0 or len(good_idx) == n:
        return time_us, x.copy(), y.copy()

    x_clean = np.interp(time_us, time_us[good_idx], x[good_idx])
    y_clean = np.interp(time_us, time_us[good_idx], y[good_idx])
    return time_us, x_clean, y_clean


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

    for fpath in ncs_files:
        rel = os.path.relpath(fpath, folder)
        try:
            lfp, _lfp_start_us = load_ncs(fpath)
            lfp = signal.resample_poly(lfp, fs_down, fs)
            if APPLY_TIME_NOTCH:
                lfp = notch_filter(lfp, fs_down, LINE_HARMONICS, NOTCH_Q)
            if APPLY_TIME_DETREND:
                lfp = detrend_signal(lfp, dtype=DETREND_TYPE)

            f, Pxx, n_clean, n_total = compute_psd_clean_epochs(
                lfp, fs_down, nperseg, mad_thresh=MAD_THRESH, low_band=LOW_BAND,
                apply_delta_theta_filter=APPLY_DELTA_THETA_FILTER)

            if APPLY_SPECTRAL_INTERP:
                f, Pxx = clean_line_noise_psd(
                    f, Pxx, LINE_HARMONICS, INTERP_HALFWIDTH)

            if freq_vec is None:
                freq_vec = f

            df        = f[1] - f[0] # type: ignore
            valid_idx = (f >= NORM_BAND[0]) & (f <= NORM_BAND[1]) # type: ignore
            total_power = np.sum(Pxx[valid_idx]) * df
            Pxx_norm    = Pxx / total_power

            psds.append(Pxx_norm)
            file_names.append(rel)
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
            animal_dir = os.path.join(save_dir, animal) # type: ignore
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
                _style_fooof_fit_ax(ax, fm, xlim=fit_xlim, # type: ignore
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


AX_LABEL_FONTSIZE = 10


def _style_fooof_fit_ax(ax, fm, xlim=(1, 20), title="Sample FOOOF fit",
                        theta_band=None):
    """Plot an already-fit FOOOF model (original spectrum, full model, aperiodic
    fit) onto `ax` with the shared color/label styling used across the script.

    Also extracts the theta peak (strongest peak within theta_band) from `fm`
    and shades/labels its [cf - bw/2, cf + bw/2] range on the axis, and
    annotates the fitted aperiodic parameters (offset, knee, exponent).
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
    ax.text(0.5, 1.01, f"R²={fm.r_squared_:.3f}, error={fm.error_:.3f}",
            transform=ax.transAxes, ha='center', va='bottom', fontsize=8)

    ap = fm.aperiodic_params_
    if len(ap) == 2:
        offset, exponent = ap
        knee = np.nan
    else:
        offset, knee, exponent = ap
    knee_str = f"{knee:.3f}" if not np.isnan(knee) else "n/a"
    ax.text(0.02, 0.03,
            f"Offset={offset:.3f}\nKnee={knee_str}\nExponent={exponent:.3f}",
            transform=ax.transAxes, ha='left', va='bottom', fontsize=7.5,
            color='#EA080C')

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

        if 'animal' in df.columns:
            animals_here = sorted(df['animal'].unique())
            for i, animal in enumerate(animals_here):
                vals = df.loc[df['animal'] == animal, key].dropna()
                ax.hist(vals, bins=20, range=xlim,
                        alpha=0.6, color=ANIMAL_COLORS[i % len(ANIMAL_COLORS)],
                        edgecolor='white', lw=0.5, label=str(animal))
            ax.legend(fontsize=7)
        else:
            ax.hist(vals_all, bins=20, range=xlim,
                    color='#AAAAAA', edgecolor='#555555', lw=0.6)
            ax.axvline(vals_all.median(), color='steelblue', lw=1.5, ls='--',
                       label=f'median = {vals_all.median():.2f}')
            ax.legend(fontsize=7)

        ax.set_title(f'n = {vals_all.size}', fontsize=8)
        ax.set_xlabel(meta['xlabel'])
        ax.set_xlim(xlim)
        ax.set_ylabel('No. of recordings')
        ax.spines[['top', 'right']].set_visible(False)

    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    fig.suptitle('Aperiodic-fit properties (all recordings)', fontsize=11)
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


# %% ==================== MAIN PIPELINE (PSD -> FOOOF -> aperiodic plots) =======

if __name__ == '__main__':

    # 1) Generate PSDs for every animal via the reference folder-walk pipeline.
    results = {}
    for label, folder in ANIMALS.items():
        print(f"=== Processing {label} ===")
        results[label] = process_animal(label, folder)

    # Persist the processed PSDs so downstream steps can be re-run without
    # redoing the (slow) .ncs -> PSD pass.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'processed_psds.pkl'), 'wb') as fh:
        pickle.dump({a: {'freqs': r[0], 'mean': r[1], 'sem': r[2],
                         'psds': r[4], 'files': r[5]} for a, r in results.items()}, fh)

    # 2) FOOOF on every individual PSD -> flat per-file results.
    #    Also saves a model-fit figure (original spectrum, full model,
    #    aperiodic fit) for every file under FIGURE_DIR/individual_fits/<animal>/.
    fooof_results = build_fooof_results(
        results, save_fits=True,
        save_dir=os.path.join(FIGURE_DIR, 'individual_fits'))

    # 3) Expand into per-property dataframe, export it, and flag poor fits.
    expanded_fooof_df = fooof_results_to_df(fooof_results, theta_band=THETA_BAND)
    expanded_fooof_df.to_csv(os.path.join(OUTPUT_DIR, 'fooof_aperiodic_results.csv'),
                             index=False)

    export_low_quality_fits(
        expanded_fooof_df,
        os.path.join(OUTPUT_DIR, 'low_quality_fooof_fits.txt'))

    # 4) Aperiodic-parameter histograms (offset, knee, exponent), by animal.
    plot_aperiodic_properties(expanded_fooof_df, save=True)

