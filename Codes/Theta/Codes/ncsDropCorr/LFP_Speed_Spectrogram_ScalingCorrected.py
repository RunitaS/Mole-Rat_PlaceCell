"""
LFP figure for every .ncs file, saved in a LFP_Speed_Spectrogram subfolder next to
the file. Top to bottom:

  A  Smoothed running speed, clipped to the LFP start and end.
  B  5 s LFP snippet, with the theta-filtered signal overlaid and colored by its
     Hilbert phase.
  C  Spectrogram of the whole session, 1-100 Hz.
  D  Same spectrogram, 1-20 Hz.

No epochs are dropped. Noise is reduced by filtering the whole trace, following
SpeedSpectrogram_v2.m: timestamp-anchored .ncs loading -> resample to 1 kHz -> mains
notch -> wavelet denoising -> linear detrend -> outlier fill. As in that code, the
median and Savitzky-Golay steps do not feed the spectrograms (APPLY_SMOOTHING): they
remove nearly all power above 30 Hz. Mains harmonics are also interpolated in the
spectrogram, as in Fig4a_PSD_v2_Simul_ScalingCorrected.py.

Unlike the .m file, the trace stays at 1 kHz, not 250 Hz (see fs_down), so every step
after the resample, including the spectrograms, runs at 1 kHz.

filloutliers, medfilt1 and smoothdata ports match MATLAB R2025a exactly on 60000
samples of real LFP. wdenoise is NOT reproduced exactly: its default is an empirical
Bayes threshold, while denoise_wavelet uses BayesShrink with the same wavelet and level.
"""

import os
import re
import sys
import shutil
import getpass
import hashlib
import datetime
import platform
import subprocess
import importlib.metadata

import numpy as np
import pandas as pd
import pywt
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from fooof.utils import interpolate_spectrum

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ---- Timestamp-anchored .ncs loading ----
# ncs_repair.py corrects the partial-record and sample-loss timing error in the
# raw .ncs files (see its header). It sits in this script's folder; the absolute
# path also finds it when running cell-by-cell from another working directory.
NCS_REPAIR_DIR = r'C:\Runita\NMR\Mole-Rat_PlaceCell\Codes\Theta\Codes\ncsDropCorr'
if NCS_REPAIR_DIR not in sys.path:
    sys.path.insert(0, NCS_REPAIR_DIR)
from ncs_repair import load_repaired_uv


# %% ==================== Configuration ==========================================

# Root folder holding one subfolder per animal, named exactly with that
# animal's ID (e.g. ROOT_DIR/1059Test/...).
ROOT_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final'

# Each figure goes in this subfolder of its .ncs file's own folder.
FIGURE_SUBFOLDER = 'LFP_Speed_Spectrogram'
# Run metadata and the code snapshot.
OUTPUT_DIR = os.path.join(ROOT_DIR, 'LFP_Speed_Spectrogram_RunInfo')

# One figure is made per .ncs file of each animal listed here.
ANIMALS_TO_INCLUDE = [
    'Fa1059',
    'Fa23BD',
    'Fa5834',
    'Fa8477',
]

# {animal_id: folder_path} -- auto-built from ANIMALS_TO_INCLUDE + ROOT_DIR.
ANIMALS = {
    animal_id: os.path.join(ROOT_DIR, animal_id)
    for animal_id in ANIMALS_TO_INCLUDE
}

# ---- Which files, and which snippet in each ----
# Keys and entries are .ncs paths relative to ROOT_DIR, e.g.
# r'Fa1059\Open\Day11\3Cntrl\CSC1ch2.ncs'.
# ONLY_FILES: make figures for just these files. Empty: every .ncs file.
ONLY_FILES = []
# SNIPPET_START: snippet start in s from the LFP start, for a file listed here.
# Otherwise the highest-scoring snippet is used. Its score is the lowest theta share
# (theta power / NORM_BAND power) across its 2 s spectrogram windows, so theta has to
# last the whole 5 s: a short large burst can't win. Snippets with no saturated samples
# and a median speed of at least SNIPPET_MIN_SPEED_CMS are preferred; every file gets a
# figure, so these are relaxed if no snippet qualifies (see snippet_criteria in the log).
SNIPPET_START = {}
SNIPPET_DUR_S  = 5.0
SNIPPET_STEP_S = 0.5    # spacing of candidate snippet starts (= spectrogram step)
SNIPPET_MIN_SPEED_CMS = 4.0   # ACG_Utils.SPEED_THRESH_CMS (moving vs immobile)
SATURATION_FRAC = 0.99  # a raw sample at or above this fraction of the int16 range is saturated

# 'session': speed covers the whole LFP recording, like the spectrograms.
# 'snippet': speed covers only the 5 s of LFP directly below it.
SPEED_SPAN = 'session'
# Extra Gaussian smoothing of the plotted speed only. Snippet choice and the median
# speed in the snippet title use the PSD code's speed, which is frame-level.
SPEED_PLOT_SIGMA_S = 0.25

# ---- Acquisition ----
fs      = 32000  # original sampling rate (Hz)
fs_down = 1000   # target sampling rate (SpeedSpectrogram_v2.m uses 250 Hz)
nperseg = int(2 * fs_down)  # 2 s spectrogram window -> df = 0.5 Hz
ADBITVOLTS_NO_HEADER = 0.000003051757812500000169  # only if a header has no -ADBitVolts
NORM_BAND  = (1.0, 100.0) # band used for relative-power normalization
THETA_BAND = (3.0, 7.0)   # Hz

SPEED_SMOOTH_WINDOW = 5     # Savitzky-Golay window (samples, odd)
SPEED_SMOOTH_POLY   = 3

# ---- Noise reduction (replaces epoch rejection) ----
# Sample-based windows have the same duration as SpeedSpectrogram_v2.m's at 250 Hz, so
# the outlier and smoothing windows assume fs_down = 1000 Hz.
LINE_HARMONICS   = [50.0, 100.0, 150.0, 200.0]  # harmonics below the 500 Hz Nyquist
NOTCH_Q          = 30.0           # PSD code's value (the .m file has 35, 50 Hz only)
INTERP_HALFWIDTH = 2.0
# At 250 Hz, level 4 zeroes the whole 62.5-125 Hz level, so what shows above 62.5 Hz is
# a mirror image of lower frequencies: measured, a false dark line at 75 Hz (the image
# of the 50 Hz notch; power at 75 Hz vs neighbours 0.3, was 1.0) and 70-90% of 60-90 Hz
# power gone. At 1 kHz the zeroed level is 250-500 Hz: no line (1.0) and >= 90% of power
# below 90 Hz is kept.
WAVELET_NAME     = 'sym4'         # wdenoise default
WAVELET_LEVEL    = 4              # wdenoise(x, 4)
DETREND_TYPE     = 'linear'
OUTLIER_WINDOW   = 400            # 0.4 s: filloutliers(x, 'linear', 'movmedian', 100) at 250 Hz
OUTLIER_THRESH   = 3.0            # local scaled MADs (filloutliers default)
# medfilt1(x, 5) + smoothdata(x, 'sgolay', 15). Off because, measured on 3 sessions,
# they leave ~1% of 30-60 Hz power and ~0% of 60-90 Hz power (theta: ~98%), which
# blanks the upper half of the 1-100 Hz spectrogram.
APPLY_SMOOTHING  = False
MEDFILT_KERNEL   = 21             # 5 samples at 250 Hz
SGOLAY_WINDOW    = 61             # 15 samples at 250 Hz
SGOLAY_POLY      = 2              # smoothdata 'sgolay' default

# ---- Theta phase ----
THETA_FILTER_ORDER = 5  # Butterworth band-pass, as in ThetaVsSpeed_ACGMethod (sosfiltfilt)

# ---- Spectrogram ----
# Same 2 s Hann window, constant detrend and density scaling as the PSD's Welch.
SPEC_NOVERLAP = nperseg - int(SNIPPET_STEP_S * fs_down)
SPEC_CMAP     = 'viridis'
SPEC_PERCENTILES = (1, 99)  # color limits, per panel

PHASE_CMAP = 'hsv'  # cyclic: -pi and +pi get the same color
THETA_ALPHA = 0.5   # opacity of the phase-colored theta trace, so the LFP shows through

# Per-animal colors — first 4 colors of matplotlib's Pastel1 palette.
ANIMAL_COLORS = {
    'Fa8477': '#FBB4AE',  # Pastel1[0] red
    'Fa1059': '#B3CDE3',  # Pastel1[1] blue
    'Fa23BD': '#CCEBC5',  # Pastel1[2] green
    'Fa5834': '#DECBE4',  # Pastel1[3] purple
}

AX_LABEL_FONTSIZE   = 10
TICK_LABEL_FONTSIZE = 10


def animal_color(label, default='#888888'):
    """ANIMAL_COLORS entry for an animal label, matched on the ID so that e.g.
    'Fa1059', '1059Test' and 'FA1059' all resolve; `default` if no ID matches."""
    label = str(label).upper()
    for animal_id, color in ANIMAL_COLORS.items():
        if animal_id[2:].upper() in label:
            return color
    return default


def _fmt(ax):
    ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)


# %% ==================== Signal-processing helpers (from the PSD code) ==========

def read_ncs_adbitvolts(fpath):
    """V per ADC count, from the '-ADBitVolts' line of the 16 kB .ncs header;
    ADBITVOLTS_NO_HEADER if the header has none (simulated files)."""
    with open(fpath, 'rb') as f:
        header = f.read(16 * 1024).decode('latin-1')
    m = re.search(r'-ADBitVolts\s+(\S+)', header)
    if m is None:
        return ADBITVOLTS_NO_HEADER
    return float(m.group(1))


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


def denoise_wavelet(x):
    """Soft-threshold the sym4 detail coefficients, with the noise level estimated
    from the finest level. BayesShrink (Chang et al. 2000), standing in for wdenoise's
    default empirical Bayes threshold; a level with no signal above noise is zeroed."""
    coeffs = pywt.wavedec(x, WAVELET_NAME, level=WAVELET_LEVEL)
    noise_sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    denoised = [coeffs[0]]
    for detail in coeffs[1:]:
        signal_var = max(np.mean(detail ** 2) - noise_sigma ** 2, 0)
        thresh = noise_sigma ** 2 / np.sqrt(signal_var) if signal_var > 0 else np.abs(detail).max()
        denoised.append(pywt.threshold(detail, thresh, 'soft'))
    return pywt.waverec(denoised, WAVELET_NAME)[:len(x)]


def fill_outliers(x, window=OUTLIER_WINDOW, thresh=OUTLIER_THRESH, chunk=20000):
    """MATLAB's filloutliers(x, 'linear', 'movmedian', window): a sample more than
    `thresh` scaled MADs from its moving median is replaced by linear interpolation.
    Windows are k/2 back and k/2 - 1 forward, shrinking at the ends.

    Returns the filled trace and the outlier mask."""
    n = len(x)
    n_before = window // 2
    n_after = window - n_before - 1
    med, mad = np.empty(n), np.empty(n)
    windows = sliding_window_view(x, window)
    for j in range(0, len(windows), chunk):
        block = windows[j:j + chunk]
        m = np.median(block, axis=1)
        med[j + n_before:j + n_before + len(block)] = m
        mad[j + n_before:j + n_before + len(block)] = np.median(np.abs(block - m[:, None]), axis=1)
    for k in [*range(n_before), *range(n - n_after, n)]:
        block = x[max(0, k - n_before):min(n, k + n_after + 1)]
        med[k] = np.median(block)
        mad[k] = np.median(np.abs(block - med[k]))
    is_outlier = np.abs(x - med) > thresh * 1.482602218505602 * mad

    filled = x.copy()
    filled[is_outlier] = np.interp(np.flatnonzero(is_outlier), np.flatnonzero(~is_outlier), x[~is_outlier])
    return filled, is_outlier


def load_clean_lfp(fpath):
    """Repaired trace in uV -> fs_down -> notch -> wavelet denoise -> detrend -> outlier
    fill (-> median filter -> Savitzky-Golay, if APPLY_SMOOTHING). Sample i of the
    returned trace is at start_us + i * 1e6 / fs_down.

    Also returns the time (s from the LFP start) of every raw sample at the ADC limit,
    found before filtering rounds it off, and the fraction of samples outlier-filled."""
    adbitvolts = read_ncs_adbitvolts(fpath)
    lfp, start_us = load_repaired_uv(fpath, adbitvolts)
    adc_limit_uv = np.iinfo(np.int16).max * adbitvolts * 1e6
    saturated_s = np.flatnonzero(np.abs(lfp) >= SATURATION_FRAC * adc_limit_uv) / fs
    lfp = signal.resample_poly(lfp, fs_down, fs)
    lfp = notch_filter(lfp, fs_down, LINE_HARMONICS, NOTCH_Q)
    lfp = denoise_wavelet(lfp)
    lfp = signal.detrend(lfp, type=DETREND_TYPE)
    lfp, is_outlier = fill_outliers(lfp)
    if APPLY_SMOOTHING:
        lfp = signal.medfilt(lfp, MEDFILT_KERNEL)
        lfp = savgol_filter(lfp, SGOLAY_WINDOW, SGOLAY_POLY)
    return lfp, start_us, saturated_s, is_outlier.mean()


def find_position_file(ncs_path):
    """Return the tracking .csv that lives alongside `ncs_path` (same folder).
    Returns None if none is found; if more than one is present, the first
    (alphabetically) is used."""
    folder = os.path.dirname(ncs_path)
    candidates = sorted(fn for fn in os.listdir(folder) if fn.lower().endswith('.csv'))
    if not candidates:
        return None
    return os.path.join(folder, candidates[0])


def compute_velocity_from_position(csv_path):
    """Smoothed running speed (cm/s) from a tracking .csv: UNIX timestamp in us
    (col A, same clock as the .ncs timestamps), x and y in cm (cols D, E).
    Frame-to-frame displacement over actual elapsed time, then Savitzky-Golay."""
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


def clean_line_noise_psd(f, Pxx, harmonics, halfwidth=2.0):
    """Interpolate the PSD across each mains harmonic (FOOOF interpolate_spectrum)."""
    nyq    = f[-1]
    ranges = [[h - halfwidth, h + halfwidth]
              for h in harmonics if (h + halfwidth) < nyq]
    if not ranges:
        return f, Pxx
    f_i, P_i = interpolate_spectrum(f, Pxx, ranges)
    return f_i, P_i


# %% ==================== Spectrogram, snippet choice, theta phase ================

def compute_spectrogram(lfp):
    """Spectrogram of the cleaned trace (uV^2/Hz), with the same window as the PSD
    and the mains harmonics interpolated in every time bin. The trace is
    reflect-padded by half a window, so that bins run from the LFP start to its end.

    Returns
    -------
    f, t : ndarray  frequency (Hz) and window-center time (s from the LFP start)
    Sxx  : ndarray  power, shape (len(f), len(t))
    """
    pad = nperseg // 2
    f, t, Sxx = signal.spectrogram(
        np.pad(lfp, pad, mode='reflect'), fs=fs_down, window='hann', nperseg=nperseg,
        noverlap=SPEC_NOVERLAP, detrend='constant', scaling='density', mode='psd')
    for k in range(Sxx.shape[1]):
        _, Sxx[:, k] = clean_line_noise_psd(f, Sxx[:, k], LINE_HARMONICS, INTERP_HALFWIDTH)
    return f, t - pad / fs_down, Sxx


def snippet_has_saturation(start_s, saturated_s):
    """True if a saturated raw sample falls inside the snippet (saturated_s is sorted)."""
    i0, i1 = np.searchsorted(saturated_s, [start_s, start_s + SNIPPET_DUR_S])
    return i1 > i0


def snippet_median_speed(start_s, t_speed, speed):
    """Median speed inside the snippet (t_speed is sorted); NaN if no tracking there."""
    i0, i1 = np.searchsorted(t_speed, [start_s, start_s + SNIPPET_DUR_S])
    return np.median(speed[i0:i1]) if i1 > i0 else np.nan


def snippet_theta_share(start_s, f, t, Sxx):
    """Theta power as a fraction of NORM_BAND power, from the spectrogram windows
    lying wholly inside the snippet. Returns (share of the mean spectrum, lowest
    share of any one window)."""
    half_win = nperseg / fs_down / 2
    in_snip = (t >= start_s + half_win - 1e-9) & (t <= start_s + SNIPPET_DUR_S - half_win + 1e-9)
    P = Sxx[:, in_snip]
    idx_theta = (f >= THETA_BAND[0]) & (f <= THETA_BAND[1])
    idx_norm = (f >= NORM_BAND[0]) & (f <= NORM_BAND[1])
    theta, norm = P[idx_theta].sum(axis=0), P[idx_norm].sum(axis=0)
    return theta.sum() / norm.sum(), (theta / norm).min()


def find_best_snippet(f, t, Sxx, duration_s, saturated_s, t_speed, speed):
    """Start (s) of the highest-scoring snippet (see the config block), its (theta
    share, score), and which criteria it met: the strictest set any snippet can meet."""
    starts = np.arange(0, duration_s - SNIPPET_DUR_S + 1e-9, SNIPPET_STEP_S)
    unsaturated = [s for s in starts if not snippet_has_saturation(s, saturated_s)]
    moving = [s for s in unsaturated
              if snippet_median_speed(s, t_speed, speed) >= SNIPPET_MIN_SPEED_CMS]
    if moving:
        candidates, criteria = moving, 'unsaturated, moving'
    elif unsaturated:
        candidates, criteria = unsaturated, 'unsaturated, NOT moving'
    else:
        candidates, criteria = list(starts), 'SATURATED'
    shares = [snippet_theta_share(s, f, t, Sxx) for s in candidates]
    best = int(np.argmax([score for _share, score in shares]))
    return candidates[best], shares[best], criteria


def theta_and_phase(lfp):
    """Theta-filtered trace (uV) and its Hilbert phase (rad, 0 at a peak, +/-pi at a
    trough), computed on the whole session so that the snippet has no filter edges."""
    sos = signal.butter(THETA_FILTER_ORDER, THETA_BAND, btype='band', fs=fs_down, output='sos')
    theta = signal.sosfiltfilt(sos, lfp)
    return theta, np.angle(signal.hilbert(theta))


# %% ==================== Per-file processing ====================================

# One dict per .ncs file seen by process_animal() (processed or skipped);
# written to the 'processed_files' sheet of the run-metadata workbook.
RUN_FILE_LOG = []


def evaluate_file(fpath, velocity_cache, snippet_start=None):
    """Clean one .ncs file and find its snippet. Returns everything the figure needs."""
    pos_path = find_position_file(fpath)
    if pos_path is None:
        raise ValueError('no tracking .csv next to this file.')
    if pos_path not in velocity_cache:
        velocity_cache[pos_path] = compute_velocity_from_position(pos_path)
    time_us, speed = velocity_cache[pos_path]

    lfp, start_us, saturated_s, outlier_frac = load_clean_lfp(fpath)
    duration_s = len(lfp) / fs_down
    if duration_s < SNIPPET_DUR_S:
        raise ValueError(f'Trace is shorter than {SNIPPET_DUR_S:g} s.')

    f, t, Sxx = compute_spectrogram(lfp)

    t_speed = (time_us - start_us) / 1e6
    in_lfp = (t_speed >= 0) & (t_speed <= duration_s)
    t_speed, speed = t_speed[in_lfp], speed[in_lfp]
    if t_speed.size < 2:
        raise ValueError('tracking does not overlap the LFP.')

    if snippet_start is None:
        snippet_start, (share, score), criteria = find_best_snippet(
            f, t, Sxx, duration_s, saturated_s, t_speed, speed)
    else:
        (share, score), criteria = snippet_theta_share(snippet_start, f, t, Sxx), 'set by hand'

    return {
        'lfp': lfp, 'f': f, 't': t, 'Sxx': Sxx, 't_speed': t_speed, 'speed': speed,
        'duration_s': duration_s, 'snippet_start': snippet_start,
        'theta_share': share, 'score': score, 'criteria': criteria,
        'snippet_saturated': snippet_has_saturation(snippet_start, saturated_s),
        'saturated_s_total': saturated_s.size / fs, 'outlier_frac': outlier_frac,
        'tracking_csv': pos_path,
    }


def process_animal(label, folder):
    """One figure per .ncs file under `folder`, saved next to that file. A file that
    fails is logged as SKIP and the run carries on."""
    ncs_files = sorted(os.path.join(root, fname)
                       for root, _dirs, files in os.walk(folder)
                       for fname in files if fname.endswith('.ncs'))
    print(f'Found {len(ncs_files)} .ncs files in {folder}')

    only_files = {os.path.normpath(p) for p in ONLY_FILES}
    snippet_starts = {os.path.normpath(p): s for p, s in SNIPPET_START.items()}
    velocity_cache = {}   # {csv_path: (time_us, speed)} -- shared across .ncs in a session
    for fpath in ncs_files:
        rel = os.path.relpath(fpath, folder)
        key = os.path.join(label, rel)
        if only_files and key not in only_files:
            continue
        try:
            res = evaluate_file(fpath, velocity_cache, snippet_starts.get(key))
            plot_file_figure(label, rel, res, os.path.join(os.path.dirname(fpath), FIGURE_SUBFOLDER))
        except Exception as e:
            print(f'  SKIP: {rel} -- {e}')
            RUN_FILE_LOG.append({'animal': label, 'status': 'SKIP', 'file': rel,
                                 'full_path': fpath, 'skip_reason': str(e)})
            continue
        finally:
            plt.close('all')
        print(f"  OK: {rel}  [snippet {res['snippet_start']:.1f} s ({res['criteria']}), "
              f"score {res['score']:.3f}; {100 * res['outlier_frac']:.2f}% samples outlier-filled]")
        RUN_FILE_LOG.append({
            'animal':              label,
            'status':              'OK',
            'file':                rel,
            'full_path':           fpath,
            'tracking_csv':        res['tracking_csv'],
            'duration_s':          res['duration_s'],
            'outlier_filled_frac': res['outlier_frac'],
            'saturated_s_total':   res['saturated_s_total'],
            'snippet_start_s':     res['snippet_start'],
            'snippet_criteria':    res['criteria'],
            'snippet_saturated':   res['snippet_saturated'],
            'snippet_theta_share': res['theta_share'],
            'snippet_score':       res['score'],
            'skip_reason':         '',
        })


# %% ==================== Figure ===================================================

def _plot_spectrogram_on_ax(ax, cax, res, freq_range, title):
    """One spectrogram panel. Color limits come from this panel's frequency range."""
    in_range = (res['f'] >= freq_range[0]) & (res['f'] <= freq_range[1])
    log_power = np.log10(res['Sxx'][in_range])
    vmin, vmax = np.nanpercentile(log_power, SPEC_PERCENTILES)
    mesh = ax.pcolormesh(res['t'], res['f'][in_range], log_power, cmap=SPEC_CMAP,
                         vmin=vmin, vmax=vmax, shading='nearest', rasterized=True)
    ax.set_xlim(0, res['duration_s'])
    ax.set_ylim(freq_range)
    ax.set_ylabel('Frequency (Hz)', fontsize=AX_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    for edge in (res['snippet_start'], res['snippet_start'] + SNIPPET_DUR_S):
        ax.axvline(edge, color='white', ls='--', lw=0.8)
    _fmt(ax)
    cbar = plt.colorbar(mesh, cax=cax)
    cbar.set_label('log10 power (\u00b5V\u00b2/Hz)', fontsize=AX_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)


def plot_file_figure(label, rel, res, out_dir):
    """Speed, LFP snippet with Hilbert theta phase, 1-100 Hz and 1-20 Hz spectrograms,
    saved as PNG + SVG in `out_dir`."""
    color = animal_color(label)
    t0, t1 = res['snippet_start'], res['snippet_start'] + SNIPPET_DUR_S

    i0, i1 = int(round(t0 * fs_down)), int(round(t1 * fs_down))
    t_snip = np.arange(i0, i1) / fs_down
    lfp_snip = res['lfp'][i0:i1]
    theta, phase = theta_and_phase(res['lfp'])
    theta_snip, phase_snip = theta[i0:i1], phase[i0:i1]

    median_speed = snippet_median_speed(t0, res['t_speed'], res['speed'])
    sigma_samples = SPEED_PLOT_SIGMA_S / np.median(np.diff(res['t_speed']))
    speed_plot = gaussian_filter1d(res['speed'], sigma_samples)

    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(4, 2, width_ratios=[1, 0.02], height_ratios=[0.8, 1.2, 1.2, 1.2],
                           hspace=0.62, wspace=0.03)
    ax_speed = fig.add_subplot(gs[0, 0])
    ax_lfp   = fig.add_subplot(gs[1, 0])
    ax_100   = fig.add_subplot(gs[2, 0])
    ax_20    = fig.add_subplot(gs[3, 0], sharex=ax_100)

    # A: speed
    ax_speed.plot(res['t_speed'], speed_plot, lw=1.0, color=color)
    if SPEED_SPAN == 'session':
        ax_speed.set_xlim(0, res['duration_s'])
        ax_speed.axvspan(t0, t1, color='gray', alpha=0.25, zorder=0)
    else:
        ax_speed.set_xlim(t0, t1)
    ax_speed.set_ylim(bottom=0)
    ax_speed.set_xlabel('Time (s)', fontsize=AX_LABEL_FONTSIZE)
    ax_speed.set_ylabel('Speed (cm/s)', fontsize=AX_LABEL_FONTSIZE)
    ax_speed.set_title('A  Smoothed running speed', fontsize=10)
    ax_speed.spines[['top', 'right']].set_visible(False)
    _fmt(ax_speed)

    # B: noise-reduced LFP + theta colored by Hilbert phase
    ax_lfp.plot(t_snip, lfp_snip, lw=0.9, color=color, label='LFP (noise-reduced)')
    # empty line: the legend entry for the theta trace, which has one color per segment
    ax_lfp.plot([], [], lw=2.0, color='0.2',
                label=f'Theta ({THETA_BAND[0]:g}-{THETA_BAND[1]:g} Hz), colored by phase')
    points = np.column_stack([t_snip, theta_snip])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    # butt caps: round caps overlap across the short 1 kHz segments, which adds up to opaque
    theta_line = LineCollection(segments, cmap=PHASE_CMAP, norm=Normalize(-np.pi, np.pi),
                                linewidths=2.0, capstyle='butt', zorder=3, alpha=THETA_ALPHA)
    theta_line.set_array(phase_snip[:-1])
    ax_lfp.add_collection(theta_line)
    ax_lfp.set_xlim(t0, t1)
    ax_lfp.set_xlabel('Time (s)', fontsize=AX_LABEL_FONTSIZE)
    ax_lfp.set_ylabel('LFP (\u00b5V)', fontsize=AX_LABEL_FONTSIZE)
    saturated_note = ', SATURATED' if res['snippet_saturated'] else ''
    ax_lfp.set_title(f'B  {SNIPPET_DUR_S:g} s LFP snippet  '
                     f'(median speed {median_speed:.1f} cm/s, '
                     f"theta share of {NORM_BAND[0]:g}-{NORM_BAND[1]:g} Hz power "
                     f"{100 * res['theta_share']:.0f}%{saturated_note})", fontsize=10)
    ax_lfp.spines[['top', 'right']].set_visible(False)
    ax_lfp.legend(fontsize=8, frameon=False, ncol=2, loc='upper right')
    y_max = max(np.abs(lfp_snip).max(), np.abs(theta_snip).max())
    ax_lfp.set_ylim(-1.05 * y_max, 1.35 * y_max)  # headroom for the legend
    _fmt(ax_lfp)

    cax = fig.add_subplot(gs[1, 1])
    cbar = plt.colorbar(theta_line, cax=cax, ticks=[-np.pi, 0, np.pi])
    cbar.ax.set_yticklabels(['-\u03c0', '0', '\u03c0'])
    cbar.set_label('Theta phase (rad)', fontsize=AX_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

    # C, D: spectrograms
    _plot_spectrogram_on_ax(ax_100, fig.add_subplot(gs[2, 1]), res, (1, 100),
                            'C  Spectrogram, whole session, 1-100 Hz')
    _plot_spectrogram_on_ax(ax_20, fig.add_subplot(gs[3, 1]), res, (1, 20),
                            'D  Spectrogram, whole session, 1-20 Hz')
    ax_100.set_yticks([1, 20, 40, 60, 80, 100])
    ax_20.set_yticks([1, 5, 10, 15, 20])
    ax_100.set_xlabel('Time (s)', fontsize=AX_LABEL_FONTSIZE)
    ax_20.set_xlabel('Time (s)', fontsize=AX_LABEL_FONTSIZE)

    fig.suptitle(f'{label}: {rel}', fontsize=11, y=0.95)

    os.makedirs(out_dir, exist_ok=True)
    stem = re.sub(r'[\\/]+', '_', os.path.splitext(rel)[0])
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(out_dir, f'{label}_{stem}_speed_LFP_spectrogram.{ext}'),
                    dpi=300, bbox_inches='tight')
    return fig


# %% ==================== Run metadata (from the PSD code) =========================

def _script_path():
    """Absolute path of this script (None when run from a notebook / bare console)."""
    p = globals().get('__file__')
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


def _package_versions(pkgs=('numpy', 'scipy', 'pandas', 'matplotlib', 'fooof', 'PyWavelets', 'openpyxl')):
    versions = {}
    for p in pkgs:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = 'not installed'
    return versions


def capture_code_provenance(run_start, snapshot_dir=None):
    """Identify the code version and environment for this run. If `snapshot_dir`
    is given, a copy of the script is saved there so the exact code can be
    recovered later. Call this at the START of the run."""
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
    """{parameter: value} -- every analysis setting in the config block."""
    names = ['ROOT_DIR', 'OUTPUT_DIR', 'FIGURE_SUBFOLDER', 'ANIMALS_TO_INCLUDE', 'ONLY_FILES',
             'SNIPPET_START', 'SNIPPET_DUR_S', 'SNIPPET_STEP_S', 'SNIPPET_MIN_SPEED_CMS',
             'SATURATION_FRAC', 'SPEED_SPAN', 'SPEED_PLOT_SIGMA_S', 'fs', 'fs_down',
             'nperseg', 'ADBITVOLTS_NO_HEADER', 'NORM_BAND', 'THETA_BAND',
             'SPEED_SMOOTH_WINDOW', 'SPEED_SMOOTH_POLY',
             'LINE_HARMONICS', 'NOTCH_Q', 'INTERP_HALFWIDTH', 'WAVELET_NAME', 'WAVELET_LEVEL',
             'DETREND_TYPE', 'OUTLIER_WINDOW', 'OUTLIER_THRESH', 'APPLY_SMOOTHING',
             'MEDFILT_KERNEL', 'SGOLAY_WINDOW', 'SGOLAY_POLY', 'THETA_FILTER_ORDER',
             'SPEC_NOVERLAP', 'SPEC_CMAP', 'SPEC_PERCENTILES', 'PHASE_CMAP', 'THETA_ALPHA',
             'ANIMAL_COLORS']
    return {name: globals()[name] for name in names}


def save_run_metadata(out_path, provenance, run_status='complete'):
    """Write the run metadata to a multi-sheet .xlsx workbook:
      - 'code_provenance' -- script identity, git state, run time, package versions
      - 'settings'        -- every analysis setting
      - 'processed_files' -- every .ncs seen, with its snippet or skip reason

    Requires the `openpyxl` package.
    """
    prov = {'run_status': run_status, **provenance,
            'metadata_written': datetime.datetime.now().isoformat(timespec='seconds')}
    sheets = {
        'code_provenance': pd.DataFrame({'item': list(prov.keys()),
                                         'value': [str(v) for v in prov.values()]}),
        'settings':        pd.DataFrame([{'parameter': k, 'value': str(v)}
                                         for k, v in _analysis_settings().items()]),
        'processed_files': pd.DataFrame(RUN_FILE_LOG),
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


# %% ==================== MAIN PIPELINE ============================================

RUN_START = datetime.datetime.now()
RUN_FILE_LOG.clear()
RUN_METADATA_PATH = os.path.join(OUTPUT_DIR, 'run_metadata.xlsx')
PROVENANCE = capture_code_provenance(
    RUN_START, snapshot_dir=os.path.join(OUTPUT_DIR, 'code_snapshot'))

for label, folder in ANIMALS.items():
    print(f"=== Processing {label} ===")
    process_animal(label, folder)
    save_run_metadata(RUN_METADATA_PATH, PROVENANCE, run_status='incomplete')

save_run_metadata(RUN_METADATA_PATH, PROVENANCE, run_status='complete')
