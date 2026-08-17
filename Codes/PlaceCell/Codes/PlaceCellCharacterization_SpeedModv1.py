# -*- coding: utf-8 -*-
"""
Batch shuffling analysis – V4 

Parameters to edit (line 103-114):
root_folder  = r'F:/Check/1059_Nest_Day10'
output_excel = r'F:/Check/1059_Nest_Day10/sir_shuff.xlsx'

fps            = 30           # tracking frame rate (Hz)
target_bin_cm  = 2.0          # bin size in cm
arena_width_cm = 60.0         # physical arena width in cm
min_occ_s      = 1.0          # exclude bins with < 1 s occupancy
MAX_GAP_US     = 50_000       # max spike–position gap in µs (50 ms)
N_BOOTSTRAP    = 1000         # circular-shift shuffles for SIR significance

MAX_GPU_UTIL_PCT = 60          # if GPU is available, wait until GPU utilization drops below this percentage before starting each bootstrap to prevent overload (set to 0 to disable waiting)
MAX_WORKERS      = 4
"""

import os
import shutil
import threading
import time
import concurrent.futures
import types
import random
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pandas as pd
from scipy.ndimage import convolve, gaussian_filter1d
from scipy.stats import pearsonr, spearmanr, shapiro, t as _t_dist

# Thread-safe Matplotlib imports for parallel rendering
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


class _Metrics(TypedDict, total=False):
    n_spikes:        int | None
    n_discarded:     int | None
    peak_fr:         float | None
    mean_fr:         float | None
    sir:             float | None
    sparsity:        float | None
    coherence:       float | None
    bootstrap_mean:  float | None
    bootstrap_p95:   float | None
    bootstrap_sig:   bool | None
    theta_modulated: bool | None
    theta_peak_freq: float | None
    speed_score:            float | None
    speed_p_value:          float | None
    speed_beta:             float | None
    speed_f0:               float | None
    speed_modulated:        bool | None
    speed_spearman_r:       float | None
    speed_spearman_p:       float | None
    speed_normality_p:      float | None
    speed_residuals_normal: bool | None
    speed_shuffle_mean:     float | None
    speed_shuffle_p95:      float | None
    speed_shuffle_p:        float | None
    speed_shuffle_sig:      bool | None
    place_cell:      bool | None
    session:         str
    unit:            str
    job_order:       int

# ── GPU availability ──────────────────────────────────────────────────────────

if TYPE_CHECKING:
    import cupy as cp                                          # type: ignore
    from cupyx.scipy.ndimage import convolve as cp_convolve    # type: ignore
    import pynvml                                              # type: ignore

try:
    import cupy as cp                                          # type: ignore[import-untyped]
    from cupyx.scipy.ndimage import convolve as cp_convolve    # type: ignore[import-untyped]
    # Validate that CUDA JIT (nvrtc) actually works before committing to GPU mode
    _t = cp.zeros((3, 3), dtype=cp.float64)
    _k = cp.ones((3, 3), dtype=cp.float64) / 9.0
    cp_convolve(_t, _k, mode='constant')
    del _t, _k
    _GPU = True
    print("CuPy detected – GPU (CUDA) acceleration enabled.")
except ImportError:
    cp          = types.SimpleNamespace()                      # type: ignore[assignment]
    # Assign dummy lambda to prevent "None cannot be called" Pylance errors
    cp_convolve = lambda *args, **kwargs: None                 # type: ignore[assignment]
    _GPU = False
    print("CuPy not found – running on CPU (install cupy-cuda12x to enable GPU).")
except Exception as _gpu_err:
    # CuPy imported but CUDA JIT unavailable (e.g. missing nvrtc*.dll)
    cp          = types.SimpleNamespace()                      # type: ignore[assignment]
    cp_convolve = lambda *args, **kwargs: None                 # type: ignore[assignment]
    _GPU = False
    print(f"CuPy found but GPU JIT unavailable ({_gpu_err}) – falling back to CPU.")

try:
    import pynvml                                              # type: ignore[import-untyped]
    pynvml.nvmlInit()
    _NVML        = True
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    pynvml       = types.SimpleNamespace()                     # type: ignore[assignment]
    _nvml_handle = None
    _NVML        = False


def _gpu_util_pct() -> int:
    if not _NVML:
        return 0
    try:
        return int(pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle).gpu)
    except Exception:
        return 0


# ── Configuration ─────────────────────────────────────────────────────────────

root_folder  = r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True'
output_excel = r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True/wut.xlsx'

# Destination for .ntt + tracking files of confirmed place cells (folder pattern
# replicated from the animal-ID folder onwards, e.g. Fa1059/Open/<session>/...)
Output_PlaceTrue = r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True_v2'

fps            = 30           # tracking frame rate (Hz)
target_bin_cm  = 2.0          # bin size in cm
arena_width_cm = 80.0         # physical arena width in cm
min_occ_s      = 1.0          # exclude bins with < 1 s occupancy
MAX_GAP_US     = 50_000       # max spike–position gap in µs (50 ms)
N_BOOTSTRAP    = 1000         # circular-shift shuffles for SIR significance

AUTOCORR_WINDOW_MS  = 500.0   # autocorrelogram half-window (ms)
AUTOCORR_BIN_MS     = 5.0     # autocorrelogram bin size (ms)
THETA_POWER_THRESH  = 2.0     # theta peak must exceed N× mean spectrum power

SPEED_MIN_CMS       = 2.0     # lowest speed (cm/s) included in speed-modulation analysis
SPEED_MAX_CMS       = 60.0   # highest speed (cm/s) included in speed-modulation analysis
SPEED_BIN_CMS       = 4.0 #2.0     # width of each speed bin (cm/s)
# Gaussian smoothing sigma (s), applied SYMMETRICALLY to both the running-speed
# trace and the instantaneous firing-rate trace before binning/regression (see
# _compute_speed_modulation). 100-300 ms is the range used in the hippocampal-
# formation speed-tuning literature (e.g. Kropff et al. 2015, "Speed cells in
# the medial entorhinal cortex", Nature -- ~250 ms Gaussian sigma on both speed
# and rate); 300 ms sits within that range.
SPEED_SMOOTH_S      = 0.3 #0.08
SPEED_MIN_BIN_FRAC  = 0.002   # discard speed bins holding < this fraction of samples
SPEED_N_SHUFFLE          = 1000   # circular time-shifts for the speed-modulation shuffle test
SPEED_SHUFFLE_MARGIN_S   = 20.0   # min |shift| (s) between real and shuffled alignment (matches SIR bootstrap convention)

MAX_GPU_UTIL_PCT = 60
MAX_WORKERS      = 4

# 'pixel' or 'cm' – set interactively at startup (see __main__ below).
# 'pixel' : tracking file has 'x'/'y'/'time' columns in pixels, converted to cm.
# 'cm'    : tracking file is a .csv with time in column A, x (cm) in column D,
#           y (cm) in column E – used directly, no pixel→cm conversion.
COORD_UNITS = 'pixel'

_gpu_semaphore = threading.Semaphore(2)

# 3 × 3 triangular (bilinear) smoothing kernel
_TRIANGULAR_KERNEL = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], dtype=np.float64) / 16.0

ntt_dtype = np.dtype([
    ('timestamp',   '<u8'),
    ('sc_number',   '<u4'),
    ('cell_number', '<u4'),
    ('params',      '<u4', (8,)),
    ('waveforms',   '<i2', (32, 4)),
])

ANIMAL_NAMES = ('Fa1059', 'Fa23BD', 'Fa8477', 'Fa5384')


def _animal_relpath(dirpath: str) -> str | None:
    """Return the portion of `dirpath` starting at the animal-ID folder
    (Fa1059 / Fa23BD / Fa8477 / Fa5384), or None if no such folder is found.
    """
    parts = os.path.normpath(dirpath).split(os.sep)
    for i, part in enumerate(parts):
        if part in ANIMAL_NAMES:
            return os.path.join(*parts[i:])
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _benjamini_hochberg(pvals: 'pd.Series') -> 'pd.Series':
    """Benjamini-Hochberg FDR-adjusted p-values. NaN entries (units where the
    speed-modulation test could not be run) are passed through as NaN and
    excluded from the correction (they're not part of the family of tests).
    """
    pvals = pd.Series(pvals, dtype=float)
    valid = pvals.notna()
    adj   = pd.Series(np.nan, index=pvals.index, dtype=float)
    m     = int(valid.sum())
    if m == 0:
        return adj

    idx_valid  = pvals[valid].sort_values().index
    ranked_p   = pvals.loc[idx_valid].to_numpy()
    ranks      = np.arange(1, m + 1)
    raw_adj    = ranked_p * m / ranks
    # Enforce monotonicity: adjusted p-values must not decrease as rank decreases.
    monotone_adj = np.minimum.accumulate(raw_adj[::-1])[::-1]
    monotone_adj = np.clip(monotone_adj, 0.0, 1.0)
    adj.loc[idx_valid] = monotone_adj
    return adj


def _wait_for_gpu_slot(poll_interval: float = 0.5):
    if not _GPU or not _NVML:
        return
    while _gpu_util_pct() >= MAX_GPU_UTIL_PCT:
        time.sleep(poll_interval)


def _triangular_smooth(fr_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Apply 3×3 triangular smoothing restricted to valid bins.
    Edge effects corrected by dividing the convolved rates by the convolved mask.
    """
    fr_in = np.where(valid_mask, fr_map, 0.0)
    mask_in = valid_mask.astype(np.float64)
    
    if _GPU:
        fr_gpu   = cp.asarray(fr_in, dtype=cp.float64)
        mask_gpu = cp.asarray(mask_in, dtype=cp.float64)
        kern_gpu = cp.asarray(_TRIANGULAR_KERNEL, dtype=cp.float64)
        _wait_for_gpu_slot()
        _gpu_semaphore.acquire()
        try:
            smoothed_fr = cp.asnumpy(
                cp_convolve(fr_gpu, kern_gpu, mode='constant', cval=0.0)
            )
            smoothed_weights = cp.asnumpy(
                cp_convolve(mask_gpu, kern_gpu, mode='constant', cval=0.0)
            )
        finally:
            _gpu_semaphore.release()
    else:
        smoothed_fr = convolve(fr_in, _TRIANGULAR_KERNEL, mode='constant', cval=0.0)
        smoothed_weights = convolve(mask_in, _TRIANGULAR_KERNEL, mode='constant', cval=0.0)
    
    # Correct edge effects by normalizing by gathered weights
    smoothed = np.zeros_like(smoothed_fr)
    valid_weights = smoothed_weights > 0
    smoothed[valid_weights] = smoothed_fr[valid_weights] / smoothed_weights[valid_weights]
    
    smoothed[~valid_mask] = 0.0
    return smoothed


# ── Theta modulation ──────────────────────────────────────────────────────────

def _compute_theta_modulation(
    spike_ts_us: np.ndarray,
    window_ms:   float = AUTOCORR_WINDOW_MS,
    bin_ms:      float = AUTOCORR_BIN_MS,
    thresh:      float = THETA_POWER_THRESH,
) -> tuple[bool, float]:
    """FFT-based theta modulation test on the spike autocorrelogram.

    Mirrors the MATLAB logic:
      1. Build a ±window_ms autocorrelogram with bin_ms resolution.
      2. One-sided power spectrum via FFT.
      3. theta_modulated = peak_power(3–7 Hz) > thresh × mean(full spectrum).

    Returns (theta_modulated, dominant_freq_in_theta_band_hz).
    spike_ts_us must be in microseconds (raw .ntt timestamps).
    """
    if len(spike_ts_us) < 10:
        return False, float('nan')

    spike_ms = np.sort(spike_ts_us) * 1e-3          # µs → ms
    n_bins   = int(round(2.0 * window_ms / bin_ms))
    autocorr = np.zeros(n_bins, dtype=np.float64)

    for i in range(len(spike_ms)):
        t_ref = spike_ms[i]
        lo    = int(np.searchsorted(spike_ms, t_ref - window_ms, side='left'))
        hi    = int(np.searchsorted(spike_ms, t_ref + window_ms, side='right'))
        diffs = spike_ms[lo:hi] - t_ref
        diffs = diffs[diffs != 0.0]
        if len(diffs) == 0:
            continue
        bin_idx = ((diffs + window_ms) / bin_ms).astype(int)
        np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)
        np.add.at(autocorr, bin_idx, 1)

    # One-sided power spectrum
    fs      = 1000.0 / bin_ms                       # sampling freq in Hz
    N       = len(autocorr)
    f_axis  = np.arange(N, dtype=float) * (fs / N)
    half    = N // 2
    f_axis  = f_axis[:half]
    power   = np.abs(np.fft.fft(autocorr)[:half]) ** 2

    # Peak in theta band (3–7 Hz)
    theta_mask = (f_axis >= 3.0) & (f_axis <= 7.0)
    if not theta_mask.any():
        return False, float('nan')

    theta_power = power[theta_mask]
    peak_power  = float(theta_power.max())
    peak_freq   = float(f_axis[theta_mask][np.argmax(theta_power)])

    theta_modulated = peak_power > thresh * float(power.mean())
    return theta_modulated, round(peak_freq, 2)


# ── Speed modulation ──────────────────────────────────────────────────────────

def _gaussian_smooth_track(arr: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Gaussian-smooth `arr` (NaN-safe): NaNs are zero-filled before filtering
    (so they don't smear through the kernel support) and restored afterwards.
    """
    finite = np.isfinite(arr)
    smoothed = gaussian_filter1d(np.where(finite, arr, 0.0),
                                  sigma=sigma_samples, mode='nearest')
    smoothed[~finite] = np.nan
    return smoothed


def _bin_rate_by_speed(speed_valid: np.ndarray, rate_valid: np.ndarray,
                        speed_bins: np.ndarray, min_bin_frac: float):
    """Bin (speed_valid, rate_valid) sample pairs into `speed_bins`-wide bins.

    Returns (bin_centres, mean_rate, n_per_bin) where mean_rate is NaN for
    empty bins and for bins holding < min_bin_frac of all in-range samples.
    """
    n_bins      = len(speed_bins) - 1
    bin_centres = 0.5 * (speed_bins[:-1] + speed_bins[1:])
    bin_idx     = np.clip(np.digitize(speed_valid, speed_bins) - 1, 0, n_bins - 1)

    n_per_bin = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
    sums      = np.bincount(bin_idx, weights=rate_valid, minlength=n_bins)
    mean_rate = np.full(n_bins, np.nan)
    nonzero   = n_per_bin > 0
    mean_rate[nonzero] = sums[nonzero] / n_per_bin[nonzero]

    total_pts = n_per_bin.sum()
    if total_pts > 0:
        mean_rate[(n_per_bin / total_pts) < min_bin_frac] = np.nan

    return bin_centres, mean_rate, n_per_bin.astype(int)


def _weighted_linregress(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    """Weighted least-squares fit y = slope*x + intercept, weighting each
    point by `w` (here: n_per_bin, since a bin mean's variance scales as
    ~1/n_per_bin -- a bin averaged from 2 samples should not carry the same
    weight as one averaged from 2000).

    Returns slope, intercept, weighted Pearson r, and a two-sided Wald
    t-test p-value for H0: slope == 0 (df = n_points - 2).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n = len(x)

    sw     = w.sum()
    x_mean = np.sum(w * x) / sw
    y_mean = np.sum(w * y) / sw
    sxx    = np.sum(w * (x - x_mean) ** 2)
    sxy    = np.sum(w * (x - x_mean) * (y - y_mean))
    syy    = np.sum(w * (y - y_mean) ** 2)

    slope     = sxy / sxx if sxx > 0 else float('nan')
    intercept = y_mean - slope * x_mean if np.isfinite(slope) else float('nan')
    r         = sxy / np.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float('nan')

    resid = y - (slope * x + intercept) if np.isfinite(slope) else np.full(n, np.nan)
    df    = n - 2
    if df > 0 and sxx > 0 and np.isfinite(slope):
        mse      = np.sum(w * resid ** 2) / df
        se_slope = np.sqrt(mse / sxx)
        tstat    = slope / se_slope if se_slope > 0 else float('inf')
        pvalue   = float(2.0 * _t_dist.sf(np.abs(tstat), df))
    else:
        pvalue = float('nan')

    return {'slope': float(slope), 'intercept': float(intercept),
            'r': float(r), 'pvalue': pvalue, 'resid': resid}


def _speed_modulation_shuffle(
    speed:               np.ndarray,
    fr_smooth:           np.ndarray,
    real_r:              float,
    min_speed_cms:       float,
    max_speed_cms:       float,
    speed_bins:          np.ndarray,
    min_bin_frac:        float,
    pos_sample_rate_hz:  float,
    n_shuffle:           int = SPEED_N_SHUFFLE,
    margin_s:            float = SPEED_SHUFFLE_MARGIN_S,
) -> dict:
    """Circular time-shift shuffle test for speed modulation (analogous to the
    SIR location-shuffle bootstrap, _run_bootstrap/_sir_from_spikes_locshuf).

    The firing-rate trace is circularly rolled by a random offset (leaving
    the speed trace, and hence its autocorrelation structure, untouched),
    the whole binning + weighted-regression pipeline is re-run, and the null
    distribution of |r| is compared against the real |r|. This breaks the
    time-locked spike-speed relationship without assuming any particular
    residual distribution, addressing the autocorrelation/non-Gaussian-
    residual concerns that make the parametric linregress-style p-value
    (from _weighted_linregress) potentially anti-conservative.
    """
    n = len(speed)
    if n < 3 or not np.isfinite(real_r):
        return {'speed_shuffle_mean': float('nan'), 'speed_shuffle_p95': float('nan'),
                'speed_shuffle_p': float('nan'), 'speed_shuffle_sig': None}

    margin_frames = int(margin_s * pos_sample_rate_hz)
    if n <= 2 * margin_frames:
        # Recording too short for the standard margin; fall back to a quarter
        # of the track so shuffles still avoid near-trivial (small) shifts.
        margin_frames = max(1, n // 4)
    if n <= margin_frames:
        return {'speed_shuffle_mean': float('nan'), 'speed_shuffle_p95': float('nan'),
                'speed_shuffle_p': float('nan'), 'speed_shuffle_sig': None}

    null_r = np.full(n_shuffle, np.nan)
    for i in range(n_shuffle):
        shift        = random.randint(margin_frames, n - margin_frames)
        fr_shifted   = np.roll(fr_smooth, shift)
        in_range_shf = ((speed > min_speed_cms) & (speed < max_speed_cms) &
                         np.isfinite(fr_shifted))
        if in_range_shf.sum() < 3:
            continue
        _, mean_rate_shf, n_per_bin_shf = _bin_rate_by_speed(
            speed[in_range_shf], fr_shifted[in_range_shf], speed_bins, min_bin_frac)
        fit_sel_shf = np.isfinite(mean_rate_shf)
        if fit_sel_shf.sum() < 3:
            continue
        centres_shf = 0.5 * (speed_bins[:-1] + speed_bins[1:])
        wls_shf = _weighted_linregress(centres_shf[fit_sel_shf], mean_rate_shf[fit_sel_shf],
                                        n_per_bin_shf[fit_sel_shf].astype(np.float64))
        null_r[i] = wls_shf['r']

    null_r = null_r[np.isfinite(null_r)]
    if len(null_r) == 0:
        return {'speed_shuffle_mean': float('nan'), 'speed_shuffle_p95': float('nan'),
                'speed_shuffle_p': float('nan'), 'speed_shuffle_sig': None}

    # Two-sided permutation p-value (add-one smoothing avoids p == 0).
    p_shuffle = float((1 + np.sum(np.abs(null_r) >= np.abs(real_r))) / (1 + len(null_r)))

    return {'speed_shuffle_mean': round(float(np.mean(null_r)), 4),
            'speed_shuffle_p95':  round(float(np.percentile(np.abs(null_r), 95)), 4),
            'speed_shuffle_p':    round(p_shuffle, 4),
            'speed_shuffle_sig':  bool(p_shuffle < 0.05)}


def _compute_speed_modulation(
    x_cm:               np.ndarray,
    y_cm:               np.ndarray,
    t_us:               np.ndarray,
    spike_ts_us:        np.ndarray,
    pos_sample_rate_hz: float,
    min_speed_cms:      float = SPEED_MIN_CMS,
    max_speed_cms:      float = SPEED_MAX_CMS,
    speed_bin_cms:      float = SPEED_BIN_CMS,
    smooth_window_s:    float = SPEED_SMOOTH_S,
    min_bin_frac:       float = SPEED_MIN_BIN_FRAC,
    n_shuffle:          int = SPEED_N_SHUFFLE,
    ntt_path:           str | None = None,
    label:              str = '',
) -> dict:
    """Relates firing rate to running speed (ports MATLAB `speed_firing_runita`).

      1. Per-frame running speed (cm/s) from consecutive positions, using the
         actual inter-frame interval rather than an assumed fixed rate.
      2. Instantaneous firing rate on the same frame base (spike count per
         inter-frame interval / interval duration). Both speed and rate are
         then Gaussian-smoothed with the SAME kernel (symmetric smoothing --
         smoothing only the rate biases the fit, see SPEED_SMOOTH_S comment).
      3. Restrict to samples with min_speed_cms < speed < max_speed_cms.
      4. Bin firing rate by speed (speed_bin_cms-wide bins), drop bins with
         < min_bin_frac of the samples, and fit rate = beta*speed + f0 to the
         binned means via WEIGHTED least squares (weight = n_per_bin), plus a
         circular-shift shuffle test and a Spearman/normality check (beta/f0
         naming kept from the MATLAB source).

    x_cm, y_cm, t_us must be same-length position tracks (t_us in µs, sorted).
    spike_ts_us must be in the same time base (µs).
    Returns speed_score (weighted Pearson r), speed_p_value (parametric),
    speed_beta (slope), speed_f0 (intercept), speed_modulated (p < 0.05),
    speed_spearman_r/p, speed_normality_p, speed_residuals_normal,
    speed_shuffle_mean/p95/p, speed_shuffle_sig.
    """
    result = {'speed_score': float('nan'), 'speed_p_value': float('nan'),
              'speed_beta': float('nan'), 'speed_f0': float('nan'),
              'speed_modulated': None,
              'speed_spearman_r': float('nan'), 'speed_spearman_p': float('nan'),
              'speed_normality_p': float('nan'), 'speed_residuals_normal': None,
              'speed_shuffle_mean': float('nan'), 'speed_shuffle_p95': float('nan'),
              'speed_shuffle_p': float('nan'), 'speed_shuffle_sig': None}

    n = len(t_us)
    if n < 3 or len(spike_ts_us) == 0:
        return result

    # ── Per-frame running speed (cm/s) ────────────────────────────────────────
    # There are n position samples, hence n-1 "inter-frame intervals" between
    # them. dt_s is the true duration of each of those intervals (in seconds),
    # computed from the actual timestamps rather than assuming a fixed frame
    # rate — the tracker's frame interval is not perfectly constant.
    dt_s = np.diff(t_us) * 1e-6
    with np.errstate(invalid='ignore', divide='ignore'):
        # Euclidean distance travelled between consecutive frames, divided by
        # how long that step took -> speed (cm/s) for each of the n-1 intervals.
        speed = np.hypot(np.diff(x_cm), np.diff(y_cm)) / dt_s
    speed[dt_s <= 0] = np.nan          # guard against zero/negative dt (duplicate or out-of-order timestamps)
    speed = np.append(speed, speed[-1])              # pad to length n (repeat last value for the final sample)
                                                    #padding at stop or start? 
                                                    # Predictive signals will be masked if assinged to (n-1, n) rather than (n, n+1)

    # ── Instantaneous firing rate on the same frame base ──────────────────────
    # Goal: turn the spike train into one firing-rate value per inter-frame
    # interval, so it lines up 1-to-1 with the `speed` array above.
    #
    # searchsorted(t_us, spike_ts_us, side='right') - 1 finds, for every spike
    # timestamp, the index of the position frame immediately *before* it —
    # i.e. which of the n-1 intervals [t[i], t[i+1]) the spike falls into.
    # side='right' means a spike landing exactly on a frame timestamp is
    # assigned to the interval that *starts* at that timestamp.
    interval_idx = np.searchsorted(t_us, spike_ts_us, side='right') - 1
    # Clip so spikes before the first frame or after the last frame don't
    # produce an out-of-bounds index; they get folded into the first/last interval.
    interval_idx = np.clip(interval_idx, 0, n - 2)
    # 50 ms spike–position gate (MAX_GAP_US, same convention as the nearest-
    # timestamp matching used for the ratemap, see ~line 681): only count a
    # spike toward an interval's rate if it lies within MAX_GAP_US of the
    # nearer of that interval's two bounding position timestamps. This drops
    # spikes recorded during tracking dropouts (large inter-frame gaps) or
    # outside the tracked time range, rather than folding them into whichever
    # interval searchsorted/clip happens to assign them to.
    dist_start  = np.abs(spike_ts_us - t_us[interval_idx])
    dist_end    = np.abs(t_us[interval_idx + 1] - spike_ts_us)
    min_dist    = np.minimum(dist_start, dist_end)
    interval_idx = interval_idx[min_dist <= MAX_GAP_US]
    # Count how many spikes fall in each of the n-1 intervals.
    counts = np.bincount(interval_idx, minlength=n - 1).astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        # Instantaneous rate for interval i = (spikes in interval i) / (duration
        # of interval i). This is a raw, un-smoothed, per-frame firing rate (Hz).
        fr_inst = counts / dt_s
    fr_inst[dt_s <= 0] = np.nan
    fr_inst = np.append(fr_inst, fr_inst[-1])         # pad to length n, same convention as `speed`

    # ── Symmetric Gaussian smoothing of BOTH speed and firing rate ───────────
    # WINDOW SIZE: controlled by `smooth_window_s` (default SPEED_SMOOTH_S,
    # 0.3 s -- within the 100-300 ms range used in hippocampal-formation
    # speed-tuning studies, e.g. Kropff et al. 2015). This is the *standard
    # deviation* (sigma) of the Gaussian kernel, expressed in seconds, and is
    # converted to samples via sigma_samples = smooth_window_s *
    # pos_sample_rate_hz (the position tracking frame rate, e.g. 30 Hz).
    # scipy's gaussian_filter1d truncates the kernel at 4*sigma by default, so
    # the *effective* smoothing window (full kernel support) spans roughly
    # ±4*sigma_samples samples.
    #
    # Both `speed` and `fr_inst` are smoothed with the IDENTICAL kernel. If
    # only the rate were smoothed (as before), the rate trace would gain
    # temporal autocorrelation and reduced residual scatter around the fit
    # that the (unsmoothed) speed trace doesn't share -- an asymmetric bias
    # that can artificially tighten the fit and inflate significance.
    sigma_samples = max(smooth_window_s * pos_sample_rate_hz, 1e-6)
    speed_smooth  = _gaussian_smooth_track(speed,   sigma_samples)
    fr_smooth     = _gaussian_smooth_track(fr_inst, sigma_samples)

    # ── Restrict to the usable speed range ─────────────────────────────────────
    # Very low speeds (near-stationary, e.g. grooming/resting) and very high
    # speeds (tracking artifacts / jumps) are excluded so the fit isn't
    # dominated by outliers or immobility-related firing (e.g. sharp-wave
    # ripples during rest). Only frames with min_speed_cms < speed < max_speed_cms
    # AND a valid (non-NaN) smoothed rate are kept.
    in_range = ((speed_smooth > min_speed_cms) & (speed_smooth < max_speed_cms) &
                np.isfinite(fr_smooth))
    if in_range.sum() < 3:
        return result

    speed_valid = speed_smooth[in_range]
    rate_valid  = fr_smooth[in_range]
    if np.std(speed_valid) == 0 or np.std(rate_valid) == 0:
        return result

    # ── Bin firing rate by speed ────────────────────────────────────────────────
    # Build speed_bin_cms-wide bin edges spanning [min_speed_cms, max_speed_cms]
    # (e.g. 2 cm/s wide bins from 2 to 90 cm/s), giving n_bins bins.
    speed_bins = np.arange(min_speed_cms, max_speed_cms + speed_bin_cms, speed_bin_cms)
    bin_centres, mean_rate, n_per_bin = _bin_rate_by_speed(
        speed_valid, rate_valid, speed_bins, min_bin_frac)

    if n_per_bin.sum() == 0:
        return result

    fit_sel = np.isfinite(mean_rate)
    if fit_sel.sum() < 3:
        return result

    # ── Weighted regression of binned mean firing rate against bin-centre
    # speed: rate = speed_beta * speed + speed_f0, weighted by n_per_bin so a
    # bin averaged from a handful of samples doesn't count as much as one
    # averaged from thousands (unweighted OLS on the raw bin means treats
    # them as equally precise, which they aren't).
    w   = n_per_bin[fit_sel].astype(np.float64)
    wls = _weighted_linregress(bin_centres[fit_sel], mean_rate[fit_sel], w)

    result['speed_score']     = round(wls['r'], 4) if np.isfinite(wls['r']) else float('nan')
    result['speed_p_value']   = round(wls['pvalue'], 4) if np.isfinite(wls['pvalue']) else float('nan')
    result['speed_beta']      = round(wls['slope'], 4) if np.isfinite(wls['slope']) else float('nan')
    result['speed_f0']        = round(wls['intercept'], 4) if np.isfinite(wls['intercept']) else float('nan')
    result['speed_modulated'] = (bool(wls['pvalue'] < 0.05)
                                  if np.isfinite(wls['pvalue']) else None)

    # ── Normality check + Spearman rank correlation ──────────────────────────
    # Shapiro-Wilk on the WLS residuals (n_bins is small, typically ~10-40,
    # well within Shapiro's valid range and more powerful there than KS).
    # Firing-rate data is Poisson-like (variance scales with the mean, more
    # so at low spike counts) and often shows a saturating/non-monotonic
    # rather than strictly linear relation to speed, so normality is not
    # guaranteed even after binning. Spearman rank correlation makes no
    # distributional assumption and captures monotonic (not just linear)
    # relationships, so it's reported alongside Pearson/WLS on every cell
    # rather than only substituted in when normality fails -- that decision
    # is left to downstream analysis, guided by speed_residuals_normal.
    resid = wls['resid']
    if fit_sel.sum() >= 3 and np.all(np.isfinite(resid)) and np.std(resid) > 0:
        try:
            _, shapiro_p = shapiro(resid)
            result['speed_normality_p']      = round(float(shapiro_p), 4)
            result['speed_residuals_normal'] = bool(shapiro_p > 0.05)
        except Exception:
            pass

    rho, p_spearman = spearmanr(bin_centres[fit_sel], mean_rate[fit_sel])
    if np.isfinite(rho):
        result['speed_spearman_r'] = round(float(rho), 4)
        result['speed_spearman_p'] = round(float(p_spearman), 4)

    # ── Circular time-shift shuffle test ──────────────────────────────────────
    # Augments the parametric WLS p-value above with a shuffle-based null
    # distribution (same logic as the SIR location-shuffle bootstrap):
    # circularly roll the firing-rate trace relative to the (fixed) speed
    # trace many times, refit, and see how often |shuffled r| >= |real r|.
    shuf = _speed_modulation_shuffle(
        speed_smooth, fr_smooth, wls['r'],
        min_speed_cms, max_speed_cms, speed_bins, min_bin_frac,
        pos_sample_rate_hz, n_shuffle=n_shuffle)
    result['speed_shuffle_mean'] = shuf['speed_shuffle_mean']
    result['speed_shuffle_p95']  = shuf['speed_shuffle_p95']
    result['speed_shuffle_p']    = shuf['speed_shuffle_p']
    result['speed_shuffle_sig']  = shuf['speed_shuffle_sig']

    # ── speed-vs-rate plot ───────────────────────────────────────────────────
    if ntt_path is not None:
        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        ax.scatter(speed_valid, rate_valid, s=3, color='0.75', alpha=0.4,
                   label='raw samples')
        ax.scatter(bin_centres[fit_sel], mean_rate[fit_sel], s=80, color='black',
                   label='binned mean')

        fit_x = np.array([bin_centres[fit_sel].min(), bin_centres[fit_sel].max()])
        fit_y = wls['slope'] * fit_x + wls['intercept']
        ax.plot(fit_x, fit_y, 'r-', label='WLS fit')

        stats_txt = (f"r (WLS) = {wls['r']:.3f}\n"
                     f"slope = {wls['slope']:.3f}\n"
                     f"intercept = {wls['intercept']:.3f}\n"
                     f"p (parametric) = {wls['pvalue']:.3g}\n"
                     f"p (shuffle) = {shuf['speed_shuffle_p']}\n"
                     f"spearman rho = {result['speed_spearman_r']}, "
                     f"p = {result['speed_spearman_p']}\n"
                     f"resid. normal (p) = {result['speed_normality_p']}")
        ax.text(0.02, 0.98, stats_txt, transform=ax.transAxes,
               va='top', ha='left', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('speed (cm/s)')
        ax.set_ylabel('firing rate (Hz)')
        ax.legend(loc='lower right', fontsize=8)
        fig.tight_layout()

        ntt_name   = os.path.splitext(os.path.basename(ntt_path))[0]
        save_dir   = os.path.join(os.path.dirname(ntt_path), 'speed modulation')
        os.makedirs(save_dir, exist_ok=True)
        lbl_suffix = f'_{label}' if label else ''
        save_path  = os.path.join(save_dir, f'{ntt_name}{lbl_suffix}_speed_modulation.png')
        fig.savefig(save_path, dpi=150)
        print(f'  [SAVED] {save_path}')

    return result


# ── Bootstrap helpers ─────────────────────────────────────────────────────────

def _sir_from_spikes_locshuf(spike_frame_indices: np.ndarray, rnd: int,
                              beh_bx: np.ndarray, beh_by: np.ndarray,
                              occ_map: np.ndarray, valid_mask: np.ndarray,
                              n_bins_x: int, n_bins_y: int) -> float:
    """Compute SIR after circularly shifting the position time series by `rnd` frames.
    Spike-to-frame assignments are kept fixed; only the location at each frame changes.
    Equivalent to MATLAB: locs_rand = [locs(rnd:end); locs(1:rnd-1)]
    """
    n_frames   = len(beh_bx)
    shuf_frame = (spike_frame_indices + rnd) % n_frames

    spike_map = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    np.add.at(spike_map, (beh_bx[shuf_frame], beh_by[shuf_frame]), 1.0)

    fr_raw = np.zeros_like(spike_map)
    np.divide(spike_map, occ_map, out=fr_raw, where=valid_mask)

    fr_smooth   = _triangular_smooth(fr_raw, valid_mask)
    total_occ_s = occ_map[valid_mask].sum()
    pi_flat     = occ_map[valid_mask] / total_occ_s
    ri_flat     = fr_smooth[valid_mask]
    r_mean      = float(np.sum(pi_flat * ri_flat))

    if r_mean <= 0:
        return 0.0
    nonzero = ri_flat > 0
    ratio   = ri_flat[nonzero] / r_mean
    return float(np.sum(pi_flat[nonzero] * ratio * np.log2(ratio)))


def _run_bootstrap(spike_frame_indices: np.ndarray, t: np.ndarray,
                   beh_bx: np.ndarray, beh_by: np.ndarray,
                   occ_map: np.ndarray, valid_mask: np.ndarray,
                   n_bins_x: int, n_bins_y: int,
                   real_sir: float, ntt_path: str,
                   label: str = '') -> dict:
    """Location-shuffling bootstrap (matches MATLAB calcSI_v3_locshuf).

    For each permutation, the position time series is circularly shifted by a
    random number of frames (at least 20 s from either end), while spike-to-frame
    assignments remain unchanged.  This decorrelates spikes from positions without
    altering the animal's occupancy statistics.
    """
    if len(spike_frame_indices) == 0:
        return {'bootstrap_mean': float('nan'),
                'bootstrap_p95':  float('nan'),
                'bootstrap_sig':  False}

    n_frames      = len(t)
    MARGIN_FRAMES = int(20 * fps)   # 20 seconds of frames at either end (matches Fenton reference)

    if n_frames <= 2 * MARGIN_FRAMES:
        return {'bootstrap_mean': float('nan'),
                'bootstrap_p95':  float('nan'),
                'bootstrap_sig':  None}

    sir_i = np.zeros(N_BOOTSTRAP, dtype=np.float64)
    for i in range(N_BOOTSTRAP):
        rnd      = random.randint(MARGIN_FRAMES, n_frames - MARGIN_FRAMES)
        sir_i[i] = _sir_from_spikes_locshuf(spike_frame_indices, rnd,
                                             beh_bx, beh_by,
                                             occ_map, valid_mask,
                                             n_bins_x, n_bins_y)

    bootstrap_mean = float(np.mean(sir_i))
    bootstrap_p95  = float(np.percentile(sir_i, 95))
    bootstrap_sig  = bool(real_sir > bootstrap_p95)

    # ── histogram plot ────────────────────────────────────────────────────────
    # Thread-safe Object-Oriented Figure generation prevents race conditions
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    hist_result = ax.hist(sir_i, 100, color='black')
    counts: np.ndarray = np.asarray(hist_result[0])
    max_count = float(counts.max()) if counts.max() > 0 else 1.0

    box_plot = ax.boxplot(sir_i, whis=[5, 95], vert=False, showfliers=False, # type: ignore
                          positions=[-max_count / 10], widths=max_count / 15)
    ax.plot([real_sir, real_sir], [0, max_count], 'r-.')

    ci95 = float(box_plot['whiskers'][1].get_xdata()[1])
    bs_av = f"Bootstrap mean SIR = {bootstrap_mean:.3f} bits/spike"
    up_ci = f"Upper 95% CI = {ci95:.3f} bits/spike"
    bs_p  = (f"Cell SIR = {real_sir:.3f} bits/spike "
             f"({'p < 0.05' if bootstrap_sig else 'ns'})")
    ax.set_title(bs_av + '\n' + up_ci + '\n' + bs_p, multialignment='center')

    ax.set_yticks([0, max_count * 0.25, max_count * 0.5,
                   max_count * 0.75, max_count])
    ax.set_yticklabels([str(round(v, 1))
                        for v in (0, max_count * 0.25, max_count * 0.5,
                                  max_count * 0.75, max_count)])
    ax.set_ylabel('count')
    ax.set_xlabel('spatial information rate (bits/spike)')
    ax.set_ylim(-max_count / 7, max_count * 1.1)
    fig.tight_layout()

    ntt_name   = os.path.splitext(os.path.basename(ntt_path))[0]
    save_dir   = os.path.join(os.path.dirname(ntt_path), 'shuffling')
    os.makedirs(save_dir, exist_ok=True)
    lbl_suffix = f'_{label}' if label else ''
    save_path  = os.path.join(save_dir, f'{ntt_name}{lbl_suffix}_bootstrapping.png')
    fig.savefig(save_path, dpi=150)
    print(f'  [SAVED] {save_path}')

    return {'bootstrap_mean': round(bootstrap_mean, 4),
            'bootstrap_p95':  round(bootstrap_p95,  4),
            'bootstrap_sig':  bootstrap_sig}


# ── Core metric computation ───────────────────────────────────────────────────

def compute_metrics(csv_path: str, ntt_path: str,
                    arena_width_cm: float, target_bin_cm: float,
                    half: str | None = None) -> tuple:

    # ── 1. Load & clean tracking ──────────────────────────────────────────────
    data = (pd.read_excel(csv_path) if csv_path.lower().endswith('.xlsx')
            else pd.read_csv(csv_path))

    if COORD_UNITS == 'cm':
        # Column A = timestamp, column D = x (cm), column E = y (cm)
        t = np.asarray(data.iloc[:, 0], dtype=float)
        x = np.asarray(data.iloc[:, 3], dtype=float)
        y = np.asarray(data.iloc[:, 4], dtype=float)
    else:
        x = np.asarray(data['x'],    dtype=float)
        y = np.asarray(data['y'],    dtype=float)
        t = np.asarray(data['time'], dtype=float)

    mask = ~np.isin(x, [1, -1])
    x, y, t = x[mask], y[mask], t[mask]

    # Pad the differential arrays with zeros/ones to prevent dropping the final frame
    dx = np.append(np.diff(x), 0)
    dy = np.append(np.diff(y), 0)
    dt = np.append(np.diff(t), 1) # pad with 1 to avoid div-by-zero on last frame
    
    dxy = np.hypot(dx, dy)
    valid_dt = dt > 0
    
    # Pre-allocate speed array to handle duplicate timestamps safely
    speed = np.zeros_like(dxy)
    speed[valid_dt] = dxy[valid_dt] / dt[valid_dt]
    
    keep = np.where(valid_dt & (speed < 0.006))[0]
    x, y, t = x[keep], y[keep], t[keep]

    order = np.argsort(t)
    x, y, t = x[order], y[order], t[order]

    if half is not None:
        mid = len(t) // 2
        if half == 'first':
            x, y, t = x[:mid], y[:mid], t[:mid]
        elif half == 'second':
            x, y, t = x[mid:], y[mid:], t[mid:]

    if len(t) == 0:
         return ({'n_spikes': 0, 'n_discarded': 0, 'peak_fr': 0.0, 'mean_fr': 0.0, 'sir': 0.0}, {})

    # ── 2. Pixel → cm conversion ──────────────────────────────────────────────
    if COORD_UNITS == 'cm':
        # Coordinates are already in cm – just zero the origin for binning.
        x_cm = x - x.min()
        y_cm = y - y.min()
    else:
        x_span = x.max() - x.min()
        y_span = y.max() - y.min()
        px_per_cm = max(x_span, y_span) / arena_width_cm

        x_cm = (x - x.min()) / px_per_cm
        y_cm = (y - y.min()) / px_per_cm

    # ── 3. Bin tracking positions ─────────────────────────────────────────────
    n_bins_x = int(np.ceil(x_cm.max() / target_bin_cm))
    n_bins_y = int(np.ceil(y_cm.max() / target_bin_cm))

    beh_bx = np.clip((x_cm / target_bin_cm).astype(int), 0, n_bins_x - 1)
    beh_by = np.clip((y_cm / target_bin_cm).astype(int), 0, n_bins_y - 1)

    # ── 4. Load spikes & nearest-timestamp assignment (50 ms gate) ────────────
    spike_data = np.memmap(ntt_path, dtype=ntt_dtype, mode='r', offset=16 * 1024)
    spike_data = spike_data[spike_data['cell_number'] != 0]  # drop unsorted/discarded cluster 0
    spike_ts   = np.sort(spike_data['timestamp'].astype(np.float64))

    if half is not None:
        t_lo = t[0]  - MAX_GAP_US
        t_hi = t[-1] + MAX_GAP_US
        spike_ts = spike_ts[(spike_ts >= t_lo) & (spike_ts <= t_hi)]

    idx   = np.searchsorted(t, spike_ts, side='left')
    idx_l = np.clip(idx - 1, 0, len(t) - 1)
    idx_r = np.clip(idx,     0, len(t) - 1)
    dist_l   = np.abs(spike_ts - t[idx_l])
    dist_r   = np.abs(spike_ts - t[idx_r])
    nearest  = np.where(dist_l <= dist_r, idx_l, idx_r)
    min_dist = np.minimum(dist_l, dist_r)

    valid_spike  = min_dist <= MAX_GAP_US
    spike_frame  = nearest[valid_spike]
    n_spikes     = int(valid_spike.sum())
    n_discarded  = int((~valid_spike).sum())

    sp_bx = beh_bx[spike_frame]
    sp_by = beh_by[spike_frame]

    # ── 5. Build occupancy and spike-count maps ────────────────────────────────
    dt_frames        = np.empty(len(t), dtype=np.float64)
    dt_frames[0]     = 1.0 / fps
    raw_dt           = np.diff(t) * 1e-6
    
    # Cap dt_frames to avoid artificial occupancy hotspots when tracking drops
    # E.g., if a gap is > ~2 frames, only credit the standard frame rate to prevent inflation
    max_frame_s      = 2.0 / fps
    dt_frames[1:]    = np.minimum(raw_dt, max_frame_s)

    occ_map   = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    spike_map = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)

    np.add.at(occ_map,   (beh_bx, beh_by), dt_frames)
    np.add.at(spike_map, (sp_bx,  sp_by),  1.0)

    valid_mask = occ_map >= min_occ_s

    # ── 6. Non-smoothed firing rate map ──────────────────────────────────────
    fr_raw = np.zeros_like(occ_map)
    fr_raw[valid_mask] = spike_map[valid_mask] / occ_map[valid_mask]

    # ── 7. Smoothed firing rate map ───────────────────────────────────────────
    fr_smooth = _triangular_smooth(fr_raw, valid_mask)

    # ── 8. Compute metrics ────────────────────────────────────────────────────
    ctx = dict(spike_ts=spike_ts[valid_spike], spike_frame=spike_frame, t=t,
               x_cm=x_cm, y_cm=y_cm,
               beh_bx=beh_bx, beh_by=beh_by,
               occ_map=occ_map, valid_mask=valid_mask,
               n_bins_x=n_bins_x, n_bins_y=n_bins_y)

    if not valid_mask.any():
        return ({'n_spikes': n_spikes, 'n_discarded': n_discarded,
                 'peak_fr': 0.0, 'mean_fr': 0.0, 'sir': 0.0,
                 'sparsity': 0.0, 'coherence': float('nan')}, ctx)

    total_occ_s = occ_map[valid_mask].sum()
    pi_flat     = occ_map[valid_mask] / total_occ_s
    ri_flat     = fr_smooth[valid_mask]
    r_mean      = float(np.sum(pi_flat * ri_flat))

    peak_fr = float(fr_smooth[valid_mask].max())
    mean_fr = r_mean

    sir = 0.0
    if r_mean > 0:
        nonzero = ri_flat > 0
        ratio   = ri_flat[nonzero] / r_mean
        sir     = float(np.sum(pi_flat[nonzero] * ratio * np.log2(ratio)))

    # Sparsity = (Σ pi ri)² / Σ pi ri²   (Skaggs et al. 1996)
    spar_num = float(np.sum(pi_flat * ri_flat))
    spar_den = float(np.sum(pi_flat * ri_flat ** 2))
    sparsity = float((spar_num ** 2) / spar_den) if spar_den > 0 else 0.0

    # Spatial coherence: non-smoothed map vs 8-neighbour mean (Fisher Z)
    valid_idx    = np.argwhere(valid_mask)
    fr_bin_vals  = []
    fr_nbr_means = []
    for bx, by in valid_idx:
        nbr_vals = [
            fr_raw[bx + dx, by + dy]
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if not (dx == 0 and dy == 0)
            and 0 <= bx + dx < n_bins_x
            and 0 <= by + dy < n_bins_y
            and valid_mask[bx + dx, by + dy]
        ]
        if nbr_vals:
            fr_bin_vals.append(fr_raw[bx, by])
            fr_nbr_means.append(float(np.mean(nbr_vals)))

    if len(fr_bin_vals) > 2:
        r_coef, _ = pearsonr(fr_bin_vals, fr_nbr_means)
        r_coef    = float(np.clip(r_coef, -0.9999, 0.9999)) # type: ignore
        coherence = float(0.5 * np.log((1 + r_coef) / (1 - r_coef)))
    else:
        coherence = float('nan')

    metrics = {
        'n_spikes':    n_spikes,
        'n_discarded': n_discarded,
        'peak_fr':     round(peak_fr,   4),
        'mean_fr':     round(mean_fr,   4),
        'sir':         round(sir,       4),
        'sparsity':    round(sparsity,  4),
        'coherence':   round(coherence, 4) if not np.isnan(coherence) else float('nan'),
    }
    return metrics, ctx


# ── Per-job wrapper (called from thread pool) ─────────────────────────────────

_print_lock = threading.Lock()

_NULL_BOOTSTRAP = {'bootstrap_mean': None, 'bootstrap_p95': None, 'bootstrap_sig': None}

def _run_job(args):
    unit_idx, total_units, job_order, dirpath, csv_path, ntt_file = args
    session_name = os.path.relpath(dirpath, root_folder)
    ntt_path     = os.path.join(dirpath, ntt_file)
    pct          = 100 * unit_idx / total_units

    with _print_lock:
        print(f'[{unit_idx}/{total_units}  {pct:.1f}%]  {session_name}  |  {ntt_file}  '
              f'(GPU {_gpu_util_pct()}%)')

    _err_row: dict = {
        'n_spikes': None, 'n_discarded': None,
        'peak_fr':  None, 'mean_fr':     None, 'sir': None,
        'sparsity': None, 'coherence':   None,
        'bootstrap_mean': None, 'bootstrap_p95': None, 'bootstrap_sig': None,
        'theta_modulated': None, 'theta_peak_freq': None,
        'speed_score': None, 'speed_p_value': None,
        'speed_beta': None, 'speed_f0': None, 'speed_modulated': None,
        'speed_spearman_r': None, 'speed_spearman_p': None,
        'speed_normality_p': None, 'speed_residuals_normal': None,
        'speed_shuffle_mean': None, 'speed_shuffle_p95': None,
        'speed_shuffle_p': None, 'speed_shuffle_sig': None,
        'session': session_name, 'unit': ntt_file,
        'job_order': job_order, 'place_cell': None,
    }

    def _build_row(half: str | None, label: str) -> dict:
        try:
            metrics, ctx = compute_metrics(csv_path, ntt_path, arena_width_cm, target_bin_cm, half=half)
        except Exception as e:
            with _print_lock:
                print(f'  ERROR in {ntt_file} [{label}]: {e}')
            return dict(_err_row)

        if not ctx:
            bootst = _NULL_BOOTSTRAP
        else:
            try:
                bootst = _run_bootstrap(
                    ctx['spike_frame'], ctx['t'], ctx['beh_bx'], ctx['beh_by'],
                    ctx['occ_map'], ctx['valid_mask'], ctx['n_bins_x'], ctx['n_bins_y'],
                    metrics.get('sir', 0.0), ntt_path, label=label,  # type: ignore
                )
            except Exception as e:
                with _print_lock:
                    print(f'  BOOTSTRAP ERROR in {ntt_file} [{label}]: {e}')
                bootst = _NULL_BOOTSTRAP

        metrics['bootstrap_mean'] = bootst.get('bootstrap_mean')
        metrics['bootstrap_p95']  = bootst.get('bootstrap_p95')
        metrics['bootstrap_sig']  = bootst.get('bootstrap_sig')

        # Theta modulation
        spike_ts = ctx.get('spike_ts') if ctx else None
        if spike_ts is not None and len(spike_ts) >= 10:
            try:
                theta_mod, theta_freq = _compute_theta_modulation(spike_ts)
                metrics['theta_modulated'] = theta_mod
                metrics['theta_peak_freq'] = theta_freq
            except Exception as e:
                with _print_lock:
                    print(f'  THETA ERROR in {ntt_file} [{label}]: {e}')
                metrics['theta_modulated'] = None
                metrics['theta_peak_freq'] = None
        else:
            metrics['theta_modulated'] = None
            metrics['theta_peak_freq'] = None

        # Speed modulation
        if ctx and ctx.get('spike_ts') is not None and len(ctx['spike_ts']) > 0:
            try:
                speed_res = _compute_speed_modulation(
                    ctx['x_cm'], ctx['y_cm'], ctx['t'], ctx['spike_ts'], fps,
                    ntt_path=ntt_path, label=label,
                )
                metrics['speed_score']            = speed_res['speed_score']
                metrics['speed_p_value']          = speed_res['speed_p_value']
                metrics['speed_beta']             = speed_res['speed_beta']
                metrics['speed_f0']               = speed_res['speed_f0']
                metrics['speed_modulated']        = speed_res['speed_modulated']
                metrics['speed_spearman_r']       = speed_res['speed_spearman_r']
                metrics['speed_spearman_p']       = speed_res['speed_spearman_p']
                metrics['speed_normality_p']      = speed_res['speed_normality_p']
                metrics['speed_residuals_normal'] = speed_res['speed_residuals_normal']
                metrics['speed_shuffle_mean']     = speed_res['speed_shuffle_mean']
                metrics['speed_shuffle_p95']      = speed_res['speed_shuffle_p95']
                metrics['speed_shuffle_p']        = speed_res['speed_shuffle_p']
                metrics['speed_shuffle_sig']      = speed_res['speed_shuffle_sig']
            except Exception as e:
                with _print_lock:
                    print(f'  SPEED ERROR in {ntt_file} [{label}]: {e}')
                for key in ('speed_score', 'speed_p_value', 'speed_beta', 'speed_f0',
                            'speed_modulated', 'speed_spearman_r', 'speed_spearman_p',
                            'speed_normality_p', 'speed_residuals_normal',
                            'speed_shuffle_mean', 'speed_shuffle_p95',
                            'speed_shuffle_p', 'speed_shuffle_sig'):
                    metrics[key] = None
        else:
            for key in ('speed_score', 'speed_p_value', 'speed_beta', 'speed_f0',
                        'speed_modulated', 'speed_spearman_r', 'speed_spearman_p',
                        'speed_normality_p', 'speed_residuals_normal',
                        'speed_shuffle_mean', 'speed_shuffle_p95',
                        'speed_shuffle_p', 'speed_shuffle_sig'):
                metrics[key] = None

        metrics['session']   = session_name
        metrics['unit']      = ntt_file
        metrics['job_order'] = job_order

        n_spikes = metrics.get('n_spikes')
        sir      = metrics.get('sir')
        peak_fr  = metrics.get('peak_fr')
        sparsity = metrics.get('sparsity')
        boot_sig = metrics.get('bootstrap_sig')

        if (n_spikes is not None) and (sir is not None) and (peak_fr is not None) and (sparsity is not None) and (boot_sig is not None):
            metrics['place_cell'] = (
                int(n_spikes)    >  50   and
                float(peak_fr)   >  1.0  and
                float(peak_fr)   < 15.0  and
                float(sir)       >  0.5  and
                float(sparsity)  <  0.9  and
                boot_sig is True
            )
        else:
            metrics['place_cell'] = None

        return metrics

    full_row   = _build_row(None,     'full')
    first_row  = _build_row('first',  'first_half')
    second_row = _build_row('second', 'second_half')

    return (full_row, first_row, second_row)
# ── Batch scan ────────────────────────────────────────────────────────────────

_PIXEL_ANSWERS = {'pixel', 'pixels', 'px'}
_CM_ANSWERS    = {'cm', 'cms', 'centimeter', 'centimeters', 'centimetre', 'centimetres'}

if __name__ == "__main__":
    _coord_answer = input("Are the tracking coordinates in pixels or cm? [pixel/cm]: ").strip().lower()
    while _coord_answer not in _PIXEL_ANSWERS | _CM_ANSWERS:
        _coord_answer = input("Please enter 'pixel' or 'cm': ").strip().lower()
    COORD_UNITS = 'pixel' if _coord_answer in _PIXEL_ANSWERS else 'cm'
    print(f"Using '{COORD_UNITS}' tracking coordinates.\n")

    all_jobs   = []
    dir_to_csv = {}
    output_excel_basename = os.path.basename(output_excel).lower()
    for dirpath, _, filenames in os.walk(root_folder):
        tracking_files_all = [f for f in filenames
                              if f.lower().endswith(('.csv', '.xlsx'))
                              and f.lower() != output_excel_basename]
        # '_cm.csv' files are the cm-converted tracking files; everything else
        # (.xlsx, or a plain .csv without that suffix) is pixel-based tracking.
        if COORD_UNITS == 'cm':
            tracking_files = [f for f in tracking_files_all if f.lower().endswith('_cm.csv')]
        else:
            tracking_files = [f for f in tracking_files_all if not f.lower().endswith('_cm.csv')]
        ntt_files      = [f for f in filenames if f.lower().endswith('.ntt')]
        if len(tracking_files) == 1 and len(ntt_files) > 0:
            csv_path = os.path.join(dirpath, tracking_files[0])
            dir_to_csv[dirpath] = csv_path
            for ntt_file in sorted(ntt_files):
                all_jobs.append((dirpath, csv_path, ntt_file))

    total_units = len(all_jobs)
    print(f'Found {total_units} unit(s) across all sessions.\n')

    job_args = [
        (idx, total_units, idx - 1, dirpath, csv_path, ntt_file)
        for idx, (dirpath, csv_path, ntt_file) in enumerate(all_jobs, start=1)
    ]

    # ── Parallel execution ────────────────────────────────────────────────────────

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_run_job, args): args for args in job_args}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: r[0]['job_order'])

    # ── Save to Excel ─────────────────────────────────────────────────────────────

    column_order = ['session', 'unit', 'n_spikes', 'n_discarded',
                    'peak_fr', 'mean_fr', 'sir', 'sparsity', 'coherence',
                    'bootstrap_mean', 'bootstrap_p95', 'bootstrap_sig',
                    'theta_modulated', 'theta_peak_freq',
                    'speed_score', 'speed_p_value', 'speed_beta', 'speed_f0', 'speed_modulated',
                    'speed_spearman_r', 'speed_spearman_p',
                    'speed_normality_p', 'speed_residuals_normal',
                    'speed_shuffle_mean', 'speed_shuffle_p95',
                    'speed_shuffle_p', 'speed_shuffle_sig',
                    'speed_p_value_fdr', 'speed_modulated_fdr',
                    'speed_shuffle_p_fdr', 'speed_shuffle_sig_fdr',
                    'place_cell']

    df_full   = pd.DataFrame([r[0] for r in results], columns=column_order)
    df_first  = pd.DataFrame([r[1] for r in results], columns=column_order)
    df_second = pd.DataFrame([r[2] for r in results], columns=column_order)

    # ── FDR (Benjamini-Hochberg) correction across the batch ───────────────────
    # A fixed alpha = 0.05 applied cell-by-cell means ~5% of non-modulated
    # cells get flagged "significant" by chance alone. Since the downstream
    # question is typically "what fraction of the population is speed-
    # modulated," BH-correct across all units tested in each sheet (each
    # half is its own independent batch of tests) -- for both the parametric
    # WLS p-value and the shuffle p-value.
    for df in (df_full, df_first, df_second):
        df['speed_p_value_fdr']     = _benjamini_hochberg(df['speed_p_value'])
        df['speed_modulated_fdr']   = (df['speed_p_value_fdr'] < 0.05).astype(object)
        df.loc[df['speed_p_value_fdr'].isna(), 'speed_modulated_fdr'] = None
        df['speed_shuffle_p_fdr']   = _benjamini_hochberg(df['speed_shuffle_p'])
        df['speed_shuffle_sig_fdr'] = (df['speed_shuffle_p_fdr'] < 0.05).astype(object)
        df.loc[df['speed_shuffle_p_fdr'].isna(), 'speed_shuffle_sig_fdr'] = None

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_full.to_excel(writer,   sheet_name='Full',        index=False)
        df_first.to_excel(writer,  sheet_name='First_Half',  index=False)
        df_second.to_excel(writer, sheet_name='Second_Half', index=False)

    print(f'\nDone. Results saved to {output_excel}')
    print(f'Total units processed : {len(df_full)}')
    print(f'Place cells found     : {df_full["place_cell"].sum()}')
    print(f'Speed-modulated (parametric, uncorrected) : {df_full["speed_modulated"].sum()}')
    print(f'Speed-modulated (parametric, FDR-corrected): {df_full["speed_modulated_fdr"].sum()}')
    print(f'Speed-modulated (shuffle, FDR-corrected)   : {df_full["speed_shuffle_sig_fdr"].sum()}')

    # ── Copy .ntt + tracking files for confirmed place cells ───────────────────
    # Replicates the folder structure from the animal-ID folder onwards
    # (Fa1059 / Fa23BD / Fa8477 / Fa5384) under Output_PlaceTrue.

    place_rows = df_full[df_full['place_cell'] == True]  # noqa: E712
    copied_tracking_dirs = set()

    for _, row in place_rows.iterrows():
        session = row['session']
        dirpath = root_folder if session == '.' else os.path.join(root_folder, session)
        ntt_path = os.path.join(dirpath, row['unit'])

        rel = _animal_relpath(dirpath)
        if rel is None:
            print(f'  [SKIP] Could not locate animal-ID folder in path: {dirpath}')
            continue

        dest_dir = os.path.join(Output_PlaceTrue, rel)
        os.makedirs(dest_dir, exist_ok=True)

        if os.path.isfile(ntt_path):
            shutil.copy2(ntt_path, dest_dir)
        else:
            print(f'  [SKIP] .ntt file not found: {ntt_path}')

        if dirpath not in copied_tracking_dirs:
            csv_path = dir_to_csv.get(dirpath)
            if csv_path and os.path.isfile(csv_path):
                shutil.copy2(csv_path, dest_dir)
            copied_tracking_dirs.add(dirpath)

    print(f'Place-cell files copied to : {Output_PlaceTrue}')