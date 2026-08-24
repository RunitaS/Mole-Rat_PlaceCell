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
from typing import TYPE_CHECKING, NamedTuple, TypedDict

import numpy as np
import pandas as pd
from scipy.ndimage import convolve, gaussian_filter1d
from scipy.stats import pearsonr, t as _t_dist

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
    speed_score:     float | None
    speed_p_value:   float | None
    speed_r2:        float | None
    speed_beta:      float | None
    speed_f0:        float | None
    speed_modulated: bool | None
    speed_shuffle_mean: float | None
    speed_shuffle_lo:   float | None
    speed_shuffle_hi:   float | None
    speed_shuffle_p:    float | None
    speed_modulated_shuffle: bool | None
    speed_shuffle_ran:  bool | None
    speed_score_td:      float | None
    speed_p_value_td:    float | None
    speed_r2_td:          float | None
    speed_shuffle_mean_td: float | None
    speed_shuffle_lo_td:   float | None
    speed_shuffle_hi_td:   float | None
    speed_shuffle_p_td:    float | None
    speed_modulated_td:    bool | None
    speed_shuffle_ran_td:  bool | None
    p_speed:         bool | None
    n_speed:         bool | None
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
output_excel = r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True/wut_SpeedShuf_pValDebug_Median.xlsx'

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
SPEED_MAX_CMS       = 80.0   # highest speed (cm/s) included in speed-modulation analysis
SPEED_BIN_CMS       = 4.0 #2.0     # width of each speed bin (cm/s)
SPEED_SMOOTH_S      = 0.3 #0.08    # Gaussian smoothing window (s) applied to instantaneous firing rate
SPEED_MIN_BIN_FRAC  = 0.002   # discard speed bins holding < this fraction of samples

SPEED_N_SHUFFLE      = 1000   # circular-shift shuffles for speed-modulation significance
SPEED_SHUFFLE_MARGIN_S = 20.0 # min circular-shift offset (s) from either end, matches SIR shuffling

POS_JUMP_THRESH_CMS  = 80.0   # frame-to-frame jumps implying a speed above this (cm/s) are tracking artifacts
POS_SMOOTH_SIGMA_SMP = 1.0   # Gaussian smoothing sigma (in samples) applied to x/y tracking position

SPEED_MOD_DOWNSAMPLE_FACTOR = 3   # downsample tracking for speed-modulation analysis only:
                                   # 30 fps -> 15 fps by keeping every alternate frame. The
                                   # 33.33 ms native bin made the speed-vs-firing-rate cloud
                                   # too scattered; speed/rate are recomputed on this coarser
                                   # frame base (rest of the pipeline, e.g. SIR/ratemap, is
                                   # unaffected and still uses the native-rate tracking).

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


# ── Tracking position smoothing ───────────────────────────────────────────────

def _smooth_tracking_position(
    x_cm:            np.ndarray,
    y_cm:            np.ndarray,
    t_us:            np.ndarray,
    jump_thresh_cms: float = POS_JUMP_THRESH_CMS,
    sigma_samples:   float = POS_SMOOTH_SIGMA_SMP,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean and smooth the x/y tracking position before any downstream use.

    Iterative jump removal + Gaussian smoothing:
      1. Frame-to-frame steps between surviving ("good") samples that imply
         a speed above `jump_thresh_cms` are treated as tracking artifacts.
         This is re-tested against the shrinking good-sample set until no
         new artifacts are found, so runs of two or more consecutive bad
         frames (e.g. a tracker glitch that lingers for a few frames) are
         caught — not just single-frame jumps relative to the immediately
         preceding raw sample.
      2. Artifact frames are filled by linear interpolation (in time) between
         the nearest surviving good samples, rather than held at the last
         good position — holding would freeze position then "snap" back at
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


# ── Weighted least squares (speed-modulation regression) ──────────────────────

class _WLSResult(NamedTuple):
    slope:     float
    intercept: float
    rvalue:    float
    pvalue:    float


def _weighted_linregress(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> _WLSResult:
    """Weighted least-squares fit of y = slope*x + intercept, weighting each
    (x, y) point by `w`.

    Used in place of scipy.stats.linregress for the speed-modulation fit:
    each point there is a speed bin's mean firing rate, but bins hold very
    different sample counts (sparse high-speed bins vs. dense low-speed
    bins). An unweighted fit gives a 50-sample bin the same leverage as a
    5000-sample bin, biasing the slope/r/p-value toward whatever the noisy,
    sparse bins happen to show. Weighting by `w = n_per_bin` corrects that.

    rvalue is the weighted Pearson correlation; pvalue is a two-tailed t-test
    on the slope (df = n_points - 2), mirroring linregress's convention.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    w_sum  = w.sum()
    x_mean = np.sum(w * x) / w_sum
    y_mean = np.sum(w * y) / w_sum
    dx, dy = x - x_mean, y - y_mean

    Sxx = np.sum(w * dx * dx)
    Syy = np.sum(w * dy * dy)
    Sxy = np.sum(w * dx * dy)

    if Sxx <= 0:
        return _WLSResult(slope=0.0, intercept=float(y_mean), rvalue=0.0, pvalue=1.0)

    slope     = Sxy / Sxx
    intercept = y_mean - slope * x_mean
    rvalue    = Sxy / np.sqrt(Sxx * Syy) if Syy > 0 else 0.0
    rvalue    = float(np.clip(rvalue, -1.0, 1.0))

    n   = len(x)
    dof = n - 2
    if dof > 0:
        resid = y - (slope * x + intercept)
        mse   = np.sum(w * resid * resid) / dof
        se_slope = np.sqrt(mse / Sxx)
        if se_slope > 0:
            tstat  = slope / se_slope
            pvalue = float(2.0 * _t_dist.sf(np.abs(tstat), dof))
        else:
            pvalue = 0.0 if slope != 0.0 else 1.0
    else:
        pvalue = float('nan')

    return _WLSResult(slope=float(slope), intercept=float(intercept),
                       rvalue=rvalue, pvalue=pvalue)


# ── Shuffle significance (shared by binned + time-domain speed modulation) ────

class _ShuffleSummary(NamedTuple):
    mean:      float
    lo:        float
    hi:        float
    p:         float
    modulated: bool


def _shuffle_ci_and_pvalue(real_score: float, shuff_scores: np.ndarray,
                            min_valid: int = 100) -> '_ShuffleSummary | None':
    """Two-tailed 2.5/97.5 shuffle CI + signed-tail p-value from a null
    distribution of circular-shift shuffle scores.

    Used identically for BOTH the binned-regression speed score and the
    time-domain (Pearson r) speed score, so the two significance tests only
    ever differ in which score they're fed — never in how significance is
    computed from it.

    Returns None if fewer than `min_valid` finite shuffle scores survive.
    """
    valid = shuff_scores[np.isfinite(shuff_scores)]
    if len(valid) < min_valid:
        return None

    n    = len(valid)
    mean = float(np.mean(valid))
    lo   = float(np.percentile(valid, 2.5))
    hi   = float(np.percentile(valid, 97.5))

    # Two-tailed p-value, built from the SAME signed-tail rule as lo/hi above
    # (rather than pooling |shuffle| >= |real| across both tails, which can
    # silently disagree with the lo/hi crossing test when the null
    # distribution is skewed): count how far into whichever tail the real
    # score sits on its own side, then double that one-sided fraction so a
    # 2.5% one-sided tail on either side reads as p = 0.05 two-sided. This
    # makes "p < 0.05" and "outside [lo, hi]" the same test by construction.
    if real_score >= 0:
        tail_count = int(np.sum(valid >= real_score))
    else:
        tail_count = int(np.sum(valid <= real_score))
    p = float(min(1.0, 2.0 * (tail_count + 1) / (n + 1)))

    return _ShuffleSummary(mean=mean, lo=lo, hi=hi, p=p, modulated=bool(p < 0.05))


# ── Speed modulation ──────────────────────────────────────────────────────────

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
    ntt_path:           str | None = None,
    label:              str = '',
) -> dict:
    """Relates firing rate to running speed (ports MATLAB `speed_firing_runita`).

      1. Per-frame running speed (cm/s) from consecutive positions, using the
         actual inter-frame interval rather than an assumed fixed rate, then
         Gaussian-smoothed with the same kernel as the firing rate (step 2)
         so the two signals being compared are smoothed on equal footing.
      2. Instantaneous firing rate on the same frame base (spike count per
         inter-frame interval / interval duration), Gaussian-smoothed.
      3. Restrict to samples with min_speed_cms < smoothed speed < max_speed_cms.
      4. Bin firing rate by (smoothed) speed (speed_bin_cms-wide bins), drop
         bins with < min_bin_frac of the samples, and fit rate = beta*speed + f0
         to the binned medians (beta/f0 naming kept from the MATLAB source).

    x_cm, y_cm, t_us must be same-length position tracks (t_us in µs, sorted).
    spike_ts_us must be in the same time base (µs).
    Returns speed_score (r), speed_p_value, speed_beta (slope),
    speed_f0 (intercept), speed_modulated (p < 0.05), and a circular-shift
    shuffle confirmation (speed_shuffle_mean/lo/hi/p, speed_modulated_shuffle)
    analogous to the SIR location-shuffling bootstrap.
    """
    result = {'speed_score': float('nan'), 'speed_p_value': float('nan'),
              'speed_r2': float('nan'),
              'speed_beta': float('nan'), 'speed_f0': float('nan'),
              'speed_modulated': None,
              'speed_shuffle_mean': float('nan'), 'speed_shuffle_lo': float('nan'),
              'speed_shuffle_hi': float('nan'), 'speed_shuffle_p': float('nan'),
              'speed_modulated_shuffle': None,
              # Time-domain correlation (Iwase et al. 2020 "speed score"): direct
              # Pearson r between the smoothed speed and smoothed firing-rate time
              # series, sample-by-sample, with no speed-binning step. Computed
              # alongside — not instead of — the binned-regression result above.
              'speed_score_td': float('nan'), 'speed_p_value_td': float('nan'),
              'speed_r2_td': float('nan'),
              'speed_shuffle_mean_td': float('nan'), 'speed_shuffle_lo_td': float('nan'),
              'speed_shuffle_hi_td': float('nan'), 'speed_shuffle_p_td': float('nan'),
              'speed_modulated_td': None,
              # p-Speed / n-Speed classification, matching Iwase et al.: real
              # speed_score_td above the 99th percentile (p-Speed) or below the
              # 1st percentile (n-Speed) of its own shuffle null distribution.
              'p_speed': None, 'n_speed': None,
              # Whether each method's initial (parametric) R²/p-value screen
              # passed and its circular-shift shuffle was therefore run at all
              # -- see the "initial significance gate" below.
              'speed_shuffle_ran': False, 'speed_shuffle_ran_td': False}

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

    # ── Gaussian smoothing of speed, matching the firing-rate smoothing ───────
    # Same kernel (sigma_samples, derived from smooth_window_s and the position
    # sample rate) and same NaN-safe zero-fill/restore trick as the firing-rate
    # smoothing below, so speed and firing rate are compared on equal footing
    # instead of a smoothed rate against raw (noisy) speed. sigma_samples is
    # computed once here and reused for the firing-rate smoothing further down.
    sigma_samples = max(smooth_window_s * pos_sample_rate_hz, 1e-6)
    speed_finite = np.isfinite(speed)
    speed_smooth = gaussian_filter1d(np.where(speed_finite, speed, 0.0),
                                      sigma=sigma_samples, mode='nearest')
    speed_smooth[~speed_finite] = np.nan

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

    # Gaussian smoothing of the instantaneous rate.
    # WINDOW SIZE: controlled by `smooth_window_s` (default SPEED_SMOOTH_S,
    # currently 0.08 s = 80 ms). This is the *standard deviation* (sigma) of
    # the Gaussian kernel, expressed in seconds, and is converted to samples
    # via sigma_samples = smooth_window_s * pos_sample_rate_hz (the position
    # tracking frame rate, e.g. 30 Hz -> sigma ≈ 2.4 samples ≈ 80 ms).
    # scipy's gaussian_filter1d truncates the kernel at 4*sigma by default, so
    # the *effective* smoothing window (full kernel support) spans roughly
    # ±4*sigma_samples samples (~±320 ms at 30 Hz), not just ±sigma.
    # NaNs (from bad dt) are temporarily zero-filled before filtering (so they
    # don't propagate NaN through the whole kernel support) and then restored
    # afterwards via the `finite` mask, so the output stays NaN exactly where
    # the input was invalid.
    # sigma_samples was already computed above (shared with the speed smoothing).
    finite = np.isfinite(fr_inst)
    fr_smooth = gaussian_filter1d(np.where(finite, fr_inst, 0.0),
                                   sigma=sigma_samples, mode='nearest')
    fr_smooth[~finite] = np.nan

    # ── Restrict to the usable speed range ─────────────────────────────────────
    # Very low speeds (near-stationary, e.g. grooming/resting) and very high
    # speeds (tracking artifacts / jumps) are excluded so the fit isn't
    # dominated by outliers or immobility-related firing (e.g. sharp-wave
    # ripples during rest). Only frames with min_speed_cms < speed < max_speed_cms
    # (evaluated on the smoothed speed, matching the smoothed rate it's compared
    # against) AND a valid (non-NaN) smoothed rate are kept.
    in_range = (speed_smooth > min_speed_cms) & (speed_smooth < max_speed_cms) & np.isfinite(fr_smooth)
    if in_range.sum() < 3:
        return result

    speed_valid = speed_smooth[in_range]
    rate_valid  = fr_smooth[in_range]
    if np.std(speed_valid) == 0 or np.std(rate_valid) == 0:
        return result

    # ── Combined 2×2 speed-analysis figure ──────────────────────────────────────
    # top-left: binned speed-modulation scatter/fit; bottom-left: its shuffle
    # histogram. top-right: time-domain speed-vs-rate correlation (time series);
    # bottom-right: its shuffle histogram. All four panels for this cell are
    # drawn into one figure and saved together, in a single 'speed modulation'
    # folder, rather than as four separate plots split across two folders.
    fig_combined = ax_bin = ax_bin_shuf = ax_td = ax_td_shuf = None
    if ntt_path is not None:
        fig_combined = Figure(figsize=(14, 10))
        FigureCanvasAgg(fig_combined)
        ax_bin      = fig_combined.add_subplot(2, 2, 1)
        ax_bin_shuf = fig_combined.add_subplot(2, 2, 3)
        ax_td       = fig_combined.add_subplot(2, 2, 2)
        ax_td_shuf  = fig_combined.add_subplot(2, 2, 4)

    # ── Time-domain correlation (Iwase et al. 2020 "speed score") ──────────────
    # Direct Pearson correlation between the smoothed speed and smoothed firing
    # rate, sample-by-sample across time — no speed-binning step, unlike the
    # binned-regression fit below. This is the method the paper actually uses.
    r_td, p_td = pearsonr(speed_valid, rate_valid)
    result['speed_score_td'] = round(float(r_td), 4)
    result['speed_r2_td']    = round(float(r_td ** 2), 4)
    # NOTE: this parametric p-value assumes independent residuals, which is
    # violated here — Gaussian-smoothed adjacent samples are strongly
    # autocorrelated — so it will be anti-conservative (falsely low). The
    # shuffle-based speed_p_value_td / speed_modulated_td set further below,
    # from circularly-shifted null data, is the one to trust for significance.
    # It's still used below as a cheap initial screen: a fit that isn't even
    # nominally significant on its face doesn't get the expensive shuffle.
    result['speed_p_value_td'] = round(float(p_td), 4)

    # ── top-right panel: time-domain speed-vs-rate correlation (time series) ───
    # Drawn unconditionally, mirroring the binned scatter/fit panel below (also
    # independent of significance), so this panel is present whether or not the
    # shuffle ends up running. The title is refined further down with the
    # shuffle-based label/p-value once (if) that shuffle completes.
    if ax_td is not None:
        time_s = (t_us - t_us[0]) * 1e-6
        ax_td.plot(time_s, speed_smooth, color='tab:blue', linewidth=0.6)
        ax_td.set_xlabel('time (s)')
        ax_td.set_ylabel('speed (cm/s)', color='tab:blue')
        ax_td.tick_params(axis='y', labelcolor='tab:blue')
        ax_td_twin = ax_td.twinx()
        ax_td_twin.plot(time_s, fr_smooth, color='tab:orange', linewidth=0.6)
        ax_td_twin.set_ylabel('firing rate (Hz)', color='tab:orange')
        ax_td_twin.tick_params(axis='y', labelcolor='tab:orange')
        ax_td.set_title(f"time-domain: r = {r_td:.3f} (parametric p = {p_td:.3g})")

    # ── Bin firing rate by speed ────────────────────────────────────────────────
    # Build speed_bin_cms-wide bin edges spanning [min_speed_cms, max_speed_cms]
    # (e.g. 2 cm/s wide bins from 2 to 90 cm/s), giving n_bins bins.
    speed_bins = np.arange(min_speed_cms, max_speed_cms + speed_bin_cms, speed_bin_cms)
    n_bins     = len(speed_bins) - 1
    # For every valid (speed, rate) sample, find which speed bin it falls in.
    # np.digitize returns 1-indexed bin numbers; subtract 1 to make it
    # 0-indexed, and clip to guard against samples exactly at/above the last
    # edge (which digitize would otherwise put in an out-of-range bin).
    bin_idx    = np.clip(np.digitize(speed_valid, speed_bins) - 1, 0, n_bins - 1)

    # bin_centres: the midpoint speed of each bin, used as the x-value in the
    # final regression (rate vs. speed).
    bin_centres = 0.5 * (speed_bins[:-1] + speed_bins[1:])
    binned_rate = np.full(n_bins, np.nan)
    n_per_bin   = np.zeros(n_bins, dtype=int)
    bin_masks   = []   # per-bin boolean membership, reused (unchanged across shuffles) below
    # For each speed bin, take the median (not mean) of the (already
    # Gaussian-smoothed) firing rate of every sample that fell in it. This
    # collapses the noisy per-frame rate/speed cloud into one representative
    # rate per speed bin, using the median so a handful of outlier frames
    # (e.g. a burst landing in an otherwise low-rate bin) don't drag the
    # bin's representative value the way a mean would.
    for b in range(n_bins):
        sel = bin_idx == b
        bin_masks.append(sel)
        n_per_bin[b] = int(sel.sum())
        if n_per_bin[b] > 0:
            binned_rate[b] = float(np.median(rate_valid[sel]))

    total_pts = n_per_bin.sum()
    if total_pts == 0:
        return result
    # Bins that hold too few samples (fraction of total < min_bin_frac, e.g.
    # 0.2%) are unreliable estimates of rate at that speed, so they're
    # discarded (set to NaN) rather than allowed to bias the regression —
    # this matters most at the high-speed tail, which is sparsely sampled.
    binned_rate[(n_per_bin / total_pts) < min_bin_frac] = np.nan   # drop under-sampled bins

    fit_sel = np.isfinite(binned_rate)
    if fit_sel.sum() < 3:
        return result

    # Weighted linear regression of binned median firing rate against
    # bin-centre speed: rate = speed_beta * speed + speed_f0, each bin
    # weighted by its sample count (n_per_bin) so sparsely-sampled bins
    # (mostly at the high-speed tail) don't get the same leverage as
    # densely-sampled ones.
    # reg.rvalue is the speed score (correlation coefficient r), reg.pvalue
    # tests whether the slope is significantly different from zero
    # (p < 0.05 => speed_modulated).
    weights_fit = n_per_bin[fit_sel].astype(np.float64)
    reg = _weighted_linregress(bin_centres[fit_sel], binned_rate[fit_sel], weights_fit)

    result['speed_score']     = round(float(reg.rvalue), 4)
    result['speed_p_value']   = round(float(reg.pvalue), 4)
    result['speed_r2']        = round(float(reg.rvalue ** 2), 4)
    result['speed_beta']      = round(float(reg.slope), 4)
    result['speed_f0']        = round(float(reg.intercept), 4)
    result['speed_modulated'] = bool(reg.pvalue < 0.05)

    # ── Initial R²/p-value significance gate ────────────────────────────────────
    # Before spending SPEED_N_SHUFFLE circular-shift iterations on a cell, first
    # check whether its real (parametric) fit is even nominally significant:
    # binned_significant from the weighted-regression p-value (reg.pvalue, same
    # test as speed_modulated above), td_significant from the raw Pearson p_td.
    # A method whose own initial fit doesn't clear p < 0.05 doesn't get shuffled
    # at all -- its shuffle_* / modulated_* / p_speed / n_speed fields are left
    # at their NaN/None defaults rather than being computed from a null test
    # that was never going to matter. The two methods are gated independently:
    # one can pass its screen and run its shuffle while the other doesn't.
    binned_significant = result['speed_modulated']
    td_significant      = bool(p_td < 0.05)

    # ── Circular-shift shuffling (confirms speed_modulated) ────────────────────
    # Same idea as the SIR location-shuffling bootstrap (_run_bootstrap): the
    # spike-count-per-interval trace is circularly shifted by a random offset
    # of at least SPEED_SHUFFLE_MARGIN_S (20 s) from either end, decorrelating
    # firing from speed while preserving each trace's own statistics. The
    # speed bin edges/membership (bin_idx) and which bins pass min_bin_frac
    # (fit_sel) depend only on `speed`, not on firing, so they are identical
    # for every shuffle and are reused rather than recomputed N_SHUFFLE times.
    n_intervals   = len(counts)
    MARGIN_FRAMES = int(SPEED_SHUFFLE_MARGIN_S * pos_sample_rate_hz)
    real_score    = float(reg.rvalue)

    binned_shuf_drawn = False   # whether the bottom-left panel got real histogram content
    td_shuf_drawn     = False   # whether the bottom-right panel got real histogram content

    if n_intervals > 2 * MARGIN_FRAMES and (binned_significant or td_significant):
        # Record which method(s) actually cleared the initial screen and
        # therefore get their null distribution built below.
        result['speed_shuffle_ran']    = binned_significant
        result['speed_shuffle_ran_td'] = td_significant

        fit_idx         = np.where(fit_sel)[0]
        n_per_bin_fit    = n_per_bin[fit_idx].astype(np.float64)
        bin_centres_fit  = bin_centres[fit_idx]
        bin_masks_fit    = [bin_masks[b] for b in fit_idx]   # same membership as the real fit, reused every shuffle
        shuff_scores     = np.full(SPEED_N_SHUFFLE, np.nan)
        shuff_scores_td  = np.full(SPEED_N_SHUFFLE, np.nan)   # null distribution for the time-domain correlation

        for i in range(SPEED_N_SHUFFLE):
            rnd             = random.randint(MARGIN_FRAMES, n_intervals - MARGIN_FRAMES)
            shifted_counts  = np.roll(counts, rnd)
            with np.errstate(invalid='ignore', divide='ignore'):
                fr_inst_s = shifted_counts / dt_s
            fr_inst_s[dt_s <= 0] = np.nan
            fr_inst_s = np.append(fr_inst_s, fr_inst_s[-1])
            fr_smooth_s = gaussian_filter1d(np.where(finite, fr_inst_s, 0.0),
                                             sigma=sigma_samples, mode='nearest')
            fr_smooth_s[~finite] = np.nan

            # Per-bin median of this shuffle's rate (matches the real-fit
            # computation above) — can't use the old bincount-sum/n trick
            # since median isn't additive across samples.
            rate_valid_s   = fr_smooth_s[in_range]

            # Time-domain null score: same shuffled rate, correlated directly
            # against the (unshuffled) smoothed speed — mirrors real_score's
            # r_td computation above, giving a matched null distribution for
            # the time-domain correlation. Only built if the time-domain fit
            # passed its own initial significance gate.
            if td_significant and np.std(rate_valid_s) > 0:
                shuff_scores_td[i] = pearsonr(speed_valid, rate_valid_s)[0]

            # Binned null score. Only built if the binned fit passed its own
            # initial significance gate.
            if binned_significant:
                binned_rate_fit = np.array([np.median(rate_valid_s[m]) for m in bin_masks_fit])
                if np.std(binned_rate_fit) != 0:
                    reg_s = _weighted_linregress(bin_centres_fit, binned_rate_fit, n_per_bin_fit)
                    shuff_scores[i] = reg_s.rvalue

        # ── bottom-left panel: binned-fit shuffle histogram ─────────────────────
        valid_shuff = shuff_scores[np.isfinite(shuff_scores)]
        summ = _shuffle_ci_and_pvalue(real_score, shuff_scores)
        if summ is not None:
            shuffle_mean, shuffle_lo, shuffle_hi, shuffle_p = summ.mean, summ.lo, summ.hi, summ.p

            result['speed_shuffle_mean']      = round(shuffle_mean, 4)
            result['speed_shuffle_lo']        = round(shuffle_lo, 4)
            result['speed_shuffle_hi']        = round(shuffle_hi, 4)
            result['speed_shuffle_p']         = round(shuffle_p, 4)
            result['speed_modulated_shuffle'] = summ.modulated

            if ax_bin_shuf is not None:
                hist_result_s = ax_bin_shuf.hist(valid_shuff, 100, color='black')
                counts_s: np.ndarray = np.asarray(hist_result_s[0])
                max_count_s = float(counts_s.max()) if counts_s.max() > 0 else 1.0

                ax_bin_shuf.plot([real_score, real_score], [0, max_count_s], 'r-.')
                # 2.5% / 97.5% shuffle CI bounds, marking the significance cutoff.
                ax_bin_shuf.axvline(shuffle_lo, color='0.5', linestyle=':', linewidth=0.8)
                ax_bin_shuf.axvline(shuffle_hi, color='0.5', linestyle=':', linewidth=0.8)
                sh_av = f"Shuffle mean r = {shuffle_mean:.3f}"
                sh_ci = f"95% CI = [{shuffle_lo:.3f}, {shuffle_hi:.3f}]"
                sh_p  = (f"Cell r = {real_score:.3f} "
                         f"({'p < 0.05' if result['speed_modulated_shuffle'] else 'ns'})")
                ax_bin_shuf.set_title(sh_av + '\n' + sh_ci + '\n' + sh_p, multialignment='center')
                ax_bin_shuf.set_ylabel('count')
                ax_bin_shuf.set_xlabel('speed score (r)')
                binned_shuf_drawn = True

        # ── Time-domain shuffle summary + p-Speed/n-Speed classification ───────
        # Uses the exact same _shuffle_ci_and_pvalue helper as the binned result
        # above — identical CI/p-value construction, only the input score differs.
        valid_shuff_td = shuff_scores_td[np.isfinite(shuff_scores_td)]
        summ_td = _shuffle_ci_and_pvalue(r_td, shuff_scores_td)
        if summ_td is not None:
            shuffle_mean_td, shuffle_lo_td, shuffle_hi_td, shuffle_p_td = (
                summ_td.mean, summ_td.lo, summ_td.hi, summ_td.p)

            result['speed_shuffle_mean_td'] = round(shuffle_mean_td, 4)
            result['speed_shuffle_lo_td']   = round(shuffle_lo_td, 4)
            result['speed_shuffle_hi_td']   = round(shuffle_hi_td, 4)
            result['speed_shuffle_p_td']    = round(shuffle_p_td, 4)
            result['speed_modulated_td']    = summ_td.modulated

            # p-Speed / n-Speed classification: uses the SAME two-tailed 2.5/97.5
            # shuffle CI as speed_modulated_td above (not the paper's one-sided
            # 1st/99th percentile) — a cell is p-Speed if its real time-domain
            # score sits above that CI, n-Speed if it sits below.
            result['p_speed'] = bool(r_td > shuffle_hi_td)
            result['n_speed'] = bool(r_td < shuffle_lo_td)

            if result['p_speed']:
                _td_label, _td_color = 'p-Speed', 'green'
            elif result['n_speed']:
                _td_label, _td_color = 'n-Speed', 'red'
            else:
                _td_label, _td_color = 'non-speed', 'black'

            # Refine the top-right panel's title now that the shuffle-based
            # label/p-value are known (the panel itself was already drawn above).
            if ax_td is not None:
                ax_td.set_title(f"{_td_label}  (r = {r_td:.3f}, shuffle p = {shuffle_p_td:.3g})",
                                 color=_td_color, fontweight='bold')

            # ── bottom-right panel: time-domain shuffle histogram ───────────────
            if ax_td_shuf is not None:
                hist_result_td = ax_td_shuf.hist(valid_shuff_td, 100, color='black')
                counts_td: np.ndarray = np.asarray(hist_result_td[0])
                max_count_td = float(counts_td.max()) if counts_td.max() > 0 else 1.0

                ax_td_shuf.plot([r_td, r_td], [0, max_count_td], color=_td_color, linestyle='-.', linewidth=1.5)
                ax_td_shuf.axvline(shuffle_lo_td, color='0.5', linestyle=':', linewidth=0.8)
                ax_td_shuf.axvline(shuffle_hi_td, color='0.5', linestyle=':', linewidth=0.8)

                title_td = (f"{_td_label}\n"
                            f"Cell r = {r_td:.3f}  "
                            f"(shuffle 95% CI = [{shuffle_lo_td:.3f}, {shuffle_hi_td:.3f}], "
                            f"p = {shuffle_p_td:.3g})")
                ax_td_shuf.set_title(title_td, color=_td_color, fontweight='bold', multialignment='center')
                ax_td_shuf.set_ylabel('count')
                ax_td_shuf.set_xlabel('time-domain speed score (r)')
                td_shuf_drawn = True

    # Shuffle panels that never ran (initial screen not significant, or not
    # enough frames for the margin) get a placeholder instead of staying blank.
    if ax_bin_shuf is not None and not binned_shuf_drawn:
        ax_bin_shuf.text(0.5, 0.5, 'binned-fit shuffle not run\n(initial fit not significant)',
                          ha='center', va='center', transform=ax_bin_shuf.transAxes)
        ax_bin_shuf.set_xticks([]); ax_bin_shuf.set_yticks([])
    if ax_td_shuf is not None and not td_shuf_drawn:
        ax_td_shuf.text(0.5, 0.5, 'time-domain shuffle not run\n(initial correlation not significant)',
                         ha='center', va='center', transform=ax_td_shuf.transAxes)
        ax_td_shuf.set_xticks([]); ax_td_shuf.set_yticks([])

    # ── top-left panel: binned speed-modulation scatter/fit ─────────────────────
    if ax_bin is not None:
        ax_bin.scatter(speed_valid, rate_valid, s=3, color='0.75', alpha=0.4,
                        label='raw samples')
        ax_bin.scatter(bin_centres[fit_sel], binned_rate[fit_sel], s=80, color='black',
                        label='binned median')

        fit_x = np.array([bin_centres[fit_sel].min(), bin_centres[fit_sel].max()])
        fit_y = reg.slope * fit_x + reg.intercept
        ax_bin.plot(fit_x, fit_y, 'r-', label='fit')

        stats_txt = (f"binned: r = {reg.rvalue:.3f}, slope = {reg.slope:.3f}, "
                     f"intercept = {reg.intercept:.3f}, p = {reg.pvalue:.3g}\n"
                     f"time-domain: r = {r_td:.3f}, p(shuffle) = {result['speed_shuffle_p_td']}")
        ax_bin.text(0.02, 0.98, stats_txt, transform=ax_bin.transAxes,
                    va='top', ha='left', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax_bin.set_xlabel('speed (cm/s)')
        ax_bin.set_ylabel('firing rate (Hz)')
        ax_bin.legend(loc='lower right', fontsize=8)

    # ── save the combined 2×2 figure ─────────────────────────────────────────────
    if fig_combined is not None:
        fig_combined.tight_layout()

        ntt_name   = os.path.splitext(os.path.basename(ntt_path))[0]
        save_dir   = os.path.join(os.path.dirname(ntt_path), 'speed modulation')
        os.makedirs(save_dir, exist_ok=True)
        lbl_suffix = f'_{label}' if label else ''
        save_path  = os.path.join(save_dir, f'{ntt_name}{lbl_suffix}_speed_modulation.png')
        fig_combined.savefig(save_path, dpi=150)
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


# ── Tracking load/clean/convert ────────────────────────────────────────────────

def _load_tracking(csv_path: str, arena_width_cm: float,
                   half: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load, clean, and pixel→cm-convert one tracking file (no position
    smoothing applied — see `_smooth_tracking_position`).

    Returns (x_cm, y_cm, t) with t in the same (µs) time base as spike
    timestamps. Arrays are empty if no valid tracking samples remain.
    """
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
        return x.astype(np.float64), y.astype(np.float64), t.astype(np.float64)

    # ── Pixel → cm conversion ──────────────────────────────────────────────────
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

    return x_cm, y_cm, t


def _instantaneous_speed(x_cm: np.ndarray, y_cm: np.ndarray, t_us: np.ndarray) -> np.ndarray:
    """Per-frame running speed (cm/s), padded to len(t_us) (repeats last value)."""
    dt_s = np.diff(t_us) * 1e-6
    with np.errstate(invalid='ignore', divide='ignore'):
        speed = np.hypot(np.diff(x_cm), np.diff(y_cm)) / dt_s
    speed[dt_s <= 0] = np.nan
    return np.append(speed, speed[-1])


def _plot_and_save_speed(csv_path: str, arena_width_cm: float) -> None:
    """Plot pre- vs. post-smoothing running speed for one tracking file and
    save the comparison next to it.
    """
    x_raw, y_raw, t = _load_tracking(csv_path, arena_width_cm)
    if len(t) < 2:
        print(f'  [SKIP] Not enough valid tracking samples to plot speed: {csv_path}')
        return

    x_smooth, y_smooth = _smooth_tracking_position(x_raw, y_raw, t)

    speed_raw    = _instantaneous_speed(x_raw,    y_raw,    t)
    speed_smooth = _instantaneous_speed(x_smooth, y_smooth, t)

    time_s = (t - t[0]) * 1e-6

    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(time_s, speed_raw,    linewidth=0.5, color='0.75', label='pre-smoothing (raw)')
    ax.plot(time_s, speed_smooth, linewidth=0.7, color='black', label='post-smoothing')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('speed (cm/s)')
    ax.set_title(os.path.splitext(os.path.basename(csv_path))[0])
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()

    csv_name  = os.path.splitext(os.path.basename(csv_path))[0]
    save_path = os.path.join(os.path.dirname(csv_path), f'{csv_name}_speed.png')
    fig.savefig(save_path, dpi=150)
    print(f'  [SAVED] {save_path}')


DT_S_TOL_MS = 1.0  # flag a frame interval as anomalous if it differs from the expected 1000/fps by more than this (ms)


def _plot_and_save_dt_s(csv_path: str, arena_width_cm: float, fps: float) -> None:
    """Plot the tracking inter-frame interval (dt_s) in the time domain for one
    tracking file and save it next to it, flagging intervals that deviate from
    the expected 1000/fps spacing (e.g. 33.33 ms at 30 fps).
    """
    _, _, t = _load_tracking(csv_path, arena_width_cm)
    if len(t) < 2:
        print(f'  [SKIP] Not enough valid tracking samples to plot dt_s: {csv_path}')
        return

    dt_ms  = np.diff(t) * 1e-3        # us -> ms
    time_s = (t[1:] - t[0]) * 1e-6    # time of each interval (end timestamp), relative to session start

    expected_ms  = 1000.0 / fps
    is_anomalous = np.abs(dt_ms - expected_ms) > DT_S_TOL_MS
    n_anom       = int(is_anomalous.sum())

    fig    = Figure()
    canvas = FigureCanvasAgg(fig)
    ax     = fig.add_subplot(111)
    ax.plot(time_s, dt_ms, marker='.', markersize=2, linestyle='None', color='0.4', alpha=0.5, label='dt_s')
    ax.axhline(expected_ms, color='green', linewidth=0.8, linestyle='--',
               label=f'expected ({expected_ms:.2f} ms)')
    if n_anom > 0:
        ax.scatter(time_s[is_anomalous], dt_ms[is_anomalous], color='red', s=10, zorder=3,
                   label=f'off by >{DT_S_TOL_MS:.1f} ms (n={n_anom})')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('inter-frame interval dt_s (ms)')
    ax.set_title(os.path.splitext(os.path.basename(csv_path))[0])
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()

    csv_name  = os.path.splitext(os.path.basename(csv_path))[0]
    save_path = os.path.join(os.path.dirname(csv_path), f'{csv_name}_dt_s.png')
    fig.savefig(save_path, dpi=150)

    pct = 100 * n_anom / len(dt_ms)
    print(f'  [SAVED] {save_path}   ({n_anom}/{len(dt_ms)} intervals off by >{DT_S_TOL_MS:.1f} ms, '
          f'{pct:.2f}%; range {dt_ms.min():.2f}-{dt_ms.max():.2f} ms)')


# ── Core metric computation ───────────────────────────────────────────────────

def compute_metrics(csv_path: str, ntt_path: str,
                    arena_width_cm: float, target_bin_cm: float,
                    half: str | None = None) -> tuple:

    # ── 1-2. Load, clean, and convert tracking ─────────────────────────────────
    x_cm, y_cm, t = _load_tracking(csv_path, arena_width_cm, half=half)

    if len(t) == 0:
         return ({'n_spikes': 0, 'n_discarded': 0, 'peak_fr': 0.0, 'mean_fr': 0.0, 'sir': 0.0}, {})

    # ── 2b. Smooth tracking position (jump removal + Gaussian smoothing) ──────
    x_cm, y_cm = _smooth_tracking_position(x_cm, y_cm, t)

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
        'speed_score': None, 'speed_p_value': None, 'speed_r2': None,
        'speed_beta': None, 'speed_f0': None, 'speed_modulated': None,
        'speed_shuffle_mean': None, 'speed_shuffle_lo': None,
        'speed_shuffle_hi': None, 'speed_shuffle_p': None,
        'speed_modulated_shuffle': None, 'speed_shuffle_ran': None,
        'speed_score_td': None, 'speed_p_value_td': None, 'speed_r2_td': None,
        'speed_shuffle_mean_td': None, 'speed_shuffle_lo_td': None,
        'speed_shuffle_hi_td': None, 'speed_shuffle_p_td': None,
        'speed_modulated_td': None, 'speed_shuffle_ran_td': None,
        'p_speed': None, 'n_speed': None,
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

        # Speed modulation — full session only (not run on first/second half splits).
        if half is None and ctx and ctx.get('spike_ts') is not None and len(ctx['spike_ts']) > 0:
            try:
                # Downsample tracking to 15 fps (keep every alternate frame) before
                # estimating speed and binning spikes for the speed-modulation fit —
                # see SPEED_MOD_DOWNSAMPLE_FACTOR.
                x_cm_ds = ctx['x_cm'][::SPEED_MOD_DOWNSAMPLE_FACTOR]
                y_cm_ds = ctx['y_cm'][::SPEED_MOD_DOWNSAMPLE_FACTOR]
                t_ds    = ctx['t'][::SPEED_MOD_DOWNSAMPLE_FACTOR]
                pos_fps_ds = fps / SPEED_MOD_DOWNSAMPLE_FACTOR
                speed_res = _compute_speed_modulation(
                    x_cm_ds, y_cm_ds, t_ds, ctx['spike_ts'], pos_fps_ds,
                    ntt_path=ntt_path, label=label,
                )
                metrics['speed_score']     = speed_res['speed_score']
                metrics['speed_p_value']   = speed_res['speed_p_value']
                metrics['speed_r2']        = speed_res['speed_r2']
                metrics['speed_beta']      = speed_res['speed_beta']
                metrics['speed_f0']        = speed_res['speed_f0']
                metrics['speed_modulated'] = speed_res['speed_modulated']
                metrics['speed_shuffle_mean']      = speed_res['speed_shuffle_mean']
                metrics['speed_shuffle_lo']        = speed_res['speed_shuffle_lo']
                metrics['speed_shuffle_hi']        = speed_res['speed_shuffle_hi']
                metrics['speed_shuffle_p']         = speed_res['speed_shuffle_p']
                metrics['speed_modulated_shuffle'] = speed_res['speed_modulated_shuffle']
                metrics['speed_shuffle_ran']       = speed_res['speed_shuffle_ran']
                metrics['speed_score_td']          = speed_res['speed_score_td']
                metrics['speed_p_value_td']        = speed_res['speed_p_value_td']
                metrics['speed_r2_td']             = speed_res['speed_r2_td']
                metrics['speed_shuffle_mean_td']   = speed_res['speed_shuffle_mean_td']
                metrics['speed_shuffle_lo_td']     = speed_res['speed_shuffle_lo_td']
                metrics['speed_shuffle_hi_td']     = speed_res['speed_shuffle_hi_td']
                metrics['speed_shuffle_p_td']      = speed_res['speed_shuffle_p_td']
                metrics['speed_modulated_td']      = speed_res['speed_modulated_td']
                metrics['speed_shuffle_ran_td']    = speed_res['speed_shuffle_ran_td']
                metrics['p_speed']                 = speed_res['p_speed']
                metrics['n_speed']                 = speed_res['n_speed']
            except Exception as e:
                with _print_lock:
                    print(f'  SPEED ERROR in {ntt_file} [{label}]: {e}')
                metrics['speed_score']     = None
                metrics['speed_p_value']   = None
                metrics['speed_r2']        = None
                metrics['speed_beta']      = None
                metrics['speed_f0']        = None
                metrics['speed_modulated'] = None
                metrics['speed_shuffle_mean']      = None
                metrics['speed_shuffle_lo']        = None
                metrics['speed_shuffle_hi']        = None
                metrics['speed_shuffle_p']         = None
                metrics['speed_modulated_shuffle'] = None
                metrics['speed_shuffle_ran']       = None
                metrics['speed_score_td']          = None
                metrics['speed_p_value_td']        = None
                metrics['speed_r2_td']             = None
                metrics['speed_shuffle_mean_td']   = None
                metrics['speed_shuffle_lo_td']     = None
                metrics['speed_shuffle_hi_td']     = None
                metrics['speed_shuffle_p_td']      = None
                metrics['speed_modulated_td']      = None
                metrics['speed_shuffle_ran_td']    = None
                metrics['p_speed']                 = None
                metrics['n_speed']                 = None
        else:
            metrics['speed_score']     = None
            metrics['speed_p_value']   = None
            metrics['speed_r2']        = None
            metrics['speed_beta']      = None
            metrics['speed_f0']        = None
            metrics['speed_modulated'] = None
            metrics['speed_shuffle_mean']      = None
            metrics['speed_shuffle_lo']        = None
            metrics['speed_shuffle_hi']        = None
            metrics['speed_shuffle_p']         = None
            metrics['speed_modulated_shuffle'] = None
            metrics['speed_shuffle_ran']       = None
            metrics['speed_score_td']          = None
            metrics['speed_p_value_td']        = None
            metrics['speed_r2_td']             = None
            metrics['speed_shuffle_mean_td']   = None
            metrics['speed_shuffle_lo_td']     = None
            metrics['speed_shuffle_hi_td']     = None
            metrics['speed_shuffle_p_td']      = None
            metrics['speed_modulated_td']      = None
            metrics['speed_shuffle_ran_td']    = None
            metrics['p_speed']                 = None
            metrics['n_speed']                 = None

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

    # ── Speed-vs-time plots (one per tracking file) ─────────────────────────────
    print(f'Generating speed and dt_s plots for {len(dir_to_csv)} tracking file(s)...')
    for csv_path in dir_to_csv.values():
        try:
            _plot_and_save_speed(csv_path, arena_width_cm)
        except Exception as e:
            print(f'  SPEED PLOT ERROR for {csv_path}: {e}')
        try:
            _plot_and_save_dt_s(csv_path, arena_width_cm, fps)
        except Exception as e:
            print(f'  DT_S PLOT ERROR for {csv_path}: {e}')
    print()

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
                    'speed_score', 'speed_p_value', 'speed_r2', 'speed_beta', 'speed_f0', 'speed_modulated',
                    'speed_shuffle_mean', 'speed_shuffle_lo', 'speed_shuffle_hi',
                    'speed_shuffle_p', 'speed_modulated_shuffle', 'speed_shuffle_ran',
                    'speed_score_td', 'speed_p_value_td', 'speed_r2_td',
                    'speed_shuffle_mean_td', 'speed_shuffle_lo_td', 'speed_shuffle_hi_td',
                    'speed_shuffle_p_td', 'speed_modulated_td', 'speed_shuffle_ran_td',
                    'p_speed', 'n_speed',
                    'place_cell']

    df_full   = pd.DataFrame([r[0] for r in results], columns=column_order)
    df_first  = pd.DataFrame([r[1] for r in results], columns=column_order)
    df_second = pd.DataFrame([r[2] for r in results], columns=column_order)

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_full.to_excel(writer,   sheet_name='Full',        index=False)
        df_first.to_excel(writer,  sheet_name='First_Half',  index=False)
        df_second.to_excel(writer, sheet_name='Second_Half', index=False)

    n_place_speed_mod = int(((df_full['place_cell'] == True) &                       # noqa: E712
                             (df_full['speed_modulated_shuffle'] == True)).sum())     # noqa: E712

    print(f'\nDone. Results saved to {output_excel}')
    print(f'Total units processed              : {len(df_full)}')
    print(f'Place cells found                  : {df_full["place_cell"].sum()}')
    print(f'Place cells also speed-modulated    : {n_place_speed_mod}  '
          f'(shuffle-confirmed, {SPEED_N_SHUFFLE} shuffles, '
          f'{SPEED_SHUFFLE_MARGIN_S:.0f}s window)')

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