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
from scipy.stats import pearsonr, linregress

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
    speed_beta:      float | None
    speed_f0:        float | None
    speed_modulated: bool | None
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

SPEED_MIN_CMS       = 4.0     # lowest speed (cm/s) included in speed-modulation analysis
SPEED_MAX_CMS       = 100.0   # highest speed (cm/s) included in speed-modulation analysis
SPEED_BIN_CMS       = 2.0     # width of each speed bin (cm/s)
SPEED_SMOOTH_S      = 0.08    # Gaussian smoothing window (s) applied to instantaneous firing rate
SPEED_MIN_BIN_FRAC  = 0.01    # discard speed bins holding < this fraction of samples

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
) -> dict:
    """Relates firing rate to running speed (ports MATLAB `speed_firing_runita`).

      1. Per-frame running speed (cm/s) from consecutive positions, using the
         actual inter-frame interval rather than an assumed fixed rate.
      2. Instantaneous firing rate on the same frame base (spike count per
         inter-frame interval / interval duration), Gaussian-smoothed.
      3. Restrict to samples with min_speed_cms < speed < max_speed_cms.
      4. Bin firing rate by speed (speed_bin_cms-wide bins), drop bins with
         < min_bin_frac of the samples, and fit rate = beta*speed + f0 to the
         binned means (beta/f0 naming kept from the MATLAB source).

    x_cm, y_cm, t_us must be same-length position tracks (t_us in µs, sorted).
    spike_ts_us must be in the same time base (µs).
    Returns speed_score (r), speed_p_value, speed_beta (slope),
    speed_f0 (intercept), speed_modulated (p < 0.05).
    """
    result = {'speed_score': float('nan'), 'speed_p_value': float('nan'),
              'speed_beta': float('nan'), 'speed_f0': float('nan'),
              'speed_modulated': None}

    n = len(t_us)
    if n < 3 or len(spike_ts_us) == 0:
        return result

    # ── Per-frame running speed (cm/s) ────────────────────────────────────────
    dt_s = np.diff(t_us) * 1e-6
    with np.errstate(invalid='ignore', divide='ignore'):
        speed = np.hypot(np.diff(x_cm), np.diff(y_cm)) / dt_s
    speed[dt_s <= 0] = np.nan
    speed = np.append(speed, speed[-1])              # pad to length n

    # ── Instantaneous firing rate on the same frame base ──────────────────────
    interval_idx = np.searchsorted(t_us, spike_ts_us, side='right') - 1
    interval_idx = np.clip(interval_idx, 0, n - 2)
    counts = np.bincount(interval_idx, minlength=n - 1).astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        fr_inst = counts / dt_s
    fr_inst[dt_s <= 0] = np.nan
    fr_inst = np.append(fr_inst, fr_inst[-1])         # pad to length n

    finite = np.isfinite(fr_inst)
    sigma_samples = max(smooth_window_s * pos_sample_rate_hz, 1e-6)
    fr_smooth = gaussian_filter1d(np.where(finite, fr_inst, 0.0),
                                   sigma=sigma_samples, mode='nearest')
    fr_smooth[~finite] = np.nan

    # ── Restrict to the usable speed range ─────────────────────────────────────
    in_range = (speed > min_speed_cms) & (speed < max_speed_cms) & np.isfinite(fr_smooth)
    if in_range.sum() < 3:
        return result

    speed_valid = speed[in_range]
    rate_valid  = fr_smooth[in_range]
    if np.std(speed_valid) == 0 or np.std(rate_valid) == 0:
        return result

    # ── Bin firing rate by speed ────────────────────────────────────────────────
    speed_bins = np.arange(min_speed_cms, max_speed_cms + speed_bin_cms, speed_bin_cms)
    n_bins     = len(speed_bins) - 1
    bin_idx    = np.clip(np.digitize(speed_valid, speed_bins) - 1, 0, n_bins - 1)

    bin_centres = 0.5 * (speed_bins[:-1] + speed_bins[1:])
    mean_rate   = np.full(n_bins, np.nan)
    n_per_bin   = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        sel = bin_idx == b
        n_per_bin[b] = int(sel.sum())
        if n_per_bin[b] > 0:
            mean_rate[b] = float(np.mean(rate_valid[sel]))

    total_pts = n_per_bin.sum()
    if total_pts == 0:
        return result
    mean_rate[(n_per_bin / total_pts) < min_bin_frac] = np.nan   # drop under-sampled bins

    fit_sel = np.isfinite(mean_rate)
    if fit_sel.sum() < 3:
        return result

    reg = linregress(bin_centres[fit_sel], mean_rate[fit_sel])

    result['speed_score']     = round(float(reg.rvalue), 4)
    result['speed_p_value']   = round(float(reg.pvalue), 4)
    result['speed_beta']      = round(float(reg.slope), 4)
    result['speed_f0']        = round(float(reg.intercept), 4)
    result['speed_modulated'] = bool(reg.pvalue < 0.05)
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
                )
                metrics['speed_score']     = speed_res['speed_score']
                metrics['speed_p_value']   = speed_res['speed_p_value']
                metrics['speed_beta']      = speed_res['speed_beta']
                metrics['speed_f0']        = speed_res['speed_f0']
                metrics['speed_modulated'] = speed_res['speed_modulated']
            except Exception as e:
                with _print_lock:
                    print(f'  SPEED ERROR in {ntt_file} [{label}]: {e}')
                metrics['speed_score']     = None
                metrics['speed_p_value']   = None
                metrics['speed_beta']      = None
                metrics['speed_f0']        = None
                metrics['speed_modulated'] = None
        else:
            metrics['speed_score']     = None
            metrics['speed_p_value']   = None
            metrics['speed_beta']      = None
            metrics['speed_f0']        = None
            metrics['speed_modulated'] = None

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
                    'place_cell']

    df_full   = pd.DataFrame([r[0] for r in results], columns=column_order)
    df_first  = pd.DataFrame([r[1] for r in results], columns=column_order)
    df_second = pd.DataFrame([r[2] for r in results], columns=column_order)

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_full.to_excel(writer,   sheet_name='Full',        index=False)
        df_first.to_excel(writer,  sheet_name='First_Half',  index=False)
        df_second.to_excel(writer, sheet_name='Second_Half', index=False)

    print(f'\nDone. Results saved to {output_excel}')
    print(f'Total units processed : {len(df_full)}')
    print(f'Place cells found     : {df_full["place_cell"].sum()}')

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