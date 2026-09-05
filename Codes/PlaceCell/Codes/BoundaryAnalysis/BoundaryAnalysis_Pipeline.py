# -*- coding: utf-8 -*-
"""
Boundary analysis pipeline – merges BoundaryAnalysis_open_v2.py,
BoundaryAnalysis_linearv3.py and
dist_from_wall_EdgeCentre_plot_proportion_plot_between_arenas_v2.py into a
single sequential run:

  1. Open-field boundary analysis  (circular arena geometry)
  2. Linear-track boundary analysis (linear-track geometry)
  3. Edge/centre comparison plots built from whichever of the two Excel
     outputs above are available

Each arena has its own hardcoded root folder / output path below. Set an
arena's *_ROOT_FOLDER to False to skip that arena's analysis entirely (the
comparison step then falls back to whatever data is available).
"""

import os
import threading
import time
import concurrent.futures
import types
import random
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.ndimage import convolve
from scipy.stats import pearsonr, ttest_ind

# Thread-safe Matplotlib imports for parallel rendering
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle


# ══════════════════════════════════════════════════════════════════════════════
# ── HARDCODED DIRECTORIES ── edit these paths, nothing else below needs to change
# Set a *_ROOT_FOLDER to False to skip that arena's boundary analysis entirely.
# ══════════════════════════════════════════════════════════════════════════════

OPEN_ROOT_FOLDER            = r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/Fa8477/Open'
OPEN_OUTPUT_EXCEL           = r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/Fa8477/Open/Open_BoundaryAnalysis_v2.xlsx'
OPEN_ARENA_WIDTH_CM         = 60.0   # physical arena diameter in cm (circular arena)
OPEN_EDGE_ZONE_THRESHOLD_CM = 9.25   # bins <= this distance from the wall are "edge"

LINEAR_ROOT_FOLDER            = r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True/Fa1059'
LINEAR_OUTPUT_EXCEL           = r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True/Fa1059/Linear_BoundaryAnalysis.xlsx'
LINEAR_ARENA_WIDTH_CM         = 80.0  # physical track length in cm
LINEAR_EDGE_ZONE_THRESHOLD_CM = 2.0   # bins closer than this to the mid-line are "centre"
LINEAR_TRACK_WIDTH_CM         = 8.0   # physical track width in cm

# Where the arena-comparison plots (step 3) are written. If an arena above is
# skipped (False), the comparison step falls back to single-arena plots using
# whichever Excel outputs exist.
COMPARISON_OUTPUT_DIR = r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data'

# ══════════════════════════════════════════════════════════════════════════════


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
    place_cell:      bool | None
    peak_bin_x_cm:   float | None
    peak_bin_y_cm:   float | None
    dist_from_wall_cm: float | None
    zone:            str | None
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
    cp_convolve = lambda *args, **kwargs: None                 # type: ignore[assignment]
    _GPU = False
    print("CuPy not found – running on CPU (install cupy-cuda12x to enable GPU).")
except Exception as _gpu_err:
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
# Shared analysis parameters (used identically for both arenas)

fps            = 30           # tracking frame rate (Hz)
target_bin_cm  = 2.0          # bin size in cm
min_occ_s      = 1.0          # exclude bins with < 1 s occupancy
MAX_GAP_US     = 50_000       # max spike–position gap in µs (50 ms)
N_BOOTSTRAP    = 1000         # circular-shift shuffles for SIR significance

AUTOCORR_WINDOW_MS  = 500.0   # autocorrelogram half-window (ms)
AUTOCORR_BIN_MS     = 5.0     # autocorrelogram bin size (ms)
THETA_POWER_THRESH  = 2.0     # theta peak must exceed N× mean spectrum power

MAX_GPU_UTIL_PCT = 60
MAX_WORKERS      = 4

# SUA / MUA classification thresholds
RPV_THRESHOLD_MS = 1.0    # refractory-period violation window (ms)
RPV_MAX_PCT      = 0.1    # max % ISI violations allowed for SUA
WF_SNR_MIN       = 3.0    # minimum waveform peak-to-peak SNR for SUA
WF_CV_MAX        = 0.4    # maximum waveform amplitude CV for SUA

# 'pixel' or 'cm' – set interactively at startup (see __main__ below).
# Tracking CSV columns (by position): A = time, B = x (pixel), C = y (pixel),
# D = x (cm), E = y (cm).
COORD_UNITS = 'pixel'

_gpu_semaphore = threading.Semaphore(2)


class ArenaConfig:
    """Bundles the per-arena settings that the shared analysis code branches on."""

    def __init__(self, arena_type: str, root_folder: str, output_excel: str,
                 arena_width_cm: float, edge_threshold_cm: float,
                 track_width_cm: float = 8.0):
        self.arena_type        = arena_type   # 'open' or 'linear'
        self.root_folder       = root_folder
        self.output_excel      = output_excel
        self.arena_width_cm    = arena_width_cm
        self.edge_threshold_cm = edge_threshold_cm
        self.track_width_cm    = track_width_cm


# ── Edge / Centre zone classification ────────────────────────────────────────

def classify_edge_centre(
    fr_smooth: np.ndarray,
    valid_mask: np.ndarray,
    n_bins_x: int,
    n_bins_y: int,
    target_bin_cm: float,
    cfg: ArenaConfig,
    max_x_cm: float | None = None,
    max_y_cm: float | None = None,
) -> dict:
    """Locate the peak-firing bin and classify it as 'edge' or 'centre'.

    Open field (circular arena):
      - The arena centre in cm is (radius, radius) where radius = diameter / 2.
      - Distance from the wall = arena_radius - distance_from_centre.
      - A bin is 'edge' when dist_from_wall <= edge_threshold_cm.

    Linear track:
      - Orientation is inferred from the tracking span (long axis = run direction).
      - A bin is 'edge' when its distance from the track's mid-line (short axis)
        is at least edge_threshold_cm, OR when its distance from either short
        (end) wall along the long axis is within edge_threshold_cm. Otherwise
        it is 'centre'.
    """
    nan = float('nan')
    result: dict = {
        'peak_bin_x_cm':    nan,
        'peak_bin_y_cm':    nan,
        'dist_from_wall_cm': nan,
        'zone':             None,
    }

    if not valid_mask.any():
        return result

    # Find the bin with the maximum smoothed firing rate among visited bins
    masked_fr = np.where(valid_mask, fr_smooth, -np.inf)
    flat_idx  = int(np.argmax(masked_fr))
    peak_bx, peak_by = np.unravel_index(flat_idx, (n_bins_x, n_bins_y))

    # Bin centre coordinates in cm
    peak_x_cm = (float(peak_bx) + 0.5) * target_bin_cm
    peak_y_cm = (float(peak_by) + 0.5) * target_bin_cm

    if cfg.arena_type == 'open':
        arena_radius_cm = cfg.arena_width_cm / 2.0
        centre_x_cm     = arena_radius_cm
        centre_y_cm     = arena_radius_cm

        dist_from_centre_cm = float(np.hypot(peak_x_cm - centre_x_cm,
                                             peak_y_cm - centre_y_cm))
        dist_from_wall_cm   = max(0.0, arena_radius_cm - dist_from_centre_cm)

        zone = 'edge' if dist_from_wall_cm <= cfg.edge_threshold_cm else 'centre'

    else:  # linear
        assert max_x_cm is not None and max_y_cm is not None
        # Determine orientation: Horizontal if X span > Y span
        if max_x_cm > max_y_cm:
            center_cm = max_y_cm / 2.0
            dist_from_center_cm = abs(peak_y_cm - center_cm)
            dist_from_end_wall_cm = min(peak_x_cm, max_x_cm - peak_x_cm)
        else:
            center_cm = max_x_cm / 2.0
            dist_from_center_cm = abs(peak_x_cm - center_cm)
            dist_from_end_wall_cm = min(peak_y_cm, max_y_cm - peak_y_cm)

        near_long_wall  = dist_from_center_cm >= cfg.edge_threshold_cm
        near_short_wall = dist_from_end_wall_cm <= cfg.edge_threshold_cm
        zone = 'edge' if (near_long_wall or near_short_wall) else 'centre'

        # Distance from closest wall (long or short side).
        dist_from_long_wall_cm = max(0.0, (cfg.track_width_cm / 2.0) - dist_from_center_cm)
        dist_from_wall_cm = min(dist_from_long_wall_cm, dist_from_end_wall_cm)

    result['peak_bin_x_cm']     = round(peak_x_cm,          3)
    result['peak_bin_y_cm']     = round(peak_y_cm,          3)
    result['dist_from_wall_cm'] = round(dist_from_wall_cm,  3)
    result['zone']               = zone
    return result


# ── SUA / MUA classification ──────────────────────────────────────────────────

def classify_sua_mua(waveforms: np.ndarray, spike_ts_us: np.ndarray) -> dict:
    """
    Classify a cluster as SUA or MUA.  SUA requires ALL three:
      1. RPV rate (% of ISIs < RPV_THRESHOLD_MS) < RPV_MAX_PCT
      2. Waveform SNR (mean peak-to-peak / 2×baseline noise SD) >= WF_SNR_MIN
      3. Waveform amplitude CV <= WF_CV_MAX

    waveforms shape: (n_spikes, 32_samples, 4_wires)
    Baseline noise is estimated from the first 4 pre-trigger samples.
    """
    nan = float('nan')
    result = {'is_sua': False, 'rpv_rate': nan, 'waveform_snr': nan, 'waveform_cv': nan}

    if len(spike_ts_us) < 2:
        return result

    isis_ms  = np.diff(np.sort(spike_ts_us)) / 1000.0
    rpv_rate = float(np.sum(isis_ms < RPV_THRESHOLD_MS) / len(isis_ms) * 100.0)
    result['rpv_rate'] = round(rpv_rate, 3)

    # Mean peak-to-peak amplitude per spike, averaged across the 4 tetrode wires
    pp_per_spike = (waveforms.max(axis=1) - waveforms.min(axis=1)).mean(axis=1)  # (n_spikes,)
    signal       = float(pp_per_spike.mean())

    # Noise SD from the first 4 pre-trigger samples (pre-spike baseline window)
    baseline_sd  = float(waveforms[:, :4, :].std())
    waveform_snr = round(signal / (2.0 * baseline_sd), 3) if baseline_sd > 0 else nan
    waveform_cv  = round(float(pp_per_spike.std() / signal), 3) if signal > 0 else nan

    result['waveform_snr'] = waveform_snr
    result['waveform_cv']  = waveform_cv

    is_sua = (
        rpv_rate < RPV_MAX_PCT
        and not np.isnan(waveform_snr) and waveform_snr >= WF_SNR_MIN
        and not np.isnan(waveform_cv)  and waveform_cv  <= WF_CV_MAX
    )
    result['is_sua'] = bool(is_sua)
    return result


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


# ── Stability (place-field split-half Pearson R) ──────────────────────────────

def _compute_stability(
    map_a: np.ndarray, valid_a: np.ndarray,
    map_b: np.ndarray, valid_b: np.ndarray,
) -> float:
    """Pixel-by-pixel Pearson R between two smoothed rate maps.

    Only bins valid (visited) in both maps and within the overlapping grid
    region are included, matching the standard split-half stability method.
    """
    rows = min(map_a.shape[0], map_b.shape[0])
    cols = min(map_a.shape[1], map_b.shape[1])

    a  = map_a[:rows, :cols]
    b  = map_b[:rows, :cols]
    va = valid_a[:rows, :cols]
    vb = valid_b[:rows, :cols]

    both_valid = va & vb
    if int(both_valid.sum()) < 3:
        return float('nan')

    a_flat = a[both_valid]
    b_flat = b[both_valid]

    if float(np.std(a_flat)) == 0.0 or float(np.std(b_flat)) == 0.0:
        return float('nan')

    r, _ = pearsonr(a_flat, b_flat)
    return round(float(r), 4)  # type: ignore


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
    random number of frames (at least 1 s from either end), while spike-to-frame
    assignments remain unchanged.  This decorrelates spikes from positions without
    altering the animal's occupancy statistics.
    """
    if len(spike_frame_indices) == 0:
        return {'bootstrap_mean': float('nan'),
                'bootstrap_p95':  float('nan'),
                'bootstrap_sig':  False}

    n_frames      = len(t)
    MARGIN_FRAMES = int(fps)   # 1 second of frames at either end

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
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    hist_result = ax.hist(sir_i, 100, color='black')
    counts: np.ndarray = np.asarray(hist_result[0])
    max_count = float(counts.max()) if counts.max() > 0 else 1.0

    box_plot = ax.boxplot(sir_i, whis=[5, 95], vert=False, showfliers=False,  # type: ignore
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


# ── Tracking load / bin (shared by spike metrics and the occupancy pre-check) ─

def _load_and_bin_tracking(csv_path: str, arena_width_cm: float, target_bin_cm: float,
                           half: str | None = None):
    """Load, clean, optionally slice (half/quartile) and spatially bin tracking data.

    Returns (t, beh_bx, beh_by, n_bins_x, n_bins_y, max_x_cm, max_y_cm) or None if
    no samples remain.
    """
    # ── 1. Load & clean tracking ──────────────────────────────────────────────
    data = pd.read_csv(csv_path)

    t = np.asarray(data.iloc[:, 0], dtype=float)
    if COORD_UNITS == 'cm':
        x = np.asarray(data.iloc[:, 3], dtype=float)   # column D
        y = np.asarray(data.iloc[:, 4], dtype=float)   # column E
    else:
        x = np.asarray(data.iloc[:, 1], dtype=float)   # column B
        y = np.asarray(data.iloc[:, 2], dtype=float)   # column C

    mask = ~np.isin(x, [1, -1])
    x, y, t = x[mask], y[mask], t[mask]

    # Pad the differential arrays with zeros/ones to prevent dropping the final frame
    dx = np.append(np.diff(x), 0)
    dy = np.append(np.diff(y), 0)
    dt = np.append(np.diff(t), 1)  # pad with 1 to avoid div-by-zero on last frame

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
        n   = len(t)
        mid = n // 2
        q   = n // 4
        if half == 'first':
            x, y, t = x[:mid],    y[:mid],    t[:mid]
        elif half == 'second':
            x, y, t = x[mid:],    y[mid:],    t[mid:]
        elif half == 'q1':
            x, y, t = x[:q],      y[:q],      t[:q]
        elif half == 'q2':
            x, y, t = x[q:2*q],   y[q:2*q],   t[q:2*q]
        elif half == 'q3':
            x, y, t = x[2*q:3*q], y[2*q:3*q], t[2*q:3*q]
        elif half == 'q4':
            x, y, t = x[3*q:],    y[3*q:],    t[3*q:]

    if len(t) == 0:
        return None

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

    return t, beh_bx, beh_by, n_bins_x, n_bins_y, float(x_cm.max()), float(y_cm.max())


def _build_occupancy_map(t: np.ndarray, beh_bx: np.ndarray, beh_by: np.ndarray,
                         n_bins_x: int, n_bins_y: int) -> np.ndarray:
    """Build the occupancy-time map (seconds per bin) from binned tracking frames."""
    dt_frames     = np.empty(len(t), dtype=np.float64)
    dt_frames[0]  = 1.0 / fps
    raw_dt        = np.diff(t) * 1e-6

    # Cap dt_frames to avoid artificial occupancy hotspots when tracking drops
    max_frame_s   = 2.0 / fps
    dt_frames[1:] = np.minimum(raw_dt, max_frame_s)

    occ_map = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    np.add.at(occ_map, (beh_bx, beh_by), dt_frames)
    return occ_map


def _zone_occupancy_pct(occ_map: np.ndarray, valid_mask: np.ndarray,
                        n_bins_x: int, n_bins_y: int,
                        target_bin_cm: float, cfg: ArenaConfig,
                        max_x_cm: float | None = None,
                        max_y_cm: float | None = None) -> tuple[float, float]:
    """Return (edge_pct, centre_pct) of total occupied time spent in each zone."""
    total_occ_s = float(occ_map[valid_mask].sum())
    if total_occ_s <= 0:
        return 0.0, 0.0

    bx_idx, by_idx = np.meshgrid(np.arange(n_bins_x), np.arange(n_bins_y), indexing='ij')
    bin_x_cm = (bx_idx.astype(np.float64) + 0.5) * target_bin_cm
    bin_y_cm = (by_idx.astype(np.float64) + 0.5) * target_bin_cm

    if cfg.arena_type == 'open':
        arena_radius_cm = cfg.arena_width_cm / 2.0
        dist_from_centre_cm = np.hypot(bin_x_cm - arena_radius_cm, bin_y_cm - arena_radius_cm)
        dist_from_wall_cm   = np.maximum(0.0, arena_radius_cm - dist_from_centre_cm)

        edge_mask    = (dist_from_wall_cm <= cfg.edge_threshold_cm) & valid_mask
        edge_occ_s   = float(occ_map[edge_mask].sum())
        centre_occ_s = total_occ_s - edge_occ_s

    else:  # linear
        assert max_x_cm is not None and max_y_cm is not None
        # Determine orientation: Horizontal if X span > Y span
        if max_x_cm > max_y_cm:
            centre_line_cm         = max_y_cm / 2.0
            dist_from_centre_cm    = np.abs(bin_y_cm - centre_line_cm)
            dist_from_end_wall_cm  = np.minimum(bin_x_cm, max_x_cm - bin_x_cm)
        else:
            centre_line_cm         = max_x_cm / 2.0
            dist_from_centre_cm    = np.abs(bin_x_cm - centre_line_cm)
            dist_from_end_wall_cm  = np.minimum(bin_y_cm, max_y_cm - bin_y_cm)

        edge_mask    = ((dist_from_centre_cm >= cfg.edge_threshold_cm) |
                        (dist_from_end_wall_cm <= cfg.edge_threshold_cm)) & valid_mask
        edge_occ_s   = float(occ_map[edge_mask].sum())
        centre_occ_s = total_occ_s - edge_occ_s

    return 100.0 * edge_occ_s / total_occ_s, 100.0 * centre_occ_s / total_occ_s


def compute_session_zone_occupancy(csv_path: str, cfg: ArenaConfig):
    """Compute the session's edge-vs-centre occupancy-time split from tracking alone.

    Returns (edge_pct, centre_pct, occ_map, valid_mask, n_bins_x, n_bins_y,
    max_x_cm, max_y_cm), or None if the tracking file yields no valid samples.
    """
    track = _load_and_bin_tracking(csv_path, cfg.arena_width_cm, target_bin_cm, half=None)
    if track is None:
        return None
    t, beh_bx, beh_by, n_bins_x, n_bins_y, max_x_cm, max_y_cm = track

    occ_map    = _build_occupancy_map(t, beh_bx, beh_by, n_bins_x, n_bins_y)
    valid_mask = occ_map >= min_occ_s

    edge_pct, centre_pct = _zone_occupancy_pct(occ_map, valid_mask, n_bins_x, n_bins_y,
                                               target_bin_cm, cfg,
                                               max_x_cm=max_x_cm, max_y_cm=max_y_cm)
    return edge_pct, centre_pct, occ_map, valid_mask, n_bins_x, n_bins_y, max_x_cm, max_y_cm


def _plot_occupancy_map(occ_map: np.ndarray, valid_mask: np.ndarray,
                        target_bin_cm: float, cfg: ArenaConfig,
                        edge_pct: float, centre_pct: float, csv_path: str,
                        max_x_cm: float | None = None,
                        max_y_cm: float | None = None) -> str:
    """Render and save a 2-D occupancy-time heat map annotated with the edge/centre split."""
    n_bins_x, n_bins_y = occ_map.shape
    plot_map = np.where(valid_mask, occ_map, np.nan)

    fig    = Figure(figsize=(6, 5))
    canvas = FigureCanvasAgg(fig)
    ax     = fig.add_subplot(111)

    extent = [0, n_bins_x * target_bin_cm, 0, n_bins_y * target_bin_cm]
    im = ax.imshow(plot_map.T, origin='lower', extent=extent, cmap='viridis')
    fig.colorbar(im, ax=ax, label='occupancy (s)')

    if cfg.arena_type == 'open':
        arena_radius_cm = cfg.arena_width_cm / 2.0
        inner_radius_cm = max(0.0, arena_radius_cm - cfg.edge_threshold_cm)
        centre_xy = (arena_radius_cm, arena_radius_cm)
        ax.add_patch(Circle(centre_xy, inner_radius_cm, fill=False,
                            edgecolor='red', linewidth=1.5, linestyle='--'))
        ax.add_patch(Circle(centre_xy, arena_radius_cm, fill=False,
                            edgecolor='white', linewidth=1.0))
    else:  # linear
        assert max_x_cm is not None and max_y_cm is not None
        if max_x_cm > max_y_cm:
            centre_line_cm = max_y_cm / 2.0
            ax.axhline(centre_line_cm - cfg.edge_threshold_cm, color='red', linewidth=1.5, linestyle='--')
            ax.axhline(centre_line_cm + cfg.edge_threshold_cm, color='red', linewidth=1.5, linestyle='--')
            ax.axvline(cfg.edge_threshold_cm,               color='red', linewidth=1.5, linestyle='--')
            ax.axvline(max_x_cm - cfg.edge_threshold_cm,     color='red', linewidth=1.5, linestyle='--')
        else:
            centre_line_cm = max_x_cm / 2.0
            ax.axvline(centre_line_cm - cfg.edge_threshold_cm, color='red', linewidth=1.5, linestyle='--')
            ax.axvline(centre_line_cm + cfg.edge_threshold_cm, color='red', linewidth=1.5, linestyle='--')
            ax.axhline(cfg.edge_threshold_cm,               color='red', linewidth=1.5, linestyle='--')
            ax.axhline(max_y_cm - cfg.edge_threshold_cm,     color='red', linewidth=1.5, linestyle='--')

    ax.set_xlim(0, n_bins_x * target_bin_cm)
    ax.set_ylim(0, n_bins_y * target_bin_cm)
    ax.set_aspect('equal')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_title(f'Occupancy map\nEdge zone: {edge_pct:.1f}%   Centre zone: {centre_pct:.1f}%')
    fig.tight_layout()

    csv_name  = os.path.splitext(os.path.basename(csv_path))[0]
    save_dir  = os.path.join(os.path.dirname(csv_path), 'occupancy_maps')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{csv_name}_occupancy_map.png')
    fig.savefig(save_path, dpi=150)
    print(f'  [SAVED] {save_path}')
    return save_path


# ── Core metric computation ───────────────────────────────────────────────────

def compute_metrics(csv_path: str, ntt_path: str, cfg: ArenaConfig,
                    half: str | None = None) -> tuple:

    track = _load_and_bin_tracking(csv_path, cfg.arena_width_cm, target_bin_cm, half=half)
    if track is None:
        return ({'n_spikes': 0, 'n_discarded': 0, 'peak_fr': 0.0, 'mean_fr': 0.0, 'sir': 0.0}, {})
    t, beh_bx, beh_by, n_bins_x, n_bins_y, max_x_cm, max_y_cm = track

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
    occ_map   = _build_occupancy_map(t, beh_bx, beh_by, n_bins_x, n_bins_y)
    spike_map = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)

    np.add.at(spike_map, (sp_bx, sp_by), 1.0)

    valid_mask = occ_map >= min_occ_s

    # ── 6. Non-smoothed firing rate map ──────────────────────────────────────
    fr_raw = np.zeros_like(occ_map)
    fr_raw[valid_mask] = spike_map[valid_mask] / occ_map[valid_mask]

    # ── 7. Smoothed firing rate map ───────────────────────────────────────────
    fr_smooth = _triangular_smooth(fr_raw, valid_mask)

    # ── 8. Compute metrics ────────────────────────────────────────────────────
    ctx = dict(spike_ts=spike_ts[valid_spike], spike_frame=spike_frame, t=t,
               beh_bx=beh_bx, beh_by=beh_by,
               occ_map=occ_map, valid_mask=valid_mask,
               fr_smooth=fr_smooth,
               n_bins_x=n_bins_x, n_bins_y=n_bins_y,
               max_x_cm=max_x_cm, max_y_cm=max_y_cm)

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
        r_coef    = float(np.clip(r_coef, -0.9999, 0.9999))  # type: ignore
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

def _run_job(args, cfg: ArenaConfig):
    unit_idx, total_units, job_order, dirpath, csv_path, ntt_file = args
    session_name = os.path.relpath(dirpath, cfg.root_folder)
    ntt_path     = os.path.join(dirpath, ntt_file)
    pct          = 100 * unit_idx / total_units

    with _print_lock:
        print(f'[{cfg.arena_type.upper()} {unit_idx}/{total_units}  {pct:.1f}%]  {session_name}  |  {ntt_file}  '
              f'(GPU {_gpu_util_pct()}%)')

    # ── SUA / MUA pre-filter ──────────────────────────────────────────────────
    try:
        _spike_data = np.memmap(ntt_path, dtype=ntt_dtype, mode='r', offset=16 * 1024)
        _waveforms  = _spike_data['waveforms'].astype(np.float64)
        _spike_ts   = _spike_data['timestamp'].astype(np.float64)
        sua_result  = classify_sua_mua(_waveforms, _spike_ts)
        del _spike_data, _waveforms, _spike_ts
    except Exception as _e:
        with _print_lock:
            print(f'  SUA CHECK ERROR in {ntt_file}: {_e}')
        sua_result = {'is_sua': None, 'rpv_rate': None, 'waveform_snr': None, 'waveform_cv': None}

    with _print_lock:
        print(f'  SUA: RPV={sua_result["rpv_rate"]}%  SNR={sua_result["waveform_snr"]}  '
              f'CV={sua_result["waveform_cv"]}  ->  '
              f'{"SUA (pass)" if sua_result["is_sua"] else "MUA – skipped"}')

    _err_row: dict = {
        'is_sua': sua_result.get('is_sua'), 'rpv_rate': sua_result.get('rpv_rate'),
        'waveform_snr': sua_result.get('waveform_snr'), 'waveform_cv': sua_result.get('waveform_cv'),
        'n_spikes': None, 'n_discarded': None,
        'peak_fr':  None, 'mean_fr':     None, 'sir': None,
        'sparsity': None, 'coherence':   None,
        'bootstrap_mean': None, 'bootstrap_p95': None, 'bootstrap_sig': None,
        'theta_modulated': None, 'theta_peak_freq': None,
        'stability_full_vs_first': None, 'stability_full_vs_second': None,
        'stability_first_vs_second': None,
        'stability_q1_vs_q2': None, 'stability_q3_vs_q4': None,
        'stability_q1_vs_q3': None, 'stability_q2_vs_q4': None,
        'session': session_name, 'unit': ntt_file,
        'job_order': job_order, 'place_cell': None,
        'peak_bin_x_cm': None, 'peak_bin_y_cm': None,
        'dist_from_wall_cm': None, 'zone': None,
    }

    if sua_result.get('is_sua') is False:
        return (dict(_err_row), dict(_err_row), dict(_err_row))

    def _build_row(half: str | None, label: str) -> tuple[dict, dict]:
        try:
            metrics, ctx = compute_metrics(csv_path, ntt_path, cfg, half=half)
        except Exception as e:
            with _print_lock:
                print(f'  ERROR in {ntt_file} [{label}]: {e}')
            return dict(_err_row), {}

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

        # ── Edge / Centre zone classification (full session only) ─────────────
        if half is None and ctx:
            zone_info = classify_edge_centre(
                fr_smooth       = ctx['fr_smooth'],
                valid_mask      = ctx['valid_mask'],
                n_bins_x        = ctx['n_bins_x'],
                n_bins_y        = ctx['n_bins_y'],
                target_bin_cm   = target_bin_cm,
                cfg             = cfg,
                max_x_cm        = ctx['max_x_cm'],
                max_y_cm        = ctx['max_y_cm'],
            )
        else:
            zone_info = {'peak_bin_x_cm': None, 'peak_bin_y_cm': None,
                         'dist_from_wall_cm': None, 'zone': None}

        metrics['peak_bin_x_cm']     = zone_info['peak_bin_x_cm']
        metrics['peak_bin_y_cm']     = zone_info['peak_bin_y_cm']
        metrics['dist_from_wall_cm'] = zone_info['dist_from_wall_cm']
        metrics['zone']              = zone_info['zone']

        return metrics, ctx

    full_row,   full_ctx   = _build_row(None,     'full')
    first_row,  first_ctx  = _build_row('first',  'first_half')
    second_row, second_ctx = _build_row('second', 'second_half')

    # ── Quartile rate maps (no bootstrap needed – used only for stability) ────
    def _get_quartile_ctx(segment: str) -> dict:
        try:
            _, ctx = compute_metrics(csv_path, ntt_path, cfg, half=segment)
            return ctx
        except Exception as e:
            with _print_lock:
                print(f'  ERROR computing {segment} map for {ntt_file}: {e}')
            return {}

    q1_ctx = _get_quartile_ctx('q1')
    q2_ctx = _get_quartile_ctx('q2')
    q3_ctx = _get_quartile_ctx('q3')
    q4_ctx = _get_quartile_ctx('q4')

    # ── Stability: pairwise Pearson R on smoothed rate maps ───────────────────
    def _safe_stability(ctx_a: dict, ctx_b: dict) -> float | None:
        map_a = ctx_a.get('fr_smooth')
        map_b = ctx_b.get('fr_smooth')
        va    = ctx_a.get('valid_mask')
        vb    = ctx_b.get('valid_mask')
        if map_a is None or map_b is None or va is None or vb is None:
            return None
        try:
            val = _compute_stability(map_a, va, map_b, vb)
            return None if (val != val) else val  # convert nan to None for Excel
        except Exception:
            return None

    full_row['stability_full_vs_first']   = _safe_stability(full_ctx, first_ctx)
    full_row['stability_full_vs_second']  = _safe_stability(full_ctx, second_ctx)
    full_row['stability_first_vs_second'] = _safe_stability(first_ctx, second_ctx)
    full_row['stability_q1_vs_q2']        = _safe_stability(q1_ctx, q2_ctx)
    full_row['stability_q3_vs_q4']        = _safe_stability(q3_ctx, q4_ctx)
    full_row['stability_q1_vs_q3']        = _safe_stability(q1_ctx, q3_ctx)
    full_row['stability_q2_vs_q4']        = _safe_stability(q2_ctx, q4_ctx)

    for _row in (full_row, first_row, second_row):
        _row['is_sua']       = sua_result.get('is_sua')
        _row['rpv_rate']     = sua_result.get('rpv_rate')
        _row['waveform_snr'] = sua_result.get('waveform_snr')
        _row['waveform_cv']  = sua_result.get('waveform_cv')

    return (full_row, first_row, second_row)


# ── Batch scan (one arena) ────────────────────────────────────────────────────

def run_boundary_analysis(cfg: ArenaConfig) -> str | None:
    """Runs the full boundary-analysis batch for one arena and writes its Excel
    output. Returns the output_excel path, or None if nothing was processed."""

    print(f'\n{"="*80}\n{cfg.arena_type.upper()} FIELD BOUNDARY ANALYSIS\n'
          f'Root folder: {cfg.root_folder}\n{"="*80}')

    all_jobs = []
    occupancy_records = []
    output_excel_basename = os.path.basename(cfg.output_excel).lower()
    for dirpath, _, filenames in os.walk(cfg.root_folder):
        tracking_files = [f for f in filenames
                          if f.lower().endswith('.csv')
                          and f.lower() != output_excel_basename]
        ntt_files      = [f for f in filenames if f.lower().endswith('.ntt')]
        if len(tracking_files) == 1 and len(ntt_files) > 0:
            csv_path     = os.path.join(dirpath, tracking_files[0])
            session_name = os.path.relpath(dirpath, cfg.root_folder)

            try:
                occ_result = compute_session_zone_occupancy(csv_path, cfg)
            except Exception as _e:
                print(f'  OCCUPANCY MAP ERROR in {session_name}: {_e}')
                occ_result = None

            if occ_result is not None:
                edge_pct, centre_pct, occ_map, valid_mask, _n_bx, _n_by, _max_x, _max_y = occ_result
                print(f'  {session_name}: edge zone = {edge_pct:.1f}%   centre zone = {centre_pct:.1f}%')
                try:
                    _plot_occupancy_map(occ_map, valid_mask, target_bin_cm, cfg,
                                        edge_pct, centre_pct, csv_path,
                                        max_x_cm=_max_x, max_y_cm=_max_y)
                except Exception as _e:
                    print(f'  OCCUPANCY PLOT ERROR in {session_name}: {_e}')
                occupancy_records.append({
                    'session': session_name, 'tracking_file': tracking_files[0],
                    'edge_pct': round(edge_pct, 2), 'centre_pct': round(centre_pct, 2),
                })
            else:
                occupancy_records.append({
                    'session': session_name, 'tracking_file': tracking_files[0],
                    'edge_pct': None, 'centre_pct': None,
                })

            for ntt_file in sorted(ntt_files):
                all_jobs.append((dirpath, csv_path, ntt_file))

    total_units = len(all_jobs)
    print(f'Found {total_units} unit(s) across all {cfg.arena_type} sessions.\n')

    job_args = [
        (idx, total_units, idx - 1, dirpath, csv_path, ntt_file)
        for idx, (dirpath, csv_path, ntt_file) in enumerate(all_jobs, start=1)
    ]

    # ── Parallel execution ────────────────────────────────────────────────────

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_run_job, args, cfg): args for args in job_args}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: r[0]['job_order'])

    # ── Save to Excel ─────────────────────────────────────────────────────────

    _base_cols = ['session', 'unit',
                  'is_sua', 'rpv_rate', 'waveform_snr', 'waveform_cv',
                  'n_spikes', 'n_discarded',
                  'peak_fr', 'mean_fr', 'sir', 'sparsity', 'coherence',
                  'bootstrap_mean', 'bootstrap_p95', 'bootstrap_sig',
                  'theta_modulated', 'theta_peak_freq', 'place_cell',
                  'peak_bin_x_cm', 'peak_bin_y_cm', 'dist_from_wall_cm', 'zone']

    column_order_full = _base_cols + [
        'stability_full_vs_first', 'stability_full_vs_second', 'stability_first_vs_second',
        'stability_q1_vs_q2', 'stability_q3_vs_q4',
        'stability_q1_vs_q3', 'stability_q2_vs_q4',
    ]
    column_order_half = [c for c in _base_cols
                         if c not in ('peak_bin_x_cm', 'peak_bin_y_cm',
                                      'dist_from_wall_cm', 'zone')]

    df_full   = pd.DataFrame([r[0] for r in results], columns=column_order_full)
    df_first  = pd.DataFrame([r[1] for r in results], columns=column_order_half)
    df_second = pd.DataFrame([r[2] for r in results], columns=column_order_half)

    def _summary_stats(df: pd.DataFrame) -> dict:
        total        = len(df)
        n_sua        = int(df['is_sua'].eq(True).sum())
        n_mua        = int(df['is_sua'].eq(False).sum())
        place        = int(df['place_cell'].sum())
        theta        = int(df['theta_modulated'].sum())
        theta_place  = int((df['place_cell'] & df['theta_modulated']).sum())
        return {
            'Metric':  ['Total units processed',
                        'SUA',
                        'MUA (excluded)',
                        'Place cells',
                        'Theta-modulated cells',
                        'Theta-modulated place cells'],
            'Count':   [total, n_sua, n_mua, place, theta, theta_place],
            'Percent': [
                '100%',
                f'{100 * n_sua / total:.1f}%'        if total else '—',
                f'{100 * n_mua / total:.1f}%'        if total else '—',
                f'{100 * place / total:.1f}%'        if total else '—',
                f'{100 * theta / total:.1f}%'        if total else '—',
                f'{100 * theta_place / total:.1f}%'  if total else '—',
            ],
        }

    # ── Edge vs Centre t-test (place cells only, full session) ────────────────
    pc_full       = df_full[df_full['place_cell'].eq(True)].copy()
    n_edge        = int((pc_full['zone'] == 'edge').sum())
    n_centre      = int((pc_full['zone'] == 'centre').sum())

    edge_dists   = pc_full.loc[pc_full['zone'] == 'edge',   'dist_from_wall_cm'].dropna().values
    centre_dists = pc_full.loc[pc_full['zone'] == 'centre', 'dist_from_wall_cm'].dropna().values

    if len(edge_dists) >= 2 and len(centre_dists) >= 2:
        t_stat, p_val = ttest_ind(edge_dists, centre_dists, equal_var=False)
        t_stat = round(float(t_stat), 4)  # type: ignore
        p_val  = round(float(p_val),  4)  # type: ignore
        sig    = 'Yes' if p_val < 0.05 else 'No'
    else:
        t_stat, p_val, sig = float('nan'), float('nan'), 'Insufficient data'

    df_zone_ttest = pd.DataFrame({
        'Metric': [
            'Arena width/diameter (cm)',
            'Edge threshold (cm)',
            'Place cells – edge zone',
            'Place cells – centre zone',
            'Mean dist_from_wall edge (cm)',
            'Mean dist_from_wall centre (cm)',
            'Welch t-statistic',
            'p-value (two-tailed)',
            'Significant (p < 0.05)',
        ],
        'Value': [
            cfg.arena_width_cm,
            cfg.edge_threshold_cm,
            n_edge,
            n_centre,
            round(float(edge_dists.mean()),   3) if len(edge_dists)   > 0 else float('nan'),  # type: ignore
            round(float(centre_dists.mean()), 3) if len(centre_dists) > 0 else float('nan'),  # type: ignore
            t_stat,
            p_val,
            sig,
        ],
    })

    df_summary_full   = pd.DataFrame(_summary_stats(df_full))
    df_summary_first  = pd.DataFrame(_summary_stats(df_first))
    df_summary_second = pd.DataFrame(_summary_stats(df_second))
    df_occupancy      = pd.DataFrame(occupancy_records,
                                     columns=['session', 'tracking_file', 'edge_pct', 'centre_pct'])

    with pd.ExcelWriter(cfg.output_excel, engine='openpyxl') as writer:
        df_full.to_excel(writer,          sheet_name='Full',              index=False)
        df_first.to_excel(writer,         sheet_name='First_Half',        index=False)
        df_second.to_excel(writer,        sheet_name='Second_Half',       index=False)
        df_summary_full.to_excel(writer,  sheet_name='Summary_Full',      index=False)
        df_summary_first.to_excel(writer, sheet_name='Summary_First',     index=False)
        df_summary_second.to_excel(writer,sheet_name='Summary_Second',    index=False)
        df_zone_ttest.to_excel(writer,    sheet_name='Zone_EdgeCentre',   index=False)
        df_occupancy.to_excel(writer,     sheet_name='Occupancy',         index=False)

    n_sua         = int(df_full['is_sua'].eq(True).sum())
    n_mua         = int(df_full['is_sua'].eq(False).sum())
    n_theta       = int(df_full['theta_modulated'].sum())
    n_theta_place = int((df_full['place_cell'] & df_full['theta_modulated']).sum())
    print(f'\nDone. Results saved to {cfg.output_excel}')
    print(f'Total units processed        : {len(df_full)}')
    print(f'SUA                          : {n_sua}')
    print(f'MUA (excluded)               : {n_mua}')
    print(f'Place cells found            : {df_full["place_cell"].sum()}')
    print(f'Theta-modulated cells        : {n_theta}')
    print(f'Theta-modulated place cells  : {n_theta_place}')
    print(f'\n── Edge / Centre zone (place cells, full session) ──')
    print(f'Edge zone cells (threshold {cfg.edge_threshold_cm} cm) : {n_edge}')
    print(f'Centre zone cells            : {n_centre}')
    print(f'Welch t-statistic            : {t_stat}')
    print(f'p-value (two-tailed)         : {p_val}')
    print(f'Significant (p < 0.05)       : {sig}')

    return cfg.output_excel


# ── Step 3: Edge/centre comparison plots ──────────────────────────────────────
# (adapted from dist_from_wall_EdgeCentre_plot_proportion_plot_between_arenas_v2.py)

mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'DejaVu Sans'],
    'font.size':         11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})

COLOR_EDGE   = '#9B87BE'   # light purple
COLOR_CENTRE = '#6B7AAD'   # blue-slate


def load_zone_data(excel_file: str):
    """Load dist_from_wall and zone columns from an Excel file, filtered to place cells."""
    df = pd.read_excel(excel_file, sheet_name=0, header=0)
    col_dist = df.columns[21]   # dist_from_wall_cm  (column V)
    col_zone = df.columns[22]   # zone               (column W)

    dist = df[col_dist].dropna()
    zone = df[col_zone].loc[dist.index]

    if 'place_cell' in df.columns:
        pc_mask = df['place_cell'].loc[dist.index].eq(True)
        dist    = dist[pc_mask]
        zone    = zone[pc_mask]

    edge_vals = np.asarray(dist[zone.str.lower().str.contains('edge', na=False)], dtype=float)
    cent_vals = np.asarray(dist[zone.str.lower().str.contains('cent', na=False)], dtype=float)
    return edge_vals, cent_vals


def plot_zone_boxplot(excel_file: str, arena_label: str, output_png: str):
    """Boxplot of dist_from_wall split by edge/centre zone, for one arena."""
    edge_vals, cent_vals = load_zone_data(excel_file)
    if len(edge_vals) == 0 or len(cent_vals) == 0:
        print(f'  [SKIP] {arena_label}: not enough edge/centre place cells to plot '
              f'(n_edge={len(edge_vals)}, n_centre={len(cent_vals)}).')
        return

    u_stat, pval = stats.mannwhitneyu(edge_vals, cent_vals, alternative='two-sided')
    n_edge, n_cent = len(edge_vals), len(cent_vals)

    if pval < 0.001:
        sig_tag = '***'; p_label = 'p<0.001'
    elif pval < 0.01:
        sig_tag = '**';  p_label = f'p={pval:.3f}'
    elif pval < 0.05:
        sig_tag = '*';   p_label = f'p={pval:.3f}'
    else:
        sig_tag = 'ns';  p_label = f'p={pval:.3f}'

    fig, ax = plt.subplots(figsize=(4.5, 6))

    groups      = [edge_vals, cent_vals]
    group_names = ['Edge', 'Centre']
    colors      = [COLOR_EDGE, COLOR_CENTRE]

    bp = ax.boxplot(
        groups,
        positions=[1, 2],
        patch_artist=True,
        widths=0.45,
        showfliers=False,
        medianprops=dict(color='black', linewidth=2.5),
        whiskerprops=dict(linewidth=1.5, color='black'),
        capprops=dict(linewidth=1.5, color='black'),
        boxprops=dict(linewidth=1.5),
    )

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(groups, colors), start=1):
        jitter = rng.uniform(-0.13, 0.13, size=len(vals))
        ax.scatter(i + jitter, vals,
                   color=color, alpha=0.55, s=20, zorder=3, edgecolors='none')

    stat_text = f'Mann-Whitney {sig_tag}: {p_label}'
    n_text    = f'n = {n_edge} edge, {n_cent} centre'
    ax.text(0.05, 0.97, stat_text, transform=ax.transAxes, ha='left', va='top',
            fontsize=8.5, color='#444444')
    ax.text(0.05, 0.91, n_text,    transform=ax.transAxes, ha='left', va='top',
            fontsize=8.5, color='#444444')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(group_names, fontsize=12)
    ax.set_ylabel('Distance from wall (cm)', fontsize=12)
    ax.set_title(f'Place cell distance from wall\nby zone ({arena_label})', fontsize=13, pad=10)
    ax.set_xlim(0.4, 2.6)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'\n{arena_label} — Mann-Whitney U = {u_stat:.1f},  {p_label}  ({sig_tag})')
    print(f'n edge = {n_edge},  n centre = {n_cent}')
    print(f'  [SAVED] {output_png}')


def plot_proportion_comparison(zone_data_by_arena: dict, output_png: str):
    """Stacked bar chart of edge/centre proportions across arenas.

    zone_data_by_arena: {arena_label: (edge_vals, cent_vals)}
    """
    def pct(edge, cent):
        total = len(edge) + len(cent)
        if total == 0:
            return 0.0, 0.0
        return 100 * len(edge) / total, 100 * len(cent) / total

    arena_labels  = list(zone_data_by_arena.keys())
    pct_edge_vals = []
    pct_cent_vals = []
    n_labels      = []
    for label in arena_labels:
        edge_vals, cent_vals = zone_data_by_arena[label]
        pe, pc = pct(edge_vals, cent_vals)
        pct_edge_vals.append(pe)
        pct_cent_vals.append(pc)
        n_labels.append(f'{label}: n={len(edge_vals) + len(cent_vals)}')

    x     = np.arange(len(arena_labels))
    width = 0.5

    fig2, ax2 = plt.subplots(figsize=(5.5, 5))

    bars_edge = ax2.bar(x, pct_edge_vals, width,
                        label='Edge', color=COLOR_EDGE, alpha=0.85, linewidth=1.2,
                        edgecolor='black')
    bars_cent = ax2.bar(x, pct_cent_vals, width,
                        label='Centre', color=COLOR_CENTRE, alpha=0.85, linewidth=1.2,
                        edgecolor='black', bottom=pct_edge_vals)

    for bar, h_edge, h_cent in zip(bars_edge, pct_edge_vals, pct_cent_vals):
        mid_x = bar.get_x() + bar.get_width() / 2
        if h_edge > 4:
            ax2.text(mid_x, h_edge / 2, f'{h_edge:.1f}%',
                     ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        if h_cent > 4:
            ax2.text(mid_x, h_edge + h_cent / 2, f'{h_cent:.1f}%',
                     ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(arena_labels, fontsize=12)
    ax2.set_ylabel('Proportion of cells (%)', fontsize=12)
    ax2.set_title('Edge vs Centre cell proportion\nacross arenas', fontsize=13, pad=10)
    ax2.set_ylim(0, 100)
    ax2.legend(frameon=False, fontsize=10)

    ax2.text(0.98, 0.97, '   '.join(n_labels),
             transform=ax2.transAxes, ha='right', va='top',
             fontsize=8.5, color='#444444')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig2)

    for label, (edge_vals, cent_vals) in zone_data_by_arena.items():
        pe, pc = pct(edge_vals, cent_vals)
        print(f'{label:14s} — Edge: {pe:.1f}%,  Centre: {pc:.1f}%  '
              f'(n={len(edge_vals) + len(cent_vals)})')
    print(f'  [SAVED] {output_png}')


# ── Batch scan ────────────────────────────────────────────────────────────────

_PIXEL_ANSWERS = {'pixel', 'pixels', 'px'}
_CM_ANSWERS    = {'cm', 'cms', 'centimeter', 'centimeters', 'centimetre', 'centimetres'}

if __name__ == "__main__":
    _coord_answer = input("Are the tracking coordinates in pixels or cm? [pixel/cm]: ").strip().lower()
    while _coord_answer not in _PIXEL_ANSWERS | _CM_ANSWERS:
        _coord_answer = input("Please enter 'pixel' or 'cm': ").strip().lower()
    COORD_UNITS = 'pixel' if _coord_answer in _PIXEL_ANSWERS else 'cm'
    print(f"Using '{COORD_UNITS}' tracking coordinates.\n")

    excel_paths: dict[str, str] = {}   # arena_label -> output_excel path

    # ── Step 1: Open field ────────────────────────────────────────────────────
    if OPEN_ROOT_FOLDER:
        open_cfg = ArenaConfig(
            arena_type='open',
            root_folder=OPEN_ROOT_FOLDER,
            output_excel=OPEN_OUTPUT_EXCEL,
            arena_width_cm=OPEN_ARENA_WIDTH_CM,
            edge_threshold_cm=OPEN_EDGE_ZONE_THRESHOLD_CM,
        )
        excel_paths['Open Field'] = run_boundary_analysis(open_cfg)
    else:
        print('\nOPEN_ROOT_FOLDER is False — skipping open-field boundary analysis.')

    # ── Step 2: Linear track ──────────────────────────────────────────────────
    if LINEAR_ROOT_FOLDER:
        linear_cfg = ArenaConfig(
            arena_type='linear',
            root_folder=LINEAR_ROOT_FOLDER,
            output_excel=LINEAR_OUTPUT_EXCEL,
            arena_width_cm=LINEAR_ARENA_WIDTH_CM,
            edge_threshold_cm=LINEAR_EDGE_ZONE_THRESHOLD_CM,
            track_width_cm=LINEAR_TRACK_WIDTH_CM,
        )
        excel_paths['Linear Track'] = run_boundary_analysis(linear_cfg)
    else:
        print('\nLINEAR_ROOT_FOLDER is False — skipping linear-track boundary analysis.')

    # ── Step 3: Edge/centre comparison plots ──────────────────────────────────
    print(f'\n{"="*80}\nEDGE / CENTRE COMPARISON PLOTS\n{"="*80}')

    zone_data_by_arena = {}
    for arena_label, excel_path in excel_paths.items():
        if not excel_path:
            continue
        edge_vals, cent_vals = load_zone_data(excel_path)
        zone_data_by_arena[arena_label] = (edge_vals, cent_vals)
        safe_name = arena_label.lower().replace(' ', '_')
        plot_zone_boxplot(
            excel_path, arena_label,
            os.path.join(COMPARISON_OUTPUT_DIR, f'dist_from_wall_EdgeCentre_{safe_name}.png'),
        )

    if len(zone_data_by_arena) >= 2:
        plot_proportion_comparison(
            zone_data_by_arena,
            os.path.join(COMPARISON_OUTPUT_DIR, 'proportion_EdgeCentre_arenas.png'),
        )
    else:
        print('\nOnly one (or zero) arena has data — skipping arena-comparison '
              'proportion plot (needs both Open Field and Linear Track results).')
