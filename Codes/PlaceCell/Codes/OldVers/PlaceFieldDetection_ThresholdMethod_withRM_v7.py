# -*- coding: utf-8 -*-
"""
Place-field isolation – batch version.

Input:
    root_folder = Output_PlaceTrue folder produced by
    'PlaceCellCharacterization_SI_Spar_Cohr_PeakFR_MeanFR_Shuffling_TwoHalvesCopmare_ThetaMod_Batch_GPU_Final.py'
    i.e. C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True
    (each sub-folder holds the confirmed-place-cell .ntt file(s) plus their tracking file).

Three arena types are supported, auto-detected from the session folder path
(the path must contain 'Open', 'Linear', or 'Circle' -- case-insensitive --
see _detect_arena_type):
    - 'Open'   : open field, 60 cm diameter.
    - 'Linear' : linear track, 80 x 8 cm.
    - 'Circle' : circular alley track, 80 cm outer diameter / 72 cm inner
                 diameter.

Open field and Linear track use a conventional 2-D Cartesian occupancy grid
(target_bin_cm x target_bin_cm bins), smoothed with an occupancy-weighted
2-D Gaussian kernel (see _gaussian_smooth_2d). No adaptive binning is used.

Circle uses 1-D angular binning instead of the 2-D grid. The animal's
position on a ring track is fundamentally 1-D (its running direction, theta,
around the ring), not 2-D. Binning (x, y) on a square Cartesian grid and
running an 8-connected-component search on it does not respect that
topology -- the ring's curvature means a square grid samples the track
unevenly, so a single true field can get sliced into several disconnected
2-D components (spurious "multifield" cells) purely from binning artifacts,
independent of any real firing discontinuity. To fix this, the ratemap is
built in angular bins: every tracking sample and spike is assigned to a bin
by its bearing (theta) around the track's fitted centre only (its radial
distance from centre is discarded), collapsing the ratemap to a single 1-D
array around the ring, smoothed with a wrap-around 1-D Gaussian kernel (see
_gaussian_smooth_circular). Field detection then finds contiguous runs of
qualifying bins on that 1-D circular array, wrapping around the theta=0/2*pi
seam, which is a topologically correct match for a closed loop and cannot
fragment a field the way the 2-D grid did.

Algorithm (per .ntt file) – "threshold method", adapted from a MATLAB
`placefield`/`getLegals` reference implementation:
    1. Keep only occupied bins whose Gaussian-smoothed firing rate is both
       >= METHOD2_RATE_THRESHOLD_FRAC of the cell's peak rate AND above
       the cell's mean firing rate.
    2. Open/Linear: 8-connected-component label the surviving bins on the
       2-D grid. Circle: label contiguous runs of the surviving bins on the
       circular (wrap-around) angular axis.
    3. Components/runs spanning >= MIN_FIELD_SIZE_BINS contiguous bins (no
       discontinuity) are reported as place fields (peak bin + firing-rate-
       weighted centre of mass, mirroring the MATLAB reference's `fieldPos`).
    This is a single non-iterative sweep (no suppression / re-bootstrapping).

Output:
    A single Excel workbook. Each .ntt file gets its own sheet (one row per
    detected field). A final 'Summary' sheet lists every .ntt file analysed,
    the number of fields found, and each field's size (cm^2) / % of the
    occupied arena it covers.
"""

#<Major fix: Skewness score to be added to the code>

import os
import re
import time
import threading
import concurrent.futures
import types

import numpy as np
import pandas as pd
from scipy.ndimage import convolve

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ── GPU availability ──────────────────────────────────────────────────────────

try:
    import cupy as cp                                          # type: ignore[import-untyped]
    from cupyx.scipy.ndimage import convolve as cp_convolve    # type: ignore[import-untyped]
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

root_folder  = r'X:\NMR_group_data\Runita\Temp\PlaceCell_True\TestRun'
output_excel = r'X:\NMR_group_data\Runita\Temp\PlaceCell_True\TestRun\PlaceFields.xlsx'

# root_folder  = r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True'
# output_excel = r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/PlaceFields.xlsx'

fps            = 30           # tracking frame rate (Hz)
target_bin_cm  = 2.0          # bin size in cm (both 2-D grid bins and angular arc-length bins)
min_occ_s      = 1.0          # exclude bins with < 1 s occupancy
MAX_GAP_US     = 50_000       # max spike-position gap in µs (50 ms)
MAX_SPEED_CM_S = 90           # frame-to-frame speed above this is treated as a tracking-jump artifact and dropped

# Arena type is auto-detected per session from the folder path (see
# _detect_arena_type). ARENA_WIDTH_CM is the physical width (cm) used only to
# convert tracking pixel coordinates to cm: the open field's diameter, the
# linear track's long axis, and the circular track's outer diameter.
ARENA_TYPES = ('Open', 'Linear', 'Circle')
ARENA_WIDTH_CM = {
    'Open':   60.0,    # 60 cm diameter open field
    'Linear': 80.0,    # 80 x 8 cm linear track (long axis)
    'Circle': 80.0,    # 80 cm outer diameter circular track
}

MIN_FIELD_SIZE_BINS = 9       # a field must span >= 9 contiguous bins (8-connected on the
                               # 2-D grid for Open/Linear, or wrap-around angular bins for Circle)

METHOD2_RATE_THRESHOLD_FRAC = 0.20    # "threshold method": bins must be >= 20% of the cell's peak rate
                                       # (mirrors pTreshold in the MATLAB placefield reference); bins must
                                       # also be above the cell's mean firing rate (see detect_place_fields_threshold_*)

MAX_GPU_UTIL_PCT = 60
MAX_WORKERS      = 4

# 'pixel' or 'cm' – set interactively at startup (see __main__ below).
COORD_UNITS = 'pixel'

_gpu_semaphore = threading.Semaphore(2)

# Hockeimer et al. 2025 (eLife 85599): ratemaps binned at 10 px (2.1 cm) per
# bin, smoothed with a Gaussian kernel of sigma = 1.5 bins. Stored here as a
# physical sigma in cm (1.5 * 2.1 cm) so it converts correctly to whatever
# bin size (target_bin_cm) this script is run with. Used for both the 2-D
# (Open/Linear) and angular (Circle) rate maps.
GAUSSIAN_SIGMA_CM = 5 #1.5 * 2.1

ntt_dtype = np.dtype([
    ('timestamp',   '<u8'),
    ('sc_number',   '<u4'),
    ('cell_number', '<u4'),
    ('params',      '<u4', (8,)),
    ('waveforms',   '<i2', (32, 4)),
])

_print_lock = threading.Lock()


# ── Arena-type detection ─────────────────────────────────────────────────────

def _detect_arena_type(path: str) -> str:
    """Identifies the arena type from a session folder path. The path must
    contain exactly one of 'Open', 'Linear', or 'Circle' (case-insensitive)
    somewhere in it -- raises ValueError otherwise, so a mis-named folder
    fails loudly instead of silently picking the wrong binning procedure."""
    lower = path.lower()
    hits = [name for name in ARENA_TYPES if name.lower() in lower]
    if len(hits) != 1:
        raise ValueError(
            f"Could not uniquely determine arena type from path {path!r} "
            f"(expected exactly one of {ARENA_TYPES} to appear in the path, found {hits})")
    return hits[0]


# ── Smoothing ──────────────────────────────────────────────────────────────────

def _wait_for_gpu_slot(poll_interval: float = 0.5):
    if not _GPU or not _NVML:
        return
    while _gpu_util_pct() >= MAX_GPU_UTIL_PCT:
        time.sleep(poll_interval)


def _gaussian_kernel_1d(sigma_bins: float) -> np.ndarray:
    """1D Gaussian kernel, sigma given in bins, truncated at 3 sigma."""
    radius = max(1, int(np.ceil(3 * sigma_bins)))
    ax = np.arange(-radius, radius + 1)
    kernel = np.exp(-(ax ** 2) / (2 * sigma_bins ** 2))
    kernel /= kernel.sum()
    return kernel


def _gaussian_smooth_circular(fr_map: np.ndarray, valid_mask: np.ndarray, bin_cm: float) -> np.ndarray:
    """Gaussian smoothing on a 1D angular rate map. The axis is circular --
    the track is a closed loop -- so it is padded by wrapping rather than
    zero-filling, which would otherwise create a spurious rate dip at the
    arbitrary theta=0/2*pi seam."""
    sigma_bins = GAUSSIAN_SIGMA_CM / bin_cm
    kernel = _gaussian_kernel_1d(sigma_bins)
    kr = kernel.shape[0] // 2

    fr_in   = np.where(valid_mask, fr_map, 0.0)
    mask_in = valid_mask.astype(np.float64)

    def _pad(arr, xp):
        return xp.pad(arr, (kr, kr), mode='wrap')

    n_theta = fr_map.shape[0]

    if _GPU:
        fr_gpu   = _pad(cp.asarray(fr_in,   dtype=cp.float64), cp)
        mask_gpu = _pad(cp.asarray(mask_in, dtype=cp.float64), cp)
        kern_gpu = cp.asarray(kernel, dtype=cp.float64)
        _wait_for_gpu_slot()
        _gpu_semaphore.acquire()
        try:
            smoothed_fr_full = cp.asnumpy(
                cp_convolve(fr_gpu, kern_gpu, mode='constant', cval=0.0)
            )
            smoothed_weights_full = cp.asnumpy(
                cp_convolve(mask_gpu, kern_gpu, mode='constant', cval=0.0)
            )
        finally:
            _gpu_semaphore.release()
    else:
        fr_pad   = _pad(fr_in,   np)
        mask_pad = _pad(mask_in, np)
        smoothed_fr_full      = convolve(fr_pad,   kernel, mode='constant', cval=0.0)
        smoothed_weights_full = convolve(mask_pad, kernel, mode='constant', cval=0.0)

    smoothed_fr      = smoothed_fr_full[kr:kr + n_theta]
    smoothed_weights = smoothed_weights_full[kr:kr + n_theta]

    smoothed = np.zeros_like(smoothed_fr)
    valid_weights = smoothed_weights > 0
    smoothed[valid_weights] = smoothed_fr[valid_weights] / smoothed_weights[valid_weights]
    smoothed[~valid_mask] = 0.0
    return smoothed


def _gaussian_smooth_2d(fr_map: np.ndarray, valid_mask: np.ndarray, bin_cm: float) -> np.ndarray:
    """Occupancy-weighted 2D Gaussian smoothing for the Open-field / Linear-
    track rate maps (used instead of adaptive binning). Rate is recovered as
    smoothed-spikes / smoothed-occupancy-weight, so unvisited bins near the
    arena edge don't bias the smoothed rate downward. Edges are zero-padded
    (no wrap-around -- unlike the circular ring track, these arenas are not
    a closed loop)."""
    sigma_bins = GAUSSIAN_SIGMA_CM / bin_cm
    kernel_1d = _gaussian_kernel_1d(sigma_bins)
    kernel_2d = np.outer(kernel_1d, kernel_1d)

    fr_in   = np.where(valid_mask, fr_map, 0.0)
    mask_in = valid_mask.astype(np.float64)

    if _GPU:
        fr_gpu   = cp.asarray(fr_in,   dtype=cp.float64)
        mask_gpu = cp.asarray(mask_in, dtype=cp.float64)
        kern_gpu = cp.asarray(kernel_2d, dtype=cp.float64)
        _wait_for_gpu_slot()
        _gpu_semaphore.acquire()
        try:
            smoothed_fr      = cp.asnumpy(cp_convolve(fr_gpu,   kern_gpu, mode='constant', cval=0.0))
            smoothed_weights = cp.asnumpy(cp_convolve(mask_gpu, kern_gpu, mode='constant', cval=0.0))
        finally:
            _gpu_semaphore.release()
    else:
        smoothed_fr      = convolve(fr_in,   kernel_2d, mode='constant', cval=0.0)
        smoothed_weights = convolve(mask_in, kernel_2d, mode='constant', cval=0.0)

    smoothed = np.zeros_like(smoothed_fr)
    valid_weights = smoothed_weights > 0
    smoothed[valid_weights] = smoothed_fr[valid_weights] / smoothed_weights[valid_weights]
    smoothed[~valid_mask] = 0.0
    return smoothed


# ── Rate-map metrics (SIR / sparsity / coherence / peak / mean) ────────────────

def _metrics_from_ratemap(fr_map: np.ndarray, occ_map: np.ndarray, valid_mask: np.ndarray) -> dict:
    total_occ_s = occ_map[valid_mask].sum()
    pi_flat     = occ_map[valid_mask] / total_occ_s
    ri_flat     = fr_map[valid_mask]
    r_mean      = float(np.sum(pi_flat * ri_flat))

    peak_fr = float(ri_flat.max()) if ri_flat.size else 0.0
    mean_fr = r_mean

    sir = 0.0
    if r_mean > 0:
        nonzero = ri_flat > 0
        ratio   = ri_flat[nonzero] / r_mean
        sir     = float(np.sum(pi_flat[nonzero] * ratio * np.log2(ratio)))

    spar_num = float(np.sum(pi_flat * ri_flat))
    spar_den = float(np.sum(pi_flat * ri_flat ** 2))
    sparsity = float((spar_num ** 2) / spar_den) if spar_den > 0 else 0.0

    return {'peak_fr': round(peak_fr, 4), 'mean_fr': round(mean_fr, 4),
            'sir': round(sir, 4), 'sparsity': round(sparsity, 4)}


# ── Ring-track geometry (angular binning, Circle arena only) ────────────────────

RADIAL_RANGE_CLIP_PCTILE = 0.5   # trim this many percentiles off each end of the
                                  # radial-distance distribution before estimating
                                  # the track width, so a handful of tracking-jitter
                                  # outliers can't stretch/blur the estimate

def _fit_ring_centre(x_cm: np.ndarray, y_cm: np.ndarray) -> tuple:
    """Algebraic (Kasa) circle fit to the tracked positions, returning
    (cx, cy) -- the track's centre in the same cm frame as x_cm/y_cm. Assumes
    the great majority of samples lie on (or near) the ring, which holds for
    a circular-alley track."""
    A = np.column_stack([x_cm, y_cm, np.ones_like(x_cm)])
    b = x_cm ** 2 + y_cm ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx = sol[0] / 2.0
    cy = sol[1] / 2.0
    return cx, cy


def _circular_mean_rad(angles_rad: np.ndarray, weights: np.ndarray) -> float:
    s = float(np.sum(weights * np.sin(angles_rad)))
    c = float(np.sum(weights * np.cos(angles_rad)))
    return float(np.arctan2(s, c)) % (2.0 * np.pi)


# ── Rate-map construction, per arena type ───────────────────────────────────────

def _build_ratemap_circular(x_cm: np.ndarray, y_cm: np.ndarray, spike_frame: np.ndarray,
                             dt_frames: np.ndarray, target_bin_cm: float) -> dict:
    """1-D angular ratemap for the Circle (ring) track -- see module
    docstring. Every sample's bearing (theta) around the track's fitted
    centre is binned; radial distance from centre is discarded."""
    cx, cy = _fit_ring_centre(x_cm, y_cm)
    r_cm        = np.hypot(x_cm - cx, y_cm - cy)
    theta_rad   = np.arctan2(y_cm - cy, x_cm - cx)             # (-pi, pi]
    theta_0_2pi = np.mod(theta_rad, 2.0 * np.pi)               # [0, 2*pi)

    r_min = float(np.percentile(r_cm, RADIAL_RANGE_CLIP_PCTILE))
    r_max = float(np.percentile(r_cm, 100.0 - RADIAL_RANGE_CLIP_PCTILE))
    track_width_cm = max(r_max - r_min, 0.0)

    r_mean = float(np.median(r_cm))
    bin_width_rad = target_bin_cm / r_mean                     # arc length ~= target_bin_cm
    n_bins_theta = max(1, int(round(2.0 * np.pi / bin_width_rad)))
    bin_width_rad = 2.0 * np.pi / n_bins_theta                 # re-close evenly around the ring

    beh_bt = np.clip((theta_0_2pi / bin_width_rad).astype(int), 0, n_bins_theta - 1)
    sp_bt  = beh_bt[spike_frame]

    occ_map   = np.zeros(n_bins_theta, dtype=np.float64)
    spike_map = np.zeros(n_bins_theta, dtype=np.float64)
    np.add.at(occ_map,   beh_bt, dt_frames)
    np.add.at(spike_map, sp_bt,  1.0)

    valid_mask = occ_map >= min_occ_s
    fr_raw = np.zeros_like(occ_map)
    fr_raw[valid_mask] = spike_map[valid_mask] / occ_map[valid_mask]
    fr_smooth = _gaussian_smooth_circular(fr_raw, valid_mask, target_bin_cm)

    return dict(beh_bt=beh_bt, occ_map=occ_map, valid_mask=valid_mask,
                fr_raw=fr_raw, fr_smooth=fr_smooth,
                n_bins_theta=n_bins_theta, bin_width_rad=bin_width_rad,
                cx=cx, cy=cy, r_min=r_min, r_max=r_max, track_width_cm=track_width_cm)


def _build_ratemap_2d(x_cm: np.ndarray, y_cm: np.ndarray, spike_frame: np.ndarray,
                       dt_frames: np.ndarray, target_bin_cm: float) -> dict:
    """Conventional 2-D Cartesian ratemap (target_bin_cm x target_bin_cm
    bins) for the Open-field and Linear-track arenas, Gaussian-smoothed
    (no adaptive binning)."""
    n_bins_x = int(np.ceil(x_cm.max() / target_bin_cm))
    n_bins_y = int(np.ceil(y_cm.max() / target_bin_cm))

    beh_bx = np.clip((x_cm / target_bin_cm).astype(int), 0, n_bins_x - 1)
    beh_by = np.clip((y_cm / target_bin_cm).astype(int), 0, n_bins_y - 1)
    sp_bx  = beh_bx[spike_frame]
    sp_by  = beh_by[spike_frame]

    occ_map   = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    spike_map = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    np.add.at(occ_map,   (beh_bx, beh_by), dt_frames)
    np.add.at(spike_map, (sp_bx,  sp_by),  1.0)

    valid_mask = occ_map >= min_occ_s
    fr_raw = np.zeros_like(occ_map)
    fr_raw[valid_mask] = spike_map[valid_mask] / occ_map[valid_mask]
    fr_smooth = _gaussian_smooth_2d(fr_raw, valid_mask, target_bin_cm)

    return dict(beh_bx=beh_bx, beh_by=beh_by, occ_map=occ_map, valid_mask=valid_mask,
                fr_raw=fr_raw, fr_smooth=fr_smooth,
                n_bins_x=n_bins_x, n_bins_y=n_bins_y)


# ── Tracking + spike loading / rate-map dispatch ────────────────────────────────

def build_ratemap(csv_path: str, ntt_path: str, arena_width_cm: float,
                   target_bin_cm: float, arena_type: str) -> tuple:
    """Loads tracking + spikes and returns (metrics, ctx). Dispatches to the
    1-D angular ratemap (Circle) or the 2-D Cartesian ratemap (Open, Linear)
    depending on arena_type -- see module docstring."""

    data = (pd.read_excel(csv_path) if csv_path.lower().endswith('.xlsx')
            else pd.read_csv(csv_path))

    if COORD_UNITS == 'cm':
        t    = np.asarray(data.iloc[:, 0], dtype=float)
        x    = np.asarray(data.iloc[:, 3], dtype=float)
        y    = np.asarray(data.iloc[:, 4], dtype=float)
        x_px = np.asarray(data['x'], dtype=float)   # raw pixel column, used only to spot the lost-tracking sentinel below
    else:
        x = np.asarray(data['x'],    dtype=float)
        y = np.asarray(data['y'],    dtype=float)
        t = np.asarray(data['time'], dtype=float)
        x_px = x

    # Lost-tracking sentinel (1 / -1) is a pixel-space convention; check it on the
    # pixel column even in 'cm' mode, since the cm-converted column almost never
    # lands on exactly 1 or -1 and the filter would otherwise be a no-op.
    mask = ~np.isin(x_px, [1, -1])
    x, y, t = x[mask], y[mask], t[mask]

    # Jump-artifact filter, evaluated in real cm/s. `t` is a raw microsecond
    # timestamp and x/y can be pixels or cm depending on COORD_UNITS, so both
    # need to be put on a common physical scale before comparing to a speed
    # threshold (a rough, unfiltered pixel-to-cm scale is fine here, same as
    # pixel_to_cm_conversion_v4.py's remove_speed_jumps: real jumps are large
    # enough to still be caught).
    dt_sec   = np.append(np.diff(t), 1) * 1e-6
    valid_dt = dt_sec > 0

    if COORD_UNITS == 'cm':
        x_cm_rough, y_cm_rough = x, y
    else:
        rough_px_per_cm = max(x.max() - x.min(), y.max() - y.min()) / arena_width_cm
        x_cm_rough = x / rough_px_per_cm
        y_cm_rough = y / rough_px_per_cm

    dxy_cm = np.hypot(np.append(np.diff(x_cm_rough), 0), np.append(np.diff(y_cm_rough), 0))

    speed_cm_s = np.zeros_like(dxy_cm)
    speed_cm_s[valid_dt] = dxy_cm[valid_dt] / dt_sec[valid_dt]

    keep = np.where(valid_dt & (speed_cm_s < MAX_SPEED_CM_S))[0]
    x, y, t = x[keep], y[keep], t[keep]

    order = np.argsort(t)
    x, y, t = x[order], y[order], t[order]

    if len(t) == 0:
        return ({'n_spikes': 0}, {})

    if COORD_UNITS == 'cm':
        x_cm = x - x.min()
        y_cm = y - y.min()
    else:
        x_span = x.max() - x.min()
        y_span = y.max() - y.min()
        px_per_cm = max(x_span, y_span) / arena_width_cm
        x_cm = (x - x.min()) / px_per_cm
        y_cm = (y - y.min()) / px_per_cm

    spike_data = np.memmap(ntt_path, dtype=ntt_dtype, mode='r', offset=16 * 1024)
    spike_ts   = np.sort(spike_data['timestamp'].astype(np.float64))

    idx   = np.searchsorted(t, spike_ts, side='left')
    idx_l = np.clip(idx - 1, 0, len(t) - 1)
    idx_r = np.clip(idx,     0, len(t) - 1)
    dist_l  = np.abs(spike_ts - t[idx_l])
    dist_r  = np.abs(spike_ts - t[idx_r])
    nearest  = np.where(dist_l <= dist_r, idx_l, idx_r)
    min_dist = np.minimum(dist_l, dist_r)

    valid_spike = min_dist <= MAX_GAP_US
    spike_frame = nearest[valid_spike]
    n_spikes    = int(valid_spike.sum())
    n_discarded = int((~valid_spike).sum())

    dt_frames     = np.empty(len(t), dtype=np.float64)
    dt_frames[0]  = 1.0 / fps
    raw_dt        = np.diff(t) * 1e-6
    max_frame_s   = 2.0 / fps
    dt_frames[1:] = np.minimum(raw_dt, max_frame_s)

    if arena_type == 'Circle':
        ctx_extra = _build_ratemap_circular(x_cm, y_cm, spike_frame, dt_frames, target_bin_cm)
    else:
        ctx_extra = _build_ratemap_2d(x_cm, y_cm, spike_frame, dt_frames, target_bin_cm)

    occ_map    = ctx_extra['occ_map']
    valid_mask = ctx_extra['valid_mask']
    fr_smooth  = ctx_extra['fr_smooth']

    ctx = dict(spike_ts=spike_ts[valid_spike], spike_frame=spike_frame, t=t, **ctx_extra)

    if not valid_mask.any():
        return ({'n_spikes': n_spikes, 'n_discarded': n_discarded,
                  'peak_fr': 0.0, 'mean_fr': 0.0, 'sir': 0.0, 'sparsity': 0.0}, ctx)

    base = _metrics_from_ratemap(fr_smooth, occ_map, valid_mask)
    base['n_spikes']    = n_spikes
    base['n_discarded'] = n_discarded
    return base, ctx


# ── Field isolation – "threshold method", 2-D grid (Open field / Linear track) ──

def _connected_components_8(qualifies: np.ndarray, n_bins_x: int, n_bins_y: int) -> list:
    """8-connected component labelling of the bins where `qualifies` is True.
    Direct analogue of the MATLAB reference's visited/getLegals flood fill."""
    visited = ~qualifies
    components = []
    for i in range(n_bins_x):
        for j in range(n_bins_y):
            if visited[i, j]:
                continue
            region = []
            stack = [(i, j)]
            visited[i, j] = True
            while stack:
                bx, by = stack.pop()
                region.append((bx, by))
                for ddx in (-1, 0, 1):
                    for ddy in (-1, 0, 1):
                        if ddx == 0 and ddy == 0:
                            continue
                        nx, ny = bx + ddx, by + ddy
                        if 0 <= nx < n_bins_x and 0 <= ny < n_bins_y and not visited[nx, ny]:
                            visited[nx, ny] = True
                            stack.append((nx, ny))
            components.append(region)
    return components


def detect_place_fields_threshold_2d(base_metrics: dict, ctx: dict, target_bin_cm: float) -> list[dict]:
    """"Threshold method" for the 2-D grid (Open field / Linear track): a bin
    only qualifies for a field if its Gaussian-smoothed rate is >=
    METHOD2_RATE_THRESHOLD_FRAC of the cell's peak rate AND above the cell's
    mean firing rate; 8-connected components of qualifying bins spanning >=
    MIN_FIELD_SIZE_BINS contiguous bins (no discontinuity) are reported as
    fields (peak bin + rate-weighted centre of mass, mirroring the MATLAB
    reference's `fieldPos`/`centreFieldSize`)."""
    valid_mask  = ctx['valid_mask']
    fr_smooth   = ctx['fr_smooth']
    n_bins_x    = ctx['n_bins_x']
    n_bins_y    = ctx['n_bins_y']

    total_valid_bins = int(valid_mask.sum())
    if total_valid_bins == 0:
        return []

    peak_fr = float(fr_smooth[valid_mask].max())
    mean_fr = float(base_metrics.get('mean_fr', 0.0))
    rate_threshold = METHOD2_RATE_THRESHOLD_FRAC * peak_fr
    min_size_bins  = MIN_FIELD_SIZE_BINS

    qualifies = valid_mask & (fr_smooth >= rate_threshold) & (fr_smooth > mean_fr)

    centre_bin = (n_bins_x / 2.0, n_bins_y / 2.0)
    best_centre_dist  = np.inf
    centre_field_idx  = None
    fields = []

    for region in _connected_components_8(qualifies, n_bins_x, n_bins_y):
        if len(region) < min_size_bins:
            continue

        bxs   = np.array([b[0] for b in region])
        bys   = np.array([b[1] for b in region])
        rates = fr_smooth[bxs, bys]

        peak_local_idx = int(np.argmax(rates))
        peak_bin = (int(bxs[peak_local_idx]), int(bys[peak_local_idx]))
        peak_val = float(rates[peak_local_idx])

        total_rate = float(rates.sum())
        com_x = float(np.sum(rates * bxs) / total_rate)
        com_y = float(np.sum(rates * bys) / total_rate)

        n_bins_field = len(region)
        area_cm2     = n_bins_field * (target_bin_cm ** 2)
        pct_area     = 100.0 * n_bins_field / total_valid_bins
        bin_coords_str = ';'.join(f'{bx}-{by}' for bx, by in sorted(region))

        dist_to_centre = float(np.hypot(com_x - centre_bin[0], com_y - centre_bin[1]))
        if dist_to_centre < best_centre_dist:
            best_centre_dist = dist_to_centre
            centre_field_idx = len(fields)

        fields.append({
            'field_number':          len(fields) + 1,
            'peak_bin_x':            peak_bin[0],
            'peak_bin_y':            peak_bin[1],
            'peak_fr_hz':            round(peak_val, 4),
            'com_bin_x':             round(com_x, 3),
            'com_bin_y':             round(com_y, 3),
            'com_cm_x':              round(com_x * target_bin_cm, 2),
            'com_cm_y':              round(com_y * target_bin_cm, 2),
            'n_bins':                n_bins_field,
            'area_cm2':              round(area_cm2, 2),
            'pct_of_occupied_area':  round(pct_area, 2),
            'bbox_x_min':            int(bxs.min()),
            'bbox_x_max':            int(bxs.max()),
            'bbox_y_min':            int(bys.min()),
            'bbox_y_max':            int(bys.max()),
            'bbox_cm_x_min':         round(bxs.min() * target_bin_cm, 2),
            'bbox_cm_x_max':         round((bxs.max() + 1) * target_bin_cm, 2),
            'bbox_cm_y_min':         round(bys.min() * target_bin_cm, 2),
            'bbox_cm_y_max':         round((bys.max() + 1) * target_bin_cm, 2),
            'bin_coords':            bin_coords_str,
            'total_occupied_bins':   total_valid_bins,
            'min_field_size_bins':   min_size_bins,
            'rate_threshold_hz':     round(rate_threshold, 4),
            'mean_fr_threshold_hz':  round(mean_fr, 4),
            'is_centre_field':       False,
        })

    if centre_field_idx is not None:
        fields[centre_field_idx]['is_centre_field'] = True

    return fields


# ── Field isolation – "threshold method", angular (Circle track) ────────────────

def _circular_runs(qualifies: np.ndarray, n_bins_theta: int) -> list:
    """Contiguous-run labelling of the angular bins where `qualifies` is
    True, wrapping around the theta=0/2*pi seam (bin 0 borders bin
    n_bins_theta-1) since the track is a closed loop. A field is never
    artificially split just because it straddles that arbitrary seam --
    the circular analogue of the 2-D 8-connected-component search, adapted
    from the MATLAB reference's visited/getLegals flood fill."""
    if not qualifies.any():
        return []
    if qualifies.all():
        return [list(range(n_bins_theta))]

    idx = np.where(qualifies)[0]
    runs = []
    current = [int(idx[0])]
    for b in idx[1:]:
        b = int(b)
        if b == current[-1] + 1:
            current.append(b)
        else:
            runs.append(current)
            current = [b]
    runs.append(current)

    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == n_bins_theta - 1:
        merged = runs[-1] + runs[0]
        runs = runs[1:-1] + [merged]

    return runs


def detect_place_fields_threshold_circular(base_metrics: dict, ctx: dict, target_bin_cm: float) -> list[dict]:
    """"Threshold method" for the Circle (ring) track: a single-pass
    contiguous-run detector adapted from the MATLAB `placefield`/`getLegals`
    reference. A bin only qualifies for a field if its smoothed rate is >=
    METHOD2_RATE_THRESHOLD_FRAC of the cell's peak rate AND above the cell's
    mean firing rate; contiguous runs of qualifying angular bins spanning >=
    MIN_FIELD_SIZE_BINS bins (no discontinuity, wrapping around the theta
    seam) are reported as fields (peak bin + rate-weighted circular centre
    of mass, mirroring the reference's `fieldPos`/`centreFieldSize`)."""
    valid_mask    = ctx['valid_mask']
    fr_smooth     = ctx['fr_smooth']
    n_bins_theta  = ctx['n_bins_theta']
    bin_width_rad = ctx['bin_width_rad']
    track_width_cm = ctx['track_width_cm']

    total_valid_bins = int(valid_mask.sum())
    if total_valid_bins == 0:
        return []

    peak_fr = float(fr_smooth[valid_mask].max())
    mean_fr = float(base_metrics.get('mean_fr', 0.0))
    rate_threshold = METHOD2_RATE_THRESHOLD_FRAC * peak_fr
    min_size_bins  = MIN_FIELD_SIZE_BINS

    qualifies = valid_mask & (fr_smooth >= rate_threshold) & (fr_smooth > mean_fr)
    theta_centers_all = (np.arange(n_bins_theta) + 0.5) * bin_width_rad

    # Legacy "centre field" heuristic (nearest field to a reference bearing)
    # kept for output-schema continuity; on a ring track there is no single
    # geometric centre bin, so this just flags the field closest to the
    # arbitrary theta=0 reference bearing.
    best_centre_dist = np.inf
    centre_field_idx = None
    fields = []

    for region in _circular_runs(qualifies, n_bins_theta):
        if len(region) < min_size_bins:
            continue

        bts   = np.array(region)
        rates = fr_smooth[bts]

        peak_local_idx = int(np.argmax(rates))
        peak_bt        = int(bts[peak_local_idx])
        peak_val       = float(rates[peak_local_idx])
        peak_theta_rad = theta_centers_all[peak_bt]

        theta_centers = theta_centers_all[bts]
        com_theta_rad = _circular_mean_rad(theta_centers, rates)

        # Angular span/bbox relative to the field's own circular mean, so a
        # field straddling the theta=0/2*pi seam still gets a small, correct
        # angular width instead of an apparent near-full-circle bbox.
        offsets_rad = np.mod(theta_centers - com_theta_rad + np.pi, 2.0 * np.pi) - np.pi
        wraps_seam  = bool((0 in bts.tolist()) and (n_bins_theta - 1 in bts.tolist()))

        n_bins_field  = len(region)
        arc_length_cm = n_bins_field * target_bin_cm
        area_cm2      = arc_length_cm * track_width_cm
        pct_area      = 100.0 * n_bins_field / total_valid_bins
        pct_circumference = 100.0 * n_bins_field / n_bins_theta
        bin_coords_str = ';'.join(str(bt) for bt in sorted(region))

        theta_diff_from_ref = ((com_theta_rad + np.pi) % (2.0 * np.pi)) - np.pi  # ref bearing = 0 rad
        dtheta_bins = abs(theta_diff_from_ref) / bin_width_rad
        if dtheta_bins < best_centre_dist:
            best_centre_dist = dtheta_bins
            centre_field_idx = len(fields)

        fields.append({
            'field_number':               len(fields) + 1,
            'peak_bin_theta':             peak_bt,
            'peak_theta_deg':             round(np.degrees(peak_theta_rad) % 360.0, 2),
            'peak_fr_hz':                 round(peak_val, 4),
            'com_theta_deg':              round(np.degrees(com_theta_rad) % 360.0, 2),
            'n_bins':                     n_bins_field,
            'arc_length_cm':              round(arc_length_cm, 2),
            'track_width_cm':             round(track_width_cm, 2),
            'area_cm2':                   round(area_cm2, 2),
            'pct_of_occupied_area':       round(pct_area, 2),
            'pct_of_track_circumference': round(pct_circumference, 2),
            'theta_span_deg':             round(float(offsets_rad.max() - offsets_rad.min()) * 180.0 / np.pi, 2),
            'bbox_theta_min_deg':         round(np.degrees(com_theta_rad + offsets_rad.min()) % 360.0, 2),
            'bbox_theta_max_deg':         round(np.degrees(com_theta_rad + offsets_rad.max()) % 360.0, 2),
            'wraps_theta_seam':           wraps_seam,
            'bin_coords':                 bin_coords_str,
            'total_occupied_bins':        total_valid_bins,
            'min_field_size_bins':        min_size_bins,
            'rate_threshold_hz':          round(rate_threshold, 4),
            'mean_fr_threshold_hz':       round(mean_fr, 4),
            'is_centre_field':            False,
        })

    if centre_field_idx is not None:
        fields[centre_field_idx]['is_centre_field'] = True

    return fields


# ── Rate-map + field-boundary plotting – 2-D grid (Open field / Linear track) ───

def _field_mask_from_bin_coords(bin_coords: str, n_bins_x: int, n_bins_y: int) -> np.ndarray:
    mask = np.zeros((n_bins_x, n_bins_y), dtype=bool)
    if not bin_coords:
        return mask
    for pair in bin_coords.split(';'):
        bx_str, by_str = pair.split('-')
        mask[int(bx_str), int(by_str)] = True
    return mask


def _plot_ratemap_with_field_boundaries_2d(ax, fr_smooth: np.ndarray, valid_mask: np.ndarray,
                                            fields: list[dict], n_bins_x: int, n_bins_y: int, title: str):
    display_map = np.ma.masked_where(~valid_mask, fr_smooth)
    im = ax.imshow(display_map.T, origin='lower', cmap='jet', interpolation='nearest')

    for field in fields:
        field_mask = _field_mask_from_bin_coords(field.get('bin_coords', ''), n_bins_x, n_bins_y)
        if not field_mask.any():
            continue
        ax.contour(field_mask.T.astype(float), levels=[0.5], colors='black', linewidths=1.5)

    ax.set_title(title)
    ax.set_xlabel('x bin')
    ax.set_ylabel('y bin')
    return im


def _save_field_ratemap_plot_2d(ctx: dict, fields_threshold: list[dict], ntt_path: str):
    """Saves one PNG per .ntt file: rate map with detected-field boundaries
    (black outlines) from the threshold method."""
    fr_smooth  = ctx['fr_smooth']
    valid_mask = ctx['valid_mask']
    n_bins_x   = ctx['n_bins_x']
    n_bins_y   = ctx['n_bins_y']

    fig = Figure(figsize=(6, 6))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    im = _plot_ratemap_with_field_boundaries_2d(ax, fr_smooth, valid_mask, fields_threshold,
                                                 n_bins_x, n_bins_y,
                                                 f'Threshold method ({len(fields_threshold)} field(s))')
    fig.colorbar(im, ax=ax, label='Hz')
    fig.tight_layout()

    ntt_name  = os.path.splitext(os.path.basename(ntt_path))[0]
    save_dir  = os.path.join(os.path.dirname(ntt_path), 'ratemap_field_plots')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{ntt_name}_ratemap_fields.png')
    fig.savefig(save_path, dpi=150)


# ── Rate-map + field-boundary plotting – angular (Circle track) ─────────────────

_FIELD_COLORS = ['tab:red', 'tab:green', 'tab:purple', 'tab:orange',
                  'tab:brown', 'tab:pink', 'tab:cyan', 'tab:olive']


def _plot_ratemap_polar(ax, fr_smooth: np.ndarray, valid_mask: np.ndarray,
                         fields: list[dict], n_bins_theta: int,
                         bin_width_rad: float, title: str):
    """Plots the 1-D angular rate map as a polar tuning curve, so the ring
    track renders as an actual ring: theta=0 and theta=2*pi coincide in
    physical space, so a field that straddles that seam still appears as one
    unbroken arc -- no wraparound artifact, unlike a flat bar/line plot."""
    theta_centers = (np.arange(n_bins_theta) + 0.5) * bin_width_rad
    rates = np.where(valid_mask, fr_smooth, np.nan)

    theta_plot = np.concatenate([theta_centers, theta_centers[:1] + 2.0 * np.pi])
    rates_plot = np.concatenate([rates, rates[:1]])

    ax.plot(theta_plot, rates_plot, color='tab:blue', linewidth=1.5)
    ax.fill(theta_plot, np.nan_to_num(rates_plot), color='tab:blue', alpha=0.15)

    for i, field in enumerate(fields):
        bin_coords = field.get('bin_coords', '')
        if not bin_coords:
            continue
        bts = [int(b) for b in bin_coords.split(';')]
        color = _FIELD_COLORS[i % len(_FIELD_COLORS)]
        ax.plot(theta_centers[bts], rates[bts], 'o', color=color, markersize=3,
                zorder=5, label=f"Field {field['field_number']}")

    ax.set_title(title)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    if fields:
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=7)


def _plot_ratemap_ring(ax, fr_smooth: np.ndarray, valid_mask: np.ndarray,
                        fields: list[dict], n_bins_theta: int, bin_width_rad: float,
                        r_min: float, r_max: float, title: str):
    """Plots the 1-D angular rate map as a binned heatmap in an actual ring
    shape (an annulus spanning the track's fitted radial extent, r_min to
    r_max), the angular analogue of the 2-D grid's imshow rate map. Each
    angular bin is one wedge of the ring, colour-coded by its smoothed rate;
    the polar r-axis is forced to start at 0 so the empty disc inside r_min
    renders as the ring's hole. Detected fields are outlined in black, the
    same way field boundaries are contoured on the 2-D grid."""
    theta_edges = np.arange(n_bins_theta + 1) * bin_width_rad
    r_edges = np.array([r_min, r_max])
    theta_mesh, r_mesh = np.meshgrid(theta_edges, r_edges)

    rates = np.ma.masked_where(~valid_mask, fr_smooth).reshape(1, n_bins_theta)
    mesh = ax.pcolormesh(theta_mesh, r_mesh, rates, cmap='jet', shading='flat')

    theta_centers = (np.arange(n_bins_theta) + 0.5) * bin_width_rad
    theta_ext = np.concatenate([theta_centers, theta_centers[:1] + 2.0 * np.pi])
    track_width = max(r_max - r_min, 1e-6)
    pad = min(0.15 * track_width, r_min) if r_min > 0 else 0.15 * track_width
    r_centers = np.array([r_min - pad, r_min, r_max, r_max + pad])

    for field in fields:
        bin_coords = field.get('bin_coords', '')
        if not bin_coords:
            continue
        bts = [int(b) for b in bin_coords.split(';')]
        field_row = np.zeros(n_bins_theta)
        field_row[bts] = 1.0
        mask_2d = np.vstack([np.zeros(n_bins_theta), field_row, field_row, np.zeros(n_bins_theta)])
        mask_ext = np.concatenate([mask_2d, mask_2d[:, :1]], axis=1)
        theta_c, r_c = np.meshgrid(theta_ext, r_centers)
        ax.contour(theta_c, r_c, mask_ext, levels=[0.5], colors='black', linewidths=1.5)

    ax.set_title(title)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_ylim(0, r_max * 1.1)
    ax.set_yticklabels([])
    return mesh


def _save_field_ratemap_plot_circular(ctx: dict, fields_threshold: list[dict], ntt_path: str):
    """Saves one PNG per .ntt file: the polar angular tuning curve with
    detected-field markers, alongside the binned rate map rendered as an
    actual ring (angular bins, field boundaries outlined) -- the angular
    analogue of the imshow rate map used for the Open field / Linear track."""
    fr_smooth     = ctx['fr_smooth']
    valid_mask    = ctx['valid_mask']
    n_bins_theta  = ctx['n_bins_theta']
    bin_width_rad = ctx['bin_width_rad']
    r_min         = ctx['r_min']
    r_max         = ctx['r_max']

    fig = Figure(figsize=(12, 6.5))
    canvas = FigureCanvasAgg(fig)
    ax_line = fig.add_subplot(121, projection='polar')
    ax_ring = fig.add_subplot(122, projection='polar')

    _plot_ratemap_polar(ax_line, fr_smooth, valid_mask, fields_threshold,
                         n_bins_theta, bin_width_rad,
                         f'Threshold method ({len(fields_threshold)} field(s))')
    im = _plot_ratemap_ring(ax_ring, fr_smooth, valid_mask, fields_threshold,
                             n_bins_theta, bin_width_rad, r_min, r_max,
                             'Binned rate map (ring)')
    fig.colorbar(im, ax=ax_ring, label='Hz', fraction=0.046, pad=0.1)

    fig.tight_layout()

    ntt_name  = os.path.splitext(os.path.basename(ntt_path))[0]
    save_dir  = os.path.join(os.path.dirname(ntt_path), 'ratemap_field_plots')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{ntt_name}_ratemap_fields.png')
    fig.savefig(save_path, dpi=150)


# ── Excel sheet-name sanitisation ───────────────────────────────────────────────

_INVALID_SHEET_CHARS = re.compile(r'[\\/*?:\[\]]')

def _safe_sheet_name(name: str, used: set) -> str:
    clean = _INVALID_SHEET_CHARS.sub('_', name)[:31]
    base = clean
    suffix = 1
    while clean.lower() in used:
        tail = f'_{suffix}'
        clean = (base[:31 - len(tail)] + tail)
        suffix += 1
    used.add(clean.lower())
    return clean


# ── Per-job wrapper ──────────────────────────────────────────────────────────

def _run_job(args):
    unit_idx, total_units, dirpath, csv_path, ntt_file = args
    session_name = os.path.relpath(dirpath, root_folder)
    ntt_path     = os.path.join(dirpath, ntt_file)
    pct          = 100 * unit_idx / total_units

    try:
        arena_type = _detect_arena_type(dirpath)
    except ValueError as e:
        with _print_lock:
            print(f'  ERROR determining arena type for {ntt_file}: {e}')
        return session_name, ntt_file, 'Unknown', [], {'error': str(e)}

    arena_width_cm = ARENA_WIDTH_CM[arena_type]

    with _print_lock:
        print(f'[{unit_idx}/{total_units}  {pct:.1f}%]  {session_name}  |  {ntt_file}  |  arena={arena_type}  (GPU {_gpu_util_pct()}%)')

    try:
        base_metrics, ctx = build_ratemap(csv_path, ntt_path, arena_width_cm, target_bin_cm, arena_type)
    except Exception as e:
        with _print_lock:
            print(f'  ERROR building rate map for {ntt_file}: {e}')
        return session_name, ntt_file, arena_type, [], {'error': str(e)}

    if not ctx or not ctx.get('valid_mask', np.array([])).any():
        return session_name, ntt_file, arena_type, [], {'error': 'no valid occupancy / tracking data'}

    try:
        if arena_type == 'Circle':
            fields_threshold = detect_place_fields_threshold_circular(base_metrics, ctx, target_bin_cm)
        else:
            fields_threshold = detect_place_fields_threshold_2d(base_metrics, ctx, target_bin_cm)
    except Exception as e:
        with _print_lock:
            print(f'  ERROR detecting fields (threshold method) in {ntt_file}: {e}')
        return session_name, ntt_file, arena_type, [], {'error': str(e)}

    try:
        if arena_type == 'Circle':
            _save_field_ratemap_plot_circular(ctx, fields_threshold, ntt_path)
        else:
            _save_field_ratemap_plot_2d(ctx, fields_threshold, ntt_path)
    except Exception as e:
        with _print_lock:
            print(f'  ERROR saving rate-map/field plot for {ntt_file}: {e}')

    summary = {
        'n_spikes':  base_metrics.get('n_spikes'),
        'base_sir':  base_metrics.get('sir'),
        'base_sparsity': base_metrics.get('sparsity'),
        'base_peak_fr':  base_metrics.get('peak_fr'),
        'base_mean_fr':  base_metrics.get('mean_fr'),
    }
    return session_name, ntt_file, arena_type, fields_threshold, summary


# ── Batch scan ────────────────────────────────────────────────────────────────

_PIXEL_ANSWERS = {'pixel', 'pixels', 'px'}
_CM_ANSWERS    = {'cm', 'cms', 'centimeter', 'centimeters', 'centimetre', 'centimetres'}

if __name__ == "__main__":
    _coord_answer = input("Are the tracking coordinates in pixels or cm? [pixel/cm]: ").strip().lower()
    while _coord_answer not in _PIXEL_ANSWERS | _CM_ANSWERS:
        _coord_answer = input("Please enter 'pixel' or 'cm': ").strip().lower()
    COORD_UNITS = 'pixel' if _coord_answer in _PIXEL_ANSWERS else 'cm'
    print(f"Using '{COORD_UNITS}' tracking coordinates.\n")

    all_jobs = []
    output_excel_basename = os.path.basename(output_excel).lower()
    for dirpath, _, filenames in os.walk(root_folder):
        tracking_files_all = [f for f in filenames
                               if f.lower().endswith(('.csv', '.xlsx'))
                               and f.lower() != output_excel_basename]
        if COORD_UNITS == 'cm':
            tracking_files = [f for f in tracking_files_all if f.lower().endswith('_cm.csv')]
        else:
            tracking_files = [f for f in tracking_files_all if not f.lower().endswith('_cm.csv')]
        ntt_files = [f for f in filenames if f.lower().endswith('.ntt')]
        if len(tracking_files) == 1 and len(ntt_files) > 0:
            csv_path = os.path.join(dirpath, tracking_files[0])
            for ntt_file in sorted(ntt_files):
                all_jobs.append((dirpath, csv_path, ntt_file))

    total_units = len(all_jobs)
    print(f'Found {total_units} unit(s) across all sessions.\n')

    job_args = [(idx, total_units, dirpath, csv_path, ntt_file)
                for idx, (dirpath, csv_path, ntt_file) in enumerate(all_jobs, start=1)]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_run_job, args): args for args in job_args}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # ── Write Excel ────────────────────────────────────────────────────────────

    field_columns_2d = [
        'field_number', 'peak_bin_x', 'peak_bin_y', 'peak_fr_hz',
        'com_bin_x', 'com_bin_y', 'com_cm_x', 'com_cm_y',
        'n_bins', 'area_cm2', 'pct_of_occupied_area',
        'bbox_x_min', 'bbox_x_max', 'bbox_y_min', 'bbox_y_max',
        'bbox_cm_x_min', 'bbox_cm_x_max', 'bbox_cm_y_min', 'bbox_cm_y_max',
        'bin_coords', 'total_occupied_bins', 'min_field_size_bins',
        'rate_threshold_hz', 'mean_fr_threshold_hz', 'is_centre_field',
    ]

    field_columns_circular = [
        'field_number', 'peak_bin_theta', 'peak_theta_deg', 'peak_fr_hz',
        'com_theta_deg', 'n_bins', 'arc_length_cm', 'track_width_cm',
        'area_cm2', 'pct_of_occupied_area', 'pct_of_track_circumference',
        'theta_span_deg', 'bbox_theta_min_deg', 'bbox_theta_max_deg', 'wraps_theta_seam',
        'bin_coords', 'total_occupied_bins', 'min_field_size_bins',
        'rate_threshold_hz', 'mean_fr_threshold_hz', 'is_centre_field',
    ]

    # ── Pre-pass: assign sheet names & build the summary rows first, so the
    # 'Summary' sheet can be written before the per-unit sheets (Excel keeps
    # sheets in write order, so writing it first makes it the first tab). ────

    used_sheet_names = set()
    summary_rows = []
    sorted_results = sorted(results, key=lambda r: (r[0], r[1]))

    for session_name, ntt_file, arena_type, fields_threshold, summary in sorted_results:
        sheet_name = _safe_sheet_name(os.path.splitext(ntt_file)[0], used_sheet_names)

        areas_thr = [f['area_cm2'] for f in fields_threshold]
        pcts_thr  = [f['pct_of_occupied_area'] for f in fields_threshold]

        summary_rows.append({
            'session': session_name,
            'unit': ntt_file,
            'arena_type': arena_type,
            'sheet_name': sheet_name,
            'n_fields_detected': len(fields_threshold),
            'field_areas_cm2':  '; '.join(f'{a:.2f}' for a in areas_thr),
            'field_pct_areas':  '; '.join(f'{p:.2f}' for p in pcts_thr),
            'total_field_area_cm2': round(sum(areas_thr), 2) if areas_thr else 0.0,
            'total_pct_area_occupied': round(sum(pcts_thr), 2) if pcts_thr else 0.0,
            'n_spikes': summary.get('n_spikes'),
            'base_sir': summary.get('base_sir'),
            'base_sparsity': summary.get('base_sparsity'),
            'base_peak_fr': summary.get('base_peak_fr'),
            'base_mean_fr': summary.get('base_mean_fr'),
            'error': summary.get('error', ''),
        })

    summary_columns = [
        'session', 'unit', 'arena_type', 'sheet_name', 'n_fields_detected',
        'field_areas_cm2', 'field_pct_areas',
        'total_field_area_cm2', 'total_pct_area_occupied',
        'n_spikes', 'base_sir', 'base_sparsity', 'base_peak_fr', 'base_mean_fr', 'error',
    ]
    df_summary = pd.DataFrame(summary_rows, columns=summary_columns)

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

        for (session_name, ntt_file, arena_type, fields_threshold, summary), row in zip(sorted_results, summary_rows):
            sheet_name = row['sheet_name']
            columns = field_columns_circular if arena_type == 'Circle' else field_columns_2d
            if fields_threshold:
                df_thr = pd.DataFrame(fields_threshold, columns=columns)
            else:
                df_thr = pd.DataFrame(columns=columns)
            df_thr.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f'\nDone. Results saved to {output_excel}')
    print(f'Total units processed : {len(results)}')
    print(f'Total fields detected (threshold method) : {sum(r["n_fields_detected"] for r in summary_rows)}')
