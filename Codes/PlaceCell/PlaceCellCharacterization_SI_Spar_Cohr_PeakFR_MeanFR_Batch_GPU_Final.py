# -*- coding: utf-8 -*-
"""
Batch place cell characterisation – V4 (Ulanovsky & Moss, Hippocampus 2011).

"""

import os
import threading
import time
import concurrent.futures

import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from scipy.stats import pearsonr

# ── GPU availability ──────────────────────────────────────────────────────────

try:
    import cupy as cp
    from cupyx.scipy.ndimage import convolve as cp_convolve
    _GPU = True
    print("CuPy detected – GPU (CUDA) acceleration enabled.")
except ImportError:
    _GPU = False
    print("CuPy not found – running on CPU (install cupy-cuda12x to enable GPU).")

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _NVML = False


def _gpu_util_pct() -> int:
    if not _NVML:
        return 0
    try:
        return pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle).gpu
    except Exception:
        return 0


# ── Configuration ─────────────────────────────────────────────────────────────

root_folder  = r'X:/NMR_group_data/Runita/Analysis/Hpc2ndEdtn/Fa8477'
output_excel = r'X:/NMR_group_data/Runita/Analysis/Hpc2ndEdtn/Fa8477/8477Peak.xlsx'

fps            = 30           # tracking frame rate (Hz)
target_bin_cm  = 2.0          # bin size in cm  (Ulanovsky & Moss: 2 × 2 cm)
arena_width_cm = 80.0         # physical arena width in cm
min_occ_s      = 1.0          # exclude bins with < 1 s occupancy
MAX_GAP_US     = 50_000       # max spike–position gap in µs  (50 ms)

MAX_GPU_UTIL_PCT = 60
MAX_WORKERS      = 4

_gpu_semaphore = threading.Semaphore(2)

# 3 × 3 triangular (bilinear) smoothing kernel  – Ulanovsky & Moss (2011)
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
    Invalid bins are zeroed before and after convolution.
    """
    fr_in = np.where(valid_mask, fr_map, 0.0)
    if _GPU:
        fr_gpu   = cp.asarray(fr_in, dtype=cp.float64)
        kern_gpu = cp.asarray(_TRIANGULAR_KERNEL, dtype=cp.float64)
        _wait_for_gpu_slot()
        _gpu_semaphore.acquire()
        try:
            smoothed = cp.asnumpy(
                cp_convolve(fr_gpu, kern_gpu, mode='constant', cval=0.0)
            )
        finally:
            _gpu_semaphore.release()
    else:
        smoothed = convolve(fr_in, _TRIANGULAR_KERNEL, mode='constant', cval=0.0)
    smoothed[~valid_mask] = 0.0
    return smoothed


# ── Core metric computation ───────────────────────────────────────────────────

def compute_metrics(csv_path: str, ntt_path: str,
                    arena_width_cm: float, target_bin_cm: float) -> dict:

    # ── 1. Load & clean tracking ──────────────────────────────────────────────
    data = (pd.read_excel(csv_path) if csv_path.lower().endswith('.xlsx')
            else pd.read_csv(csv_path))

    x = np.asarray(data['x'],    dtype=float)
    y = np.asarray(data['y'],    dtype=float)
    t = np.asarray(data['time'], dtype=float)   # UNIX timestamps in µs

    # drop sentinel values
    mask = ~np.isin(x, [1, -1])
    x, y, t = x[mask], y[mask], t[mask]

    # velocity filter in pixel space (threshold ≈ 0.006 px/µs → ~200 cm/s)
    dxy = np.hypot(np.diff(x), np.diff(y))
    dt  = np.diff(t)
    keep = np.where((dt > 0) & (dxy / dt < 0.006))[0]
    x, y, t = x[keep], y[keep], t[keep]

    # ensure monotonically increasing time (guard against out-of-order frames)
    order = np.argsort(t)
    x, y, t = x[order], y[order], t[order]

    # ── 2. Pixel → cm conversion ──────────────────────────────────────────────
    # Use whichever axis spans more pixels as the 80-cm reference
    x_span = x.max() - x.min()
    y_span = y.max() - y.min()
    px_per_cm = max(x_span, y_span) / arena_width_cm

    x_cm = (x - x.min()) / px_per_cm   # shift to [0, arena_width_cm]
    y_cm = (y - y.min()) / px_per_cm

    # ── 3. Bin tracking positions ─────────────────────────────────────────────
    n_bins_x = int(np.ceil(x_cm.max() / target_bin_cm)) + 1
    n_bins_y = int(np.ceil(y_cm.max() / target_bin_cm)) + 1

    beh_bx = (x_cm / target_bin_cm).astype(int)
    beh_by = (y_cm / target_bin_cm).astype(int)

    # ── 4. Load spikes & nearest-timestamp assignment (50 ms gate) ────────────
    spike_data = np.memmap(ntt_path, dtype=ntt_dtype, mode='r', offset=16 * 1024)
    spike_ts   = spike_data['timestamp'].astype(np.float64)  # µs

    # For each spike, find the nearest tracking timestamp
    idx   = np.searchsorted(t, spike_ts, side='left')
    idx_l = np.clip(idx - 1, 0, len(t) - 1)
    idx_r = np.clip(idx,     0, len(t) - 1)
    dist_l   = np.abs(spike_ts - t[idx_l])
    dist_r   = np.abs(spike_ts - t[idx_r])
    nearest  = np.where(dist_l <= dist_r, idx_l, idx_r)
    min_dist = np.minimum(dist_l, dist_r)

    valid_spike  = min_dist <= MAX_GAP_US   # True → within 50 ms
    spike_frame  = nearest[valid_spike]     # tracking-frame index per kept spike
    n_spikes     = int(valid_spike.sum())
    n_discarded  = int((~valid_spike).sum())

    sp_bx = beh_bx[spike_frame]
    sp_by = beh_by[spike_frame]

    # ── 5. Build occupancy and spike-count maps ────────────────────────────────
    frame_dur = 1.0 / fps   # seconds per tracking frame

    occ_map   = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    spike_map = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)

    np.add.at(occ_map,   (beh_bx, beh_by), frame_dur)
    np.add.at(spike_map, (sp_bx,  sp_by),  1.0)

    # exclude bins with < 1 s occupancy  (Ulanovsky & Moss 2011)
    valid_mask = occ_map >= min_occ_s

    # ── 6. Non-smoothed firing rate map ──────────────────────────────────────
    fr_raw = np.where(valid_mask, spike_map / occ_map, 0.0)

    # ── 7. Smoothed firing rate map (3 × 3 triangular kernel) ─────────────────
    fr_smooth = _triangular_smooth(fr_raw, valid_mask)

    # ── 8. Compute metrics ────────────────────────────────────────────────────
    if not valid_mask.any():
        return {
            'n_spikes':    n_spikes,
            'n_discarded': n_discarded,
            'peak_fr':     0.0,
            'mean_fr':     0.0,
            'sir':         0.0,
            'coherence':   float('nan'),
            'sparsity':    0.0,
        }

    total_occ_s = occ_map[valid_mask].sum()     # total session time in valid bins
    pi_flat     = occ_map[valid_mask] / total_occ_s   # occupancy probability  pi
    ri_flat     = fr_smooth[valid_mask]               # smoothed FR  ri  (Hz)
    r_mean      = float(np.sum(pi_flat * ri_flat))    # overall mean firing rate r

    # Peak and mean firing rate (from smoothed map)
    peak_fr = float(fr_smooth[valid_mask].max())
    mean_fr = r_mean

    # Spatial information  SI (bits/spike) = Σ pi (ri/r) log2(ri/r)
    # Ulanovsky & Moss eq.; Skaggs et al. 1993
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
    # Ulanovsky & Moss: "coherence was computed from non-smoothed maps"
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
        r_coef    = float(np.clip(r_coef, -0.9999, 0.9999))
        coherence = float(0.5 * np.log((1 + r_coef) / (1 - r_coef)))   # Fisher Z
    else:
        coherence = float('nan')

    return {
        'n_spikes':    n_spikes,
        'n_discarded': n_discarded,
        'peak_fr':     round(peak_fr,   4),
        'mean_fr':     round(mean_fr,   4),
        'sir':         round(sir,       4),
        'coherence':   round(coherence, 4),
        'sparsity':    round(sparsity,  4),
    }


# ── GPU throttle helper ───────────────────────────────────────────────────────

def _wait_for_gpu_slot(poll_interval: float = 0.5):   # noqa: F811
    if not _GPU or not _NVML:
        return
    while _gpu_util_pct() >= MAX_GPU_UTIL_PCT:
        time.sleep(poll_interval)


# ── Per-job wrapper (called from thread pool) ─────────────────────────────────

_print_lock = threading.Lock()


def _run_job(args):
    unit_idx, total_units, job_order, dirpath, csv_path, ntt_file = args
    session_name = os.path.relpath(dirpath, root_folder)
    ntt_path     = os.path.join(dirpath, ntt_file)
    pct          = 100 * unit_idx / total_units

    with _print_lock:
        print(f'[{unit_idx}/{total_units}  {pct:.1f}%]  {session_name}  |  {ntt_file}  '
              f'(GPU {_gpu_util_pct()}%)')

    try:
        metrics = compute_metrics(csv_path, ntt_path, arena_width_cm, target_bin_cm)
    except Exception as e:
        with _print_lock:
            print(f'  ERROR in {ntt_file}: {e}')
        metrics = {k: None for k in
                   ['n_spikes', 'n_discarded', 'peak_fr', 'mean_fr',
                    'sir', 'coherence', 'sparsity']}

    metrics['session']   = session_name
    metrics['unit']      = ntt_file
    metrics['job_order'] = job_order   # for restoring insertion order

    if None not in (metrics['n_spikes'], metrics['sir'], metrics['peak_fr']):
        metrics['place_cell'] = (
            metrics['n_spikes'] > 50 and
            metrics['sir']      > 0.5 and
            metrics['peak_fr']  > 1.0
        )
    else:
        metrics['place_cell'] = None

    return metrics


# ── Batch scan ────────────────────────────────────────────────────────────────

all_jobs = []
for dirpath, _, filenames in os.walk(root_folder):
    tracking_files = [f for f in filenames if f.lower().endswith(('.csv', '.xlsx'))]
    ntt_files      = [f for f in filenames if f.lower().endswith('.ntt')]
    if len(tracking_files) == 1 and len(ntt_files) > 0:
        csv_path = os.path.join(dirpath, tracking_files[0])
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

results.sort(key=lambda r: r['job_order'])

# ── Save to Excel ─────────────────────────────────────────────────────────────

column_order = ['session', 'unit', 'n_spikes', 'n_discarded',
                'peak_fr', 'mean_fr', 'sir', 'coherence', 'sparsity', 'place_cell']

df = pd.DataFrame(results, columns=column_order)
df.to_excel(output_excel, index=False)

print(f'\nDone. Results saved to {output_excel}')
print(f'Total units processed : {len(df)}')
print(f'Place cells found     : {df["place_cell"].sum()}')
