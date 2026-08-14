# -*- coding: utf-8 -*-
"""
Place-field isolation – batch version.

Input:
    root_folder = Output_PlaceTrue folder produced by
    'PlaceCellCharacterization_SI_Spar_Cohr_PeakFR_MeanFR_Shuffling_TwoHalvesCopmare_ThetaMod_Batch_GPU_Final.py'
    i.e. C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True
    (each sub-folder holds the confirmed-place-cell .ntt file(s) plus their tracking file).

Algorithm (per .ntt file):

    Part 1 – detect the current highest place field
        1. Find the bin with the peak (smoothed) firing rate.
        2. Region-grow (8-connected) to all neighbouring occupied bins whose
           firing rate is >= 20% of that peak bin's rate.
        3. The region (including the peak bin) must contain >= 4% of the total
           number of occupied bins. If not, move to the bin with the next
           highest firing rate and repeat until a field satisfies the criterion
           (or no candidate bin remains).
        4. Characterise the field: area (bins & cm^2), % of the occupied arena
           it covers, its boundaries (bounding box + exact bin list).

    Part 2 – iterate to find every field
        5. Set every bin belonging to already-detected fields to the firing
           rate of the (originally) lowest-firing occupied bin ("turned off").
        6. Recompute SIR / sparsity / peak / mean firing rate and rerun the
           location-shuffling bootstrap on this modified map.
        7. If the cell is still spatially selective (same place-cell criteria
           as the source characterization script), go back to step 1 on the
           modified map to isolate the next field. Otherwise stop.

Output:
    A single Excel workbook. Each .ntt file gets its own sheet (one row per
    detected field). A final 'Summary' sheet lists every .ntt file analysed,
    the number of fields found, and each field's size (cm^2) / % of the
    occupied arena it covers.
"""

import os
import re
import time
import random
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

root_folder  = r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True'
output_excel = r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/PlaceFields.xlsx'

fps            = 30           # tracking frame rate (Hz)
target_bin_cm  = 2.0          # bin size in cm
arena_width_cm = 80.0         # physical arena width in cm
min_occ_s      = 1.0          # exclude bins with < 1 s occupancy
MAX_GAP_US     = 50_000       # max spike-position gap in µs (50 ms)
N_BOOTSTRAP    = 1000         # circular-shift shuffles for SIR significance

FIELD_RATE_FRAC     = 0.20    # neighbouring bins must be >= 20% of the peak bin
MIN_FIELD_SIZE_FRAC = 0.04    # a field must span >= 4% of all occupied bins

MAX_GPU_UTIL_PCT = 60
MAX_WORKERS      = 4

# 'pixel' or 'cm' – set interactively at startup (see __main__ below).
COORD_UNITS = 'pixel'

_gpu_semaphore = threading.Semaphore(2)

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

_print_lock = threading.Lock()


# ── Smoothing ──────────────────────────────────────────────────────────────────

def _wait_for_gpu_slot(poll_interval: float = 0.5):
    if not _GPU or not _NVML:
        return
    while _gpu_util_pct() >= MAX_GPU_UTIL_PCT:
        time.sleep(poll_interval)


def _triangular_smooth(fr_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    fr_in   = np.where(valid_mask, fr_map, 0.0)
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
        smoothed_fr      = convolve(fr_in,   _TRIANGULAR_KERNEL, mode='constant', cval=0.0)
        smoothed_weights = convolve(mask_in, _TRIANGULAR_KERNEL, mode='constant', cval=0.0)

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


def _sir_only(fr_map: np.ndarray, occ_map: np.ndarray, valid_mask: np.ndarray) -> float:
    total_occ_s = occ_map[valid_mask].sum()
    pi_flat     = occ_map[valid_mask] / total_occ_s
    ri_flat     = fr_map[valid_mask]
    r_mean      = float(np.sum(pi_flat * ri_flat))
    if r_mean <= 0:
        return 0.0
    nonzero = ri_flat > 0
    ratio   = ri_flat[nonzero] / r_mean
    return float(np.sum(pi_flat[nonzero] * ratio * np.log2(ratio)))


# ── Bootstrap (location-shuffle), with optional field suppression ──────────────

def _sir_from_spikes_locshuf(spike_frame_indices: np.ndarray, rnd: int,
                              beh_bx: np.ndarray, beh_by: np.ndarray,
                              occ_map: np.ndarray, valid_mask: np.ndarray,
                              n_bins_x: int, n_bins_y: int,
                              field_mask: np.ndarray | None = None,
                              baseline_rate: float = 0.0) -> float:
    """SIR after circularly shifting position by `rnd` frames (spike-frame
    assignment fixed). If `field_mask` is given, those bins are clamped to
    `baseline_rate` after smoothing, exactly as done for the real map when
    isolating additional fields."""
    n_frames   = len(beh_bx)
    shuf_frame = (spike_frame_indices + rnd) % n_frames

    spike_map = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    np.add.at(spike_map, (beh_bx[shuf_frame], beh_by[shuf_frame]), 1.0)

    fr_raw = np.zeros_like(spike_map)
    np.divide(spike_map, occ_map, out=fr_raw, where=valid_mask)

    fr_smooth = _triangular_smooth(fr_raw, valid_mask)
    if field_mask is not None and field_mask.any():
        fr_smooth = fr_smooth.copy()
        fr_smooth[field_mask] = baseline_rate

    return _sir_only(fr_smooth, occ_map, valid_mask)


def _run_bootstrap(spike_frame_indices: np.ndarray, t: np.ndarray,
                    beh_bx: np.ndarray, beh_by: np.ndarray,
                    occ_map: np.ndarray, valid_mask: np.ndarray,
                    n_bins_x: int, n_bins_y: int,
                    real_sir: float, ntt_path: str, label: str = '',
                    field_mask: np.ndarray | None = None,
                    baseline_rate: float = 0.0,
                    save_plot: bool = True) -> dict:
    if len(spike_frame_indices) == 0:
        return {'bootstrap_mean': float('nan'), 'bootstrap_p95': float('nan'), 'bootstrap_sig': False}

    n_frames      = len(t)
    MARGIN_FRAMES = int(20 * fps)

    if n_frames <= 2 * MARGIN_FRAMES:
        return {'bootstrap_mean': float('nan'), 'bootstrap_p95': float('nan'), 'bootstrap_sig': None}

    sir_i = np.zeros(N_BOOTSTRAP, dtype=np.float64)
    for i in range(N_BOOTSTRAP):
        rnd      = random.randint(MARGIN_FRAMES, n_frames - MARGIN_FRAMES)
        sir_i[i] = _sir_from_spikes_locshuf(spike_frame_indices, rnd,
                                             beh_bx, beh_by, occ_map, valid_mask,
                                             n_bins_x, n_bins_y,
                                             field_mask=field_mask, baseline_rate=baseline_rate)

    bootstrap_mean = float(np.mean(sir_i))
    bootstrap_p95  = float(np.percentile(sir_i, 95))
    bootstrap_sig  = bool(real_sir > bootstrap_p95)

    if save_plot:
        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        hist_result = ax.hist(sir_i, 100, color='black')
        counts: np.ndarray = np.asarray(hist_result[0])
        max_count = float(counts.max()) if counts.max() > 0 else 1.0

        box_plot = ax.boxplot(sir_i, whis=[5, 95], vert=False, showfliers=False,  # type: ignore
                               positions=[-max_count / 10], widths=max_count / 15)
        ax.plot([real_sir, real_sir], [0, max_count], 'r-.')

        bs_av = f"Bootstrap mean SIR = {bootstrap_mean:.3f} bits/spike"
        bs_p  = (f"Cell SIR = {real_sir:.3f} bits/spike "
                 f"({'p < 0.05' if bootstrap_sig else 'ns'})")
        ax.set_title(bs_av + '\n' + bs_p, multialignment='center')
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

    return {'bootstrap_mean': round(bootstrap_mean, 4),
            'bootstrap_p95':  round(bootstrap_p95, 4),
            'bootstrap_sig':  bootstrap_sig}


def _is_place_cell(n_spikes, peak_fr, sir, sparsity, bootstrap_sig) -> bool | None:
    if None in (n_spikes, peak_fr, sir, sparsity, bootstrap_sig):
        return None
    return (int(n_spikes) > 50 and
            1.0 < float(peak_fr) < 15.0 and
            float(sir) > 0.5 and
            float(sparsity) < 0.9 and
            bootstrap_sig is True)


# ── Tracking + spike loading / rate-map construction ────────────────────────────

def build_ratemap(csv_path: str, ntt_path: str,
                   arena_width_cm: float, target_bin_cm: float) -> tuple:
    """Loads tracking + spikes and returns (metrics, ctx) where ctx carries
    everything needed for field detection & bootstrapping: fr_raw, fr_smooth,
    occ_map, valid_mask, spike_frame indices, t, beh_bx, beh_by, n_bins_x/y."""

    data = (pd.read_excel(csv_path) if csv_path.lower().endswith('.xlsx')
            else pd.read_csv(csv_path))

    if COORD_UNITS == 'cm':
        t = np.asarray(data.iloc[:, 0], dtype=float)
        x = np.asarray(data.iloc[:, 3], dtype=float)
        y = np.asarray(data.iloc[:, 4], dtype=float)
    else:
        x = np.asarray(data['x'],    dtype=float)
        y = np.asarray(data['y'],    dtype=float)
        t = np.asarray(data['time'], dtype=float)

    mask = ~np.isin(x, [1, -1])
    x, y, t = x[mask], y[mask], t[mask]

    dx = np.append(np.diff(x), 0)
    dy = np.append(np.diff(y), 0)
    dt = np.append(np.diff(t), 1)

    dxy = np.hypot(dx, dy)
    valid_dt = dt > 0

    speed = np.zeros_like(dxy)
    speed[valid_dt] = dxy[valid_dt] / dt[valid_dt]

    keep = np.where(valid_dt & (speed < 0.006))[0]
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

    n_bins_x = int(np.ceil(x_cm.max() / target_bin_cm))
    n_bins_y = int(np.ceil(y_cm.max() / target_bin_cm))

    beh_bx = np.clip((x_cm / target_bin_cm).astype(int), 0, n_bins_x - 1)
    beh_by = np.clip((y_cm / target_bin_cm).astype(int), 0, n_bins_y - 1)

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

    sp_bx = beh_bx[spike_frame]
    sp_by = beh_by[spike_frame]

    dt_frames     = np.empty(len(t), dtype=np.float64)
    dt_frames[0]  = 1.0 / fps
    raw_dt        = np.diff(t) * 1e-6
    max_frame_s   = 2.0 / fps
    dt_frames[1:] = np.minimum(raw_dt, max_frame_s)

    occ_map   = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    spike_map = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)

    np.add.at(occ_map,   (beh_bx, beh_by), dt_frames)
    np.add.at(spike_map, (sp_bx,  sp_by),  1.0)

    valid_mask = occ_map >= min_occ_s

    fr_raw = np.zeros_like(occ_map)
    fr_raw[valid_mask] = spike_map[valid_mask] / occ_map[valid_mask]
    fr_smooth = _triangular_smooth(fr_raw, valid_mask)

    ctx = dict(spike_ts=spike_ts[valid_spike], spike_frame=spike_frame, t=t,
               beh_bx=beh_bx, beh_by=beh_by,
               occ_map=occ_map, valid_mask=valid_mask,
               fr_raw=fr_raw, fr_smooth=fr_smooth,
               n_bins_x=n_bins_x, n_bins_y=n_bins_y)

    if not valid_mask.any():
        return ({'n_spikes': n_spikes, 'n_discarded': n_discarded,
                  'peak_fr': 0.0, 'mean_fr': 0.0, 'sir': 0.0, 'sparsity': 0.0}, ctx)

    base = _metrics_from_ratemap(fr_smooth, occ_map, valid_mask)
    base['n_spikes']    = n_spikes
    base['n_discarded'] = n_discarded
    return base, ctx


# ── Field isolation ──────────────────────────────────────────────────────────

def _flood_fill(seed, fr_map, valid_mask, field_mask, threshold, n_bins_x, n_bins_y):
    visited = set()
    stack = [seed]
    visited.add(seed)
    while stack:
        bx, by = stack.pop()
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                if ddx == 0 and ddy == 0:
                    continue
                nx, ny = bx + ddx, by + ddy
                if not (0 <= nx < n_bins_x and 0 <= ny < n_bins_y):
                    continue
                nb = (nx, ny)
                if nb in visited or not valid_mask[nx, ny] or field_mask[nx, ny]:
                    continue
                if fr_map[nx, ny] >= threshold:
                    visited.add(nb)
                    stack.append(nb)
    return visited


def _find_one_field(fr_map, valid_mask, field_mask, min_size_bins, n_bins_x, n_bins_y):
    """Part 1, steps 1-3: peak bin -> region grow -> size check -> next-highest
    bin on failure. Returns the winning set of (bx, by) bins, or None."""
    bx_idx, by_idx = np.where(valid_mask & ~field_mask)
    if bx_idx.size == 0:
        return None
    rates = fr_map[bx_idx, by_idx]
    order = np.argsort(rates)[::-1]  # descending

    for k in order:
        seed = (int(bx_idx[k]), int(by_idx[k]))
        peak_val = fr_map[seed]
        if peak_val <= 0:
            break
        threshold = FIELD_RATE_FRAC * peak_val
        region = _flood_fill(seed, fr_map, valid_mask, field_mask, threshold, n_bins_x, n_bins_y)
        if len(region) >= min_size_bins:
            return region
    return None


def detect_place_fields(base_metrics: dict, ctx: dict, ntt_path: str,
                         target_bin_cm: float) -> list[dict]:
    occ_map    = ctx['occ_map']
    valid_mask = ctx['valid_mask']
    fr_smooth  = ctx['fr_smooth']
    n_bins_x   = ctx['n_bins_x']
    n_bins_y   = ctx['n_bins_y']
    spike_frame = ctx['spike_frame']
    t          = ctx['t']
    beh_bx     = ctx['beh_bx']
    beh_by     = ctx['beh_by']
    n_spikes   = base_metrics.get('n_spikes')

    total_valid_bins = int(valid_mask.sum())
    min_size_bins    = max(1, int(np.ceil(MIN_FIELD_SIZE_FRAC * total_valid_bins)))
    baseline_rate    = float(fr_smooth[valid_mask].min())

    field_mask  = np.zeros_like(valid_mask, dtype=bool)
    current_fr  = fr_smooth.copy()
    fields      = []
    field_num   = 1

    while True:
        region = _find_one_field(current_fr, valid_mask, field_mask, min_size_bins, n_bins_x, n_bins_y)
        if region is None:
            with _print_lock:
                print(f'  [{os.path.basename(ntt_path)}] no further field found after field {field_num - 1}.')
            break

        bxs = np.array([b[0] for b in region])
        bys = np.array([b[1] for b in region])
        rates_in_region = current_fr[bxs, bys]
        peak_local_idx = int(np.argmax(rates_in_region))
        peak_bin = (int(bxs[peak_local_idx]), int(bys[peak_local_idx]))
        peak_val = float(rates_in_region[peak_local_idx])

        n_bins_field = len(region)
        area_cm2     = n_bins_field * (target_bin_cm ** 2)
        pct_area     = 100.0 * n_bins_field / total_valid_bins

        bin_coords_str = ';'.join(f'{bx}-{by}' for bx, by in sorted(region))

        # ── suppress this field and re-test spatial selectivity ────────────────
        for b in region:
            field_mask[b[0], b[1]] = True
        suppressed_fr = current_fr.copy()
        suppressed_fr[field_mask] = baseline_rate

        after = _metrics_from_ratemap(suppressed_fr, occ_map, valid_mask)
        boot  = _run_bootstrap(spike_frame, t, beh_bx, beh_by, occ_map, valid_mask,
                                n_bins_x, n_bins_y, after['sir'], ntt_path,
                                label=f'field{field_num}_removed',
                                field_mask=field_mask, baseline_rate=baseline_rate)
        after.update(boot)
        still_place_cell = _is_place_cell(n_spikes, after['peak_fr'], after['sir'],
                                           after['sparsity'], after['bootstrap_sig'])

        fields.append({
            'field_number':          field_num,
            'peak_bin_x':            peak_bin[0],
            'peak_bin_y':            peak_bin[1],
            'peak_fr_hz':            round(peak_val, 4),
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
            'sir_after_removal':            after['sir'],
            'sparsity_after_removal':       after['sparsity'],
            'peak_fr_after_removal':        after['peak_fr'],
            'mean_fr_after_removal':        after['mean_fr'],
            'bootstrap_mean_after_removal': after.get('bootstrap_mean'),
            'bootstrap_p95_after_removal':  after.get('bootstrap_p95'),
            'bootstrap_sig_after_removal':  after.get('bootstrap_sig'),
            'still_place_cell_after_removal': still_place_cell,
        })

        if not still_place_cell:
            break

        current_fr = suppressed_fr
        field_num += 1

    return fields


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

    with _print_lock:
        print(f'[{unit_idx}/{total_units}  {pct:.1f}%]  {session_name}  |  {ntt_file}  (GPU {_gpu_util_pct()}%)')

    try:
        base_metrics, ctx = build_ratemap(csv_path, ntt_path, arena_width_cm, target_bin_cm)
    except Exception as e:
        with _print_lock:
            print(f'  ERROR building rate map for {ntt_file}: {e}')
        return session_name, ntt_file, [], {'error': str(e)}

    if not ctx or not ctx.get('valid_mask', np.array([])).any():
        return session_name, ntt_file, [], {'error': 'no valid occupancy / tracking data'}

    try:
        fields = detect_place_fields(base_metrics, ctx, ntt_path, target_bin_cm)
    except Exception as e:
        with _print_lock:
            print(f'  ERROR detecting fields in {ntt_file}: {e}')
        return session_name, ntt_file, [], {'error': str(e)}

    summary = {
        'n_spikes':  base_metrics.get('n_spikes'),
        'base_sir':  base_metrics.get('sir'),
        'base_sparsity': base_metrics.get('sparsity'),
        'base_peak_fr':  base_metrics.get('peak_fr'),
        'base_mean_fr':  base_metrics.get('mean_fr'),
    }
    return session_name, ntt_file, fields, summary


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

    field_columns = [
        'field_number', 'peak_bin_x', 'peak_bin_y', 'peak_fr_hz',
        'n_bins', 'area_cm2', 'pct_of_occupied_area',
        'bbox_x_min', 'bbox_x_max', 'bbox_y_min', 'bbox_y_max',
        'bbox_cm_x_min', 'bbox_cm_x_max', 'bbox_cm_y_min', 'bbox_cm_y_max',
        'bin_coords', 'total_occupied_bins', 'min_field_size_bins',
        'sir_after_removal', 'sparsity_after_removal',
        'peak_fr_after_removal', 'mean_fr_after_removal',
        'bootstrap_mean_after_removal', 'bootstrap_p95_after_removal',
        'bootstrap_sig_after_removal', 'still_place_cell_after_removal',
    ]

    # ── Pre-pass: assign sheet names & build the summary rows first, so the
    # 'Summary' sheet can be written before the per-unit sheets (Excel keeps
    # sheets in write order, so writing it first makes it the first tab). ────

    used_sheet_names = set()
    summary_rows = []
    sorted_results = sorted(results, key=lambda r: (r[0], r[1]))

    for session_name, ntt_file, fields, summary in sorted_results:
        sheet_name = _safe_sheet_name(os.path.splitext(ntt_file)[0], used_sheet_names)

        areas = [f['area_cm2'] for f in fields]
        pcts  = [f['pct_of_occupied_area'] for f in fields]

        summary_rows.append({
            'session': session_name,
            'unit': ntt_file,
            'sheet_name': sheet_name,
            'n_fields_detected': len(fields),
            'field_areas_cm2':  '; '.join(f'{a:.2f}' for a in areas),
            'field_pct_areas':  '; '.join(f'{p:.2f}' for p in pcts),
            'total_field_area_cm2': round(sum(areas), 2) if areas else 0.0,
            'total_pct_area_occupied': round(sum(pcts), 2) if pcts else 0.0,
            'n_spikes': summary.get('n_spikes'),
            'base_sir': summary.get('base_sir'),
            'base_sparsity': summary.get('base_sparsity'),
            'base_peak_fr': summary.get('base_peak_fr'),
            'base_mean_fr': summary.get('base_mean_fr'),
            'error': summary.get('error', ''),
        })

    summary_columns = [
        'session', 'unit', 'sheet_name', 'n_fields_detected',
        'field_areas_cm2', 'field_pct_areas',
        'total_field_area_cm2', 'total_pct_area_occupied',
        'n_spikes', 'base_sir', 'base_sparsity', 'base_peak_fr', 'base_mean_fr', 'error',
    ]
    df_summary = pd.DataFrame(summary_rows, columns=summary_columns)

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

        for (session_name, ntt_file, fields, summary), row in zip(sorted_results, summary_rows):
            sheet_name = row['sheet_name']
            if fields:
                df = pd.DataFrame(fields, columns=field_columns)
            else:
                df = pd.DataFrame(columns=field_columns)
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f'\nDone. Results saved to {output_excel}')
    print(f'Total units processed : {len(results)}')
    print(f'Total fields detected  : {sum(r["n_fields_detected"] for r in summary_rows)}')
