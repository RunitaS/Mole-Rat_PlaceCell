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
import threading
import time
import concurrent.futures
import types
import random
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pandas as pd
from scipy.ndimage import convolve

# Thread-safe Matplotlib imports for parallel rendering
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


class _Metrics(TypedDict, total=False):
    n_spikes:        int | None
    n_discarded:     int | None
    peak_fr:         float | None
    mean_fr:         float | None
    sir:             float | None
    bootstrap_mean:  float | None
    bootstrap_p95:   float | None
    bootstrap_sig:   bool | None
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

root_folder  = r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True/Fa23BD'
output_excel = r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True/Fa23BD/sir_shuff_23BD_Opn.xlsx'

fps            = 30           # tracking frame rate (Hz)
target_bin_cm  = 2.0          # bin size in cm
arena_width_cm = 80.0         # physical arena width in cm
min_occ_s      = 1.0          # exclude bins with < 1 s occupancy
MAX_GAP_US     = 50_000       # max spike–position gap in µs (50 ms)
N_BOOTSTRAP    = 1000         # circular-shift shuffles for SIR significance

MAX_GPU_UTIL_PCT = 60
MAX_WORKERS      = 4

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
                   real_sir: float, ntt_path: str) -> dict:
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
    # Thread-safe Object-Oriented Figure generation prevents race conditions
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    hist_result = ax.hist(sir_i, 100, color='black')
    counts: np.ndarray = np.asarray(hist_result[0])
    max_count = float(counts.max()) if counts.max() > 0 else 1.0

    box_plot = ax.boxplot(sir_i, whis=[5, 95], vert=False, showfliers=False,
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

    ntt_name = os.path.splitext(os.path.basename(ntt_path))[0]
    save_dir  = os.path.join(os.path.dirname(ntt_path), 'shuffling')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{ntt_name}_bootstrapping.png')
    fig.savefig(save_path, dpi=150)
    print(f'  [SAVED] {save_path}')

    return {'bootstrap_mean': round(bootstrap_mean, 4),
            'bootstrap_p95':  round(bootstrap_p95,  4),
            'bootstrap_sig':  bootstrap_sig}


# ── Core metric computation ───────────────────────────────────────────────────

def compute_metrics(csv_path: str, ntt_path: str,
                    arena_width_cm: float, target_bin_cm: float) -> tuple:

    # ── 1. Load & clean tracking ──────────────────────────────────────────────
    data = (pd.read_excel(csv_path) if csv_path.lower().endswith('.xlsx')
            else pd.read_csv(csv_path))

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
    
    if len(t) == 0:
         return ({'n_spikes': 0, 'n_discarded': 0, 'peak_fr': 0.0, 'mean_fr': 0.0, 'sir': 0.0}, {})

    # ── 2. Pixel → cm conversion ──────────────────────────────────────────────
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
               beh_bx=beh_bx, beh_by=beh_by,
               occ_map=occ_map, valid_mask=valid_mask,
               n_bins_x=n_bins_x, n_bins_y=n_bins_y)

    if not valid_mask.any():
        return ({'n_spikes': n_spikes, 'n_discarded': n_discarded,
                 'peak_fr': 0.0, 'mean_fr': 0.0, 'sir': 0.0}, ctx)

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

    metrics = {
        'n_spikes':    n_spikes,
        'n_discarded': n_discarded,
        'peak_fr':     round(peak_fr, 4),
        'mean_fr':     round(mean_fr, 4),
        'sir':         round(sir,     4),
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

    # ── main metrics ──────────────────────────────────────────────────────────
    try:
        metrics, ctx = compute_metrics(csv_path, ntt_path, arena_width_cm, target_bin_cm)
    except Exception as e:
        with _print_lock:
            print(f'  ERROR in {ntt_file}: {e}')
        metrics: _Metrics = {
            'n_spikes': None, 'n_discarded': None,
            'peak_fr':  None, 'mean_fr':     None, 'sir': None,
            'bootstrap_mean': None, 'bootstrap_p95': None, 'bootstrap_sig': None,
            'session': session_name, 'unit': ntt_file,
            'job_order': job_order, 'place_cell': None,
        }
        return metrics

    # ── bootstrap ─────────────────────────────────────────────────────────────
    if not ctx:
        bootst = _NULL_BOOTSTRAP
    else:
        try:
            bootst = _run_bootstrap(
                ctx['spike_frame'], ctx['t'], ctx['beh_bx'], ctx['beh_by'],
                ctx['occ_map'], ctx['valid_mask'], ctx['n_bins_x'], ctx['n_bins_y'],
                metrics.get('sir', 0.0), ntt_path,
            )
        except Exception as e:
            with _print_lock:
                print(f'  BOOTSTRAP ERROR in {ntt_file}: {e}')
            bootst = _NULL_BOOTSTRAP

    # # ── explicit dictionary assignment instead of .update() for Pylance ───────
    # metrics['bootstrap_mean'] = bootst.get('bootstrap_mean')
    # metrics['bootstrap_p95']  = bootst.get('bootstrap_p95')
    # metrics['bootstrap_sig']  = bootst.get('bootstrap_sig')
    
    # metrics['session']   = session_name
    # metrics['unit']      = ntt_file
    # metrics['job_order'] = job_order

    # # ── safer dictionary access for optional TypedDict items ──────────────────
    # if None not in (metrics.get('n_spikes'), metrics.get('sir'),
    #                 metrics.get('peak_fr'),  metrics.get('bootstrap_sig')):
    #     metrics['place_cell'] = (
    #         int(metrics.get('n_spikes', 0))    > 50  and
    #         float(metrics.get('sir', 0.0))     > 0.5 and
    #         float(metrics.get('peak_fr', 0.0)) > 1.0 and
    #         metrics.get('bootstrap_sig') is True
    #     )
    # else:
    #     metrics['place_cell'] = None

    # return metrics

# ── explicit dictionary assignment instead of .update() for Pylance ───────
    metrics['bootstrap_mean'] = bootst.get('bootstrap_mean')
    metrics['bootstrap_p95']  = bootst.get('bootstrap_p95')
    metrics['bootstrap_sig']  = bootst.get('bootstrap_sig')
    
    metrics['session']   = session_name
    metrics['unit']      = ntt_file
    metrics['job_order'] = job_order

    # ── Type-narrowing for strict Pylance checks ──────────────────────────────
    n_spikes = metrics.get('n_spikes')
    sir      = metrics.get('sir')
    peak_fr  = metrics.get('peak_fr')
    boot_sig = metrics.get('bootstrap_sig')

    if (n_spikes is not None) and (sir is not None) and (peak_fr is not None) and (boot_sig is not None):
        metrics['place_cell'] = (
            int(n_spikes) > 50 and
            float(sir) > 0.5 and
            float(peak_fr) > 1.0 and
            boot_sig is True
        )
    else:
        metrics['place_cell'] = None

    return metrics
# ── Batch scan ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_jobs = []
    output_excel_basename = os.path.basename(output_excel).lower()
    for dirpath, _, filenames in os.walk(root_folder):
        tracking_files = [f for f in filenames
                          if f.lower().endswith(('.csv', '.xlsx'))
                          and f.lower() != output_excel_basename]
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
                    'peak_fr', 'mean_fr', 'sir',
                    'bootstrap_mean', 'bootstrap_p95', 'bootstrap_sig', 'place_cell']

    df = pd.DataFrame(results, columns=column_order)
    df.to_excel(output_excel, index=False)

    print(f'\nDone. Results saved to {output_excel}')
    print(f'Total units processed : {len(df)}')
    print(f'Place cells found     : {df["place_cell"].sum()}')