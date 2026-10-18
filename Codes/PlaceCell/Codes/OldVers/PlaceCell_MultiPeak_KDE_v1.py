# -*- coding: utf-8 -*-
"""
Multi-peak quantification within isolated place fields – KDE method

Runs only on confirmed place cells: cells with place_cell == True on the 'Full'
sheet of the PlaceCell_Main Excel, whose .ntt + tracking files were copied into
the Output_PlaceTrue folder. Nothing else from PlaceCell_Main is re-run: each
cell's full-session ratemap is rebuilt exactly as in PlaceCell_Main (same
tracking cleaning, immobility filter, binning and Gaussian smoothing -- the
settings in the "Ratemap settings" block below must match that run), and the
fields PlaceCell_Main already isolated (PlaceFields sheet, `bin_coords`) are
reused as-is. A ratemap_check column flags any field whose recomputed smoothed
peak does not reproduce the Excel peak_fr_hz (i.e. the settings drifted).

Per isolated field:
  1. Fit a weighted Gaussian KDE (scipy.stats.gaussian_kde, Scott's rule) on
     the field's bin centres, each bin weighted by its RAW firing rate. Scaled
     by the summed rate, the KDE is in Hz (a Scott-bandwidth smoothing of the
     field's own rate map), so peak heights read as firing rates.
  2. Find the KDE's local maxima inside the field (8-neighbourhood on the 2-D
     grid; left/right neighbour on the Circle track's 1-D angular map). The
     KDE is also evaluated on a 1-bin ring around the field, so a field-edge
     bin only counts as a peak if the KDE really falls off outside it.
     A peak is counted if its KDE height is >= PEAK_REL_HEIGHT_FRAC of the
     field's highest peak.
  3. >= 2 counted peaks -> multi_peak = True, and the field is cut at the
     troughs between its peaks by marker-based watershed on the KDE (every bin
     joins the counted peak it is reached from by descending the KDE; peaks
     below the cut-off are absorbed into their neighbours). On the 1-D Circle
     map the cut falls exactly at the lowest point between neighbouring peaks;
     on the 2-D grid it follows the valley floor between the peaks.

Outputs (folder MULTIPEAK_SUBDIR inside Output_PlaceTrue):
  output_excel   'Fields'    one row per field, multi_peak True/False
                 'SubFields' one row per sub-field of every multi-peak field
                 'Cells'     one row per cell
                 'Settings'  parameters used for this run
  ratemap_dir    Gaussian-smoothed ratemap of every cell: field bins only (other
                 visited bins gray, unvisited white), field outlines black,
                 trough cuts white, counted peaks as stars; next to it the KDE
                 map with the same boundaries. Sorted into MultiPeak/ and
                 SinglePeak/ subfolders.
"""

import os
import heapq
import threading
import concurrent.futures
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.ndimage import convolve, gaussian_filter1d
from scipy.stats import gaussian_kde

# Thread-safe Matplotlib imports for parallel rendering
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as path_effects

# ── Directories (edit these per run) ──────────────────────────────────────────

# Output_PlaceTrue folder of the PlaceCell_Main run (root of the confirmed
# place cells, animal-ID folders onwards, e.g. Fa1059/Open/<session>/...)
Output_PlaceTrue   = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_CI99_2'
# Excel written by the SAME PlaceCell_Main run (needs the 'Full' and 'PlaceFields' sheets)
place_fields_excel = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\PC_SmoothCompare\CrossCheck_DoubleRun\All_TT_PlaceChar_SiSkaggsAdptBin_CI99.xlsx'

# ── Multi-peak settings ───────────────────────────────────────────────────────
PEAK_REL_HEIGHT_FRAC = 0.75     # a KDE peak is counted if its height is >= this
                                # fraction of the field's highest KDE peak
MIN_TROUGH_DIP_FRAC  = 0.0      # ...and if the KDE dips below (1 - this) x its height on the
                                # way to any higher peak (0.0: any real dip; flat ridges and
                                # plateaus never count as a second peak)
KDE_BW_METHOD        = 'scott'  # gaussian_kde bw_method ('scott', 'silverman' or a scalar factor)
MIN_KDE_POINTS       = 3        # fields with fewer bins of non-zero RAW rate get no KDE and are
                                # reported as single-peak (kde_note says why)

# Output folder (inside Output_PlaceTrue) -- named after the peak cut-off so
# runs with different PEAK_REL_HEIGHT_FRAC don't overwrite each other.
MULTIPEAK_SUBDIR = f'MultiPeak_KDE_Rel{int(round(PEAK_REL_HEIGHT_FRAC * 100))}'
output_dir   = os.path.join(Output_PlaceTrue, MULTIPEAK_SUBDIR)
output_excel = os.path.join(output_dir, 'MultiPeak_KDE_Fields.xlsx')
ratemap_dir  = os.path.join(output_dir, 'ratemaps_GaussianSmoothed')

MAX_WORKERS = 4

# ── Ratemap settings (must match the PlaceCell_Main run that wrote the Excel) ─
fps                  = 30       # tracking frame rate (Hz)
target_bin_cm        = 2.0      # bin size in cm
arena_width_cm       = 80.0     # physical arena width in cm (pixel coordinates only)
use_min_occ_s        = False    # valid bins: >= min_occ_s seconds occupancy
min_occ_s            = 0.5
use_min_visits       = True     # valid bins: visited >= min_visits times
min_visits           = 2
MAX_GAP_US           = 50_000   # max spike–position gap in µs (50 ms)
POS_JUMP_THRESH_CMS  = 90.0     # tracking-artifact speed threshold (cm/s)
POS_SMOOTH_SIGMA_SMP = 5.0      # Gaussian smoothing sigma (samples) of x/y position
IMMOBILITY_SPEED_CMS = 0.5      # frames slower than this (cm/s) are dropped
COORD_UNITS          = 'cm'     # 'cm': *_cm.csv (time col A, x col D, y col E); 'pixel': x/y/time columns
GAUSSIAN_SIGMA_CM    = 3        # ratemap Gaussian smoothing sigma (cm)
RADIAL_RANGE_CLIP_PCTILE = 0.5  # Circle track: percentile trimmed off each end of radial distance

# Saved to the 'Settings' sheet of output_excel.
RUN_CONFIG_VARS = [
    'Output_PlaceTrue', 'place_fields_excel', 'output_excel', 'ratemap_dir',
    'PEAK_REL_HEIGHT_FRAC', 'MIN_TROUGH_DIP_FRAC', 'KDE_BW_METHOD', 'MIN_KDE_POINTS',
    'MAX_WORKERS',
    'fps', 'target_bin_cm', 'arena_width_cm',
    'use_min_occ_s', 'min_occ_s', 'use_min_visits', 'min_visits',
    'MAX_GAP_US', 'POS_JUMP_THRESH_CMS', 'POS_SMOOTH_SIGMA_SMP', 'IMMOBILITY_SPEED_CMS',
    'COORD_UNITS', 'GAUSSIAN_SIGMA_CM', 'RADIAL_RANGE_CLIP_PCTILE',
]

ntt_dtype = np.dtype([
    ('timestamp',   '<u8'),
    ('sc_number',   '<u4'),
    ('cell_number', '<u4'),
    ('params',      '<u4', (8,)),
    ('waveforms',   '<i2', (32, 4)),
])

_print_lock = threading.Lock()


# ── Ratemap reconstruction (copied from PlaceCell_Main, CPU only) ─────────────

def _bin_visit_counts(flat_idx: np.ndarray, n_bins: int) -> np.ndarray:
    """Number of visits per bin: a visit starts whenever the animal enters a bin
    (first frame, or bin index differs from the previous frame)."""
    flat_idx = np.asarray(flat_idx)
    if flat_idx.size == 0:
        return np.zeros(n_bins, dtype=np.int64)
    starts = np.ones(flat_idx.size, dtype=bool)
    starts[1:] = flat_idx[1:] != flat_idx[:-1]
    return np.bincount(flat_idx[starts], minlength=n_bins)


def _valid_bin_mask(occ: np.ndarray, visits: np.ndarray) -> np.ndarray:
    """Valid bins under the enabled criteria (use_min_occ_s / use_min_visits)."""
    mask = occ > 0
    if use_min_occ_s:
        mask &= occ >= min_occ_s
    if use_min_visits:
        mask &= visits >= min_visits
    return mask


def _gaussian_kernel(sigma_bins: float) -> np.ndarray:
    """2D Gaussian kernel, sigma given in bins, truncated at 3 sigma."""
    radius = max(1, int(np.ceil(3 * sigma_bins)))
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax, indexing='ij')
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_bins ** 2))
    kernel /= kernel.sum()
    return kernel


def _gaussian_kernel_1d(sigma_bins: float) -> np.ndarray:
    """1D Gaussian kernel, sigma given in bins, truncated at 3 sigma."""
    radius = max(1, int(np.ceil(3 * sigma_bins)))
    ax = np.arange(-radius, radius + 1)
    kernel = np.exp(-(ax ** 2) / (2 * sigma_bins ** 2))
    kernel /= kernel.sum()
    return kernel


def _gaussian_smooth(fr_map: np.ndarray, valid_mask: np.ndarray, bin_cm: float) -> np.ndarray:
    """Gaussian smoothing restricted to valid bins, edge-corrected by dividing
    the convolved rates by the convolved mask."""
    kernel = _gaussian_kernel(GAUSSIAN_SIGMA_CM / bin_cm)
    fr_in   = np.where(valid_mask, fr_map, 0.0)
    mask_in = valid_mask.astype(np.float64)
    smoothed_fr      = convolve(fr_in,   kernel, mode='constant', cval=0.0)
    smoothed_weights = convolve(mask_in, kernel, mode='constant', cval=0.0)

    smoothed = np.zeros_like(smoothed_fr)
    valid_weights = smoothed_weights > 0
    smoothed[valid_weights] = smoothed_fr[valid_weights] / smoothed_weights[valid_weights]
    smoothed[~valid_mask] = 0.0
    return smoothed


def _gaussian_smooth_circular(fr_map: np.ndarray, valid_mask: np.ndarray, bin_cm: float) -> np.ndarray:
    """Gaussian smoothing on a 1D angular rate map, padded by wrapping (the
    track is a closed loop)."""
    kernel = _gaussian_kernel_1d(GAUSSIAN_SIGMA_CM / bin_cm)
    kr = kernel.shape[0] // 2
    n_theta = fr_map.shape[0]

    fr_pad   = np.pad(np.where(valid_mask, fr_map, 0.0), (kr, kr), mode='wrap')
    mask_pad = np.pad(valid_mask.astype(np.float64),     (kr, kr), mode='wrap')
    smoothed_fr      = convolve(fr_pad,   kernel, mode='constant', cval=0.0)[kr:kr + n_theta]
    smoothed_weights = convolve(mask_pad, kernel, mode='constant', cval=0.0)[kr:kr + n_theta]

    smoothed = np.zeros_like(smoothed_fr)
    valid_weights = smoothed_weights > 0
    smoothed[valid_weights] = smoothed_fr[valid_weights] / smoothed_weights[valid_weights]
    smoothed[~valid_mask] = 0.0
    return smoothed


def _load_tracking(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load, clean, and pixel→cm-convert one tracking file (same as
    PlaceCell_Main._load_tracking, full session). Returns (x_cm, y_cm, t_us)."""
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

    if len(t) >= 2:
        if COORD_UNITS == 'cm':
            units_per_cm = 1.0
        else:
            x_span = np.percentile(x, 99.5) - np.percentile(x, 0.5)
            y_span = np.percentile(y, 99.5) - np.percentile(y, 0.5)
            units_per_cm = max(x_span, y_span) / arena_width_cm
        jump_thresh_per_us = POS_JUMP_THRESH_CMS * units_per_cm * 1e-6

        dt_step   = np.diff(t)
        dxy_step  = np.hypot(np.diff(x), np.diff(y))
        valid_step = dt_step > 0
        speed_step = np.zeros_like(dxy_step)
        speed_step[valid_step] = dxy_step[valid_step] / dt_step[valid_step]

        speed    = np.append(speed_step, speed_step[-1])
        valid_dt = np.append(valid_step, valid_step[-1])

        keep = np.where(valid_dt & (speed < jump_thresh_per_us))[0]
        x, y, t = x[keep], y[keep], t[keep]

    order = np.argsort(t)
    x, y, t = x[order], y[order], t[order]

    if len(t) == 0:
        return x.astype(np.float64), y.astype(np.float64), t.astype(np.float64)

    if COORD_UNITS == 'cm':
        x_cm = x - x.min()
        y_cm = y - y.min()
    else:
        px_per_cm = max(x.max() - x.min(), y.max() - y.min()) / arena_width_cm
        x_cm = (x - x.min()) / px_per_cm
        y_cm = (y - y.min()) / px_per_cm

    return x_cm, y_cm, t


def _smooth_tracking_position(x_cm: np.ndarray, y_cm: np.ndarray,
                              t_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Iterative jump removal, linear interpolation over artifact frames, then
    Gaussian smoothing (same as PlaceCell_Main._smooth_tracking_position)."""
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

        newly_bad = step_speed > POS_JUMP_THRESH_CMS
        if not newly_bad.any():
            break
        bad[good_idx[1:][newly_bad]] = True

    good_idx = np.where(~bad)[0]
    if len(good_idx) == 0 or len(good_idx) == n:
        x_clean, y_clean = x_cm.copy(), y_cm.copy()
    else:
        x_clean = np.interp(t_us, t_us[good_idx], x_cm[good_idx])
        y_clean = np.interp(t_us, t_us[good_idx], y_cm[good_idx])

    x_smooth = gaussian_filter1d(x_clean, sigma=POS_SMOOTH_SIGMA_SMP, mode='nearest')
    y_smooth = gaussian_filter1d(y_clean, sigma=POS_SMOOTH_SIGMA_SMP, mode='nearest')
    return x_smooth, y_smooth


def _instantaneous_speed(x_cm: np.ndarray, y_cm: np.ndarray, t_us: np.ndarray) -> np.ndarray:
    """Per-frame running speed (cm/s); the last frame gets the second-last frame's speed."""
    dt_s = np.diff(t_us) * 1e-6
    with np.errstate(invalid='ignore', divide='ignore'):
        speed = np.hypot(np.diff(x_cm), np.diff(y_cm)) / dt_s
    speed[dt_s <= 0] = np.nan
    return np.append(speed, speed[-1])


def _load_session_track(csv_path: str) -> dict | None:
    """Tracking part of PlaceCell_Main.compute_metrics (steps 1-3), shared by
    every unit of a session: cleaned + smoothed position, per-frame occupancy
    time, immobility mask and position bins."""
    x_cm, y_cm, t = _load_tracking(csv_path)
    if len(t) == 0:
        return None
    x_cm, y_cm = _smooth_tracking_position(x_cm, y_cm, t)

    dt_frames     = np.empty(len(t), dtype=np.float64)
    dt_frames[0]  = 1.0 / fps
    dt_frames[1:] = np.minimum(np.diff(t) * 1e-6, 2.0 / fps)

    if len(t) >= 2:
        moving = ~(_instantaneous_speed(x_cm, y_cm, t) < IMMOBILITY_SPEED_CMS)   # NaN speed kept
    else:
        moving = np.ones(len(t), dtype=bool)
    if not moving.any():
        return None

    n_bins_x = max(1, int(np.ceil(x_cm.max() / target_bin_cm)))
    n_bins_y = max(1, int(np.ceil(y_cm.max() / target_bin_cm)))
    beh_bx = np.clip((x_cm / target_bin_cm).astype(int), 0, n_bins_x - 1)
    beh_by = np.clip((y_cm / target_bin_cm).astype(int), 0, n_bins_y - 1)

    return dict(x_cm=x_cm, y_cm=y_cm, t=t, dt_frames=dt_frames, moving=moving,
                beh_bx=beh_bx, beh_by=beh_by, n_bins_x=n_bins_x, n_bins_y=n_bins_y)


def _build_unit_ratemap(track: dict, ntt_path: str) -> dict:
    """Spike part of PlaceCell_Main.compute_metrics (steps 4-7): 50 ms
    nearest-frame assignment, immobility filter, occupancy / spike maps, valid
    bins, raw and Gaussian-smoothed ratemaps (moving frames only)."""
    t, moving = track['t'], track['moving']
    n_bins_x, n_bins_y = track['n_bins_x'], track['n_bins_y']

    spike_data = np.memmap(ntt_path, dtype=ntt_dtype, mode='r', offset=16 * 1024)
    spike_data = spike_data[spike_data['cell_number'] != 0]  # drop unsorted/discarded cluster 0
    spike_ts   = np.sort(spike_data['timestamp'].astype(np.float64))

    idx   = np.searchsorted(t, spike_ts, side='left')
    idx_l = np.clip(idx - 1, 0, len(t) - 1)
    idx_r = np.clip(idx,     0, len(t) - 1)
    dist_l  = np.abs(spike_ts - t[idx_l])
    dist_r  = np.abs(spike_ts - t[idx_r])
    nearest = np.where(dist_l <= dist_r, idx_l, idx_r)
    valid_spike  = np.minimum(dist_l, dist_r) <= MAX_GAP_US
    moving_spike = valid_spike & moving[nearest]

    new_frame_idx = np.cumsum(moving) - 1
    spike_frame   = new_frame_idx[nearest[moving_spike]]
    beh_bx    = track['beh_bx'][moving]
    beh_by    = track['beh_by'][moving]
    dt_frames = track['dt_frames'][moving]

    occ_map   = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    spike_map = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    np.add.at(occ_map,   (beh_bx, beh_by), dt_frames)
    np.add.at(spike_map, (beh_bx[spike_frame], beh_by[spike_frame]), 1.0)

    visit_map  = _bin_visit_counts(beh_bx * n_bins_y + beh_by,
                                   n_bins_x * n_bins_y).reshape(n_bins_x, n_bins_y)
    valid_mask = _valid_bin_mask(occ_map, visit_map)

    fr_raw = np.zeros_like(occ_map)
    fr_raw[valid_mask] = spike_map[valid_mask] / occ_map[valid_mask]
    fr_smooth = _gaussian_smooth(fr_raw, valid_mask, target_bin_cm)

    return dict(x_cm=track['x_cm'][moving], y_cm=track['y_cm'][moving],
                spike_frame=spike_frame, dt_frames=dt_frames,
                occ_map=occ_map, valid_mask=valid_mask,
                fr_raw=fr_raw, fr_smooth=fr_smooth,
                n_bins_x=n_bins_x, n_bins_y=n_bins_y)


def _fit_ring_centre(x_cm: np.ndarray, y_cm: np.ndarray) -> tuple:
    """Algebraic (Kasa) circle fit to the tracked positions -> (cx, cy)."""
    A = np.column_stack([x_cm, y_cm, np.ones_like(x_cm)])
    b = x_cm ** 2 + y_cm ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol[0] / 2.0, sol[1] / 2.0


def _build_ratemap_circular(ctx: dict) -> dict:
    """1-D angular ratemap of the Circle track (same as
    PlaceCell_Main._build_ratemap_circular)."""
    x_cm, y_cm  = ctx['x_cm'], ctx['y_cm']
    spike_frame = ctx['spike_frame']
    dt_frames   = ctx['dt_frames']

    cx, cy = _fit_ring_centre(x_cm, y_cm)
    r_cm        = np.hypot(x_cm - cx, y_cm - cy)
    theta_0_2pi = np.mod(np.arctan2(y_cm - cy, x_cm - cx), 2.0 * np.pi)

    r_min = float(np.percentile(r_cm, RADIAL_RANGE_CLIP_PCTILE))
    r_max = float(np.percentile(r_cm, 100.0 - RADIAL_RANGE_CLIP_PCTILE))

    bin_width_rad = target_bin_cm / max(float(np.median(r_cm)), 1e-6)
    n_bins_theta  = max(1, int(round(2.0 * np.pi / bin_width_rad)))
    bin_width_rad = 2.0 * np.pi / n_bins_theta

    beh_bt = np.clip((theta_0_2pi / bin_width_rad).astype(int), 0, n_bins_theta - 1)

    occ_map   = np.zeros(n_bins_theta, dtype=np.float64)
    spike_map = np.zeros(n_bins_theta, dtype=np.float64)
    np.add.at(occ_map,   beh_bt, dt_frames)
    np.add.at(spike_map, beh_bt[spike_frame], 1.0)

    valid_mask = _valid_bin_mask(occ_map, _bin_visit_counts(beh_bt, n_bins_theta))
    fr_raw = np.zeros_like(occ_map)
    fr_raw[valid_mask] = spike_map[valid_mask] / occ_map[valid_mask]
    fr_smooth = _gaussian_smooth_circular(fr_raw, valid_mask, target_bin_cm)

    return dict(valid_mask=valid_mask, fr_raw=fr_raw, fr_smooth=fr_smooth,
                n_bins_theta=n_bins_theta, bin_width_rad=bin_width_rad,
                r_min=r_min, r_max=r_max, track_width_cm=max(r_max - r_min, 0.0))


# ── Multi-peak detection (KDE + watershed) ────────────────────────────────────

def _watershed(values: np.ndarray, adj: list, seeds: list) -> np.ndarray:
    """Marker-based watershed (priority flood, highest value first) over the
    field nodes 0..len(values)-1. `adj[i]` lists node i's neighbours (indices
    >= len(values) are outside the field and ignored); seeds[k] gets label k+1.
    Each node takes the label of the basin that reaches it first while
    descending the surface, so basins meet at the troughs."""
    n_field = len(values)
    labels = np.zeros(n_field, dtype=int)
    heap = []
    for k, s in enumerate(seeds, start=1):
        labels[s] = k
        heapq.heappush(heap, (-values[s], s))
    while heap:
        _, i = heapq.heappop(heap)
        for j in adj[i]:
            if j < n_field and labels[j] == 0:
                labels[j] = labels[i]
                heapq.heappush(heap, (-values[j], j))
    return labels


def _merge_levels(rate: np.ndarray, adj: list) -> np.ndarray:
    """Union-find over the field bins in descending KDE order (0-D topological
    persistence). For every bin, the KDE level at which the region it tops
    merges into a region with a higher top, i.e. the trough that has to be
    crossed to reach a higher peak. Non-peak bins merge at their own height;
    the field's highest bin never merges (inf)."""
    n_field = len(rate)
    parent  = np.full(n_field, -1)
    top     = np.arange(n_field)                 # highest bin of each root's region
    merge   = np.full(n_field, np.inf)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in np.argsort(-rate, kind='stable'):
        parent[i] = i
        for j in adj[i]:
            if j >= n_field or parent[j] < 0:
                continue
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            # Higher top survives; exact ties go to the lower bin index, the
            # same order the local maxima are ranked in (_kde_multipeak).
            ti, tj = top[ri], top[rj]
            hi, lo = (ri, rj) if (rate[ti], -ti) >= (rate[tj], -tj) else (rj, ri)
            merge[top[lo]] = rate[i]
            parent[lo] = hi
    return merge


def _no_kde_result(n_field: int, w: np.ndarray, note: str) -> dict:
    """Single-peak result for a field the KDE can't be fitted on (too few
    non-zero-rate bins): the whole field is one sub-field, peak at the
    highest raw-rate bin."""
    peak = int(np.argmax(w))
    return dict(rate=np.full(n_field, np.nan), labels=np.ones(n_field, dtype=int),
                local_max=[peak], counted=[peak], trough_rel={peak: np.nan},
                ref_hz=np.nan, peak_fallback=False, saddles={},
                kde_axes='', kde_factor=np.nan, kde_neff=np.nan,
                kde_bw_bins=np.array([np.nan]), kde_n_points=int((w > 0).sum()),
                kde_note=note)


def _kde_multipeak(pts_ext: np.ndarray, n_field: int, adj: list, w: np.ndarray) -> dict:
    """KDE peak search + trough cut for one field.

    pts_ext : (d, n_ext) bin-centre coordinates (in bins) -- the field's n_field
              bins first, then a 1-bin ring around the field.
    adj     : for each field bin, the indices (into pts_ext) of its neighbours.
    w       : raw firing rate of each field bin (KDE weights).
    """
    d   = pts_ext.shape[0]
    pos = w > 0
    if pos.sum() < MIN_KDE_POINTS:
        return _no_kde_result(n_field, w, f'{int(pos.sum())} non-zero-rate bin(s) < '
                                          f'MIN_KDE_POINTS: no KDE, single peak')

    # Collinear non-zero-rate bins (a 1-bin-wide strip, or too few points)
    # make the 2-D covariance singular: fit a 1-D KDE along the field's long
    # (first principal) axis instead.
    X   = pts_ext[:, :n_field][:, pos]
    cov = np.atleast_2d(np.cov(X, aweights=w[pos]))
    evals, evecs = np.linalg.eigh(cov)
    if evals.max() <= 0:
        return _no_kde_result(n_field, w, 'all non-zero-rate bins at one location: no KDE, single peak')
    if d > 1 and evals.min() <= 1e-6 * evals.max():
        axis      = evecs[:, -1]
        proj      = axis[None, :]
        kde_axes  = f'long axis ({axis[0]:+.2f}, {axis[1]:+.2f})'
    else:
        proj      = np.eye(d)
        kde_axes  = 'theta' if d == 1 else 'x,y'
    kde = gaussian_kde(proj @ X, bw_method=KDE_BW_METHOD, weights=w[pos])

    # Σ r_i K(x - x_i) with K integrating to 1 over one bin -> rate in Hz.
    rate_ext = kde(proj @ pts_ext) * float(w.sum())
    rate     = rate_ext[:n_field]

    # Local maxima inside the field (>= every neighbour, ring bins included,
    # so a field-edge bin only counts if the KDE really falls off outside it).
    # Adjacent maxima can only be exact ties -- keep one of them.
    cand = [i for i in range(n_field) if all(rate_ext[i] >= rate_ext[j] for j in adj[i])]
    cand.sort(key=lambda i: -rate[i])
    local_max = []
    for i in cand:
        if not any(i in adj[p] for p in local_max):
            local_max.append(i)

    # The KDE's maximum can sit just outside a non-convex field; then fall back
    # to the field's highest KDE bin as its single peak.
    peak_fallback = not local_max
    if peak_fallback:
        local_max = [int(np.argmax(rate))]

    # A peak counts if it is tall enough AND a real trough separates it from
    # every higher peak (trough_rel = deepest-needed crossing / own height).
    merge      = _merge_levels(rate, adj)
    ref_hz     = float(rate[local_max[0]])
    trough_rel = {i: (np.nan if i == local_max[0] else float(merge[i] / rate[i]))
                  for i in local_max}
    counted = [local_max[0]] + [
        i for i in local_max[1:]
        if rate[i] >= PEAK_REL_HEIGHT_FRAC * ref_hz
        and merge[i] < (1.0 - MIN_TROUGH_DIP_FRAC) * rate[i]]
    labels  = _watershed(rate, adj, counted)

    # Saddle between two touching sub-fields = highest crossing point between
    # them (max over touching bin pairs of the lower of the two KDE values).
    saddles: dict[tuple[int, int], float] = {}
    for i in range(n_field):
        for j in adj[i]:
            if j < n_field and labels[i] and labels[j] and labels[i] < labels[j]:
                key = (int(labels[i]), int(labels[j]))
                saddles[key] = max(saddles.get(key, -np.inf), float(min(rate[i], rate[j])))

    return dict(rate=rate, labels=labels, local_max=local_max, counted=counted,
                trough_rel=trough_rel, ref_hz=ref_hz, peak_fallback=peak_fallback,
                saddles=saddles, kde_axes=kde_axes, kde_factor=float(kde.factor),
                kde_neff=float(kde.neff),
                kde_bw_bins=np.sqrt(np.diag(np.atleast_2d(kde.covariance))),
                kde_n_points=int(pos.sum()), kde_note='')


def _analyse_field_2d(nodes: list, fr_raw: np.ndarray) -> dict:
    """KDE multi-peak analysis of one 2-D grid field (nodes = (bx, by) bins)."""
    index = {n: k for k, n in enumerate(nodes)}
    ext_nodes = list(nodes)
    adj = []
    for bx, by in nodes:
        nbrs = []
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                if ddx == 0 and ddy == 0:
                    continue
                nb = (bx + ddx, by + ddy)
                k = index.get(nb)
                if k is None:                     # ring bin just outside the field
                    k = len(ext_nodes)
                    index[nb] = k
                    ext_nodes.append(nb)
                nbrs.append(k)
        adj.append(nbrs)

    pts_ext = np.asarray(ext_nodes, dtype=float).T + 0.5
    idx = np.asarray(nodes)
    return _kde_multipeak(pts_ext, len(nodes), adj, fr_raw[idx[:, 0], idx[:, 1]])


def _order_ring_run(bts: list) -> np.ndarray:
    """Angular bins of one Circle field in track order: a field straddling the
    theta=0/2*pi seam is rotated to start after its gap, so positions along
    the returned array are contiguous."""
    bts = np.sort(np.asarray(bts, dtype=int))
    gaps = np.where(np.diff(bts) > 1)[0]
    if len(gaps) == 0:
        return bts
    g = gaps[0]
    return np.concatenate([bts[g + 1:], bts[:g + 1]])


def _analyse_field_circular(ordered: np.ndarray, fr_raw: np.ndarray) -> dict:
    """KDE multi-peak analysis of one Circle field along the track (1-D,
    positions 0..L-1 in track order, plus one ring bin at each end)."""
    L = len(ordered)
    pos_ext = np.concatenate([np.arange(L), [-1, L]]).astype(float) + 0.5
    adj = [[i - 1 if i > 0 else L, i + 1 if i < L - 1 else L + 1] for i in range(L)]
    return _kde_multipeak(pos_ext[None, :], L, adj, fr_raw[ordered])


def _join(vals, fmt='{:.4f}') -> str:
    return ';'.join(fmt.format(v) for v in vals)


def _analyse_cell(ctx: dict, cc: dict | None, fdf: pd.DataFrame,
                  arena_type: str, session: str, unit: str) -> tuple[list, list, list]:
    """Runs the multi-peak analysis on every field of one cell.
    Returns (field_rows, sub_rows, plot_fields)."""
    field_rows, sub_rows, plot_fields = [], [], []
    circle = arena_type == 'Circle'

    for _, f in fdf.sort_values('field_number').iterrows():
        fn = int(f['field_number'])
        coords = str(f['bin_coords'])

        if circle:
            n_theta = cc['n_bins_theta']
            bw_rad  = cc['bin_width_rad']
            bts = [int(b) for b in coords.split(';')]
            if min(bts) < 0 or max(bts) >= n_theta:
                raise ValueError(f'field {fn}: bin_coords outside the {n_theta}-bin angular map')
            nodes = _order_ring_run(bts)
            res   = _analyse_field_circular(nodes, cc['fr_raw'])
            sm, raw = cc['fr_smooth'][nodes], cc['fr_raw'][nodes]
            bin_str = [str(int(b)) for b in nodes]
            theta_c = (nodes + 0.5) * bw_rad
        else:
            nodes = sorted(tuple(int(v) for v in p.split('-')) for p in coords.split(';'))
            idx = np.asarray(nodes)
            if (idx.min() < 0 or idx[:, 0].max() >= ctx['n_bins_x']
                    or idx[:, 1].max() >= ctx['n_bins_y']):
                raise ValueError(f'field {fn}: bin_coords outside the '
                                 f'{ctx["n_bins_x"]}x{ctx["n_bins_y"]} ratemap grid')
            res = _analyse_field_2d(nodes, ctx['fr_raw'])
            sm, raw = ctx['fr_smooth'][idx[:, 0], idx[:, 1]], ctx['fr_raw'][idx[:, 0], idx[:, 1]]
            bin_str = [f'{bx}-{by}' for bx, by in nodes]

        rate, labels, counted = res['rate'], res['labels'], res['counted']
        ref_hz   = res['ref_hz']
        n_peaks  = len(counted)
        multi    = n_peaks >= 2
        n_field  = len(nodes)

        # Sanity check: the rebuilt smoothed map must reproduce PlaceCell_Main's
        # field peak (peak_fr_hz, rounded to 4 dp in the Excel).
        recomputed_peak = float(sm.max())
        excel_peak      = float(f['peak_fr_hz'])
        ratemap_check   = bool(np.isclose(recomputed_peak, excel_peak, rtol=1e-3, atol=1e-3))

        peak_hz = [float(rate[c]) for c in counted]
        troughs = {}
        for (a, b), s in sorted(res['saddles'].items()):
            troughs[(a, b)] = s / min(peak_hz[a - 1], peak_hz[b - 1])

        field_rows.append({
            'session': session, 'unit': unit, 'arena_type': arena_type,
            'field_number': fn,
            'multi_peak': multi,
            'n_peaks': n_peaks,
            'n_kde_local_maxima': len(res['local_max']),
            'peak_fallback': res['peak_fallback'],
            'kde_peak_hz': round(ref_hz, 4),
            'peak_threshold_hz': round(PEAK_REL_HEIGHT_FRAC * ref_hz, 4),
            'peak_kde_hz': _join(peak_hz),
            'peak_bins': ';'.join(bin_str[c] for c in counted),
            'local_max_rel_heights': _join([rate[i] / ref_hz for i in res['local_max']], '{:.3f}'),
            'local_max_trough_rel': _join([res['trough_rel'][i] for i in res['local_max']], '{:.3f}'),
            'shallowest_trough_rel': round(max(troughs.values()), 4) if troughs else np.nan,
            'troughs_rel': ';'.join(f'{a}-{b}:{v:.3f}' for (a, b), v in troughs.items()),
            'kde_n_points': res['kde_n_points'],
            'kde_n_eff': round(res['kde_neff'], 2),
            'kde_factor': round(res['kde_factor'], 4),
            'kde_axes': res['kde_axes'],
            'kde_bw_cm': _join(res['kde_bw_bins'] * target_bin_cm, '{:.2f}'),
            'kde_note': res['kde_note'],
            'n_bins': n_field,
            'area_cm2': f.get('area_cm2'),
            'pct_of_occupied_area': f.get('pct_of_occupied_area'),
            'peak_fr_hz': excel_peak,
            'recomputed_peak_fr_hz': round(recomputed_peak, 4),
            'ratemap_check': ratemap_check,
            'bin_coords': ';'.join(sorted(bin_str, key=lambda s: tuple(int(v) for v in s.split('-')))),
        })

        plot_fields.append(dict(field_number=fn, nodes=nodes, rate=rate, labels=labels,
                                counted=counted, local_max=res['local_max'],
                                ref_hz=ref_hz, multi=multi))

        if not multi:
            continue

        # ── Sub-field rows (multi-peak fields only) ───────────────────────────
        for k in range(1, n_peaks + 1):
            members = np.where(labels == k)[0]
            seed    = counted[k - 1]
            loc     = members[int(np.argmax(sm[members]))]
            sm_m    = sm[members]
            touching = [s for (a, b), s in res['saddles'].items() if k in (a, b)]
            max_saddle = max(touching) if touching else np.nan

            row = {
                'session': session, 'unit': unit, 'arena_type': arena_type,
                'field_number': fn, 'sub_field_number': k, 'sub_field_id': f'{fn}.{k}',
                'kde_peak_hz': round(peak_hz[k - 1], 4),
                'kde_peak_rel': round(peak_hz[k - 1] / ref_hz, 4),
                'peak_fr_hz': round(float(sm[loc]), 4),
                'peak_fr_hz_raw': round(float(raw[loc]), 4),
                'field_raw_max_fr_hz': round(float(raw[members].max()), 4),
                'n_bins': len(members),
                'pct_of_field_area': round(100.0 * len(members) / n_field, 2),
                'max_saddle_hz': round(max_saddle, 4),
                'max_saddle_rel': round(max_saddle / peak_hz[k - 1], 4),
            }
            if circle:
                com = np.arctan2(np.sum(sm_m * np.sin(theta_c[members])),
                                 np.sum(sm_m * np.cos(theta_c[members]))) % (2.0 * np.pi)
                row.update({
                    'kde_peak_bin_theta': int(nodes[seed]),
                    'kde_peak_theta_deg': round(float(np.degrees(theta_c[seed])), 2),
                    'peak_bin_theta': int(nodes[loc]),
                    'peak_theta_deg': round(float(np.degrees(theta_c[loc])), 2),
                    'com_theta_deg': round(float(np.degrees(com)), 2),
                    'arc_length_cm': round(len(members) * target_bin_cm, 2),
                    'area_cm2': round(len(members) * target_bin_cm * cc['track_width_cm'], 2),
                    'bin_coords': ';'.join(bin_str[m] for m in members),
                })
            else:
                bxs, bys = idx[members, 0], idx[members, 1]
                com_x = float(np.sum(sm_m * bxs) / np.sum(sm_m))
                com_y = float(np.sum(sm_m * bys) / np.sum(sm_m))
                row.update({
                    'kde_peak_bin_x': nodes[seed][0], 'kde_peak_bin_y': nodes[seed][1],
                    'peak_bin_x': nodes[loc][0], 'peak_bin_y': nodes[loc][1],
                    'com_bin_x': round(com_x, 3), 'com_bin_y': round(com_y, 3),
                    'com_cm_x': round(com_x * target_bin_cm, 2),
                    'com_cm_y': round(com_y * target_bin_cm, 2),
                    'area_cm2': round(len(members) * target_bin_cm ** 2, 2),
                    'bin_coords': ';'.join(bin_str[m] for m in members),
                })
            sub_rows.append(row)

    return field_rows, sub_rows, plot_fields


# ── Plotting ──────────────────────────────────────────────────────────────────

_GRAY_CMAP = ListedColormap(['#bdbdbd'])     # background (visited, non-field) bins
_TXT_FX    = [path_effects.withStroke(linewidth=2, foreground='black')]


def _mask_outline_segments(mask: np.ndarray) -> list:
    """Bin-edge segments outlining the True region of a 2-D mask (bin (i, j)
    spans [i, i+1] x [j, j+1])."""
    p = np.pad(mask, 1)
    segs = []
    for i, j in zip(*np.nonzero(p[1:, 1:-1] != p[:-1, 1:-1])):     # vertical edges at x = i
        segs.append([(i, j), (i, j + 1)])
    for i, j in zip(*np.nonzero(p[1:-1, 1:] != p[1:-1, :-1])):     # horizontal edges at y = j
        segs.append([(i, j), (i + 1, j)])
    return segs


def _label_cut_segments(sub_id: np.ndarray) -> list:
    """Bin-edge segments between neighbouring bins with different non-zero labels."""
    segs = []
    a, b = sub_id[:-1, :], sub_id[1:, :]
    for i, j in zip(*np.nonzero((a > 0) & (b > 0) & (a != b))):
        segs.append([(i + 1, j), (i + 1, j + 1)])
    a, b = sub_id[:, :-1], sub_id[:, 1:]
    for i, j in zip(*np.nonzero((a > 0) & (b > 0) & (a != b))):
        segs.append([(i, j + 1), (i + 1, j + 1)])
    return segs


def _field_summary_title(plot_fields: list) -> str:
    return '  '.join(f"F{pf['field_number']}: {len(pf['counted'])} peak"
                     f"{'s' if len(pf['counted']) != 1 else ''}" for pf in plot_fields)


def _plot_cell_2d(ctx: dict, plot_fields: list, title: str, save_path: str) -> None:
    nx, ny     = ctx['n_bins_x'], ctx['n_bins_y']
    valid_mask = ctx['valid_mask']

    in_field = np.zeros((nx, ny), dtype=bool)
    sub_id   = np.zeros((nx, ny), dtype=int)      # unique id per (field, sub-field)
    kde_map  = np.full((nx, ny), np.nan)
    uid = 0
    for pf in plot_fields:
        for (bx, by), lab, val in zip(pf['nodes'], pf['labels'], pf['rate']):
            in_field[bx, by] = True
            kde_map[bx, by]  = val
            sub_id[bx, by]   = uid + max(int(lab), 1)
        uid += max(int(pf['labels'].max()), 1)

    bg       = np.where(valid_mask & ~in_field, 1.0, np.nan)
    outline  = _mask_outline_segments(in_field)
    cuts     = _label_cut_segments(sub_id)
    extent   = (0, nx, 0, ny)

    fig = Figure(figsize=(13, 5.8))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    panels = ((axes[0], ctx['fr_smooth'], f'Gaussian-smoothed ratemap (σ = {GAUSSIAN_SIGMA_CM} cm)'),
              (axes[1], kde_map,          f'Field KDE ({KDE_BW_METHOD}), peaks ≥ {PEAK_REL_HEIGHT_FRAC:.0%} of field max'))
    for ax, data, panel_title in panels:
        ax.imshow(bg.T, origin='lower', extent=extent, cmap=_GRAY_CMAP,
                  vmin=0, vmax=1, interpolation='nearest')
        shown = np.ma.masked_where(~in_field, np.nan_to_num(data))
        finite = data[in_field & np.isfinite(data)]
        vmax  = float(finite.max()) if finite.size and finite.max() > 0 else 1.0
        im = ax.imshow(shown.T, origin='lower', extent=extent, cmap='jet',
                       vmin=0, vmax=vmax, interpolation='nearest')
        fig.colorbar(im, ax=ax, label='Firing rate (Hz)', fraction=0.046, pad=0.04)

        ax.add_collection(LineCollection(outline, colors='black', linewidths=2.0))
        if cuts:
            ax.add_collection(LineCollection(cuts, colors='white', linewidths=2.5))

        for pf in plot_fields:
            for i in pf['local_max']:
                if i not in pf['counted']:
                    bx, by = pf['nodes'][i]
                    ax.plot(bx + 0.5, by + 0.5, 'x', color='black', ms=6, mew=1.5)
            for k, i in enumerate(pf['counted'], start=1):
                bx, by = pf['nodes'][i]
                ax.plot(bx + 0.5, by + 0.5, '*', ms=13, mfc='white', mec='black', mew=1.0)
                label = f"{pf['field_number']}.{k}" if pf['multi'] else f"F{pf['field_number']}"
                ax.text(bx + 0.5, by + 1.6, label, color='white', fontsize=8,
                        ha='center', va='bottom', path_effects=_TXT_FX)

        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_aspect('equal')
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel('x bin')
        ax.set_ylabel('y bin')

    fig.suptitle(f'{title}\n{_field_summary_title(plot_fields)}   '
                 f'(★ counted peak, × peak < {PEAK_REL_HEIGHT_FRAC:.0%}, white line = trough cut)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)


def _nan_at_wrap(deg: np.ndarray, *ys: np.ndarray):
    """Insert NaN where the angle wraps 360 -> 0, so a seam-straddling field
    is drawn as two pieces instead of a line across the whole axis."""
    breaks = np.where(np.diff(deg) < 0)[0] + 1
    return (np.insert(deg.astype(float), breaks, np.nan),
            *(np.insert(y.astype(float), breaks, np.nan) for y in ys))


def _plot_cell_circular(cc: dict, plot_fields: list, title: str, save_path: str) -> None:
    n          = cc['n_bins_theta']
    bw         = cc['bin_width_rad']
    r_min      = cc['r_min']
    r_max      = cc['r_max']
    valid_mask = cc['valid_mask']
    fr_smooth  = cc['fr_smooth']
    r_mid      = 0.5 * (r_min + r_max)

    in_field = np.zeros(n, dtype=bool)
    for pf in plot_fields:
        in_field[pf['nodes']] = True

    fig = Figure(figsize=(14, 5.8))
    FigureCanvasAgg(fig)
    ax_ring = fig.add_subplot(1, 2, 1, projection='polar')
    ax_lin  = fig.add_subplot(1, 2, 2)

    # ── Ring heatmap ──────────────────────────────────────────────────────────
    theta_mesh, r_mesh = np.meshgrid(np.arange(n + 1) * bw, np.array([r_min, r_max]))
    bg = np.ma.masked_invalid(np.where(valid_mask & ~in_field, 1.0, np.nan).reshape(1, n))
    ax_ring.pcolormesh(theta_mesh, r_mesh, bg, cmap=_GRAY_CMAP, vmin=0, vmax=1, shading='flat')
    vmax = float(fr_smooth[in_field].max()) if in_field.any() else 1.0
    mesh = ax_ring.pcolormesh(theta_mesh, r_mesh,
                              np.ma.masked_where(~in_field, fr_smooth).reshape(1, n),
                              cmap='jet', vmin=0, vmax=vmax, shading='flat')
    fig.colorbar(mesh, ax=ax_ring, label='Firing rate (Hz)', fraction=0.046, pad=0.08)

    for pf in plot_fields:
        L  = len(pf['nodes'])
        t0 = pf['nodes'][0] * bw
        t1 = t0 + L * bw
        th = np.linspace(t0, t1, max(20, 4 * L))
        for r in (r_min, r_max):
            ax_ring.plot(th, np.full_like(th, r), color='black', lw=2.0)
        for t in (t0, t1):
            ax_ring.plot([t, t], [r_min, r_max], color='black', lw=2.0)
        for p in range(1, L):
            if pf['labels'][p] != pf['labels'][p - 1]:
                ax_ring.plot([t0 + p * bw] * 2, [r_min, r_max], color='white', lw=2.5)
        for i in pf['local_max']:
            if i not in pf['counted']:
                ax_ring.plot(t0 + (i + 0.5) * bw, r_mid, 'x', color='black', ms=6, mew=1.5)
        for k, i in enumerate(pf['counted'], start=1):
            ax_ring.plot(t0 + (i + 0.5) * bw, r_mid, '*', ms=13, mfc='white', mec='black')
            label = f"{pf['field_number']}.{k}" if pf['multi'] else f"F{pf['field_number']}"
            ax_ring.text(t0 + (i + 0.5) * bw, r_max * 1.12, label, fontsize=8,
                         ha='center', va='center')

    ax_ring.set_theta_zero_location('E')
    ax_ring.set_theta_direction(1)
    ax_ring.set_ylim(0, r_max * 1.2)
    ax_ring.set_yticklabels([])
    ax_ring.set_title(f'Gaussian-smoothed angular ratemap (σ = {GAUSSIAN_SIGMA_CM} cm)', fontsize=10)

    # ── Linearised profile: smoothed rate + field KDE vs track angle ──────────
    deg_per_bin = np.degrees(bw)
    all_deg = (np.arange(n) + 0.5) * deg_per_bin
    ax_lin.plot(all_deg, np.where(valid_mask, fr_smooth, np.nan), color='#bdbdbd', lw=1.2,
                label='smoothed ratemap')
    for n_pf, pf in enumerate(plot_fields):
        nodes = np.asarray(pf['nodes'])
        deg = (nodes + 0.5) * deg_per_bin
        x, y_sm, y_kde = _nan_at_wrap(deg, fr_smooth[nodes], pf['rate'])
        ax_lin.plot(x, y_sm, color='tab:blue', lw=2.0, label='field (smoothed)' if n_pf == 0 else None)
        ax_lin.plot(x, y_kde, color='tab:red', lw=1.5, ls='--', label='field KDE' if n_pf == 0 else None)
        x_thr, y_thr = _nan_at_wrap(deg, np.full(len(nodes), PEAK_REL_HEIGHT_FRAC * pf['ref_hz']))
        ax_lin.plot(x_thr, y_thr, color='tab:red', lw=0.8, ls=':',
                    label=f'{PEAK_REL_HEIGHT_FRAC:.0%} of KDE max' if n_pf == 0 else None)
        for p in range(1, len(nodes)):
            if pf['labels'][p] != pf['labels'][p - 1]:
                ax_lin.axvline(((nodes[0] + p) % n) * deg_per_bin, color='black', lw=1.2, ls='-.')
        for i in pf['local_max']:
            if i not in pf['counted']:
                ax_lin.plot(deg[i], pf['rate'][i], 'x', color='black', ms=6, mew=1.5)
        for k, i in enumerate(pf['counted'], start=1):
            ax_lin.plot(deg[i], pf['rate'][i], '*', ms=13, mfc='white', mec='black')
            label = f"{pf['field_number']}.{k}" if pf['multi'] else f"F{pf['field_number']}"
            ax_lin.annotate(label, (deg[i], pf['rate'][i]), textcoords='offset points',
                            xytext=(0, 8), ha='center', fontsize=8)
    ax_lin.set_xlim(0, 360)
    ax_lin.set_xlabel('Track angle (deg)')
    ax_lin.set_ylabel('Firing rate (Hz)')
    ax_lin.set_title('Field profiles (dash-dot line = trough cut)', fontsize=10)
    ax_lin.legend(fontsize=7, loc='upper right')

    fig.suptitle(f'{title}\n{_field_summary_title(plot_fields)}   '
                 f'(★ counted peak, × peak < {PEAK_REL_HEIGHT_FRAC:.0%}, white line = trough cut)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)


# ── Per-session job (called from thread pool) ─────────────────────────────────

def _run_session(args):
    sess_idx, n_sessions, session, units = args
    dirpath = os.path.join(Output_PlaceTrue, session)
    field_rows, sub_rows, cell_rows = [], [], []

    with _print_lock:
        print(f'[{sess_idx}/{n_sessions}]  {session}  ({len(units)} cell(s))')

    if not os.path.isdir(dirpath):
        with _print_lock:
            print(f'  [SKIP] session folder not found: {dirpath}')
        return field_rows, sub_rows, cell_rows

    names = os.listdir(dirpath)
    if COORD_UNITS == 'cm':
        tracking = [f for f in names if f.lower().endswith('_cm.csv')]
    else:
        tracking = [f for f in names if f.lower().endswith(('.csv', '.xlsx'))
                    and not f.lower().endswith('_cm.csv')]
    if len(tracking) != 1:
        with _print_lock:
            print(f'  [SKIP] expected 1 tracking file in {dirpath}, found {len(tracking)}')
        return field_rows, sub_rows, cell_rows

    try:
        track = _load_session_track(os.path.join(dirpath, tracking[0]))
    except Exception as e:
        with _print_lock:
            print(f'  TRACKING ERROR in {session}: {e}')
        return field_rows, sub_rows, cell_rows
    if track is None:
        with _print_lock:
            print(f'  [SKIP] no usable tracking samples in {session}')
        return field_rows, sub_rows, cell_rows

    for unit, fdf in units:
        ntt_path = os.path.join(dirpath, unit)
        if not os.path.isfile(ntt_path):
            with _print_lock:
                print(f'  [SKIP] .ntt not found: {ntt_path}')
            continue
        arena_type = str(fdf['arena_type'].iloc[0])
        try:
            ctx = _build_unit_ratemap(track, ntt_path)
            cc  = _build_ratemap_circular(ctx) if arena_type == 'Circle' else None
            f_rows, s_rows, plot_fields = _analyse_cell(ctx, cc, fdf, arena_type, session, unit)
        except Exception as e:
            with _print_lock:
                print(f'  ERROR in {session} | {unit}: {e}')
            continue

        n_multi = sum(r['multi_peak'] for r in f_rows)
        check   = all(r['ratemap_check'] for r in f_rows)
        if not check:
            with _print_lock:
                print(f'  [WARN] {session} | {unit}: recomputed field peak != Excel peak_fr_hz '
                      f'(check the ratemap settings match the PlaceCell_Main run)')

        sub_dir   = os.path.join(ratemap_dir, 'MultiPeak' if n_multi else 'SinglePeak')
        stem      = session.replace('\\', '_').replace('/', '_') + '__' + os.path.splitext(unit)[0]
        save_path = os.path.join(sub_dir, stem + '.png')
        try:
            title = f'{session}  |  {unit}'
            if cc is not None:
                _plot_cell_circular(cc, plot_fields, title, save_path)
            else:
                _plot_cell_2d(ctx, plot_fields, title, save_path)
        except Exception as e:
            with _print_lock:
                print(f'  PLOT ERROR in {session} | {unit}: {e}')
            save_path = ''

        field_rows += f_rows
        sub_rows   += s_rows
        cell_rows.append({
            'session': session, 'unit': unit, 'arena_type': arena_type,
            'n_fields': len(f_rows),
            'n_multi_peak_fields': int(n_multi),
            'has_multi_peak_field': bool(n_multi),
            'max_n_peaks': max(r['n_peaks'] for r in f_rows),
            'ratemap_check': check,
            'ratemap_png': save_path,
        })

    return field_rows, sub_rows, cell_rows


# ── Batch run ─────────────────────────────────────────────────────────────────

FIELD_COLUMNS = ['session', 'unit', 'arena_type', 'field_number',
                 'multi_peak', 'n_peaks', 'n_kde_local_maxima', 'peak_fallback',
                 'kde_peak_hz', 'peak_threshold_hz', 'peak_kde_hz', 'peak_bins',
                 'local_max_rel_heights', 'local_max_trough_rel',
                 'shallowest_trough_rel', 'troughs_rel',
                 'kde_n_points', 'kde_n_eff', 'kde_factor', 'kde_axes', 'kde_bw_cm', 'kde_note',
                 'n_bins', 'area_cm2', 'pct_of_occupied_area',
                 'peak_fr_hz', 'recomputed_peak_fr_hz', 'ratemap_check', 'bin_coords']

SUBFIELD_COLUMNS = ['session', 'unit', 'arena_type', 'field_number',
                    'sub_field_number', 'sub_field_id',
                    'kde_peak_hz', 'kde_peak_rel', 'max_saddle_hz', 'max_saddle_rel',
                    'peak_fr_hz', 'peak_fr_hz_raw', 'field_raw_max_fr_hz',
                    'n_bins', 'area_cm2', 'pct_of_field_area',
                    # 2-D grid (Open field / Linear track)
                    'kde_peak_bin_x', 'kde_peak_bin_y', 'peak_bin_x', 'peak_bin_y',
                    'com_bin_x', 'com_bin_y', 'com_cm_x', 'com_cm_y',
                    # Angular (Circle track)
                    'kde_peak_bin_theta', 'kde_peak_theta_deg', 'peak_bin_theta',
                    'peak_theta_deg', 'com_theta_deg', 'arc_length_cm',
                    'bin_coords']

CELL_COLUMNS = ['session', 'unit', 'arena_type', 'n_fields', 'n_multi_peak_fields',
                'has_multi_peak_field', 'max_n_peaks', 'ratemap_check', 'ratemap_png']


if __name__ == "__main__":
    os.makedirs(os.path.join(ratemap_dir, 'MultiPeak'),  exist_ok=True)
    os.makedirs(os.path.join(ratemap_dir, 'SinglePeak'), exist_ok=True)

    df_full   = pd.read_excel(place_fields_excel, sheet_name='Full')
    df_fields = pd.read_excel(place_fields_excel, sheet_name='PlaceFields')

    # PlaceFields also lists cells that were overruled AFTER field detection
    # (per-field / total area caps) -- keep only final place_cell == True cells.
    place_keys = set(zip(df_full.loc[df_full['place_cell'] == True, 'session'],   # noqa: E712
                         df_full.loc[df_full['place_cell'] == True, 'unit']))     # noqa: E712
    keep = np.array([(s, u) in place_keys for s, u in zip(df_fields['session'], df_fields['unit'])],
                    dtype=bool)
    df_fields = df_fields[keep & df_fields['bin_coords'].notna().to_numpy()]

    sessions: dict[str, list] = {}
    for (session, unit), fdf in df_fields.groupby(['session', 'unit'], sort=True):
        sessions.setdefault(session, []).append((unit, fdf))
    n_cells = sum(len(u) for u in sessions.values())
    print(f'{len(place_keys)} place cell(s) in Excel; {n_cells} with fields, '
          f'across {len(sessions)} session(s).\n')

    job_args = [(i, len(sessions), session, units)
                for i, (session, units) in enumerate(sessions.items(), start=1)]

    all_fields, all_subs, all_cells = [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for f_rows, s_rows, c_rows in executor.map(_run_session, job_args):
            all_fields += f_rows
            all_subs   += s_rows
            all_cells  += c_rows

    out_fields = pd.DataFrame(all_fields, columns=FIELD_COLUMNS)
    out_subs   = pd.DataFrame(all_subs,   columns=SUBFIELD_COLUMNS)
    out_cells  = pd.DataFrame(all_cells,  columns=CELL_COLUMNS)
    out_settings = pd.DataFrame(
        [{'parameter': 'script_name',   'value': os.path.basename(__file__)},
         {'parameter': 'run_timestamp', 'value': datetime.now().isoformat(timespec='seconds')}]
        + [{'parameter': name, 'value': str(globals()[name])} for name in RUN_CONFIG_VARS])

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        out_fields.to_excel(writer,   sheet_name='Fields',    index=False)
        out_subs.to_excel(writer,     sheet_name='SubFields', index=False)
        out_cells.to_excel(writer,    sheet_name='Cells',     index=False)
        out_settings.to_excel(writer, sheet_name='Settings',  index=False)

    n_multi_fields = int(out_fields['multi_peak'].sum())
    n_multi_cells  = int(out_cells['has_multi_peak_field'].sum())
    n_bad_check    = int((~out_cells['ratemap_check'].astype(bool)).sum())
    print(f'\nDone. Results saved to {output_excel}')
    print(f'Ratemaps saved to                   : {ratemap_dir}')
    print(f'Cells analysed                      : {len(out_cells)} / {n_cells}')
    print(f'Fields analysed                     : {len(out_fields)}')
    print(f'Multi-peak fields (>= 2 peaks at '
          f'{PEAK_REL_HEIGHT_FRAC:.0%}) : {n_multi_fields}  ({len(out_subs)} sub-fields)')
    print(f'Cells with >= 1 multi-peak field    : {n_multi_cells}')
    if n_bad_check:
        print(f'[WARN] {n_bad_check} cell(s) failed ratemap_check -- their recomputed '
              f'smoothed ratemap does not match the Excel; check the ratemap settings.')
