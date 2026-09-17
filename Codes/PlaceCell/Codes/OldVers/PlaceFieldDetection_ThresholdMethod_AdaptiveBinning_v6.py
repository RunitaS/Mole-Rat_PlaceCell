# -*- coding: utf-8 -*-
"""
Place-field isolation – batch version.

Input:
    root_folder = Output_PlaceTrue folder produced by
    'PlaceCellCharacterization_SI_Spar_Cohr_PeakFR_MeanFR_Shuffling_TwoHalvesCopmare_ThetaMod_Batch_GPU_Final.py'
    i.e. C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True
    (each sub-folder holds the confirmed-place-cell .ntt file(s) plus their tracking file).

Algorithm (per .ntt file) – "threshold method", adapted from a MATLAB
`placefield`/`getLegals` reference implementation:
    1. Keep only occupied bins whose adaptively-binned firing rate (Skaggs &
       McNaughton 1998 criterion, ported from Roddy Grieves' rate_mapper.m
       'adaptive' method) is both >= METHOD2_RATE_THRESHOLD_FRAC of the
       cell's peak rate AND above the cell's mean firing rate.
    2. 8-connected-component label the surviving bins.
    3. Components spanning >= MIN_FIELD_SIZE_BINS contiguous (8-connected,
       no discontinuity) bins are reported as place fields (peak bin +
       firing-rate-weighted centre of mass, mirroring the MATLAB
       reference's `fieldPos`).
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
import threading
import concurrent.futures
import types

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ── GPU status (informational only – adaptive binning runs on CPU) ────────────

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
MAX_SPEED_CM_S = 90           # frame-to-frame speed above this is treated as a tracking-jump artifact and dropped

MIN_FIELD_SIZE_BINS = 9       # a field must span >= 9 contiguous (8-connected) bins

METHOD2_RATE_THRESHOLD_FRAC = 0.20    # "threshold method": bins must be >= 10% of the cell's peak rate
                                       # (mirrors pTreshold in the MATLAB placefield reference); bins must
                                       # also be above the cell's mean firing rate (see detect_place_fields_threshold)

MAX_WORKERS      = 4

# 'pixel' or 'cm' – set interactively at startup (see __main__ below).
COORD_UNITS = 'pixel'

# Adaptive binning (Skaggs & McNaughton 1998, eq. 11), ported from Roddy
# Grieves' rate_mapper.m 'adaptive' method (github.com/RoddyMGrieves/rate_mapper).
ADAPTIVE_ALPHA_BASE = 0.0001  # "a" in r >= a / (n * sqrt(s)); applied to raw
                               # occupancy SAMPLE COUNTS (n), not occupancy time
ADAPTIVE_MINDIST_CM = 4.0     # checkpoint radius (cm): if the occupancy TIME
                               # enclosed within this fixed radius doesn't reach
                               # min_occ_s, the bin is invalid outright
ADAPTIVE_MAXDIST_CM = 64.0    # max search radius (cm) before an adaptively
                               # binned bin gives up expanding

ntt_dtype = np.dtype([
    ('timestamp',   '<u8'),
    ('sc_number',   '<u4'),
    ('cell_number', '<u4'),
    ('params',      '<u4', (8,)),
    ('waveforms',   '<i2', (32, 4)),
])

_print_lock = threading.Lock()


# ── Adaptive binning ───────────────────────────────────────────────────────────

def _adaptive_binned_ratemap(occ_map: np.ndarray, spike_map: np.ndarray,
                              occ_count_map: np.ndarray,
                              min_occ_s: float, fps: float, target_bin_cm: float,
                              alpha_base: float = ADAPTIVE_ALPHA_BASE,
                              mindist_cm: float = ADAPTIVE_MINDIST_CM,
                              maxdist_cm: float = ADAPTIVE_MAXDIST_CM) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive binning firing rate map, ported from Roddy Grieves' rate_mapper.m
    'adaptive' method (Skaggs & McNaughton 1998, eq. 11; see
    https://github.com/RoddyMGrieves/rate_mapper). For every bin, a circular
    neighbourhood (radius r, in bin units) is grown outward until

        r >= a / (n * sqrt(s))

    where a = alpha_base, n is the occupancy SAMPLE COUNT enclosed so far
    (occ_count_map -- a raw count of position samples, not seconds), and s is
    the spike count enclosed so far. Equivalently, squaring both sides:
    rsq * n^2 * s >= a^-2, which is what's evaluated below. Once the radius
    is chosen, the bin's rate is spikes / occupancy-TIME (occ_map, in
    seconds) enclosed within it.

    A bin with zero spikes in its neighbourhood never gets substituted with
    "1 spike" to dodge a zero threshold -- the criterion simply never
    triggers while s=0, so the neighbourhood keeps growing until a spike is
    found or maxdist is hit.

    Two validity gates, both taken directly from rate_mapper.m:
      - mindist checkpoint: if the occupancy TIME enclosed within a small
        fixed `mindist_cm` radius doesn't reach min_occ_s, the bin is invalid
        outright, regardless of how far expansion would otherwise go.
      - maxdist cap: the search stops at `maxdist_cm`; if the criterion was
        never satisfied by then, the bin uses whatever radius accumulated up
        to that cap.

    Every bin within reach of some visited bin is evaluated (an unvisited
    bin can still borrow a valid estimate from nearby visited bins). Bins
    with no visited bin within maxdist are skipped outright via a distance
    transform on the occupancy mask.
    """
    n_i, n_j = occ_map.shape
    mindist_rsq = (mindist_cm / target_bin_cm) ** 2
    maxdist_rsq = (maxdist_cm / target_bin_cm) ** 2
    max_rsq_int = max(int(np.ceil(maxdist_rsq)), 1)

    fr_map     = np.zeros_like(occ_map)
    valid_mask = np.zeros(occ_map.shape, dtype=bool)

    if spike_map.sum() <= 0 or occ_count_map.sum() <= 0:
        return fr_map, valid_mask

    occ_flat       = occ_map.ravel()          # occupancy TIME (s) per bin
    occ_count_flat = occ_count_map.ravel()    # occupancy SAMPLE COUNT per bin
    spike_flat     = spike_map.ravel()

    I, J = np.meshgrid(np.arange(n_i), np.arange(n_j), indexing='ij')
    I = I.ravel(); J = J.ravel()

    # Candidate squared radii, ascending, starting at 1 bin^2 -- Grieves'
    # smallest tested radius is a full bin width (r=1), not r=0.
    rsq_values   = np.arange(1, max_rsq_int + 1, dtype=np.float64)
    thresh_sq    = (1.0 / alpha_base) ** 2

    visited      = occ_count_map > 0
    dist_visited = distance_transform_edt(~visited)   # 0 at visited bins
    reachable    = dist_visited <= np.sqrt(maxdist_rsq)

    for i0, j0 in np.argwhere(reachable):
        d2 = (I - i0) ** 2 + (J - j0) ** 2
        order         = np.argsort(d2, kind='stable')
        d2_sorted     = d2[order]
        occ_cum       = np.cumsum(occ_flat[order])
        occ_count_cum = np.cumsum(occ_count_flat[order])
        spike_cum     = np.cumsum(spike_flat[order])

        # mindist checkpoint gate: fixed radius, independent of the
        # adaptively-chosen radius found below
        chk = int(np.searchsorted(d2_sorted, mindist_rsq, side='right'))
        chk = min(max(chk, 1), len(d2_sorted))
        if occ_cum[chk - 1] < min_occ_s:
            continue   # invalid regardless of how far expansion would go

        idx = np.searchsorted(d2_sorted, rsq_values, side='right')
        idx = np.clip(idx, 1, len(d2_sorted))
        occ_time_at_rsq  = occ_cum[idx - 1]
        occ_count_at_rsq = occ_count_cum[idx - 1]
        spike_at_rsq     = spike_cum[idx - 1]

        satisfied = rsq_values * (occ_count_at_rsq ** 2) * spike_at_rsq >= thresh_sq
        k = int(np.argmax(satisfied)) if satisfied.any() else max_rsq_int - 1

        n_occ_final    = occ_time_at_rsq[k]
        n_spikes_final = spike_at_rsq[k]

        if n_occ_final >= min_occ_s:
            fr_map[i0, j0]     = n_spikes_final / n_occ_final
            valid_mask[i0, j0] = True

    return fr_map, valid_mask


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


# ── Tracking + spike loading / rate-map construction ────────────────────────────

def build_ratemap(csv_path: str, ntt_path: str,
                   arena_width_cm: float, target_bin_cm: float) -> tuple:
    """Loads tracking + spikes and returns (metrics, ctx) where ctx carries
    everything needed for field detection: fr_adaptive, occ_map,
    valid_mask, spike_frame indices, t, beh_bx, beh_by, n_bins_x/y."""

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

    occ_map       = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    occ_count_map = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    spike_map     = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)

    np.add.at(occ_map,       (beh_bx, beh_by), dt_frames)
    np.add.at(occ_count_map, (beh_bx, beh_by), 1.0)
    np.add.at(spike_map,     (sp_bx,  sp_by),  1.0)

    # Adaptive-binned firing rate map (replaces the fixed-bin + Gaussian-
    # smoothing pipeline): each bin's neighbourhood radius grows until it
    # encloses enough occupancy and spikes to satisfy the Skaggs & McNaughton
    # (1998) adaptive-binning criterion -- see _adaptive_binned_ratemap.
    fr_adaptive, valid_mask = _adaptive_binned_ratemap(
        occ_map, spike_map, occ_count_map, min_occ_s, fps, target_bin_cm)

    ctx = dict(spike_ts=spike_ts[valid_spike], spike_frame=spike_frame, t=t,
               beh_bx=beh_bx, beh_by=beh_by,
               occ_map=occ_map, valid_mask=valid_mask,
               fr_adaptive=fr_adaptive,
               n_bins_x=n_bins_x, n_bins_y=n_bins_y)

    if not valid_mask.any():
        return ({'n_spikes': n_spikes, 'n_discarded': n_discarded,
                  'peak_fr': 0.0, 'mean_fr': 0.0, 'sir': 0.0, 'sparsity': 0.0}, ctx)

    base = _metrics_from_ratemap(fr_adaptive, occ_map, valid_mask)
    base['n_spikes']    = n_spikes
    base['n_discarded'] = n_discarded
    return base, ctx


# ── Field isolation – "threshold method" (MATLAB placefield reference) ─────────

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


def detect_place_fields_threshold(base_metrics: dict, ctx: dict, target_bin_cm: float) -> list[dict]:
    """"Threshold method": a single-pass connected-component detector adapted
    from the MATLAB `placefield`/`getLegals` reference. A bin only qualifies
    for a field if its adaptively-binned rate is >= METHOD2_RATE_THRESHOLD_FRAC of the
    cell's peak rate AND above the cell's mean firing rate; 8-connected
    components of qualifying bins spanning >= MIN_FIELD_SIZE_BINS contiguous
    bins (no discontinuity) are reported as fields (peak bin + rate-weighted
    centre of mass, mirroring the reference's `fieldPos`/`centreFieldSize`)."""
    valid_mask = ctx['valid_mask']
    fr_adaptive  = ctx['fr_adaptive']
    n_bins_x   = ctx['n_bins_x']
    n_bins_y   = ctx['n_bins_y']

    total_valid_bins = int(valid_mask.sum())
    if total_valid_bins == 0:
        return []

    peak_fr = float(fr_adaptive[valid_mask].max())
    mean_fr = float(base_metrics.get('mean_fr', 0.0))
    rate_threshold = METHOD2_RATE_THRESHOLD_FRAC * peak_fr
    min_size_bins  = MIN_FIELD_SIZE_BINS

    qualifies = valid_mask & (fr_adaptive >= rate_threshold) & (fr_adaptive > mean_fr)

    centre_bin = (n_bins_x / 2.0, n_bins_y / 2.0)
    best_centre_dist  = np.inf
    centre_field_idx  = None
    fields = []

    for region in _connected_components_8(qualifies, n_bins_x, n_bins_y):
        if len(region) < min_size_bins:
            continue

        bxs   = np.array([b[0] for b in region])
        bys   = np.array([b[1] for b in region])
        rates = fr_adaptive[bxs, bys]

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


# ── Rate-map + field-boundary plotting ──────────────────────────────────────────

def _field_mask_from_bin_coords(bin_coords: str, n_bins_x: int, n_bins_y: int) -> np.ndarray:
    mask = np.zeros((n_bins_x, n_bins_y), dtype=bool)
    if not bin_coords:
        return mask
    for pair in bin_coords.split(';'):
        bx_str, by_str = pair.split('-')
        mask[int(bx_str), int(by_str)] = True
    return mask


def _plot_ratemap_with_field_boundaries(ax, fr_adaptive: np.ndarray, valid_mask: np.ndarray,
                                         fields: list[dict], n_bins_x: int, n_bins_y: int, title: str):
    display_map = np.ma.masked_where(~valid_mask, fr_adaptive)
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


def _save_field_ratemap_plot(ctx: dict, fields_threshold: list[dict], ntt_path: str):
    """Saves one PNG per .ntt file: rate map with detected-field boundaries
    (black outlines) from the threshold method."""
    fr_adaptive  = ctx['fr_adaptive']
    valid_mask = ctx['valid_mask']
    n_bins_x   = ctx['n_bins_x']
    n_bins_y   = ctx['n_bins_y']

    fig = Figure(figsize=(6, 6))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    im = _plot_ratemap_with_field_boundaries(ax, fr_adaptive, valid_mask, fields_threshold,
                                              n_bins_x, n_bins_y,
                                              f'Threshold method ({len(fields_threshold)} field(s))')
    fig.colorbar(im, ax=ax, label='Hz')
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
        fields_threshold = detect_place_fields_threshold(base_metrics, ctx, target_bin_cm)
    except Exception as e:
        with _print_lock:
            print(f'  ERROR detecting fields (threshold method) in {ntt_file}: {e}')
        return session_name, ntt_file, [], {'error': str(e)}

    try:
        _save_field_ratemap_plot(ctx, fields_threshold, ntt_path)
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
    return session_name, ntt_file, fields_threshold, summary


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

    field_columns_threshold = [
        'field_number', 'peak_bin_x', 'peak_bin_y', 'peak_fr_hz',
        'com_bin_x', 'com_bin_y', 'com_cm_x', 'com_cm_y',
        'n_bins', 'area_cm2', 'pct_of_occupied_area',
        'bbox_x_min', 'bbox_x_max', 'bbox_y_min', 'bbox_y_max',
        'bbox_cm_x_min', 'bbox_cm_x_max', 'bbox_cm_y_min', 'bbox_cm_y_max',
        'bin_coords', 'total_occupied_bins', 'min_field_size_bins',
        'rate_threshold_hz', 'mean_fr_threshold_hz', 'is_centre_field',
    ]

    # ── Pre-pass: assign sheet names & build the summary rows first, so the
    # 'Summary' sheet can be written before the per-unit sheets (Excel keeps
    # sheets in write order, so writing it first makes it the first tab). ────

    used_sheet_names = set()
    summary_rows = []
    sorted_results = sorted(results, key=lambda r: (r[0], r[1]))

    for session_name, ntt_file, fields_threshold, summary in sorted_results:
        sheet_name = _safe_sheet_name(os.path.splitext(ntt_file)[0], used_sheet_names)

        areas_thr = [f['area_cm2'] for f in fields_threshold]
        pcts_thr  = [f['pct_of_occupied_area'] for f in fields_threshold]

        summary_rows.append({
            'session': session_name,
            'unit': ntt_file,
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
        'session', 'unit', 'sheet_name', 'n_fields_detected',
        'field_areas_cm2', 'field_pct_areas',
        'total_field_area_cm2', 'total_pct_area_occupied',
        'n_spikes', 'base_sir', 'base_sparsity', 'base_peak_fr', 'base_mean_fr', 'error',
    ]
    df_summary = pd.DataFrame(summary_rows, columns=summary_columns)

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

        for (session_name, ntt_file, fields_threshold, summary), row in zip(sorted_results, summary_rows):
            sheet_name = row['sheet_name']
            if fields_threshold:
                df_thr = pd.DataFrame(fields_threshold, columns=field_columns_threshold)
            else:
                df_thr = pd.DataFrame(columns=field_columns_threshold)
            df_thr.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f'\nDone. Results saved to {output_excel}')
    print(f'Total units processed : {len(results)}')
    print(f'Total fields detected (threshold method) : {sum(r["n_fields_detected"] for r in summary_rows)}')
