# -*- coding: utf-8 -*-
"""
Mean Rate Map & Quadrant Analysis (S1H + Figure 1B/D, after Muessig et al.)

Reproduces, for place cells pooled across multiple recording days:
  - Fig S1H : overall mean (unsmoothed) firing-rate map per arena, fine spatial bins.
  - Fig 1B  : quadrant-wise proportion of place-field peak locations.
  - Fig 1D  : quadrant-wise mean firing rate.

Arenas (edit ARENA_CONFIGS['root'] below):
  1. circular_track : 1D circular track, outer dia 80 cm, inner dia 72 cm -> arc quadrants
  2. linear_track    : 80 x 8 cm linear track (vertical sessions auto-rotated 90 deg CCW) -> rectangle quadrants
  3. open_field      : circular open field, dia 60 cm -> pie quadrants

Tracking load/clean/smooth (pixel<->cm handling, jump removal, Gaussian smoothing), spike-position
matching (50 ms gate) and place-cell qualification (n_spikes>50, 1<peak_fr<15 Hz, SIR>0.5,
sparsity<0.9, location-shuffle bootstrap significant) are ported from
PlaceCellCharacterization_SpeedModv3_DownsampledPos15.py.

Folder layout expected under each arena's root (same convention as the reference scripts):
  root/.../<session>/  containing exactly one tracking file (.csv or .xlsx) and one or more .ntt files.

Every arena's spatial bins are represented as a single flat index (0..n_bins-1); this lets rate-map
construction, SIR/sparsity, and the bootstrap significance test share one implementation across the
2D open field, the 1D (wrap-around) ring, and the 1D linear track -- only the coordinate transform,
smoothing kernel and plotting differ per arena.
"""

import os
import random
import concurrent.futures

import numpy as np
import pandas as pd
from scipy.ndimage import convolve, gaussian_filter1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.patches import Wedge, Rectangle

# ============================================================================
# CONFIGURATION -- edit root folders + geometry below
# ============================================================================

ARENA_CONFIGS = {
    'open_field': dict(
        root=r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True/Fa23BD',
        shape='circle',
        diameter_cm=60.0,
    ),
    'circular_track': dict(
        root=r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True/Fa8477',
        shape='ring',
        outer_diameter_cm=80.0,
        inner_diameter_cm=72.0,
    ),
    'linear_track': dict(
        root=r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True/Fa1059',
        shape='linear',
        length_cm=80.0,
        width_cm=8.0,
    ),
}

OUTPUT_DIR = r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True/MeanRM_Quad'

fps            = 30           # tracking frame rate (Hz)
target_bin_cm  = 2.0          # spatial bin size (cm / along-track cm), fine-map resolution
min_occ_s      = 1.0          # exclude bins with < 1 s occupancy
MAX_GAP_US     = 50_000       # max spike-position gap (us)
N_BOOTSTRAP    = 1000         # circular-shift shuffles for SIR significance (reduce for faster runs)
MAX_WORKERS    = 4

POS_JUMP_THRESH_CMS  = 80.0   # frame-to-frame jumps implying a speed above this (cm/s) are tracking artifacts
POS_SMOOTH_SIGMA_SMP = 1.0    # Gaussian smoothing sigma (samples) applied to x/y tracking position

# 'pixel' or 'cm' -- set interactively at startup (see __main__).
COORD_UNITS = 'pixel'
_PIXEL_ANSWERS = {'pixel', 'pixels', 'px'}
_CM_ANSWERS    = {'cm', 'cms', 'centimeter', 'centimeters', 'centimetre', 'centimetres'}

ntt_dtype = np.dtype([
    ('timestamp',   '<u8'),
    ('sc_number',   '<u4'),
    ('cell_number', '<u4'),
    ('params',      '<u4', (8,)),
    ('waveforms',   '<i2', (32, 4)),
])


# ============================================================================
# Tracking load / clean / smooth -- ported from
# PlaceCellCharacterization_SpeedModv3_DownsampledPos15.py
# ============================================================================

def _load_tracking(csv_path: str, arena_width_cm: float) -> tuple:
    """Load, clean and pixel->cm convert one tracking file.

    Returns (x_cm, y_cm, t) with t in the same (us) time base as spike timestamps.
    """
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
        return x.astype(np.float64), y.astype(np.float64), t.astype(np.float64)

    if COORD_UNITS == 'cm':
        x_cm = x - x.min()
        y_cm = y - y.min()
    else:
        x_span = x.max() - x.min()
        y_span = y.max() - y.min()
        px_per_cm = max(x_span, y_span) / arena_width_cm
        x_cm = (x - x.min()) / px_per_cm
        y_cm = (y - y.min()) / px_per_cm

    return x_cm, y_cm, t


def _smooth_tracking_position(x_cm: np.ndarray, y_cm: np.ndarray, t_us: np.ndarray,
                              jump_thresh_cms: float = POS_JUMP_THRESH_CMS,
                              sigma_samples: float = POS_SMOOTH_SIGMA_SMP) -> tuple:
    """Iterative jump removal (interpolated) + Gaussian smoothing of x/y tracking."""
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


# ============================================================================
# Triangular smoothing kernels (2D for the open field, 1D for ring/linear)
# ============================================================================

_TRIANGULAR_KERNEL_2D = np.array([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]], dtype=np.float64) / 16.0
_TRIANGULAR_KERNEL_1D = np.array([1.0, 2.0, 1.0]) / 4.0


def _triangular_smooth_2d(fr_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    fr_in   = np.where(valid_mask, fr_map, 0.0)
    mask_in = valid_mask.astype(np.float64)
    smoothed_fr = convolve(fr_in,   _TRIANGULAR_KERNEL_2D, mode='constant', cval=0.0)
    smoothed_w  = convolve(mask_in, _TRIANGULAR_KERNEL_2D, mode='constant', cval=0.0)
    smoothed = np.zeros_like(smoothed_fr)
    vw = smoothed_w > 0
    smoothed[vw] = smoothed_fr[vw] / smoothed_w[vw]
    smoothed[~valid_mask] = 0.0
    return smoothed


def _triangular_smooth_1d(fr: np.ndarray, valid: np.ndarray, wrap: bool = False) -> np.ndarray:
    mode = 'wrap' if wrap else 'constant'
    fr_in   = np.where(valid, fr, 0.0)
    mask_in = valid.astype(np.float64)
    smoothed_fr = convolve(fr_in,   _TRIANGULAR_KERNEL_1D, mode=mode, cval=0.0)
    smoothed_w  = convolve(mask_in, _TRIANGULAR_KERNEL_1D, mode=mode, cval=0.0)
    smoothed = np.zeros_like(smoothed_fr)
    vw = smoothed_w > 0
    smoothed[vw] = smoothed_fr[vw] / smoothed_w[vw]
    smoothed[~valid] = 0.0
    return smoothed


# ============================================================================
# Arena geometry handlers
#
# Each handler converts (x_cm, y_cm) into a single flat spatial-bin index per
# position sample (0..n_bins-1), so the rate-map / SIR / bootstrap machinery
# below is written once and shared by the 2D open field, the 1D wrap-around
# ring, and the 1D linear track.
# ============================================================================

class OpenFieldHandler:
    def __init__(self, cfg: dict, bin_cm: float):
        self.diameter = cfg['diameter_cm']
        self.bin_cm   = bin_cm
        self.nx = int(np.ceil(self.diameter / bin_cm))
        self.ny = self.nx
        self.n_bins = self.nx * self.ny
        self.cx = self.diameter / 2.0
        self.cy = self.diameter / 2.0
        self.arena_width_cm = self.diameter

        xs = (np.arange(self.nx) + 0.5) * bin_cm
        ys = (np.arange(self.ny) + 0.5) * bin_cm
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        r = np.hypot(XX - self.cx, YY - self.cy)
        self.geom_valid = (r <= self.diameter / 2.0).ravel()

    def orient(self, x_cm, y_cm):
        return x_cm, y_cm

    def to_bins(self, x_cm, y_cm):
        bx = np.clip((x_cm / self.bin_cm).astype(int), 0, self.nx - 1)
        by = np.clip((y_cm / self.bin_cm).astype(int), 0, self.ny - 1)
        flat = bx * self.ny + by
        r = np.hypot(x_cm - self.cx, y_cm - self.cy)
        sample_valid = r <= (self.diameter / 2.0 + self.bin_cm)
        return flat, sample_valid

    def smooth(self, fr_flat, valid_flat):
        fr2 = fr_flat.reshape(self.nx, self.ny)
        v2  = valid_flat.reshape(self.nx, self.ny)
        return _triangular_smooth_2d(fr2, v2).ravel()

    def quadrant_of_bin(self):
        xs = (np.arange(self.nx) + 0.5) * self.bin_cm
        ys = (np.arange(self.ny) + 0.5) * self.bin_cm
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        ang = np.degrees(np.arctan2(YY - self.cy, XX - self.cx)) % 360.0
        return ((ang // 90).astype(int) % 4).ravel()

    def plot_fine(self, ax, values_flat, valid_flat, cmap, norm):
        grid = np.full(self.n_bins, np.nan)
        m = valid_flat & self.geom_valid
        grid[m] = values_flat[m]
        grid2 = grid.reshape(self.nx, self.ny)
        im = ax.imshow(np.ma.masked_invalid(grid2.T), origin='lower',
                        extent=[0, self.diameter, 0, self.diameter],
                        cmap=cmap, norm=norm)
        ax.set_aspect('equal')
        ax.axis('off')
        return im

    def plot_quadrants(self, ax, quadrant_values, cmap, norm):
        for q in range(4):
            val = quadrant_values[q]
            color = cmap(norm(val)) if np.isfinite(val) else 'lightgray'
            w = Wedge((self.cx, self.cy), self.diameter / 2.0, q * 90, (q + 1) * 90,
                      facecolor=color, edgecolor='k', linewidth=1)
            ax.add_patch(w)
        ax.set_xlim(0, self.diameter)
        ax.set_ylim(0, self.diameter)
        ax.set_aspect('equal')
        ax.axis('off')


class CircularTrackHandler:
    def __init__(self, cfg: dict, bin_cm: float):
        self.outer_d = cfg['outer_diameter_cm']
        self.inner_d = cfg['inner_diameter_cm']
        self.outer_r = self.outer_d / 2.0
        self.inner_r = self.inner_d / 2.0
        self.mean_r  = (self.outer_r + self.inner_r) / 2.0
        self.cx = self.outer_r
        self.cy = self.outer_r
        self.arena_width_cm = self.outer_d
        self.radial_tol_cm = 4.0

        circumference = 2 * np.pi * self.mean_r
        self.n_bins = max(8, int(round(circumference / bin_cm)))
        self.bin_width_deg = 360.0 / self.n_bins

    def orient(self, x_cm, y_cm):
        return x_cm, y_cm

    def to_bins(self, x_cm, y_cm):
        r = np.hypot(x_cm - self.cx, y_cm - self.cy)
        theta = np.degrees(np.arctan2(y_cm - self.cy, x_cm - self.cx)) % 360.0
        bin_idx = np.clip((theta / self.bin_width_deg).astype(int), 0, self.n_bins - 1)
        on_track = (r >= self.inner_r - self.radial_tol_cm) & (r <= self.outer_r + self.radial_tol_cm)
        return bin_idx, on_track

    def smooth(self, fr_flat, valid_flat):
        return _triangular_smooth_1d(fr_flat, valid_flat, wrap=True)

    def quadrant_of_bin(self):
        centers_deg = (np.arange(self.n_bins) + 0.5) * self.bin_width_deg
        return (centers_deg // 90).astype(int) % 4

    def plot_fine(self, ax, values_flat, valid_flat, cmap, norm):
        theta_edges = np.linspace(0, 2 * np.pi, self.n_bins + 1)
        r_edges = np.array([self.inner_r, self.outer_r])
        vals = np.where(valid_flat, values_flat, np.nan)[None, :]
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        pcm = ax.pcolormesh(theta_edges, r_edges, vals, cmap=cmap, norm=norm, shading='auto')
        ax.set_ylim(0, self.outer_r + 5)
        ax.set_yticklabels([])
        ax.grid(False)
        return pcm

    def plot_quadrants(self, ax, quadrant_values, cmap, norm):
        for q in range(4):
            val = quadrant_values[q]
            color = cmap(norm(val)) if np.isfinite(val) else 'lightgray'
            w = Wedge((self.cx, self.cy), self.outer_r, q * 90, (q + 1) * 90,
                      width=self.outer_r - self.inner_r,
                      facecolor=color, edgecolor='k', linewidth=1)
            ax.add_patch(w)
        ax.set_xlim(0, self.outer_d)
        ax.set_ylim(0, self.outer_d)
        ax.set_aspect('equal')
        ax.axis('off')


class LinearTrackHandler:
    def __init__(self, cfg: dict, bin_cm: float):
        self.length = cfg['length_cm']
        self.width  = cfg['width_cm']
        self.n_bins = max(4, int(round(self.length / bin_cm)))
        self.bin_cm = self.length / self.n_bins
        self.arena_width_cm = self.length

    def orient(self, x_cm, y_cm):
        """Vertical-session tracks (long axis along y) are rotated 90 deg CCW so every
        linear-track recording pools onto the same length-cm axis regardless of the
        physical orientation of the track in the room."""
        if len(x_cm) == 0:
            return x_cm, y_cm
        x_span = x_cm.max() - x_cm.min()
        y_span = y_cm.max() - y_cm.min()
        if y_span > x_span:
            xr = -y_cm
            yr = x_cm
            xr = xr - xr.min()
            yr = yr - yr.min()
            return xr, yr
        return x_cm, y_cm

    def to_bins(self, x_cm, y_cm):
        pos = np.clip(x_cm, 0, self.length)
        bin_idx = np.clip((pos / self.bin_cm).astype(int), 0, self.n_bins - 1)
        sample_valid = np.ones_like(pos, dtype=bool)
        return bin_idx, sample_valid

    def smooth(self, fr_flat, valid_flat):
        return _triangular_smooth_1d(fr_flat, valid_flat, wrap=False)

    def quadrant_of_bin(self):
        centers = (np.arange(self.n_bins) + 0.5) * self.bin_cm
        return np.clip((centers // (self.length / 4.0)).astype(int), 0, 3)

    def plot_fine(self, ax, values_flat, valid_flat, cmap, norm):
        vals = np.where(valid_flat, values_flat, np.nan)[None, :]
        edges = np.linspace(0, self.length, self.n_bins + 1)
        pcm = ax.pcolormesh(edges, [0, self.width], vals, cmap=cmap, norm=norm, shading='auto')
        ax.set_aspect('equal')
        ax.axis('off')
        return pcm

    def plot_quadrants(self, ax, quadrant_values, cmap, norm):
        seg = self.length / 4.0
        for q in range(4):
            val = quadrant_values[q]
            color = cmap(norm(val)) if np.isfinite(val) else 'lightgray'
            rect = Rectangle((q * seg, 0), seg, self.width,
                             facecolor=color, edgecolor='k', linewidth=1)
            ax.add_patch(rect)
        ax.set_xlim(0, self.length)
        ax.set_ylim(0, self.width)
        ax.set_aspect('auto')
        ax.axis('off')


def make_handler(cfg: dict):
    shape = cfg['shape']
    if shape == 'circle':
        return OpenFieldHandler(cfg, target_bin_cm)
    if shape == 'ring':
        return CircularTrackHandler(cfg, target_bin_cm)
    if shape == 'linear':
        return LinearTrackHandler(cfg, target_bin_cm)
    raise ValueError(f'Unknown arena shape: {shape}')


# ============================================================================
# Core per-cell rate map + place-cell qualification
# (generalized flat-bin port of compute_metrics / _run_bootstrap from
#  PlaceCellCharacterization_SpeedModv3_DownsampledPos15.py)
# ============================================================================

def compute_cell_ratemap(x_cm: np.ndarray, y_cm: np.ndarray, t: np.ndarray,
                         spike_ts: np.ndarray, handler) -> dict | None:
    x_cm, y_cm = handler.orient(x_cm, y_cm)
    bin_idx, sample_valid = handler.to_bins(x_cm, y_cm)
    t = t[sample_valid]
    bin_idx = bin_idx[sample_valid]
    if len(t) < 2:
        return None

    idx   = np.searchsorted(t, spike_ts, side='left')
    idx_l = np.clip(idx - 1, 0, len(t) - 1)
    idx_r = np.clip(idx,     0, len(t) - 1)
    dist_l  = np.abs(spike_ts - t[idx_l])
    dist_r  = np.abs(spike_ts - t[idx_r])
    nearest = np.where(dist_l <= dist_r, idx_l, idx_r)
    min_dist = np.minimum(dist_l, dist_r)

    valid_spike = min_dist <= MAX_GAP_US
    spike_frame = nearest[valid_spike]
    n_spikes    = int(valid_spike.sum())

    n = len(t)
    dt_frames = np.empty(n, dtype=np.float64)
    dt_frames[0] = 1.0 / fps
    raw_dt = np.diff(t) * 1e-6
    dt_frames[1:] = np.minimum(raw_dt, 2.0 / fps)

    n_bins = handler.n_bins
    occ_map   = np.zeros(n_bins, dtype=np.float64)
    spike_map = np.zeros(n_bins, dtype=np.float64)
    np.add.at(occ_map,   bin_idx, dt_frames)
    np.add.at(spike_map, bin_idx[spike_frame], 1.0)

    occ_valid  = occ_map >= min_occ_s
    geom_valid = getattr(handler, 'geom_valid', np.ones(n_bins, dtype=bool))
    valid = occ_valid & geom_valid

    fr_raw = np.zeros(n_bins, dtype=np.float64)
    fr_raw[valid] = spike_map[valid] / occ_map[valid]
    fr_smooth = handler.smooth(fr_raw, valid)

    result = dict(n_spikes=n_spikes, fr_raw=fr_raw, fr_smooth=fr_smooth,
                  occ_map=occ_map, valid=valid, bin_idx=bin_idx,
                  spike_frame=spike_frame, t=t, n_bins=n_bins)

    if not valid.any():
        result.update(peak_fr=0.0, mean_fr=0.0, sir=0.0, sparsity=0.0)
        return result

    total_occ = occ_map[valid].sum()
    pi = occ_map[valid] / total_occ
    ri = fr_smooth[valid]
    r_mean = float(np.sum(pi * ri))
    peak_fr = float(fr_smooth[valid].max())

    sir = 0.0
    if r_mean > 0:
        nz = ri > 0
        ratio = ri[nz] / r_mean
        sir = float(np.sum(pi[nz] * ratio * np.log2(ratio)))

    spar_num = float(np.sum(pi * ri))
    spar_den = float(np.sum(pi * ri ** 2))
    sparsity = float((spar_num ** 2) / spar_den) if spar_den > 0 else 0.0

    result.update(peak_fr=round(peak_fr, 4), mean_fr=round(r_mean, 4),
                  sir=round(sir, 4), sparsity=round(sparsity, 4))
    return result


def run_bootstrap_generic(handler, cell: dict, real_sir: float, n_bootstrap: int = N_BOOTSTRAP) -> dict:
    """Location-shuffling bootstrap (Fenton circular-shift method), generalized to any
    flat-bin arena: the position->bin mapping is fixed, only which frame each spike is
    credited to is circularly shifted."""
    bin_idx     = cell['bin_idx']
    spike_frame = cell['spike_frame']
    t           = cell['t']
    occ_map     = cell['occ_map']
    valid       = cell['valid']
    n_bins      = cell['n_bins']

    if len(spike_frame) == 0:
        return dict(bootstrap_mean=float('nan'), bootstrap_p95=float('nan'), bootstrap_sig=False)

    n_frames = len(t)
    MARGIN_FRAMES = int(20 * fps)
    if n_frames <= 2 * MARGIN_FRAMES:
        return dict(bootstrap_mean=float('nan'), bootstrap_p95=float('nan'), bootstrap_sig=None)

    sir_i = np.zeros(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        rnd = random.randint(MARGIN_FRAMES, n_frames - MARGIN_FRAMES)
        shuf_frame = (spike_frame + rnd) % n_frames

        spike_map = np.zeros(n_bins, dtype=np.float64)
        np.add.at(spike_map, bin_idx[shuf_frame], 1.0)

        fr_raw = np.zeros(n_bins, dtype=np.float64)
        fr_raw[valid] = spike_map[valid] / occ_map[valid]
        fr_smooth = handler.smooth(fr_raw, valid)

        total_occ = occ_map[valid].sum()
        pi = occ_map[valid] / total_occ
        ri = fr_smooth[valid]
        r_mean = float(np.sum(pi * ri))
        if r_mean <= 0:
            sir_i[i] = 0.0
            continue
        nz = ri > 0
        ratio = ri[nz] / r_mean
        sir_i[i] = float(np.sum(pi[nz] * ratio * np.log2(ratio)))

    bootstrap_mean = float(np.mean(sir_i))
    bootstrap_p95  = float(np.percentile(sir_i, 95))
    bootstrap_sig  = bool(real_sir > bootstrap_p95)
    return dict(bootstrap_mean=round(bootstrap_mean, 4),
                bootstrap_p95=round(bootstrap_p95, 4),
                bootstrap_sig=bootstrap_sig)


def process_unit(csv_path: str, ntt_path: str, ntt_file: str, session_name: str, handler) -> dict | None:
    x_cm, y_cm, t = _load_tracking(csv_path, handler.arena_width_cm)
    if len(t) < 2:
        return None
    x_cm, y_cm = _smooth_tracking_position(x_cm, y_cm, t)

    spike_data = np.memmap(ntt_path, dtype=ntt_dtype, mode='r', offset=16 * 1024)
    spike_data = spike_data[spike_data['cell_number'] != 0]
    spike_ts = np.sort(spike_data['timestamp'].astype(np.float64))
    if len(spike_ts) == 0:
        return None

    cell = compute_cell_ratemap(x_cm, y_cm, t, spike_ts, handler)
    if cell is None:
        return None

    boot = run_bootstrap_generic(handler, cell, cell['sir'])
    boot_sig = boot.get('bootstrap_sig')
    place_cell = bool(
        boot_sig is True and
        cell['n_spikes'] > 50 and
        1.0 < cell['peak_fr'] < 15.0 and
        cell['sir'] > 0.5 and
        cell['sparsity'] < 0.9
    )

    peak_bin = None
    if cell['valid'].any():
        masked = np.where(cell['valid'], cell['fr_smooth'], -np.inf)
        peak_bin = int(np.argmax(masked))

    return dict(
        session=session_name, unit=ntt_file,
        n_spikes=cell['n_spikes'], peak_fr=cell['peak_fr'], mean_fr=cell['mean_fr'],
        sir=cell['sir'], sparsity=cell['sparsity'],
        bootstrap_sig=boot_sig, place_cell=place_cell,
        fr_raw=cell['fr_raw'], valid=cell['valid'], occ_map=cell['occ_map'],
        peak_bin=peak_bin,
    )


# ============================================================================
# Batch scan per arena
# ============================================================================

def collect_arena_results(arena_key: str, cfg: dict) -> tuple:
    handler = make_handler(cfg)
    root = cfg['root']

    jobs = []
    for dirpath, _, filenames in os.walk(root):
        tracking_files_all = [f for f in filenames if f.lower().endswith(('.csv', '.xlsx'))]
        if COORD_UNITS == 'cm':
            tracking_files = [f for f in tracking_files_all if f.lower().endswith('_cm.csv')]
        else:
            tracking_files = [f for f in tracking_files_all if not f.lower().endswith('_cm.csv')]
        ntt_files = [f for f in filenames if f.lower().endswith('.ntt')]
        if len(tracking_files) == 1 and len(ntt_files) > 0:
            csv_path = os.path.join(dirpath, tracking_files[0])
            for ntt_file in sorted(ntt_files):
                jobs.append((dirpath, csv_path, ntt_file))

    def _job(args):
        dirpath, csv_path, ntt_file = args
        session_name = os.path.relpath(dirpath, root)
        ntt_path = os.path.join(dirpath, ntt_file)
        try:
            return process_unit(csv_path, ntt_path, ntt_file, session_name, handler)
        except Exception as e:
            print(f'  ERROR [{arena_key}] {session_name}/{ntt_file}: {e}')
            return None

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, r in enumerate(executor.map(_job, jobs), start=1):
            print(f'[{arena_key} {i}/{len(jobs)}]', end='\r')
            if r is not None:
                results.append(r)

    n_place = sum(1 for r in results if r['place_cell'])
    print(f'\n[{arena_key}] {len(jobs)} units scanned, {len(results)} processed, {n_place} place cells.')
    return handler, results


# ============================================================================
# Pooling across cells / days
# ============================================================================

def pool_fine_map(handler, results: list) -> tuple:
    place = [r for r in results if r['place_cell']]
    if not place:
        return np.full(handler.n_bins, np.nan), np.zeros(handler.n_bins, dtype=bool)
    stack = np.full((len(place), handler.n_bins), np.nan)
    for i, r in enumerate(place):
        stack[i, r['valid']] = r['fr_raw'][r['valid']]
    mean_map = np.nanmean(stack, axis=0)
    any_valid = ~np.all(np.isnan(stack), axis=0)
    return mean_map, any_valid


def pool_quadrant_peak_proportion(handler, results: list) -> tuple:
    place = [r for r in results if r['place_cell'] and r['peak_bin'] is not None]
    q_of_bin = handler.quadrant_of_bin()
    counts = np.zeros(4)
    for r in place:
        counts[q_of_bin[r['peak_bin']]] += 1
    total = counts.sum()
    pct = counts / total * 100.0 if total > 0 else np.full(4, np.nan)
    return pct, int(total)


def pool_quadrant_mean_rate(handler, results: list) -> np.ndarray:
    place = [r for r in results if r['place_cell']]
    if not place:
        return np.full(4, np.nan)
    q_of_bin = handler.quadrant_of_bin()
    per_cell = np.full((len(place), 4), np.nan)
    for i, r in enumerate(place):
        valid, occ, fr = r['valid'], r['occ_map'], r['fr_raw']
        for q in range(4):
            sel = valid & (q_of_bin == q)
            occ_q = occ[sel].sum()
            if occ_q <= 0:
                continue
            spk_q = (fr[sel] * occ[sel]).sum()
            per_cell[i, q] = spk_q / occ_q
    return np.nanmean(per_cell, axis=0)


# ============================================================================
# Plotting
# ============================================================================

_ARENA_ORDER = ['open_field', 'circular_track', 'linear_track']
_ARENA_TITLES = {'open_field': 'Open Field', 'circular_track': 'Circular Track', 'linear_track': 'Linear Track'}


def _get_cmap(name: str):
    try:
        base = matplotlib.colormaps[name]
    except Exception:
        base = plt.get_cmap(name)
    return base.copy() if hasattr(base, 'copy') else base


def make_cmap_norm(values) -> tuple:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if vmax <= vmin:
            vmax = vmin + 1e-6
    cmap = _get_cmap('jet')
    cmap.set_bad('white')
    return cmap, Normalize(vmin=vmin, vmax=vmax)


def plot_fig_S1H(arena_handlers: dict, arena_results: dict, save_path: str):
    fig = plt.figure(figsize=(15, 5))
    for i, key in enumerate(_ARENA_ORDER):
        handler = arena_handlers[key]
        mean_map, valid = pool_fine_map(handler, arena_results[key])
        cmap, norm = make_cmap_norm(mean_map[valid])

        proj = 'polar' if key == 'circular_track' else None
        ax = fig.add_subplot(1, 3, i + 1, projection=proj)
        pcm = handler.plot_fine(ax, mean_map, valid, cmap, norm)

        n_place = sum(1 for r in arena_results[key] if r['place_cell'])
        ax.set_title(f'{_ARENA_TITLES[key]}\n(n={n_place} place cells)')
        fig.colorbar(pcm, ax=ax, shrink=0.7, label='Firing rate (Hz)')

    fig.suptitle('Fig S1H -- Overall mean firing rate maps (unsmoothed, place cells pooled across days)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_fig_1BD(arena_handlers: dict, arena_results: dict, save_path: str):
    peak_pct, rate = {}, {}
    for key in _ARENA_ORDER:
        handler = arena_handlers[key]
        peak_pct[key] = pool_quadrant_peak_proportion(handler, arena_results[key])
        rate[key]     = pool_quadrant_mean_rate(handler, arena_results[key])

    cmap_b, norm_b = make_cmap_norm(np.concatenate([v[0] for v in peak_pct.values()]))
    cmap_d, norm_d = make_cmap_norm(np.concatenate(list(rate.values())))

    fig = plt.figure(figsize=(15, 10))
    axes_b, axes_d = [], []
    for i, key in enumerate(_ARENA_ORDER):
        handler = arena_handlers[key]

        ax_b = fig.add_subplot(2, 3, i + 1)
        handler.plot_quadrants(ax_b, peak_pct[key][0], cmap_b, norm_b)
        ax_b.set_title(f'{_ARENA_TITLES[key]}\nPeak proportion (%), n={peak_pct[key][1]}')
        axes_b.append(ax_b)

        ax_d = fig.add_subplot(2, 3, i + 4)
        handler.plot_quadrants(ax_d, rate[key], cmap_d, norm_d)
        ax_d.set_title(f'{_ARENA_TITLES[key]}\nMean firing rate (Hz)')
        axes_d.append(ax_d)

    sm_b = cm.ScalarMappable(cmap=cmap_b, norm=norm_b); sm_b.set_array([])
    fig.colorbar(sm_b, ax=axes_b, shrink=0.6, label='% of peaks')
    sm_d = cm.ScalarMappable(cmap=cmap_d, norm=norm_d); sm_d.set_array([])
    fig.colorbar(sm_d, ax=axes_d, shrink=0.6, label='Mean firing rate (Hz)')

    fig.suptitle('Figure 1B/D -- Quadrant-wise peak location (top) and mean firing rate (bottom)')
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def export_excel(arena_handlers: dict, arena_results: dict, out_path: str):
    rows = []
    for key, results in arena_results.items():
        handler = arena_handlers[key]
        q_of_bin = handler.quadrant_of_bin()
        for r in results:
            q = int(q_of_bin[r['peak_bin']]) if r['peak_bin'] is not None else None
            rows.append(dict(
                arena=key, session=r['session'], unit=r['unit'],
                n_spikes=r['n_spikes'], peak_fr=r['peak_fr'], mean_fr=r['mean_fr'],
                sir=r['sir'], sparsity=r['sparsity'],
                bootstrap_sig=r['bootstrap_sig'], place_cell=r['place_cell'],
                peak_quadrant=q,
            ))
    df = pd.DataFrame(rows)
    df.to_excel(out_path, index=False)
    print(f'[SAVED] {out_path}')


# ============================================================================
# __main__
# ============================================================================

if __name__ == '__main__':
    _coord_answer = input("Are the tracking coordinates in pixels or cm? [pixel/cm]: ").strip().lower()
    while _coord_answer not in _PIXEL_ANSWERS | _CM_ANSWERS:
        _coord_answer = input("Please enter 'pixel' or 'cm': ").strip().lower()
    COORD_UNITS = 'pixel' if _coord_answer in _PIXEL_ANSWERS else 'cm'
    print(f"Using '{COORD_UNITS}' tracking coordinates.\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    arena_handlers, arena_results = {}, {}
    for key, cfg in ARENA_CONFIGS.items():
        print(f'\n=== Processing arena: {key} ===')
        handler, results = collect_arena_results(key, cfg)
        arena_handlers[key] = handler
        arena_results[key]  = results

    plot_fig_S1H(arena_handlers, arena_results, os.path.join(OUTPUT_DIR, 'FigS1H_MeanRateMaps.png'))
    plot_fig_1BD(arena_handlers, arena_results, os.path.join(OUTPUT_DIR, 'Fig1_BD_QuadrantAnalysis.png'))
    export_excel(arena_handlers, arena_results, os.path.join(OUTPUT_DIR, 'PlaceCell_MeanRateMap_Summary.xlsx'))

    print('\nDone.')
