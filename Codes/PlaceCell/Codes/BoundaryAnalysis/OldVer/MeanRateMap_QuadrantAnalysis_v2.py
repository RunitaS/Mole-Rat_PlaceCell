# -*- coding: utf-8 -*-
"""
Mean Rate Map & Quadrant Analysis (S1H + Figure 1B/D, after Muessig et al.)

Reproduces, for place cells pooled across multiple recording days:
  - Fig S1H : overall mean (unsmoothed) firing-rate map per arena, fine spatial bins
              (2 x 2 cm; genuinely 2D for every arena, including the linear track's
              length x width).
  - Fig 1B/D: quadrant mean maps (Fig 1A method) -- the whole-arena map is cut into
              4 regions, each registered into one reference region's coordinates,
              and the 4 registered copies are averaged bin-by-bin, giving one small
              *multi-bin* heatmap (not a single scalar per quadrant):
                * open_field      : mirror-reflection fold (x and y independently) -> small square heatmap
                * circular_track  : 90 deg angular roll fold -> small arc heatmap (no distinct walls)
                * linear_track    : mirror-reflection fold (length x width) -> small rectangle heatmap
              Fig 1B pools each cell's one peak location into the folded grid (proportion
              of peaks per bin); Fig 1D folds the single overall (Fig S1H) mean rate map.

Arenas (edit ARENA_CONFIGS['root'] below):
  1. circular_track : 1D circular track, outer dia 80 cm, inner dia 72 cm
  2. linear_track    : 80 x 8 cm linear track (vertical sessions auto-rotated 90 deg CCW)
  3. open_field      : circular open field, dia 60 cm

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
# Quadrant mean-map folding (Muessig et al. Figure 1A)
#
# "The full map is divided into quadrants rotated around the center of the
# environment (c) such that all walls a are mapped onto a' and all walls b
# are mapped onto b'." I.e. the whole-arena binned map is cut into 4 regions,
# each region is registered into one reference region's coordinate frame, and
# the 4 registered copies are averaged bin-by-bin -- producing a single
# small *multi-bin* map, not a scalar per region.
#
# Each handler builds a `quad_idx_flat` array (length n_bins) mapping every
# whole-arena bin to its bin index in the folded reference region (or -1 if
# the bin has no fold partner, e.g. an odd leftover center row/column), plus
# `n_quad_bins`. These two generic helpers then do the actual folding; only
# how `quad_idx_flat` is built (rotation, roll, or reflection) differs per
# arena's symmetry.
# ============================================================================

def _fold_peak_bin(quad_idx_flat: np.ndarray, bin_idx: int) -> int:
    return int(quad_idx_flat[bin_idx])


def _fold_mean_map(quad_idx_flat: np.ndarray, n_quad_bins: int,
                    values_flat: np.ndarray, valid_flat: np.ndarray) -> tuple:
    q = quad_idx_flat
    m = (q >= 0) & valid_flat
    sums = np.zeros(n_quad_bins, dtype=np.float64)
    weights = np.zeros(n_quad_bins, dtype=np.float64)
    if m.any():
        np.add.at(sums, q[m], values_flat[m])
        np.add.at(weights, q[m], 1.0)
    out = np.full(n_quad_bins, np.nan)
    wm = weights > 0
    out[wm] = sums[wm] / weights[wm]
    return out, wm


def _build_reflect_quadrant_fold(nx: int, ny: int) -> tuple:
    """Registers each of the 4 quadrants of an (nx, ny) bin grid onto one reference
    quadrant [0:nx//2, 0:ny//2] by mirror-reflecting each axis about its own midpoint
    independently, so a quadrant's bx-boundary always lands on the reference's
    bx-boundary and its by-boundary always lands on the reference's by-boundary --
    i.e. wall a always maps onto wall a', wall b always onto wall b', for all 4
    quadrants (Muessig et al. Fig 1A).

    A plain 90 deg rotation of the whole grid does NOT do this: rotating swaps the
    two axes, so for 2 of the 4 quadrants (those related to the reference by 90 or
    270 deg, as opposed to the diagonal 180 deg one) a quadrant's bx-wall ends up
    registered against the reference's by-wall instead of its bx-wall -- the two
    wall sides get cross-matched rather than matched straight.
    """
    half_x, half_y = nx // 2, ny // 2
    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    local_i, local_j = np.meshgrid(np.arange(half_x), np.arange(half_y), indexing='ij')
    local_flat = local_i * half_y + local_j

    transforms = [
        (I,                J),
        (nx - 1 - I,       J),
        (I,                ny - 1 - J),
        (nx - 1 - I,       ny - 1 - J),
    ]
    quad_idx = np.full((nx, ny), -1, dtype=int)
    for Ti, Tj in transforms:
        sub_i = Ti[:half_x, :half_y]
        sub_j = Tj[:half_x, :half_y]
        quad_idx[sub_i, sub_j] = local_flat

    n_quad_bins = half_x * half_y
    return quad_idx.ravel(), n_quad_bins, (half_x, half_y)


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

        self._build_quadrant_fold()

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

    def _build_quadrant_fold(self):
        """Folds the 4 quadrants of the (nx, ny) bin grid onto one reference quadrant
        via independent mirror reflection of each axis (see _build_reflect_quadrant_fold)
        so that wall a always maps onto wall a' and wall b always onto wall b', for all
        4 quadrants -- per Muessig et al. Fig 1A."""
        self._quad_idx_flat, self.n_quad_bins, self.quad_shape = \
            _build_reflect_quadrant_fold(self.nx, self.ny)

    def fold_peak_bin(self, bin_idx):
        return _fold_peak_bin(self._quad_idx_flat, bin_idx)

    def fold_mean_map(self, values_flat, valid_flat):
        return _fold_mean_map(self._quad_idx_flat, self.n_quad_bins, values_flat, valid_flat)

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

    def plot_quadrant_map(self, ax, values_flat, valid_flat, cmap, norm):
        half = self.quad_shape[0]
        grid = np.full(self.n_quad_bins, np.nan)
        grid[valid_flat] = values_flat[valid_flat]
        grid2 = grid.reshape(half, half)
        side = half * self.bin_cm
        im = ax.imshow(np.ma.masked_invalid(grid2.T), origin='lower',
                        extent=[0, side, 0, side], cmap=cmap, norm=norm)
        ax.set_aspect('equal')
        ax.axis('off')
        return im


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
        # kept as a multiple of 4 so the quadrant fold below splits into 4 exactly
        # equal-length arcs (see _build_quadrant_fold)
        self.n_bins = max(8, 4 * int(round(circumference / bin_cm / 4.0)))
        self.bin_width_deg = 360.0 / self.n_bins

        self._build_quadrant_fold()

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

    def _build_quadrant_fold(self):
        """Per the user's choice for arenas without distinct walls: split the ring
        into 4 equal 90 deg arcs (arbitrary angular quartering) and fold them onto
        one reference arc, keeping the along-track bins within that arc intact."""
        qn = self.n_bins // 4
        quad_idx = np.empty(self.n_bins, dtype=int)
        for k in range(4):
            seg = np.arange(k * qn, (k + 1) * qn)
            quad_idx[seg] = np.arange(qn)
        self._quad_idx_flat = quad_idx
        self.n_quad_bins = qn

    def fold_peak_bin(self, bin_idx):
        return _fold_peak_bin(self._quad_idx_flat, bin_idx)

    def fold_mean_map(self, values_flat, valid_flat):
        return _fold_mean_map(self._quad_idx_flat, self.n_quad_bins, values_flat, valid_flat)

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

    def plot_quadrant_map(self, ax, values_flat, valid_flat, cmap, norm):
        theta_edges = np.linspace(0, np.pi / 2, self.n_quad_bins + 1)
        r_edges = np.array([self.inner_r, self.outer_r])
        vals = np.where(valid_flat, values_flat, np.nan)[None, :]
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        pcm = ax.pcolormesh(theta_edges, r_edges, vals, cmap=cmap, norm=norm, shading='auto')
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        ax.set_ylim(0, self.outer_r + 5)
        ax.set_yticklabels([])
        ax.grid(False)
        return pcm


class LinearTrackHandler:
    def __init__(self, cfg: dict, bin_cm: float):
        self.length = cfg['length_cm']
        self.width  = cfg['width_cm']
        # genuine 2D binning: bins along the track length AND across its width, so
        # the fine map has multiple rows in both dimensions (not just columns along
        # length with a single uniform row across width).
        self.nx = max(4, int(round(self.length / bin_cm)))   # along length
        self.ny = max(2, int(round(self.width  / bin_cm)))   # across width
        self.bin_cm_x = self.length / self.nx
        self.bin_cm_y = self.width  / self.ny
        self.n_bins = self.nx * self.ny
        self.arena_width_cm = self.length

        self._build_quadrant_fold()

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
        px = np.clip(x_cm, 0, self.length)
        py = np.clip(y_cm, 0, self.width)
        bx = np.clip((px / self.bin_cm_x).astype(int), 0, self.nx - 1)
        by = np.clip((py / self.bin_cm_y).astype(int), 0, self.ny - 1)
        flat = bx * self.ny + by
        sample_valid = np.ones_like(px, dtype=bool)
        return flat, sample_valid

    def smooth(self, fr_flat, valid_flat):
        fr2 = fr_flat.reshape(self.nx, self.ny)
        v2  = valid_flat.reshape(self.nx, self.ny)
        return _triangular_smooth_2d(fr2, v2).ravel()

    def _build_quadrant_fold(self):
        """A linear track's two axes (length vs. width) aren't interchangeable like a
        square's, so folding uses mirror reflection about each axis' own midpoint
        (see _build_reflect_quadrant_fold): each 'quadrant' = one length-end x one
        width-side corner, analogous to the paper's wall-corner quadrants."""
        self._quad_idx_flat, self.n_quad_bins, self.quad_shape = \
            _build_reflect_quadrant_fold(self.nx, self.ny)

    def fold_peak_bin(self, bin_idx):
        return _fold_peak_bin(self._quad_idx_flat, bin_idx)

    def fold_mean_map(self, values_flat, valid_flat):
        return _fold_mean_map(self._quad_idx_flat, self.n_quad_bins, values_flat, valid_flat)

    def plot_fine(self, ax, values_flat, valid_flat, cmap, norm):
        grid = np.full(self.n_bins, np.nan)
        grid[valid_flat] = values_flat[valid_flat]
        grid2 = grid.reshape(self.nx, self.ny)
        im = ax.imshow(np.ma.masked_invalid(grid2.T), origin='lower',
                        extent=[0, self.length, 0, self.width],
                        aspect='equal', cmap=cmap, norm=norm)
        ax.axis('off')
        return im

    def plot_quadrant_map(self, ax, values_flat, valid_flat, cmap, norm):
        half_x, half_y = self.quad_shape
        grid = np.full(self.n_quad_bins, np.nan)
        grid[valid_flat] = values_flat[valid_flat]
        grid2 = grid.reshape(half_x, half_y)
        im = ax.imshow(np.ma.masked_invalid(grid2.T), origin='lower',
                        extent=[0, half_x * self.bin_cm_x, 0, half_y * self.bin_cm_y],
                        aspect='equal', cmap=cmap, norm=norm)
        ax.axis('off')
        return im


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
    """Fig 1B: proportion of place-cell peaks, folded onto one reference quadrant
    (Fig 1A). Each cell contributes its one peak bin, remapped into the reference
    quadrant's local coordinates -- unlike the mean-rate map below, a peak is a
    single location so it is not itself averaged across the 4 folded regions."""
    place = [r for r in results if r['place_cell'] and r['peak_bin'] is not None]
    counts = np.zeros(handler.n_quad_bins)
    for r in place:
        q = handler.fold_peak_bin(r['peak_bin'])
        if q >= 0:
            counts[q] += 1
    total = counts.sum()
    pct = counts / total * 100.0 if total > 0 else np.full(handler.n_quad_bins, np.nan)
    return pct, int(total)


def pool_quadrant_mean_rate(handler, results: list) -> tuple:
    """Fig 1D: quadrant mean rate map, obtained by folding (Fig 1A) the single overall
    unsmoothed mean rate map across all place cells (same map as Fig S1H) -- per the
    Methods, Fig 1D/1E use "the overall mean rate maps for all cells" (unsmoothed)."""
    mean_map, valid = pool_fine_map(handler, results)
    return handler.fold_mean_map(np.where(valid, mean_map, 0.0), valid)


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

    cmap_b, norm_b = make_cmap_norm(np.concatenate([pct for pct, _ in peak_pct.values()]))
    cmap_d, norm_d = make_cmap_norm(np.concatenate([vals[valid] for vals, valid in rate.values()]))

    fig = plt.figure(figsize=(15, 10))
    axes_b, axes_d = [], []
    for i, key in enumerate(_ARENA_ORDER):
        handler = arena_handlers[key]
        proj = 'polar' if key == 'circular_track' else None

        pct, total = peak_pct[key]
        pct_valid = np.isfinite(pct)
        ax_b = fig.add_subplot(2, 3, i + 1, projection=proj)
        handler.plot_quadrant_map(ax_b, np.nan_to_num(pct), pct_valid, cmap_b, norm_b)
        ax_b.set_title(f'{_ARENA_TITLES[key]}\nPeak proportion (%), n={total}')
        axes_b.append(ax_b)

        vals, valid = rate[key]
        ax_d = fig.add_subplot(2, 3, i + 4, projection=proj)
        handler.plot_quadrant_map(ax_d, np.nan_to_num(vals), valid, cmap_d, norm_d)
        ax_d.set_title(f'{_ARENA_TITLES[key]}\nMean firing rate (Hz)')
        axes_d.append(ax_d)

    sm_b = cm.ScalarMappable(cmap=cmap_b, norm=norm_b); sm_b.set_array([])
    fig.colorbar(sm_b, ax=axes_b, shrink=0.6, label='% of peaks')
    sm_d = cm.ScalarMappable(cmap=cmap_d, norm=norm_d); sm_d.set_array([])
    fig.colorbar(sm_d, ax=axes_d, shrink=0.6, label='Mean firing rate (Hz)')

    fig.suptitle('Figure 1B/D -- Quadrant mean maps: folded peak-location proportion (top) and mean firing rate (bottom)')
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def export_excel(arena_handlers: dict, arena_results: dict, out_path: str):
    rows = []
    for key, results in arena_results.items():
        handler = arena_handlers[key]
        for r in results:
            q = handler.fold_peak_bin(r['peak_bin']) if r['peak_bin'] is not None else None
            if q is not None and q < 0:
                q = None
            rows.append(dict(
                arena=key, session=r['session'], unit=r['unit'],
                n_spikes=r['n_spikes'], peak_fr=r['peak_fr'], mean_fr=r['mean_fr'],
                sir=r['sir'], sparsity=r['sparsity'],
                bootstrap_sig=r['bootstrap_sig'], place_cell=r['place_cell'],
                peak_quadrant_bin=q,
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
