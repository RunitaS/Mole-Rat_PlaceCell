# -*- coding: utf-8 -*-
"""
Mean Rate Map & Quadrant Analysis (S1H + Figure 1B/D, after Muessig et al.)

Reproduces, for place cells pooled across multiple recording days:
  - Fig S1H : overall mean field-index map per arena, fine spatial bins (2 x 2 cm;
              genuinely 2D for every arena, including the linear track's length x
              width). Each cell's map is first normalized 0-1 to its own peak and
              minimum (smoothed) firing rate (field_index_map, place-cell method, as
              in ThetaMod_PhasePrecession_Stats_v3.py) before pooling across cells --
              this keeps a cell's contribution from being over- or under-weighted by
              its absolute peak rate or by where its field happens to sit in the map,
              which raw-Hz pooling does not.
  - Fig 1B/D: quadrant mean maps (Fig 1A method) -- the whole-arena map is cut into
              4 regions, each registered into one reference region's coordinates,
              and the 4 registered copies are averaged bin-by-bin, giving one small
              *multi-bin* heatmap (not a single scalar per quadrant):
                * open_field      : mirror-reflection fold (x and y independently) -> small square heatmap
                * circular_track  : mirror-reflection fold (arc-length x track-width, about the
                                    two axes of symmetry) -> small arc x width heatmap
                * linear_track    : mirror-reflection fold (length x width) -> small rectangle heatmap
              Fig 1B pools each cell's one peak location into the folded grid (proportion
              of peaks per bin); Fig 1D folds the single overall (Fig S1H) mean field-index map.

Arenas (edit ROOT_DIRECTORY / ARENA_CONFIGS geometry below):
  1. circular_track : annular track, outer dia 80 cm, inner dia 72 cm (4 cm wide) -- binned
                      genuinely 2D like the other two arenas: arc-length around the ring
                      (wrap-around) x radial position across the track width, both in the
                      same 2 x 2 cm bins used elsewhere, rather than a single 1D angular bin
                      spanning the whole track width.
  2. linear_track    : 80 x 8 cm linear track (vertical sessions auto-rotated 90 deg CCW)
  3. open_field      : circular open field, dia 60 cm

Arena type is auto-detected per session from its path: any 'Open' / 'Linear' / 'Circle'
path component (case-insensitive) under ROOT_DIRECTORY routes that session to open_field /
linear_track / circular_track respectively (see ARENA_FOLDER_KEYWORDS, _detect_arena_key).
There is no per-arena root folder anymore -- every animal/arena/day/session lives under the
single ROOT_DIRECTORY tree, e.g.:
  ROOT_DIRECTORY/Fa8477/Open/Day8/1Cntrl    -> open_field
  ROOT_DIRECTORY/Fa1059/Linear/Day10/1_0    -> linear_track
  ROOT_DIRECTORY/Fa8477/Circle/Day5/3Rot    -> circular_track

Tracking load/clean/smooth (pixel<->cm handling, jump removal, Gaussian smoothing), spike-position
matching (50 ms gate) and place-cell qualification (n_spikes>50, 1<peak_fr<15 Hz, SIR>0.5,
sparsity<0.9, location-shuffle bootstrap significant) are ported from
PlaceCellCharacterization_SpeedModv3_DownsampledPos15.py.

Folder layout expected under ROOT_DIRECTORY (same convention as the reference scripts):
  ROOT_DIRECTORY/.../<Open|Linear|Circle>/.../<session>/  containing exactly one tracking
  file (.csv or .xlsx) and one or more .ntt files.

Every arena's spatial bins are represented as a single flat index (0..n_bins-1); this lets rate-map
construction, SIR/sparsity, and the bootstrap significance test share one implementation across the
2D open field, the 2D (wrap-around in arc-length only) circular track, and the 2D linear track --
only the coordinate transform, smoothing kernel and plotting differ per arena.
"""

import os
import random
import concurrent.futures

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# ============================================================================
# CONFIGURATION -- edit root folder + geometry below
# ============================================================================

# Single root under which every animal/arena/day/session lives. Arena type is auto-detected
# per session from its path (see ARENA_FOLDER_KEYWORDS / _detect_arena_key below) -- no
# per-arena root folders needed any more.
ROOT_DIRECTORY = r'X:\NMR_group_data\Runita\Analysis\Thesis\Data\All_TT_PlaceTrue'

# Maps a lowercased path-component name to the arena it identifies. A session is routed to
# an arena if any component of its path (relative to ROOT_DIRECTORY) matches one of these
# keys case-insensitively, e.g. '.../Fa8477/Open/Day8/1Cntrl' -> open_field.
ARENA_FOLDER_KEYWORDS = {
    'open':     'open_field',
    'linear':   'linear_track',
    'circle':   'circular_track',
    'circular': 'circular_track',
}

# NOTE: 'shape' here selects the geometry handler (make_handler below: 'circle' ->
# OpenFieldHandler, 'ring' -> CircularTrackHandler, 'linear' -> LinearTrackHandler) -- it is
# unrelated to the ARENA_FOLDER_KEYWORDS folder-name keywords above (e.g. the 'Circle'-named
# session folders hold the *ring* track and route to shape='ring', not shape='circle').
# Do not "fix" these to match the folder names.
ARENA_CONFIGS = {
    'open_field': dict(
        shape='circle',
        diameter_cm=60.0,
    ),
    'circular_track': dict(
        shape='ring',
        outer_diameter_cm=80.0,
        inner_diameter_cm=72.0,
    ),
    'linear_track': dict(
        shape='linear',
        length_cm=80.0,
        width_cm=8.0,
    ),
}


def _detect_arena_key(dirpath: str) -> str:
    """Infers which arena a session directory belongs to from its path, by looking for an
    'Open' / 'Linear' / 'Circle' (or 'Circular') path component, case-insensitive, anywhere
    in dirpath. Returns None if no such component is found."""
    for part in os.path.normpath(dirpath).split(os.sep):
        arena_key = ARENA_FOLDER_KEYWORDS.get(part.strip().lower())
        if arena_key is not None:
            return arena_key
    return None

OUTPUT_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Data\All_TT_PlaceTrue\MeanRM_Quad'

fps            = 30           # tracking frame rate (Hz)
target_bin_cm  = 2.0          # spatial bin size (cm / along-track cm), fine-map resolution
min_occ_s      = 1.0          # exclude bins with < 1 s occupancy
COVERAGE_FRACTION = 0.50      # DEBUG: temporarily lowered from 0.80 to test whether the
                               # rest of the pipeline (bootstrap, place-cell qualification,
                               # pooling, quadrant fold, plotting) runs end-to-end on
                               # circular-track sessions that the 80% threshold was excluding
                               # (e.g. 2NoRot at ~78% bin coverage) -- a session must have
                               # >= min_occ_s occupancy in at least this fraction of the
                               # arena's total spatial bins (per handler, see
                               # `total_arena_bins`/`coverage_threshold_bins`), else every file
                               # (unit) from that session is skipped -- computed dynamically per
                               # handler rather than hardcoded, so it tracks target_bin_cm/geometry
MAX_GAP_US     = 50_000       # max spike-position gap (us)
N_BOOTSTRAP    = 1000         # circular-shift shuffles for SIR significance (reduce for faster runs)
MAX_WORKERS    = 4

POS_JUMP_THRESH_CMS  = 80.0   # frame-to-frame jumps implying a speed above this (cm/s) are tracking artifacts
POS_SMOOTH_SIGMA_SMP = 1.0    # Gaussian smoothing sigma (samples) applied to x/y tracking position
RATEMAP_SMOOTH_SIGMA_BINS = 1.0  # Gaussian smoothing sigma (bins) applied to rate maps

# Place-field extraction (pass-index 'place' filter-band criteria, auto_filter_band in
# pass_index_parser.m: field = area with raw rate >= 20% of the raw peak): a bin qualifies
# if its RAW (unsmoothed) rate exceeds both the cell's mean firing rate and this fraction
# of the raw rate map's peak; a connected run of qualifying bins is only kept as a field if
# it spans at least MIN_FIELD_BINS contiguous bins.
FIELD_PEAK_FRAC = 0.20
MIN_FIELD_BINS  = 7

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
# Gaussian smoothing kernel (2D for all three arenas -- open field, circular track, linear
# track -- with wrap-around optionally applied along one axis for the circular track)
# ============================================================================

def _gaussian_smooth_2d(fr_map: np.ndarray, valid_mask: np.ndarray,
                         sigma: float = RATEMAP_SMOOTH_SIGMA_BINS,
                         wrap_x: bool = False) -> np.ndarray:
    """2D Gaussian smoothing over a (bx, by) bin grid. wrap_x=True treats axis 0 (bx) as
    wrap-around (e.g. the circular track's arc-length axis, a closed ring) while axis 1
    (by) stays bounded -- a cylinder topology -- by passing scipy's per-axis `mode`."""
    mode = ('wrap' if wrap_x else 'constant', 'constant')
    fr_in   = np.where(valid_mask, fr_map, 0.0)
    mask_in = valid_mask.astype(np.float64)
    smoothed_fr = gaussian_filter(fr_in,   sigma=sigma, mode=mode, cval=0.0)
    smoothed_w  = gaussian_filter(mask_in, sigma=sigma, mode=mode, cval=0.0)
    smoothed = np.zeros_like(smoothed_fr)
    vw = smoothed_w > 0
    smoothed[vw] = smoothed_fr[vw] / smoothed_w[vw]
    smoothed[~valid_mask] = 0.0
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


def _connected_components_2d_flat(qualifies_flat: np.ndarray, nx: int, ny: int,
                                   wrap_x: bool = False) -> list:
    """8-connected component labelling over a flat (bx*ny+by)-indexed boolean array
    (open field / linear track / circular track grid), returning each component as a
    list of flat bin indices. wrap_x=True additionally connects bx=0 to bx=nx-1 (a
    cylinder topology, for the circular track's wrap-around arc-length axis). Direct
    analogue of the threshold-method field detector's visited/flood-fill
    (PlaceFieldDetection_ThresholdMethod_withRM_v5.py)."""
    qualifies_2d = qualifies_flat.reshape(nx, ny)
    visited = ~qualifies_2d
    components = []
    for i in range(nx):
        for j in range(ny):
            if visited[i, j]:
                continue
            region = []
            stack = [(i, j)]
            visited[i, j] = True
            while stack:
                bx, by = stack.pop()
                region.append(bx * ny + by)
                for ddx in (-1, 0, 1):
                    for ddy in (-1, 0, 1):
                        if ddx == 0 and ddy == 0:
                            continue
                        ni, nj = bx + ddx, by + ddy
                        if wrap_x:
                            ni = ni % nx
                        if 0 <= ni < nx and 0 <= nj < ny and not visited[ni, nj]:
                            visited[ni, nj] = True
                            stack.append((ni, nj))
            components.append(region)
    return components


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


def _build_reflect_quadrant_fold_arc(nx: int) -> tuple:
    """Arc-length analogue of _build_reflect_quadrant_fold's per-axis reflection: registers
    each of the 4 angular quadrants of a circular-track arena's arc-length (bx) axis onto
    one reference quadrant by mirror-reflection, the same registration principle used for
    the open field and linear track, instead of matching quadrants by a fixed rotation
    offset.

    The ring is cut into 4 equal arcs by two axes of symmetry (0/180 deg and 90/270 deg,
    the ring's analogue of a rectangle's two wall-pairs). Bins in quadrants 0 and 2 (each
    starting right after an axis) keep increasing local index with angle; bins in
    quadrants 1 and 3 have their local index reversed. This mirrors each quadrant about
    its own nearest axis of symmetry, so a bin's distance from that axis always lands on
    the same local index as in the reference quadrant -- wall a always maps onto wall a'
    -- for all 4 quadrants, rather than a plain rotation which would cross-match a bin
    near one axis in one quadrant against a bin near the opposite axis in another.
    """
    qn = nx // 4
    quad_idx = np.empty(nx, dtype=int)
    local_fwd = np.arange(qn)
    local_rev = local_fwd[::-1]
    for k in range(4):
        seg = np.arange(k * qn, (k + 1) * qn)
        quad_idx[seg] = local_fwd if k % 2 == 0 else local_rev
    return quad_idx, qn


def _build_reflect_quadrant_fold_cylinder(nx: int, ny: int) -> tuple:
    """2D (arc-length x radial-width) analogue of _build_reflect_quadrant_fold for the
    circular track: folds the 4 angular quadrants via mirror-reflection about the ring's
    two axes of symmetry (_build_reflect_quadrant_fold_arc, applied to the bx/arc-length
    axis), while the by/radial-width axis is carried through unchanged.

    This is not an arbitrary simplification: reflecting a point about a diameter of the
    ring (a line through the center) preserves its distance from the center exactly, so a
    bin's radial position (by) is invariant under the very reflection that defines the
    quadrants -- only its arc-length position (bx) moves. Unlike the open field/linear
    track (where both axes are reflected/halved), here only bx is folded into a quarter
    (qn = nx // 4 bins) and the full ny radial bins are kept in the folded map.
    """
    arc_quad_idx, qn = _build_reflect_quadrant_fold_arc(nx)
    bx_idx, by_idx = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    quad_idx_2d = arc_quad_idx[bx_idx] * ny + by_idx
    n_quad_bins = qn * ny
    return quad_idx_2d.ravel(), n_quad_bins, (qn, ny)


# ============================================================================
# Arena geometry handlers
#
# Each handler converts (x_cm, y_cm) into a single flat spatial-bin index per
# position sample (0..n_bins-1), so the rate-map / SIR / bootstrap machinery
# below is written once and shared by the 2D open field, the 2D circular track
# (wrap-around along its arc-length axis only), and the 2D linear track.
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

        # distance from each bin centre to the arena's (single, circular) wall, for the
        # boundary-preference KDE analysis (analyze_wall_distance_kde)
        self.dist_to_wall_flat = np.clip(self.diameter / 2.0 - r, 0.0, None).ravel()
        self.max_dist_to_wall = self.diameter / 2.0

        # distance from each bin centre to the arena CENTER (0 at center, diameter/2 at
        # the wall) -- the boundary-preference analysis' x-axis for this arena
        # (analyze_boundary_firing)
        self.dist_from_center_flat = r.ravel()
        self.max_dist_from_center = self.diameter / 2.0

        # total bins actually inside the circular arena (excludes the corner bins of the
        # bounding nx*ny grid that geom_valid already masks out) -- the denominator for the
        # 80% coverage criterion (COVERAGE_FRACTION)
        self.total_arena_bins = int(self.geom_valid.sum())
        self.coverage_threshold_bins = int(np.ceil(COVERAGE_FRACTION * self.total_arena_bins))

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
        return _gaussian_smooth_2d(fr2, v2).ravel()

    def connected_components(self, qualifies_flat):
        return _connected_components_2d_flat(qualifies_flat, self.nx, self.ny)

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

    def overlay_quadrant_significance(self, ax, sig_mask: np.ndarray):
        """Outlines, on top of an existing plot_quadrant_map axes, the folded bins where the
        quadrant-fold KDE analysis found the observed map's bootstrap CI to significantly
        exceed the occupancy-null KDE (analyze_quadrant_kde)."""
        half = self.quad_shape[0]
        sig2d = sig_mask.reshape(half, half).astype(float)
        if sig2d.max() > 0:
            side = half * self.bin_cm
            ax.contour(sig2d.T, levels=[0.5], colors='black', linewidths=1.2,
                       extent=[0, side, 0, side], origin='lower')


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
        self.track_width_cm = self.outer_r - self.inner_r

        # genuine 2D binning, same procedure as the open field / linear track: bins along
        # the track's arc-length (wrap-around, like the linear track's length axis) AND
        # bins across its radial width (like the linear track's width axis), both in the
        # same bin_cm (2 x 2 cm) resolution -- rather than a single 1D angular bin that
        # spans the whole track width.
        circumference = 2 * np.pi * self.mean_r
        self.circumference_cm = circumference
        # kept as a multiple of 4 so the quadrant fold below splits into 4 exactly
        # equal-length arcs (see _build_quadrant_fold)
        self.nx = max(8, 4 * int(round(circumference / bin_cm / 4.0)))   # along arc-length
        self.ny = max(2, int(round(self.track_width_cm / bin_cm)))       # across radial width
        self.bin_width_deg = 360.0 / self.nx
        self.bin_cm_y = self.track_width_cm / self.ny
        self.n_bins = self.nx * self.ny

        # distance from each bin's radial position to the nearest of the track's two
        # edges (inner or outer rim), for the boundary-preference KDE analysis
        # (analyze_wall_distance_kde). Independent of arc position (bx): reflecting a
        # point about a diameter of the ring preserves its radial distance, so only by
        # (radial bin) matters here, mirroring _build_reflect_quadrant_fold_cylinder's
        # observation that by is invariant under the quadrant-fold reflection.
        by_centers_cm = (np.arange(self.ny) + 0.5) * self.bin_cm_y
        dist_by = np.minimum(by_centers_cm, self.track_width_cm - by_centers_cm)
        self.dist_to_wall_flat = np.tile(dist_by, self.nx)
        self.max_dist_to_wall = self.track_width_cm / 2.0

        # boundary-preference analysis (analyze_boundary_firing): the track is only 2
        # bins wide radially, so instead of a continuous distance-to-wall x-axis, each
        # bin is classified inner-ring vs outer-ring (by < ny//2 -- for the actual
        # ny=2 config this is exactly the inner rim bin vs the outer rim bin), and the
        # x-axis becomes arc-length position around the ring (0..circumference_cm),
        # same convention as the linear track's length axis.
        bx_centers_cm = (np.arange(self.nx) + 0.5) * (self.circumference_cm / self.nx)
        self.arc_pos_cm_flat = np.repeat(bx_centers_cm, self.ny)
        self.inner_side_flat = np.tile(np.arange(self.ny) < (self.ny // 2), self.nx)

        # every bin of the nx*ny grid is on the track (no out-of-bounds corners to mask),
        # so all n_bins count toward the 80% coverage criterion (COVERAGE_FRACTION)
        self.total_arena_bins = self.n_bins
        self.coverage_threshold_bins = int(np.ceil(COVERAGE_FRACTION * self.total_arena_bins))

        self._build_quadrant_fold()

    def orient(self, x_cm, y_cm):
        return x_cm, y_cm

    def to_bins(self, x_cm, y_cm):
        r = np.hypot(x_cm - self.cx, y_cm - self.cy)
        theta = np.degrees(np.arctan2(y_cm - self.cy, x_cm - self.cx)) % 360.0
        bx = np.clip((theta / self.bin_width_deg).astype(int), 0, self.nx - 1)
        rel_r = np.clip(r - self.inner_r, 0.0, self.track_width_cm)
        by = np.clip((rel_r / self.bin_cm_y).astype(int), 0, self.ny - 1)
        flat = bx * self.ny + by
        on_track = (r >= self.inner_r - self.radial_tol_cm) & (r <= self.outer_r + self.radial_tol_cm)
        return flat, on_track

    def smooth(self, fr_flat, valid_flat):
        fr2 = fr_flat.reshape(self.nx, self.ny)
        v2  = valid_flat.reshape(self.nx, self.ny)
        return _gaussian_smooth_2d(fr2, v2, wrap_x=True).ravel()

    def connected_components(self, qualifies_flat):
        return _connected_components_2d_flat(qualifies_flat, self.nx, self.ny, wrap_x=True)

    def _build_quadrant_fold(self):
        """Splits the ring's arc-length axis into 4 equal 90 deg arcs and folds them onto
        one reference arc via mirror reflection about the ring's two axes of symmetry,
        while carrying the radial-width axis through unchanged (see
        _build_reflect_quadrant_fold_cylinder) -- the same registration principle used for
        the open field and linear track, rather than matching arcs by a fixed rotation
        offset."""
        self._quad_idx_flat, self.n_quad_bins, self.quad_shape = \
            _build_reflect_quadrant_fold_cylinder(self.nx, self.ny)

    def fold_peak_bin(self, bin_idx):
        return _fold_peak_bin(self._quad_idx_flat, bin_idx)

    def fold_mean_map(self, values_flat, valid_flat):
        return _fold_mean_map(self._quad_idx_flat, self.n_quad_bins, values_flat, valid_flat)

    def plot_fine(self, ax, values_flat, valid_flat, cmap, norm):
        theta_edges = np.linspace(0, 2 * np.pi, self.nx + 1)
        r_edges = np.linspace(self.inner_r, self.outer_r, self.ny + 1)
        grid = np.where(valid_flat, values_flat, np.nan).reshape(self.nx, self.ny)
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        pcm = ax.pcolormesh(theta_edges, r_edges, grid.T, cmap=cmap, norm=norm, shading='auto')
        ax.set_ylim(0, self.outer_r + 5)
        ax.set_yticklabels([])
        ax.grid(False)
        return pcm

    def plot_quadrant_map(self, ax, values_flat, valid_flat, cmap, norm):
        qn, ny = self.quad_shape
        theta_edges = np.linspace(0, np.pi / 2, qn + 1)
        r_edges = np.linspace(self.inner_r, self.outer_r, ny + 1)
        grid = np.where(valid_flat, values_flat, np.nan).reshape(qn, ny)
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        pcm = ax.pcolormesh(theta_edges, r_edges, grid.T, cmap=cmap, norm=norm, shading='auto')
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        ax.set_ylim(0, self.outer_r + 5)
        ax.set_yticklabels([])
        ax.grid(False)
        return pcm

    def overlay_quadrant_significance(self, ax, sig_mask: np.ndarray):
        """Polar analogue of OpenFieldHandler.overlay_quadrant_significance: outlines the
        folded (arc-length x radial-width) bins flagged significant by analyze_quadrant_kde,
        on top of an existing plot_quadrant_map polar axes."""
        qn, ny = self.quad_shape
        sig2d = sig_mask.reshape(qn, ny).astype(float)
        if sig2d.max() > 0:
            theta_centers = (np.arange(qn) + 0.5) * ((np.pi / 2) / qn)
            r_centers = self.inner_r + (np.arange(ny) + 0.5) * ((self.outer_r - self.inner_r) / ny)
            ax.contour(theta_centers, r_centers, sig2d.T, levels=[0.5], colors='black', linewidths=1.2)


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

        # distance from each bin centre to the nearest of the rectangle's 4 walls (2
        # length-ends + 2 width-sides), for the boundary-preference KDE analysis
        # (analyze_wall_distance_kde) -- same "nearest wall, either axis" convention as
        # classify_edge_centre's dist_from_wall_cm in BoundaryAnalysis_Pipeline_v2.py.
        bx_centers_cm = (np.arange(self.nx) + 0.5) * self.bin_cm_x
        by_centers_cm = (np.arange(self.ny) + 0.5) * self.bin_cm_y
        dist_x = np.minimum(bx_centers_cm, self.length - bx_centers_cm)
        dist_y = np.minimum(by_centers_cm, self.width - by_centers_cm)
        DX, DY = np.meshgrid(dist_x, dist_y, indexing='ij')
        self.dist_to_wall_flat = np.minimum(DX, DY).ravel()
        self.max_dist_to_wall = min(self.length, self.width) / 2.0

        # distance from each bin centre to the MIDPOINT of the long (length) wall only
        # (width axis ignored) -- the boundary-preference analysis' x-axis for this
        # arena (analyze_boundary_firing). 0 at the track's midpoint, length/2 (40 cm)
        # at either end; both ends fold onto the same distance value.
        dist_from_mid_x = np.abs(bx_centers_cm - self.length / 2.0)
        self.dist_from_mid_flat = np.repeat(dist_from_mid_x, self.ny)
        self.max_dist_from_mid = self.length / 2.0

        # every bin of the nx*ny grid is on the track (no out-of-bounds corners to mask),
        # so all n_bins count toward the 80% coverage criterion (COVERAGE_FRACTION)
        self.total_arena_bins = self.n_bins
        self.coverage_threshold_bins = int(np.ceil(COVERAGE_FRACTION * self.total_arena_bins))

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
        return _gaussian_smooth_2d(fr2, v2).ravel()

    def connected_components(self, qualifies_flat):
        return _connected_components_2d_flat(qualifies_flat, self.nx, self.ny)

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

    def overlay_quadrant_significance(self, ax, sig_mask: np.ndarray):
        """Rectangular analogue of OpenFieldHandler.overlay_quadrant_significance, using this
        handler's own per-axis (length vs. width) bin size for the extent."""
        half_x, half_y = self.quad_shape
        sig2d = sig_mask.reshape(half_x, half_y).astype(float)
        if sig2d.max() > 0:
            ax.contour(sig2d.T, levels=[0.5], colors='black', linewidths=1.2,
                       extent=[0, half_x * self.bin_cm_x, 0, half_y * self.bin_cm_y], origin='lower')


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

def field_index_map(fr_flat: np.ndarray, valid_flat: np.ndarray) -> np.ndarray:
    """Normalize a (flat-bin) rate map into a 0-1 field index map: (rate - min) / (peak -
    min) over the valid bins, place-cell method of field_index_map in
    ThetaMod_PhasePrecession_Stats_v3.py. Pooling these across cells (instead of raw Hz)
    keeps a cell with a low peak rate or an off-center field from being over- or
    under-weighted relative to the rest of the pool."""
    fi = np.full(fr_flat.shape, np.nan, dtype=np.float64)
    if not valid_flat.any():
        return fi
    vals = fr_flat[valid_flat]
    vmin, vmax = float(vals.min()), float(vals.max())
    rng = vmax - vmin
    fi[valid_flat] = (vals - vmin) / rng if rng > 0 else 0.0
    return fi


def extract_place_field_mask(cell: dict, handler) -> np.ndarray:
    """Place-field bins via the pass-index 'place' filter-band criteria (auto_filter_band
    in pass_index_parser.m), adapted to a discrete bin grid instead of a continuous filter
    band: a bin qualifies if its RAW (unsmoothed) rate exceeds both the cell's mean firing
    rate and FIELD_PEAK_FRAC (20%) of the RAW rate map's peak (mirroring auto_filter_band's
    `rmap > 0.2 * peak`), and only connected runs of qualifying bins spanning >=
    MIN_FIELD_BINS (7) contiguous bins are kept as a field (adapting auto_filter_band's
    area-based field radius, which has no direct meaning on a discrete bin grid, into a
    minimum-size connected-component criterion instead)."""
    valid  = cell['valid']
    fr_raw = cell['fr_raw']
    field_mask = np.zeros(handler.n_bins, dtype=bool)
    if not valid.any():
        return field_mask

    peak_raw = float(fr_raw[valid].max())
    mean_fr  = cell['mean_fr']
    qualifies = valid & (fr_raw > mean_fr) & (fr_raw > FIELD_PEAK_FRAC * peak_raw)
    if not qualifies.any():
        return field_mask

    for component in handler.connected_components(qualifies):
        if len(component) >= MIN_FIELD_BINS:
            field_mask[component] = True
    return field_mask


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
    fi_map = field_index_map(fr_smooth, valid)

    result = dict(n_spikes=n_spikes, fr_raw=fr_raw, fr_smooth=fr_smooth, fi_map=fi_map,
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

    # Coverage criterion: the occupancy map (cell['valid']) depends only on tracking, not on
    # this unit's spikes, so it is identical for every unit in this session -- this check
    # therefore skips every file (unit) from a session whose tracking did not cover enough of
    # the arena (COVERAGE_FRACTION), not just this one unit.
    covered_bins = int(cell['valid'].sum())
    if covered_bins < handler.coverage_threshold_bins:
        print(f'  [SKIP low coverage] {session_name}/{ntt_file}: '
              f'{covered_bins}/{handler.total_arena_bins} bins covered '
              f'({covered_bins / handler.total_arena_bins:.0%}), '
              f'need >= {handler.coverage_threshold_bins} ({COVERAGE_FRACTION:.0%})')
        return None

    boot = run_bootstrap_generic(handler, cell, cell['sir'])
    boot_sig = boot.get('bootstrap_sig')
    place_cell = bool(
        boot_sig is True and
        cell['n_spikes'] > 50 and
        1.0 < cell['peak_fr'] < 25.0 and
        cell['sir'] > 0.5 and
        cell['sparsity'] < 0.75
    )

    peak_bin = None
    if cell['valid'].any():
        masked = np.where(cell['valid'], cell['fr_smooth'], -np.inf)
        peak_bin = int(np.argmax(masked))

    field_mask = extract_place_field_mask(cell, handler)

    return dict(
        session=session_name, unit=ntt_file,
        n_spikes=cell['n_spikes'], peak_fr=cell['peak_fr'], mean_fr=cell['mean_fr'],
        sir=cell['sir'], sparsity=cell['sparsity'],
        bootstrap_sig=boot_sig, place_cell=place_cell,
        fr_raw=cell['fr_raw'], fr_smooth=cell['fr_smooth'], fi_map=cell['fi_map'], valid=cell['valid'], occ_map=cell['occ_map'],
        peak_bin=peak_bin, field_mask=field_mask,
    )


# ============================================================================
# Batch scan per arena
# ============================================================================

def collect_arena_results(arena_key: str, cfg: dict) -> tuple:
    handler = make_handler(cfg)
    root = ROOT_DIRECTORY

    jobs = []
    for dirpath, _, filenames in os.walk(root):
        if _detect_arena_key(dirpath) != arena_key:
            continue
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
    """Pools each place cell's field index map (0-1 normalized to its own peak and
    minimum firing rate, see field_index_map) rather than raw Hz -- so a cell's
    contribution to the pooled map does not depend on its absolute peak rate or on
    where its field happens to sit relative to other cells' fields."""
    place = [r for r in results if r['place_cell']]
    if not place:
        return np.full(handler.n_bins, np.nan), np.zeros(handler.n_bins, dtype=bool)
    stack = np.full((len(place), handler.n_bins), np.nan)
    for i, r in enumerate(place):
        stack[i, r['valid']] = r['fi_map'][r['valid']]
    mean_map = np.nanmean(stack, axis=0)
    any_valid = ~np.all(np.isnan(stack), axis=0)
    return mean_map, any_valid


def pool_field_only_map(handler, results: list) -> tuple:
    """Mean field-index map pooled ONLY over each place cell's own extracted place-field
    bins (extract_place_field_mask) -- bins a cell never fielded in do not dilute its
    contribution, per Mean_RM_Fixes.md ('Use only place fields to construct the mean RM').
    Like pool_fine_map, each cell is normalized to its own 0-1 field index (fi_map) before
    pooling, so a cell's contribution does not depend on its absolute peak rate. Pooling is
    occupancy-weighted per bin (weight = seconds of occupancy, proportional to the number of
    position samples collected there) rather than an unweighted mean across cells, so a
    bin's pooled value is not biased by cells that happened to oversample (or undersample)
    that bin relative to others."""
    place = [r for r in results if r['place_cell']]
    if not place:
        return np.full(handler.n_bins, np.nan), np.zeros(handler.n_bins, dtype=bool)

    sums    = np.zeros(handler.n_bins, dtype=np.float64)
    weights = np.zeros(handler.n_bins, dtype=np.float64)
    for r in place:
        m = r['field_mask'] & r['valid']
        if not m.any():
            continue
        w = r['occ_map'][m]
        sums[m]    += r['fi_map'][m] * w
        weights[m] += w

    mean_map = np.full(handler.n_bins, np.nan)
    any_valid = weights > 0
    mean_map[any_valid] = sums[any_valid] / weights[any_valid]
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
    """Fig 1D: quadrant mean field-index map, obtained by folding (Fig 1A) the single
    overall mean field-index map across all place cells (same map as Fig S1H) -- each
    cell's map is first normalized 0-1 to its own peak and minimum firing rate
    (field_index_map) before pooling, per the Methods' intent that Fig 1D/1E compare
    field shape rather than absolute rate across cells."""
    mean_map, valid = pool_fine_map(handler, results)
    return handler.fold_mean_map(np.where(valid, mean_map, 0.0), valid)


def pool_quadrant_field_only_rate(handler, results: list) -> tuple:
    """Field-only analogue of pool_quadrant_mean_rate: folds (Fig 1A) the place-field-only,
    occupancy-weighted mean field-index map (pool_field_only_map) into the same reference
    quadrant, instead of the unrestricted overall mean map -- the third of the three folded
    rate maps (alongside pool_quadrant_mean_rate and pool_quadrant_peak_proportion) that the
    quadrant-fold KDE analysis below compares against an occupancy-null KDE."""
    mean_map, valid = pool_field_only_map(handler, results)
    return handler.fold_mean_map(np.where(valid, mean_map, 0.0), valid)


# ============================================================================
# Boundary-preference analysis
#
# For each arena and each of 3 pooled maps, builds an averaged-firing-vs-position tuning
# curve along one geometric axis specific to the arena, Gaussian-kernel-smooths it, and
# tests whether it shows any peak not explained by how much time/area the animal sampled
# at each position:
#   open_field     : x = distance from the arena CENTER (0..30 cm), handler.dist_from_center_flat.
#   linear_track   : x = distance from the MIDPOINT of the 80 cm long wall (0..40 cm; the
#                    width axis is dropped and both track ends fold onto one axis),
#                    handler.dist_from_mid_flat.
#   circular_track : only 2 bins wide radially, so there is no continuous radial axis to
#                    fit a curve over. Each bin is classified inner-ring vs outer-ring
#                    (handler.inner_side_flat) and x = arc-length position around the ring
#                    (handler.arc_pos_cm_flat, wrap-around) -- two curves per map type.
#
#   'overall' : the Fig S1H mean field-index map (pool_fine_map), UNWEIGHTED mean across
#               spatial bins sharing a position (mirrors pool_fine_map's own unweighted
#               pooling across cells).
#   'field'   : the place-field-only mean map (pool_field_only_map), occupancy-WEIGHTED
#               mean across spatial bins sharing a position (mirrors pool_field_only_map's
#               own occupancy weighting).
#   'peak'    : each place cell's single peak-firing bin, as % of place-cell peaks per
#               position (a point-process histogram, unfolded -- unlike Fig 1B's
#               quadrant-folded peak proportions).
#
# Significance: the null is the same bin-average-then-smooth curve built from pooled
# occupancy (dwell-time) instead of firing, then -- like the observed curve -- rescaled to
# % of its own total so the two are comparable despite different native units (field-index
# vs. seconds). A cell-identity bootstrap (resampling place cells with replacement) gives
# the observed curve's sampling uncertainty; a position is flagged when the ENTIRE
# bootstrap CI (its 2.5th percentile) lies above the null, and contiguous flagged runs are
# reported as significant boundary peaks.
#
# Note: because 'overall'/'field' now AVERAGE (rather than density-weight) bins into each
# position, arena geometry/bin-count no longer biases the curve's mean the way it did under
# the old weighted-KDE-of-raw-samples design -- the occupancy null here mainly flags
# positions the animal barely sampled (noisy estimates) and behavioral sampling bias (e.g.
# thigmotaxis), rather than correcting a geometric mean-bias. 'peak' stays a count/mass-type
# measure, so its occupancy null plays the same bias-correcting role as before.
# ============================================================================

BOUNDARY_BIN_CM            = target_bin_cm            # position-bin resolution for the averaged curve
BOUNDARY_SMOOTH_SIGMA_BINS = RATEMAP_SMOOTH_SIGMA_BINS # Gaussian kernel width, in position-bins
N_KDE_BOOTSTRAP             = 1000
KDE_ALPHA                   = 0.05   # two-sided: flag where the observed curve's 2.5th percentile exceeds the null


def _gaussian_smooth_1d_nan(values: np.ndarray, valid: np.ndarray, sigma_bins: float,
                             wrap: bool = False) -> np.ndarray:
    """NaN-safe 1D Gaussian smoothing (same trick as _gaussian_smooth_2d): smooth
    value*mask and mask separately via gaussian_filter1d, then divide. wrap=True treats
    the axis as a closed ring (the circular track's arc-length position)."""
    mode = 'wrap' if wrap else 'constant'
    v_in = np.where(valid, values, 0.0)
    w_in = valid.astype(np.float64)
    sv = gaussian_filter1d(v_in, sigma=sigma_bins, mode=mode)
    sw = gaussian_filter1d(w_in, sigma=sigma_bins, mode=mode)
    out = np.zeros_like(sv)
    vw = sw > 1e-9
    out[vw] = sv[vw] / sw[vw]
    return out


def _normalize_pct(curve: np.ndarray) -> np.ndarray:
    """Rescales a curve to sum to 100 across position-bins, so curves built from
    different native units (Hz-like field-index vs. seconds of occupancy) or different
    aggregation types (mean vs. count) become directly comparable in shape."""
    total = curve.sum()
    return curve / total * 100.0 if total > 0 else curve


def _distance_bin_average(pos_flat: np.ndarray, value_flat: np.ndarray, weight_flat: np.ndarray,
                           valid_mask: np.ndarray, bin_edges: np.ndarray) -> tuple:
    """Groups bins of pos_flat into bin_edges intervals and computes a WEIGHTED MEAN of
    value_flat within each group (weight_flat all-ones gives an unweighted mean). Returns
    (centers, mean_per_bin, group_valid) -- group_valid is False for empty groups."""
    n_bins = len(bin_edges) - 1
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    m = (valid_mask & np.isfinite(pos_flat) & np.isfinite(value_flat) & np.isfinite(weight_flat)
         & (weight_flat >= 0))
    pos, val, w = pos_flat[m], value_flat[m], weight_flat[m]

    sums    = np.zeros(n_bins)
    weights = np.zeros(n_bins)
    if len(pos):
        group = np.clip(np.searchsorted(bin_edges, pos, side='right') - 1, 0, n_bins - 1)
        np.add.at(sums,    group, val * w)
        np.add.at(weights, group, w)

    mean_per_bin = np.zeros(n_bins)
    group_valid  = weights > 0
    mean_per_bin[group_valid] = sums[group_valid] / weights[group_valid]
    return centers, mean_per_bin, group_valid


def _position_sum(pos_flat: np.ndarray, value_flat: np.ndarray, valid_mask: np.ndarray,
                   bin_edges: np.ndarray) -> tuple:
    """Groups bins of pos_flat into bin_edges intervals and SUMS value_flat within each
    group (a mass-like aggregate, unlike _distance_bin_average's mean). Returns
    (centers, sum_per_bin, all-True group_valid -- a zero-sum bin is a legitimate value)."""
    n_bins = len(bin_edges) - 1
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    m = valid_mask & np.isfinite(pos_flat) & np.isfinite(value_flat)
    pos, val = pos_flat[m], value_flat[m]
    sums = np.zeros(n_bins)
    if len(pos):
        group = np.clip(np.searchsorted(bin_edges, pos, side='right') - 1, 0, n_bins - 1)
        np.add.at(sums, group, val)
    return centers, sums, np.ones(n_bins, dtype=bool)


def _position_histogram(pos_values: np.ndarray, bin_edges: np.ndarray) -> tuple:
    """Counts of raw scalar pos_values (e.g. one per place-cell peak) falling into each
    bin_edges interval. Returns (centers, counts_per_bin, all-True group_valid)."""
    n_bins = len(bin_edges) - 1
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts = np.zeros(n_bins)
    if len(pos_values):
        group = np.clip(np.searchsorted(bin_edges, pos_values, side='right') - 1, 0, n_bins - 1)
        np.add.at(counts, group, 1.0)
    return centers, counts, np.ones(n_bins, dtype=bool)


def _pooled_occupancy(handler, results: list, field_only: bool) -> tuple:
    """Pooled dwell-time (s) per bin across all place cells in `results` -- optionally
    restricted to each cell's own extracted place-field bins, mirroring
    pool_field_only_map's restriction."""
    place = [r for r in results if r['place_cell']]
    occ_sum   = np.zeros(handler.n_bins, dtype=np.float64)
    valid_any = np.zeros(handler.n_bins, dtype=bool)
    for r in place:
        m = (r['field_mask'] & r['valid']) if field_only else r['valid']
        occ_sum[m] += r['occ_map'][m]
        valid_any |= m
    return occ_sum, valid_any


_MAP_ORDER  = ['overall', 'peak', 'field']
_MAP_TITLES = {'overall': 'Overall mean rate map', 'peak': 'Peak mean rate map',
               'field': 'Place-field mean rate map'}

_ARENA_XLABEL = {
    'open_field':      'Distance from center (cm)',
    'linear_track':    'Distance from track midpoint (cm)',
    'circular_track':  'Position along track (cm)',
}


def _arena_position_axis(handler, arena_key: str) -> tuple:
    """Returns (pos_flat, bin_edges, wrap) -- the geometric x-axis for one arena's
    boundary-preference curve (see module comment above)."""
    if arena_key == 'open_field':
        edges = np.arange(0.0, handler.max_dist_from_center + BOUNDARY_BIN_CM, BOUNDARY_BIN_CM)
        return handler.dist_from_center_flat, edges, False
    elif arena_key == 'linear_track':
        edges = np.arange(0.0, handler.max_dist_from_mid + BOUNDARY_BIN_CM, BOUNDARY_BIN_CM)
        return handler.dist_from_mid_flat, edges, False
    elif arena_key == 'circular_track':
        edges = np.linspace(0.0, handler.circumference_cm, handler.nx + 1)
        return handler.arc_pos_cm_flat, edges, True
    raise ValueError(arena_key)


def _build_observed_curve(handler, cells: list, map_type: str, pos_flat: np.ndarray,
                           bin_edges: np.ndarray, side_mask: np.ndarray, wrap: bool) -> tuple:
    """Returns (curve [% of its own total], n_samples) for one map type, built from
    `cells` (the real place-cell population, or one bootstrap resample of it)."""
    if map_type == 'overall':
        mean_map, valid = pool_fine_map(handler, cells)
        valid = valid & side_mask
        _, avg, avg_valid = _distance_bin_average(pos_flat, mean_map, np.ones(handler.n_bins), valid, bin_edges)
        curve = _normalize_pct(_gaussian_smooth_1d_nan(avg, avg_valid, BOUNDARY_SMOOTH_SIGMA_BINS, wrap))
        return curve, int(valid.sum())
    elif map_type == 'field':
        mean_map, valid = pool_field_only_map(handler, cells)
        occ_field, _ = _pooled_occupancy(handler, cells, field_only=True)
        valid = valid & side_mask
        _, avg, avg_valid = _distance_bin_average(pos_flat, mean_map, occ_field, valid, bin_edges)
        curve = _normalize_pct(_gaussian_smooth_1d_nan(avg, avg_valid, BOUNDARY_SMOOTH_SIGMA_BINS, wrap))
        return curve, int(valid.sum())
    else:  # 'peak'
        peak_bins = [r['peak_bin'] for r in cells
                     if r.get('peak_bin') is not None and side_mask[r['peak_bin']]]
        pos_vals = pos_flat[peak_bins] if peak_bins else np.array([])
        _, counts, cvalid = _position_histogram(pos_vals, bin_edges)
        curve = _normalize_pct(_gaussian_smooth_1d_nan(counts, cvalid, BOUNDARY_SMOOTH_SIGMA_BINS, wrap))
        return curve, len(peak_bins)


def _build_null_curve(handler, cells: list, map_type: str, pos_flat: np.ndarray,
                       bin_edges: np.ndarray, side_mask: np.ndarray, wrap: bool) -> np.ndarray:
    """Occupancy-derived null curve (% of its own total), paired with _build_observed_curve
    so mean-type observed curves ('overall'/'field') get a mean-type null and the
    count-type observed curve ('peak') gets a sum-type (mass-like) null."""
    if map_type == 'overall':
        occ_all, valid = _pooled_occupancy(handler, cells, field_only=False)
        valid = valid & side_mask
        _, avg, avg_valid = _distance_bin_average(pos_flat, occ_all, np.ones(handler.n_bins), valid, bin_edges)
        return _normalize_pct(_gaussian_smooth_1d_nan(avg, avg_valid, BOUNDARY_SMOOTH_SIGMA_BINS, wrap))
    elif map_type == 'field':
        occ_field, valid = _pooled_occupancy(handler, cells, field_only=True)
        valid = valid & side_mask
        _, avg, avg_valid = _distance_bin_average(pos_flat, occ_field, np.ones(handler.n_bins), valid, bin_edges)
        return _normalize_pct(_gaussian_smooth_1d_nan(avg, avg_valid, BOUNDARY_SMOOTH_SIGMA_BINS, wrap))
    else:  # 'peak'
        occ_all, valid = _pooled_occupancy(handler, cells, field_only=False)
        valid = valid & side_mask
        _, tot, tvalid = _position_sum(pos_flat, occ_all, valid, bin_edges)
        return _normalize_pct(_gaussian_smooth_1d_nan(tot, tvalid, BOUNDARY_SMOOTH_SIGMA_BINS, wrap))


def _find_significant_peaks(grid: np.ndarray, med: np.ndarray, lo: np.ndarray,
                             null_curve: np.ndarray, min_run_points: int = 3) -> list:
    """Contiguous grid runs where the observed curve's bootstrap lower bound exceeds the
    occupancy null; each run's peak = the observed-curve local maximum within it."""
    sig = lo > null_curve
    labeled, n_labels = label(sig)
    peaks = []
    for lbl in range(1, n_labels + 1):
        idx = np.where(labeled == lbl)[0]
        if len(idx) < min_run_points:
            continue
        peak_i = idx[np.argmax(med[idx])]
        peaks.append(dict(
            start_cm=round(float(grid[idx[0]]), 2),
            end_cm=round(float(grid[idx[-1]]), 2),
            peak_dist_cm=round(float(grid[peak_i]), 2),
            peak_pct=round(float(med[peak_i]), 4),
            null_pct=round(float(null_curve[peak_i]), 4),
        ))
    return peaks


def analyze_boundary_firing(handler, results: list, map_type: str, arena_key: str,
                             side_mask: np.ndarray | None = None,
                             n_boot: int = N_KDE_BOOTSTRAP,
                             rng: np.random.Generator | None = None) -> dict:
    """Boundary-preference analysis for one arena x one pooled map (x one side, for the
    circular track): see module-level comment above for the curve / null / bootstrap /
    peak-detection design."""
    place_cells = [r for r in results if r['place_cell']]
    pos_flat, bin_edges, wrap = _arena_position_axis(handler, arena_key)
    if side_mask is None:
        side_mask = np.ones(handler.n_bins, dtype=bool)
    grid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_pos = len(grid)

    out = dict(map_type=map_type, grid=grid, n_place_cells=len(place_cells), n_samples=0,
               real_med=np.zeros(n_pos), real_lo=np.zeros(n_pos), real_hi=np.zeros(n_pos),
               null_curve=np.zeros(n_pos), peaks=[])
    if not place_cells:
        return out

    real_curve, n_samples = _build_observed_curve(handler, place_cells, map_type, pos_flat,
                                                    bin_edges, side_mask, wrap)
    out['n_samples'] = n_samples
    if n_samples < 2:
        return out

    out['null_curve'] = _build_null_curve(handler, place_cells, map_type, pos_flat,
                                           bin_edges, side_mask, wrap)

    rng = rng if rng is not None else np.random.default_rng(0)
    n = len(place_cells)
    curves = np.zeros((n_boot, n_pos))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_cells = [place_cells[i] for i in idx]
        curves[b], _ = _build_observed_curve(handler, boot_cells, map_type, pos_flat, bin_edges, side_mask, wrap)
    out['real_lo']  = np.percentile(curves, 100 * KDE_ALPHA / 2.0, axis=0)
    out['real_hi']  = np.percentile(curves, 100 * (1.0 - KDE_ALPHA / 2.0), axis=0)
    out['real_med'] = np.percentile(curves, 50, axis=0)

    out['peaks'] = _find_significant_peaks(grid, out['real_med'], out['real_lo'], out['null_curve'])
    return out


def plot_boundary_firing(results: dict, save_path: str):
    fig, axes = plt.subplots(len(_MAP_ORDER), len(_ARENA_ORDER), figsize=(15, 12), squeeze=False)
    for col, arena_key in enumerate(_ARENA_ORDER):
        for row, map_type in enumerate(_MAP_ORDER):
            ax = axes[row][col]
            if arena_key == 'circular_track':
                n_place, n_sig_total = 0, 0
                for side, color, ls in (('inner', '#C0392B', '-'), ('outer', '#2E86C1', '--')):
                    res = results[arena_key][map_type][side]
                    grid = res['grid']
                    ax.plot(grid, res['null_curve'], color=color, ls=':', lw=1, alpha=0.6,
                            label=f'{side} occupancy null')
                    ax.plot(grid, res['real_med'], color=color, lw=2, ls=ls, label=f'{side} observed')
                    ax.fill_between(grid, res['real_lo'], res['real_hi'], color=color, alpha=0.15)
                    for pk in res['peaks']:
                        ax.axvspan(pk['start_cm'], pk['end_cm'], color=color, alpha=0.15)
                    n_place = res['n_place_cells']
                    n_sig_total += len(res['peaks'])
                title_extra = f"(n={n_place} cells, {n_sig_total} sig. peak{'s' if n_sig_total != 1 else ''})"
            else:
                res = results[arena_key][map_type]
                grid = res['grid']
                ax.plot(grid, res['null_curve'], color='0.4', ls='--', lw=1.5, label='Occupancy null')
                ax.plot(grid, res['real_med'], color='#C0392B', lw=2, label='Observed')
                ax.fill_between(grid, res['real_lo'], res['real_hi'], color='#C0392B', alpha=0.2)
                for pk in res['peaks']:
                    ax.axvspan(pk['start_cm'], pk['end_cm'], color='gold', alpha=0.35)
                    ax.axvline(pk['peak_dist_cm'], color='#B8860B', lw=1, ls=':')
                n_sig = len(res['peaks'])
                title_extra = f"(n={res['n_place_cells']} cells, {n_sig} sig. peak{'s' if n_sig != 1 else ''})"

            ax.set_title(f"{_ARENA_TITLES[arena_key]} -- {_MAP_TITLES[map_type]}\n{title_extra}", fontsize=9)
            ax.set_xlabel(_ARENA_XLABEL[arena_key], fontsize=8)
            ax.set_ylabel('% of curve total', fontsize=8)
            if row == 0 and (col == 0 or arena_key == 'circular_track'):
                ax.legend(fontsize=6, loc='upper right')

    fig.suptitle('Boundary-preference analysis: avg. firing (or % peaks) vs. position, kernel-smoothed\n'
                 '(shaded band = position range where the observed curve significantly exceeds the occupancy null)')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def export_boundary_firing_summary(results: dict, out_path: str):
    rows = []
    for arena_key, by_map in results.items():
        for map_type, res_or_sides in by_map.items():
            sides = res_or_sides if arena_key == 'circular_track' else {None: res_or_sides}
            for side, res in sides.items():
                if not res['peaks']:
                    rows.append(dict(arena=arena_key, map_type=map_type, side=side,
                                      n_place_cells=res['n_place_cells'], n_samples=res['n_samples'],
                                      peak_rank=None, start_cm=None, end_cm=None,
                                      peak_dist_cm=None, peak_pct=None, null_pct=None))
                    continue
                for i, pk in enumerate(res['peaks'], start=1):
                    rows.append(dict(arena=arena_key, map_type=map_type, side=side,
                                      n_place_cells=res['n_place_cells'], n_samples=res['n_samples'],
                                      peak_rank=i, **pk))
    df = pd.DataFrame(rows)
    df.to_excel(out_path, index=False)
    print(f'[SAVED] {out_path}')


def run_boundary_firing_analysis(arena_handlers: dict, arena_results: dict, out_dir: str,
                                  n_boot: int = N_KDE_BOOTSTRAP) -> dict:
    results = {}
    for arena_key in _ARENA_ORDER:
        handler, res = arena_handlers[arena_key], arena_results[arena_key]
        if arena_key == 'circular_track':
            results[arena_key] = {
                map_type: {
                    'inner': analyze_boundary_firing(handler, res, map_type, arena_key,
                                                      side_mask=handler.inner_side_flat, n_boot=n_boot),
                    'outer': analyze_boundary_firing(handler, res, map_type, arena_key,
                                                      side_mask=~handler.inner_side_flat, n_boot=n_boot),
                }
                for map_type in _MAP_ORDER
            }
        else:
            results[arena_key] = {
                map_type: analyze_boundary_firing(handler, res, map_type, arena_key, n_boot=n_boot)
                for map_type in _MAP_ORDER
            }

    plot_boundary_firing(results, os.path.join(out_dir, 'WallDistance_KDE.png'))
    export_boundary_firing_summary(results, os.path.join(out_dir, 'WallDistance_KDE_Peaks.xlsx'))

    for arena_key in _ARENA_ORDER:
        for map_type in _MAP_ORDER:
            sides = results[arena_key][map_type] if arena_key == 'circular_track' else {None: results[arena_key][map_type]}
            for side, res in sides.items():
                tag = f'{arena_key}/{map_type}' + (f'/{side}' if side else '')
                n_sig = len(res['peaks'])
                peak_str = ', '.join(f"{p['peak_dist_cm']:.1f} cm" for p in res['peaks'])
                print(f'[{tag}] n={res["n_place_cells"]} place cells, '
                      f'{n_sig} significant boundary peak(s)' + (f': {peak_str}' if n_sig else ''))
    return results


# ============================================================================
# Quadrant-fold KDE analysis
#
# 2D analogue of the boundary-preference analysis above (analyze_boundary_firing), applied
# directly to the small Fig 1B/D quadrant-folded grid (handler.quad_shape) instead of a 1D
# distance-from-boundary curve: each of the three folded rate maps --
#   'overall' : pool_quadrant_mean_rate      (folded overall mean field-index map, Fig 1D)
#   'field'   : pool_quadrant_field_only_rate (folded place-field-only mean field-index map)
#   'peak'    : pool_quadrant_peak_proportion (folded % of place-cell peaks, Fig 1B)
# is Gaussian-kernel-smoothed (spatial KDE) on the folded grid, bootstrapped (resampling
# place cells with replacement) to get a per-bin CI, and compared against a matching KDE of
# the folded occupancy null (pool_quadrant_occupancy) -- field-restricted to pair with
# 'field', unrestricted for 'overall'/'peak', mirroring how analyze_boundary_firing pairs
# each map type with its own null. A folded bin is flagged significant where the observed
# map's CI lower bound exceeds the null KDE; contiguous flagged bins are reported as clusters
# (same connected-component logic used for place fields, _connected_components_2d_flat).
# ============================================================================

def pool_quadrant_occupancy(handler, results: list, field_only: bool = False) -> tuple:
    """Occupancy null for the quadrant-fold KDE: pools each place cell's dwell-time map
    (_pooled_occupancy, optionally restricted to its own place-field bins to match the
    'field' map type) and folds it (Fig 1A) into the same reference quadrant as
    pool_quadrant_mean_rate / pool_quadrant_field_only_rate / pool_quadrant_peak_proportion,
    giving a shape-matched null to KDE-smooth and compare against."""
    occ_sum, valid = _pooled_occupancy(handler, results, field_only=field_only)
    return handler.fold_mean_map(np.where(valid, occ_sum, 0.0), valid)


def _kde_smooth_quadrant(handler, values_flat: np.ndarray, valid_flat: np.ndarray,
                          sigma: float = RATEMAP_SMOOTH_SIGMA_BINS) -> np.ndarray:
    """Spatial (2D) kernel density estimate over a quadrant-folded map: reshapes the flat
    folded values to the fold's own (qx, qy) grid and NaN-safe Gaussian-smooths them
    (_gaussian_smooth_2d, no wrap -- a folded quadrant is a bounded quarter-region even for
    the circular track, whose *unfolded* grid wraps but whose folded quarter-arc does not)."""
    qx, qy = handler.quad_shape
    grid = values_flat.reshape(qx, qy)
    vmask = valid_flat.reshape(qx, qy)
    return _gaussian_smooth_2d(grid, vmask, sigma=sigma, wrap_x=False).ravel()


_QUADRANT_MAP_BUILDERS = {
    'overall': pool_quadrant_mean_rate,
    'field':   pool_quadrant_field_only_rate,
    'peak':    None,  # handled separately below (pool_quadrant_peak_proportion returns (pct, total), not (vals, valid))
}


def _build_quadrant_observed_kde(handler, cells: list, map_type: str) -> np.ndarray:
    """Builds one quadrant-folded map (pool_quadrant_*), KDE-smooths it on the folded grid,
    and rescales it to % of its own total -- the 2D analogue of _build_observed_curve, run
    once on the real place-cell population and once per bootstrap resample."""
    if map_type == 'peak':
        pct, _ = pool_quadrant_peak_proportion(handler, cells)
        vals, valid = pct, np.isfinite(pct)
    else:
        vals, valid = _QUADRANT_MAP_BUILDERS[map_type](handler, cells)
    vals = np.where(valid, vals, 0.0)
    smoothed = _kde_smooth_quadrant(handler, vals, valid)
    return _normalize_pct(smoothed)


def _build_quadrant_null_kde(handler, cells: list, map_type: str) -> np.ndarray:
    """Occupancy-null counterpart of _build_quadrant_observed_kde: folds pooled dwell-time
    (field-restricted for 'field', unrestricted for 'overall'/'peak') into the same
    quadrant grid, KDE-smooths it, and rescales to % of its own total."""
    occ_vals, occ_valid = pool_quadrant_occupancy(handler, cells, field_only=(map_type == 'field'))
    occ_vals = np.where(occ_valid, occ_vals, 0.0)
    smoothed = _kde_smooth_quadrant(handler, occ_vals, occ_valid)
    return _normalize_pct(smoothed)


def analyze_quadrant_kde(handler, results: list, map_type: str,
                          n_boot: int = N_KDE_BOOTSTRAP,
                          rng: np.random.Generator | None = None) -> dict:
    """Quadrant-fold KDE analysis for one map type (see module comment above): bootstraps a
    per-bin CI for the KDE-smoothed folded map and flags folded bins where it significantly
    exceeds the KDE-smoothed occupancy null."""
    place_cells = [r for r in results if r['place_cell']]
    qx, qy = handler.quad_shape
    n_quad = handler.n_quad_bins

    out = dict(map_type=map_type, quad_shape=(qx, qy), n_place_cells=len(place_cells),
               real_med=np.full(n_quad, np.nan), real_lo=np.full(n_quad, np.nan),
               real_hi=np.full(n_quad, np.nan), null_kde=np.full(n_quad, np.nan),
               sig_mask=np.zeros(n_quad, dtype=bool), clusters=[])
    if not place_cells:
        return out

    out['null_kde'] = _build_quadrant_null_kde(handler, place_cells, map_type)

    rng = rng if rng is not None else np.random.default_rng(0)
    n = len(place_cells)
    curves = np.zeros((n_boot, n_quad))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_cells = [place_cells[i] for i in idx]
        curves[b] = _build_quadrant_observed_kde(handler, boot_cells, map_type)
    out['real_lo']  = np.percentile(curves, 100 * KDE_ALPHA / 2.0, axis=0)
    out['real_hi']  = np.percentile(curves, 100 * (1.0 - KDE_ALPHA / 2.0), axis=0)
    out['real_med'] = np.percentile(curves, 50, axis=0)

    sig = out['real_lo'] > out['null_kde']
    out['sig_mask'] = sig
    out['clusters'] = [
        dict(n_bins=len(region),
             peak_local_idx=int(region[np.argmax(out['real_med'][region])]),
             peak_pct=round(float(np.max(out['real_med'][region])), 4))
        for region in _connected_components_2d_flat(sig, qx, qy, wrap_x=False)
    ]
    return out


def plot_quadrant_kde(arena_handlers: dict, results: dict, save_path: str):
    """Plots one clean 3x3 grid (map type x arena) of the KDE-smoothed observed
    quadrant-fold map, each with a black contour marking folded bins where the bootstrap
    CI significantly exceeds the occupancy-null KDE (analyze_quadrant_kde) -- instead of a
    full second row of null-map panels per map type, a smaller strip directly beneath each
    row shows the difference (observed KDE minus null KDE) on a diverging colormap, giving
    an at-a-glance summary of where/how much the observed map departs from the null without
    doubling the figure."""
    n_map = len(_MAP_ORDER)
    fig = plt.figure(figsize=(13, 4.4 * n_map))
    gs = fig.add_gridspec(2 * n_map, 3, height_ratios=[3, 1] * n_map, hspace=0.6, wspace=0.15)

    for mi, map_type in enumerate(_MAP_ORDER):
        main_row, diff_row = 2 * mi, 2 * mi + 1

        main_vals = np.concatenate([results[k][map_type]['real_med'] for k in _ARENA_ORDER])
        cmap_main, norm_main = make_cmap_norm(main_vals)

        diffs = {k: results[k][map_type]['real_med'] - results[k][map_type]['null_kde']
                 for k in _ARENA_ORDER}
        diff_all = np.concatenate(list(diffs.values()))
        diff_abs_max = float(np.nanmax(np.abs(diff_all))) if np.isfinite(diff_all).any() else 1.0
        diff_abs_max = max(diff_abs_max, 1e-6)
        cmap_diff = _get_cmap('RdBu_r')
        cmap_diff.set_bad('white')
        norm_diff = Normalize(vmin=-diff_abs_max, vmax=diff_abs_max)

        main_axes, diff_axes = [], []
        for ci, arena_key in enumerate(_ARENA_ORDER):
            handler = arena_handlers[arena_key]
            res = results[arena_key][map_type]
            proj = 'polar' if arena_key == 'circular_track' else None

            ax_main = fig.add_subplot(gs[main_row, ci], projection=proj)
            main_valid = np.isfinite(res['real_med'])
            handler.plot_quadrant_map(ax_main, np.nan_to_num(res['real_med']), main_valid, cmap_main, norm_main)
            handler.overlay_quadrant_significance(ax_main, res['sig_mask'])
            n_sig = len(res['clusters'])
            ax_main.set_title(f"{_ARENA_TITLES[arena_key]} -- {_MAP_TITLES[map_type]}\n"
                               f"n={res['n_place_cells']} cells, "
                               f"{n_sig} sig. cluster{'s' if n_sig != 1 else ''}", fontsize=8)
            main_axes.append(ax_main)

            ax_diff = fig.add_subplot(gs[diff_row, ci], projection=proj)
            diff_valid = np.isfinite(res['real_med']) & np.isfinite(res['null_kde'])
            handler.plot_quadrant_map(ax_diff, np.nan_to_num(diffs[arena_key]), diff_valid, cmap_diff, norm_diff)
            ax_diff.set_title('observed - null', fontsize=7)
            diff_axes.append(ax_diff)

        sm_main = cm.ScalarMappable(cmap=cmap_main, norm=norm_main); sm_main.set_array([])
        fig.colorbar(sm_main, ax=main_axes, shrink=0.7, pad=0.02, label='% of curve total')
        sm_diff = cm.ScalarMappable(cmap=cmap_diff, norm=norm_diff); sm_diff.set_array([])
        fig.colorbar(sm_diff, ax=diff_axes, shrink=0.6, pad=0.02, label='obs - null (pct pts)')

    fig.suptitle('Quadrant-fold KDE: Gaussian-smoothed folded rate maps (black contour = folded bin where the\n'
                 "bootstrap CI significantly exceeds the occupancy-null KDE); smaller row below each = observed minus null")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def export_quadrant_kde_summary(results: dict, out_path: str):
    rows = []
    for arena_key, by_map in results.items():
        for map_type, res in by_map.items():
            qx, qy = res['quad_shape']
            if not res['clusters']:
                rows.append(dict(arena=arena_key, map_type=map_type,
                                  n_place_cells=res['n_place_cells'],
                                  cluster_rank=None, n_bins=None,
                                  peak_row=None, peak_col=None, peak_pct=None))
                continue
            for i, cl in enumerate(res['clusters'], start=1):
                peak_row, peak_col = divmod(cl['peak_local_idx'], qy)
                rows.append(dict(arena=arena_key, map_type=map_type,
                                  n_place_cells=res['n_place_cells'],
                                  cluster_rank=i, n_bins=cl['n_bins'],
                                  peak_row=peak_row, peak_col=peak_col,
                                  peak_pct=cl['peak_pct']))
    df = pd.DataFrame(rows)
    df.to_excel(out_path, index=False)
    print(f'[SAVED] {out_path}')


def run_quadrant_kde_analysis(arena_handlers: dict, arena_results: dict, out_dir: str,
                               n_boot: int = N_KDE_BOOTSTRAP) -> dict:
    results = {}
    for arena_key in _ARENA_ORDER:
        handler, res = arena_handlers[arena_key], arena_results[arena_key]
        results[arena_key] = {
            map_type: analyze_quadrant_kde(handler, res, map_type, n_boot=n_boot)
            for map_type in _MAP_ORDER
        }

    plot_quadrant_kde(arena_handlers, results, os.path.join(out_dir, 'QuadrantFold_KDE.png'))
    export_quadrant_kde_summary(results, os.path.join(out_dir, 'QuadrantFold_KDE_Clusters.xlsx'))

    for arena_key in _ARENA_ORDER:
        for map_type in _MAP_ORDER:
            res = results[arena_key][map_type]
            n_sig = len(res['clusters'])
            print(f'[{arena_key}/{map_type} quadrant KDE] n={res["n_place_cells"]} place cells, '
                  f'{n_sig} significant cluster(s) vs. occupancy null')
    return results


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
        fig.colorbar(pcm, ax=ax, shrink=0.7, label='Field index (a.u.)')

    fig.suptitle('Fig S1H -- Overall mean field index maps (place cells pooled across days)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_field_only_mean_maps(arena_handlers: dict, arena_results: dict, save_path: str):
    """Additional mean field-index maps built ONLY from each place cell's extracted
    place-field bins (pool_field_only_map), occupancy-weighted to avoid bias from bins
    some cells oversampled relative to others. Normalized to each cell's own peak (see
    pool_field_only_map), so values are in the 0-1 field index, not raw Hz."""
    fig = plt.figure(figsize=(15, 5))
    for i, key in enumerate(_ARENA_ORDER):
        handler = arena_handlers[key]
        mean_map, valid = pool_field_only_map(handler, arena_results[key])
        cmap, norm = make_cmap_norm(mean_map[valid])

        proj = 'polar' if key == 'circular_track' else None
        ax = fig.add_subplot(1, 3, i + 1, projection=proj)
        pcm = handler.plot_fine(ax, mean_map, valid, cmap, norm)

        n_place = sum(1 for r in arena_results[key] if r['place_cell'])
        ax.set_title(f'{_ARENA_TITLES[key]}\n(n={n_place} place cells)')
        fig.colorbar(pcm, ax=ax, shrink=0.7, label='Mean field index (a.u.)')

    fig.suptitle('Field-only mean field-index maps (occupancy-weighted, place-field bins only)')
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
        ax_d.set_title(f'{_ARENA_TITLES[key]}\nMean field index (a.u.)')
        axes_d.append(ax_d)

    sm_b = cm.ScalarMappable(cmap=cmap_b, norm=norm_b); sm_b.set_array([])
    fig.colorbar(sm_b, ax=axes_b, shrink=0.6, label='% of peaks')
    sm_d = cm.ScalarMappable(cmap=cmap_d, norm=norm_d); sm_d.set_array([])
    fig.colorbar(sm_d, ax=axes_d, shrink=0.6, label='Mean field index (a.u.)')

    fig.suptitle('Figure 1B/D -- Quadrant mean maps: folded peak-location proportion (top) and mean field index (bottom)')
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def _debug_occupancy_and_rawrate(x_cm: np.ndarray, y_cm: np.ndarray, t: np.ndarray,
                                  spike_ts: np.ndarray, handler) -> dict | None:
    """Same bin-assignment + spike-matching arithmetic as compute_cell_ratemap, but with
    NO min_occ_s / geom_valid masking -- a bin counts as 'visited' the moment occ_map > 0.
    This is deliberately more permissive than the real pipeline so raw bin-assignment bugs
    (gaps, duplicated wrap, off-center ring, wrong radial split) show up even in
    sparsely-sampled bins that compute_cell_ratemap would mask out as invalid."""
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

    n = len(t)
    dt_frames = np.empty(n, dtype=np.float64)
    dt_frames[0] = 1.0 / fps
    dt_frames[1:] = np.minimum(np.diff(t) * 1e-6, 2.0 / fps)

    n_bins = handler.n_bins
    occ_map   = np.zeros(n_bins, dtype=np.float64)
    spike_map = np.zeros(n_bins, dtype=np.float64)
    np.add.at(occ_map,   bin_idx, dt_frames)
    np.add.at(spike_map, bin_idx[spike_frame], 1.0)

    visited = occ_map > 0
    fr_raw = np.zeros(n_bins, dtype=np.float64)
    fr_raw[visited] = spike_map[visited] / occ_map[visited]
    return dict(occ_map=occ_map, spike_map=spike_map, fr_raw=fr_raw, visited=visited,
                n_spikes=int(valid_spike.sum()))


def debug_plot_circular_track_raw(cfg: dict, out_dir: str) -> None:
    """Diagnostic for the circular-track bin-assignment bug: for every session, plots the
    RAW occupancy map (no min_occ_s threshold), and for every unit in it the RAW
    (unsmoothed) firing-rate map, on the handler's actual (arc-length x radial-width)
    polar grid -- bypassing the 80% coverage criterion, min_occ_s masking, and
    place-cell qualification entirely, so every unit gets a saved plot regardless of
    whether the real pipeline would keep it."""
    handler = CircularTrackHandler(cfg, target_bin_cm)
    root = ROOT_DIRECTORY
    os.makedirs(out_dir, exist_ok=True)

    print(f'[debug] circular_track grid: nx={handler.nx} (arc bins), ny={handler.ny} (radial bins), '
          f'n_bins={handler.n_bins}, inner_r={handler.inner_r:.2f}, outer_r={handler.outer_r:.2f}, '
          f'bin_width_deg={handler.bin_width_deg:.3f}, bin_cm_y={handler.bin_cm_y:.3f}')

    for dirpath, _, filenames in os.walk(root):
        if _detect_arena_key(dirpath) != 'circular_track':
            continue
        tracking_files_all = [f for f in filenames if f.lower().endswith(('.csv', '.xlsx'))]
        if COORD_UNITS == 'cm':
            tracking_files = [f for f in tracking_files_all if f.lower().endswith('_cm.csv')]
        else:
            tracking_files = [f for f in tracking_files_all if not f.lower().endswith('_cm.csv')]
        ntt_files = [f for f in filenames if f.lower().endswith('.ntt')]
        if len(tracking_files) != 1 or not ntt_files:
            continue

        session_name = os.path.relpath(dirpath, root)
        safe_name = session_name.replace(os.sep, '_').replace('/', '_')
        csv_path = os.path.join(dirpath, tracking_files[0])
        try:
            x_cm, y_cm, t = _load_tracking(csv_path, handler.arena_width_cm)
        except Exception as e:
            print(f'  ERROR loading tracking [{session_name}]: {e}')
            continue
        if len(t) < 2:
            print(f'  [SKIP no tracking] {session_name}')
            continue
        x_cm, y_cm = _smooth_tracking_position(x_cm, y_cm, t)

        bin_idx_all, on_track_all = handler.to_bins(x_cm, y_cm)
        dt_all = np.empty(len(t), dtype=np.float64)
        dt_all[0] = 1.0 / fps
        dt_all[1:] = np.minimum(np.diff(t) * 1e-6, 2.0 / fps)
        occ_all = np.zeros(handler.n_bins, dtype=np.float64)
        np.add.at(occ_all, bin_idx_all[on_track_all], dt_all[on_track_all])
        visited_all = occ_all > 0
        pct_on_track = float(on_track_all.mean() * 100.0)
        pct_bins_visited = float(visited_all.sum()) / handler.n_bins * 100.0
        print(f'  [{session_name}] {len(t)} tracking samples, {pct_on_track:.1f}% on_track, '
              f'{int(visited_all.sum())}/{handler.n_bins} bins visited ({pct_bins_visited:.1f}%)')

        if visited_all.any():
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(1, 1, 1, projection='polar')
            cmap, norm = make_cmap_norm(occ_all[visited_all])
            pcm = handler.plot_fine(ax, occ_all, visited_all, cmap, norm)
            ax.set_title(f'{session_name}\nRaw occupancy (s) -- {pct_on_track:.0f}% samples on-track, '
                         f'{pct_bins_visited:.0f}% bins visited')
            fig.colorbar(pcm, ax=ax, shrink=0.7, label='Occupancy (s)')
            fig.tight_layout()
            occ_path = os.path.join(out_dir, f'Occupancy_{safe_name}.png')
            fig.savefig(occ_path, dpi=150)
            plt.close(fig)
            print(f'    [SAVED] {occ_path}')
        else:
            print(f'    [SKIP plot] {session_name}: no bins visited at all')

        for ntt_file in sorted(ntt_files):
            ntt_path = os.path.join(dirpath, ntt_file)
            try:
                spike_data = np.memmap(ntt_path, dtype=ntt_dtype, mode='r', offset=16 * 1024)
                spike_data = spike_data[spike_data['cell_number'] != 0]
                spike_ts = np.sort(spike_data['timestamp'].astype(np.float64))
            except Exception as e:
                print(f'    ERROR reading {ntt_file}: {e}')
                continue
            if len(spike_ts) == 0:
                continue

            debug_cell = _debug_occupancy_and_rawrate(x_cm, y_cm, t, spike_ts, handler)
            if debug_cell is None or not debug_cell['visited'].any():
                print(f'    [SKIP plot] {session_name}/{ntt_file}: no visited bins')
                continue

            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(1, 1, 1, projection='polar')
            visited = debug_cell['visited']
            cmap, norm = make_cmap_norm(debug_cell['fr_raw'][visited])
            pcm = handler.plot_fine(ax, debug_cell['fr_raw'], visited, cmap, norm)
            ax.set_title(f'{session_name}/{ntt_file}\nRaw rate map (Hz), n_spikes={debug_cell["n_spikes"]}')
            fig.colorbar(pcm, ax=ax, shrink=0.7, label='Raw rate (Hz)')
            fig.tight_layout()
            unit_safe = os.path.splitext(ntt_file)[0]
            rm_path = os.path.join(out_dir, f'RawRM_{safe_name}_{unit_safe}.png')
            fig.savefig(rm_path, dpi=150)
            plt.close(fig)
            print(f'    [SAVED] {rm_path}')


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


def run_circular_track_pipeline_test(cfg: dict, out_dir: str) -> None:
    """DEBUG: runs the FULL downstream pipeline (bootstrap, coverage filtering at
    whatever COVERAGE_FRACTION is currently set to, place-cell qualification, pooling,
    quadrant fold, plotting, excel export) for the circular_track arena ALONE -- to check
    that the rest of the analysis actually completes on this data once the coverage
    threshold is lowered (e.g. sessions like 2NoRot at ~78% bin coverage, excluded by the
    original 80% threshold), without waiting on the other two arenas."""
    os.makedirs(out_dir, exist_ok=True)
    handler, results = collect_arena_results('circular_track', cfg)

    n_processed = len(results)
    n_place = sum(1 for r in results if r['place_cell'])
    print(f'[circular_track pipeline test] COVERAGE_FRACTION={COVERAGE_FRACTION:.0%}, '
          f'{n_processed} units processed, {n_place} place cells.')
    if n_processed == 0:
        print('  No units passed the coverage criterion -- nothing to plot.')
        return

    # Fig S1H equivalent: overall mean field-index map (place cells only)
    mean_map, valid = pool_fine_map(handler, results)
    if valid.any():
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1, projection='polar')
        cmap, norm = make_cmap_norm(mean_map[valid])
        pcm = handler.plot_fine(ax, mean_map, valid, cmap, norm)
        ax.set_title(f'Circular Track -- mean field index (n={n_place} place cells)')
        fig.colorbar(pcm, ax=ax, shrink=0.7, label='Field index (a.u.)')
        fig.tight_layout()
        p = os.path.join(out_dir, 'CircularTrack_MeanFieldIndex.png')
        fig.savefig(p, dpi=200)
        plt.close(fig)
        print(f'  [SAVED] {p}')
    else:
        print('  [SKIP] no valid bins in pooled fine map')

    # Field-only mean rate map
    mean_map_fo, valid_fo = pool_field_only_map(handler, results)
    if valid_fo.any():
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1, projection='polar')
        cmap, norm = make_cmap_norm(mean_map_fo[valid_fo])
        pcm = handler.plot_fine(ax, mean_map_fo, valid_fo, cmap, norm)
        ax.set_title(f'Circular Track -- field-only mean rate (n={n_place} place cells)')
        fig.colorbar(pcm, ax=ax, shrink=0.7, label='Mean firing rate (Hz)')
        fig.tight_layout()
        p = os.path.join(out_dir, 'CircularTrack_FieldOnlyMeanRate.png')
        fig.savefig(p, dpi=200)
        plt.close(fig)
        print(f'  [SAVED] {p}')
    else:
        print('  [SKIP] no valid bins in field-only map')

    # Quadrant fold (Fig 1B/D): peak proportion + mean field index
    pct, total = pool_quadrant_peak_proportion(handler, results)
    rate_vals, rate_valid = pool_quadrant_mean_rate(handler, results)
    fig = plt.figure(figsize=(10, 5))
    pct_valid = np.isfinite(pct)
    ax1 = fig.add_subplot(1, 2, 1, projection='polar')
    cmap1, norm1 = make_cmap_norm(pct[pct_valid])
    handler.plot_quadrant_map(ax1, np.nan_to_num(pct), pct_valid, cmap1, norm1)
    ax1.set_title(f'Peak proportion (%), n={total}')
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    cmap2, norm2 = make_cmap_norm(rate_vals[rate_valid])
    handler.plot_quadrant_map(ax2, np.nan_to_num(rate_vals), rate_valid, cmap2, norm2)
    ax2.set_title('Mean field index (a.u.)')
    fig.suptitle('Circular Track -- quadrant fold (Fig 1B/D)')
    p = os.path.join(out_dir, 'CircularTrack_QuadrantFold.png')
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f'  [SAVED] {p}')

    export_excel({'circular_track': handler}, {'circular_track': results},
                 os.path.join(out_dir, 'CircularTrack_Summary.xlsx'))


# ============================================================================
# __main__
# ============================================================================

def run_full_pipeline(out_dir: str) -> None:
    """Runs the full pipeline (bootstrap, coverage filtering at COVERAGE_FRACTION,
    place-cell qualification, pooling, quadrant fold, plotting, excel export) for all
    three arenas (open_field, circular_track, linear_track), producing the combined
    Fig S1H / field-only / Fig 1B-D figures side by side across arenas."""
    os.makedirs(out_dir, exist_ok=True)

    arena_handlers, arena_results = {}, {}
    for key in _ARENA_ORDER:
        handler, results = collect_arena_results(key, ARENA_CONFIGS[key])
        arena_handlers[key] = handler
        arena_results[key] = results

    plot_fig_S1H(arena_handlers, arena_results,
                 os.path.join(out_dir, 'FigS1H_MeanFieldIndex.png'))
    plot_field_only_mean_maps(arena_handlers, arena_results,
                               os.path.join(out_dir, 'FieldOnly_MeanFieldIndex.png'))
    plot_fig_1BD(arena_handlers, arena_results,
                 os.path.join(out_dir, 'Fig1BD_QuadrantFold.png'))
    export_excel(arena_handlers, arena_results,
                 os.path.join(out_dir, 'AllArenas_Summary.xlsx'))
    run_boundary_firing_analysis(arena_handlers, arena_results, out_dir)
    run_quadrant_kde_analysis(arena_handlers, arena_results, out_dir)


if __name__ == '__main__':
    _coord_answer = input("Are the tracking coordinates in pixels or cm? [pixel/cm]: ").strip().lower()
    while _coord_answer not in _PIXEL_ANSWERS | _CM_ANSWERS:
        _coord_answer = input("Please enter 'pixel' or 'cm': ").strip().lower()
    COORD_UNITS = 'pixel' if _coord_answer in _PIXEL_ANSWERS else 'cm'
    print(f"Using '{COORD_UNITS}' tracking coordinates.\n")

    run_full_pipeline(os.path.join(OUTPUT_DIR, 'AllArenas'))

    print('\nDone.')
