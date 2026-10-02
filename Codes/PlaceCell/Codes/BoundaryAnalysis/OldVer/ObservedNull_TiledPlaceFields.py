# -*- coding: utf-8 -*-
"""
Observed null for the pooled mean maps (overall mean field-index map, field-only mean field-index
map, peak proportion map) and their 2D KDEs.

Question: if the animals had travelled exactly the trajectories they did travel (tracking files),
but every place cell had a perfect place field, the fields tiling the arena uniformly with no two
fields sharing a bin, what would the three pooled mean maps -- and their KDEs -- look like?

Step 1  Trajectory / occupancy (ported unchanged from MeanOccupancyMap_Tracking_v2.py):
        every *_cm.csv under ROOT_DIRECTORY is loaded, cleaned, smoothed, open-field centred and
        Rotate-corrected, oriented and binned (2 x 2 cm). All sessions of an arena are chained
        file after file into ONE continuous trajectory (TRAJ_FILE_GAP_S between files); the
        occupancy map is that concatenated trajectory's dwell time (s) per bin.

Step 2  Perfect place fields (tile_arena): the arena's bins are partitioned into non-overlapping,
        contiguous fields of FIELD_BINS (3 x 3 = 9) bins, enlarged by 1..MAX_EXTRA_BINS (3) bins
        only where the geometry requires it:
          - a fully valid rectangular bin grid (linear track 40 x 4, circular track 120 x 2,
            arc-length x width, wrapping) is tiled exactly by the smallest, then squarest,
            w x h block of 9..12 bins (both sides >= 2 bins) that divides the grid. A 3 x 3 block
            cannot fit a 4- or 2-bin-wide track, so both tracks get 5 x 2 = 10-bin fields.
          - the open field (30 x 30 grid masked to the 60 cm disc) keeps a 3 x 3 lattice; the
            bins of the blocks cut by the wall are shared out by a capacity-constrained
            nearest-centre assignment, so every field holds 9..12 bins and every disc bin
            belongs to exactly one field.
        One simulated cell per field. Its firing rate is FIELD_PEAK_RATE (1) at the field's peak
        bin (the field bin nearest the field centroid; exact ties alternate between fields so
        no side of the arena is favoured) and falls linearly with distance d from it,
            rate = 1 - d / (d_max + 1),
        so it is still > 0 in the field's outermost bin and reaches 0 one bin beyond it. Every
        bin outside the field is 0.

Step 3  Every cell 'fires' along the concatenated trajectory at the rate its field predicts for
        the bin the animal occupies on each frame (rate * dt expected spikes per frame).
        Rate map = spikes / occupancy per bin; then the same per-cell steps as
        MeanRateMap_QuadrantAnalysis_v23.py: bins with < min_occ_s occupancy are invalid,
        Gaussian smoothing (sigma RATEMAP_SMOOTH_SIGMA_BINS, wraps on the circular track;
        SMOOTH_SIM_RATEMAPS = False skips it), 0-1 field-index normalisation, peak bin taken
        from the raw map. Field-only bins = bins at >= FIELD_PEAK_FRAC (20 %) of the cell's
        peak rate. The cells are pooled with v23's pooling rules into the three observed-null
        maps.

Step 4  v23's KDE (scipy gaussian_kde, Scott's rule, edge-corrected) applied to each null map
        over the bins the trajectory validly sampled. The KDE NPZs use the same keys as v23's,
        so real and null kernels can be paired.

Outputs (in OUTPUT_DIR):
  ObservedNull_FieldTiling.png              -- the tiled fields of every arena (outlines, peak
                                               bins) and the perfect rate map they define
  ObservedNull_Occupancy_Trajectory.png     -- concatenated occupancy and trajectory per arena
  Null_MeanMaps_AllTypes.png                -- the three observed-null maps (rows) x arenas
  Null_KDE_FigS1H_MeanFieldIndex.png / .npz    -- KDE of the overall mean null map
  Null_KDE_FieldOnly_MeanFieldIndex.png / .npz -- KDE of the field-only mean null map
  Null_KDE_PeakProportion_Map.png / .npz       -- KDE of the peak proportion null map
  (all maps except the KDEs are drawn in true arena geometry, the circular track as a ring)
  ObservedNull_Maps.npz                     -- null maps, domains, field labels, tuning, occupancy
  ObservedNull_Summary.xlsx                 -- per-arena tiling summary + per-session tracking
"""

import os
import threading
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label
from scipy.optimize import linear_sum_assignment
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT_DIRECTORY = r'X:\NMR_group_data\Runita\Analysis\Mean_KDE_Open_PascalOldenburg\Open_KDE\CorrectedData\Data\SessionTypeSorted_PC\Open\Cntrl'
# Same folder as MeanRateMap_QuadrantAnalysis_v23.py's figures, so real and null maps sit together
# (every file written here is prefixed Null_ / ObservedNull_, so nothing of v23's is overwritten)
OUTPUT_DIR = os.path.join(ROOT_DIRECTORY, 'MeanRM_Quad_v23', 'AllArenas')

ARENA_FOLDER_KEYWORDS = {
    'open':     'open_field',
    'linear':   'linear_track',
    'circle':   'circular_track',
    'circular': 'circular_track',
}

ARENA_CONFIGS = {
    'open_field':     dict(shape='circle', diameter_cm=60.0),
    'circular_track': dict(shape='ring', outer_diameter_cm=80.0, inner_diameter_cm=72.0),
    'linear_track':   dict(shape='linear', length_cm=80.0, width_cm=8.0),
}

fps           = 30      # tracking frame rate (Hz)
target_bin_cm = 2.0     # spatial bin size (cm)

# --- Step 1: tracking (same values as MeanOccupancyMap_Tracking_v2.py) ---
POS_JUMP_THRESH_CMS  = 90.0   # frame-to-frame jumps faster than this (cm/s) are tracking artifacts
POS_SMOOTH_SIGMA_SMP = 1.0    # Gaussian smoothing sigma (samples) of x/y tracking
TRAJ_FILE_GAP_S      = 1.0 / fps   # gap between the end of one file and the start of the next
TRAJ_LINE_KW = dict(color='0.55', lw=0.25, alpha=0.35, solid_joinstyle='round', rasterized=True)

CENTRE_OPEN_FIELD_TRACKING = True
ROTATE_SESSION_KEYWORD     = 'rotate'
ROTATE_SESSION_CCW_DEG     = 0.0

# --- Step 2: perfect place fields ---
FIELD_SIDE_BINS = 5       # nominal field: FIELD_SIDE_BINS x FIELD_SIDE_BINS bins
FIELD_BINS      = FIELD_SIDE_BINS * FIELD_SIDE_BINS
MAX_EXTRA_BINS  = 3       # a field may grow to FIELD_BINS + MAX_EXTRA_BINS bins to fit the geometry
FIELD_PEAK_RATE = 1.0     # rate at the field's peak bin

# --- Step 3: per-cell rate maps (same values as MeanRateMap_QuadrantAnalysis_v23.py) ---
min_occ_s                 = 1      # bins with less occupancy (s) are invalid
SMOOTH_SIM_RATEMAPS       = True
RATEMAP_SMOOTH_SIGMA_BINS = 3.0
FIELD_PEAK_FRAC           = 0.20   # field-only bins: rate >= this fraction of the cell's peak rate

# --- Step 4: KDE (same values as v23) ---
KDE_BW_METHOD       = 'scott'
KDE_EDGE_CORRECTION = True

_ARENA_ORDER  = ['open_field', 'circular_track', 'linear_track']
_ARENA_TITLES = {'open_field': 'Open Field', 'circular_track': 'Circular Track',
                 'linear_track': 'Linear Track'}


def _detect_arena_key(dirpath: str):
    for part in os.path.normpath(dirpath).split(os.sep):
        arena_key = ARENA_FOLDER_KEYWORDS.get(part.strip().lower())
        if arena_key is not None:
            return arena_key
    return None


# ============================================================================
# Step 1 -- tracking load / clean / smooth / frame corrections
# (ported unchanged from MeanOccupancyMap_Tracking_v2.py)
# ============================================================================

def _load_tracking(csv_path: str) -> tuple:
    """Load and clean one cm tracking file. Returns (x_cm, y_cm, t) with t in us."""
    data = (pd.read_excel(csv_path) if csv_path.lower().endswith('.xlsx')
            else pd.read_csv(csv_path))
    t = np.asarray(data.iloc[:, 0], dtype=float)
    x = np.asarray(data.iloc[:, 3], dtype=float)
    y = np.asarray(data.iloc[:, 4], dtype=float)

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
        return x, y, t
    return x - x.min(), y - y.min(), t


def _smooth_tracking_position(x_cm, y_cm, t_us) -> tuple:
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
    return (gaussian_filter1d(x_clean, sigma=POS_SMOOTH_SIGMA_SMP, mode='nearest'),
            gaussian_filter1d(y_clean, sigma=POS_SMOOTH_SIGMA_SMP, mode='nearest'))


def _estimate_disc_centre(x_cm, y_cm, radius_cm, n_sectors=72, wall_frac=0.90,
                          min_sectors=12, max_shift_frac=0.25) -> tuple:
    """Centre of a circular arena of known radius, fitted to the tracking's outer envelope.
    Returns (cx, cy, ok)."""
    cx0 = 0.5 * (float(x_cm.min()) + float(x_cm.max()))
    cy0 = 0.5 * (float(y_cm.min()) + float(y_cm.max()))
    if len(x_cm) < min_sectors:
        return cx0, cy0, False

    theta = np.arctan2(y_cm - cy0, x_cm - cx0)
    r = np.hypot(x_cm - cx0, y_cm - cy0)
    sector = np.clip(((theta + np.pi) / (2 * np.pi) * n_sectors).astype(int), 0, n_sectors - 1)
    order = np.lexsort((r, sector))
    last = np.flatnonzero(np.append(np.diff(sector[order]) != 0, True))
    env = order[last]
    env = env[r[env] >= wall_frac * radius_cm]
    if len(env) < min_sectors:
        return cx0, cy0, False

    ang = np.sort(theta[env])
    gaps = np.append(np.diff(ang), ang[0] + 2 * np.pi - ang[-1])
    if float(gaps.max()) > np.pi:
        return cx0, cy0, False

    px, py = x_cm[env].astype(np.float64), y_cm[env].astype(np.float64)
    cx, cy = cx0, cy0
    for _ in range(100):
        dx, dy = px - cx, py - cy
        rr = np.hypot(dx, dy)
        keep = rr > 1e-9
        if keep.sum() < min_sectors:
            return cx0, cy0, False
        cx_new = float(px[keep].mean() - radius_cm * (dx[keep] / rr[keep]).mean())
        cy_new = float(py[keep].mean() - radius_cm * (dy[keep] / rr[keep]).mean())
        converged = abs(cx_new - cx) < 1e-7 and abs(cy_new - cy) < 1e-7
        cx, cy = cx_new, cy_new
        if converged:
            break

    if not (np.isfinite(cx) and np.isfinite(cy)):
        return cx0, cy0, False
    if np.hypot(cx - cx0, cy - cy0) > max_shift_frac * radius_cm:
        return cx0, cy0, False
    return cx, cy, True


_CENTRE_WARN_LOCK = threading.Lock()


def _centre_open_field_tracking(x_cm, y_cm, session_dir, handler) -> tuple:
    if not CENTRE_OPEN_FIELD_TRACKING or _detect_arena_key(session_dir) != 'open_field' or len(x_cm) == 0:
        return x_cm, y_cm
    cx, cy, ok = _estimate_disc_centre(x_cm, y_cm, handler.diameter / 2.0)
    if not ok:
        with _CENTRE_WARN_LOCK:
            print(f'  [centre fit failed] {session_dir}: frame left as loaded')
        return x_cm, y_cm
    return x_cm + (handler.cx - cx), y_cm + (handler.cy - cy)


def _apply_session_rotation(x_cm, y_cm, session_dir, handler) -> tuple:
    if (_detect_arena_key(session_dir) != 'open_field'
            or ROTATE_SESSION_KEYWORD not in os.path.basename(os.path.normpath(session_dir)).lower()
            or len(x_cm) == 0):
        return x_cm, y_cm
    th = np.radians(ROTATE_SESSION_CCW_DEG)
    dx, dy = x_cm - handler.cx, y_cm - handler.cy
    return (handler.cx + dx * np.cos(th) - dy * np.sin(th),
            handler.cy + dx * np.sin(th) + dy * np.cos(th))


def _session_positions(csv_path: str, handler) -> tuple:
    """Load -> clean -> smooth -> open-field centring -> Rotate fix."""
    x_cm, y_cm, t = _load_tracking(csv_path)
    if len(t) < 2:
        return x_cm, y_cm, t
    x_cm, y_cm = _smooth_tracking_position(x_cm, y_cm, t)
    session_dir = os.path.dirname(csv_path)
    x_cm, y_cm = _centre_open_field_tracking(x_cm, y_cm, session_dir, handler)
    x_cm, y_cm = _apply_session_rotation(x_cm, y_cm, session_dir, handler)
    return x_cm, y_cm, t


# ============================================================================
# Smoothing + arena geometry handlers (binning from MeanOccupancyMap_Tracking_v2.py, KDE
# geometry / smoothing / plotting from MeanRateMap_QuadrantAnalysis_v23.py -- same bin grids)
# ============================================================================

def _gaussian_smooth_2d(fr_map, valid_mask, sigma, wrap_x=False):
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


class OpenFieldHandler:
    wrap_x = False

    def __init__(self, cfg, bin_cm):
        self.diameter = cfg['diameter_cm']
        self.bin_cm = bin_cm
        self.nx = self.ny = int(np.ceil(self.diameter / bin_cm))
        self.n_bins = self.nx * self.ny
        self.cx = self.cy = self.diameter / 2.0
        c = (np.arange(self.nx) + 0.5) * bin_cm
        XX, YY = np.meshgrid(c, c, indexing='ij')
        self.geom_valid = (np.hypot(XX - self.cx, YY - self.cy) <= self.diameter / 2.0).ravel()

    def orient(self, x_cm, y_cm):
        return x_cm, y_cm

    def to_bins(self, x_cm, y_cm):
        bx = np.clip((x_cm / self.bin_cm).astype(int), 0, self.nx - 1)
        by = np.clip((y_cm / self.bin_cm).astype(int), 0, self.ny - 1)
        sample_valid = np.hypot(x_cm - self.cx, y_cm - self.cy) <= (self.diameter / 2.0 + self.bin_cm)
        return bx * self.ny + by, sample_valid

    def bin_centres_xy(self):
        xs = (np.arange(self.nx) + 0.5) * self.bin_cm
        ys = (np.arange(self.ny) + 0.5) * self.bin_cm
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        return np.vstack([XX.ravel(), YY.ravel()])

    def edge_to_xy(self, u, v):
        """Bin-grid coordinates (bin (i, j) spans u in [i, i+1], v in [j, j+1]) -> arena cm."""
        return np.asarray(u) * self.bin_cm, np.asarray(v) * self.bin_cm

    def draw_outline(self, ax):
        ax.add_patch(matplotlib.patches.Circle((self.cx, self.cy), self.diameter / 2.0,
                                               fill=False, edgecolor='0.2', lw=1.0, zorder=5))

    def bin_areas_cm2(self):
        return np.full(self.n_bins, self.bin_cm ** 2)

    def smooth(self, fr_flat, valid_flat):
        return _gaussian_smooth_2d(fr_flat.reshape(self.nx, self.ny), valid_flat.reshape(self.nx, self.ny),
                                   RATEMAP_SMOOTH_SIGMA_BINS).ravel()

    def plot_trajectory(self, ax, x_cm, y_cm):
        ax.plot(x_cm, y_cm, **TRAJ_LINE_KW)
        ax.add_patch(matplotlib.patches.Circle((self.cx, self.cy), self.diameter / 2.0,
                                               fill=False, edgecolor='0.35', lw=1.0, zorder=5))
        ax.set_xlim(0, self.diameter)
        ax.set_ylim(0, self.diameter)
        ax.set_aspect('equal')
        ax.axis('off')

    def plot_bins_2d(self, ax, values_flat, valid_flat, cmap, norm):
        grid = np.where(valid_flat & self.geom_valid, values_flat, np.nan).reshape(self.nx, self.ny)
        x_edges = np.arange(self.nx + 1) * self.bin_cm
        y_edges = np.arange(self.ny + 1) * self.bin_cm
        pcm = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(grid.T),
                            cmap=cmap, norm=norm, shading='flat')
        ax.add_patch(matplotlib.patches.Circle((self.cx, self.cy), self.diameter / 2.0,
                                               fill=False, edgecolor='0.35', lw=1.0, zorder=5))
        ax.set_aspect('equal')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        return pcm


class CircularTrackHandler:
    wrap_x = True

    def __init__(self, cfg, bin_cm):
        self.outer_r = cfg['outer_diameter_cm'] / 2.0
        self.inner_r = cfg['inner_diameter_cm'] / 2.0
        self.cx = self.cy = self.outer_r
        self.radial_tol_cm = 4.0
        self.track_width_cm = self.outer_r - self.inner_r
        self.circumference_cm = 2 * np.pi * (self.outer_r + self.inner_r) / 2.0
        self.nx = max(8, 4 * int(round(self.circumference_cm / bin_cm / 4.0)))
        self.ny = max(2, int(round(self.track_width_cm / bin_cm)))
        self.bin_width_deg = 360.0 / self.nx
        self.bin_cm_y = self.track_width_cm / self.ny
        self.n_bins = self.nx * self.ny
        self.geom_valid = np.ones(self.n_bins, dtype=bool)

    def orient(self, x_cm, y_cm):
        return x_cm, y_cm

    def to_bins(self, x_cm, y_cm):
        r = np.hypot(x_cm - self.cx, y_cm - self.cy)
        theta = np.degrees(np.arctan2(y_cm - self.cy, x_cm - self.cx)) % 360.0
        bx = np.clip((theta / self.bin_width_deg).astype(int), 0, self.nx - 1)
        rel_r = np.clip(r - self.inner_r, 0.0, self.track_width_cm)
        by = np.clip((rel_r / self.bin_cm_y).astype(int), 0, self.ny - 1)
        on_track = (r >= self.inner_r - self.radial_tol_cm) & (r <= self.outer_r + self.radial_tol_cm)
        return bx * self.ny + by, on_track

    def bin_centres_xy(self):
        theta = np.radians((np.arange(self.nx) + 0.5) * self.bin_width_deg)
        r = self.inner_r + (np.arange(self.ny) + 0.5) * self.bin_cm_y
        TH, R = np.meshgrid(theta, r, indexing='ij')
        return np.vstack([(self.cx + R * np.cos(TH)).ravel(),
                          (self.cy + R * np.sin(TH)).ravel()])

    def edge_to_xy(self, u, v):
        """Bin-grid coordinates (u = angular bin, v = radial bin) -> room cm on the actual ring."""
        th = np.radians(np.asarray(u) * self.bin_width_deg)
        r = self.inner_r + np.asarray(v) * self.bin_cm_y
        return self.cx + r * np.cos(th), self.cy + r * np.sin(th)

    def draw_outline(self, ax):
        for rr in (self.inner_r, self.outer_r):
            ax.add_patch(matplotlib.patches.Circle((self.cx, self.cy), rr, fill=False,
                                                   edgecolor='0.2', lw=0.8, zorder=5))

    def bin_areas_cm2(self):
        dth = np.radians(self.bin_width_deg)
        r_lo = self.inner_r + np.arange(self.ny) * self.bin_cm_y
        r_hi = r_lo + self.bin_cm_y
        return np.tile(0.5 * dth * (r_hi ** 2 - r_lo ** 2), self.nx)

    def smooth(self, fr_flat, valid_flat):
        return _gaussian_smooth_2d(fr_flat.reshape(self.nx, self.ny), valid_flat.reshape(self.nx, self.ny),
                                   RATEMAP_SMOOTH_SIGMA_BINS, wrap_x=True).ravel()

    def plot_trajectory(self, ax, x_cm, y_cm):
        # Cartesian axes (not polar): polar Line2D segments are drawn as straight chords in
        # display space, which cut across the ring. Same orientation as edge_to_xy (0 deg = East, CCW).
        ax.plot(x_cm, y_cm, **TRAJ_LINE_KW)
        for rr in (self.inner_r, self.outer_r):
            ax.add_patch(matplotlib.patches.Circle((self.cx, self.cy), rr, fill=False,
                                                   edgecolor='0.35', lw=0.8, zorder=5))
        lim = self.outer_r + 5
        ax.set_xlim(self.cx - lim, self.cx + lim)
        ax.set_ylim(self.cy - lim, self.cy + lim)
        ax.set_aspect('equal')
        ax.axis('off')

    def plot_bins_2d(self, ax, values_flat, valid_flat, cmap, norm):
        grid = np.where(valid_flat, values_flat, np.nan).reshape(self.nx, self.ny)
        x_edges = np.linspace(0.0, self.circumference_cm, self.nx + 1)
        y_edges = np.linspace(0.0, self.track_width_cm, self.ny + 1)
        pcm = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(grid.T),
                            cmap=cmap, norm=norm, shading='flat')
        ax.set_aspect('auto')
        ax.set_xlabel('Arc length at mean radius (cm; 0 = East, CCW)')
        ax.set_ylabel('From inner wall (cm)')
        return pcm


class LinearTrackHandler:
    wrap_x = False

    def __init__(self, cfg, bin_cm):
        self.length, self.width = cfg['length_cm'], cfg['width_cm']
        self.nx = max(4, int(round(self.length / bin_cm)))
        self.ny = max(2, int(round(self.width / bin_cm)))
        self.bin_cm_x = self.length / self.nx
        self.bin_cm_y = self.width / self.ny
        self.n_bins = self.nx * self.ny
        self.geom_valid = np.ones(self.n_bins, dtype=bool)

    def orient(self, x_cm, y_cm):
        """Vertical tracks (long axis along y) are rotated 90 deg CCW onto the length axis."""
        if len(x_cm) == 0:
            return x_cm, y_cm
        if (y_cm.max() - y_cm.min()) > (x_cm.max() - x_cm.min()):
            xr, yr = -y_cm, x_cm
            return xr - xr.min(), yr - yr.min()
        return x_cm, y_cm

    def to_bins(self, x_cm, y_cm):
        bx = np.clip((np.clip(x_cm, 0, self.length) / self.bin_cm_x).astype(int), 0, self.nx - 1)
        by = np.clip((np.clip(y_cm, 0, self.width) / self.bin_cm_y).astype(int), 0, self.ny - 1)
        return bx * self.ny + by, np.ones(len(x_cm), dtype=bool)

    def bin_centres_xy(self):
        xs = (np.arange(self.nx) + 0.5) * self.bin_cm_x
        ys = (np.arange(self.ny) + 0.5) * self.bin_cm_y
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        return np.vstack([XX.ravel(), YY.ravel()])

    def edge_to_xy(self, u, v):
        return np.asarray(u) * self.bin_cm_x, np.asarray(v) * self.bin_cm_y

    def draw_outline(self, ax):
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), self.length, self.width,
                                                  fill=False, edgecolor='0.2', lw=1.0, zorder=5))

    def bin_areas_cm2(self):
        return np.full(self.n_bins, self.bin_cm_x * self.bin_cm_y)

    def smooth(self, fr_flat, valid_flat):
        return _gaussian_smooth_2d(fr_flat.reshape(self.nx, self.ny), valid_flat.reshape(self.nx, self.ny),
                                   RATEMAP_SMOOTH_SIGMA_BINS).ravel()

    def plot_trajectory(self, ax, x_cm, y_cm):
        ax.plot(x_cm, y_cm, **TRAJ_LINE_KW)
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), self.length, self.width,
                                                  fill=False, edgecolor='0.35', lw=1.0, zorder=5))
        ax.set_xlim(-1, self.length + 1)
        ax.set_ylim(-1, self.width + 1)
        ax.set_aspect('equal')
        ax.axis('off')

    def plot_bins_2d(self, ax, values_flat, valid_flat, cmap, norm):
        grid = np.where(valid_flat, values_flat, np.nan).reshape(self.nx, self.ny)
        x_edges = np.arange(self.nx + 1) * self.bin_cm_x
        y_edges = np.arange(self.ny + 1) * self.bin_cm_y
        pcm = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(grid.T),
                            cmap=cmap, norm=norm, shading='flat')
        ax.set_aspect('auto')
        ax.set_xlabel('Track length (cm)')
        ax.set_ylabel('Track width (cm)')
        return pcm


def make_handler(cfg: dict):
    return {'circle': OpenFieldHandler, 'ring': CircularTrackHandler,
            'linear': LinearTrackHandler}[cfg['shape']](cfg, target_bin_cm)


# ============================================================================
# Step 1 -- one concatenated trajectory + occupancy map per arena
# ============================================================================

def session_trajectory(csv_path: str, handler):
    """One session's oriented, on-arena samples, their flat bin and their dwell time dt (s) --
    same dt / binning arithmetic as MeanOccupancyMap_Tracking_v2.session_occupancy. None if the
    tracking is unusable."""
    x_cm, y_cm, t = _session_positions(csv_path, handler)
    if len(t) < 2:
        return None
    x_cm, y_cm = handler.orient(x_cm, y_cm)
    bin_idx, sample_valid = handler.to_bins(x_cm, y_cm)
    x_cm, y_cm = x_cm[sample_valid], y_cm[sample_valid]
    t, bin_idx = t[sample_valid], bin_idx[sample_valid]
    if len(t) < 2:
        return None

    dt = np.empty(len(t), dtype=np.float64)
    dt[0] = 1.0 / fps
    dt[1:] = np.minimum(np.diff(t) * 1e-6, 2.0 / fps)
    return dict(x=x_cm, y=y_cm, t_us=t, bin_idx=bin_idx, dt=dt)


def chain_trajectories(trajs: list) -> tuple:
    """Concatenate per-session (x, y, t_us) in order into one trajectory (x, y, t_s). Each
    session's clock is shifted so its first sample comes TRAJ_FILE_GAP_S after the previous
    session's last sample."""
    xs, ys, ts = [], [], []
    t_next = 0.0
    for x, y, t_us in trajs:
        t_s = (t_us - t_us[0]) * 1e-6 + t_next
        xs.append(x)
        ys.append(y)
        ts.append(t_s)
        t_next = t_s[-1] + TRAJ_FILE_GAP_S
    if not ts:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ts)


def find_tracking_files(arena_key: str) -> list:
    """(session_name, csv_path) for every folder of this arena holding exactly one *_cm.csv."""
    out_dir = os.path.normcase(os.path.abspath(OUTPUT_DIR))
    sessions = []
    for dirpath, dirnames, filenames in os.walk(ROOT_DIRECTORY):
        dirnames[:] = [d for d in dirnames
                       if os.path.normcase(os.path.abspath(os.path.join(dirpath, d))) != out_dir]
        if _detect_arena_key(dirpath) != arena_key:
            continue
        tracking = [f for f in filenames if f.lower().endswith('_cm.csv')]
        if len(tracking) == 1:
            sessions.append((os.path.relpath(dirpath, ROOT_DIRECTORY),
                             os.path.join(dirpath, tracking[0])))
        elif len(tracking) > 1:
            print(f'  [SKIP] {dirpath}: {len(tracking)} *_cm.csv files, expected 1')
    return sorted(sessions)


def collect_arena_trajectory(arena_key: str, handler) -> tuple:
    """(traj, rows): every session of this arena chained into one trajectory, with the flat bin
    and dwell time of every frame and the resulting occupancy map (s per bin)."""
    sessions, rows = [], []
    for session_name, csv_path in find_tracking_files(arena_key):
        try:
            s = session_trajectory(csv_path, handler)
        except Exception as e:
            print(f'  ERROR [{arena_key}] {session_name}: {e}')
            continue
        if s is None or not handler.geom_valid[s['bin_idx']].any():
            print(f'  [SKIP no tracking] [{arena_key}] {session_name}')
            continue
        sessions.append(s)
        rows.append(dict(arena=arena_key, session=session_name,
                         tracking_file=os.path.basename(csv_path),
                         n_frames=len(s['dt']), duration_s=round(float(s['dt'].sum()), 2)))
    print(f'[{arena_key}] {len(sessions)} sessions used')
    if not sessions:
        return None, rows

    x, y, t_s = chain_trajectories([(s['x'], s['y'], s['t_us']) for s in sessions])
    bin_idx = np.concatenate([s['bin_idx'] for s in sessions])
    dt = np.concatenate([s['dt'] for s in sessions])
    occ = np.bincount(bin_idx, weights=dt, minlength=handler.n_bins)
    return dict(x=x, y=y, t_s=t_s, bin_idx=bin_idx, dt=dt, occ=occ, n_sessions=len(sessions)), rows


# ============================================================================
# Step 2 -- perfect place fields tiling the arena
# ============================================================================

def _exact_block_tiling(handler):
    """Tile a fully valid nx x ny grid with identical w x h blocks: the smallest block of
    FIELD_BINS..FIELD_BINS+MAX_EXTRA_BINS bins (both sides >= 2), squarest first, that divides the
    grid exactly. Returns (labels, block_ij, description) or None if no block fits."""
    if not handler.geom_valid.all():
        return None
    nx, ny = handler.nx, handler.ny
    for n in range(FIELD_BINS, FIELD_BINS + MAX_EXTRA_BINS + 1):
        shapes = sorted(((w, n // w) for w in range(2, n // 2 + 1) if n % w == 0 and n // w >= 2),
                        key=lambda s: max(s) / min(s))
        for w, h in shapes:
            if nx % w or ny % h:
                continue
            BX, BY = np.meshgrid(np.arange(nx) // w, np.arange(ny) // h, indexing='ij')
            labels = (BX * (ny // h) + BY).ravel()
            block_ij = np.array([(i, j) for i in range(nx // w) for j in range(ny // h)])
            return labels, block_ij, f'exact {w} x {h}-bin blocks'
    return None


def _lattice_assignment_tiling(handler):
    """FIELD_SIDE_BINS x FIELD_SIDE_BINS lattice over the bin grid; every block lying wholly inside
    the arena is one field, seeded at its centre. Extra seeds sit at the centroids of the
    largest wall-cut blocks, so the arena holds floor(n_valid / FIELD_BINS) fields. Each valid
    bin is then assigned to a seed by a min-cost assignment (squared distance) in which every
    seed must take FIELD_BINS bins and may take up to MAX_EXTRA_BINS more; the wall-cut seeds
    are re-centred on their fields and the assignment repeated until stable."""
    s = FIELD_SIDE_BINS
    nx, ny = handler.nx, handler.ny
    pts = np.argwhere(handler.geom_valid.reshape(nx, ny))          # (n_valid, 2) bin (ix, iy)
    flat = pts[:, 0] * ny + pts[:, 1]
    n_fields = len(pts) // FIELD_BINS
    if n_fields == 0 or len(pts) > n_fields * (FIELD_BINS + MAX_EXTRA_BINS):
        raise ValueError(f'{len(pts)} arena bins cannot be split into fields of '
                         f'{FIELD_BINS}..{FIELD_BINS + MAX_EXTRA_BINS} bins')

    ox, oy = (nx % s) // 2, (ny % s) // 2
    blk = np.stack([(pts[:, 0] - ox) // s, (pts[:, 1] - oy) // s], axis=1)
    keys, inv, counts = np.unique(blk, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    full = np.flatnonzero(counts == s * s)
    cut = np.flatnonzero(counts < s * s)
    cut = cut[np.argsort(-counts[cut], kind='stable')]
    seed_blocks = np.concatenate([full, cut[:n_fields - len(full)]])
    seeds = np.array([pts[inv == b].mean(axis=0) for b in seed_blocks], dtype=np.float64)

    big = 1e6   # bonus that makes the first FIELD_BINS slots of every seed mandatory
    for _ in range(100):
        d2 = ((pts[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=-1)
        cost = np.concatenate([d2 - big] * FIELD_BINS + [d2] * MAX_EXTRA_BINS, axis=1)
        rows, cols = linear_sum_assignment(cost)
        lab = np.empty(len(pts), dtype=int)
        lab[rows] = cols % n_fields
        new = seeds.copy()
        for k in range(len(full), n_fields):
            new[k] = pts[lab == k].mean(axis=0)
        if np.allclose(new, seeds):
            break
        seeds = new

    labels = np.full(handler.n_bins, -1, dtype=int)
    labels[flat] = lab
    return labels, keys[seed_blocks], (f'{s} x {s}-bin lattice, wall-cut blocks reassigned '
                                       f'({FIELD_BINS}..{FIELD_BINS + MAX_EXTRA_BINS} bins/field)')


def _field_tuning(handler, labels: np.ndarray, block_ij: np.ndarray) -> tuple:
    """(tuning (n_fields, n_bins), peak_bin (n_fields,)). Peak = field bin nearest the field
    centroid; exact ties are broken by the parity of the field's block index, so tied fields
    alternate (e.g. inner/outer half of the 2-bin-wide circular track). Rate falls linearly with
    distance from the peak, reaching 0 one bin beyond the field's farthest bin."""
    ix, iy = np.divmod(np.arange(handler.n_bins), handler.ny)
    n_fields = len(block_ij)
    tuning = np.zeros((n_fields, handler.n_bins), dtype=np.float64)
    peak_bin = np.empty(n_fields, dtype=int)
    for k in range(n_fields):
        members = np.flatnonzero(labels == k)
        px, py = ix[members], iy[members]
        d_c = np.hypot(px - px.mean(), py - py.mean())
        cand = members[np.isclose(d_c, d_c.min())]
        peak = cand[int(np.sum(block_ij[k])) % len(cand)]
        d = np.hypot(px - ix[peak], py - iy[peak])
        tuning[k, members] = FIELD_PEAK_RATE * (1.0 - d / (d.max() + 1.0))
        peak_bin[k] = peak
    return tuning, peak_bin


def tile_arena(handler) -> dict:
    tiled = _exact_block_tiling(handler)
    if tiled is None:
        tiled = _lattice_assignment_tiling(handler)
    labels, block_ij, mode = tiled

    # Every arena bin in exactly one field, every field contiguous and 9..12 bins
    assert np.all((labels >= 0) == handler.geom_valid), 'tiling does not cover the arena exactly'
    sizes = np.bincount(labels[labels >= 0], minlength=len(block_ij))
    assert sizes.min() >= FIELD_BINS and sizes.max() <= FIELD_BINS + MAX_EXTRA_BINS, sizes
    lab2 = labels.reshape(handler.nx, handler.ny)
    split = [k for k in range(len(block_ij)) if label(lab2 == k)[1] != 1]
    if split:
        print(f'  [WARN] {len(split)} non-contiguous fields: {split}')

    tuning, peak_bin = _field_tuning(handler, labels, block_ij)
    return dict(labels=labels, block_ij=block_ij, mode=mode, sizes=sizes,
                tuning=tuning, peak_bin=peak_bin, n_fields=len(block_ij))


# ============================================================================
# Step 3 -- simulated cells fire along the concatenated trajectory
# ============================================================================

def field_index_map(fr_flat: np.ndarray, valid_flat: np.ndarray) -> np.ndarray:
    fi = np.full(fr_flat.shape, np.nan, dtype=np.float64)
    if not valid_flat.any():
        return fi
    vals = fr_flat[valid_flat]
    vmin, vmax = float(vals.min()), float(vals.max())
    rng = vmax - vmin
    fi[valid_flat] = (vals - vmin) / rng if rng > 0 else 0.0
    return fi


def simulate_cells(handler, traj: dict, tiling: dict) -> tuple:
    """One rate map per tiled field. On every frame of the concatenated trajectory the cell emits
    rate(bin) * dt spikes, rate(bin) being its field's rate at the bin the animal is in. Cells
    whose field the trajectory never validly sampled stay silent and are left out (returned
    count), as a cell with no spikes would never have been recorded as a place cell."""
    bin_idx, dt, occ_map = traj['bin_idx'], traj['dt'], traj['occ']
    valid = (occ_map >= min_occ_s) & handler.geom_valid
    results, n_silent = [], 0
    for k in range(tiling['n_fields']):
        spikes_per_frame = tiling['tuning'][k, bin_idx] * dt
        spike_map = np.bincount(bin_idx, weights=spikes_per_frame, minlength=handler.n_bins)

        fr_raw = np.zeros(handler.n_bins, dtype=np.float64)
        fr_raw[valid] = spike_map[valid] / occ_map[valid]
        if not np.any(fr_raw[valid] > 0):
            n_silent += 1
            continue
        fr_map = handler.smooth(fr_raw, valid) if SMOOTH_SIM_RATEMAPS else fr_raw
        fi_map = field_index_map(fr_map, valid)
        field_mask = valid & (fr_map >= FIELD_PEAK_FRAC * float(fr_map[valid].max()))
        peak_bin_raw = int(np.argmax(np.where(valid, fr_raw, -np.inf)))
        results.append(dict(cell=k, valid=valid, fr_raw=fr_raw, fr_smooth=fr_map, fi_map=fi_map,
                            field_mask=field_mask, peak_bin_raw=peak_bin_raw))
    return results, n_silent


# ============================================================================
# Pooling across simulated cells (same rules as v23)
# ============================================================================

def pool_fine_map(handler, results: list) -> tuple:
    if not results:
        return np.full(handler.n_bins, np.nan), np.zeros(handler.n_bins, dtype=bool)
    stack = np.full((len(results), handler.n_bins), np.nan)
    for i, r in enumerate(results):
        stack[i, r['valid']] = r['fi_map'][r['valid']]
    with warnings.catch_warnings():   # bins no cell sampled are all-NaN -> NaN, as intended
        warnings.simplefilter('ignore', RuntimeWarning)
        mean_map = np.nanmean(stack, axis=0)
    any_valid = ~np.all(np.isnan(stack), axis=0)
    return mean_map, any_valid


def pool_field_only_map(handler, results: list) -> tuple:
    if not results:
        return np.full(handler.n_bins, np.nan), np.zeros(handler.n_bins, dtype=bool)
    fi_sum    = np.zeros(handler.n_bins, dtype=np.float64)
    n_sampled = np.zeros(handler.n_bins, dtype=np.float64)
    n_field   = np.zeros(handler.n_bins, dtype=np.int64)
    for r in results:
        valid = r['valid']
        n_sampled[valid] += 1.0
        m = r['field_mask'] & valid
        fi_sum[m] += r['fi_map'][m]
        n_field[m] += 1

    mean_map = np.full(handler.n_bins, np.nan)
    in_any_field = (n_field > 0) & (n_sampled > 0)
    mean_map[in_any_field] = fi_sum[in_any_field] / n_sampled[in_any_field]
    return mean_map, in_any_field


def pool_peak_proportion_map(handler, results: list) -> tuple:
    n_cells = len(results)
    visited = np.zeros(handler.n_bins, dtype=bool)
    counts  = np.zeros(handler.n_bins, dtype=np.float64)
    for r in results:
        visited |= r['valid']
        counts[r['peak_bin_raw']] += 1.0

    pct_map = np.full(handler.n_bins, np.nan)
    if n_cells > 0:
        pct_map[visited] = counts[visited] / n_cells * 100.0
    return pct_map, visited, n_cells


def _union_valid(handler, results: list) -> np.ndarray:
    visited = np.zeros(handler.n_bins, dtype=bool)
    for r in results:
        visited |= r['valid']
    return visited


# ============================================================================
# Step 4 -- 2D kernel density estimate of the null maps (v23: gaussian_kde, Scott's rule)
# ============================================================================

def _kernel_mass_in_domain(eval_pts: np.ndarray, dom_pts: np.ndarray, dom_area: np.ndarray,
                           cov: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(cov)
    norm_factor = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(cov)))
    d = eval_pts[:, :, None] - dom_pts[:, None, :]
    q = np.einsum('imn,ij,jmn->mn', d, inv, d)
    return norm_factor * (np.exp(-0.5 * q) @ dom_area)


def kde_density_map(handler, values_flat: np.ndarray, domain_flat: np.ndarray,
                    values_are_mass: bool = False):
    domain_flat = domain_flat.astype(bool)
    pts  = handler.bin_centres_xy()
    area = handler.bin_areas_cm2()

    vals = np.where(domain_flat & np.isfinite(values_flat), values_flat, 0.0)
    vals = np.clip(vals, 0.0, None)
    mass = vals if values_are_mass else vals * area
    fit = mass > 0
    if fit.sum() < 3:
        return None
    try:
        kde = gaussian_kde(pts[:, fit], bw_method=KDE_BW_METHOD, weights=mass[fit])
    except (np.linalg.LinAlgError, ValueError):
        return None

    ev = pts[:, domain_flat]
    dens_raw = kde(ev)
    dens = dens_raw
    if KDE_EDGE_CORRECTION:
        kmass = _kernel_mass_in_domain(ev, ev, area[domain_flat], kde.covariance)
        dens = np.where(kmass > 0, dens_raw / np.where(kmass > 0, kmass, 1.0), np.nan)

    density_raw = np.full(handler.n_bins, np.nan)
    density     = np.full(handler.n_bins, np.nan)
    density_raw[domain_flat] = dens_raw
    density[domain_flat]     = dens

    total = float(mass.sum())
    scaled = density * total * (area if values_are_mass else 1.0)

    return dict(density=density, density_raw=density_raw, scaled=scaled,
                factor=float(kde.factor), neff=float(kde.neff), cov=kde.covariance)


_MAP_KINDS = {
    'overall': (lambda h, res: (pool_fine_map(h, res)[0], False),
                'Overall mean field index map', 'field index', 'Field index (a.u.)',
                'Null_FigS1H_MeanFieldIndex'),
    'field_only': (lambda h, res: (pool_field_only_map(h, res)[0], False),
                   'Field-only mean field index map', 'field index',
                   'Field index, background = 0 (a.u.)', 'Null_FieldOnly_MeanFieldIndex'),
    'peak': (lambda h, res: (pool_peak_proportion_map(h, res)[0], True),
             'Peak proportion map', '% of cells per bin', '% of cells with peak in bin',
             'Null_PeakProportion_Map'),
}


# ============================================================================
# Plotting / export
# ============================================================================

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


def _no_data(ax, key, why='no sessions'):
    ax.set_title(f'{_ARENA_TITLES[key]}\n({why})')
    ax.axis('off')


# --- Bins drawn in true arena geometry (room cm; the circular track as an actual ring) ---

_EDGE_PTS = 6   # points per bin side, so the circular track's bin edges follow the ring's arcs


def _bin_polygons(handler, flat_bins: np.ndarray) -> np.ndarray:
    """(n, 4 * _EDGE_PTS, 2) outline of every bin in flat_bins, in arena cm."""
    s = np.linspace(0.0, 1.0, _EDGE_PTS)
    du = np.concatenate([s, np.ones_like(s), s[::-1], np.zeros_like(s)])
    dv = np.concatenate([np.zeros_like(s), s, np.ones_like(s), s[::-1]])
    ix, iy = np.divmod(flat_bins, handler.ny)
    x, y = handler.edge_to_xy(ix[:, None] + du, iy[:, None] + dv)
    return np.stack([x, y], axis=-1)


def _field_boundary_segments(handler, labels: np.ndarray) -> list:
    """Every bin edge separating two different fields, or a field from outside the arena."""
    nx, ny = handler.nx, handler.ny
    lab = labels.reshape(nx, ny)

    def at(i, j):
        if handler.wrap_x:
            i %= nx
        return lab[i, j] if (0 <= i < nx and 0 <= j < ny) else -1

    s = np.linspace(0.0, 1.0, _EDGE_PTS)
    segs = []
    for i in (range(nx) if handler.wrap_x else range(-1, nx)):     # edge u = i + 1
        for j in range(ny):
            a, b = at(i, j), at(i + 1, j)
            if a != b and max(a, b) >= 0:
                segs.append(np.column_stack(handler.edge_to_xy(np.full_like(s, i + 1), j + s)))
    for i in range(nx):                                             # edge v = j + 1
        for j in range(-1, ny):
            a, b = at(i, j), at(i, j + 1)
            if a != b and max(a, b) >= 0:
                segs.append(np.column_stack(handler.edge_to_xy(i + s, np.full_like(s, j + 1))))
    return segs


def _field_colours(handler, labels: np.ndarray, n_fields: int) -> np.ndarray:
    """(n_fields, 4) RGBA, coloured greedily so no two touching fields (8-neighbourhood) match."""
    nx, ny = handler.nx, handler.ny
    lab = labels.reshape(nx, ny)
    nbrs = [set() for _ in range(n_fields)]
    for i in range(nx):
        for j in range(ny):
            a = lab[i, j]
            if a < 0:
                continue
            for di, dj in ((1, -1), (1, 0), (1, 1), (0, 1)):
                i2, j2 = i + di, j + dj
                if handler.wrap_x:
                    i2 %= nx
                if 0 <= i2 < nx and 0 <= j2 < ny and lab[i2, j2] >= 0 and lab[i2, j2] != a:
                    nbrs[a].add(lab[i2, j2])
                    nbrs[lab[i2, j2]].add(a)
    palette = _get_cmap('tab10')(np.arange(10))
    colour = np.full(n_fields, -1)
    for k in range(n_fields):
        used = {colour[n] for n in nbrs[k]}
        colour[k] = next(c for c in ((3 * k + o) % 10 for o in range(10)) if c not in used)
    return palette[colour]


def _draw_bins(ax, handler, shown: np.ndarray, values=None, cmap=None, norm=None, facecolors=None):
    """Fill the bins in shown, coloured by values (cmap / norm) or by per-bin facecolors."""
    bins = np.flatnonzero(shown)
    if facecolors is None:
        pc = PolyCollection(_bin_polygons(handler, bins), array=values[bins], cmap=cmap, norm=norm,
                            edgecolors='face', linewidths=0.3)
    else:
        pc = PolyCollection(_bin_polygons(handler, bins), facecolors=facecolors[bins],
                            edgecolors='face', linewidths=0.3)
    ax.add_collection(pc)
    handler.draw_outline(ax)
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.axis('off')
    return pc


def _draw_field_boundaries(ax, handler, labels: np.ndarray, **line_kw):
    ax.add_collection(LineCollection(_field_boundary_segments(handler, labels), zorder=4, **line_kw))


def _arena_grid(fig, n_rows):
    """n_rows x 3 arena columns; the 80 x 8 cm linear track gets the widest column."""
    return fig.add_gridspec(n_rows, 3, width_ratios=[1.0, 1.0, 1.6])


def _hbar(fig, mappable, ax, label):
    fig.colorbar(mappable, ax=ax, orientation='horizontal', fraction=0.05, pad=0.03, label=label)


def plot_field_tiling(arena_data: dict, save_path: str):
    """Row 1: the field every arena bin belongs to (black = field outlines, dot = peak bin).
    Row 2: the perfect rate map those fields define. Independent of the tracking data, so every
    arena is drawn whether or not it has sessions."""
    fig = plt.figure(figsize=(20, 12))
    gs = _arena_grid(fig, 2)
    for ci, key in enumerate(_ARENA_ORDER):
        h, tiling = arena_data[key]['handler'], arena_data[key]['tiling']
        arena, labels = h.geom_valid, tiling['labels']
        sizes, counts = np.unique(tiling['sizes'], return_counts=True)

        ax = fig.add_subplot(gs[0, ci])
        fc = np.zeros((h.n_bins, 4))
        fc[arena] = _field_colours(h, labels, tiling['n_fields'])[labels[arena]]
        _draw_bins(ax, h, arena, facecolors=fc)
        _draw_field_boundaries(ax, h, labels, colors='k', linewidths=0.8)
        px, py = h.edge_to_xy(*(np.array(np.divmod(tiling['peak_bin'], h.ny)) + 0.5))
        ax.scatter(px, py, s=10, c='k', zorder=6)
        ax.set_title(f"{_ARENA_TITLES[key]}: {tiling['n_fields']} non-overlapping fields\n"
                     f"{tiling['mode']}\nbins per field: "
                     + ', '.join(f'{s} ({n} fields)' for s, n in zip(sizes, counts)), fontsize=9)

        ax = fig.add_subplot(gs[1, ci])
        rate = tiling['tuning'].sum(axis=0)
        cmap, norm = make_cmap_norm(rate[arena])
        pc = _draw_bins(ax, h, arena, rate, cmap, norm)
        _draw_field_boundaries(ax, h, labels, colors='k', linewidths=0.5, alpha=0.6)
        ax.set_title(f'Perfect place fields: rate {FIELD_PEAK_RATE:g} at the peak bin,\n'
                     'falling linearly to 0 one bin beyond the field', fontsize=9)
        _hbar(fig, pc, ax, 'rate')

    fig.suptitle(f'Observed null -- Step 2: perfect place fields tiling each arena '
                 f'({target_bin_cm:g} x {target_bin_cm:g} cm bins; dot = peak bin)')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_occupancy_trajectory(arena_data: dict, save_path: str):
    """Row 1: occupancy of the concatenated trajectory (s per bin). Row 2: that trajectory."""
    fig = plt.figure(figsize=(20, 12))
    gs = _arena_grid(fig, 2)
    for ci, key in enumerate(_ARENA_ORDER):
        h, traj = arena_data[key]['handler'], arena_data[key]['traj']
        ax_occ, ax_trj = fig.add_subplot(gs[0, ci]), fig.add_subplot(gs[1, ci])
        if traj is None:
            _no_data(ax_occ, key)
            _no_data(ax_trj, key)
            continue
        occ_ok = (traj['occ'] > 0) & h.geom_valid
        cmap, norm = make_cmap_norm(traj['occ'][occ_ok])
        pc = _draw_bins(ax_occ, h, occ_ok, traj['occ'], cmap, norm)
        n_invalid = int((h.geom_valid & (traj['occ'] < min_occ_s)).sum())
        ax_occ.set_title(f"{_ARENA_TITLES[key]}: concatenated occupancy\n({traj['n_sessions']} sessions, "
                         f"{traj['dt'].sum() / 60.0:.1f} min; {n_invalid} arena bins < {min_occ_s:g} s "
                         f"= invalid)", fontsize=9)
        _hbar(fig, pc, ax_occ, 'occupancy (s)')
        h.plot_trajectory(ax_trj, traj['x'], traj['y'])
        ax_trj.set_title('Concatenated trajectory (sessions chained)', fontsize=9)

    fig.suptitle('Observed null -- Step 1: concatenated trajectory and occupancy')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_null_maps(arena_data: dict, save_path: str):
    """Rows: overall mean, field-only mean, peak proportion null maps; columns: arenas. Thin lines
    are the tiled field outlines; blank bins had < min_occ_s occupancy."""
    fig = plt.figure(figsize=(20, 17))
    gs = _arena_grid(fig, len(_MAP_KINDS))
    for ri, (pool_fn, title, _, cbar_label, _) in enumerate(_MAP_KINDS.values()):
        for ci, key in enumerate(_ARENA_ORDER):
            d = arena_data[key]
            ax = fig.add_subplot(gs[ri, ci])
            if not d['results']:
                _no_data(ax, key, 'no sessions' if d['traj'] is None else 'no simulated cells')
                continue
            h = d['handler']
            values, _ = pool_fn(h, d['results'])
            shown = np.isfinite(values) & h.geom_valid
            cmap, norm = make_cmap_norm(values[shown])
            pc = _draw_bins(ax, h, shown, values, cmap, norm)
            _draw_field_boundaries(ax, h, d['tiling']['labels'], colors='k', linewidths=0.4, alpha=0.35)
            ax.set_title(f"{_ARENA_TITLES[key]} -- {title}\n(n={len(d['results'])} simulated cells)",
                         fontsize=9)
            _hbar(fig, pc, ax, cbar_label)

    fig.suptitle('Observed null mean maps: perfect tiled place fields x real concatenated trajectory\n'
                 f'(thin lines = tiled field outlines; blank = < {min_occ_s:g} s occupancy)')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_kde_maps(arena_data: dict, map_kind: str, save_path: str):
    pool_fn, title, units, _, _ = _MAP_KINDS[map_kind]
    fig = plt.figure(figsize=(17, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.7])
    arena_axes = {'open_field':     fig.add_subplot(gs[:, 0]),
                  'circular_track': fig.add_subplot(gs[0, 1]),
                  'linear_track':   fig.add_subplot(gs[1, 1])}
    saved = {}
    for key in _ARENA_ORDER:
        d = arena_data[key]
        ax = arena_axes[key]
        if not d['results']:
            _no_data(ax, key, 'no simulated cells')
            continue
        handler, results = d['handler'], d['results']

        values, values_are_mass = pool_fn(handler, results)
        domain = _union_valid(handler, results)
        kde_out = kde_density_map(handler, values, domain, values_are_mass) if domain.any() else None
        if kde_out is None:
            _no_data(ax, key, 'no KDE: too few data')
            continue

        density = kde_out['density']
        cmap, norm_c = make_cmap_norm(density[domain])
        pcm = handler.plot_bins_2d(ax, density, domain & np.isfinite(density), cmap, norm_c)
        sd = np.sqrt(np.diag(kde_out['cov']))
        ax.set_title(f'{_ARENA_TITLES[key]} (n={len(results)} simulated cells, '
                     f'{handler.nx} x {handler.ny} bins)\n'
                     f'Scott factor={kde_out["factor"]:.3f}, neff={kde_out["neff"]:.0f}, '
                     f'BW sd={sd[0]:.1f} x {sd[1]:.1f} cm (room x, y)', fontsize=9)
        fig.colorbar(pcm, ax=ax, shrink=0.8 if key == 'open_field' else 1.0,
                     label='KDE density (cm$^{-2}$)')

        for k in ('density', 'density_raw', 'scaled', 'cov'):
            saved[f'{key}_{k}'] = kde_out[k]
        saved[f'{key}_factor'] = kde_out['factor']
        saved[f'{key}_neff'] = kde_out['neff']
        saved[f'{key}_domain'] = domain
        d.setdefault('kde', {})[map_kind] = kde_out

    corr = 'edge-corrected' if KDE_EDGE_CORRECTION else 'no edge correction'
    fig.suptitle(f'Observed null -- {title} -- 2D Gaussian KDE (Scott\'s rule, {corr})')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')

    if saved:
        npz_path = os.path.splitext(save_path)[0] + '.npz'
        np.savez_compressed(npz_path, scaled_units=units, **saved)
        print(f'[SAVED] {npz_path}')


def export_null_maps(arena_data: dict, out_path: str):
    saved = {}
    for key in _ARENA_ORDER:
        d = arena_data[key]
        if not d['results']:
            continue
        h, res = d['handler'], d['results']
        saved[f'{key}_overall'], saved[f'{key}_overall_valid'] = pool_fine_map(h, res)
        saved[f'{key}_field_only'], saved[f'{key}_field_only_valid'] = pool_field_only_map(h, res)
        saved[f'{key}_peak'], saved[f'{key}_peak_valid'], _ = pool_peak_proportion_map(h, res)
        saved[f'{key}_domain'] = _union_valid(h, res)
        saved[f'{key}_field_labels'] = d['tiling']['labels']
        saved[f'{key}_field_peak_bin'] = d['tiling']['peak_bin']
        saved[f'{key}_field_tuning'] = d['tiling']['tuning']
        saved[f'{key}_occupancy_s'] = d['traj']['occ']
        saved[f'{key}_nx_ny'] = np.array([h.nx, h.ny])
    if saved:
        np.savez_compressed(out_path, **saved)
        print(f'[SAVED] {out_path}')


def export_excel(arena_data: dict, out_path: str):
    arena_rows, session_rows = [], []
    for key in _ARENA_ORDER:
        d = arena_data[key]
        session_rows += d['rows']
        if d['traj'] is None:
            arena_rows.append(dict(arena=key, n_sessions=0))
            continue
        h, tiling, traj = d['handler'], d['tiling'], d['traj']
        sizes, counts = np.unique(tiling['sizes'], return_counts=True)
        row = dict(arena=key, n_sessions=traj['n_sessions'],
                   total_time_min=round(float(traj['dt'].sum()) / 60.0, 2),
                   grid_nx=h.nx, grid_ny=h.ny, arena_bins=int(h.geom_valid.sum()),
                   arena_bins_valid=int(((traj['occ'] >= min_occ_s) & h.geom_valid).sum()),
                   tiling=tiling['mode'], n_fields=tiling['n_fields'],
                   bins_per_field=', '.join(f'{s}:{n}' for s, n in zip(sizes, counts)),
                   n_cells_pooled=len(d['results']), n_cells_silent=d['n_silent'],
                   smoothed=SMOOTH_SIM_RATEMAPS)
        for kind, kde_out in d.get('kde', {}).items():
            row[f'kde_{kind}_scott_factor'] = round(kde_out['factor'], 4)
            row[f'kde_{kind}_neff'] = round(kde_out['neff'], 1)
        arena_rows.append(row)
    with pd.ExcelWriter(out_path) as xw:
        pd.DataFrame(arena_rows).to_excel(xw, sheet_name='Arenas', index=False)
        pd.DataFrame(session_rows).to_excel(xw, sheet_name='Sessions', index=False)
    print(f'[SAVED] {out_path}')


# ============================================================================
# __main__ Pipeline
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    arena_data = {}
    for key in _ARENA_ORDER:
        handler = make_handler(ARENA_CONFIGS[key])
        traj, rows = collect_arena_trajectory(key, handler)                    # Step 1
        d = dict(handler=handler, traj=traj, rows=rows, tiling=tile_arena(handler),   # Step 2
                 results=[], n_silent=0)
        print(f"[{key}] {d['tiling']['n_fields']} fields ({d['tiling']['mode']})")
        if traj is not None:
            d['results'], d['n_silent'] = simulate_cells(handler, traj, d['tiling'])   # Step 3
            print(f"[{key}] {len(d['results'])} cells pooled, {d['n_silent']} silent "
                  f"(field never sampled)")
        arena_data[key] = d

    plot_field_tiling(arena_data, os.path.join(OUTPUT_DIR, 'ObservedNull_FieldTiling.png'))
    plot_occupancy_trajectory(arena_data, os.path.join(OUTPUT_DIR, 'ObservedNull_Occupancy_Trajectory.png'))
    plot_null_maps(arena_data, os.path.join(OUTPUT_DIR, 'Null_MeanMaps_AllTypes.png'))
    for kind, (*_, stem) in _MAP_KINDS.items():
        plot_kde_maps(arena_data, kind, os.path.join(OUTPUT_DIR,                # Step 4
                                                     stem.replace('Null_', 'Null_KDE_') + '.png'))
    export_null_maps(arena_data, os.path.join(OUTPUT_DIR, 'ObservedNull_Maps.npz'))
    export_excel(arena_data, os.path.join(OUTPUT_DIR, 'ObservedNull_Summary.xlsx'))


if __name__ == '__main__':
    main()
    print('\nDone.')
