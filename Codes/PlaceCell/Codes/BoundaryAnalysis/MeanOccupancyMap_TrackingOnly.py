# -*- coding: utf-8 -*-
"""
Mean occupancy maps from tracking data ONLY (no neural data anywhere).

For every session folder under ROOT_DIRECTORY that holds exactly one cm tracking file
(*_cm.csv), this script builds that session's occupancy (dwell-time) map on the same 2 x 2 cm
bin grids used by MeanRateMap_QuadrantAnalysis_v19_Rayleigh.py, then averages the maps across
all sessions of each arena:
  - open_field     : circular open field, dia 60 cm (2D x/y grid, corners outside the circle masked)
  - circular_track : annular track 80/72 cm (arc-length x radial-width grid, wraps around the ring)
  - linear_track   : 80 x 8 cm (length x width grid; vertical sessions rotated 90 deg)

Arena type is detected from the path ('Open' / 'Linear' / 'Circle' path component), exactly as
in v19. The tracking load / clean / smooth steps and the open-field frame corrections (centring
on the arena centre, and the 'Rotate' session rotation) are ported unchanged from v19, so these
occupancy maps sit in the same frame as that script's rate maps.

Two averages are plotted per arena:
  - mean % of session time per bin : each session's map is first normalised to sum to 100 %,
                                     so every session counts equally regardless of its length
  - mean dwell time per bin (s)    : plain mean of the raw per-session occupancy maps
A bin a session never visited is NaN in that session's map, so each bin's mean is taken only
over the sessions that visited it (NaN-aware mean); bins no session ever visited stay NaN and
are left blank in the plots.

No session-level coverage criteria are applied: every tracking file found is used.

Outputs (in OUTPUT_DIR):
  MeanOccupancy_AllArenas.png   -- 2 rows (mean % time, mean seconds) x 3 arenas
  MeanOccupancy_Sessions.xlsx   -- one row per session used (arena, duration, bins visited)
"""

import os
import threading

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.colors import Normalize

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT_DIRECTORY = r'C:\Runita\SessionType_Sorted\Open\Rotate'
OUTPUT_DIR = os.path.join(ROOT_DIRECTORY, 'MeanOccupancyMaps')

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

POS_JUMP_THRESH_CMS  = 90.0   # frame-to-frame jumps faster than this (cm/s) are tracking artifacts
POS_SMOOTH_SIGMA_SMP = 1.0    # Gaussian smoothing sigma (samples) of x/y tracking

# Gaussian smoothing (bins) of the final mean occupancy maps; 0 = plot the raw (unsmoothed) mean.
# Uses the same NaN-safe, arena-topology-aware smoothing as v19's rate maps.
OCC_SMOOTH_SIGMA_BINS = 0.0

CENTRE_OPEN_FIELD_TRACKING = True
ROTATE_SESSION_KEYWORD     = 'rotate'
ROTATE_SESSION_CCW_DEG     = 0.0   # same value as v19 (was 120.0)

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
# Tracking load / clean / smooth / frame corrections (ported from v19)
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
    """Centre of a circular arena of known radius, fitted to the tracking's outer envelope
    (see v19 for the full rationale). Returns (cx, cy, ok)."""
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
    """Load -> clean -> smooth -> open-field centring -> Rotate fix (same order as v19)."""
    x_cm, y_cm, t = _load_tracking(csv_path)
    if len(t) < 2:
        return x_cm, y_cm, t
    x_cm, y_cm = _smooth_tracking_position(x_cm, y_cm, t)
    session_dir = os.path.dirname(csv_path)
    x_cm, y_cm = _centre_open_field_tracking(x_cm, y_cm, session_dir, handler)
    x_cm, y_cm = _apply_session_rotation(x_cm, y_cm, session_dir, handler)
    return x_cm, y_cm, t


# ============================================================================
# Smoothing + arena geometry handlers (binning / plotting only, ported from v19)
# ============================================================================

def _gaussian_smooth_2d(values, valid_mask, sigma, wrap_x=False):
    mode = ('wrap' if wrap_x else 'constant', 'constant')
    sv = gaussian_filter(np.where(valid_mask, values, 0.0), sigma=sigma, mode=mode, cval=0.0)
    sw = gaussian_filter(valid_mask.astype(np.float64), sigma=sigma, mode=mode, cval=0.0)
    out = np.zeros_like(sv)
    ok = sw > 0
    out[ok] = sv[ok] / sw[ok]
    out[~valid_mask] = 0.0
    return out


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

    def plot_fine(self, ax, values_flat, valid_flat, cmap, norm):
        grid = np.full(self.n_bins, np.nan)
        m = valid_flat & self.geom_valid
        grid[m] = values_flat[m]
        im = ax.imshow(np.ma.masked_invalid(grid.reshape(self.nx, self.ny).T), origin='lower',
                       extent=[0, self.diameter, 0, self.diameter], cmap=cmap, norm=norm)
        ax.add_patch(matplotlib.patches.Circle((self.cx, self.cy), self.diameter / 2.0,
                                               fill=False, edgecolor='0.35', lw=1.0, zorder=5))
        ax.set_aspect('equal')
        ax.axis('off')
        return im


class CircularTrackHandler:
    wrap_x = True

    def __init__(self, cfg, bin_cm):
        self.outer_r = cfg['outer_diameter_cm'] / 2.0
        self.inner_r = cfg['inner_diameter_cm'] / 2.0
        self.cx = self.cy = self.outer_r
        self.radial_tol_cm = 4.0
        self.track_width_cm = self.outer_r - self.inner_r
        circumference = 2 * np.pi * (self.outer_r + self.inner_r) / 2.0
        self.nx = max(8, 4 * int(round(circumference / bin_cm / 4.0)))
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

    def plot_fine(self, ax, values_flat, valid_flat, cmap, norm):
        grid = np.full(self.n_bins, np.nan)
        grid[valid_flat] = values_flat[valid_flat]
        im = ax.imshow(np.ma.masked_invalid(grid.reshape(self.nx, self.ny).T), origin='lower',
                       extent=[0, self.length, 0, self.width], aspect='equal', cmap=cmap, norm=norm)
        ax.axis('off')
        return im


def make_handler(cfg: dict):
    return {'circle': OpenFieldHandler, 'ring': CircularTrackHandler,
            'linear': LinearTrackHandler}[cfg['shape']](cfg, target_bin_cm)


# ============================================================================
# Occupancy
# ============================================================================

def session_occupancy(csv_path: str, handler):
    """Dwell time (s) per flat bin for one session, or None if the tracking is unusable.
    Same dt / binning arithmetic as v19's compute_cell_ratemap, without any spikes."""
    x_cm, y_cm, t = _session_positions(csv_path, handler)
    if len(t) < 2:
        return None
    x_cm, y_cm = handler.orient(x_cm, y_cm)
    bin_idx, sample_valid = handler.to_bins(x_cm, y_cm)
    t, bin_idx = t[sample_valid], bin_idx[sample_valid]
    if len(t) < 2:
        return None

    dt = np.empty(len(t), dtype=np.float64)
    dt[0] = 1.0 / fps
    dt[1:] = np.minimum(np.diff(t) * 1e-6, 2.0 / fps)

    occ = np.zeros(handler.n_bins, dtype=np.float64)
    np.add.at(occ, bin_idx, dt)
    occ[(occ <= 0) | ~handler.geom_valid] = np.nan   # unvisited (or outside-arena) bins -> NaN
    return occ


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


def collect_arena_occupancy(arena_key: str) -> dict:
    handler = make_handler(ARENA_CONFIGS[arena_key])
    maps, rows = [], []
    for session_name, csv_path in find_tracking_files(arena_key):
        try:
            occ = session_occupancy(csv_path, handler)
        except Exception as e:
            print(f'  ERROR [{arena_key}] {session_name}: {e}')
            continue
        if occ is None or not np.isfinite(occ).any():
            print(f'  [SKIP no tracking] [{arena_key}] {session_name}')
            continue
        maps.append(occ)
        visited = int(np.isfinite(occ[handler.geom_valid]).sum())
        rows.append(dict(arena=arena_key, session=session_name,
                         tracking_file=os.path.basename(csv_path),
                         total_time_s=round(float(np.nansum(occ)), 2),
                         bins_visited=visited, arena_bins=int(handler.geom_valid.sum()),
                         pct_bins_visited=round(100.0 * visited / handler.geom_valid.sum(), 1)))
    print(f'[{arena_key}] {len(maps)} sessions used')

    if maps:
        stack = np.array(maps)
        visited_any = np.isfinite(stack).any(axis=0) & handler.geom_valid
        mean_s = np.full(handler.n_bins, np.nan)
        mean_pct = np.full(handler.n_bins, np.nan)
        # NaN-aware means: each bin is averaged only over the sessions that visited it
        pct_stack = stack / np.nansum(stack, axis=1, keepdims=True) * 100.0
        mean_s[visited_any] = np.nanmean(stack[:, visited_any], axis=0)
        mean_pct[visited_any] = np.nanmean(pct_stack[:, visited_any], axis=0)
    else:
        mean_s = mean_pct = np.full(handler.n_bins, np.nan)
        visited_any = np.zeros(handler.n_bins, dtype=bool)

    if OCC_SMOOTH_SIGMA_BINS > 0 and visited_any.any():
        def _sm(v):
            return _gaussian_smooth_2d(v.reshape(handler.nx, handler.ny),
                                       visited_any.reshape(handler.nx, handler.ny),
                                       OCC_SMOOTH_SIGMA_BINS, handler.wrap_x).ravel()
        mean_s, mean_pct = _sm(mean_s), _sm(mean_pct)

    return dict(handler=handler, mean_s=mean_s, mean_pct=mean_pct, valid=visited_any,
                n_sessions=len(maps), rows=rows)


# ============================================================================
# Plotting / export
# ============================================================================

def _cmap_norm(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    vmin, vmax = (float(vals.min()), float(vals.max())) if len(vals) else (0.0, 1.0)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    cmap = matplotlib.colormaps['jet'].copy()
    cmap.set_bad('white')
    return cmap, Normalize(vmin=vmin, vmax=vmax)


def plot_mean_occupancy(arena_data: dict, save_path: str):
    rows = [('mean_pct', 'Mean % of session time per bin', '% of session time'),
            ('mean_s',   'Mean dwell time per bin (s)',    'Dwell time (s)')]
    fig = plt.figure(figsize=(15, 10))
    for ri, (key, row_title, cbar_label) in enumerate(rows):
        for ci, arena_key in enumerate(_ARENA_ORDER):
            d = arena_data[arena_key]
            proj = 'polar' if arena_key == 'circular_track' else None
            ax = fig.add_subplot(2, 3, ri * 3 + ci + 1, projection=proj)
            if not d['valid'].any():
                ax.axis('off')
                ax.set_title(f"{_ARENA_TITLES[arena_key]}\n(no sessions)")
                continue
            cmap, norm = _cmap_norm(d[key][d['valid']])
            im = d['handler'].plot_fine(ax, d[key], d['valid'], cmap, norm)
            ax.set_title(f"{_ARENA_TITLES[arena_key]} -- {row_title}\n(n={d['n_sessions']} sessions)",
                         fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.7, label=cbar_label)

    smooth_note = (f'Gaussian-smoothed, sigma {OCC_SMOOTH_SIGMA_BINS:g} bins' if OCC_SMOOTH_SIGMA_BINS > 0
                   else 'unsmoothed')
    fig.suptitle(f'Mean occupancy maps from tracking data ({target_bin_cm:g} x {target_bin_cm:g} cm bins, '
                 f'{smooth_note}; blank = never visited)')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    arena_data = {k: collect_arena_occupancy(k) for k in _ARENA_ORDER}
    plot_mean_occupancy(arena_data, os.path.join(OUTPUT_DIR, 'MeanOccupancy_AllArenas.png'))

    xlsx = os.path.join(OUTPUT_DIR, 'MeanOccupancy_Sessions.xlsx')
    pd.DataFrame([r for k in _ARENA_ORDER for r in arena_data[k]['rows']]).to_excel(xlsx, index=False)
    print(f'[SAVED] {xlsx}')


if __name__ == '__main__':
    main()
    print('\nDone.')
