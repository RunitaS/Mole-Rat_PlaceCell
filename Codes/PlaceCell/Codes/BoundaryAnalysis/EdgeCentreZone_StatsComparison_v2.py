# -*- coding: utf-8 -*-
"""
Edge vs centre place cells: every place cell is assigned to the zone of the arena that holds the
PEAK BIN OF ITS RAW RATE MAP, and every place-cell characterization metric is compared between the
two zones.

ZONES (2 x 2 cm bins; same definitions as MeanRateMap_QuadrantAnalysis_v27_5 / the TAC6
BoundaryAnalysis_open.py that produced dist_from_wall_cm / zone):
  Open   circular open field, dia 60 cm
           Edge   = bin centre <= OPEN_EDGE_ZONE_THRESHOLD_CM (9.25 cm) from the wall
           Centre = every other bin
  Linear 80 x 8 cm track (4 bins across the width; vertical sessions rotated to horizontal)
           Edge   = the 2 outer rows across the width (bin centre >= LINEAR_EDGE_ZONE_THRESHOLD_CM,
                    2 cm, from the midline) + the bins within 2 cm of either end of the track
           Centre = the 2 middle rows, away from the track ends
  Circle annular track, outer dia 80 cm, inner dia 72 cm (4 cm wide -> 2 radial rows of 2 x 2 cm
         bins, arc length x radial position)
           Outer  = outer ring of bins
           Inner  = inner ring of bins

PEAK BIN: the valid bin with the highest rate in the cell's RAW rate map (spikes / occupancy per
bin, no smoothing or adaptive binning), built exactly as the 'raw' map of v27_5 from the .ntt and
tracking files in PLACE_CELL_DIR: tracking cleaned (> 90 cm/s jumps interpolated) and Gaussian
smoothed, open-field tracking centred on the arena, spikes matched to the nearest frame (50 ms
gate), speed filter 0.5-90 cm/s on spikes (and occupancy, SPEED_FILTER_OCCUPANCY), valid bins =
bins visited >= MIN_VISITS times.

DWELL CRITERION: per session, the share of the (speed-filtered) dwell time in valid bins spent in
each of the two zones. A session is SKIPPED (none of its cells analysed) unless BOTH zones hold
>= MIN_ZONE_DWELL_PCT (20 %) of the dwell time.

COMPARISONS -- Edge vs Centre (Open, Linear) / Outer vs Inner (Circle), each arena separately:
  1. Arena            : all cells of the arena                     -> OUTPUT_DIR\\Arena\\<arena>
  2. Session type     : the cells of each session type of an arena -> OUTPUT_DIR\\SessionType\\<arena>_<type>
  3. Animal-controlled: for each entry of v10's ANIMAL_CONTROLLED_COMPARISONS, each of its arenas,
                        only the cells of its animals              -> OUTPUT_DIR\\AnimalControlledArenaType\\<name>\\<arena>

Metrics, tests, plots and histograms are those of ArenaSessionType_StatsComparison_v10.py, which
is imported (not copied) with 'zone' as the grouping column: continuous metrics (peak_fr, sir,
sparsity, coherence, stability score [+ per stability type], speed score, mean_fr, field number /
area / % area) with CONTINUOUS_TEST ('KW' = Mann-Whitney U for two zones, or 'LMM'), proportions
(theta modulation, coherence shuffle, stability significance, speed cells) and the speed-direction
classification with chi-square tests. Place-cell yield is left out: only place cells are in
PLACE_CELL_DIR, so non-place cells have no zone.

TEMPORAL STABILITY (TS_METRICS, every comparison above): from TEMPORAL_STABILITY_EXCEL, the
output of TemporalStabilityMap_v4_OccWgh.py (per field bin: % of visits with >= 1 spike). Only
each cell's HIGHEST-PEAK FIELD is used: the field holding the field bin with the highest raw rate
(n_spikes / occupancy_s of the PerBin sheet); that bin is the field's peak bin. Per cell:
  ts_mean_pct         mean   % across the bins of that field (PerField mean_pct_across_bins)
  ts_median_pct       median % across the bins of that field (PerField median_pct_across_bins)
  ts_occ_wmean_pct    occupancy-weighted mean / median % across the bins of that field, each bin
  ts_occ_wmedian_pct  weighted by its dwell time (PerField occ_wmean_pct_across_bins /
                      occ_wmedian_pct_across_bins), so poorly sampled bins count less
  ts_pct_at_peak_bin  % at the field's peak bin (PerBin pct_visits_with_spike)
The cell's zone is still that of its raw-map peak bin above. Same CONTINUOUS_TEST, box plots
(TemporalStability_<metric>.png) and histograms as the other continuous metrics; tables
<prefix>_TS_Desc / _TS_Omnibus / _TS_PostHoc. Cells absent from TEMPORAL_STABILITY_EXCEL (no
field detected there) are left out of these tests only.

Cells are matched between INPUT_EXCEL (the 'session' + 'unit' columns of the 'Full' and
'PlaceFields' sheets) and PLACE_CELL_DIR by the session path from the animal folder onwards
(e.g. 'Fa1059\\Linear\\Day10\\1_0') and the .ntt file name, case-insensitively.

Outputs (OUTPUT_DIR):
  EdgeCentre_Stats.xlsx    ZoneCounts   (cells per zone and mean dwell per zone, every comparison)
                           ZoneSessions (dwell per zone of every session, passed / skipped)
                           ZoneCells    (peak bin + zone + temporal stability of every unit in
                                         PLACE_CELL_DIR)
                           Ar_<arena>_*, Ses_<A><type>_*, AC_<arenas>_<A>_*  (v10 tables + TS_*)
  EdgeCentre_AllStats.csv  all statistics tables in one long-format CSV
  plus the box/bar plots and Histograms\\ of every comparison in the folders above.

INPUT_EXCEL, PLACE_CELL_DIR and TEMPORAL_STABILITY_EXCEL are only read.
"""

import os
import re
import sys

import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb
from scipy.ndimage import gaussian_filter1d

# CellClusteredStats_Utils.py lives one folder up, ArenaSessionType_StatsComparison_v10.py in
# its OldVers\ folder.
_CODES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_CODES_DIR, os.path.join(_CODES_DIR, 'OldVers')]
import ArenaSessionType_StatsComparison_v10 as asc   # noqa: E402
import CellClusteredStats_Utils as ccs               # noqa: E402

# ── Parameters ──────────────────────────────────────────────────────────────

INPUT_EXCEL    = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\PC_SmoothCompare\CrossCheck_DoubleRun\All_TT_PlaceChar_SiSkaggsAdptBin_Alpha200.xlsx'
PLACE_CELL_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Alpha200'
OUTPUT_DIR     = os.path.join(PLACE_CELL_DIR, 'Results', 'EdgeCentreZone')

# Output of TemporalStabilityMap_v4_OccWgh.py (PerField + PerBin sheets) on PLACE_CELL_DIR.
TEMPORAL_STABILITY_EXCEL = os.path.join(PLACE_CELL_DIR, 'Results', 'TemporalStabilityStats',
                                        'TemporalStability.xlsx')

# Temporal stability of each cell's highest-peak field, compared between the zones.
TS_METRICS = {
    'ts_mean_pct':        'Temporal stability, mean across the bins of the highest-peak field '
                          '(% visits with >= 1 spike)',
    'ts_median_pct':      'Temporal stability, median across the bins of the highest-peak field '
                          '(% visits with >= 1 spike)',
    'ts_occ_wmean_pct':   'Temporal stability, occupancy-weighted mean across the bins of the '
                          'highest-peak field (% visits with >= 1 spike)',
    'ts_occ_wmedian_pct': 'Temporal stability, occupancy-weighted median across the bins of the '
                          'highest-peak field (% visits with >= 1 spike)',
    'ts_pct_at_peak_bin': 'Temporal stability at the peak bin of the highest-peak field '
                          '(% visits with >= 1 spike)',
}

# A session is analysed only if BOTH zones hold >= this % of its dwell time.
MIN_ZONE_DWELL_PCT = 20.0

# Test for the continuous metrics ('KW' or 'LMM'; see ArenaSessionType_StatsComparison_v10.py).
CONTINUOUS_TEST = 'KW'

# Zones of each arena: (wall-side zone, inner zone), in plotting order.
ZONES = {
    'Open':   ('Edge', 'Centre'),
    'Linear': ('Edge', 'Centre'),
    'Circle': ('Outer', 'Inner'),
}

# Arena geometry (as in v27_5 ARENA_CONFIGS)
BIN_CM                        = 2.0
OPEN_DIAMETER_CM              = 60.0
OPEN_EDGE_ZONE_THRESHOLD_CM   = 9.25
LINEAR_LENGTH_CM              = 80.0
LINEAR_WIDTH_CM               = 8.0
LINEAR_EDGE_ZONE_THRESHOLD_CM = 2.0
CIRCLE_OUTER_DIAMETER_CM      = 80.0
CIRCLE_INNER_DIAMETER_CM      = 72.0

# Tracking and raw rate map (as in v27_5)
COORD_UNITS            = 'cm'     # 'cm': tracking file *_cm.csv (time, ., ., x, y); else pixels
FPS                    = 30
MAX_GAP_US             = 50_000   # max spike-to-frame gap (µs)
POS_JUMP_THRESH_CMS    = 90.0
POS_SMOOTH_SIGMA_SMP   = 5.0
CENTRE_OPEN_FIELD_TRACKING = True
SPEED_FILTER_ENABLED   = True
MIN_SPEED_CMS          = 0.5
MAX_SPEED_CMS          = 90.0
SPEED_FILTER_OCCUPANCY = True
USE_MIN_OCC_S          = False    # valid bins: >= MIN_OCC_S seconds occupancy
MIN_OCC_S              = 1.0
USE_MIN_VISITS         = True     # valid bins: >= MIN_VISITS separate visits
MIN_VISITS             = 2

# Zone colors in the plots: the inner zone (Centre / Inner) takes its arena's color from v10's
# ARENA_COLORS (Circle '#FA7FFA', Linear '#7FFAFA', Open '#7FB3E5'); the wall-side zone
# (Edge / Outer) a darker shade of it, every tone mixed with this fraction of black.
EDGE_DARKEN = 0.35

ntt_dtype = np.dtype([
    ('timestamp',   '<u8'),
    ('sc_number',   '<u4'),
    ('cell_number', '<u4'),
    ('params',      '<u4', (8,)),
    ('waveforms',   '<i2', (32, 4)),
])


# ── Arena geometry ──────────────────────────────────────────────────────────
# Every arena's bins are one flat index (bx * ny + by). Per bin: geom_valid (inside the arena),
# wall_zone (True = Edge / Outer), dist_to_wall (cm) and bin_x / bin_y (cm; open field: x, y;
# linear track: along, across the track; circle: arc length at the mean radius, distance from
# the inner wall).

class OpenFieldGeometry:
    def __init__(self):
        d = OPEN_DIAMETER_CM
        self.nx = self.ny = int(np.ceil(d / BIN_CM))
        self.n_bins = self.nx * self.ny
        self.cx = self.cy = d / 2.0
        self.arena_width_cm = d

        XX, YY = np.meshgrid((np.arange(self.nx) + 0.5) * BIN_CM,
                             (np.arange(self.ny) + 0.5) * BIN_CM, indexing='ij')
        r = np.hypot(XX - self.cx, YY - self.cy)
        self.geom_valid = (r <= d / 2.0).ravel()
        self.dist_to_wall = np.clip(d / 2.0 - r, 0.0, None).ravel()
        self.wall_zone = self.dist_to_wall <= OPEN_EDGE_ZONE_THRESHOLD_CM
        self.bin_x, self.bin_y = XX.ravel(), YY.ravel()

    def orient(self, x_cm, y_cm):
        return x_cm, y_cm

    def to_bins(self, x_cm, y_cm):
        bx = np.clip((x_cm / BIN_CM).astype(int), 0, self.nx - 1)
        by = np.clip((y_cm / BIN_CM).astype(int), 0, self.ny - 1)
        on_arena = np.hypot(x_cm - self.cx, y_cm - self.cy) <= OPEN_DIAMETER_CM / 2.0 + BIN_CM
        return bx * self.ny + by, on_arena


class CircularTrackGeometry:
    def __init__(self):
        self.outer_r = CIRCLE_OUTER_DIAMETER_CM / 2.0
        self.inner_r = CIRCLE_INNER_DIAMETER_CM / 2.0
        self.cx = self.cy = self.outer_r
        self.arena_width_cm = CIRCLE_OUTER_DIAMETER_CM
        self.radial_tol_cm = 4.0
        self.track_width_cm = self.outer_r - self.inner_r
        mean_r = (self.outer_r + self.inner_r) / 2.0

        circumference = 2 * np.pi * mean_r
        self.nx = max(8, 4 * int(round(circumference / BIN_CM / 4.0)))
        self.ny = max(2, int(round(self.track_width_cm / BIN_CM)))
        self.n_bins = self.nx * self.ny
        self.bin_width_deg = 360.0 / self.nx
        self.bin_cm_y = self.track_width_cm / self.ny
        self.geom_valid = np.ones(self.n_bins, dtype=bool)

        r_from_inner = (np.arange(self.ny) + 0.5) * self.bin_cm_y
        self.dist_to_wall = np.tile(np.minimum(r_from_inner, self.track_width_cm - r_from_inner),
                                    self.nx)
        self.wall_zone = np.tile(np.arange(self.ny) >= self.ny // 2, self.nx)   # outer ring
        self.bin_x = np.repeat((np.arange(self.nx) + 0.5) * circumference / self.nx, self.ny)
        self.bin_y = np.tile(r_from_inner, self.nx)

    def orient(self, x_cm, y_cm):
        return x_cm, y_cm

    def to_bins(self, x_cm, y_cm):
        r = np.hypot(x_cm - self.cx, y_cm - self.cy)
        theta = np.degrees(np.arctan2(y_cm - self.cy, x_cm - self.cx)) % 360.0
        bx = np.clip((theta / self.bin_width_deg).astype(int), 0, self.nx - 1)
        rel_r = np.clip(r - self.inner_r, 0.0, self.track_width_cm)
        by = np.clip((rel_r / self.bin_cm_y).astype(int), 0, self.ny - 1)
        on_track = ((r >= self.inner_r - self.radial_tol_cm)
                    & (r <= self.outer_r + self.radial_tol_cm))
        return bx * self.ny + by, on_track


class LinearTrackGeometry:
    def __init__(self):
        self.length, self.width = LINEAR_LENGTH_CM, LINEAR_WIDTH_CM
        self.nx = max(4, int(round(self.length / BIN_CM)))
        self.ny = max(2, int(round(self.width / BIN_CM)))
        self.n_bins = self.nx * self.ny
        self.bin_cm_x = self.length / self.nx
        self.bin_cm_y = self.width / self.ny
        self.arena_width_cm = self.length
        self.geom_valid = np.ones(self.n_bins, dtype=bool)

        XX, YY = np.meshgrid((np.arange(self.nx) + 0.5) * self.bin_cm_x,
                             (np.arange(self.ny) + 0.5) * self.bin_cm_y, indexing='ij')
        DX = np.minimum(XX, self.length - XX)     # distance from the nearer track end
        DY = np.minimum(YY, self.width - YY)      # distance from the nearer long wall
        self.dist_to_wall = np.minimum(DX, DY).ravel()
        dist_from_midline = self.width / 2.0 - DY
        self.wall_zone = ((dist_from_midline >= LINEAR_EDGE_ZONE_THRESHOLD_CM)
                          | (DX <= LINEAR_EDGE_ZONE_THRESHOLD_CM)).ravel()
        self.bin_x, self.bin_y = XX.ravel(), YY.ravel()

    def orient(self, x_cm, y_cm):
        """Vertical sessions rotated 90 deg CCW so the track runs along x."""
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


def make_geometries() -> dict:
    return {'Open': OpenFieldGeometry(), 'Linear': LinearTrackGeometry(),
            'Circle': CircularTrackGeometry()}


# ── Tracking load / clean / smooth (as in v27_5) ────────────────────────────

def _load_tracking(csv_path: str, arena_width_cm: float) -> tuple:
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

    dxy = np.hypot(np.append(np.diff(x), 0), np.append(np.diff(y), 0))
    dt = np.append(np.diff(t), 1)
    valid_dt = dt > 0
    speed = np.zeros_like(dxy)
    speed[valid_dt] = dxy[valid_dt] / dt[valid_dt]
    keep = np.where(valid_dt & (speed < 0.006))[0]
    x, y, t = x[keep], y[keep], t[keep]

    order = np.argsort(t)
    x, y, t = x[order], y[order], t[order]
    if len(t) == 0:
        return x, y, t

    if COORD_UNITS == 'cm':
        return x - x.min(), y - y.min(), t
    px_per_cm = max(x.max() - x.min(), y.max() - y.min()) / arena_width_cm
    return (x - x.min()) / px_per_cm, (y - y.min()) / px_per_cm, t


def _smooth_tracking_position(x_cm: np.ndarray, y_cm: np.ndarray, t_us: np.ndarray) -> tuple:
    """Frames after a > POS_JUMP_THRESH_CMS jump are dropped and interpolated, then the
    trajectory is Gaussian smoothed (sigma POS_SMOOTH_SIGMA_SMP samples)."""
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


def _speed_mask(x_cm: np.ndarray, y_cm: np.ndarray, t_us: np.ndarray) -> np.ndarray:
    n = len(t_us)
    if not SPEED_FILTER_ENABLED or n < 2:
        return np.ones(n, dtype=bool)
    dt_s = np.diff(t_us) * 1e-6
    step_cm = np.hypot(np.diff(x_cm), np.diff(y_cm))
    speed = np.full(n - 1, np.nan)
    ok = dt_s > 0
    speed[ok] = step_cm[ok] / dt_s[ok]
    speed = np.concatenate(([speed[0]], speed))
    return (speed >= MIN_SPEED_CMS) & (speed <= MAX_SPEED_CMS)


def _bin_visit_counts(flat_idx: np.ndarray, n_bins: int) -> np.ndarray:
    """Visits per bin: a visit starts whenever the animal enters a bin."""
    if flat_idx.size == 0:
        return np.zeros(n_bins, dtype=np.int64)
    starts = np.ones(flat_idx.size, dtype=bool)
    starts[1:] = flat_idx[1:] != flat_idx[:-1]
    return np.bincount(flat_idx[starts], minlength=n_bins)


# ── Session occupancy, zone dwell and per-cell raw-map peak ─────────────────

def session_occupancy(csv_path: str, geom, arena: str) -> dict | None:
    """One session's on-arena frames (time, flat bin, speed-filter flag), occupancy map (s) and
    valid bins. Every cell of a session shares these. None if the tracking is unusable."""
    x, y, t = _load_tracking(csv_path, geom.arena_width_cm)
    if len(t) < 2:
        return None
    x, y = _smooth_tracking_position(x, y, t)
    if arena == 'Open' and CENTRE_OPEN_FIELD_TRACKING:
        x = x + (geom.cx - 0.5 * (x.min() + x.max()))
        y = y + (geom.cy - 0.5 * (y.min() + y.max()))
    x, y = geom.orient(x, y)
    moving = _speed_mask(x, y, t)
    bin_idx, on_arena = geom.to_bins(x, y)
    t, bin_idx, moving = t[on_arena], bin_idx[on_arena], moving[on_arena]
    if len(t) < 2:
        return None

    dt = np.empty(len(t), dtype=np.float64)
    dt[0] = 1.0 / FPS
    dt[1:] = np.minimum(np.diff(t) * 1e-6, 2.0 / FPS)
    occ_frame = moving if SPEED_FILTER_OCCUPANCY else np.ones(len(t), dtype=bool)

    occ_map = np.zeros(geom.n_bins, dtype=np.float64)
    np.add.at(occ_map, bin_idx[occ_frame], dt[occ_frame])
    visits = _bin_visit_counts(bin_idx[occ_frame], geom.n_bins)

    valid = (occ_map > 0) & geom.geom_valid
    if USE_MIN_OCC_S:
        valid &= occ_map >= MIN_OCC_S
    if USE_MIN_VISITS:
        valid &= visits >= MIN_VISITS
    return dict(t=t, bin_idx=bin_idx, moving=moving, occ_map=occ_map, valid=valid)


def zone_dwell_pct(geom, sess: dict) -> tuple:
    """(% dwell in the wall-side zone, % in the inner zone, total dwell s), over valid bins."""
    occ, valid = sess['occ_map'], sess['valid']
    total = float(occ[valid].sum())
    if total <= 0:
        return np.nan, np.nan, 0.0
    wall = float(occ[valid & geom.wall_zone].sum())
    return 100.0 * wall / total, 100.0 * (total - wall) / total, total


def raw_map_peak(sess: dict, spike_ts: np.ndarray, geom) -> dict:
    """Peak bin of the raw rate map (spikes / occupancy per valid bin) of one cell."""
    t = sess['t']
    idx = np.searchsorted(t, spike_ts, side='left')
    idx_l = np.clip(idx - 1, 0, len(t) - 1)
    idx_r = np.clip(idx,     0, len(t) - 1)
    dist_l = np.abs(spike_ts - t[idx_l])
    dist_r = np.abs(spike_ts - t[idx_r])
    nearest = np.where(dist_l <= dist_r, idx_l, idx_r)
    ok = (np.minimum(dist_l, dist_r) <= MAX_GAP_US) & sess['moving'][nearest]

    spike_map = np.bincount(sess['bin_idx'][nearest[ok]], minlength=geom.n_bins).astype(float)
    valid, occ = sess['valid'], sess['occ_map']
    rate = np.zeros(geom.n_bins, dtype=np.float64)
    rate[valid] = spike_map[valid] / occ[valid]

    out = dict(n_spikes_raw_map=int(ok.sum()), peak_bin=None, peak_rate_raw_hz=np.nan,
               peak_bin_x_cm=np.nan, peak_bin_y_cm=np.nan, peak_dist_from_wall_cm=np.nan,
               peak_in_wall_zone=None)
    if not valid.any() or rate[valid].max() <= 0:
        return out
    peak = int(np.argmax(np.where(valid, rate, -np.inf)))
    out.update(peak_bin=peak, peak_rate_raw_hz=round(float(rate[peak]), 4),
               peak_bin_x_cm=round(float(geom.bin_x[peak]), 3),
               peak_bin_y_cm=round(float(geom.bin_y[peak]), 3),
               peak_dist_from_wall_cm=round(float(geom.dist_to_wall[peak]), 3),
               peak_in_wall_zone=bool(geom.wall_zone[peak]))
    return out


def _load_spike_ts(ntt_path: str) -> np.ndarray:
    spikes = np.memmap(ntt_path, dtype=ntt_dtype, mode='r', offset=16 * 1024)
    spikes = spikes[spikes['cell_number'] != 0]
    return np.sort(spikes['timestamp'].astype(np.float64))


# ── Matching cells between INPUT_EXCEL and PLACE_CELL_DIR ───────────────────

def _session_key(path) -> str:
    """Session path from the animal folder onwards, lower case, '/'-separated
    ('X:\\...\\Fa1059\\Linear\\Day10\\1_0' and 'Fa1059\\Linear\\Day10\\1_0' -> 'fa1059/linear/day10/1_0')."""
    parts = [p.strip().lower() for p in re.split(r'[\\/]+', str(path).strip())
             if p.strip() not in ('', '.')]
    for i, p in enumerate(parts):
        if re.match(ccs.ANIMAL_PATTERN, p, flags=re.IGNORECASE):
            return '/'.join(parts[i:])
    return '/'.join(parts)


def _unit_key(unit) -> str:
    return os.path.basename(str(unit).strip()).lower()


def _tracking_files(filenames: list) -> list:
    tracking = [f for f in filenames if f.lower().endswith(('.csv', '.xlsx'))]
    if COORD_UNITS == 'cm':
        return [f for f in tracking if f.lower().endswith('_cm.csv')]
    return [f for f in tracking if not f.lower().endswith('_cm.csv')]


def scan_place_cell_dir(geoms: dict) -> tuple:
    """Zone dwell of every session folder in PLACE_CELL_DIR and the raw-map peak bin / zone of
    every .ntt unit in it. Returns (sessions_df, cells_df)."""
    out_dir = os.path.normcase(os.path.abspath(OUTPUT_DIR))
    folders = []
    for dirpath, dirnames, filenames in os.walk(PLACE_CELL_DIR):
        dirnames[:] = [d for d in dirnames
                       if os.path.normcase(os.path.abspath(os.path.join(dirpath, d))) != out_dir]
        ntt_files = sorted(f for f in filenames if f.lower().endswith('.ntt'))
        if ntt_files:
            folders.append((dirpath, _tracking_files(filenames), ntt_files))

    session_rows, cell_rows = [], []
    for i, (dirpath, tracking, ntt_files) in enumerate(sorted(folders), start=1):
        rel = os.path.relpath(dirpath, PLACE_CELL_DIR)
        arena = asc.extract_arena_type(rel)
        srow = dict(session=rel, _skey=_session_key(rel), arena_type=arena,
                    session_type=asc.extract_session_type(rel, arena),
                    animal=ccs._path_token(rel, ccs.ANIMAL_PATTERN), n_units=len(ntt_files),
                    total_dwell_s=np.nan, wall_zone_dwell_pct=np.nan,
                    inner_zone_dwell_pct=np.nan, passes_dwell=False, note='')
        print(f'[{i}/{len(folders)}] {rel}  ({arena}, {len(ntt_files)} unit(s))')

        sess = None
        if arena is None:
            srow['note'] = 'no recognizable arena in path'
        elif len(tracking) != 1:
            srow['note'] = f'{len(tracking)} tracking files (need exactly 1)'
        else:
            try:
                sess = session_occupancy(os.path.join(dirpath, tracking[0]), geoms[arena], arena)
            except Exception as e:
                srow['note'] = f'tracking error: {e}'
            if sess is None and not srow['note']:
                srow['note'] = 'unusable tracking'
        if sess is not None:
            wall_pct, inner_pct, total_s = zone_dwell_pct(geoms[arena], sess)
            passes = bool(wall_pct >= MIN_ZONE_DWELL_PCT and inner_pct >= MIN_ZONE_DWELL_PCT)
            srow.update(total_dwell_s=round(total_s, 2), wall_zone_dwell_pct=round(wall_pct, 2),
                        inner_zone_dwell_pct=round(inner_pct, 2), passes_dwell=passes,
                        note='' if passes else f'a zone has < {MIN_ZONE_DWELL_PCT:g} % dwell')
        if srow['note']:
            print(f'    SKIPPED: {srow["note"]}')
        session_rows.append(srow)

        for ntt in ntt_files:
            crow = dict(session=rel, unit=ntt, _skey=srow['_skey'], _ukey=_unit_key(ntt),
                        arena_type=arena, session_type=srow['session_type'],
                        animal=srow['animal'], session_passes_dwell=srow['passes_dwell'],
                        wall_zone_dwell_pct=srow['wall_zone_dwell_pct'],
                        inner_zone_dwell_pct=srow['inner_zone_dwell_pct'],
                        peak_zone=None, zone=None, excluded_because=srow['note'])
            if sess is not None:
                try:
                    peak = raw_map_peak(sess, _load_spike_ts(os.path.join(dirpath, ntt)),
                                        geoms[arena])
                except Exception as e:
                    peak = {}
                    crow['excluded_because'] = f'spike/rate map error: {e}'
                crow.update(peak)
                if peak.get('peak_in_wall_zone') is not None:
                    crow['peak_zone'] = ZONES[arena][0 if peak['peak_in_wall_zone'] else 1]
                elif not crow['excluded_because']:
                    crow['excluded_because'] = 'no spikes in valid bins'
                if srow['passes_dwell']:
                    crow['zone'] = crow['peak_zone']
            cell_rows.append(crow)

    cells_df = pd.DataFrame(cell_rows).drop(columns=['peak_bin', 'peak_in_wall_zone'],
                                            errors='ignore')
    return pd.DataFrame(session_rows), cells_df


def attach_zones(df: pd.DataFrame, cells_df: pd.DataFrame, what: str) -> pd.DataFrame:
    """Add the 'zone' (+ peak-bin) columns of cells_df to the Excel rows in df; rows with no zone
    (unit not in PLACE_CELL_DIR, session skipped, no spikes) are dropped, with a report."""
    df = df.copy()
    df['_skey'] = df['session'].map(_session_key)
    df['_ukey'] = df['unit'].map(_unit_key)
    zcols = ['_skey', '_ukey', 'zone', 'peak_zone', 'peak_rate_raw_hz', 'peak_bin_x_cm',
             'peak_bin_y_cm', 'peak_dist_from_wall_cm', 'wall_zone_dwell_pct',
             'inner_zone_dwell_pct', 'excluded_because']
    z = cells_df[[c for c in zcols if c in cells_df.columns]]
    dup = z.duplicated(['_skey', '_ukey'], keep='first')
    if dup.any():
        print(f'WARNING: {int(dup.sum())} unit(s) appear twice in PLACE_CELL_DIR under the same '
              f'animal-relative path; the first is used.')
        z = z[~dup]
    merged = df.merge(z, on=['_skey', '_ukey'], how='left', indicator=True)

    not_found = merged['_merge'] == 'left_only'
    no_zone = ~not_found & merged['zone'].isna()
    print(f'\n{what}: {len(merged)} place-cell row(s) in the Excel file; '
          f'{int(merged["zone"].notna().sum())} assigned to a zone, '
          f'{int(not_found.sum())} not found in PLACE_CELL_DIR, {int(no_zone.sum())} excluded:')
    for reason, n in merged.loc[no_zone, 'excluded_because'].fillna('').value_counts().items():
        print(f'    {n:4d}  {reason or "unknown"}')
    if not_found.any():
        print('  Not found in PLACE_CELL_DIR (first 20):')
        for s, u in merged.loc[not_found, ['session', 'unit']].head(20).itertuples(index=False):
            print(f'    {s} | {u}')

    out = merged[merged['zone'].notna()].drop(columns=['_merge'])
    bad = out.apply(lambda r: r['zone'] not in ZONES.get(r['arena_type'], ()), axis=1)
    if bad.any():
        print(f'WARNING: {int(bad.sum())} row(s) whose Excel arena does not match their zone; dropped.')
        out = out[~bad]
    return out


# ── Temporal stability of each cell's highest-peak field ────────────────────

def load_temporal_stability(path: str) -> pd.DataFrame:
    """One row per cell of TemporalStabilityMap_v4_OccWgh.py's output: its highest-peak field
    (the field holding the field bin with the highest raw rate, n_spikes / occupancy_s; first
    bin on ties), that field's peak bin, and the TS_METRICS of both. Keyed by _skey / _ukey."""
    bins = pd.read_excel(path, sheet_name='PerBin')
    fields = pd.read_excel(path, sheet_name='PerField')

    bins['ts_peak_bin_rate_hz'] = bins['n_spikes'] / bins['occupancy_s'].where(bins['occupancy_s'] > 0)
    bins = bins.dropna(subset=['ts_peak_bin_rate_hz'])
    peak = bins.loc[bins.groupby(['session', 'unit'], sort=False)['ts_peak_bin_rate_hz'].idxmax()]
    peak = peak[['session', 'unit', 'field_number', 'ts_peak_bin_rate_hz', 'n_visits',
                 'pct_visits_with_spike']].rename(columns={
        'field_number': 'ts_field_number', 'n_visits': 'ts_peak_bin_n_visits',
        'pct_visits_with_spike': 'ts_pct_at_peak_bin'})

    n_fields = fields.groupby(['session', 'unit']).size().rename('ts_n_fields').reset_index()
    fields = fields[['session', 'unit', 'field_number', 'n_field_bins', 'mean_pct_across_bins',
                     'median_pct_across_bins', 'occ_wmean_pct_across_bins',
                     'occ_wmedian_pct_across_bins']].rename(columns={
        'field_number': 'ts_field_number', 'n_field_bins': 'ts_n_field_bins',
        'mean_pct_across_bins': 'ts_mean_pct', 'median_pct_across_bins': 'ts_median_pct',
        'occ_wmean_pct_across_bins': 'ts_occ_wmean_pct',
        'occ_wmedian_pct_across_bins': 'ts_occ_wmedian_pct'})

    out = (peak.merge(fields, on=['session', 'unit', 'ts_field_number'], how='left')
               .merge(n_fields, on=['session', 'unit'], how='left'))
    out['ts_peak_bin_rate_hz'] = out['ts_peak_bin_rate_hz'].round(4)
    out['_skey'] = out['session'].map(_session_key)
    out['_ukey'] = out['unit'].map(_unit_key)
    dup = out.duplicated(['_skey', '_ukey'], keep='first')
    if dup.any():
        print(f'WARNING: {int(dup.sum())} unit(s) appear twice in TEMPORAL_STABILITY_EXCEL under '
              f'the same animal-relative path; the first is used.')
    out = out[~dup].drop(columns=['session', 'unit'])
    print(f'Temporal stability: {len(out)} cell(s) with a field in {path}')
    return out


# ── Comparisons ─────────────────────────────────────────────────────────────

def zone_colors(arena: str) -> dict:
    """GROUP_COLORS entries of the two zones of `arena`: inner zone = the arena's color, wall-side
    zone = the same tones darkened by EDGE_DARKEN."""
    base = asc.ARENA_COLORS[arena]
    wall, inner = ZONES[arena]
    darker = {k: to_hex(tuple(c * (1.0 - EDGE_DARKEN) for c in to_rgb(base[k])))
              for k in ('face', 'dark', 'light')}
    return {inner: base, wall: asc._shade(darker['face'], darker['dark'], darker['light'])}


def _zone_count_row(info: dict, arena: str, df: pd.DataFrame, sessions_df: pd.DataFrame) -> dict:
    """Place-cell rows / unique cells per zone and the mean session dwell per zone of the
    sessions those cells come from."""
    wall, inner = ZONES[arena]
    sess = sessions_df[sessions_df['_skey'].isin(df['_skey'].unique())]
    row = dict(info, wall_zone=wall, inner_zone=inner, n_sessions=len(sess))
    for z in (wall, inner):
        sub = df[df['zone'] == z]
        row[f'n_rows_{z}'] = len(sub)
        row[f'n_cells_{z}'] = sub['cell_id'].nunique()
    n_rows = len(df)
    row['pct_rows_wall_zone'] = round(100.0 * row[f'n_rows_{wall}'] / n_rows, 2) if n_rows else np.nan
    row['mean_session_dwell_pct_wall_zone'] = round(float(sess['wall_zone_dwell_pct'].mean()), 2) \
        if len(sess) else np.nan
    row['mean_session_dwell_pct_inner_zone'] = round(float(sess['inner_zone_dwell_pct'].mean()), 2) \
        if len(sess) else np.nan
    return row


def compare_temporal_stability(grp, df: pd.DataFrame, plots_dir: str) -> dict:
    """Zone comparison of every TS_METRICS metric (v10's compare_groups, as in
    TemporalStabilityMap_v4_OccWgh._compare): a box plot per metric in `plots_dir` and a
    histogram in `plots_dir`\\Histograms. Cells with no temporal-stability field are left out.
    Returns {'TS_Desc', 'TS_Omnibus', 'TS_PostHoc'}."""
    df = df.dropna(subset=list(TS_METRICS), how='all')
    print(f'\n{grp.title} -- temporal stability of the highest-peak field; place cells per zone:')
    print(df[grp.col].value_counts().reindex(grp.order).fillna(0).astype(int))
    if not len(df):
        print('No place cells with temporal stability; skipping.')
        return {}

    hist_dir = os.path.join(plots_dir, 'Histograms')
    os.makedirs(hist_dir, exist_ok=True)
    all_desc, all_omnibus, all_posthoc = [], [], []
    for col, label in TS_METRICS.items():
        desc_df, omnibus, posthoc_df = asc.compare_groups(df, col, label, grp)
        all_desc.append(desc_df)
        all_omnibus.append(omnibus)
        if len(posthoc_df):
            all_posthoc.append(posthoc_df)
        asc.plot_metric(df, col, label, os.path.join(plots_dir, f'TemporalStability_{col}.png'), grp,
                        stats_lines=asc._continuous_stats_lines(desc_df, omnibus, posthoc_df, grp),
                        posthoc_df=posthoc_df)
        asc.plot_histogram_by_group(df, col, label,
                                    os.path.join(hist_dir, f'Hist_TemporalStability_{col}.png'), grp)

    omnibus_df = pd.DataFrame(all_omnibus)
    asc._print_significant(omnibus_df, f'{grp.title}, temporal stability', 45, 16)
    return {
        'TS_Desc':    pd.concat(all_desc, ignore_index=True),
        'TS_Omnibus': omnibus_df,
        'TS_PostHoc': pd.concat(all_posthoc, ignore_index=True) if all_posthoc else pd.DataFrame(),
    }


def run_zone_comparison(info: dict, arena: str, full_df: pd.DataFrame, field_df: pd.DataFrame,
                        title: str, plots_dir: str, sheet_prefix: str,
                        sessions_df: pd.DataFrame, tables: dict, counts: list):
    """Edge vs centre (Outer vs Inner) comparison of one subset of cells with v10's
    run_comparison and of their temporal stability (compare_temporal_stability); the tables are
    stored in `tables` under '<sheet_prefix>_<name>' with the subset description (`info`) in the
    first columns."""
    counts.append(_zone_count_row(info, arena, full_df, sessions_df))
    asc.GROUP_COLORS.update(zone_colors(arena))   # Open and Linear share the zone names
    grp = asc.Grouping('zone', list(ZONES[arena]), 'zones', title)
    results = asc.run_comparison(grp, full_df, field_df, full_df, plots_dir)
    if len(full_df):
        results.update(compare_temporal_stability(grp, full_df, plots_dir))
    for name, tbl in results.items():
        if len(tbl):
            for i, (k, v) in enumerate(info.items()):
                if k not in tbl.columns:
                    tbl.insert(i, k, v)
        tables[f'{sheet_prefix}_{name}'] = tbl


def _write_workbook(path: str, sheets: dict):
    short = {}
    for name in sheets:
        s = name[:31]
        if s in short.values():
            raise ValueError(f'Sheet name {s!r} (from {name!r}) is not unique after Excel\'s '
                             f'31-character limit.')
        short[name] = s
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for name, tbl in sheets.items():
            tbl.to_excel(writer, sheet_name=short[name], index=False)


def main():
    asc.CONTINUOUS_TEST = CONTINUOUS_TEST
    asc.CATEGORICAL_METRICS_ALL_UNITS = {}   # place-cell yield: non-place cells have no zone

    print(f'Scanning {PLACE_CELL_DIR} ...')
    sessions_df, cells_df = scan_place_cell_dir(make_geometries())
    n_pass = int(sessions_df['passes_dwell'].sum())
    print(f'\n{len(sessions_df)} session folder(s), {n_pass} pass the dwell criterion '
          f'(both zones >= {MIN_ZONE_DWELL_PCT:g} %).')

    print(f'\nLoading {INPUT_EXCEL} ...')
    full = pd.read_excel(INPUT_EXCEL, sheet_name='Full')
    fields = pd.read_excel(INPUT_EXCEL, sheet_name='PlaceFields')
    full_df = attach_zones(asc.build_full_metrics_df(full), cells_df, 'Full sheet')
    field_df = attach_zones(asc.build_field_metrics_df(full, fields), cells_df, 'PlaceFields sheet')

    print(f'\nLoading {TEMPORAL_STABILITY_EXCEL} ...')
    ts_df = load_temporal_stability(TEMPORAL_STABILITY_EXCEL)
    cells_df = cells_df.merge(ts_df, on=['_skey', '_ukey'], how='left')
    full_df = full_df.merge(ts_df, on=['_skey', '_ukey'], how='left')

    print('\nPlace cells by arena and zone (Full sheet):')
    print(pd.crosstab(full_df['arena_type'], full_df['zone'], margins=True))
    print('\nOf those, with temporal stability of their highest-peak field:')
    has_ts = full_df[full_df['ts_field_number'].notna()]
    print(pd.crosstab(has_ts['arena_type'], has_ts['zone'], margins=True))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tables, counts = {}, []

    # ── 1. Each arena, all its cells ────────────────────────────────────────
    for arena in asc.ARENA_TYPES:
        def _sel(d):
            return d[d['arena_type'] == arena]
        info = dict(level='Arena', arena_type=arena, session_type='all', animals='all')
        run_zone_comparison(info, arena, _sel(full_df), _sel(field_df), f'{arena} arena',
                            os.path.join(OUTPUT_DIR, 'Arena', arena), f'Ar_{arena}',
                            sessions_df, tables, counts)

    # ── 2. Each session type of each arena ──────────────────────────────────
    for arena in asc.ARENA_TYPES:
        for stype in asc.ARENA_SESSION_TYPES[arena]:
            def _sel(d):
                return d[(d['arena_type'] == arena) & (d['session_type'] == stype)]
            info = dict(level='SessionType', arena_type=arena, session_type=stype, animals='all')
            run_zone_comparison(info, arena, _sel(full_df), _sel(field_df),
                                f'{arena} arena, {asc._label(stype)}',
                                os.path.join(OUTPUT_DIR, 'SessionType', f'{arena}_{stype}'),
                                f'Ses_{arena[0]}{stype}', sessions_df, tables, counts)

    # ── 3. Animal-controlled: each arena, only the animals recorded in all compared arenas ──
    for name, (animals, arenas) in asc.ANIMAL_CONTROLLED_COMPARISONS.items():
        wanted = {a.lower() for a in animals}
        ordered = [a for a in asc.ARENA_TYPES if a in arenas]
        tag = 'AC_' + ''.join(a[0] for a in ordered)      # e.g. 'AC_CO'
        for arena in ordered:
            def _sel(d):
                return d[(d['arena_type'] == arena)
                         & d['animal'].astype(str).str.lower().isin(wanted)]
            info = dict(level='AnimalControlled', arena_type=arena, session_type='all',
                        animals=', '.join(animals), comparison=name)
            run_zone_comparison(info, arena, _sel(full_df), _sel(field_df),
                                f'{" + ".join(animals)} only, {arena} arena',
                                os.path.join(OUTPUT_DIR, asc.ANIMAL_CONTROLLED_DIR, name, arena),
                                f'{tag}_{arena[0]}', sessions_df, tables, counts)

    # ── Save ────────────────────────────────────────────────────────────────
    counts_df = pd.DataFrame(counts)
    sheets = {'ZoneCounts': counts_df,
              'ZoneSessions': sessions_df.drop(columns=['_skey']),
              'ZoneCells': cells_df.drop(columns=['_skey', '_ukey'])}
    sheets.update(tables)
    stats_xlsx = os.path.join(OUTPUT_DIR, 'EdgeCentre_Stats.xlsx')
    _write_workbook(stats_xlsx, sheets)
    print(f'\nZone assignment and all statistics saved to {stats_xlsx}')

    parts = []
    for name, tbl in tables.items():
        if tbl is not None and len(tbl):
            tbl = tbl.copy()
            tbl.insert(0, 'table', name)
            parts.append(tbl)
    if parts:
        stats_csv = os.path.join(OUTPUT_DIR, 'EdgeCentre_AllStats.csv')
        pd.concat(parts, ignore_index=True).to_csv(stats_csv, index=False, encoding='utf-8-sig')
        print(f'All statistics also saved to {stats_csv}')

    print('\nCells per zone:')
    print(counts_df.to_string(index=False))
    print(f'\nDone. Plots saved under {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
