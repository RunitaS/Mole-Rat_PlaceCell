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

COMPARISONS -- Edge vs Centre (Open, Linear) / Outer vs Inner (Circle), tested within each subset
of cells (panel) separately:
  1. Arena            : all cells of each arena                    -> OUTPUT_DIR\\Arena
  2. Session type     : the cells of each session type of an arena -> OUTPUT_DIR\\SessionType
  3. Animal-controlled: for each entry of v10's ANIMAL_CONTROLLED_COMPARISONS, each of its arenas,
                        only the cells of its animals              -> OUTPUT_DIR\\AnimalControlledArenaType\\<name>
All the panels of a comparison share ONE plot per metric: the two zones of every panel side by
side (wall-side zone in a darker shade of the arena's color, inner zone in the arena's color),
with the panel's significance bracket above its pair. Histograms (<folder>\\Histograms): one panel
per subset with both zones on shared bins, one row per arena (speed histograms: one row per arena
and zone).

ZONE SCHEMATICS: OUTPUT_DIR\\ZoneSchematic_<arena>.png, the 2 x 2 cm bins of each arena shaded by
zone in the plot colors.

Metrics and tests are those of ArenaSessionType_StatsComparison_v10.py, which is imported (not
copied) with 'zone' as the grouping column: continuous metrics (peak_fr, sir,
sparsity, coherence, stability score [+ per stability type], speed score, mean_fr, field number /
area / % area) with CONTINUOUS_TEST ('KW' = Mann-Whitney U for two zones, or 'LMM'), proportions
(theta modulation, coherence shuffle, stability significance, speed cells) and the speed-direction
classification with chi-square tests. Place-cell yield is left out: only place cells are in
PLACE_CELL_DIR, so non-place cells have no zone.

TEMPORAL STABILITY (TS_METRICS, every panel above): from TEMPORAL_STABILITY_EXCEL, the
output of TemporalStabilityMap_v6_MapPassStability.py (per field bin: % of visits with >= 1
spike). Except for ts_occ_wmean_pct_all_bins, only each cell's HIGHEST-PEAK FIELD is used: the
field holding the field bin with the highest raw rate (n_spikes / occupancy_s of the PerBin
sheet); that bin is the field's peak bin. Per cell:
  ts_mean_pct         mean   % across the bins of that field (PerField mean_pct_across_bins)
  ts_median_pct       median % across the bins of that field (PerField median_pct_across_bins)
  ts_occ_wmean_pct    occupancy-weighted mean / median % across the bins of that field, each bin
  ts_occ_wmedian_pct  weighted by its dwell time (PerField occ_wmean_pct_across_bins /
                      occ_wmedian_pct_across_bins), so poorly sampled bins count less
  ts_occ_wmean_pct_mapnorm   sum over the bins of that field of % x dwell, divided by the dwell
                      of ALL valid bins of the map (= ts_occ_wmean_pct x the fraction of map
                      dwell spent in the field; PerField occ_wmean_pct_mapnorm)
  ts_occ_wmean_pct_all_bins  occupancy-weighted mean % over EVERY visited valid bin of the map,
                      in or out of a field (PerCell occ_wmean_pct_all_bins; not field-specific)
  ts_pct_at_peak_bin  % at the field's peak bin (PerBin pct_visits_with_spike)
  LAP (PASS) STABILITY: a pass is one entry -> exit run through a field; it is stable if it holds
  >= LAP_MIN_SPIKES spikes (v6's MIN_SPIKES_PER_PASS). Score = stable passes / passes (0-1).
  ts_lap_stability            that field's score (PerField lap_stability_score)
  ts_lap_stability_pooled     passes through EVERY field of the cell pooled (PerCell
                              lap_stability_score; not field-specific)
  ts_lap_stability_field_mean mean of the per-field scores of the cell, every field weighted
                              equally (PerCell mean_lap_stability_across_fields)
The two map-wide measures have no median: rescaling the weights leaves a weighted median
unchanged, and PerBin holds field bins only. The cell's zone is still that of its raw-map peak
bin above. Same CONTINUOUS_TEST, box plots
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
  ZoneSchematic_<arena>.png
  plus the box/bar plots and Histograms\\ of every comparison in the folders above.

INPUT_EXCEL, PLACE_CELL_DIR and TEMPORAL_STABILITY_EXCEL are only read.
"""

import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Circle, Patch, Rectangle, Wedge
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter1d

# CellClusteredStats_Utils.py lives one folder up, ArenaSessionType_StatsComparison_v10.py in
# its OldVers\ folder.
_CODES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_CODES_DIR, os.path.join(_CODES_DIR, 'OldVers')]
import ArenaSessionType_StatsComparison_v10 as asc   # noqa: E402  (sets the Agg backend)
import CellClusteredStats_Utils as ccs               # noqa: E402
import matplotlib.pyplot as plt                      # noqa: E402

# ── Parameters ──────────────────────────────────────────────────────────────

INPUT_EXCEL    = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\PC_SmoothCompare\CrossCheck_DoubleRun\All_TT_PlaceChar_SiSkaggsAdptBin_Alpha200.xlsx'
PLACE_CELL_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Alpha200'
OUTPUT_DIR     = os.path.join(PLACE_CELL_DIR, 'Results', 'EdgeCentreZone_SomatosenControl')

# Output of TemporalStabilityMap_v6_MapPassStability.py (PerCell + PerField + PerBin sheets) on
# PLACE_CELL_DIR.
TEMPORAL_STABILITY_EXCEL = os.path.join(PLACE_CELL_DIR, 'Results', 'TemporalStability.xlsx')

# MIN_SPIKES_PER_PASS of the v6 run that made TEMPORAL_STABILITY_EXCEL (plot labels only: the
# lap stability scores are read from the workbook).
LAP_MIN_SPIKES = 5

# Temporal stability of each cell's highest-peak field (ts_occ_wmean_pct_all_bins: whole map),
# compared between the zones.
TS_METRICS = {
    'ts_mean_pct':        'Temporal stability, mean across the bins of the highest-peak field '
                          '(% visits with >= 1 spike)',
    'ts_median_pct':      'Temporal stability, median across the bins of the highest-peak field '
                          '(% visits with >= 1 spike)',
    'ts_occ_wmean_pct':   'Temporal stability, occupancy-weighted mean across the bins of the '
                          'highest-peak field (% visits with >= 1 spike)',
    'ts_occ_wmedian_pct': 'Temporal stability, occupancy-weighted median across the bins of the '
                          'highest-peak field (% visits with >= 1 spike)',
    'ts_occ_wmean_pct_mapnorm':  'Temporal stability of the highest-peak field weighted by dwell, '
                                 'normalised to whole-map dwell (%)',
    'ts_occ_wmean_pct_all_bins': 'Temporal stability, occupancy-weighted mean across all visited '
                                 'map bins (% visits with >= 1 spike)',
    'ts_pct_at_peak_bin':'Temporal stability at the peak bin of the highest-peak field '
                          '(% visits with >= 1 spike)',
    'ts_lap_stability':   f'Lap stability of the highest-peak field (fraction of passes with '
                          f'>= {LAP_MIN_SPIKES} spikes)',
    'ts_lap_stability_pooled':     f'Lap stability, passes pooled over all fields (fraction of '
                                   f'passes with >= {LAP_MIN_SPIKES} spikes)',
    'ts_lap_stability_field_mean': f'Mean lap stability across fields (fraction of passes with '
                                   f'>= {LAP_MIN_SPIKES} spikes)',
}

# A session is analysed only if BOTH zones hold >= this % of its dwell time.
MIN_ZONE_DWELL_PCT = 40.0

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
OPEN_EDGE_ZONE_THRESHOLD_CM   = 4#9.25
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

# Combined box/bar plots: x-axis width per zone box (inches) and the gap between panels (in box
# widths).
ZONE_BOX_W = 0.75
PANEL_GAP  = 0.7

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
    """One row per cell of TemporalStabilityMap_v6_MapPassStability.py's output: its highest-peak
    field (the field holding the field bin with the highest raw rate, n_spikes / occupancy_s;
    first bin on ties), that field's peak bin, the TS_METRICS of both and the cell's whole-map
    occupancy-weighted mean (PerCell). Keyed by _skey / _ukey."""
    bins = pd.read_excel(path, sheet_name='PerBin')
    fields = pd.read_excel(path, sheet_name='PerField')
    cells = pd.read_excel(path, sheet_name='PerCell')
    for sheet, df, cols in (('PerField', fields, ['occ_wmean_pct_mapnorm', 'n_passes',
                                                  'n_passes_with_min_spikes', 'lap_stability_score']),
                            ('PerCell', cells, ['occ_wmean_pct_all_bins', 'lap_stability_score',
                                                'mean_lap_stability_across_fields'])):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f'{sheet} sheet of {path} has no column(s) {missing}; it must be the '
                           f'output of TemporalStabilityMap_v6_MapPassStability.py.')

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
                     'occ_wmedian_pct_across_bins', 'occ_wmean_pct_mapnorm', 'n_passes',
                     'n_passes_with_min_spikes', 'lap_stability_score']].rename(columns={
        'field_number': 'ts_field_number', 'n_field_bins': 'ts_n_field_bins',
        'mean_pct_across_bins': 'ts_mean_pct', 'median_pct_across_bins': 'ts_median_pct',
        'occ_wmean_pct_across_bins': 'ts_occ_wmean_pct',
        'occ_wmedian_pct_across_bins': 'ts_occ_wmedian_pct',
        'occ_wmean_pct_mapnorm': 'ts_occ_wmean_pct_mapnorm',
        'n_passes': 'ts_n_passes', 'n_passes_with_min_spikes': 'ts_n_passes_with_min_spikes',
        'lap_stability_score': 'ts_lap_stability'})
    cells = cells.drop_duplicates(['session', 'unit'])[
        ['session', 'unit', 'n_map_bins_visited', 'occupancy_s_map', 'occ_wmean_pct_all_bins',
         'lap_stability_score', 'mean_lap_stability_across_fields']].rename(columns={
        'n_map_bins_visited': 'ts_n_map_bins_visited', 'occupancy_s_map': 'ts_occupancy_s_map',
        'occ_wmean_pct_all_bins': 'ts_occ_wmean_pct_all_bins',
        'lap_stability_score': 'ts_lap_stability_pooled',
        'mean_lap_stability_across_fields': 'ts_lap_stability_field_mean'})

    out = (peak.merge(fields, on=['session', 'unit', 'ts_field_number'], how='left')
               .merge(n_fields, on=['session', 'unit'], how='left')
               .merge(cells, on=['session', 'unit'], how='left'))
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


# Colors of every (arena, zone); main() also adds them to v10's GROUP_COLORS so that its
# _point_colors finds them.
ZONE_COLORS = {(arena, z): c for arena in ZONES for z, c in zone_colors(arena).items()}


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


class Metric(NamedTuple):
    """One compared metric: plot file stem, table-name template ('{}' -> Desc / Omnibus /
    PostHoc), kind ('box', 'cat' or 'speed_dir'), cell table ('full' or 'field'), column, label,
    stability type the cells are restricted to (None = all cells) and whether it gets a
    histogram."""
    stem: str
    table: str
    kind: str
    source: str
    col: str
    label: str
    stab_type: str | None = None
    hist: bool = False


def metric_specs() -> list:
    """The metrics of v10's run_comparison (same plot file and table names), then TS_METRICS."""
    specs = [Metric(f'Full_{c}', '{}', 'box', 'full', c, l, hist=c in asc.HIST_FULL_METRICS)
             for c, l in asc.FULL_METRICS.items()]
    specs += [Metric(f'Fields_{c}', '{}', 'box', 'field', c, l, hist=True)
              for c, l in asc.FIELD_METRICS.items()]
    specs += [Metric(f'Full_{c}', '{}_Extra', 'box', 'full', c, l)
              for c, l in asc.EXTRA_FULL_METRICS.items()]
    specs += [Metric(f'Full_stability_score_{k}', 'StabType_{}', 'box', 'full', 'stability_score',
                     f'Stability score ({d})', stab_type=k)
              for k, d in asc.STABILITY_TYPES.items()]
    specs += [Metric(f'Categorical_{c}', 'Cat{}', 'cat', 'full', c, l)
              for c, l in asc.CATEGORICAL_METRICS_PLACE_CELLS.items()]
    specs.append(Metric('Categorical_speed_direction', 'SpeedDir_{}', 'speed_dir', 'full',
                        'speed_class', 'Speed-direction classification (speed score)'))
    specs += [Metric(f'TemporalStability_{c}', 'TS_{}', 'box', 'full', c, l, hist=True)
              for c, l in TS_METRICS.items()]
    return specs


@dataclass
class ZonePanel:
    """One subset of cells of a comparison (an arena, a session type, ...): its Edge vs Centre
    statistics and its label in the combined plots (e.g. 'Linear' or 'Linear 90°')."""
    arena: str
    label: str
    full_df: pd.DataFrame
    field_df: pd.DataFrame
    stats: dict = field(default_factory=dict)   # {Metric.stem: (desc_df, omnibus, posthoc_df)}

    @property
    def grp(self):
        return asc.Grouping('zone', list(ZONES[self.arena]), 'zones', self.label)

    def cells(self, m: Metric) -> pd.DataFrame:
        df = self.field_df if m.source == 'field' else self.full_df
        return df[df['stability_type'] == m.stab_type] if m.stab_type else df

    def values(self, m: Metric, zone: str) -> pd.DataFrame:
        """Cells of `zone` with a value of m.col (numeric)."""
        sub = self.cells(m)
        sub = sub[sub['zone'] == zone].copy()
        sub[m.col] = pd.to_numeric(sub[m.col], errors='coerce')
        return sub.dropna(subset=[m.col])


def zone_stats(panel: ZonePanel, metrics: list) -> dict:
    """{Metric.stem: (desc_df, omnibus, posthoc_df)}: v10's test of every metric between the two
    zones of `panel`."""
    out = {}
    for m in metrics:
        df = panel.cells(m)
        if m.kind == 'box':
            out[m.stem] = asc.compare_groups(df, m.col, m.label, panel.grp)
        elif m.kind == 'cat':
            out[m.stem] = asc.compare_categorical(df, m.col, m.label, panel.grp)
        else:
            out[m.stem] = asc.compare_speed_direction(df, panel.grp)
    return out


def stats_tables(stats: dict, metrics: list) -> dict:
    """One panel's stats as v10's tables ('Desc', 'Omnibus', 'PostHoc', 'Desc_Extra', ...)."""
    parts = {}
    for m in metrics:
        desc_df, omnibus, posthoc_df = stats[m.stem]
        d, o, p = parts.setdefault(m.table, ([], [], []))
        d.append(desc_df)
        o.append(omnibus)
        if len(posthoc_df):
            p.append(posthoc_df)
    tables = {}
    for tmpl, (d, o, p) in parts.items():
        tables[tmpl.format('Desc')]    = pd.concat(d, ignore_index=True)
        tables[tmpl.format('Omnibus')] = pd.DataFrame(o)
        tables[tmpl.format('PostHoc')] = pd.concat(p, ignore_index=True) if p else pd.DataFrame()
    return tables


def zone_panel(info: dict, arena: str, label: str, title: str, full_df: pd.DataFrame,
               field_df: pd.DataFrame, sheet_prefix: str, metrics: list,
               sessions_df: pd.DataFrame, tables: dict, counts: list):
    """Edge vs centre (Outer vs Inner) statistics of one subset of cells; the tables are stored in
    `tables` under '<sheet_prefix>_<name>' with the subset description (`info`) in the first
    columns. Returns the ZonePanel, or None if the subset has no place cells."""
    counts.append(_zone_count_row(info, arena, full_df, sessions_df))
    print(f'\n{"=" * 70}\n{title} -- {" vs ".join(ZONES[arena])}\n{"=" * 70}')
    print(full_df['zone'].value_counts().reindex(ZONES[arena]).fillna(0).astype(int).to_string())
    if not len(full_df):
        print('No place cells; skipping.')
        return None

    panel = ZonePanel(arena, label, full_df, field_df)
    panel.stats = zone_stats(panel, metrics)
    results = stats_tables(panel.stats, metrics)
    asc._print_significant(pd.concat([t for n, t in results.items() if n.endswith('Omnibus')],
                                     ignore_index=True), title, 65, 30)
    for name, tbl in results.items():
        if len(tbl):
            for i, (k, v) in enumerate(info.items()):
                if k not in tbl.columns:
                    tbl.insert(i, k, v)
        tables[f'{sheet_prefix}_{name}'] = tbl
    return panel


# ── Combined plots: every panel of a comparison in one figure ───────────────
# Box / bar plots: the two zones of each panel side by side (v10 style), the panel label under
# the pair and the panel's significance bracket above it. Histograms: v10's panels on shared
# bins, both zones in each.

def _slots(panels: list) -> list:
    """(panel, zone, x) of every box / bar: each panel's two zones side by side, PANEL_GAP
    between panels."""
    slots, x = [], 0.0
    for p in panels:
        for z in ZONES[p.arena]:
            x += 1.0
            slots.append((p, z, x))
        x += PANEL_GAP
    return slots


def _stats_lines(panels: list, m: Metric, fmt) -> list:
    """v10's stats text of every panel (fmt = asc._continuous_stats_lines, ...), each under a
    header line with the panel label."""
    lines = []
    for p in panels:
        lines.append(f'── {p.label} ──')
        lines += fmt(*p.stats[m.stem], p.grp)
    return lines


def _draw_stats_top(ax_info, stats_lines: list, legend_rows: int):
    """Stats text at the TOP of the info panel, under the legend rows (v10's _draw_stats puts it
    at the bottom, which leaves a gap under the plot when there are many lines; the empty space
    at the bottom is cropped by the tight bounding box when saving)."""
    if stats_lines:
        ax_info.annotate('\n'.join(stats_lines), xy=(0.5, 1.0), xycoords='axes fraction',
                         xytext=(0, -(legend_rows * asc.LEGEND_ROW_IN * 72 + 6)),
                         textcoords='offset points', ha='center', va='top',
                         fontsize=asc.FS_STATS, linespacing=1.3)


def _suptitle(fig, text: str):
    """Figure title just above the panels (call after tight_layout)."""
    top = max(ax.get_tightbbox().transformed(fig.transFigure.inverted()).y1 for ax in fig.axes
              if ax.axison)
    fig.suptitle(text, y=top + 0.005, va='bottom', fontsize=asc.FS_TITLE, fontweight='bold')


def _combined_fig(panels: list, stats_lines: list, legend_rows: int = 0):
    """v10's _new_fig, sized for the slots of `panels`; the stats text is drawn."""
    n_slots = 2 * len(panels) + PANEL_GAP * (len(panels) - 1)
    fig_w = max(n_slots * ZONE_BOX_W + asc.AXIS_PAD_W, 4.0)
    stats_lines = asc._wrap_stats(stats_lines, fig_w)
    fig, ax, ax_info = asc._new_fig((fig_w, 5.5), legend_rows=legend_rows, stats_lines=stats_lines)
    _draw_stats_top(ax_info, stats_lines, legend_rows)
    return fig, ax, ax_info


def _label_slots(ax, panels: list, slots: list, ns: list):
    """Zone + n under every box / bar, the panel label under each pair."""
    ax.set_xticks([x for _, _, x in slots])
    ax.set_xticklabels([f'{z}\n(n={n})' for (_, z, _), n in zip(slots, ns)])
    ax.set_xlim(slots[0][2] - 0.7, slots[-1][2] + 0.7)
    for p in panels:
        xs = [x for q, _, x in slots if q is p]
        ax.annotate(p.label, xy=(np.mean(xs), 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -40), textcoords='offset points', ha='center', va='top',
                    fontsize=asc.FS_LABEL - 1, fontweight='bold')


def _finish_combined(fig, ax, panels, slots, ns, ylabel: str, title: str, out_path: str):
    _label_slots(ax, panels, slots, ns)
    ax.set_ylabel(ylabel, fontsize=asc.FS_LABEL, labelpad=8)
    asc._set_title(ax, title)
    asc._style_axes(ax)
    fig.tight_layout()
    asc._save(fig, out_path)


def _percent_brackets(ax, panels: list, slots: list, heights: list, m: Metric):
    """v10's _set_percent_ylim with one significance line per panel, above its pair of bars."""
    tops = []
    for p in panels:
        pos = {z: x for q, z, x in slots if q is p}
        bar_top = max(h for (q, _, _), h in zip(slots, heights) if q is p)
        top = asc._draw_sig_brackets(ax, pos, p.stats[m.stem][2], y_base=bar_top + 4,
                                     y_step=8, y_text=3)
        if top is not None:
            tops.append(top)
    ax.set_ylim(0, max([100] + [t + 8 for t in tops]))
    ax.set_yticks(np.arange(0, 101, 20))


def plot_zone_boxes(panels: list, m: Metric, out_path: str, title: str):
    """v10's plot_metric with every panel's two zones in one plot."""
    slots = _slots(panels)
    data = [p.values(m, z) for p, z, _ in slots]
    if not any(len(d) for d in data):
        return

    has_legend = m.col in asc.SIGNIFICANCE_METRICS
    fig, ax, ax_info = _combined_fig(
        panels, _stats_lines(panels, m, asc._continuous_stats_lines),
        legend_rows=len(asc._SIGNIFICANCE_LEGENDS[m.col]) if has_legend else 0)

    rng = np.random.default_rng(0)
    for (p, z, x), sub in zip(slots, data):
        if not len(sub):
            continue
        vals = sub[m.col].to_numpy(dtype=float)
        bp = ax.boxplot([vals], positions=[x], showfliers=False, patch_artist=True, widths=0.55,
                        manage_ticks=False,
                        boxprops=dict(linewidth=asc.BOX_LW, color=asc.PAL_BLACK),
                        whiskerprops=dict(linewidth=asc.BOX_LW, color=asc.PAL_BLACK),
                        capprops=dict(linewidth=asc.BOX_LW, color=asc.PAL_BLACK),
                        medianprops=dict(linewidth=asc.MEDIAN_LW, color=asc.PAL_BLACK),
                        zorder=2)
        bp['boxes'][0].set_facecolor(ZONE_COLORS[(p.arena, z)]['face'])
        bp['boxes'][0].set_edgecolor(asc.PAL_BLACK)
        colors = asc._point_colors(m.col, sub, (p.arena, z))
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), x) + jitter, vals, s=22,
                   facecolors=[fc for fc, _ec in colors], edgecolors=[ec for _fc, ec in colors],
                   linewidths=0.5, zorder=3)

    # Significance line above each panel's pair of zones.
    y_lo, y_hi = ax.get_ylim()
    span = y_hi - y_lo
    tops = []
    for p in panels:
        pairs = [(z, x, sub) for (q, z, x), sub in zip(slots, data) if q is p and len(sub)]
        if not pairs:
            continue
        top = asc._draw_sig_brackets(ax, {z: x for z, x, _ in pairs}, p.stats[m.stem][2],
                                     y_base=max(sub[m.col].max() for *_, sub in pairs) + 0.06 * span,
                                     y_step=0.10 * span, y_text=0.03 * span)
        if top is not None:
            tops.append(top)
    if tops:
        ax.set_ylim(y_lo, max(y_hi, max(tops) + 0.08 * span))

    if has_legend:
        p0 = panels[0]
        asc._add_significance_legend(ax_info, m.col, ZONE_COLORS[(p0.arena, ZONES[p0.arena][1])])
    _finish_combined(fig, ax, panels, slots, [len(d) for d in data],
                     textwrap.fill(m.label, 30), f'{title}: {m.label}', out_path)


def plot_zone_bars(panels: list, m: Metric, out_path: str, title: str):
    """v10's plot_categorical with every panel's two zones in one plot."""
    slots = _slots(panels)
    pct, ns = [], []
    for p, z, _ in slots:
        cells = p.cells(m)
        y = cells.loc[cells['zone'] == z, m.col].apply(asc._to_tri_bool).dropna()
        ns.append(len(y))
        pct.append(100.0 * (y == True).sum() / len(y) if len(y) else 0.0)  # noqa: E712
    if not any(ns):
        return

    fig, ax, _ = _combined_fig(panels, _stats_lines(panels, m, asc._categorical_stats_lines))
    ax.bar([x for *_, x in slots], pct,
           color=[ZONE_COLORS[(p.arena, z)]['face'] for p, z, _ in slots],
           edgecolor=asc.BAR_EDGE, linewidth=asc.BAR_EDGE_LW, width=asc.BAR_WIDTH, zorder=2)
    _percent_brackets(ax, panels, slots, pct, m)
    _finish_combined(fig, ax, panels, slots, ns, '% of cells', f'{title}: {m.label}', out_path)


def plot_zone_speed_direction(panels: list, m: Metric, out_path: str, title: str):
    """v10's plot_speed_direction with every panel's two zones in one plot: p-Speed in the darker
    and n-Speed in the lighter tone of each zone's color, non-speed gray."""
    slots = _slots(panels)
    pct_p, pct_n, ns = [], [], []
    for p, z, _ in slots:
        cls = p.full_df.loc[p.full_df['zone'] == z, 'speed_class'].dropna()
        ns.append(len(cls))
        pct_p.append(100.0 * (cls == 'p').sum() / len(cls) if len(cls) else 0.0)
        pct_n.append(100.0 * (cls == 'n').sum() / len(cls) if len(cls) else 0.0)
    if not any(ns):
        return
    pct_non = [100.0 - a - b if n else 0.0 for a, b, n in zip(pct_p, pct_n, ns)]

    fig, ax, ax_info = _combined_fig(panels, _stats_lines(panels, m, asc._speed_dir_stats_lines),
                                     legend_rows=1)
    xs = [x for *_, x in slots]
    colors = [ZONE_COLORS[(p.arena, z)] for p, z, _ in slots]
    bar_kw = dict(edgecolor=asc.BAR_EDGE, linewidth=asc.BAR_EDGE_LW, width=asc.BAR_WIDTH, zorder=2)
    ax.bar(xs, pct_p, color=[c['dark'] for c in colors], label='p-Speed', **bar_kw)
    ax.bar(xs, pct_n, bottom=pct_p, color=[c['light'] for c in colors], label='n-Speed', **bar_kw)
    ax.bar(xs, pct_non, bottom=[a + b for a, b in zip(pct_p, pct_n)], color=asc.SIG_GRAY['face'],
           label='non-speed', **bar_kw)
    _percent_brackets(ax, panels, slots, [100.0 if n else 0.0 for n in ns], m)
    handles, labels = ax.get_legend_handles_labels()
    ax_info.legend(handles, labels, loc='upper center', fontsize=asc.FS_LEGEND, ncol=3)
    _finish_combined(fig, ax, panels, slots, ns, '% of place cells', f'{title}: {m.label}',
                     out_path)


def _hist_rows(panels: list) -> list:
    """Histogram panel layout: one row per arena, or a single row if every arena has one panel."""
    rows = [r for r in ([p for p in panels if p.arena == a] for a in asc.ARENA_TYPES) if r]
    return [[p for r in rows for p in r]] if all(len(r) == 1 for r in rows) else rows


def _grid_fig(rows: list, stats_lines: list, legend_rows: int = 0):
    """v10's _new_panel_fig with a grid of panels (one per entry of the lists in `rows`) on a
    shared x-axis. Returns (fig, axes as rows, ax_info); the stats text is drawn."""
    n_cols = max(len(r) for r in rows)
    fig_w = max(asc.HIST_PANEL_W * n_cols, 7)
    stats_lines = asc._wrap_stats(stats_lines, fig_w)
    info_h = len(stats_lines) * asc.STATS_LINE_IN + legend_rows * asc.LEGEND_ROW_IN + 0.4
    fig = plt.figure(figsize=(fig_w, asc.HIST_PANEL_H * len(rows) + info_h))
    gs = fig.add_gridspec(len(rows) + 1, n_cols,
                          height_ratios=[asc.HIST_PANEL_H] * len(rows) + [info_h])
    axes, first = [], None
    for i, row in enumerate(rows):
        axes.append([])
        for j in range(len(row)):
            ax = fig.add_subplot(gs[i, j], sharex=first)
            first = ax if first is None else first
            axes[-1].append(ax)
    ax_info = fig.add_subplot(gs[-1, :])
    ax_info.axis('off')
    _draw_stats_top(ax_info, stats_lines, legend_rows)
    return fig, axes, ax_info


def _zone_hist_panel(ax, p: ZonePanel, vals: list, bins):
    """v10's histogram panel with both zones of `p` side by side in every bin, and a dashed
    (wall-side zone) / dotted (inner zone) line at each zone's median."""
    zones = ZONES[p.arena]
    present = [(z, v) for z, v in zip(zones, vals) if len(v)]
    if present:
        counts, _, _ = ax.hist([v for _, v in present], bins=bins,
                               color=[ZONE_COLORS[(p.arena, z)]['face'] for z, _ in present],
                               edgecolor=asc.BAR_EDGE, linewidth=asc.BAR_EDGE_LW, zorder=2)
        ax.set_ylim(0, max(np.max(counts), 1) * 1.3)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    for k, (z, v) in enumerate(present):
        tone = ZONE_COLORS[(p.arena, z)]['dark']
        med = np.median(v)
        ax.axvline(med, color=tone, linestyle=(0, (3, 2)) if z == zones[0] else (0, (1, 1.5)),
                   linewidth=1.5 * asc.AXIS_LW, zorder=4)
        ax.text(0.98, 0.99 - 0.09 * k, f'{z} med={med:.3g}', transform=ax.transAxes,
                ha='right', va='top', fontsize=asc.FS_MED, color=tone)
    ax.set_title(f'{p.label}\n' + ', '.join(f'{z} n={len(v)}' for z, v in zip(zones, vals)),
                 fontsize=asc.FS_LABEL, pad=10)


def plot_zone_histograms(panels: list, m: Metric, out_path: str, title: str):
    """v10's plot_histogram_by_group: one panel per subset (rows = arenas), both zones on bins
    shared by every panel."""
    vals = {p.label: [p.values(m, z)[m.col].to_numpy(dtype=float) for z in ZONES[p.arena]]
            for p in panels}
    all_vals = np.concatenate([v for vs in vals.values() for v in vs])
    if not len(all_vals):
        return
    bins = np.histogram_bin_edges(all_vals, bins=asc.N_BINS)

    rows = _hist_rows(panels)
    fig, axes, _ = _grid_fig(rows, _stats_lines(panels, m, asc._continuous_stats_lines))
    for row, ax_row in zip(rows, axes):
        for p, ax in zip(row, ax_row):
            _zone_hist_panel(ax, p, vals[p.label], bins)
            ax.set_xlabel(textwrap.fill(m.label, 22), fontsize=asc.FS_LABEL, labelpad=6)
            asc._style_axes(ax)
        ax_row[0].set_ylabel('Count', fontsize=asc.FS_LABEL, labelpad=6)
    fig.tight_layout()
    _suptitle(fig, title)
    asc._save(fig, out_path)


def plot_zone_speed_histograms(panels: list, col: str, label: str, out_path: str, title: str):
    """v10's plot_speed_histograms_by_group: speed cells only, p-type (green) / n-type (red)
    stacked, one panel per subset and zone (rows = arena x zone)."""
    speed = {}
    for p in panels:
        sub = p.full_df[p.full_df['speed_class'].isin(['p', 'n'])].copy()
        sub[col] = pd.to_numeric(sub[col], errors='coerce')
        speed[p.label] = sub.dropna(subset=[col])
    all_vals = pd.concat(speed.values())[col].to_numpy(dtype=float)
    if not len(all_vals):
        print(f'  {title}: no speed cells for {label}; skipping plot.')
        return
    bins = np.histogram_bin_edges(all_vals, bins=asc.N_BINS)

    # Stats text: descriptives per zone and p/n type, then the zone comparison of the pooled
    # speed cells of each panel.
    stats_lines = []
    for p in panels:
        sub = speed[p.label]
        stats_lines.append(f'── {p.label} ──')
        for z in ZONES[p.arena]:
            for name, k in (('p-type', 'p'), ('n-type', 'n')):
                v = sub.loc[(sub['zone'] == z) & (sub['speed_class'] == k), col]
                stats_lines.append(f'{z} {name} (n={len(v)}): {asc._desc_str(v)}')
        _, omnibus, posthoc_df = asc.compare_groups(sub, col, label, p.grp)
        stats_lines.append(f'{" vs ".join(ZONES[p.arena])} (all speed-modulated cells pooled):')
        stats_lines += asc._test_lines(omnibus, posthoc_df)

    rows = [[(p, ZONES[p.arena][i]) for p in row] for row in _hist_rows(panels) for i in (0, 1)]
    fig, axes, ax_info = _grid_fig(rows, stats_lines, legend_rows=1)
    for row, ax_row in zip(rows, axes):
        for (p, z), ax in zip(row, ax_row):
            sub = speed[p.label]
            p_vals = sub.loc[(sub['zone'] == z) & (sub['speed_class'] == 'p'), col].to_numpy(float)
            n_vals = sub.loc[(sub['zone'] == z) & (sub['speed_class'] == 'n'), col].to_numpy(float)
            if len(p_vals) + len(n_vals):
                ax.hist([p_vals, n_vals], bins=bins, stacked=True,
                        color=[asc.P_COLOR['face'], asc.N_COLOR['face']],
                        edgecolor=asc.BAR_EDGE, linewidth=asc.BAR_EDGE_LW, zorder=2)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_title(f'{p.label} {z}\n(n={len(p_vals) + len(n_vals)}: {len(p_vals)} p-type, '
                         f'{len(n_vals)} n-type)', fontsize=asc.FS_LABEL, pad=10)
            ax.set_xlabel(textwrap.fill(label, 22), fontsize=asc.FS_LABEL, labelpad=6)
            asc._style_axes(ax)
        ax_row[0].set_ylabel('Count', fontsize=asc.FS_LABEL, labelpad=6)
    handles = [Patch(facecolor=asc.P_COLOR['face'], edgecolor='none', label='p-type speed cell'),
               Patch(facecolor=asc.N_COLOR['face'], edgecolor='none', label='n-type speed cell')]
    ax_info.legend(handles=handles, loc='upper center', ncol=2, fontsize=asc.FS_LEGEND)
    fig.tight_layout()
    _suptitle(fig, f'{title}: {label} (speed cells only)')
    asc._save(fig, out_path)


def plot_comparison(panels: list, metrics: list, title: str, plots_dir: str):
    """Every combined plot of one comparison: box / bar plots in `plots_dir`, histograms in
    `plots_dir`\\Histograms."""
    if not panels:
        print(f'\n{title}: no place cells; no plots.')
        return
    hist_dir = os.path.join(plots_dir, 'Histograms')
    os.makedirs(hist_dir, exist_ok=True)
    plot = {'box': plot_zone_boxes, 'cat': plot_zone_bars, 'speed_dir': plot_zone_speed_direction}
    for m in metrics:
        plot[m.kind](panels, m, os.path.join(plots_dir, f'{m.stem}.png'), title)
        if m.hist:
            plot_zone_histograms(panels, m, os.path.join(hist_dir, f'Hist_{m.stem}.png'), title)
    for col, label in asc.SPEED_METRICS.items():
        plot_zone_speed_histograms(panels, col, label,
                                   os.path.join(hist_dir, f'Hist_Speed_{col}.png'), title)
    print(f'\n{title}: plots saved to {plots_dir}')


# ── Zone schematics ─────────────────────────────────────────────────────────

ARENA_SIZE_NOTES = {
    'Open':   f'dia {OPEN_DIAMETER_CM:g} cm',
    'Linear': f'{LINEAR_LENGTH_CM:g} x {LINEAR_WIDTH_CM:g} cm',
    'Circle': f'outer dia {CIRCLE_OUTER_DIAMETER_CM:g} cm, inner dia {CIRCLE_INNER_DIAMETER_CM:g} cm',
}
ZONE_NOTES = {   # (wall-side zone, inner zone)
    'Open':   (f'bin centre ≤ {OPEN_EDGE_ZONE_THRESHOLD_CM:g} cm from the wall', 'all other bins'),
    'Linear': (f'2 outer rows + bins within {LINEAR_EDGE_ZONE_THRESHOLD_CM:g} cm of the track ends',
               '2 middle rows, away from the ends'),
    'Circle': ('outer ring of bins', 'inner ring of bins'),
}


def _bin_patches(geom, arena: str) -> tuple:
    """(patches, flat bin indices) of every bin inside the arena, drawn in cm."""
    patches, idx = [], np.flatnonzero(geom.geom_valid)
    for b in idx:
        bx, by = divmod(int(b), geom.ny)
        if arena == 'Circle':
            t0 = bx * geom.bin_width_deg
            patches.append(Wedge((geom.cx, geom.cy), geom.inner_r + (by + 1) * geom.bin_cm_y,
                                 t0, t0 + geom.bin_width_deg, width=geom.bin_cm_y))
        elif arena == 'Linear':
            patches.append(Rectangle((bx * geom.bin_cm_x, by * geom.bin_cm_y),
                                     geom.bin_cm_x, geom.bin_cm_y))
        else:
            patches.append(Rectangle((bx * BIN_CM, by * BIN_CM), BIN_CM, BIN_CM))
    return patches, idx


def plot_zone_schematics(geoms: dict, out_dir: str):
    """OUTPUT_DIR\\ZoneSchematic_<arena>.png: every bin of the arena shaded by its zone (wall-side
    zone darker), with the arena walls (and, open field, the edge-zone threshold) outlined."""
    for arena, geom in geoms.items():
        wall, inner = ZONES[arena]
        face = {True: ZONE_COLORS[(arena, wall)]['face'], False: ZONE_COLORS[(arena, inner)]['face']}
        patches, idx = _bin_patches(geom, arena)
        fig, ax = plt.subplots(figsize=(9.0, 2.2) if arena == 'Linear' else (5.5, 5.5))
        ax.add_collection(PatchCollection(patches, facecolors=[face[bool(w)] for w in geom.wall_zone[idx]],
                                          edgecolors='white', linewidths=0.4, zorder=2))

        wall_kw = dict(fill=False, edgecolor=asc.PAL_BLACK, linewidth=asc.BOX_LW, zorder=3)
        handles = [Patch(facecolor=face[True], edgecolor='none', label=f'{wall}: {ZONE_NOTES[arena][0]}'),
                   Patch(facecolor=face[False], edgecolor='none', label=f'{inner}: {ZONE_NOTES[arena][1]}')]
        if arena == 'Linear':
            ax.add_patch(Rectangle((0, 0), geom.length, geom.width, **wall_kw))
            w, h = geom.length, geom.width
            ax.set_xlabel('Along the track (cm)', fontsize=asc.FS_LABEL)
            ax.set_yticks([0, geom.width])
            ax.set_ylabel('Width\n(cm)', fontsize=asc.FS_LABEL, rotation=0, ha='right', va='center')
        else:
            radii = ((OPEN_DIAMETER_CM / 2.0,) if arena == 'Open' else (geom.outer_r, geom.inner_r))
            for r in radii:
                ax.add_patch(Circle((geom.cx, geom.cy), r, **wall_kw))
            if arena == 'Open':
                thr = OPEN_DIAMETER_CM / 2.0 - OPEN_EDGE_ZONE_THRESHOLD_CM
                ax.add_patch(Circle((geom.cx, geom.cy), thr, linestyle=(0, (3, 2)), **wall_kw))
                handles.append(Patch(facecolor='none', edgecolor=asc.PAL_BLACK, linestyle=(0, (3, 2)),
                                     label=f'{OPEN_EDGE_ZONE_THRESHOLD_CM:g} cm from the wall'))
            w = h = 2.0 * radii[0]
            ax.set_xlabel('x (cm)', fontsize=asc.FS_LABEL)
            ax.set_ylabel('y (cm)', fontsize=asc.FS_LABEL)

        pad = 2.0
        ax.set_xlim(-pad, w + pad)
        ax.set_ylim(-pad, h + pad)
        ax.set_aspect('equal')
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.03, 0.5),
                  fontsize=asc.FS_LEGEND)
        ax.set_title(f'{arena} arena: {wall} vs {inner} zone\n'
                     f'({ARENA_SIZE_NOTES[arena]}; {BIN_CM:g} x {BIN_CM:g} cm bins)',
                     fontsize=asc.FS_TITLE - 2, pad=12)
        asc._style_axes(ax)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f'ZoneSchematic_{arena}.png')
        asc._save(fig, out_path)
        print(f'Zone schematic saved to {out_path}')


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
    asc.GROUP_COLORS.update(ZONE_COLORS)
    metrics = metric_specs()

    geoms = make_geometries()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_zone_schematics(geoms, OUTPUT_DIR)

    print(f'Scanning {PLACE_CELL_DIR} ...')
    sessions_df, cells_df = scan_place_cell_dir(geoms)
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

    tables, counts = {}, []
    stat_args = (metrics, sessions_df, tables, counts)

    # ── 1. Each arena, all its cells ────────────────────────────────────────
    panels = []
    for arena in asc.ARENA_TYPES:
        def _sel(d):
            return d[d['arena_type'] == arena]
        info = dict(level='Arena', arena_type=arena, session_type='all', animals='all')
        panels.append(zone_panel(info, arena, arena, f'{arena} arena', _sel(full_df),
                                 _sel(field_df), f'Ar_{arena}', *stat_args))
    plot_comparison([p for p in panels if p], metrics, 'Edge vs centre, all cells',
                    os.path.join(OUTPUT_DIR, 'Arena'))

    # ── 2. Each session type of each arena ──────────────────────────────────
    panels = []
    for arena in asc.ARENA_TYPES:
        for stype in asc.ARENA_SESSION_TYPES[arena]:
            def _sel(d):
                return d[(d['arena_type'] == arena) & (d['session_type'] == stype)]
            info = dict(level='SessionType', arena_type=arena, session_type=stype, animals='all')
            panels.append(zone_panel(info, arena, f'{arena} {asc._label(stype)}',
                                     f'{arena} arena, {asc._label(stype)}', _sel(full_df),
                                     _sel(field_df), f'Ses_{arena[0]}{stype}', *stat_args))
    plot_comparison([p for p in panels if p], metrics, 'Edge vs centre, by session type',
                    os.path.join(OUTPUT_DIR, 'SessionType'))

    # ── 3. Animal-controlled: each arena, only the animals recorded in all compared arenas ──
    for name, (animals, arenas) in asc.ANIMAL_CONTROLLED_COMPARISONS.items():
        wanted = {a.lower() for a in animals}
        ordered = [a for a in asc.ARENA_TYPES if a in arenas]
        tag = 'AC_' + ''.join(a[0] for a in ordered)      # e.g. 'AC_CO'
        panels = []
        for arena in ordered:
            def _sel(d):
                return d[(d['arena_type'] == arena)
                         & d['animal'].astype(str).str.lower().isin(wanted)]
            info = dict(level='AnimalControlled', arena_type=arena, session_type='all',
                        animals=', '.join(animals), comparison=name)
            panels.append(zone_panel(info, arena, arena, f'{" + ".join(animals)} only, {arena} arena',
                                     _sel(full_df), _sel(field_df), f'{tag}_{arena[0]}',
                                     *stat_args))
        plot_comparison([p for p in panels if p], metrics,
                        f'Edge vs centre, {" + ".join(animals)} only',
                        os.path.join(OUTPUT_DIR, asc.ANIMAL_CONTROLLED_DIR, name))

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
