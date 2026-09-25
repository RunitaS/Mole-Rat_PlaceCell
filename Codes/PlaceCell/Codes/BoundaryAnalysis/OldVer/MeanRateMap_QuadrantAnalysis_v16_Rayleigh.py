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

Tracking load/clean/smooth (pixel<->cm handling, jump removal, Gaussian smoothing) and
spike-position matching (50 ms gate) are ported from
PlaceCellCharacterization_SpeedModv3_DownsampledPos15.py.

CELL SELECTION: this script does NOT re-test cells for place-cell qualification. Every .ntt
file under ROOT_DIRECTORY is taken as an already-qualified place cell -- qualification
(n_spikes, peak rate, SIR, sparsity, shuffle significance) was done by the upstream pipeline
that populated that folder, and re-applying a second, differently-parameterized screen here
would silently drop cells that pipeline accepted. The ONLY inclusion criteria applied below
are the two session-level coverage criteria (COVERAGE_FRACTION and MIN_ZONE_COVERAGE_PCT),
which are about how well the animal sampled the arena, not about the cell. So every unit of
every coverage-passing session is pooled into every map and every statistic.

The per-cell metrics (n_spikes, peak_fr, SIR, sparsity) and the location-shuffle bootstrap
are still computed and written to the summary workbook as diagnostics, but they no longer
gate anything. ListIncludedSessionsCells.py reports exactly which sessions and cells this
leaves in the analysis.

Folder layout expected under ROOT_DIRECTORY (same convention as the reference scripts):
  ROOT_DIRECTORY/.../<Open|Linear|Circle>/.../<session>/  containing exactly one tracking
  file (.csv or .xlsx) and one or more .ntt files.

Every arena's spatial bins are represented as a single flat index (0..n_bins-1); this lets rate-map
construction, SIR/sparsity, and the bootstrap significance test share one implementation across the
2D open field, the 2D (wrap-around in arc-length only) circular track, and the 2D linear track --
only the coordinate transform, smoothing kernel and plotting differ per arena.
"""

import os
import re
import hashlib
import random
import threading
import concurrent.futures

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label
from scipy.stats import rankdata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# ============================================================================
# CONFIGURATION -- edit root folder + geometry below
# ============================================================================

# Single root under which every animal/arena/day/session lives. Arena type is auto-detected
# per session from its path (see ARENA_FOLDER_KEYWORDS / _detect_arena_key below) -- no
# per-arena root folders needed any more.
ROOT_DIRECTORY = r'X:\NMR_group_data\Runita\Analysis\Thesis\Data_v2_Accepted\SessionType_Sorted\Open\Cntrl'

# Output folder for all figures / workbooks (same location as before, now derived from the root).
OUTPUT_DIR = os.path.join(ROOT_DIRECTORY, 'MeanRM_Quad_Rayleigh2')

# Whole-arena bin coverage criterion (session-level). True: a session must have >= min_occ_s
# occupancy in at least COVERAGE_FRACTION of the arena's total spatial bins (per handler, see
# `total_arena_bins`/`coverage_threshold_bins`), else every file (unit) from that session is
# skipped -- computed dynamically per handler, so it tracks target_bin_cm/geometry.
# False: the criterion is not applied.
USE_BIN_COVERAGE_CRITERION = True
COVERAGE_FRACTION = 0.01      # DEBUG: temporarily lowered from 0.80 to test whether the
                               # rest of the pipeline (bootstrap, place-cell qualification,
                               # pooling, quadrant fold, plotting) runs end-to-end on
                               # circular-track sessions that the 80% threshold was excluding
                               # (e.g. 2NoRot at ~78% bin coverage)

# Edge/centre zone occupancy criterion (session-level). True: a session is only included if the
# animal's tracked occupancy covers BOTH the edge zone AND the centre zone by at least
# MIN_ZONE_COVERAGE_PCT of the session's total occupied time; a failing session has every unit
# in it skipped. False: the criterion is not applied (sessions are not screened by zone
# coverage, and the tracking is not even read for it). See session_zone_coverage /
# collect_arena_results.
USE_ZONE_COVERAGE_CRITERION = True
MIN_ZONE_COVERAGE_PCT       = 0.1

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


# ── Open-field arena centring: _load_tracking anchors each session's coordinate frame on that
# session's own tracking bounding box (`x - x.min()`), so the point the handler treats as the
# arena centre -- (cx, cy) = (diameter/2, diameter/2) -- is the true centre only when the
# animal actually reached the wall on all four sides. Often it did not: measured tracking
# spans across the open-field sessions in this dataset run 58.6-60.0 cm in x but only
# 53.3-60.0 cm in y against a 60 cm arena, putting the assumed centre up to ~3.4 cm off the
# real one.
#
# Every radius-based quantity in the pipeline is measured from (cx, cy) -- geom_valid,
# dist_to_wall_flat, dist_from_center_flat, edge_zone_flat (a 9.25 cm edge band) and to_bins'
# sample_valid -- so that offset biases the edge-vs-centre split of EVERY open-field session,
# not just the rotated ones, and it is also what made _apply_session_rotation turn the data
# about the wrong axis.
#
# The correction is applied once, at load time (_session_positions): estimate where the arena
# centre actually lies in the session's own frame and translate the tracking so it lands on
# (cx, cy). Every downstream radius then means what it claims, and rotating about (cx, cy)
# becomes rotating about the physical axis the arena was turned about.
#
# Set to False to restore the old behaviour (assume the tracking bounding box is the arena's).
CENTRE_OPEN_FIELD_TRACKING = True

_CENTRE_WARN_LOCK = threading.Lock()
_CENTRE_WARNED    = set()


def _estimate_disc_centre(x_cm: np.ndarray, y_cm: np.ndarray, radius_cm: float,
                          n_sectors: int = 72, wall_frac: float = 0.90,
                          min_sectors: int = 12,
                          max_shift_frac: float = 0.25) -> tuple:
    """Estimates the centre of a circular arena of KNOWN radius from one session's tracking.

    Works from the tracking's OUTER ENVELOPE rather than its bounding box or its centroid.
    The envelope is the farthest sample in each of `n_sectors` angular sectors around a
    starting guess -- the points where the animal came closest to the wall. Sectors whose
    farthest sample does not reach `wall_frac * radius_cm` are dropped: in those directions
    the animal never approached the wall, so the sector says nothing about where the wall is
    and keeping it would drag the estimate inwards. Using the bounding box instead is exactly
    the assumption that fails here, because it reads an under-sampled side as a near wall.

    The centre is then the least-squares fit of a circle of the known radius to the surviving
    envelope points. Setting the gradient of sum_i (|p_i - c| - R)^2 to zero gives the fixed
    point c = mean(p_i) - R * mean((p_i - c) / |p_i - c|), iterated from the bounding-box
    midpoint (which is already exact whenever the tracking does reach the wall all round --
    the common case in this dataset, and then this function is a no-op).

    Returns (cx, cy, ok). `ok` is False -- and the bounding-box midpoint returned unchanged --
    when the fit is not trustworthy:
      * fewer than `min_sectors` sectors reach the wall, so the fit is under-constrained;
      * the surviving sectors leave an angular gap wider than pi, i.e. they all sit within one
        half-plane, which leaves the centre unidentifiable along the perpendicular direction;
      * the fit wants to move the centre by more than `max_shift_frac * radius_cm`, which in
        this geometry means it has run away rather than found a better centre.
    Callers are expected to leave the frame untouched when `ok` is False, rather than apply a
    guess that could be worse than the status quo.
    """
    cx0 = 0.5 * (float(x_cm.min()) + float(x_cm.max()))
    cy0 = 0.5 * (float(y_cm.min()) + float(y_cm.max()))
    if len(x_cm) < min_sectors:
        return cx0, cy0, False

    theta = np.arctan2(y_cm - cy0, x_cm - cx0)
    r = np.hypot(x_cm - cx0, y_cm - cy0)
    sector = np.clip(((theta + np.pi) / (2 * np.pi) * n_sectors).astype(int), 0, n_sectors - 1)

    # one index per occupied sector: sort by (sector, r) and take the last sample of each run
    order = np.lexsort((r, sector))
    s_sorted = sector[order]
    last = np.flatnonzero(np.append(np.diff(s_sorted) != 0, True))
    env = order[last]
    env = env[r[env] >= wall_frac * radius_cm]
    if len(env) < min_sectors:
        return cx0, cy0, False

    # identifiability: the wall-reaching sectors must surround the start point
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
        ux, uy = dx[keep] / rr[keep], dy[keep] / rr[keep]
        cx_new = float(px[keep].mean() - radius_cm * ux.mean())
        cy_new = float(py[keep].mean() - radius_cm * uy.mean())
        converged = abs(cx_new - cx) < 1e-7 and abs(cy_new - cy) < 1e-7
        cx, cy = cx_new, cy_new
        if converged:
            break

    if not (np.isfinite(cx) and np.isfinite(cy)):
        return cx0, cy0, False
    if np.hypot(cx - cx0, cy - cy0) > max_shift_frac * radius_cm:
        return cx0, cy0, False
    return cx, cy, True


def _centre_open_field_tracking(x_cm: np.ndarray, y_cm: np.ndarray, session_dir: str,
                                handler) -> tuple:
    """Translates an OPEN FIELD session's tracking so the estimated arena centre lands on the
    handler's (cx, cy), the point every radius-based measure in the pipeline is taken from
    (see the CENTRE_OPEN_FIELD_TRACKING comment above). Any other arena is returned unchanged
    -- the linear track's frame is defined by the track's own extent and the ring track is
    handled by CircularTrackHandler, so neither wants this.

    A translation (unlike the rotation below) does not preserve distance from (cx, cy), which
    is the whole point: it is what puts each sample at its true distance from the arena centre.
    When the centre cannot be estimated reliably the frame is left exactly as it was, so such a
    session keeps its previous behaviour rather than taking on an unfounded shift."""
    if not CENTRE_OPEN_FIELD_TRACKING:
        return x_cm, y_cm
    if _detect_arena_key(session_dir) != 'open_field' or len(x_cm) == 0:
        return x_cm, y_cm

    cx, cy, ok = _estimate_disc_centre(x_cm, y_cm, handler.diameter / 2.0)
    if not ok:
        with _CENTRE_WARN_LOCK:
            if session_dir not in _CENTRE_WARNED:
                _CENTRE_WARNED.add(session_dir)
                print(f'  [centre fit failed] {session_dir}: tracking does not reach the wall '
                      f'in enough directions to locate the arena centre -- frame left as '
                      f'loaded (radius measured from the tracking bounding box)')
        return x_cm, y_cm

    return x_cm + (handler.cx - cx), y_cm + (handler.cy - cy)


# ── Open-field 'Rotate' sessions: in the open field the arena (and its cue) is physically
# rotated by 120 deg between the control/zero sessions and the rotation session, so a
# 'Rotate' session's tracking sits in a frame turned 120 deg relative to the others. Those
# sessions' positions are rotated back by 120 deg CCW about the arena centre before any
# binning, so every open-field session pools onto one common arena frame (the same idea as
# LinearTrackHandler.orient's 90 deg fix for vertical tracks, but keyed off the session
# folder name rather than the tracking's own aspect ratio).
#
# A session is a rotation session when its own (deepest) folder name contains this keyword,
# case-insensitively -- e.g. '3Rotate', '2_Rotate', '1Rotate' all match, while '1Cntrl' /
# '2Zero' do not. The keyword is deliberately the full 'rotate' and not 'rot', so a
# circular-track 'NoRot' folder can never be matched by accident.
ROTATE_SESSION_KEYWORD  = 'rotate'
ROTATE_SESSION_CCW_DEG  = 0.0 #120.0


def _is_rotate_session(session_dir: str) -> bool:
    """True when this session's own folder name marks it as a rotation session (see
    ROTATE_SESSION_KEYWORD). Only the deepest component is tested, so an ancestor folder
    that happens to contain the keyword does not flag every session beneath it."""
    return ROTATE_SESSION_KEYWORD in os.path.basename(os.path.normpath(session_dir)).lower()


def _apply_session_rotation(x_cm: np.ndarray, y_cm: np.ndarray, session_dir: str,
                            handler) -> tuple:
    """Rotates an OPEN FIELD 'Rotate' session's tracking by ROTATE_SESSION_CCW_DEG counter-
    clockwise about the arena centre, so it pools onto the same frame as that day's control
    sessions. Any other arena, or any non-rotation session, is returned unchanged.

    PRECONDITION: the frame must already have been centred by _centre_open_field_tracking, so
    that (handler.cx, handler.cy) really is the arena centre -- the physical axis the arena was
    turned about, and the same point to_bins/geom_valid/edge_zone_flat measure radius from.
    Rotating about (cx, cy) preserves every sample's distance from it, so this step leaves
    radius-derived quantities (the edge/centre split, geom_valid membership, distance-to-wall)
    mathematically unchanged and only re-assigns angular position. Without the centring step
    that invariant does not hold: (cx, cy) would be the midpoint of the session's tracking
    bounding box, and rotating about it would silently translate the data relative to the real
    arena centre.

    Note that the invariance is exact in continuous coordinates but not after binning: the
    square bin lattice is not rotation-invariant, so which bins clear min_occ_s shifts by a
    fraction of a percent. That is why the rotation must not decide session inclusion on its
    own -- see the note in session_zone_coverage.
    """
    if _detect_arena_key(session_dir) != 'open_field' or not _is_rotate_session(session_dir):
        return x_cm, y_cm
    if len(x_cm) == 0:
        return x_cm, y_cm

    theta = np.radians(ROTATE_SESSION_CCW_DEG)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx = x_cm - handler.cx
    dy = y_cm - handler.cy
    xr = handler.cx + dx * cos_t - dy * sin_t
    yr = handler.cy + dx * sin_t + dy * cos_t
    return xr, yr


fps           = 30           # tracking frame rate (Hz)
target_bin_cm  = 2.0          # spatial bin size (cm / along-track cm), fine-map resolution
min_occ_s      = 1          # exclude bins with < 0.5 s occupancy
# COVERAGE_FRACTION / USE_BIN_COVERAGE_CRITERION (whole-arena bin coverage) are set in the
# CONFIGURATION block at the top of the file.

# ── Zone-wise (edge vs centre) session coverage criterion -- ported from
# BoundaryAnalysis_Pipeline_v3.py's MIN_ZONE_COVERAGE_PCT / classify_edge_centre. In addition
# to the whole-arena COVERAGE_FRACTION check above, a session is only included if the
# animal's tracked occupancy covers BOTH the edge zone AND the centre zone by at least this
# percentage of the session's total occupied time -- a session that fails this bar has its
# whole folder (every unit in it) skipped, computed from tracking alone before any spikes are
# loaded (see session_zone_coverage / collect_arena_results). Switch and threshold
# (USE_ZONE_COVERAGE_CRITERION, MIN_ZONE_COVERAGE_PCT) are set in the CONFIGURATION block at
# the top of the file.

OPEN_EDGE_ZONE_THRESHOLD_CM   = 9.25  # open field: bins within this distance of the wall are 'edge'
LINEAR_EDGE_ZONE_THRESHOLD_CM = 2.0   # linear track: bins within this distance of either long wall
                                       # (width axis) or either end wall (length axis) are 'edge'
# circular track needs no threshold: with only ny bins radially, a bin is 'edge' when nearer the
# outer wall than the inner wall (CircularTrackHandler.edge_zone_flat / inner_side_flat)

MAX_GAP_US     = 50_000       # max spike-position gap (us)
N_BOOTSTRAP    = 1000         # circular-shift shuffles for SIR significance (reduce for faster runs)
MAX_WORKERS    = 4

# Master seed for the per-unit SIR shuffle bootstrap (run_bootstrap_generic). Each unit draws
# its shifts from its own random.Random, seeded from this plus the unit's identity
# (_stable_seed), so the reported bootstrap_sig is reproducible run to run and independent of
# how the ThreadPoolExecutor happened to interleave units. Before this, the bootstrap drew
# from the unseeded global `random` module, which made the bootstrap_sig column of
# AllArenas_Summary.xlsx differ on every run, for every arena -- noise that is easy to mistake
# for a real effect when diffing two runs. Change it only to check robustness to the seed.
BOOTSTRAP_SEED = 0

POS_JUMP_THRESH_CMS  = 90.0   # frame-to-frame jumps implying a speed above this (cm/s) are tracking artifacts
POS_SMOOTH_SIGMA_SMP = 1.0    # Gaussian smoothing sigma (samples) applied to x/y tracking position
RATEMAP_SMOOTH_SIGMA_BINS = 3.0  # Gaussian smoothing sigma (bins) applied to rate maps

# Place-field extraction (pass-index 'place' filter-band criteria, auto_filter_band in
# pass_index_parser.m: field = area with raw rate >= 20% of the raw peak): a bin qualifies
# if its RAW (unsmoothed) rate exceeds both the cell's mean firing rate and this fraction
# of the raw rate map's peak; a connected run of qualifying bins is only kept as a field if
# it spans at least MIN_FIELD_BINS contiguous bins.
FIELD_PEAK_FRAC = 0.20
MIN_FIELD_BINS  = 9

# ── Colour scaling of the quadrant-fold figures (Fig1BD_QuadrantFold.png,
# QuadrantFold_KDE.png). Both are 3-arena grids, and this decides whether the three arenas
# in a row share one colour scale or each gets its own:
#
#   'shared'    -- one normalization per row, pooled over all three arenas (the original
#                  behaviour). Colours ARE comparable between arenas. The cost is that every
#                  panel's colours depend on all three arenas' data, so a change confined to
#                  one arena silently repaints the other two even though their numbers are
#                  bit-identical -- which is very easy to misread as "my change broke the
#                  linear track". If you are diffing two runs, expect this.
#   'per_arena' -- each panel normalized to its own range, with its own colorbar. Panels are
#                  then independent, so an open-field change cannot touch the linear/circular
#                  panels at all. The cost is that colours are NOT comparable between arenas,
#                  and a panel whose true dynamic range is narrow (the folded mean field-index
#                  maps span only ~0.17-0.25 a.u.) gets that narrow band stretched across the
#                  whole colormap, which makes near-flat maps look highly structured.
#
# Default 'shared' to keep the published figures looking as they always have; switch to
# 'per_arena' when you specifically need panel independence.
QUADRANT_COLOUR_SCALE = 'shared'

# 'pixel' or 'cm'
COORD_UNITS = 'cm'

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


def _session_positions(csv_path: str, handler) -> tuple:
    """THE entry point for one session's tracking: load, clean, smooth, then apply the
    open-field frame corrections (_centre_open_field_tracking, then _apply_session_rotation).
    Returns (x_cm, y_cm, t) ready for handler.orient / handler.to_bins.

    Every consumer -- the coverage screen, the per-unit rate maps, the debug maps and the
    external ListIncludedSessionsCells.py audit -- must go through here rather than calling
    _load_tracking/_smooth_tracking_position itself. When those corrections lived at the call
    sites instead, two paths (the audit script and the debug map builder) silently skipped
    them and reported a different set of included sessions than the pipeline actually used.

    The order matters: centring first (it defines where the arena centre is), rotation second
    (it turns the data about that centre). Both are no-ops outside the open field."""
    x_cm, y_cm, t = _load_tracking(csv_path, handler.arena_width_cm)
    if len(t) < 2:
        return x_cm, y_cm, t
    x_cm, y_cm = _smooth_tracking_position(x_cm, y_cm, t)
    session_dir = os.path.dirname(csv_path)
    x_cm, y_cm = _centre_open_field_tracking(x_cm, y_cm, session_dir, handler)
    x_cm, y_cm = _apply_session_rotation(x_cm, y_cm, session_dir, handler)
    return x_cm, y_cm, t


def _stable_seed(*parts) -> int:
    """Deterministic 64-bit seed from a unit's identity (session name + file name).

    Python's built-in hash() is salted per process unless PYTHONHASHSEED is pinned, so it
    cannot be used for results that must reproduce across runs. Seeding per unit rather than
    seeding the `random` module once also makes the bootstrap independent of thread
    scheduling: collect_arena_results runs units through a ThreadPoolExecutor, and a single
    shared generator would hand out a different shuffle sequence to each unit on every run."""
    key = '|'.join(str(p) for p in parts).encode('utf-8')
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'big') ^ BOOTSTRAP_SEED


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


def _draw_bin_outline(ax, mask_2d: np.ndarray, x_centers: np.ndarray, y_centers: np.ndarray,
                       wrap_x: bool = False):
    """Outlines the True bins of a folded (nx, ny) mask on top of an already-drawn quadrant map
    (imshow for the open field/linear track, pcolormesh for the circular track's polar axes).

    x_centers/y_centers are the BIN CENTRE coordinates of the map underneath, in that map's own
    units (cm, or theta/radius for the polar axes): contour interpolates between the grid nodes
    it is given, so a 0.5-level crossing between a significant bin's centre and its
    non-significant neighbour's centre lands exactly on the edge the two bins share.

    The mask is zero-padded by one bin (and the centre coordinates extended to match) so that a
    significant bin at the edge of the folded grid gets a closed outline rather than a line that
    stops at the data boundary -- without the padding there is no node beyond that bin for the
    level to cross between, so the outline is left open along the grid's own border. A white
    underlay keeps the black line readable over the dark blue and dark red ends of 'jet'.

    wrap_x=True treats axis 0 as a closed ring (the circular track's whole, unfolded arc-length
    axis, theta=0 meeting theta=2*pi) and pads that border from the OPPOSITE end of the mask
    instead of zero, so a significant run straddling the seam gets one continuous outline
    instead of two that each stop dead at the plot's edge."""
    if not mask_2d.any():
        return
    nx, ny = mask_2d.shape
    padded = np.zeros((nx + 2, ny + 2), dtype=float)
    padded[1:-1, 1:-1] = mask_2d.astype(float)
    if wrap_x:
        padded[0, 1:-1]  = mask_2d[-1, :].astype(float)
        padded[-1, 1:-1] = mask_2d[0, :].astype(float)

    def _extend(centers):
        step = centers[1] - centers[0] if len(centers) > 1 else 1.0
        return np.concatenate(([centers[0] - step], centers, [centers[-1] + step]))

    xp, yp = _extend(np.asarray(x_centers, dtype=float)), _extend(np.asarray(y_centers, dtype=float))
    # the padding ring sits outside the map, so restore the limits the map itself set rather
    # than letting contour autoscale a one-bin white margin around every panel
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.contour(xp, yp, padded.T, levels=[0.5], colors='white', linewidths=3.0, zorder=5)
    ax.contour(xp, yp, padded.T, levels=[0.5], colors='black', linewidths=1.6, zorder=6)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


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

        # edge/centre zone classification (MIN_ZONE_COVERAGE_PCT criterion): a bin is 'edge'
        # when within OPEN_EDGE_ZONE_THRESHOLD_CM of the wall, else 'centre' -- same rule as
        # classify_edge_centre's open-field branch in BoundaryAnalysis_Pipeline_v3.py
        self.edge_zone_flat = self.dist_to_wall_flat <= OPEN_EDGE_ZONE_THRESHOLD_CM

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
        # outline of the physical arena wall (diameter = 60 cm)
        ax.add_patch(matplotlib.patches.Circle((self.cx, self.cy), self.diameter / 2.0,
                                               fill=False, edgecolor='0.35', lw=1.0, zorder=5))
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
        centers = (np.arange(half) + 0.5) * self.bin_cm
        _draw_bin_outline(ax, sig_mask.reshape(half, half), centers, centers)

    def overlay_fine_significance(self, ax, sig_mask: np.ndarray):
        """Whole-arena analogue of overlay_quadrant_significance: outlines, on top of an
        existing plot_fine axes, the bins where the fine-map KDE analysis (analyze_fine_kde)
        found the observed (unfolded) map's bootstrap CI to significantly exceed the
        occupancy-null KDE."""
        x_centers = (np.arange(self.nx) + 0.5) * self.bin_cm
        y_centers = (np.arange(self.ny) + 0.5) * self.bin_cm
        _draw_bin_outline(ax, sig_mask.reshape(self.nx, self.ny), x_centers, y_centers)


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

        # edge/centre zone classification (MIN_ZONE_COVERAGE_PCT criterion): the outer-wall-
        # touching ring of bins is 'edge', the inner-wall-touching ring is 'centre' -- same
        # rule as classify_edge_centre's circular-track branch in
        # BoundaryAnalysis_Pipeline_v3.py (no threshold needed, the track is only ny bins
        # wide radially)
        self.edge_zone_flat = ~self.inner_side_flat

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
        theta_centers = (np.arange(qn) + 0.5) * ((np.pi / 2) / qn)
        r_centers = self.inner_r + (np.arange(ny) + 0.5) * ((self.outer_r - self.inner_r) / ny)
        _draw_bin_outline(ax, sig_mask.reshape(qn, ny), theta_centers, r_centers)

    def overlay_fine_significance(self, ax, sig_mask: np.ndarray):
        """Whole-ring analogue of overlay_quadrant_significance: outlines, on top of an
        existing plot_fine polar axes, the bins flagged significant by analyze_fine_kde.
        wrap_x=True because this is the unfolded arc-length axis, a closed ring (theta=0
        meets theta=2*pi), unlike the folded quarter-arc overlay_quadrant_significance draws."""
        theta_centers = (np.arange(self.nx) + 0.5) * (2 * np.pi / self.nx)
        r_centers = self.inner_r + (np.arange(self.ny) + 0.5) * ((self.outer_r - self.inner_r) / self.ny)
        _draw_bin_outline(ax, sig_mask.reshape(self.nx, self.ny), theta_centers, r_centers, wrap_x=True)


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

        # edge/centre zone classification (MIN_ZONE_COVERAGE_PCT criterion): a bin is 'edge'
        # when it is at least LINEAR_EDGE_ZONE_THRESHOLD_CM from the track's midline (width
        # axis) OR within LINEAR_EDGE_ZONE_THRESHOLD_CM of either end wall (length axis),
        # else 'centre' -- same rule as classify_edge_centre's linear-track branch in
        # BoundaryAnalysis_Pipeline_v3.py
        dist_from_midline_cm = (self.width / 2.0) - DY
        edge_zone_2d = ((dist_from_midline_cm >= LINEAR_EDGE_ZONE_THRESHOLD_CM) |
                        (DX <= LINEAR_EDGE_ZONE_THRESHOLD_CM))
        self.edge_zone_flat = edge_zone_2d.ravel()

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
        handler's own per-axis (length vs. width) bin size."""
        half_x, half_y = self.quad_shape
        x_centers = (np.arange(half_x) + 0.5) * self.bin_cm_x
        y_centers = (np.arange(half_y) + 0.5) * self.bin_cm_y
        _draw_bin_outline(ax, sig_mask.reshape(half_x, half_y), x_centers, y_centers)

    def overlay_fine_significance(self, ax, sig_mask: np.ndarray):
        """Whole-track analogue of overlay_quadrant_significance: outlines, on top of an
        existing plot_fine axes, the bins flagged significant by analyze_fine_kde, using this
        handler's own per-axis (length vs. width) bin size."""
        x_centers = (np.arange(self.nx) + 0.5) * self.bin_cm_x
        y_centers = (np.arange(self.ny) + 0.5) * self.bin_cm_y
        _draw_bin_outline(ax, sig_mask.reshape(self.nx, self.ny), x_centers, y_centers)


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


def run_bootstrap_generic(handler, cell: dict, real_sir: float, n_bootstrap: int = N_BOOTSTRAP,
                          seed: int | None = None) -> dict:
    """Location-shuffling bootstrap (Fenton circular-shift method), generalized to any
    flat-bin arena: the position->bin mapping is fixed, only which frame each spike is
    credited to is circularly shifted.

    `seed` selects this unit's own random.Random stream (see _stable_seed / BOOTSTRAP_SEED);
    callers pass a value derived from the unit's identity so the result is reproducible and
    does not depend on thread interleaving."""
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

    rand = random.Random(BOOTSTRAP_SEED if seed is None else seed)
    sir_i = np.zeros(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        rnd = rand.randint(MARGIN_FRAMES, n_frames - MARGIN_FRAMES)
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


def _zone_occupancy_pct(handler, occ_map: np.ndarray, valid_mask: np.ndarray) -> tuple:
    """Edge vs centre split of total occupied time (percent of the session's total occupied
    time spent in each zone), ported from BoundaryAnalysis_Pipeline_v3.py's
    _zone_occupancy_pct -- used by the MIN_ZONE_COVERAGE_PCT session-screening criterion."""
    total_occ_s = float(occ_map[valid_mask].sum())
    if total_occ_s <= 0:
        return 0.0, 0.0
    edge_mask    = handler.edge_zone_flat & valid_mask
    edge_occ_s   = float(occ_map[edge_mask].sum())
    centre_occ_s = total_occ_s - edge_occ_s
    return 100.0 * edge_occ_s / total_occ_s, 100.0 * centre_occ_s / total_occ_s


def session_zone_coverage(csv_path: str, handler) -> tuple | None:
    """Computes one session's edge-vs-centre occupancy-time split from tracking alone (no
    spike data needed), for the MIN_ZONE_COVERAGE_PCT session-screening criterion --
    tracking is identical for every unit in a session, so this is computed once per session
    rather than once per unit. Returns (edge_pct, centre_pct), or None if the tracking file
    yields no valid samples."""
    # _session_positions applies the open-field frame corrections (centring, then the 120 deg
    # 'Rotate' fix) so this screen sees exactly the positions the rate maps will.
    #
    # CAVEAT worth knowing when reading this screen's numbers: centring genuinely changes the
    # edge/centre split (it puts each sample at its true distance from the arena centre), but
    # the rotation should not -- it preserves distance from that centre exactly. In binned
    # form it still nudges the result by a few tenths of a percent, because the square bin
    # lattice is not rotation-invariant, and a session sitting on a threshold can therefore
    # flip inclusion purely on frame choice. If that matters for a future dataset, screen on
    # the centred-but-unrotated positions; it is the same quantity, measured without the
    # lattice noise.
    x_cm, y_cm, t = _session_positions(csv_path, handler)
    if len(t) < 2:
        return None
    x_cm, y_cm = handler.orient(x_cm, y_cm)
    bin_idx, sample_valid = handler.to_bins(x_cm, y_cm)
    t = t[sample_valid]
    bin_idx = bin_idx[sample_valid]
    if len(t) < 2:
        return None

    n = len(t)
    dt_frames = np.empty(n, dtype=np.float64)
    dt_frames[0] = 1.0 / fps
    raw_dt = np.diff(t) * 1e-6
    dt_frames[1:] = np.minimum(raw_dt, 2.0 / fps)

    occ_map = np.zeros(handler.n_bins, dtype=np.float64)
    np.add.at(occ_map, bin_idx, dt_frames)

    occ_valid  = occ_map >= min_occ_s
    geom_valid = getattr(handler, 'geom_valid', np.ones(handler.n_bins, dtype=bool))
    valid_mask = occ_valid & geom_valid

    return _zone_occupancy_pct(handler, occ_map, valid_mask)


def process_unit(csv_path: str, ntt_path: str, ntt_file: str, session_name: str, handler) -> dict | None:
    x_cm, y_cm, t = _session_positions(csv_path, handler)
    if len(t) < 2:
        return None

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
    # the arena (COVERAGE_FRACTION), not just this one unit. Not applied when
    # USE_BIN_COVERAGE_CRITERION is False.
    covered_bins = int(cell['valid'].sum())
    if USE_BIN_COVERAGE_CRITERION and covered_bins < handler.coverage_threshold_bins:
        print(f'  [SKIP low coverage] {session_name}/{ntt_file}: '
              f'{covered_bins}/{handler.total_arena_bins} bins covered '
              f'({covered_bins / handler.total_arena_bins:.0%}), '
              f'need >= {handler.coverage_threshold_bins} ({COVERAGE_FRACTION:.0%})')
        return None

    # No place-cell qualification here: every unit under ROOT_DIRECTORY has already been
    # qualified as a place cell by the upstream pipeline that built that folder, so every
    # unit of every coverage-passing session enters the analysis. The bootstrap below is
    # kept purely as a reported diagnostic (bootstrap_sig in the summary workbook) -- it no
    # longer gates anything.
    boot = run_bootstrap_generic(handler, cell, cell['sir'],
                                 seed=_stable_seed(session_name, ntt_file))
    boot_sig = boot.get('bootstrap_sig')

    peak_bin = None
    if cell['valid'].any():
        masked = np.where(cell['valid'], cell['fr_smooth'], -np.inf)
        peak_bin = int(np.argmax(masked))

    field_mask = extract_place_field_mask(cell, handler)

    return dict(
        session=session_name, unit=ntt_file,
        n_spikes=cell['n_spikes'], peak_fr=cell['peak_fr'], mean_fr=cell['mean_fr'],
        sir=cell['sir'], sparsity=cell['sparsity'],
        bootstrap_sig=boot_sig,
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
            session_name = os.path.relpath(dirpath, root)

            # Zone-wise (edge/centre) coverage screen (MIN_ZONE_COVERAGE_PCT): computed once
            # per session from tracking alone -- a session whose tracking does not cover both
            # zones by at least MIN_ZONE_COVERAGE_PCT has its whole folder skipped, before any
            # spikes are loaded. Skipped entirely when USE_ZONE_COVERAGE_CRITERION is False.
            if USE_ZONE_COVERAGE_CRITERION:
                try:
                    zone_cov = session_zone_coverage(csv_path, handler)
                except Exception as e:
                    print(f'  ZONE COVERAGE ERROR [{arena_key}] {session_name}: {e}')
                    zone_cov = None
                if zone_cov is None:
                    print(f'  [SKIP no tracking] [{arena_key}] {session_name}: could not compute zone coverage')
                    continue
                edge_pct, centre_pct = zone_cov
                if edge_pct < MIN_ZONE_COVERAGE_PCT or centre_pct < MIN_ZONE_COVERAGE_PCT:
                    print(f'  [SKIP zone coverage] [{arena_key}] {session_name}: '
                          f'edge={edge_pct:.1f}%  centre={centre_pct:.1f}%  '
                          f'(need >= {MIN_ZONE_COVERAGE_PCT:.0f}% in each zone)')
                    continue

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

    print(f'\n[{arena_key}] {len(jobs)} units scanned, {len(results)} processed '
          f'(all of them pooled -- cells are pre-qualified upstream).')
    return handler, results


# ============================================================================
# Pooling across cells / days
# ============================================================================

def pool_fine_map(handler, results: list) -> tuple:
    """Pools each place cell's field index map (0-1 normalized to its own peak and
    minimum firing rate, see field_index_map) rather than raw Hz -- so a cell's
    contribution to the pooled map does not depend on its absolute peak rate or on
    where its field happens to sit relative to other cells' fields."""
    place = results   # every unit is already a qualified place cell (see module docstring)
    if not place:
        return np.full(handler.n_bins, np.nan), np.zeros(handler.n_bins, dtype=bool)
    stack = np.full((len(place), handler.n_bins), np.nan)
    for i, r in enumerate(place):
        stack[i, r['valid']] = r['fi_map'][r['valid']]
    mean_map = np.nanmean(stack, axis=0)
    any_valid = ~np.all(np.isnan(stack), axis=0)
    return mean_map, any_valid


def pool_field_only_map(handler, results: list) -> tuple:
    """Field-density-weighted mean field-index map, pooled ONLY over each place cell's own
    extracted place-field bins (extract_place_field_mask) -- bins a cell never fielded in do
    not dilute its contribution, per Mean_RM_Fixes.md ('Use only place fields to construct
    the mean RM'). Like pool_fine_map, each cell is normalized to its own 0-1 field index
    (fi_map) before pooling, so a cell's contribution does not depend on its absolute peak
    rate.

    Each bin's value is (sum of field index over every field covering that bin) / (total
    number of (cell, bin) field-coverage pairs pooled across the whole map). This is
    algebraically (# fields covering this bin / total field-bin instances) * (mean field
    index of those fields) -- so a bin where many fields overlap is weighted up relative to
    a bin only one weak field passes through, AND a bin covered by high-field-index (near
    each field's own peak) contributions is weighted up relative to one covered only by
    field edges. This replaces the older occupancy-weighted-mean design, which normalized
    per bin so field-overlap density could never show through."""
    place = results   # every unit is already a qualified place cell (see module docstring)
    if not place:
        return np.full(handler.n_bins, np.nan), np.zeros(handler.n_bins, dtype=bool)

    fi_sum   = np.zeros(handler.n_bins, dtype=np.float64)
    n_fields = np.zeros(handler.n_bins, dtype=np.float64)
    for r in place:
        m = r['field_mask'] & r['valid']
        if not m.any():
            continue
        fi_sum[m]   += r['fi_map'][m]
        n_fields[m] += 1.0

    total_field_bin_instances = n_fields.sum()
    mean_map = np.full(handler.n_bins, np.nan)
    any_valid = n_fields > 0
    if total_field_bin_instances > 0:
        mean_map[any_valid] = fi_sum[any_valid] / total_field_bin_instances
    return mean_map, any_valid


def pool_quadrant_peak_proportion(handler, results: list) -> tuple:
    """Fig 1B: proportion of place-cell peaks, folded onto one reference quadrant
    (Fig 1A). Each cell contributes its one peak bin, remapped into the reference
    quadrant's local coordinates -- unlike the mean-rate map below, a peak is a
    single location so it is not itself averaged across the 4 folded regions."""
    place = [r for r in results if r['peak_bin'] is not None]
    counts = np.zeros(handler.n_quad_bins)
    for r in place:
        q = handler.fold_peak_bin(r['peak_bin'])
        if q >= 0:
            counts[q] += 1
    total = counts.sum()
    pct = counts / total * 100.0 if total > 0 else np.full(handler.n_quad_bins, np.nan)
    return pct, int(total)


def pool_peak_proportion_fine(handler, results: list) -> tuple:
    """Unfolded analogue of pool_quadrant_peak_proportion: % of place-cell peaks per bin
    across the WHOLE (unfolded) arena grid, instead of one reference quadrant -- the 'peak'
    map type for the fine-map KDE analysis below (analyze_fine_kde)."""
    place = [r for r in results if r['peak_bin'] is not None]
    counts = np.zeros(handler.n_bins)
    for r in place:
        counts[r['peak_bin']] += 1
    total = counts.sum()
    pct = counts / total * 100.0 if total > 0 else np.full(handler.n_bins, np.nan)
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
    field-density-weighted mean field-index map (pool_field_only_map) into the same reference
    quadrant, instead of the unrestricted overall mean map -- the third of the three folded
    rate maps (alongside pool_quadrant_mean_rate and pool_quadrant_peak_proportion) that the
    quadrant-fold KDE analysis below compares against an occupancy-null KDE."""
    mean_map, valid = pool_field_only_map(handler, results)
    return handler.fold_mean_map(np.where(valid, mean_map, 0.0), valid)


# ============================================================================
# Uniform-field null (expected maps if place fields tile the arena uniformly)
#
# The original null was the pooled dwell-time map itself. That is not what a population of
# place cells would produce under "no spatial preference": a cell contributes to the overall
# / field-only / peak maps through its FIELDS, whose footprint has a finite size, so bins near
# a wall (which a fixed-size field can only reach from one side) are covered by fewer
# possible fields than bins in the middle of the arena, whatever the animal's dwell time.
#
# Generative model used here (build_uniform_field_null):
#   * Every arena has one fixed field footprint of FIELD_FOOTPRINT_BINS[arena] contiguous
#     bins (linear 7, circular 9, open field 12). A footprint may be placed at ANY position
#     where it fits wholly inside the arena (wrapping round the ring for the circular track),
#     and every such placement is equally likely -- i.e. field locations are uniform over the
#     arena. Mirror-image variants of a footprint are averaged so the shape has no handedness.
#   * Occupancy is applied PER SESSION (each session that survived the coverage criteria is
#     represented once per unit it contributed, exactly as the real pooling does it): the
#     field the session can actually observe is the footprint's intersection with that
#     session's valid bins (occupancy >= min_occ_s); a placement whose observed part is
#     smaller than MIN_FIELD_BINS would not have been detected as a field and is dropped.
#   * 'overall' : each synthetic cell has field index 1 inside its observed field and 0 in
#                 the rest of that session's valid bins -> expected mean = probability a valid
#                 bin lies in a field (the analogue of pool_fine_map's nanmean over cells).
#   * 'field'   : expected field-coverage counts, normalised like pool_field_only_map.
#   * 'peak'    : the peak bin of a flat-topped field is not defined by the rate map, so it is
#                 drawn within the observed field with probability proportional to the
#                 session's PROPORTION OF DWELL TIME in each bin (PEAK_WITHIN_FIELD).
#
# NULL_MODE selects which null the KDE analyses treat as primary; the other is always
# computed as well so the two can be compared (see export_null_comparison).
# ============================================================================

NULL_MODE = 'uniform_field'    # 'uniform_field' (this model) or 'occupancy' (old pooled dwell-time null)

# field footprint size in bins, per arena (see model description above)
FIELD_FOOTPRINT_BINS = {'linear_track': 7, 'circular_track': 9, 'open_field': 12}

# 'occupancy': peak bin drawn within the field in proportion to session dwell-time proportion
# 'uniform'  : peak bin drawn uniformly over the field's observed bins
PEAK_WITHIN_FIELD = 'occupancy'


def _arena_key_of(handler) -> str:
    if isinstance(handler, OpenFieldHandler):
        return 'open_field'
    if isinstance(handler, CircularTrackHandler):
        return 'circular_track'
    if isinstance(handler, LinearTrackHandler):
        return 'linear_track'
    raise TypeError(f'Unknown handler type: {type(handler)}')


def _footprint_variants(handler, arena_key: str) -> list:
    """Field footprint(s) for one arena as (n, 2) int arrays of (dx, dy) bin offsets.

    open_field   : the n bins nearest a bin CORNER (12 -> a 4x4 block with its 4 corners cut,
                   the most compact 12-bin shape; symmetric so a single variant).
    tracks       : the field spans the track's width and runs along its length/arc (the tracks
                   are only ny bins wide): full-width columns, the last one partial and centred.
                   Its 4 mirror images (length flip, width flip) are all returned."""
    n = FIELD_FOOTPRINT_BINS[arena_key]
    if arena_key == 'open_field':
        span = 2 * int(np.ceil(np.sqrt(n) / 2.0)) + 2
        ii, jj = np.meshgrid(np.arange(span), np.arange(span), indexing='ij')
        d = np.hypot(ii + 0.5 - span / 2.0, jj + 0.5 - span / 2.0).ravel()
        take = np.argsort(d, kind='stable')[:n]
        pts = np.stack([ii.ravel()[take], jj.ravel()[take]], axis=1)
        return [pts - pts.min(axis=0)]

    ny = handler.ny
    cols = int(np.ceil(n / ny))
    last = n - (cols - 1) * ny
    r0 = (ny - last) // 2
    pts = [(c, r) for c in range(cols - 1) for r in range(ny)]
    pts += [(cols - 1, r0 + r) for r in range(last)]
    variants, seen = [], set()
    for fx in (False, True):
        for fy in (False, True):
            v = frozenset(((cols - 1 - c) if fx else c, (ny - 1 - r) if fy else r) for c, r in pts)
            if v not in seen:
                seen.add(v)
                variants.append(np.array(sorted(v), dtype=int))
    return variants


def _uniform_field_placements(handler, arena_key: str) -> tuple:
    """Every allowed placement of the arena's field footprint. Returns (F, w): F is a
    (n_placements, n_bins) bool matrix of footprint bins, w the placement weights (sum 1;
    each mirror variant gets equal total weight, placements within a variant are uniform)."""
    nx, ny = handler.nx, handler.ny
    wrap = (arena_key == 'circular_track')
    geom = getattr(handler, 'geom_valid', np.ones(handler.n_bins, dtype=bool))
    rows, weights = [], []
    for offs in _footprint_variants(handler, arena_key):
        dx, dy = offs[:, 0], offs[:, 1]
        sx_range = range(nx) if wrap else range(0, nx - int(dx.max()))
        sy_range = range(0, ny - int(dy.max()))
        v_rows = []
        for sx in sx_range:
            bx = (sx + dx) % nx if wrap else sx + dx
            for sy in sy_range:
                flat = bx * ny + (sy + dy)
                if not geom[flat].all():
                    continue
                m = np.zeros(handler.n_bins, dtype=bool)
                m[flat] = True
                v_rows.append(m)
        rows.extend(v_rows)
        weights.extend([1.0 / len(v_rows)] * len(v_rows))
    if not rows:
        raise ValueError(f'{arena_key}: no placement of the {FIELD_FOOTPRINT_BINS[arena_key]}-bin '
                         f'field footprint fits inside the arena')
    w = np.asarray(weights)
    return np.array(rows), w / w.sum()


def build_uniform_field_null(handler, results: list) -> dict:
    """Expected overall / field-only / peak maps for uniformly tiled fields, given each
    session's occupancy (model description above). Returns
        overall, field : (values, valid) flat maps, directly comparable to pool_fine_map /
                         pool_field_only_map
        peak_pct       : % of peaks per bin, comparable to pool_peak_proportion_fine
        field_weight   : sum over synthetic fields of dwell time (s) in each field bin -- the
                         null's analogue of _pooled_occupancy(field_only=True)
        n_cells, n_sessions, n_placements
    """
    arena_key = _arena_key_of(handler)
    F, bw = _uniform_field_placements(handler, arena_key)

    by_session = {}
    for r in results:
        entry = by_session.setdefault(r['session'], [r, 0])
        entry[1] += 1

    n = handler.n_bins
    cov_sum    = np.zeros(n)   # sum over cells of P(bin in that cell's observed field)
    valid_sum  = np.zeros(n)   # number of cells for which the bin is valid
    fld_weight = np.zeros(n)
    peak_mass  = np.zeros(n)
    n_cells = n_sessions = 0
    for r, n_s in by_session.values():
        V = r['valid']
        O = F & V[None, :]
        keep = O.sum(axis=1) >= MIN_FIELD_BINS
        if not keep.any():
            continue
        w = np.where(keep, bw, 0.0)
        w = w / w.sum()
        Of = O.astype(np.float64)
        cov = w @ Of

        occ = np.where(V, r['occ_map'], 0.0)
        pi = occ / occ.sum() if PEAK_WITHIN_FIELD == 'occupancy' else V.astype(np.float64)
        denom = Of @ pi
        ok = keep & (denom > 0)
        wp = np.where(ok, w / np.where(ok, denom, 1.0), 0.0)
        peak_s = pi * (wp @ Of)          # P(peak = bin), sums to ~1 over the session's fields

        cov_sum    += n_s * cov
        valid_sum  += n_s * V
        fld_weight += n_s * cov * occ
        peak_mass  += n_s * peak_s
        n_cells    += n_s
        n_sessions += 1

    overall = np.zeros(n)
    m = valid_sum > 0
    overall[m] = cov_sum[m] / valid_sum[m]
    total_cov = cov_sum.sum()
    field = cov_sum / total_cov if total_cov > 0 else np.zeros(n)
    pct = (peak_mass / peak_mass.sum() * 100.0) if peak_mass.sum() > 0 else np.full(n, np.nan)

    return dict(overall=(overall, m & (cov_sum > 0)), field=(field, cov_sum > 0),
                field_weight=fld_weight, peak_pct=pct,
                n_cells=n_cells, n_sessions=n_sessions, n_placements=len(F))


def _uniform_null_fine_map(handler, cells: list, map_type: str) -> tuple:
    """(values, valid) of the uniform-field null for one map type on the whole-arena grid --
    same interface as the observed builders in _FINE_MAP_BUILDERS / pool_peak_proportion_fine."""
    nul = build_uniform_field_null(handler, cells)
    if map_type == 'peak':
        pct = nul['peak_pct']
        return pct, np.isfinite(pct)
    return nul[map_type]


def _uniform_null_quadrant_map(handler, cells: list, map_type: str) -> tuple:
    """(values, valid) of the uniform-field null folded onto the reference quadrant, using the
    same fold operations as the observed pool_quadrant_* maps."""
    vals, valid = _uniform_null_fine_map(handler, cells, map_type)
    if map_type == 'peak':
        q = handler._quad_idx_flat
        counts = np.zeros(handler.n_quad_bins)
        m = (q >= 0) & valid
        np.add.at(counts, q[m], vals[m])
        total = counts.sum()
        pct = counts / total * 100.0 if total > 0 else np.full(handler.n_quad_bins, np.nan)
        return pct, np.isfinite(pct)
    return handler.fold_mean_map(np.where(valid, vals, 0.0), valid)


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
#   'field'   : the place-field-only mean map (pool_field_only_map), FIELD-DENSITY-weighted
#               mean across spatial bins sharing a position (mirrors pool_field_only_map's
#               own field-overlap-count x field-index weighting).
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
    place = results   # every unit is already a qualified place cell (see module docstring)
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
                       bin_edges: np.ndarray, side_mask: np.ndarray, wrap: bool,
                       null_mode: str | None = None) -> np.ndarray:
    """Null curve (% of its own total), paired with _build_observed_curve so mean-type
    observed curves ('overall'/'field') get a mean-type null and the count-type observed
    curve ('peak') gets a sum-type (mass-like) null.

    null_mode 'occupancy' is the original pooled dwell-time null; 'uniform_field' is the
    expected-map null of build_uniform_field_null, pushed through exactly the same
    bin-average -> smooth -> normalise steps as the observed curve."""
    if (NULL_MODE if null_mode is None else null_mode) == 'uniform_field':
        nul = build_uniform_field_null(handler, cells)
        if map_type == 'overall':
            vals, valid = nul['overall']
            _, avg, avg_valid = _distance_bin_average(pos_flat, vals, np.ones(handler.n_bins),
                                                       valid & side_mask, bin_edges)
        elif map_type == 'field':
            vals, valid = nul['field']
            _, avg, avg_valid = _distance_bin_average(pos_flat, vals, nul['field_weight'],
                                                       valid & side_mask, bin_edges)
        else:  # 'peak'
            pct = nul['peak_pct']
            _, avg, avg_valid = _position_sum(pos_flat, np.nan_to_num(pct),
                                               np.isfinite(pct) & side_mask, bin_edges)
        return _normalize_pct(_gaussian_smooth_1d_nan(avg, avg_valid, BOUNDARY_SMOOTH_SIGMA_BINS, wrap))

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
    place_cells = results   # every unit is already a qualified place cell (see module docstring)
    pos_flat, bin_edges, wrap = _arena_position_axis(handler, arena_key)
    if side_mask is None:
        side_mask = np.ones(handler.n_bins, dtype=bool)
    grid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_pos = len(grid)

    out = dict(map_type=map_type, grid=grid, n_place_cells=len(place_cells), n_samples=0,
               real_med=np.zeros(n_pos), real_lo=np.zeros(n_pos), real_hi=np.zeros(n_pos),
               null_curve=np.zeros(n_pos), null_curve_uniform=np.zeros(n_pos),
               null_curve_occ=np.zeros(n_pos), peaks=[], peaks_uniform=[], peaks_occ=[])
    if not place_cells:
        return out

    real_curve, n_samples = _build_observed_curve(handler, place_cells, map_type, pos_flat,
                                                    bin_edges, side_mask, wrap)
    out['n_samples'] = n_samples
    if n_samples < 2:
        return out

    # both nulls are always built (for export_null_comparison); NULL_MODE picks the primary
    out['null_curve_uniform'] = _build_null_curve(handler, place_cells, map_type, pos_flat,
                                                   bin_edges, side_mask, wrap, null_mode='uniform_field')
    out['null_curve_occ']     = _build_null_curve(handler, place_cells, map_type, pos_flat,
                                                   bin_edges, side_mask, wrap, null_mode='occupancy')
    out['null_curve'] = out['null_curve_uniform'] if NULL_MODE == 'uniform_field' else out['null_curve_occ']

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

    out['peaks_uniform'] = _find_significant_peaks(grid, out['real_med'], out['real_lo'], out['null_curve_uniform'])
    out['peaks_occ']     = _find_significant_peaks(grid, out['real_med'], out['real_lo'], out['null_curve_occ'])
    out['peaks'] = out['peaks_uniform'] if NULL_MODE == 'uniform_field' else out['peaks_occ']
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
                    ax.plot(grid, res['null_curve_uniform'], color=color, ls=':', lw=1.2, alpha=0.9,
                            label=f'{side} uniform-field null')
                    ax.plot(grid, res['null_curve_occ'], color=color, ls='-.', lw=0.8, alpha=0.35,
                            label=f'{side} occupancy null (old)')
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
                ax.plot(grid, res['null_curve_uniform'], color='0.3', ls='--', lw=1.5, label='Uniform-field null')
                ax.plot(grid, res['null_curve_occ'], color='0.65', ls=':', lw=1.2, label='Occupancy null (old)')
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

    null_name = 'uniform-field null' if NULL_MODE == 'uniform_field' else 'occupancy null'
    fig.suptitle('Boundary-preference analysis: avg. firing (or % peaks) vs. position, kernel-smoothed\n'
                 f'(shaded band = position range where the observed curve significantly exceeds the {null_name})')
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


def _build_quadrant_null_kde(handler, cells: list, map_type: str,
                              null_mode: str | None = None) -> np.ndarray:
    """Null counterpart of _build_quadrant_observed_kde, KDE-smoothed on the folded grid and
    rescaled to % of its own total. null_mode 'occupancy' folds pooled dwell-time
    (field-restricted for 'field', unrestricted for 'overall'/'peak'); 'uniform_field' folds
    the expected maps of build_uniform_field_null instead (_uniform_null_quadrant_map)."""
    if (NULL_MODE if null_mode is None else null_mode) == 'uniform_field':
        vals, valid = _uniform_null_quadrant_map(handler, cells, map_type)
        vals = np.where(valid, vals, 0.0)
        return _normalize_pct(_kde_smooth_quadrant(handler, vals, valid))
    occ_vals, occ_valid = pool_quadrant_occupancy(handler, cells, field_only=(map_type == 'field'))
    occ_vals = np.where(occ_valid, occ_vals, 0.0)
    smoothed = _kde_smooth_quadrant(handler, occ_vals, occ_valid)
    return _normalize_pct(smoothed)


def _quadrant_data_mask(handler, cells: list, map_type: str) -> np.ndarray:
    """Folded bins that genuinely carry data for this map type: the intersection of the
    observed map's and the occupancy null's fold validity (_fold_mean_map's weight mask).

    The KDE pipeline itself cannot be asked this afterwards -- it zero-fills empty bins
    (_build_quadrant_observed_kde's np.where(valid, ., 0.0), then _gaussian_smooth_2d's
    smoothed[~valid_mask] = 0.0), so an unsampled bin comes out as a finite 0.0 that is
    indistinguishable from a real zero. Plots need this mask to leave such bins blank
    instead of painting them at the colormap's vmin -- for the open field they are mostly
    the corner bins outside the circular arena (geom_valid), plus any never-visited bin."""
    if map_type == 'peak':
        pct, _ = pool_quadrant_peak_proportion(handler, cells)
        obs_valid = np.isfinite(pct)
    else:
        _, obs_valid = _QUADRANT_MAP_BUILDERS[map_type](handler, cells)
    _, null_valid = pool_quadrant_occupancy(handler, cells, field_only=(map_type == 'field'))
    # the mask is shared by both nulls so the two significance calls are made over the same
    # bins: a bin the uniform-field null gives zero coverage cannot be tested against it
    _, uf_valid = _uniform_null_quadrant_map(handler, cells, map_type)
    return obs_valid & null_valid & uf_valid


def analyze_quadrant_kde(handler, results: list, map_type: str,
                          n_boot: int = N_KDE_BOOTSTRAP,
                          rng: np.random.Generator | None = None) -> dict:
    """Quadrant-fold KDE analysis for one map type (see module comment above): bootstraps a
    per-bin CI for the KDE-smoothed folded map and flags folded bins where it significantly
    exceeds the KDE-smoothed occupancy null."""
    place_cells = results   # every unit is already a qualified place cell (see module docstring)
    qx, qy = handler.quad_shape
    n_quad = handler.n_quad_bins

    nan_q, no_q = np.full(n_quad, np.nan), np.zeros(n_quad, dtype=bool)
    out = dict(map_type=map_type, quad_shape=(qx, qy), n_place_cells=len(place_cells),
               real_med=nan_q.copy(), real_lo=nan_q.copy(),
               real_hi=nan_q.copy(), null_kde=nan_q.copy(),
               null_kde_uniform=nan_q.copy(), null_kde_occ=nan_q.copy(),
               sig_mask=no_q.copy(), sig_mask_uniform=no_q.copy(), sig_mask_occ=no_q.copy(),
               depl_mask_uniform=no_q.copy(), depl_mask_occ=no_q.copy(),
               quad_valid=no_q.copy(), clusters=[], clusters_uniform=[], clusters_occ=[])
    if not place_cells:
        return out

    # both nulls are always built (for export_null_comparison); NULL_MODE picks the primary
    out['null_kde_uniform'] = _build_quadrant_null_kde(handler, place_cells, map_type, null_mode='uniform_field')
    out['null_kde_occ']     = _build_quadrant_null_kde(handler, place_cells, map_type, null_mode='occupancy')
    out['null_kde']   = out['null_kde_uniform'] if NULL_MODE == 'uniform_field' else out['null_kde_occ']
    out['quad_valid'] = _quadrant_data_mask(handler, place_cells, map_type)

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

    # Restrict to genuinely sampled bins (quad_valid): real_lo/null_kde are both built by
    # zero-filling unsampled bins before Gaussian smoothing (_build_quadrant_observed_kde /
    # _build_quadrant_null_kde), so real activity bleeds into never-visited bins -- for the
    # open field these are the folded quadrant's corner bins that lie outside the circular
    # arena (see _quadrant_data_mask). The observed and null curves don't bleed in the same
    # proportion, so real_lo > null_kde can spuriously hold there; without this mask those
    # bins get flagged and outlined even though they are physically outside the arena.
    for tag in ('uniform', 'occ'):
        null = out[f'null_kde_{tag}']
        sig = (out['real_lo'] > null) & out['quad_valid']
        out[f'sig_mask_{tag}']   = sig
        out[f'depl_mask_{tag}']  = (out['real_hi'] < null) & out['quad_valid']
        out[f'clusters_{tag}'] = [
            dict(n_bins=len(region),
                 peak_local_idx=int(region[np.argmax(out['real_med'][region])]),
                 peak_pct=round(float(np.max(out['real_med'][region])), 4))
            for region in _connected_components_2d_flat(sig, qx, qy, wrap_x=False)
        ]
    primary = 'uniform' if NULL_MODE == 'uniform_field' else 'occ'
    out['sig_mask'] = out[f'sig_mask_{primary}']
    out['clusters'] = out[f'clusters_{primary}']
    return out


def plot_quadrant_kde(arena_handlers: dict, results: dict, save_path: str):
    """Plots one clean 3x3 grid (map type x arena) of the KDE-smoothed observed
    quadrant-fold map, each with a black contour marking folded bins where the bootstrap
    CI significantly exceeds the occupancy-null KDE (analyze_quadrant_kde) -- instead of a
    full second row of null-map panels per map type, a smaller strip directly beneath each
    row shows the difference (observed KDE minus null KDE) on a diverging colormap, giving
    an at-a-glance summary of where/how much the observed map departs from the null without
    doubling the figure.

    Colour scaling follows QUADRANT_COLOUR_SCALE -- see that constant for the trade-off."""
    shared = (QUADRANT_COLOUR_SCALE == 'shared')
    n_map = len(_MAP_ORDER)
    if shared:
        fig = plt.figure(figsize=(13, 4.4 * n_map))
        gs = fig.add_gridspec(2 * n_map, 3, height_ratios=[3, 1] * n_map, hspace=0.6, wspace=0.15)
    else:
        # per-panel colorbars steal width from the axes they attach to, so the figure is wider
        # and the diff row taller than in 'shared' mode -- otherwise the short 'observed -
        # null' panels come out too small to read
        fig = plt.figure(figsize=(15, 4.8 * n_map))
        gs = fig.add_gridspec(2 * n_map, 3, height_ratios=[3, 1.4] * n_map, hspace=0.55, wspace=0.28)

    for mi, map_type in enumerate(_MAP_ORDER):
        main_row, diff_row = 2 * mi, 2 * mi + 1

        # Scales are built from the genuinely sampled bins only (quad_valid) -- the KDE
        # zero-fills unsampled bins, which would otherwise pin vmin at 0 and compress the
        # range (see _quadrant_data_mask).
        diffs = {k: results[k][map_type]['real_med'] - results[k][map_type]['null_kde']
                 for k in _ARENA_ORDER}
        if shared:
            row_main = make_cmap_norm(np.concatenate(
                [results[k][map_type]['real_med'][results[k][map_type]['quad_valid']]
                 for k in _ARENA_ORDER]))
            row_diff = _diverging_cmap_norm(np.concatenate(
                [diffs[k][results[k][map_type]['quad_valid']] for k in _ARENA_ORDER]))

        main_axes, diff_axes = [], []
        for ci, arena_key in enumerate(_ARENA_ORDER):
            handler = arena_handlers[arena_key]
            res = results[arena_key][map_type]
            proj = 'polar' if arena_key == 'circular_track' else None
            qv = res['quad_valid']
            diff = diffs[arena_key]

            if shared:
                cmap_main, norm_main = row_main
                cmap_diff, norm_diff = row_diff
            else:
                cmap_main, norm_main = make_cmap_norm(res['real_med'][qv])
                cmap_diff, norm_diff = _diverging_cmap_norm(diff[qv])

            ax_main = fig.add_subplot(gs[main_row, ci], projection=proj)
            # quad_valid, not isfinite(real_med): the KDE zero-fills unsampled bins, so
            # isfinite() is True everywhere and those bins would render at vmin instead of
            # being left blank
            main_valid = qv & np.isfinite(res['real_med'])
            im_main = handler.plot_quadrant_map(ax_main, np.nan_to_num(res['real_med']),
                                                main_valid, cmap_main, norm_main)
            handler.overlay_quadrant_significance(ax_main, res['sig_mask'])
            n_sig = len(res['clusters'])
            ax_main.set_title(f"{_ARENA_TITLES[arena_key]} -- {_MAP_TITLES[map_type]}\n"
                               f"n={res['n_place_cells']} cells, "
                               f"{n_sig} sig. cluster{'s' if n_sig != 1 else ''}", fontsize=8)
            main_axes.append(ax_main)

            ax_diff = fig.add_subplot(gs[diff_row, ci], projection=proj)
            diff_valid = qv & np.isfinite(res['real_med']) & np.isfinite(res['null_kde'])
            im_diff = handler.plot_quadrant_map(ax_diff, np.nan_to_num(diff), diff_valid,
                                                cmap_diff, norm_diff)
            handler.overlay_quadrant_significance(ax_diff, res['sig_mask'])
            ax_diff.set_title('observed - null', fontsize=7)
            diff_axes.append(ax_diff)

            if not shared:
                fig.colorbar(im_main, ax=ax_main, fraction=0.07, aspect=26, pad=0.04,
                             label='% of curve total')
                fig.colorbar(im_diff, ax=ax_diff, fraction=0.07, aspect=14, pad=0.04,
                             label='obs - null (pct pts)')

        if shared:
            # one colorbar per row, spanning the three arena panels that share its scale
            fig.colorbar(_mappable(*row_main), ax=main_axes, shrink=0.7, pad=0.02,
                         label='% of curve total')
            fig.colorbar(_mappable(*row_diff), ax=diff_axes, shrink=0.6, pad=0.02,
                         label='obs - null (pct pts)')

    scale_note = ('Colour scales are SHARED across arenas within each row -- comparable between '
                  'arenas, but every panel\'s colours\ndepend on all three arenas\' data.'
                  if shared else
                  'Colour scales are PER PANEL -- read each against its own colorbar, not across arenas.')
    null_name = 'uniform-field null' if NULL_MODE == 'uniform_field' else 'occupancy-null'
    fig.suptitle('Quadrant-fold KDE: Gaussian-smoothed folded rate maps (black contour = folded bin where the\n'
                 f'bootstrap CI significantly exceeds the {null_name} KDE); smaller row below each = observed minus null.\n'
                 + scale_note)
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
# Fine-map (whole, unfolded arena) KDE analysis
#
# Same design as the quadrant-fold KDE analysis above (analyze_quadrant_kde), applied
# directly to the WHOLE arena grid (handler.n_bins) instead of its Fig 1B/D quadrant fold --
# i.e. the same significance test run on the Fig S1H overall mean map, the field-only mean
# map, and the (unfolded) peak-location proportion map:
#   'overall' : pool_fine_map          (overall mean field-index map, Fig S1H)
#   'field'   : pool_field_only_map    (place-field-only mean field-index map)
#   'peak'    : pool_peak_proportion_fine (% of place-cell peaks per bin, unfolded)
# is Gaussian-kernel-smoothed (spatial KDE) on the full grid, bootstrapped (resampling place
# cells with replacement) to get a per-bin CI, and compared against a matching KDE of the
# pooled occupancy null (_pooled_occupancy) -- field-restricted to pair with 'field',
# unrestricted for 'overall'/'peak'. A bin is flagged significant where the observed map's CI
# lower bound exceeds the null KDE; contiguous flagged bins are reported as clusters via each
# handler's own connected_components (which already knows the circular track's arc-length
# wrap-around).
# ============================================================================

_FINE_MAP_BUILDERS = {
    'overall': pool_fine_map,
    'field':   pool_field_only_map,
    'peak':    None,  # handled separately below (pool_peak_proportion_fine returns (pct, total), not (vals, valid))
}


def _kde_smooth_fine(handler, values_flat: np.ndarray, valid_flat: np.ndarray) -> np.ndarray:
    """Spatial (2D) KDE over the WHOLE (unfolded) arena grid -- fine-map analogue of
    _kde_smooth_quadrant. Reuses each handler's own smooth() rather than re-deriving the
    reshape/wrap logic here, since it already applies the right topology per arena (e.g.
    wrap_x=True for the circular track's arc-length axis)."""
    return handler.smooth(values_flat, valid_flat)


def _build_fine_observed_kde(handler, cells: list, map_type: str) -> np.ndarray:
    """Fine-map analogue of _build_quadrant_observed_kde: builds one whole-arena pooled map
    (pool_fine_map / pool_field_only_map / pool_peak_proportion_fine), KDE-smooths it over
    the full grid, and rescales it to % of its own total -- run once on the real place-cell
    population and once per bootstrap resample."""
    if map_type == 'peak':
        pct, _ = pool_peak_proportion_fine(handler, cells)
        vals, valid = pct, np.isfinite(pct)
    else:
        vals, valid = _FINE_MAP_BUILDERS[map_type](handler, cells)
    vals = np.where(valid, vals, 0.0)
    smoothed = _kde_smooth_fine(handler, vals, valid)
    return _normalize_pct(smoothed)


def _build_fine_null_kde(handler, cells: list, map_type: str,
                          null_mode: str | None = None) -> np.ndarray:
    """Null counterpart of _build_fine_observed_kde, KDE-smoothed over the same whole-arena
    grid and rescaled to % of its own total. null_mode 'occupancy' smooths the pooled
    dwell-time map (field-restricted for 'field', unrestricted for 'overall'/'peak');
    'uniform_field' smooths the expected map of build_uniform_field_null instead."""
    if (NULL_MODE if null_mode is None else null_mode) == 'uniform_field':
        vals, valid = _uniform_null_fine_map(handler, cells, map_type)
        vals = np.where(valid, vals, 0.0)
        return _normalize_pct(_kde_smooth_fine(handler, vals, valid))
    occ_vals, occ_valid = _pooled_occupancy(handler, cells, field_only=(map_type == 'field'))
    occ_vals = np.where(occ_valid, occ_vals, 0.0)
    smoothed = _kde_smooth_fine(handler, occ_vals, occ_valid)
    return _normalize_pct(smoothed)


def _fine_data_mask(handler, cells: list, map_type: str) -> np.ndarray:
    """Whole-arena analogue of _quadrant_data_mask: bins that genuinely carry data for this
    map type, i.e. the intersection of the observed map's and the occupancy null's own
    validity. The KDE pipeline zero-fills never-visited bins before smoothing (e.g. the open
    field's out-of-circle corner bins of its bounding nx*ny grid), so without this mask such a
    bin would come back as a finite 0.0 indistinguishable from a real zero, and could get
    flagged (or plotted) as if it were real data."""
    if map_type == 'peak':
        pct, _ = pool_peak_proportion_fine(handler, cells)
        obs_valid = np.isfinite(pct)
    else:
        _, obs_valid = _FINE_MAP_BUILDERS[map_type](handler, cells)
    _, null_valid = _pooled_occupancy(handler, cells, field_only=(map_type == 'field'))
    # shared by both nulls so their significance calls cover the same bins (see
    # _quadrant_data_mask)
    _, uf_valid = _uniform_null_fine_map(handler, cells, map_type)
    return obs_valid & null_valid & uf_valid


def analyze_fine_kde(handler, results: list, map_type: str,
                      n_boot: int = N_KDE_BOOTSTRAP,
                      rng: np.random.Generator | None = None) -> dict:
    """Whole-arena analogue of analyze_quadrant_kde: bootstraps a per-bin CI for the
    KDE-smoothed overall/field/peak rate map (instead of its quadrant fold) and flags bins
    where it significantly exceeds the KDE-smoothed occupancy null."""
    place_cells = results   # every unit is already a qualified place cell (see module docstring)
    n_fine = handler.n_bins

    nan_f, no_f = np.full(n_fine, np.nan), np.zeros(n_fine, dtype=bool)
    out = dict(map_type=map_type, n_place_cells=len(place_cells),
               real_med=nan_f.copy(), real_lo=nan_f.copy(),
               real_hi=nan_f.copy(), null_kde=nan_f.copy(),
               null_kde_uniform=nan_f.copy(), null_kde_occ=nan_f.copy(),
               sig_mask=no_f.copy(), sig_mask_uniform=no_f.copy(), sig_mask_occ=no_f.copy(),
               depl_mask_uniform=no_f.copy(), depl_mask_occ=no_f.copy(),
               fine_valid=no_f.copy(), clusters=[], clusters_uniform=[], clusters_occ=[])
    if not place_cells:
        return out

    # both nulls are always built (for export_null_comparison); NULL_MODE picks the primary
    out['null_kde_uniform'] = _build_fine_null_kde(handler, place_cells, map_type, null_mode='uniform_field')
    out['null_kde_occ']     = _build_fine_null_kde(handler, place_cells, map_type, null_mode='occupancy')
    out['null_kde']   = out['null_kde_uniform'] if NULL_MODE == 'uniform_field' else out['null_kde_occ']
    out['fine_valid'] = _fine_data_mask(handler, place_cells, map_type)

    rng = rng if rng is not None else np.random.default_rng(0)
    n = len(place_cells)
    curves = np.zeros((n_boot, n_fine))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_cells = [place_cells[i] for i in idx]
        curves[b] = _build_fine_observed_kde(handler, boot_cells, map_type)
    out['real_lo']  = np.percentile(curves, 100 * KDE_ALPHA / 2.0, axis=0)
    out['real_hi']  = np.percentile(curves, 100 * (1.0 - KDE_ALPHA / 2.0), axis=0)
    out['real_med'] = np.percentile(curves, 50, axis=0)

    # Restrict to genuinely sampled bins (fine_valid), same reasoning as
    # analyze_quadrant_kde's sig mask -- see _fine_data_mask.
    for tag in ('uniform', 'occ'):
        null = out[f'null_kde_{tag}']
        sig = (out['real_lo'] > null) & out['fine_valid']
        out[f'sig_mask_{tag}']  = sig
        out[f'depl_mask_{tag}'] = (out['real_hi'] < null) & out['fine_valid']
        out[f'clusters_{tag}'] = [
            dict(n_bins=len(region),
                 peak_local_idx=int(region[np.argmax(out['real_med'][region])]),
                 peak_pct=round(float(np.max(out['real_med'][region])), 4))
            for region in handler.connected_components(sig)
        ]
    primary = 'uniform' if NULL_MODE == 'uniform_field' else 'occ'
    out['sig_mask'] = out[f'sig_mask_{primary}']
    out['clusters'] = out[f'clusters_{primary}']
    return out


def plot_fine_kde(arena_handlers: dict, results: dict, save_path: str):
    """Fine-map analogue of plot_quadrant_kde: same KDE-smoothed-map-plus-significance-
    contour design (analyze_fine_kde), drawn over the whole arena (handler.plot_fine /
    overlay_fine_significance) instead of the Fig 1B/D quadrant fold.

    Colour scaling follows QUADRANT_COLOUR_SCALE -- see that constant for the trade-off."""
    shared = (QUADRANT_COLOUR_SCALE == 'shared')
    n_map = len(_MAP_ORDER)
    if shared:
        fig = plt.figure(figsize=(13, 4.4 * n_map))
        gs = fig.add_gridspec(2 * n_map, 3, height_ratios=[3, 1] * n_map, hspace=0.6, wspace=0.15)
    else:
        fig = plt.figure(figsize=(15, 4.8 * n_map))
        gs = fig.add_gridspec(2 * n_map, 3, height_ratios=[3, 1.4] * n_map, hspace=0.55, wspace=0.28)

    for mi, map_type in enumerate(_MAP_ORDER):
        main_row, diff_row = 2 * mi, 2 * mi + 1

        diffs = {k: results[k][map_type]['real_med'] - results[k][map_type]['null_kde']
                 for k in _ARENA_ORDER}
        if shared:
            row_main = make_cmap_norm(np.concatenate(
                [results[k][map_type]['real_med'][results[k][map_type]['fine_valid']]
                 for k in _ARENA_ORDER]))
            row_diff = _diverging_cmap_norm(np.concatenate(
                [diffs[k][results[k][map_type]['fine_valid']] for k in _ARENA_ORDER]))

        main_axes, diff_axes = [], []
        for ci, arena_key in enumerate(_ARENA_ORDER):
            handler = arena_handlers[arena_key]
            res = results[arena_key][map_type]
            proj = 'polar' if arena_key == 'circular_track' else None
            fv = res['fine_valid']
            diff = diffs[arena_key]

            if shared:
                cmap_main, norm_main = row_main
                cmap_diff, norm_diff = row_diff
            else:
                cmap_main, norm_main = make_cmap_norm(res['real_med'][fv])
                cmap_diff, norm_diff = _diverging_cmap_norm(diff[fv])

            ax_main = fig.add_subplot(gs[main_row, ci], projection=proj)
            main_valid = fv & np.isfinite(res['real_med'])
            im_main = handler.plot_fine(ax_main, np.nan_to_num(res['real_med']),
                                        main_valid, cmap_main, norm_main)
            handler.overlay_fine_significance(ax_main, res['sig_mask'])
            n_sig = len(res['clusters'])
            ax_main.set_title(f"{_ARENA_TITLES[arena_key]} -- {_MAP_TITLES[map_type]}\n"
                               f"n={res['n_place_cells']} cells, "
                               f"{n_sig} sig. cluster{'s' if n_sig != 1 else ''}", fontsize=8)
            main_axes.append(ax_main)

            ax_diff = fig.add_subplot(gs[diff_row, ci], projection=proj)
            diff_valid = fv & np.isfinite(res['real_med']) & np.isfinite(res['null_kde'])
            im_diff = handler.plot_fine(ax_diff, np.nan_to_num(diff), diff_valid,
                                        cmap_diff, norm_diff)
            handler.overlay_fine_significance(ax_diff, res['sig_mask'])
            ax_diff.set_title('observed - null', fontsize=7)
            diff_axes.append(ax_diff)

            if not shared:
                fig.colorbar(im_main, ax=ax_main, fraction=0.07, aspect=26, pad=0.04,
                             label='% of map total')
                fig.colorbar(im_diff, ax=ax_diff, fraction=0.07, aspect=14, pad=0.04,
                             label='obs - null (pct pts)')

        if shared:
            fig.colorbar(_mappable(*row_main), ax=main_axes, shrink=0.7, pad=0.02,
                         label='% of map total')
            fig.colorbar(_mappable(*row_diff), ax=diff_axes, shrink=0.6, pad=0.02,
                         label='obs - null (pct pts)')

    scale_note = ('Colour scales are SHARED across arenas within each row -- comparable between '
                  'arenas, but every panel\'s colours\ndepend on all three arenas\' data.'
                  if shared else
                  'Colour scales are PER PANEL -- read each against its own colorbar, not across arenas.')
    null_name = 'uniform-field null' if NULL_MODE == 'uniform_field' else 'occupancy-null'
    fig.suptitle('Fine-map KDE: Gaussian-smoothed WHOLE-ARENA rate maps (black contour = bin where the\n'
                 f'bootstrap CI significantly exceeds the {null_name} KDE); smaller row below each = observed minus null.\n'
                 + scale_note)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def export_fine_kde_summary(arena_handlers: dict, results: dict, out_path: str):
    rows = []
    for arena_key, by_map in results.items():
        ny = arena_handlers[arena_key].ny
        for map_type, res in by_map.items():
            if not res['clusters']:
                rows.append(dict(arena=arena_key, map_type=map_type,
                                  n_place_cells=res['n_place_cells'],
                                  cluster_rank=None, n_bins=None,
                                  peak_row=None, peak_col=None, peak_pct=None))
                continue
            for i, cl in enumerate(res['clusters'], start=1):
                peak_row, peak_col = divmod(cl['peak_local_idx'], ny)
                rows.append(dict(arena=arena_key, map_type=map_type,
                                  n_place_cells=res['n_place_cells'],
                                  cluster_rank=i, n_bins=cl['n_bins'],
                                  peak_row=peak_row, peak_col=peak_col,
                                  peak_pct=cl['peak_pct']))
    df = pd.DataFrame(rows)
    df.to_excel(out_path, index=False)
    print(f'[SAVED] {out_path}')


def run_fine_kde_analysis(arena_handlers: dict, arena_results: dict, out_dir: str,
                           n_boot: int = N_KDE_BOOTSTRAP) -> dict:
    results = {}
    for arena_key in _ARENA_ORDER:
        handler, res = arena_handlers[arena_key], arena_results[arena_key]
        results[arena_key] = {
            map_type: analyze_fine_kde(handler, res, map_type, n_boot=n_boot)
            for map_type in _MAP_ORDER
        }

    plot_fine_kde(arena_handlers, results, os.path.join(out_dir, 'FineMap_KDE.png'))
    export_fine_kde_summary(arena_handlers, results, os.path.join(out_dir, 'FineMap_KDE_Clusters.xlsx'))

    for arena_key in _ARENA_ORDER:
        for map_type in _MAP_ORDER:
            res = results[arena_key][map_type]
            n_sig = len(res['clusters'])
            print(f'[{arena_key}/{map_type} fine-map KDE] n={res["n_place_cells"]} place cells, '
                  f'{n_sig} significant cluster(s) vs. occupancy null')
    return results


# ============================================================================
# Rayleigh vector analysis (open field only) -- DESCRIPTIVE ONLY
#
# NOTE: the output of this section (Rayleigh_OpenField*.png / .xlsx) is DESCRIPTIVE: it reports the
# resultant length R and mean direction of the angularly-binned pooled map but NO p-value. The
# old weighted-Rayleigh p (z = n R^2 with n = number of spatial bins, or of significant bins) was
# removed because it is not valid: the weights are rates rather than counts, n is arbitrary, the
# smoothed bins are not independent, and the pooled map is a single sample. The inferential
# tests are in "Directional-preference tests across cells" (run_directional_vector_analysis).
#
# Summarises whether an OPEN-FIELD pooled mean rate map ('overall', 'peak', 'field' -- the same
# three map types as the KDE analyses above, taken on the whole unfolded grid) has a
# directional bias about the arena centre:
#   1. Centre: the middle of the map's own circumference -- the midpoint of the extent of
#      the ring of bins that sit on the circle's outline (handler.geom_valid), NOT the
#      centroid of the bins that happen to hold data. The centre therefore does not depend
#      on the centre bin (or any other bin) being valid; with an even nx it falls on a
#      corner shared by 4 bins, so no bin sits at the centre and every bin has a defined angle.
#   2. Every valid bin's centre is given an angle about that centre (deg CCW from +x, the
#      same orientation handler.plot_fine draws) and is assigned to exactly ONE angular bin
#      of RAYLEIGH_BIN_DEG (half-open [lo, hi), so a bin on a boundary is never counted
#      twice, and none are dropped).
#   3. Each angular bin's magnitude is the sum (or mean, RAYLEIGH_MAGNITUDE) of its bins'
#      map values, each first multiplied by a radial weight that grows with the bin's
#      distance from the centre (RAYLEIGH_RADIAL_POWER), so wall-side bins count for more.
#   4. Weighted resultant (R, mean direction) of the angular bins' magnitudes (angle = angular
#      bin centre). Descriptive only -- no significance test.
# ============================================================================

RAYLEIGH_BIN_DEG   = 30.0
RAYLEIGH_MAGNITUDE = 'sum'    # 'sum' or 'mean' of the map values of the bins in each angular bin
RAYLEIGH_ALPHA     = 0.05
RAYLEIGH_ARENA_KEY = 'open_field'
# Each bin's value is weighted by (distance from centre / circle radius) ** RAYLEIGH_RADIAL_POWER
# before entering its angular bin, so wall-side bins pull the resultant vector harder than
# centre-side bins. 0 = no weighting (every bin equal), 1 = weight grows linearly with distance
# (0 at the centre, 1 at the wall), 2 = quadratically (favours the wall more strongly).
RAYLEIGH_RADIAL_POWER = 1.0


def _circle_centre_from_circumference(handler) -> tuple:
    """(cx_bin, cy_bin) centre of the arena circle in bin-index units (bin i's centre is at i),
    from the circumference bins of handler.geom_valid: those inside the circle with at least one
    4-neighbour outside it (or off the grid). The centre is the midpoint of the circumference's
    extent along each axis, so it is set by the arena outline alone and is the same whether or
    not the bin(s) at the centre hold data. Returns also the circumference mask (nx, ny)."""
    inside = handler.geom_valid.reshape(handler.nx, handler.ny)
    padded = np.pad(inside, 1, constant_values=False)
    all_neighbours_inside = (padded[:-2, 1:-1] & padded[2:, 1:-1] &
                             padded[1:-1, :-2] & padded[1:-1, 2:])
    circumference = inside & ~all_neighbours_inside
    bx, by = np.nonzero(circumference)
    cx_bin = 0.5 * (bx.min() + bx.max())
    cy_bin = 0.5 * (by.min() + by.max())
    return float(cx_bin), float(cy_bin), circumference


def angular_bin_magnitudes(handler, values_flat: np.ndarray, valid_flat: np.ndarray,
                           bin_deg: float = RAYLEIGH_BIN_DEG,
                           magnitude: str = RAYLEIGH_MAGNITUDE,
                           radial_power: float = RAYLEIGH_RADIAL_POWER) -> dict:
    """Assigns every valid bin of a whole-arena map to one angular bin about the circle's centre
    and returns the per-angular-bin magnitude (see the section comment above). Each bin's value
    is multiplied by (r / circle radius) ** radial_power before it is summed/averaged, so bins
    nearer the wall count for more than bins nearer the centre; `magnitude_unweighted` is the
    same aggregation without that weight."""
    n_ang = int(round(360.0 / bin_deg))
    cx_bin, cy_bin, circumference = _circle_centre_from_circumference(handler)

    bx, by = np.meshgrid(np.arange(handler.nx), np.arange(handler.ny), indexing='ij')
    dx, dy = (bx - cx_bin).ravel(), (by - cy_bin).ravel()
    use = valid_flat & handler.geom_valid & np.isfinite(values_flat) & (np.hypot(dx, dy) > 1e-9)

    ang_deg = np.degrees(np.arctan2(dy, dx)) % 360.0
    # rounding first keeps a bin sitting exactly on a boundary from flipping sides on
    # floating-point noise; the modulo folds 360.0 back into the first angular bin
    ang_bin = np.floor(np.round(ang_deg, 9) / bin_deg).astype(int) % n_ang

    # radial weight: 0 at the centre, 1 on the circumference (r / circle radius, both measured from
    # the circumference-derived centre), raised to radial_power. power 0 = no weighting.
    r = np.hypot(dx, dy)
    cbx, cby = np.nonzero(circumference)
    r_circle = float(np.hypot(cbx - cx_bin, cby - cy_bin).max())
    weight = np.minimum(r / r_circle, 1.0) ** radial_power

    def _per_angular_bin(vals):
        sums = np.zeros(n_ang)
        np.add.at(sums, ang_bin[use], vals[use])
        if magnitude == 'mean':
            return np.divide(sums, counts, out=np.zeros(n_ang), where=counts > 0)
        return sums

    counts = np.zeros(n_ang, dtype=int)
    np.add.at(counts, ang_bin[use], 1)
    assert counts.sum() == int(use.sum())      # each spatial bin lands in exactly one angular bin
    mags_unweighted = _per_angular_bin(values_flat)
    mags = _per_angular_bin(values_flat * weight)
    return dict(magnitude=mags, magnitude_unweighted=mags_unweighted, n_spatial_bins=counts,
                bin_centres_deg=(np.arange(n_ang) + 0.5) * bin_deg,
                centre_bin=(cx_bin, cy_bin), circumference=circumference, n_used=int(use.sum()))


def weighted_resultant(angles_deg: np.ndarray, weights: np.ndarray, bin_deg: float = 0.0) -> dict:
    """DESCRIPTIVE weighted resultant of circular data: `weights` are the magnitudes at
    `angles_deg`. bin_deg > 0 applies the standard correction for grouped angles (Zar) to R.
    Deliberately returns no p-value -- see the section comment above."""
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = w.sum()
    if total <= 0:
        return dict(R=np.nan, mean_dir_deg=np.nan)
    th = np.radians(angles_deg)
    C, S = (w * np.cos(th)).sum(), (w * np.sin(th)).sum()
    R = np.hypot(C, S) / total
    if bin_deg > 0:
        d = np.radians(bin_deg)
        R = min(1.0, R * (d / 2.0) / np.sin(d / 2.0))
    return dict(R=float(R), mean_dir_deg=float(np.degrees(np.arctan2(S, C)) % 360.0))


def _rayleigh_on_map(handler, vals: np.ndarray, valid: np.ndarray, n_cells: int,
                     map_type: str) -> dict:
    """Shared core: angular-bin magnitudes + descriptive resultant of one whole-arena map."""
    if n_cells == 0 or not valid.any():
        return dict(map_type=map_type, n_place_cells=n_cells, ang=None, stats=None,
                    map=vals, map_valid=valid)
    ang = angular_bin_magnitudes(handler, np.where(valid, vals, 0.0), valid)
    stats = weighted_resultant(ang['bin_centres_deg'], ang['magnitude'], bin_deg=RAYLEIGH_BIN_DEG)
    return dict(map_type=map_type, n_place_cells=n_cells, ang=ang, stats=stats,
                map=vals, map_valid=valid)


def analyze_rayleigh(handler, results: list, map_type: str) -> dict:
    """Descriptive resultant of one pooled whole-arena map ('overall'/'peak'/'field')."""
    if map_type == 'peak':
        vals, _ = pool_peak_proportion_fine(handler, results)
        valid = np.isfinite(vals)
    else:
        vals, valid = _FINE_MAP_BUILDERS[map_type](handler, results)
    return _rayleigh_on_map(handler, vals, valid, len(results), map_type)


def analyze_rayleigh_kde(handler, results: list, map_type: str, kde_res: dict,
                         sig_only: bool = False) -> dict:
    """Descriptive resultant of the fine-map KDE map drawn by plot_fine_kde: the bootstrap
    median KDE-smoothed map (kde_res['real_med'] from analyze_fine_kde), restricted to the bins
    that KDE analysis treats as genuinely sampled (kde_res['fine_valid']). Each bin enters with
    its firing-rate value times the radial (wall-proximity) weight of angular_bin_magnitudes.

    sig_only=True further restricts it to the bins whose bootstrap CI significantly exceeds
    the null (kde_res['sig_mask'], the NULL_MODE null -- the bins outlined in black on the KDE
    figure)."""
    vals = kde_res['real_med']
    valid = kde_res['fine_valid'] & np.isfinite(vals)
    if sig_only:
        valid &= kde_res['sig_mask']
    return _rayleigh_on_map(handler, vals, valid, len(results), map_type)


def plot_rayleigh(handler, results: dict, save_path: str, kind_label: str = 'pooled map'):
    """Per map type: the angular-bin magnitudes as a polar histogram with the resultant vector
    (length = R, scaled to the histogram's radius), plus the map itself with the circumference
    centre marked."""
    fig = plt.figure(figsize=(5 * len(_MAP_ORDER), 9))
    for mi, map_type in enumerate(_MAP_ORDER):
        res = results[map_type]
        ax_map = fig.add_subplot(2, len(_MAP_ORDER), mi + 1)
        ax_pol = fig.add_subplot(2, len(_MAP_ORDER), len(_MAP_ORDER) + mi + 1, projection='polar')
        if res['ang'] is None:
            ax_map.axis('off'); ax_pol.axis('off')
            ax_map.set_title(f'{_MAP_TITLES[map_type]}\n(no data / no significant bins)')
            continue

        vals, valid = res['map'], res['map_valid']
        cmap, norm = make_cmap_norm(vals[valid])
        handler.plot_fine(ax_map, vals, valid, cmap, norm)
        cxb, cyb = res['ang']['centre_bin']
        ax_map.plot((cxb + 0.5) * handler.bin_cm, (cyb + 0.5) * handler.bin_cm, 'k+', ms=12, mew=2)

        ang, st = res['ang'], res['stats']
        th = np.radians(ang['bin_centres_deg'])
        mag = ang['magnitude']
        ax_pol.bar(th, mag, width=np.radians(RAYLEIGH_BIN_DEG), color='0.6', edgecolor='0.3', lw=0.4)
        rmax = float(mag.max()) if mag.max() > 0 else 1.0
        ax_pol.annotate('', xy=(np.radians(st['mean_dir_deg']), st['R'] * rmax), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='k', lw=2))
        ax_pol.set_title(f"R={st['R']:.3f}, mean dir={st['mean_dir_deg']:.0f} deg\n"
                         f"(descriptive -- see DirectionalVector_* for tests)", fontsize=9)
        ax_map.set_title(f"{_ARENA_TITLES[RAYLEIGH_ARENA_KEY]} -- {_MAP_TITLES[map_type]}\n"
                         f"(n={res['n_place_cells']} place cells)", fontsize=9)

    fig.suptitle(f'Descriptive resultant vector, {kind_label} ({RAYLEIGH_BIN_DEG:g} deg bins, '
                 f'{RAYLEIGH_MAGNITUDE} magnitude, radial weight r^{RAYLEIGH_RADIAL_POWER:g})')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def export_rayleigh_summary(results_by_kind: dict, out_path: str):
    """results_by_kind: {'pooled': {map_type: analyze_rayleigh(...)}, 'kde': {...}} -- one row per
    (kind, map_type) on the 'Rayleigh' sheet, one row per angular bin on 'AngularBins'."""
    summary_rows, bin_rows = [], []
    for kind, map_type in [(k, mt) for k in results_by_kind for mt in _MAP_ORDER]:
        res = results_by_kind[kind][map_type]
        if res['ang'] is None:
            summary_rows.append(dict(arena=RAYLEIGH_ARENA_KEY, map_kind=kind, map_type=map_type,
                                     n_place_cells=res['n_place_cells']))
            continue
        ang, st = res['ang'], res['stats']
        summary_rows.append(dict(arena=RAYLEIGH_ARENA_KEY, map_kind=kind, map_type=map_type,
                                 n_place_cells=res['n_place_cells'],
                                 centre_bin_x=ang['centre_bin'][0], centre_bin_y=ang['centre_bin'][1],
                                 bin_deg=RAYLEIGH_BIN_DEG, magnitude=RAYLEIGH_MAGNITUDE,
                                 radial_power=RAYLEIGH_RADIAL_POWER,
                                 n_spatial_bins=ang['n_used'], R=st['R'],
                                 mean_dir_deg=st['mean_dir_deg']))
        for k, (c, m, mu, nb) in enumerate(zip(ang['bin_centres_deg'], ang['magnitude'],
                                               ang['magnitude_unweighted'], ang['n_spatial_bins'])):
            bin_rows.append(dict(map_kind=kind, map_type=map_type, angular_bin=k,
                                 bin_start_deg=k * RAYLEIGH_BIN_DEG,
                                 bin_centre_deg=c, n_spatial_bins=int(nb), magnitude=float(m),
                                 magnitude_unweighted=float(mu)))
    with pd.ExcelWriter(out_path) as xw:
        pd.DataFrame(summary_rows).to_excel(xw, sheet_name='Rayleigh', index=False)
        pd.DataFrame(bin_rows).to_excel(xw, sheet_name='AngularBins', index=False)
    print(f'[SAVED] {out_path}')


def run_rayleigh_analysis(arena_handlers: dict, arena_results: dict, out_dir: str,
                          fine_kde_results: dict | None = None) -> dict:
    """Descriptive resultant-vector summary (plots + workbook, no p-values) of every map type of
    the open field only: the pooled mean rate maps and, when fine_kde_results
    (run_fine_kde_analysis' output) is given, the fine-map KDE maps of plot_fine_kde as well.
    Inference lives in run_directional_vector_analysis."""
    handler, cells = arena_handlers[RAYLEIGH_ARENA_KEY], arena_results[RAYLEIGH_ARENA_KEY]
    results_by_kind = {'pooled': {mt: analyze_rayleigh(handler, cells, mt) for mt in _MAP_ORDER}}
    if fine_kde_results is not None:
        kde = fine_kde_results[RAYLEIGH_ARENA_KEY]
        results_by_kind['kde'] = {mt: analyze_rayleigh_kde(handler, cells, mt, kde[mt])
                                  for mt in _MAP_ORDER}
        results_by_kind['kde_sig'] = {mt: analyze_rayleigh_kde(handler, cells, mt, kde[mt], sig_only=True)
                                      for mt in _MAP_ORDER}

    plot_files = {'pooled':  ('Rayleigh_OpenField.png', 'pooled mean rate maps'),
                  'kde':     ('Rayleigh_OpenField_KDE.png', 'fine-map KDE (bootstrap median)'),
                  'kde_sig': ('Rayleigh_OpenField_KDE_SigBins.png',
                              'fine-map KDE, significant bins only (rate x wall-proximity weight)')}
    for kind, results in results_by_kind.items():
        plot_rayleigh(handler, results, os.path.join(out_dir, plot_files[kind][0]), plot_files[kind][1])
    export_rayleigh_summary(results_by_kind, os.path.join(out_dir, 'Rayleigh_OpenField.xlsx'))

    for kind, results in results_by_kind.items():
        for map_type in _MAP_ORDER:
            st = results[map_type]['stats']
            tag = f'{RAYLEIGH_ARENA_KEY}/{kind}/{map_type} resultant (descriptive)'
            if st is None:
                print(f'[{tag}] no data / no significant bins')
            else:
                print(f"[{tag}] R={st['R']:.4f}, mean dir={st['mean_dir_deg']:.1f} deg")
    return results_by_kind


# ============================================================================
# Directional-preference tests across cells (open field only)
#
# Replaces the invalid weighted-Rayleigh p-values above. The independent unit of the data is the
# cell (and, above it, the animal), NOT the spatial bin, so the inference is done on ONE VECTOR
# PER CELL:
#
#   v_i = sum_j f_ij * d_j / sum_j f_ij   -   mean_j(d_j)  [over the cell's occupancy-passing bins]
#
# where d_j = (bin-centre j - arena centre) / arena radius, f_ij is the cell's weight in bin j,
# and the sum runs over bins that pass occupancy (cell['valid'], i.e. >= min_occ_s and inside the
# arena). v_i is the rate-weighted centroid of the cell's firing relative to the arena centre:
# its ANGLE is the preferred direction and its LENGTH (0-1, in arena radii) carries both how far
# from the centre and how concentrated the firing is -- so rate AND distance are both preserved
# with no angular binning. The second term (VECTOR_SUBTRACT_COVERAGE) removes the centroid of
# the cell's own sampled bins, so a flat map gives v_i = 0 and an animal that skipped one side of
# the arena does not produce a spurious direction.
#
# Three kinds of per-cell weight f_ij are analysed, for each of the three map types:
#   'rate'    : the cell's own map -- overall: fi_map on valid bins; field: fi_map inside the
#               cell's place field; peak: a single 1 at the cell's peak bin.
#   'kde'     : that same map after the fine-map KDE step (handler.smooth, the per-cell analogue of
#               analyze_fine_kde's KDE smoothing).
#   'kde_sig' : the 'kde' map restricted to the bins the POOLED fine-map KDE found significantly
#               above the null (kde_res['sig_mask']). CAVEAT: those bins were selected using the
#               same cells being tested, so the test is optimistic (selection bias) and only
#               descriptive; it has no pooled-map permutation counterpart for the same reason.
#
# Tests (all rank/permutation based; the primary p is always the permutation p):
#   * Moore's modified Rayleigh test on ANIMAL-MEAN vectors (angle, length): each animal's cells
#     are averaged first, so n = number of animals and there is no pseudoreplication. p is by
#     Monte-Carlo under Moore's null (uniform random angles, length ranks fixed); the asymptotic
#     p = exp(-3 R*^2) is reported alongside.
#   * Moore's test on all CELL vectors with an animal-CLUSTERED permutation: the statistic is
#     computed on every cell, but the null rotates all cells of an animal together by one random
#     angle, so cells of one animal cannot each count as independent evidence.
#   * Moore's test on all cell vectors treating cells as independent -- REFERENCE ONLY
#     (pseudoreplicated: cells of one animal/session are not independent).
#   * Pooled-map permutation ('rate' and 'kde' kinds): every cell's map and its occupancy mask are
#     independently rotated/reflected by one of the 8 symmetries of the square grid (exact, no
#     interpolation, about the arena centre), the pooled map is rebuilt and its resultant length
#     recomputed. The observed pooled R is compared to that null: spatial autocorrelation and the
#     occupancy masks are preserved and no sample size enters. (Cells are treated as independent
#     here, so it is subject to the same pseudoreplication caveat as the naive Moore test.)
#
# Moore's test is scale-free (uses ranks of the lengths) and, like Rayleigh, only detects a
# UNIMODAL directional bias -- opposite or 4-fold symmetric preferences give R* ~ 0.
# ============================================================================

# regex applied to the session name (path relative to ROOT_DIRECTORY) to extract the animal ID,
# e.g. 'Open\Fa1059_Day10_3Rotate' -> 'Fa1059'; falls back to the first path component
ANIMAL_ID_REGEX = r'\b(Fa[0-9A-Za-z]+?)_'
VECTOR_SUBTRACT_COVERAGE = True   # subtract the centroid of the cell's own sampled bins (see above)
MOORE_N_PERM             = 10000  # Monte-Carlo draws for Moore's test
CLUSTER_N_PERM          = 10000  # animal-clustered rotation permutations
POOLED_N_PERM            = 2000   # pooled-map D4 permutations
VECTOR_SEED              = 0
_VECTOR_KINDS            = ('rate', 'kde', 'kde_sig')
_VECTOR_KIND_TITLES = {'rate': 'per-cell mean rate maps', 'kde': 'per-cell fine-KDE maps',
                       'kde_sig': 'per-cell fine-KDE maps, significant bins only'}


def _animal_id(session_name: str) -> str:
    m = re.search(ANIMAL_ID_REGEX, session_name)
    return m.group(1) if m else os.path.normpath(session_name).split(os.sep)[0]


def _bin_offsets(handler) -> tuple:
    """(dx, dy) per flat bin: bin-centre offset from the arena centre in units of the arena
    radius, so a vector's length is 0 at the centre and ~1 at the wall."""
    cx_bin, cy_bin, _ = _circle_centre_from_circumference(handler)
    bx, by = np.meshgrid(np.arange(handler.nx), np.arange(handler.ny), indexing='ij')
    radius_bins = handler.diameter / 2.0 / handler.bin_cm
    return (bx.ravel() - cx_bin) / radius_bins, (by.ravel() - cy_bin) / radius_bins


def _cell_weight_map(handler, cell: dict, map_type: str, kind: str, sig_mask=None):
    """(weights f_j, support mask) of one cell for a map type / kind, or None if the cell has no
    usable map (no field, no peak, nothing left inside the significant bins). Only occupancy-
    passing bins (cell['valid'] & handler.geom_valid) can carry weight."""
    valid = cell['valid'] & handler.geom_valid
    fi = np.where(valid & np.isfinite(cell['fi_map']), cell['fi_map'], 0.0)
    if map_type == 'overall':
        w, sup = fi, valid
    elif map_type == 'field':
        sup = valid & cell['field_mask']
        w = np.where(sup, fi, 0.0)
    else:  # 'peak'
        pb = cell['peak_bin']
        if pb is None or not valid[pb]:
            return None
        sup = valid
        w = np.zeros(handler.n_bins)
        w[pb] = 1.0
    if not sup.any():
        return None
    if kind in ('kde', 'kde_sig'):
        w = np.where(sup, handler.smooth(w, sup), 0.0)
    if kind == 'kde_sig':
        sup = sup & sig_mask
        w = np.where(sup, w, 0.0)
    return (w, sup) if w.sum() > 0 else None


def _weighted_offset(dxy: tuple, w: np.ndarray, ref_mask: np.ndarray):
    """Centroid of weights `w` minus (if VECTOR_SUBTRACT_COVERAGE) the centroid of `ref_mask`."""
    s = w.sum()
    if not s > 0 or not ref_mask.any():
        return None
    vx, vy = (w * dxy[0]).sum() / s, (w * dxy[1]).sum() / s
    if VECTOR_SUBTRACT_COVERAGE:
        vx -= dxy[0][ref_mask].mean()
        vy -= dxy[1][ref_mask].mean()
    return float(vx), float(vy)


def _moore_pvalue_mc(ranks: np.ndarray, r_obs: float, rng: np.random.Generator,
                     n_perm: int) -> float:
    """Monte-Carlo p of Moore's R* under H0: the angles are uniform on the circle (independent of
    the lengths), with the length ranks held fixed -- that is Moore's null, from which his tables
    come. It is NOT a shuffle of the ranks over the observed angles: that conditions on the
    observed angles, so it tests independence of length and angle, and cannot reject when the
    angles all point the same way. The observed arrangement is counted, so p >= 1/(n_perm+1)."""
    n = len(ranks)
    count, done = 0, 0
    while done < n_perm:
        chunk = min(2000, n_perm - done)
        th = rng.uniform(0.0, 2 * np.pi, size=(chunk, n))
        count += int((np.hypot(np.cos(th) @ ranks, np.sin(th) @ ranks) / n ** 1.5 >= r_obs - 1e-12).sum())
        done += chunk
    return (1 + count) / (n_perm + 1)


def moore_test(theta: np.ndarray, rho: np.ndarray, rng: np.random.Generator,
               n_perm: int = MOORE_N_PERM) -> dict:
    """Moore's (1980) modified Rayleigh test on n vectors (angle theta [rad], length rho).
    Lengths are replaced by their ranks (ties averaged); C = sum rank*cos(theta),
    S = sum rank*sin(theta), R* = sqrt(C^2 + S^2) / n^1.5. Under H0, 6 R*^2 is asymptotically
    chi-square(2), i.e. p ~ exp(-3 R*^2) (reported as p_asymptotic, not reliable for small n); the
    primary p (`p_perm`) is the Monte-Carlo p under uniform random angles (_moore_pvalue_mc)."""
    theta, rho = np.asarray(theta, float), np.asarray(rho, float)
    n = len(theta)
    if n < 3:
        return dict(n=n, Rstar=np.nan, mean_dir_deg=np.nan, p_asymptotic=np.nan, p_perm=np.nan,
                    perm_method='n<3')
    ranks = rankdata(rho)
    C, S = float(ranks @ np.cos(theta)), float(ranks @ np.sin(theta))
    r_obs = np.hypot(C, S) / n ** 1.5
    return dict(n=n, Rstar=float(r_obs), mean_dir_deg=float(np.degrees(np.arctan2(S, C)) % 360.0),
                p_asymptotic=float(min(1.0, np.exp(-3.0 * r_obs ** 2))),
                p_perm=_moore_pvalue_mc(ranks, r_obs, rng, n_perm),
                perm_method=f'monte_carlo_uniform_angles x{n_perm}')


def moore_test_clustered(theta: np.ndarray, rho: np.ndarray, animal_idx: np.ndarray,
                         rng: np.random.Generator, n_perm: int = CLUSTER_N_PERM) -> dict:
    """Moore's R* on every cell vector, with an animal-clustered null: each permutation rotates ALL
    cells of an animal by one common random angle (keeping the within-animal structure and the
    length ranks), i.e. it asks whether the animals' orientations are non-uniform, not whether
    cells are. Needs >= 3 animals."""
    theta, rho = np.asarray(theta, float), np.asarray(rho, float)
    n = len(theta)
    n_anim = int(animal_idx.max()) + 1 if n else 0
    if n < 3 or n_anim < 3:
        return dict(n=n, n_animals=n_anim, Rstar=np.nan, mean_dir_deg=np.nan, p_perm=np.nan,
                    perm_method='n_animals<3')
    ranks = rankdata(rho)
    C, S = float(ranks @ np.cos(theta)), float(ranks @ np.sin(theta))
    r_obs = np.hypot(C, S) / n ** 1.5
    count, done = 0, 0
    while done < n_perm:
        chunk = min(2000, n_perm - done)
        phi = rng.uniform(0.0, 2 * np.pi, size=(chunk, n_anim))
        th = theta[None, :] + phi[:, animal_idx]
        count += int((np.hypot(np.cos(th) @ ranks, np.sin(th) @ ranks) / n ** 1.5 >= r_obs - 1e-12).sum())
        done += chunk
    return dict(n=n, n_animals=n_anim, Rstar=float(r_obs),
                mean_dir_deg=float(np.degrees(np.arctan2(S, C)) % 360.0),
                p_perm=(1 + count) / (n_perm + 1), perm_method='animal_cluster_rotation')


def _d4_variants(arr_flat: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """The 8 symmetries of the square grid (4 rotations x optional mirror) of a flat (nx*ny) map,
    as (8, n_bins). Variant 0 is the identity. Exact on the lattice, about the grid centre."""
    a = arr_flat.reshape(nx, ny)
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out.append(r.ravel())
        out.append(r[::-1].ravel())
    return np.array(out)


def pooled_map_permutation(handler, cells: list, map_type: str, kind: str, dxy: tuple,
                           rng: np.random.Generator, n_perm: int = POOLED_N_PERM) -> dict:
    """Pooled-map permutation null of the resultant length (see section comment). `cells` must
    be the cells that have a per-cell vector for this map type, so the pooled map and the
    per-cell tests use the same cells."""
    nx, ny = handler.nx, handler.ny
    cx_bin, cy_bin, _ = _circle_centre_from_circumference(handler)
    if nx != ny or abs(cx_bin - (nx - 1) / 2.0) > 1e-9 or abs(cy_bin - (ny - 1) / 2.0) > 1e-9:
        return dict(R_obs=np.nan, p_perm=np.nan, note='arena centre not on the grid centre')
    usable = [(c, wm) for c in cells if (wm := _cell_weight_map(handler, c, map_type, 'rate')) is not None]
    n = len(usable)
    if n == 0:
        return dict(R_obs=np.nan, p_perm=np.nan, note='no cells')

    # per cell, per symmetry: numerator weights A, support B (normaliser / smoothing support)
    # and occupancy-valid V (defines the coverage reference centroid)
    A = np.zeros((n, 8, handler.n_bins))
    B = np.zeros_like(A)
    V = np.zeros_like(A)
    for i, (c, (w, sup)) in enumerate(usable):
        valid = c['valid'] & handler.geom_valid
        A[i] = _d4_variants(w, nx, ny)
        B[i] = _d4_variants(sup.astype(float), nx, ny)
        V[i] = _d4_variants(valid.astype(float), nx, ny)

    def _R(idx):
        ar = np.arange(n)
        a, b, v = A[ar, idx].sum(0), B[ar, idx].sum(0), V[ar, idx].sum(0)
        sup = b > 0
        # 'overall' pools as a per-bin mean over the cells valid there (pool_fine_map); the other
        # two are sums (pool_field_only_map / pool_peak_proportion_fine), whose common scale
        # cancels in the centroid
        F = np.where(sup, a / np.where(sup, b, 1.0), 0.0) if map_type == 'overall' else a
        if kind == 'kde':
            F = np.where(sup, handler.smooth(np.where(sup, F, 0.0), sup), 0.0)
        vec = _weighted_offset(dxy, F, v > 0)
        return (np.nan, np.nan) if vec is None else (float(np.hypot(*vec)),
                                                     float(np.degrees(np.arctan2(vec[1], vec[0])) % 360.0))

    r_obs, dir_obs = _R(np.zeros(n, dtype=int))
    r_null = np.array([_R(rng.integers(0, 8, size=n))[0] for _ in range(n_perm)])
    return dict(R_obs=r_obs, mean_dir_deg=dir_obs,
                p_perm=float((1 + np.sum(r_null >= r_obs - 1e-12)) / (n_perm + 1)),
                R_null_p95=float(np.nanpercentile(r_null, 95)), n_perm=n_perm, note='')


def plot_cell_vectors(kind: str, data_by_map: dict, save_path: str):
    """One polar panel per map type: every cell's vector (angle, length in arena radii) as a dot
    coloured by animal, the animal-mean vectors as arrows, and the test p-values in the title."""
    fig = plt.figure(figsize=(5.2 * len(_MAP_ORDER), 5.6))
    animals_all = sorted({a for d in data_by_map.values() for a in d['animals']})
    colours = {a: plt.get_cmap('tab10')(i % 10) for i, a in enumerate(animals_all)}
    for mi, map_type in enumerate(_MAP_ORDER):
        ax = fig.add_subplot(1, len(_MAP_ORDER), mi + 1, projection='polar')
        d = data_by_map.get(map_type)
        if d is None or len(d['theta']) == 0:
            ax.set_title(f'{_MAP_TITLES[map_type]}\n(no cells)', fontsize=9)
            continue
        rmax = max(float(d['rho'].max()), float(d['animal_rho'].max()), 1e-6) * 1.1
        for a in np.unique(d['animals']):
            m = d['animals'] == a
            ax.scatter(d['theta'][m], d['rho'][m], s=14, alpha=0.5, color=colours[a],
                       label=f'{a} (n={m.sum()})')
        for a, th, r in zip(d['animal_labels'], d['animal_theta'], d['animal_rho']):
            ax.annotate('', xy=(th, r), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=colours[a], lw=2.2))
        ax.set_ylim(0, rmax)
        t = d['tests']
        ax.set_title(f"{_MAP_TITLES[map_type]}\n"
                     f"Moore (animal means, n={t['animal']['n']}): p={t['animal']['p_perm']:.3g}\n"
                     f"Moore (cells, animal-clustered, n={t['cluster']['n']}): p={t['cluster']['p_perm']:.3g}",
                     fontsize=8)
        if mi == 0:
            ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(-0.25, 1.15))
    fig.suptitle(f'Per-cell direction vectors, {_VECTOR_KIND_TITLES[kind]}\n'
                 'dots = cells, arrows = animal means; length in arena radii'
                 + ('  [bins selected on the same cells -- p optimistic]' if kind == 'kde_sig' else ''),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def run_directional_vector_analysis(handler, cells: list, out_dir: str,
                                    fine_kde_results: dict | None = None) -> dict:
    """Per-cell vectors + Moore's test across animals / cells + pooled-map permutation, for the
    open field (see the section comment). Writes DirectionalVector_OpenField.xlsx (sheets Tests,
    CellVectors, AnimalVectors) and one DirectionalVector_OpenField_<kind>.png per kind."""
    rng = np.random.default_rng(VECTOR_SEED)
    dxy = _bin_offsets(handler)
    kde = fine_kde_results[RAYLEIGH_ARENA_KEY] if fine_kde_results is not None else None
    kinds = [k for k in _VECTOR_KINDS if k != 'kde_sig' or kde is not None]
    cell_animals = [_animal_id(c['session']) for c in cells]

    test_rows, cell_rows, animal_rows, results = [], [], [], {}
    for kind in kinds:
        data_by_map = {}
        for map_type in _MAP_ORDER:
            sig = kde[map_type]['sig_mask'] if (kind == 'kde_sig' and kde is not None) else None
            used, vx, vy = [], [], []
            for i, c in enumerate(cells):
                wm = _cell_weight_map(handler, c, map_type, kind, sig)
                vec = _weighted_offset(dxy, wm[0], c['valid'] & handler.geom_valid) if wm else None
                if vec is not None:
                    used.append(i); vx.append(vec[0]); vy.append(vec[1])
            if not used:
                continue
            vx, vy = np.array(vx), np.array(vy)
            theta, rho = np.arctan2(vy, vx), np.hypot(vx, vy)
            animals = np.array([cell_animals[i] for i in used])
            animal_labels = sorted(set(animals))
            animal_idx = np.array([animal_labels.index(a) for a in animals])

            # animal-mean vectors: average the (x, y) vectors of an animal's cells, then Moore's
            # test on (angle, length) of those means
            avx = np.array([vx[animals == a].mean() for a in animal_labels])
            avy = np.array([vy[animals == a].mean() for a in animal_labels])
            a_theta, a_rho = np.arctan2(avy, avx), np.hypot(avx, avy)

            t_animal  = moore_test(a_theta, a_rho, rng)
            t_cluster = moore_test_clustered(theta, rho, animal_idx, rng)
            t_naive   = moore_test(theta, rho, rng)
            t_pooled  = (pooled_map_permutation(handler, [cells[i] for i in used], map_type, kind, dxy, rng)
                         if kind in ('rate', 'kde') else None)

            sel_note = 'bins selected on the same cells: p optimistic' if kind == 'kde_sig' else ''
            common = dict(kind=kind, map_type=map_type, n_cells=len(used), n_animals=len(animal_labels))
            test_rows.append(dict(common, test='Moore, animal-mean vectors', n_units=t_animal['n'],
                                  statistic_Rstar=t_animal['Rstar'], mean_dir_deg=t_animal['mean_dir_deg'],
                                  p_asymptotic=t_animal['p_asymptotic'], p_perm=t_animal['p_perm'],
                                  perm_method=t_animal['perm_method'], note=sel_note))
            test_rows.append(dict(common, test='Moore, cell vectors, animal-clustered permutation',
                                  n_units=t_cluster['n'], statistic_Rstar=t_cluster['Rstar'],
                                  mean_dir_deg=t_cluster['mean_dir_deg'], p_asymptotic=np.nan,
                                  p_perm=t_cluster['p_perm'], perm_method=t_cluster['perm_method'],
                                  note=sel_note))
            test_rows.append(dict(common, test='Moore, cell vectors, cells independent (reference only)',
                                  n_units=t_naive['n'], statistic_Rstar=t_naive['Rstar'],
                                  mean_dir_deg=t_naive['mean_dir_deg'],
                                  p_asymptotic=t_naive['p_asymptotic'], p_perm=t_naive['p_perm'],
                                  perm_method=t_naive['perm_method'],
                                  note='pseudoreplicated' + ('; ' + sel_note if sel_note else '')))
            if t_pooled is not None:
                test_rows.append(dict(common, test='Pooled-map D4 permutation (resultant length)',
                                      n_units=len(used), statistic_Rstar=t_pooled['R_obs'],
                                      mean_dir_deg=t_pooled.get('mean_dir_deg', np.nan),
                                      p_asymptotic=np.nan, p_perm=t_pooled['p_perm'],
                                      perm_method=f"D4 x{t_pooled.get('n_perm', 0)}",
                                      note=(t_pooled['note'] + '; ' if t_pooled['note'] else '')
                                           + 'statistic is the pooled resultant length R, not R*; '
                                           + f"null 95th pct R = {t_pooled.get('R_null_p95', np.nan):.4f}; "
                                           + 'cells treated as independent'))

            for j, i in enumerate(used):
                cell_rows.append(dict(kind=kind, map_type=map_type, session=cells[i]['session'],
                                      unit=cells[i]['unit'], animal=animals[j], vx=vx[j], vy=vy[j],
                                      angle_deg=float(np.degrees(theta[j]) % 360.0), length=rho[j]))
            for a, x, y, th, r in zip(animal_labels, avx, avy, a_theta, a_rho):
                animal_rows.append(dict(kind=kind, map_type=map_type, animal=a,
                                        n_cells=int((animals == a).sum()), vx=x, vy=y,
                                        angle_deg=float(np.degrees(th) % 360.0), length=r))

            data_by_map[map_type] = dict(theta=theta, rho=rho, animals=animals,
                                         animal_labels=animal_labels, animal_theta=a_theta,
                                         animal_rho=a_rho,
                                         tests=dict(animal=t_animal, cluster=t_cluster, naive=t_naive,
                                                    pooled=t_pooled))
        results[kind] = data_by_map
        if data_by_map:
            plot_cell_vectors(kind, data_by_map,
                              os.path.join(out_dir, f'DirectionalVector_OpenField_{kind}.png'))

    out_path = os.path.join(out_dir, 'DirectionalVector_OpenField.xlsx')
    with pd.ExcelWriter(out_path) as xw:
        pd.DataFrame(test_rows).to_excel(xw, sheet_name='Tests', index=False)
        pd.DataFrame(cell_rows).to_excel(xw, sheet_name='CellVectors', index=False)
        pd.DataFrame(animal_rows).to_excel(xw, sheet_name='AnimalVectors', index=False)
    print(f'[SAVED] {out_path}')

    for r in test_rows:
        if r['test'].startswith(('Moore, animal', 'Pooled')):
            print(f"[{RAYLEIGH_ARENA_KEY}/{r['kind']}/{r['map_type']}] {r['test']}: "
                  f"n={r['n_units']} (animals={r['n_animals']}), stat={r['statistic_Rstar']:.4f}, "
                  f"dir={r['mean_dir_deg']:.1f} deg, p_perm={r['p_perm']:.4g}"
                  + (' -> significant' if np.isfinite(r['p_perm']) and r['p_perm'] < RAYLEIGH_ALPHA else ''))
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


def _diverging_cmap_norm(values) -> tuple:
    """Symmetric-about-zero RdBu_r scale for the observed-minus-null difference panels, so
    zero is always the colormap's white midpoint and equal departures either side read the
    same. Counterpart of make_cmap_norm for signed quantities."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    abs_max = float(np.max(np.abs(vals))) if len(vals) else 1.0
    abs_max = max(abs_max, 1e-6)
    cmap = _get_cmap('RdBu_r')
    cmap.set_bad('white')
    return cmap, Normalize(vmin=-abs_max, vmax=abs_max)


def _mappable(cmap, norm):
    """A ScalarMappable carrying (cmap, norm), for a colorbar that is attached to several
    axes at once rather than to one image."""
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm


def plot_fig_S1H(arena_handlers: dict, arena_results: dict, save_path: str):
    fig = plt.figure(figsize=(15, 5))
    for i, key in enumerate(_ARENA_ORDER):
        handler = arena_handlers[key]
        mean_map, valid = pool_fine_map(handler, arena_results[key])
        cmap, norm = make_cmap_norm(mean_map[valid])

        proj = 'polar' if key == 'circular_track' else None
        ax = fig.add_subplot(1, 3, i + 1, projection=proj)
        pcm = handler.plot_fine(ax, mean_map, valid, cmap, norm)

        n_place = len(arena_results[key])
        ax.set_title(f'{_ARENA_TITLES[key]}\n(n={n_place} place cells)')
        fig.colorbar(pcm, ax=ax, shrink=0.7, label='Field index (a.u.)')

    fig.suptitle('Fig S1H -- Overall mean field index maps (place cells pooled across days)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_field_only_mean_maps(arena_handlers: dict, arena_results: dict, save_path: str):
    """Additional mean field-index maps built ONLY from each place cell's extracted
    place-field bins (pool_field_only_map), weighted by both how many fields overlap each
    bin and those fields' own field index (see pool_field_only_map's docstring). Each cell
    is normalized to its own peak before pooling (fi_map), so values are a field-density x
    field-index composite, not raw Hz."""
    fig = plt.figure(figsize=(15, 5))
    for i, key in enumerate(_ARENA_ORDER):
        handler = arena_handlers[key]
        mean_map, valid = pool_field_only_map(handler, arena_results[key])
        cmap, norm = make_cmap_norm(mean_map[valid])

        proj = 'polar' if key == 'circular_track' else None
        ax = fig.add_subplot(1, 3, i + 1, projection=proj)
        pcm = handler.plot_fine(ax, mean_map, valid, cmap, norm)

        n_place = len(arena_results[key])
        ax.set_title(f'{_ARENA_TITLES[key]}\n(n={n_place} place cells)')
        fig.colorbar(pcm, ax=ax, shrink=0.7, label='Field density x field index (a.u.)')

    fig.suptitle('Field-only mean field-index maps (field-density-weighted, place-field bins only)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_fig_1BD(arena_handlers: dict, arena_results: dict, save_path: str):
    """Colour scaling follows QUADRANT_COLOUR_SCALE -- see that constant for the trade-off."""
    shared = (QUADRANT_COLOUR_SCALE == 'shared')
    peak_pct, rate = {}, {}
    for key in _ARENA_ORDER:
        handler = arena_handlers[key]
        peak_pct[key] = pool_quadrant_peak_proportion(handler, arena_results[key])
        rate[key]     = pool_quadrant_mean_rate(handler, arena_results[key])

    if shared:
        cmap_b, norm_b = make_cmap_norm(np.concatenate([pct for pct, _ in peak_pct.values()]))
        cmap_d, norm_d = make_cmap_norm(np.concatenate([vals[valid] for vals, valid in rate.values()]))

    fig = plt.figure(figsize=(15, 10))
    axes_b, axes_d = [], []
    for i, key in enumerate(_ARENA_ORDER):
        handler = arena_handlers[key]
        proj = 'polar' if key == 'circular_track' else None

        pct, total = peak_pct[key]
        pct_valid = np.isfinite(pct)
        vals, valid = rate[key]
        if not shared:
            # per-arena colour scales + colorbars, for the reason spelled out in
            # plot_quadrant_kde: a normalization pooled over the arenas makes an
            # open-field-only change repaint the linear and circular panels even though
            # their values did not move
            cmap_b, norm_b = make_cmap_norm(pct[pct_valid])
            cmap_d, norm_d = make_cmap_norm(vals[valid])

        ax_b = fig.add_subplot(2, 3, i + 1, projection=proj)
        im_b = handler.plot_quadrant_map(ax_b, np.nan_to_num(pct), pct_valid, cmap_b, norm_b)
        ax_b.set_title(f'{_ARENA_TITLES[key]}\nPeak proportion (%), n={total}')
        axes_b.append(ax_b)

        ax_d = fig.add_subplot(2, 3, i + 4, projection=proj)
        im_d = handler.plot_quadrant_map(ax_d, np.nan_to_num(vals), valid, cmap_d, norm_d)
        ax_d.set_title(f'{_ARENA_TITLES[key]}\nMean field index (a.u.)')
        axes_d.append(ax_d)

        if not shared:
            fig.colorbar(im_b, ax=ax_b, shrink=0.75, pad=0.04, label='% of peaks')
            fig.colorbar(im_d, ax=ax_d, shrink=0.75, pad=0.04, label='Mean field index (a.u.)')

    if shared:
        fig.colorbar(_mappable(cmap_b, norm_b), ax=axes_b, shrink=0.6, label='% of peaks')
        fig.colorbar(_mappable(cmap_d, norm_d), ax=axes_d, shrink=0.6, label='Mean field index (a.u.)')

    scale_note = ('Colour scales are SHARED across arenas within each row.' if shared else
                  'Colour scales are per panel -- read each against its own colorbar, not across arenas.')
    fig.suptitle('Figure 1B/D -- Quadrant mean maps: folded peak-location proportion (top) and mean field index '
                 f'(bottom).\n{scale_note}')
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
            # via _session_positions, so the debug maps are built on the same frame-corrected
            # positions as the real ones -- calling _load_tracking/_smooth_tracking_position
            # directly here used to skip the open-field centring and 120 deg rotation, which
            # made every debug map of a 'Rotate' session disagree with its real counterpart
            x_cm, y_cm, t = _session_positions(csv_path, handler)
        except Exception as e:
            print(f'  ERROR loading tracking [{session_name}]: {e}')
            continue
        if len(t) < 2:
            print(f'  [SKIP no tracking] {session_name}')
            continue

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
                bootstrap_sig=r['bootstrap_sig'],
                peak_quadrant_bin=q,
            ))
    df = pd.DataFrame(rows)
    df.to_excel(out_path, index=False)
    print(f'[SAVED] {out_path}')


def run_circular_track_pipeline_test(cfg: dict, out_dir: str) -> None:
    """DEBUG: runs the FULL downstream pipeline (bootstrap, coverage filtering at
    whatever COVERAGE_FRACTION is currently set to, pooling, quadrant fold, plotting,
    excel export) for the circular_track arena ALONE -- to check
    that the rest of the analysis actually completes on this data once the coverage
    threshold is lowered (e.g. sessions like 2NoRot at ~78% bin coverage, excluded by the
    original 80% threshold), without waiting on the other two arenas."""
    os.makedirs(out_dir, exist_ok=True)
    handler, results = collect_arena_results('circular_track', cfg)

    n_processed = len(results)
    n_place = n_processed
    print(f'[circular_track pipeline test] COVERAGE_FRACTION={COVERAGE_FRACTION:.0%}, '
          f'{n_processed} units processed (all pooled).')
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
        fig.colorbar(pcm, ax=ax, shrink=0.7, label='Field density x field index (a.u.)')
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
# Uniform-field null: expected-map figure + comparison against the old occupancy null
# ============================================================================

def plot_uniform_field_null_maps(arena_handlers: dict, arena_results: dict, save_path: str):
    """Raw (unsmoothed) expected overall / field-only / peak-percentage maps for fields
    uniformly tiling each arena, given that arena's sessions' occupancy (see
    build_uniform_field_null). Colour scale is per panel: the three map types are in different
    units, and arenas differ in bin count."""
    fig = plt.figure(figsize=(15, 4.6 * len(_MAP_ORDER)))
    for ci, key in enumerate(_ARENA_ORDER):
        handler = arena_handlers[key]
        nul = build_uniform_field_null(handler, arena_results[key])
        for mi, map_type in enumerate(_MAP_ORDER):
            if map_type == 'peak':
                vals, valid = np.nan_to_num(nul['peak_pct']), np.isfinite(nul['peak_pct'])
                label = '% of peaks'
            else:
                vals, valid = nul[map_type]
                label = ('P(bin in field)' if map_type == 'overall' else 'share of field coverage')
            valid = valid & getattr(handler, 'geom_valid', np.ones(handler.n_bins, dtype=bool))
            cmap, norm = make_cmap_norm(vals[valid])
            proj = 'polar' if key == 'circular_track' else None
            ax = fig.add_subplot(len(_MAP_ORDER), 3, mi * 3 + ci + 1, projection=proj)
            pcm = handler.plot_fine(ax, vals, valid, cmap, norm)
            ax.set_title(f"{_ARENA_TITLES[key]} -- expected {_MAP_TITLES[map_type]}\n"
                         f"{FIELD_FOOTPRINT_BINS[key]}-bin fields, {nul['n_placements']} placements, "
                         f"{nul['n_cells']} cells / {nul['n_sessions']} sessions", fontsize=8)
            fig.colorbar(pcm, ax=ax, shrink=0.7, label=label)
    fig.suptitle('Expected maps under uniformly tiled place fields, given each session\'s occupancy\n'
                 f'(peak bin drawn within a field by "{PEAK_WITHIN_FIELD}"; unsmoothed)')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_null_comparison(arena_handlers: dict, results: dict, save_path: str, kind: str):
    """Where the two nulls disagree about significance. Per bin: significant vs BOTH nulls,
    only the uniform-field null, only the old occupancy null, or neither. kind is 'fine'
    (whole arena, results of run_fine_kde_analysis) or 'quad' (folded, run_quadrant_kde_analysis)."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    valid_key = 'fine_valid' if kind == 'fine' else 'quad_valid'
    colours = ['#e6e6e6', '#404040', '#D55E00', '#0072B2']
    cmap = ListedColormap(colours)
    cmap.set_bad('white')
    norm = Normalize(vmin=-0.5, vmax=3.5)

    fig = plt.figure(figsize=(15, 4.2 * len(_MAP_ORDER)))
    for mi, map_type in enumerate(_MAP_ORDER):
        for ci, key in enumerate(_ARENA_ORDER):
            handler = arena_handlers[key]
            res = results[key][map_type]
            su, so = res['sig_mask_uniform'], res['sig_mask_occ']
            codes = np.zeros(len(su))
            codes[su & so] = 1
            codes[su & ~so] = 2
            codes[~su & so] = 3
            proj = 'polar' if key == 'circular_track' else None
            ax = fig.add_subplot(len(_MAP_ORDER), 3, mi * 3 + ci + 1, projection=proj)
            plot = handler.plot_fine if kind == 'fine' else handler.plot_quadrant_map
            plot(ax, codes, res[valid_key], cmap, norm)
            ax.set_title(f"{_ARENA_TITLES[key]} -- {_MAP_TITLES[map_type]}\n"
                         f"sig. bins: uniform-field {int(su.sum())}, occupancy {int(so.sum())}, "
                         f"both {int((su & so).sum())}", fontsize=8)
    fig.legend(handles=[Patch(color=colours[1], label='significant vs both nulls'),
                        Patch(color=colours[2], label='only vs uniform-field null'),
                        Patch(color=colours[3], label='only vs occupancy null (old)'),
                        Patch(color=colours[0], label='not significant')],
               loc='lower center', ncol=4, fontsize=9)
    fig.suptitle(f"{'Whole-arena' if kind == 'fine' else 'Quadrant-fold'} KDE: significance under the two nulls")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def _null_comparison_row(analysis, arena_key, map_type, side, n_tested, res, null_u, null_o, tested,
                          sig_u, sig_o, depl_u, depl_o, clusters_u, clusters_o) -> dict:
    a, b = null_u[tested], null_o[tested]
    corr = float(np.corrcoef(a, b)[0, 1]) if tested.sum() > 2 and a.std() > 0 and b.std() > 0 else np.nan
    return dict(analysis=analysis, arena=arena_key, map_type=map_type, side=side,
                n_place_cells=res['n_place_cells'], n_tested=int(n_tested),
                n_sig_uniform=int(sig_u.sum()), n_sig_occupancy=int(sig_o.sum()),
                n_sig_both=int((sig_u & sig_o).sum()),
                n_only_uniform=int((sig_u & ~sig_o).sum()), n_only_occupancy=int((~sig_u & sig_o).sum()),
                n_depleted_uniform=int(depl_u.sum()), n_depleted_occupancy=int(depl_o.sum()),
                n_clusters_uniform=clusters_u, n_clusters_occupancy=clusters_o,
                null_corr=round(corr, 4) if np.isfinite(corr) else np.nan,
                null_total_variation_pct=round(float(0.5 * np.abs(a - b).sum()), 4))


def export_null_comparison(boundary_res: dict, quad_res: dict, fine_res: dict, out_path: str):
    """One row per analysis x arena x map type (x side): how many positions/bins are
    significantly ABOVE each null (observed bootstrap 2.5th percentile > null), how many are
    significantly BELOW (97.5th percentile < null, 'depleted'), the overlap of the two
    significance calls, and how different the two null curves themselves are (correlation and
    total variation in percentage points over the tested positions)."""
    rows = []
    for arena_key, by_map in boundary_res.items():
        for map_type, res_or_sides in by_map.items():
            sides = res_or_sides if arena_key == 'circular_track' else {None: res_or_sides}
            for side, res in sides.items():
                nu, no = res['null_curve_uniform'], res['null_curve_occ']
                rows.append(_null_comparison_row(
                    'boundary_1D', arena_key, map_type, side, len(res['grid']), res, nu, no,
                    np.ones(len(res['grid']), dtype=bool),
                    res['real_lo'] > nu, res['real_lo'] > no, res['real_hi'] < nu, res['real_hi'] < no,
                    len(res['peaks_uniform']), len(res['peaks_occ'])))
    for analysis, results, valid_key in (('quadrant_fold', quad_res, 'quad_valid'),
                                         ('fine_map', fine_res, 'fine_valid')):
        for arena_key, by_map in results.items():
            for map_type, res in by_map.items():
                v = res[valid_key]
                rows.append(_null_comparison_row(
                    analysis, arena_key, map_type, None, v.sum(), res,
                    res['null_kde_uniform'], res['null_kde_occ'], v,
                    res['sig_mask_uniform'], res['sig_mask_occ'],
                    res['depl_mask_uniform'], res['depl_mask_occ'],
                    len(res['clusters_uniform']), len(res['clusters_occ'])))
    df = pd.DataFrame(rows)
    df.to_excel(out_path, index=False)
    print(f'[SAVED] {out_path}')
    with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 200):
        print(df.drop(columns=['n_place_cells']).to_string(index=False))


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
    plot_uniform_field_null_maps(arena_handlers, arena_results,
                                 os.path.join(out_dir, 'UniformFieldNull_ExpectedMaps.png'))
    boundary_res = run_boundary_firing_analysis(arena_handlers, arena_results, out_dir)
    quad_res     = run_quadrant_kde_analysis(arena_handlers, arena_results, out_dir)
    fine_res     = run_fine_kde_analysis(arena_handlers, arena_results, out_dir)
    run_rayleigh_analysis(arena_handlers, arena_results, out_dir, fine_kde_results=fine_res)
    run_directional_vector_analysis(arena_handlers[RAYLEIGH_ARENA_KEY],
                                    arena_results[RAYLEIGH_ARENA_KEY], out_dir,
                                    fine_kde_results=fine_res)

    plot_null_comparison(arena_handlers, quad_res,
                         os.path.join(out_dir, 'QuadrantFold_KDE_NullComparison.png'), 'quad')
    plot_null_comparison(arena_handlers, fine_res,
                         os.path.join(out_dir, 'FineMap_KDE_NullComparison.png'), 'fine')
    export_null_comparison(boundary_res, quad_res, fine_res,
                           os.path.join(out_dir, 'NullComparison_Significance.xlsx'))


if __name__ == '__main__':
    print(f"Using '{COORD_UNITS}' tracking coordinates.\n")

    run_full_pipeline(os.path.join(OUTPUT_DIR, 'AllArenas'))

    print('\nDone.')
