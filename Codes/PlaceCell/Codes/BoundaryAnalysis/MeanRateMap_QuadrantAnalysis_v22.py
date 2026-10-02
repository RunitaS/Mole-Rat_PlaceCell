# -*- coding: utf-8 -*-
"""
Mean Rate Map Analysis (S1H, after Muessig et al.)

Reproduces, for place cells pooled across multiple recording days:
  - Fig S1H : overall mean field-index map per arena, fine spatial bins (2 x 2 cm;
              genuinely 2D for every arena, including the linear track's length x
              width). Each cell's map is first normalized 0-1 to its own peak and
              minimum (smoothed) firing rate (field_index_map, place-cell method, as
              in ThetaMod_PhasePrecession_Stats_v3.py) before pooling across cells --
              this keeps a cell's contribution from being over- or under-weighted by
              its absolute peak rate or by where its field happens to sit in the map,
              which raw-Hz pooling does not.

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

VELOCITY FILTER: after smoothing, speed is computed frame by frame from the smoothed trajectory,
and every spike whose matched frame has speed < MIN_SPEED_CMS (0.5 cm/s, immobility -- non-spatial
SWR / consolidation firing) or > MAX_SPEED_CMS (90 cm/s, tracking artifact) is excluded from ALL
analyses; with SPEED_FILTER_OCCUPANCY those frames are dropped from occupancy as well (see
_speed_mask, compute_cell_ratemap).

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
import hashlib
import random
import concurrent.futures

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.colors import Normalize

# ============================================================================
# CONFIGURATION -- edit root folder + geometry below
# ============================================================================

# Single root under which every animal/arena/day/session lives. Arena type is auto-detected
# per session from its path (see ARENA_FOLDER_KEYWORDS / _detect_arena_key below) -- no
# per-arena root folders needed any more.
ROOT_DIRECTORY = r'X:\NMR_group_data\Runita\Analysis\Mean_KDE_Open_PascalOldenburg\Open_KDE\CorrectedData\Data\SessionTypeSorted_PC\Open\Cntrl'

# Output folder for all figures / workbooks (same location as before, now derived from the root).
OUTPUT_DIR = os.path.join(ROOT_DIRECTORY, 'MeanRM_Quad_CorrRayleigh_Test')

# Whole-arena bin coverage criterion (session-level). True: a session must have >= min_occ_s
# occupancy in at least COVERAGE_FRACTION of the arena's total spatial bins (per handler, see
# `total_arena_bins`/`coverage_threshold_bins`), else every file (unit) from that session is
# skipped -- computed dynamically per handler, so it tracks target_bin_cm/geometry.
# False: the criterion is not applied.
USE_BIN_COVERAGE_CRITERION = False
COVERAGE_FRACTION = 0.01      # DEBUG: temporarily lowered from 0.80 to test whether the
                               # rest of the pipeline (bootstrap, place-cell qualification,
                               # pooling, plotting) runs end-to-end on
                               # circular-track sessions that the 80% threshold was excluding
                               # (e.g. 2NoRot at ~78% bin coverage)

# Edge/centre zone occupancy criterion (session-level). True: a session is only included if the
# animal's tracked occupancy covers BOTH the edge zone AND the centre zone by at least
# MIN_ZONE_COVERAGE_PCT of the session's total occupied time; a failing session has every unit
# in it skipped. False: the criterion is not applied (sessions are not screened by zone
# coverage, and the tracking is not even read for it). See session_zone_coverage /
# collect_arena_results.
USE_ZONE_COVERAGE_CRITERION = False
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


# ── Open-field arena centring: _load_tracking anchors each session's coordinate frame at the
# tracking's own minimum (`x - x.min()`, `y - y.min()`), so the tracking's midpoint sits at
# (x_span/2, y_span/2). The handler treats (cx, cy) = (diameter/2, diameter/2) as the arena
# centre, and the two only coincide when the tracking spans exactly one diameter on both axes.
# Measured spans in this dataset run 58.6-60.0 cm in x but only 53.3-60.0 cm in y against a
# 60 cm arena, which would leave the assumed centre up to ~3.4 cm off.
#
# Every radius-based quantity in the pipeline is measured from (cx, cy) -- geom_valid,
# edge_zone_flat (a 9.25 cm edge band) and to_bins' sample_valid -- so that offset would bias
# the edge-vs-centre split of every open-field session.
#
# The correction is applied once, at load time (_session_positions): take the midpoint of the
# tracking's x range and y range, ((x.min() + x.max()) / 2, (y.min() + y.max()) / 2), and
# translate the tracking so that midpoint lands on (cx, cy).
#
# Set to False to skip the translation (tracking stays anchored at its own minimum).
CENTRE_OPEN_FIELD_TRACKING = True


def _centre_open_field_tracking(x_cm: np.ndarray, y_cm: np.ndarray, session_dir: str,
                                handler) -> tuple:
    """Translates an OPEN FIELD session's tracking so the midpoint of its x and y ranges lands
    on the handler's (cx, cy), the point every radius-based measure in the pipeline is taken
    from (see the CENTRE_OPEN_FIELD_TRACKING comment above). Any other arena is returned
    unchanged -- the linear track's frame is defined by the track's own extent and the ring
    track is handled by CircularTrackHandler, so neither wants this."""
    if not CENTRE_OPEN_FIELD_TRACKING:
        return x_cm, y_cm
    if _detect_arena_key(session_dir) != 'open_field' or len(x_cm) == 0:
        return x_cm, y_cm

    mid_x = 0.5 * (float(x_cm.min()) + float(x_cm.max()))
    mid_y = 0.5 * (float(y_cm.min()) + float(y_cm.max()))
    return x_cm + (handler.cx - mid_x), y_cm + (handler.cy - mid_y)


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
POS_SMOOTH_SIGMA_SMP = 5.0    # Gaussian smoothing sigma (samples) applied to x/y tracking position

# ── Velocity filter (applied before ANY analysis). Spikes fired while the animal is immobile
# are dominated by non-spatial sharp-wave-ripple (SWR) / memory-consolidation firing, which
# biases the pooled firing density; implausibly fast frames are tracking artifacts. Speed is
# computed from the SMOOTHED trajectory as frame-by-frame displacement / frame interval
# (~33 ms at fps=30), and every spike whose matched tracking frame has speed outside
# [MIN_SPEED_CMS, MAX_SPEED_CMS] is discarded -- from the rate maps, SIR/sparsity, the
# bootstrap, peak/field extraction, and so from every pooled map and statistic downstream
# (see _speed_mask / compute_cell_ratemap).
SPEED_FILTER_ENABLED = True
MIN_SPEED_CMS        = 0.5    # below this the animal is treated as immobile (SWR-prone)
MAX_SPEED_CMS        = 90.0   # above this the frame is treated as a tracking artifact
# True: the same out-of-range frames are also removed from OCCUPANCY (standard practice) --
# otherwise the immobile time would stay in the rate denominator with its spikes removed,
# artificially lowering the rate wherever the animal rests (typically near walls).
# False: only the spikes are removed; occupancy uses every frame.
SPEED_FILTER_OCCUPANCY = True
RATEMAP_SMOOTH_SIGMA_BINS = 3.0  # Gaussian smoothing sigma (bins) applied to rate maps

# Place-field extraction (pass-index 'place' filter-band criteria, auto_filter_band in
# pass_index_parser.m: field = area with rate >= 20% of the peak): a bin qualifies if its
# SMOOTHED rate exceeds both the cell's mean firing rate and this fraction of the smoothed
# rate map's peak; a connected run of qualifying bins is only kept as a field if it spans at
# least MIN_FIELD_BINS contiguous bins.
FIELD_PEAK_FRAC = 0.20
MIN_FIELD_BINS  = 9

# ── 2D kernel density estimate of each pooled map (overall mean, field-only mean, peak
# proportion) -- see kde_density_map. scipy.stats.gaussian_kde is fit to the physical (cm)
# bin-centre coordinates, weighted by each bin's firing mass (map value x bin area; raw peak
# counts for the peak-proportion map), with the bandwidth chosen by Scott's rule:
# H = neff^(-1/(d+4))^2 * weighted covariance, neff = (sum w)^2 / sum w^2, d = 2.
KDE_BW_METHOD = 'scott'
# True: divide the KDE at each point by the fraction of its kernel that falls inside the
# sampled arena, so bins near the walls are not biased low by kernel mass leaking out of the
# arena (important for an edge-vs-centre analysis). False: plain scipy KDE.
KDE_EDGE_CORRECTION = True

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
    open-field centring correction (_centre_open_field_tracking, a no-op outside the open
    field). Returns (x_cm, y_cm, t) ready for handler.orient / handler.to_bins.

    Every consumer -- the coverage screen, the per-unit rate maps, the debug maps and the
    external ListIncludedSessionsCells.py audit -- must go through here rather than calling
    _load_tracking/_smooth_tracking_position itself. When those corrections lived at the call
    sites instead, two paths (the audit script and the debug map builder) silently skipped
    them and reported a different set of included sessions than the pipeline actually used."""
    x_cm, y_cm, t = _load_tracking(csv_path, handler.arena_width_cm)
    if len(t) < 2:
        return x_cm, y_cm, t
    x_cm, y_cm = _smooth_tracking_position(x_cm, y_cm, t)
    session_dir = os.path.dirname(csv_path)
    x_cm, y_cm = _centre_open_field_tracking(x_cm, y_cm, session_dir, handler)
    return x_cm, y_cm, t


def _speed_mask(x_cm: np.ndarray, y_cm: np.ndarray, t_us: np.ndarray) -> np.ndarray:
    """Per-frame boolean: True where the animal's speed lies within
    [MIN_SPEED_CMS, MAX_SPEED_CMS] (all True when SPEED_FILTER_ENABLED is False).

    Speed of frame i is the displacement of the smoothed trajectory from frame i-1 to i
    divided by the actual timestamp interval (so dropped frames are handled correctly);
    frame 0 takes frame 1's speed. Must be called on the FULL, contiguous tracking series
    (before any off-arena samples are removed), otherwise removed samples would create
    artificial jumps."""
    n = len(t_us)
    if not SPEED_FILTER_ENABLED or n < 2:
        return np.ones(n, dtype=bool)
    dt_s = np.diff(t_us) * 1e-6
    step_cm = np.hypot(np.diff(x_cm), np.diff(y_cm))
    speed = np.full(n - 1, np.nan)
    ok = dt_s > 0
    speed[ok] = step_cm[ok] / dt_s[ok]
    speed = np.concatenate(([speed[0]], speed))
    # NaN speed (zero dt) compares False on both sides -> frame excluded
    return (speed >= MIN_SPEED_CMS) & (speed <= MAX_SPEED_CMS)


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

        # edge/centre zone classification (MIN_ZONE_COVERAGE_PCT criterion): a bin is 'edge'
        # when within OPEN_EDGE_ZONE_THRESHOLD_CM of the wall, else 'centre' -- same rule as
        # classify_edge_centre's open-field branch in BoundaryAnalysis_Pipeline_v3.py
        dist_to_wall = np.clip(self.diameter / 2.0 - r, 0.0, None).ravel()
        self.edge_zone_flat = dist_to_wall <= OPEN_EDGE_ZONE_THRESHOLD_CM

        # total bins actually inside the circular arena (excludes the corner bins of the
        # bounding nx*ny grid that geom_valid already masks out) -- the denominator for the
        # 80% coverage criterion (COVERAGE_FRACTION)
        self.total_arena_bins = int(self.geom_valid.sum())
        self.coverage_threshold_bins = int(np.ceil(COVERAGE_FRACTION * self.total_arena_bins))

    def orient(self, x_cm, y_cm):
        return x_cm, y_cm

    def to_bins(self, x_cm, y_cm):
        bx = np.clip((x_cm / self.bin_cm).astype(int), 0, self.nx - 1)
        by = np.clip((y_cm / self.bin_cm).astype(int), 0, self.ny - 1)
        flat = bx * self.ny + by
        r = np.hypot(x_cm - self.cx, y_cm - self.cy)
        sample_valid = r <= (self.diameter / 2.0 + self.bin_cm)
        return flat, sample_valid

    def bin_centres_xy(self):
        """(2, n_bins) Cartesian cm centre of every flat bin (for kde_density_map)."""
        xs = (np.arange(self.nx) + 0.5) * self.bin_cm
        ys = (np.arange(self.ny) + 0.5) * self.bin_cm
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        return np.vstack([XX.ravel(), YY.ravel()])

    def bin_areas_cm2(self):
        return np.full(self.n_bins, self.bin_cm ** 2)

    def smooth(self, fr_flat, valid_flat):
        fr2 = fr_flat.reshape(self.nx, self.ny)
        v2  = valid_flat.reshape(self.nx, self.ny)
        return _gaussian_smooth_2d(fr2, v2).ravel()

    def connected_components(self, qualifies_flat):
        return _connected_components_2d_flat(qualifies_flat, self.nx, self.ny)

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

    def plot_bins_2d(self, ax, values_flat, valid_flat, cmap, norm):
        """Flat bin-grid heatmap (one cell per 2 x 2 cm bin, cm axes); bins outside the
        arena or not in valid_flat are left blank."""
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
        # kept as a multiple of 4 (preserves the established arc-bin layout)
        self.nx = max(8, 4 * int(round(circumference / bin_cm / 4.0)))   # along arc-length
        self.ny = max(2, int(round(self.track_width_cm / bin_cm)))       # across radial width
        self.bin_width_deg = 360.0 / self.nx
        self.bin_cm_y = self.track_width_cm / self.ny
        self.n_bins = self.nx * self.ny

        # each bin classified inner-ring vs outer-ring (by < ny//2 -- for the actual ny=2
        # config this is exactly the inner rim bin vs the outer rim bin)
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

    def bin_centres_xy(self):
        """(2, n_bins) Cartesian cm centre of every (arc, radial) bin -- same angle convention
        as to_bins. The KDE works in room coordinates, so the ring's wrap-around is handled
        naturally without any periodic kernel."""
        theta = np.radians((np.arange(self.nx) + 0.5) * self.bin_width_deg)
        r = self.inner_r + (np.arange(self.ny) + 0.5) * self.bin_cm_y
        TH, R = np.meshgrid(theta, r, indexing='ij')
        return np.vstack([(self.cx + R * np.cos(TH)).ravel(),
                          (self.cy + R * np.sin(TH)).ravel()])

    def bin_areas_cm2(self):
        """Annular-sector area of each bin (outer radial bins are slightly larger)."""
        dth = np.radians(self.bin_width_deg)
        r_lo = self.inner_r + np.arange(self.ny) * self.bin_cm_y
        r_hi = r_lo + self.bin_cm_y
        return np.tile(0.5 * dth * (r_hi ** 2 - r_lo ** 2), self.nx)

    def smooth(self, fr_flat, valid_flat):
        fr2 = fr_flat.reshape(self.nx, self.ny)
        v2  = valid_flat.reshape(self.nx, self.ny)
        return _gaussian_smooth_2d(fr2, v2, wrap_x=True).ravel()

    def connected_components(self, qualifies_flat):
        return _connected_components_2d_flat(qualifies_flat, self.nx, self.ny, wrap_x=True)

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

    def plot_bins_2d(self, ax, values_flat, valid_flat, cmap, norm):
        """The ring unrolled onto its actual (arc-length x radial-width) bin grid: x is
        arc-length at the mean radius (0 = East, counter-clockwise, same angle convention as
        to_bins), y is distance from the inner wall. Aspect is left free -- the track is ~150
        x 4 cm, so at equal aspect the bins would be unreadable."""
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

        # per-bin distance to the nearer end wall (DX, length axis) and nearer side wall
        # (DY, width axis)
        bx_centers_cm = (np.arange(self.nx) + 0.5) * self.bin_cm_x
        by_centers_cm = (np.arange(self.ny) + 0.5) * self.bin_cm_y
        dist_x = np.minimum(bx_centers_cm, self.length - bx_centers_cm)
        dist_y = np.minimum(by_centers_cm, self.width - by_centers_cm)
        DX, DY = np.meshgrid(dist_x, dist_y, indexing='ij')

        # edge/centre zone classification (MIN_ZONE_COVERAGE_PCT criterion): a bin is 'edge'
        # when it is at least LINEAR_EDGE_ZONE_THRESHOLD_CM from the track's midline (width
        # axis) OR within LINEAR_EDGE_ZONE_THRESHOLD_CM of either end wall (length axis),
        # else 'centre' -- same rule as classify_edge_centre's linear-track branch in
        # BoundaryAnalysis_Pipeline_v3.py
        dist_from_midline_cm = (self.width / 2.0) - DY
        edge_zone_2d = ((dist_from_midline_cm >= LINEAR_EDGE_ZONE_THRESHOLD_CM) |
                        (DX <= LINEAR_EDGE_ZONE_THRESHOLD_CM))
        self.edge_zone_flat = edge_zone_2d.ravel()

        # every bin of the nx*ny grid is on the track (no out-of-bounds corners to mask),
        # so all n_bins count toward the 80% coverage criterion (COVERAGE_FRACTION)
        self.total_arena_bins = self.n_bins
        self.coverage_threshold_bins = int(np.ceil(COVERAGE_FRACTION * self.total_arena_bins))

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

    def bin_centres_xy(self):
        """(2, n_bins) cm centre of every flat bin (length, width) (for kde_density_map)."""
        xs = (np.arange(self.nx) + 0.5) * self.bin_cm_x
        ys = (np.arange(self.ny) + 0.5) * self.bin_cm_y
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        return np.vstack([XX.ravel(), YY.ravel()])

    def bin_areas_cm2(self):
        return np.full(self.n_bins, self.bin_cm_x * self.bin_cm_y)

    def smooth(self, fr_flat, valid_flat):
        fr2 = fr_flat.reshape(self.nx, self.ny)
        v2  = valid_flat.reshape(self.nx, self.ny)
        return _gaussian_smooth_2d(fr2, v2).ravel()

    def connected_components(self, qualifies_flat):
        return _connected_components_2d_flat(qualifies_flat, self.nx, self.ny)

    def plot_fine(self, ax, values_flat, valid_flat, cmap, norm):
        grid = np.full(self.n_bins, np.nan)
        grid[valid_flat] = values_flat[valid_flat]
        grid2 = grid.reshape(self.nx, self.ny)
        im = ax.imshow(np.ma.masked_invalid(grid2.T), origin='lower',
                        extent=[0, self.length, 0, self.width],
                        aspect='equal', cmap=cmap, norm=norm)
        ax.axis('off')
        return im

    def plot_bins_2d(self, ax, values_flat, valid_flat, cmap, norm):
        """Flat (length x width) bin-grid heatmap with cm axes. Aspect is left free so the
        4 rows of width bins stay readable on an 80 x 8 cm track."""
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
    band: a bin qualifies if its SMOOTHED rate exceeds both the cell's mean firing rate
    (occupancy-weighted mean of the same smoothed map) and FIELD_PEAK_FRAC (20%) of the
    smoothed rate map's peak (mirroring auto_filter_band's `rmap > 0.2 * peak`), and only
    connected runs of qualifying bins spanning >= MIN_FIELD_BINS (9) contiguous bins are kept
    as a field (adapting auto_filter_band's area-based field radius, which has no direct
    meaning on a discrete bin grid, into a minimum-size connected-component criterion
    instead)."""
    valid     = cell['valid']
    fr_smooth = cell['fr_smooth']
    field_mask = np.zeros(handler.n_bins, dtype=bool)
    if not valid.any():
        return field_mask

    peak_smooth = float(fr_smooth[valid].max())
    mean_fr     = cell['mean_fr']
    qualifies = valid & (fr_smooth > mean_fr) & (fr_smooth > FIELD_PEAK_FRAC * peak_smooth)
    if not qualifies.any():
        return field_mask

    for component in handler.connected_components(qualifies):
        if len(component) >= MIN_FIELD_BINS:
            field_mask[component] = True
    return field_mask


def compute_cell_ratemap(x_cm: np.ndarray, y_cm: np.ndarray, t: np.ndarray,
                         spike_ts: np.ndarray, handler) -> dict | None:
    x_cm, y_cm = handler.orient(x_cm, y_cm)
    moving = _speed_mask(x_cm, y_cm, t)
    bin_idx, sample_valid = handler.to_bins(x_cm, y_cm)
    t = t[sample_valid]
    bin_idx = bin_idx[sample_valid]
    moving = moving[sample_valid]
    if len(t) < 2:
        return None

    # spikes are matched against every on-arena frame (not only the in-speed ones), so a spike
    # fired during immobility is matched to its own immobile frame and rejected, rather than
    # being re-credited to the nearest in-speed frame
    idx   = np.searchsorted(t, spike_ts, side='left')
    idx_l = np.clip(idx - 1, 0, len(t) - 1)
    idx_r = np.clip(idx,     0, len(t) - 1)
    dist_l  = np.abs(spike_ts - t[idx_l])
    dist_r  = np.abs(spike_ts - t[idx_r])
    nearest = np.where(dist_l <= dist_r, idx_l, idx_r)
    min_dist = np.minimum(dist_l, dist_r)

    matched     = min_dist <= MAX_GAP_US
    valid_spike = matched & moving[nearest]
    n_spikes    = int(valid_spike.sum())
    n_spikes_speed_excluded = int((matched & ~moving[nearest]).sum())

    n = len(t)
    dt_frames = np.empty(n, dtype=np.float64)
    dt_frames[0] = 1.0 / fps
    raw_dt = np.diff(t) * 1e-6
    dt_frames[1:] = np.minimum(raw_dt, 2.0 / fps)

    if SPEED_FILTER_OCCUPANCY:
        # drop out-of-speed frames entirely; spike frame indices are remapped onto the kept
        # frames so the bootstrap's circular shift also only moves spikes among in-speed frames
        new_index = np.cumsum(moving) - 1
        spike_frame = new_index[nearest[valid_spike]]
        t, bin_idx, dt_frames = t[moving], bin_idx[moving], dt_frames[moving]
        if len(t) < 2:
            return None
    else:
        spike_frame = nearest[valid_spike]

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

    result = dict(n_spikes=n_spikes, n_spikes_speed_excluded=n_spikes_speed_excluded, fr_raw=fr_raw, fr_smooth=fr_smooth, fi_map=fi_map,
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
    # _session_positions applies the open-field centring correction, so this screen sees
    # exactly the positions the rate maps will.
    x_cm, y_cm, t = _session_positions(csv_path, handler)
    if len(t) < 2:
        return None
    x_cm, y_cm = handler.orient(x_cm, y_cm)
    moving = _speed_mask(x_cm, y_cm, t)
    bin_idx, sample_valid = handler.to_bins(x_cm, y_cm)
    t = t[sample_valid]
    bin_idx = bin_idx[sample_valid]
    moving = moving[sample_valid]
    if len(t) < 2:
        return None

    n = len(t)
    dt_frames = np.empty(n, dtype=np.float64)
    dt_frames[0] = 1.0 / fps
    raw_dt = np.diff(t) * 1e-6
    dt_frames[1:] = np.minimum(raw_dt, 2.0 / fps)

    # same occupancy the rate maps use (velocity-filtered when SPEED_FILTER_OCCUPANCY)
    if SPEED_FILTER_OCCUPANCY:
        bin_idx, dt_frames = bin_idx[moving], dt_frames[moving]

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
    peak_bin_raw = None
    if cell['valid'].any():
        masked = np.where(cell['valid'], cell['fr_smooth'], -np.inf)
        peak_bin = int(np.argmax(masked))
        # peak of the RAW (unsmoothed) rate map, for the peak-proportion map
        masked_raw = np.where(cell['valid'], cell['fr_raw'], -np.inf)
        peak_bin_raw = int(np.argmax(masked_raw))

    field_mask = extract_place_field_mask(cell, handler)

    return dict(
        session=session_name, unit=ntt_file,
        n_spikes=cell['n_spikes'], n_spikes_speed_excluded=cell['n_spikes_speed_excluded'],
        peak_fr=cell['peak_fr'], mean_fr=cell['mean_fr'],
        sir=cell['sir'], sparsity=cell['sparsity'],
        bootstrap_sig=boot_sig,
        fr_raw=cell['fr_raw'], fr_smooth=cell['fr_smooth'], fi_map=cell['fi_map'], valid=cell['valid'], occ_map=cell['occ_map'],
        peak_bin=peak_bin, peak_bin_raw=peak_bin_raw, field_mask=field_mask,
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
    """Field-only mean field-index map: the overall mean map (pool_fine_map) with each cell's
    out-of-field background removed, per Mean_RM_Fixes.md ('Use only place fields to
    construct the mean RM'). Each cell contributes its 0-1 field index (fi_map) inside its own
    extracted place-field bins (extract_place_field_mask) and 0 in every other bin it
    sampled, so sub-threshold background firing adds nothing.

    Each bin's value is (sum of field index over the cells with a field there) / (number of
    cells that sampled that bin) -- the same per-bin denominator as pool_fine_map. This is
    algebraically (fraction of sampling cells with a field at this bin) * (mean field index
    of those fields), so field-overlap density still shows through, but the value is a
    per-bin average bounded to [0, 1]: it does not depend on how many cells were recorded,
    how large fields are elsewhere in the arena, or how many sessions happened to sample the
    bin, and can therefore be compared between conditions and against the overall map.

    Bins that are not part of a place field in ANY cell are set to NaN and marked invalid,
    so they are left empty in the plot instead of being drawn as 0."""
    place = results   # every unit is already a qualified place cell (see module docstring)
    if not place:
        return np.full(handler.n_bins, np.nan), np.zeros(handler.n_bins, dtype=bool)

    fi_sum   = np.zeros(handler.n_bins, dtype=np.float64)
    n_sampled = np.zeros(handler.n_bins, dtype=np.float64)
    n_field   = np.zeros(handler.n_bins, dtype=np.int64)
    for r in place:
        valid = r['valid']
        n_sampled[valid] += 1.0
        m = r['field_mask'] & valid
        fi_sum[m] += r['fi_map'][m]
        n_field[m] += 1

    # Bins that fall inside no cell's place field are NaN (left blank in the plot) rather
    # than 0, so the map only shows bins covered by at least one place field.
    mean_map = np.full(handler.n_bins, np.nan)
    in_any_field = (n_field > 0) & (n_sampled > 0)
    mean_map[in_any_field] = fi_sum[in_any_field] / n_sampled[in_any_field]
    return mean_map, in_any_field


def pool_peak_proportion_map(handler, results: list) -> tuple:
    """Peak-proportion map: each place cell contributes the single bin holding the peak of
    its RAW (unsmoothed) rate map (peak_bin_raw). Each bin's value is the percentage of all
    place cells in this arena whose raw peak falls in that bin -- e.g. 2 of 100 cells -> 2.
    Every bin visited in at least one session of this arena (valid for at least one cell) is
    shown; visited bins that hold no cell's peak get 0. Returns (pct_map, visited, n_cells)."""
    n_cells = len(results)
    visited = np.zeros(handler.n_bins, dtype=bool)
    counts  = np.zeros(handler.n_bins, dtype=np.float64)
    for r in results:
        visited |= r['valid']
        if r['peak_bin_raw'] is not None:
            counts[r['peak_bin_raw']] += 1.0

    pct_map = np.full(handler.n_bins, np.nan)
    if n_cells > 0:
        pct_map[visited] = counts[visited] / n_cells * 100.0
    return pct_map, visited, n_cells


# ============================================================================
# 2D kernel density estimate of the pooled maps (scipy gaussian_kde, Scott's rule)
# ============================================================================

def _kernel_mass_in_domain(eval_pts: np.ndarray, dom_pts: np.ndarray, dom_area: np.ndarray,
                           cov: np.ndarray) -> np.ndarray:
    """For each evaluation point p, the fraction of a Gaussian kernel N(p, cov) that falls
    inside the domain, approximated by summing kernel x bin area over the domain bins."""
    inv = np.linalg.inv(cov)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(cov)))
    d = eval_pts[:, :, None] - dom_pts[:, None, :]              # (2, n_eval, n_dom)
    q = np.einsum('imn,ij,jmn->mn', d, inv, d)
    return norm * (np.exp(-0.5 * q) @ dom_area)


def kde_density_map(handler, values_flat: np.ndarray, domain_flat: np.ndarray,
                    values_are_mass: bool = False) -> dict | None:
    """Fits a 2D Gaussian KDE (scipy.stats.gaussian_kde, bandwidth by KDE_BW_METHOD =
    Scott's rule) to a pooled flat-bin map and evaluates it at every domain bin.

    The map is treated as a spatial distribution of firing: the data points are the
    Cartesian cm centres of the bins (handler.bin_centres_xy), and each bin's weight is its
    firing MASS --
      * rate-like maps (overall / field-only mean field index; values_are_mass=False):
        map value x bin area, so the estimate is a density per cm^2 and is not skewed by the
        ring track's unequal bin areas;
      * count-like maps (peak proportion; values_are_mass=True): the map value itself (% of
        cells whose peak is in that bin), i.e. the KDE of the cells' peak locations.
    Bins with zero / NaN value carry no weight. With weights, scipy's Scott factor uses the
    effective sample size neff = (sum w)^2 / sum w^2 and the weighted covariance of the bin
    coordinates, so the kernel is anisotropic where the map is (e.g. along the linear track).

    domain_flat is the set of bins the animal sampled; the KDE is only evaluated (and, with
    KDE_EDGE_CORRECTION, renormalized) there -- every other bin is NaN.

    Returns dict with
      density     : edge-corrected KDE density (1/cm^2), NaN outside the domain
      density_raw : plain scipy KDE density (1/cm^2), NaN outside the domain
      scaled      : density rescaled back into the input map's units (field index, or % of
                    cells per bin), i.e. a KDE-smoothed version of the map itself
      factor, neff, cov (bandwidth covariance, cm^2), n_fit
    or None if the KDE cannot be fit (too few weighted bins / singular covariance)."""
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
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f'  [KDE skipped] could not fit KDE: {e}')
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

    # back into the input's units: a constant map c over the domain returns exactly c
    total = float(mass.sum())
    scaled = density * total * (area if values_are_mass else 1.0)

    return dict(density=density, density_raw=density_raw, scaled=scaled,
                factor=float(kde.factor), neff=float(kde.neff), cov=kde.covariance,
                n_fit=int(fit.sum()))


def _union_valid(handler, results: list) -> np.ndarray:
    visited = np.zeros(handler.n_bins, dtype=bool)
    for r in results:
        visited |= r['valid']
    return visited


# map kind -> (pooling function returning (values, values_are_mass), figure title, input units)
_KDE_MAP_KINDS = {
    'overall': (lambda h, res: (pool_fine_map(h, res)[0], False),
                'Overall mean field index map', 'field index'),
    'field_only': (lambda h, res: (pool_field_only_map(h, res)[0], False),
                   'Field-only mean field index map', 'field index'),
    'peak': (lambda h, res: (pool_peak_proportion_map(h, res)[0], True),
             'Peak proportion map', '% of cells per bin'),
}


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

        n_place = len(arena_results[key])
        ax.set_title(f'{_ARENA_TITLES[key]}\n(n={n_place} place cells)')
        fig.colorbar(pcm, ax=ax, shrink=0.7, label='Field index (a.u.)')

    fig.suptitle('Fig S1H -- Overall mean field index maps (place cells pooled across days)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_field_only_mean_maps(arena_handlers: dict, arena_results: dict, save_path: str):
    """Additional mean field-index maps with each cell's out-of-field background set to 0
    (pool_field_only_map): same per-bin averaging as Fig S1H, but only place-field bins
    contribute a non-zero field index. Values are 0-1 field index (a.u.), not raw Hz."""
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
        fig.colorbar(pcm, ax=ax, shrink=0.7, label='Field index, background = 0 (a.u.)')

    fig.suptitle('Field-only mean field-index maps (out-of-field bins set to 0; '
                 'bins in no cell\'s field left blank)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_peak_proportion_maps(arena_handlers: dict, arena_results: dict, save_path: str):
    """Heatmap of pool_peak_proportion_map per arena: % of place cells whose raw rate-map
    peak falls in each bin (0 in visited bins with no peak, blank in never-visited bins)."""
    fig = plt.figure(figsize=(15, 5))
    for i, key in enumerate(_ARENA_ORDER):
        handler = arena_handlers[key]
        pct_map, visited, n_cells = pool_peak_proportion_map(handler, arena_results[key])
        cmap, norm = make_cmap_norm(pct_map[visited])

        proj = 'polar' if key == 'circular_track' else None
        ax = fig.add_subplot(1, 3, i + 1, projection=proj)
        pcm = handler.plot_fine(ax, pct_map, visited, cmap, norm)

        ax.set_title(f'{_ARENA_TITLES[key]}\n(n={n_cells} place cells)')
        fig.colorbar(pcm, ax=ax, shrink=0.7, label='% of place cells with peak in bin')

    fig.suptitle('Peak proportion maps (raw rate-map peak bin per place cell)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_kde_maps(arena_handlers: dict, arena_results: dict, map_kind: str, save_path: str):
    """2D KDE (Scott's rule, kde_density_map) of one pooled map type ('overall',
    'field_only' or 'peak') per arena, plotted as density per cm^2 on each arena's flat 2D
    bin grid (handler.plot_bins_2d -- the circular track unrolled to arc-length x radial
    width). The numeric maps (edge-corrected density, raw density, and the density rescaled
    into the source map's units) plus each arena's bandwidth are saved next to the figure as
    an .npz."""
    pool_fn, title, units = _KDE_MAP_KINDS[map_kind]
    fig = plt.figure(figsize=(17, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.7])
    arena_axes = {'open_field':     fig.add_subplot(gs[:, 0]),
                  'circular_track': fig.add_subplot(gs[0, 1]),
                  'linear_track':   fig.add_subplot(gs[1, 1])}
    saved = {}
    for key in _ARENA_ORDER:
        handler = arena_handlers[key]
        results = arena_results[key]
        ax = arena_axes[key]

        values, values_are_mass = pool_fn(handler, results)
        domain = _union_valid(handler, results)
        kde_out = kde_density_map(handler, values, domain, values_are_mass) if domain.any() else None
        if kde_out is None:
            ax.set_title(f'{_ARENA_TITLES[key]}\n(no KDE: too few data)')
            ax.axis('off')
            continue

        density = kde_out['density']
        cmap, norm = make_cmap_norm(density[domain])
        pcm = handler.plot_bins_2d(ax, density, domain & np.isfinite(density), cmap, norm)
        sd = np.sqrt(np.diag(kde_out['cov']))
        ax.set_title(f'{_ARENA_TITLES[key]} (n={len(results)} place cells, '
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

    corr = 'edge-corrected' if KDE_EDGE_CORRECTION else 'no edge correction'
    fig.suptitle(f'{title} -- 2D Gaussian KDE (Scott\'s rule, {corr})')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')

    if saved:
        npz_path = os.path.splitext(save_path)[0] + '.npz'
        np.savez_compressed(npz_path, scaled_units=units, **saved)
        print(f'[SAVED] {npz_path}')


def _debug_occupancy_and_rawrate(x_cm: np.ndarray, y_cm: np.ndarray, t: np.ndarray,
                                  spike_ts: np.ndarray, handler) -> dict | None:
    """Same bin-assignment + spike-matching arithmetic as compute_cell_ratemap, but with
    NO min_occ_s / geom_valid masking -- a bin counts as 'visited' the moment occ_map > 0.
    This is deliberately more permissive than the real pipeline so raw bin-assignment bugs
    (gaps, duplicated wrap, off-center ring, wrong radial split) show up even in
    sparsely-sampled bins that compute_cell_ratemap would mask out as invalid."""
    x_cm, y_cm = handler.orient(x_cm, y_cm)
    moving = _speed_mask(x_cm, y_cm, t)
    bin_idx, sample_valid = handler.to_bins(x_cm, y_cm)
    t = t[sample_valid]
    bin_idx = bin_idx[sample_valid]
    moving = moving[sample_valid]
    if len(t) < 2:
        return None

    idx   = np.searchsorted(t, spike_ts, side='left')
    idx_l = np.clip(idx - 1, 0, len(t) - 1)
    idx_r = np.clip(idx,     0, len(t) - 1)
    dist_l  = np.abs(spike_ts - t[idx_l])
    dist_r  = np.abs(spike_ts - t[idx_r])
    nearest = np.where(dist_l <= dist_r, idx_l, idx_r)
    min_dist = np.minimum(dist_l, dist_r)
    valid_spike = (min_dist <= MAX_GAP_US) & moving[nearest]
    spike_frame = nearest[valid_spike]

    n = len(t)
    dt_frames = np.empty(n, dtype=np.float64)
    dt_frames[0] = 1.0 / fps
    dt_frames[1:] = np.minimum(np.diff(t) * 1e-6, 2.0 / fps)
    if SPEED_FILTER_OCCUPANCY:
        dt_frames[~moving] = 0.0

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
            # directly here used to skip the open-field centring, which made debug maps
            # disagree with their real counterparts
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
        for r in results:
            rows.append(dict(
                arena=key, session=r['session'], unit=r['unit'],
                n_spikes=r['n_spikes'], n_spikes_speed_excluded=r.get('n_spikes_speed_excluded'),
                peak_fr=r['peak_fr'], mean_fr=r['mean_fr'],
                sir=r['sir'], sparsity=r['sparsity'],
                bootstrap_sig=r['bootstrap_sig'],
            ))
    df = pd.DataFrame(rows)
    df.to_excel(out_path, index=False)
    print(f'[SAVED] {out_path}')


def run_circular_track_pipeline_test(cfg: dict, out_dir: str) -> None:
    """DEBUG: runs the FULL downstream pipeline (bootstrap, coverage filtering at
    whatever COVERAGE_FRACTION is currently set to, pooling, plotting,
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
        ax.set_title(f'Circular Track -- field-only mean field index (n={n_place} place cells)')
        fig.colorbar(pcm, ax=ax, shrink=0.7, label='Field index, background = 0 (a.u.)')
        fig.tight_layout()
        p = os.path.join(out_dir, 'CircularTrack_FieldOnlyMeanRate.png')
        fig.savefig(p, dpi=200)
        plt.close(fig)
        print(f'  [SAVED] {p}')
    else:
        print('  [SKIP] no valid bins in field-only map')

    # Peak proportion map (raw rate-map peak bin per place cell)
    pct_map, visited, n_cells = pool_peak_proportion_map(handler, results)
    if visited.any():
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1, projection='polar')
        cmap, norm = make_cmap_norm(pct_map[visited])
        pcm = handler.plot_fine(ax, pct_map, visited, cmap, norm)
        ax.set_title(f'Circular Track -- peak proportion (n={n_cells} place cells)')
        fig.colorbar(pcm, ax=ax, shrink=0.7, label='% of place cells with peak in bin')
        fig.tight_layout()
        p = os.path.join(out_dir, 'CircularTrack_PeakProportion.png')
        fig.savefig(p, dpi=200)
        plt.close(fig)
        print(f'  [SAVED] {p}')
    else:
        print('  [SKIP] no visited bins for peak proportion map')

    export_excel({'circular_track': handler}, {'circular_track': results},
                 os.path.join(out_dir, 'CircularTrack_Summary.xlsx'))


# ============================================================================
# __main__
# ============================================================================

def run_full_pipeline(out_dir: str) -> None:
    """Runs the full pipeline (bootstrap, coverage filtering at COVERAGE_FRACTION,
    place-cell qualification, pooling, plotting, excel export) for all
    three arenas (open_field, circular_track, linear_track), producing the combined
    Fig S1H / field-only / peak-proportion figures side by side across arenas."""
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
    plot_peak_proportion_maps(arena_handlers, arena_results,
                              os.path.join(out_dir, 'PeakProportion_Map.png'))
    plot_kde_maps(arena_handlers, arena_results, 'overall',
                  os.path.join(out_dir, 'KDE_FigS1H_MeanFieldIndex.png'))
    plot_kde_maps(arena_handlers, arena_results, 'field_only',
                  os.path.join(out_dir, 'KDE_FieldOnly_MeanFieldIndex.png'))
    plot_kde_maps(arena_handlers, arena_results, 'peak',
                  os.path.join(out_dir, 'KDE_PeakProportion_Map.png'))
    export_excel(arena_handlers, arena_results,
                 os.path.join(out_dir, 'AllArenas_Summary.xlsx'))


if __name__ == '__main__':
    print(f"Using '{COORD_UNITS}' tracking coordinates.\n")

    run_full_pipeline(os.path.join(OUTPUT_DIR, 'AllArenas'))

    print('\nDone.')
