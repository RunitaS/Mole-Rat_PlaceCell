# -*- coding: utf-8 -*-
"""
Mean Rate Map Analysis (after Muessig et al.) + KDE + occupancy-matched spike null + Duong (2013)

PIPELINE (run_full_pipeline), per arena:
  1. MEAN MAPS -- the pooled maps of the recorded place cells (described below): overall mean
     field index, field-only mean field index and peak proportion.
  2. KDE -- a 2D Gaussian KDE (scipy gaussian_kde, Scott's rule, edge-corrected over the bins the
     cells validly sampled) of each of the three pooled maps.
  3. OBSERVED NULL -- occupancy-matched spike null run through the full pipeline (described in
     detail directly below): N_NULL_ITER null population maps and null KDEs per map kind, Monte
     Carlo statistics (2D divergence / distance, enrichment and z maps with a global envelope,
     distance-to-wall projection) of every real KDE against them.
  4. DUONG TEST -- local significant differences between each real KDE and the null KDE of
     stage 3 (Duong 2013, "Local significant differences from nonparametric two-sample tests",
     J. Nonparametric Statistics 25:3, 635-645); see the DUONG section below. Clusters where the
     real density is significantly above the null are drawn in GREEN_ABOVE, below in
     MAGENTA_BELOW.

=================================================================================================
OBSERVED NULL -- OCCUPANCY-MATCHED SPIKE NULL RUN THROUGH THE FULL PIPELINE (stage 3, run_obs_null)
=================================================================================================
Question asked: what would the three pooled maps (and their KDEs) look like if every cell fired at
a CONSTANT rate, so that the only spatial structure left is what each cell's real trajectory and
this pipeline itself produce? The pipeline creates structure on its own: low-dwell bins (often
next to walls) give noisy rates, so the raw-map peak tends to land there; the 0-1 field-index
normalisation, the field threshold and smoothing all interact with the occupancy. A comparison of
the real maps with the combined occupancy density alone would not reveal these biases; this null
reproduces them, because the null cells go through exactly the same code as the real cells.

Step 0 -- combined occupancy (combined_occupancy; figure ObsNull_CombinedOccupancy.png)
  * Every session's occupancy map is the dwell time per bin (s) its cells' rate maps were divided
    by (speed-filtered frames when SPEED_FILTER_OCCUPANCY). Sessions are already in one common
    arena frame: open-field tracking is centred on the arena (CENTRE_OPEN_FIELD_TRACKING),
    vertical linear-track sessions are rotated, the circular track is binned in arc x radius.
  * Each session's map is divided by its own total -> a dwell PROBABILITY map, so long sessions do
    not dominate.
  * Sessions are averaged with weight 1 each (OCC_SESSION_WEIGHTING = 'equal') or with the number
    of cells the session contributes ('cells', so the occupancy matches the cell sample).
  * The same 2D KDE as the real maps (kde_density_map: Scott's rule, edge-corrected, evaluated over
    the bins the cells validly sampled) is fitted to this combined dwell map -> p_occ(x). p_occ is
    shown for reference and, in the hybrid mode only, sets the spike weights of step 1.

Step 1 -- one null cell per real cell (simulate_null_cell)
  * Real trajectory kept: the null cell uses the real cell's own occupancy map (its own session's
    frames, binned and speed-filtered exactly as for its real rate map).
  * Real spike count kept: n = the number of spikes that entered the real cell's rate map (matched
    to a tracking frame within MAX_GAP_US and passing the speed filter).
  * Spikes redistributed uniformly in time (NULL_SPIKE_MODE = 'uniform', null (b): constant rate):
    each of the n spikes is assigned independently to a tracking frame f with probability
    dt_f / T, T = summed dt of the frames on which a spike can be counted (speed-passing frames),
    i.e. a constant firing rate n / T across the session. Only a spike's BIN enters the rate map and
    the frames of one bin are interchangeable, so this is drawn exactly (not approximated) as one
    multinomial draw of the n spikes over bins with p_b = (time spent in bin b) / T
    (spike_occ_map, recorded per cell by compute_cell_ratemap).
    NULL_SPIKE_MODE = 'occupancy' (hybrid null (a)): frame weight dt_f * p_occ(bin of f), i.e.
    p_b proportional to (time in bin b) x p_occ(b). The expected rate then follows the combined
    occupancy density ("fields sit where the animals spend time"), while the sampling noise is
    still each cell's own. Bins outside the p_occ domain (invalid in every session) get weight 0.
  * Identical pipeline, same functions as the real cells (nothing re-implemented):
    rate_maps_from_counts -- min_occ_s dwell threshold + arena mask, rate = spikes / dwell,
    RATEMAP_SMOOTH_SIGMA_BINS Gaussian smoothing, 0-1 field-index map, occupancy-weighted mean
    rate; then _cell_peaks_and_field -- raw-map peak bin and the place-field mask
    (> mean rate, > FIELD_PEAK_FRAC x peak, >= MIN_FIELD_BINS connected bins).

Step 2 -- one null population = one iteration (_null_iterations)
  * One null cell for every real cell of the arena (same sessions, same cells), pooled with the
    same functions as the real data: pool_fine_map (overall mean field index), pool_field_only_map
    (field-only mean field index) and pool_peak_proportion_map (% of cells with their peak per bin).
  * Each null population map is fitted with the same KDE as its real counterpart (kde_density_map:
    bandwidth by Scott's rule on that map, same edge correction, same domain = bins the cells
    validly sampled, which is identical to the real domain because validity depends on occupancy
    only).
  * Every iteration has its own random generator seeded with (NULL_SEED, arena, iteration), so the
    result does not depend on how the iterations are split over NULL_MAX_WORKERS processes.

Step 3 -- repeat N_NULL_ITER (1000) times -> per (map kind, arena) N null population maps and N null
  KDEs. Iterations whose KDE cannot be fitted (fewer than 3 bins with mass, possible for the
  field-only map) are dropped and the number used is reported.

Step 4 -- statistics, real KDE vs null KDEs (obs_null_compare), per map kind x arena
  * Every KDE becomes a mass distribution over the common domain (bins where the real KDE and all
    null KDEs are defined): g(x) = f(x) area(x) / sum_x f(x) area(x). The null reference is the
    mean null distribution r(x) = mean_i g_i(x). Each null iteration i is compared with the
    leave-one-out reference r_-i (mean of the other N-1 iterations, sd likewise), so no null is
    ever compared with a reference that contains itself.
  * Method 1 statistics (2D):
      JSD_bits          Jensen-Shannon divergence (bits) between g and r.
      SlicedW1_cm       Wasserstein-1 / earth mover's distance between g and r, as the sliced
                        Wasserstein distance: the mean over N_SLICED_DIRECTIONS projection
                        directions (room x, y) of the exact 1D W1 of the projected distributions.
                        (Exact 2D optimal transport over ~10^3 bins x 1000 iterations x 9 maps is
                        too slow; the sliced distance is a proper metric in the same units, cm.)
      Enrichment map    log2(g(x) / r(x)).
      z map             z(x) = (g(x) - r(x)) / sd_null(x).
      MaxAbsZ           max_x |z(x)| -- the global-envelope statistic.
    Global envelope (max statistic, Westfall-Young): c = (1 - NULL_ALPHA) quantile of the null
    MaxAbsZ_i. Bins with z > c (z < -c) are significantly ABOVE (BELOW) the null with family-wise
    error <= NULL_ALPHA over all bins together; they are split into 8-connected clusters (wrapping
    on the circular track) and outlined GREEN_ABOVE / MAGENTA_BELOW, as for the Duong test. The
    uncorrected pointwise p(x) = (1 + #{|z_i(x)| >= |z(x)|}) / (N + 1) is saved too.
  * Method 2 statistics (projection on distance to the nearest wall, handler.dist_to_wall):
      MeanDist2Wall_cm  mass-weighted mean distance to the wall.
      EdgeZoneFrac      fraction of the mass in the edge zone (handler.edge_zone_flat: open field
                        <= OPEN_EDGE_ZONE_THRESHOLD_CM from the wall; circular track outer half of
                        the track width; linear track ends and side strips, see
                        LINEAR_EDGE_ZONE_THRESHOLD_CM).
      Dist2Wall_W1_cm   1D Wasserstein distance between the distance-to-wall distributions of g
                        and r.
      Distance profile  mass per DIST_PROFILE_BIN_CM distance band, real vs the null mean with its
                        2.5-97.5 % pointwise envelope. The 1D distributions are read off the 2D
                        KDEs (not from a second, 1D KDE) so real and null share one smoothing.
  * Monte Carlo p-values with N valid iterations:
      distances (JSD_bits, SlicedW1_cm, Dist2Wall_W1_cm, MaxAbsZ), one-sided:
          p = (1 + #{T_i >= T_obs}) / (N + 1)
      directional statistics (MeanDist2Wall_cm, EdgeZoneFrac), two-sided:
          p = min(1, 2 min(p_upper, p_lower)), p_upper = (1 + #{s_i >= s_obs}) / (N + 1),
          p_lower = (1 + #{s_i <= s_obs}) / (N + 1); the sign of (observed - null mean) gives the
          direction.

Step 5 -- hand-off to the Duong test (stage 4): the null KDE of each map kind is the same KDE
  fitted to the MEAN null population map (average of the N null population maps, NaN -> 0 inside
  the domain). With DUONG_SAMPLE_SIZE_MODE = 'cells', n2 = (cells) x (N iterations).

What this null preserves: each cell's trajectory, dwell times, speed filtering, spike count, and
the session structure of the pooled sample (cells recorded together share their occupancy).
What it removes: all spatial tuning (uniform mode) or all tuning other than occupancy (hybrid),
and the temporal structure of the spike train -- null spikes are independent (Poisson-like), so
bursts are not reproduced and the null rate maps are somewhat less noisy than bursty real cells
with the same spike count would be.

Runtime: N_NULL_ITER x (cells) simulated rate maps plus N_NULL_ITER x 3 KDE fits per arena; the
iterations run in NULL_MAX_WORKERS processes (falls back to serial if a process pool cannot start,
e.g. in some interactive consoles). Set RUN_OBS_NULL = False to skip stages 3 and 4.
=================================================================================================

Fig S1H: overall mean field-index map per arena, fine spatial bins (2 x 2 cm; genuinely 2D for
every arena, including the linear track's length x width). Each cell's map is first normalized
0-1 to its own peak and minimum (smoothed) firing rate (field_index_map, place-cell method, as in
ThetaMod_PhasePrecession_Stats_v3.py) before pooling across cells -- this keeps a cell's
contribution from being over- or under-weighted by its absolute peak rate or by where its field
happens to sit in the map, which raw-Hz pooling does not.

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
_speed_mask, _session_frames, compute_cell_ratemap).

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

Outputs (OUTPUT_DIR/AllArenas):
  Mean maps      WholeRM_MeanFieldIndex.png, FieldOnly_MeanFieldIndex.png, PeakProportion_Map.png,
                 AllArenas_Summary.xlsx
  KDE            KDE_<stem>.png / .npz, stem = FigS1H_MeanFieldIndex, FieldOnly_MeanFieldIndex,
                 PeakProportion_Map
  Observed null  ObsNull_CombinedOccupancy.png
                 ObsNull_<kind>.png      rows = arenas; real pooled map, mean null pooled map,
                                         real KDE, mean null KDE, log2 enrichment, z map with
                                         global-envelope clusters
                 ObsNull_Statistics.png  null distribution of every statistic, real value in red
                 ObsNull_DistToWall.png  distance-to-wall profiles, real vs null 95 % envelope
                 ObsNull_Stats.xlsx      sheets Summary, GlobalEnvelope, Clusters, NullIterations
                 ObsNull_Results.npz     null maps / KDEs per iteration, z, p, enrichment, masks
  Duong test     DuongTest_<kind>.png, DuongTest_Clusters.xlsx, DuongTest_Results.npz
"""

import os
import time
import hashlib
import random
import concurrent.futures
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import chi2, gaussian_kde, norm as normal_dist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm, to_rgba
from matplotlib.lines import Line2D

# ============================================================================
# CONFIGURATION -- edit root folder + geometry below
# ============================================================================

ROOT_DIRECTORY = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\PC_True_irSparADptBin_Corrected'
OUTPUT_DIR = os.path.join(ROOT_DIRECTORY, 'MeanRM_Quad_ObsShuffle_v27')

USE_BIN_COVERAGE_CRITERION = False
COVERAGE_FRACTION = 0.01

USE_ZONE_COVERAGE_CRITERION = False
MIN_ZONE_COVERAGE_PCT       = 0.1

ARENA_FOLDER_KEYWORDS = {
    'open':     'open_field',
    'linear':   'linear_track',
    'circle':   'circular_track',
    'circular': 'circular_track',
}

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
    for part in os.path.normpath(dirpath).split(os.sep):
        arena_key = ARENA_FOLDER_KEYWORDS.get(part.strip().lower())
        if arena_key is not None:
            return arena_key
    return None

CENTRE_OPEN_FIELD_TRACKING = True

def _centre_open_field_tracking(x_cm: np.ndarray, y_cm: np.ndarray, session_dir: str,
                                handler) -> tuple:
    if not CENTRE_OPEN_FIELD_TRACKING:
        return x_cm, y_cm
    if _detect_arena_key(session_dir) != 'open_field' or len(x_cm) == 0:
        return x_cm, y_cm

    mid_x = 0.5 * (float(x_cm.min()) + float(x_cm.max()))
    mid_y = 0.5 * (float(y_cm.min()) + float(y_cm.max()))
    return x_cm + (handler.cx - mid_x), y_cm + (handler.cy - mid_y)


fps           = 30
target_bin_cm  = 2.0
min_occ_s      = 1

OPEN_EDGE_ZONE_THRESHOLD_CM   = 9.25
LINEAR_EDGE_ZONE_THRESHOLD_CM = 2.0

MAX_GAP_US     = 50_000
N_BOOTSTRAP    = 1000
MAX_WORKERS    = 4
BOOTSTRAP_SEED = 0

POS_JUMP_THRESH_CMS  = 90.0
POS_SMOOTH_SIGMA_SMP = 5.0

SPEED_FILTER_ENABLED = True
MIN_SPEED_CMS        = 0.5
MAX_SPEED_CMS        = 90.0
SPEED_FILTER_OCCUPANCY = True
RATEMAP_SMOOTH_SIGMA_BINS = 3.0

FIELD_PEAK_FRAC = 0.20
MIN_FIELD_BINS  = 9

KDE_BW_METHOD = 'scott'
KDE_EDGE_CORRECTION = True

COORD_UNITS = 'cm'

# --- Observed occupancy-matched spike null (stage 3; see the header) ---
RUN_OBS_NULL          = True
N_NULL_ITER           = 1000       # null populations (iterations) per arena
NULL_SPIKE_MODE       = 'uniform'  # 'uniform': constant rate in time (null b); 'occupancy': hybrid,
                                   #   spike weight per frame = dt x combined-occupancy KDE (null a)
OCC_SESSION_WEIGHTING = 'equal'    # combined occupancy: 'equal' per session, or 'cells' per session
NULL_SEED             = 20261005
NULL_MAX_WORKERS      = 4          # worker processes for the null iterations (1 = serial)
NULL_ALPHA            = 0.05       # family-wise level of the global (max |z|) envelope
N_SLICED_DIRECTIONS   = 36         # projection directions of the sliced Wasserstein distance
DIST_PROFILE_BIN_CM   = 2.0        # distance-to-wall band width of the Method 2 profile

# --- Duong (2013) local test, real KDE vs null KDE (stage 4) ---
DUONG_ALPHA            = 0.05     # family-wise level per (map kind, arena), Hochberg step-up
DUONG_SAMPLE_SIZE_MODE = 'neff'   # 'neff' or 'cells' (see the DUONG section)
DUONG_RENORMALISE_ON_COMMON_DOMAIN = True
DUONG_MIN_CLUSTER_BINS = 1        # clusters smaller than this are not outlined / reported
GREEN_ABOVE   = '#39FF14'         # fluorescent green: real significantly ABOVE null
MAGENTA_BELOW = '#FF00FF'         # fluorescent magenta: real significantly BELOW null
OUTLINE_LW    = 2.2
R_K_GAUSS_2D  = 1.0 / (4.0 * np.pi)   # R(K) = integral of K^2 for the standard bivariate normal

ntt_dtype = np.dtype([
    ('timestamp',   '<u8'),
    ('sc_number',   '<u4'),
    ('cell_number', '<u4'),
    ('params',      '<u4', (8,)),
    ('waveforms',   '<i2', (32, 4)),
])


# ============================================================================
# Tracking load / clean / smooth
# ============================================================================

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
    x_cm, y_cm, t = _load_tracking(csv_path, handler.arena_width_cm)
    if len(t) < 2:
        return x_cm, y_cm, t
    x_cm, y_cm = _smooth_tracking_position(x_cm, y_cm, t)
    session_dir = os.path.dirname(csv_path)
    x_cm, y_cm = _centre_open_field_tracking(x_cm, y_cm, session_dir, handler)
    return x_cm, y_cm, t


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


def _session_frames(csv_path: str, handler) -> dict | None:
    """One session's on-arena tracking frames: oriented position, flat bin, dwell time dt (s),
    the speed-filter flag `moving` and `occ_frame` (frames that count toward occupancy). Shared by
    the rate maps and the zone-coverage criterion, so both bin and time-weight the trajectory
    identically. None if the tracking is unusable."""
    x_cm, y_cm, t = _session_positions(csv_path, handler)
    if len(t) < 2:
        return None
    x_cm, y_cm = handler.orient(x_cm, y_cm)
    moving = _speed_mask(x_cm, y_cm, t)
    bin_idx, sample_valid = handler.to_bins(x_cm, y_cm)
    x_cm, y_cm, t = x_cm[sample_valid], y_cm[sample_valid], t[sample_valid]
    bin_idx, moving = bin_idx[sample_valid], moving[sample_valid]
    if len(t) < 2:
        return None

    dt_frames = np.empty(len(t), dtype=np.float64)
    dt_frames[0] = 1.0 / fps
    dt_frames[1:] = np.minimum(np.diff(t) * 1e-6, 2.0 / fps)
    occ_frame = moving if SPEED_FILTER_OCCUPANCY else np.ones(len(t), dtype=bool)
    return dict(x=x_cm, y=y_cm, t=t, bin_idx=bin_idx, dt=dt_frames, moving=moving,
                occ_frame=occ_frame)


def _stable_seed(*parts) -> int:
    key = '|'.join(str(p) for p in parts).encode('utf-8')
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'big') ^ BOOTSTRAP_SEED


# ============================================================================
# Gaussian smoothing kernel
# ============================================================================

def _gaussian_smooth_2d(fr_map: np.ndarray, valid_mask: np.ndarray,
                         sigma: float = RATEMAP_SMOOTH_SIGMA_BINS,
                         wrap_x: bool = False) -> np.ndarray:
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
# ============================================================================

class OpenFieldHandler:
    wrap_x = False

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

        self.dist_to_wall = np.clip(self.diameter / 2.0 - r, 0.0, None).ravel()
        self.edge_zone_flat = self.dist_to_wall <= OPEN_EDGE_ZONE_THRESHOLD_CM

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
        ax.add_patch(matplotlib.patches.Circle((self.cx, self.cy), self.diameter / 2.0,
                                               fill=False, edgecolor='0.35', lw=1.0, zorder=5))
        ax.set_aspect('equal')
        ax.axis('off')
        return im

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

        circumference = 2 * np.pi * self.mean_r
        self.circumference_cm = circumference
        self.nx = max(8, 4 * int(round(circumference / bin_cm / 4.0)))
        self.ny = max(2, int(round(self.track_width_cm / bin_cm)))
        self.bin_width_deg = 360.0 / self.nx
        self.bin_cm_y = self.track_width_cm / self.ny
        self.n_bins = self.nx * self.ny
        self.geom_valid = np.ones(self.n_bins, dtype=bool)

        r_from_inner = (np.arange(self.ny) + 0.5) * self.bin_cm_y
        self.dist_to_wall = np.tile(np.minimum(r_from_inner, self.track_width_cm - r_from_inner),
                                    self.nx)

        self.inner_side_flat = np.tile(np.arange(self.ny) < (self.ny // 2), self.nx)
        self.edge_zone_flat = ~self.inner_side_flat

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

    def __init__(self, cfg: dict, bin_cm: float):
        self.length = cfg['length_cm']
        self.width  = cfg['width_cm']
        self.nx = max(4, int(round(self.length / bin_cm)))
        self.ny = max(2, int(round(self.width  / bin_cm)))
        self.bin_cm_x = self.length / self.nx
        self.bin_cm_y = self.width  / self.ny
        self.n_bins = self.nx * self.ny
        self.arena_width_cm = self.length
        self.geom_valid = np.ones(self.n_bins, dtype=bool)

        bx_centers_cm = (np.arange(self.nx) + 0.5) * self.bin_cm_x
        by_centers_cm = (np.arange(self.ny) + 0.5) * self.bin_cm_y
        dist_x = np.minimum(bx_centers_cm, self.length - bx_centers_cm)
        dist_y = np.minimum(by_centers_cm, self.width - by_centers_cm)
        DX, DY = np.meshgrid(dist_x, dist_y, indexing='ij')
        self.dist_to_wall = np.minimum(DX, DY).ravel()

        dist_from_midline_cm = (self.width / 2.0) - DY
        edge_zone_2d = ((dist_from_midline_cm >= LINEAR_EDGE_ZONE_THRESHOLD_CM) |
                        (DX <= LINEAR_EDGE_ZONE_THRESHOLD_CM))
        self.edge_zone_flat = edge_zone_2d.ravel()

        self.total_arena_bins = self.n_bins
        self.coverage_threshold_bins = int(np.ceil(COVERAGE_FRACTION * self.total_arena_bins))

    def orient(self, x_cm, y_cm):
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
# Core per-cell rate map
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


def extract_place_field_mask(cell: dict, handler) -> np.ndarray:
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


def rate_maps_from_counts(spike_map: np.ndarray, occ_map: np.ndarray, handler) -> dict:
    """Rate map, smoothed rate map, field-index map and SIR / sparsity from per-bin spike counts
    and occupancy (s). Bins with < min_occ_s occupancy or outside the arena are invalid."""
    n_bins = handler.n_bins
    valid = (occ_map >= min_occ_s) & handler.geom_valid

    fr_raw = np.zeros(n_bins, dtype=np.float64)
    fr_raw[valid] = spike_map[valid] / occ_map[valid]
    fr_smooth = handler.smooth(fr_raw, valid)
    fi_map = field_index_map(fr_smooth, valid)

    result = dict(fr_raw=fr_raw, fr_smooth=fr_smooth, fi_map=fi_map, occ_map=occ_map, valid=valid)
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


def _cell_peaks_and_field(cell: dict, handler) -> dict:
    """Peak bins of the smoothed and raw maps, and the place-field mask, as pooled below."""
    peak_bin = None
    peak_bin_raw = None
    if cell['valid'].any():
        peak_bin = int(np.argmax(np.where(cell['valid'], cell['fr_smooth'], -np.inf)))
        peak_bin_raw = int(np.argmax(np.where(cell['valid'], cell['fr_raw'], -np.inf)))
    return dict(peak_bin=peak_bin, peak_bin_raw=peak_bin_raw,
                field_mask=extract_place_field_mask(cell, handler))


def compute_cell_ratemap(frames: dict, spike_ts: np.ndarray, handler) -> dict | None:
    t, bin_idx, moving, dt_frames = frames['t'], frames['bin_idx'], frames['moving'], frames['dt']

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

    if SPEED_FILTER_OCCUPANCY:
        new_index = np.cumsum(moving) - 1
        spike_frame = new_index[nearest[valid_spike]]
        t, bin_idx, dt_frames = t[moving], bin_idx[moving], dt_frames[moving]
        if len(t) < 2:
            return None
        spike_ok = np.ones(len(t), dtype=bool)
    else:
        spike_frame = nearest[valid_spike]
        spike_ok = moving

    n_bins = handler.n_bins
    occ_map   = np.zeros(n_bins, dtype=np.float64)
    spike_map = np.zeros(n_bins, dtype=np.float64)
    np.add.at(occ_map,   bin_idx, dt_frames)
    np.add.at(spike_map, bin_idx[spike_frame], 1.0)

    # time per bin on the frames a spike can be counted on (speed-passing frames): the
    # constant-rate spike distribution of the occupancy-matched null (stage 3)
    spike_occ_map = np.zeros(n_bins, dtype=np.float64)
    np.add.at(spike_occ_map, bin_idx[spike_ok], dt_frames[spike_ok])

    result = rate_maps_from_counts(spike_map, occ_map, handler)
    result.update(n_spikes=n_spikes, n_spikes_speed_excluded=n_spikes_speed_excluded,
                  bin_idx=bin_idx, spike_frame=spike_frame, t=t, n_bins=n_bins,
                  spike_occ_map=spike_occ_map)
    return result


def run_bootstrap_generic(handler, cell: dict, real_sir: float, n_bootstrap: int = N_BOOTSTRAP,
                          seed: int | None = None) -> dict:
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
    total_occ_s = float(occ_map[valid_mask].sum())
    if total_occ_s <= 0:
        return 0.0, 0.0
    edge_mask    = handler.edge_zone_flat & valid_mask
    edge_occ_s   = float(occ_map[edge_mask].sum())
    centre_occ_s = total_occ_s - edge_occ_s
    return 100.0 * edge_occ_s / total_occ_s, 100.0 * centre_occ_s / total_occ_s


def session_zone_coverage(csv_path: str, handler) -> tuple | None:
    frames = _session_frames(csv_path, handler)
    if frames is None:
        return None
    occ = frames['occ_frame']
    occ_map = np.zeros(handler.n_bins, dtype=np.float64)
    np.add.at(occ_map, frames['bin_idx'][occ], frames['dt'][occ])
    valid_mask = (occ_map >= min_occ_s) & handler.geom_valid
    return _zone_occupancy_pct(handler, occ_map, valid_mask)


def process_unit(csv_path: str, ntt_path: str, ntt_file: str, session_name: str, handler) -> dict | None:
    frames = _session_frames(csv_path, handler)
    if frames is None:
        return None

    spike_data = np.memmap(ntt_path, dtype=ntt_dtype, mode='r', offset=16 * 1024)
    spike_data = spike_data[spike_data['cell_number'] != 0]
    spike_ts = np.sort(spike_data['timestamp'].astype(np.float64))
    if len(spike_ts) == 0:
        return None

    cell = compute_cell_ratemap(frames, spike_ts, handler)
    if cell is None:
        return None

    covered_bins = int(cell['valid'].sum())
    if USE_BIN_COVERAGE_CRITERION and covered_bins < handler.coverage_threshold_bins:
        return None

    boot = run_bootstrap_generic(handler, cell, cell['sir'],
                                 seed=_stable_seed(session_name, ntt_file))

    return dict(
        session=session_name, unit=ntt_file,
        n_spikes=cell['n_spikes'], n_spikes_speed_excluded=cell['n_spikes_speed_excluded'],
        peak_fr=cell['peak_fr'], mean_fr=cell['mean_fr'],
        sir=cell['sir'], sparsity=cell['sparsity'],
        bootstrap_sig=boot.get('bootstrap_sig'),
        fr_raw=cell['fr_raw'], fr_smooth=cell['fr_smooth'], fi_map=cell['fi_map'], valid=cell['valid'], occ_map=cell['occ_map'],
        spike_occ_map=cell['spike_occ_map'],
        **_cell_peaks_and_field(cell, handler),
    )


# ============================================================================
# Session discovery + batch scan per arena
# ============================================================================

def find_arena_sessions(arena_key: str, handler) -> list:
    """(session_name, dirpath, csv_path, ntt_files) for every folder of this arena holding exactly
    one tracking file and at least one .ntt file (and, with USE_ZONE_COVERAGE_CRITERION, passing
    the zone-coverage criterion)."""
    out_dir = os.path.normcase(os.path.abspath(OUTPUT_DIR))
    sessions = []
    for dirpath, dirnames, filenames in os.walk(ROOT_DIRECTORY):
        dirnames[:] = [d for d in dirnames
                       if os.path.normcase(os.path.abspath(os.path.join(dirpath, d))) != out_dir]
        if _detect_arena_key(dirpath) != arena_key:
            continue
        tracking_files_all = [f for f in filenames if f.lower().endswith(('.csv', '.xlsx'))]
        if COORD_UNITS == 'cm':
            tracking_files = [f for f in tracking_files_all if f.lower().endswith('_cm.csv')]
        else:
            tracking_files = [f for f in tracking_files_all if not f.lower().endswith('_cm.csv')]
        ntt_files = sorted(f for f in filenames if f.lower().endswith('.ntt'))
        if len(tracking_files) != 1 or not ntt_files:
            continue
        csv_path = os.path.join(dirpath, tracking_files[0])

        if USE_ZONE_COVERAGE_CRITERION:
            try:
                zone_cov = session_zone_coverage(csv_path, handler)
            except Exception:
                zone_cov = None
            if zone_cov is None:
                continue
            edge_pct, centre_pct = zone_cov
            if edge_pct < MIN_ZONE_COVERAGE_PCT or centre_pct < MIN_ZONE_COVERAGE_PCT:
                continue

        sessions.append((os.path.relpath(dirpath, ROOT_DIRECTORY), dirpath, csv_path, ntt_files))
    return sorted(sessions)


def collect_arena_results(handler, sessions: list) -> list:
    jobs = [(session_name, dirpath, csv_path, ntt_file)
            for session_name, dirpath, csv_path, ntt_files in sessions for ntt_file in ntt_files]

    def _job(args):
        session_name, dirpath, csv_path, ntt_file = args
        ntt_path = os.path.join(dirpath, ntt_file)
        try:
            return process_unit(csv_path, ntt_path, ntt_file, session_name, handler)
        except Exception as e:
            return None

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for r in executor.map(_job, jobs):
            if r is not None:
                results.append(r)
    return results


# ============================================================================
# Pooling across cells / days
# ============================================================================

def pool_fine_map(handler, results: list) -> tuple:
    place = results
    if not place:
        return np.full(handler.n_bins, np.nan), np.zeros(handler.n_bins, dtype=bool)
    stack = np.full((len(place), handler.n_bins), np.nan)
    for i, r in enumerate(place):
        stack[i, r['valid']] = r['fi_map'][r['valid']]
    with warnings.catch_warnings():   # bins no cell sampled are all-NaN -> NaN, as intended
        warnings.simplefilter('ignore', RuntimeWarning)
        mean_map = np.nanmean(stack, axis=0)
    any_valid = ~np.all(np.isnan(stack), axis=0)
    return mean_map, any_valid


def pool_field_only_map(handler, results: list) -> tuple:
    place = results
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
        if r['peak_bin_raw'] is not None:
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


# map kind -> (pool fn -> (values, values_are_mass), title, KDE 'scaled' units, map colourbar
#              label, file stem: KDE figure = KDE_<stem>)
_KDE_MAP_KINDS = {
    'overall': (lambda h, res: (pool_fine_map(h, res)[0], False),
                'Overall mean field index map', 'field index', 'Field index (a.u.)',
                'FigS1H_MeanFieldIndex'),
    'field_only': (lambda h, res: (pool_field_only_map(h, res)[0], False),
                   'Field-only mean field index map', 'field index',
                   'Field index, background = 0 (a.u.)', 'FieldOnly_MeanFieldIndex'),
    'peak': (lambda h, res: (pool_peak_proportion_map(h, res)[0], True),
             'Peak proportion map', '% of cells per bin', '% of cells with peak in bin',
             'PeakProportion_Map'),
}


# ============================================================================
# STAGE 2 -- 2D kernel density estimate of the pooled maps (scipy gaussian_kde, Scott's rule)
# ============================================================================

def _kernel_mass_in_domain(eval_pts: np.ndarray, dom_pts: np.ndarray, dom_area: np.ndarray,
                           cov: np.ndarray) -> np.ndarray:
    """m(x) = sum over domain bins y of N(x - y; 0, cov) * area(y): the share of a Gaussian kernel
    of covariance cov centred at x that falls inside the domain (Riemann sum over bins)."""
    inv = np.linalg.inv(cov)
    norm_factor = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(cov)))
    d = eval_pts[:, :, None] - dom_pts[:, None, :]
    q = np.einsum('imn,ij,jmn->mn', d, inv, d)
    return norm_factor * (np.exp(-0.5 * q) @ dom_area)


def kde_density_map(handler, values_flat: np.ndarray, domain_flat: np.ndarray,
                    values_are_mass: bool = False) -> dict | None:
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
                factor=float(kde.factor), neff=float(kde.neff), cov=kde.covariance,
                domain=domain_flat)


def arena_kde(handler, results: list, map_kind: str) -> dict | None:
    """KDE of one pooled map over the bins the cells validly sampled (None if too few data)."""
    if not results:
        return None
    values, values_are_mass = _KDE_MAP_KINDS[map_kind][0](handler, results)
    domain = _union_valid(handler, results)
    return kde_density_map(handler, values, domain, values_are_mass) if domain.any() else None


# ============================================================================
# STAGE 3 -- Observed occupancy-matched spike null (Method 4; full description in the header)
# ============================================================================

# statistic -> test: 'upper' = one-sided (distances to the null), 'two-sided' = directional
_NULL_SCALAR_STATS = {
    'JSD_bits':         'upper',
    'SlicedW1_cm':      'upper',
    'MaxAbsZ':          'upper',
    'Dist2Wall_W1_cm':  'upper',
    'MeanDist2Wall_cm': 'two-sided',
    'EdgeZoneFrac':     'two-sided',
}


def combined_occupancy(handler, results: list) -> tuple:
    """Step 0. Combined dwell probability map of the arena: each session's occupancy (the occ_map
    its cells' rate maps were divided by) normalised to sum 1 over the arena, then averaged across
    sessions with weight 1 ('equal') or the session's number of cells ('cells').
    Returns (combined map, number of sessions)."""
    if OCC_SESSION_WEIGHTING not in ('equal', 'cells'):
        raise ValueError(f'Unknown OCC_SESSION_WEIGHTING {OCC_SESSION_WEIGHTING!r}')
    per_session = {}
    for r in results:
        per_session.setdefault(r['session'], [r['occ_map'], 0])[1] += 1

    combined = np.zeros(handler.n_bins, dtype=np.float64)
    w_total = 0.0
    for occ, n_cells in per_session.values():
        occ = np.where(handler.geom_valid, occ, 0.0)
        total = float(occ.sum())
        if total <= 0:
            continue
        w = 1.0 if OCC_SESSION_WEIGHTING == 'equal' else float(n_cells)
        combined += w * occ / total
        w_total += w
    if w_total > 0:
        combined /= w_total
    return combined, len(per_session)


def null_spike_probabilities(results: list, occ_kde: dict | None) -> np.ndarray:
    """Step 1. (n_cells, n_bins): probability that one null spike of each cell falls in each bin.
    'uniform' -- time per bin over the spike-countable frames / total (constant rate in time);
    'occupancy' -- the same time weights x the combined-occupancy KDE density of the bin."""
    W = np.stack([r['spike_occ_map'] for r in results]).astype(np.float64)
    if NULL_SPIKE_MODE == 'occupancy':
        if occ_kde is None:
            raise ValueError("NULL_SPIKE_MODE = 'occupancy' needs the combined-occupancy KDE")
        p_occ = np.clip(np.nan_to_num(occ_kde['density'], nan=0.0), 0.0, None)
        hybrid = W * p_occ[None, :]
        keep_uniform = hybrid.sum(axis=1) <= 0      # no overlap with the p_occ domain
        W = np.where(keep_uniform[:, None], W, hybrid)
    elif NULL_SPIKE_MODE != 'uniform':
        raise ValueError(f'Unknown NULL_SPIKE_MODE {NULL_SPIKE_MODE!r}')
    return W / W.sum(axis=1, keepdims=True)


def simulate_null_cell(p_bins: np.ndarray, n_spikes: int, occ_map: np.ndarray, handler,
                       rng: np.random.Generator) -> dict:
    """Step 1. One null cell: the real spike count redistributed over the real trajectory
    (multinomial over bins with p_bins), then the identical rate-map / field / peak pipeline."""
    spike_map = rng.multinomial(n_spikes, p_bins).astype(np.float64)
    cell = rate_maps_from_counts(spike_map, occ_map, handler)
    cell.update(_cell_peaks_and_field(cell, handler))
    return cell


def _null_iterations(handler, arena_index: int, p_bins: np.ndarray, n_spikes: np.ndarray,
                     occ_maps: np.ndarray, iterations: list) -> dict:
    """Steps 2-3 for a chunk of iterations (top-level so worker processes can run it).
    Returns the pooled null maps and their KDE densities, shape (kinds, iterations, n_bins)."""
    kinds = list(_KDE_MAP_KINDS)
    n_it = len(iterations)
    maps = np.full((len(kinds), n_it, handler.n_bins), np.nan, dtype=np.float32)
    dens = np.full((len(kinds), n_it, handler.n_bins), np.nan, dtype=np.float32)
    neff = np.full((len(kinds), n_it), np.nan)
    for j, it in enumerate(iterations):
        rng = np.random.default_rng([NULL_SEED, arena_index, int(it)])
        cells = [simulate_null_cell(p_bins[c], int(n_spikes[c]), occ_maps[c], handler, rng)
                 for c in range(len(n_spikes))]
        domain = _union_valid(handler, cells)
        for k, kind in enumerate(kinds):
            values, values_are_mass = _KDE_MAP_KINDS[kind][0](handler, cells)
            maps[k, j] = values
            kde = kde_density_map(handler, values, domain, values_are_mass) if domain.any() else None
            if kde is not None:
                dens[k, j] = kde['density']
                neff[k, j] = kde['neff']
    return dict(iterations=np.asarray(iterations, dtype=int), maps=maps, dens=dens, neff=neff)


def simulate_null_populations(handler, arena_key: str, results: list, occ_kde: dict | None,
                              n_iter: int) -> dict:
    """Steps 1-3 for one arena: n_iter null populations, run in NULL_MAX_WORKERS processes."""
    p_bins   = null_spike_probabilities(results, occ_kde)
    n_spikes = np.array([r['n_spikes'] for r in results], dtype=int)
    occ_maps = np.stack([r['occ_map'] for r in results])
    args = (handler, _ARENA_ORDER.index(arena_key), p_bins, n_spikes, occ_maps)
    chunks = [c.tolist() for c in np.array_split(np.arange(n_iter), max(1, min(n_iter, 4 * NULL_MAX_WORKERS)))
              if len(c)]

    t0 = time.time()
    parts = []

    def _progress():
        done = sum(len(p['iterations']) for p in parts)
        print(f'  [obs-null {arena_key}] {done}/{n_iter} iterations ({time.time() - t0:.0f} s)')

    if NULL_MAX_WORKERS > 1:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=NULL_MAX_WORKERS) as ex:
                for fut in concurrent.futures.as_completed([ex.submit(_null_iterations, *args, c)
                                                            for c in chunks]):
                    parts.append(fut.result())
                    _progress()
        except Exception as e:
            print(f'  [WARN] process pool failed ({type(e).__name__}: {e}); running serially')
            parts = []
    if not parts:
        for c in chunks:
            parts.append(_null_iterations(*args, c))
            _progress()

    parts.sort(key=lambda p: p['iterations'][0])
    return dict(iterations=np.concatenate([p['iterations'] for p in parts]),
                maps=np.concatenate([p['maps'] for p in parts], axis=1),
                dens=np.concatenate([p['dens'] for p in parts], axis=1),
                neff=np.concatenate([p['neff'] for p in parts], axis=1))


def _mean_null_map(maps: np.ndarray, domain: np.ndarray) -> np.ndarray:
    """Step 5. Mean null population map: average over iterations, NaN (no field / no peak) = 0
    inside the domain, NaN outside."""
    out = np.full(maps.shape[-1], np.nan)
    out[domain] = np.mean(np.nan_to_num(maps[:, domain].astype(np.float64), nan=0.0), axis=0)
    return out


def _w1_1d(P: np.ndarray, Q: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Exact 1D Wasserstein-1 distance between the rows of P and Q (masses on support points u):
    integral of |CDF_P - CDF_Q| du."""
    o = np.argsort(u, kind='stable')
    gaps = np.diff(u[o])
    cdf_diff = np.cumsum((P - Q)[:, o], axis=1)[:, :-1]
    return np.abs(cdf_diff) @ gaps


def _null_scalar_stats(P: np.ndarray, Q: np.ndarray, xy: np.ndarray, dist: np.ndarray,
                       edge: np.ndarray) -> dict:
    """Method 1-2 statistics of the mass distributions in the rows of P against the reference
    rows of Q (both (k, m), each row summing to 1). MaxAbsZ is added by the caller."""
    M = 0.5 * (P + Q)
    with np.errstate(divide='ignore', invalid='ignore'):
        kl_p = np.where(P > 0, P * np.log2(P / M), 0.0).sum(axis=1)
        kl_q = np.where(Q > 0, Q * np.log2(Q / M), 0.0).sum(axis=1)
    sliced = np.zeros(len(P))
    for th in np.pi * np.arange(N_SLICED_DIRECTIONS) / N_SLICED_DIRECTIONS:
        sliced += _w1_1d(P, Q, np.cos(th) * xy[0] + np.sin(th) * xy[1])
    return {'JSD_bits':         0.5 * (kl_p + kl_q),
            'SlicedW1_cm':      sliced / N_SLICED_DIRECTIONS,
            'Dist2Wall_W1_cm':  _w1_1d(P, Q, dist),
            'MeanDist2Wall_cm': P @ dist,
            'EdgeZoneFrac':     P @ edge.astype(np.float64)}


def _mc_pvalues(obs: float, null: np.ndarray, test: str) -> dict:
    n = len(null)
    p_upper = (1.0 + np.sum(null >= obs)) / (n + 1.0)
    p_lower = (1.0 + np.sum(null <= obs)) / (n + 1.0)
    p = p_upper if test == 'upper' else min(1.0, 2.0 * min(p_upper, p_lower))
    return dict(p=float(p), p_upper=float(p_upper), p_lower=float(p_lower))


def obs_null_compare(handler, real_kde: dict, null_dens: np.ndarray) -> dict | None:
    """Step 4. Real KDE vs the null KDEs of one (map kind, arena); null_dens (iterations, n_bins)."""
    fitted = np.any(np.isfinite(null_dens), axis=1)
    iterations = np.flatnonzero(fitted)
    D = null_dens[fitted].astype(np.float64)
    n = len(D)
    if n < 3:
        return None

    common = (real_kde['domain'] & np.isfinite(real_kde['density'])
              & np.all(np.isfinite(D), axis=0))
    area = handler.bin_areas_cm2()[common]
    xy   = handler.bin_centres_xy()[:, common]
    dist = handler.dist_to_wall[common]
    edge = handler.edge_zone_flat[common]

    g = np.clip(real_kde['density'][common], 0.0, None) * area
    g = g / g.sum()
    G = np.clip(D[:, common], 0.0, None) * area
    G = G / G.sum(axis=1, keepdims=True)

    # reference = mean null; each null against its leave-one-out reference
    S1, S2 = G.sum(axis=0), (G ** 2).sum(axis=0)
    r  = S1 / n
    sd = np.sqrt(np.clip((S2 - n * r ** 2) / (n - 1), 0.0, None))
    R_loo  = (S1[None, :] - G) / (n - 1)
    sd_loo = np.sqrt(np.clip((S2[None, :] - G ** 2 - (n - 1) * R_loo ** 2) / (n - 2), 0.0, None))

    z = np.divide(g - r, sd, out=np.zeros_like(g), where=sd > 0)
    Z = np.divide(G - R_loo, sd_loo, out=np.zeros_like(G), where=sd_loo > 0)
    p_point = (1.0 + np.sum(np.abs(Z) >= np.abs(z)[None, :], axis=0)) / (n + 1.0)

    T_null = np.abs(Z).max(axis=1)
    T_obs  = float(np.abs(z).max())
    c_crit = float(np.quantile(T_null, 1.0 - NULL_ALPHA))

    stats_obs  = {k: float(v[0]) for k, v in _null_scalar_stats(g[None, :], r[None, :], xy, dist, edge).items()}
    stats_null = _null_scalar_stats(G, R_loo, xy, dist, edge)
    stats_obs['MaxAbsZ'], stats_null['MaxAbsZ'] = T_obs, T_null
    pvals = {s: _mc_pvalues(stats_obs[s], stats_null[s], test) for s, test in _NULL_SCALAR_STATS.items()}

    # Method 2 distance-to-wall profile (mass per band)
    n_bands = int(np.floor(dist.max() / DIST_PROFILE_BIN_CM)) + 1
    band = np.minimum((dist / DIST_PROFILE_BIN_CM).astype(int), n_bands - 1)
    onehot = np.zeros((len(dist), n_bands))
    onehot[np.arange(len(dist)), band] = 1.0
    prof_null = G @ onehot

    with np.errstate(divide='ignore', invalid='ignore'):
        enrichment = np.log2(g / r)

    out = dict(common=common, iterations=iterations, n_iter=n,
               f_real=g / area, f_null=r / area, mass_real=g, mass_null=r, sd_null=sd,
               diff=(g - r) / area, z=z, p=p_point, enrichment=enrichment,
               T_obs=T_obs, T_null=T_null, c_crit=c_crit,
               p_global=float((1.0 + np.sum(T_null >= T_obs)) / (n + 1.0)),
               stats_obs=stats_obs, stats_null=stats_null, pvals=pvals,
               prof_centres=(np.arange(n_bands) + 0.5) * DIST_PROFILE_BIN_CM,
               prof_real=g @ onehot, prof_null_mean=prof_null.mean(axis=0),
               prof_null_lo=np.percentile(prof_null, 2.5, axis=0),
               prof_null_hi=np.percentile(prof_null, 97.5, axis=0))
    for tag, sel in (('above', z > c_crit), ('below', z < -c_crit)):
        mask = np.zeros(handler.n_bins, dtype=bool)
        mask[np.flatnonzero(common)[sel]] = True
        out[f'lab_{tag}'], out[f'n_{tag}'] = _duong_cluster_labels(handler, mask)
    return out


def run_obs_null(arena_handlers: dict, arena_results: dict, kdes: dict) -> tuple:
    """Stage 3 for every arena with cells. Fills kdes['null'][kind][arena] with the KDE of the
    mean null population map (for the Duong test) and returns (obs_null, n_null_cells), where
    obs_null[arena] holds the occupancy, the null stacks and obs_null_compare per map kind."""
    obs_null = {}
    for arena in _ARENA_ORDER:
        results = arena_results[arena]
        if not results:
            continue
        handler = arena_handlers[arena]
        domain = _union_valid(handler, results)
        if not domain.any():
            continue

        occ_comb, n_sessions = combined_occupancy(handler, results)
        occ_kde = kde_density_map(handler, occ_comb, domain, values_are_mass=True)
        print(f'[obs-null {arena}] {len(results)} cells, {n_sessions} sessions, '
              f'{N_NULL_ITER} iterations, mode = {NULL_SPIKE_MODE}')
        sim = simulate_null_populations(handler, arena, results, occ_kde, N_NULL_ITER)

        entry = dict(occ_combined=occ_comb, occ_kde=occ_kde, n_sessions=n_sessions,
                     n_cells=len(results), domain=domain, sim=sim, mean_maps={}, compare={})
        for k, kind in enumerate(_KDE_MAP_KINDS):
            values_are_mass = _KDE_MAP_KINDS[kind][0](handler, results)[1]
            mean_map = _mean_null_map(sim['maps'][k], domain)
            entry['mean_maps'][kind] = mean_map
            null_kde = kde_density_map(handler, mean_map, domain, values_are_mass)
            if null_kde is not None:
                kdes['null'].setdefault(kind, {})[arena] = null_kde

            real_kde = kdes['real'].get(kind, {}).get(arena)
            res = obs_null_compare(handler, real_kde, sim['dens'][k]) if real_kde is not None else None
            entry['compare'][kind] = res
            if res is not None:
                print(f'[obs-null {kind}] {arena}: N = {res["n_iter"]}, JSD p = '
                      f'{res["pvals"]["JSD_bits"]["p"]:.3g}, global envelope p = '
                      f'{res["p_global"]:.3g}, clusters above = {res["n_above"]}, '
                      f'below = {res["n_below"]}')
        obs_null[arena] = entry

    n_null_cells = {arena: e['n_cells'] * len(e['sim']['iterations']) for arena, e in obs_null.items()}
    return obs_null, n_null_cells


def export_obs_null(arena_handlers: dict, obs_null: dict, out_dir: str):
    summary, envelope, clusters, iters, saved = [], [], [], [], {}
    for arena, entry in obs_null.items():
        handler = arena_handlers[arena]
        saved[f'{arena}_occupancy_combined'] = entry['occ_combined']
        if entry['occ_kde'] is not None:
            saved[f'{arena}_occupancy_kde'] = entry['occ_kde']['density']
        for k, (kind, res) in enumerate(entry['compare'].items()):
            pre = f'{kind}_{arena}_'
            saved[pre + 'null_maps'] = entry['sim']['maps'][k]
            saved[pre + 'null_density'] = entry['sim']['dens'][k]
            saved[pre + 'null_mean_map'] = entry['mean_maps'][kind]
            if res is None:
                continue
            for s, test in _NULL_SCALAR_STATS.items():
                null = res['stats_null'][s]
                summary.append(dict(
                    map_kind=kind, arena=arena, statistic=s, test=test,
                    observed=res['stats_obs'][s], null_mean=float(np.mean(null)),
                    null_sd=float(np.std(null, ddof=1)),
                    null_q025=float(np.percentile(null, 2.5)), null_q975=float(np.percentile(null, 97.5)),
                    direction='real > null' if res['stats_obs'][s] > np.mean(null) else 'real < null',
                    **res['pvals'][s], n_iter=res['n_iter'], n_cells=entry['n_cells'],
                    n_sessions=entry['n_sessions'], null_mode=NULL_SPIKE_MODE))
            envelope.append(dict(
                map_kind=kind, arena=arena, alpha=NULL_ALPHA, m_points=int(res['common'].sum()),
                n_iter=res['n_iter'], max_abs_z_obs=res['T_obs'], c_crit=res['c_crit'],
                p_global=res['p_global'],
                n_bins_above=int(np.sum(res['z'] > res['c_crit'])),
                n_bins_below=int(np.sum(res['z'] < -res['c_crit'])),
                n_clusters_above=res['n_above'], n_clusters_below=res['n_below']))
            clusters += duong_cluster_rows(kind, arena, handler, res)
            for i, it in enumerate(res['iterations']):
                iters.append(dict(map_kind=kind, arena=arena, iteration=int(it),
                                  **{s: float(res['stats_null'][s][i]) for s in _NULL_SCALAR_STATS}))

            for name in ('z', 'p', 'enrichment', 'f_real', 'f_null', 'sd_null'):
                arr = np.full(handler.n_bins, np.nan)
                arr[res['common']] = res[name]
                saved[pre + name] = arr
            saved[pre + 'common_domain'] = res['common']
            saved[pre + 'cluster_above'] = res['lab_above']
            saved[pre + 'cluster_below'] = res['lab_below']
            saved[pre + 'iterations_used'] = res['iterations']

    if not summary:
        print('[obs-null] nothing to export')
        return
    xlsx = os.path.join(out_dir, 'ObsNull_Stats.xlsx')
    with pd.ExcelWriter(xlsx) as xw:
        pd.DataFrame(summary).to_excel(xw, sheet_name='Summary', index=False)
        pd.DataFrame(envelope).to_excel(xw, sheet_name='GlobalEnvelope', index=False)
        pd.DataFrame(clusters).reindex(columns=_CLUSTER_COLS).to_excel(xw, sheet_name='Clusters', index=False)
        pd.DataFrame(iters).to_excel(xw, sheet_name='NullIterations', index=False)
    print(f'[SAVED] {xlsx}')
    npz = os.path.join(out_dir, 'ObsNull_Results.npz')
    np.savez_compressed(npz, null_mode=NULL_SPIKE_MODE, **saved)
    print(f'[SAVED] {npz}')


# ============================================================================
# STAGE 4 -- Duong (2013) local significant differences, real KDE vs null KDE
# ============================================================================
#
# Per map kind x arena:
#   1. Evaluation points x_j = bin centres of the COMMON domain (real domain & null domain, both
#      densities finite and their raw KDE > 0). With DUONG_RENORMALISE_ON_COMMON_DOMAIN both
#      densities are rescaled to integrate to 1 over that common domain, so the comparison is of
#      where the mass sits, not of how much of it fell outside.
#   2. Local statistic U(x) = [f1(x) - f2(x)]^2, f1 = real KDE, f2 = null KDE. Under the
#      local null H0(x): f1(x) = f2(x),  X^2(x) = U(x) / sigma_U^2(x)  ->  chi^2 with 1 df, where
#          sigma_U^2(x) = R(K) [ n1^-1 |H1|^-1/2 f1(x) + n2^-1 |H2|^-1/2 f2(x) ],
#      R(K) = 1 / (4 pi) for the 2D Gaussian kernel. Boundary term: the KDEs are edge-corrected
#      (raw KDE divided by the kernel mass m_H(x) inside the domain). For the Gaussian kernel
#      K_H^2 = R(K) |H|^-1/2 K_{H/2}, so the variance of the corrected estimate is the paper's
#      term multiplied by m_{H/2}(x) / m_H(x)^2 -- 1 in the interior, > 1 at a wall. For an
#      uncorrected KDE the factor reduces to m_{H/2}(x). The m_H computed here is checked against
#      the correction kde_density_map applied (density / density_raw) as a geometry sanity check.
#   3. p_j = P(chi^2_1 >= X^2_j); Hochberg (1988) step-up over all m evaluation points of the
#      (kind, arena) family: j* = max{ j : p_(j) <= alpha / (m - j + 1) }, reject p_(1..j*).
#   4. Rejected points with f1 > f2: real density significantly ABOVE the null;
#      f1 < f2: significantly BELOW. Each sign is split into 8-connected clusters of bins
#      (wrapping around the circular track).
#
# Sample sizes n1, n2 (DUONG_SAMPLE_SIZE_MODE):
#   'neff'  -- Kish effective sample size of each weighted KDE. For the peak-proportion maps this
#              is ~ the number of cells, a genuine sample of peak locations. For the
#              overall and field-only maps the weighted points are BINS of a smoothed map, which
#              are spatially correlated, so neff overstates the independent information there ->
#              the test is liberal for those two map kinds; check with 'cells'.
#   'cells' -- n1 = number of real place cells pooled, n2 = number of null cells pooled
#              (cells x N_NULL_ITER: the null KDE is that of the mean of N_NULL_ITER null
#              population maps). Conservative for the bin-weighted maps.

def _duong_kde_side(handler, kde_out: dict, common: np.ndarray, n: float) -> dict:
    """Density and its asymptotic variance on the common evaluation points for one KDE."""
    dens, raw = kde_out['density'], kde_out['density_raw']
    H   = np.asarray(kde_out['cov'], dtype=float)
    own = kde_out['domain']
    xy, area = handler.bin_centres_xy(), handler.bin_areas_cm2()

    ev, src = xy[:, common], xy[:, own]
    m_H  = _kernel_mass_in_domain(ev, src, area[own], H)
    m_H2 = _kernel_mass_in_domain(ev, src, area[own], H / 2.0)

    f, f_raw = dens[common], raw[common]
    c = f / f_raw                                    # multiplier the KDE applied to the raw estimate
    corrected = not np.allclose(c, 1.0, rtol=1e-6)
    edge_check = float(np.max(np.abs(c * m_H - 1.0))) if corrected else 0.0

    f_true = f_raw / m_H                             # boundary-unbiased estimate of f(x)
    var = c ** 2 * R_K_GAUSS_2D / (n * np.sqrt(np.linalg.det(H))) * f_true * m_H2

    scale = 1.0
    if DUONG_RENORMALISE_ON_COMMON_DOMAIN:
        scale = 1.0 / float(np.sum(f * area[common]))
    return dict(f=f * scale, var=var * scale ** 2, scale=scale, H=H, n=n,
                corrected=corrected, edge_check=edge_check,
                mass_outside_common=float(1.0 - np.sum(dens[common] * area[common])
                                          / np.nansum(dens[own] * area[own])))


def hochberg_reject(p: np.ndarray, alpha: float) -> tuple:
    """Hochberg (1988) step-up. Returns (reject mask, p cut-off or nan, j*)."""
    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    ok = p_sorted <= alpha / (m - np.arange(1, m + 1) + 1)
    reject = np.zeros(m, dtype=bool)
    if not ok.any():
        return reject, float('nan'), 0
    j_star = int(np.flatnonzero(ok).max()) + 1
    reject[order[:j_star]] = True
    return reject, float(p_sorted[j_star - 1]), j_star


def _duong_cluster_labels(handler, mask_flat: np.ndarray) -> tuple:
    """8-connected clusters of mask_flat (wrapping on the circular track) with at least
    DUONG_MIN_CLUSTER_BINS bins. (labels, n), label -1 = no cluster."""
    labels = np.full(handler.n_bins, -1, dtype=int)
    n = 0
    for component in handler.connected_components(mask_flat):
        if len(component) >= DUONG_MIN_CLUSTER_BINS:
            labels[component] = n
            n += 1
    return labels, n


def duong_compare_arena(handler, real_kde: dict, null_kde: dict, n_real: float,
                        n_null: float) -> dict:
    common = (real_kde['domain'] & null_kde['domain']
              & np.isfinite(real_kde['density']) & np.isfinite(null_kde['density'])
              & (real_kde['density_raw'] > 0) & (null_kde['density_raw'] > 0))
    real = _duong_kde_side(handler, real_kde, common, n_real)
    null = _duong_kde_side(handler, null_kde, common, n_null)

    diff = real['f'] - null['f']
    z = diff / np.sqrt(real['var'] + null['var'])
    x2 = z ** 2
    p = chi2.sf(x2, df=1)
    reject, p_cut, j_star = hochberg_reject(p, DUONG_ALPHA)

    out = dict(common=common, real=real, null=null, f_real=real['f'], f_null=null['f'],
               diff=diff, z=z, x2=x2, p=p, reject=reject, p_cut=p_cut, j_star=j_star)
    for tag, sel in (('above', reject & (diff > 0)), ('below', reject & (diff < 0))):
        mask = np.zeros(handler.n_bins, dtype=bool)
        mask[np.flatnonzero(common)[sel]] = True
        out[f'lab_{tag}'], out[f'n_{tag}'] = _duong_cluster_labels(handler, mask)
    return out


def duong_cluster_rows(kind: str, arena: str, handler, res: dict) -> list:
    xy, area_all = handler.bin_centres_xy(), handler.bin_areas_cm2()
    rows = []
    idx = np.flatnonzero(res['common'])
    for direction, labels, n in (('real > null', res['lab_above'], res['n_above']),
                                 ('real < null', res['lab_below'], res['n_below'])):
        for k in range(n):
            sel = labels[idx] == k                      # positions within the common domain
            b = idx[sel]
            a = area_all[b]
            dw = handler.dist_to_wall[b]
            row = dict(map_kind=kind, arena=arena, direction=direction, cluster=k + 1,
                       n_bins=int(len(b)), area_cm2=round(float(a.sum()), 2),
                       centroid_x_cm=round(float(np.sum(xy[0, b] * a) / a.sum()), 2),
                       centroid_y_cm=round(float(np.sum(xy[1, b] * a) / a.sum()), 2),
                       mean_dist_to_wall_cm=round(float(np.sum(dw * a) / a.sum()), 2),
                       min_dist_to_wall_cm=round(float(dw.min()), 2),
                       max_dist_to_wall_cm=round(float(dw.max()), 2),
                       mean_real_density=float(np.mean(res['f_real'][sel])),
                       mean_null_density=float(np.mean(res['f_null'][sel])),
                       mean_ratio_real_over_null=round(float(np.mean(res['f_real'][sel] / res['f_null'][sel])), 4),
                       excess_mass=round(float(np.sum((res['f_real'][sel] - res['f_null'][sel]) * a)), 5),
                       peak_abs_z=round(float(np.max(np.abs(res['z'][sel]))), 3),
                       min_p=float(np.min(res['p'][sel])))
            if isinstance(handler, CircularTrackHandler):
                th = np.arctan2(xy[1, b] - handler.cy, xy[0, b] - handler.cx)
                ang = np.arctan2(np.sum(np.sin(th) * a), np.sum(np.cos(th) * a))
                row['centroid_angle_deg'] = round(float(np.degrees(ang) % 360.0), 1)
            rows.append(row)
    return rows


_CLUSTER_COLS = ['map_kind', 'arena', 'direction', 'cluster', 'n_bins', 'area_cm2',
                 'centroid_x_cm', 'centroid_y_cm', 'centroid_angle_deg',
                 'mean_dist_to_wall_cm', 'min_dist_to_wall_cm', 'max_dist_to_wall_cm',
                 'mean_real_density', 'mean_null_density', 'mean_ratio_real_over_null',
                 'excess_mass', 'peak_abs_z', 'min_p']


def run_duong_tests(arena_handlers: dict, kdes: dict, real_results: dict, n_null_cells: dict,
                    out_dir: str):
    """Duong test of every real KDE against its null KDE; figures, workbook and npz.
    kdes = {'real': {kind: {arena: kde}}, 'null': {kind: {arena: kde}}}; a (kind, arena) missing
    either KDE is skipped. n_null_cells = {arena: number of null cells behind the null KDE}."""
    test_rows, cluster_rows, saved = [], [], {}
    for kind, (_, title, *_) in _KDE_MAP_KINDS.items():
        results = {}
        for arena in _ARENA_ORDER:
            real_kde = kdes['real'].get(kind, {}).get(arena)
            null_kde = kdes['null'].get(kind, {}).get(arena)
            if real_kde is None or null_kde is None:
                continue
            handler = arena_handlers[arena]
            if DUONG_SAMPLE_SIZE_MODE == 'neff':
                n_real, n_null = real_kde['neff'], null_kde['neff']
            elif DUONG_SAMPLE_SIZE_MODE == 'cells':
                n_real, n_null = float(len(real_results[arena])), float(n_null_cells[arena])
            else:
                raise ValueError(f'Unknown DUONG_SAMPLE_SIZE_MODE {DUONG_SAMPLE_SIZE_MODE!r}')
            res = duong_compare_arena(handler, real_kde, null_kde, n_real, n_null)
            results[arena] = res

            for side in ('real', 'null'):
                if res[side]['corrected'] and res[side]['edge_check'] > 0.02:
                    print(f'  [WARN] {kind}/{arena}/{side}: edge-correction mass here differs from '
                          f'the KDE\'s by up to {100 * res[side]["edge_check"]:.1f} %')
            print(f'[duong {kind}] {arena}: m = {res["common"].sum()}, rejected = '
                  f'{res["reject"].sum()} (p cut-off {res["p_cut"]:.3g}), clusters above = '
                  f'{res["n_above"]}, below = {res["n_below"]}')

            test_rows.append(dict(
                map_kind=kind, arena=arena, alpha=DUONG_ALPHA, sample_size_mode=DUONG_SAMPLE_SIZE_MODE,
                n1_real=round(res['real']['n'], 2), n2_null=round(res['null']['n'], 2),
                H1=np.round(res['real']['H'], 3).tolist(), H2=np.round(res['null']['H'], 3).tolist(),
                m_points=int(res['common'].sum()),
                real_domain_bins=int(real_kde['domain'].sum()),
                null_domain_bins=int(null_kde['domain'].sum()),
                real_mass_outside_common=round(res['real']['mass_outside_common'], 4),
                null_mass_outside_common=round(res['null']['mass_outside_common'], 4),
                hochberg_j_star=res['j_star'], hochberg_p_cutoff=res['p_cut'],
                n_rejected=int(res['reject'].sum()),
                n_bins_real_above=int(np.sum(res['reject'] & (res['diff'] > 0))),
                n_bins_real_below=int(np.sum(res['reject'] & (res['diff'] < 0))),
                n_clusters_above=res['n_above'], n_clusters_below=res['n_below'],
                min_p=float(res['p'].min()), max_abs_z=round(float(np.abs(res['z']).max()), 3),
                edge_check_real=round(res['real']['edge_check'], 5),
                edge_check_null=round(res['null']['edge_check'], 5)))
            cluster_rows += duong_cluster_rows(kind, arena, handler, res)

            pre = f'{kind}_{arena}_'
            for name, vals in (('f_real', res['f_real']), ('f_null', res['f_null']),
                               ('var_real', res['real']['var']), ('var_null', res['null']['var']),
                               ('z', res['z']), ('p', res['p'])):
                arr = np.full(handler.n_bins, np.nan)
                arr[res['common']] = vals
                saved[pre + name] = arr
            rej = np.zeros(handler.n_bins, dtype=bool)
            rej[res['common']] = res['reject']
            saved[pre + 'reject'] = rej
            saved[pre + 'common_domain'] = res['common']
            saved[pre + 'cluster_above'] = res['lab_above']
            saved[pre + 'cluster_below'] = res['lab_below']

        if results:
            plot_duong_kind(title, arena_handlers, results,
                            os.path.join(out_dir, f'DuongTest_{kind}.png'))

    if not test_rows:
        print('[duong] skipped: no (map kind, arena) with both a real and a null KDE')
        return
    xlsx = os.path.join(out_dir, 'DuongTest_Clusters.xlsx')
    with pd.ExcelWriter(xlsx) as xw:
        pd.DataFrame(test_rows).to_excel(xw, sheet_name='Tests', index=False)
        pd.DataFrame(cluster_rows).reindex(columns=_CLUSTER_COLS).to_excel(
            xw, sheet_name='Clusters', index=False)
    print(f'[SAVED] {xlsx}')
    npz = os.path.join(out_dir, 'DuongTest_Results.npz')
    np.savez_compressed(npz, **saved)
    print(f'[SAVED] {npz}')


# ============================================================================
# Plotting
# ============================================================================

_ARENA_ORDER = ['open_field', 'circular_track', 'linear_track']
_ARENA_TITLES = {'open_field': 'Open Field', 'circular_track': 'Circular Track', 'linear_track': 'Linear Track'}
_DUONG_ROW_HEIGHT = {'open_field': 1.0, 'circular_track': 1.0, 'linear_track': 0.45}
_EDGE_PTS = 6   # points per bin side, so the circular track's bin edges follow the ring's arcs


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

def _bin_polygons(handler, flat_bins: np.ndarray) -> np.ndarray:
    """(n, 4 * _EDGE_PTS, 2) outline of every bin in flat_bins, in arena cm."""
    s = np.linspace(0.0, 1.0, _EDGE_PTS)
    du = np.concatenate([s, np.ones_like(s), s[::-1], np.zeros_like(s)])
    dv = np.concatenate([np.zeros_like(s), s, np.ones_like(s), s[::-1]])
    ix, iy = np.divmod(flat_bins, handler.ny)
    x, y = handler.edge_to_xy(ix[:, None] + du, iy[:, None] + dv)
    return np.stack([x, y], axis=-1)


def _field_boundary_segments(handler, labels: np.ndarray) -> list:
    """Every bin edge separating two different labels, or a labelled bin from an unlabelled one
    or from outside the arena (label -1 = none)."""
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


def _hbar(fig, mappable, ax, label):
    fig.colorbar(mappable, ax=ax, orientation='horizontal', fraction=0.05, pad=0.03, label=label)


# --- Stage 1: mean map figures ---

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

    fig.suptitle('Field-only mean field-index maps')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_peak_proportion_maps(arena_handlers: dict, arena_results: dict, save_path: str):
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

    fig.suptitle('Peak proportion maps')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


# --- Stage 2: KDE figures (+ npz) ---

def plot_kde_maps(arena_handlers: dict, arena_results: dict, map_kind: str, save_path: str,
                  cell_label: str = 'place cells', title_prefix: str = '') -> dict:
    """KDE of one pooled map kind for every arena: figure, npz next to it, and the KDEs
    (arena -> kde_density_map output) for the Duong test."""
    _, title, units, _, _ = _KDE_MAP_KINDS[map_kind]
    fig = plt.figure(figsize=(17, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.7])
    arena_axes = {'open_field':     fig.add_subplot(gs[:, 0]),
                  'circular_track': fig.add_subplot(gs[0, 1]),
                  'linear_track':   fig.add_subplot(gs[1, 1])}
    saved, kdes = {}, {}
    for key in _ARENA_ORDER:
        handler = arena_handlers[key]
        results = arena_results[key]
        ax = arena_axes[key]

        kde_out = arena_kde(handler, results, map_kind)
        if kde_out is None:
            _no_data(ax, key, 'no KDE: too few data')
            continue
        kdes[key] = kde_out

        density, domain = kde_out['density'], kde_out['domain']
        cmap, norm_c = make_cmap_norm(density[domain])
        pcm = handler.plot_bins_2d(ax, density, domain & np.isfinite(density), cmap, norm_c)
        sd = np.sqrt(np.diag(kde_out['cov']))
        ax.set_title(f'{_ARENA_TITLES[key]} (n={len(results)} {cell_label}, '
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
    fig.suptitle(f'{title_prefix}{title} -- 2D Gaussian KDE (Scott\'s rule, {corr})')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')

    if saved:
        npz_path = os.path.splitext(save_path)[0] + '.npz'
        np.savez_compressed(npz_path, scaled_units=units, **saved)
        print(f'[SAVED] {npz_path}')
    return kdes


# --- Stage 3: Duong test figures ---

def _duong_outline_segments(handler, mask_flat: np.ndarray) -> list:
    """Every bin side separating a bin of mask_flat from a bin outside it, in room cm."""
    return _field_boundary_segments(handler, np.where(mask_flat, 0, -1))


def _draw_duong_clusters(ax, handler, res: dict):
    halo = [pe.Stroke(linewidth=OUTLINE_LW + 1.8, foreground='black'), pe.Normal()]
    for tag, colour in (('above', GREEN_ABOVE), ('below', MAGENTA_BELOW)):
        mask = res[f'lab_{tag}'] >= 0
        if mask.any():
            lc = LineCollection(_duong_outline_segments(handler, mask), colors=colour,
                                linewidths=OUTLINE_LW, capstyle='round', joinstyle='round', zorder=6)
            lc.set_path_effects(halo)
            ax.add_collection(lc)


def plot_duong_kind(title: str, arena_handlers: dict, results: dict, save_path: str):
    """Rows = arenas; columns = real KDE, null KDE, difference, signed z, significant clusters.
    Fluorescent green (outline, and fill in the last column) = real significantly ABOVE the null;
    magenta = significantly BELOW."""
    arenas = [a for a in _ARENA_ORDER if a in results]
    ratios = [_DUONG_ROW_HEIGHT[a] for a in arenas]
    fig = plt.figure(figsize=(28, 1.2 + 5.6 * sum(ratios)))
    gs = fig.add_gridspec(len(arenas), 5, height_ratios=ratios)
    dens_cmap = plt.get_cmap('Greys')
    div_cmap = plt.get_cmap('RdBu_r')

    for r, arena in enumerate(arenas):
        handler, res = arena_handlers[arena], results[arena]
        common = res['common']
        both = np.concatenate([res['f_real'], res['f_null']])
        dens_norm = Normalize(vmin=float(both.min()), vmax=float(both.max()))
        dmax = float(np.max(np.abs(res['diff']))) or 1e-12
        zmax = float(np.max(np.abs(res['z']))) or 1.0
        z_star = float(normal_dist.isf(res['p_cut'] / 2.0)) if np.isfinite(res['p_cut']) else float('nan')

        panels = [
            (res['f_real'], dens_cmap, dens_norm, 'density (cm$^{-2}$)',
             f"{_ARENA_TITLES[arena]} -- REAL KDE\n"
             f"n1 = {res['real']['n']:.1f}, H1 sd = {np.sqrt(res['real']['H'][0, 0]):.1f} x "
             f"{np.sqrt(res['real']['H'][1, 1]):.1f} cm"),
            (res['f_null'], dens_cmap, dens_norm, 'density (cm$^{-2}$)',
             f"NULL KDE\n"
             f"n2 = {res['null']['n']:.1f}, H2 sd = {np.sqrt(res['null']['H'][0, 0]):.1f} x "
             f"{np.sqrt(res['null']['H'][1, 1]):.1f} cm"),
            (res['diff'], div_cmap, TwoSlopeNorm(0.0, -dmax, dmax), 'real - null (cm$^{-2}$)',
             'Difference  f$_{real}$ - f$_{null}$'),
            (res['z'], div_cmap, TwoSlopeNorm(0.0, -zmax, zmax), 'signed z = (f$_1$ - f$_2$) / $\\sigma_U$',
             f"Duong local test, Hochberg alpha = {DUONG_ALPHA:g}\n"
             f"m = {common.sum()} points, {res['reject'].sum()} rejected "
             + (f"(|z| >= {z_star:.2f})" if np.isfinite(z_star) else '(none)')
             + f"; clusters: {res['n_above']} above, {res['n_below']} below"),
        ]
        for c, (vals, cmap, nrm, cbar_label, ttl) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            full = np.full(handler.n_bins, np.nan)
            full[common] = vals
            pc = _draw_bins(ax, handler, common, full, cmap, nrm)
            _draw_duong_clusters(ax, handler, res)
            ax.set_title(ttl, fontsize=9)
            _hbar(fig, pc, ax, cbar_label)

        ax = fig.add_subplot(gs[r, 4])
        colours = np.tile(to_rgba('0.88'), (handler.n_bins, 1))
        colours[res['lab_above'] >= 0] = to_rgba(GREEN_ABOVE)
        colours[res['lab_below'] >= 0] = to_rgba(MAGENTA_BELOW)
        _draw_bins(ax, handler, common, facecolors=colours)
        ax.set_title(f"Significant clusters\n{res['n_above']} real > null (green), "
                     f"{res['n_below']} real < null (magenta)", fontsize=9)

    halo = [pe.Stroke(linewidth=OUTLINE_LW + 1.8, foreground='black'), pe.Normal()]
    handles = [Line2D([], [], color=GREEN_ABOVE, lw=OUTLINE_LW, path_effects=halo,
                      label='Real significantly ABOVE null'),
               Line2D([], [], color=MAGENTA_BELOW, lw=OUTLINE_LW, path_effects=halo,
                      label='Real significantly BELOW null')]
    fig.legend(handles=handles, loc='upper right', ncol=2, frameon=False, fontsize=10)
    renorm = 'renormalised on the common domain' if DUONG_RENORMALISE_ON_COMMON_DOMAIN else 'as estimated'
    fig.suptitle(f'{title}: real vs null KDE -- Duong (2013) local significant differences\n'
                 f'(densities {renorm}; n = {DUONG_SAMPLE_SIZE_MODE}; blank = outside the common domain)',
                 x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97 - 0.15 / (1.2 + 5.6 * sum(ratios))))
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


# --- Stage 3: observed-null figures ---

def _full(handler, common: np.ndarray, vals: np.ndarray) -> np.ndarray:
    arr = np.full(handler.n_bins, np.nan)
    arr[common] = vals
    return arr


def plot_combined_occupancy(arena_handlers: dict, obs_null: dict, save_path: str):
    """Row 1: combined dwell probability per bin; row 2: its KDE p_occ(x)."""
    arenas = [a for a in _ARENA_ORDER if a in obs_null]
    if not arenas:
        return
    fig = plt.figure(figsize=(6.0 * len(arenas), 11))
    for c, arena in enumerate(arenas):
        handler, entry = arena_handlers[arena], obs_null[arena]
        occ, dom = entry['occ_combined'], entry['domain']
        shown = (occ > 0) & handler.geom_valid
        cmap, nrm = make_cmap_norm(occ[shown])
        ax = fig.add_subplot(2, len(arenas), c + 1)
        _hbar(fig, _draw_bins(ax, handler, shown, occ, cmap, nrm), ax, 'dwell probability per bin')
        ax.set_title(f'{_ARENA_TITLES[arena]} -- combined occupancy\n'
                     f'{entry["n_sessions"]} sessions, weighting = {OCC_SESSION_WEIGHTING}', fontsize=9)
        ax = fig.add_subplot(2, len(arenas), len(arenas) + c + 1)
        if entry['occ_kde'] is None:
            _no_data(ax, arena, 'no occupancy KDE')
            continue
        dens = entry['occ_kde']['density']
        cmap, nrm = make_cmap_norm(dens[dom])
        _hbar(fig, _draw_bins(ax, handler, dom & np.isfinite(dens), dens, cmap, nrm), ax,
              'p$_{occ}$ (cm$^{-2}$)')
        ax.set_title('Occupancy KDE p$_{occ}$(x) (Scott\'s rule, edge-corrected)', fontsize=9)
    fig.suptitle('Combined occupancy (dwell probability, sessions normalised before averaging)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_obs_null_kind(kind: str, arena_handlers: dict, arena_results: dict, obs_null: dict,
                       save_path: str):
    """Rows = arenas; columns = real pooled map, mean null pooled map, real KDE, mean null KDE,
    log2 enrichment, z map. Green / magenta outlines = global-envelope clusters above / below."""
    _, title, _, map_label, _ = _KDE_MAP_KINDS[kind]
    arenas = [a for a in _ARENA_ORDER if a in obs_null and obs_null[a]['compare'].get(kind) is not None]
    if not arenas:
        return
    ratios = [_DUONG_ROW_HEIGHT[a] for a in arenas]
    fig = plt.figure(figsize=(33, 1.2 + 5.6 * sum(ratios)))
    gs = fig.add_gridspec(len(arenas), 6, height_ratios=ratios)
    div_cmap = plt.get_cmap('RdBu_r')

    for r, arena in enumerate(arenas):
        handler, entry = arena_handlers[arena], obs_null[arena]
        res, dom, common = entry['compare'][kind], entry['domain'], entry['compare'][kind]['common']
        real_map = _KDE_MAP_KINDS[kind][0](handler, arena_results[arena])[0]
        null_map = entry['mean_maps'][kind]
        map_cmap, map_norm = make_cmap_norm(np.concatenate([real_map[dom], null_map[dom]]))
        den_cmap, den_norm = make_cmap_norm(np.concatenate([res['f_real'], res['f_null']]))
        enr = res['enrichment'][np.isfinite(res['enrichment'])]
        emax = float(np.max(np.abs(enr))) if enr.size and np.max(np.abs(enr)) > 0 else 1.0
        zmax = max(float(np.max(np.abs(res['z']))), res['c_crit'], 1e-6)

        panels = [
            (dom, real_map, map_cmap, map_norm, map_label, False,
             f'{_ARENA_TITLES[arena]} -- REAL pooled map (n = {entry["n_cells"]} cells)'),
            (dom, null_map, map_cmap, map_norm, map_label, False,
             f'NULL pooled map, mean of {res["n_iter"]} iterations ({NULL_SPIKE_MODE})'),
            (common, _full(handler, common, res['f_real']), den_cmap, den_norm,
             'density (cm$^{-2}$)', False, 'REAL KDE'),
            (common, _full(handler, common, res['f_null']), den_cmap, den_norm,
             'density (cm$^{-2}$)', False, 'NULL KDE (mean of the null KDEs)'),
            (common, _full(handler, common, res['enrichment']), div_cmap,
             TwoSlopeNorm(0.0, -emax, emax), 'log$_2$(real / null)', True,
             f'Enrichment; JSD = {res["stats_obs"]["JSD_bits"]:.3g} bits '
             f'(p = {res["pvals"]["JSD_bits"]["p"]:.3g}), sliced W1 = '
             f'{res["stats_obs"]["SlicedW1_cm"]:.2f} cm (p = {res["pvals"]["SlicedW1_cm"]["p"]:.3g})'),
            (common, _full(handler, common, res['z']), div_cmap, TwoSlopeNorm(0.0, -zmax, zmax),
             'z = (real - null) / sd$_{null}$', True,
             f'z map; global envelope |z| > {res["c_crit"]:.2f} (alpha = {NULL_ALPHA:g}), '
             f'p$_{{global}}$ = {res["p_global"]:.3g}\nclusters: {res["n_above"]} above, '
             f'{res["n_below"]} below'),
        ]
        for c, (shown, vals, cmap, nrm, cbar_label, outline, ttl) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            pc = _draw_bins(ax, handler, shown, vals, cmap, nrm)
            if outline:
                _draw_duong_clusters(ax, handler, res)
            ax.set_title(ttl, fontsize=9)
            _hbar(fig, pc, ax, cbar_label)

    halo = [pe.Stroke(linewidth=OUTLINE_LW + 1.8, foreground='black'), pe.Normal()]
    handles = [Line2D([], [], color=GREEN_ABOVE, lw=OUTLINE_LW, path_effects=halo,
                      label='Real significantly ABOVE null (global envelope)'),
               Line2D([], [], color=MAGENTA_BELOW, lw=OUTLINE_LW, path_effects=halo,
                      label='Real significantly BELOW null (global envelope)')]
    fig.legend(handles=handles, loc='upper right', ncol=2, frameon=False, fontsize=10)
    fig.suptitle(f'{title}: real vs occupancy-matched spike null (full pipeline re-run per '
                 f'iteration)', x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97 - 0.15 / (1.2 + 5.6 * sum(ratios))))
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_obs_null_stats(obs_null: dict, save_path: str):
    """One row per (map kind, arena), one column per statistic: null histogram, real value red."""
    rows = [(kind, arena, entry['compare'][kind]) for kind in _KDE_MAP_KINDS
            for arena, entry in obs_null.items() if entry['compare'].get(kind) is not None]
    if not rows:
        return
    stats = list(_NULL_SCALAR_STATS)
    fig, axes = plt.subplots(len(rows), len(stats), figsize=(3.3 * len(stats), 2.5 * len(rows)),
                             squeeze=False)
    for i, (kind, arena, res) in enumerate(rows):
        for j, s in enumerate(stats):
            ax = axes[i, j]
            vals = np.asarray(res['stats_null'][s], dtype=float)
            vals = vals[np.isfinite(vals)]
            # a (near-)constant null (spread at floating-point noise level) cannot hold 40 bins
            degenerate = vals.size == 0 or np.ptp(vals) <= 1e-9 * max(1.0, np.abs(vals).max())
            if vals.size:
                ax.hist(vals, bins=1 if degenerate else 40, color='0.65')
            if degenerate:
                ax.text(0.02, 0.95, 'null ~constant', transform=ax.transAxes, fontsize=7,
                        va='top', color='0.3')
            ax.axvline(res['stats_obs'][s], color='red', lw=1.8)
            ax.set_title(f'{kind} | {_ARENA_TITLES[arena]}\n{s}: p = {res["pvals"][s]["p"]:.3g} '
                         f'({_NULL_SCALAR_STATS[s]})', fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(f'Null distributions ({NULL_SPIKE_MODE} spike null, grey) vs real (red)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_obs_null_dist_profiles(obs_null: dict, save_path: str):
    """Method 2: KDE mass per distance-to-wall band, real vs null mean and 95 % envelope."""
    arenas = [a for a in _ARENA_ORDER if a in obs_null]
    kinds = list(_KDE_MAP_KINDS)
    if not arenas:
        return
    fig, axes = plt.subplots(len(kinds), len(arenas), figsize=(5.5 * len(arenas), 3.6 * len(kinds)),
                             squeeze=False)
    for i, kind in enumerate(kinds):
        for j, arena in enumerate(arenas):
            ax, res = axes[i, j], obs_null[arena]['compare'].get(kind)
            if res is None:
                _no_data(ax, arena, 'no comparison')
                continue
            x = res['prof_centres']
            ax.fill_between(x, res['prof_null_lo'], res['prof_null_hi'], color='0.8',
                            label='null 2.5-97.5 %')
            ax.plot(x, res['prof_null_mean'], 'k--', lw=1.2, label='null mean')
            ax.plot(x, res['prof_real'], 'r-', lw=1.8, label='real')
            ax.set_xlabel('distance to wall (cm)')
            ax.set_ylabel('fraction of KDE mass')
            pv = res['pvals']
            ax.set_title(f'{_KDE_MAP_KINDS[kind][1]} | {_ARENA_TITLES[arena]}\n'
                         f'mean dist {res["stats_obs"]["MeanDist2Wall_cm"]:.2f} cm '
                         f'(p = {pv["MeanDist2Wall_cm"]["p"]:.3g}), W1 p = '
                         f'{pv["Dist2Wall_W1_cm"]["p"]:.3g}', fontsize=8)
            ax.legend(fontsize=7)
    fig.suptitle('Distance-to-wall profiles of the KDEs: real vs occupancy-matched spike null')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


# ============================================================================
# Export
# ============================================================================

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


# ============================================================================
# __main__ Pipeline
# ============================================================================

def run_full_pipeline(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    arena_handlers = {key: make_handler(ARENA_CONFIGS[key]) for key in _ARENA_ORDER}
    arena_sessions = {key: find_arena_sessions(key, arena_handlers[key]) for key in _ARENA_ORDER}
    for key in _ARENA_ORDER:
        print(f'[{key}] {len(arena_sessions[key])} sessions, '
              f'{sum(len(s[3]) for s in arena_sessions[key])} units')

    # 1. Mean rate maps: overall, field-only, peak proportion
    arena_results = {}
    for key in _ARENA_ORDER:
        arena_results[key] = collect_arena_results(arena_handlers[key], arena_sessions[key])
        print(f'[{key}] {len(arena_results[key])} place cells pooled')
    plot_fig_S1H(arena_handlers, arena_results, os.path.join(out_dir, 'WholeRM_MeanFieldIndex.png'))
    plot_field_only_mean_maps(arena_handlers, arena_results, os.path.join(out_dir, 'FieldOnly_MeanFieldIndex.png'))
    plot_peak_proportion_maps(arena_handlers, arena_results, os.path.join(out_dir, 'PeakProportion_Map.png'))

    # 2. KDE of every pooled map
    kdes = {'real': {}, 'null': {}}
    for kind, (*_, stem) in _KDE_MAP_KINDS.items():
        kdes['real'][kind] = plot_kde_maps(arena_handlers, arena_results, kind,
                                           os.path.join(out_dir, f'KDE_{stem}.png'))

    # 3. Observed occupancy-matched spike null (fills kdes['null'] for stage 4)
    # 4. Duong (2013) local test: real KDE vs the KDE of the mean null population map
    if RUN_OBS_NULL:
        obs_null, n_null_cells = run_obs_null(arena_handlers, arena_results, kdes)
        plot_combined_occupancy(arena_handlers, obs_null, os.path.join(out_dir, 'ObsNull_CombinedOccupancy.png'))
        for kind in _KDE_MAP_KINDS:
            plot_obs_null_kind(kind, arena_handlers, arena_results, obs_null,
                               os.path.join(out_dir, f'ObsNull_{kind}.png'))
        plot_obs_null_stats(obs_null, os.path.join(out_dir, 'ObsNull_Statistics.png'))
        plot_obs_null_dist_profiles(obs_null, os.path.join(out_dir, 'ObsNull_DistToWall.png'))
        export_obs_null(arena_handlers, obs_null, out_dir)
        run_duong_tests(arena_handlers, kdes, arena_results, n_null_cells, out_dir)

    export_excel(arena_handlers, arena_results, os.path.join(out_dir, 'AllArenas_Summary.xlsx'))


if __name__ == '__main__':
    run_full_pipeline(os.path.join(OUTPUT_DIR, 'AllArenas'))
    print('\nDone.')
