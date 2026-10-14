# -*- coding: utf-8 -*-
"""
Mean Rate Map Analysis (after Muessig et al.) + KDE + METHOD 2 bin-shuffle bootstrap of the
Duong (2013) local test

PIPELINE (run_full_pipeline), per arena:
  1. MEAN MAPS -- the pooled maps of the recorded place cells (described below): overall mean
     field index, field-only mean field index and peak proportion.
  2. KDE -- a 2D Gaussian KDE (scipy gaussian_kde, Scott's rule, edge-corrected over the bins the
     cells validly sampled) of each of the three pooled maps.
  3. BIN-SHUFFLE BOOTSTRAP, METHOD 2 (run_shuffle_duong; described in detail directly below) --
     each pooled map's bins shuffled N_SHUFFLE times, every shuffled map fitted with the same KDE
     and compared with the real KDE by the Duong test (Duong 2013, "Local significant differences
     from nonparametric two-sample tests", J. Nonparametric Statistics 25:3, 635-645; see the
     DUONG section below). Regions where the real density is consistently significantly above
     the shuffles are outlined in GREEN_ABOVE, consistently below in MAGENTA_BELOW.
  4. RAYLEIGH (v27_5; open field and circular track; run right after stage 1) -- each of the three
     pooled maps is averaged in 12 deg sectors and the sector magnitudes are tested with a unimodal
     and a bimodal (axial) Rayleigh test. Open field: the map is first multiplied by a wall
     multiplier (0 at the centre, 1 at the wall) and sectors are averaged with an area correction
     (ring-equalised sector means). Circular track: plain sector means, no multiplier or correction.
     See the RAYLEIGH section. Outputs Rayleigh_OpenField.png / .xlsx and
     Rayleigh_CircularTrack.png / .xlsx (sheets Tests, Sectors).
  5. LINEAR TRACK ARM PREFERENCE (v27_5; after all session-type runs) -- per linear session type
     (0 / 180 = horizontal arm, 90 / 270 = vertical arm of the + maze) x map kind, the Method 2
     consistently-above (green) and -below (magenta) bins are split at the middle of the track:
     topographic index (left - right or top - down) and magnetic north index (mN side - mS side),
     each / number of significant bins, in the raw tracking frame. See the ARM PREFERENCE section.
     Outputs OUTPUT_DIR/SessionType/Linear_ArmPreference_<kind>.png and Linear_ArmPreference.xlsx.

METHOD 2 -- BIN-SHUFFLE BOOTSTRAP OF THE DUONG TEST (stage 3), per map kind x arena:
  Step 1 -- shuffle (shuffle_map_values): the values the KDE is fitted to (the pooled map inside
     the KDE domain = the bins the cells validly sampled, unsampled field-only / peak bins = 0) are
     randomly permuted across the domain bins. The value distribution and the domain are kept;
     where each value sits is randomised.
  Step 2 -- the shuffled map is fitted with exactly the real KDE (kde_density_map: Scott's rule,
     same edge correction, same domain).
  Step 3 -- Duong test, real KDE vs shuffled KDE (duong_compare_arena, Hochberg at DUONG_ALPHA):
     every evaluation bin is significantly ABOVE, significantly BELOW or not different.
  Step 4 -- repeat N_SHUFFLE (1000) times. Per bin: f_above = fraction of shuffles in which it was
     significantly above, f_below likewise.
  Step 5 -- consistency: a bin is consistently ABOVE (green) if f_above >= SHUFFLE_CONSISTENCY,
     consistently BELOW (magenta) if f_below >= SHUFFLE_CONSISTENCY. The default 0.95 is "more than
     95 % of the shuffles", per direction (two directions -> 2.5 % / 97.5 % if you read it as a
     95 % interval of the sign that excludes "not different": set 0.975 for that stricter rule).
     Consistent bins are split into 8-connected clusters (wrapping on the circular track).
  Also reported: a pointwise percentile envelope, i.e. bins where the real (renormalised) density
     lies above the 97.5th or below the 2.5th percentile of the N_SHUFFLE shuffled densities.
What this null tests: shuffling removes all spatial structure but not occupancy -- the shuffled
map is uniform in expectation over the sampled bins, whatever the animal's dwell time there. It
therefore asks "is there more / less mass here than a spatially random map with the same values
would give?" (clustering), not "more than occupancy would give" (see Method 1 for that).

Fig S1H: overall mean field-index map per arena, fine spatial bins (2 x 2 cm; genuinely 2D for
every arena, including the linear track's length x width). Each cell's map is first normalized
0-1 to its own peak and minimum ADAPTIVE-BINNED firing rate (field_index_map, place-cell method,
as in ThetaMod_PhasePrecession_Stats_v3.py) before pooling across cells -- this keeps a cell's
contribution from being over- or under-weighted by its absolute peak rate or by where its field
happens to sit in the map, which raw-Hz pooling does not.

ADAPTIVE-BINNED RATE MAP (v3; replaces the Gaussian-smoothed map of v2 everywhere): Yartsev &
Ulanovsky 2013 / Grieves rate_mapper.m 'yadaptive', ported from PlaceCell_Main_v18_AdptBin_Bin2.py
(the pipeline that qualified these cells). For every valid bin a disc (radius in bin units,
r^2 = 0, 1, 2, ...) grows until the occupancy it encloses is >= ADAPTIVE_DISC_OCC_S, or r^2 reaches
ADAPTIVE_MAX_RSQ - 1; the bin's rate = spikes in disc / occupancy in disc. Only valid bins contribute
to a disc; on the circular track the disc wraps around the ring. Bins whose final disc holds
< ADAPTIVE_MIN_OCC_S are dropped from the cell's valid bins. This map feeds the field-index map,
the field-only map (place-field mask), the peak-proportion map (peak bin), peak / mean rate, SIR,
sparsity and the SIR bootstrap. Valid bins follow the same criteria as that pipeline (visits >=
min_visits by default; see use_min_occ_s / use_min_visits) -- with a 1 s occupancy floor every
valid bin would already hold >= ADAPTIVE_DISC_OCC_S and the adaptive map would equal the raw map.

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

SESSION TYPES (v27_4): within each arena the session type is detected from the label of the
session's last folder (_detect_session_type; same rules as ArenaSessionType_StatsComparison_v8.py):
  open_field     : Cntrl, Rotate, Zero          (e.g. '1Cntrl', '2Rotate')
  linear_track   : 0, 90, 180, 270              (digits after the last '_', e.g. '3_180')
  circular_track : Stnd, NoRot, Rot             (e.g. '2NoRot', '4Stnd'; 'NoRot' tested first)
Every cell's rate map is computed once; the whole pipeline (mean maps, KDE, Method 2 shuffle,
summary workbook) is then run on all cells of each arena (OUTPUT_DIR/AllArenas, as before;
RUN_ALL_ARENAS) and separately on the cells of every session type (OUTPUT_DIR/SessionType/
<Arena>_<type>, e.g. Linear_180; RUN_SESSION_TYPES). Sessions whose type is not recognised stay
in AllArenas and are left out of the session-type runs (listed in a warning).

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
only the coordinate transform, adaptive-disc wrap-around and plotting differ per arena.

Outputs (OUTPUT_DIR/AllArenas, and the same files, for that one arena, in every
OUTPUT_DIR/SessionType/<Arena>_<type> folder):
  Mean maps      WholeRM_MeanFieldIndex.png, FieldOnly_MeanFieldIndex.png, PeakProportion_Map.png,
                 AllArenas_Summary.xlsx (<Arena>_<type>_Summary.xlsx in a session-type folder)
  KDE            KDE_<stem>.png / .npz, stem = FigS1H_MeanFieldIndex, FieldOnly_MeanFieldIndex,
                 PeakProportion_Map
  Method 2       ShuffleDuong_<kind>.png (rows = arenas; real KDE, one shuffled map, mean shuffled
                 KDE, signed consistency f_above - f_below, consistent clusters, pointwise
                 envelope), ShuffleDuong_Clusters.xlsx (sheets Tests, Clusters),
                 ShuffleDuong_Results.npz
"""

import os
import re
import textwrap
import functools
import hashlib
import random
import time
import concurrent.futures
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csr_matrix
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

ROOT_DIRECTORY = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\PC_True_SiAdptBin_CI99_Bin2'
OUTPUT_DIR = os.path.join(ROOT_DIRECTORY, 'MeanRM_Quad_M2_BinShuffle_AdptBin')

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

# Run the pipeline on all cells of each arena (OUTPUT_DIR/AllArenas) and/or separately on the
# cells of every session type (OUTPUT_DIR/SessionType/<Arena>_<type>).
RUN_ALL_ARENAS    = True
RUN_SESSION_TYPES = True

# Session types within each arena, in output order (as in ArenaSessionType_StatsComparison_v8.py).
ARENA_SESSION_TYPES = {
    'open_field':     ['Cntrl', 'Rotate', 'Zero'],
    'linear_track':   ['0', '90', '180', '270'],
    'circular_track': ['Stnd', 'NoRot', 'Rot'],
}
ARENA_FOLDER_NAMES = {'open_field': 'Open', 'linear_track': 'Linear', 'circular_track': 'Circle'}

def _detect_session_type(session_path: str, arena_key: str):
    """Session type named in the last folder of session_path ('1Cntrl' -> 'Cntrl' for the open
    field, '3_180' -> '180' for the linear track, '2NoRot' -> 'NoRot' for the circular track), or
    None if it can't be recognised."""
    label = re.split(r'[\\/]+', str(session_path).strip())[-1].strip().lower()
    if arena_key == 'open_field':
        for t in ARENA_SESSION_TYPES['open_field']:
            if t.lower() in label:
                return t
    elif arena_key == 'linear_track':
        m = re.search(r'_(270|180|90|0)$', label)
        if m:
            return m.group(1)
    elif arena_key == 'circular_track':
        # 'norot' must be tested before 'rot', which it contains.
        if 'norot' in label:
            return 'NoRot'
        if 'stnd' in label:
            return 'Stnd'
        if 'rot' in label:
            return 'Rot'
    return None


def _session_type_label(arena_key: str, session_type: str) -> str:
    """'Linear_180', 'Open_Cntrl', ... -- the session-type folder name."""
    return f'{ARENA_FOLDER_NAMES[arena_key]}_{session_type}'

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

# Valid-bin criteria (as in PlaceCell_Main_v18_AdptBin_Bin2.py). A bin is valid only if it is
# inside the arena, has non-zero occupancy and passes every enabled criterion below.
use_min_occ_s  = False   # True: exclude bins with < min_occ_s seconds occupancy
min_occ_s      = 1       #       (with >= ADAPTIVE_DISC_OCC_S the adaptive map = the raw map)
use_min_visits = True    # True: exclude bins visited < min_visits times
min_visits     = 2       #       a "visit" = one contiguous entry into the bin

# Adaptive binning (Yartsev & Ulanovsky 2013; Grieves rate_mapper.m 'yadaptive')
ADAPTIVE_DISC_OCC_S = 1.0     # occupancy (s) a disc must enclose
ADAPTIVE_MAX_RSQ    = 200     # disc stops growing at r^2 = ADAPTIVE_MAX_RSQ - 1 (bins^2)
ADAPTIVE_MIN_OCC_S  = 0.5     # bins whose final disc holds < this occupancy (s) become invalid

# SIR / sparsity sums restricted to valid bins with >= SI_MIN_OCC_S raw occupancy (real data and
# every bootstrap shuffle alike), as in PlaceCell_Main_v18_AdptBin_Bin2.py.
USE_SI_MIN_OCC = True
SI_MIN_OCC_S   = 0.5

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

FIELD_PEAK_FRAC = 0.20
MIN_FIELD_BINS  = 9

KDE_BW_METHOD = 'scott'
KDE_EDGE_CORRECTION = True

COORD_UNITS = 'cm'

# --- Method 2 bin-shuffle bootstrap of the Duong test (stage 3; see the header) ---
N_SHUFFLE           = 1000
SHUFFLE_CONSISTENCY = 0.95     # fraction of shuffles a bin must be significant in, per direction
SHUFFLE_ENVELOPE    = (2.5, 97.5)   # percentiles of the pointwise shuffle envelope
SHUFFLE_SEED        = 20261006
SHUFFLE_MAX_WORKERS = 4        # worker processes over the (map kind, arena) jobs (1 = serial)

# --- Duong (2013) local test, real KDE vs null KDE ---
DUONG_ALPHA            = 0.05     # family-wise level per (map kind, arena), Hochberg step-up
DUONG_SAMPLE_SIZE_MODE = 'neff'   # 'neff' or 'cells' (see the DUONG section)
DUONG_RENORMALISE_ON_COMMON_DOMAIN = True
DUONG_MIN_CLUSTER_BINS = 1        # clusters smaller than this are not outlined / reported
GREEN_ABOVE   = '#39FF14'         # fluorescent green: real significantly ABOVE null
MAGENTA_BELOW = '#FF00FF'         # fluorescent magenta: real significantly BELOW null
OUTLINE_LW    = 2.2
R_K_GAUSS_2D  = 1.0 / (4.0 * np.pi)   # R(K) = integral of K^2 for the standard bivariate normal

# --- Rayleigh test of the mean maps (open field: wall-weighted; circular track; see the RAYLEIGH section) ---
RUN_RAYLEIGH             = True
RAYLEIGH_SECTOR_DEG      = 12.0    # angular sector width (30 sectors)
RAYLEIGH_RING_WIDTH_CM   = 6.0     # radial ring width of the area correction (5 rings in the 60 cm field)
RAYLEIGH_AREA_CORRECTION = True    # sector value plotted / primary: True = ring-equalised, False = plain bin mean
RAYLEIGH_N_PERM          = 10000   # sector-permutation null of the mean resultant length
RAYLEIGH_ALPHA           = 0.05    # on p_perm, for the unimodal / bimodal / none pattern call
RAYLEIGH_SEED            = 20261013

# --- Linear track arm preference of the Duong-consistent bins (see the ARM PREFERENCE section) ---
RUN_LINEAR_ARM_PREFERENCE = True    # needs RUN_SESSION_TYPES
# Arm of the + maze and magnetic-north side of every linear session type, in the raw tracking
# frame (x to the right, y UP): 0 / 180 = horizontal (-) arm, 90 / 270 = vertical (|) arm.
LINEAR_ARM    = {'0': 'horizontal', '180': 'horizontal', '90': 'vertical', '270': 'vertical'}
LINEAR_MN_SIDE = {'0': 'left', '180': 'right', '90': 'top', '270': 'bottom'}

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
# Valid bins + adaptive binning (Yartsev & Ulanovsky 2013; Grieves rate_mapper.m 'yadaptive')
# ============================================================================

def _bin_visit_counts(flat_idx: np.ndarray, n_bins: int) -> np.ndarray:
    """Number of visits per bin: a visit starts whenever the animal enters a bin
    (first frame, or bin index differs from the previous frame)."""
    flat_idx = np.asarray(flat_idx)
    if flat_idx.size == 0:
        return np.zeros(n_bins, dtype=np.int64)
    starts = np.ones(flat_idx.size, dtype=bool)
    starts[1:] = flat_idx[1:] != flat_idx[:-1]
    return np.bincount(flat_idx[starts], minlength=n_bins)


def _valid_bin_mask(occ_map: np.ndarray, visits: np.ndarray, handler) -> np.ndarray:
    """Valid bins under the enabled criteria (use_min_occ_s / use_min_visits), inside the arena."""
    mask = (occ_map > 0) & handler.geom_valid
    if use_min_occ_s:
        mask &= occ_map >= min_occ_s
    if use_min_visits:
        mask &= visits >= min_visits
    return mask


@functools.lru_cache(maxsize=8)
def _adaptive_disc_geometry(nx: int, ny: int, wrap_x: bool, max_rsq: int) -> tuple:
    """Disc geometry for _adaptive_operator, cached per grid.

    Returns (target, n_enclosed):
      target     : (n_offsets, n_bins) flat index of every bin shifted by the k-th offset, offsets
                   sorted by squared distance (so a cumsum over rows = disc sums); -1 where the
                   shifted bin falls off the grid. With wrap_x the first axis wraps (circular track).
      n_enclosed : (max_rsq,) number of offsets with d^2 <= rsq, rsq = 0..max_rsq-1.
    """
    rad = int(np.floor(np.sqrt(max_rsq - 1)))
    ax = np.arange(-rad, rad + 1)
    dx, dy = np.meshgrid(ax, ax, indexing='ij')
    dx, dy = dx.ravel(), dy.ravel()
    d2 = dx ** 2 + dy ** 2
    keep = d2 <= max_rsq - 1
    order = np.argsort(d2[keep], kind='stable')
    dx, dy, d2 = dx[keep][order], dy[keep][order], d2[keep][order]

    bx, by = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    tx = bx.ravel()[None, :] + dx[:, None]
    ty = by.ravel()[None, :] + dy[:, None]
    if wrap_x:
        tx = tx % nx
        on_grid = (ty >= 0) & (ty < ny)
    else:
        on_grid = (tx >= 0) & (tx < nx) & (ty >= 0) & (ty < ny)
    target = np.where(on_grid, tx * ny + ty, -1)

    n_enclosed = np.searchsorted(d2, np.arange(max_rsq), side='right')
    return target, n_enclosed


def _adaptive_operator(occ_map: np.ndarray, valid: np.ndarray, handler) -> dict:
    """Adaptive-binning discs for one cell's occupancy (they depend on occupancy only, so the SIR
    bootstrap reuses them for every shuffle).

    For every valid bin a disc (radius in bin units, r^2 = 0, 1, 2, ...) grows until
    occ_disc >= ADAPTIVE_DISC_OCC_S, or r^2 reaches ADAPTIVE_MAX_RSQ - 1. Only valid bins contribute
    to a disc. Evaluated for all radii at once from cumulative disc sums, then the first radius
    meeting the criterion is taken -- identical to growing r^2 one step at a time.

    Returns dict(W, occ_disc, ok): W (n_bins x n_bins, sparse) sums a valid-only per-bin map over
    each bin's final disc; occ_disc = occupancy in that disc; ok = valid bins whose disc holds
    >= ADAPTIVE_MIN_OCC_S (the adaptive map's valid bins).
    """
    n_bins = handler.n_bins
    target, n_enclosed = _adaptive_disc_geometry(handler.nx, handler.ny, bool(handler.wrap_x),
                                                 ADAPTIVE_MAX_RSQ)
    on_grid = target >= 0
    occ_v = np.where(valid, occ_map, 0.0)
    occ_g = np.where(on_grid, occ_v[np.where(on_grid, target, 0)], 0.0)
    occ_cum = np.cumsum(occ_g, axis=0)[n_enclosed - 1]           # (max_rsq, n_bins)

    satisfied = occ_cum >= ADAPTIVE_DISC_OCC_S
    k = np.where(satisfied.any(axis=0), satisfied.argmax(axis=0), ADAPTIVE_MAX_RSQ - 1)
    cols = np.arange(n_bins)
    occ_disc = occ_cum[k, cols]
    ok = valid & (occ_disc >= ADAPTIVE_MIN_OCC_S)

    in_disc = (np.arange(target.shape[0])[:, None] < n_enclosed[k][None, :]) & on_grid & ok[None, :]
    rows = np.broadcast_to(cols, target.shape)[in_disc]
    W = csr_matrix((np.ones(rows.size), (rows, target[in_disc])), shape=(n_bins, n_bins))
    return dict(W=W, occ_disc=occ_disc, ok=ok)


def _adaptive_rate(spike_map: np.ndarray, valid: np.ndarray, op: dict) -> np.ndarray:
    """Adaptive-binned rate = spikes in disc / occupancy in disc (0 outside op['ok'])."""
    spk_disc = op['W'] @ np.where(valid, spike_map, 0.0)
    rate = np.zeros_like(spk_disc)
    ok = op['ok']
    rate[ok] = spk_disc[ok] / op['occ_disc'][ok]
    return rate


def _si_bin_mask(occ_map: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Bins entering the SIR / sparsity sums (see USE_SI_MIN_OCC), selected on RAW occupancy."""
    if USE_SI_MIN_OCC:
        return valid & (occ_map >= SI_MIN_OCC_S)
    return valid


def _sir_sparsity(rate: np.ndarray, occ_map: np.ndarray, valid: np.ndarray) -> tuple:
    """Skaggs SIR (bits/spike) and sparsity of the adaptive-binned map, pi = raw occupancy
    normalised over the bins in the sum. Returns (sir, sparsity)."""
    m = _si_bin_mask(occ_map, valid)
    if not m.any() or occ_map[m].sum() <= 0:
        return 0.0, 0.0
    pi = occ_map[m] / occ_map[m].sum()
    ri = rate[m]
    r_mean = float(np.sum(pi * ri))
    sir = 0.0
    if r_mean > 0:
        nz = ri > 0
        ratio = ri[nz] / r_mean
        sir = float(np.sum(pi[nz] * ratio * np.log2(ratio)))
    spar_den = float(np.sum(pi * ri ** 2))
    sparsity = float(r_mean ** 2 / spar_den) if spar_den > 0 else 0.0
    return sir, sparsity


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
    """Place-field bins on the adaptive-binned map: rate > the cell's mean rate and
    > FIELD_PEAK_FRAC x its peak, in 8-connected components of >= MIN_FIELD_BINS bins."""
    valid    = cell['valid']
    fr_adapt = cell['fr_adapt']
    field_mask = np.zeros(handler.n_bins, dtype=bool)
    if not valid.any():
        return field_mask

    peak_adapt = float(fr_adapt[valid].max())
    mean_fr    = cell['mean_fr']
    qualifies = valid & (fr_adapt > mean_fr) & (fr_adapt > FIELD_PEAK_FRAC * peak_adapt)
    if not qualifies.any():
        return field_mask

    for component in handler.connected_components(qualifies):
        if len(component) >= MIN_FIELD_BINS:
            field_mask[component] = True
    return field_mask


def rate_maps_from_counts(spike_map: np.ndarray, occ_map: np.ndarray, visits: np.ndarray,
                          handler) -> dict:
    """Raw rate map, adaptive-binned rate map, field-index map and peak / mean rate, SIR and
    sparsity (all from the adaptive-binned map) from per-bin spike counts, occupancy (s) and
    visits. 'valid_occ' = bins passing the valid-bin criteria (_valid_bin_mask); 'valid' = the
    subset whose adaptive disc reached ADAPTIVE_MIN_OCC_S -- the bins every later analysis uses."""
    n_bins = handler.n_bins
    valid_occ = _valid_bin_mask(occ_map, visits, handler)

    fr_raw = np.zeros(n_bins, dtype=np.float64)
    fr_raw[valid_occ] = spike_map[valid_occ] / occ_map[valid_occ]

    adapt_op = _adaptive_operator(occ_map, valid_occ, handler)
    valid = adapt_op['ok']
    fr_adapt = _adaptive_rate(spike_map, valid_occ, adapt_op)
    fi_map = field_index_map(fr_adapt, valid)

    result = dict(fr_raw=fr_raw, fr_adapt=fr_adapt, fi_map=fi_map, occ_map=occ_map,
                  valid=valid, valid_occ=valid_occ, adapt_op=adapt_op)
    if not valid.any():
        result.update(peak_fr=0.0, mean_fr=0.0, sir=0.0, sparsity=0.0)
        return result

    pi = occ_map[valid] / occ_map[valid].sum()
    mean_fr = float(np.sum(pi * fr_adapt[valid]))
    peak_fr = float(fr_adapt[valid].max())
    sir, sparsity = _sir_sparsity(fr_adapt, occ_map, valid)

    result.update(peak_fr=round(peak_fr, 4), mean_fr=round(mean_fr, 4),
                  sir=round(sir, 4), sparsity=round(sparsity, 4))
    return result


def _cell_peaks_and_field(cell: dict, handler) -> dict:
    """Peak bin of the adaptive-binned map and the place-field mask, as pooled below."""
    peak_bin = None
    if cell['valid'].any():
        peak_bin = int(np.argmax(np.where(cell['valid'], cell['fr_adapt'], -np.inf)))
    return dict(peak_bin=peak_bin, field_mask=extract_place_field_mask(cell, handler))


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
    else:
        spike_frame = nearest[valid_spike]

    n_bins = handler.n_bins
    occ_map   = np.zeros(n_bins, dtype=np.float64)
    spike_map = np.zeros(n_bins, dtype=np.float64)
    np.add.at(occ_map,   bin_idx, dt_frames)
    np.add.at(spike_map, bin_idx[spike_frame], 1.0)
    visits = _bin_visit_counts(bin_idx, n_bins)

    result = rate_maps_from_counts(spike_map, occ_map, visits, handler)
    result.update(n_spikes=n_spikes, n_spikes_speed_excluded=n_spikes_speed_excluded,
                  bin_idx=bin_idx, spike_frame=spike_frame, t=t, n_bins=n_bins)
    return result


def run_bootstrap_generic(handler, cell: dict, real_sir: float, n_bootstrap: int = N_BOOTSTRAP,
                          seed: int | None = None) -> dict:
    bin_idx     = cell['bin_idx']
    spike_frame = cell['spike_frame']
    t           = cell['t']
    occ_map     = cell['occ_map']
    valid       = cell['valid']
    valid_occ   = cell['valid_occ']
    adapt_op    = cell['adapt_op']     # discs depend on occupancy only -> same for every shuffle
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

        fr_adapt = _adaptive_rate(spike_map, valid_occ, adapt_op)
        sir_i[i] = _sir_sparsity(fr_adapt, occ_map, valid)[0]

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
    visits = _bin_visit_counts(frames['bin_idx'][occ], handler.n_bins)
    valid_mask = _valid_bin_mask(occ_map, visits, handler)
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
        fr_raw=cell['fr_raw'], fr_adapt=cell['fr_adapt'], fi_map=cell['fi_map'], valid=cell['valid'], occ_map=cell['occ_map'],
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
        if r['peak_bin'] is not None:
            counts[r['peak_bin']] += 1.0

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
# Duong (2013) local significant differences, real KDE vs null KDE (used by stage 3)
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
#              overall and field-only maps the weighted points are BINS of an adaptive-binned map, which
#              are spatially correlated, so neff overstates the independent information there ->
#              the test is liberal for those two map kinds; check with 'cells'.
#   'cells' -- n1 = number of real place cells pooled, n2 = number of null cells pooled.
#              Conservative for the bin-weighted maps.

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


# ============================================================================
# STAGE 3 -- METHOD 2 bin-shuffle bootstrap of the Duong test (see the header)
# ============================================================================

def shuffle_map_values(values: np.ndarray, domain: np.ndarray, rng) -> np.ndarray:
    """Step 1. The values the KDE is fitted to (NaN -> 0 inside the domain) randomly permuted
    across the domain bins."""
    out = np.where(domain & np.isfinite(values), values, 0.0)
    idx = np.flatnonzero(domain)
    out[idx] = out[idx][rng.permutation(len(idx))]
    return out


def shuffle_duong_job(handler, values: np.ndarray, domain: np.ndarray, values_are_mass: bool,
                      real_kde: dict, n_cells: float, seed, n_iter: int) -> dict:
    """Steps 1-4 for one (map kind, arena): n_iter shuffled maps, their KDEs and their Duong tests
    against the real KDE. Per bin: number of shuffles it was significantly above / below / in the
    common domain, plus every shuffle's renormalised density and signed z (float32)."""
    rng = np.random.default_rng(seed)
    n_bins = handler.n_bins
    n_above = np.zeros(n_bins, dtype=np.int64)
    n_below = np.zeros(n_bins, dtype=np.int64)
    n_common = np.zeros(n_bins, dtype=np.int64)
    f_shuf = np.full((n_iter, n_bins), np.nan, dtype=np.float32)
    z_shuf = np.full((n_iter, n_bins), np.nan, dtype=np.float32)
    f_real = np.full(n_bins, np.nan)
    n_rejected, factors, example = [], [], None
    for it in range(n_iter):
        sv = shuffle_map_values(values, domain, rng)
        shuf_kde = kde_density_map(handler, sv, domain, values_are_mass)
        if shuf_kde is None:
            continue
        if DUONG_SAMPLE_SIZE_MODE == 'neff':
            n1, n2 = real_kde['neff'], shuf_kde['neff']
        elif DUONG_SAMPLE_SIZE_MODE == 'cells':
            n1 = n2 = n_cells
        else:
            raise ValueError(f'Unknown DUONG_SAMPLE_SIZE_MODE {DUONG_SAMPLE_SIZE_MODE!r}')
        res = duong_compare_arena(handler, real_kde, shuf_kde, n1, n2)
        common = res['common']
        idx = np.flatnonzero(common)
        n_common[common] += 1
        n_above[idx[res['reject'] & (res['diff'] > 0)]] += 1
        n_below[idx[res['reject'] & (res['diff'] < 0)]] += 1
        f_shuf[it, common] = res['f_null']
        z_shuf[it, common] = res['z']
        f_real[common] = res['f_real']
        n_rejected.append(int(res['reject'].sum()))
        factors.append(shuf_kde['factor'])
        if example is None:
            example = sv
    return dict(n_above=n_above, n_below=n_below, n_common=n_common, f_shuf=f_shuf, z_shuf=z_shuf,
                f_real=f_real, n_rejected=np.asarray(n_rejected), factors=np.asarray(factors),
                example=example, domain=domain, real_factor=real_kde['factor'])


def summarise_shuffle(handler, job: dict) -> dict:
    """Step 5. Fractions of shuffles significant in each direction, consistent bins and their
    clusters, and the pointwise percentile envelope of the shuffled densities."""
    seen = job['n_common'] > 0
    f_above = np.where(seen, job['n_above'] / np.maximum(job['n_common'], 1), np.nan)
    f_below = np.where(seen, job['n_below'] / np.maximum(job['n_common'], 1), np.nan)
    cons_above = seen & (f_above >= SHUFFLE_CONSISTENCY)
    cons_below = seen & (f_below >= SHUFFLE_CONSISTENCY)
    lab_above, n_cl_above = _duong_cluster_labels(handler, cons_above)
    lab_below, n_cl_below = _duong_cluster_labels(handler, cons_below)
    with warnings.catch_warnings():   # bins outside every common domain are all-NaN, as intended
        warnings.simplefilter('ignore', RuntimeWarning)
        env_lo, env_hi = np.nanpercentile(job['f_shuf'], SHUFFLE_ENVELOPE, axis=0)
        f_shuf_mean = np.nanmean(job['f_shuf'], axis=0)
        z_lo, z_med, z_hi = np.nanpercentile(job['z_shuf'], (SHUFFLE_ENVELOPE[0], 50.0,
                                                             SHUFFLE_ENVELOPE[1]), axis=0)
    f_real = job['f_real']
    return dict(job, seen=seen, f_above=f_above, f_below=f_below,
                cons_above=cons_above, cons_below=cons_below,
                lab_above=lab_above, n_above_cl=n_cl_above, lab_below=lab_below, n_below_cl=n_cl_below,
                env_lo=env_lo, env_hi=env_hi, f_shuf_mean=f_shuf_mean, z_lo=z_lo, z_med=z_med, z_hi=z_hi,
                env_above=seen & (f_real > env_hi), env_below=seen & (f_real < env_lo))


def shuffle_cluster_rows(kind: str, arena: str, handler, s: dict) -> list:
    xy, area_all = handler.bin_centres_xy(), handler.bin_areas_cm2()
    rows = []
    for direction, labels, n, frac in (('real > shuffle', s['lab_above'], s['n_above_cl'], s['f_above']),
                                       ('real < shuffle', s['lab_below'], s['n_below_cl'], s['f_below'])):
        for k in range(n):
            b = np.flatnonzero(labels == k)
            a = area_all[b]
            dw = handler.dist_to_wall[b]
            row = dict(map_kind=kind, arena=arena, direction=direction, cluster=k + 1,
                       n_bins=int(len(b)), area_cm2=round(float(a.sum()), 2),
                       centroid_x_cm=round(float(np.sum(xy[0, b] * a) / a.sum()), 2),
                       centroid_y_cm=round(float(np.sum(xy[1, b] * a) / a.sum()), 2),
                       mean_dist_to_wall_cm=round(float(np.sum(dw * a) / a.sum()), 2),
                       min_dist_to_wall_cm=round(float(dw.min()), 2),
                       max_dist_to_wall_cm=round(float(dw.max()), 2),
                       mean_frac_shuffles_significant=round(float(np.mean(frac[b])), 4),
                       min_frac_shuffles_significant=round(float(np.min(frac[b])), 4),
                       mean_real_density=float(np.mean(s['f_real'][b])),
                       mean_shuffle_density=float(np.mean(s['f_shuf_mean'][b])),
                       mean_ratio_real_over_shuffle=round(float(np.mean(s['f_real'][b] / s['f_shuf_mean'][b])), 4),
                       median_z=round(float(np.median(s['z_med'][b])), 3))
            if isinstance(handler, CircularTrackHandler):
                th = np.arctan2(xy[1, b] - handler.cy, xy[0, b] - handler.cx)
                ang = np.arctan2(np.sum(np.sin(th) * a), np.sum(np.cos(th) * a))
                row['centroid_angle_deg'] = round(float(np.degrees(ang) % 360.0), 1)
            rows.append(row)
    return rows


def _run_shuffle_jobs(jobs: dict) -> dict:
    """Run every (map kind, arena) job in SHUFFLE_MAX_WORKERS processes; serial if a process pool
    cannot start (e.g. in some interactive consoles) or SHUFFLE_MAX_WORKERS = 1."""
    if SHUFFLE_MAX_WORKERS > 1 and len(jobs) > 1:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(SHUFFLE_MAX_WORKERS, len(jobs))) as ex:
                futures = {key: ex.submit(shuffle_duong_job, *args) for key, args in jobs.items()}
                return {key: fut.result() for key, fut in futures.items()}
        except Exception as e:
            print(f'[shuffle] process pool failed ({type(e).__name__}: {e}); running serially')
    return {key: shuffle_duong_job(*args) for key, args in jobs.items()}


def run_shuffle_duong(arena_handlers: dict, arena_results: dict, real_kdes: dict, out_dir: str,
                      title_prefix: str = '') -> dict:
    """Steps 1-5 for every map kind x arena with a real KDE; figures, workbook and npz.
    real_kdes = {kind: {arena: kde}} as returned by plot_kde_maps. Returns the consistent bins,
    {(kind, arena): {'cons_above', 'cons_below'}}."""
    jobs = {}
    for k_i, (kind, (pool_fn, *_)) in enumerate(_KDE_MAP_KINDS.items()):
        for a_i, arena in enumerate(_ARENA_ORDER):
            real_kde = real_kdes.get(kind, {}).get(arena)
            if real_kde is None:
                continue
            handler, results = arena_handlers[arena], arena_results[arena]
            values, values_are_mass = pool_fn(handler, results)
            jobs[(kind, arena)] = (handler, values, _union_valid(handler, results), values_are_mass,
                                   real_kde, float(len(results)), [SHUFFLE_SEED, k_i, a_i], N_SHUFFLE)
    if not jobs:
        print('[shuffle] skipped: no (map kind, arena) with a real KDE')
        return {}
    print(f'[shuffle] {len(jobs)} (map kind, arena) jobs x {N_SHUFFLE} shuffles')
    t0 = time.time()
    outputs = _run_shuffle_jobs(jobs)
    print(f'[shuffle] done in {time.time() - t0:.0f} s')

    test_rows, cluster_rows, saved, consistent = [], [], {}, {}
    for kind, (_, title, *_) in _KDE_MAP_KINDS.items():
        summaries = {}
        for arena in _ARENA_ORDER:
            if (kind, arena) not in outputs:
                continue
            handler = arena_handlers[arena]
            s = summarise_shuffle(handler, outputs[(kind, arena)])
            summaries[arena] = s
            consistent[(kind, arena)] = dict(cons_above=s['cons_above'], cons_below=s['cons_below'])
            n_done = len(s['n_rejected'])
            print(f'[shuffle {kind}] {arena}: {n_done} shuffles, {int(s["cons_above"].sum())} bins '
                  f'consistently above ({s["n_above_cl"]} clusters), {int(s["cons_below"].sum())} below '
                  f'({s["n_below_cl"]} clusters)')
            test_rows.append(dict(
                map_kind=kind, arena=arena, n_shuffles=n_done, alpha=DUONG_ALPHA,
                sample_size_mode=DUONG_SAMPLE_SIZE_MODE, consistency=SHUFFLE_CONSISTENCY,
                m_points=int(s['seen'].sum()), n_cells=len(arena_results[arena]),
                real_scott_factor=round(s['real_factor'], 4),
                shuffle_scott_factor_mean=round(float(np.mean(s['factors'])), 4),
                mean_rejected_per_shuffle=round(float(np.mean(s['n_rejected'])), 2),
                frac_shuffles_any_rejection=round(float(np.mean(s['n_rejected'] > 0)), 4),
                n_bins_consistent_above=int(s['cons_above'].sum()),
                n_bins_consistent_below=int(s['cons_below'].sum()),
                n_clusters_above=s['n_above_cl'], n_clusters_below=s['n_below_cl'],
                n_bins_envelope_above=int(s['env_above'].sum()),
                n_bins_envelope_below=int(s['env_below'].sum()),
                max_frac_above=round(float(np.nanmax(s['f_above'])), 4),
                max_frac_below=round(float(np.nanmax(s['f_below'])), 4)))
            cluster_rows += shuffle_cluster_rows(kind, arena, handler, s)

            pre = f'{kind}_{arena}_'
            for name in ('f_real', 'f_shuf_mean', 'env_lo', 'env_hi', 'z_lo', 'z_med', 'z_hi',
                         'f_above', 'f_below', 'cons_above', 'cons_below', 'lab_above', 'lab_below',
                         'env_above', 'env_below', 'n_rejected', 'factors', 'example', 'domain'):
                if s[name] is not None:
                    saved[pre + name] = s[name]
        if summaries:
            plot_shuffle_kind(title_prefix + title, arena_handlers, summaries,
                              os.path.join(out_dir, f'ShuffleDuong_{kind}.png'))

    cluster_cols = ['map_kind', 'arena', 'direction', 'cluster', 'n_bins', 'area_cm2',
                    'centroid_x_cm', 'centroid_y_cm', 'centroid_angle_deg',
                    'mean_dist_to_wall_cm', 'min_dist_to_wall_cm', 'max_dist_to_wall_cm',
                    'mean_frac_shuffles_significant', 'min_frac_shuffles_significant',
                    'mean_real_density', 'mean_shuffle_density', 'mean_ratio_real_over_shuffle',
                    'median_z']
    xlsx = os.path.join(out_dir, 'ShuffleDuong_Clusters.xlsx')
    with pd.ExcelWriter(xlsx) as xw:
        pd.DataFrame(test_rows).to_excel(xw, sheet_name='Tests', index=False)
        pd.DataFrame(cluster_rows).reindex(columns=cluster_cols).to_excel(
            xw, sheet_name='Clusters', index=False)
    print(f'[SAVED] {xlsx}')
    npz = os.path.join(out_dir, 'ShuffleDuong_Results.npz')
    np.savez_compressed(npz, **saved)
    print(f'[SAVED] {npz}')
    return consistent


# ============================================================================
# STAGE 4 -- RAYLEIGH test of the mean maps (open field: wall-weighted; circular track)
# ============================================================================
# Per map kind (overall, field-only, peak proportion), on the pooled map inside the bins the cells
# validly sampled (unsampled field-only bins = 0, as for the KDE).
# CIRCULAR TRACK: no multiplier and no area correction (the track is 4 cm wide, so there is no
#   centre-to-wall growth in bin number): every bin goes, by the angle of its centre, into exactly
#   one RAYLEIGH_SECTOR_DEG sector (4 angular x 2 radial bins per 12 deg sector), the sector value
#   is the plain mean of its bins, and step 4 below is applied as for the open field.
# OPEN FIELD:
#   1. wall multiplier rho = distance of the bin centre from the arena centre / arena radius
#      (0 at the centre, 1 at the wall); weighted value = map value x rho.
#   2. every bin goes, by the angle of its centre, into exactly one RAYLEIGH_SECTOR_DEG sector.
#   3. sector magnitude. A sector is a wedge, so its outer part holds far more bins than its inner
#      part (bins at radius r ~ r): a plain mean over the sector's bins weights the periphery by area
#      on top of rho. AREA CORRECTION (ring-equalised): the sector is cut into RAYLEIGH_RING_WIDTH_CM
#      rings, the weighted values are averaged within each (sector, ring) cell, and the sector value
#      is the mean of its ring means -- every radius counts once, so rho is the only radial weighting.
#      A (sector, ring) cell with no sampled bin (2 sectors have no bin centre in the innermost ring)
#      takes that ring's mean over the other sectors, which adds no directional signal. The plain
#      bin mean is reported alongside.
#   4. Rayleigh test on the sector magnitudes as weights of the sector-centre angles:
#      unimodal -- mean resultant length R of the angles; bimodal (axial, two modes 180 deg apart) --
#      R of the doubled angles, mean axis = half its direction. R is corrected for grouping into
#      sectors (Zar 2010, eq. 26.16). Two p-values:
#        p_rayleigh: Rayleigh Z = n R^2 with n = number of sectors (Zar 2010, eq. 27.4). The weights are
#                    map values, not counts, so n is a convention -- read it as approximate.
#        p_perm:     sector-permutation null -- the sector magnitudes shuffled across sectors
#                    RAYLEIGH_N_PERM times; p = (1 + #null R >= real R) / (N + 1). Unit-free; preferred.
#      pattern: of the tests with p_perm < RAYLEIGH_ALPHA, the one with the larger R (a single sharp
#      peak also gives a significant axial test, so a significant bimodal p alone is not bimodality).
#      Adjacent sectors of a smooth map are correlated and the permutation breaks that, so both p's
#      are somewhat liberal; very small p's are the robust ones.

def _bin_sectors(handler) -> tuple:
    """Per bin: distance from the arena centre (cm) and sector index (-1 outside the arena, by the
    angle of the bin centre, 0 = +x, CCW); and the number of sectors."""
    xy = handler.bin_centres_xy()
    dx, dy = xy[0] - handler.cx, xy[1] - handler.cy
    theta = np.arctan2(dy, dx) % (2.0 * np.pi)
    n_sec = int(round(360.0 / RAYLEIGH_SECTOR_DEG))
    sector = np.minimum((np.degrees(theta) // RAYLEIGH_SECTOR_DEG).astype(int), n_sec - 1)
    sector[~handler.geom_valid] = -1
    return np.hypot(dx, dy), sector, n_sec


def _open_field_polar(handler) -> tuple:
    """Open field, per bin: wall multiplier rho (0 = centre, 1 = wall), sector and ring index (-1
    outside the arena), and the numbers of sectors and rings."""
    r, sector, n_sec = _bin_sectors(handler)
    radius = handler.diameter / 2.0
    rho = np.clip(r / radius, 0.0, 1.0)
    n_ring = int(np.ceil(radius / RAYLEIGH_RING_WIDTH_CM))
    ring = np.minimum((r // RAYLEIGH_RING_WIDTH_CM).astype(int), n_ring - 1)
    ring[~handler.geom_valid] = -1
    return rho, sector, ring, n_sec, n_ring


def rayleigh_sector_profile(handler, values: np.ndarray, ring_equalise: bool) -> dict:
    """Steps 2-3. Sector magnitudes of the map (NaN = not sampled): plain bin mean, and with
    ring_equalise (open field) the ring-equalised mean."""
    _, sector, n_sec = _bin_sectors(handler)
    ok = np.isfinite(values) & (sector >= 0)
    s, x = sector[ok], values[ok]
    n_bins = np.bincount(s, minlength=n_sec)
    plain = np.where(n_bins > 0, np.bincount(s, x, n_sec) / np.maximum(n_bins, 1), np.nan)
    centres = np.radians((np.arange(n_sec) + 0.5) * RAYLEIGH_SECTOR_DEG)
    if not ring_equalise:
        return dict(centres=centres, plain=plain, n_bins=n_bins)

    _, _, ring, _, n_ring = _open_field_polar(handler)
    k = ring[ok]
    cell_sum = np.zeros((n_sec, n_ring))
    cell_n = np.zeros((n_sec, n_ring))
    np.add.at(cell_sum, (s, k), x)
    np.add.at(cell_n, (s, k), 1.0)
    cell_mean = np.where(cell_n > 0, cell_sum / np.maximum(cell_n, 1.0), np.nan)
    with warnings.catch_warnings():   # rings with no sampled bin at all are all-NaN -> dropped
        warnings.simplefilter('ignore', RuntimeWarning)
        ring_mean = np.nanmean(cell_mean, axis=0)
    rings = np.isfinite(ring_mean)
    empty = (cell_n == 0) & rings
    filled = np.where(empty, ring_mean, cell_mean)[:, rings]
    ring_eq = filled.mean(axis=1) if rings.any() else np.full(n_sec, np.nan)
    return dict(centres=centres, ring_equalised=ring_eq, plain=plain, n_bins=n_bins,
                n_imputed=int(empty.sum()), n_rings=int(rings.sum()))


def _resultant(angles: np.ndarray, w: np.ndarray, m: int) -> tuple:
    """Mean resultant length and direction (rad) of the angles x m, weighted by w (w may be
    (n_perm, n) for many weightings at once)."""
    c = w @ np.cos(m * angles)
    s = w @ np.sin(m * angles)
    return np.hypot(c, s) / w.sum(axis=-1), np.arctan2(s, c)


def _rayleigh_p(r: float, n: int) -> float:
    """Rayleigh test p (Zar 2010, eq. 27.4) for mean resultant length r of n angles."""
    rn = n * r
    return float(np.clip(np.exp(np.sqrt(1.0 + 4.0 * n + 4.0 * (n ** 2 - rn ** 2)) - (1.0 + 2.0 * n)),
                         0.0, 1.0))


def rayleigh_test(centres: np.ndarray, mags: np.ndarray, seed) -> dict:
    """Step 4. Unimodal (m = 1) and bimodal-axial (m = 2) Rayleigh tests of the sector magnitudes."""
    ok = np.isfinite(mags)
    a, w = centres[ok], np.clip(mags[ok], 0.0, None)
    n = len(a)
    out = dict(n_sectors=n)
    if n < 3 or w.sum() <= 0:
        return out
    perms = np.argsort(np.random.default_rng(seed).random((RAYLEIGH_N_PERM, n)), axis=1)
    d = np.radians(RAYLEIGH_SECTOR_DEG)
    for m, tag in ((1, 'uni'), (2, 'bi')):
        r, mu = _resultant(a, w, m)
        r_null, _ = _resultant(a, w[perms], m)
        dm = m * d                                   # grouping interval of the m-fold angles
        r_corr = min(1.0, r * (dm / 2.0) / np.sin(dm / 2.0))
        direction = np.degrees(mu / m) % (360.0 / m)
        out.update({f'{tag}_R': r, f'{tag}_R_corr': r_corr, f'{tag}_Z': n * r_corr ** 2,
                    f'{tag}_p_rayleigh': _rayleigh_p(r_corr, n),
                    f'{tag}_p_perm': (1.0 + np.sum(r_null >= r)) / (RAYLEIGH_N_PERM + 1.0),
                    f'{tag}_null_R_95': float(np.percentile(r_null, 95.0)),
                    f'{tag}_dir_deg': direction})
    out['bi_axis2_deg'] = out['bi_dir_deg'] + 180.0
    # A single sharp peak also has an axial component (the bimodal test can be significant for a
    # unimodal profile), so the pattern is the significant test with the larger R.
    sig = [(out[f'{t}_R_corr'], label) for t, label in (('uni', 'unimodal'), ('bi', 'bimodal (axial)'))
           if out[f'{t}_p_perm'] < RAYLEIGH_ALPHA]
    out['pattern'] = max(sig)[1] if sig else 'none'
    return out


def _rayleigh_map(handler, results: list, pool_fn, wall_weighted: bool) -> np.ndarray:
    """Step 1. Pooled map inside the sampled bins (NaN -> 0 there), NaN elsewhere; x the wall
    multiplier if wall_weighted (open field)."""
    values, _ = pool_fn(handler, results)
    domain = _union_valid(handler, results) & handler.geom_valid
    values = np.where(domain, np.where(np.isfinite(values), values, 0.0), np.nan)
    return values * _open_field_polar(handler)[0] if wall_weighted else values


_RAYLEIGH_FILE = {'open_field': 'Rayleigh_OpenField', 'circular_track': 'Rayleigh_CircularTrack'}


def run_rayleigh(arena: str, handler, results: list, out_dir: str, title_prefix: str = ''):
    """Steps 1-4 for the three map kinds of one arena (open field or circular track); figure +
    workbook."""
    open_field = arena == 'open_field'
    methods = ('ring_equalised', 'plain') if open_field else ('plain',)
    primary = 'ring_equalised' if (open_field and RAYLEIGH_AREA_CORRECTION) else 'plain'
    test_rows, sector_rows, panels = [], [], {}
    for k_i, (kind, (pool_fn, title, *_)) in enumerate(_KDE_MAP_KINDS.items()):
        values = _rayleigh_map(handler, results, pool_fn, wall_weighted=open_field)
        prof = rayleigh_sector_profile(handler, values, ring_equalise=open_field)
        stats = {}
        for method in methods:
            st = rayleigh_test(prof['centres'], prof[method], [RAYLEIGH_SEED, k_i])
            stats[method] = st
            row = dict(map_kind=kind, arena=arena, n_cells=len(results), sector_mean=method,
                       primary=method == primary)
            if open_field:
                row.update(n_rings=prof['n_rings'] if method == 'ring_equalised' else None,
                           n_imputed_sector_rings=prof['n_imputed'] if method == 'ring_equalised' else None)
            row.update({key: (round(v, 6) if isinstance(v, float) else v) for key, v in st.items()})
            test_rows.append(row)
        st = stats[primary]
        if 'uni_R' in st:
            print(f'[rayleigh {arena} {kind}] {primary}: unimodal R={st["uni_R_corr"]:.3f} at {st["uni_dir_deg"]:.0f} deg, '
                  f'p_perm={st["uni_p_perm"]:.4g}; bimodal R={st["bi_R_corr"]:.3f} axis '
                  f'{st["bi_dir_deg"]:.0f}/{st["bi_axis2_deg"]:.0f} deg, p_perm={st["bi_p_perm"]:.4g} '
                  f'-> {st["pattern"]}')
        for j, c in enumerate(prof['centres']):
            sector_rows.append(dict(map_kind=kind, sector=j + 1, centre_deg=round(np.degrees(c), 1),
                                    n_bins=int(prof['n_bins'][j]),
                                    **{method: prof[method][j] for method in methods}))
        panels[kind] = dict(title=title, values=values, centres=prof['centres'],
                            mags=prof[primary], stats=st)

    plot_rayleigh(arena, handler, panels, len(results), primary,
                  os.path.join(out_dir, f'{_RAYLEIGH_FILE[arena]}.png'), title_prefix)
    xlsx = os.path.join(out_dir, f'{_RAYLEIGH_FILE[arena]}.xlsx')
    with pd.ExcelWriter(xlsx) as xw:
        pd.DataFrame(test_rows).to_excel(xw, sheet_name='Tests', index=False)
        pd.DataFrame(sector_rows).to_excel(xw, sheet_name='Sectors', index=False)
    print(f'[SAVED] {xlsx}')


# ============================================================================
# STAGE 5 -- LINEAR TRACK ARM PREFERENCE of the Duong-consistent bins (topographic vs magnetic)
# ============================================================================
# The linear track is one open arm of a + maze: 0 / 180 deg sessions use the horizontal (-) arm,
# 90 / 270 deg sessions the vertical (|) arm; within an arm the physical space is the same and only
# the magnetic north (mN) side differs (LINEAR_MN_SIDE). Per session type (its own Method 2 run) x
# map kind x direction (consistently ABOVE = green, BELOW = magenta), the consistent bins are split
# at the middle of the track length:
#   horizontal arm: topographic index = (n left  - n right) / n significant
#   vertical arm:   topographic index = (n top   - n down)  / n significant
#   both:           magnetic index    = (n mN side - n mS side) / n significant
# i.e. the magnetic index is +topographic when mN is on the left / top side and -topographic
# otherwise. Undefined (NaN) without significant bins.
# Positions are those of the raw tracking (x to the right, y up). Horizontal sessions are not
# rotated, so left = the lower half of the map's x. Vertical sessions are rotated on loading
# (LinearTrackHandler.orient: map x = max(y) - y), a rigid relabelling of the bins that does not
# change the Duong result, so top (high y) = the lower half of the map's x; the bins are mapped
# back to their real side, not re-tested on a rotated map.

def _first_half_label(arm: str) -> tuple:
    """Real sides of the lower / upper half of the map's x for this arm."""
    return ('left', 'right') if arm == 'horizontal' else ('top', 'bottom')


def linear_arm_preference(handler, consistent_by_type: dict) -> list:
    """consistent_by_type = {session type: {(kind, 'linear_track'): {'cons_above', 'cons_below'}}};
    one row per session type x map kind x direction."""
    first_half = handler.bin_centres_xy()[0] < handler.length / 2.0
    rows = []
    for stype in ARENA_SESSION_TYPES['linear_track']:
        cons = consistent_by_type.get(stype)
        if cons is None:
            continue
        arm, mn_side = LINEAR_ARM[stype], LINEAR_MN_SIDE[stype]
        side_a, side_b = _first_half_label(arm)
        if mn_side not in (side_a, side_b):
            raise ValueError(f'LINEAR_MN_SIDE[{stype!r}] = {mn_side!r} is not a side of the {arm} arm')
        mn_sign = 1.0 if mn_side == side_a else -1.0
        for kind in _KDE_MAP_KINDS:
            c = cons.get((kind, 'linear_track'))
            if c is None:
                continue
            for direction, tag in (('higher', 'cons_above'), ('lower', 'cons_below')):
                sig = c[tag]
                n_a, n_b = int(np.sum(sig & first_half)), int(np.sum(sig & ~first_half))
                n = n_a + n_b
                topo = (n_a - n_b) / n if n else np.nan
                rows.append(dict(session_type=stype, arm=arm, map_kind=kind, duong_bins=direction,
                                 n_significant=n, side_1=side_a, n_side_1=n_a, side_2=side_b, n_side_2=n_b,
                                 mN_side=mn_side, n_mN=n_a if mn_sign > 0 else n_b,
                                 n_mS=n_b if mn_sign > 0 else n_a,
                                 topographic_index=topo, magnetic_index=mn_sign * topo))
    return rows


def run_linear_arm_preference(handler, consistent_by_type: dict, out_dir: str):
    rows = linear_arm_preference(handler, consistent_by_type)
    if not rows:
        print('[arm preference] skipped: no linear session type with Duong results')
        return
    df = pd.DataFrame(rows)
    for kind, (_, title, *_) in _KDE_MAP_KINDS.items():
        sub = df[df['map_kind'] == kind]
        if len(sub):
            plot_linear_arm_preference(sub, title, os.path.join(out_dir, f'Linear_ArmPreference_{kind}.png'))
    xlsx = os.path.join(out_dir, 'Linear_ArmPreference.xlsx')
    df.to_excel(xlsx, index=False)
    print(f'[SAVED] {xlsx}')




# ============================================================================
# Plotting
# ============================================================================

_ARENA_ORDER = ['open_field', 'circular_track', 'linear_track']
_ARENA_TITLES = {'open_field': 'Open Field', 'circular_track': 'Circular Track', 'linear_track': 'Linear Track'}
_DUONG_ROW_HEIGHT = {'open_field': 1.0, 'circular_track': 1.0, 'linear_track': 0.45}
_EDGE_PTS = 6   # points per bin side, so the circular track's bin edges follow the ring's arcs
_KDE_PANEL_SIZE = {'open_field': (7.5, 7.0), 'circular_track': (12.0, 4.5),   # (w, h) inches of one
                   'linear_track': (12.0, 4.0)}                               # KDE panel when alone


def _get_cmap(name: str):
    try:
        base = matplotlib.colormaps[name]
    except Exception:
        base = plt.get_cmap(name)
    return base.copy() if hasattr(base, 'copy') else base


def make_cmap_norm(values, fixed_range: tuple | None = None) -> tuple:
    """jet colormap + Normalize over the data range, or over fixed_range = (vmin, vmax) if given."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if fixed_range is not None:
        vmin, vmax = fixed_range
    elif len(vals) == 0:
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


def _wrapped_suptitle(fig, text: str):
    """Suptitle wrapped to the figure width (~9 characters per inch), so a single-arena figure
    doesn't clip it."""
    fig.suptitle(textwrap.fill(text, max(40, int(fig.get_figwidth() * 9))))


def _arenas_in(arena_results: dict) -> list:
    """Arenas to plot, in _ARENA_ORDER: every arena run (AllArenas) or the one arena of a
    session-type run."""
    return [k for k in _ARENA_ORDER if k in arena_results]


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

def plot_fig_S1H(arena_handlers: dict, arena_results: dict, save_path: str, title_prefix: str = ''):
    arenas = _arenas_in(arena_results)
    fig = plt.figure(figsize=(max(5 * len(arenas), 7), 5))
    for i, key in enumerate(arenas):
        handler = arena_handlers[key]
        mean_map, valid = pool_fine_map(handler, arena_results[key])
        cmap, norm = make_cmap_norm(mean_map[valid], fixed_range=(0.0, 1.0))

        proj = 'polar' if key == 'circular_track' else None
        ax = fig.add_subplot(1, len(arenas), i + 1, projection=proj)
        pcm = handler.plot_fine(ax, mean_map, valid, cmap, norm)

        n_place = len(arena_results[key])
        ax.set_title(f'{_ARENA_TITLES[key]}\n(n={n_place} place cells)')
        fig.colorbar(pcm, ax=ax, shrink=0.7, label='Field index (a.u.)')

    _wrapped_suptitle(fig, f'{title_prefix}Fig S1H -- Overall mean field index maps, adaptive-binned '
                           '(place cells pooled across days)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_field_only_mean_maps(arena_handlers: dict, arena_results: dict, save_path: str,
                              title_prefix: str = ''):
    arenas = _arenas_in(arena_results)
    fig = plt.figure(figsize=(max(5 * len(arenas), 7), 5))
    for i, key in enumerate(arenas):
        handler = arena_handlers[key]
        mean_map, valid = pool_field_only_map(handler, arena_results[key])
        cmap, norm = make_cmap_norm(mean_map[valid], fixed_range=(0.0, 1.0))

        proj = 'polar' if key == 'circular_track' else None
        ax = fig.add_subplot(1, len(arenas), i + 1, projection=proj)
        pcm = handler.plot_fine(ax, mean_map, valid, cmap, norm)

        n_place = len(arena_results[key])
        ax.set_title(f'{_ARENA_TITLES[key]}\n(n={n_place} place cells)')
        fig.colorbar(pcm, ax=ax, shrink=0.7, label='Field index, background = 0 (a.u.)')

    _wrapped_suptitle(fig, f'{title_prefix}Field-only mean field-index maps, adaptive-binned')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_peak_proportion_maps(arena_handlers: dict, arena_results: dict, save_path: str,
                              title_prefix: str = ''):
    arenas = _arenas_in(arena_results)
    fig = plt.figure(figsize=(max(5 * len(arenas), 7), 5))
    for i, key in enumerate(arenas):
        handler = arena_handlers[key]
        pct_map, visited, n_cells = pool_peak_proportion_map(handler, arena_results[key])
        cmap, norm = make_cmap_norm(pct_map[visited])

        proj = 'polar' if key == 'circular_track' else None
        ax = fig.add_subplot(1, len(arenas), i + 1, projection=proj)
        pcm = handler.plot_fine(ax, pct_map, visited, cmap, norm)

        ax.set_title(f'{_ARENA_TITLES[key]}\n(n={n_cells} place cells)')
        fig.colorbar(pcm, ax=ax, shrink=0.7, label='% of place cells with peak in bin')

    _wrapped_suptitle(fig, f'{title_prefix}Peak proportion maps')
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
    arenas = _arenas_in(arena_results)
    if len(arenas) == len(_ARENA_ORDER):
        fig = plt.figure(figsize=(17, 7.5))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.7])
        arena_axes = {'open_field':     fig.add_subplot(gs[:, 0]),
                      'circular_track': fig.add_subplot(gs[0, 1]),
                      'linear_track':   fig.add_subplot(gs[1, 1])}
    else:   # session-type run: one panel per arena present, sized to that arena's 2D bin layout
        sizes = [_KDE_PANEL_SIZE[a] for a in arenas]
        fig, axs = plt.subplots(1, len(arenas), squeeze=False,
                                figsize=(sum(w for w, _ in sizes), max(h for _, h in sizes)),
                                gridspec_kw={'width_ratios': [w for w, _ in sizes]})
        arena_axes = dict(zip(arenas, axs[0]))
    saved, kdes = {}, {}
    for key in arenas:
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


def plot_shuffle_kind(title: str, arena_handlers: dict, summaries: dict, save_path: str):
    """Rows = arenas; columns = real KDE, one shuffled map, mean shuffled KDE, signed consistency
    f_above - f_below, consistent clusters, pointwise percentile envelope. Outlines: fluorescent
    green = consistently significantly ABOVE the shuffles, magenta = consistently BELOW."""
    arenas = [a for a in _ARENA_ORDER if a in summaries]
    ratios = [_DUONG_ROW_HEIGHT[a] for a in arenas]
    fig = plt.figure(figsize=(33, 1.2 + 5.6 * sum(ratios)))
    gs = fig.add_gridspec(len(arenas), 6, height_ratios=ratios)
    dens_cmap = plt.get_cmap('Greys')
    div_cmap = plt.get_cmap('RdBu_r')

    def fills(ax, handler, shown, above, below):
        colours = np.tile(to_rgba('0.88'), (handler.n_bins, 1))
        colours[above] = to_rgba(GREEN_ABOVE)
        colours[below] = to_rgba(MAGENTA_BELOW)
        _draw_bins(ax, handler, shown, facecolors=colours)

    for r, arena in enumerate(arenas):
        handler, s = arena_handlers[arena], summaries[arena]
        seen, n_done = s['seen'], len(s['n_rejected'])
        both = np.concatenate([s['f_real'][seen], s['f_shuf_mean'][seen]])
        dens_norm = Normalize(vmin=float(both.min()), vmax=float(both.max()))

        ax = fig.add_subplot(gs[r, 0])
        pc = _draw_bins(ax, handler, seen, s['f_real'], dens_cmap, dens_norm)
        _draw_duong_clusters(ax, handler, s)
        ax.set_title(f"{_ARENA_TITLES[arena]} -- REAL KDE\n(Scott factor {s['real_factor']:.3f})", fontsize=9)
        _hbar(fig, pc, ax, 'density (cm$^{-2}$)')

        ax = fig.add_subplot(gs[r, 1])
        ex_shown = s['domain'] & np.isfinite(s['example'])
        cmap, nrm = make_cmap_norm(s['example'][ex_shown])
        pc = _draw_bins(ax, handler, ex_shown, s['example'], cmap, nrm)
        ax.set_title('One shuffled pooled map\n(values permuted across the sampled bins)', fontsize=9)
        _hbar(fig, pc, ax, 'shuffled map value')

        ax = fig.add_subplot(gs[r, 2])
        pc = _draw_bins(ax, handler, seen, s['f_shuf_mean'], dens_cmap, dens_norm)
        _draw_duong_clusters(ax, handler, s)
        ax.set_title(f"Mean SHUFFLED KDE ({n_done} shuffles)\n"
                     f"(mean Scott factor {np.mean(s['factors']):.3f})", fontsize=9)
        _hbar(fig, pc, ax, 'density (cm$^{-2}$)')

        ax = fig.add_subplot(gs[r, 3])
        pc = _draw_bins(ax, handler, seen, s['f_above'] - s['f_below'], div_cmap, TwoSlopeNorm(0.0, -1.0, 1.0))
        _draw_duong_clusters(ax, handler, s)
        ax.set_title(f"Fraction of shuffles significant: above - below\n"
                     f"(Duong, Hochberg alpha = {DUONG_ALPHA:g}; outlined where >= {SHUFFLE_CONSISTENCY:g})",
                     fontsize=9)
        _hbar(fig, pc, ax, 'f$_{above}$ - f$_{below}$')

        ax = fig.add_subplot(gs[r, 4])
        fills(ax, handler, seen, s['lab_above'] >= 0, s['lab_below'] >= 0)
        ax.set_title(f"Consistent clusters (significant in >= {100 * SHUFFLE_CONSISTENCY:g} % of shuffles)\n"
                     f"{s['n_above_cl']} real > shuffle (green), {s['n_below_cl']} real < shuffle (magenta)",
                     fontsize=9)

        ax = fig.add_subplot(gs[r, 5])
        fills(ax, handler, seen, s['env_above'], s['env_below'])
        lo, hi = SHUFFLE_ENVELOPE
        ax.set_title(f"Pointwise envelope: real > {hi:g}th (green) / < {lo:g}th (magenta)\n"
                     f"percentile of the shuffled densities ({int(s['env_above'].sum())} / "
                     f"{int(s['env_below'].sum())} bins; not corrected for multiple bins)", fontsize=9)

    halo = [pe.Stroke(linewidth=OUTLINE_LW + 1.8, foreground='black'), pe.Normal()]
    handles = [Line2D([], [], color=GREEN_ABOVE, lw=OUTLINE_LW, path_effects=halo,
                      label='Real consistently significantly ABOVE shuffles'),
               Line2D([], [], color=MAGENTA_BELOW, lw=OUTLINE_LW, path_effects=halo,
                      label='Real consistently significantly BELOW shuffles')]
    fig.legend(handles=handles, loc='upper right', ncol=2, frameon=False, fontsize=10)
    renorm = 'renormalised on the common domain' if DUONG_RENORMALISE_ON_COMMON_DOMAIN else 'as estimated'
    fig.suptitle(f'{title}: real vs bin-shuffled KDEs -- Duong (2013) test repeated over {N_SHUFFLE} shuffles\n'
                 f'(densities {renorm}; n = {DUONG_SAMPLE_SIZE_MODE}; blank = outside the common domain)',
                 x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97 - 0.15 / (1.2 + 5.6 * sum(ratios))))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


# --- Stage 4: Rayleigh figure ---

RAYLEIGH_UNI_COLOUR = '#D55E00'   # unimodal mean vector
RAYLEIGH_BI_COLOUR  = '#000000'   # bimodal mean axis


def plot_rayleigh(arena: str, handler, panels: dict, n_cells: int, sector_mean: str, save_path: str,
                  title_prefix: str = ''):
    """Columns = map kinds. Top: the tested map (open field: wall-weighted) with the sectors (and
    the open field's rings). Bottom: sector magnitudes (polar bars) with the unimodal mean vector
    and the bimodal mean axis, each drawn with length R x the largest sector magnitude."""
    open_field = arena == 'open_field'
    fig = plt.figure(figsize=(6 * len(panels), 12))
    r_in, r_out = (0.0, handler.diameter / 2.0) if open_field else (handler.inner_r, handler.outer_r)
    width = np.radians(RAYLEIGH_SECTOR_DEG)
    for i, (kind, p) in enumerate(panels.items()):
        ax = fig.add_subplot(2, len(panels), i + 1)
        shown = np.isfinite(p['values'])
        cmap = _get_cmap('Blues')
        pc = _draw_bins(ax, handler, shown, p['values'], cmap,
                        Normalize(0.0, max(float(np.nanmax(p['values'])), 1e-6)))
        for c in p['centres'] - width / 2.0:
            ax.plot(handler.cx + np.array([r_in, r_out]) * np.cos(c),
                    handler.cy + np.array([r_in, r_out]) * np.sin(c), color='0.6', lw=0.4, zorder=4)
        if open_field:
            for rr in np.arange(RAYLEIGH_RING_WIDTH_CM, r_out, RAYLEIGH_RING_WIDTH_CM):
                ax.add_patch(matplotlib.patches.Circle((handler.cx, handler.cy), rr, fill=False,
                                                       edgecolor='0.6', lw=0.4, ls='--', zorder=4))
            ax.set_title(f'{p["title"]}\nx wall multiplier (0 centre -> 1 wall)', fontsize=10)
            _hbar(fig, pc, ax, 'map value x distance multiplier')
        else:
            ax.set_title(f'{p["title"]}\n({RAYLEIGH_SECTOR_DEG:g} deg sectors)', fontsize=10)
            _hbar(fig, pc, ax, 'map value')

        ax = fig.add_subplot(2, len(panels), len(panels) + i + 1, projection='polar')
        st = p['stats']
        mags = np.nan_to_num(p['mags'])
        ax.bar(p['centres'], mags, width=width * 0.92, color='#6A9FCB', edgecolor='none')
        top = float(mags.max()) if mags.max() > 0 else 1.0
        ax.set_ylim(0, top * 1.05)
        ax.set_yticklabels([])
        if 'uni_R' in st:
            th1 = np.radians(st['uni_dir_deg'])
            ax.annotate('', xy=(th1, st['uni_R_corr'] * top), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='-|>', color=RAYLEIGH_UNI_COLOUR, lw=2.2))
            th2 = np.radians(st['bi_dir_deg'])
            ax.plot([th2, 0, th2 + np.pi], [st['bi_R_corr'] * top, 0, st['bi_R_corr'] * top],
                    color=RAYLEIGH_BI_COLOUR, lw=2.2, ls='--')
            ax.set_title(f'Unimodal: R = {st["uni_R_corr"]:.3f}, dir = {st["uni_dir_deg"]:.0f} deg\n'
                         f'p_perm = {st["uni_p_perm"]:.3g}, p_Rayleigh = {st["uni_p_rayleigh"]:.3g}\n'
                         f'Bimodal: R = {st["bi_R_corr"]:.3f}, axis = {st["bi_dir_deg"]:.0f}/'
                         f'{st["bi_axis2_deg"]:.0f} deg\n'
                         f'p_perm = {st["bi_p_perm"]:.3g}, p_Rayleigh = {st["bi_p_rayleigh"]:.3g}\n'
                         f'Pattern: {st["pattern"]}', fontsize=9, pad=14)

    handles = [Line2D([], [], color=RAYLEIGH_UNI_COLOUR, lw=2.2, label='Unimodal mean vector'),
               Line2D([], [], color=RAYLEIGH_BI_COLOUR, lw=2.2, ls='--', label='Bimodal (axial) mean axis')]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=10)
    maps = 'the wall-weighted mean maps' if open_field else 'the mean maps'
    _wrapped_suptitle(fig, f'{title_prefix}{_ARENA_TITLES[arena]} (n={n_cells} place cells) -- Rayleigh '
                           f'tests of {maps}, {RAYLEIGH_SECTOR_DEG:g} deg sectors, '
                           f'sector value = {sector_mean.replace("_", "-")} mean; vector / axis length '
                           f'= R x largest sector')
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


# --- Stage 5: linear track arm preference figure ---

ARM_COLOURS = {'horizontal': '#4C78A8', 'vertical': '#E8853D'}


def plot_linear_arm_preference(df: pd.DataFrame, title: str, save_path: str):
    """One map kind. Columns: topographic index (left - right for 0 / 180, top - down for
    90 / 270) | magnetic index (mN - mS); rows: Duong bins consistently higher | lower firing."""
    present = set(df['session_type'])   # grouped by arm: 0, 180 | 90, 270
    stypes = [t for a in ('horizontal', 'vertical') for t in ARENA_SESSION_TYPES['linear_track']
              if LINEAR_ARM[t] == a and t in present]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5), sharey=True)
    for r, (direction, colour) in enumerate((('higher', GREEN_ABOVE), ('lower', MAGENTA_BELOW))):
        sub = df[df['duong_bins'] == direction].set_index('session_type')
        for c, (col, ylab) in enumerate((('topographic_index', 'Topographic preference index'),
                                         ('magnetic_index', 'Magnetic north preference index'))):
            ax = axs[r, c]
            ticks = []
            for i, t in enumerate(stypes):
                row = sub.loc[t] if t in sub.index else None
                if c == 0:
                    lab = f'{t}°\n(left - right)' if LINEAR_ARM[t] == 'horizontal' else f'{t}°\n(top - down)'
                else:
                    lab = f'{t}°\n(mN {LINEAR_MN_SIDE[t]})'
                ticks.append(lab)
                if row is None or not np.isfinite(row[col]):
                    ax.text(i, 0.04, 'no sig.\nbins', ha='center', va='bottom', fontsize=8, color='0.4')
                    continue
                v = float(row[col])
                ax.bar(i, v, width=0.6, color=ARM_COLOURS[LINEAR_ARM[t]])
                ax.text(i, v + (0.04 if v >= 0 else -0.04), f'{v:+.2f}\n(n={int(row["n_significant"])})',
                        ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
            ax.axhline(0.0, color='0.3', lw=0.8)
            ax.set_xticks(range(len(stypes)))
            ax.set_xticklabels(ticks, fontsize=9)
            ax.set_xlim(-0.6, len(stypes) - 0.4)
            ax.set_ylim(-1.3, 1.3)
            ax.set_ylabel(ylab)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title(f'{"Topographic" if c == 0 else "Magnetic"} -- Duong bins consistently '
                         f'{direction.upper()} than shuffles', fontsize=10)
            ax.add_patch(matplotlib.patches.Rectangle((0, 1.0), 1, 0.012, transform=ax.transAxes,
                                                      color=colour, clip_on=False))
    handles = [matplotlib.patches.Patch(color=ARM_COLOURS[a], label=f'{a.capitalize()} arm')
               for a in ('horizontal', 'vertical')]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=10)
    _wrapped_suptitle(fig, f'Linear track arm preference -- {title}: (side 1 - side 2) / significant bins, '
                           f'track split at its middle; raw tracking frame (x right, y up)')
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(save_path, dpi=200)
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
                arena=key, session_type=r.get('session_type'), session=r['session'], unit=r['unit'],
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

def collect_all_results(arena_handlers: dict) -> dict:
    """Per-cell rate maps of every arena (computed once, shared by the all-arena and the
    session-type runs), each cell tagged with its session type ('session_type', None if not
    recognised)."""
    arena_sessions = {key: find_arena_sessions(key, arena_handlers[key]) for key in _ARENA_ORDER}
    arena_results = {}
    for key in _ARENA_ORDER:
        print(f'[{key}] {len(arena_sessions[key])} sessions, '
              f'{sum(len(s[3]) for s in arena_sessions[key])} units')
        arena_results[key] = collect_arena_results(arena_handlers[key], arena_sessions[key])
        for r in arena_results[key]:
            r['session_type'] = _detect_session_type(r['session'], key)
        print(f'[{key}] {len(arena_results[key])} place cells pooled')

        unknown = sorted({r['session'] for r in arena_results[key] if r['session_type'] is None})
        if unknown:
            print(f'WARNING [{key}]: {len(unknown)} session(s) with no recognisable session type '
                  f'({ARENA_SESSION_TYPES[key]}); kept in AllArenas, left out of the session-type runs:')
            for s in unknown:
                print(f'    {s}')
        counts = {t: sum(r['session_type'] == t for r in arena_results[key])
                  for t in ARENA_SESSION_TYPES[key]}
        print(f'[{key}] place cells per session type: {counts}')
    return arena_results


def run_full_pipeline(arena_handlers: dict, arena_results: dict, out_dir: str, label: str = '',
                      summary_name: str = 'AllArenas_Summary.xlsx') -> None:
    """Stages 1-3 + summary workbook on arena_results ({arena: cells}), saved to out_dir. label
    (e.g. 'Linear_180') prefixes the figure titles."""
    os.makedirs(out_dir, exist_ok=True)
    prefix = f'{label} -- ' if label else ''
    print(f'\n{"=" * 70}\n{label or "All arenas"} -> {out_dir}\n{"=" * 70}')

    # 1. Mean rate maps: overall, field-only, peak proportion
    plot_fig_S1H(arena_handlers, arena_results, os.path.join(out_dir, 'WholeRM_MeanFieldIndex.png'),
                 prefix)
    plot_field_only_mean_maps(arena_handlers, arena_results,
                              os.path.join(out_dir, 'FieldOnly_MeanFieldIndex.png'), prefix)
    plot_peak_proportion_maps(arena_handlers, arena_results,
                              os.path.join(out_dir, 'PeakProportion_Map.png'), prefix)

    # 4. Rayleigh tests of the mean maps (open field, circular track; run before the slow shuffles)
    if RUN_RAYLEIGH:
        for arena in _RAYLEIGH_FILE:
            if arena_results.get(arena):
                run_rayleigh(arena, arena_handlers[arena], arena_results[arena], out_dir, prefix)

    # 2. KDE of every pooled map
    kdes = {'real': {}, 'null': {}}
    for kind, (*_, stem) in _KDE_MAP_KINDS.items():
        kdes['real'][kind] = plot_kde_maps(arena_handlers, arena_results, kind,
                                           os.path.join(out_dir, f'KDE_{stem}.png'),
                                           title_prefix=prefix)

    # 3. Method 2: real KDE vs N_SHUFFLE bin-shuffled KDEs, Duong test per shuffle
    consistent = run_shuffle_duong(arena_handlers, arena_results, kdes['real'], out_dir,
                                   title_prefix=prefix)

    export_excel(arena_handlers, arena_results, os.path.join(out_dir, summary_name))
    return consistent


if __name__ == '__main__':
    arena_handlers = {key: make_handler(ARENA_CONFIGS[key]) for key in _ARENA_ORDER}
    arena_results = collect_all_results(arena_handlers)

    # All cells of each arena, every session type pooled
    if RUN_ALL_ARENAS:
        run_full_pipeline(arena_handlers, arena_results, os.path.join(OUTPUT_DIR, 'AllArenas'))

    # Every session type of every arena on its own
    if RUN_SESSION_TYPES:
        linear_consistent = {}
        for key in _ARENA_ORDER:
            for stype in ARENA_SESSION_TYPES[key]:
                label = _session_type_label(key, stype)
                cells = [r for r in arena_results[key] if r['session_type'] == stype]
                if not cells:
                    print(f'\n[{label}] no place cells; skipped')
                    continue
                consistent = run_full_pipeline(arena_handlers, {key: cells},
                                               os.path.join(OUTPUT_DIR, 'SessionType', label), label,
                                               summary_name=f'{label}_Summary.xlsx')
                if key == 'linear_track':
                    linear_consistent[stype] = consistent

        # 5. Linear track arm preference across the four linear session types
        if RUN_LINEAR_ARM_PREFERENCE:
            run_linear_arm_preference(arena_handlers['linear_track'], linear_consistent,
                                      os.path.join(OUTPUT_DIR, 'SessionType'))
    print('\nDone.')
