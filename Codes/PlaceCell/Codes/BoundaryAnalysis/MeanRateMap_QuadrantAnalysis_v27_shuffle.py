# -*- coding: utf-8 -*-
"""
Mean Rate Map Analysis (S1H, after Muessig et al.) + KDE + Duong (2013) local test

PIPELINE (run_full_pipeline), per arena:
  1. MEAN MAPS -- the pooled maps of the recorded place cells (described below): overall mean
     field index, field-only mean field index and peak proportion.
  2. KDE -- a 2D Gaussian KDE (scipy gaussian_kde, Scott's rule, edge-corrected over the bins the
     cells validly sampled) of each of the three pooled maps.
  3. DUONG TEST -- local significant differences between each real KDE and a null KDE (Duong 2013,
     "Local significant differences from nonparametric two-sample tests", J. Nonparametric
     Statistics 25:3, 635-645); see the DUONG section below. No null model is built in this
     script, so the test is skipped until null KDEs are supplied to run_duong_tests. Clusters
     where the real density is significantly above the null are drawn in GREEN_ABOVE, below in
     MAGENTA_BELOW.

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
  Duong test     DuongTest_<kind>.png, DuongTest_Clusters.xlsx, DuongTest_Results.npz
                 (only when null KDEs are supplied)
"""

# Standard library: file-system paths, directory walking and joining.
import os
# Standard library: cryptographic hashing, used only to derive a reproducible per-cell random seed.
import hashlib
# Standard library: pseudo-random number generator used for the circular-shift bootstrap.
import random
# Standard library: thread pool used to process many units (.ntt files) in parallel.
import concurrent.futures
# Standard library: lets us silence expected RuntimeWarnings (e.g. nanmean of an all-NaN column).
import warnings

# NumPy: all array maths (binning, rates, masks, linear algebra).
import numpy as np
# pandas: reads tracking .csv/.xlsx files and writes the Excel summary workbooks.
import pandas as pd
# SciPy Gaussian filters: 2D filter for rate-map smoothing, 1D filter for trajectory smoothing.
from scipy.ndimage import gaussian_filter, gaussian_filter1d
# SciPy stats: chi-square survival fn (Duong p-values), weighted Gaussian KDE, standard normal (z cut-off).
from scipy.stats import chi2, gaussian_kde, norm as normal_dist

# Matplotlib core package.
import matplotlib
# Select the non-interactive 'Agg' backend so figures are rendered straight to files (no GUI window).
matplotlib.use('Agg')
# pyplot: figure / axes creation interface.
import matplotlib.pyplot as plt
# patches: Circle / Rectangle shapes used to draw arena outlines.
import matplotlib.patches
# patheffects: black "halo" stroke drawn under the coloured Duong cluster outlines.
import matplotlib.patheffects as pe
# LineCollection draws many outline segments at once; PolyCollection draws many filled bin polygons.
from matplotlib.collections import LineCollection, PolyCollection
# Normalize = linear colour scaling; TwoSlopeNorm = diverging scale centred on 0; to_rgba = colour -> RGBA tuple.
from matplotlib.colors import Normalize, TwoSlopeNorm, to_rgba
# Line2D: proxy artists used to build the figure legend.
from matplotlib.lines import Line2D

# ============================================================================
# CONFIGURATION -- edit root folder + geometry below
# ============================================================================

# Top-level folder that is walked recursively to find every session (tracking file + .ntt files).
ROOT_DIRECTORY = r'X:\NMR_group_data\Runita\Analysis\Mean_KDE_Open_PascalOldenburg\Open_KDE\CorrectedData\Data\SessionTypeSorted_PC\Open\Cntrl'
# Folder (inside ROOT_DIRECTORY) where every figure / workbook / npz is written.
OUTPUT_DIR = os.path.join(ROOT_DIRECTORY, 'MeanRM_Quad_ObsNullRealFIelds')

# Switch: if True, a unit is dropped when its number of valid (>= min_occ_s) bins is below COVERAGE_FRACTION of the arena.
USE_BIN_COVERAGE_CRITERION = False
# Minimum fraction of arena bins that must be validly sampled (only used when the switch above is True).
COVERAGE_FRACTION = 0.01

# Switch: if True, a session is dropped unless both the edge zone and the centre zone received enough occupancy.
USE_ZONE_COVERAGE_CRITERION = False
# Minimum % of total occupancy time required in EACH zone (edge and centre) when the switch above is True.
MIN_ZONE_COVERAGE_PCT       = 0.1

# Map from a folder name (lower-case) to the arena it denotes; used to classify every session by its path.
ARENA_FOLDER_KEYWORDS = {
    # A folder called "Open" -> open-field arena.
    'open':     'open_field',
    # A folder called "Linear" -> linear track.
    'linear':   'linear_track',
    # A folder called "Circle" -> circular (annular) track.
    'circle':   'circular_track',
    # A folder called "Circular" -> circular (annular) track.
    'circular': 'circular_track',
# End of the keyword dictionary.
}

# Physical geometry of every arena (all in cm).
ARENA_CONFIGS = {
    # Open field: a circular floor.
    'open_field': dict(
        # Shape tag used by make_handler to pick OpenFieldHandler.
        shape='circle',
        # Diameter of the open field = 60 cm (radius 30 cm).
        diameter_cm=60.0,
    # End of open-field config.
    ),
    # Circular track: an annulus (ring).
    'circular_track': dict(
        # Shape tag used by make_handler to pick CircularTrackHandler.
        shape='ring',
        # Outer wall diameter = 80 cm (outer radius 40 cm).
        outer_diameter_cm=80.0,
        # Inner wall diameter = 72 cm (inner radius 36 cm) -> track 4 cm wide.
        inner_diameter_cm=72.0,
    # End of circular-track config.
    ),
    # Linear track: a rectangle.
    'linear_track': dict(
        # Shape tag used by make_handler to pick LinearTrackHandler.
        shape='linear',
        # Track length = 80 cm.
        length_cm=80.0,
        # Track width = 8 cm.
        width_cm=8.0,
    # End of linear-track config.
    ),
# End of arena geometry dictionary.
}

# Classify a directory path as one of the arenas by looking for an arena keyword among its folder names.
def _detect_arena_key(dirpath: str) -> str:
    # Normalise the path and split it into its individual folder names; check each folder in turn.
    for part in os.path.normpath(dirpath).split(os.sep):
        # Look the folder name up (whitespace-stripped, lower-cased) in the keyword table; None if not a keyword.
        arena_key = ARENA_FOLDER_KEYWORDS.get(part.strip().lower())
        # The first folder that matches a keyword decides the arena...
        if arena_key is not None:
            # ...so return that arena key immediately.
            return arena_key
    # No folder in the path names an arena -> unclassified.
    return None

# Switch: if True, open-field tracking is translated so the trajectory is centred on the arena centre.
CENTRE_OPEN_FIELD_TRACKING = True

# Shift open-field coordinates so the midpoint of the trajectory's bounding box sits at the arena centre.
def _centre_open_field_tracking(x_cm: np.ndarray, y_cm: np.ndarray, session_dir: str,
                                # handler supplies the arena centre (cx, cy).
                                handler) -> tuple:
    # If centring is disabled globally...
    if not CENTRE_OPEN_FIELD_TRACKING:
        # ...return the coordinates unchanged.
        return x_cm, y_cm
    # Only open-field sessions are centred, and only if there is at least one sample...
    if _detect_arena_key(session_dir) != 'open_field' or len(x_cm) == 0:
        # ...otherwise return the coordinates unchanged.
        return x_cm, y_cm

    # Midpoint of the x range of the trajectory: (min + max) / 2.
    mid_x = 0.5 * (float(x_cm.min()) + float(x_cm.max()))
    # Midpoint of the y range of the trajectory: (min + max) / 2.
    mid_y = 0.5 * (float(y_cm.min()) + float(y_cm.max()))
    # Translate every sample by (arena centre - trajectory midpoint) in x and y.
    return x_cm + (handler.cx - mid_x), y_cm + (handler.cy - mid_y)


# Camera frame rate (frames per second) -- used for default frame dwell time and bootstrap margins.
fps           = 30
# Side length of a spatial bin (cm): every arena is binned at ~2 x 2 cm.
target_bin_cm  = 2.0
# Minimum occupancy (seconds) for a bin to count as validly sampled.
min_occ_s      = 0.5 #1

# Open field: bins whose centre is within 9.25 cm of the wall form the "edge zone".
OPEN_EDGE_ZONE_THRESHOLD_CM   = 9.25
# Linear track: distance threshold (cm) that defines the edge zone (long walls and track ends).
LINEAR_EDGE_ZONE_THRESHOLD_CM = 2.0

# Max allowed gap (microseconds) between a spike and its nearest tracking frame (50 ms) for the spike to be kept.
MAX_GAP_US     = 50_000
# Number of circular-shift shuffles in the spatial-information bootstrap.
N_BOOTSTRAP    = 1000
# Number of worker threads that process units in parallel.
MAX_WORKERS    = 4
# Global seed mixed into every per-cell bootstrap seed.
BOOTSTRAP_SEED = 0

# Any step between consecutive tracking samples faster than this (cm/s) is treated as a tracking jump.
POS_JUMP_THRESH_CMS  = 90.0
# SD (in samples) of the 1D Gaussian used to smooth the x and y trajectories.
POS_SMOOTH_SIGMA_SMP = 5.0

# Switch: apply the running-speed filter.
SPEED_FILTER_ENABLED = True
# Frames slower than this (cm/s) are "immobile" and excluded.
MIN_SPEED_CMS        = 0.5
# Frames faster than this (cm/s) are treated as artefacts and excluded.
MAX_SPEED_CMS        = 90.0
# Switch: also remove speed-filtered frames from the occupancy (not only from the spikes).
SPEED_FILTER_OCCUPANCY = True
# SD (in bins) of the 2D Gaussian used to smooth rate maps (3 bins = 6 cm).
RATEMAP_SMOOTH_SIGMA_BINS = 3.0

# A place-field bin must exceed this fraction (20 %) of the cell's smoothed peak rate.
FIELD_PEAK_FRAC = 0.20
# A place field must contain at least this many connected bins.
MIN_FIELD_BINS  = 9

# Bandwidth rule for scipy gaussian_kde: Scott's rule (factor = n_eff^(-1/(d+4))).
KDE_BW_METHOD = 'scott'
# Switch: divide the KDE by the share of the kernel lying inside the sampled domain (boundary correction).
KDE_EDGE_CORRECTION = True

# Units of the tracking file: 'cm' = already in cm (columns 0,3,4); anything else = pixels with named columns.
COORD_UNITS = 'cm'

# --- Duong (2013) local test, real KDE vs null KDE (stage 3) ---
# Family-wise significance level for each (map kind, arena) test family (Hochberg step-up).
DUONG_ALPHA            = 0.05     # family-wise level per (map kind, arena), Hochberg step-up
# How the sample sizes n1, n2 in the Duong variance are chosen: Kish effective n ('neff') or number of cells ('cells').
DUONG_SAMPLE_SIZE_MODE = 'neff'   # 'neff' or 'cells' (see the DUONG section)
# Switch: rescale both densities to integrate to 1 over the common evaluation domain before comparing.
DUONG_RENORMALISE_ON_COMMON_DOMAIN = True
# Minimum number of bins for a significant cluster to be outlined / reported.
DUONG_MIN_CLUSTER_BINS = 1        # clusters smaller than this are not outlined / reported
# Colour used for clusters where the real density is significantly ABOVE the null.
GREEN_ABOVE   = '#39FF14'         # fluorescent green: real significantly ABOVE null
# Colour used for clusters where the real density is significantly BELOW the null.
MAGENTA_BELOW = '#FF00FF'         # fluorescent magenta: real significantly BELOW null
# Line width of the cluster outlines.
OUTLINE_LW    = 2.2
# Roughness R(K) = integral of K^2 for the standard bivariate Gaussian kernel = 1 / (4*pi).
R_K_GAUSS_2D  = 1.0 / (4.0 * np.pi)   # R(K) = integral of K^2 for the standard bivariate normal

# Binary record layout of a Neuralynx .ntt (tetrode spike) file; one record per spike.
ntt_dtype = np.dtype([
    # Spike time, unsigned 64-bit little-endian integer, in microseconds.
    ('timestamp',   '<u8'),
    # Acquisition-entity (spike channel) number, unsigned 32-bit.
    ('sc_number',   '<u4'),
    # Cluster / cell ID assigned by spike sorting (0 = unassigned), unsigned 32-bit.
    ('cell_number', '<u4'),
    # 8 feature values computed by the acquisition system, unsigned 32-bit each.
    ('params',      '<u4', (8,)),
    # Waveform: 32 samples x 4 tetrode channels, signed 16-bit.
    ('waveforms',   '<i2', (32, 4)),
# End of the record definition.
])


# ============================================================================
# Tracking load / clean / smooth
# ============================================================================

# Load a tracking file, drop invalid / jump samples, sort by time and convert to cm with origin at (min x, min y).
def _load_tracking(csv_path: str, arena_width_cm: float) -> tuple:
    # Read the tracking table: Excel reader for .xlsx...
    data = (pd.read_excel(csv_path) if csv_path.lower().endswith('.xlsx')
            # ...CSV reader for everything else.
            else pd.read_csv(csv_path))

    # If the file is already in cm, columns are taken by position:
    if COORD_UNITS == 'cm':
        # column 0 = timestamp (microseconds), as float.
        t = np.asarray(data.iloc[:, 0], dtype=float)
        # column 3 = x position (cm), as float.
        x = np.asarray(data.iloc[:, 3], dtype=float)
        # column 4 = y position (cm), as float.
        y = np.asarray(data.iloc[:, 4], dtype=float)
    # Otherwise (pixel files) columns are taken by name:
    else:
        # 'x' column = x position (pixels).
        x = np.asarray(data['x'],    dtype=float)
        # 'y' column = y position (pixels).
        y = np.asarray(data['y'],    dtype=float)
        # 'time' column = timestamp (microseconds).
        t = np.asarray(data['time'], dtype=float)

    # Mark samples whose x equals the tracker's "lost" sentinel values 1 or -1 as invalid (keep all others).
    mask = ~np.isin(x, [1, -1])
    # Keep only the non-sentinel samples in x, y and t.
    x, y, t = x[mask], y[mask], t[mask]

    # Forward x-difference to the next sample (x[i+1]-x[i]); last sample gets 0.
    dx = np.append(np.diff(x), 0)
    # Forward y-difference to the next sample; last sample gets 0.
    dy = np.append(np.diff(y), 0)
    # Forward time difference to the next sample; last sample gets 1 (dummy positive value).
    dt = np.append(np.diff(t), 1)

    # Euclidean step length to the next sample: sqrt(dx^2 + dy^2).
    dxy = np.hypot(dx, dy)
    # Steps with a strictly positive time difference (duplicate / backwards timestamps are invalid).
    valid_dt = dt > 0
    # Speed array initialised to 0.
    speed = np.zeros_like(dxy)
    # Speed = step length / dt for steps with valid dt (units: cm per microsecond in 'cm' mode).
    speed[valid_dt] = dxy[valid_dt] / dt[valid_dt]

    # Keep samples with valid dt and speed < 0.9 units/us (= 900 cm/s in cm mode): removes gross glitches only.
    keep = np.where(valid_dt & (speed < 0.9))[0]
    # Apply the keep index to x, y and t.
    x, y, t = x[keep], y[keep], t[keep]

    # Indices that sort the samples by increasing time.
    order = np.argsort(t)
    # Reorder x, y and t chronologically.
    x, y, t = x[order], y[order], t[order]

    # If nothing survived the cleaning...
    if len(t) == 0:
        # ...return the (empty) arrays as float64.
        return x.astype(np.float64), y.astype(np.float64), t.astype(np.float64)

    # Coordinates already in cm:
    if COORD_UNITS == 'cm':
        # shift x so the smallest x is 0.
        x_cm = x - x.min()
        # shift y so the smallest y is 0.
        y_cm = y - y.min()
    # Coordinates in pixels:
    else:
        # x extent of the trajectory (pixels).
        x_span = x.max() - x.min()
        # y extent of the trajectory (pixels).
        y_span = y.max() - y.min()
        # Pixels per cm = larger extent / physical arena width (assumes the animal covered the full width).
        px_per_cm = max(x_span, y_span) / arena_width_cm
        # Shift x to start at 0 and convert to cm.
        x_cm = (x - x.min()) / px_per_cm
        # Shift y to start at 0 and convert to cm.
        y_cm = (y - y.min()) / px_per_cm

    # Return cleaned position (cm) and time (microseconds).
    return x_cm, y_cm, t


# Remove tracking jumps (> jump_thresh_cms) iteratively, interpolate over them, then Gaussian-smooth x and y.
def _smooth_tracking_position(x_cm: np.ndarray, y_cm: np.ndarray, t_us: np.ndarray,
                              # Speed threshold (cm/s) defining a jump.
                              jump_thresh_cms: float = POS_JUMP_THRESH_CMS,
                              # Gaussian smoothing SD in samples.
                              sigma_samples: float = POS_SMOOTH_SIGMA_SMP) -> tuple:
    # Number of tracking samples.
    n = len(x_cm)
    # With fewer than 2 samples no speed can be computed...
    if n < 2:
        # ...return unchanged copies.
        return x_cm.copy(), y_cm.copy()

    # Boolean flag per sample: True = sample is a jump / bad.
    bad = np.zeros(n, dtype=bool)
    # Repeat the jump detection (at most n times) until no new jump is found.
    for _ in range(n):
        # Indices of the samples still considered good.
        good_idx = np.where(~bad)[0]
        # Need at least two good samples to form a step...
        if len(good_idx) < 2:
            # ...otherwise stop.
            break
        # Time between consecutive good samples, converted from microseconds to seconds.
        dt_good = np.diff(t_us[good_idx]) * 1e-6
        # Suppress divide-by-zero / invalid warnings for dt = 0 steps.
        with np.errstate(invalid='ignore', divide='ignore'):
            # Speed of each step between consecutive good samples (cm/s) = distance / time.
            step_speed = np.hypot(np.diff(x_cm[good_idx]), np.diff(y_cm[good_idx])) / dt_good
        # Steps with non-positive dt get speed 0 (not considered jumps).
        step_speed[dt_good <= 0] = 0.0

        # Steps faster than the jump threshold.
        newly_bad = step_speed > jump_thresh_cms
        # If there are no jumps left...
        if not newly_bad.any():
            # ...the cleaning has converged; stop.
            break
        # Mark the LATER sample of every jump step as bad (good_idx[1:] are the step end points).
        bad[good_idx[1:][newly_bad]] = True

    # Indices of the final good samples.
    good_idx = np.where(~bad)[0]
    # If no sample is good or all samples are good, interpolation is impossible / unnecessary:
    if len(good_idx) == 0 or len(good_idx) == n:
        # keep the original coordinates.
        x_clean, y_clean = x_cm.copy(), y_cm.copy()
    # Otherwise replace bad samples by linear interpolation in time between good ones:
    else:
        # x at every timestamp, linearly interpolated from the good samples.
        x_clean = np.interp(t_us, t_us[good_idx], x_cm[good_idx])
        # y at every timestamp, linearly interpolated from the good samples.
        y_clean = np.interp(t_us, t_us[good_idx], y_cm[good_idx])

    # Smooth x with a 1D Gaussian (SD = sigma_samples); edges padded by repeating the end value.
    x_smooth = gaussian_filter1d(x_clean, sigma=sigma_samples, mode='nearest')
    # Smooth y the same way.
    y_smooth = gaussian_filter1d(y_clean, sigma=sigma_samples, mode='nearest')
    # Return the cleaned, smoothed trajectory.
    return x_smooth, y_smooth


# Full position pipeline for one session: load -> jump-clean + smooth -> centre (open field only).
def _session_positions(csv_path: str, handler) -> tuple:
    # Load and clean the tracking file (cm, microseconds).
    x_cm, y_cm, t = _load_tracking(csv_path, handler.arena_width_cm)
    # With fewer than 2 samples nothing more can be done...
    if len(t) < 2:
        # ...return what was loaded.
        return x_cm, y_cm, t
    # Remove jumps, interpolate and Gaussian-smooth the trajectory.
    x_cm, y_cm = _smooth_tracking_position(x_cm, y_cm, t)
    # Folder containing the tracking file (= the session folder).
    session_dir = os.path.dirname(csv_path)
    # Centre the trajectory on the arena centre if this is an open-field session.
    x_cm, y_cm = _centre_open_field_tracking(x_cm, y_cm, session_dir, handler)
    # Return the final position and time arrays.
    return x_cm, y_cm, t


# Per-frame boolean: True where the animal's running speed lies within [MIN_SPEED_CMS, MAX_SPEED_CMS].
def _speed_mask(x_cm: np.ndarray, y_cm: np.ndarray, t_us: np.ndarray) -> np.ndarray:
    # Number of frames.
    n = len(t_us)
    # If the speed filter is off or there are too few frames to compute speed...
    if not SPEED_FILTER_ENABLED or n < 2:
        # ...every frame passes.
        return np.ones(n, dtype=bool)
    # Time between consecutive frames, microseconds -> seconds.
    dt_s = np.diff(t_us) * 1e-6
    # Distance travelled between consecutive frames (cm).
    step_cm = np.hypot(np.diff(x_cm), np.diff(y_cm))
    # Speed per step, initialised to NaN (NaN never passes the filter).
    speed = np.full(n - 1, np.nan)
    # Steps with positive dt.
    ok = dt_s > 0
    # Speed (cm/s) = distance / time for those steps.
    speed[ok] = step_cm[ok] / dt_s[ok]
    # Assign each frame the speed of the step that ENDS at it; frame 0 copies the first step's speed.
    speed = np.concatenate(([speed[0]], speed))
    # Frame passes if MIN_SPEED_CMS <= speed <= MAX_SPEED_CMS.
    return (speed >= MIN_SPEED_CMS) & (speed <= MAX_SPEED_CMS)


# Build the per-frame table for one session: position, bin, dwell time, moving flag, occupancy flag.
def _session_frames(csv_path: str, handler) -> dict | None:
    """One session's on-arena tracking frames: oriented position, flat bin, dwell time dt (s),
    the speed-filter flag `moving` and `occ_frame` (frames that count toward occupancy). Shared by
    the rate maps and the zone-coverage criterion, so both bin and time-weight the trajectory
    identically. None if the tracking is unusable."""
    # Load, clean, smooth and centre the trajectory.
    x_cm, y_cm, t = _session_positions(csv_path, handler)
    # Unusable tracking (< 2 samples)...
    if len(t) < 2:
        # ...signal failure.
        return None
    # Arena-specific orientation (linear track: rotate vertical sessions to horizontal).
    x_cm, y_cm = handler.orient(x_cm, y_cm)
    # Speed filter flag for every frame.
    moving = _speed_mask(x_cm, y_cm, t)
    # Flat spatial bin index of every frame, and whether the frame lies on/near the arena.
    bin_idx, sample_valid = handler.to_bins(x_cm, y_cm)
    # Discard off-arena frames from position and time.
    x_cm, y_cm, t = x_cm[sample_valid], y_cm[sample_valid], t[sample_valid]
    # Discard off-arena frames from the bin index and moving flag.
    bin_idx, moving = bin_idx[sample_valid], moving[sample_valid]
    # If fewer than 2 on-arena frames remain...
    if len(t) < 2:
        # ...signal failure.
        return None

    # Allocate the dwell time (seconds) attributed to each frame.
    dt_frames = np.empty(len(t), dtype=np.float64)
    # First frame: one nominal frame period (1/fps s).
    dt_frames[0] = 1.0 / fps
    # Other frames: time since previous frame (s), capped at 2 frame periods so tracking gaps don't inflate occupancy.
    dt_frames[1:] = np.minimum(np.diff(t) * 1e-6, 2.0 / fps)
    # Frames counted in occupancy: only moving frames if SPEED_FILTER_OCCUPANCY, otherwise all frames.
    occ_frame = moving if SPEED_FILTER_OCCUPANCY else np.ones(len(t), dtype=bool)
    # Return everything as a dict keyed by name.
    return dict(x=x_cm, y=y_cm, t=t, bin_idx=bin_idx, dt=dt_frames, moving=moving,
                # (continuation) occupancy flag.
                occ_frame=occ_frame)


# Deterministic 64-bit seed from arbitrary identifiers (e.g. session, unit) so each cell's bootstrap is reproducible.
def _stable_seed(*parts) -> int:
    # Join the parts with '|' into one string and encode it as UTF-8 bytes.
    key = '|'.join(str(p) for p in parts).encode('utf-8')
    # 8-byte BLAKE2b hash -> big-endian integer, XOR-ed with the global BOOTSTRAP_SEED.
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'big') ^ BOOTSTRAP_SEED


# ============================================================================
# Gaussian smoothing kernel
# ============================================================================

# Normalised (masked) 2D Gaussian smoothing: smooth(rate * mask) / smooth(mask), so unsampled bins don't drag rates to 0.
def _gaussian_smooth_2d(fr_map: np.ndarray, valid_mask: np.ndarray,
                         # Kernel SD in bins.
                         sigma: float = RATEMAP_SMOOTH_SIGMA_BINS,
                         # True = first axis is periodic (circular track's angular axis).
                         wrap_x: bool = False) -> np.ndarray:
    # Boundary mode per axis: axis 0 wraps around if periodic, else zero-padded; axis 1 always zero-padded.
    mode = ('wrap' if wrap_x else 'constant', 'constant')
    # Rate map with invalid bins set to 0 (so they add nothing to the numerator).
    fr_in   = np.where(valid_mask, fr_map, 0.0)
    # Validity mask as 1.0 / 0.0 weights.
    mask_in = valid_mask.astype(np.float64)
    # Numerator: Gaussian-smoothed masked rate map.
    smoothed_fr = gaussian_filter(fr_in,   sigma=sigma, mode=mode, cval=0.0)
    # Denominator: Gaussian-smoothed mask = local weight of valid bins under the kernel.
    smoothed_w  = gaussian_filter(mask_in, sigma=sigma, mode=mode, cval=0.0)
    # Output array, initialised to 0.
    smoothed = np.zeros_like(smoothed_fr)
    # Bins where the kernel touched at least some valid bins.
    vw = smoothed_w > 0
    # Normalised smoothed rate = numerator / denominator (weighted local mean of valid rates).
    smoothed[vw] = smoothed_fr[vw] / smoothed_w[vw]
    # Invalid bins are forced back to 0 in the output.
    smoothed[~valid_mask] = 0.0
    # Return the smoothed 2D map.
    return smoothed


# Find 8-connected groups of True bins in a flat (nx*ny) mask, optionally wrapping along the first axis.
def _connected_components_2d_flat(qualifies_flat: np.ndarray, nx: int, ny: int,
                                   # True = axis 0 is periodic (circular track).
                                   wrap_x: bool = False) -> list:
    # Reshape the flat mask to its (nx, ny) grid (flat index = i * ny + j).
    qualifies_2d = qualifies_flat.reshape(nx, ny)
    # Bins that do NOT qualify are pre-marked as visited so they are never added to a component.
    visited = ~qualifies_2d
    # List of components; each component is a list of flat bin indices.
    components = []
    # Scan every row index i...
    for i in range(nx):
        # ...and every column index j.
        for j in range(ny):
            # Skip bins already assigned (or not qualifying).
            if visited[i, j]:
                # Move to the next bin.
                continue
            # Start a new component.
            region = []
            # Depth-first-search stack seeded with this bin.
            stack = [(i, j)]
            # Mark the seed as visited.
            visited[i, j] = True
            # Grow the component until the stack is empty.
            while stack:
                # Take the next bin to expand.
                bx, by = stack.pop()
                # Add its flat index to the component.
                region.append(bx * ny + by)
                # Look at the 3 x 3 neighbourhood: row offsets -1, 0, +1...
                for ddx in (-1, 0, 1):
                    # ...column offsets -1, 0, +1.
                    for ddy in (-1, 0, 1):
                        # Skip the centre bin itself.
                        if ddx == 0 and ddy == 0:
                            # Next offset.
                            continue
                        # Neighbour's grid coordinates.
                        ni, nj = bx + ddx, by + ddy
                        # On a periodic axis...
                        if wrap_x:
                            # ...wrap the row index around (bin -1 -> nx-1, bin nx -> 0).
                            ni = ni % nx
                        # If the neighbour is inside the grid and is an unvisited qualifying bin...
                        if 0 <= ni < nx and 0 <= nj < ny and not visited[ni, nj]:
                            # ...mark it visited...
                            visited[ni, nj] = True
                            # ...and push it for expansion.
                            stack.append((ni, nj))
            # The component is complete; store it.
            components.append(region)
    # Return all components.
    return components


# ============================================================================
# Arena geometry handlers
# ============================================================================

# Geometry, binning, smoothing and plotting for the circular open field.
class OpenFieldHandler:
    # The open field has no periodic axis.
    wrap_x = False

    # Build the square bin grid covering the circular floor and its per-bin geometric properties.
    def __init__(self, cfg: dict, bin_cm: float):
        # Arena diameter (cm).
        self.diameter = cfg['diameter_cm']
        # Bin side length (cm).
        self.bin_cm   = bin_cm
        # Number of bins along x = ceil(diameter / bin size) (60/2 = 30).
        self.nx = int(np.ceil(self.diameter / bin_cm))
        # Square grid: same number of bins along y.
        self.ny = self.nx
        # Total number of bins in the grid (30 x 30 = 900).
        self.n_bins = self.nx * self.ny
        # Arena centre x (cm) = radius.
        self.cx = self.diameter / 2.0
        # Arena centre y (cm) = radius.
        self.cy = self.diameter / 2.0
        # Physical width used for pixel->cm scaling = diameter.
        self.arena_width_cm = self.diameter

        # x coordinate (cm) of every bin centre: (i + 0.5) * bin size.
        xs = (np.arange(self.nx) + 0.5) * bin_cm
        # y coordinate (cm) of every bin centre.
        ys = (np.arange(self.ny) + 0.5) * bin_cm
        # 2D grids of bin-centre coordinates, indexed [i, j] = [x bin, y bin].
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        # Distance of every bin centre from the arena centre.
        r = np.hypot(XX - self.cx, YY - self.cy)
        # Bins whose centre lies inside the circle are geometrically valid (flattened).
        self.geom_valid = (r <= self.diameter / 2.0).ravel()

        # Distance from each bin centre to the circular wall = radius - r (clipped at 0), flattened.
        self.dist_to_wall = np.clip(self.diameter / 2.0 - r, 0.0, None).ravel()
        # Edge zone = bins within OPEN_EDGE_ZONE_THRESHOLD_CM (9.25 cm) of the wall.
        self.edge_zone_flat = self.dist_to_wall <= OPEN_EDGE_ZONE_THRESHOLD_CM

        # Number of bins inside the arena.
        self.total_arena_bins = int(self.geom_valid.sum())
        # Minimum number of validly sampled bins for the (optional) bin-coverage criterion.
        self.coverage_threshold_bins = int(np.ceil(COVERAGE_FRACTION * self.total_arena_bins))

    # Orientation step (none needed for the open field).
    def orient(self, x_cm, y_cm):
        # Return coordinates unchanged.
        return x_cm, y_cm

    # Convert positions (cm) to flat bin indices and flag samples lying on/near the arena.
    def to_bins(self, x_cm, y_cm):
        # x bin = floor(x / bin size), clipped into [0, nx-1].
        bx = np.clip((x_cm / self.bin_cm).astype(int), 0, self.nx - 1)
        # y bin = floor(y / bin size), clipped into [0, ny-1].
        by = np.clip((y_cm / self.bin_cm).astype(int), 0, self.ny - 1)
        # Flat index = x bin * ny + y bin.
        flat = bx * self.ny + by
        # Distance of each sample from the arena centre.
        r = np.hypot(x_cm - self.cx, y_cm - self.cy)
        # Sample is valid if within radius + one bin (2 cm tolerance).
        sample_valid = r <= (self.diameter / 2.0 + self.bin_cm)
        # Return bin indices and validity flags.
        return flat, sample_valid

    # Coordinates (cm) of all bin centres as a 2 x n_bins array (row 0 = x, row 1 = y).
    def bin_centres_xy(self):
        # Bin-centre x coordinates.
        xs = (np.arange(self.nx) + 0.5) * self.bin_cm
        # Bin-centre y coordinates.
        ys = (np.arange(self.ny) + 0.5) * self.bin_cm
        # 2D coordinate grids in [i, j] order.
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        # Flatten and stack into shape (2, n_bins) in flat-index order.
        return np.vstack([XX.ravel(), YY.ravel()])

    # Convert bin-grid coordinates (u, v) to arena cm.
    def edge_to_xy(self, u, v):
        """Bin-grid coordinates (bin (i, j) spans u in [i, i+1], v in [j, j+1]) -> arena cm."""
        # Multiply grid units by the bin size.
        return np.asarray(u) * self.bin_cm, np.asarray(v) * self.bin_cm

    # Draw the arena wall as a circle.
    def draw_outline(self, ax):
        # Add an unfilled dark-grey circle of the arena radius at the arena centre.
        ax.add_patch(matplotlib.patches.Circle((self.cx, self.cy), self.diameter / 2.0,
                                               # (continuation) style: no fill, grey edge, 1 pt, on top.
                                               fill=False, edgecolor='0.2', lw=1.0, zorder=5))

    # Area (cm^2) of every bin.
    def bin_areas_cm2(self):
        # All bins are bin_cm x bin_cm squares (4 cm^2).
        return np.full(self.n_bins, self.bin_cm ** 2)

    # Smooth a flat rate map over the valid bins.
    def smooth(self, fr_flat, valid_flat):
        # Reshape rates to the (nx, ny) grid.
        fr2 = fr_flat.reshape(self.nx, self.ny)
        # Reshape the validity mask to the grid.
        v2  = valid_flat.reshape(self.nx, self.ny)
        # Masked Gaussian smoothing (no wrap), flattened back.
        return _gaussian_smooth_2d(fr2, v2).ravel()

    # Connected components of a flat mask on this grid.
    def connected_components(self, qualifies_flat):
        # 8-connected components, no wrap-around.
        return _connected_components_2d_flat(qualifies_flat, self.nx, self.ny)

    # Plot a flat map as an image clipped to valid in-arena bins.
    def plot_fine(self, ax, values_flat, valid_flat, cmap, norm):
        # Start with all-NaN (NaN = drawn as 'bad' colour, white).
        grid = np.full(self.n_bins, np.nan)
        # Bins to show = valid AND inside the circle.
        m = valid_flat & self.geom_valid
        # Copy the values of the shown bins.
        grid[m] = values_flat[m]
        # Reshape to the (nx, ny) grid.
        grid2 = grid.reshape(self.nx, self.ny)
        # Show the transposed grid (rows = y) with origin bottom-left, spanning 0..diameter cm in both axes.
        im = ax.imshow(np.ma.masked_invalid(grid2.T), origin='lower',
                        # (continuation) image extent in cm.
                        extent=[0, self.diameter, 0, self.diameter],
                        # (continuation) colour map and normalisation.
                        cmap=cmap, norm=norm)
        # Draw the arena wall on top.
        ax.add_patch(matplotlib.patches.Circle((self.cx, self.cy), self.diameter / 2.0,
                                               # (continuation) outline style.
                                               fill=False, edgecolor='0.35', lw=1.0, zorder=5))
        # Equal x/y scaling so the circle is round.
        ax.set_aspect('equal')
        # Hide axes ticks and frame.
        ax.axis('off')
        # Return the image for the colour bar.
        return im

    # Plot a flat map as a pcolormesh with cm axes (used for KDE figures).
    def plot_bins_2d(self, ax, values_flat, valid_flat, cmap, norm):
        # Values where valid and inside the arena, NaN elsewhere, reshaped to (nx, ny).
        grid = np.where(valid_flat & self.geom_valid, values_flat, np.nan).reshape(self.nx, self.ny)
        # x bin edges (cm): 0, 2, ..., 60.
        x_edges = np.arange(self.nx + 1) * self.bin_cm
        # y bin edges (cm).
        y_edges = np.arange(self.ny + 1) * self.bin_cm
        # Draw each bin as a flat-coloured cell (transposed so rows = y), NaN masked.
        pcm = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(grid.T),
                            # (continuation) colour map, normalisation, flat shading.
                            cmap=cmap, norm=norm, shading='flat')
        # Draw the arena wall.
        ax.add_patch(matplotlib.patches.Circle((self.cx, self.cy), self.diameter / 2.0,
                                               # (continuation) outline style.
                                               fill=False, edgecolor='0.35', lw=1.0, zorder=5))
        # Equal aspect ratio.
        ax.set_aspect('equal')
        # x-axis label.
        ax.set_xlabel('x (cm)')
        # y-axis label.
        ax.set_ylabel('y (cm)')
        # Return the mesh for the colour bar.
        return pcm


# Geometry, binning, smoothing and plotting for the annular (circular) track; axis 0 = angle, axis 1 = radius.
class CircularTrackHandler:
    # The angular axis is periodic.
    wrap_x = True

    # Build the angle x radius bin grid of the ring.
    def __init__(self, cfg: dict, bin_cm: float):
        # Outer diameter (cm).
        self.outer_d = cfg['outer_diameter_cm']
        # Inner diameter (cm).
        self.inner_d = cfg['inner_diameter_cm']
        # Outer radius (40 cm).
        self.outer_r = self.outer_d / 2.0
        # Inner radius (36 cm).
        self.inner_r = self.inner_d / 2.0
        # Mean radius of the track (38 cm).
        self.mean_r  = (self.outer_r + self.inner_r) / 2.0
        # Ring centre x (cm) = outer radius (coordinates start at 0 at the outer wall).
        self.cx = self.outer_r
        # Ring centre y (cm) = outer radius.
        self.cy = self.outer_r
        # Width used for pixel->cm scaling = outer diameter.
        self.arena_width_cm = self.outer_d
        # Radial tolerance (cm) beyond either wall within which samples still count as on-track.
        self.radial_tol_cm = 4.0
        # Track width (cm) = outer radius - inner radius (4 cm).
        self.track_width_cm = self.outer_r - self.inner_r

        # Circumference at the mean radius = 2*pi*38 (~238.8 cm).
        circumference = 2 * np.pi * self.mean_r
        # Store it.
        self.circumference_cm = circumference
        # Angular bins: circumference / bin size, rounded to a multiple of 4 (>= 8) -> 120 bins.
        self.nx = max(8, 4 * int(round(circumference / bin_cm / 4.0)))
        # Radial bins: track width / bin size, rounded (>= 2) -> 2 bins.
        self.ny = max(2, int(round(self.track_width_cm / bin_cm)))
        # Angular width of one bin in degrees (360/120 = 3 deg).
        self.bin_width_deg = 360.0 / self.nx
        # Radial width of one bin (cm) (4/2 = 2 cm).
        self.bin_cm_y = self.track_width_cm / self.ny
        # Total bins (120 x 2 = 240).
        self.n_bins = self.nx * self.ny
        # Every bin of the ring is geometrically valid.
        self.geom_valid = np.ones(self.n_bins, dtype=bool)

        # Radial distance of each radial-bin centre from the inner wall (1 cm, 3 cm).
        r_from_inner = (np.arange(self.ny) + 0.5) * self.bin_cm_y
        # Distance to the nearer wall (inner or outer) per radial bin, repeated for every angular bin.
        self.dist_to_wall = np.tile(np.minimum(r_from_inner, self.track_width_cm - r_from_inner),
                                    # (continuation) number of repetitions = number of angular bins.
                                    self.nx)

        # Inner half of the track: radial bin index < ny/2, tiled over all angles.
        self.inner_side_flat = np.tile(np.arange(self.ny) < (self.ny // 2), self.nx)
        # Edge zone = outer half of the track (bins next to the outer wall).
        self.edge_zone_flat = ~self.inner_side_flat

        # Number of arena bins = all bins.
        self.total_arena_bins = self.n_bins
        # Minimum validly sampled bins for the optional coverage criterion.
        self.coverage_threshold_bins = int(np.ceil(COVERAGE_FRACTION * self.total_arena_bins))

    # Orientation step (none for the ring).
    def orient(self, x_cm, y_cm):
        # Return unchanged.
        return x_cm, y_cm

    # Convert (x, y) cm to flat (angle, radius) bin indices and an on-track flag.
    def to_bins(self, x_cm, y_cm):
        # Radial distance of each sample from the ring centre.
        r = np.hypot(x_cm - self.cx, y_cm - self.cy)
        # Polar angle in degrees, 0 = East, counter-clockwise, mapped into [0, 360).
        theta = np.degrees(np.arctan2(y_cm - self.cy, x_cm - self.cx)) % 360.0
        # Angular bin = floor(theta / bin width), clipped into [0, nx-1].
        bx = np.clip((theta / self.bin_width_deg).astype(int), 0, self.nx - 1)
        # Radial position measured from the inner wall, clipped to the track width.
        rel_r = np.clip(r - self.inner_r, 0.0, self.track_width_cm)
        # Radial bin = floor(rel_r / radial bin width), clipped into [0, ny-1].
        by = np.clip((rel_r / self.bin_cm_y).astype(int), 0, self.ny - 1)
        # Flat index = angular bin * ny + radial bin.
        flat = bx * self.ny + by
        # On-track if radius lies within [inner - tol, outer + tol] = [32, 44] cm.
        on_track = (r >= self.inner_r - self.radial_tol_cm) & (r <= self.outer_r + self.radial_tol_cm)
        # Return bin indices and on-track flags.
        return flat, on_track

    # Cartesian (cm) coordinates of every bin centre as a 2 x n_bins array.
    def bin_centres_xy(self):
        # Centre angle (radians) of each angular bin.
        theta = np.radians((np.arange(self.nx) + 0.5) * self.bin_width_deg)
        # Centre radius (cm) of each radial bin.
        r = self.inner_r + (np.arange(self.ny) + 0.5) * self.bin_cm_y
        # Grids of angle and radius in [i, j] order.
        TH, R = np.meshgrid(theta, r, indexing='ij')
        # x = cx + R cos(theta), flattened (row 0).
        return np.vstack([(self.cx + R * np.cos(TH)).ravel(),
                          # y = cy + R sin(theta), flattened (row 1).
                          (self.cy + R * np.sin(TH)).ravel()])

    # Convert bin-grid coordinates (u = angular bin units, v = radial bin units) to room cm on the ring.
    def edge_to_xy(self, u, v):
        """Bin-grid coordinates (u = angular bin, v = radial bin) -> room cm on the actual ring."""
        # Angle (radians) = u * angular bin width.
        th = np.radians(np.asarray(u) * self.bin_width_deg)
        # Radius = inner radius + v * radial bin width.
        r = self.inner_r + np.asarray(v) * self.bin_cm_y
        # Polar -> Cartesian about the ring centre.
        return self.cx + r * np.cos(th), self.cy + r * np.sin(th)

    # Draw the inner and outer walls.
    def draw_outline(self, ax):
        # For each wall radius...
        for rr in (self.inner_r, self.outer_r):
            # ...add an unfilled grey circle.
            ax.add_patch(matplotlib.patches.Circle((self.cx, self.cy), rr, fill=False,
                                                   # (continuation) outline style.
                                                   edgecolor='0.2', lw=0.8, zorder=5))

    # Area (cm^2) of every annular-sector bin.
    def bin_areas_cm2(self):
        # Angular bin width in radians.
        dth = np.radians(self.bin_width_deg)
        # Inner radius of each radial bin.
        r_lo = self.inner_r + np.arange(self.ny) * self.bin_cm_y
        # Outer radius of each radial bin.
        r_hi = r_lo + self.bin_cm_y
        # Sector area = 0.5 * dtheta * (r_hi^2 - r_lo^2), repeated for every angular bin.
        return np.tile(0.5 * dth * (r_hi ** 2 - r_lo ** 2), self.nx)

    # Smooth a flat rate map with wrap-around along the angular axis.
    def smooth(self, fr_flat, valid_flat):
        # Rates as (nx, ny) grid.
        fr2 = fr_flat.reshape(self.nx, self.ny)
        # Validity as (nx, ny) grid.
        v2  = valid_flat.reshape(self.nx, self.ny)
        # Masked Gaussian smoothing, periodic in angle, flattened back.
        return _gaussian_smooth_2d(fr2, v2, wrap_x=True).ravel()

    # Connected components with wrap-around in angle.
    def connected_components(self, qualifies_flat):
        # 8-connected components, angular axis periodic.
        return _connected_components_2d_flat(qualifies_flat, self.nx, self.ny, wrap_x=True)

    # Plot a flat map on a polar axis as a ring.
    def plot_fine(self, ax, values_flat, valid_flat, cmap, norm):
        # Angular bin edges (radians) 0..2*pi.
        theta_edges = np.linspace(0, 2 * np.pi, self.nx + 1)
        # Radial bin edges (cm) from inner to outer wall.
        r_edges = np.linspace(self.inner_r, self.outer_r, self.ny + 1)
        # Valid values, NaN elsewhere, as (nx, ny) grid.
        grid = np.where(valid_flat, values_flat, np.nan).reshape(self.nx, self.ny)
        # Angle 0 points East.
        ax.set_theta_zero_location('E')
        # Angles increase counter-clockwise.
        ax.set_theta_direction(1)
        # Polar pcolormesh (transposed: rows = radius).
        pcm = ax.pcolormesh(theta_edges, r_edges, grid.T, cmap=cmap, norm=norm, shading='auto')
        # Radial axis from 0 to just beyond the outer wall.
        ax.set_ylim(0, self.outer_r + 5)
        # Hide radial tick labels.
        ax.set_yticklabels([])
        # Hide the polar grid.
        ax.grid(False)
        # Return the mesh for the colour bar.
        return pcm

    # Plot a flat map "unrolled": arc length (x) versus distance from inner wall (y).
    def plot_bins_2d(self, ax, values_flat, valid_flat, cmap, norm):
        # Valid values, NaN elsewhere, as (nx, ny) grid.
        grid = np.where(valid_flat, values_flat, np.nan).reshape(self.nx, self.ny)
        # x edges = arc length at the mean radius (cm).
        x_edges = np.linspace(0.0, self.circumference_cm, self.nx + 1)
        # y edges = distance from inner wall (cm).
        y_edges = np.linspace(0.0, self.track_width_cm, self.ny + 1)
        # Flat-shaded mesh (transposed), NaN masked.
        pcm = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(grid.T),
                            # (continuation) colour map, normalisation, flat shading.
                            cmap=cmap, norm=norm, shading='flat')
        # Free aspect ratio (the strip is very long and thin).
        ax.set_aspect('auto')
        # x-axis label.
        ax.set_xlabel('Arc length at mean radius (cm; 0 = East, CCW)')
        # y-axis label.
        ax.set_ylabel('From inner wall (cm)')
        # Return mesh for the colour bar.
        return pcm


# Geometry, binning, smoothing and plotting for the rectangular linear track.
class LinearTrackHandler:
    # No periodic axis.
    wrap_x = False

    # Build the length x width bin grid.
    def __init__(self, cfg: dict, bin_cm: float):
        # Track length (cm).
        self.length = cfg['length_cm']
        # Track width (cm).
        self.width  = cfg['width_cm']
        # Bins along the length: round(80/2) = 40 (at least 4).
        self.nx = max(4, int(round(self.length / bin_cm)))
        # Bins across the width: round(8/2) = 4 (at least 2).
        self.ny = max(2, int(round(self.width  / bin_cm)))
        # Exact bin length (cm) so the bins tile the track exactly.
        self.bin_cm_x = self.length / self.nx
        # Exact bin width (cm).
        self.bin_cm_y = self.width  / self.ny
        # Total bins (40 x 4 = 160).
        self.n_bins = self.nx * self.ny
        # Width used for pixel->cm scaling = track length (the larger extent).
        self.arena_width_cm = self.length
        # Every bin is inside the track.
        self.geom_valid = np.ones(self.n_bins, dtype=bool)

        # x coordinates (cm) of bin centres.
        bx_centers_cm = (np.arange(self.nx) + 0.5) * self.bin_cm_x
        # y coordinates (cm) of bin centres.
        by_centers_cm = (np.arange(self.ny) + 0.5) * self.bin_cm_y
        # Distance of each x centre to the nearer end wall.
        dist_x = np.minimum(bx_centers_cm, self.length - bx_centers_cm)
        # Distance of each y centre to the nearer long wall.
        dist_y = np.minimum(by_centers_cm, self.width - by_centers_cm)
        # 2D grids of those distances in [i, j] order.
        DX, DY = np.meshgrid(dist_x, dist_y, indexing='ij')
        # Distance to the nearest wall of any kind, flattened.
        self.dist_to_wall = np.minimum(DX, DY).ravel()

        # Distance of each bin centre from the track's long midline = width/2 - distance to long wall.
        dist_from_midline_cm = (self.width / 2.0) - DY
        # Edge zone = bins >= 2 cm off the midline (the rows next to the long walls)...
        edge_zone_2d = ((dist_from_midline_cm >= LINEAR_EDGE_ZONE_THRESHOLD_CM) |
                        # ...OR bins within 2 cm of an end wall.
                        (DX <= LINEAR_EDGE_ZONE_THRESHOLD_CM))
        # Flatten the edge-zone mask.
        self.edge_zone_flat = edge_zone_2d.ravel()

        # Number of arena bins.
        self.total_arena_bins = self.n_bins
        # Minimum validly sampled bins for the optional coverage criterion.
        self.coverage_threshold_bins = int(np.ceil(COVERAGE_FRACTION * self.total_arena_bins))

    # Make the track horizontal: rotate sessions recorded with the track vertical.
    def orient(self, x_cm, y_cm):
        # Nothing to orient if there are no samples.
        if len(x_cm) == 0:
            # Return as is.
            return x_cm, y_cm
        # Extent of the trajectory in x.
        x_span = x_cm.max() - x_cm.min()
        # Extent of the trajectory in y.
        y_span = y_cm.max() - y_cm.min()
        # If the trajectory is taller than wide, the track was vertical:
        if y_span > x_span:
            # rotate 90 deg CCW: new x = -old y...
            xr = -y_cm
            # ...new y = old x.
            yr = x_cm
            # Shift the new x so it starts at 0.
            xr = xr - xr.min()
            # Shift the new y so it starts at 0.
            yr = yr - yr.min()
            # Return the rotated trajectory.
            return xr, yr
        # Already horizontal: return unchanged.
        return x_cm, y_cm

    # Convert positions to flat bin indices (every sample is kept).
    def to_bins(self, x_cm, y_cm):
        # Clip x into [0, length].
        px = np.clip(x_cm, 0, self.length)
        # Clip y into [0, width].
        py = np.clip(y_cm, 0, self.width)
        # Length bin = floor(x / bin length), clipped into [0, nx-1].
        bx = np.clip((px / self.bin_cm_x).astype(int), 0, self.nx - 1)
        # Width bin = floor(y / bin width), clipped into [0, ny-1].
        by = np.clip((py / self.bin_cm_y).astype(int), 0, self.ny - 1)
        # Flat index = length bin * ny + width bin.
        flat = bx * self.ny + by
        # All samples are valid (positions outside are clipped to the edge bins).
        sample_valid = np.ones_like(px, dtype=bool)
        # Return bin indices and validity flags.
        return flat, sample_valid

    # Coordinates (cm) of all bin centres, 2 x n_bins.
    def bin_centres_xy(self):
        # Bin-centre x coordinates.
        xs = (np.arange(self.nx) + 0.5) * self.bin_cm_x
        # Bin-centre y coordinates.
        ys = (np.arange(self.ny) + 0.5) * self.bin_cm_y
        # Coordinate grids in [i, j] order.
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        # Flatten and stack.
        return np.vstack([XX.ravel(), YY.ravel()])

    # Convert bin-grid coordinates (u, v) to cm.
    def edge_to_xy(self, u, v):
        # Scale each axis by its bin size.
        return np.asarray(u) * self.bin_cm_x, np.asarray(v) * self.bin_cm_y

    # Draw the track as a rectangle.
    def draw_outline(self, ax):
        # Unfilled rectangle from (0, 0) of size length x width.
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), self.length, self.width,
                                                  # (continuation) outline style.
                                                  fill=False, edgecolor='0.2', lw=1.0, zorder=5))

    # Area (cm^2) of every bin.
    def bin_areas_cm2(self):
        # All bins have area bin_cm_x * bin_cm_y.
        return np.full(self.n_bins, self.bin_cm_x * self.bin_cm_y)

    # Smooth a flat rate map over the valid bins.
    def smooth(self, fr_flat, valid_flat):
        # Rates as (nx, ny) grid.
        fr2 = fr_flat.reshape(self.nx, self.ny)
        # Validity as (nx, ny) grid.
        v2  = valid_flat.reshape(self.nx, self.ny)
        # Masked Gaussian smoothing (no wrap), flattened.
        return _gaussian_smooth_2d(fr2, v2).ravel()

    # Connected components on this grid.
    def connected_components(self, qualifies_flat):
        # 8-connected, no wrap.
        return _connected_components_2d_flat(qualifies_flat, self.nx, self.ny)

    # Plot a flat map as an image of the track.
    def plot_fine(self, ax, values_flat, valid_flat, cmap, norm):
        # All-NaN grid.
        grid = np.full(self.n_bins, np.nan)
        # Fill the valid bins.
        grid[valid_flat] = values_flat[valid_flat]
        # Reshape to (nx, ny).
        grid2 = grid.reshape(self.nx, self.ny)
        # Show transposed grid (rows = width), origin bottom-left, in cm.
        im = ax.imshow(np.ma.masked_invalid(grid2.T), origin='lower',
                        # (continuation) extent in cm.
                        extent=[0, self.length, 0, self.width],
                        # (continuation) equal aspect, colour map, normalisation.
                        aspect='equal', cmap=cmap, norm=norm)
        # Hide axes.
        ax.axis('off')
        # Return image for the colour bar.
        return im

    # Plot a flat map as a pcolormesh with cm axes.
    def plot_bins_2d(self, ax, values_flat, valid_flat, cmap, norm):
        # Valid values, NaN elsewhere, as (nx, ny).
        grid = np.where(valid_flat, values_flat, np.nan).reshape(self.nx, self.ny)
        # x edges (cm).
        x_edges = np.arange(self.nx + 1) * self.bin_cm_x
        # y edges (cm).
        y_edges = np.arange(self.ny + 1) * self.bin_cm_y
        # Flat-shaded mesh, transposed, NaN masked.
        pcm = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(grid.T),
                            # (continuation) colour map, normalisation, flat shading.
                            cmap=cmap, norm=norm, shading='flat')
        # Free aspect ratio.
        ax.set_aspect('auto')
        # x-axis label.
        ax.set_xlabel('Track length (cm)')
        # y-axis label.
        ax.set_ylabel('Track width (cm)')
        # Return mesh for the colour bar.
        return pcm


# Factory: create the right handler class for an arena config.
def make_handler(cfg: dict):
    # Shape tag of the arena.
    shape = cfg['shape']
    # 'circle' -> open field.
    if shape == 'circle':
        # Open-field handler with 2 cm bins.
        return OpenFieldHandler(cfg, target_bin_cm)
    # 'ring' -> circular track.
    if shape == 'ring':
        # Circular-track handler with ~2 cm bins.
        return CircularTrackHandler(cfg, target_bin_cm)
    # 'linear' -> linear track.
    if shape == 'linear':
        # Linear-track handler with 2 cm bins.
        return LinearTrackHandler(cfg, target_bin_cm)
    # Any other tag is a configuration error.
    raise ValueError(f'Unknown arena shape: {shape}')


# ============================================================================
# Core per-cell rate map
# ============================================================================

# Field index: min-max normalise a cell's smoothed rate map to 0..1 over its valid bins.
def field_index_map(fr_flat: np.ndarray, valid_flat: np.ndarray) -> np.ndarray:
    # Output initialised to NaN (invalid bins stay NaN).
    fi = np.full(fr_flat.shape, np.nan, dtype=np.float64)
    # If the cell has no valid bin...
    if not valid_flat.any():
        # ...return the all-NaN map.
        return fi
    # Rates of the valid bins.
    vals = fr_flat[valid_flat]
    # Minimum and maximum valid rate.
    vmin, vmax = float(vals.min()), float(vals.max())
    # Dynamic range of the map.
    rng = vmax - vmin
    # FI = (rate - min) / (max - min); a perfectly flat map gets 0 everywhere.
    fi[valid_flat] = (vals - vmin) / rng if rng > 0 else 0.0
    # Return the field-index map.
    return fi


# Place-field mask of one cell: connected groups (>= MIN_FIELD_BINS) of bins above mean rate and above 20 % of peak.
def extract_place_field_mask(cell: dict, handler) -> np.ndarray:
    # Valid-bin mask of the cell.
    valid     = cell['valid']
    # Smoothed rate map of the cell.
    fr_smooth = cell['fr_smooth']
    # Field mask initialised to all False.
    field_mask = np.zeros(handler.n_bins, dtype=bool)
    # No valid bins -> no field.
    if not valid.any():
        # Return the empty mask.
        return field_mask

    # Peak smoothed rate over valid bins.
    peak_smooth = float(fr_smooth[valid].max())
    # Occupancy-weighted mean rate of the cell.
    mean_fr     = cell['mean_fr']
    # Candidate field bins: valid AND rate > mean rate AND rate > FIELD_PEAK_FRAC * peak.
    qualifies = valid & (fr_smooth > mean_fr) & (fr_smooth > FIELD_PEAK_FRAC * peak_smooth)
    # No candidate bin -> no field.
    if not qualifies.any():
        # Return the empty mask.
        return field_mask

    # For every 8-connected group of candidate bins...
    for component in handler.connected_components(qualifies):
        # ...that is at least MIN_FIELD_BINS (9) bins large...
        if len(component) >= MIN_FIELD_BINS:
            # ...mark its bins as place field.
            field_mask[component] = True
    # Return the union of all accepted fields.
    return field_mask


# Rate maps and spatial metrics (peak, mean, Skaggs information, sparsity) from per-bin spike counts and occupancy.
def rate_maps_from_counts(spike_map: np.ndarray, occ_map: np.ndarray, handler) -> dict:
    """Rate map, smoothed rate map, field-index map and SIR / sparsity from per-bin spike counts
    and occupancy (s). Bins with < min_occ_s occupancy or outside the arena are invalid."""
    # Number of bins of this arena.
    n_bins = handler.n_bins
    # Valid bins: occupancy >= min_occ_s (1 s) AND inside the arena.
    valid = (occ_map >= min_occ_s) & handler.geom_valid

    # Raw rate map initialised to 0.
    fr_raw = np.zeros(n_bins, dtype=np.float64)
    # Raw rate (Hz) = spike count / occupancy time, valid bins only.
    fr_raw[valid] = spike_map[valid] / occ_map[valid]
    # Masked Gaussian smoothing of the raw rate map.
    fr_smooth = handler.smooth(fr_raw, valid)
    # Field-index map = min-max normalised smoothed map.
    fi_map = field_index_map(fr_smooth, valid)

    # Collect the maps into the result dict.
    result = dict(fr_raw=fr_raw, fr_smooth=fr_smooth, fi_map=fi_map, occ_map=occ_map, valid=valid)
    # If there are no valid bins...
    if not valid.any():
        # ...all metrics are 0.
        result.update(peak_fr=0.0, mean_fr=0.0, sir=0.0, sparsity=0.0)
        # Return early.
        return result

    # Total valid occupancy time (s).
    total_occ = occ_map[valid].sum()
    # Occupancy probability of each valid bin: p_i = t_i / sum(t).
    pi = occ_map[valid] / total_occ
    # Smoothed rate of each valid bin: r_i.
    ri = fr_smooth[valid]
    # Occupancy-weighted mean rate: r_mean = sum(p_i * r_i).
    r_mean = float(np.sum(pi * ri))
    # Peak smoothed rate.
    peak_fr = float(fr_smooth[valid].max())

    # Spatial information (bits/spike) defaults to 0.
    sir = 0.0
    # Only defined if the cell fires at all.
    if r_mean > 0:
        # Bins with non-zero rate (0 * log 0 is taken as 0).
        nz = ri > 0
        # Relative rate r_i / r_mean.
        ratio = ri[nz] / r_mean
        # Skaggs information: SI = sum p_i * (r_i/r_mean) * log2(r_i/r_mean).
        sir = float(np.sum(pi[nz] * ratio * np.log2(ratio)))

    # Sparsity numerator term: sum(p_i * r_i).
    spar_num = float(np.sum(pi * ri))
    # Sparsity denominator: sum(p_i * r_i^2).
    spar_den = float(np.sum(pi * ri ** 2))
    # Sparsity = (sum p_i r_i)^2 / sum p_i r_i^2 (0 if denominator is 0).
    sparsity = float((spar_num ** 2) / spar_den) if spar_den > 0 else 0.0

    # Store metrics rounded to 4 decimals.
    result.update(peak_fr=round(peak_fr, 4), mean_fr=round(r_mean, 4),
                  # (continuation) SI and sparsity.
                  sir=round(sir, 4), sparsity=round(sparsity, 4))
    # Return maps + metrics.
    return result


# Peak bin of the smoothed map, peak bin of the raw map and the place-field mask of one cell.
def _cell_peaks_and_field(cell: dict, handler) -> dict:
    """Peak bins of the smoothed and raw maps, and the place-field mask, as pooled below."""
    # Default: no smoothed peak bin.
    peak_bin = None
    # Default: no raw peak bin.
    peak_bin_raw = None
    # Only if the cell has valid bins:
    if cell['valid'].any():
        # Index of the maximum smoothed rate among valid bins (invalid bins set to -inf).
        peak_bin = int(np.argmax(np.where(cell['valid'], cell['fr_smooth'], -np.inf)))
        # Index of the maximum raw rate among valid bins.
        peak_bin_raw = int(np.argmax(np.where(cell['valid'], cell['fr_raw'], -np.inf)))
    # Return both peaks and the place-field mask.
    return dict(peak_bin=peak_bin, peak_bin_raw=peak_bin_raw,
                # (continuation) place-field mask.
                field_mask=extract_place_field_mask(cell, handler))


# Assign spikes to tracking frames, apply the speed filter, build occupancy & spike-count maps, then rate maps.
def compute_cell_ratemap(frames: dict, spike_ts: np.ndarray, handler) -> dict | None:
    # Unpack frame times, frame bins, moving flags and dwell times.
    t, bin_idx, moving, dt_frames = frames['t'], frames['bin_idx'], frames['moving'], frames['dt']

    # For each spike, the first frame index whose time is >= the spike time.
    idx   = np.searchsorted(t, spike_ts, side='left')
    # Candidate frame just before the spike (clipped into range).
    idx_l = np.clip(idx - 1, 0, len(t) - 1)
    # Candidate frame at/after the spike (clipped into range).
    idx_r = np.clip(idx,     0, len(t) - 1)
    # Time distance from the spike to the left candidate.
    dist_l  = np.abs(spike_ts - t[idx_l])
    # Time distance from the spike to the right candidate.
    dist_r  = np.abs(spike_ts - t[idx_r])
    # Nearest frame (ties go to the left frame).
    nearest = np.where(dist_l <= dist_r, idx_l, idx_r)
    # Time distance to the nearest frame.
    min_dist = np.minimum(dist_l, dist_r)

    # Spike is matched if its nearest frame is within MAX_GAP_US (50 ms).
    matched     = min_dist <= MAX_GAP_US
    # Spike is used if matched AND its frame passes the speed filter.
    valid_spike = matched & moving[nearest]
    # Number of spikes used.
    n_spikes    = int(valid_spike.sum())
    # Number of matched spikes discarded by the speed filter (diagnostic).
    n_spikes_speed_excluded = int((matched & ~moving[nearest]).sum())

    # If immobile/artefact frames are also removed from occupancy:
    if SPEED_FILTER_OCCUPANCY:
        # Map each original frame index to its index in the moving-only array (cumulative count - 1).
        new_index = np.cumsum(moving) - 1
        # Re-index the used spikes' frames into the moving-only array.
        spike_frame = new_index[nearest[valid_spike]]
        # Keep only moving frames in time, bin index and dwell time.
        t, bin_idx, dt_frames = t[moving], bin_idx[moving], dt_frames[moving]
        # Too few moving frames -> unusable.
        if len(t) < 2:
            # Signal failure.
            return None
    # Otherwise all frames stay in occupancy:
    else:
        # Spike frames index the full frame array.
        spike_frame = nearest[valid_spike]

    # Number of bins of the arena.
    n_bins = handler.n_bins
    # Occupancy map (seconds per bin), zeros.
    occ_map   = np.zeros(n_bins, dtype=np.float64)
    # Spike-count map, zeros.
    spike_map = np.zeros(n_bins, dtype=np.float64)
    # Occupancy: add each frame's dwell time to its bin (unbuffered, handles repeated bins).
    np.add.at(occ_map,   bin_idx, dt_frames)
    # Spike counts: add 1 to the bin of each used spike's frame.
    np.add.at(spike_map, bin_idx[spike_frame], 1.0)

    # Rate maps + metrics from counts and occupancy.
    result = rate_maps_from_counts(spike_map, occ_map, handler)
    # Attach spike counts and the frame-level data needed by the bootstrap.
    result.update(n_spikes=n_spikes, n_spikes_speed_excluded=n_spikes_speed_excluded,
                  # (continuation) frame bins, spike frames, frame times, number of bins.
                  bin_idx=bin_idx, spike_frame=spike_frame, t=t, n_bins=n_bins)
    # Return the cell record.
    return result


# Circular-shift shuffle test: is the cell's spatial information above the 95th percentile of shifted-spike SIs?
def run_bootstrap_generic(handler, cell: dict, real_sir: float, n_bootstrap: int = N_BOOTSTRAP,
                          # Per-cell random seed (None -> global seed).
                          seed: int | None = None) -> dict:
    # Bin index of every (occupancy) frame.
    bin_idx     = cell['bin_idx']
    # Frame index of every used spike.
    spike_frame = cell['spike_frame']
    # Frame times.
    t           = cell['t']
    # Occupancy map (kept fixed during shuffling).
    occ_map     = cell['occ_map']
    # Valid bins (kept fixed during shuffling).
    valid       = cell['valid']
    # Number of bins.
    n_bins      = cell['n_bins']

    # No spikes -> test undefined.
    if len(spike_frame) == 0:
        # NaN statistics, not significant.
        return dict(bootstrap_mean=float('nan'), bootstrap_p95=float('nan'), bootstrap_sig=False)

    # Number of frames in the session.
    n_frames = len(t)
    # Minimum shift = 20 s worth of frames (600 frames).
    MARGIN_FRAMES = int(20 * fps)
    # Session too short to shift by >= 20 s on both sides...
    if n_frames <= 2 * MARGIN_FRAMES:
        # ...test cannot be run (sig = None).
        return dict(bootstrap_mean=float('nan'), bootstrap_p95=float('nan'), bootstrap_sig=None)

    # Seeded random generator (per-cell seed if given).
    rand = random.Random(BOOTSTRAP_SEED if seed is None else seed)
    # Array of shuffled SI values.
    sir_i = np.zeros(n_bootstrap, dtype=np.float64)
    # Repeat n_bootstrap times:
    for i in range(n_bootstrap):
        # Random shift (frames) between 20 s and (session length - 20 s), inclusive.
        rnd = rand.randint(MARGIN_FRAMES, n_frames - MARGIN_FRAMES)
        # Circularly shift every spike's frame index by the same amount (wrap at session end).
        shuf_frame = (spike_frame + rnd) % n_frames

        # Shuffled spike-count map, zeros.
        spike_map = np.zeros(n_bins, dtype=np.float64)
        # Count shifted spikes per bin.
        np.add.at(spike_map, bin_idx[shuf_frame], 1.0)

        # Shuffled raw rate map, zeros.
        fr_raw = np.zeros(n_bins, dtype=np.float64)
        # Raw rate = shifted counts / (unchanged) occupancy in valid bins.
        fr_raw[valid] = spike_map[valid] / occ_map[valid]
        # Smooth exactly as for the real map.
        fr_smooth = handler.smooth(fr_raw, valid)

        # Total valid occupancy (s).
        total_occ = occ_map[valid].sum()
        # Occupancy probabilities p_i.
        pi = occ_map[valid] / total_occ
        # Shuffled smoothed rates r_i.
        ri = fr_smooth[valid]
        # Mean rate sum(p_i r_i).
        r_mean = float(np.sum(pi * ri))
        # If the mean rate is not positive...
        if r_mean <= 0:
            # ...SI of this shuffle is 0...
            sir_i[i] = 0.0
            # ...go to the next shuffle.
            continue
        # Non-zero-rate bins.
        nz = ri > 0
        # Relative rate r_i / r_mean.
        ratio = ri[nz] / r_mean
        # Skaggs SI of this shuffle.
        sir_i[i] = float(np.sum(pi[nz] * ratio * np.log2(ratio)))

    # Mean of the shuffle distribution.
    bootstrap_mean = float(np.mean(sir_i))
    # 95th percentile of the shuffle distribution.
    bootstrap_p95  = float(np.percentile(sir_i, 95))
    # Significant if the real SI exceeds the 95th percentile.
    bootstrap_sig  = bool(real_sir > bootstrap_p95)
    # Return rounded statistics and the significance flag.
    return dict(bootstrap_mean=round(bootstrap_mean, 4),
                # (continuation) 95th percentile.
                bootstrap_p95=round(bootstrap_p95, 4),
                # (continuation) significance flag.
                bootstrap_sig=bootstrap_sig)


# Percentage of valid occupancy time spent in the edge zone and in the centre zone.
def _zone_occupancy_pct(handler, occ_map: np.ndarray, valid_mask: np.ndarray) -> tuple:
    # Total occupancy over valid bins (s).
    total_occ_s = float(occ_map[valid_mask].sum())
    # No occupancy -> 0 %, 0 %.
    if total_occ_s <= 0:
        # Return zeros.
        return 0.0, 0.0
    # Valid bins in the edge zone.
    edge_mask    = handler.edge_zone_flat & valid_mask
    # Occupancy in the edge zone (s).
    edge_occ_s   = float(occ_map[edge_mask].sum())
    # Occupancy in the centre zone = total - edge.
    centre_occ_s = total_occ_s - edge_occ_s
    # Return (edge %, centre %).
    return 100.0 * edge_occ_s / total_occ_s, 100.0 * centre_occ_s / total_occ_s


# Session-level zone coverage (edge %, centre %) from the tracking file alone.
def session_zone_coverage(csv_path: str, handler) -> tuple | None:
    # Per-frame session table.
    frames = _session_frames(csv_path, handler)
    # Unusable tracking...
    if frames is None:
        # ...no coverage.
        return None
    # Frames counted in occupancy.
    occ = frames['occ_frame']
    # Occupancy map zeros.
    occ_map = np.zeros(handler.n_bins, dtype=np.float64)
    # Sum dwell time per bin over the occupancy frames.
    np.add.at(occ_map, frames['bin_idx'][occ], frames['dt'][occ])
    # Valid bins: >= min_occ_s and inside the arena.
    valid_mask = (occ_map >= min_occ_s) & handler.geom_valid
    # Edge / centre occupancy percentages.
    return _zone_occupancy_pct(handler, occ_map, valid_mask)


# Full processing of one unit (.ntt file) in one session -> per-cell result dict (or None).
def process_unit(csv_path: str, ntt_path: str, ntt_file: str, session_name: str, handler) -> dict | None:
    # Per-frame session table (tracking is re-loaded for each unit).
    frames = _session_frames(csv_path, handler)
    # Unusable tracking...
    if frames is None:
        # ...skip the unit.
        return None

    # Memory-map the .ntt file as spike records, skipping its 16 kB text header.
    spike_data = np.memmap(ntt_path, dtype=ntt_dtype, mode='r', offset=16 * 1024)
    # Keep only spikes assigned to a cluster (cell_number != 0).
    spike_data = spike_data[spike_data['cell_number'] != 0]
    # Spike timestamps (microseconds) as float, sorted ascending.
    spike_ts = np.sort(spike_data['timestamp'].astype(np.float64))
    # No spikes...
    if len(spike_ts) == 0:
        # ...skip.
        return None

    # Occupancy, spike maps, rate maps and metrics for this cell.
    cell = compute_cell_ratemap(frames, spike_ts, handler)
    # Rate map could not be built...
    if cell is None:
        # ...skip.
        return None

    # Number of validly sampled bins.
    covered_bins = int(cell['valid'].sum())
    # Optional bin-coverage criterion: too few sampled bins...
    if USE_BIN_COVERAGE_CRITERION and covered_bins < handler.coverage_threshold_bins:
        # ...skip.
        return None

    # Circular-shift bootstrap of SI with a reproducible seed from (session, unit) -- diagnostic only.
    boot = run_bootstrap_generic(handler, cell, cell['sir'],
                                 # (continuation) per-cell seed.
                                 seed=_stable_seed(session_name, ntt_file))

    # Assemble the per-cell record.
    return dict(
        # Identifiers.
        session=session_name, unit=ntt_file,
        # Spike counts.
        n_spikes=cell['n_spikes'], n_spikes_speed_excluded=cell['n_spikes_speed_excluded'],
        # Peak and mean rate.
        peak_fr=cell['peak_fr'], mean_fr=cell['mean_fr'],
        # Spatial information and sparsity.
        sir=cell['sir'], sparsity=cell['sparsity'],
        # Bootstrap significance flag.
        bootstrap_sig=boot.get('bootstrap_sig'),
        # Maps: raw rate, smoothed rate, field index, validity, occupancy.
        fr_raw=cell['fr_raw'], fr_smooth=cell['fr_smooth'], fi_map=cell['fi_map'], valid=cell['valid'], occ_map=cell['occ_map'],
        # Peak bins and place-field mask merged into the record.
        **_cell_peaks_and_field(cell, handler),
    # End of the record.
    )


# ============================================================================
# Session discovery + batch scan per arena
# ============================================================================

# Walk ROOT_DIRECTORY and list every session folder belonging to one arena.
def find_arena_sessions(arena_key: str, handler) -> list:
    """(session_name, dirpath, csv_path, ntt_files) for every folder of this arena holding exactly
    one tracking file and at least one .ntt file (and, with USE_ZONE_COVERAGE_CRITERION, passing
    the zone-coverage criterion)."""
    # Normalised absolute path of the output folder (so it can be excluded from the walk).
    out_dir = os.path.normcase(os.path.abspath(OUTPUT_DIR))
    # Accumulator of session tuples.
    sessions = []
    # Recursively visit every folder under ROOT_DIRECTORY.
    for dirpath, dirnames, filenames in os.walk(ROOT_DIRECTORY):
        # Prune the output folder from further descent (in-place edit of dirnames).
        dirnames[:] = [d for d in dirnames
                       # (continuation) keep sub-folders that are not the output folder.
                       if os.path.normcase(os.path.abspath(os.path.join(dirpath, d))) != out_dir]
        # Skip folders whose path does not name this arena.
        if _detect_arena_key(dirpath) != arena_key:
            # Next folder.
            continue
        # All .csv / .xlsx files in the folder.
        tracking_files_all = [f for f in filenames if f.lower().endswith(('.csv', '.xlsx'))]
        # In cm mode...
        if COORD_UNITS == 'cm':
            # ...only files ending in '_cm.csv' are tracking files.
            tracking_files = [f for f in tracking_files_all if f.lower().endswith('_cm.csv')]
        # In pixel mode...
        else:
            # ...the tracking file is any csv/xlsx NOT ending in '_cm.csv'.
            tracking_files = [f for f in tracking_files_all if not f.lower().endswith('_cm.csv')]
        # All .ntt files (units) in the folder, sorted by name.
        ntt_files = sorted(f for f in filenames if f.lower().endswith('.ntt'))
        # A session needs exactly one tracking file and at least one unit...
        if len(tracking_files) != 1 or not ntt_files:
            # ...otherwise skip the folder.
            continue
        # Full path of the tracking file.
        csv_path = os.path.join(dirpath, tracking_files[0])

        # Optional session-level zone-coverage criterion:
        if USE_ZONE_COVERAGE_CRITERION:
            # Try to compute the edge / centre occupancy...
            try:
                # ...from the tracking file.
                zone_cov = session_zone_coverage(csv_path, handler)
            # Any error...
            except Exception:
                # ...counts as unusable.
                zone_cov = None
            # Unusable tracking...
            if zone_cov is None:
                # ...skip the session.
                continue
            # Unpack edge % and centre %.
            edge_pct, centre_pct = zone_cov
            # Either zone under-sampled...
            if edge_pct < MIN_ZONE_COVERAGE_PCT or centre_pct < MIN_ZONE_COVERAGE_PCT:
                # ...skip the session.
                continue

        # Record (session name relative to ROOT, folder, tracking path, unit list).
        sessions.append((os.path.relpath(dirpath, ROOT_DIRECTORY), dirpath, csv_path, ntt_files))
    # Return sessions sorted by name.
    return sorted(sessions)


# Process every unit of every session of one arena in parallel; return the successful per-cell records.
def collect_arena_results(handler, sessions: list, arena_key: str = '') -> list:
    # One job per (session, unit).
    jobs = [(session_name, dirpath, csv_path, ntt_file)
            # (continuation) iterate sessions, then the units within each session.
            for session_name, dirpath, csv_path, ntt_files in sessions for ntt_file in ntt_files]

    # Worker function executed by a thread for one job; returns (record or None, reason the unit was dropped or None).
    def _job(args):
        # Unpack the job tuple.
        session_name, dirpath, csv_path, ntt_file = args
        # Full path of the .ntt file.
        ntt_path = os.path.join(dirpath, ntt_file)
        # Try to process the unit...
        try:
            # ...returning its record (or None).
            r = process_unit(csv_path, ntt_path, ntt_file, session_name, handler)
        # Any error drops the unit...
        except Exception as e:
            # ...and its exception type and message are reported as the reason.
            return None, f'error: {type(e).__name__}: {e}'
        # process_unit returned None: unusable tracking, no sorted spikes, too few moving frames or coverage criterion.
        if r is None:
            # Report the unit as excluded.
            return None, 'excluded (unusable tracking, no spikes, too few moving frames or coverage criterion)'
        # Successful unit: record and no drop reason.
        return r, None

    # Accumulator of successful records.
    results = []
    # Accumulator of (session, unit, reason) for every dropped unit.
    dropped = []
    # Thread pool with MAX_WORKERS threads.
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Run all jobs; results come back in job order, so each can be paired with its job tuple.
        for job, (r, reason) in zip(jobs, executor.map(_job, jobs)):
            # Keep only successful units...
            if r is not None:
                # ...append the record.
                results.append(r)
            # Otherwise...
            else:
                # ...remember which unit was dropped and why.
                dropped.append((job[0], job[3], reason))
    # Report how many of the units were dropped...
    print(f'[{arena_key}] {len(dropped)} of {len(jobs)} units dropped')
    # ...and list each dropped unit with its reason.
    for session_name, ntt_file, reason in dropped:
        # One line per dropped unit.
        print(f'    [DROPPED] {session_name} / {ntt_file}: {reason}')
    # Return all per-cell records of this arena.
    return results


# ============================================================================
# Pooling across cells / days
# ============================================================================

# Overall mean field-index map: per bin, mean FI over the cells that validly sampled that bin.
def pool_fine_map(handler, results: list) -> tuple:
    # All cells are pooled.
    place = results
    # No cells...
    if not place:
        # ...all-NaN map and an all-False validity mask.
        return np.full(handler.n_bins, np.nan), np.zeros(handler.n_bins, dtype=bool)
    # Matrix cells x bins, initialised to NaN.
    stack = np.full((len(place), handler.n_bins), np.nan)
    # For each cell...
    for i, r in enumerate(place):
        # ...fill its row with its FI values in its valid bins (other bins stay NaN).
        stack[i, r['valid']] = r['fi_map'][r['valid']]
    # Silence the warning raised for bins no cell sampled (all NaN column).
    with warnings.catch_warnings():   # bins no cell sampled are all-NaN -> NaN, as intended
        # Ignore RuntimeWarnings inside this block.
        warnings.simplefilter('ignore', RuntimeWarning)
        # Column-wise mean ignoring NaN = mean FI over cells that sampled each bin.
        mean_map = np.nanmean(stack, axis=0)
    # Bins sampled by at least one cell.
    any_valid = ~np.all(np.isnan(stack), axis=0)
    # Return pooled map and its validity mask.
    return mean_map, any_valid


# Field-only mean map: per bin, sum of FI inside fields / number of cells that sampled the bin (non-field = 0).
def pool_field_only_map(handler, results: list) -> tuple:
    # All cells are pooled.
    place = results
    # No cells...
    if not place:
        # ...all-NaN map and all-False mask.
        return np.full(handler.n_bins, np.nan), np.zeros(handler.n_bins, dtype=bool)

    # Per-bin sum of FI over cells whose field covers the bin.
    fi_sum   = np.zeros(handler.n_bins, dtype=np.float64)
    # Per-bin number of cells that validly sampled the bin.
    n_sampled = np.zeros(handler.n_bins, dtype=np.float64)
    # Per-bin number of cells whose field covers the bin.
    n_field   = np.zeros(handler.n_bins, dtype=np.int64)
    # For each cell:
    for r in place:
        # its valid bins,
        valid = r['valid']
        # count it as having sampled those bins,
        n_sampled[valid] += 1.0
        # its field bins (restricted to valid bins),
        m = r['field_mask'] & valid
        # add its FI in the field bins to the sum,
        fi_sum[m] += r['fi_map'][m]
        # and count it as a field cell in those bins.
        n_field[m] += 1

    # Output map initialised to NaN.
    mean_map = np.full(handler.n_bins, np.nan)
    # Bins inside at least one cell's field (and sampled by at least one cell).
    in_any_field = (n_field > 0) & (n_sampled > 0)
    # Mean = FI sum over field cells / all sampling cells (non-field sampling cells contribute 0).
    mean_map[in_any_field] = fi_sum[in_any_field] / n_sampled[in_any_field]
    # Return the field-only map and its mask.
    return mean_map, in_any_field


# Peak-proportion map: % of cells whose raw-rate peak lies in each bin.
def pool_peak_proportion_map(handler, results: list) -> tuple:
    # Number of cells.
    n_cells = len(results)
    # Union of valid bins across cells.
    visited = np.zeros(handler.n_bins, dtype=bool)
    # Number of peaks per bin.
    counts  = np.zeros(handler.n_bins, dtype=np.float64)
    # For each cell:
    for r in results:
        # add its valid bins to the union,
        visited |= r['valid']
        # and if it has a raw peak bin...
        if r['peak_bin_raw'] is not None:
            # ...count one peak there.
            counts[r['peak_bin_raw']] += 1.0

    # Output map initialised to NaN.
    pct_map = np.full(handler.n_bins, np.nan)
    # If there are cells...
    if n_cells > 0:
        # ...percentage = peaks in bin / number of cells * 100 (sampled bins only).
        pct_map[visited] = counts[visited] / n_cells * 100.0
    # Return map, mask and number of cells.
    return pct_map, visited, n_cells


# Union of the valid-bin masks of all cells (= KDE domain).
def _union_valid(handler, results: list) -> np.ndarray:
    # Start with no bins.
    visited = np.zeros(handler.n_bins, dtype=bool)
    # For each cell...
    for r in results:
        # ...OR in its valid bins.
        visited |= r['valid']
    # Return the union.
    return visited


# map kind -> (pool fn -> (values, values_are_mass), title, KDE 'scaled' units, map colourbar
#              label, file stem: KDE figure = KDE_<stem>)
# Registry of the three pooled-map kinds that are KDE-smoothed (and Duong-tested).
_KDE_MAP_KINDS = {
    # Overall mean FI map: values are intensities (multiplied by bin area to become mass).
    'overall': (lambda h, res: (pool_fine_map(h, res)[0], False),
                # (continuation) title, scaled units, colour-bar label,
                'Overall mean field index map', 'field index', 'Field index (a.u.)',
                # (continuation) file stem.
                'FigS1H_MeanFieldIndex'),
    # Field-only mean FI map: values are intensities.
    'field_only': (lambda h, res: (pool_field_only_map(h, res)[0], False),
                   # (continuation) title, scaled units,
                   'Field-only mean field index map', 'field index',
                   # (continuation) colour-bar label, file stem.
                   'Field index, background = 0 (a.u.)', 'FieldOnly_MeanFieldIndex'),
    # Peak-proportion map: values are already masses (% of cells per bin).
    'peak': (lambda h, res: (pool_peak_proportion_map(h, res)[0], True),
             # (continuation) title, scaled units, colour-bar label,
             'Peak proportion map', '% of cells per bin', '% of cells with peak in bin',
             # (continuation) file stem.
             'PeakProportion_Map'),
# End of the registry.
}



# ============================================================================
# STAGE 2 -- 2D kernel density estimate of the pooled maps (scipy gaussian_kde, Scott's rule)
# ============================================================================

# Fraction of a Gaussian kernel (covariance cov) centred at each evaluation point that lies inside the domain bins.
def _kernel_mass_in_domain(eval_pts: np.ndarray, dom_pts: np.ndarray, dom_area: np.ndarray,
                           # Kernel covariance (2 x 2).
                           cov: np.ndarray) -> np.ndarray:
    """m(x) = sum over domain bins y of N(x - y; 0, cov) * area(y): the share of a Gaussian kernel
    of covariance cov centred at x that falls inside the domain (Riemann sum over bins)."""
    # Inverse covariance (precision matrix).
    inv = np.linalg.inv(cov)
    # Bivariate normal normalising constant 1 / (2*pi*sqrt(det cov)).
    norm_factor = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(cov)))
    # Pairwise difference vectors x - y, shape (2, n_eval, n_domain).
    d = eval_pts[:, :, None] - dom_pts[:, None, :]
    # Squared Mahalanobis distance (x-y)^T inv (x-y) for every pair, shape (n_eval, n_domain).
    q = np.einsum('imn,ij,jmn->mn', d, inv, d)
    # Gaussian density for every pair times bin area, summed over domain bins -> kernel mass per evaluation point.
    return norm_factor * (np.exp(-0.5 * q) @ dom_area)


# Weighted 2D Gaussian KDE of one pooled map over its sampled domain, optionally edge-corrected.
def kde_density_map(handler, values_flat: np.ndarray, domain_flat: np.ndarray,
                    # True if values are already masses (counts / %), False if intensities per bin.
                    values_are_mass: bool = False) -> dict | None:
    # Make sure the domain mask is boolean.
    domain_flat = domain_flat.astype(bool)
    # Bin-centre coordinates (cm), shape (2, n_bins).
    pts  = handler.bin_centres_xy()
    # Bin areas (cm^2).
    area = handler.bin_areas_cm2()

    # Map values inside the domain and finite; 0 elsewhere.
    vals = np.where(domain_flat & np.isfinite(values_flat), values_flat, 0.0)
    # Negative values set to 0 (weights must be non-negative).
    vals = np.clip(vals, 0.0, None)
    # Mass of each bin: value itself if already mass, else value * bin area (integral over the bin).
    mass = vals if values_are_mass else vals * area
    # Bins with positive mass are the KDE data points.
    fit = mass > 0
    # Need at least 3 weighted points for a 2D KDE...
    if fit.sum() < 3:
        # ...otherwise give up.
        return None
    # Try to fit the KDE:
    try:
        # Weighted Gaussian KDE at the bin centres, weights = mass, bandwidth = Scott's rule on the weighted covariance.
        kde = gaussian_kde(pts[:, fit], bw_method=KDE_BW_METHOD, weights=mass[fit])
    # Singular covariance (e.g. all points on a line) or bad input...
    except (np.linalg.LinAlgError, ValueError) as e:
        # ...no KDE.
        return None

    # Evaluation points = centres of the domain bins.
    ev = pts[:, domain_flat]
    # Raw KDE density (per cm^2) at each domain bin centre.
    dens_raw = kde(ev)
    # Default: no edge correction.
    dens = dens_raw
    # With edge correction:
    if KDE_EDGE_CORRECTION:
        # Share of the KDE kernel (covariance = KDE covariance) that falls inside the domain, per evaluation point.
        kmass = _kernel_mass_in_domain(ev, ev, area[domain_flat], kde.covariance)
        # Corrected density = raw density / kernel mass (NaN where the mass is 0).
        dens = np.where(kmass > 0, dens_raw / np.where(kmass > 0, kmass, 1.0), np.nan)

    # Full-size raw density map, NaN outside the domain.
    density_raw = np.full(handler.n_bins, np.nan)
    # Full-size corrected density map, NaN outside the domain.
    density     = np.full(handler.n_bins, np.nan)
    # Fill raw density in the domain.
    density_raw[domain_flat] = dens_raw
    # Fill corrected density in the domain.
    density[domain_flat]     = dens

    # Total mass of the map.
    total = float(mass.sum())
    # Density converted back to map units: density * total mass (* bin area for mass maps -> % per bin).
    scaled = density * total * (area if values_are_mass else 1.0)

    # Return densities, scaled map, Scott factor, effective n, kernel covariance and domain.
    return dict(density=density, density_raw=density_raw, scaled=scaled,
                # (continuation) bandwidth factor, Kish effective sample size, covariance,
                factor=float(kde.factor), neff=float(kde.neff), cov=kde.covariance,
                # (continuation) evaluation domain.
                domain=domain_flat)


# KDE of one pooled map kind for one arena.
def arena_kde(handler, results: list, map_kind: str) -> dict | None:
    """KDE of one pooled map over the bins the cells validly sampled (None if too few data)."""
    # No cells -> no KDE.
    if not results:
        # Return None.
        return None
    # Pool the cells into the requested map; also get whether the values are masses.
    values, values_are_mass = _KDE_MAP_KINDS[map_kind][0](handler, results)
    # Domain = union of the cells' valid bins.
    domain = _union_valid(handler, results)
    # KDE if the domain is non-empty, else None.
    return kde_density_map(handler, values, domain, values_are_mass) if domain.any() else None


# ============================================================================
# STAGE 3 -- Duong (2013) local significant differences, real KDE vs null KDE
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
#   'cells' -- n1 = number of real place cells pooled, n2 = number of null cells pooled.
#              Conservative for the bin-weighted maps.

# One side (real or null) of the Duong test: density and its asymptotic variance at the common points.
def _duong_kde_side(handler, kde_out: dict, common: np.ndarray, n: float) -> dict:
    """Density and its asymptotic variance on the common evaluation points for one KDE."""
    # Corrected and raw density maps of this KDE.
    dens, raw = kde_out['density'], kde_out['density_raw']
    # Bandwidth matrix H = KDE kernel covariance.
    H   = np.asarray(kde_out['cov'], dtype=float)
    # This KDE's own domain (where its kernels were renormalised).
    own = kde_out['domain']
    # Bin-centre coordinates and bin areas.
    xy, area = handler.bin_centres_xy(), handler.bin_areas_cm2()

    # Evaluation points (common domain) and source points (own domain).
    ev, src = xy[:, common], xy[:, own]
    # Kernel mass inside the own domain with bandwidth H, at each common point.
    m_H  = _kernel_mass_in_domain(ev, src, area[own], H)
    # Kernel mass inside the own domain with bandwidth H/2 (needed for the variance of K_H^2).
    m_H2 = _kernel_mass_in_domain(ev, src, area[own], H / 2.0)

    # Corrected and raw densities at the common points.
    f, f_raw = dens[common], raw[common]
    # Multiplier the KDE applied to the raw estimate (1/m_H if edge-corrected, 1 otherwise).
    c = f / f_raw                                    # multiplier the KDE applied to the raw estimate
    # Was any edge correction applied?
    corrected = not np.allclose(c, 1.0, rtol=1e-6)
    # Sanity check: max |c * m_H - 1| should be ~0 if both corrections agree.
    edge_check = float(np.max(np.abs(c * m_H - 1.0))) if corrected else 0.0

    # Boundary-unbiased density estimate f(x) = raw / m_H.
    f_true = f_raw / m_H                             # boundary-unbiased estimate of f(x)
    # Variance of the (corrected) KDE: c^2 * R(K) / (n * |H|^1/2) * f(x) * m_{H/2}(x).
    var = c ** 2 * R_K_GAUSS_2D / (n * np.sqrt(np.linalg.det(H))) * f_true * m_H2

    # Default: no renormalisation.
    scale = 1.0
    # Optionally make the density integrate to 1 over the common domain:
    if DUONG_RENORMALISE_ON_COMMON_DOMAIN:
        # scale = 1 / sum(f * area) over common bins.
        scale = 1.0 / float(np.sum(f * area[common]))
    # Return the scaled density, scaled variance (scale^2) and diagnostics.
    return dict(f=f * scale, var=var * scale ** 2, scale=scale, H=H, n=n,
                # (continuation) correction flag and sanity-check value,
                corrected=corrected, edge_check=edge_check,
                # (continuation) share of this KDE's mass that lies outside the common domain:
                mass_outside_common=float(1.0 - np.sum(dens[common] * area[common])
                                          # (continuation) divided by the KDE's total mass over its own domain.
                                          / np.nansum(dens[own] * area[own])))


# Hochberg (1988) step-up multiple-comparison procedure.
def hochberg_reject(p: np.ndarray, alpha: float) -> tuple:
    """Hochberg (1988) step-up. Returns (reject mask, p cut-off or nan, j*)."""
    # Number of tests m.
    m = len(p)
    # Indices that sort the p-values ascending.
    order = np.argsort(p)
    # Sorted p-values p_(1) <= ... <= p_(m).
    p_sorted = p[order]
    # Hochberg condition for each rank j: p_(j) <= alpha / (m - j + 1).
    ok = p_sorted <= alpha / (m - np.arange(1, m + 1) + 1)
    # Rejection mask, initially nothing rejected.
    reject = np.zeros(m, dtype=bool)
    # If no rank satisfies the condition...
    if not ok.any():
        # ...reject nothing.
        return reject, float('nan'), 0
    # j* = largest rank satisfying the condition (1-based).
    j_star = int(np.flatnonzero(ok).max()) + 1
    # Reject the j* smallest p-values.
    reject[order[:j_star]] = True
    # Return mask, p cut-off p_(j*) and j*.
    return reject, float(p_sorted[j_star - 1]), j_star


# Label 8-connected clusters of a flat mask (size >= DUONG_MIN_CLUSTER_BINS).
def _duong_cluster_labels(handler, mask_flat: np.ndarray) -> tuple:
    """8-connected clusters of mask_flat (wrapping on the circular track) with at least
    DUONG_MIN_CLUSTER_BINS bins. (labels, n), label -1 = no cluster."""
    # Label per bin, -1 = not in a cluster.
    labels = np.full(handler.n_bins, -1, dtype=int)
    # Cluster counter.
    n = 0
    # For every connected component of the mask...
    for component in handler.connected_components(mask_flat):
        # ...that is large enough...
        if len(component) >= DUONG_MIN_CLUSTER_BINS:
            # ...give its bins the next label...
            labels[component] = n
            # ...and increment the counter.
            n += 1
    # Return labels and number of clusters.
    return labels, n


# Run the Duong local test of one real KDE against one null KDE on one arena.
def duong_compare_arena(handler, real_kde: dict, null_kde: dict, n_real: float,
                        # Null sample size.
                        n_null: float) -> dict:
    # Common domain: in both domains...
    common = (real_kde['domain'] & null_kde['domain']
              # (continuation) ...both corrected densities finite...
              & np.isfinite(real_kde['density']) & np.isfinite(null_kde['density'])
              # (continuation) ...and both raw densities positive.
              & (real_kde['density_raw'] > 0) & (null_kde['density_raw'] > 0))
    # Density and variance of the real KDE at the common points.
    real = _duong_kde_side(handler, real_kde, common, n_real)
    # Density and variance of the null KDE at the common points.
    null = _duong_kde_side(handler, null_kde, common, n_null)

    # Density difference f1 - f2.
    diff = real['f'] - null['f']
    # Signed z = (f1 - f2) / sqrt(var1 + var2).
    z = diff / np.sqrt(real['var'] + null['var'])
    # Test statistic X^2 = z^2.
    x2 = z ** 2
    # p-value = P(chi^2 with 1 df >= X^2).
    p = chi2.sf(x2, df=1)
    # Hochberg multiple-comparison correction over all common points.
    reject, p_cut, j_star = hochberg_reject(p, DUONG_ALPHA)

    # Collect results.
    out = dict(common=common, real=real, null=null, f_real=real['f'], f_null=null['f'],
               # (continuation) difference, z, X^2, p, rejection mask, cut-off and j*.
               diff=diff, z=z, x2=x2, p=p, reject=reject, p_cut=p_cut, j_star=j_star)
    # Separately for "real above null" and "real below null" rejected points:
    for tag, sel in (('above', reject & (diff > 0)), ('below', reject & (diff < 0))):
        # Full-size mask, all False.
        mask = np.zeros(handler.n_bins, dtype=bool)
        # Set the selected common-domain points to True at their bin positions.
        mask[np.flatnonzero(common)[sel]] = True
        # Cluster labels and number of clusters for this sign.
        out[f'lab_{tag}'], out[f'n_{tag}'] = _duong_cluster_labels(handler, mask)
    # Return the full test result.
    return out


# One table row per significant cluster, with size, location and effect summaries.
def duong_cluster_rows(kind: str, arena: str, handler, res: dict) -> list:
    # Bin-centre coordinates and bin areas.
    xy, area_all = handler.bin_centres_xy(), handler.bin_areas_cm2()
    # Accumulator of rows.
    rows = []
    # Flat bin indices of the common-domain points (position k in the test arrays <-> bin idx[k]).
    idx = np.flatnonzero(res['common'])
    # For each direction with its labels and cluster count:
    for direction, labels, n in (('real > null', res['lab_above'], res['n_above']),
                                 # (continuation) second direction.
                                 ('real < null', res['lab_below'], res['n_below'])):
        # For each cluster k:
        for k in range(n):
            # Positions (within the common-domain arrays) belonging to cluster k.
            sel = labels[idx] == k                      # positions within the common domain
            # Their flat bin indices.
            b = idx[sel]
            # Their bin areas.
            a = area_all[b]
            # Their distance to the nearest wall.
            dw = handler.dist_to_wall[b]
            # Build the row:
            row = dict(map_kind=kind, arena=arena, direction=direction, cluster=k + 1,
                       # number of bins and total area (cm^2),
                       n_bins=int(len(b)), area_cm2=round(float(a.sum()), 2),
                       # area-weighted centroid x,
                       centroid_x_cm=round(float(np.sum(xy[0, b] * a) / a.sum()), 2),
                       # area-weighted centroid y,
                       centroid_y_cm=round(float(np.sum(xy[1, b] * a) / a.sum()), 2),
                       # area-weighted mean distance to wall,
                       mean_dist_to_wall_cm=round(float(np.sum(dw * a) / a.sum()), 2),
                       # minimum distance to wall,
                       min_dist_to_wall_cm=round(float(dw.min()), 2),
                       # maximum distance to wall,
                       max_dist_to_wall_cm=round(float(dw.max()), 2),
                       # mean real density in the cluster,
                       mean_real_density=float(np.mean(res['f_real'][sel])),
                       # mean null density in the cluster,
                       mean_null_density=float(np.mean(res['f_null'][sel])),
                       # mean of the per-bin real/null ratio,
                       mean_ratio_real_over_null=round(float(np.mean(res['f_real'][sel] / res['f_null'][sel])), 4),
                       # excess probability mass = sum((f_real - f_null) * area),
                       excess_mass=round(float(np.sum((res['f_real'][sel] - res['f_null'][sel]) * a)), 5),
                       # largest |z| in the cluster,
                       peak_abs_z=round(float(np.max(np.abs(res['z'][sel]))), 3),
                       # smallest p in the cluster.
                       min_p=float(np.min(res['p'][sel])))
            # On the circular track also report the cluster's angular position:
            if isinstance(handler, CircularTrackHandler):
                # Polar angle of each bin centre about the ring centre.
                th = np.arctan2(xy[1, b] - handler.cy, xy[0, b] - handler.cx)
                # Area-weighted circular mean angle = atan2(sum a sin, sum a cos).
                ang = np.arctan2(np.sum(np.sin(th) * a), np.sum(np.cos(th) * a))
                # Store it in degrees, 0..360.
                row['centroid_angle_deg'] = round(float(np.degrees(ang) % 360.0), 1)
            # Add the row.
            rows.append(row)
    # Return all cluster rows.
    return rows


# Run every real-vs-null Duong test; write figures, a cluster workbook and an npz of all arrays.
def run_duong_tests(arena_handlers: dict, kdes: dict, real_results: dict, null_results: dict,
                    # Output folder.
                    out_dir: str):
    """Duong test of every real KDE against its null KDE; figures, workbook and npz.
    kdes = {'real': {kind: {arena: kde}}, 'null': {kind: {arena: kde}}}; a (kind, arena) missing
    either KDE is skipped."""
    # Per-test summary rows, per-cluster rows, arrays to save.
    test_rows, cluster_rows, saved = [], [], {}
    # For each map kind (and its title):
    for kind, (_, title, *_) in _KDE_MAP_KINDS.items():
        # Results of this kind, per arena.
        results = {}
        # For each arena:
        for arena in _ARENA_ORDER:
            # Real KDE of this kind/arena (None if missing).
            real_kde = kdes['real'].get(kind, {}).get(arena)
            # Null KDE of this kind/arena (None if missing).
            null_kde = kdes['null'].get(kind, {}).get(arena)
            # If either is missing...
            if real_kde is None or null_kde is None:
                # ...skip.
                continue
            # Geometry handler of the arena.
            handler = arena_handlers[arena]
            # Sample sizes from the KDEs' effective n...
            if DUONG_SAMPLE_SIZE_MODE == 'neff':
                # ...n1 = neff(real), n2 = neff(null).
                n_real, n_null = real_kde['neff'], null_kde['neff']
            # ...or from the number of cells:
            elif DUONG_SAMPLE_SIZE_MODE == 'cells':
                # n1 = real cells, n2 = null cells.
                n_real, n_null = float(len(real_results[arena])), float(len(null_results[arena]))
            # Any other setting is an error.
            else:
                # Raise.
                raise ValueError(f'Unknown DUONG_SAMPLE_SIZE_MODE {DUONG_SAMPLE_SIZE_MODE!r}')
            # Run the local test.
            res = duong_compare_arena(handler, real_kde, null_kde, n_real, n_null)
            # Store it.
            results[arena] = res

            # For each side:
            for side in ('real', 'null'):
                # Warn if the two edge-correction computations disagree by > 2 %.
                if res[side]['corrected'] and res[side]['edge_check'] > 0.02:
                    # Print the warning (first part).
                    print(f'  [WARN] {kind}/{arena}/{side}: edge-correction mass here differs from '
                          # (continuation) second part with the discrepancy in %.
                          f'the KDE\'s by up to {100 * res[side]["edge_check"]:.1f} %')
            # Print a summary of the test: number of points, rejections, cut-off...
            print(f'[duong {kind}] {arena}: m = {res["common"].sum()}, rejected = '
                  # (continuation) ...p cut-off and cluster counts...
                  f'{res["reject"].sum()} (p cut-off {res["p_cut"]:.3g}), clusters above = '
                  # (continuation) ...below.
                  f'{res["n_above"]}, below = {res["n_below"]}')

            # Append a summary row for this test:
            test_rows.append(dict(
                # kind, arena, alpha, sample-size mode,
                map_kind=kind, arena=arena, alpha=DUONG_ALPHA, sample_size_mode=DUONG_SAMPLE_SIZE_MODE,
                # n1 and n2,
                n1_real=round(res['real']['n'], 2), n2_null=round(res['null']['n'], 2),
                # bandwidth matrices,
                H1=np.round(res['real']['H'], 3).tolist(), H2=np.round(res['null']['H'], 3).tolist(),
                # number of tested points,
                m_points=int(res['common'].sum()),
                # real domain size,
                real_domain_bins=int(real_kde['domain'].sum()),
                # null domain size,
                null_domain_bins=int(null_kde['domain'].sum()),
                # real mass outside the common domain,
                real_mass_outside_common=round(res['real']['mass_outside_common'], 4),
                # null mass outside the common domain,
                null_mass_outside_common=round(res['null']['mass_outside_common'], 4),
                # Hochberg j* and p cut-off,
                hochberg_j_star=res['j_star'], hochberg_p_cutoff=res['p_cut'],
                # number of rejected points,
                n_rejected=int(res['reject'].sum()),
                # rejected points with real > null,
                n_bins_real_above=int(np.sum(res['reject'] & (res['diff'] > 0))),
                # rejected points with real < null,
                n_bins_real_below=int(np.sum(res['reject'] & (res['diff'] < 0))),
                # cluster counts,
                n_clusters_above=res['n_above'], n_clusters_below=res['n_below'],
                # smallest p and largest |z|,
                min_p=float(res['p'].min()), max_abs_z=round(float(np.abs(res['z']).max()), 3),
                # edge-correction sanity check (real),
                edge_check_real=round(res['real']['edge_check'], 5),
                # edge-correction sanity check (null).
                edge_check_null=round(res['null']['edge_check'], 5)))
            # Append this test's cluster rows.
            cluster_rows += duong_cluster_rows(kind, arena, handler, res)

            # Key prefix for saved arrays.
            pre = f'{kind}_{arena}_'
            # For each per-point quantity:
            for name, vals in (('f_real', res['f_real']), ('f_null', res['f_null']),
                               # (continuation) variances,
                               ('var_real', res['real']['var']), ('var_null', res['null']['var']),
                               # (continuation) z and p.
                               ('z', res['z']), ('p', res['p'])):
                # Full-size array of NaN...
                arr = np.full(handler.n_bins, np.nan)
                # ...with the common-domain values filled in...
                arr[res['common']] = vals
                # ...saved under prefix + name.
                saved[pre + name] = arr
            # Full-size rejection mask.
            rej = np.zeros(handler.n_bins, dtype=bool)
            # Fill rejected common-domain points.
            rej[res['common']] = res['reject']
            # Save the rejection mask.
            saved[pre + 'reject'] = rej
            # Save the common domain.
            saved[pre + 'common_domain'] = res['common']
            # Save the "above" cluster labels.
            saved[pre + 'cluster_above'] = res['lab_above']
            # Save the "below" cluster labels.
            saved[pre + 'cluster_below'] = res['lab_below']

        # If any arena was tested for this kind...
        if results:
            # ...plot the Duong figure for this kind.
            plot_duong_kind(title, arena_handlers, results,
                            # (continuation) output path.
                            os.path.join(out_dir, f'DuongTest_{kind}.png'))

    # Nothing tested (e.g. no null KDEs supplied)...
    if not test_rows:
        # ...report and stop.
        print('[duong] skipped: no (map kind, arena) with both a real and a null KDE')
        # Return without writing files.
        return
    # Column order of the cluster sheet.
    cluster_cols = ['map_kind', 'arena', 'direction', 'cluster', 'n_bins', 'area_cm2',
                    # (continuation) location columns,
                    'centroid_x_cm', 'centroid_y_cm', 'centroid_angle_deg',
                    # (continuation) wall-distance columns,
                    'mean_dist_to_wall_cm', 'min_dist_to_wall_cm', 'max_dist_to_wall_cm',
                    # (continuation) density columns,
                    'mean_real_density', 'mean_null_density', 'mean_ratio_real_over_null',
                    # (continuation) effect / significance columns.
                    'excess_mass', 'peak_abs_z', 'min_p']
    # Workbook path.
    xlsx = os.path.join(out_dir, 'DuongTest_Clusters.xlsx')
    # Open an Excel writer:
    with pd.ExcelWriter(xlsx) as xw:
        # Sheet 'Tests' = one row per test.
        pd.DataFrame(test_rows).to_excel(xw, sheet_name='Tests', index=False)
        # Sheet 'Clusters' = one row per cluster, columns in the fixed order.
        pd.DataFrame(cluster_rows).reindex(columns=cluster_cols).to_excel(
            # (continuation) writer, sheet name, no index.
            xw, sheet_name='Clusters', index=False)
    # Report.
    print(f'[SAVED] {xlsx}')
    # npz path.
    npz = os.path.join(out_dir, 'DuongTest_Results.npz')
    # Save all arrays compressed.
    np.savez_compressed(npz, **saved)
    # Report.
    print(f'[SAVED] {npz}')



# ============================================================================
# Plotting
# ============================================================================

# Order in which arenas are processed and plotted.
_ARENA_ORDER = ['open_field', 'circular_track', 'linear_track']
# Display names of the arenas.
_ARENA_TITLES = {'open_field': 'Open Field', 'circular_track': 'Circular Track', 'linear_track': 'Linear Track'}
# Relative row heights in the Duong figure (the linear track is a thin strip).
_DUONG_ROW_HEIGHT = {'open_field': 1.0, 'circular_track': 1.0, 'linear_track': 0.45}
# Points per bin side when drawing bin polygons/edges (lets ring bins follow the arcs).
_EDGE_PTS = 6   # points per bin side, so the circular track's bin edges follow the ring's arcs


# Fetch a named colour map as a modifiable copy.
def _get_cmap(name: str):
    # Try the modern colormap registry...
    try:
        # ...look the map up by name.
        base = matplotlib.colormaps[name]
    # Older matplotlib versions...
    except Exception:
        # ...use plt.get_cmap instead.
        base = plt.get_cmap(name)
    # Return a copy (so set_bad doesn't change the global map) if copying is supported.
    return base.copy() if hasattr(base, 'copy') else base


# 'jet' colour map with white for NaN, normalised to the finite min..max of values.
def make_cmap_norm(values) -> tuple:
    # Values as a float array.
    vals = np.asarray(values, dtype=float)
    # Keep only finite values.
    vals = vals[np.isfinite(vals)]
    # No finite values...
    if len(vals) == 0:
        # ...default range 0..1.
        vmin, vmax = 0.0, 1.0
    # Otherwise...
    else:
        # ...range = data min..max.
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        # Constant data...
        if vmax <= vmin:
            # ...widen the range by a tiny amount to avoid a zero-width scale.
            vmax = vmin + 1e-6
    # Copy of the 'jet' colour map.
    cmap = _get_cmap('jet')
    # NaN / masked bins are shown white.
    cmap.set_bad('white')
    # Return colour map and linear normalisation.
    return cmap, Normalize(vmin=vmin, vmax=vmax)


# Mark an axis as having no data.
def _no_data(ax, key, why='no sessions'):
    # Title = arena name plus the reason.
    ax.set_title(f'{_ARENA_TITLES[key]}\n({why})')
    # Hide the axis.
    ax.axis('off')


# --- Bins drawn in true arena geometry (room cm; the circular track as an actual ring) ---

# Outline polygon of each requested bin in arena cm.
def _bin_polygons(handler, flat_bins: np.ndarray) -> np.ndarray:
    """(n, 4 * _EDGE_PTS, 2) outline of every bin in flat_bins, in arena cm."""
    # Parameter 0..1 along one bin side, _EDGE_PTS points.
    s = np.linspace(0.0, 1.0, _EDGE_PTS)
    # u offsets walking the 4 sides: bottom (0->1), right (1), top (1->0), left (0).
    du = np.concatenate([s, np.ones_like(s), s[::-1], np.zeros_like(s)])
    # v offsets for the same walk: bottom (0), right (0->1), top (1), left (1->0).
    dv = np.concatenate([np.zeros_like(s), s, np.ones_like(s), s[::-1]])
    # Grid indices (i, j) of each flat bin: i = flat // ny, j = flat % ny.
    ix, iy = np.divmod(flat_bins, handler.ny)
    # Convert every outline point (i + du, j + dv) from grid units to cm.
    x, y = handler.edge_to_xy(ix[:, None] + du, iy[:, None] + dv)
    # Stack into (n_bins, n_points, 2).
    return np.stack([x, y], axis=-1)


# Line segments (cm) along every bin edge that separates different labels (boundary outlines).
def _field_boundary_segments(handler, labels: np.ndarray) -> list:
    """Every bin edge separating two different labels, or a labelled bin from an unlabelled one
    or from outside the arena (label -1 = none)."""
    # Grid dimensions.
    nx, ny = handler.nx, handler.ny
    # Labels as (nx, ny) grid.
    lab = labels.reshape(nx, ny)

    # Label at grid position (i, j), with wrap-around and -1 outside the grid.
    def at(i, j):
        # Periodic first axis...
        if handler.wrap_x:
            # ...wrap i.
            i %= nx
        # Label inside the grid, -1 outside.
        return lab[i, j] if (0 <= i < nx and 0 <= j < ny) else -1

    # Parameter 0..1 along an edge.
    s = np.linspace(0.0, 1.0, _EDGE_PTS)
    # Accumulator of segments.
    segs = []
    # Edges between bins (i, j) and (i+1, j), i.e. at u = i + 1 (from -1 to include the left border unless wrapping).
    for i in (range(nx) if handler.wrap_x else range(-1, nx)):     # edge u = i + 1
        # For every j.
        for j in range(ny):
            # Labels on both sides of the edge.
            a, b = at(i, j), at(i + 1, j)
            # If they differ and at least one side is labelled...
            if a != b and max(a, b) >= 0:
                # ...add the edge from (i+1, j) to (i+1, j+1), converted to cm.
                segs.append(np.column_stack(handler.edge_to_xy(np.full_like(s, i + 1), j + s)))
    # Edges between bins (i, j) and (i, j+1), i.e. at v = j + 1.
    for i in range(nx):                                             # edge v = j + 1
        # j from -1 so the bottom border is included.
        for j in range(-1, ny):
            # Labels on both sides.
            a, b = at(i, j), at(i, j + 1)
            # If they differ and at least one is labelled...
            if a != b and max(a, b) >= 0:
                # ...add the edge from (i, j+1) to (i+1, j+1) in cm.
                segs.append(np.column_stack(handler.edge_to_xy(i + s, np.full_like(s, j + 1))))
    # Return all segments.
    return segs



# Draw selected bins as filled polygons, coloured by values (colour map) or explicit face colours.
def _draw_bins(ax, handler, shown: np.ndarray, values=None, cmap=None, norm=None, facecolors=None):
    """Fill the bins in shown, coloured by values (cmap / norm) or by per-bin facecolors."""
    # Flat indices of the bins to draw.
    bins = np.flatnonzero(shown)
    # Colour by data values:
    if facecolors is None:
        # Polygon collection with per-bin values mapped through cmap / norm.
        pc = PolyCollection(_bin_polygons(handler, bins), array=values[bins], cmap=cmap, norm=norm,
                            # (continuation) edges same colour as face, thin.
                            edgecolors='face', linewidths=0.3)
    # Colour by explicit colours:
    else:
        # Polygon collection with fixed face colours.
        pc = PolyCollection(_bin_polygons(handler, bins), facecolors=facecolors[bins],
                            # (continuation) edges same colour as face, thin.
                            edgecolors='face', linewidths=0.3)
    # Add polygons to the axis.
    ax.add_collection(pc)
    # Draw the arena walls.
    handler.draw_outline(ax)
    # Rescale the view to the drawn content.
    ax.autoscale_view()
    # Equal aspect.
    ax.set_aspect('equal')
    # Hide axes.
    ax.axis('off')
    # Return the collection for a colour bar.
    return pc


# Horizontal colour bar under an axis.
def _hbar(fig, mappable, ax, label):
    # Add a thin horizontal colour bar with a label.
    fig.colorbar(mappable, ax=ax, orientation='horizontal', fraction=0.05, pad=0.03, label=label)


# --- Stage 1: mean map figures ---

# Figure S1H: overall mean field-index map for each arena.
def plot_fig_S1H(arena_handlers: dict, arena_results: dict, save_path: str):
    # 15 x 5 inch figure.
    fig = plt.figure(figsize=(15, 5))
    # For each arena in order (panel index i):
    for i, key in enumerate(_ARENA_ORDER):
        # Arena handler.
        handler = arena_handlers[key]
        # Pooled overall mean FI map and its validity mask.
        mean_map, valid = pool_fine_map(handler, arena_results[key])
        # Colour scale over the valid values.
        cmap, norm = make_cmap_norm(mean_map[valid])

        # Circular track drawn on a polar axis; others Cartesian.
        proj = 'polar' if key == 'circular_track' else None
        # Subplot i+1 of a 1 x 3 grid.
        ax = fig.add_subplot(1, 3, i + 1, projection=proj)
        # Draw the map.
        pcm = handler.plot_fine(ax, mean_map, valid, cmap, norm)

        # Number of cells pooled.
        n_place = len(arena_results[key])
        # Panel title.
        ax.set_title(f'{_ARENA_TITLES[key]}\n(n={n_place} place cells)')
        # Colour bar.
        fig.colorbar(pcm, ax=ax, shrink=0.7, label='Field index (a.u.)')

    # Figure title.
    fig.suptitle('Fig S1H -- Overall mean field index maps (place cells pooled across days)')
    # Tidy layout.
    fig.tight_layout()
    # Save at 200 dpi.
    fig.savefig(save_path, dpi=200)
    # Free memory.
    plt.close(fig)
    # Report.
    print(f'[SAVED] {save_path}')


# Field-only mean field-index map for each arena.
def plot_field_only_mean_maps(arena_handlers: dict, arena_results: dict, save_path: str):
    # 15 x 5 inch figure.
    fig = plt.figure(figsize=(15, 5))
    # For each arena:
    for i, key in enumerate(_ARENA_ORDER):
        # Arena handler.
        handler = arena_handlers[key]
        # Pooled field-only map and mask.
        mean_map, valid = pool_field_only_map(handler, arena_results[key])
        # Colour scale.
        cmap, norm = make_cmap_norm(mean_map[valid])

        # Polar axis for the circular track.
        proj = 'polar' if key == 'circular_track' else None
        # Subplot.
        ax = fig.add_subplot(1, 3, i + 1, projection=proj)
        # Draw.
        pcm = handler.plot_fine(ax, mean_map, valid, cmap, norm)

        # Number of cells.
        n_place = len(arena_results[key])
        # Title.
        ax.set_title(f'{_ARENA_TITLES[key]}\n(n={n_place} place cells)')
        # Colour bar.
        fig.colorbar(pcm, ax=ax, shrink=0.7, label='Field index, background = 0 (a.u.)')

    # Figure title.
    fig.suptitle('Field-only mean field-index maps')
    # Layout.
    fig.tight_layout()
    # Save.
    fig.savefig(save_path, dpi=200)
    # Close.
    plt.close(fig)
    # Report.
    print(f'[SAVED] {save_path}')


# Peak-proportion map for each arena.
def plot_peak_proportion_maps(arena_handlers: dict, arena_results: dict, save_path: str):
    # 15 x 5 inch figure.
    fig = plt.figure(figsize=(15, 5))
    # For each arena:
    for i, key in enumerate(_ARENA_ORDER):
        # Arena handler.
        handler = arena_handlers[key]
        # % of cells peaking per bin, sampled-bin mask, number of cells.
        pct_map, visited, n_cells = pool_peak_proportion_map(handler, arena_results[key])
        # Colour scale.
        cmap, norm = make_cmap_norm(pct_map[visited])

        # Polar axis for the circular track.
        proj = 'polar' if key == 'circular_track' else None
        # Subplot.
        ax = fig.add_subplot(1, 3, i + 1, projection=proj)
        # Draw.
        pcm = handler.plot_fine(ax, pct_map, visited, cmap, norm)

        # Title.
        ax.set_title(f'{_ARENA_TITLES[key]}\n(n={n_cells} place cells)')
        # Colour bar.
        fig.colorbar(pcm, ax=ax, shrink=0.7, label='% of place cells with peak in bin')

    # Figure title.
    fig.suptitle('Peak proportion maps')
    # Layout.
    fig.tight_layout()
    # Save.
    fig.savefig(save_path, dpi=200)
    # Close.
    plt.close(fig)
    # Report.
    print(f'[SAVED] {save_path}')


# --- Stage 2: KDE figures (+ npz) ---

# KDE of one map kind for all arenas: figure + npz; returns the KDEs for the Duong test.
def plot_kde_maps(arena_handlers: dict, arena_results: dict, map_kind: str, save_path: str,
                  # Text for the cell count in titles, and an optional title prefix.
                  cell_label: str = 'place cells', title_prefix: str = '') -> dict:
    """KDE of one pooled map kind for every arena: figure, npz next to it, and the KDEs
    (arena -> kde_density_map output) for the Duong test."""
    # Title and 'scaled' units of this map kind.
    _, title, units, _, _ = _KDE_MAP_KINDS[map_kind]
    # 17 x 7.5 inch figure.
    fig = plt.figure(figsize=(17, 7.5))
    # 2 x 2 grid, right column 1.7x wider.
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.7])
    # Open field spans the whole left column...
    arena_axes = {'open_field':     fig.add_subplot(gs[:, 0]),
                  # ...circular track top right...
                  'circular_track': fig.add_subplot(gs[0, 1]),
                  # ...linear track bottom right.
                  'linear_track':   fig.add_subplot(gs[1, 1])}
    # Arrays to save and KDEs to return.
    saved, kdes = {}, {}
    # For each arena:
    for key in _ARENA_ORDER:
        # Handler.
        handler = arena_handlers[key]
        # Cells of this arena.
        results = arena_results[key]
        # Axis for this arena.
        ax = arena_axes[key]

        # KDE of the pooled map.
        kde_out = arena_kde(handler, results, map_kind)
        # No KDE possible...
        if kde_out is None:
            # ...label the panel...
            _no_data(ax, key, 'no KDE: too few data')
            # ...and move on.
            continue
        # Keep the KDE for the Duong test.
        kdes[key] = kde_out

        # Corrected density and domain.
        density, domain = kde_out['density'], kde_out['domain']
        # Colour scale over the domain densities.
        cmap, norm_c = make_cmap_norm(density[domain])
        # Draw the density on the bin grid (circular track unrolled).
        pcm = handler.plot_bins_2d(ax, density, domain & np.isfinite(density), cmap, norm_c)
        # Kernel SD along x and y = sqrt of the covariance diagonal.
        sd = np.sqrt(np.diag(kde_out['cov']))
        # Title line 1: arena, n cells, grid size...
        ax.set_title(f'{_ARENA_TITLES[key]} (n={len(results)} {cell_label}, '
                     # (continuation) grid size.
                     f'{handler.nx} x {handler.ny} bins)\n'
                     # (continuation) line 2: Scott factor and effective n,
                     f'Scott factor={kde_out["factor"]:.3f}, neff={kde_out["neff"]:.0f}, '
                     # (continuation) kernel SDs; small font.
                     f'BW sd={sd[0]:.1f} x {sd[1]:.1f} cm (room x, y)', fontsize=9)
        # Colour bar (shorter for the tall open-field panel).
        fig.colorbar(pcm, ax=ax, shrink=0.8 if key == 'open_field' else 1.0,
                     # (continuation) label.
                     label='KDE density (cm$^{-2}$)')

        # Save density, raw density, scaled map and covariance...
        for k in ('density', 'density_raw', 'scaled', 'cov'):
            # ...under '<arena>_<name>'.
            saved[f'{key}_{k}'] = kde_out[k]
        # Save the Scott factor.
        saved[f'{key}_factor'] = kde_out['factor']
        # Save the effective n.
        saved[f'{key}_neff'] = kde_out['neff']
        # Save the domain.
        saved[f'{key}_domain'] = domain

    # Text describing the edge-correction setting.
    corr = 'edge-corrected' if KDE_EDGE_CORRECTION else 'no edge correction'
    # Figure title.
    fig.suptitle(f'{title_prefix}{title} -- 2D Gaussian KDE (Scott\'s rule, {corr})')
    # Layout.
    fig.tight_layout()
    # Save.
    fig.savefig(save_path, dpi=200)
    # Close.
    plt.close(fig)
    # Report.
    print(f'[SAVED] {save_path}')

    # If any arena had a KDE...
    if saved:
        # ...npz path = figure path with .npz extension...
        npz_path = os.path.splitext(save_path)[0] + '.npz'
        # ...save all arrays plus the units string...
        np.savez_compressed(npz_path, scaled_units=units, **saved)
        # ...and report.
        print(f'[SAVED] {npz_path}')
    # Return the KDEs (arena -> KDE dict).
    return kdes


# --- Stage 3: Duong test figures ---

# Outline segments of a boolean mask (boundary between mask and non-mask bins).
def _duong_outline_segments(handler, mask_flat: np.ndarray) -> list:
    """Every bin side separating a bin of mask_flat from a bin outside it, in room cm."""
    # Mask -> labels 0 / -1, then reuse the boundary-segment routine.
    return _field_boundary_segments(handler, np.where(mask_flat, 0, -1))


# Draw green (above) and magenta (below) outlines of significant clusters.
def _draw_duong_clusters(ax, handler, res: dict):
    # Path effect: thicker black stroke underneath, then the normal line (halo).
    halo = [pe.Stroke(linewidth=OUTLINE_LW + 1.8, foreground='black'), pe.Normal()]
    # For each direction and its colour:
    for tag, colour in (('above', GREEN_ABOVE), ('below', MAGENTA_BELOW)):
        # Bins belonging to any cluster of this direction.
        mask = res[f'lab_{tag}'] >= 0
        # If there are any...
        if mask.any():
            # ...build a line collection of their outline segments...
            lc = LineCollection(_duong_outline_segments(handler, mask), colors=colour,
                                # (continuation) line width, rounded caps/joins, drawn on top.
                                linewidths=OUTLINE_LW, capstyle='round', joinstyle='round', zorder=6)
            # ...apply the halo...
            lc.set_path_effects(halo)
            # ...and add it to the axis.
            ax.add_collection(lc)


# Duong figure for one map kind: rows = arenas; columns = real, null, difference, z, clusters.
def plot_duong_kind(title: str, arena_handlers: dict, results: dict, save_path: str):
    """Rows = arenas; columns = real KDE, null KDE, difference, signed z, significant clusters.
    Fluorescent green (outline, and fill in the last column) = real significantly ABOVE the null;
    magenta = significantly BELOW."""
    # Arenas that were tested, in standard order.
    arenas = [a for a in _ARENA_ORDER if a in results]
    # Their row heights.
    ratios = [_DUONG_ROW_HEIGHT[a] for a in arenas]
    # Figure height scales with the total row height.
    fig = plt.figure(figsize=(28, 1.2 + 5.6 * sum(ratios)))
    # Grid: one row per arena, 5 columns.
    gs = fig.add_gridspec(len(arenas), 5, height_ratios=ratios)
    # Grey scale for densities.
    dens_cmap = plt.get_cmap('Greys')
    # Diverging red-blue for difference and z.
    div_cmap = plt.get_cmap('RdBu_r')

    # For each arena row r:
    for r, arena in enumerate(arenas):
        # Handler and test result.
        handler, res = arena_handlers[arena], results[arena]
        # Common domain mask.
        common = res['common']
        # Real and null densities together (shared colour scale).
        both = np.concatenate([res['f_real'], res['f_null']])
        # Shared density normalisation min..max.
        dens_norm = Normalize(vmin=float(both.min()), vmax=float(both.max()))
        # Symmetric range for the difference (tiny fallback if all zero).
        dmax = float(np.max(np.abs(res['diff']))) or 1e-12
        # Symmetric range for z (fallback 1).
        zmax = float(np.max(np.abs(res['z']))) or 1.0
        # |z| threshold equivalent to the Hochberg p cut-off (two-sided), NaN if nothing rejected.
        z_star = float(normal_dist.isf(res['p_cut'] / 2.0)) if np.isfinite(res['p_cut']) else float('nan')

        # Definitions of the first four panels: (values, cmap, norm, colour-bar label, title).
        panels = [
            # Panel 1: real KDE density.
            (res['f_real'], dens_cmap, dens_norm, 'density (cm$^{-2}$)',
             # (continuation) title: arena name...
             f"{_ARENA_TITLES[arena]} -- REAL KDE\n"
             # (continuation) ...n1 and kernel SD x...
             f"n1 = {res['real']['n']:.1f}, H1 sd = {np.sqrt(res['real']['H'][0, 0]):.1f} x "
             # (continuation) ...kernel SD y.
             f"{np.sqrt(res['real']['H'][1, 1]):.1f} cm"),
            # Panel 2: null KDE density.
            (res['f_null'], dens_cmap, dens_norm, 'density (cm$^{-2}$)',
             # (continuation) title...
             f"NULL KDE\n"
             # (continuation) ...n2 and kernel SD x...
             f"n2 = {res['null']['n']:.1f}, H2 sd = {np.sqrt(res['null']['H'][0, 0]):.1f} x "
             # (continuation) ...kernel SD y.
             f"{np.sqrt(res['null']['H'][1, 1]):.1f} cm"),
            # Panel 3: difference real - null on a diverging scale centred at 0.
            (res['diff'], div_cmap, TwoSlopeNorm(0.0, -dmax, dmax), 'real - null (cm$^{-2}$)',
             # (continuation) title.
             'Difference  f$_{real}$ - f$_{null}$'),
            # Panel 4: signed z on a diverging scale centred at 0.
            (res['z'], div_cmap, TwoSlopeNorm(0.0, -zmax, zmax), 'signed z = (f$_1$ - f$_2$) / $\\sigma_U$',
             # (continuation) title: alpha...
             f"Duong local test, Hochberg alpha = {DUONG_ALPHA:g}\n"
             # (continuation) ...number of points and rejections...
             f"m = {common.sum()} points, {res['reject'].sum()} rejected "
             # (continuation) ...|z| threshold (or none)...
             + (f"(|z| >= {z_star:.2f})" if np.isfinite(z_star) else '(none)')
             # (continuation) ...cluster counts.
             + f"; clusters: {res['n_above']} above, {res['n_below']} below"),
        # End of the panel list.
        ]
        # Draw each of the four panels in column c:
        for c, (vals, cmap, nrm, cbar_label, ttl) in enumerate(panels):
            # Axis at row r, column c.
            ax = fig.add_subplot(gs[r, c])
            # Full-size NaN array...
            full = np.full(handler.n_bins, np.nan)
            # ...with the common-domain values filled.
            full[common] = vals
            # Draw common-domain bins coloured by value.
            pc = _draw_bins(ax, handler, common, full, cmap, nrm)
            # Overlay significant-cluster outlines.
            _draw_duong_clusters(ax, handler, res)
            # Panel title.
            ax.set_title(ttl, fontsize=9)
            # Horizontal colour bar.
            _hbar(fig, pc, ax, cbar_label)

        # Fifth column: cluster map.
        ax = fig.add_subplot(gs[r, 4])
        # Light grey for all bins by default.
        colours = np.tile(to_rgba('0.88'), (handler.n_bins, 1))
        # Green for "real above null" cluster bins.
        colours[res['lab_above'] >= 0] = to_rgba(GREEN_ABOVE)
        # Magenta for "real below null" cluster bins.
        colours[res['lab_below'] >= 0] = to_rgba(MAGENTA_BELOW)
        # Draw common-domain bins with these colours.
        _draw_bins(ax, handler, common, facecolors=colours)
        # Title with cluster counts.
        ax.set_title(f"Significant clusters\n{res['n_above']} real > null (green), "
                     # (continuation) below count.
                     f"{res['n_below']} real < null (magenta)", fontsize=9)

    # Halo effect for the legend lines.
    halo = [pe.Stroke(linewidth=OUTLINE_LW + 1.8, foreground='black'), pe.Normal()]
    # Legend proxy line for "above"...
    handles = [Line2D([], [], color=GREEN_ABOVE, lw=OUTLINE_LW, path_effects=halo,
                      # (continuation) label.
                      label='Real significantly ABOVE null'),
               # ...and for "below".
               Line2D([], [], color=MAGENTA_BELOW, lw=OUTLINE_LW, path_effects=halo,
                      # (continuation) label.
                      label='Real significantly BELOW null')]
    # Figure legend top right.
    fig.legend(handles=handles, loc='upper right', ncol=2, frameon=False, fontsize=10)
    # Text describing the renormalisation setting.
    renorm = 'renormalised on the common domain' if DUONG_RENORMALISE_ON_COMMON_DOMAIN else 'as estimated'
    # Figure title (line 1)...
    fig.suptitle(f'{title}: real vs null KDE -- Duong (2013) local significant differences\n'
                 # (continuation) ...line 2 with settings...
                 f'(densities {renorm}; n = {DUONG_SAMPLE_SIZE_MODE}; blank = outside the common domain)',
                 # (continuation) ...left-aligned.
                 x=0.01, ha='left', fontsize=12)
    # Layout, leaving room at the top for the title.
    fig.tight_layout(rect=(0, 0, 1, 0.97 - 0.15 / (1.2 + 5.6 * sum(ratios))))
    # Save.
    fig.savefig(save_path, dpi=200)
    # Close.
    plt.close(fig)
    # Report.
    print(f'[SAVED] {save_path}')


# ============================================================================
# Export
# ============================================================================

# Write one row per pooled cell (identifiers + diagnostics) to an Excel file.
def export_excel(arena_handlers: dict, arena_results: dict, out_path: str):
    # Accumulator of rows.
    rows = []
    # For each arena and its cells:
    for key, results in arena_results.items():
        # For each cell:
        for r in results:
            # Append a row:
            rows.append(dict(
                # arena, session, unit,
                arena=key, session=r['session'], unit=r['unit'],
                # spike counts,
                n_spikes=r['n_spikes'], n_spikes_speed_excluded=r.get('n_spikes_speed_excluded'),
                # peak and mean rate,
                peak_fr=r['peak_fr'], mean_fr=r['mean_fr'],
                # spatial information and sparsity,
                sir=r['sir'], sparsity=r['sparsity'],
                # bootstrap significance.
                bootstrap_sig=r['bootstrap_sig'],
            # End of row.
            ))
    # Rows -> DataFrame.
    df = pd.DataFrame(rows)
    # Write to Excel without the index.
    df.to_excel(out_path, index=False)
    # Report.
    print(f'[SAVED] {out_path}')


# ============================================================================
# __main__ Pipeline
# ============================================================================

# Whole analysis: discover sessions, build per-cell maps, pool, plot, KDE, Duong test, export.
def run_full_pipeline(out_dir: str) -> None:
    # Create the output folder if needed.
    os.makedirs(out_dir, exist_ok=True)

    # One geometry handler per arena.
    arena_handlers = {key: make_handler(ARENA_CONFIGS[key]) for key in _ARENA_ORDER}
    # Session list per arena.
    arena_sessions = {key: find_arena_sessions(key, arena_handlers[key]) for key in _ARENA_ORDER}
    # For each arena...
    for key in _ARENA_ORDER:
        # ...print the number of sessions...
        print(f'[{key}] {len(arena_sessions[key])} sessions, '
              # (continuation) ...and the total number of units.
              f'{sum(len(s[3]) for s in arena_sessions[key])} units')

    # 1. Mean rate maps: overall, field-only, peak proportion
    # Per-arena list of per-cell records.
    arena_results = {}
    # For each arena:
    for key in _ARENA_ORDER:
        # Process all its units (dropped units are counted and listed).
        arena_results[key] = collect_arena_results(arena_handlers[key], arena_sessions[key], key)
        # Report the number of cells pooled.
        print(f'[{key}] {len(arena_results[key])} place cells pooled')
    # Figure: overall mean FI maps.
    plot_fig_S1H(arena_handlers, arena_results, os.path.join(out_dir, 'WholeRM_MeanFieldIndex.png'))
    # Figure: field-only mean FI maps.
    plot_field_only_mean_maps(arena_handlers, arena_results, os.path.join(out_dir, 'FieldOnly_MeanFieldIndex.png'))
    # Figure: peak-proportion maps.
    plot_peak_proportion_maps(arena_handlers, arena_results, os.path.join(out_dir, 'PeakProportion_Map.png'))

    # 2. KDE of every pooled map
    # Container for real and null KDEs, per kind, per arena.
    kdes = {'real': {}, 'null': {}}
    # For each map kind (and its file stem):
    for kind, (*_, stem) in _KDE_MAP_KINDS.items():
        # Compute, plot and store the real KDEs of this kind.
        kdes['real'][kind] = plot_kde_maps(arena_handlers, arena_results, kind,
                                           # (continuation) output path KDE_<stem>.png.
                                           os.path.join(out_dir, f'KDE_{stem}.png'))

    # 3. Duong (2013) local test: real KDE vs null KDE. No null model is built here; fill
    #    kdes['null'][kind] (arena -> KDE, e.g. from plot_kde_maps on null cells) and null_results
    #    (arena -> null cells, used by DUONG_SAMPLE_SIZE_MODE = 'cells') to run it.
    # Null cells per arena (empty: no null model in this script).
    null_results = {}
    # Run the Duong tests (skipped because kdes['null'] is empty).
    run_duong_tests(arena_handlers, kdes, arena_results, null_results, out_dir)

    # Write the per-cell summary workbook.
    export_excel(arena_handlers, arena_results, os.path.join(out_dir, 'AllArenas_Summary.xlsx'))


# Run only when executed as a script (not when imported).
if __name__ == '__main__':
    # Run the pipeline, writing into OUTPUT_DIR/AllArenas.
    run_full_pipeline(os.path.join(OUTPUT_DIR, 'AllArenas'))
    # Final message.
    print('\nDone.')
