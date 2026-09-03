# -*- coding: utf-8 -*-
"""
Combined theta-modulation + phase-precession pipeline, run sequentially per
unit (one .ntt file = one already-isolated unit):

  Step 1 (visualization only): polar plot of spike counts vs. instantaneous
    theta phase, phase estimated via the Hilbert transform of the
    bandpass-filtered LFP.
  Step 2: Theta Modulation Index (TMI) for that unit (Frank et al. 2001:
    TMI = 1 - the minimum of the smoothed, normalized theta-phase
    histogram), tested for significance via that paper's shuffling
    procedure -- each spike is assigned an independent random phase in
    [0, 360), except that consecutive spikes < 50 ms apart and in the same
    real-data phase bin are kept together and given the same shuffled
    phase, preserving burst structure -- (as well as the classic Rayleigh
    test / mean resultant length on the same phases, reported alongside
    for QC).
  Step 3: only for units found to be significantly theta-modulated in Step 2
    (TMI shuffle test), run the Pass Index phase-precession analysis
    (Climer, Newman & Hasselmo 2013, Eur J Neurosci 38:2526-2541, with
    Kempter et al. 2012, J Neurosci Methods 207:113-124 for the
    circular-linear regression) to test whether the cell's spatial firing
    phase-precesses against theta.

ROOT_FOLDER is searched recursively; every folder that directly contains at
least one .ncs and at least one .ntt file is treated as a session. Data
layout expected per session folder:
    *.ncs   Neuralynx continuous (LFP) file. The first one (natural sort of
            filename) is used as the theta reference channel.
    *.ntt   Neuralynx tetrode spike files, one file per already-isolated
            unit. Every .ntt file in the folder is processed.
    tracking .csv file, auto-detected as the first .csv in the folder (only
            needed for Step 3). Column order is fixed: col A = timestamp,
            col B = x (pixels, unused), col C = y (pixels, unused), col D =
            x (cm), col E = y (cm). Columns D/E are used directly; no
            pixel-to-cm conversion is performed. A session with no tracking
            file still runs Steps 1-2; Step 3 is skipped for it.

Assumption: tracking timestamps are on the same absolute clock as the
Neuralynx acquisition system. Set TRACKING_TIME_UNIT below to match the
units the timestamp column is actually stored in.

Output: one row per unit is appended to a single summary table written to
ROOT_FOLDER/theta_phase.xlsx, covering Step 1's polar-plot statistics
(MRL, preferred phase, Rayleigh p), Step 2's TMI and its shuffle p-value,
and Step 3's phase-precession fit (rho, p, slope, is_precessing,
is_recessing) where it ran. A significant negative-slope fit is labeled
theta-phase precessing (is_precessing); a significant positive-slope fit
of the same magnitude range is labeled theta-phase recessing
(is_recessing) rather than discarded. Per-unit plots (polar plot from
Step 1, and the 6-panel Pass Index summary from Step 3, when run) are
saved to <session_folder>/ThetaPhasePrecession_Combined/.

If ROOT_FOLDER's cell sessions are laid out as
<root>/<animal>/<arena>/DayN/<session>, with <arena> one of ARENA_LABELS
('Circle', 'Linear', 'Open'), the significantly precessing/recessing
cells (pooled across animals within each arena) are additionally compared
across arenas on r^2 (rho^2 of the circular-linear fit), |slope|
(deg/pass), and phase range (deg actually swept, |slope| times the
observed pass-index extent), via Kruskal-Wallis + pairwise Mann-Whitney U
tests. Plot saved to ROOT_FOLDER/ArenaComparison_PhasePrecession.png;
stats tables added as 'ArenaComparison_Omnibus' / 'ArenaComparison_Pairwise'
sheets in the summary workbook. The same arena comparison (Kruskal-Wallis +
pairwise Mann-Whitney U) is run on TMI for all significantly theta-modulated
cells (TMI_Significant), pooled across animals within each arena, saved to
ROOT_FOLDER/ArenaComparison_TMI.png with 'ArenaComparison_TMI_Omnibus' /
'ArenaComparison_TMI_Pairwise' sheets.

Requires: numpy, scipy, pandas, matplotlib, openpyxl (for writing .xlsx).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.optimize import minimize, minimize_scalar
from scipy.special import erf

# ============================================================================
# Configuration -- EDIT THESE
# ============================================================================

ROOT_FOLDER = Path(r"X:/NMR_group_data/Runita/Analysis/Thesis")
OUTPUT_EXCEL_NAME = 'theta_phase.xlsx'   # written to ROOT_FOLDER

TRACKING_TIME_UNIT = 'us'     # 'us', 'ms', or 's' -- units of the tracking timestamp column

# --- Step 3: Pass Index phase-precession parameters ---
METHOD = 'place'              # 'place' (recommended for place cells) or 'grid'
BINSIDE = 'auto'              # spatial bin size (position units, e.g. cm). 'auto' = 4
SMTH_WIDTH = 'auto'           # rate-map Gaussian smoothing width. 'auto' = 3 * BINSIDE
FILTER_BAND = 'auto'          # (low, high) cycles/unit-distance for the spatial filter
LFP_FILTER_BAND = (3.0, 7.0) # Hz, theta band used for both the theta-modulation test and LFP phase
SLOPE_BNDS = None             # optional (low, high) bound on precession slope (cycles/unit)
MIN_SPIKES_FOR_FIT = 50       # skip circular-linear fit if fewer spikes than this

# --- Step 1/2: theta phase-locking / modulation parameters ---
PHASE_BIN_SIZE_DEG = 12             # degrees per polar-histogram bin (360 must be divisible by this)
ALPHA = 0.05                        # Rayleigh-test / TMI-shuffle significance threshold
MIN_SPIKES_FOR_TMI = 8              # min spikes required to run the polar-plot / TMI battery
N_TMI_SHUFFLES = 1000               # number of surrogate random-phase draws for the TMI shuffle test
TMI_BURST_MAX_GAP_SEC = 0.05        # consecutive spikes closer than this (Frank et al. 2001: 50 ms) AND
TMI_BURST_PHASE_BIN_DEG = 36        # in the same real-phase bin this wide are kept together as one
                                     # burst and given the same shuffled phase (calc_tmi's own bin width)
RANDOM_SEED = 0                     # seed for the TMI shuffle test's RNG, for reproducibility

NCS_SAMPLES_PER_RECORD = 512
HEADER_BYTES = 16 * 1024
DEFAULT_ADBITVOLTS = 0.000000195

# --- Arena comparison (Circle/Linear/Open) parameters ---
# Data layout: ROOT_FOLDER/<animal>/<arena>/DayN/<session>, e.g.
# .../Fa23BD/Open/Day1/1Cntrl -- the arena is read off whichever path
# component (case-insensitively) matches one of these labels.
ARENA_LABELS = ('Circle', 'Linear', 'Open')


# ============================================================================
# Neuralynx file I/O
# ============================================================================

def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', path.name)]


def _read_header_text(path: Path) -> str:
    with open(path, 'rb') as fh:
        raw = fh.read(HEADER_BYTES)
    return raw.decode('latin-1', errors='replace')


def _header_field(header_text: str, name: str, default=None, cast=float):
    for line in header_text.splitlines():
        line = line.strip()
        if line.startswith(f'-{name}'):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return cast(parts[1])
                except ValueError:
                    return default
    return default


def load_ncs(path: Path):
    """Load a Neuralynx .ncs file. Returns (samples_uV, timestamps_s, fs_hz).

    Assumes continuously-sampled, full 512-sample records (standard for an
    uninterrupted Neuralynx CSC recording), matching the rest of this repo's
    .ncs readers.
    """
    header_text = _read_header_text(path)
    adbitvolts = _header_field(header_text, 'ADBitVolts', default=DEFAULT_ADBITVOLTS)

    ncs_dtype = np.dtype([
        ('TimeStamp', '<u8'),
        ('ChannelNumber', '<u4'),
        ('SampleFreq', '<u4'),
        ('NumValidSamples', '<u4'),
        ('Samples', '<i2', (NCS_SAMPLES_PER_RECORD,)),
    ])
    records = np.fromfile(path, dtype=ncs_dtype, offset=HEADER_BYTES)
    if len(records) == 0:
        raise ValueError(f'No records found in {path}')

    fs = float(records['SampleFreq'][0])
    if fs <= 0:
        fs = _header_field(header_text, 'SamplingFrequency', default=32000.0)

    sample_period_us = 1e6 / fs
    offsets_us = np.arange(NCS_SAMPLES_PER_RECORD) * sample_period_us
    timestamps_us = (records['TimeStamp'][:, None].astype(np.float64) + offsets_us[None, :]).ravel()
    samples_uV = (records['Samples'].astype(np.float64) * adbitvolts * 1e6).ravel()

    return samples_uV, timestamps_us / 1e6, fs


def load_ntt_spike_times(path: Path):
    """Load a Neuralynx .ntt file. Returns dict {cell_number: spike_times_s}.

    cell_number 0 (unsorted/noise in MClust convention) is always dropped;
    only sorted, non-zero clusters are returned as units.
    """
    ntt_dtype = np.dtype([
        ('timestamp', '<u8'),
        ('sc_number', '<u4'),
        ('cell_number', '<u4'),
        ('params', '<u4', (8,)),
        ('waveforms', '<i2', (32, 4)),
    ])
    records = np.memmap(path, dtype=ntt_dtype, mode='r', offset=HEADER_BYTES)
    timestamps_s = records['timestamp'].astype(np.float64) / 1e6
    cell_numbers = records['cell_number'].astype(np.int64)

    units = {}
    unique_cells = np.unique(cell_numbers)
    unique_cells = unique_cells[unique_cells != 0]
    for cell in unique_cells:
        units[int(cell)] = np.sort(timestamps_s[cell_numbers == cell])
    return units


# ============================================================================
# Tracking file I/O
# ============================================================================

def _find_tracking_file(folder: Path) -> Path:
    candidates = sorted(folder.glob('*.csv'))
    if not candidates:
        raise FileNotFoundError(f'No .csv tracking file found in {folder}')
    return candidates[0]


def load_tracking(path: Path, time_unit: str = 'us'):
    """Load tracking coordinates already in cm.

    Column order is fixed: col 0 = timestamp, col 1 = x (pixels, unused),
    col 2 = y (pixels, unused), col 3 = x (cm), col 4 = y (cm).

    Returns (pos_ts_s, pos_xy_cm) sorted by ascending timestamp, with NaN and
    duplicate-timestamp rows removed.
    """
    probe = pd.read_csv(path, header=None, nrows=1)

    has_header = False
    for val in probe.iloc[0, :5]:
        try:
            float(val)
        except (TypeError, ValueError):
            has_header = True
            break

    df = pd.read_csv(path, header=0 if has_header else None)

    data = df.iloc[:, [0, 3, 4]].to_numpy(dtype=np.float64)
    time_scale = {'us': 1e-6, 'ms': 1e-3, 's': 1.0}[time_unit]
    pos_ts = data[:, 0] * time_scale
    pos_xy = data[:, 1:3]

    valid = ~np.any(np.isnan(data), axis=1)
    pos_ts, pos_xy = pos_ts[valid], pos_xy[valid]

    order = np.argsort(pos_ts, kind='stable')
    pos_ts, pos_xy = pos_ts[order], pos_xy[order]
    keep = np.concatenate(([True], np.diff(pos_ts) > 0))
    return pos_ts[keep], pos_xy[keep]


# ============================================================================
# Rate map / field index
# ============================================================================

def _fill_nan_nearest(arr: np.ndarray) -> np.ndarray:
    mask = np.isnan(arr)
    if not mask.any():
        return arr
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


def spk_pos(pos_ts, pos_xy, spk_ts):
    """Nearest tracked position at each spike time."""
    idx = np.searchsorted(pos_ts, spk_ts)
    idx = np.clip(idx, 1, len(pos_ts) - 1)
    left, right = idx - 1, idx
    use_left = np.abs(spk_ts - pos_ts[left]) <= np.abs(pos_ts[right] - spk_ts)
    idx = np.where(use_left, left, right)
    return pos_xy[idx], idx


def rate_map(pos_ts, pos_xy, spk_xy, binside, smth_width):
    """Occupancy-normalized 2D rate map. Returns (map, occupancy, x_edges, y_edges)."""
    mins = np.floor(pos_xy.min(axis=0) / binside) * binside
    maxs = np.ceil(pos_xy.max(axis=0) / binside) * binside
    x_edges = np.arange(mins[0], maxs[0] + binside, binside)
    y_edges = np.arange(mins[1], maxs[1] + binside, binside)

    dt = float(np.mean(np.diff(pos_ts)))
    occupancy, _, _ = np.histogram2d(pos_xy[:, 0], pos_xy[:, 1], bins=[x_edges, y_edges])
    occupancy = occupancy * dt

    spk_counts, _, _ = np.histogram2d(spk_xy[:, 0], spk_xy[:, 1], bins=[x_edges, y_edges])
    with np.errstate(invalid='ignore', divide='ignore'):
        rmap = spk_counts / occupancy
    rmap[occupancy == 0] = np.nan

    rmap = _fill_nan_nearest(rmap)
    sigma = smth_width / binside / 2.0
    rmap = gaussian_filter(rmap, sigma=sigma, truncate=3.0, mode='nearest')
    rmap[occupancy == 0] = 0.0

    return rmap, occupancy, x_edges, y_edges


def field_index_map(rmap, occupancy, method):
    """Normalize the rate map into a 0-1 field index map."""
    fi_map = rmap.copy()
    fi_map[occupancy == 0] = np.nan
    valid = ~np.isnan(fi_map)

    if method == 'grid':
        vals = fi_map[valid]
        order = np.argsort(vals)
        ranked = np.empty_like(vals)
        ranked[order] = np.linspace(0, 1, len(vals))
        fi_map[valid] = ranked
    else:  # 'place'
        vmin, vmax = np.nanmin(fi_map), np.nanmax(fi_map)
        rng = vmax - vmin
        fi_map[valid] = (fi_map[valid] - vmin) / rng if rng > 0 else 0.0

    return fi_map


def field_index_per_position(pos_xy, fi_map, x_edges, y_edges):
    """Field-index value of the occupied bin at every tracked position sample."""
    xi = np.clip(np.digitize(pos_xy[:, 0], x_edges) - 1, 0, len(x_edges) - 2)
    yi = np.clip(np.digitize(pos_xy[:, 1], y_edges) - 1, 0, len(y_edges) - 2)
    field_index = fi_map[xi, yi]
    return _fill_nan_nearest(field_index)


def sample_along_arc(pos_ts, pos_xy, field_index):
    """Resample field_index at evenly spaced steps along the arc length (path
    distance) traversed, as in pass_index_parser.m sample_along_arc."""
    arc = np.concatenate(([0.0], np.cumsum(np.sqrt(np.sum(np.diff(pos_xy, axis=0) ** 2, axis=1)))))
    moving = np.concatenate(([True], np.diff(arc) > 0))
    arc_m, ts_m = arc[moving], pos_ts[moving]

    cc = np.linspace(0, arc_m.max(), len(ts_m))
    ts2 = np.interp(cc, arc_m, ts_m)
    resampled = np.interp(ts2, pos_ts, field_index)
    return cc, ts2, resampled


def _interp_nearest_extrap(x_ref, y_ref, x_query):
    idx = np.searchsorted(x_ref, x_query)
    idx = np.clip(idx, 1, len(x_ref) - 1)
    left, right = idx - 1, idx
    use_left = np.abs(x_query - x_ref[left]) <= np.abs(x_ref[right] - x_query)
    idx = np.where(use_left, left, right)
    return y_ref[idx]


def bandpass_filter(data, low, high, fs, order=3):
    """Zero-phase Butterworth bandpass, always via second-order sections.

    SOS form avoids the b/a transfer-function representation's numerical
    instability (spurious pole at z=1) for narrow or near-DC bands. Wn is
    clamped into the valid open interval (0, 1) relative to Nyquist so a
    degenerate auto-selected band (e.g. from a very sparse/diffuse field
    estimate) raises a clear error instead of a cryptic scipy exception.
    """
    nyq = fs / 2.0
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 1 - 1e-6)
    if low_n >= high_n:
        raise ValueError(
            f'invalid filter band after clamping to Nyquist: '
            f'requested ({low:.4g}, {high:.4g}), fs={fs:.4g} -> '
            f'normalized ({low_n:.4g}, {high_n:.4g})'
        )
    sos = signal.butter(order, [low_n, high_n], btype='band', output='sos')
    filtered = signal.sosfiltfilt(sos, data)
    if np.any(np.isnan(filtered)):
        filtered = signal.sosfilt(sos, data)
    return filtered


def auto_filter_band(method, rmap, occupancy, binside, n_dims=2):
    if method == 'grid':
        return (1.0 / (2 * 170.0), 1.0 / (26.7 / 8.0))
    # 'place': field diameter from area with >=10% peak rate
    peak = np.nanmax(rmap)
    volume = float(np.sum(rmap > 0.1 * peak)) * binside ** n_dims
    r = np.sqrt(volume / np.pi)  # n_dims == 2 case of pass_index_parser.m
    r = max(r, 1e-6)
    return (1.0 / (6.0 * r), 3.0 / r)


# ============================================================================
# Linear-circular regression (Kempter et al. 2012 / anglereg.m)
# ============================================================================

def anglereg(x: np.ndarray, theta: np.ndarray, bnds=None):
    """Linear-circular regression. Returns (slope_cycles_per_unit, intercept_rad)."""
    theta = np.mod(theta, 2 * np.pi)
    x = np.asarray(x, dtype=np.float64)

    X = np.column_stack([np.ones_like(x), x])

    def phi_cost(phi):
        wrapped = np.mod(theta + phi, 2 * np.pi)
        beta, *_ = np.linalg.lstsq(X, wrapped, rcond=None)
        resid = wrapped - X @ beta
        return float(np.sum(resid ** 2))

    phi_opt = minimize_scalar(phi_cost, bounds=(0, 2 * np.pi), method='bounded',
                               options={'xatol': 1e-10}).x
    wrapped = np.mod(theta + phi_opt, 2 * np.pi)
    beta, *_ = np.linalg.lstsq(X, wrapped, rcond=None)
    slope0 = beta[1]

    n = len(x)

    def neg_resultant(s):
        return -np.sqrt((np.sum(np.cos(theta - 2 * np.pi * s * x)) / n) ** 2 +
                        (np.sum(np.sin(theta - 2 * np.pi * s * x)) / n) ** 2)

    if bnds is None:
        r1 = minimize(lambda s: neg_resultant(s[0]), x0=[slope0], method='Nelder-Mead')
        alt0 = -1.0 / slope0 if slope0 != 0 else 1.0
        r2 = minimize(lambda s: neg_resultant(s[0]), x0=[alt0], method='Nelder-Mead')
        s = float(r1.x[0]) if r1.fun < r2.fun else float(r2.x[0])
    else:
        res = minimize_scalar(neg_resultant, bounds=bnds, method='bounded')
        s = float(res.x)

    b = np.arctan2(np.sum(np.sin(theta - 2 * np.pi * s * x)),
                   np.sum(np.cos(theta - 2 * np.pi * s * x)))
    return s, b


def kempter_lincirc(x, theta, s=None, b=None, slope_bnds=None):
    """Linear-circular correlation (Kempter et al. 2012). Returns (rho, p, s, b)."""
    x = np.asarray(x, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    good = ~np.isnan(x) & ~np.isnan(theta)
    x, theta = x[good], theta[good]

    if len(x) == 0:
        return np.nan, np.nan, np.nan, np.nan

    if s is None:
        s, b = anglereg(x, theta, slope_bnds)

    n = len(x)
    phi = np.mod(s * x, 2 * np.pi)
    theta_w = np.mod(theta, 2 * np.pi)
    phi_bar = np.angle(np.sum(np.exp(1j * phi)) / n)
    theta_bar = np.angle(np.sum(np.exp(1j * theta_w)) / n)

    num = np.sum(np.sin(theta_w - theta_bar) * np.sin(phi - phi_bar))
    den = np.sqrt(np.sum(np.sin(theta_w - theta_bar) ** 2) * np.sum(np.sin(phi - phi_bar) ** 2))
    rho = (np.abs(num / den) if den > 0 else 0.0) * np.sign(s)

    def lam(i, j):
        return np.sum((np.sin(phi - phi_bar) ** i) * (np.sin(theta_w - theta_bar) ** j)) / n

    l20, l02, l22 = lam(2, 0), lam(0, 2), lam(2, 2)
    z = rho * np.sqrt(n * l20 * l02 / l22) if l22 > 0 else 0.0
    p = 1 - erf(np.abs(z) / np.sqrt(2))

    return rho, p, s, b


# ============================================================================
# Step 3: Pass index phase-precession computation for one unit
# ============================================================================

def compute_pass_index(pos_ts, pos_xy, spk_ts, lfp_ts, lfp_sig, lfp_fs,
                        method='place', binside='auto', smth_width='auto',
                        filter_band='auto', lfp_filter_band=(6.0, 10.0),
                        slope_bnds=None):
    n_dims = pos_xy.shape[1]
    if binside == 'auto':
        binside = 2.0 * n_dims
    if smth_width == 'auto':
        smth_width = 3.0 * binside

    spk_xy, _ = spk_pos(pos_ts, pos_xy, spk_ts)
    rmap, occupancy, x_edges, y_edges = rate_map(pos_ts, pos_xy, spk_xy, binside, smth_width)
    fi_map = field_index_map(rmap, occupancy, method)
    field_index = field_index_per_position(pos_xy, fi_map, x_edges, y_edges)

    cc, ts2, resampled = sample_along_arc(pos_ts, pos_xy, field_index)

    if filter_band == 'auto':
        filter_band = auto_filter_band(method, rmap, occupancy, binside, n_dims)
    fs_arc = 1.0 / np.mean(np.diff(cc))
    filtered_field_index = bandpass_filter(resampled, filter_band[0], filter_band[1], fs_arc)

    pass_index_trace = np.angle(signal.hilbert(filtered_field_index)) / np.pi
    unwrapped = np.unwrap(pass_index_trace * np.pi)
    spk_unwrapped = _interp_nearest_extrap(ts2, unwrapped, spk_ts)
    spk_pass_index = (np.mod(spk_unwrapped + np.pi, 2 * np.pi) - np.pi) / np.pi

    filtered_lfp = bandpass_filter(lfp_sig, lfp_filter_band[0], lfp_filter_band[1], lfp_fs)
    lfp_phase = np.angle(signal.hilbert(filtered_lfp))
    unwrapped_lfp_phase = np.unwrap(lfp_phase)
    spk_theta_phase = np.mod(np.interp(spk_ts, lfp_ts, unwrapped_lfp_phase) + np.pi, 2 * np.pi) - np.pi

    rho, p, s, b = kempter_lincirc(spk_pass_index, spk_theta_phase, slope_bnds=slope_bnds)
    slope_deg_per_pass = np.rad2deg(2 * np.pi * s) if not np.isnan(s) else np.nan
    is_significant_fit = bool((not np.isnan(p)) and p < 0.05)
    # Negative slope: theta-phase precessing (spike phase advances to earlier
    # phase over the field pass). Positive slope with the same significance
    # and magnitude criteria: theta-phase recessing (phase moves later).
    is_precessing = bool(is_significant_fit and -1440 < slope_deg_per_pass < -22)
    is_recessing = bool(is_significant_fit and 22 < slope_deg_per_pass < 1440)

    # r^2 of the circular-linear fit (rho is Kempter et al.'s circular-linear
    # correlation coefficient, the circular analogue of a linear r).
    r_squared = rho ** 2 if not np.isnan(rho) else np.nan
    # Phase range: total degrees of phase actually swept by this cell's
    # spikes, i.e. the fit slope times the observed extent of pass index
    # (not the full -1..1 range, which the cell may not have fully covered).
    pass_index_range = float(np.ptp(spk_pass_index)) if len(spk_pass_index) > 0 else np.nan
    phase_range_deg = (abs(slope_deg_per_pass) * pass_index_range
                        if not (np.isnan(slope_deg_per_pass) or np.isnan(pass_index_range))
                        else np.nan)

    # Density map: spike density over (pass index, LFP phase), occupancy-normalized
    lfp_pass_index = _interp_nearest_extrap(pos_ts, np.interp(pos_ts, ts2, pass_index_trace), lfp_ts)
    pi_edges = np.linspace(-1, 1, 41)
    ph_edges = np.linspace(0, 2 * np.pi, 101)
    dt_lfp = float(np.mean(np.diff(lfp_ts)))
    occ_density, _, _ = np.histogram2d(lfp_pass_index, np.mod(lfp_phase, 2 * np.pi),
                                        bins=[pi_edges, ph_edges])
    occ_density *= dt_lfp
    spk_density, _, _ = np.histogram2d(spk_pass_index, np.mod(spk_theta_phase, 2 * np.pi),
                                        bins=[pi_edges, ph_edges])
    with np.errstate(invalid='ignore', divide='ignore'):
        density = spk_density / occ_density
    density[~np.isfinite(density)] = 0.0
    density = gaussian_filter(density, sigma=1.5, truncate=2.0, mode='nearest')

    return {
        'rate_map': rmap, 'occupancy': occupancy, 'x_edges': x_edges, 'y_edges': y_edges,
        'field_index_map': fi_map,
        'spk_xy': spk_xy,
        'spk_pass_index': spk_pass_index, 'spk_theta_phase': spk_theta_phase,
        'rho': rho, 'p': p, 's': s, 'b': b,
        'slope_deg_per_pass': slope_deg_per_pass,
        'r_squared': r_squared, 'phase_range_deg': phase_range_deg,
        'is_precessing': is_precessing, 'is_recessing': is_recessing,
        'density': density, 'pi_edges': pi_edges, 'ph_edges': ph_edges,
        'n_spikes': len(spk_ts),
    }


# ============================================================================
# Steps 1 & 2: theta phase-locking, Theta Modulation Index, shuffle test
# ============================================================================

def circ_r(alpha_rad):
    """Mean resultant vector length (circ_r.m, unweighted/unbinned case)."""
    return float(np.abs(np.mean(np.exp(1j * alpha_rad))))


def circ_mean(alpha_rad):
    """Circular mean (circ_mean.m)."""
    return float(np.angle(np.mean(np.exp(1j * alpha_rad))))


def circ_rtest(alpha_rad):
    """Rayleigh test for non-uniformity of circular data (circ_rtest.m)."""
    n = len(alpha_rad)
    r = circ_r(alpha_rad)
    R = n * r
    z = R ** 2 / n
    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))
    return float(pval), float(z)


def _gauss_kernel(n=11, sigma=1.0):
    """An n-point 1D Gaussian smoothing kernel (AditiPrecessionUtils.gauss)."""
    r = np.arange(-(n // 2), n // 2 + 1)
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(r.astype(float) ** 2) / (2 * sigma ** 2))


def _gauss_smooth_1d(y, n=11, sigma=1.0):
    """Edge-padded Gaussian convolution smoothing (AditiPrecessionUtils.gaussSmooth)."""
    g = _gauss_kernel(n, sigma)
    y = np.asarray(y, dtype=float)
    padded = np.pad(y, (len(g),), mode='edge')
    smoothed = np.convolve(padded, g, 'same')
    return smoothed[len(g):len(smoothed) - len(g)]


def calc_tmi(spike_phase_deg, num_cycles=5, bin_width_deg=36, return_hist=False):
    """Theta Modulation Index (AditiPrecessionUtils.calcTMI): tiles spike
    phases across num_cycles repeated 360-degree cycles, bins/smooths the
    histogram, keeps the middle two cycles, max-normalizes, and defines
    TMI = 1 - trough of the normalized histogram (near 1 = strongly
    theta-modulated firing, near 0 = phase-uniform firing).
    """
    phases = spike_phase_deg[~np.isnan(spike_phase_deg)]
    tiled = np.concatenate([phases + j * 360 for j in range(num_cycles)])
    edges = np.arange(0, num_cycles * 360, bin_width_deg)
    counts, edges = np.histogram(tiled, bins=edges)

    smcounts = _gauss_smooth_1d(counts.astype(float), n=7, sigma=0.5)
    edges2 = edges[:-1]
    mask = (edges2 >= 360) & (edges2 <= 1080)
    edges2 = edges2[mask] - 360
    counts2 = smcounts[mask]
    normcounts = counts2 / np.max(counts2)
    tmi = float(1 - np.min(normcounts))

    if return_hist:
        bincentres = edges2 + bin_width_deg / 2
        return tmi, edges2, bincentres, normcounts
    return tmi


def find_phase_peak_valley(spike_phase_deg):
    """Phase peak and valley (AditiPrecessionUtils.findPhaseValley), located
    on the same smoothed/normalized phase histogram calc_tmi uses, so the
    reported peak/valley phases are consistent with the TMI score."""
    tmi, _edges2, bincentres, normcounts = calc_tmi(spike_phase_deg, return_hist=True)
    peak_phase_deg = float(bincentres[np.argmax(normcounts)] % 360)
    valley_phase_deg = float(bincentres[np.argmin(normcounts)] % 360)
    return peak_phase_deg, valley_phase_deg, tmi


def assign_spike_phase(spk_ts, lfp_ts, lfp_phase_unwrapped):
    """Interpolated Hilbert theta phase (rad, wrapped to [0, 2*pi)) at each
    spike time. Interpolates cos/sin of the unwrapped phase separately (not
    the angle itself) so the 0/2*pi wraparound doesn't corrupt the
    interpolated value. Spikes outside the LFP's time range are clamped to
    the nearest end sample (np.interp's default extrapolation)."""
    cos_i = np.interp(spk_ts, lfp_ts, np.cos(lfp_phase_unwrapped))
    sin_i = np.interp(spk_ts, lfp_ts, np.sin(lfp_phase_unwrapped))
    return np.mod(np.arctan2(sin_i, cos_i), 2 * np.pi)


def _burst_groups(spk_ts, phase_deg, max_gap_sec=TMI_BURST_MAX_GAP_SEC,
                   phase_bin_deg=TMI_BURST_PHASE_BIN_DEG):
    """Group spikes into bursts for the shuffle test below: consecutive
    spikes (in time order) are merged into the same group if they are
    < max_gap_sec apart AND fall in the same real-data phase bin (Frank et
    al. 2001's rule for keeping burst spikes assigned to the same phase
    under shuffling). Returns a group-id array aligned to spk_ts's original
    order (arbitrary integers, not necessarily contiguous)."""
    order = np.argsort(spk_ts)
    ts_sorted = spk_ts[order]
    bin_idx_sorted = np.floor(phase_deg[order] / phase_bin_deg).astype(np.int64)

    same_burst = (np.diff(ts_sorted) < max_gap_sec) & (np.diff(bin_idx_sorted) == 0)
    group_id_sorted = np.concatenate(([0], np.cumsum(~same_burst)))

    group_id = np.empty_like(group_id_sorted)
    group_id[order] = group_id_sorted
    return group_id


def shuffle_tmi_significance(spk_ts, phase_deg, observed_tmi, rng, n_shuffles=N_TMI_SHUFFLES):
    """Null distribution for the TMI (Frank et al. 2001): each of n_shuffles
    surrogates assigns every spike an independent random phase in [0, 360),
    except that consecutive spikes forming a burst (see _burst_groups) are
    kept together and given the same shuffled phase, preserving burst
    temporal structure while destroying true theta-phase alignment.

    Returns
    -------
    pval         : fraction of shuffle TMIs >= observed_tmi (add-one
                   smoothed so p is never exactly 0).
    shuffle_tmis : ndarray of the n_shuffles shuffle TMI scores.
    """
    group_id = _burst_groups(spk_ts, phase_deg)
    n_groups = int(group_id.max()) + 1

    shuffle_tmis = np.empty(n_shuffles)
    for i in range(n_shuffles):
        group_phases = rng.uniform(0.0, 360.0, size=n_groups)
        shuffle_tmis[i] = calc_tmi(group_phases[group_id])

    pval = float((np.sum(shuffle_tmis >= observed_tmi) + 1) / (n_shuffles + 1))
    return pval, shuffle_tmis


def compute_theta_modulation(spk_ts, lfp_ts, lfp_phase_unwrapped, rng):
    """Steps 1 & 2 for one unit: Hilbert-phase polar-plot statistics (MRL,
    preferred phase, Rayleigh test) plus the Theta Modulation Index and its
    shuffle-test significance.

    Returns (metrics dict, phase_deg array of per-spike theta phase, for
    plotting the Step 1 polar histogram).
    """
    phase_rad = assign_spike_phase(spk_ts, lfp_ts, lfp_phase_unwrapped)
    phase_deg = np.degrees(phase_rad)
    n = len(phase_deg)

    metrics = dict(n_spikes_theta=n)
    if n < MIN_SPIKES_FOR_TMI:
        metrics.update(MRL=np.nan, PreferredPhase_deg=np.nan, Rayleigh_p=np.nan,
                        SignificantThetaModulation=False, PhasePeak_deg=np.nan,
                        PhaseValley_deg=np.nan, TMI=np.nan, TMI_shuffle_p=np.nan,
                        TMI_Significant=False)
        return metrics, phase_deg

    mrl = circ_r(phase_rad)
    pref_phase_deg = np.degrees(circ_mean(phase_rad)) % 360
    rayleigh_p, _ = circ_rtest(phase_rad)

    peak_phase_deg, valley_phase_deg, tmi = find_phase_peak_valley(phase_deg)
    tmi_pval, _shuffle_dist = shuffle_tmi_significance(spk_ts, phase_deg, tmi, rng)
    tmi_sig = bool(np.isfinite(tmi_pval) and tmi_pval < ALPHA)

    metrics.update(MRL=mrl, PreferredPhase_deg=pref_phase_deg, Rayleigh_p=rayleigh_p,
                    SignificantThetaModulation=bool(rayleigh_p < ALPHA),
                    PhasePeak_deg=peak_phase_deg, PhaseValley_deg=valley_phase_deg,
                    TMI=tmi, TMI_shuffle_p=tmi_pval, TMI_Significant=tmi_sig)
    return metrics, phase_deg


# ============================================================================
# Plotting
# ============================================================================

def plot_polar_theta(phase_deg, mrl, pref_phase_deg, rayleigh_p, is_sig, title, out_path: Path,
                      bin_size_deg=PHASE_BIN_SIZE_DEG):
    """Step 1 (visualization only): polar histogram of spike counts vs.
    Hilbert-transform theta phase, with the MRL/preferred-phase vector."""
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection='polar')
    edges_deg = np.arange(0, 360 + bin_size_deg, bin_size_deg)
    edges_rad = np.radians(edges_deg)
    counts, _ = np.histogram(np.radians(phase_deg), bins=edges_rad)
    width = np.radians(bin_size_deg)

    if is_sig:
        face_color, vec_color, vec_width = '#B3B3B3', 'red', 4
    else:
        face_color, vec_color, vec_width = '#D9D9D9', '#4D4D4D', 2

    ax.bar(edges_rad[:-1], counts, width=width, align='edge',
           facecolor=face_color, edgecolor='black', linewidth=0.5)
    max_count = counts.max() if counts.max() > 0 else 1
    ax.plot([0, np.radians(pref_phase_deg)], [0, mrl * max_count],
            color=vec_color, linewidth=vec_width)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    sig_str = 'SIGNIFICANT' if is_sig else 'not significant'
    ax.set_title(
        f"{title}\nn={len(phase_deg)} spikes | MRL={mrl:.3f} | pref={pref_phase_deg:.0f} deg\n"
        f"Rayleigh p={rayleigh_p:.3g} ({sig_str})", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_phase_histogram(phase_deg, is_sig, title, out_path: Path,
                          bin_size_deg=PHASE_BIN_SIZE_DEG):
    """Linear histogram of spike counts vs. Hilbert-transform theta phase
    bin (same phase data and bin width as plot_polar_theta, shown over two
    repeated 360-degree cycles for readability)."""
    edges_deg = np.arange(0, 360 + bin_size_deg, bin_size_deg)
    counts, _ = np.histogram(phase_deg, bins=edges_deg)
    centers_deg = edges_deg[:-1] + bin_size_deg / 2

    centers_dup = np.concatenate([centers_deg, centers_deg + 360])
    counts_dup = np.concatenate([counts, counts])

    face_color = '#B3B3B3' if is_sig else '#D9D9D9'

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(centers_dup, counts_dup, width=bin_size_deg, align='center',
           facecolor=face_color, edgecolor='black', linewidth=0.5)
    ax.set_xlim(0, 720)
    ax.set_xticks(np.arange(0, 720 + 1, 180))
    ax.set_xlabel('Theta phase (deg)')
    ax.set_ylabel('Spike count')
    sig_str = 'SIGNIFICANT' if is_sig else 'not significant'
    ax.set_title(f'{title}\nn={len(phase_deg)} spikes ({sig_str})', fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_unit_summary(pos_xy, results, title, out_path: Path):
    """Step 3: 6-panel Pass Index phase-precession summary figure."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(pos_xy[:, 0], pos_xy[:, 1], color='0.8', linewidth=0.5, zorder=1)
    sc = ax.scatter(results['spk_xy'][:, 0], results['spk_xy'][:, 1],
                     c=results['spk_pass_index'], cmap='hsv', s=10, vmin=-1, vmax=1, zorder=2)
    fig.colorbar(sc, ax=ax, label='Pass index')
    ax.set_title('Trajectory + spikes (color = pass index)')
    ax.set_aspect('equal')

    ax = axes[0, 1]
    im = ax.imshow(results['rate_map'].T, origin='lower', cmap='jet',
                    extent=[results['x_edges'][0], results['x_edges'][-1],
                            results['y_edges'][0], results['y_edges'][-1]])
    fig.colorbar(im, ax=ax, label='Rate (Hz)')
    ax.set_title('Rate map')
    ax.set_aspect('equal')

    ax = axes[0, 2]
    im = ax.imshow(results['field_index_map'].T, origin='lower', cmap='hot', vmin=0, vmax=1,
                    extent=[results['x_edges'][0], results['x_edges'][-1],
                            results['y_edges'][0], results['y_edges'][-1]])
    fig.colorbar(im, ax=ax, label='Field index')
    ax.set_title('Field index map')
    ax.set_aspect('equal')

    ax = axes[1, 0]
    pi_dup = np.concatenate([results['spk_pass_index'], results['spk_pass_index']])
    phase_dup = np.concatenate([np.rad2deg(np.mod(results['spk_theta_phase'], 2 * np.pi)),
                                 np.rad2deg(np.mod(results['spk_theta_phase'], 2 * np.pi)) + 360])
    ax.scatter(pi_dup, phase_dup, s=8, alpha=0.6)
    s, b = results['s'], results['b']
    if not np.isnan(s):
        xg = np.linspace(-1, 1, 500)
        phi = np.mod(2 * np.pi * s * xg + b, 2 * np.pi)
        phi[np.abs(np.diff(phi, prepend=phi[0])) > np.pi] = np.nan
        ax.plot(xg, np.rad2deg(phi), 'r', linewidth=2)
        ax.plot(xg, np.rad2deg(phi) + 360, 'r', linewidth=2)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 720)
    ax.set_xlabel('Pass index')
    ax.set_ylabel('LFP phase (deg)')
    ax.set_title(f'rho={results["rho"]:.2f}  p={results["p"]:.3g}\n'
                 f'slope={results["slope_deg_per_pass"]:.1f} deg/pass  '
                 f'precessing={results["is_precessing"]}  recessing={results["is_recessing"]}')

    ax = axes[1, 1]
    ph_centers = np.rad2deg(0.5 * (results['ph_edges'][:-1] + results['ph_edges'][1:]))
    pi_centers = 0.5 * (results['pi_edges'][:-1] + results['pi_edges'][1:])
    dens = results['density']
    dens_dup = np.concatenate([dens, dens], axis=1)
    ph_dup = np.concatenate([ph_centers, ph_centers + 360])
    im = ax.pcolormesh(pi_centers, ph_dup, dens_dup.T, cmap='jet', shading='auto')
    fig.colorbar(im, ax=ax, label='Rate (Hz)')
    ax.set_xlabel('Pass index')
    ax.set_ylabel('LFP phase (deg)')
    ax.set_title('Density map')

    axes[1, 2].axis('off')
    axes[1, 2].text(0.0, 0.9, f"n_spikes = {results['n_spikes']}", fontsize=11)
    axes[1, 2].text(0.0, 0.75, f"rho = {results['rho']:.3f}", fontsize=11)
    axes[1, 2].text(0.0, 0.6, f"p = {results['p']:.4g}", fontsize=11)
    axes[1, 2].text(0.0, 0.45, f"slope = {results['slope_deg_per_pass']:.2f} deg/pass", fontsize=11)
    axes[1, 2].text(0.0, 0.3, f"is_precessing = {results['is_precessing']}", fontsize=11)
    axes[1, 2].text(0.0, 0.15, f"is_recessing = {results['is_recessing']}", fontsize=11)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Batch main
# ============================================================================

def find_session_folders(root: Path) -> list[Path]:
    """Recursively find folders directly containing both .ncs and .ntt files."""
    ncs_parents = {p.parent for p in root.rglob('*.ncs')}
    return sorted(folder for folder in ncs_parents if any(folder.glob('*.ntt')))


def detect_arena(folder_path: Path) -> str | None:
    """Arena (Circle/Linear/Open) read off a case-insensitive match to
    ARENA_LABELS among folder_path's components, per the data layout
    ROOT_FOLDER/<animal>/<arena>/DayN/<session>. Returns None if no
    component matches (e.g. a differently organized dataset)."""
    parts_lower = {part.lower() for part in folder_path.parts}
    for label in ARENA_LABELS:
        if label.lower() in parts_lower:
            return label
    return None


# ============================================================================
# Arena comparison (Circle vs Linear vs Open) for significantly precessing/
# recessing cells, animals pooled within each arena
# ============================================================================

ARENA_COMPARISON_METRICS = {
    'r_squared': 'R² (ρ²)',
    'abs_slope_deg_per_pass': 'Slope magnitude (deg/pass)',
    'phase_range_deg': 'Phase range (deg)',
}
TMI_COMPARISON_METRICS = {
    'TMI': 'Theta Modulation Index (TMI)',
}
ARENA_COLORS = {'Circle': '#4C72B0', 'Linear': '#DD8452', 'Open': '#55A868'}


def compare_arenas(df_sig: pd.DataFrame, metrics: dict = ARENA_COMPARISON_METRICS):
    """Kruskal-Wallis omnibus test + pairwise Mann-Whitney U post-hoc tests,
    one per metric, across whichever arenas are present in df_sig['Arena'].
    Returns (omnibus_df, pairwise_df)."""
    arenas_present = [a for a in ARENA_LABELS if (df_sig['Arena'] == a).sum() > 0]

    omnibus_rows, pairwise_rows = [], []
    for col, label in metrics.items():
        groups = {a: df_sig.loc[df_sig['Arena'] == a, col].dropna().to_numpy() for a in arenas_present}
        groups = {a: v for a, v in groups.items() if len(v) > 0}

        if len(groups) >= 2:
            kw_stat, kw_p = stats.kruskal(*groups.values())
        else:
            kw_stat, kw_p = np.nan, np.nan
        omnibus_rows.append(dict(
            Metric=label, Test='Kruskal-Wallis', N_arenas=len(groups), Statistic=kw_stat, p_value=kw_p,
            N_per_arena=', '.join(f'{a}={len(v)}' for a, v in groups.items())))

        arena_names = list(groups.keys())
        for i in range(len(arena_names)):
            for j in range(i + 1, len(arena_names)):
                a1, a2 = arena_names[i], arena_names[j]
                v1, v2 = groups[a1], groups[a2]
                u_stat, p_pair = stats.mannwhitneyu(v1, v2, alternative='two-sided')
                pairwise_rows.append(dict(Metric=label, Arena1=a1, Arena2=a2, N1=len(v1), N2=len(v2),
                                           Test='Mann-Whitney U', Statistic=u_stat, p_value=p_pair))

    omnibus_df = pd.DataFrame(omnibus_rows,
                               columns=['Metric', 'Test', 'N_arenas', 'Statistic', 'p_value', 'N_per_arena'])
    pairwise_df = pd.DataFrame(pairwise_rows,
                                columns=['Metric', 'Arena1', 'Arena2', 'N1', 'N2', 'Test', 'Statistic', 'p_value'])
    return omnibus_df, pairwise_df


def plot_arena_comparison(df_sig: pd.DataFrame, pairwise_df: pd.DataFrame, out_path: Path,
                           metrics: dict = ARENA_COMPARISON_METRICS):
    """Boxplot + jittered points per arena for each metric, with pairwise
    Mann-Whitney p-value brackets, one panel per metric."""
    arenas_present = [a for a in ARENA_LABELS if (df_sig['Arena'] == a).sum() > 0]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 5.5), squeeze=False)
    axes = axes[0]

    for ax, (col, label) in zip(axes, metrics.items()):
        data = [df_sig.loc[df_sig['Arena'] == a, col].dropna().to_numpy() for a in arenas_present]

        bp = ax.boxplot(data, positions=range(len(arenas_present)), widths=0.5,
                         patch_artist=True, showfliers=False)
        for patch, arena in zip(bp['boxes'], arenas_present):
            patch.set_facecolor(ARENA_COLORS.get(arena, '#999999'))
            patch.set_alpha(0.5)

        for i, (arena, vals) in enumerate(zip(arenas_present, data)):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, color=ARENA_COLORS.get(arena, '#999999'),
                       edgecolor='black', linewidth=0.3, s=25, zorder=3)

        ax.set_xticks(range(len(arenas_present)))
        ax.set_xticklabels([f'{a}\n(n={len(v)})' for a, v in zip(arenas_present, data)])
        ax.set_ylabel(label)
        ax.set_title(label)

        finite_vals = [v for v in data if len(v) > 0]
        y_max = max((v.max() for v in finite_vals), default=1.0)
        y_min = min((v.min() for v in finite_vals), default=0.0)
        y_span = (y_max - y_min) if y_max > y_min else max(abs(y_max), 1.0)
        step = y_span * 0.12

        sub = pairwise_df[pairwise_df['Metric'] == label]
        level = 0
        for _, row in sub.iterrows():
            if row['Arena1'] not in arenas_present or row['Arena2'] not in arenas_present:
                continue
            i1, i2 = arenas_present.index(row['Arena1']), arenas_present.index(row['Arena2'])
            y = y_max + step * (level + 1)
            ax.plot([i1, i1, i2, i2], [y - step * 0.2, y, y, y - step * 0.2], color='black', linewidth=1)
            p = row['p_value']
            p_str = 'n/a' if not np.isfinite(p) else (f'p={p:.3g}' if p >= 0.001 else 'p<0.001')
            ax.text((i1 + i2) / 2, y + step * 0.05, p_str, ha='center', va='bottom', fontsize=9)
            level += 1
        if level > 0:
            ax.set_ylim(y_min - y_span * 0.05, y_max + step * (level + 1.5))

    fig.suptitle('Phase precession/recession by arena (Circle vs Linear vs Open)\n'
                  'significantly precessing/recessing cells, animals pooled within arena',
                  fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def process_session(data_folder: Path, rng) -> list[dict]:
    output_dir = data_folder / 'ThetaMod_PhasePrecession'
    output_dir.mkdir(parents=True, exist_ok=True)

    ncs_files = sorted(data_folder.glob('*.ncs'), key=_natural_key)
    theta_ncs = ncs_files[0]
    print(f'Using LFP file: {theta_ncs.name}')
    lfp_sig, lfp_ts, lfp_fs = load_ncs(theta_ncs)
    filtered_lfp = bandpass_filter(lfp_sig, LFP_FILTER_BAND[0], LFP_FILTER_BAND[1], lfp_fs)
    lfp_phase_unwrapped = np.unwrap(np.angle(signal.hilbert(filtered_lfp)))

    # Tracking is only needed for Step 3; a missing tracking file should not
    # block Steps 1-2 for this session.
    pos_ts = pos_xy = None
    t_start = t_stop = None
    try:
        tracking_path = _find_tracking_file(data_folder)
        print(f'Using tracking file: {tracking_path.name}')
        pos_ts, pos_xy = load_tracking(tracking_path, TRACKING_TIME_UNIT)
        t_start = max(pos_ts.min(), lfp_ts.min())
        t_stop = min(pos_ts.max(), lfp_ts.max())
    except FileNotFoundError as exc:
        print(f'  {exc} -- Step 3 (phase precession) will be skipped for this session.')

    session_label = '_'.join(data_folder.parts[-3:])
    ntt_files = sorted(data_folder.glob('*.ntt'), key=_natural_key)

    rows = []
    for ntt_path in ntt_files:
        print(f'Processing: {ntt_path.name}')
        units = load_ntt_spike_times(ntt_path)
        for cell_number, spk_ts in units.items():
            unit_label = f'{ntt_path.stem}_cell{cell_number}' if len(units) > 1 else ntt_path.stem
            row = dict(Session=session_label, FolderPath=str(data_folder), Unit=unit_label,
                       ntt_file=ntt_path.name, cell_number=cell_number, n_spikes_total=len(spk_ts))

            # ---- Steps 1 & 2: theta phase-locking polar plot + TMI shuffle test ----
            metrics, phase_deg = compute_theta_modulation(spk_ts, lfp_ts, lfp_phase_unwrapped, rng)
            row.update(metrics)

            if metrics['n_spikes_theta'] >= MIN_SPIKES_FOR_TMI:
                polar_path = output_dir / f'{unit_label}_PolarPlot.png'
                plot_polar_theta(phase_deg, metrics['MRL'], metrics['PreferredPhase_deg'],
                                  metrics['Rayleigh_p'], metrics['SignificantThetaModulation'],
                                  unit_label, polar_path)

                hist_path = output_dir / f'{unit_label}_PhaseHistogram.png'
                plot_phase_histogram(phase_deg, metrics['SignificantThetaModulation'],
                                      unit_label, hist_path)
                print(f'  {unit_label}: TMI={metrics["TMI"]:.3f} '
                      f'(p={metrics["TMI_shuffle_p"]:.3g}, '
                      f'{"theta-modulated" if metrics["TMI_Significant"] else "not theta-modulated"})')
            else:
                print(f'  {unit_label}: only {metrics["n_spikes_theta"]} spikes with LFP coverage '
                      f'(< MIN_SPIKES_FOR_TMI={MIN_SPIKES_FOR_TMI}), skipping polar plot / TMI test')

            # ---- Step 3: phase precession, gated on Step 2's TMI significance ----
            row['PrecessionTested'] = False
            row['PrecessionSkippedReason'] = ''
            if not metrics['TMI_Significant']:
                row['PrecessionSkippedReason'] = 'not significantly theta-modulated (TMI shuffle test)'
            elif pos_ts is None:
                row['PrecessionSkippedReason'] = 'no tracking file found for this session'
            else:
                spk_ts_overlap = spk_ts[(spk_ts >= t_start) & (spk_ts <= t_stop)]
                if len(spk_ts_overlap) < MIN_SPIKES_FOR_FIT:
                    row['PrecessionSkippedReason'] = (
                        f'only {len(spk_ts_overlap)} spikes in tracking/LFP overlap window '
                        f'(< MIN_SPIKES_FOR_FIT={MIN_SPIKES_FOR_FIT})')
                else:
                    try:
                        results = compute_pass_index(
                            pos_ts, pos_xy, spk_ts_overlap, lfp_ts, lfp_sig, lfp_fs,
                            method=METHOD, binside=BINSIDE, smth_width=SMTH_WIDTH,
                            filter_band=FILTER_BAND, lfp_filter_band=LFP_FILTER_BAND,
                            slope_bnds=SLOPE_BNDS,
                        )
                        png_path = output_dir / f'{unit_label}_PassIndex.png'
                        plot_unit_summary(pos_xy, results, unit_label, png_path)
                        row['PrecessionTested'] = True
                        row.update({
                            'PassIndex_n_spikes': results['n_spikes'], 'rho': results['rho'],
                            'precession_p': results['p'],
                            'slope_deg_per_pass': results['slope_deg_per_pass'],
                            'r_squared': results['r_squared'],
                            'phase_range_deg': results['phase_range_deg'],
                            'is_precessing': results['is_precessing'],
                            'is_recessing': results['is_recessing'],
                        })
                        print(f'  {unit_label}: PRECESSION rho={results["rho"]:.3f}  '
                              f'p={results["p"]:.3g}  slope={results["slope_deg_per_pass"]:.1f} deg/pass  '
                              f'precessing={results["is_precessing"]}  recessing={results["is_recessing"]}')
                    except Exception as exc:
                        row['PrecessionSkippedReason'] = f'ERROR ({exc})'
                        print(f'  {unit_label}: phase-precession ERROR ({exc})')

            rows.append(row)

    return rows


def build_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Population-level summary counts/percentages for the 'Summary' sheet.

    Each row's Count/Denominator/Percent together cover one pair of
    (number, percentage) items: total cells; SignificantThetaModulation;
    TMI_Significant (theta-modulated); PrecessionTested; and is_precessing
    counted two ways -- out of all theta-modulated (TMI_Significant) cells,
    and out of only the subset that was actually precession-tested (some
    theta-modulated cells are skipped, e.g. no tracking file or too few
    overlapping spikes).
    """
    def pct(n, d):
        return (100.0 * n / d) if d > 0 else np.nan

    total_cells = len(df)
    n_sig_theta = int((df['SignificantThetaModulation'] == True).sum())      # noqa: E712
    n_tmi_sig = int((df['TMI_Significant'] == True).sum())                   # noqa: E712
    n_precession_tested = int((df['PrecessionTested'] == True).sum())        # noqa: E712
    n_precessing = int((df['is_precessing'] == True).sum())                  # noqa: E712

    rows = [
        dict(Metric='SignificantThetaModulation (Rayleigh) cells',
             Count=n_sig_theta, Denominator=total_cells, DenominatorLabel='total cells',
             Percent=pct(n_sig_theta, total_cells)),
        dict(Metric='TMI_Significant (theta-modulated) cells',
             Count=n_tmi_sig, Denominator=total_cells, DenominatorLabel='total cells',
             Percent=pct(n_tmi_sig, total_cells)),
        dict(Metric='PrecessionTested cells',
             Count=n_precession_tested, Denominator=total_cells, DenominatorLabel='total cells',
             Percent=pct(n_precession_tested, total_cells)),
        dict(Metric='is_precessing cells (of theta-modulated cells)',
             Count=n_precessing, Denominator=n_tmi_sig, DenominatorLabel='theta-modulated cells',
             Percent=pct(n_precessing, n_tmi_sig)),
        dict(Metric='is_precessing cells (of precession-tested cells)',
             Count=n_precessing, Denominator=n_precession_tested,
             DenominatorLabel='precession-tested cells',
             Percent=pct(n_precessing, n_precession_tested)),
    ]
    return pd.DataFrame(rows, columns=['Metric', 'Count', 'Denominator', 'DenominatorLabel', 'Percent'])


def main():
    if 360 % PHASE_BIN_SIZE_DEG != 0:
        raise ValueError('PHASE_BIN_SIZE_DEG must divide 360 evenly.')

    rng = np.random.default_rng(RANDOM_SEED)

    session_folders = find_session_folders(ROOT_FOLDER)
    if not session_folders:
        raise FileNotFoundError(f'No folders with both .ncs and .ntt files found under {ROOT_FOLDER}')

    all_rows = []
    for data_folder in session_folders:
        print(f'\n=== Session: {data_folder} ===')
        try:
            all_rows.extend(process_session(data_folder, rng))
        except Exception as exc:
            print(f'ERROR processing {data_folder}: {exc}')
            continue

    columns = ['Session', 'FolderPath', 'Unit', 'ntt_file', 'cell_number', 'n_spikes_total',
               'n_spikes_theta', 'MRL', 'PreferredPhase_deg', 'Rayleigh_p',
               'SignificantThetaModulation', 'PhasePeak_deg', 'PhaseValley_deg', 'TMI',
               'TMI_shuffle_p', 'TMI_Significant', 'PrecessionTested', 'PassIndex_n_spikes',
               'rho', 'r_squared', 'precession_p', 'slope_deg_per_pass', 'phase_range_deg',
               'is_precessing', 'is_recessing', 'PrecessionSkippedReason']
    df = pd.DataFrame(all_rows, columns=columns)
    summary_df = build_summary_stats(df)

    # ---- Arena comparison (Circle vs Linear vs Open) ----
    df['Arena'] = df['FolderPath'].apply(lambda p: detect_arena(Path(p)))
    sig_mask = (df['is_precessing'] == True) | (df['is_recessing'] == True)  # noqa: E712
    df_sig = df.loc[sig_mask & df['Arena'].notna()].copy()
    df_sig['abs_slope_deg_per_pass'] = df_sig['slope_deg_per_pass'].abs()

    arena_omnibus_df = pd.DataFrame()
    arena_pairwise_df = pd.DataFrame()
    arenas_with_data = sorted(df_sig['Arena'].unique()) if not df_sig.empty else []
    if len(arenas_with_data) >= 2:
        arena_omnibus_df, arena_pairwise_df = compare_arenas(df_sig)
        arena_plot_path = ROOT_FOLDER / 'ArenaComparison_PhasePrecession.png'
        plot_arena_comparison(df_sig, arena_pairwise_df, arena_plot_path)
        print(f'\nArena comparison ({", ".join(arenas_with_data)}) plot saved to {arena_plot_path}')
    else:
        print(f'\nFewer than 2 arenas with significantly precessing/recessing cells found '
              f'({arenas_with_data}) -- skipping arena comparison plot.')

    # ---- Arena comparison of TMI (significantly theta-modulated cells) ----
    df_tmi_sig = df.loc[(df['TMI_Significant'] == True) & df['Arena'].notna()].copy()  # noqa: E712

    tmi_omnibus_df = pd.DataFrame()
    tmi_pairwise_df = pd.DataFrame()
    arenas_with_tmi_data = sorted(df_tmi_sig['Arena'].unique()) if not df_tmi_sig.empty else []
    if len(arenas_with_tmi_data) >= 2:
        tmi_omnibus_df, tmi_pairwise_df = compare_arenas(df_tmi_sig, TMI_COMPARISON_METRICS)
        tmi_plot_path = ROOT_FOLDER / 'ArenaComparison_TMI.png'
        plot_arena_comparison(df_tmi_sig, tmi_pairwise_df, tmi_plot_path, TMI_COMPARISON_METRICS)
        print(f'\nArena TMI comparison ({", ".join(arenas_with_tmi_data)}) plot saved to {tmi_plot_path}')
    else:
        print(f'\nFewer than 2 arenas with significantly theta-modulated cells found '
              f'({arenas_with_tmi_data}) -- skipping TMI arena comparison plot.')

    excel_path = ROOT_FOLDER / OUTPUT_EXCEL_NAME
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='ThetaPhase', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        if not arena_omnibus_df.empty:
            arena_omnibus_df.to_excel(writer, sheet_name='ArenaComparison_Omnibus', index=False)
        if not arena_pairwise_df.empty:
            arena_pairwise_df.to_excel(writer, sheet_name='ArenaComparison_Pairwise', index=False)
        if not tmi_omnibus_df.empty:
            tmi_omnibus_df.to_excel(writer, sheet_name='ArenaComparison_TMI_Omnibus', index=False)
        if not tmi_pairwise_df.empty:
            tmi_pairwise_df.to_excel(writer, sheet_name='ArenaComparison_TMI_Pairwise', index=False)
    print(f'\nDone. {len(df)} unit(s) processed. Summary saved to {excel_path}')


if __name__ == '__main__':
    main()
