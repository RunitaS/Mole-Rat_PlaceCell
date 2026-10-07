# -*- coding: utf-8 -*-
"""
Pass Index phase precession analysis, adapted from Climer, Newman & Hasselmo
(2013, Eur J Neurosci 38:2526-2541) MATLAB toolbox (jrclimer/Pass_Index,
https://github.com/jrclimer/Pass_Index -- a local copy is in
./Pass_Index-master for reference). Linear-circular regression follows
Kempter et al. (2012, J Neurosci Methods 207:113-124).

ROOT_FOLDER is searched recursively; every folder that directly contains at
least one .ncs and at least one .ntt file is treated as a session and
processed independently, with its own output written alongside it. Data
layout expected per session folder:
    *.ncs   Neuralynx continuous (LFP) files. The first one (natural sort of
            filename) is used as the theta reference channel.
    *.ntt   Neuralynx tetrode spike files, one file per already-isolated unit
            (as produced by this repo's SUA_MUA_classification.py /
            concat_ntt.py pipeline). Every .ntt file in the folder is
            processed as one unit.
    tracking .csv file, auto-detected as the first .csv in the folder.
            Column order is fixed: col A = timestamp, col B = x (pixels,
            unused), col C = y (pixels, unused), col D = x (cm), col E = y
            (cm) (column names are ignored). Columns D/E are used directly;
            no pixel-to-cm conversion is performed.

Assumption: tracking timestamps are on the same absolute clock as the
Neuralynx acquisition system (typical when tracking is exported from the
same recording system). Set TRACKING_TIME_UNIT below to match the units the
timestamp column is actually stored in.

Requires: numpy, scipy, pandas, matplotlib.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import signal
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.optimize import minimize, minimize_scalar
from scipy.special import erf

# ============================================================================
# Configuration -- EDIT THESE
# ============================================================================

ROOT_FOLDER = Path(r"C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True")

TRACKING_TIME_UNIT = 'us'     # 'us', 'ms', or 's' -- units of the tracking timestamp column

METHOD = 'place'              # 'place' (recommended for place cells) or 'grid'
BINSIDE = 'auto'              # spatial bin size (position units, e.g. cm). 'auto' = 4
SMTH_WIDTH = 'auto'           # rate-map Gaussian smoothing width. 'auto' = 3 * BINSIDE
FILTER_BAND = 'auto'          # (low, high) cycles/unit-distance for the spatial filter
LFP_FILTER_BAND = (3.0, 7.0) # Hz, theta band used for LFP phase
SLOPE_BNDS = None             # optional (low, high) bound on precession slope (cycles/unit)

MIN_SPIKES_FOR_FIT = 50       # skip circular-linear fit if fewer spikes than this
NCS_SAMPLES_PER_RECORD = 512
HEADER_BYTES = 16 * 1024
DEFAULT_ADBITVOLTS = 0.000000195


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
# Pass index computation for one unit
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
    is_precessing = bool((not np.isnan(p)) and p < 0.05 and -1440 < slope_deg_per_pass < -22)

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
        'slope_deg_per_pass': slope_deg_per_pass, 'is_precessing': is_precessing,
        'density': density, 'pi_edges': pi_edges, 'ph_edges': ph_edges,
        'n_spikes': len(spk_ts),
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_unit_summary(pos_xy, results, title, out_path: Path):
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
                 f'precessing={results["is_precessing"]}')

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


def process_session(data_folder: Path):
    output_dir = data_folder / 'PhasePrecession_PassIndex'
    output_dir.mkdir(parents=True, exist_ok=True)

    ncs_files = sorted(data_folder.glob('*.ncs'), key=_natural_key)
    theta_ncs = ncs_files[0]
    print(f'Using LFP file: {theta_ncs.name}')
    lfp_sig, lfp_ts, lfp_fs = load_ncs(theta_ncs)

    tracking_path = _find_tracking_file(data_folder)
    print(f'Using tracking file: {tracking_path.name}')
    pos_ts, pos_xy = load_tracking(tracking_path, TRACKING_TIME_UNIT)

    t_start = max(pos_ts.min(), lfp_ts.min())
    t_stop = min(pos_ts.max(), lfp_ts.max())

    ntt_files = sorted(data_folder.glob('*.ntt'), key=_natural_key)

    summary_rows = []
    for ntt_path in ntt_files:
        print(f'Processing: {ntt_path.name}')
        units = load_ntt_spike_times(ntt_path)
        for cell_number, spk_ts in units.items():
            spk_ts = spk_ts[(spk_ts >= t_start) & (spk_ts <= t_stop)]
            unit_label = f'{ntt_path.stem}_cell{cell_number}' if len(units) > 1 else ntt_path.stem

            if len(spk_ts) < MIN_SPIKES_FOR_FIT:
                print(f'  {unit_label}: only {len(spk_ts)} spikes in overlap window, skipping')
                continue

            try:
                results = compute_pass_index(
                    pos_ts, pos_xy, spk_ts, lfp_ts, lfp_sig, lfp_fs,
                    method=METHOD, binside=BINSIDE, smth_width=SMTH_WIDTH,
                    filter_band=FILTER_BAND, lfp_filter_band=LFP_FILTER_BAND,
                    slope_bnds=SLOPE_BNDS,
                )
            except Exception as exc:
                print(f'  {unit_label}: ERROR ({exc})')
                continue

            print(f'  {unit_label}: rho={results["rho"]:.3f}  p={results["p"]:.3g}  '
                  f'slope={results["slope_deg_per_pass"]:.1f} deg/pass  '
                  f'precessing={results["is_precessing"]}')

            png_path = output_dir / f'{unit_label}_PassIndex.png'
            plot_unit_summary(pos_xy, results, unit_label, png_path)

            summary_rows.append({
                'unit': unit_label, 'ntt_file': ntt_path.name, 'cell_number': cell_number,
                'n_spikes': results['n_spikes'], 'rho': results['rho'], 'p': results['p'],
                'slope_deg_per_pass': results['slope_deg_per_pass'],
                'is_precessing': results['is_precessing'],
            })

    df = pd.DataFrame(summary_rows)
    csv_path = output_dir / 'PassIndex_Summary.csv'
    df.to_csv(csv_path, index=False)
    print(f'Done. {len(df)} unit(s) processed. Summary saved to {csv_path}')


def main():
    session_folders = find_session_folders(ROOT_FOLDER)
    if not session_folders:
        raise FileNotFoundError(f'No folders with both .ncs and .ntt files found under {ROOT_FOLDER}')

    for data_folder in session_folders:
        print(f'\n=== Session: {data_folder} ===')
        try:
            process_session(data_folder)
        except Exception as exc:
            print(f'ERROR processing {data_folder}: {exc}')
            continue


if __name__ == '__main__':
    main()
