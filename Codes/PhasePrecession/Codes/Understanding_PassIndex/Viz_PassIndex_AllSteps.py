# -*- coding: utf-8 -*-
"""
Step-by-step diagnostic visualization of the Pass Index phase-precession
algorithm (Climer, Newman & Hasselmo 2013, with Kempter et al. 2012 for the
circular-linear regression), following Pass_Index_Algo.md Steps 0-10.

Unlike ThetaMod_PhasePRecession.py (batch, whole ROOT_FOLDER) or
Viz_PassIndexTraversals.py (batch, Step 6 only), this script runs on ONE
user-specified .ntt file and produces one PNG per algorithm step so every
intermediate transformation can be inspected individually:

    00_RawData_Overview.png          Step 0: raw LFP / tracking / spike raster
    01_SpikePositions.png            Step 1: spk_pos (nearest-position lookup)
    02_RateMap.png                   Step 2: occupancy -> raw rate -> smoothed rate
    03_FieldIndexMap.png             Step 3: field_index_map + per-position trace
    04_ArcLengthResampling.png       Step 4: sample_along_arc
    05_BandpassFilter.png            Step 5: spatial bandpass filter + spectrum
    06_HilbertPassIndex.png          Step 6: Hilbert phase -> pass index
    06b_PassTraversalsOverlay.png    Step 6: individual pass traversals overlaid
    07_ThetaPhase.png                Step 7: LFP theta-phase extraction
    08_CircularLinearRegression.png  Step 8: anglereg / kempter_lincirc
    09_Classification.png            Step 9: precession classification
    10_DensityMap.png                Step 10: pass-index x LFP-phase density map

All low-level computation (file I/O, rate map, field index, arc resampling,
bandpass filtering, circular-linear regression) is imported unchanged from
ThetaMod_PhasePRecession.py so the numbers here match the production
pipeline exactly; only the intermediate arrays that module normally discards
are kept here for plotting (same approach as Viz_PassIndexTraversals.py).

Requires: numpy, scipy, matplotlib.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter

from Understanding_PassIndex.ThetaMod_PhasePRecession import (
    TRACKING_TIME_UNIT, METHOD, BINSIDE, SMTH_WIDTH, FILTER_BAND,
    LFP_FILTER_BAND, SLOPE_BNDS, MIN_SPIKES_FOR_FIT,
    _natural_key, _find_tracking_file, _fill_nan_nearest, _interp_nearest_extrap,
    load_ncs, load_ntt_spike_times, load_tracking,
    spk_pos, field_index_map, field_index_per_position, sample_along_arc,
    bandpass_filter, auto_filter_band, anglereg, kempter_lincirc,
)

# ============================================================================
# Configuration -- EDIT THESE
# ============================================================================

# The ONE .ntt file to visualize (not a whole folder). Its parent folder must
# also contain the session's .ncs (LFP) file and a tracking .csv.
NTT_FILE = Path(r"C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True/Fa1059/Day8/2_180/TT4_SS_17.ntt")

# Which sorted cluster in that .ntt file to use. None = auto-pick the
# sorted, non-zero cluster with the most spikes (cluster 0/noise is always
# excluded by load_ntt_spike_times).
CELL_NUMBER = None

# Duration (s) of the zoomed raw-trace window shown in Steps 0 and 7.
ZOOM_WINDOW_SEC = 2.0

# Range of candidate slopes (cycles per pass-index unit) swept when plotting
# the Step 8 resultant-length diagnostic curve.
SLOPE_SWEEP_RANGE = (-4.0, 4.0)


# ============================================================================
# Step-by-step computation (mirrors compute_pass_index, keeping intermediates)
# ============================================================================

def rate_map_steps(pos_ts, pos_xy, spk_xy, binside, smth_width):
    """Step 2, with every intermediate kept: occupancy, raw spike counts,
    raw (holey) rate, nearest-neighbor-filled rate, and the final smoothed
    rate map -- same math as rate_map() in ThetaMod_PhasePRecession.py."""
    mins = np.floor(pos_xy.min(axis=0) / binside) * binside
    maxs = np.ceil(pos_xy.max(axis=0) / binside) * binside
    x_edges = np.arange(mins[0], maxs[0] + binside, binside)
    y_edges = np.arange(mins[1], maxs[1] + binside, binside)

    dt = float(np.mean(np.diff(pos_ts)))
    occupancy, _, _ = np.histogram2d(pos_xy[:, 0], pos_xy[:, 1], bins=[x_edges, y_edges])
    occupancy = occupancy * dt

    spk_counts, _, _ = np.histogram2d(spk_xy[:, 0], spk_xy[:, 1], bins=[x_edges, y_edges])
    with np.errstate(invalid='ignore', divide='ignore'):
        raw_rate = spk_counts / occupancy
    raw_rate_nan = raw_rate.copy()
    raw_rate_nan[occupancy == 0] = np.nan

    filled_rate = _fill_nan_nearest(raw_rate_nan)
    sigma = smth_width / binside / 2.0
    smoothed = gaussian_filter(filled_rate, sigma=sigma, truncate=3.0, mode='nearest')
    final_map = smoothed.copy()
    final_map[occupancy == 0] = 0.0

    return dict(x_edges=x_edges, y_edges=y_edges, occupancy=occupancy,
                spk_counts=spk_counts, raw_rate_nan=raw_rate_nan,
                filled_rate=filled_rate, smoothed=smoothed, rate_map=final_map)


def phi_cost_curve(x, theta, n=200):
    """Step 8 diagnostic: anglereg's initial phase-offset search, phi_cost(phi)
    over the full [0, 2*pi) range (same objective anglereg minimizes)."""
    X = np.column_stack([np.ones_like(x), x])
    phis = np.linspace(0, 2 * np.pi, n)
    costs = np.empty(n)
    for i, phi in enumerate(phis):
        wrapped = np.mod(theta + phi, 2 * np.pi)
        beta, *_ = np.linalg.lstsq(X, wrapped, rcond=None)
        resid = wrapped - X @ beta
        costs[i] = np.sum(resid ** 2)
    return phis, costs


def resultant_length_curve(x, theta, s_range, n=400):
    """Step 8 diagnostic: mean resultant vector length of theta - 2*pi*s*x as
    a function of candidate slope s (the objective anglereg actually
    maximizes once the rough initial slope has been found)."""
    n_pts = len(x)
    s_vals = np.linspace(s_range[0], s_range[1], n)
    resultants = np.empty(n)
    for i, s in enumerate(s_vals):
        c = np.sum(np.cos(theta - 2 * np.pi * s * x)) / n_pts
        si = np.sum(np.sin(theta - 2 * np.pi * s * x)) / n_pts
        resultants[i] = np.sqrt(c ** 2 + si ** 2)
    return s_vals, resultants


def segment_passes(unwrapped, min_samples=5):
    """One pass = one contiguous run between successive floor(2*pi) boundary
    crossings of the unwrapped Hilbert phase (same definition
    Viz_PassIndexTraversals.py uses for a single field traversal)."""
    cycle = np.floor((unwrapped + np.pi) / (2 * np.pi)).astype(np.int64)
    boundaries = np.where(np.diff(cycle) != 0)[0] + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(unwrapped)]))
    return [(s, e) for s, e in zip(starts, stops) if (e - s) >= min_samples]


def compute_all_steps(pos_ts, pos_xy, spk_ts, lfp_ts, lfp_sig, lfp_fs,
                       method=METHOD, binside=BINSIDE, smth_width=SMTH_WIDTH,
                       filter_band=FILTER_BAND, lfp_filter_band=LFP_FILTER_BAND,
                       slope_bnds=SLOPE_BNDS):
    """Runs the full pipeline once, keeping every intermediate array needed
    to plot Steps 1-10."""
    n_dims = pos_xy.shape[1]
    if binside == 'auto':
        binside = 2.0 * n_dims
    if smth_width == 'auto':
        smth_width = 3.0 * binside

    out = dict(binside=binside, smth_width=smth_width)

    # --- Step 1 ---
    spk_xy, spk_pos_idx = spk_pos(pos_ts, pos_xy, spk_ts)
    out.update(spk_xy=spk_xy, spk_pos_idx=spk_pos_idx)

    # --- Step 2 ---
    rm = rate_map_steps(pos_ts, pos_xy, spk_xy, binside, smth_width)
    out.update(rm)

    # --- Step 3 ---
    fi_map = field_index_map(rm['rate_map'], rm['occupancy'], method)
    field_index = field_index_per_position(pos_xy, fi_map, rm['x_edges'], rm['y_edges'])
    out.update(field_index_map=fi_map, field_index=field_index)

    # --- Step 4 ---
    arc = np.concatenate(([0.0], np.cumsum(np.sqrt(np.sum(np.diff(pos_xy, axis=0) ** 2, axis=1)))))
    cc, ts2, resampled = sample_along_arc(pos_ts, pos_xy, field_index)
    out.update(arc=arc, cc=cc, ts2=ts2, resampled=resampled)

    # --- Step 5 ---
    if filter_band == 'auto':
        filter_band = auto_filter_band(method, rm['rate_map'], rm['occupancy'], binside, n_dims)
    fs_arc = 1.0 / np.mean(np.diff(cc))
    filtered_field_index = bandpass_filter(resampled, filter_band[0], filter_band[1], fs_arc)
    out.update(filter_band=filter_band, fs_arc=fs_arc, filtered_field_index=filtered_field_index)

    # --- Step 6 ---
    analytic = signal.hilbert(filtered_field_index)
    pass_index_trace = np.angle(analytic) / np.pi
    envelope = np.abs(analytic)
    unwrapped = np.unwrap(pass_index_trace * np.pi)
    spk_unwrapped = _interp_nearest_extrap(ts2, unwrapped, spk_ts)
    spk_pass_index = (np.mod(spk_unwrapped + np.pi, 2 * np.pi) - np.pi) / np.pi
    passes = segment_passes(unwrapped)
    out.update(pass_index_trace=pass_index_trace, envelope=envelope, unwrapped=unwrapped,
               spk_pass_index=spk_pass_index, passes=passes)

    # --- Step 7 ---
    filtered_lfp = bandpass_filter(lfp_sig, lfp_filter_band[0], lfp_filter_band[1], lfp_fs)
    lfp_phase = np.angle(signal.hilbert(filtered_lfp))
    unwrapped_lfp_phase = np.unwrap(lfp_phase)
    spk_theta_phase = np.mod(np.interp(spk_ts, lfp_ts, unwrapped_lfp_phase) + np.pi, 2 * np.pi) - np.pi
    out.update(filtered_lfp=filtered_lfp, lfp_phase=lfp_phase, spk_theta_phase=spk_theta_phase)

    # --- Step 8 ---
    s, b = anglereg(spk_pass_index, spk_theta_phase, slope_bnds)
    rho, p, s, b = kempter_lincirc(spk_pass_index, spk_theta_phase, s=s, b=b)
    out.update(s=s, b=b, rho=rho, p=p)

    # --- Step 9 ---
    slope_deg_per_pass = np.rad2deg(2 * np.pi * s) if not np.isnan(s) else np.nan
    is_significant = bool((not np.isnan(p)) and p < 0.05)
    is_precessing = bool(is_significant and -1440 < slope_deg_per_pass < -22)
    out.update(slope_deg_per_pass=slope_deg_per_pass, is_significant=is_significant,
               is_precessing=is_precessing)

    # --- Step 10 ---
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
    out.update(density=density, pi_edges=pi_edges, ph_edges=ph_edges)

    return out


# ============================================================================
# Plotting -- one function per step
# ============================================================================

def _zoom_window(t_center, lfp_ts, half_width):
    lo = max(lfp_ts.min(), t_center - half_width)
    hi = min(lfp_ts.max(), t_center + half_width)
    mask = (lfp_ts >= lo) & (lfp_ts <= hi)
    return lo, hi, mask


def plot_step00_raw_overview(pos_ts, pos_xy, spk_ts, lfp_ts, lfp_sig, unit_label, out_path):
    t_center = spk_ts[0] if len(spk_ts) else 0.5 * (lfp_ts[0] + lfp_ts[-1])
    lo, hi, mask = _zoom_window(t_center, lfp_ts, ZOOM_WINDOW_SEC / 2.0)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    fig.suptitle(f'{unit_label}\nStep 0: raw data', fontsize=12, fontweight='bold')

    ax = axes[0]
    sc = ax.scatter(pos_xy[:, 0], pos_xy[:, 1], c=pos_ts, cmap='viridis', s=2)
    fig.colorbar(sc, ax=ax, label='Time (s)')
    ax.set_title('Tracked trajectory (color = time)')
    ax.set_xlabel('x (pos units)')
    ax.set_ylabel('y (pos units)')
    ax.set_aspect('equal')

    ax = axes[1]
    ax.plot(lfp_ts[mask], lfp_sig[mask], color='0.2', linewidth=0.8)
    ax.set_title(f'Raw LFP (theta reference channel), zoomed window [{lo:.2f}, {hi:.2f}] s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('LFP (uV)')

    ax = axes[2]
    ax.eventplot([spk_ts], colors='k', linewidths=0.6)
    ax.axvspan(lo, hi, color='red', alpha=0.15, label='zoomed window above')
    ax.set_yticks([])
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Spike raster, n={len(spk_ts)} spikes')
    ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_step01_spike_positions(pos_ts, pos_xy, spk_ts, spk_xy, unit_label, out_path):
    idx = np.searchsorted(pos_ts, spk_ts)
    idx = np.clip(idx, 1, len(pos_ts) - 1)
    left, right = idx - 1, idx
    dt_err = np.minimum(np.abs(spk_ts - pos_ts[left]), np.abs(pos_ts[right] - spk_ts))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{unit_label}\nStep 1: spk_pos (nearest-in-time position lookup)',
                 fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(pos_xy[:, 0], pos_xy[:, 1], color='0.8', linewidth=0.5, zorder=1)
    ax.scatter(spk_xy[:, 0], spk_xy[:, 1], color='red', s=10, zorder=2)
    ax.set_title(f'Spike positions (n={len(spk_ts)}) on trajectory')
    ax.set_aspect('equal')

    ax = axes[1]
    ax.hist(dt_err * 1000, bins=50, color='steelblue', edgecolor='k', linewidth=0.3)
    ax.set_xlabel('|spike time - matched tracking sample time| (ms)')
    ax.set_ylabel('Count')
    ax.set_title('Time error of nearest-position match\n(should be << behavior camera frame period)')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_step02_rate_map(rm, unit_label, out_path):
    extent = [rm['x_edges'][0], rm['x_edges'][-1], rm['y_edges'][0], rm['y_edges'][-1]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig.suptitle(f'{unit_label}\nStep 2: occupancy-normalized rate map', fontsize=12, fontweight='bold')

    ax = axes[0, 0]
    im = ax.imshow(rm['occupancy'].T, origin='lower', cmap='viridis', extent=extent)
    fig.colorbar(im, ax=ax, label='Occupancy (s)')
    ax.set_title('Occupancy (time spent per bin)')
    ax.set_aspect('equal')

    ax = axes[0, 1]
    im = ax.imshow(rm['spk_counts'].T, origin='lower', cmap='viridis', extent=extent)
    fig.colorbar(im, ax=ax, label='Spike count')
    ax.set_title('Raw spike counts per bin')
    ax.set_aspect('equal')

    ax = axes[1, 0]
    masked = np.ma.masked_invalid(rm['raw_rate_nan'])
    im = ax.imshow(masked.T, origin='lower', cmap='jet', extent=extent)
    fig.colorbar(im, ax=ax, label='Rate (Hz)')
    ax.set_title('Raw rate (spikes/occupancy)\nwhite = unvisited (NaN before fill)')
    ax.set_aspect('equal')

    ax = axes[1, 1]
    im = ax.imshow(rm['rate_map'].T, origin='lower', cmap='jet', extent=extent)
    fig.colorbar(im, ax=ax, label='Rate (Hz)')
    ax.set_title('Final: NaN-filled + Gaussian-smoothed\n(never-visited bins re-zeroed)')
    ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_step03_field_index(pos_ts, pos_xy, rm, fi_map, field_index, method, unit_label, out_path):
    extent = [rm['x_edges'][0], rm['x_edges'][-1], rm['y_edges'][0], rm['y_edges'][-1]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'{unit_label}\nStep 3: field index map (method={method})', fontsize=12, fontweight='bold')

    ax = axes[0]
    im = ax.imshow(fi_map.T, origin='lower', cmap='hot', vmin=0, vmax=1, extent=extent)
    fig.colorbar(im, ax=ax, label='Field index [0-1]')
    ax.set_title('field_index_map\n(0 = field edge/floor, 1 = peak)')
    ax.set_aspect('equal')

    ax = axes[1]
    valid_rates = rm['rate_map'][rm['occupancy'] > 0]
    ax.hist(valid_rates, bins=40, color='0.7', edgecolor='k', linewidth=0.3)
    if method == 'place':
        ax.axvline(valid_rates.min(), color='blue', linestyle='--', label='min -> field index 0')
        ax.axvline(valid_rates.max(), color='red', linestyle='--', label='max -> field index 1')
        ax.set_title("method='place': min-max normalize occupied-bin rates")
    else:
        ax.set_title("method='grid': rank/percentile-transform occupied-bin rates")
    ax.set_xlabel('Smoothed rate (Hz), occupied bins only')
    ax.set_ylabel('Bin count')
    ax.legend(fontsize=8)

    ax = axes[2]
    sc = ax.scatter(pos_xy[:, 0], pos_xy[:, 1], c=field_index, cmap='hot', vmin=0, vmax=1, s=3)
    fig.colorbar(sc, ax=ax, label='Field index at position')
    ax.set_title('field_index_per_position\n(looked up at every tracked sample)')
    ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_step04_arc_resampling(pos_ts, arc, field_index, cc, ts2, resampled, unit_label, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    fig.suptitle(f'{unit_label}\nStep 4: resample field index onto uniform arc length',
                 fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(pos_ts, arc, color='0.3', linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative arc length (pos units)')
    ax.set_title('Path distance traveled vs. time (flat stretches = animal not moving, dropped)')

    ax = axes[1]
    ax.plot(pos_ts, field_index, color='0.3', linewidth=0.6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Field index')
    ax.set_title('Field index vs. TIME (unevenly spaced in distance -- before resampling)')

    ax = axes[2]
    ax.plot(cc, resampled, color='darkorange', linewidth=0.8)
    ax.set_xlabel('Arc length traveled (pos units)')
    ax.set_ylabel('Field index')
    ax.set_title('Field index vs. ARC LENGTH, evenly resampled (this feeds Step 5)')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_step05_bandpass(cc, resampled, filtered_field_index, fs_arc, filter_band, unit_label, out_path):
    freqs_r, psd_r = signal.periodogram(resampled, fs=fs_arc, window='hann')
    freqs_f, psd_f = signal.periodogram(filtered_field_index, fs=fs_arc, window='hann')

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    fig.suptitle(f'{unit_label}\nStep 5: spatial bandpass filter (band = '
                 f'[{filter_band[0]:.4f}, {filter_band[1]:.4f}] cycles/unit distance)',
                 fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(cc, resampled, color='0.6', linewidth=0.8, label='resampled field index (input)')
    ax.plot(cc, filtered_field_index, color='darkgreen', linewidth=1.2, label='bandpass-filtered (output)')
    ax.set_xlabel('Arc length (pos units)')
    ax.set_ylabel('Field index')
    ax.set_title('Signal before / after zero-phase Butterworth bandpass')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.semilogy(freqs_r, psd_r + 1e-20, color='0.6', linewidth=1, label='input spectrum')
    ax.semilogy(freqs_f, psd_f + 1e-20, color='darkgreen', linewidth=1, label='filtered spectrum')
    ax.axvspan(filter_band[0], filter_band[1], color='red', alpha=0.15, label='passband')
    ax.set_xlim(0, min(freqs_r.max(), filter_band[1] * 5 + 1e-6))
    ax.set_xlabel('Spatial frequency (cycles / unit distance)')
    ax.set_ylabel('Power spectral density')
    ax.set_title('Power spectrum: passband isolates ~one oscillation per field traversal')
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_step06_hilbert(cc, filtered_field_index, envelope, pass_index_trace, unwrapped,
                         unit_label, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(f'{unit_label}\nStep 6: Hilbert transform -> pass index', fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(cc, filtered_field_index, color='darkgreen', linewidth=0.9, label='filtered field index')
    ax.plot(cc, envelope, color='0.3', linewidth=0.9, linestyle='--', label='Hilbert envelope |analytic signal|')
    ax.plot(cc, -envelope, color='0.3', linewidth=0.9, linestyle='--')
    ax.set_ylabel('Field index')
    ax.set_title('Analytic signal: filtered signal + its instantaneous amplitude envelope')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(cc, pass_index_trace, color='purple', linewidth=0.8)
    ax.axhline(-1, color='0.6', linestyle=':', linewidth=0.8)
    ax.axhline(0, color='0.6', linestyle=':', linewidth=0.8)
    ax.axhline(1, color='0.6', linestyle=':', linewidth=0.8)
    ax.set_ylabel('Pass index\n(angle(hilbert)/pi)')
    ax.set_title('-1 = entering field, 0 = field center, +1 = leaving field, then wraps to -1 again')

    ax = axes[2]
    ax.plot(cc, unwrapped / np.pi, color='teal', linewidth=0.8)
    ax.set_xlabel('Arc length (pos units)')
    ax.set_ylabel('Unwrapped phase / pi')
    ax.set_title('Unwrapped phase (continuous across consecutive passes; monotonic per direction of travel)')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_step06b_pass_traversals(pass_index_trace, passes, unit_label, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap('viridis')
    n_passes = len(passes)

    for i, (s, e) in enumerate(passes):
        trace = pass_index_trace[s:e]
        progress = np.linspace(0, 1, len(trace))
        color = cmap(i / max(n_passes - 1, 1))
        ax.plot(progress, trace, color=color, alpha=0.5, linewidth=1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(n_passes - 1, 1)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Pass number (chronological)')

    ax.axhline(-1, color='0.6', linestyle='--', linewidth=0.8)
    ax.axhline(0, color='0.6', linestyle='--', linewidth=0.8)
    ax.axhline(1, color='0.6', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Normalized progress through pass (0 = start, 1 = end)')
    ax.set_ylabel('Pass index (Hilbert phase / pi)')
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f'{unit_label}\nStep 6 (all traversals): {n_passes} individual field passes overlaid')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_step07_theta_phase(spk_ts, lfp_ts, lfp_sig, filtered_lfp, lfp_phase, spk_theta_phase,
                             lfp_filter_band, unit_label, out_path):
    t_center = spk_ts[0] if len(spk_ts) else 0.5 * (lfp_ts[0] + lfp_ts[-1])
    lo, hi, mask = _zoom_window(t_center, lfp_ts, ZOOM_WINDOW_SEC / 2.0)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    fig.suptitle(f'{unit_label}\nStep 7: LFP theta-phase extraction '
                 f'(band = {lfp_filter_band[0]}-{lfp_filter_band[1]} Hz)', fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(lfp_ts[mask], lfp_sig[mask], color='0.5', linewidth=0.8, label='raw LFP')
    ax.plot(lfp_ts[mask], filtered_lfp[mask], color='crimson', linewidth=1.2, label='theta-filtered LFP')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('LFP (uV)')
    ax.set_title(f'Raw vs. theta-bandpassed LFP, zoomed window [{lo:.2f}, {hi:.2f}] s')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(lfp_ts[mask], filtered_lfp[mask], color='crimson', linewidth=1.0, label='filtered LFP')
    ax2.plot(lfp_ts[mask], np.rad2deg(lfp_phase[mask]), color='navy', linewidth=1.0, label='Hilbert phase')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Filtered LFP (uV)', color='crimson')
    ax2.set_ylabel('Instantaneous theta phase (deg)', color='navy')
    ax.set_title('Instantaneous Hilbert phase tracks the filtered LFP oscillation')

    ax = axes[2]
    ax.hist(np.rad2deg(spk_theta_phase), bins=36, color='steelblue', edgecolor='k', linewidth=0.3)
    ax.set_xlabel('Spike theta phase (deg, wrapped to (-180, 180])')
    ax.set_ylabel('Spike count')
    ax.set_title(f'Theta phase at each spike time (n={len(spk_theta_phase)} spikes)')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_step08_regression(spk_pass_index, spk_theta_phase, s, b, unit_label, out_path):
    phis, phi_costs = phi_cost_curve(spk_pass_index, np.mod(spk_theta_phase, 2 * np.pi))
    s_vals, resultants = resultant_length_curve(spk_pass_index, np.mod(spk_theta_phase, 2 * np.pi),
                                                 SLOPE_SWEEP_RANGE)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f'{unit_label}\nStep 8: circular-linear regression (Kempter et al. 2012)',
                 fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(np.rad2deg(phis), phi_costs, color='0.3', linewidth=1)
    ax.set_xlabel('Candidate phase offset phi (deg)')
    ax.set_ylabel('Sum-squared residual')
    ax.set_title('Initial rough slope search:\nphi that best "unrolls" the wrapped phase')

    ax = axes[1]
    ax.plot(s_vals, resultants, color='0.3', linewidth=1)
    if not np.isnan(s):
        ax.axvline(s, color='red', linestyle='--', label=f'chosen slope s={s:.3f}')
    ax.set_xlabel('Candidate slope s (cycles / pass-index unit)')
    ax.set_ylabel('Mean resultant vector length')
    ax.set_title('Refinement: slope maximizing resultant length\n(insensitive to 2*pi wrap ambiguity)')
    ax.legend(fontsize=8)

    ax = axes[2]
    pi_dup = np.concatenate([spk_pass_index, spk_pass_index])
    phase_dup = np.concatenate([np.rad2deg(np.mod(spk_theta_phase, 2 * np.pi)),
                                 np.rad2deg(np.mod(spk_theta_phase, 2 * np.pi)) + 360])
    ax.scatter(pi_dup, phase_dup, s=8, alpha=0.6)
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
    ax.set_title('Final fit: theta phase vs. pass index\n(two theta cycles shown for readability)')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_step09_classification(results, unit_label, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axis('off')
    fig.suptitle(f'{unit_label}\nStep 9: precession classification', fontsize=12, fontweight='bold')

    slope = results['slope_deg_per_pass']
    lines = [
        f"rho (circular-linear corr.) = {results['rho']:.3f}",
        f"p-value = {results['p']:.4g}   (significant: p < 0.05 -> {results['p'] < 0.05})",
        f"slope = {slope:.2f} deg / full pass",
        f"slope in range (-1440, -22) deg/pass -> {-1440 < slope < -22}",
        "",
        f"is_precessing = significant AND slope in range",
        f"             = {results['is_precessing']}",
    ]
    for i, line in enumerate(lines):
        ax.text(0.02, 0.9 - i * 0.11, line, fontsize=11,
                fontweight='bold' if 'is_precessing' in line else 'normal',
                transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_step10_density(density, pi_edges, ph_edges, unit_label, out_path):
    ph_centers = np.rad2deg(0.5 * (ph_edges[:-1] + ph_edges[1:]))
    pi_centers = 0.5 * (pi_edges[:-1] + pi_edges[1:])
    dens_dup = np.concatenate([density, density], axis=1)
    ph_dup = np.concatenate([ph_centers, ph_centers + 360])

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(pi_centers, ph_dup, dens_dup.T, cmap='jet', shading='auto')
    fig.colorbar(im, ax=ax, label='Occupancy-normalized spike rate (Hz)')
    ax.set_xlabel('Pass index')
    ax.set_ylabel('LFP phase (deg)')
    ax.set_title(f'{unit_label}\nStep 10: pass index x LFP phase density map')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    ntt_path = NTT_FILE
    if not ntt_path.exists():
        raise FileNotFoundError(f'NTT_FILE does not exist: {ntt_path}')
    data_folder = ntt_path.parent

    ncs_files = sorted(data_folder.glob('*.ncs'), key=_natural_key)
    if not ncs_files:
        raise FileNotFoundError(f'No .ncs (LFP) file found in {data_folder}')
    theta_ncs = ncs_files[0]
    print(f'Using LFP file: {theta_ncs.name}')
    lfp_sig, lfp_ts, lfp_fs = load_ncs(theta_ncs)

    tracking_path = _find_tracking_file(data_folder)
    print(f'Using tracking file: {tracking_path.name}')
    pos_ts, pos_xy = load_tracking(tracking_path, TRACKING_TIME_UNIT)

    units = load_ntt_spike_times(ntt_path)
    units.pop(0, None)  # cluster 0 = unsorted noise (MClust convention); never analyzed
    assert 0 not in units, 'cluster 0 (noise) leaked into units -- should be unreachable'
    if not units:
        raise ValueError(f'{ntt_path.name} has no sorted (non-zero) clusters.')
    if CELL_NUMBER is None:
        cell_number = max(units, key=lambda c: len(units[c]))
    else:
        if CELL_NUMBER == 0:
            raise ValueError('CELL_NUMBER = 0 is unsorted noise and is never analyzed; '
                              'pick a real sorted cluster.')
        if CELL_NUMBER not in units:
            raise ValueError(f'Cell {CELL_NUMBER} not found in {ntt_path.name}. '
                              f'Available: {sorted(units)}')
        cell_number = CELL_NUMBER
    assert cell_number != 0
    spk_ts = units[cell_number]

    t_start = max(pos_ts.min(), lfp_ts.min())
    t_stop = min(pos_ts.max(), lfp_ts.max())
    spk_ts = spk_ts[(spk_ts >= t_start) & (spk_ts <= t_stop)]
    unit_label = f'{ntt_path.stem}_cell{cell_number}'
    print(f'Unit: {unit_label}  ({len(spk_ts)} spikes in tracking/LFP overlap window)')
    if len(spk_ts) < MIN_SPIKES_FOR_FIT:
        print(f'WARNING: only {len(spk_ts)} spikes (< MIN_SPIKES_FOR_FIT={MIN_SPIKES_FOR_FIT}); '
              f'Step 8-10 fit will be unreliable but plots will still be generated.')

    results = compute_all_steps(pos_ts, pos_xy, spk_ts, lfp_ts, lfp_sig, lfp_fs)

    output_dir = data_folder / 'PassIndex_StepByStep' / unit_label
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Writing step plots to: {output_dir}')

    plot_step00_raw_overview(pos_ts, pos_xy, spk_ts, lfp_ts, lfp_sig, unit_label,
                              output_dir / '00_RawData_Overview.png')
    plot_step01_spike_positions(pos_ts, pos_xy, spk_ts, results['spk_xy'], unit_label,
                                 output_dir / '01_SpikePositions.png')
    plot_step02_rate_map(results, unit_label, output_dir / '02_RateMap.png')
    plot_step03_field_index(pos_ts, pos_xy, results, results['field_index_map'],
                             results['field_index'], METHOD, unit_label,
                             output_dir / '03_FieldIndexMap.png')
    plot_step04_arc_resampling(pos_ts, results['arc'], results['field_index'], results['cc'],
                                results['ts2'], results['resampled'], unit_label,
                                output_dir / '04_ArcLengthResampling.png')
    plot_step05_bandpass(results['cc'], results['resampled'], results['filtered_field_index'],
                          results['fs_arc'], results['filter_band'], unit_label,
                          output_dir / '05_BandpassFilter.png')
    plot_step06_hilbert(results['cc'], results['filtered_field_index'], results['envelope'],
                         results['pass_index_trace'], results['unwrapped'], unit_label,
                         output_dir / '06_HilbertPassIndex.png')
    plot_step06b_pass_traversals(results['pass_index_trace'], results['passes'], unit_label,
                                  output_dir / '06b_PassTraversalsOverlay.png')
    plot_step07_theta_phase(spk_ts, lfp_ts, lfp_sig, results['filtered_lfp'], results['lfp_phase'],
                             results['spk_theta_phase'], LFP_FILTER_BAND, unit_label,
                             output_dir / '07_ThetaPhase.png')
    plot_step08_regression(results['spk_pass_index'], results['spk_theta_phase'], results['s'],
                            results['b'], unit_label, output_dir / '08_CircularLinearRegression.png')
    plot_step09_classification(results, unit_label, output_dir / '09_Classification.png')
    plot_step10_density(results['density'], results['pi_edges'], results['ph_edges'], unit_label,
                         output_dir / '10_DensityMap.png')

    print(f'rho={results["rho"]:.3f}  p={results["p"]:.3g}  '
          f'slope={results["slope_deg_per_pass"]:.1f} deg/pass  '
          f'precessing={results["is_precessing"]}')
    print('Done.')


if __name__ == '__main__':
    main()
