# -*- coding: utf-8 -*-
"""
Lap rasters for place cells in the open field – V1

2-D counterpart of LapRaster_Linearized_v2. An open field has no laps and no
single axis that suits every pass, so each pass gets its own 1-D coordinate:
the signed path length from the pass's closest approach to the field peak.
Passes cut through a field at different angles and offsets, so every raster row
is shaded by the rate the rest of the session predicts along that pass's path.

For every .ntt under <place_cell_folder>\\<animal>\\Open:
  1. Rebuild trajectory, spike frames and bin grid with
     PlaceCell_Main_v21.compute_metrics (same cleaning, smoothing, immobility
     filter and spike assignment as the run that detected the fields).
  2. Detect the place fields with detect_place_fields_threshold_2d.
  3. Passes: one contiguous run of moving-only frames inside the field (entry
     -> exit through any bin), as in TemporalStabilityMap_v6. Then:
       - runs separated by <= MERGE_GAP_S of real time are merged into one pass
         (an animal skimming a 2-D field boundary would otherwise fragment it);
       - passes shorter than MIN_PASS_S are dropped;
       - passes cut by the session start / end are kept and flagged;
       - a pause inside a pass does not end it (those frames are not in the
         moving-only sequence); passes with a gap > PAUSE_FLAG_S are flagged.
  4. Per pass, from a ratemap built WITHOUT that pass (leave-one-out, so a pass
     does not predict itself; Gaussian-smoothed raw rate, as the field map):
       expected_spikes   sum over the pass's frames of rate x dwell
       z                 (observed - expected) / sqrt(expected)
     z is only read on informative passes, expected_spikes >= MIN_EXPECTED_SPIKES.
     Category, from the bins the pass visited:
       Central     at least one frame in the field's peak bin
       Peripheral  every other pass
     peak_bin_dist_cm is the centre-to-centre distance from the pass's nearest
     bin to the peak bin, so it is 0 exactly for Central passes.
  5. Epochs: N_EPOCHS spans of equal moving time (not equal clock time). Each
     epoch's map is occupancy-weighted (smoothed spikes / smoothed dwell): an
     epoch has too little dwell per bin for the smoothed raw rate, where one
     spike in a single-frame bin reads as a very high rate. It is correlated
     with the map of the rest of the session on the bins visited in both. A field that moves by less than its own width still
     correlates well, so each epoch also gets a field centre of mass (COM): the
     rate-weighted mean position of the visited bins of the field, grown by
     COM_DILATE_BINS bins so the COM can leave the session outline, that reach
     >= COM_RATE_FRAC of that epoch's highest rate there. Epochs with less than
     MIN_COM_DWELL_S of dwell or no spikes in that area have no COM.
     Reference from simulated single-Gaussian cells on real tracking (about 280
     spikes): stable cells shift a median of 3.3 cm, 95th percentile 8.9 cm; at
     that cutoff 28% of 10 cm and 70% of 20 cm total drifts are detected. This is
     not a per-cell null: real fields are irregular and can shift more by chance.

Figure (one per cell, saved in LAP_SUBDIR next to its .ntt):
  row 1  trajectory + spikes coloured by time; session ratemap
  row 2  per field: distance to field peak vs time, spikes in red. A visit is a
         dip towards 0; no dip = no visit, a dip without spikes = no firing.
  row 3  per field: lap raster (pass # on y, top = first; x = signed path length,
         0 = closest approach; shading = leave-one-out rate along the path;
         spikes as red ticks; +/- CONTEXT_S s of path in gray, spikes on it in
         dark blue), then observed (bar) vs expected (black tick) spikes, then z
         with its running mean.
         A dark row without ticks is a true miss; a pale row without ticks is
         only a pass through the edge. The strip left of each raster is the
         pass's distance to the peak bin (magma: darkest = through the peak bin).
  row 4  per field: the same raster sorted by distance to the peak bin. If bands
         only form here and not in row 3, the pattern comes from path geometry,
         not time.
  row 5  ratemap per epoch on one colour scale, with fields outlined and each
         field's COM marked. Unvisited bins are blank.

Tables (output_excel):
  PerPass   one row per pass: times, flags, category, distance to the peak bin,
            closest approach, observed / expected spikes, z
  PerField  per field: pass counts, hit rate vs the Poisson expectation,
            overdispersion (variance of z; simulated Poisson cells give about
            1.2, not 1, because the leave-one-out map is itself noisy), the
            largest share of the field's in-pass spikes from a single pass (a
            field built from one visit), Spearman rho of z vs pass number
            (rate drift), COM shift between the first
            and last usable epoch and its largest pairwise range (location
            drift), and lap stability both on this script's passes and with
            TemporalStabilityMap_v6's rule (no merging, session-edge passes
            dropped) so the two can be compared
  PerEpoch  per epoch: time range, correlation with the rest of the session,
            rate, dwell and COM of each field

Analysis settings (bin size, smoothing, filters, field thresholds, COORD_UNITS, ...)
come from PlaceCell_Main_v21; only the directories below are set here.
"""

import os
import concurrent.futures
import threading

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, convolve
from scipy.stats import spearmanr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib import colormaps
from matplotlib.patches import Patch

import PlaceCell_Main_v21 as pc


# ── Directories (edit these per run) ──────────────────────────────────────────

# Place-cell folder (Output_PlaceTrue of a PlaceCell_Main_v21 run): every .ntt
# under <animal>\Open in it is analysed.
place_cell_folder = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Alpha200'

# Per-pass / per-field / per-epoch results table.
output_excel = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Alpha200\Results\LapRaster_OpenField.xlsx'

# Figures go in this subfolder of each session folder (next to the .ntt).
LAP_SUBDIR = 'lap_raster_openfield'

# ── Settings ──────────────────────────────────────────────────────────────────

MIN_PASS_S          = 0.1   # passes shorter than this (s) are border flicker, dropped; 0 = keep all
MERGE_GAP_S         = 0.5   # field exits this short (s, real time) do not end a pass; 0 = no merging
PAUSE_FLAG_S        = 1.0   # a pass with a tracking gap (immobility) longer than this (s) is flagged
PEAK_DIST_MAX_CM    = 15.0  # strip colour scale: passes this far (cm) or more from the peak bin are lightest
MIN_SPIKES_PER_PASS = 5     # lap stability: a pass counts as stable with >= this many spikes
MIN_EXPECTED_SPIKES = 1.0   # informative pass: expected spikes >= this; only these enter z statistics
CONTEXT_S           = 1.0   # path drawn (gray) this long before / after each pass
N_EPOCHS            = 5     # equal-moving-time epochs in the filmstrip
COM_DILATE_BINS     = 3     # field area for the epoch COM: the field grown by this many bins
COM_RATE_FRAC       = 0.5   # epoch COM uses bins >= this fraction of the epoch's rate maximum there
MIN_COM_DWELL_S     = 2.0   # no epoch COM with less dwell than this (s) in that area
Z_DISPLAY_CLIP      = 6.0   # z bars are clipped to +/- this (the tables keep the real value)
Z_TREND_PASSES      = 9     # running mean of z over this many passes
RASTER_X_STEP_CM    = 1.0   # x resolution of the raster shading (cm)

FIELD_COLORS =['tab:green', 'tab:purple', 'tab:cyan', 'tab:olive', 'tab:brown', 'tab:pink']
SPIKE_COLOR         = 'red'
CONTEXT_SPIKE_COLOR = 'navy'
TIME_CMAP           = 'viridis'
Z_ABOVE_COLOR       = 'tab:red'
Z_BELOW_COLOR       = 'tab:blue'

# One sequential scale for rate in the raster shading and the filmstrip; dropped
# the palest and darkest ends so that 0 Hz differs from no data (white) and the
# red spike ticks stay visible on the highest rates.
RATE_CMAP = LinearSegmentedColormap.from_list(
    'rate_blues', colormaps['Blues'](np.linspace(0.12, 0.78, 256)))
RATE_CMAP.set_bad('white')

# Distance to the peak bin in the strip; dropped magma's palest end so that the
# farthest passes still show against the white figure background.
PEAK_DIST_CMAP = LinearSegmentedColormap.from_list(
    'peak_dist_magma', colormaps['magma'](np.linspace(0.0, 0.92, 256)))

MAX_WORKERS = pc.MAX_WORKERS

_print_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_tracking_csv(dirpath: str) -> str | None:
    """Same tracking-file selection as PlaceCell_Main_v21's batch scan."""
    tracking = [f for f in os.listdir(dirpath) if f.lower().endswith(('.csv', '.xlsx'))]
    if pc.COORD_UNITS == 'cm':
        tracking = [f for f in tracking if f.lower().endswith('_cm.csv')]
    else:
        tracking = [f for f in tracking if not f.lower().endswith('_cm.csv')]
    return os.path.join(dirpath, tracking[0]) if len(tracking) == 1 else None


def _contour_field(ax, field_mask: np.ndarray, bin_cm: float, color: str,
                   upsample: int = 8) -> None:
    """Outline a 2-D field mask along bin edges on cm axes (as in
    TemporalStabilityMap_v6_MapPassStability._contour_field)."""
    up = np.kron(np.pad(field_mask.astype(float), 1), np.ones((upsample, upsample)))
    xc = ((np.arange(up.shape[0]) + 0.5) / upsample - 1) * bin_cm
    yc = ((np.arange(up.shape[1]) + 0.5) / upsample - 1) * bin_cm
    ax.contour(xc, yc, up.T, levels=[0.5], colors=color, linewidths=1.5)


def _smooth_rate(spk: np.ndarray, occ: np.ndarray, valid: np.ndarray,
                 kernel: np.ndarray) -> np.ndarray:
    """Gaussian-smoothed raw rate, as pc._gaussian_smooth, but evaluated on every
    bin the kernel reaches (NaN elsewhere) instead of being zeroed outside
    `valid`: a pass may be the only sample of a bin it should still be predicted in."""
    raw = np.zeros_like(occ)
    raw[valid] = spk[valid] / occ[valid]
    num = convolve(raw, kernel, mode='constant', cval=0.0)
    den = convolve(valid.astype(float), kernel, mode='constant', cval=0.0)
    out = np.full_like(occ, np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def _smooth_rate_occ(spk: np.ndarray, occ: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Occupancy-weighted rate: smoothed spikes / smoothed dwell (NaN where the
    kernel reaches no dwell). Used for epochs, which are sparsely sampled."""
    num = convolve(spk, kernel, mode='constant', cval=0.0)
    den = convolve(occ, kernel, mode='constant', cval=0.0)
    out = np.full_like(occ, np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def _fmt(value, suffix: str = '') -> str:
    """Number for a figure title; 'n/a' for NaN (e.g. no informative passes)."""
    return 'n/a' if pd.isna(value) else f'{value}{suffix}'


def _fields_2d(metrics: dict, ctx: dict, bin_cm: float) -> list[dict]:
    """Place fields as masks, with peak bin / position / rate from the field map."""
    fields, _ = pc.detect_place_fields_threshold_2d(metrics, ctx, bin_cm)
    out = []
    for f in fields:
        peak_bin = (int(f['peak_bin_x']), int(f['peak_bin_y']))
        out.append({'field_number': f['field_number'],
                    'mask': pc._field_mask_from_bin_coords(f['bin_coords'], ctx['n_bins_x'],
                                                           ctx['n_bins_y']),
                    'peak_bin': peak_bin,
                    'peak_xy': ((peak_bin[0] + 0.5) * bin_cm, (peak_bin[1] + 0.5) * bin_cm),
                    'peak_rate': float(ctx['fr_field'][peak_bin])})
    return out


# ── Passes ────────────────────────────────────────────────────────────────────

def _pass_runs(in_field: np.ndarray, t_s: np.ndarray) -> list[list[int]]:
    """Contiguous runs of in-field frames as [first, last, n_fragments]; runs
    separated by <= MERGE_GAP_S of real time are merged."""
    edges  = np.diff(np.r_[0, in_field.astype(np.int8), 0])
    starts = np.flatnonzero(edges == 1)
    ends   = np.flatnonzero(edges == -1) - 1
    runs: list[list[int]] = []
    for a, b in zip(starts, ends):
        if runs and t_s[a] - t_s[runs[-1][1]] <= MERGE_GAP_S:
            runs[-1][1] = int(b)
            runs[-1][2] += 1
        else:
            runs.append([int(a), int(b), 1])
    return runs


def _v6_rule_score(in_field: np.ndarray, dt_frames: np.ndarray, spike_frame: np.ndarray) -> tuple[int, int]:
    """TemporalStabilityMap_v6._pass_stats: no merging, session-edge passes
    dropped. Returns (n_passes, n_passes_with_min_spikes)."""
    if not in_field.any():
        return 0, 0
    edges  = np.diff(np.r_[0, in_field.astype(np.int8), 0])
    starts = np.flatnonzero(edges == 1)
    ends   = np.flatnonzero(edges == -1) - 1
    cum_dt = np.r_[0.0, np.cumsum(dt_frames)]
    dur    = cum_dt[ends + 1] - cum_dt[starts]
    spk = np.sort(spike_frame)
    n_spk = (np.searchsorted(spk, ends, side='right')
             - np.searchsorted(spk, starts, side='left'))
    keep = (dur >= MIN_PASS_S) & (starts > 0) & (ends < in_field.size - 1)
    return int(keep.sum()), int((keep & (n_spk >= MIN_SPIKES_PER_PASS)).sum())


def _passes(fld: dict, ctx: dict, t_s: np.ndarray, path: np.ndarray, kernel: np.ndarray,
            bin_cm: float) -> list[dict]:
    """Every pass through one field, in time order, with its leave-one-out
    expected spikes. Keys starting with '_' are per-frame arrays for the figure."""
    bx, by, dt = ctx['beh_bx'], ctx['beh_by'], ctx['dt_frames']
    x, y = ctx['x_cm'], ctx['y_cm']
    spk_sorted = np.sort(ctx['spike_frame'])
    in_field = fld['mask'][bx, by]
    if not in_field.any():
        return []
    cum_dt = np.r_[0.0, np.cumsum(dt)]
    dist = np.hypot(x - fld['peak_xy'][0], y - fld['peak_xy'][1])
    last = in_field.size - 1

    rows = []
    for a, b, n_frag in _pass_runs(in_field, t_s):
        dur = float(cum_dt[b + 1] - cum_dt[a])
        if dur < MIN_PASS_S:
            continue
        i0, i1 = np.searchsorted(spk_sorted, a, 'left'), np.searchsorted(spk_sorted, b, 'right')
        pass_spk = spk_sorted[i0:i1]

        # Ratemap without this pass.
        occ_p = np.zeros_like(ctx['occ_map'])
        spk_p = np.zeros_like(ctx['occ_map'])
        np.add.at(occ_p, (bx[a:b + 1], by[a:b + 1]), dt[a:b + 1])
        np.add.at(spk_p, (bx[pass_spk], by[pass_spk]), 1.0)
        occ_loo = ctx['occ_map'] - occ_p
        spk_loo = ctx['spike_map'] - spk_p
        rate = _smooth_rate(spk_loo, occ_loo, ctx['valid_mask'] & (occ_loo > 1e-9), kernel)
        rate_frames = rate[bx[a:b + 1], by[a:b + 1]]

        expected = float(np.nansum(rate_frames * dt[a:b + 1]))
        observed = int(pass_spk.size)
        z = (observed - expected) / np.sqrt(expected) if expected > 0 else np.nan

        ca = a + int(np.argmin(dist[a:b + 1]))             # closest approach to the peak
        reached = float(ctx['fr_field'][bx[a:b + 1], by[a:b + 1]].max()) / fld['peak_rate']
        peak_bin_dist = bin_cm * float(np.hypot(bx[a:b + 1] - fld['peak_bin'][0],
                                                by[a:b + 1] - fld['peak_bin'][1]).min())
        net = float(np.hypot(x[b] - x[a], y[b] - y[a]))
        pause =float(np.diff(t_s[a:b + 1]).max()) if b > a else 0.0

        rows.append({'_start': a, '_end': b, '_ca': ca, '_rate': rate_frames,
                     't_start_s': round(float(t_s[a]), 3), 't_end_s': round(float(t_s[b]), 3),
                     'duration_s': round(dur, 3),
                     'n_fragments': n_frag,
                     'session_edge': bool(a == 0 or b == last),
                     'max_pause_s': round(pause, 3),
                     'paused': bool(pause > PAUSE_FLAG_S),
                     'category': 'Central' if peak_bin_dist == 0 else 'Peripheral',
                     'peak_bin_dist_cm': round(peak_bin_dist, 2),
                     'reached_rate_frac': round(reached, 3),
                     'closest_approach_cm': round(float(dist[ca]), 2),
                     'path_cm': round(float(path[b] - path[a]), 2),
                     'net_displacement_cm': round(net, 2),
                     'n_spikes': observed,
                     'expected_spikes': round(expected, 3),
                     'z': round(float(z), 3) if not np.isnan(z) else np.nan,
                     'informative': bool(expected >= MIN_EXPECTED_SPIKES)})
    return rows


def _field_summary(base: dict, fld: dict, passes: list[dict], ctx: dict) -> dict:
    counts   = np.array([p['n_spikes'] for p in passes])
    expected = np.array([p['expected_spikes'] for p in passes])
    z        = np.array([p['z'] for p in passes], dtype=float)
    info     = np.array([p['informative'] for p in passes], dtype=bool)
    cats     = [p['category'] for p in passes]
    n = len(passes)
    central = counts[[c == 'Central' for c in cats]] if n else counts
    z_info = z[info] if n else z
    if z_info.size >= 3 and np.ptp(z_info) > 0:
        rho, pval = spearmanr(np.arange(z_info.size), z_info)
    else:
        rho, pval = np.nan, np.nan
    in_field = fld['mask'][ctx['beh_bx'], ctx['beh_by']]
    n_v6, n_v6_ok = _v6_rule_score(in_field, ctx['dt_frames'], ctx['spike_frame'])
    return {**base, 'field_number': fld['field_number'],
            'n_field_bins': int(fld['mask'].sum()),
            'peak_x_cm': round(fld['peak_xy'][0], 2), 'peak_y_cm': round(fld['peak_xy'][1], 2),
            'peak_rate_hz': round(fld['peak_rate'], 3),
            'n_passes': n,
            'n_central': cats.count('Central'),
            'n_peripheral': cats.count('Peripheral'),
            'n_session_edge': sum(p['session_edge'] for p in passes),
            'n_paused': sum(p['paused'] for p in passes),
            'n_merged': sum(p['n_fragments'] > 1 for p in passes),
            'n_informative': int(info.sum()),
            'observed_spikes': int(counts.sum()),
            'expected_spikes': round(float(expected.sum()), 2),
            'hit_rate_informative': round(float((counts[info] > 0).mean()), 3) if info.any() else np.nan,
            'poisson_hit_rate_informative':
                round(float((1.0 - np.exp(-expected[info])).mean()), 3) if info.any() else np.nan,
            'overdispersion_var_z': round(float(np.var(z_info, ddof=1)), 3) if z_info.size >= 3 else np.nan,
            'top_pass_spike_frac':
                round(float(counts.max() / counts.sum()), 3) if n and counts.sum() > 0 else np.nan,
            'spearman_rho_z_vs_pass': round(float(rho), 3) if not np.isnan(rho) else np.nan,
            'spearman_p': round(float(pval), 4) if not np.isnan(pval) else np.nan,
            'lap_stability_score':
                round(float((counts >= MIN_SPIKES_PER_PASS).mean()), 3) if n else np.nan,
            'lap_stability_central':
                round(float((central >= MIN_SPIKES_PER_PASS).mean()), 3) if central.size else np.nan,
            'n_epochs_with_com': fld['n_epochs_with_com'],
            'com_shift_first_last_cm': fld['com_shift_first_last_cm'],
            'com_range_cm': fld['com_range_cm'],
            'n_passes_v6_rule': n_v6,
            'lap_stability_v6_rule': round(n_v6_ok / n_v6, 3) if n_v6 else np.nan}


# ── Epochs ────────────────────────────────────────────────────────────────────

def _epoch_com(rate: np.ndarray, occ: np.ndarray, spk: np.ndarray, area: np.ndarray,
               bin_cm: float) -> tuple[float, float]:
    """Rate-weighted COM (cm) of the visited bins of `area` that reach
    COM_RATE_FRAC of the epoch's highest rate there; NaN if undersampled."""
    seen = area & (occ > 0)
    if occ[seen].sum() < MIN_COM_DWELL_S or spk[seen].sum() == 0:
        return np.nan, np.nan
    r = np.where(seen, rate, 0.0)
    if r.max() <= 0:
        return np.nan, np.nan
    keep = r >= COM_RATE_FRAC * r.max()
    ix, iy = np.nonzero(keep)
    w = r[keep]
    return (float(np.sum(w * (ix + 0.5)) / w.sum()) * bin_cm,
            float(np.sum(w * (iy + 0.5)) / w.sum()) * bin_cm)


def _com_shift(fld: dict, epochs: list[dict]) -> None:
    """Field COM drift across epochs, stored on the field: first -> last usable
    epoch, and the largest distance between any two usable epochs."""
    n = fld['field_number']
    xy = np.array([(ep[f'field{n}_com_x_cm'], ep[f'field{n}_com_y_cm']) for ep in epochs])
    xy = xy[~np.isnan(xy[:, 0])]
    fld['n_epochs_with_com'] = int(len(xy))
    fld['com_shift_first_last_cm'] = (round(float(np.linalg.norm(xy[-1] - xy[0])), 2)
                                      if len(xy) >= 2 else np.nan)
    fld['com_range_cm'] = (round(float(np.linalg.norm(xy[:, None] - xy[None, :], axis=2).max()), 2)
                           if len(xy) >= 2 else np.nan)


def _epochs(ctx: dict, fields: list[dict], t_s: np.ndarray, kernel: np.ndarray,
            bin_cm: float) -> list[dict]:
    """N_EPOCHS spans of equal moving time: map, correlation with the rest of the
    session (bins visited in both), rate, dwell and COM of each field."""
    bx, by, dt = ctx['beh_bx'], ctx['beh_by'], ctx['dt_frames']
    cum = np.cumsum(dt)
    cuts = np.searchsorted(cum, cum[-1] * np.arange(1, N_EPOCHS) / N_EPOCHS)
    frames = np.r_[0, cuts, bx.size]
    spike_frame = ctx['spike_frame']
    out = []
    for k in range(N_EPOCHS):
        f0, f1 = int(frames[k]), int(frames[k + 1])
        occ = np.zeros_like(ctx['occ_map'])
        np.add.at(occ, (bx[f0:f1], by[f0:f1]), dt[f0:f1])
        s = spike_frame[(spike_frame >= f0) & (spike_frame < f1)]
        spk = np.zeros_like(occ)
        np.add.at(spk, (bx[s], by[s]), 1.0)
        rate = _smooth_rate_occ(spk, occ, kernel)

        occ_rest = np.clip(ctx['occ_map'] - occ, 0.0, None)
        spk_rest = ctx['spike_map'] - spk
        rate_rest = _smooth_rate_occ(spk_rest, occ_rest, kernel)
        joint = (occ > 0) & (occ_rest > 1e-9)
        r = np.nan
        if joint.sum() >= pc.STABILITY_MIN_BINS and rate[joint].std() > 0 and rate_rest[joint].std() > 0:
            r = float(np.corrcoef(rate[joint], rate_rest[joint])[0, 1])

        row = {'epoch': k + 1, 't_start_min': round(float(t_s[f0]) / 60.0, 2),
               't_end_min': round(float(t_s[f1 - 1]) / 60.0, 2),
               'n_spikes': int(s.size), 'n_joint_bins': int(joint.sum()),
               'r_vs_rest': round(r, 3) if not np.isnan(r) else np.nan, '_rate': rate, '_occ': occ}
        for fld in fields:
            occ_f = float(occ[fld['mask']].sum())
            row[f"field{fld['field_number']}_dwell_s"] = round(occ_f, 2)
            row[f"field{fld['field_number']}_rate_hz"] = (
                round(float(spk[fld['mask']].sum()) / occ_f, 3) if occ_f > 0 else np.nan)
            cx, cy = _epoch_com(rate, occ, spk,
                                binary_dilation(fld['mask'], iterations=COM_DILATE_BINS), bin_cm)
            row[f"field{fld['field_number']}_com_x_cm"] = round(cx, 2) if not np.isnan(cx) else np.nan
            row[f"field{fld['field_number']}_com_y_cm"] = round(cy, 2) if not np.isnan(cy) else np.nan
        out.append(row)
    return out


def analyse_cell(metrics: dict, ctx: dict, bin_cm: float) -> dict | None:
    """Fields, passes and epochs of one cell; None if no field is detected."""
    fields = _fields_2d(metrics, ctx, bin_cm)
    if not fields:
        return None
    t_s = (ctx['t'] - ctx['t_all'][0]) * 1e-6             # s from session start
    path = np.r_[0.0, np.cumsum(np.hypot(np.diff(ctx['x_cm']), np.diff(ctx['y_cm'])))]
    kernel = pc._gaussian_kernel(pc.GAUSSIAN_SIGMA_CM / bin_cm)
    for fld in fields:
        fld['passes'] = _passes(fld, ctx, t_s, path, kernel, bin_cm)
    epochs = _epochs(ctx, fields, t_s, kernel, bin_cm)
    for fld in fields:
        _com_shift(fld, epochs)
    return {'fields': fields, 't_s': t_s, 'path': path, 'epochs': epochs}


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_trajectory(ax, fig, res: dict, ctx: dict, bin_cm: float, extent: list) -> None:
    """Trajectory in gray, spikes coloured by time, fields outlined."""
    spike_frame = ctx['spike_frame']
    ax.plot(ctx['x_cm_all'], ctx['y_cm_all'], color='0.8', linewidth=0.4)
    sc = ax.scatter(ctx['x_cm'][spike_frame], ctx['y_cm'][spike_frame],
                    c=res['t_s'][spike_frame] / 60.0, cmap=TIME_CMAP, s=5, linewidths=0, zorder=3)
    for k, fld in enumerate(res['fields']):
        _contour_field(ax, fld['mask'], bin_cm, FIELD_COLORS[k % len(FIELD_COLORS)])
    fig.colorbar(sc, ax=ax, label='spike time (min)', fraction=0.046, pad=0.02)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_title('Trajectory + spikes coloured by time (fields outlined)')


def _plot_session_map(ax, fig, res: dict, ctx: dict, bin_cm: float, extent: list) -> None:
    """Session ratemap with fields outlined and their peaks starred."""
    im = ax.imshow(np.ma.masked_where(~ctx['valid_mask'], ctx['fr_field']).T, origin='lower',
                   extent=extent, cmap=RATE_CMAP, vmin=0, interpolation='nearest')
    for k, fld in enumerate(res['fields']):
        _contour_field(ax, fld['mask'], bin_cm, FIELD_COLORS[k % len(FIELD_COLORS)])
        ax.plot(*fld['peak_xy'], marker='*', color='white', markeredgecolor='black',
                markersize=11, linestyle='none')
        ax.annotate(str(fld['field_number']), fld['peak_xy'], xytext=(6, 6),
                    textcoords='offset points', fontsize=9, fontweight='bold')
    fig.colorbar(im, ax=ax, label='rate (Hz)', fraction=0.046, pad=0.02)
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_title(f'Session ratemap (Gaussian, sigma = {pc.GAUSSIAN_SIGMA_CM:g} cm)')


def _plot_dist_time(ax, fld: dict, res: dict, ctx: dict, color: str) -> None:
    """Distance to the field peak vs time (moving frames): in-field frames in the
    field's colour, spikes in red. Lines are broken at pauses and tracking gaps."""
    t_min = res['t_s'] / 60.0
    dist = np.hypot(ctx['x_cm'] - fld['peak_xy'][0], ctx['y_cm'] - fld['peak_xy'][1])
    in_field = fld['mask'][ctx['beh_bx'], ctx['beh_by']]
    cut = np.flatnonzero(np.diff(res['t_s']) > 0.5) + 1
    pts = np.column_stack([t_min, dist])
    ax.add_collection(LineCollection([s for s in np.split(pts, cut) if len(s) > 1],
                                     colors='0.75', linewidths=0.4))
    ax.plot(t_min[in_field], dist[in_field], '.', color=color, markersize=1.0)
    sf = ctx['spike_frame']
    ax.plot(t_min[sf], dist[sf], '.', color=SPIKE_COLOR, markersize=2.5)
    ax.set_xlim(t_min[0], t_min[-1])
    ax.set_ylim(0, np.percentile(dist, 99.5))
    ax.set_xlabel('time (min)')
    ax.set_ylabel('distance to peak (cm)')
    ax.set_title(f"Field {fld['field_number']}: distance to peak vs time "
                 f"(in-field frames coloured, spikes red)", fontsize=9)


def _raster_image(passes: list[dict], res: dict, xlim: float) -> tuple[np.ndarray, np.ndarray]:
    """(n_passes, n_x) leave-one-out rate along each pass on the signed-path axis;
    NaN where the pass did not go."""
    grid = np.arange(-xlim, xlim + RASTER_X_STEP_CM, RASTER_X_STEP_CM)
    img = np.full((len(passes), grid.size), np.nan)
    path = res['path']
    for i, p in enumerate(passes):
        xs = path[p['_start']:p['_end'] + 1] - path[p['_ca']]
        rate = np.nan_to_num(p['_rate'], nan=0.0)
        inside = (grid >= xs[0]) & (grid <= xs[-1])
        if inside.any():
            img[i, inside] = np.interp(grid[inside], xs, rate)
        elif abs(xs.mean()) <= xlim:                       # shorter than one grid step
            img[i, int(np.argmin(np.abs(grid - xs.mean())))] = rate.mean()
    return img, grid


def _plot_raster(ax, passes: list[dict], order: np.ndarray, img: np.ndarray, grid: np.ndarray,
                 fld: dict, res: dict, ctx: dict, xlim: float, title: str, fontsize: float = 9,
                 loc: str = 'center') -> None:
    """Passes in the given order (first at top): shading = leave-one-out rate along
    the path, gray = context, spikes as ticks that scale with the row height."""
    t_s, path = res['t_s'], res['path']
    spk = np.sort(ctx['spike_frame'])
    n = len(passes)
    step = RASTER_X_STEP_CM
    ax.imshow(img[order], cmap=RATE_CMAP, vmin=0, vmax=fld['peak_rate'], aspect='auto',
              interpolation='nearest',
              extent=(grid[0] - step / 2, grid[-1] + step / 2, n + 0.5, 0.5))
    ctx_segs, spike_segs, ctx_spike_segs = [], [], []
    for row, idx in enumerate(order, start=1):
        p = passes[idx]
        a, b = p['_start'], p['_end']
        c0 = int(np.searchsorted(t_s, t_s[a] - CONTEXT_S, 'left'))
        c1 = int(np.searchsorted(t_s, t_s[b] + CONTEXT_S, 'right')) - 1
        p0 = path[p['_ca']]
        if c0 < a:
            ctx_segs.append([(path[c0] - p0, row), (path[a] - p0, row)])
        if c1 > b:
            ctx_segs.append([(path[b] - p0, row), (path[c1] - p0, row)])
        f_ctx = spk[np.searchsorted(spk, c0, 'left'):np.searchsorted(spk, c1, 'right')]
        for f in f_ctx:
            pos = path[f] - p0
            seg = [(pos, row - 0.42), (pos, row + 0.42)]
            (spike_segs if a <= f <= b else ctx_spike_segs).append(seg)
    ax.add_collection(LineCollection(ctx_segs, colors='0.7', linewidths=0.6, zorder=2))
    ax.add_collection(LineCollection(ctx_spike_segs, colors=CONTEXT_SPIKE_COLOR, linewidths=0.8, zorder=3))
    ax.add_collection(LineCollection(spike_segs, colors=SPIKE_COLOR, linewidths=1.0, zorder=4))
    ax.axvline(0, color='0.35', ls='--', lw=0.7, zorder=1)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(n + 0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('signed path length from closest approach (cm)')
    ax.set_title(title, fontsize=fontsize, loc=loc)


def _plot_strip(ax, passes: list[dict], order: np.ndarray, labels: list[str], ylabel: str) -> None:
    """Narrow strip left of a raster: distance to the peak bin, pause and session-edge flags, y labels."""
    n = len(passes)
    dist = np.array([[passes[i]['peak_bin_dist_cm']] for i in order])
    ax.imshow(dist, cmap=PEAK_DIST_CMAP, vmin=0, vmax=PEAK_DIST_MAX_CM,
              aspect='auto', interpolation='nearest', extent=(0, 1, n + 0.5, 0.5))
    rows = np.arange(1, n + 1)
    paused = np.array([passes[i]['paused'] for i in order])
    edge = np.array([passes[i]['session_edge'] for i in order])
    ax.plot(np.full(paused.sum(), 1.5), rows[paused], 's', color='black', markersize=2.5)
    ax.plot(np.full(edge.sum(), 1.5), rows[edge], 'x', color='black', markersize=4)
    ax.set_xlim(0, 2)
    ax.set_ylim(n + 0.5, 0.5)
    ax.set_xticks([])
    step = max(1, int(np.ceil(n / 25)))
    ticks = list(range(1, n + 1, step))
    ax.set_yticks(ticks)
    ax.set_yticklabels([labels[k - 1] for k in ticks], fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    for s in ('top', 'right', 'bottom'):
        ax.spines[s].set_visible(False)


def _plot_counts(ax, passes: list[dict]) -> None:
    """Observed spikes as bars, expected as black ticks, square-root count axis."""
    n = len(passes)
    rows = np.arange(1, n + 1)
    ax.barh(rows, [p['n_spikes'] for p in passes], height=0.8, color=SPIKE_COLOR, alpha=0.75)
    exp = np.array([p['expected_spikes'] for p in passes])
    ax.add_collection(LineCollection([[(e, r - 0.45), (e, r + 0.45)] for e, r in zip(exp, rows)],
                                     colors='black', linewidths=1.0))
    top = max(max(p['n_spikes'] for p in passes), float(exp.max()), 1.0)
    ax.set_xscale('function', functions=(lambda v: np.sqrt(np.maximum(v, 0)), np.square))
    ax.set_xlim(0, top * 1.05)
    ax.set_xticks([t for t in (1, 5, 20, 100, 500) if t <= top * 1.05])
    ax.tick_params(axis='x', labelsize=7)
    ax.set_ylim(n + 0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('spikes\nbar: observed\ntick: expected', fontsize=7)


def _plot_z(ax, passes: list[dict]) -> None:
    """z per pass (informative passes coloured by sign, others pale gray) and the
    running mean of z over informative passes."""
    n = len(passes)
    rows = np.arange(1, n + 1)
    z = np.array([p['z'] for p in passes], dtype=float)
    info = np.array([p['informative'] for p in passes], dtype=bool)
    shown = np.clip(np.nan_to_num(z), -Z_DISPLAY_CLIP, Z_DISPLAY_CLIP)
    colors = np.where(~info, '0.88', np.where(z >= 0, Z_ABOVE_COLOR, Z_BELOW_COLOR))
    ax.barh(rows, shown, height=0.8, color=colors)
    ax.axvline(0, color='0.3', lw=0.7)
    trend = (pd.Series(np.where(info, z, np.nan))
             .rolling(Z_TREND_PASSES, center=True, min_periods=3).mean().to_numpy())
    ax.plot(np.clip(trend, -Z_DISPLAY_CLIP, Z_DISPLAY_CLIP), rows, color='black', lw=1.4)
    ax.set_xlim(-Z_DISPLAY_CLIP, Z_DISPLAY_CLIP)
    ax.set_ylim(n + 0.5, 0.5)
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=7)
    ax.set_xlabel(f'z\nline: {Z_TREND_PASSES}-pass mean', fontsize=7)


def _plot_filmstrip(fig, gs, res: dict, bin_cm: float, extent: list) -> None:
    """Ratemap per epoch on one colour scale; unvisited bins blank, fields outlined.
    The panels stay at their 1-field size, left-aligned, however wide the figure is."""
    # An epoch can fire above the session map's peak; the 99th percentile keeps its
    # structure visible without letting one poorly sampled bin set the scale.
    visited = np.concatenate([ep['_rate'][ep['_occ'] > 0] for ep in res['epochs']])
    vmax = float(np.nanpercentile(visited, 99)) if visited.size else 1.0
    fill = max(fig.get_figwidth() - 16.0, 0.0) / 16.0
    sub = gs.subgridspec(1, N_EPOCHS + 2, width_ratios=[1] * N_EPOCHS + [0.05, N_EPOCHS * fill + 1e-6],
                         wspace=0.08)
    im = None
    for k, ep in enumerate(res['epochs']):
        ax = fig.add_subplot(sub[k])
        im = ax.imshow(np.ma.masked_where(ep['_occ'] <= 0, ep['_rate']).T, origin='lower',
                       extent=extent, cmap=RATE_CMAP, vmin=0, vmax=vmax, interpolation='nearest')
        for j, fld in enumerate(res['fields']):
            color = FIELD_COLORS[j % len(FIELD_COLORS)]
            _contour_field(ax, fld['mask'], bin_cm, color)
            com = (ep[f"field{fld['field_number']}_com_x_cm"], ep[f"field{fld['field_number']}_com_y_cm"])
            if not np.isnan(com[0]):
                ax.plot(*com, marker='P', color=color, markeredgecolor='white', markersize=8,
                        linestyle='none', zorder=5)
        r = f"{ep['r_vs_rest']:.2f}" if not np.isnan(ep['r_vs_rest']) else 'n/a'
        ax.set_title(f"{ep['t_start_min']:.1f}–{ep['t_end_min']:.1f} min\n"
                     f"r vs rest = {r},  {ep['n_spikes']} spikes", fontsize=9)
        ax.set_xlabel('x (cm)')
        if k == 0:
            ax.set_ylabel('y (cm)')
        else:
            ax.set_yticklabels([])
    fig.colorbar(im, cax=fig.add_subplot(sub[N_EPOCHS]), label='rate (Hz), top = 99th pct')


def _legend_handles() -> list:
    shades = ([Patch(color=PEAK_DIST_CMAP(0.0), label='through peak bin')]
              + [Patch(color=PEAK_DIST_CMAP(frac), label=f'{frac * PEAK_DIST_MAX_CM:g} cm from peak bin')
                 for frac in (1 / 3, 2 / 3)]
              + [Patch(color=PEAK_DIST_CMAP(1.0), label=f'>= {PEAK_DIST_MAX_CM:g} cm from peak bin')])
    return (shades
            + [Line2D([], [], color='black', marker='s', ls='none', ms=3, label=f'paused > {PAUSE_FLAG_S:g} s'),
               Line2D([], [], color='black', marker='x', ls='none', ms=4, label='cut by session edge'),
               Line2D([], [], color='0.7', lw=1, label=f'+/- {CONTEXT_S:g} s context'),
               Line2D([], [], color=SPIKE_COLOR, lw=1.2, label='spike in pass'),
               Line2D([], [], color=CONTEXT_SPIKE_COLOR, lw=1.2, label='spike in context')])


def make_figure(res: dict, ctx: dict, metrics: dict, summaries: list[dict], bin_cm: float,
                title: str, save_path: str) -> None:
    """One figure per cell; see the module docstring for the rows."""
    fields = res['fields']
    n_f = len(fields)
    max_passes = max(max(len(f['passes']) for f in fields), 1)
    extent = [0, ctx['n_bins_x'] * bin_cm, 0, ctx['n_bins_y'] * bin_cm]

    h_raster = float(np.clip(0.13 * max_passes + 1.5, 4.0, 14.0))
    h_sorted = float(np.clip(0.06 * max_passes + 1.5, 3.0, 6.0))
    heights = [5.0, 2.4, h_raster, h_sorted, 3.4]
    fig_w = max(16.0, 9.0 * n_f)
    fig_h = sum(heights) + 2.8
    fig = Figure(figsize=(fig_w, fig_h))
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(5, 1, height_ratios=heights, hspace=0.42)

    # Row 1 stays at its 1-field size, left-aligned, however wide the figure is.
    gs_top = gs[0].subgridspec(1, 3, width_ratios=[8.0, 8.0, fig_w - 16.0 + 1e-6], wspace=0.15)
    _plot_trajectory(fig.add_subplot(gs_top[0]), fig, res, ctx, bin_cm, extent)
    _plot_session_map(fig.add_subplot(gs_top[1]), fig, res, ctx, bin_cm, extent)

    gs_d = gs[1].subgridspec(1, n_f, wspace=0.18)
    gs_r = gs[2].subgridspec(1, n_f, wspace=0.18)
    gs_s = gs[3].subgridspec(1, n_f, wspace=0.18)
    for k, (fld, summary) in enumerate(zip(fields, summaries)):
        color = FIELD_COLORS[k % len(FIELD_COLORS)]
        _plot_dist_time(fig.add_subplot(gs_d[k]), fld, res, ctx, color)
        passes = fld['passes']
        n = len(passes)
        if not passes:
            fig.add_subplot(gs_r[k]).set_title(f"Field {fld['field_number']}: no passes")
            fig.add_subplot(gs_s[k]).axis('off')
            continue

        path = res['path']
        reach = np.array([max(abs(path[p['_start']] - path[p['_ca']]),
                              abs(path[p['_end']] - path[p['_ca']])) for p in passes])
        xlim = max(10.0, float(np.ceil(np.percentile(reach, 95))) + 5.0)
        n_clip = int((reach > xlim).sum())
        img, grid = _raster_image(passes, res, xlim)

        # Time-ordered raster with its strip and side panels.
        sub = gs_r[k].subgridspec(1, 4, width_ratios=[0.22, 5, 1.0, 1.0], wspace=0.05)
        order = np.arange(n)
        labels = [f"{i}  ({p['t_start_s'] / 60:.1f} min)" for i, p in enumerate(passes, start=1)]
        _plot_strip(fig.add_subplot(sub[0]), passes, order, labels, 'pass # (chronological; start time)')
        ax = fig.add_subplot(sub[1])
        _plot_raster(ax, passes, order, img, grid, fld, res, ctx, xlim,
                     f"Field {fld['field_number']}, time order: {n} passes "
                     f"({summary['n_central']} central, {summary['n_informative']} informative)"
                     + (f", {n_clip} extend past the axis" if n_clip else '') + '\n'
                     f"informative passes with >= 1 spike: {_fmt(summary['hit_rate_informative'])} "
                     f"(Poisson: {_fmt(summary['poisson_hit_rate_informative'])});  "
                     f"var z = {_fmt(summary['overdispersion_var_z'])}\n"
                     f"Spearman rho(z, pass #) = {_fmt(summary['spearman_rho_z_vs_pass'])};  "
                     f"COM shift = {_fmt(summary['com_shift_first_last_cm'], ' cm')};  "
                     f"top pass = {_fmt(summary['top_pass_spike_frac'])} of spikes\n"
                     f"shading: rate predicted without that pass, 0–{fld['peak_rate']:.1f} Hz; white = no path",
                     fontsize=8, loc='left')
        if k == 0:
            ax.legend(handles=_legend_handles(), loc='lower left', fontsize=7, framealpha=0.85)
        _plot_counts(fig.add_subplot(sub[2]), passes)
        _plot_z(fig.add_subplot(sub[3]), passes)

        # Same raster sorted by distance to the peak bin (ties by closest approach),
        # on the x scale of the one above.
        sub = gs_s[k].subgridspec(1, 4, width_ratios=[0.22, 5, 1.0, 1.0], wspace=0.05)
        peak_dist = np.array([p['peak_bin_dist_cm'] for p in passes])
        by_dist = np.lexsort(([p['closest_approach_cm'] for p in passes], peak_dist))
        d_labels = [f'{peak_dist[i]:.0f} cm' for i in by_dist]
        _plot_strip(fig.add_subplot(sub[0]), passes, by_dist, d_labels, 'distance to peak bin')
        _plot_raster(fig.add_subplot(sub[1]), passes, by_dist, img, grid, fld, res, ctx, xlim,
                     f"Field {fld['field_number']} sorted by distance to the peak bin", fontsize=8)

    _plot_filmstrip(fig, gs[4], res, bin_cm, extent)

    fig.suptitle(f'{title}  |  {n_f} field(s), {metrics.get("n_spikes")} spikes (moving)\n'
                 f'Passes merged across exits <= {MERGE_GAP_S:g} s, >= {MIN_PASS_S:g} s long; '
                 f'informative = expected spikes >= {MIN_EXPECTED_SPIKES:g}; '
                 f'Central = through the peak bin; '
                 f'{N_EPOCHS} epochs of equal moving time (plus = field COM)',
                 fontsize=11, y=1 - 0.2 / fig_h, va='top')
    fig.subplots_adjust(left=1.3 / fig_w, right=0.97, top=1 - 1.3 / fig_h, bottom=0.6 / fig_h)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)


# ── Per-cell job ──────────────────────────────────────────────────────────────

def _run_cell(args) -> tuple[list[dict], list[dict], list[dict]]:
    """Open-field lap raster figure for one place cell. Returns (field_rows,
    pass_rows, epoch_rows)."""
    idx, total, dirpath, unit = args
    session  = os.path.relpath(dirpath, place_cell_folder)
    ntt_path = os.path.join(dirpath, unit)
    with _print_lock:
        print(f'[{idx}/{total}]  {session}  |  {unit}')

    csv_path = _find_tracking_csv(dirpath)
    if csv_path is None:
        with _print_lock:
            print(f'  [SKIP] no single tracking file found in {dirpath}')
        return [], [], []

    bin_cm = pc.target_bin_cm
    metrics, ctx = pc.compute_metrics(csv_path, ntt_path, pc.arena_width_cm, bin_cm)
    if not ctx or not ctx['valid_mask'].any():
        with _print_lock:
            print(f'  [SKIP] no valid occupancy for {unit}')
        return [], [], []

    res = analyse_cell(metrics, ctx, bin_cm)
    if res is None:
        with _print_lock:
            print(f'  [SKIP] no place field detected for {unit}')
        return [], [], []

    base = {'session': session, 'unit': unit,
            'session_type': os.path.basename(dirpath).split('_', 1)[-1]}
    summaries = [_field_summary(base, fld, fld['passes'], ctx) for fld in res['fields']]

    ntt_name = os.path.splitext(unit)[0]
    save_path = os.path.join(dirpath, LAP_SUBDIR, f'{ntt_name}_lap_raster_openfield.png')
    make_figure(res, ctx, metrics, summaries, bin_cm, f'{session}  |  {ntt_name}', save_path)
    with _print_lock:
        print(f'  [SAVED] {save_path}')

    pass_rows = [{**base, 'field_number': fld['field_number'], 'pass_number': i,
                  **{k: v for k, v in p.items() if not k.startswith('_')}}
                 for fld in res['fields'] for i, p in enumerate(fld['passes'], start=1)]
    epoch_rows = [{**base, **{k: v for k, v in ep.items() if not k.startswith('_')}}
                  for ep in res['epochs']]
    return summaries, pass_rows, epoch_rows


# ── Batch ─────────────────────────────────────────────────────────────────────

def run_all() -> None:
    """Lap raster for every Open-arena .ntt under place_cell_folder; writes
    output_excel (PerField, PerPass, PerEpoch)."""
    jobs = []
    for animal in pc.ANIMAL_NAMES:
        for dirpath, _, filenames in os.walk(os.path.join(place_cell_folder, animal, 'Open')):
            jobs += [(dirpath, unit) for unit in sorted(filenames) if unit.lower().endswith('.ntt')]
    job_args = [(i, len(jobs), *job) for i, job in enumerate(jobs, start=1)]
    print(f'Found {len(jobs)} Open-arena place cell(s) in {place_cell_folder}\n')

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_run_cell, a): a for a in job_args}
        for future in concurrent.futures.as_completed(futures):
            a = futures[future]
            try:
                results.append((a[0], future.result()))
            except Exception as e:
                print(f'  ERROR in {a[2]} | {a[3]}: {e}')
    results.sort(key=lambda r: r[0])

    field_rows = [f for _, r in results for f in r[0]]
    pass_rows  = [p for _, r in results for p in r[1]]
    epoch_rows = [e for _, r in results for e in r[2]]
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        pd.DataFrame(field_rows).to_excel(writer, sheet_name='PerField', index=False)
        pd.DataFrame(pass_rows).to_excel(writer,  sheet_name='PerPass',  index=False)
        pd.DataFrame(epoch_rows).to_excel(writer, sheet_name='PerEpoch', index=False)

    n_cells = len({(f['session'], f['unit']) for f in field_rows})
    print(f'\nDone. {n_cells}/{len(jobs)} cell(s) plotted.')
    print(f'Results saved to {output_excel}')


if __name__ == "__main__":
    run_all()
