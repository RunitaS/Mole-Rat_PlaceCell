# -*- coding: utf-8 -*-
"""
Lap rasters for place cells in every arena (Open, Linear, Circle) – V3

One script for the open field (was LapRaster_OpenField_v3_AdptBinFlds) and the
1-D tracks (was LapRaster_Linearized_v2), with one pass method, one shading and
the same figure rows in every arena. Every rate comes from one map type,
RATEMAP_TYPE (set right below the imports): Skaggs adaptive-binned or
Gaussian-smoothed.

For every .ntt under <place_cell_folder>\\<animal>\\<arena> (arena in ARENAS):
  1. Rebuild trajectory, spike frames and bin grid with
     PlaceCell_Main_v22.compute_metrics (same cleaning, smoothing, immobility
     filter and spike assignment as the run that detected the fields).
  2. Place fields on the RATEMAP_TYPE map. A field must span >= 9 contiguous
     bins and hold >= MIN_FIELD_SPIKES moving spikes; the fields left are
     numbered 1, 2, ...
       Open     2-D map, 8-connected bins (detect_place_fields_threshold_2d)
       Linear   2-D map as Open; each field becomes the range of long-axis bins
                it spans, its peak bin the long-axis bin of the 2-D peak
       Circle   angular map, contiguous angular bins
                (detect_place_fields_threshold_circular)
  3. Track position (1-D tracks only), as in LapRaster_Linearized_v2:
       Linear 0 / 180   position = x (cm)
       Linear 90 / 270  vertical track, rotated 90 deg counter-clockwise,
                        (x, y) -> (L - y, x), L = the session's y extent; the
                        2-D panels are drawn rotated the same way
       Circle           arc length along the ring, theta / bin_width_rad * bin_cm
     A Circle field across the theta = 0 seam is drawn with the seam moved to
     the opposite side of the ring (tick labels still give absolute position).
  4. Passes (all arenas): one contiguous run of moving-only frames inside the
     field (Open: the field's 2-D bins; Linear / Circle: the field's track bins),
     entry -> exit through any bin. Then:
       - runs separated by <= MERGE_GAP_S of real time are merged into one pass;
       - passes shorter than MIN_PASS_S are dropped;
       - passes cut by the session start / end are kept and flagged;
       - a pause inside a pass does not end it (those frames are not in the
         moving-only sequence); passes with a gap > PAUSE_FLAG_S are flagged.
  5. Per pass, from a ratemap built WITHOUT that pass (leave-one-out, so a pass
     does not predict itself):
       expected_spikes   sum over the pass's frames of rate x dwell
       z                 (observed - expected) / sqrt(expected)
     z is only read on informative passes, expected_spikes >= MIN_EXPECTED_SPIKES.
     The leave-one-out map is RATEMAP_TYPE on
       Open             the 2-D bins
       Linear, Circle   the 1-D track bins. Skaggs: a window of +/- w bins grows
                        until r_cm * occ * sqrt(spikes) >= pc.SKAGGS_ALPHA,
                        r_cm = (w + 0.5) * bin_cm, as in PlaceCell_Main_v22's
                        Circle map (wraps on the Circle only). Gaussian: 1-D
                        kernel, sigma = pc.GAUSSIAN_SIGMA_CM.
     Category:
       Central     the pass went through the field's peak bin (Open: a frame in
                   the peak bin; Linear / Circle: the peak bin lies between the
                   pass's lowest and highest track position)
       Peripheral  every other pass
     peak_bin_dist_cm is the centre-to-centre distance from the pass's nearest
     bin to the peak bin, so it is 0 exactly for Central passes.
     Linear / Circle passes also get v2's traversal label:
       Full traversal        entered at one end of the field and left at the
                             other, or covered both end bins of the field
       Partial, reached peak not full, Central
       Partial, missed peak  not full, Peripheral
  6. Epochs: N_EPOCHS spans of equal moving time (not equal clock time). Each
     epoch's RATEMAP_TYPE map is correlated with the same map of the rest of the
     session on the bins visited in both, and each field gets a centre of mass
     (COM): the rate-weighted mean position of the visited bins of the field,
     grown by COM_DILATE_BINS bins, that reach >= COM_RATE_FRAC of that epoch's
     highest rate there. Epochs with less than MIN_COM_DWELL_S of dwell or no
     spikes in that area have no COM.
       Open, Linear     2-D map (Linear 90 / 270: COM in the rotated frame, so
                        com_x_cm is track position); COM shift = distance in cm
       Circle           angular map, drawn back on the ring; COM = circular
                        rate-weighted mean, as arc position (com_pos_cm); COM
                        shift = arc distance in cm
     Open-field reference from stable Poisson cells simulated from each real
     cell's session map on its own tracking (one per cell, 271 cells), first ->
     last epoch COM shift: Skaggs alpha = 200 median 4.5 cm, 95th percentile
     12.9 cm; Gaussian median 6.0 cm, 95th percentile 18.8 cm. Pooled reference,
     not a per-cell null; no such reference exists yet for the 1-D tracks.

Figure (one per cell, saved in LAP_SUBDIR next to its .ntt):
  row 1  trajectory + spikes coloured by time; session RATEMAP_TYPE map (the map
         the fields are drawn on). Linear: one above the other, rotated to
         horizontal for 90 / 270. Circle: the angular map drawn on the ring.
         No frame, x / y ticks or axis labels; titles and colour bars stay.
  row 2  per field: lap raster in time order (pass # on y, top = first;
         shading = leave-one-out rate along the pass; spikes as red ticks;
         +/- CONTEXT_S s of path in gray, spikes on it in dark blue), title with
         the field's pass counts and summary statistics; raster key to the
         right of the first field.
           Open             x = signed path length from the pass's closest
                            approach to the field peak (0 = closest approach)
           Linear, Circle   x = track position; dashed = peak bin, dotted =
                            field edges
         A dark row without ticks is a true miss; a pale row without ticks is
         only a pass through the edge. The strip left of each raster is the
         pass's distance to the peak bin (magma: darkest = through the peak
         bin); square = pause, x = session edge, o = full traversal (1-D).
  row 3  RATEMAP_TYPE map per epoch on one colour scale, fields outlined, each
         field's COM marked; unvisited bins blank. Side by side in all arenas
         (Linear across the full width). No 1-D (linearized) epoch maps. No
         frame, x / y ticks or axis labels.

v3 (figure only; analysis and tables as v2): dropped the distance-to-peak vs
time panels (Open), the 1-D session map and track position vs time (Linear /
Circle), the observed / expected spike-count and z panels beside each raster,
and the raster sorted by distance to the peak bin. Both row-1 maps lost their
frame and x / y scales.
v2: epoch maps without frame and scales; imports PlaceCell_Main_v22 (was v21).
v22 only adds the within-field firing rate to its own batch output (_run_job);
compute_metrics, the ratemaps and the field detection used here are unchanged.

Tables (output_excel; one row set for all arenas, arena_type column):
  PerPass   one row per pass: times, flags, category, distance to the peak bin,
            closest approach, observed / expected spikes, z; Linear / Circle
            also traversal, entry / exit side, direction, entry / exit position
  PerField  per field: pass counts, hit rate vs the Poisson expectation,
            overdispersion (variance of z; the simulated open-field Poisson cells
            above give a median of 1.1 on the Skaggs map and 1.3 on the Gaussian
            map, not 1, because the leave-one-out map is itself noisy), largest
            share of the field's in-pass spikes from one pass, Spearman rho of z
            vs pass number (rate drift), COM shift between the first and last
            usable epoch and its largest pairwise range (location drift), lap
            stability on this script's passes and with TemporalStabilityMap_v6's
            rule (no merging, session-edge passes dropped). Linear / Circle also
            field extent, counts per traversal label and lap stability on full
            traversals. peak_rate_hz is from the map that shades the rasters
            (Linear: the 1-D map at the peak bin).
  PerEpoch  per epoch: time range, correlation with the rest of the session,
            rate, dwell and COM of each field

Analysis settings (bin size, smoothing, filters, field thresholds, COORD_UNITS, ...)
come from PlaceCell_Main_v22; only RATEMAP_TYPE, the Skaggs alpha, the minimum
field size and the directories below are set here.
"""

import os
import re
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
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib import colormaps
from matplotlib.patches import Patch

import PlaceCell_Main_v22 as pc


# ── Rate map (edit this per run) ──────────────────────────────────────────────

# The one map type behind everything here: field detection, session map,
# leave-one-out expected spikes, epoch maps, their correlations and COMs.
#   'Skaggs_adaptive_binned' : adaptive disc (1-D: window) grows until
#                              r_cm * occ_s * sqrt(spikes) >= pc.SKAGGS_ALPHA
#   'gaussian_smooth'        : Gaussian-smoothed raw rate, sigma = pc.GAUSSIAN_SIGMA_CM
RATEMAP_TYPE = 'Skaggs_adaptive_binned'
if RATEMAP_TYPE not in ('Skaggs_adaptive_binned', 'gaussian_smooth'):
    raise ValueError(f"RATEMAP_TYPE must be 'Skaggs_adaptive_binned' or 'gaussian_smooth', "
                     f"got {RATEMAP_TYPE!r}")

# PlaceCell_Main_v22 defaults to a different map per metric and SKAGGS_ALPHA = 50;
# compute_metrics and _build_ratemap_circular build the field map and the mean
# rate of the field threshold from these.
pc.SI_RATEMAP_TYPE                  = RATEMAP_TYPE
pc.COHERENCE_STABILITY_RATEMAP_TYPE = RATEMAP_TYPE
pc.FIRING_RATE_RATEMAP_TYPE         = RATEMAP_TYPE
pc.PLACE_FIELD_RATEMAP_TYPE         = RATEMAP_TYPE
pc.SKAGGS_ALPHA                     = 200.0   # cm * s * sqrt(spike) (Alpha200 run)
pc.MIN_FIELD_SIZE_BINS              = 9       # a field must span >= this many contiguous bins


# ── Directories (edit these per run) ──────────────────────────────────────────

# Place-cell folder (Output_PlaceTrue of a PlaceCell_Main_v22 run): every .ntt
# under <animal>\<arena> in it is analysed, for each arena in ARENAS.
place_cell_folder = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final'

# Per-pass / per-field / per-epoch results table, all arenas.
output_excel = rf'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final\Results\LapRaster_AllArenas_{RATEMAP_TYPE}.xlsx'

# Figures go in this subfolder of each session folder (next to the .ntt).
LAP_SUBDIR = f'lap_raster_{RATEMAP_TYPE}'

# ── Settings ──────────────────────────────────────────────────────────────────

ARENAS              = ('Open', 'Linear', 'Circle')   # arenas analysed
LINEAR_ROTATE_TYPES = ('90', '270')  # Linear session types recorded on the vertical track

MIN_FIELD_SPIKES    = 10    # a field must hold >= this many moving spikes across its bins
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
RASTER_X_STEP_CM    = 1.0   # x resolution of the raster shading (cm)

FIELD_COLORS = ['tab:green', 'tab:purple', 'tab:cyan', 'tab:olive', 'tab:brown', 'tab:pink']
SPIKE_COLOR         = 'red'
CONTEXT_SPIKE_COLOR = 'navy'
TIME_CMAP           = 'viridis'

# Session RATEMAP_TYPE map (row 1) and the expected (leave-one-out) rate
# shading in the rasters: one viridis scale; white = no data / no path.
RATE_CMAP = colormaps['viridis'].copy()
RATE_CMAP.set_bad('white')
RASTER_SHADE_ALPHA = 0.45   # opacity of the raster shading, so the spike ticks on top stay visible
# Epoch rate maps (bottom row); white = unvisited.
EPOCH_MAP_CMAP = colormaps['jet'].copy()
EPOCH_MAP_CMAP.set_bad('white')
# Linear filmstrip columns: N_EPOCHS maps, colourbar, room for its label.
LINEAR_FILM_RATIOS = [1] * N_EPOCHS + [0.03, 0.35]

# Distance to the peak bin in the strip; dropped magma's palest end so that the
# farthest passes still show against the white figure background.
PEAK_DIST_CMAP = LinearSegmentedColormap.from_list(
    'peak_dist_magma', colormaps['magma'](np.linspace(0.0, 0.92, 256)))

MAX_WORKERS = pc.MAX_WORKERS

_print_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_tracking_csv(dirpath: str) -> str | None:
    """Same tracking-file selection as PlaceCell_Main_v22's batch scan."""
    tracking = [f for f in os.listdir(dirpath) if f.lower().endswith(('.csv', '.xlsx'))]
    if pc.COORD_UNITS == 'cm':
        tracking = [f for f in tracking if f.lower().endswith('_cm.csv')]
    else:
        tracking = [f for f in tracking if not f.lower().endswith('_cm.csv')]
    return os.path.join(dirpath, tracking[0]) if len(tracking) == 1 else None


def _arena_and_session(dirpath: str) -> tuple[str | None, str | None]:
    """('Open' | 'Linear' | 'Circle' | None, session type) from the session
    folder path, e.g. '...\\Linear\\Day10\\4_90' -> ('Linear', '90'). Session
    type: Linear the trailing angle, Circle the folder name, Open the folder
    name after its first '_'."""
    parts = [p for p in re.split(r'[\\/]+', dirpath) if p]
    lower = [p.lower() for p in parts]
    label = parts[-1] if parts else ''
    if 'linear' in lower:
        m = re.search(r'_(270|180|90|0)$', label)
        return 'Linear', (m.group(1) if m else None)
    if 'circle' in lower:
        return 'Circle', label.lower()
    if 'open' in lower:
        return 'Open', label.split('_', 1)[-1]
    return None, None


def _map_label() -> str:
    """RATEMAP_TYPE for figure titles."""
    if RATEMAP_TYPE == 'Skaggs_adaptive_binned':
        return f'Skaggs adaptive, alpha = {pc.SKAGGS_ALPHA:g}'
    return f'Gaussian, sigma = {pc.GAUSSIAN_SIGMA_CM:g} cm'


def _fmt(value, suffix: str = '') -> str:
    """Number for a figure title; 'n/a' for NaN (e.g. no informative passes)."""
    return 'n/a' if pd.isna(value) else f'{value}{suffix}'


def _rnd(value: float, nd: int = 2) -> float:
    """Rounded value for a table; NaN stays NaN."""
    return round(float(value), nd) if np.isfinite(value) else np.nan


def _runs(bins: np.ndarray, n_bins: int, circular: bool) -> list[tuple[int, int]]:
    """Contiguous runs (first, last) of sorted bin indices; on a circular axis
    a run through bin n_bins-1 -> 0 is joined into one (first > last)."""
    bins = np.unique(bins)
    cut  = np.flatnonzero(np.diff(bins) > 1) + 1
    runs = [(int(r[0]), int(r[-1])) for r in np.split(bins, cut)]
    if circular and len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == n_bins - 1:
        runs = [(runs[-1][0], runs[0][1])] + runs[1:-1]
    return runs


def _segments(x: np.ndarray, y: np.ndarray, period: float | None) -> list[np.ndarray]:
    """Polyline -> list of (n, 2) segments, broken where x jumps by more than
    half the period (Circle track crossing the display seam)."""
    if x.size == 0:
        return []
    pts = np.column_stack([x, y])
    if period is None or x.size < 2:
        return [pts]
    cut = np.flatnonzero(np.abs(np.diff(x)) > 0.5 * period) + 1
    return [s for s in np.split(pts, cut) if len(s) > 1]


def _contour_field(ax, field_mask: np.ndarray, bin_cm: float, color: str,
                   upsample: int = 8) -> None:
    """Outline a 2-D field mask along bin edges on cm axes (as in
    TemporalStabilityMap_v6_MapPassStability._contour_field)."""
    up = np.kron(np.pad(field_mask.astype(float), 1), np.ones((upsample, upsample)))
    xc = ((np.arange(up.shape[0]) + 0.5) / upsample - 1) * bin_cm
    yc = ((np.arange(up.shape[1]) + 0.5) / upsample - 1) * bin_cm
    ax.contour(xc, yc, up.T, levels=[0.5], colors=color, linewidths=1.5)


# ── Display geometry ──────────────────────────────────────────────────────────
# geo: how a cell's arena is drawn in 2-D. kind 'open' | 'linear' | 'circle';
# x_all / y_all / x / y the trajectory (all / moving frames) in display cm;
# extent; rotated (Linear 90 / 270) with L the session's y extent; ring =
# (cx, cy, r_min, r_max, bin_width_rad) on the Circle; session_rate /
# session_show the session RATEMAP_TYPE map (2-D, unrotated; Circle: angular).

def _disp2d(geo: dict, arr: np.ndarray) -> np.ndarray:
    """2-D map in display orientation: bin (bx, by) -> (n_by-1-by, bx) for
    Linear 90 / 270 (90 deg counter-clockwise), unchanged otherwise."""
    return np.rot90(arr) if geo['rotated'] else arr


def _disp_xy(geo: dict, x, y) -> tuple:
    """Session cm -> display cm ((x, y) -> (L - y, x) for Linear 90 / 270)."""
    return (geo['L'] - y, x) if geo['rotated'] else (x, y)


def _ring_xy(geo: dict, theta, radius=None) -> tuple:
    """Point on the Circle ring at angle theta (rad), mid-track by default."""
    cx, cy, r0, r1, _ = geo['ring']
    r = 0.5 * (r0 + r1) if radius is None else radius
    return cx + r * np.cos(theta), cy + r * np.sin(theta)


def _ring_outline(ax, geo: dict, bins: np.ndarray, color: str, lw: float = 1.5) -> None:
    """Outline the ring sector of each run of angular field bins."""
    cx, cy, r0, r1, bw = geo['ring']
    n = geo['n_lin']
    for b0, b1 in _runs(bins, n, circular=True):
        b1u = b1 if b1 >= b0 else b1 + n
        th = np.linspace(b0 * bw, (b1u + 1) * bw, 60)
        xs = np.r_[r1 * np.cos(th), r0 * np.cos(th[::-1]), r1 * np.cos(th[:1])] + cx
        ys = np.r_[r1 * np.sin(th), r0 * np.sin(th[::-1]), r1 * np.sin(th[:1])] + cy
        ax.plot(xs, ys, color=color, lw=lw)


def _outline_fields(ax, geo: dict, fields: list[dict], bin_cm: float) -> None:
    for k, fld in enumerate(fields):
        color = FIELD_COLORS[k % len(FIELD_COLORS)]
        if geo['kind'] == 'circle':
            _ring_outline(ax, geo, fld['bins'], color)
        else:
            _contour_field(ax, fld['mask_disp'], bin_cm, color)


def _set_map_axes(ax, geo: dict) -> None:
    ext = geo['extent']
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect('equal', adjustable='box')


def _draw_map(ax, geo: dict, rate: np.ndarray, show: np.ndarray, vmax: float | None = None,
              cmap=RATE_CMAP):
    """RATEMAP_TYPE map on its arena: 2-D bins (rotated for Linear 90 / 270), or
    the angular bins drawn back on the Circle ring. Bins outside `show` blank."""
    if geo['kind'] == 'circle':
        cx, cy, r0, r1, bw = geo['ring']
        th = np.arange(rate.size + 1) * bw
        R = np.array([r0, r1])
        X = cx + np.outer(R, np.cos(th))
        Y = cy + np.outer(R, np.sin(th))
        vals = np.ma.masked_where(~(show & np.isfinite(rate)), np.nan_to_num(rate))
        im = ax.pcolormesh(X, Y, vals[None, :], cmap=cmap, vmin=0, vmax=vmax, shading='flat')
    else:
        r = _disp2d(geo, np.where(show, rate, np.nan))
        im = ax.imshow(np.ma.masked_invalid(r).T, origin='lower', extent=geo['extent'],
                       cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
    _set_map_axes(ax, geo)
    return im


# ── Rate maps ─────────────────────────────────────────────────────────────────

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


def _adaptive_rate(spk: np.ndarray, occ: np.ndarray, valid: np.ndarray, bin_cm: float) -> np.ndarray:
    """Skaggs adaptive rate, as pc._skaggs_adaptive_binned_ratemap, but evaluated
    on every bin (NaN where the final disc holds < pc.ADAPTIVE_MIN_OCC_S) instead
    of only on `valid`, for the same reason as _smooth_rate."""
    spk_disc, occ_disc, disc_r_bins = pc._adaptive_disc_sums(spk, occ, valid)
    r_cm = (disc_r_bins + 0.5) * bin_cm
    satisfied = r_cm[:, None] * occ_disc * np.sqrt(spk_disc) >= pc.SKAGGS_ALPHA
    k = np.where(satisfied.any(axis=0), satisfied.argmax(axis=0), pc.ADAPTIVE_MAX_RSQ - 1)
    cols = np.arange(occ.size)
    spk_fin = spk_disc[k, cols].reshape(occ.shape)
    occ_fin = occ_disc[k, cols].reshape(occ.shape)
    out = np.full_like(occ, np.nan)
    ok = occ_fin >= pc.ADAPTIVE_MIN_OCC_S
    out[ok] = spk_fin[ok] / occ_fin[ok]
    return out


def _rate_map(spk: np.ndarray, occ: np.ndarray, valid: np.ndarray, bin_cm: float) -> np.ndarray:
    """2-D RATEMAP_TYPE map of any subset of the session (one pass left out, an
    epoch, the rest of the session); only `valid` bins contribute."""
    if RATEMAP_TYPE == 'Skaggs_adaptive_binned':
        return _adaptive_rate(spk, occ, valid, bin_cm)
    return _smooth_rate(spk, occ, valid, pc._gaussian_kernel(pc.GAUSSIAN_SIGMA_CM / bin_cm))


def _window_sums_1d(spk: np.ndarray, occ: np.ndarray, valid: np.ndarray,
                    circular: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spikes and occupancy in the window of +/- w bins around every track bin,
    w = 0 .. max_w (rows), as pc._adaptive_ratemap_circular. Wraps on the Circle;
    on the Linear track the window is cut at the track ends. Returns
    (spk_win, occ_win, widths)."""
    n = occ.size
    max_w = int(np.floor(np.sqrt(pc.ADAPTIVE_MAX_RSQ - 1)))
    if circular:
        max_w = min(max_w, (n - 1) // 2)
    widths = np.arange(max_w + 1)
    # Offsets 0, +1, -1, +2, -2, ...: cumsum row 2w is the window of half-width w.
    shifts = np.concatenate([[0], np.stack([widths[1:], -widths[1:]], axis=1).ravel()]).astype(int)
    idx = np.arange(n)[None, :] + shifts[:, None]
    s = np.where(valid, spk, 0.0)
    o = np.where(valid, occ, 0.0)
    if circular:
        s_g, o_g = s[idx % n], o[idx % n]
    else:
        inside = (idx >= 0) & (idx < n)
        idx_c = np.clip(idx, 0, n - 1)
        s_g = np.where(inside, s[idx_c], 0.0)
        o_g = np.where(inside, o[idx_c], 0.0)
    return np.cumsum(s_g, axis=0)[0::2], np.cumsum(o_g, axis=0)[0::2], widths


def _rate_map_1d(spk: np.ndarray, occ: np.ndarray, valid: np.ndarray, bin_cm: float,
                 circular: bool) -> np.ndarray:
    """1-D RATEMAP_TYPE map on the track bins, evaluated on every bin (NaN where
    there is no estimate) like the 2-D maps above; only `valid` bins contribute."""
    n = occ.size
    out = np.full(n, np.nan)
    if RATEMAP_TYPE == 'Skaggs_adaptive_binned':
        spk_w, occ_w, widths = _window_sums_1d(spk, occ, valid, circular)
        r_cm = (widths + 0.5) * bin_cm
        satisfied = r_cm[:, None] * occ_w * np.sqrt(spk_w) >= pc.SKAGGS_ALPHA
        k = np.where(satisfied.any(axis=0), satisfied.argmax(axis=0), widths[-1])
        cols = np.arange(n)
        spk_f, occ_f = spk_w[k, cols], occ_w[k, cols]
        ok = occ_f >= pc.ADAPTIVE_MIN_OCC_S
        out[ok] = spk_f[ok] / occ_f[ok]
        return out
    kernel = pc._gaussian_kernel_1d(pc.GAUSSIAN_SIGMA_CM / bin_cm)
    kr = kernel.size // 2
    raw = np.zeros(n)
    raw[valid] = spk[valid] / occ[valid]
    w = valid.astype(float)
    if circular:
        num = np.convolve(np.pad(raw, kr, mode='wrap'), kernel, mode='valid')
        den = np.convolve(np.pad(w, kr, mode='wrap'), kernel, mode='valid')
    else:
        num = np.convolve(raw, kernel, mode='same')
        den = np.convolve(w, kernel, mode='same')
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


# ── Fields ────────────────────────────────────────────────────────────────────

def _fields_open(metrics: dict, ctx: dict, bin_cm: float) -> tuple[list[dict], dict]:
    """Open-field place fields with >= MIN_FIELD_SPIKES spikes as 2-D masks, with
    peak bin / position / rate from the field map; and the display geometry."""
    geo = {'kind': 'open', 'rotated': False, 'ring': None,
           'x_all': ctx['x_cm_all'], 'y_all': ctx['y_cm_all'], 'x': ctx['x_cm'], 'y': ctx['y_cm'],
           'extent': [0, ctx['n_bins_x'] * bin_cm, 0, ctx['n_bins_y'] * bin_cm],
           'session_rate': ctx['fr_field'], 'session_show': ctx['field_valid']}
    found, _ = pc.detect_place_fields_threshold_2d(metrics, ctx, bin_cm)
    out = []
    for f in found:
        mask = pc._field_mask_from_bin_coords(f['bin_coords'], ctx['n_bins_x'], ctx['n_bins_y'])
        if ctx['spike_map'][mask].sum() < MIN_FIELD_SPIKES:
            continue
        peak_bin = (int(f['peak_bin_x']), int(f['peak_bin_y']))
        peak_xy = ((peak_bin[0] + 0.5) * bin_cm, (peak_bin[1] + 0.5) * bin_cm)
        out.append({'field_number': len(out) + 1,
                    'mask': mask, 'mask_disp': mask,
                    'peak_bin': peak_bin, 'peak_xy': peak_xy, 'peak_xy_disp': peak_xy,
                    'peak_rate': float(ctx['fr_field'][peak_bin]),
                    'shade_max': float(ctx['fr_field'][peak_bin]),
                    'in_field': mask[ctx['beh_bx'], ctx['beh_by']]})
    return out, geo


def _linearize(metrics: dict, ctx: dict, arena: str, session_type: str | None,
               bin_cm: float) -> dict | None:
    """Track coordinates, 1-D maps, fields and display geometry of a Linear /
    Circle session; None if no place field is detected. Keys:
      pos, lin_bin          per moving frame: track position (cm) and track bin
      n_lin, track_len      number of track bins, track length (cm)
      circular, rotated     Circle track; Linear 90 / 270 rotated to horizontal
      occ, spk, valid       1-D occupancy (s), spike counts, valid bins
      rate                  1-D RATEMAP_TYPE session map
      fields                [{field_number, bins, peak_bin, peak_rate, ...}]
      geo                   display geometry (see _disp2d)
    """
    n_bx, n_by = ctx['n_bins_x'], ctx['n_bins_y']
    circular = arena == 'Circle'
    lin: dict = {'circular': circular, 'rotated': False}
    spike_frame = ctx['spike_frame']

    if circular:
        cc = pc._build_ratemap_circular(ctx, bin_cm)
        found, _ = pc.detect_place_fields_threshold_circular(cc, bin_cm)
        theta = np.mod(np.arctan2(ctx['y_cm'] - cc['cy'], ctx['x_cm'] - cc['cx']), 2.0 * np.pi)
        n = cc['n_bins_theta']
        lin.update(pos=theta / cc['bin_width_rad'] * bin_cm, lin_bin=cc['beh_bt'], n_lin=n,
                   occ=cc['occ_map'], spk=cc['spike_map'], valid=cc['valid_mask'])
        geo = {'kind': 'circle', 'rotated': False, 'n_lin': n,
               'ring': (cc['cx'], cc['cy'], cc['r_min'], cc['r_max'], cc['bin_width_rad']),
               'x_all': ctx['x_cm_all'], 'y_all': ctx['y_cm_all'], 'x': ctx['x_cm'], 'y': ctx['y_cm'],
               'extent': [0, n_bx * bin_cm, 0, n_by * bin_cm]}
        raw_fields = [{'bins': np.array(sorted(int(b) for b in f['bin_coords'].split(';'))),
                       'peak_bin': int(f['peak_bin_theta'])} for f in found]
    else:
        found, _ = pc.detect_place_fields_threshold_2d(metrics, ctx, bin_cm)
        rotate = session_type in LINEAR_ROTATE_TYPES
        if (n_by > n_bx) != rotate:
            with _print_lock:
                print(f'  [WARN] Linear session type {session_type!r} but track extent is '
                      f'{n_bx} x {n_by} bins; rotation = {rotate}')
        L = n_by * bin_cm
        geo = {'kind': 'linear', 'rotated': rotate, 'L': L, 'ring': None,
               'session_rate': ctx['fr_field'], 'session_show': ctx['field_valid']}
        if rotate:
            # 90 deg counter-clockwise: (x, y) -> (L - y, x); bin (bx, by) -> (n_by-1-by, bx).
            n = n_by
            lin_bin = n_by - 1 - ctx['beh_by']
            long_idx = lambda bx, by: n_by - 1 - by
            geo['extent'] = [0, n_by * bin_cm, 0, n_bx * bin_cm]
        else:
            n = n_bx
            lin_bin = ctx['beh_bx']
            long_idx = lambda bx, by: bx
            geo['extent'] = [0, n_bx * bin_cm, 0, n_by * bin_cm]
        geo['x_all'], geo['y_all'] = _disp_xy(geo, ctx['x_cm_all'], ctx['y_cm_all'])
        geo['x'], geo['y'] = _disp_xy(geo, ctx['x_cm'], ctx['y_cm'])
        occ = np.bincount(lin_bin, weights=ctx['dt_frames'], minlength=n).astype(float)
        spk = np.bincount(lin_bin[spike_frame], minlength=n).astype(float)
        lin.update(rotated=rotate, pos=geo['x'], lin_bin=lin_bin, n_lin=n, occ=occ, spk=spk,
                   valid=pc._valid_bin_mask(occ, pc._bin_visit_counts(lin_bin, n)))
        raw_fields = []
        for f in found:
            mask = pc._field_mask_from_bin_coords(f['bin_coords'], n_bx, n_by)
            bxs, bys = np.nonzero(mask)
            pbx, pby = int(f['peak_bin_x']), int(f['peak_bin_y'])
            raw_fields.append({'bins': np.unique(long_idx(bxs, bys)),
                               'peak_bin': int(long_idx(pbx, pby)),
                               'mask': mask, 'mask_disp': _disp2d(geo, mask),
                               'peak_xy_disp': _disp_xy(geo, (pbx + 0.5) * bin_cm, (pby + 0.5) * bin_cm)})

    lin['track_len'] = lin['n_lin'] * bin_cm
    lin['rate'] = _rate_map_1d(lin['spk'], lin['occ'], lin['valid'], bin_cm, circular)
    if circular:
        geo['session_rate'] = lin['rate']
        geo['session_show'] = lin['valid'] & np.isfinite(lin['rate'])

    fields = []
    for rf in raw_fields:
        n_spk = ctx['spike_map'][rf['mask']].sum() if 'mask' in rf else lin['spk'][rf['bins']].sum()
        if n_spk < MIN_FIELD_SPIKES:
            continue
        r = lin['rate']
        peak_rate = (float(r[rf['peak_bin']]) if np.isfinite(r[rf['peak_bin']]) else
                     float(np.nanmax(r[rf['bins']])) if np.isfinite(r[rf['bins']]).any() else 0.0)
        # Raster colour scale: the track map's highest rate in the field (Linear:
        # the 2-D peak's long-axis bin need not hold the 1-D maximum).
        shade_max = float(np.nanmax(r[rf['bins']])) if np.isfinite(r[rf['bins']]).any() else 0.0
        fld = {**rf, 'field_number': len(fields) + 1, 'peak_rate': peak_rate,
               'shade_max': max(shade_max, peak_rate),
               'in_field': np.isin(lin['lin_bin'], rf['bins'])}
        if circular:
            fld['peak_xy_disp'] = _ring_xy(geo, (rf['peak_bin'] + 0.5) * geo['ring'][4])
        fld['disp'] = _field_display(fld, lin, bin_cm)
        fields.append(fld)
    if not fields:
        return None
    lin['fields'] = fields
    lin['geo'] = geo
    return lin


def _field_display(fld: dict, lin: dict, bin_cm: float) -> dict:
    """Display frame for one track field. Linear: absolute position. Circle:
    absolute position too, unless the field straddles the theta = 0 seam --
    then the seam is moved to the opposite side of the ring (shift s), so the
    field is contiguous; pos_disp = (pos - s) mod C + s."""
    C, bins = lin['track_len'], fld['bins']
    s = 0.0
    if lin['circular'] and 0 in bins and lin['n_lin'] - 1 in bins:
        ang = 2.0 * np.pi * (bins + 0.5) / lin['n_lin']
        centre = (np.angle(np.mean(np.exp(1j * ang))) % (2.0 * np.pi)) / (2.0 * np.pi) * C
        s = centre - 0.5 * C
    to_disp = (lambda p: np.mod(p - s, C) + s) if lin['circular'] else (lambda p: p)
    centres = to_disp((bins + 0.5) * bin_cm)
    return {'shift': s, 'to_disp': to_disp,
            'lo': float(centres.min() - 0.5 * bin_cm), 'hi': float(centres.max() + 0.5 * bin_cm),
            'end_bins': (int(bins[np.argmin(centres)]), int(bins[np.argmax(centres)])),
            'peak': float(to_disp((fld['peak_bin'] + 0.5) * bin_cm))}


def _abs_cm(v: float, lin: dict) -> float:
    """Displayed position -> absolute track position (wraps only on the Circle)."""
    return float(np.mod(v, lin['track_len'])) if lin['circular'] else float(v)


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


def _pass_common(a: int, b: int, n_frag: int, dur: float, last: int, t_s: np.ndarray,
                 observed: int, expected: float) -> dict:
    """Columns every arena shares: times, flags, observed / expected spikes, z."""
    z = (observed - expected) / np.sqrt(expected) if expected > 0 else np.nan
    pause = float(np.diff(t_s[a:b + 1]).max()) if b > a else 0.0
    return {'t_start_s': round(float(t_s[a]), 3), 't_end_s': round(float(t_s[b]), 3),
            'duration_s': round(dur, 3),
            'n_fragments': n_frag,
            'session_edge': bool(a == 0 or b == last),
            'max_pause_s': round(pause, 3),
            'paused': bool(pause > PAUSE_FLAG_S),
            'n_spikes': observed,
            'expected_spikes': round(expected, 3),
            'z': round(float(z), 3) if not np.isnan(z) else np.nan,
            'informative': bool(expected >= MIN_EXPECTED_SPIKES)}


def _passes_open(fld: dict, ctx: dict, t_s: np.ndarray, path: np.ndarray, bin_cm: float) -> list[dict]:
    """Every pass through one open-field field, in time order, with its
    leave-one-out expected spikes from the 2-D map. Keys starting with '_' are
    for the figure."""
    bx, by, dt = ctx['beh_bx'], ctx['beh_by'], ctx['dt_frames']
    x, y = ctx['x_cm'], ctx['y_cm']
    spk_sorted = np.sort(ctx['spike_frame'])
    in_field = fld['in_field']
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
        rate = _rate_map(spk_loo, occ_loo, ctx['valid_mask'] & (occ_loo > 1e-9), bin_cm)
        rate_frames = rate[bx[a:b + 1], by[a:b + 1]]
        expected = float(np.nansum(rate_frames * dt[a:b + 1]))

        ca = a + int(np.argmin(dist[a:b + 1]))             # closest approach to the peak
        reached = float(ctx['fr_field'][bx[a:b + 1], by[a:b + 1]].max()) / fld['peak_rate']
        peak_bin_dist = bin_cm * float(np.hypot(bx[a:b + 1] - fld['peak_bin'][0],
                                                by[a:b + 1] - fld['peak_bin'][1]).min())
        common = _pass_common(a, b, n_frag, dur, last, t_s, int(pass_spk.size), expected)
        rows.append({'_start': a, '_end': b, '_ca': ca, '_rate': rate_frames,
                     **{k: common[k] for k in ('t_start_s', 't_end_s', 'duration_s', 'n_fragments',
                                               'session_edge', 'max_pause_s', 'paused')},
                     'category': 'Central' if peak_bin_dist == 0 else 'Peripheral',
                     'peak_bin_dist_cm': round(peak_bin_dist, 2),
                     'reached_rate_frac': round(reached, 3),
                     'closest_approach_cm': round(float(dist[ca]), 2),
                     'path_cm': round(float(path[b] - path[a]), 2),
                     'net_displacement_cm': round(float(np.hypot(x[b] - x[a], y[b] - y[a])), 2),
                     **{k: common[k] for k in ('n_spikes', 'expected_spikes', 'z', 'informative')}})
    return rows


def _passes_track(fld: dict, lin: dict, ctx: dict, t_s: np.ndarray, path: np.ndarray,
                  bin_cm: float) -> list[dict]:
    """Every pass through one Linear / Circle field, in time order, with its
    leave-one-out expected spikes from the 1-D track map, its Central /
    Peripheral category and v2's traversal label. Keys starting with '_' are
    for the figure."""
    in_field = fld['in_field']
    if not in_field.any():
        return []
    disp, lb, dt = fld['disp'], lin['lin_bin'], ctx['dt_frames']
    n, circular = lin['n_lin'], lin['circular']
    pos = disp['to_disp'](lin['pos'])
    # Track bin of every frame in this field's display frame (Circle: unwrapped
    # around the field), and the peak bin in the same frame.
    dbin = np.rint(disp['to_disp']((lb + 0.5) * bin_cm) / bin_cm - 0.5).astype(int)
    pk_dbin = int(np.rint(disp['peak'] / bin_cm - 0.5))
    mid = 0.5 * (disp['lo'] + disp['hi'])
    spk_sorted = np.sort(ctx['spike_frame'])
    cum_dt = np.r_[0.0, np.cumsum(dt)]
    last = in_field.size - 1

    def side(i: int) -> str:
        return 'session edge' if i < 0 or i > last else ('low' if pos[i] < mid else 'high')

    rows = []
    for a, b, n_frag in _pass_runs(in_field, t_s):
        dur = float(cum_dt[b + 1] - cum_dt[a])
        if dur < MIN_PASS_S:
            continue
        i0, i1 = np.searchsorted(spk_sorted, a, 'left'), np.searchsorted(spk_sorted, b, 'right')
        pass_spk = spk_sorted[i0:i1]

        # Track ratemap without this pass.
        occ_p = np.bincount(lb[a:b + 1], weights=dt[a:b + 1], minlength=n)
        spk_p = np.bincount(lb[pass_spk], minlength=n).astype(float)
        occ_loo = lin['occ'] - occ_p
        spk_loo = lin['spk'] - spk_p
        rate = _rate_map_1d(spk_loo, occ_loo, lin['valid'] & (occ_loo > 1e-9), bin_cm, circular)
        rate_frames = rate[lb[a:b + 1]]
        expected = float(np.nansum(rate_frames * dt[a:b + 1]))

        p, pb, db = pos[a:b + 1], lb[a:b + 1], dbin[a:b + 1]
        ca = a + int(np.argmin(np.abs(p - disp['peak'])))   # closest approach to the peak
        peak_bin_dist = bin_cm * float(max(0, pk_dbin - db.max(), db.min() - pk_dbin))
        central = peak_bin_dist == 0
        r_seen = lin['rate'][pb]
        reached = (float(np.nanmax(r_seen)) / fld['peak_rate']
                   if np.isfinite(r_seen).any() and fld['peak_rate'] > 0 else np.nan)

        entry, exit_ = side(a - 1), side(b + 1)
        through = {entry, exit_} == {'low', 'high'}
        full = through or all(e in pb for e in disp['end_bins'])
        if through:
            up = entry == 'low'
            direction = ('CCW' if up else 'CW') if circular else ('L->R' if up else 'R->L')
        else:
            direction = 'return' if entry == exit_ else ''

        common = _pass_common(a, b, n_frag, dur, last, t_s, int(pass_spk.size), expected)
        rows.append({'_start': a, '_end': b, '_ca': ca, '_rate': rate_frames, '_rate_map': rate,
                     **{k: common[k] for k in ('t_start_s', 't_end_s', 'duration_s', 'n_fragments',
                                               'session_edge', 'max_pause_s', 'paused')},
                     'category': 'Central' if central else 'Peripheral',
                     'traversal': ('Full traversal' if full else
                                   'Partial, reached peak' if central else 'Partial, missed peak'),
                     'entry_side': entry, 'exit_side': exit_, 'direction': direction,
                     'peak_bin_dist_cm': round(peak_bin_dist, 2),
                     'reached_rate_frac': _rnd(reached, 3),
                     'closest_approach_cm': round(float(abs(pos[ca] - disp['peak'])), 2),
                     'path_cm': round(float(path[b] - path[a]), 2),
                     'entry_pos_cm': round(_abs_cm(p[0], lin), 2),
                     'exit_pos_cm': round(_abs_cm(p[-1], lin), 2),
                     **{k: common[k] for k in ('n_spikes', 'expected_spikes', 'z', 'informative')}})
    return rows


def _field_summary(base: dict, fld: dict, ctx: dict, lin: dict | None) -> dict:
    passes   = fld['passes']
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
    n_v6, n_v6_ok = _v6_rule_score(fld['in_field'], ctx['dt_frames'], ctx['spike_frame'])

    if lin is None:
        where = {'n_field_bins': int(fld['mask'].sum()),
                 'peak_x_cm': round(fld['peak_xy'][0], 2), 'peak_y_cm': round(fld['peak_xy'][1], 2)}
    else:
        d = fld['disp']
        where = {'n_field_bins': int(len(fld['bins'])),
                 'field_lo_cm': round(_abs_cm(d['lo'], lin), 2),
                 'field_hi_cm': round(_abs_cm(d['hi'], lin), 2),
                 'peak_cm': round(_abs_cm(d['peak'], lin), 2)}
    row = {**base, 'field_number': fld['field_number'], **where,
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
               round(float((central >= MIN_SPIKES_PER_PASS).mean()), 3) if central.size else np.nan}
    if lin is not None:
        trav = [p['traversal'] for p in passes]
        full = counts[[t == 'Full traversal' for t in trav]] if n else counts
        row.update({'n_full_traversals': trav.count('Full traversal'),
                    'n_partial_reached_peak': trav.count('Partial, reached peak'),
                    'n_partial_missed_peak': trav.count('Partial, missed peak'),
                    'lap_stability_full_traversals':
                        round(float((full >= MIN_SPIKES_PER_PASS).mean()), 3) if full.size else np.nan})
    row.update({'n_epochs_with_com': fld['n_epochs_with_com'],
                'com_shift_first_last_cm': fld['com_shift_first_last_cm'],
                'com_range_cm': fld['com_range_cm'],
                'n_passes_v6_rule': n_v6,
                'lap_stability_v6_rule': round(n_v6_ok / n_v6, 3) if n_v6 else np.nan})
    return row


# ── Epochs ────────────────────────────────────────────────────────────────────

def _epoch_bounds(dt: np.ndarray) -> np.ndarray:
    """Frame boundaries of N_EPOCHS spans of equal moving time."""
    cum = np.cumsum(dt)
    cuts = np.searchsorted(cum, cum[-1] * np.arange(1, N_EPOCHS) / N_EPOCHS)
    return np.r_[0, cuts, dt.size]


def _corr_vs_rest(rate, rate_rest, occ, occ_rest) -> tuple[float, int]:
    """Pearson r of an epoch map with the rest-of-session map on bins visited in
    both (>= pc.STABILITY_MIN_BINS of them), and the number of those bins."""
    joint = (occ > 0) & (occ_rest > 1e-9) & np.isfinite(rate) & np.isfinite(rate_rest)
    r = np.nan
    if joint.sum() >= pc.STABILITY_MIN_BINS and rate[joint].std() > 0 and rate_rest[joint].std() > 0:
        r = float(np.corrcoef(rate[joint], rate_rest[joint])[0, 1])
    return r, int(joint.sum())


def _epoch_com_2d(rate: np.ndarray, occ: np.ndarray, spk: np.ndarray, area: np.ndarray,
                  bin_cm: float) -> tuple[float, float]:
    """Rate-weighted COM (cm) of the visited bins of `area` that reach
    COM_RATE_FRAC of the epoch's highest rate there; NaN if undersampled."""
    seen = area & (occ > 0)
    if occ[seen].sum() < MIN_COM_DWELL_S or spk[seen].sum() == 0:
        return np.nan, np.nan
    r = np.where(seen, np.nan_to_num(rate), 0.0)
    if r.max() <= 0:
        return np.nan, np.nan
    keep = r >= COM_RATE_FRAC * r.max()
    ix, iy = np.nonzero(keep)
    w = r[keep]
    return (float(np.sum(w * (ix + 0.5)) / w.sum()) * bin_cm,
            float(np.sum(w * (iy + 0.5)) / w.sum()) * bin_cm)


def _epoch_com_ring(rate: np.ndarray, occ: np.ndarray, spk: np.ndarray, area: np.ndarray,
                    track_len: float) -> float:
    """Circle analogue of _epoch_com_2d: rate-weighted circular mean of the
    angular bins, as arc position (cm); NaN if undersampled."""
    seen = area & (occ > 0)
    if occ[seen].sum() < MIN_COM_DWELL_S or spk[seen].sum() == 0:
        return np.nan
    r = np.where(seen, np.nan_to_num(rate), 0.0)
    if r.max() <= 0:
        return np.nan
    keep = np.flatnonzero(r >= COM_RATE_FRAC * r.max())
    ang = 2.0 * np.pi * (keep + 0.5) / r.size
    mean = np.angle(np.sum(r[keep] * np.exp(1j * ang))) % (2.0 * np.pi)
    return float(mean / (2.0 * np.pi) * track_len)


def _epochs_2d(ctx: dict, fields: list[dict], t_s: np.ndarray, bin_cm: float, geo: dict) -> list[dict]:
    """Open / Linear epochs on the 2-D map: map, correlation with the rest of the
    session (bins visited in both), rate, dwell and COM of each field (display
    frame, so Linear 90 / 270 COM x = track position)."""
    bx, by, dt = ctx['beh_bx'], ctx['beh_by'], ctx['dt_frames']
    frames = _epoch_bounds(dt)
    spike_frame = ctx['spike_frame']
    out = []
    for k in range(N_EPOCHS):
        f0, f1 = int(frames[k]), int(frames[k + 1])
        occ = np.zeros_like(ctx['occ_map'])
        np.add.at(occ, (bx[f0:f1], by[f0:f1]), dt[f0:f1])
        s = spike_frame[(spike_frame >= f0) & (spike_frame < f1)]
        spk = np.zeros_like(occ)
        np.add.at(spk, (bx[s], by[s]), 1.0)
        rate = _rate_map(spk, occ, ctx['valid_mask'] & (occ > 0), bin_cm)

        occ_rest = np.clip(ctx['occ_map'] - occ, 0.0, None)
        spk_rest = ctx['spike_map'] - spk
        rate_rest = _rate_map(spk_rest, occ_rest, ctx['valid_mask'] & (occ_rest > 1e-9), bin_cm)
        r, n_joint = _corr_vs_rest(rate, rate_rest, occ, occ_rest)

        row = {'epoch': k + 1, 't_start_min': round(float(t_s[f0]) / 60.0, 2),
               't_end_min': round(float(t_s[f1 - 1]) / 60.0, 2),
               'n_spikes': int(s.size), 'n_joint_bins': n_joint,
               'r_vs_rest': _rnd(r, 3), '_rate': rate, '_occ': occ, '_com': {}}
        for fld in fields:
            n = fld['field_number']
            occ_f = float(occ[fld['mask']].sum())
            row[f'field{n}_dwell_s'] = round(occ_f, 2)
            row[f'field{n}_rate_hz'] = (round(float(spk[fld['mask']].sum()) / occ_f, 3)
                                        if occ_f > 0 else np.nan)
            cx, cy = _epoch_com_2d(rate, occ, spk,
                                   binary_dilation(fld['mask'], iterations=COM_DILATE_BINS), bin_cm)
            cx, cy = _disp_xy(geo, cx, cy)
            row[f'field{n}_com_x_cm'] = _rnd(cx)
            row[f'field{n}_com_y_cm'] = _rnd(cy)
            row['_com'][n] = (cx, cy)
        out.append(row)
    return out


def _epochs_ring(ctx: dict, lin: dict, fields: list[dict], t_s: np.ndarray, bin_cm: float) -> list[dict]:
    """Circle epochs on the angular map, as _epochs_2d; COM as arc position."""
    lb, dt, spike_frame = lin['lin_bin'], ctx['dt_frames'], ctx['spike_frame']
    n_lin, C, geo = lin['n_lin'], lin['track_len'], lin['geo']
    frames = _epoch_bounds(dt)
    out = []
    for k in range(N_EPOCHS):
        f0, f1 = int(frames[k]), int(frames[k + 1])
        occ = np.bincount(lb[f0:f1], weights=dt[f0:f1], minlength=n_lin).astype(float)
        s = spike_frame[(spike_frame >= f0) & (spike_frame < f1)]
        spk = np.bincount(lb[s], minlength=n_lin).astype(float)
        rate = _rate_map_1d(spk, occ, lin['valid'] & (occ > 0), bin_cm, True)

        occ_rest = np.clip(lin['occ'] - occ, 0.0, None)
        spk_rest = lin['spk'] - spk
        rate_rest = _rate_map_1d(spk_rest, occ_rest, lin['valid'] & (occ_rest > 1e-9), bin_cm, True)
        r, n_joint = _corr_vs_rest(rate, rate_rest, occ, occ_rest)

        row = {'epoch': k + 1, 't_start_min': round(float(t_s[f0]) / 60.0, 2),
               't_end_min': round(float(t_s[f1 - 1]) / 60.0, 2),
               'n_spikes': int(s.size), 'n_joint_bins': n_joint,
               'r_vs_rest': _rnd(r, 3), '_rate': rate, '_occ': occ, '_com': {}}
        for fld in fields:
            n = fld['field_number']
            fb = fld['bins']
            occ_f = float(occ[fb].sum())
            row[f'field{n}_dwell_s'] = round(occ_f, 2)
            row[f'field{n}_rate_hz'] = (round(float(spk[fb].sum()) / occ_f, 3)
                                        if occ_f > 0 else np.nan)
            area = np.zeros(n_lin, dtype=bool)
            area[(fb[:, None] + np.arange(-COM_DILATE_BINS, COM_DILATE_BINS + 1)[None, :]) % n_lin] = True
            pos = _epoch_com_ring(rate, occ, spk, area, C)
            row[f'field{n}_com_pos_cm'] = _rnd(pos)
            row['_com'][n] = (_ring_xy(geo, 2.0 * np.pi * pos / C) if np.isfinite(pos)
                              else (np.nan, np.nan))
        out.append(row)
    return out


def _com_shift(fld: dict, epochs: list[dict], ring_len: float | None) -> None:
    """Field COM drift across epochs, stored on the field: first -> last usable
    epoch, and the largest distance between any two usable epochs (cm; arc
    distance on the Circle, ring_len = its length)."""
    n = fld['field_number']
    if ring_len is None:
        xy = np.array([(ep[f'field{n}_com_x_cm'], ep[f'field{n}_com_y_cm']) for ep in epochs], dtype=float)
        xy = xy[~np.isnan(xy[:, 0])]
        d = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
    else:
        xy = np.array([ep[f'field{n}_com_pos_cm'] for ep in epochs], dtype=float)
        xy = xy[~np.isnan(xy)]
        d = np.abs(xy[:, None] - xy[None, :])
        d = np.minimum(d, ring_len - d)
    fld['n_epochs_with_com'] = int(len(xy))
    fld['com_shift_first_last_cm'] = round(float(d[0, -1]), 2) if len(xy) >= 2 else np.nan
    fld['com_range_cm'] = round(float(d.max()), 2) if len(xy) >= 2 else np.nan


def analyse_cell(metrics: dict, ctx: dict, arena: str, session_type: str | None,
                 bin_cm: float) -> dict | None:
    """Fields, passes and epochs of one cell; None if no field is detected."""
    t_s = (ctx['t'] - ctx['t_all'][0]) * 1e-6             # s from session start
    path = np.r_[0.0, np.cumsum(np.hypot(np.diff(ctx['x_cm']), np.diff(ctx['y_cm'])))]
    if arena == 'Open':
        fields, geo = _fields_open(metrics, ctx, bin_cm)
        if not fields:
            return None
        lin = None
        for fld in fields:
            fld['passes'] = _passes_open(fld, ctx, t_s, path, bin_cm)
        epochs = _epochs_2d(ctx, fields, t_s, bin_cm, geo)
    else:
        lin = _linearize(metrics, ctx, arena, session_type, bin_cm)
        if lin is None:
            return None
        fields, geo = lin['fields'], lin['geo']
        for fld in fields:
            fld['passes'] = _passes_track(fld, lin, ctx, t_s, path, bin_cm)
        epochs = (_epochs_ring(ctx, lin, fields, t_s, bin_cm) if lin['circular']
                  else _epochs_2d(ctx, fields, t_s, bin_cm, geo))
    ring_len = lin['track_len'] if lin is not None and lin['circular'] else None
    for fld in fields:
        _com_shift(fld, epochs, ring_len)
    return {'arena': arena, 'session_type': session_type, 'fields': fields, 't_s': t_s,
            'path': path, 'epochs': epochs, 'geo': geo, 'lin': lin}


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_trajectory(ax, fig, res: dict, ctx: dict, bin_cm: float) -> None:
    """Trajectory in gray, spikes coloured by time, fields outlined."""
    geo = res['geo']
    sf = ctx['spike_frame']
    ax.plot(geo['x_all'], geo['y_all'], color='0.8', linewidth=0.4)
    sc = ax.scatter(geo['x'][sf], geo['y'][sf], c=res['t_s'][sf] / 60.0, cmap=TIME_CMAP,
                    s=5, linewidths=0, zorder=3)
    _outline_fields(ax, geo, res['fields'], bin_cm)
    fig.colorbar(sc, ax=ax, label='spike time (min)', fraction=0.046, pad=0.02)
    _set_map_axes(ax, geo)
    # No frame, ticks or x / y labels; the title and colour bar stay.
    ax.set_axis_off()
    ax.set_title('Trajectory + spikes coloured by time (fields outlined)'
                 + (' – vertical track rotated to horizontal' if geo['rotated'] else ''))


def _plot_session_map(ax, fig, res: dict, bin_cm: float) -> None:
    """Session ratemap with fields outlined and their peaks starred."""
    geo = res['geo']
    im = _draw_map(ax, geo, geo['session_rate'], geo['session_show'])
    _outline_fields(ax, geo, res['fields'], bin_cm)
    for fld in res['fields']:
        ax.plot(*fld['peak_xy_disp'], marker='*', color='white', markeredgecolor='black',
                markersize=11, linestyle='none')
        ax.annotate(str(fld['field_number']), fld['peak_xy_disp'], xytext=(6, 6),
                    textcoords='offset points', fontsize=9, fontweight='bold')
    fig.colorbar(im, ax=ax, label='rate (Hz)', fraction=0.046, pad=0.02)
    # No frame, ticks or x / y labels; the title and colour bar stay.
    ax.set_axis_off()
    ax.set_title(f'Session ratemap ({_map_label()}'
                 + ('; angular map drawn on the ring' if geo['kind'] == 'circle' else '') + ')')


def _raster_image_open(passes: list[dict], res: dict, xlim: float) -> tuple[np.ndarray, np.ndarray]:
    """(n_passes, n_x) leave-one-out rate along each open-field pass on the
    signed-path axis; NaN where the pass did not go."""
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


def _raster_image_track(passes: list[dict], pos_disp: np.ndarray, lin: dict, xlim: tuple,
                        bin_cm: float) -> tuple[np.ndarray, np.ndarray]:
    """(n_passes, n_x) leave-one-out track rate over the stretch of track each
    pass covered (lowest -> highest position); NaN elsewhere."""
    step = RASTER_X_STEP_CM
    grid = np.arange(xlim[0] + 0.5 * step, xlim[1], step)
    g = np.mod(grid, lin['track_len']) if lin['circular'] else grid
    gbin = np.clip((g / bin_cm).astype(int), 0, lin['n_lin'] - 1)
    img = np.full((len(passes), grid.size), np.nan)
    for i, p in enumerate(passes):
        xs = pos_disp[p['_start']:p['_end'] + 1]
        rate = np.nan_to_num(p['_rate_map'], nan=0.0)
        inside = (grid >= xs.min()) & (grid <= xs.max())
        if inside.any():
            img[i, inside] = rate[gbin[inside]]
        else:                                              # shorter than one grid step
            j = int(np.argmin(np.abs(grid - xs.mean())))
            img[i, j] = rate[gbin[j]]
    return img, grid


def _plot_raster(ax, passes: list[dict], order: np.ndarray, img: np.ndarray, grid: np.ndarray,
                 vmax: float, xof, res: dict, ctx: dict, xlim: tuple, title: str, xlabel: str,
                 period: float | None = None, vlines: tuple = (), fontsize: float = 9,
                 loc: str = 'center') -> None:
    """Passes in the given order (first at top): shading = leave-one-out rate
    along the path, gray = context, spikes as ticks that scale with the row
    height. xof(pass, frames) gives the x of those frames on this raster's axis."""
    t_s = res['t_s']
    spk = np.sort(ctx['spike_frame'])
    n = len(passes)
    step = RASTER_X_STEP_CM
    ax.imshow(img[order], cmap=RATE_CMAP, vmin=0, vmax=vmax, aspect='auto', alpha=RASTER_SHADE_ALPHA,
              interpolation='nearest',
              extent=(grid[0] - step / 2, grid[-1] + step / 2, n + 0.5, 0.5))
    ctx_segs, spike_segs, ctx_spike_segs = [], [], []
    for row, idx in enumerate(order, start=1):
        p = passes[idx]
        a, b = p['_start'], p['_end']
        c0 = int(np.searchsorted(t_s, t_s[a] - CONTEXT_S, 'left'))
        c1 = int(np.searchsorted(t_s, t_s[b] + CONTEXT_S, 'right')) - 1
        if c0 < a:
            ctx_segs += _segments(xof(p, np.arange(c0, a + 1)), np.full(a - c0 + 1, row), period)
        if c1 > b:
            ctx_segs += _segments(xof(p, np.arange(b, c1 + 1)), np.full(c1 - b + 1, row), period)
        f_ctx = spk[np.searchsorted(spk, c0, 'left'):np.searchsorted(spk, c1, 'right')]
        for f, pos in zip(f_ctx, xof(p, f_ctx)):
            seg = [(pos, row - 0.42), (pos, row + 0.42)]
            (spike_segs if a <= f <= b else ctx_spike_segs).append(seg)
    ax.add_collection(LineCollection(ctx_segs, colors='0.7', linewidths=0.6, zorder=2))
    ax.add_collection(LineCollection(ctx_spike_segs, colors=CONTEXT_SPIKE_COLOR, linewidths=0.8, zorder=3))
    ax.add_collection(LineCollection(spike_segs, colors=SPIKE_COLOR, linewidths=1.0, zorder=4))
    for x, style in vlines:
        ax.axvline(x, zorder=1, **style)
    ax.set_xlim(*xlim)
    ax.set_ylim(n + 0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=fontsize, loc=loc)


def _plot_strip(ax, passes: list[dict], order: np.ndarray, labels: list[str], ylabel: str) -> None:
    """Narrow strip left of a raster: distance to the peak bin, pause / session-
    edge / full-traversal flags, y labels."""
    n = len(passes)
    dist = np.array([[passes[i]['peak_bin_dist_cm']] for i in order])
    ax.imshow(dist, cmap=PEAK_DIST_CMAP, vmin=0, vmax=PEAK_DIST_MAX_CM,
              aspect='auto', interpolation='nearest', extent=(0, 1, n + 0.5, 0.5))
    rows = np.arange(1, n + 1)
    paused = np.array([passes[i]['paused'] for i in order])
    edge = np.array([passes[i]['session_edge'] for i in order])
    ax.plot(np.full(paused.sum(), 1.5), rows[paused], 's', color='black', markersize=2.5)
    ax.plot(np.full(edge.sum(), 1.5), rows[edge], 'x', color='black', markersize=4)
    xmax = 2.0
    if 'traversal' in passes[0]:
        full = np.array([passes[i]['traversal'] == 'Full traversal' for i in order])
        ax.plot(np.full(full.sum(), 2.3), rows[full], 'o', color='black', mfc='none',
                markersize=2.8, markeredgewidth=0.7)
        xmax = 2.8
    ax.set_xlim(0, xmax)
    ax.set_ylim(n + 0.5, 0.5)
    ax.set_xticks([])
    step = max(1, int(np.ceil(n / 25)))
    ticks = list(range(1, n + 1, step))
    ax.set_yticks(ticks)
    ax.set_yticklabels([labels[k - 1] for k in ticks], fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    for s in ('top', 'right', 'bottom'):
        ax.spines[s].set_visible(False)


def _plot_filmstrip(fig, gs, res: dict, bin_cm: float) -> None:
    """RATEMAP_TYPE map per epoch on one colour scale; unvisited bins blank,
    fields outlined, COM marked. All arenas side by side. Open / Circle: square
    panels at their 1-field size, left-aligned. Linear: across the full width."""
    geo, epochs = res['geo'], res['epochs']
    # An epoch can fire above the session map's peak; the 99th percentile keeps its
    # structure visible without letting one poorly sampled bin set the scale.
    visited = np.concatenate([ep['_rate'][ep['_occ'] > 0] for ep in epochs])
    visited = visited[np.isfinite(visited)]
    vmax = float(np.percentile(visited, 99)) if visited.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    if geo['kind'] == 'linear':
        # Last column: room for the colourbar label, written horizontally because
        # the bar is only as tall as the thin track maps.
        sub = gs.subgridspec(1, N_EPOCHS + 2, width_ratios=LINEAR_FILM_RATIOS, wspace=0.08)
        axes = [fig.add_subplot(sub[k]) for k in range(N_EPOCHS)]
        cax = fig.add_subplot(sub[N_EPOCHS])
    else:
        fill = max(fig.get_figwidth() - 16.0, 0.0) / 16.0
        sub = gs.subgridspec(1, N_EPOCHS + 2,
                             width_ratios=[1] * N_EPOCHS + [0.05, N_EPOCHS * fill + 1e-6], wspace=0.08)
        axes = [fig.add_subplot(sub[k]) for k in range(N_EPOCHS)]
        cax = fig.add_subplot(sub[N_EPOCHS])
    im = None
    for k, (ax, ep) in enumerate(zip(axes, epochs)):
        im = _draw_map(ax, geo, ep['_rate'], ep['_occ'] > 0, vmax, cmap=EPOCH_MAP_CMAP)
        _outline_fields(ax, geo, res['fields'], bin_cm)
        for j, fld in enumerate(res['fields']):
            cx, cy = ep['_com'][fld['field_number']]
            if np.isfinite(cx):
                ax.plot(cx, cy, marker='P', color=FIELD_COLORS[j % len(FIELD_COLORS)],
                        markeredgecolor='white', markersize=8, linestyle='none', zorder=5)
        r = f"{ep['r_vs_rest']:.2f}" if not np.isnan(ep['r_vs_rest']) else 'n/a'
        head = f"{ep['t_start_min']:.1f}–{ep['t_end_min']:.1f} min"
        tail = f"r vs rest = {r},  {ep['n_spikes']} spikes"
        ax.set_title(f'{head}\n{tail}', fontsize=9)
        # No frame, ticks or x / y labels on the epoch maps; the title stays.
        ax.set_axis_off()
    if geo['kind'] == 'linear':
        cb = fig.colorbar(im, cax=cax)
        cb.locator = MaxNLocator(2)
        cb.update_ticks()
        cb.ax.tick_params(labelsize=6)
        cb.set_label('rate (Hz)\ntop = 99th pct', rotation=0, ha='left', va='center',
                     fontsize=7, labelpad=6)
    else:
        fig.colorbar(im, cax=cax, label='rate (Hz), top = 99th pct')


def _legend_handles(track: bool) -> list:
    shades = ([Patch(color=PEAK_DIST_CMAP(0.0), label='through peak bin')]
              + [Patch(color=PEAK_DIST_CMAP(frac), label=f'{frac * PEAK_DIST_MAX_CM:g} cm from peak bin')
                 for frac in (1 / 3, 2 / 3)]
              + [Patch(color=PEAK_DIST_CMAP(1.0), label=f'>= {PEAK_DIST_MAX_CM:g} cm from peak bin')])
    flags = [Line2D([], [], color='black', marker='s', ls='none', ms=3, label=f'paused > {PAUSE_FLAG_S:g} s'),
             Line2D([], [], color='black', marker='x', ls='none', ms=4, label='cut by session edge')]
    if track:
        flags += [Line2D([], [], color='black', marker='o', mfc='none', ls='none', ms=4,
                         label='full traversal'),
                  Line2D([], [], color='0.35', ls='--', lw=0.8, label='peak bin'),
                  Line2D([], [], color='0.35', ls=':', lw=1.0, label='field edges')]
    return (shades + flags
            + [Line2D([], [], color='0.7', lw=1, label=f'+/- {CONTEXT_S:g} s context'),
               Line2D([], [], color=SPIKE_COLOR, lw=1.2, label='spike in pass'),
               Line2D([], [], color=CONTEXT_SPIKE_COLOR, lw=1.2, label='spike in context')])


def make_figure(res: dict, ctx: dict, metrics: dict, summaries: list[dict], bin_cm: float,
                title: str, save_path: str) -> None:
    """One figure per cell; see the module docstring for the rows."""
    geo, lin, fields = res['geo'], res['lin'], res['fields']
    kind = geo['kind']
    n_f = len(fields)
    max_passes = max(max(len(f['passes']) for f in fields), 1)
    fig_w = max(16.0, 9.0 * n_f)

    h_raster = float(np.clip(0.13 * max_passes + 1.5, 4.0, 14.0))
    if kind == 'linear':
        aspect = geo['extent'][3] / geo['extent'][1]
        h_2d = float(np.clip(0.8 * fig_w * aspect + 0.9, 1.6, 6.0))
        h_top = 2 * h_2d + 0.6
        # Epoch maps side by side across the plot width (left margin 1.3 in,
        # right edge 0.97). The row is made a little shorter than a map at full
        # column width, so each map fills the row height and the colourbar
        # matches it; titles and x labels sit in the gaps between rows.
        n_col = len(LINEAR_FILM_RATIOS)
        units = sum(LINEAR_FILM_RATIOS) * (1 + 0.08 * (n_col - 1) / n_col)
        panel_w = (0.97 * fig_w - 1.3) / units
        h_film = float(np.clip(0.95 * panel_w * aspect, 0.25, 5.0))
    else:
        h_top, h_film = 5.0, 3.4
    rows = ['top', 'raster', 'film']
    row_h = {'top': h_top, 'raster': h_raster, 'film': h_film}
    heights = [row_h[r] for r in rows]
    hspace = 0.42
    if kind == 'linear':
        # GridSpec takes the row gaps out of the rows, so a row gets less than its
        # nominal height (fig_h = sum + 2.8, margins 1.3 + 0.6 in). Scale the film
        # row up so its maps get the h_film they need to fill their columns.
        n, i = len(rows), rows.index('film')
        for _ in range(20):
            s = sum(heights)
            heights[i] = h_film * (n + (n - 1) * hspace) * s / (n * (s + 0.9))
    fig_h = sum(heights) + 2.8
    fig = Figure(figsize=(fig_w, fig_h))
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(len(rows), 1, height_ratios=heights, hspace=hspace)
    G = {r: gs[i] for i, r in enumerate(rows)}

    # Row 1: trajectory + session map. Open / Circle stay at their 1-field size,
    # left-aligned, however wide the figure is; Linear one above the other.
    if kind == 'linear':
        gs_top = G['top'].subgridspec(2, 1, hspace=0.55)
        ax_t, ax_m = fig.add_subplot(gs_top[0]), fig.add_subplot(gs_top[1])
    else:
        gs_top = G['top'].subgridspec(1, 3, width_ratios=[8.0, 8.0, fig_w - 16.0 + 1e-6], wspace=0.15)
        ax_t, ax_m = fig.add_subplot(gs_top[0]), fig.add_subplot(gs_top[1])
    _plot_trajectory(ax_t, fig, res, ctx, bin_cm)
    _plot_session_map(ax_m, fig, res, bin_cm)

    if kind != 'open':
        C = lin['track_len']
        track_xlim = (0.0, C) if lin['circular'] else (0.0, max(C, pc.arena_width_cm))

    gs_r = G['raster'].subgridspec(1, n_f, wspace=0.18)
    path = res['path']
    for k, (fld, summary) in enumerate(zip(fields, summaries)):
        passes = fld['passes']
        n = len(passes)
        if not passes:
            fig.add_subplot(gs_r[k]).set_title(f"Field {fld['field_number']}: no passes")
            continue

        if kind == 'open':
            reach = np.array([max(abs(path[p['_start']] - path[p['_ca']]),
                                  abs(path[p['_end']] - path[p['_ca']])) for p in passes])
            half = max(10.0, float(np.ceil(np.percentile(reach, 95))) + 5.0)
            n_clip = int((reach > half).sum())
            xlim = (-half, half)
            img, grid = _raster_image_open(passes, res, half)
            xof = lambda p, fr: path[fr] - path[p['_ca']]
            period, ticks = None, None
            vlines = ((0.0, dict(color='0.35', ls='--', lw=0.7)),)
            xlabel = 'signed path length from closest approach (cm)'
            head = (f"Field {fld['field_number']}, time order: {n} passes "
                    f"({summary['n_central']} central, {summary['n_informative']} informative)"
                    + (f", {n_clip} extend past the axis" if n_clip else ''))
        else:
            disp, C = fld['disp'], lin['track_len']
            xlim = (disp['shift'], disp['shift'] + C) if lin['circular'] else track_xlim
            pos_disp = disp['to_disp'](lin['pos'])
            img, grid = _raster_image_track(passes, pos_disp, lin, xlim, bin_cm)
            xof = lambda p, fr: pos_disp[fr]
            period = C if lin['circular'] else None
            vlines = ((disp['peak'], dict(color='0.35', ls='--', lw=0.8)),
                      (disp['lo'], dict(color='0.35', ls=':', lw=1.0)),
                      (disp['hi'], dict(color='0.35', ls=':', lw=1.0)))
            seam = lin['circular'] and disp['shift'] != 0
            ticks = np.sort(disp['to_disp'](np.arange(0.0, C, 50.0))) if seam else None
            xlabel = 'track position (cm' + ('; seam moved opposite this field)' if seam else ')')
            head = (f"Field {fld['field_number']}: {summary['field_lo_cm']:.0f}–"
                    f"{summary['field_hi_cm']:.0f} cm, peak {summary['peak_cm']:.0f} cm;  time order: "
                    f"{n} passes ({summary['n_central']} central, {summary['n_full_traversals']} full "
                    f"traversals, {summary['n_informative']} informative)")

        def _ticks(ax):
            if ticks is not None:
                ax.set_xticks(ticks)
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{np.mod(v, C):.0f}'))

        # Time-ordered raster with its strip; the raster key goes in the slot to
        # the right of the first field's raster, so it covers no data.
        sub = gs_r[k].subgridspec(1, 3, width_ratios=[0.22, 5, 2.0], wspace=0.05)
        order = np.arange(n)
        labels = [f"{i}  ({p['t_start_s'] / 60:.1f} min)" for i, p in enumerate(passes, start=1)]
        _plot_strip(fig.add_subplot(sub[0]), passes, order, labels, 'pass # (chronological; start time)')
        ax = fig.add_subplot(sub[1])
        vmax = fld['shade_max'] if fld['shade_max'] > 0 else 1.0
        _plot_raster(ax, passes, order, img, grid, vmax, xof, res, ctx, xlim,
                     head + '\n'
                     f"informative passes with >= 1 spike: {_fmt(summary['hit_rate_informative'])} "
                     f"(Poisson: {_fmt(summary['poisson_hit_rate_informative'])});  "
                     f"var z = {_fmt(summary['overdispersion_var_z'])}\n"
                     f"Spearman rho(z, pass #) = {_fmt(summary['spearman_rho_z_vs_pass'])};  "
                     f"COM shift = {_fmt(summary['com_shift_first_last_cm'], ' cm')};  "
                     f"top pass = {_fmt(summary['top_pass_spike_frac'])} of spikes\n"
                     f"shading: rate predicted without that pass, 0–{vmax:.2g} Hz; "
                     f"white = no path",
                     xlabel, period=period, vlines=vlines, fontsize=8, loc='left')
        _ticks(ax)
        if k == 0:
            ax_key = fig.add_subplot(sub[2])
            ax_key.axis('off')
            ax_key.legend(handles=_legend_handles(kind != 'open'), loc='upper left',
                          bbox_to_anchor=(0.0, 1.0), borderaxespad=0.0, fontsize=7,
                          title='raster key', title_fontsize=7, frameon=True)

    _plot_filmstrip(fig, G['film'], res, bin_cm)

    if kind == 'open':
        note = 'Open field'
    elif kind == 'circle':
        note = 'Circle: position = arc length along the ring'
    else:
        note = (f"Linear {res['session_type']}°"
                + (': vertical track rotated 90° CCW, position = L - y' if geo['rotated'] else ': position = x'))
    fig.suptitle(f'{title}  |  {note}  |  {n_f} field(s), {metrics.get("n_spikes")} spikes (moving)  |  '
                 f'{_map_label()}\n'
                 f'Passes merged across exits <= {MERGE_GAP_S:g} s, >= {MIN_PASS_S:g} s long; '
                 f'informative = expected spikes >= {MIN_EXPECTED_SPIKES:g}; '
                 f'Central = through the peak bin; '
                 f'{N_EPOCHS} epochs of equal moving time (plus = field COM)',
                 fontsize=11, y=1 - 0.2 / fig_h, va='top')
    fig.subplots_adjust(left=1.3 / fig_w, right=0.97, top=1 - 1.3 / fig_h, bottom=0.6 / fig_h)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)


# ── Per-cell job ──────────────────────────────────────────────────────────────

def lap_raster_cell(metrics: dict, ctx: dict, dirpath: str, session: str, unit: str,
                    save_path: str) -> tuple[list[dict], list[dict], list[dict]] | None:
    """Lap raster figure for one place cell from its compute_metrics output
    (run with this module's pc settings), saved to save_path. Arena and session
    type come from dirpath (the session folder); session labels the figure and
    the rows. Returns (field_rows, pass_rows, epoch_rows), or None if no place
    field is detected."""
    arena, session_type = _arena_and_session(dirpath)
    if arena is None:
        raise ValueError(f'no Open / Linear / Circle folder in {dirpath}')
    bin_cm = pc.target_bin_cm
    res = analyse_cell(metrics, ctx, arena, session_type, bin_cm)
    if res is None:
        return None

    lin = res['lin']
    base = {'session': session, 'unit': unit, 'arena_type': arena, 'session_type': session_type}
    if lin is not None:
        base.update(rotated_90_ccw=lin['rotated'], track_len_cm=round(lin['track_len'], 2))
    summaries = [_field_summary(base, fld, ctx, lin) for fld in res['fields']]

    ntt_name = os.path.splitext(unit)[0]
    make_figure(res, ctx, metrics, summaries, bin_cm, f'{session}  |  {ntt_name}', save_path)

    pass_rows = [{**base, 'field_number': fld['field_number'], 'pass_number': i,
                  **{k: v for k, v in p.items() if not k.startswith('_')}}
                 for fld in res['fields'] for i, p in enumerate(fld['passes'], start=1)]
    epoch_rows = [{**base, **{k: v for k, v in ep.items() if not k.startswith('_')}}
                  for ep in res['epochs']]
    return summaries, pass_rows, epoch_rows


def _run_cell(args) -> tuple[list[dict], list[dict], list[dict]]:
    """Lap raster figure for one place cell. Returns (field_rows, pass_rows,
    epoch_rows)."""
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

    metrics, ctx = pc.compute_metrics(csv_path, ntt_path, pc.arena_width_cm, pc.target_bin_cm)
    if not ctx or not ctx['valid_mask'].any():
        with _print_lock:
            print(f'  [SKIP] no valid occupancy for {unit}')
        return [], [], []

    save_path = os.path.join(dirpath, LAP_SUBDIR, f'{os.path.splitext(unit)[0]}_lap_raster.png')
    rows = lap_raster_cell(metrics, ctx, dirpath, session, unit, save_path)
    if rows is None:
        with _print_lock:
            print(f'  [SKIP] no place field detected for {unit}')
        return [], [], []
    with _print_lock:
        print(f'  [SAVED] {save_path}')
    return rows


# ── Batch ─────────────────────────────────────────────────────────────────────

def run_all() -> None:
    """Lap raster for every .ntt under <place_cell_folder>\\<animal>\\<arena>,
    arena in ARENAS; writes output_excel (PerField, PerPass, PerEpoch)."""
    jobs = []
    for animal in pc.ANIMAL_NAMES:
        for arena in ARENAS:
            for dirpath, _, filenames in os.walk(os.path.join(place_cell_folder, animal, arena)):
                if _arena_and_session(dirpath)[0] != arena:
                    continue
                jobs += [(dirpath, unit) for unit in sorted(filenames) if unit.lower().endswith('.ntt')]
    job_args = [(i, len(jobs), *job) for i, job in enumerate(jobs, start=1)]
    print(f'Found {len(jobs)} place cell(s) ({", ".join(ARENAS)}) in {place_cell_folder}\n')

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
