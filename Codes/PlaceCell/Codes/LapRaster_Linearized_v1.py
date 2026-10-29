# -*- coding: utf-8 -*-
"""
Linearized lap rasters for place cells on the 1-D tracks (Linear, Circle) – V1

Checks the stability of firing across laps: every pass through every place
field is drawn as one row (chronological), position along the linearized track
on x, with the cell's spikes (or none) on that pass.

For every .ntt under place_cell_folder whose session is on the Linear or Circle
track (Open-arena sessions are skipped):
  1. Rebuild trajectory, spike frames and bin grid with
     PlaceCell_Main_v21.compute_metrics (same cleaning, smoothing, immobility
     filter and spike assignment as the run that detected the fields).
  2. Detect the place fields with PlaceCell_Main_v21's threshold method
     (Linear: detect_place_fields_threshold_2d; Circle: _circular).
  3. Linearize:
       Linear 0 / 180   : position = x (cm), track along x already.
       Linear 90 / 270  : the track is vertical; the trajectory is rotated
                          90 deg counter-clockwise, (x, y) -> (L - y, x) with
                          L = the session's y extent, so the top of the
                          vertical track becomes position 0. The 2-D map in
                          the figure is drawn rotated the same way.
       Circle           : position = arc length along the ring,
                          theta / bin_width_rad * bin_cm (0 .. ~circumference),
                          on the same angular bins as the circular fields.
     A 2-D Linear field becomes the range of long-axis bins it spans.
  4. Passes: one pass is one contiguous run of moving-only frames inside the
     field's 1-D extent (entry -> exit through either end). EVERY pass is kept
     (only passes shorter than MIN_PASS_S, i.e. border flicker, are dropped),
     whether or not it went through the field's peak bin:
       Full traversal        entered at one end of the field and left at the
                             other, or covered both end bins of the field (a
                             field against the track end: in to the wall and back)
       Partial, reached peak turned back (or session edge) after the peak bin
       Partial, missed peak  turned back before reaching the peak bin
     Passes cut by the start / end of the session are kept and flagged
     (entry_side / exit_side = 'session edge').

Figure (one per cell, saved in LAP_SUBDIR next to its .ntt):
  row 1  trajectory + spikes, 2-D (Linear 90 / 270 rotated to horizontal),
         fields outlined
  row 2  1-D ratemap along the linearized track (raw + Gaussian-smoothed)
  row 3  linearized position vs time, spikes in red
  row 4  per field: lap raster (pass # on y, chronological, top = first;
         context of +/- CONTEXT_S s around each pass in light gray, the pass
         itself coloured by category, spikes as red ticks) + spikes per pass
Circle fields that straddle the theta = 0 seam are drawn with the seam moved
to the opposite side of the ring (tick labels still give absolute position).

Tables (output_excel):
  PerPass   one row per pass: times, entry / exit side, direction, category,
            spikes in the pass, positions
  PerField  per field: pass counts per category, passes with >= 1 spike, lap
            stability (fraction of passes with >= MIN_SPIKES_PER_PASS spikes;
            all passes and full traversals only), Spearman rho of spikes per
            pass vs pass number (drift across the session)

Analysis settings (bin size, smoothing, filters, field thresholds, COORD_UNITS, ...)
come from PlaceCell_Main_v21; only the directories below are set here.
"""

import os
import re
import concurrent.futures
import threading

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

import PlaceCell_Main_v21 as pc


# ── Directories (edit these per run) ──────────────────────────────────────────

# Place-cell folder (Output_PlaceTrue of a PlaceCell_Main_v21 run): every .ntt
# under it on the Linear / Circle track is analysed.
place_cell_folder = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Alpha200'

# Per-pass / per-field results table.
output_excel = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Alpha200\Results\LapRaster_Linearized.xlsx'

# Figures go in this subfolder of each session folder (next to the .ntt).
LAP_SUBDIR = 'lap_raster_linearized'

# ── Settings ──────────────────────────────────────────────────────────────────

MIN_PASS_S          = 0.1   # passes shorter than this (s) are border flicker, dropped; 0 = keep all
MIN_SPIKES_PER_PASS = 5     # lap stability: a pass counts as stable with >= this many spikes
CONTEXT_S           = 1.0   # trajectory drawn (light gray) this long before / after each pass
LINEAR_ROTATE_TYPES = ('90', '270')   # Linear session types recorded on the vertical track

CATEGORY_COLORS = {'Full traversal':        'black',
                   'Partial, reached peak': 'tab:blue',
                   'Partial, missed peak':  'tab:orange'}
FIELD_COLORS = ['tab:green', 'tab:purple', 'tab:cyan', 'tab:olive', 'tab:brown', 'tab:pink']
SPIKE_COLOR         = 'red'
CONTEXT_SPIKE_COLOR = 'lightcoral'

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


def _arena_and_session(dirpath: str) -> tuple[str | None, str | None]:
    """('Linear' | 'Circle' | None, session type) from the session folder path,
    e.g. '...\\Linear\\Day10\\4_90' -> ('Linear', '90'). Session type is the
    trailing angle for Linear, the last folder name for Circle."""
    parts = [p.lower() for p in re.split(r'[\\/]+', dirpath) if p]
    label = parts[-1] if parts else ''
    if 'linear' in parts:
        m = re.search(r'_(270|180|90|0)$', label)
        return 'Linear', (m.group(1) if m else None)
    if 'circle' in parts:
        return 'Circle', label
    return None, None


def _runs(bins: np.ndarray, n_bins: int, circular: bool) -> list[tuple[int, int]]:
    """Contiguous runs (first, last) of sorted bin indices; on a circular axis
    a run through bin n_bins-1 -> 0 is joined into one (first > last)."""
    bins = np.unique(bins)
    cut  = np.flatnonzero(np.diff(bins) > 1) + 1
    runs = [(int(r[0]), int(r[-1])) for r in np.split(bins, cut)]
    if circular and len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == n_bins - 1:
        runs = [(runs[-1][0], runs[0][1])] + runs[1:-1]
    return runs


def _shade_runs(ax, runs: list[tuple[int, int]], bin_cm: float, track_len: float,
                color: str, alpha: float = 0.15) -> None:
    """axvspan every bin run (absolute track coordinates; wrapped runs split)."""
    for b0, b1 in runs:
        spans = [(b0, b1)] if b0 <= b1 else [(b0, track_len / bin_cm - 1), (0, b1)]
        for s0, s1 in spans:
            ax.axvspan(s0 * bin_cm, (s1 + 1) * bin_cm, color=color, alpha=alpha, lw=0)


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


def _smooth_1d(rate: np.ndarray, valid: np.ndarray, bin_cm: float, circular: bool) -> np.ndarray:
    """Gaussian smoothing (sigma = pc.GAUSSIAN_SIGMA_CM) of a 1-D ratemap over
    its valid bins; wraps around on the Circle track."""
    if circular:
        return pc._gaussian_smooth_circular(rate, valid, bin_cm)
    kernel = pc._gaussian_kernel_1d(pc.GAUSSIAN_SIGMA_CM / bin_cm)
    num = np.convolve(np.where(valid, rate, 0.0), kernel, mode='same')
    den = np.convolve(valid.astype(float), kernel, mode='same')
    out = np.zeros_like(rate)
    ok = valid & (den > 0)
    out[ok] = num[ok] / den[ok]
    return out


# ── Linearization ─────────────────────────────────────────────────────────────

def _linearize(metrics: dict, ctx: dict, arena: str, session_type: str | None,
               bin_cm: float) -> dict | None:
    """1-D track coordinates, 1-D field bins and 2-D display coordinates.

    Returns None if no place field is detected. Keys:
      pos, lin_bin        per moving frame: position (cm) and 1-D bin
      n_lin, track_len    number of 1-D bins, track length (cm)
      circular, rotated   Circle track; Linear 90 / 270 rotated to horizontal
      fields              list of {field_number, bins, peak_bin, mask2d / arc}
      x2d_all, y2d_all, x2d, y2d, extent    2-D trajectory for display
    """
    out = {'circular': arena == 'Circle', 'rotated': False}
    n_bx, n_by = ctx['n_bins_x'], ctx['n_bins_y']

    if arena == 'Circle':
        ctx_circ = pc._build_ratemap_circular(ctx, bin_cm)
        fields, _ = pc.detect_place_fields_threshold_circular(ctx_circ, bin_cm)
        theta = np.mod(np.arctan2(ctx['y_cm'] - ctx_circ['cy'], ctx['x_cm'] - ctx_circ['cx']),
                       2.0 * np.pi)
        n_lin = ctx_circ['n_bins_theta']
        out.update(pos=theta / ctx_circ['bin_width_rad'] * bin_cm, lin_bin=ctx_circ['beh_bt'],
                   n_lin=n_lin, track_len=n_lin * bin_cm,
                   x2d_all=ctx['x_cm_all'], y2d_all=ctx['y_cm_all'],
                   x2d=ctx['x_cm'], y2d=ctx['y_cm'],
                   extent=[0, n_bx * bin_cm, 0, n_by * bin_cm],
                   ring=(ctx_circ['cx'], ctx_circ['cy'],
                         0.5 * (ctx_circ['r_min'] + ctx_circ['r_max']),
                         ctx_circ['bin_width_rad']))
        out['fields'] = [{'field_number': f['field_number'],
                          'bins': np.array(sorted(int(b) for b in f['bin_coords'].split(';'))),
                          'peak_bin': int(f['peak_bin_theta'])} for f in fields]
    else:
        fields, _ = pc.detect_place_fields_threshold_2d(metrics, ctx, bin_cm)
        rotate = session_type in LINEAR_ROTATE_TYPES
        if (n_by > n_bx) != rotate:
            with _print_lock:
                print(f'  [WARN] Linear session type {session_type!r} but track extent is '
                      f'{n_bx} x {n_by} bins; rotation = {rotate}')
        if rotate:
            # 90 deg counter-clockwise: (x, y) -> (L - y, x); bin (bx, by) -> (n_by-1-by, bx).
            L = n_by * bin_cm
            out.update(rotated=True, n_lin=n_by, lin_bin=n_by - 1 - ctx['beh_by'],
                       x2d_all=L - ctx['y_cm_all'], y2d_all=ctx['x_cm_all'],
                       x2d=L - ctx['y_cm'], y2d=ctx['x_cm'],
                       extent=[0, n_by * bin_cm, 0, n_bx * bin_cm])
            long_idx = lambda bx, by: n_by - 1 - by
        else:
            out.update(n_lin=n_bx, lin_bin=ctx['beh_bx'],
                       x2d_all=ctx['x_cm_all'], y2d_all=ctx['y_cm_all'],
                       x2d=ctx['x_cm'], y2d=ctx['y_cm'],
                       extent=[0, n_bx * bin_cm, 0, n_by * bin_cm])
            long_idx = lambda bx, by: bx
        out['pos'] = out['x2d']
        out['track_len'] = out['n_lin'] * bin_cm
        out['fields'] = []
        for f in fields:
            pairs = [tuple(int(v) for v in p.split('-')) for p in f['bin_coords'].split(';')]
            mask = np.zeros((n_bx, n_by), dtype=bool)
            for bx, by in pairs:
                mask[bx, by] = True
            out['fields'].append({
                'field_number': f['field_number'],
                'bins': np.unique([long_idx(bx, by) for bx, by in pairs]),
                'peak_bin': int(long_idx(f['peak_bin_x'], f['peak_bin_y'])),
                'mask2d': np.rot90(mask) if rotate else mask})

    return out if out['fields'] else None


def _field_display(fld: dict, lin: dict, bin_cm: float) -> dict:
    """Display frame for one field. Linear: absolute position. Circle: absolute
    position too, unless the field straddles the theta = 0 seam -- then the
    seam is moved to the opposite side of the ring (shift s), so the field is
    contiguous; pos_disp = (pos - s) mod C + s."""
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

def _passes(fld: dict, disp: dict, lin: dict, t_s: np.ndarray, dt_frames: np.ndarray,
            spike_frame: np.ndarray) -> list[dict]:
    """Every pass through one field on the moving-only frame sequence.

    A pass is a contiguous run of frames whose 1-D bin is in the field. Its
    entry / exit side is the side of the field ('low' / 'high' end of the
    displayed track) of the frame just before / after it, or 'session edge'
    if the run touches the first / last frame. A pass is a full traversal if
    it entered at one end and left at the other, or covered both end bins of
    the field (a field against the track end can only be left the way the
    animal came in). Passes shorter than MIN_PASS_S are dropped; all others
    are returned in time order."""
    in_field = np.isin(lin['lin_bin'], fld['bins'])
    if not in_field.any():
        return []
    pos   = disp['to_disp'](lin['pos'])
    mid   = 0.5 * (disp['lo'] + disp['hi'])
    edges = np.diff(np.r_[0, in_field.astype(np.int8), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    cum_dt = np.r_[0.0, np.cumsum(dt_frames)]
    spk = np.sort(spike_frame)
    last = in_field.size - 1

    def side(i: int) -> str:
        return 'session edge' if i < 0 or i > last else ('low' if pos[i] < mid else 'high')

    rows = []
    for a, b in zip(starts, ends):
        dur = float(cum_dt[b + 1] - cum_dt[a])
        if dur < MIN_PASS_S:
            continue
        p, pb = pos[a:b + 1], lin['lin_bin'][a:b + 1]
        entry, exit_ = side(a - 1), side(b + 1)
        through = {entry, exit_} == {'low', 'high'}
        full = through or all(e in pb for e in disp['end_bins'])
        reached = fld['peak_bin'] in pb or p.min() <= disp['peak'] <= p.max()
        category = ('Full traversal' if full else
                    'Partial, reached peak' if reached else 'Partial, missed peak')
        if through:
            up = entry == 'low'
            direction = ('CCW' if up else 'CW') if lin['circular'] else ('L->R' if up else 'R->L')
        else:
            direction = 'return' if entry == exit_ else ''
        rows.append({'start': int(a), 'end': int(b),
                     't_start_s': round(float(t_s[a]), 3), 't_end_s': round(float(t_s[b]), 3),
                     'duration_s': round(dur, 3), 'entry_side': entry, 'exit_side': exit_,
                     'direction': direction, 'category': category,
                     'reached_peak_bin': bool(reached),
                     'n_spikes': int(np.searchsorted(spk, b, 'right') - np.searchsorted(spk, a, 'left')),
                     'entry_pos_cm': round(_abs_cm(p[0], lin), 2),
                     'exit_pos_cm': round(_abs_cm(p[-1], lin), 2)})
    return rows


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_2d(ax, lin: dict, spike_frame: np.ndarray, bin_cm: float) -> None:
    """Trajectory + spikes in 2-D (rotated for Linear 90 / 270), fields outlined."""
    ext = lin['extent']
    ax.plot(lin['x2d_all'], lin['y2d_all'], color='0.6', linewidth=0.5, label='trajectory')
    ax.plot(lin['x2d'][spike_frame], lin['y2d'][spike_frame], '.', color=SPIKE_COLOR,
            markersize=3, label='spikes')
    for k, fld in enumerate(lin['fields']):
        color = FIELD_COLORS[k % len(FIELD_COLORS)]
        if lin['circular']:
            cx, cy, r, bw = lin['ring']
            for b0, b1 in _runs(fld['bins'], lin['n_lin'], circular=True):
                b1u = b1 if b1 >= b0 else b1 + lin['n_lin']
                th = np.linspace(b0 * bw, (b1u + 1) * bw, 50)
                ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), color=color, lw=6, alpha=0.45,
                        solid_capstyle='butt')
        else:
            _contour_field(ax, fld['mask2d'], bin_cm, color)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (cm)' + (', rotated 90° CCW' if lin['rotated'] else ''))
    ax.set_ylabel('y (cm)')
    ax.set_title('Trajectory + spikes' + (' (vertical track rotated to horizontal)'
                                         if lin['rotated'] else ''))
    ax.legend(loc='upper right', ncol=2, fontsize=8)


def _plot_profile(ax, lin: dict, spike_frame: np.ndarray, dt_frames: np.ndarray,
                  bin_cm: float, xlim: tuple) -> None:
    """1-D ratemap along the linearized track (raw steps + Gaussian-smoothed)."""
    n = lin['n_lin']
    occ = np.bincount(lin['lin_bin'], weights=dt_frames, minlength=n)
    spk = np.bincount(lin['lin_bin'][spike_frame], minlength=n).astype(float)
    valid = pc._valid_bin_mask(occ, pc._bin_visit_counts(lin['lin_bin'], n))
    raw = np.zeros(n)
    raw[valid] = spk[valid] / occ[valid]
    smooth = _smooth_1d(raw, valid, bin_cm, lin['circular'])
    centres = (np.arange(n) + 0.5) * bin_cm
    ax.step(centres, np.where(valid, raw, np.nan), where='mid', color='0.6', lw=0.8, label='raw')
    ax.plot(centres, np.where(valid, smooth, np.nan), color='black', lw=1.5,
            label=f'smoothed (sigma = {pc.GAUSSIAN_SIGMA_CM:g} cm)')
    for k, fld in enumerate(lin['fields']):
        _shade_runs(ax, _runs(fld['bins'], n, lin['circular']), bin_cm, lin['track_len'],
                    FIELD_COLORS[k % len(FIELD_COLORS)], alpha=0.2)
    ax.set_xlim(*xlim)
    ax.set_ylim(bottom=0)
    ax.set_ylabel('rate (Hz)')
    ax.set_title('Linearized ratemap (shaded: place fields)')
    ax.legend(loc='upper right', fontsize=8)


def _plot_pos_time(ax, lin: dict, t_s: np.ndarray, spike_frame: np.ndarray,
                   bin_cm: float, xlim: tuple) -> None:
    """Linearized position vs time (moving frames), spikes in red, time downward."""
    t_min = t_s / 60.0
    segs = _segments(lin['pos'], t_min, lin['track_len'] if lin['circular'] else None)
    ax.add_collection(LineCollection(segs, colors='0.6', linewidths=0.4))
    ax.plot(lin['pos'][spike_frame], t_min[spike_frame], '.', color=SPIKE_COLOR, markersize=2.5)
    for k, fld in enumerate(lin['fields']):
        _shade_runs(ax, _runs(fld['bins'], lin['n_lin'], lin['circular']), bin_cm,
                    lin['track_len'], FIELD_COLORS[k % len(FIELD_COLORS)])
    ax.set_xlim(*xlim)
    ax.set_ylim(t_min[-1], t_min[0])
    ax.set_ylabel('time (min)')
    ax.set_title('Linearized position vs time (moving frames; red = spikes)')


def _plot_raster(ax, ax_bar, fld: dict, disp: dict, passes: list[dict], lin: dict,
                 t_s: np.ndarray, spike_frame: np.ndarray, color: str, xlim: tuple,
                 summary: dict) -> None:
    """One row per pass (pass 1 at top): +/- CONTEXT_S s of trajectory in light
    gray, the pass coloured by category, spikes as ticks; spikes per pass as
    bars on ax_bar."""
    pos = disp['to_disp'](lin['pos'])
    period = lin['track_len'] if lin['circular'] else None
    spk = np.sort(spike_frame)
    ctx_segs, pass_segs, pass_cols = [], [], []
    sx, sy, cx, cy = [], [], [], []
    for i, p in enumerate(passes, start=1):
        a, b = p['start'], p['end']
        c0 = int(np.searchsorted(t_s, t_s[a] - CONTEXT_S, 'left'))
        c1 = int(np.searchsorted(t_s, t_s[b] + CONTEXT_S, 'right')) - 1
        ctx_segs += _segments(pos[c0:c1 + 1], np.full(c1 - c0 + 1, i), period)
        segs = _segments(pos[a:b + 1], np.full(b - a + 1, i), period)
        if b == a:                                     # single-frame pass: tiny horizontal tick
            segs = [np.array([[pos[a] - 0.3, i], [pos[a] + 0.3, i]])]
        pass_segs += segs
        pass_cols += [CATEGORY_COLORS[p['category']]] * len(segs)
        f_ctx = spk[np.searchsorted(spk, c0, 'left'):np.searchsorted(spk, c1, 'right')]
        inside = (f_ctx >= a) & (f_ctx <= b)
        sx += list(pos[f_ctx[inside]]);  sy += [i] * int(inside.sum())
        cx += list(pos[f_ctx[~inside]]); cy += [i] * int((~inside).sum())

    ax.axvspan(disp['lo'], disp['hi'], color=color, alpha=0.15, lw=0)
    ax.axvline(disp['peak'], color=color, ls='--', lw=1)
    ax.add_collection(LineCollection(ctx_segs, colors='0.8', linewidths=0.8))
    ax.add_collection(LineCollection(pass_segs, colors=pass_cols, linewidths=1.6))
    ax.plot(cx, cy, '|', color=CONTEXT_SPIKE_COLOR, markersize=6, markeredgewidth=1)
    ax.plot(sx, sy, '|', color=SPIKE_COLOR, markersize=7, markeredgewidth=1.4)

    n = len(passes)
    ax.set_xlim(*xlim)
    ax.set_ylim(n + 0.5, 0.5)
    step = max(1, int(np.ceil(n / 25)))
    ticks = list(range(1, n + 1, step))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{k}  ({passes[k - 1]['t_start_s'] / 60:.1f} min)" for k in ticks],
                       fontsize=7)
    ax.set_ylabel('pass # (chronological; start time)')
    ax.set_xlabel('linearized position (cm)')
    ax.set_title(f"Field {fld['field_number']}: {_abs_cm(disp['lo'], lin):.0f}–"
                 f"{_abs_cm(disp['hi'], lin):.0f} cm, peak "
                 f"{_abs_cm(disp['peak'], lin):.0f} cm  |  {n} passes "
                 f"({summary['n_full_traversals']} full)\n"
                 f">= 1 spike: {summary['n_passes_with_spike']}/{n}   "
                 f">= {MIN_SPIKES_PER_PASS} spikes: {summary['n_passes_with_min_spikes']}/{n} = "
                 f"{summary['lap_stability_score']}   "
                 f"Spearman rho(spikes, pass #) = {summary['spearman_rho']}",
                 fontsize=9)

    counts = np.array([p['n_spikes'] for p in passes])
    ax_bar.barh(np.arange(1, n + 1), counts, height=0.8,
                color=np.where(counts >= MIN_SPIKES_PER_PASS, SPIKE_COLOR, '0.6'))
    ax_bar.axvline(MIN_SPIKES_PER_PASS, color='black', ls=':', lw=1)
    ax_bar.set_ylim(n + 0.5, 0.5)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel('spikes in pass')
    ax_bar.set_xlim(left=0)


# ── Per-cell job ──────────────────────────────────────────────────────────────

def _field_summary(base: dict, fld: dict, disp: dict, passes: list[dict], lin: dict) -> dict:
    counts = np.array([p['n_spikes'] for p in passes])
    cats   = [p['category'] for p in passes]
    full   = counts[[c == 'Full traversal' for c in cats]] if passes else counts
    n = len(passes)
    rho, pval = (spearmanr(np.arange(n), counts) if n >= 3 and np.ptp(counts) > 0
                 else (np.nan, np.nan))
    return {**base, 'field_number': fld['field_number'],
            'field_lo_cm': round(_abs_cm(disp['lo'], lin), 2),
            'field_hi_cm': round(_abs_cm(disp['hi'], lin), 2),
            'peak_cm': round(_abs_cm(disp['peak'], lin), 2),
            'n_passes': n,
            'n_full_traversals': cats.count('Full traversal'),
            'n_partial_reached_peak': cats.count('Partial, reached peak'),
            'n_partial_missed_peak': cats.count('Partial, missed peak'),
            'n_session_edge_passes': sum('session edge' in (p['entry_side'], p['exit_side'])
                                         for p in passes),
            'n_passes_with_spike': int((counts > 0).sum()),
            'pct_passes_with_spike': round(100.0 * (counts > 0).mean(), 2) if n else np.nan,
            'n_passes_with_min_spikes': int((counts >= MIN_SPIKES_PER_PASS).sum()),
            'lap_stability_score':
                round(float((counts >= MIN_SPIKES_PER_PASS).mean()), 3) if n else np.nan,
            'lap_stability_full_traversals':
                round(float((full >= MIN_SPIKES_PER_PASS).mean()), 3) if full.size else np.nan,
            'mean_spikes_per_pass': round(float(counts.mean()), 3) if n else np.nan,
            'spearman_rho': round(float(rho), 3) if not np.isnan(rho) else np.nan,
            'spearman_p': round(float(pval), 4) if not np.isnan(pval) else np.nan}


def _run_cell(args) -> tuple[list[dict], list[dict]]:
    """Lap raster figure for one place cell. Returns (field_rows, pass_rows)."""
    idx, total, dirpath, unit = args
    session  = os.path.relpath(dirpath, place_cell_folder)
    ntt_path = os.path.join(dirpath, unit)
    arena, session_type = _arena_and_session(dirpath)
    with _print_lock:
        print(f'[{idx}/{total}]  {session}  |  {unit}')

    csv_path = _find_tracking_csv(dirpath)
    if csv_path is None:
        with _print_lock:
            print(f'  [SKIP] no single tracking file found in {dirpath}')
        return [], []

    bin_cm = pc.target_bin_cm
    metrics, ctx = pc.compute_metrics(csv_path, ntt_path, pc.arena_width_cm, bin_cm)
    if not ctx or not ctx['valid_mask'].any():
        with _print_lock:
            print(f'  [SKIP] no valid occupancy for {unit}')
        return [], []

    lin = _linearize(metrics, ctx, arena, session_type, bin_cm)
    if lin is None:
        with _print_lock:
            print(f'  [SKIP] no place field detected for {unit}')
        return [], []

    spike_frame = ctx['spike_frame']
    t_s = (ctx['t'] - ctx['t_all'][0]) * 1e-6          # s from session start
    C = lin['track_len']
    xlim = (0.0, C) if lin['circular'] else (0.0, max(C, pc.arena_width_cm))
    base = {'session': session, 'unit': unit, 'arena_type': arena,
            'session_type': session_type, 'rotated_90_ccw': lin['rotated'],
            'track_len_cm': round(C, 2)}

    per_field = []
    for fld in lin['fields']:
        disp = _field_display(fld, lin, bin_cm)
        passes = _passes(fld, disp, lin, t_s, ctx['dt_frames'], spike_frame)
        per_field.append((fld, disp, passes, _field_summary(base, fld, disp, passes, lin)))

    # ── Figure ────────────────────────────────────────────────────────────────
    n_f = len(per_field)
    max_passes = max(len(p) for _, _, p, _ in per_field)
    fig_w = max(14.0, 7.5 * n_f)
    h_2d = (fig_w * (lin['extent'][3] / lin['extent'][1]) + 0.8) if not lin['circular'] else 5.0
    h_2d = min(max(h_2d, 1.8), 6.0)
    h_raster = max(4.0, 0.13 * max_passes + 1.5)
    heights = [h_2d, 2.0, 3.5, h_raster]
    fig = Figure(figsize=(fig_w, sum(heights) + 1.6))
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(4, 1, height_ratios=heights, hspace=0.45)

    _plot_2d(fig.add_subplot(gs[0]), lin, spike_frame, bin_cm)
    _plot_profile(fig.add_subplot(gs[1]), lin, spike_frame, ctx['dt_frames'], bin_cm, xlim)
    _plot_pos_time(fig.add_subplot(gs[2]), lin, t_s, spike_frame, bin_cm, xlim)

    gs_r = gs[3].subgridspec(1, n_f, wspace=0.3)
    for k, (fld, disp, passes, summary) in enumerate(per_field):
        gs_f = gs_r[k].subgridspec(1, 2, width_ratios=[5, 1], wspace=0.05)
        ax, ax_bar = fig.add_subplot(gs_f[0]), fig.add_subplot(gs_f[1])
        color = FIELD_COLORS[k % len(FIELD_COLORS)]
        if not passes:
            ax.set_title(f"Field {fld['field_number']}: no passes")
            ax_bar.axis('off')
            continue
        f_xlim = (disp['shift'], disp['shift'] + C) if lin['circular'] else xlim
        _plot_raster(ax, ax_bar, fld, disp, passes, lin, t_s, spike_frame, color, f_xlim, summary)
        if lin['circular'] and disp['shift']:
            ax.set_xticks(np.sort(disp['to_disp'](np.arange(0.0, C, 50.0))))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{np.mod(v, C):.0f}'))
            ax.set_xlabel('linearized position (cm; seam moved opposite this field)')
        if k == 0:
            handles = [Line2D([], [], color=c, lw=2, label=l) for l, c in CATEGORY_COLORS.items()]
            handles += [Line2D([], [], color='0.8', lw=1, label=f'+/- {CONTEXT_S:g} s context'),
                        Line2D([], [], color=SPIKE_COLOR, marker='|', ls='none', ms=8,
                               label='spike in pass'),
                        Line2D([], [], color=CONTEXT_SPIKE_COLOR, marker='|', ls='none', ms=8,
                               label='spike in context')]
            ax.legend(handles=handles, loc='lower left', fontsize=7, framealpha=0.85)

    ntt_name = os.path.splitext(unit)[0]
    lin_note = ('Circle: position = arc length along the ring' if lin['circular'] else
                f"Linear {session_type}°" + (': vertical track rotated 90° CCW, position = L - y'
                                             if lin['rotated'] else ': position = x'))
    fig.suptitle(f'{session}  |  {ntt_name}  |  {lin_note}  |  {n_f} field(s), '
                 f'{metrics.get("n_spikes")} spikes (moving)\n'
                 f'Every pass through each field (>= {MIN_PASS_S:g} s), full or partial; '
                 f'lap stability = passes with >= {MIN_SPIKES_PER_PASS} spikes / passes',
                 fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.98, top=1 - 1.2 / (sum(heights) + 1.6), bottom=0.04)

    save_dir = os.path.join(dirpath, LAP_SUBDIR)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{ntt_name}_lap_raster.png')
    fig.savefig(save_path, dpi=150)
    with _print_lock:
        print(f'  [SAVED] {save_path}')

    field_rows = [s for _, _, _, s in per_field]
    pass_rows = [{**base, 'field_number': fld['field_number'], 'pass_number': i,
                  **{k: v for k, v in p.items() if k not in ('start', 'end')}}
                 for fld, _, passes, _ in per_field for i, p in enumerate(passes, start=1)]
    return field_rows, pass_rows


# ── Batch ─────────────────────────────────────────────────────────────────────

def run_all() -> None:
    """Lap raster for every Linear / Circle .ntt under place_cell_folder;
    writes output_excel (PerField, PerPass)."""
    jobs = []
    for dirpath, _, filenames in os.walk(place_cell_folder):
        if _arena_and_session(dirpath)[0] is None:
            continue
        jobs += [(dirpath, unit) for unit in sorted(filenames) if unit.lower().endswith('.ntt')]
    job_args = [(i, len(jobs), *job) for i, job in enumerate(jobs, start=1)]
    print(f'Found {len(jobs)} Linear / Circle place cell(s) in {place_cell_folder}\n')

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
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        pd.DataFrame(field_rows).to_excel(writer, sheet_name='PerField', index=False)
        pd.DataFrame(pass_rows).to_excel(writer,  sheet_name='PerPass',  index=False)

    n_cells = len({(f['session'], f['unit']) for f in field_rows})
    print(f'\nDone. {n_cells}/{len(jobs)} cell(s) plotted.')
    print(f'Results saved to {output_excel}')


if __name__ == "__main__":
    run_all()
