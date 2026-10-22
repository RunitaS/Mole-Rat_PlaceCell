# -*- coding: utf-8 -*-
"""
Peak-firing-bin sequences (Linear / Circle) and place-field silhouettes (Open)

Runs on the confirmed place cells that PlaceCell_Main copied into
Output_PlaceTrue (.ntt + tracking file per session folder, animal-ID folders
onwards, e.g. Fa1059/Linear/Day8/1_90/...). Every cell's full-session ratemap
is rebuilt with PlaceCell_Main's own functions (imported from
PLACECELL_MAIN_PATH, so its settings -- bin size, smoothing, Skaggs alpha,
field thresholds -- are the ones used here), once for each map in MAP_TYPES:
  'raw'                     RAW (unsmoothed) ratemap
  'gaussian_smooth'         Gaussian-smoothed ratemap
  'Skaggs_adaptive_binned'  Skaggs adaptive-binned ratemap
  'trajectory'              no ratemap, no binning: the cell's own trajectory
                            (gray line, full smoothed track as in
                            PlaceCell_Main's ratemap figure) with its spikes
                            (moving frames only) as red dots
Place fields are re-isolated on that same map with PlaceCell_Main's threshold
method (detect_place_fields_threshold_2d / _circular). The peak firing bin is
always taken from the RAW map, whatever the map type, so every map type of a
cell shares the same peak bin (and so the same peak position). 'trajectory'
takes its peak position and field start from the 'raw' map, so its cells are
stacked in the same order as the raw sequence.

Linear track (2-D grid map, 4-5 bins across the track)
  1. Sessions whose folder ends in _90 or _270 (e.g. '1_90', '3_270') run
     vertically: every map is rotated 90 deg counter-clockwise (np.rot90) before
     anything else, so all tracks run along x.
  2. The peak firing bin is the argmax of the 2-D RAW map; its x bin is the
     cell's linearised peak position.
  3. Plotted rows are the map averaged across the y bins (valid bins only).
  4. Full-track check: PlaceCell_Main sizes each session's grid from its own
     tracking span (x - x.min() ... x.max()), so a session where the animal
     did not cover the whole track gets a shorter map whose 0 is wherever the
     animal's lowest position was. A cell passes only if its map extent and
     its span of valid x bins (RAW map, i.e. occupancy only, so all map types
     keep the same cells) are both within LINEAR_LENGTH_TOL_CM of
     LINEAR_TRACK_LENGTH_CM (columns track_extent_cm, track_covered_cm,
     full_track); with EXCLUDE_PARTIAL_TRACKS, failing cells are left out of
     the sequence plot (still listed in PeakBins.xlsx, no sequence_rank).
Circle track
  The 1-D angular map of PlaceCell_Main (_build_ratemap_circular); linear
  position = bearing around the fitted ring centre, 0 deg = theta 0. Peak bin
  = argmax of the RAW angular map.
Sequence plot (one per arena and map type, all animals pooled)
  Cells sorted by peak position, cell 1 (peak nearest position 0) at the
  bottom; cells sharing a peak position are ordered by the first bin of the
  field holding the peak (earlier start first). For a Circle field crossing the
  0/360 seam, that first bin is counted back from the peak and so lies before 0.
  Each row is normalised to its own maximum (NORMALIZE_ROWS).
  'trajectory': each row holds the cell's trajectory, linearised position on
  x and the across-track coordinate (Linear: y after rotation; Circle: radial
  distance from the ring centre, r_min to r_max) squeezed into the row height.

Open arena (one plot per animal and map type)
  Every field of every cell drawn as a filled silhouette (animal colour,
  reduced alpha, thin outline); a dot marks the cell's peak firing bin, so only
  the field holding the peak gets one. Gray outline = bins visited in any of
  the animal's Open sessions.
  'trajectory': the trajectories of all the animal's Open sessions (gray)
  overlaid, with the spikes of all its cells as red dots.

Outputs (folder OUTPUT_SUBDIR inside Output_PlaceTrue):
  PeakBins.xlsx  'PeakBins'  one row per cell and ratemap type ('trajectory'
                             rows would repeat the 'raw' ones, so are left out)
                 'Settings'  parameters used for this run
  figures        <Arena>_PeakSequence_<map>.png/.svg,
                 Open_FieldSilhouettes_<animal>_<map>.png/.svg
                 (<map> = 'trajectory' for the trajectory figures)
"""

import os
import re
import threading
import importlib.util
import concurrent.futures
from datetime import datetime

import numpy as np
import pandas as pd

# Thread-safe Matplotlib imports
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb

# ── Directories (edit these per run) ──────────────────────────────────────────

# Output_PlaceTrue folder of the PlaceCell_Main run (root of the confirmed
# place cells, animal-ID folders onwards, e.g. Fa1059/Linear/Day8/1_90/...)
Output_PlaceTrue = r'X:/NMR_group_data/Runita/Analysis/Thesis/Corr_Data_SpkQltyFilt/SpikeQualityFilt/SpikeQualityFilt_v1_PlaceCellData/PC_True_SiSkaggsAdptBin_Alpha200'

# PlaceCell_Main version whose ratemap / field functions and settings are used
# -- should be the one that wrote Output_PlaceTrue.
PLACECELL_MAIN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'PlaceCell_Main_v21.py')

OUTPUT_SUBDIR = 'Results/PeakSequence_FieldSilhouettes'

# ── Settings ──────────────────────────────────────────────────────────────────
TRAJECTORY_MAP = 'trajectory'   # unbinned trajectory + spikes, not a ratemap
MAP_TYPES = ('raw', 'gaussian_smooth', 'Skaggs_adaptive_binned', TRAJECTORY_MAP)

ARENAS = ('Linear', 'Circle', 'Open')       # arena folder right below the animal-ID folder
ROTATE_SESSION_ANGLES = ('90', '270')       # Linear sessions rotated 90 deg counter-clockwise

LINEAR_TRACK_LENGTH_CM = 80.0   # physical Linear track length (0 to this, in cm)
LINEAR_LENGTH_TOL_CM   = 4.0    # allowed shortfall / overshoot of map extent and covered span
EXCLUDE_PARTIAL_TRACKS = True   # drop cells failing the full-track check from the sequence plot

NORMALIZE_ROWS = True     # sequence plots: each row / its own max (False: Hz, shared colour scale)
SEQUENCE_CMAP  = 'jet'

FIELD_ALPHA    = 0.4      # Open: silhouette fill alpha
OUTLINE_LW     = 0.5      # Open: silhouette outline width (pt)
OUTLINE_DARKEN = 0.45     # Open: outline / peak dot = animal colour darkened by this fraction
PEAK_DOT_SIZE  = 10       # Open: peak-bin dot size (pt^2)

TRAJ_LINE_COLOR  = '0.6'  # trajectory map: path colour
TRAJ_LW          = 0.3    # trajectory map: path width (pt)
TRAJ_SPIKE_COLOR = 'red'  # trajectory map: spike dot colour
TRAJ_SPIKE_SIZE  = 2      # trajectory map: spike dot size (pt^2)
TRAJ_ROW_FILL    = 0.9    # trajectory sequence: fraction of each row's height the track spans

FIG_FORMATS = ('png', 'svg')
FIG_DPI     = 300
MAX_WORKERS = 4

ANIMAL_COLORS = {
    'Fa8477': '#FBB4AE',  # Pastel1[0] red
    'Fa1059': '#B3CDE3',  # Pastel1[1] blue
    'Fa23BD': '#CCEBC5',  # Pastel1[2] green
    'Fa5834': '#DECBE4',  # Pastel1[3] purple
}


def animal_color(label, default='#888888'):
    """ANIMAL_COLORS entry for an animal label, matched on the ID so that e.g.
    'Fa1059', '1059Test' and 'FA1059' all resolve; `default` if no ID matches."""
    label = str(label).upper()
    for animal_id, color in ANIMAL_COLORS.items():
        if animal_id[2:].upper() in label:
            return color
    return default


# Saved to the 'Settings' sheet of the output Excel.
RUN_CONFIG_VARS = [
    'Output_PlaceTrue', 'PLACECELL_MAIN_PATH', 'OUTPUT_SUBDIR', 'MAP_TYPES',
    'ROTATE_SESSION_ANGLES', 'LINEAR_TRACK_LENGTH_CM', 'LINEAR_LENGTH_TOL_CM',
    'EXCLUDE_PARTIAL_TRACKS', 'NORMALIZE_ROWS', 'SEQUENCE_CMAP',
    'FIELD_ALPHA', 'OUTLINE_LW', 'OUTLINE_DARKEN', 'PEAK_DOT_SIZE',
    'TRAJ_LINE_COLOR', 'TRAJ_LW', 'TRAJ_SPIKE_COLOR', 'TRAJ_SPIKE_SIZE', 'TRAJ_ROW_FILL',
]
# PlaceCell_Main settings that shape the maps and fields used here.
PCM_CONFIG_VARS = [
    'target_bin_cm', 'fps', 'COORD_UNITS', 'GAUSSIAN_SIGMA_CM',
    'use_min_occ_s', 'min_occ_s', 'use_min_visits', 'min_visits',
    'IMMOBILITY_SPEED_CMS', 'POS_SMOOTH_SIGMA_SMP', 'POS_JUMP_THRESH_CMS',
    'ADAPTIVE_MAX_RSQ', 'ADAPTIVE_MIN_OCC_S', 'SKAGGS_ALPHA',
    'MIN_FIELD_SIZE_BINS', 'METHOD2_RATE_THRESHOLD_FRAC', 'RADIAL_RANGE_CLIP_PCTILE',
]


def _load_placecell_main(path: str):
    spec = importlib.util.spec_from_file_location('PlaceCell_Main', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load PlaceCell_Main from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pcm = _load_placecell_main(PLACECELL_MAIN_PATH)
bin_cm = pcm.target_bin_cm

_MAP_LABELS = {
    'raw':                    'Raw (unsmoothed)',
    'gaussian_smooth':       f'Gaussian-smoothed (sigma = {pcm.GAUSSIAN_SIGMA_CM:g} cm)',
    'Skaggs_adaptive_binned': f'Skaggs adaptive-binned (alpha = {pcm.SKAGGS_ALPHA:g})',
    TRAJECTORY_MAP:           'Trajectory + spikes (unbinned)',
}

_print_lock = threading.Lock()


# ── Unit discovery ────────────────────────────────────────────────────────────

def _session_angle(dirpath: str) -> str | None:
    """Linear-track session angle from the folder name, e.g. '3_270' -> '270'."""
    m = re.search(r'_(0|90|180|270)$', os.path.basename(os.path.normpath(dirpath)))
    return m.group(1) if m else None


def _tracking_files(filenames: list[str]) -> list[str]:
    """Tracking file(s) in a session folder, picked as in PlaceCell_Main."""
    tracking = [f for f in filenames if f.lower().endswith(('.csv', '.xlsx'))]
    if pcm.COORD_UNITS == 'cm':
        return [f for f in tracking if f.lower().endswith('_cm.csv')]
    return [f for f in tracking if not f.lower().endswith('_cm.csv')]


def _find_units(root: str) -> list[dict]:
    """Every .ntt under root/<animal>/<arena>/... whose folder holds exactly one
    tracking file. Only animal-ID folders are walked, so output folders of
    other scripts inside Output_PlaceTrue are skipped."""
    units = []
    for animal in sorted(os.listdir(root)):
        animal_dir = os.path.join(root, animal)
        if animal not in pcm.ANIMAL_NAMES or not os.path.isdir(animal_dir):
            continue
        for dirpath, _, filenames in os.walk(animal_dir):
            arena_folder = os.path.relpath(dirpath, animal_dir).split(os.sep)[0].lower()
            arena = next((a for a in ARENAS if a.lower() == arena_folder), None)
            tracking = _tracking_files(filenames)
            ntt_files = sorted(f for f in filenames if f.lower().endswith('.ntt'))
            if arena is None or len(tracking) != 1 or not ntt_files:
                continue
            angle = _session_angle(dirpath)
            for ntt_file in ntt_files:
                units.append({
                    'animal':   animal,
                    'arena':    arena,
                    'session':  os.path.relpath(dirpath, root),
                    'unit':     ntt_file,
                    'csv_path': os.path.join(dirpath, tracking[0]),
                    'ntt_path': os.path.join(dirpath, ntt_file),
                    'session_angle': angle,
                    'rotated_90ccw': arena == 'Linear' and angle in ROTATE_SESSION_ANGLES,
                })
    return units


# ── Per-cell peak bins ────────────────────────────────────────────────────────

def _base_record(unit: dict, map_type: str) -> dict:
    return {k: unit[k] for k in ('animal', 'arena', 'session', 'unit',
                                 'session_angle', 'rotated_90ccw')} | {'map_type': map_type}


def _grid_record(unit: dict, metrics: dict, ctx: dict, map_type: str) -> dict:
    """Peak bin (from the RAW map) and fields (from the `map_type` map) of one
    2-D grid map (Linear / Open). Linear maps of the ROTATE_SESSION_ANGLES
    sessions are rotated before any other step."""
    rate, pi_occ = pcm._ratemap(map_type, ctx['spike_map'], ctx['occ_map'], ctx['valid_mask'],
                                bin_cm, fr_raw=ctx['fr_raw'], fr_smooth=ctx['fr_smooth'])
    maps = {'valid_mask':   ctx['valid_mask'],
            'fr_firing':    ctx['fr_firing'],
            'firing_valid': ctx['firing_valid'],
            'fr_field':     rate,
            'field_valid':  pcm._map_valid(pi_occ, ctx['valid_mask']),
            'fr_raw':       ctx['fr_raw'],
            'raw_valid':    pcm._map_valid(ctx['occ_map'], ctx['valid_mask'])}
    if unit['rotated_90ccw']:
        # np.rot90 on an [x, y] map = counter-clockwise rotation in the x-y plane
        maps = {k: np.rot90(v) for k, v in maps.items()}
    rate, valid = maps['fr_field'], maps['field_valid']
    raw, raw_valid = maps['fr_raw'], maps['raw_valid']
    n_bins_x, n_bins_y = rate.shape

    if unit['arena'] == 'Linear' and n_bins_x < n_bins_y:
        with _print_lock:
            print(f'  [WARN] Linear map still vertical ({n_bins_x} x {n_bins_y} bins) '
                  f'after rotation step: {unit["session"]} | {unit["unit"]}')

    fields, _ = pcm.detect_place_fields_threshold_2d(
        metrics, maps | {'n_bins_x': n_bins_x, 'n_bins_y': n_bins_y}, bin_cm)
    field_masks = [pcm._field_mask_from_bin_coords(f['bin_coords'], n_bins_x, n_bins_y)
                   for f in fields]

    # Peak firing bin from the RAW map for every map type.
    px, py = (int(v) for v in np.unravel_index(np.argmax(np.where(raw_valid, raw, -np.inf)),
                                               raw.shape))
    peak_field = next((f for f, m in zip(fields, field_masks) if m[px, py]), None)
    start_bin = peak_field['bbox_x_min'] if peak_field is not None else px

    rec = _base_record(unit, map_type) | {
        'n_bins_x': n_bins_x, 'n_bins_y': n_bins_y,
        'peak_bin_x': px, 'peak_bin_y': py,
        'peak_rate_hz': round(float(raw[px, py]), 4),
        'map_rate_at_peak_hz': round(float(rate[px, py]), 4) if valid[px, py] else np.nan,
        'n_fields': len(fields),
        'peak_in_field': peak_field is not None,
        'field_masks': field_masks, 'peak_xy_cm': ((px + 0.5) * bin_cm, (py + 0.5) * bin_cm),
        'valid': valid,
    }
    if unit['arena'] == 'Linear':
        # Linearised position = x; plotted row = mean across the track's y bins.
        n_valid = valid.sum(axis=1)
        row_sum = np.where(valid, rate, 0.0).sum(axis=1)
        profile = np.full(n_bins_x, np.nan)
        np.divide(row_sum, n_valid, out=profile, where=n_valid > 0)
        # Full-track check: grid extent and the span of valid x bins must both
        # match the physical track length. RAW valid bins (occupancy only), so
        # the check gives the same answer for every map type.
        valid_x = np.flatnonzero(raw_valid.any(axis=1))
        extent_cm = n_bins_x * bin_cm
        covered_cm = (valid_x[-1] - valid_x[0] + 1) * bin_cm if valid_x.size else 0.0
        full_track = (abs(extent_cm - LINEAR_TRACK_LENGTH_CM) <= LINEAR_LENGTH_TOL_CM
                      and covered_cm >= LINEAR_TRACK_LENGTH_CM - LINEAR_LENGTH_TOL_CM)
        rec |= {'track_extent_cm': extent_cm,
                'track_covered_cm': covered_cm,
                'full_track': full_track,
                'pos_unit': 'cm',
                'peak_pos': (px + 0.5) * bin_cm,
                'field_start_bin': start_bin,
                'field_start_pos': start_bin * bin_cm,
                'profile': profile,
                'edges': np.arange(n_bins_x + 1) * bin_cm}
    return rec


def _circle_record(unit: dict, ctx: dict, cc: dict, map_type: str) -> dict:
    """Peak bin (from the RAW map) and fields (from the `map_type` map) of one
    Circle-track angular map."""
    n_theta = cc['n_bins_theta']
    spike_map = np.bincount(cc['beh_bt'][ctx['spike_frame']], minlength=n_theta).astype(np.float64)
    rate, pi_occ = pcm._ratemap_circular(map_type, spike_map, cc['occ_map'],
                                         cc['valid_mask'], bin_cm)
    valid = pcm._map_valid(pi_occ, cc['valid_mask'])
    fields, _ = pcm.detect_place_fields_threshold_circular(
        cc | {'fr_field': rate, 'field_valid': valid}, bin_cm)

    # Peak firing bin from the RAW map for every map type.
    raw, raw_occ = pcm._ratemap_circular('raw', spike_map, cc['occ_map'], cc['valid_mask'], bin_cm)
    raw_valid = pcm._map_valid(raw_occ, cc['valid_mask'])
    pk = int(np.argmax(np.where(raw_valid, raw, -np.inf)))
    peak_field_bins = next((bts for bts in ({int(b) for b in f['bin_coords'].split(';')}
                                            for f in fields) if pk in bts), None)
    # First field bin = walk back from the peak along the field, wrapping
    # around the seam; it can lie before 0 (negative) in linear coordinates.
    n_back = 0
    if peak_field_bins is not None:
        while n_back < n_theta - 1 and (pk - n_back - 1) % n_theta in peak_field_bins:
            n_back += 1
    bin_deg = 360.0 / n_theta

    return _base_record(unit, map_type) | {
        'n_bins_theta': n_theta, 'peak_bin_theta': pk,
        'peak_rate_hz': round(float(raw[pk]), 4),
        'map_rate_at_peak_hz': round(float(rate[pk]), 4) if valid[pk] else np.nan,
        'n_fields': len(fields),
        'peak_in_field': peak_field_bins is not None,
        'pos_unit': 'deg',
        'peak_pos': (pk + 0.5) * bin_deg,
        'field_start_bin': pk - n_back,
        'field_start_pos': (pk - n_back) * bin_deg,
        'profile': np.where(valid, rate, np.nan),
        'edges': np.arange(n_theta + 1) * bin_deg,
    }


def _trajectory_record(unit: dict, metrics: dict, ctx: dict, cc: dict | None) -> dict:
    """Unbinned trajectory + spike positions of one cell. Peak position and
    field start come from the 'raw' record, so the cell sorts exactly as in the
    raw sequence. Path = full smoothed track (ctx['x_cm_all'], as in
    PlaceCell_Main's ratemap figure); spikes = moving-frame spikes, the ones
    behind every ratemap. Grid coordinates match the binned maps (Linear
    ROTATE_SESSION_ANGLES sessions rotated the same way)."""
    x, y = ctx['x_cm_all'], ctx['y_cm_all']
    sx, sy = ctx['x_cm'][ctx['spike_frame']], ctx['y_cm'][ctx['spike_frame']]

    if unit['arena'] == 'Circle':
        assert cc is not None
        rec = _circle_record(unit, ctx, cc, 'raw')

        def _polar(px, py):
            theta = np.degrees(np.mod(np.arctan2(py - cc['cy'], px - cc['cx']), 2.0 * np.pi))
            return theta, np.hypot(px - cc['cx'], py - cc['cy'])

        traj, spikes = _polar(x, y), _polar(sx, sy)
        cross_range = (cc['r_min'], cc['r_max'])
    else:
        rec = _grid_record(unit, metrics, ctx, 'raw')
        if unit['rotated_90ccw']:
            # Continuous form of np.rot90 on an [x, y] map: (x, y) -> (Y - y, x),
            # Y = the unrotated map's y extent.
            y_ext = ctx['n_bins_y'] * bin_cm
            x, y, sx, sy = y_ext - y, x, y_ext - sy, sx
        traj, spikes = (x, y), (sx, sy)
        cross_range = (0.0, rec['n_bins_y'] * bin_cm)

    return rec | {'map_type': TRAJECTORY_MAP, 'traj': traj, 'spikes': spikes,
                  'cross_range': cross_range}


def _process_unit(args) -> list[dict]:
    idx, total, unit = args
    with _print_lock:
        print(f'[{idx}/{total}]  {unit["session"]}  |  {unit["unit"]}')
    try:
        metrics, ctx = pcm.compute_metrics(unit['csv_path'], unit['ntt_path'],
                                           pcm.arena_width_cm, bin_cm)
        if not ctx or not ctx['valid_mask'].any():
            with _print_lock:
                print(f'  [SKIP] no valid bins: {unit["session"]} | {unit["unit"]}')
            return []
        ratemap_types = [mt for mt in MAP_TYPES if mt != TRAJECTORY_MAP]
        cc = None
        if unit['arena'] == 'Circle':
            cc = pcm._build_ratemap_circular(ctx, bin_cm)
            recs = [_circle_record(unit, ctx, cc, mt) for mt in ratemap_types]
        else:
            recs = [_grid_record(unit, metrics, ctx, mt) for mt in ratemap_types]
        if TRAJECTORY_MAP in MAP_TYPES:
            recs.append(_trajectory_record(unit, metrics, ctx, cc))
        return recs
    except Exception as e:
        with _print_lock:
            print(f'  ERROR in {unit["session"]} | {unit["unit"]}: {e}')
        return []


# ── Plots ─────────────────────────────────────────────────────────────────────

def _save_fig(fig: Figure, save_stem: str) -> None:
    for ext in FIG_FORMATS:
        fig.savefig(f'{save_stem}.{ext}', dpi=FIG_DPI)
    print(f'  [SAVED] {save_stem}.{"/".join(FIG_FORMATS)}')


def _plot_sequence(recs: list[dict], arena: str, map_type: str, save_stem: str) -> None:
    """Stacked peak-position sequence; `recs` already sorted (cell 1 first)."""
    n = len(recs)
    rows = []
    for r in recs:
        prof = r['profile']
        if NORMALIZE_ROWS:
            peak = np.nanmax(prof) if np.isfinite(prof).any() else 0.0
            prof = prof / peak if peak > 0 else prof
        rows.append(np.ma.masked_invalid(prof))
    vmax = 1.0 if NORMALIZE_ROWS else max(float(row.max()) for row in rows)

    fig = Figure(figsize=(6, 7))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    for k, (r, row) in enumerate(zip(recs, rows), start=1):
        mesh = ax.pcolormesh(r['edges'], [k - 0.5, k + 0.5], row[None, :], cmap=SEQUENCE_CMAP,
                             vmin=0.0, vmax=vmax, shading='flat', rasterized=True)

    ax.set_ylim(0.5, n + 0.5)
    ax.set_yticks([1, n] if n > 1 else [1])
    ax.set_ylabel('Cell')
    ax.set_xlim(0, max(r['edges'][-1] for r in recs))
    if arena == 'Circle':
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlabel('Linearized position (deg)')
    else:
        ax.set_xlabel('Linearized position (cm)')
    fig.colorbar(mesh, ax=ax,
                 label='Normalized firing rate' if NORMALIZE_ROWS else 'Firing rate (Hz)')
    ax.set_title(f'{arena} track, {_MAP_LABELS[map_type]}\n'
                 f'n = {n} place cells', fontsize=10)
    fig.tight_layout()
    _save_fig(fig, save_stem)


def _darken(color: str, frac: float) -> tuple:
    return tuple(c * (1.0 - frac) for c in to_rgb(color))


def _plot_open(recs: list[dict], animal: str, map_type: str, save_stem: str) -> None:
    """All fields of all cells as overlapping silhouettes, dot at each cell's peak
    bin; gray outline = bins visited (valid) in any of these sessions."""
    fill = animal_color(animal)
    edge = _darken(fill, OUTLINE_DARKEN)

    fig = Figure(figsize=(7, 7))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    def _outline(mask: np.ndarray, **kwargs) -> None:
        # Contour at 0.5 between bin centres of a zero-padded mask = mask edge.
        xc = (np.arange(-1, mask.shape[0] + 1) + 0.5) * bin_cm
        yc = (np.arange(-1, mask.shape[1] + 1) + 0.5) * bin_cm
        m = np.pad(mask, 1).astype(float).T
        if 'alpha' in kwargs:
            ax.contourf(xc, yc, m, levels=[0.5, 1.5], colors=[fill], alpha=kwargs.pop('alpha'))
        ax.contour(xc, yc, m, levels=[0.5], **kwargs)

    for r in recs:
        for mask in r['field_masks']:
            _outline(mask, alpha=FIELD_ALPHA, colors=[edge], linewidths=OUTLINE_LW)

    n_x = max(r['n_bins_x'] for r in recs)
    n_y = max(r['n_bins_y'] for r in recs)
    visited = np.zeros((n_x, n_y), dtype=bool)
    for r in recs:
        visited[:r['n_bins_x'], :r['n_bins_y']] |= r['valid']
    _outline(visited, colors=['0.6'], linewidths=0.8)

    peaks = np.array([r['peak_xy_cm'] for r in recs])
    ax.scatter(peaks[:, 0], peaks[:, 1], s=PEAK_DOT_SIZE, color=edge, linewidths=0, zorder=3)

    x_max, y_max = n_x * bin_cm, n_y * bin_cm
    ax.set_xlim(-0.02 * x_max, 1.02 * x_max)
    ax.set_ylim(-0.02 * y_max, 1.02 * y_max)
    ax.set_aspect('equal')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    n_fields = sum(r['n_fields'] for r in recs)
    ax.set_title(f'{animal}, Open arena, {_MAP_LABELS[map_type]}\n'
                 f'n = {len(recs)} place cells, {n_fields} fields', fontsize=10)
    fig.tight_layout()
    _save_fig(fig, save_stem)


def _plot_sequence_trajectory(recs: list[dict], arena: str, save_stem: str) -> None:
    """Stacked unbinned trajectories, one row per cell (same order as
    _plot_sequence): linearised position on x, across-track coordinate squeezed
    into the row; spikes as dots. Circle paths are broken at the 0/360 seam."""
    n = len(recs)
    fig = Figure(figsize=(6, 7))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    def _to_row(v: np.ndarray, lo: float, hi: float, k: int) -> np.ndarray:
        frac = np.clip((v - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.full_like(v, 0.5)
        return k + TRAJ_ROW_FILL * (frac - 0.5)

    for k, r in enumerate(recs, start=1):
        lo, hi = r['cross_range']
        pos, cross = (np.asarray(a, dtype=float) for a in r['traj'])
        row_y = _to_row(cross, lo, hi, k)
        if arena == 'Circle':
            jumps = np.flatnonzero(np.abs(np.diff(pos)) > 180.0) + 1
            pos, row_y = np.insert(pos, jumps, np.nan), np.insert(row_y, jumps, np.nan)
        ax.plot(pos, row_y, color=TRAJ_LINE_COLOR, linewidth=TRAJ_LW, rasterized=True)
        sp_pos, sp_cross = r['spikes']
        ax.scatter(sp_pos, _to_row(np.asarray(sp_cross, dtype=float), lo, hi, k),
                   s=TRAJ_SPIKE_SIZE, color=TRAJ_SPIKE_COLOR, linewidths=0, zorder=3,
                   rasterized=True)

    ax.set_ylim(0.5, n + 0.5)
    ax.set_yticks([1, n] if n > 1 else [1])
    ax.set_ylabel('Cell')
    ax.set_xlim(0, max(r['edges'][-1] for r in recs))
    if arena == 'Circle':
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlabel('Linearized position (deg)')
    else:
        ax.set_xlabel('Linearized position (cm)')
    ax.set_title(f'{arena} track, {_MAP_LABELS[TRAJECTORY_MAP]}\n'
                 f'n = {n} place cells', fontsize=10)
    fig.tight_layout()
    _save_fig(fig, save_stem)


def _plot_open_trajectory(recs: list[dict], animal: str, save_stem: str) -> None:
    """Trajectories of all of the animal's Open sessions (gray, each session
    drawn once) overlaid, with the spikes of all its cells as dots."""
    fig = Figure(figsize=(7, 7))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    drawn_sessions = set()
    for r in recs:
        if r['session'] not in drawn_sessions:
            ax.plot(*r['traj'], color=TRAJ_LINE_COLOR, linewidth=TRAJ_LW, rasterized=True)
            drawn_sessions.add(r['session'])
    for r in recs:
        ax.scatter(*r['spikes'], s=TRAJ_SPIKE_SIZE, color=TRAJ_SPIKE_COLOR, linewidths=0,
                   zorder=3, rasterized=True)

    x_max = max(r['n_bins_x'] for r in recs) * bin_cm
    y_max = max(r['n_bins_y'] for r in recs) * bin_cm
    ax.set_xlim(-0.02 * x_max, 1.02 * x_max)
    ax.set_ylim(-0.02 * y_max, 1.02 * y_max)
    ax.set_aspect('equal')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_title(f'{animal}, Open arena, {_MAP_LABELS[TRAJECTORY_MAP]}\n'
                 f'n = {len(recs)} place cells, {len(drawn_sessions)} sessions', fontsize=10)
    fig.tight_layout()
    _save_fig(fig, save_stem)


# ── Main ──────────────────────────────────────────────────────────────────────

_PLOT_ONLY_KEYS = ('field_masks', 'peak_xy_cm', 'profile', 'edges', 'valid',
                   'traj', 'spikes', 'cross_range')

_TABLE_COLUMNS = ['animal', 'arena', 'session', 'unit', 'session_angle', 'rotated_90ccw',
                  'map_type', 'sequence_rank',
                  'peak_rate_hz', 'map_rate_at_peak_hz', 'peak_pos', 'pos_unit', 'field_start_bin', 'field_start_pos',
                  'n_fields', 'peak_in_field',
                  # Linear full-track check
                  'track_extent_cm', 'track_covered_cm', 'full_track',
                  # 2-D grid (Linear after rotation / Open)
                  'n_bins_x', 'n_bins_y', 'peak_bin_x', 'peak_bin_y',
                  # Circle
                  'n_bins_theta', 'peak_bin_theta']


def run(root: str, output_dir: str, units: list[dict] | None = None) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    if units is None:
        units = _find_units(root)
    print(f'Found {len(units)} place cell(s) under {root}\n')

    job_args = [(i, len(units), u) for i, u in enumerate(units, start=1)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        records = [rec for recs in executor.map(_process_unit, job_args) for rec in recs]

    partial = {}
    for r in records:
        if r['arena'] == 'Linear' and not r['full_track']:
            partial.setdefault(r['session'], (r['track_extent_cm'], r['track_covered_cm'], set()))[2].add(r['unit'])
    if partial:
        print(f'\n[WARN] {len(partial)} Linear session(s) do not cover 0-{LINEAR_TRACK_LENGTH_CM:g} cm '
              f'(+/- {LINEAR_LENGTH_TOL_CM:g} cm)'
              + ('; their cells are excluded from the sequence plots:' if EXCLUDE_PARTIAL_TRACKS else ':'))
        for session, (extent_cm, covered_cm, cells) in sorted(partial.items()):
            print(f'  {session}: map extent {extent_cm:g} cm, valid span {covered_cm:g} cm, '
                  f'{len(cells)} cell(s)')
        print()

    for map_type in MAP_TYPES:
        for arena in ('Linear', 'Circle'):
            recs = [r for r in records if r['arena'] == arena and r['map_type'] == map_type
                    and not (arena == 'Linear' and EXCLUDE_PARTIAL_TRACKS and not r['full_track'])]
            if not recs:
                continue
            recs.sort(key=lambda r: (r['peak_pos'], r['field_start_pos']))
            for rank, r in enumerate(recs, start=1):
                r['sequence_rank'] = rank
            save_stem = os.path.join(output_dir, f'{arena}_PeakSequence_{map_type}')
            if map_type == TRAJECTORY_MAP:
                _plot_sequence_trajectory(recs, arena, save_stem)
            else:
                _plot_sequence(recs, arena, map_type, save_stem)

        open_recs = [r for r in records if r['arena'] == 'Open' and r['map_type'] == map_type]
        for animal in sorted({r['animal'] for r in open_recs}):
            animal_recs = [r for r in open_recs if r['animal'] == animal]
            save_stem = os.path.join(output_dir, f'Open_FieldSilhouettes_{animal}_{map_type}')
            if map_type == TRAJECTORY_MAP:
                _plot_open_trajectory(animal_recs, animal, save_stem)
            else:
                _plot_open(animal_recs, animal, map_type, save_stem)

    # 'trajectory' records repeat the 'raw' ones (peak, fields), so stay out of the table.
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in _PLOT_ONLY_KEYS}
                       for r in records if r['map_type'] != TRAJECTORY_MAP],
                      columns=_TABLE_COLUMNS)
    settings = ([{'parameter': 'script_name', 'value': os.path.basename(__file__)},
                 {'parameter': 'run_timestamp', 'value': datetime.now().isoformat(timespec='seconds')}]
                + [{'parameter': name, 'value': str(globals()[name])} for name in RUN_CONFIG_VARS]
                + [{'parameter': f'PlaceCell_Main.{name}', 'value': str(getattr(pcm, name))}
                   for name in PCM_CONFIG_VARS])
    excel_path = os.path.join(output_dir, 'PeakBins.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='PeakBins', index=False)
        pd.DataFrame(settings).to_excel(writer, sheet_name='Settings', index=False)
    print(f'\n[SAVED] {excel_path}')

    n_outside = int((~df['peak_in_field'].astype(bool)).sum())
    if n_outside:
        print(f'{n_outside} cell/map row(s) have their peak bin outside every detected field '
              f'(peak_in_field = False); their field_start_bin is the peak bin itself.')
    return df


if __name__ == '__main__':
    run(Output_PlaceTrue, os.path.join(Output_PlaceTrue, OUTPUT_SUBDIR))
