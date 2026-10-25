# -*- coding: utf-8 -*-
"""
Temporal stability maps for place cells – V2

V2 adds a statistical comparison of each cell's mean and median temporal
stability across its field bins (mean_pct_across_bins, median_pct_across_bins)
and of its temporal stability at the peak bin of the raw ratemap
(pct_at_peak_bin; see step 5 below) -- each metric tested separately, with its
own box plot and histogram -- at the three levels of
ArenaSessionType_StatsComparison_v10.py, with the same tests and plot style:
  1. session types within each arena (Open: Cntrl/Rotate/Zero, Linear:
     0/90/180/270, Circle: Stnd/NoRot/Rot),
  2. arena types (Circle / Linear / Open, all session types pooled),
  3. animal-controlled arena types (only animals recorded in both arenas).
Test: Kruskal-Wallis omnibus + pairwise Mann-Whitney U, Holm-Bonferroni
corrected (one value per cell and session, a continuous %). Chi-square is not
used: pooling visits into a fired / not-fired table would treat every visit as
independent, although visits are nested within cells. As in v10, a unit recorded
in several sessions of a day is counted once per session (rows treated as
independent; each table's 'model_note' reports how many cells repeat).
Outputs, all in STATS_DIR:
    SessionType\\<arena>\\                        box plots (+ Histograms\\)
    ArenaType\\                                   box plots (+ Histograms\\)
    AnimalControlledArenaType\\<A>_vs_<B>\\        box plots (+ Histograms\\)
    AnimalControlledArenaType\\<A>_vs_<B>\\ExampleMaps\\   best / mean / worst
        N_EXAMPLE_MAPS cells per arena by EXAMPLE_METRIC (copies of their maps,
        labelled with animal, session and unit)
    AnimalControlledArenaType\\AnimalControlled_Stats.xlsx
    TemporalStability_Stats.xlsx, TemporalStability_AllStats.csv
With RUN_MAPS = False the maps are not rebuilt; the stats use the PerCell sheet
of an existing output_excel.

V4 adds occupancy-weighted summaries across field bins, so that a bin the animal
spent little time in (e.g. 1 visit, 1 spike -> 100%) counts less than a
well-sampled bin: occ_wmean_pct_across_bins / occ_wmedian_pct_across_bins are
the mean / median of the per-bin % weighted by each bin's dwell time
(occupancy_s, moving-only frames). The weighted median is the smallest bin %
at which the cumulative dwell time reaches half the total. Both are added to
the PerCell / PerField sheets and to STATS_METRICS.
Two further summaries refer the weights to the whole ratemap (all valid bins)
instead of the field bins only (rescaling the weights alone would cancel in the
weighted mean, so these differ from it by construction):
  occ_wmean_pct_mapnorm   sum over field bins of % x dwell, divided by the dwell
                          of ALL valid bins (= occ_wmean_pct_across_bins x the
                          fraction of map dwell spent in the field); PerCell + PerField
  occ_wmean_pct_all_bins  occupancy-weighted mean % over EVERY visited valid bin,
                          in or out of a field; PerCell only

V6 adds temporal stability per PASS through a field (lap stability), next to
the per-bin stability above. A pass is one contiguous run of moving-only
frames inside the field: it starts when the animal enters any bin of the field
and ends when it leaves the field through any bin (moving between bins of the
same field does not end it). Passes shorter than MIN_PASS_S, and passes cut by
the start / end of the session (no observed entry or exit), are dropped. A pass
'fired' if >= MIN_SPIKES_PER_PASS spikes were assigned to its frames.
  lap_stability_score   n passes with >= MIN_SPIKES_PER_PASS spikes / n passes
                        (0-1), per field (PerField). On PerCell it pools the
                        passes through every field of the cell.
  mean_lap_stability_across_fields   PerCell: mean of the cell's per-field
                        lap_stability_score (every field weighted equally)
Both PerCell columns are in STATS_METRICS; each field's score is written on its
field in the stability panel and listed in the figure title.

For every .ntt under place_cell_folder (all already classified as place cells,
so no place-cell check is done here):
  1. Rebuild the session trajectory, spike positions and bin grid with
     PlaceCell_Main_v21.compute_metrics (same cleaning, smoothing, immobility
     filter and spike assignment as the run that detected the fields).
  2. Detect its place fields with PlaceCell_Main_v21's threshold method
     (detect_place_fields_threshold_2d / _circular) on those maps -- no Excel needed.
  3. For every field bin, split the trajectory into visits: a visit is one
     contiguous run of tracking frames inside that bin (entry -> exit), the
     same definition PlaceCell_Main_v21 uses for its visit count.
  4. Per field bin: % of visits during which the cell fired >= 1 spike
     (a spike belongs to a visit if it was assigned to one of its frames).
  5. Peak bin: the valid bin with the highest rate in the RAW (unsmoothed)
     ratemap (Circle track: raw angular map), whether or not it lies in a
     field; its % of visits with >= 1 spike is pct_at_peak_bin (white star
     on the raw and stability maps).

Figure (one per cell, saved next to its .ntt in TEMPORAL_STABILITY_SUBDIR):
  trajectory + spikes | raw ratemap | Gaussian-smoothed ratemap | temporal stability map
Stability map: field bins coloured by %; visited non-field bins gray; never-visited
bins blank. Circle-track fields are angular (1-D) bins, so on the Circle track
the stability map is drawn as a ring (see _plot_stability_ring).

Analysis settings (bin size, smoothing, filters, field thresholds, COORD_UNITS, ...)
come from PlaceCell_Main_v21; only the directories below are set here.
"""

import os
import concurrent.futures
import threading

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.image import imread
from mpl_toolkits.axes_grid1 import make_axes_locatable

import PlaceCell_Main_v21 as pc
import ArenaSessionType_StatsComparison_v10 as asc   # group parsing, tests, plots


#Add occupancy control
# ── Directories (edit these per run) ──────────────────────────────────────────

# Place-cell folder (Output_PlaceTrue of a PlaceCell_Main_v21 run): every .ntt under
# it is analysed; each session folder also holds its tracking file.
place_cell_folder = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Alpha200'

# Per-cell / per-field / per-bin results table.
output_excel = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Alpha200\Results\TemporalStability.xlsx'

# Figures go in this subfolder of each session folder (next to the .ntt).
TEMPORAL_STABILITY_SUBDIR = 'temporal_stability'

# Statistical comparison across session types / arenas / animal-controlled arenas.
STATS_DIR  = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Alpha200\Results\TemporalStabilityStats'
STATS_XLSX = os.path.join(STATS_DIR, 'TemporalStability_Stats.xlsx')
STATS_CSV  = os.path.join(STATS_DIR, 'TemporalStability_AllStats.csv')

# True: build the maps + output_excel, then run the stats on them.
# False: skip the maps and run the stats on the PerCell sheet of output_excel.
RUN_MAPS = True

# ── Settings ──────────────────────────────────────────────────────────────────

MIN_VISIT_S     = 0.1      # visits shorter than this (s) are ignored; 0 = keep every visit
MIN_PASS_S      = 0.1      # field passes shorter than this (s) are ignored; 0 = keep every pass
MIN_SPIKES_PER_PASS = 5    # a pass counts as stable with >= this many spikes
STABILITY_CMAP  = 'jet'
NON_FIELD_COLOR = '0.75'   # gray for visited bins outside every place field

# Per-cell metrics compared (PerCell columns). Each gets a box plot + histogram.
STATS_METRICS = {
    'mean_pct_across_bins':   'Mean temporal stability across field bins (% visits with >= 1 spike)',
    'median_pct_across_bins': 'Median temporal stability across field bins (% visits with >= 1 spike)',
    'occ_wmean_pct_across_bins':   'Occupancy-weighted mean temporal stability across field bins (% visits with >= 1 spike)',
    'occ_wmedian_pct_across_bins': 'Occupancy-weighted median temporal stability across field bins (% visits with >= 1 spike)',
    'occ_wmean_pct_mapnorm':  'Field temporal stability weighted by dwell, normalised to whole-map dwell (%)',
    'occ_wmean_pct_all_bins': 'Occupancy-weighted mean temporal stability across all visited map bins (% visits with >= 1 spike)',
    'pct_at_peak_bin':       'Temporal stability at the raw-ratemap peak bin (% visits with >= 1 spike)',
    'lap_stability_score':   f'Lap stability, passes pooled over all fields '
                             f'(fraction of passes with >= {MIN_SPIKES_PER_PASS} spikes)',
    'mean_lap_stability_across_fields':
                             f'Mean lap stability across fields '
                             f'(fraction of passes with >= {MIN_SPIKES_PER_PASS} spikes)',
}
asc.CONTINUOUS_TEST = 'KW'   # Kruskal-Wallis + pairwise Mann-Whitney U (Holm)

# Example maps for each animal-controlled comparison: per arena, the
# N_EXAMPLE_MAPS cells with the highest (best) and lowest (worst) EXAMPLE_METRIC
# and the N_EXAMPLE_MAPS closest to the arena's mean (mean), saved in
# EXAMPLE_MAPS_SUBDIR next to that comparison's box plots.
EXAMPLE_METRIC      = 'mean_pct_across_bins'
N_EXAMPLE_MAPS      = 3
EXAMPLE_MAPS_SUBDIR = 'ExampleMaps'

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


def _visit_stats(flat_bin: np.ndarray, dt_frames: np.ndarray, spike_frame: np.ndarray,
                 n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin visit counts on the moving-only frame sequence.

    A visit starts whenever the bin index differs from the previous frame and
    lasts until the animal leaves that bin. A visit 'fired' if >= 1 spike was
    assigned (nearest frame) to one of its frames. Visits shorter than
    MIN_VISIT_S are dropped.

    Returns (n_visits, n_visits_fired), each of length n_bins.
    """
    if flat_bin.size == 0:
        return np.zeros(n_bins, dtype=np.int64), np.zeros(n_bins, dtype=np.int64)

    starts = np.flatnonzero(np.r_[True, flat_bin[1:] != flat_bin[:-1]])
    ends   = np.r_[starts[1:] - 1, flat_bin.size - 1]
    visit_bin = flat_bin[starts]
    visit_dur = np.add.reduceat(dt_frames, starts)

    spk = np.sort(spike_frame)
    n_spk = (np.searchsorted(spk, ends, side='right')
             - np.searchsorted(spk, starts, side='left'))

    keep = visit_dur >= MIN_VISIT_S
    n_visits = np.bincount(visit_bin[keep], minlength=n_bins)
    n_fired  = np.bincount(visit_bin[keep & (n_spk > 0)], minlength=n_bins)
    return n_visits, n_fired


def _pass_stats(in_field: np.ndarray, dt_frames: np.ndarray,
                spike_frame: np.ndarray) -> tuple[int, int]:
    """Passes through one field on the moving-only frame sequence.

    `in_field` is True on frames inside any bin of the field. A pass is one
    contiguous run of True frames (entry -> exit through any bin). Passes
    shorter than MIN_PASS_S, or touching the first / last frame (entry or exit
    not observed), are dropped. A pass is stable if >= MIN_SPIKES_PER_PASS
    spikes were assigned (nearest frame) to its frames.

    Returns (n_passes, n_passes_with_min_spikes).
    """
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


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Smallest value at which the cumulative weight (values sorted ascending)
    reaches half the total weight. NaN if the weights sum to 0."""
    total = weights.sum()
    if total <= 0:
        return np.nan
    order = np.argsort(values, kind='stable')
    cum_w = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum_w, 0.5 * total)])


def _field_bins(bin_coords: str, is_circle: bool) -> list[tuple[int, ...]]:
    """bin_coords string -> list of bin tuples: (bx, by) for 2-D, (bt,) for Circle."""
    if is_circle:
        return [(int(b),) for b in bin_coords.split(';')]
    return [tuple(int(v) for v in pair.split('-')) for pair in bin_coords.split(';')]


def _contour_field(ax, field_mask: np.ndarray, bin_cm: float, upsample: int = 8) -> None:
    """Outline a 2-D field mask along bin edges on a cm-extent imshow. The mask
    is padded (so fields touching the grid edge still close) and upsampled (so
    the contour follows bin edges instead of cutting corners between centres)."""
    up = np.kron(np.pad(field_mask.astype(float), 1), np.ones((upsample, upsample)))
    xc = ((np.arange(up.shape[0]) + 0.5) / upsample - 1) * bin_cm
    yc = ((np.arange(up.shape[1]) + 0.5) / upsample - 1) * bin_cm
    ax.contour(xc, yc, up.T, levels=[0.5], colors='black', linewidths=1.2)


def _colorbar(fig, ax, mappable, label: str) -> None:
    """Colorbar matched to the axes height (thin linear-track maps included)."""
    cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.1)
    fig.colorbar(mappable, cax=cax, label=label)


def _mark_peak(ax, x: float, y: float) -> None:
    """White star on the raw-ratemap peak bin (x, y in the axes' data units)."""
    ax.plot(x, y, marker='*', color='white', markeredgecolor='black', markersize=14,
            linestyle='none')


def _plot_rate(ax, fig,rate: np.ndarray, valid: np.ndarray, extent: list, title: str):
    im = ax.imshow(np.ma.masked_where(~valid, rate).T, origin='lower', extent=extent,
                   cmap='jet', interpolation='nearest')
    _colorbar(fig, ax, im, 'firing rate (Hz)')
    ax.set_title(title)
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')


def _plot_stability_2d(ax, fig, pct: np.ndarray, field_mask: np.ndarray,
                       valid: np.ndarray, extent: list):
    gray = np.ma.masked_where(~(valid & ~field_mask), np.ones_like(pct))
    ax.imshow(gray.T, origin='lower', extent=extent,
              cmap=ListedColormap([NON_FIELD_COLOR]), interpolation='nearest')
    im = ax.imshow(np.ma.masked_where(~field_mask, pct).T, origin='lower', extent=extent,
                   cmap=STABILITY_CMAP, vmin=0, vmax=100, interpolation='nearest')
    _colorbar(fig, ax, im, '% visits with >= 1 spike')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')


def _plot_stability_ring(ax, fig, pct: np.ndarray, field_mask: np.ndarray,
                         valid: np.ndarray, ctx_circ: dict):
    """Angular analogue of _plot_stability_2d, drawn as a ring like
    PlaceCell_Main_v21._plot_ratemap_ring."""
    n_theta, bw = ctx_circ['n_bins_theta'], ctx_circ['bin_width_rad']
    r_min, r_max = ctx_circ['r_min'], ctx_circ['r_max']
    theta_mesh, r_mesh = np.meshgrid(np.arange(n_theta + 1) * bw, [r_min, r_max])

    gray = np.ma.masked_where(~(valid & ~field_mask), np.ones(n_theta)).reshape(1, n_theta)
    ax.pcolormesh(theta_mesh, r_mesh, gray, cmap=ListedColormap([NON_FIELD_COLOR]),
                  shading='flat')
    field = np.ma.masked_where(~field_mask, pct).reshape(1, n_theta)
    mesh = ax.pcolormesh(theta_mesh, r_mesh, field, cmap=STABILITY_CMAP,
                         vmin=0, vmax=100, shading='flat')
    fig.colorbar(mesh, ax=ax, label='% visits with >= 1 spike', fraction=0.046, pad=0.1)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_ylim(0, r_max * 1.1)
    ax.set_rticks([])


# ── Per-cell job ──────────────────────────────────────────────────────────────

def _run_cell(args) -> tuple[dict | None, list[dict], list[dict]]:
    """Temporal stability map for one place cell.

    Returns (cell_row, field_rows, bin_rows); cell_row is None if skipped.
    """
    idx, total, dirpath, unit = args
    session  = os.path.relpath(dirpath, place_cell_folder)
    ntt_path = os.path.join(dirpath, unit)
    with _print_lock:
        print(f'[{idx}/{total}]  {session}  |  {unit}')

    csv_path = _find_tracking_csv(dirpath)
    if csv_path is None:
        with _print_lock:
            print(f'  [SKIP] no single tracking file found in {dirpath}')
        return None, [], []

    metrics, ctx = pc.compute_metrics(csv_path, ntt_path, pc.arena_width_cm, pc.target_bin_cm)
    if not ctx or not ctx['valid_mask'].any():
        with _print_lock:
            print(f'  [SKIP] no valid occupancy for {unit}')
        return None, [], []

    bin_cm      = pc.target_bin_cm
    n_bx, n_by  = ctx['n_bins_x'], ctx['n_bins_y']
    is_circle   = pc._is_circle_arena(dirpath)
    arena_type  = 'Circle' if is_circle else '2D'
    spike_frame = ctx['spike_frame']

    # ── Place fields + the bin sequence / grid they are defined on ───────────
    # Same detection as PlaceCell_Main_v21._run_job (Circle track: angular bins).
    if is_circle:
        ctx_circ  = pc._build_ratemap_circular(ctx, bin_cm)
        fields, _ = pc.detect_place_fields_threshold_circular(ctx_circ, bin_cm)
        flat_bin  = ctx_circ['beh_bt']
        grid      = (ctx_circ['n_bins_theta'],)
        valid     = ctx_circ['valid_mask']
        occ       = ctx_circ['occ_map']
    else:
        ctx_circ  = None
        fields, _ = pc.detect_place_fields_threshold_2d(metrics, ctx, bin_cm)
        flat_bin  = ctx['beh_bx'] * n_by + ctx['beh_by']
        grid      = (n_bx, n_by)
        valid     = ctx['valid_mask']
        occ       = ctx['occ_map']

    if not fields:
        with _print_lock:
            print(f'  [SKIP] no place field detected for {unit}')
        return None, [], []

    field_mask = np.zeros(grid, dtype=bool)
    field_id   = np.zeros(grid, dtype=np.int64)       # field_number per bin, 0 = no field
    for f in fields:
        for bn in _field_bins(f['bin_coords'], is_circle):
            field_mask[bn] = True
            field_id[bn]   = f['field_number']

    # ── Visits per bin + fraction with >= 1 spike ────────────────────────────
    n_visits, n_fired = _visit_stats(flat_bin, ctx['dt_frames'], spike_frame, int(np.prod(grid)))
    n_visits = n_visits.reshape(grid)
    n_fired  = n_fired.reshape(grid)
    spk_bins = np.bincount(flat_bin[spike_frame], minlength=int(np.prod(grid))).reshape(grid)

    pct = np.full(grid, np.nan)
    has_visits = field_mask & (n_visits > 0)
    pct[has_visits] = 100.0 * n_fired[has_visits] / n_visits[has_visits]

    # ── Temporal stability at the raw-ratemap peak bin ───────────────────────
    # Peak over all valid bins of the RAW (unsmoothed) ratemap, first bin on ties.
    # Circle track: the raw angular map (spikes / occupancy per angular bin, as in
    # PlaceCell_Main_v21._ratemap_circular). The peak need not lie in a field.
    if is_circle:
        raw_rate = np.zeros(grid)
        np.divide(spk_bins, occ, out=raw_rate, where=valid)
    else:
        raw_rate = ctx['fr_raw']
    peak_bn = tuple(int(b) for b in
                    np.unravel_index(np.argmax(np.where(valid, raw_rate, -np.inf)), grid))
    peak_pct = (100.0 * n_fired[peak_bn] / n_visits[peak_bn]) if n_visits[peak_bn] else np.nan

    # ── Tables ────────────────────────────────────────────────────────────────
    bin_rows = []
    for bn in zip(*np.nonzero(field_mask)):
        row = {'session': session, 'unit': unit, 'arena_type': arena_type,
               'field_number': int(field_id[bn])}
        if is_circle:
            row['bin_theta'] = int(bn[0])
        else:
            row['bin_x'], row['bin_y'] = int(bn[0]), int(bn[1])
        row.update({'n_visits': int(n_visits[bn]), 'n_visits_with_spike': int(n_fired[bn]),
                    'pct_visits_with_spike': round(float(pct[bn]), 2) if has_visits[bn] else np.nan,
                    'n_spikes': int(spk_bins[bn]), 'occupancy_s': round(float(occ[bn]), 3)})
        bin_rows.append(row)

    occ_map_total = float(occ[valid].sum())                # dwell (s) over the whole ratemap

    def _summary(mask: np.ndarray) -> dict:
        p = pct[mask & has_visits]
        w = occ[mask & has_visits]                         # dwell time (s) per bin
        v, fired = int(n_visits[mask].sum()), int(n_fired[mask].sum())
        n_pass, n_pass_ok = _pass_stats(mask.ravel()[flat_bin], ctx['dt_frames'], spike_frame)
        return {'n_field_bins': int(mask.sum()),
                'mean_pct_across_bins':   round(float(np.mean(p)), 2)   if p.size else np.nan,
                'median_pct_across_bins': round(float(np.median(p)), 2) if p.size else np.nan,
                'occ_wmean_pct_across_bins':
                    round(float(np.sum(p * w) / w.sum()), 2) if w.sum() > 0 else np.nan,
                'occ_wmedian_pct_across_bins': round(_weighted_median(p, w), 2),
                'occ_wmean_pct_mapnorm':
                    round(float(np.sum(p * w) / occ_map_total), 2) if occ_map_total > 0 else np.nan,
                'occupancy_s_field_bins': round(float(occ[mask].sum()), 3),
                'n_visits_total': v, 'n_visits_with_spike_total': fired,
                'pooled_pct_visits_with_spike': round(100.0 * fired / v, 2) if v else np.nan,
                'n_passes': n_pass, 'n_passes_with_min_spikes': n_pass_ok,
                'lap_stability_score': round(n_pass_ok / n_pass, 3) if n_pass else np.nan}

    field_rows = [{'session': session, 'unit': unit, 'arena_type': arena_type,
                   'field_number': int(fn), **_summary(field_id == fn)}
                  for fn in sorted(np.unique(field_id[field_mask]))]
    cell_row = {'session': session, 'unit': unit, 'arena_type': arena_type,
                'n_fields': len(field_rows), 'n_spikes': metrics.get('n_spikes'),
                'peak_fr': metrics.get('peak_fr'), 'mean_fr': metrics.get('mean_fr'),
                'sir': metrics.get('sir'), **_summary(field_mask)}
    field_laps = [fr['lap_stability_score'] for fr in field_rows
                  if not np.isnan(fr['lap_stability_score'])]
    cell_row['mean_lap_stability_across_fields'] = (round(float(np.mean(field_laps)), 3)
                                                    if field_laps else np.nan)

    # Whole-map occupancy-weighted stability: every visited valid bin, in or out of a field.
    map_bins = valid & (n_visits > 0)
    p_map = 100.0 * n_fired[map_bins] / n_visits[map_bins]
    w_map = occ[map_bins]
    cell_row.update({'n_map_bins_visited': int(map_bins.sum()),
                     'occupancy_s_map': round(occ_map_total, 3),
                     'occ_wmean_pct_all_bins':
                         round(float(np.sum(p_map * w_map) / w_map.sum()), 2)
                         if w_map.sum() > 0 else np.nan})
    if is_circle:
        cell_row['peak_bin_theta'] = peak_bn[0]
    else:
        cell_row['peak_bin_x'], cell_row['peak_bin_y'] = peak_bn
    cell_row.update({'peak_raw_fr': round(float(raw_rate[peak_bn]), 3),
                     'peak_bin_in_field': bool(field_mask[peak_bn]),
                     'peak_bin_n_visits': int(n_visits[peak_bn]),
                     'peak_bin_n_visits_with_spike': int(n_fired[peak_bn]),
                     'pct_at_peak_bin': round(float(peak_pct), 2)})

    # ── Figure ────────────────────────────────────────────────────────────────
    # Wide arenas (linear track) stack the four panels vertically; others sit in one row.
    aspect = n_by / n_bx
    if aspect < 0.5:
        n_rows, n_cols = 4, 1
        fig = Figure(figsize=(14, 4 * (14 * aspect + 1.5) + 1))
    else:
        n_rows, n_cols = 1, 4
        fig = Figure(figsize=(26, 6.5))
    FigureCanvasAgg(fig)
    extent = [0, n_bx * bin_cm, 0, n_by * bin_cm]

    ax_traj = fig.add_subplot(n_rows, n_cols, 1)
    ax_traj.plot(ctx['x_cm_all'], ctx['y_cm_all'], color='0.6', linewidth=0.5, label='trajectory')
    ax_traj.plot(ctx['x_cm'][spike_frame], ctx['y_cm'][spike_frame], '.', color='red',
                 markersize=3, label='spikes')
    ax_traj.set_xlim(extent[0], extent[1])
    ax_traj.set_ylim(extent[2], extent[3])
    ax_traj.set_aspect('equal', adjustable='box')
    ax_traj.set_xlabel('x (cm)')
    ax_traj.set_ylabel('y (cm)')
    ax_traj.set_title('Trajectory + spikes')
    ax_traj.legend(loc='upper right', ncol=2, fontsize=8)
    make_axes_locatable(ax_traj).append_axes('right', size='3%', pad=0.1).axis('off')  # align with maps

    ax_raw = fig.add_subplot(n_rows, n_cols, 2)
    _plot_rate(ax_raw, fig, ctx['fr_raw'], ctx['valid_mask'], extent, 'Raw ratemap')
    ax_smooth = fig.add_subplot(n_rows, n_cols, 3)
    _plot_rate(ax_smooth, fig, ctx['fr_smooth'], ctx['valid_mask'], extent,
               f'Gaussian-smoothed (sigma = {pc.GAUSSIAN_SIGMA_CM:g} cm)')

    stab_title = 'Temporal stability (field bins)'
    if is_circle:
        assert ctx_circ is not None
        ax_stab = fig.add_subplot(n_rows, n_cols, 4, projection='polar')
        _plot_stability_ring(ax_stab, fig, pct, field_mask, valid, ctx_circ)
        _mark_peak(ax_stab, (peak_bn[0] + 0.5) * ctx_circ['bin_width_rad'],
                   0.5 * (ctx_circ['r_min'] + ctx_circ['r_max']))
        stab_title += ', angular'
    else:
        ax_stab = fig.add_subplot(n_rows, n_cols, 4)
        _plot_stability_2d(ax_stab, fig, pct, field_mask, valid, extent)
        for fn in np.unique(field_id[field_mask]):
            for ax in (ax_raw, ax_smooth, ax_stab):
                _contour_field(ax, field_id == fn, bin_cm)
        for ax in (ax_raw, ax_stab):
            _mark_peak(ax, (peak_bn[0] + 0.5) * bin_cm, (peak_bn[1] + 0.5) * bin_cm)
    ax_stab.set_title(f'{stab_title}\n* = raw-ratemap peak bin: {cell_row["pct_at_peak_bin"]}% '
                      f'of {cell_row["peak_bin_n_visits"]} visits')

    # Lap stability label on each field (2-D: bin centroid; Circle: circular-mean angle).
    for fr in field_rows:
        bins = np.argwhere(field_id == fr['field_number'])
        if is_circle:
            assert ctx_circ is not None
            ang = (bins[:, 0] + 0.5) * ctx_circ['bin_width_rad']
            xy = (np.angle(np.mean(np.exp(1j * ang))) % (2 * np.pi),
                  0.5 * (ctx_circ['r_min'] + ctx_circ['r_max']))
        else:
            xy = tuple((bins.mean(axis=0) + 0.5) * bin_cm)
        ax_stab.annotate(f"F{fr['field_number']}: {fr['n_passes_with_min_spikes']}/{fr['n_passes']}"
                         f" = {fr['lap_stability_score']}", xy=xy, ha='center', va='center',
                         fontsize=8, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=0.8))

    ntt_name = os.path.splitext(unit)[0]
    lap_line = '  |  '.join(f"F{fr['field_number']}: {fr['n_passes_with_min_spikes']}/"
                            f"{fr['n_passes']} passes = {fr['lap_stability_score']}"
                            for fr in field_rows)
    fig.suptitle(f"{session}  |  {ntt_name}\n"
                 f"{cell_row['n_fields']} field(s), {cell_row['n_field_bins']} bins  |  "
                 f"mean across bins = {cell_row['mean_pct_across_bins']}%, "
                 f"median = {cell_row['median_pct_across_bins']}%, "
                 f"occ-weighted mean = {cell_row['occ_wmean_pct_across_bins']}%, "
                 f"occ-weighted median = {cell_row['occ_wmedian_pct_across_bins']}%, "
                 f"pooled = {cell_row['pooled_pct_visits_with_spike']}% of "
                 f"{cell_row['n_visits_total']} visits with >= 1 spike\n"
                 f"Lap stability (passes with >= {MIN_SPIKES_PER_PASS} spikes): {lap_line}  ||  "
                 f"all fields pooled = {cell_row['lap_stability_score']}, "
                 f"mean across fields = {cell_row['mean_lap_stability_across_fields']}",
                 fontsize=10)
    fig.tight_layout()

    save_dir = os.path.join(dirpath, TEMPORAL_STABILITY_SUBDIR)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{ntt_name}_temporal_stability.png')
    fig.savefig(save_path, dpi=150)
    with _print_lock:
        print(f'  [SAVED] {save_path}')

    return cell_row, field_rows, bin_rows


# ── Statistical comparison (same tests / plots as ArenaSessionType_StatsComparison_v10) ──

def _compare(grp, df: pd.DataFrame, plots_dir: str) -> dict:
    """Compare every STATS_METRICS metric across the groups of `grp`
    (asc.Grouping): Kruskal-Wallis + Holm-corrected Mann-Whitney U, a box plot
    in `plots_dir` and a histogram in `plots_dir`\\Histograms. `df` must already
    be restricted to the cells to compare. Returns {table_name: table}."""
    name = grp.title or grp.noun.capitalize()
    print(f'\n{"=" * 70}\n{name} -- comparing {grp.noun}: '
          f'{[asc._label(g) for g in grp.order]}\n{"=" * 70}')
    print('Place cells per group:')
    print(df[grp.col].value_counts().reindex(grp.order).fillna(0).astype(int))
    if not len(df):
        print('No place cells; skipping.')
        return {}

    hist_dir = os.path.join(plots_dir, 'Histograms')
    os.makedirs(hist_dir, exist_ok=True)
    all_desc, all_omnibus, all_posthoc = [], [], []
    for col, label in STATS_METRICS.items():
        desc_df, omnibus, posthoc_df = asc.compare_groups(df, col, label, grp)
        all_desc.append(desc_df)
        all_omnibus.append(omnibus)
        if len(posthoc_df):
            all_posthoc.append(posthoc_df)
        asc.plot_metric(df, col, label, os.path.join(plots_dir, f'TemporalStability_{col}.png'), grp,
                        stats_lines=asc._continuous_stats_lines(desc_df, omnibus, posthoc_df, grp),
                        posthoc_df=posthoc_df)
        asc.plot_histogram_by_group(df, col, label,
                                    os.path.join(hist_dir, f'Hist_TemporalStability_{col}.png'), grp)

    omnibus_df = pd.DataFrame(all_omnibus)
    asc._print_significant(omnibus_df, name, 45, 16)
    print(f'Plots saved to {plots_dir}')
    return {
        'Desc':    pd.concat(all_desc, ignore_index=True),
        'Omnibus': omnibus_df,
        'PostHoc': pd.concat(all_posthoc, ignore_index=True) if all_posthoc else pd.DataFrame(),
    }


def _map_path(session: str, unit: str) -> str:
    """Temporal stability figure written by _run_cell for this cell."""
    return os.path.join(place_cell_folder, session, TEMPORAL_STABILITY_SUBDIR,
                        f'{os.path.splitext(unit)[0]}_temporal_stability.png')


def _pick_examples(d: pd.DataFrame, n: int) -> list[tuple[str, int, pd.Series]]:
    """(category, rank, row) for the n best, n closest-to-mean and n worst rows
    of d by EXAMPLE_METRIC. A cell is picked once: Best first, then Worst, then
    Mean from the rest (small groups give fewer maps)."""
    d = d.dropna(subset=[EXAMPLE_METRIC]).sort_values(
        [EXAMPLE_METRIC, 'session', 'unit'], ascending=[False, True, True], kind='stable')
    best  = d.head(n)
    worst = d.drop(best.index).tail(n).iloc[::-1]
    rest  = d.drop(best.index.union(worst.index))
    mean  = rest.loc[(rest[EXAMPLE_METRIC] - d[EXAMPLE_METRIC].mean()).abs()
                     .sort_values(kind='stable').index[:n]]
    return [(cat, rank, row) for cat, part in (('Best', best), ('Mean', mean), ('Worst', worst))
            for rank, (_, row) in enumerate(part.iterrows(), start=1)]


def _save_example_maps(grp, df: pd.DataFrame, out_dir: str) -> None:
    """Copy the best / mean / worst N_EXAMPLE_MAPS temporal stability maps of
    each group of `grp` into `out_dir`, each with a banner naming the animal,
    session, unit and its EXAMPLE_METRIC value. Old PNGs in `out_dir` are
    removed first so a re-run never leaves stale examples."""
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.lower().endswith('.png'):
            os.remove(os.path.join(out_dir, f))

    for g in grp.order:
        d = df[df[grp.col] == g]
        group_mean = d[EXAMPLE_METRIC].mean()
        for cat, rank, row in _pick_examples(d, N_EXAMPLE_MAPS):
            src = _map_path(row['session'], row['unit'])
            if not os.path.isfile(src):
                print(f'  [SKIP example] map not found: {src}')
                continue
            img = imread(src)
            h_in, w_in = img.shape[0] / 150, img.shape[1] / 150   # maps are saved at dpi=150
            banner_in = 1.0
            fig = Figure(figsize=(w_in, h_in + banner_in))
            FigureCanvasAgg(fig)
            ax = fig.add_axes((0, 0, 1, h_in / (h_in + banner_in)))
            ax.imshow(img)
            ax.axis('off')
            ntt_name = os.path.splitext(row['unit'])[0]
            fig.text(0.5, 1 - 0.5 * banner_in / (h_in + banner_in),
                     f'{asc._label(g)}  |  {cat} {rank} of {N_EXAMPLE_MAPS}  |  '
                     f'Animal: {row["animal"]}  |  Session: {row["session"]}  |  Unit: {ntt_name}\n'
                     f'{EXAMPLE_METRIC} = {row[EXAMPLE_METRIC]:.2f}%  '
                     f'({asc._label(g)} mean = {group_mean:.2f}%)',
                     ha='center', va='center', fontsize=14, fontweight='bold')
            session_tag = '_'.join(p for p in str(row['session']).replace('/', '\\').split('\\') if p)
            fig.savefig(os.path.join(out_dir, f'{g}_{cat}{rank}_{session_tag}_{ntt_name}.png'),
                        dpi=150)
    print(f'Example maps saved to {out_dir}')


def _save_tables(tables: dict, xlsx_path: str) -> None:
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        for sheet_name, tbl in tables.items():
            tbl.to_excel(writer, sheet_name=sheet_name[:31], index=False)   # Excel cap: 31 chars
    print(f'Statistics saved to {xlsx_path}')


def run_stats(cell_df: pd.DataFrame) -> None:
    """Session types within each arena, arena types, and animal-controlled
    arena types (asc.ANIMAL_CONTROLLED_COMPARISONS), all saved under STATS_DIR."""
    missing = [c for c in STATS_METRICS if c not in cell_df.columns]
    if missing:
        raise KeyError(f'PerCell table has no column(s) {missing}; '
                       f'rebuild it with RUN_MAPS = True.')
    df = asc._tag_groups(cell_df, 'place cell(s)')   # arena_type, session_type, animal, cell_id
    print('\nPlace cells by arena and session type:')
    print(pd.crosstab(df['arena_type'], df['session_type'].fillna('unrecognized'), margins=True))

    os.makedirs(STATS_DIR, exist_ok=True)
    all_tables = {}

    # ── 1. Session types within each arena ──────────────────────────────────
    for arena in asc.ARENA_TYPES:
        grp = asc.session_grouping(arena)
        sub = df[(df['arena_type'] == arena) & df['session_type'].isin(grp.order)]
        for name, tbl in _compare(grp, sub, os.path.join(STATS_DIR, 'SessionType', arena)).items():
            if len(tbl) and 'arena_type' not in tbl.columns:
                tbl.insert(0, 'arena_type', arena)
            all_tables[f'Ses{arena}_{name}'] = tbl

    # ── 2. Arena types as a whole (all session types pooled) ────────────────
    for name, tbl in _compare(asc.ARENA_GROUPING, df, os.path.join(STATS_DIR, 'ArenaType')).items():
        all_tables[f'Arena_{name}'] = tbl

    # ── 3. Arena types, only animals recorded in all compared arenas ────────
    ac_dir = os.path.join(STATS_DIR, asc.ANIMAL_CONTROLLED_DIR)
    ac_tables = {}
    for name, (animals, arenas) in asc.ANIMAL_CONTROLLED_COMPARISONS.items():
        grp = asc.animal_controlled_grouping(animals, arenas)
        wanted = {a.lower() for a in animals}
        sub = df[df['animal'].astype(str).str.lower().isin(wanted) & df['arena_type'].isin(grp.order)]
        print(f'\n{name}: place cells per animal and arena:')
        print(pd.crosstab(sub['animal'], sub['arena_type'], margins=True))
        tag = 'AC' + ''.join(grp.order)   # e.g. 'ACCircleOpen'
        for tname, tbl in _compare(grp, sub, os.path.join(ac_dir, name)).items():
            if len(tbl):
                tbl.insert(0, 'animals', ', '.join(animals))
            ac_tables[f'{tag}_{tname}'] = tbl
        if len(sub):
            _save_example_maps(grp, sub, os.path.join(ac_dir, name, EXAMPLE_MAPS_SUBDIR))
    if ac_tables:
        os.makedirs(ac_dir, exist_ok=True)
        _save_tables(ac_tables, os.path.join(ac_dir, 'AnimalControlled_Stats.xlsx'))
    all_tables.update(ac_tables)

    # ── Save all statistics (one workbook + one long-format CSV) ────────────
    _save_tables(all_tables, STATS_XLSX)
    parts = [tbl.assign(table=sheet_name)[['table', *tbl.columns]]
             for sheet_name, tbl in all_tables.items() if len(tbl)]
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(STATS_CSV, index=False, encoding='utf-8-sig')
        print(f'All statistics also saved to {STATS_CSV}')


# ── Batch ─────────────────────────────────────────────────────────────────────

def run_maps() -> pd.DataFrame:
    """Temporal stability map for every .ntt under place_cell_folder; writes
    output_excel and returns its PerCell table."""
    jobs = []
    for dirpath, _, filenames in os.walk(place_cell_folder):
        jobs += [(dirpath, unit) for unit in sorted(filenames) if unit.lower().endswith('.ntt')]
    job_args = [(i, len(jobs), *job) for i, job in enumerate(jobs, start=1)]
    print(f'Found {len(jobs)} place cell(s) in {place_cell_folder}\n')

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

    cell_rows  = [r[0] for _, r in results if r[0] is not None]
    field_rows = [f for _, r in results for f in r[1]]
    bin_rows   = [b for _, r in results for b in r[2]]

    cell_df = pd.DataFrame(cell_rows)
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        cell_df.to_excel(writer,                  sheet_name='PerCell',  index=False)
        pd.DataFrame(field_rows).to_excel(writer, sheet_name='PerField', index=False)
        pd.DataFrame(bin_rows).to_excel(writer,   sheet_name='PerBin',   index=False)

    print(f'\nDone. {len(cell_rows)}/{len(jobs)} cell(s) plotted.')
    print(f'Results saved to {output_excel}')
    return cell_df


if __name__ == "__main__":
    if RUN_MAPS:
        cell_df = run_maps()
    else:
        print(f'Loading PerCell sheet of {output_excel} ...')
        cell_df = pd.read_excel(output_excel, sheet_name='PerCell')
    run_stats(cell_df)
    print(f'\nDone. Statistics and plots saved under {STATS_DIR}')
