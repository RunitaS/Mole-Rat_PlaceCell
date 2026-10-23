# -*- coding: utf-8 -*-
"""
Temporal stability maps for place cells – V1

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
from mpl_toolkits.axes_grid1 import make_axes_locatable

import PlaceCell_Main_v21 as pc

# ── Directories (edit these per run) ──────────────────────────────────────────

# Place-cell folder (Output_PlaceTrue of a PlaceCell_Main_v21 run): every .ntt under
# it is analysed; each session folder also holds its tracking file.
place_cell_folder = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Alpha200'

# Per-cell / per-field / per-bin results table.
output_excel = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Alpha200\Results\TemporalStability.xlsx'

# Figures go in this subfolder of each session folder (next to the .ntt).
TEMPORAL_STABILITY_SUBDIR = 'temporal_stability'

# ── Settings ──────────────────────────────────────────────────────────────────

MIN_VISIT_S     = 0.0      # visits shorter than this (s) are ignored; 0 = keep every visit
STABILITY_CMAP  = 'jet'
NON_FIELD_COLOR = '0.75'   # gray for visited bins outside every place field

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


def _plot_rate(ax, fig, rate: np.ndarray, valid: np.ndarray, extent: list, title: str):
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

    def _summary(mask: np.ndarray) -> dict:
        p = pct[mask & has_visits]
        v, fired = int(n_visits[mask].sum()), int(n_fired[mask].sum())
        return {'n_field_bins': int(mask.sum()),
                'mean_pct_across_bins':   round(float(np.mean(p)), 2)   if p.size else np.nan,
                'median_pct_across_bins': round(float(np.median(p)), 2) if p.size else np.nan,
                'n_visits_total': v, 'n_visits_with_spike_total': fired,
                'pooled_pct_visits_with_spike': round(100.0 * fired / v, 2) if v else np.nan}

    field_rows = [{'session': session, 'unit': unit, 'arena_type': arena_type,
                   'field_number': int(fn), **_summary(field_id == fn)}
                  for fn in sorted(np.unique(field_id[field_mask]))]
    cell_row = {'session': session, 'unit': unit, 'arena_type': arena_type,
                'n_fields': len(field_rows), 'n_spikes': metrics.get('n_spikes'),
                'peak_fr': metrics.get('peak_fr'), 'mean_fr': metrics.get('mean_fr'),
                'sir': metrics.get('sir'), **_summary(field_mask)}

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
        stab_title += ', angular'
    else:
        ax_stab = fig.add_subplot(n_rows, n_cols, 4)
        _plot_stability_2d(ax_stab, fig, pct, field_mask, valid, extent)
        for fn in np.unique(field_id[field_mask]):
            for ax in (ax_raw, ax_smooth, ax_stab):
                _contour_field(ax, field_id == fn, bin_cm)
    ax_stab.set_title(stab_title)

    ntt_name = os.path.splitext(unit)[0]
    fig.suptitle(f"{session}  |  {ntt_name}\n"
                 f"{cell_row['n_fields']} field(s), {cell_row['n_field_bins']} bins  |  "
                 f"mean across bins = {cell_row['mean_pct_across_bins']}%, "
                 f"pooled = {cell_row['pooled_pct_visits_with_spike']}% of "
                 f"{cell_row['n_visits_total']} visits with >= 1 spike", fontsize=10)
    fig.tight_layout()

    save_dir = os.path.join(dirpath, TEMPORAL_STABILITY_SUBDIR)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{ntt_name}_temporal_stability.png')
    fig.savefig(save_path, dpi=150)
    with _print_lock:
        print(f'  [SAVED] {save_path}')

    return cell_row, field_rows, bin_rows


# ── Batch ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
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

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        pd.DataFrame(cell_rows).to_excel(writer,  sheet_name='PerCell',  index=False)
        pd.DataFrame(field_rows).to_excel(writer, sheet_name='PerField', index=False)
        pd.DataFrame(bin_rows).to_excel(writer,   sheet_name='PerBin',   index=False)

    print(f'\nDone. {len(cell_rows)}/{len(jobs)} cell(s) plotted.')
    print(f'Results saved to {output_excel}')
