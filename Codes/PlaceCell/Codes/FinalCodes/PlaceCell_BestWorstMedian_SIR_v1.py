# -*- coding: utf-8 -*-
"""
Example place cells by spatial information rate (SIR): best / worst / median – V1

For each arena (Circle, Linear, Open), picks from the 'Full' sheet of a
PlaceCell_Main_v21 workbook (place_cell == True rows only):
  Best    3 cells with the highest SIR
  Worst   3 cells with the lowest SIR
  Mean    3 cells whose SIR is closest to that arena's MEDIAN SIR (SIR is
          right-skewed, so the median is the typical cell); cells already
          picked as Best / Worst are excluded
and draws each cell as one row of four panels, left to right:
  1. trajectory (gray) with spike locations (red)
  2. Gaussian-smoothed ratemap, full session
  3. Gaussian-smoothed ratemap, first half of the session
  4. Gaussian-smoothed ratemap, second half of the session
Every ratemap panel is scaled to ITS OWN peak (colour scale 0 .. that map's
max, peak given in the panel title); no panel shares a colour scale.

Maps are rebuilt with PlaceCell_Main_v21.compute_metrics (same cleaning,
smoothing, immobility filter, spike assignment and valid bins as the run that
wrote the workbook). The half maps are PlaceCell_Main_v21._split_half_maps:
both halves on the full-session grid, each within its own valid bins -- the
maps the split-half stability score is computed from. Circle sessions are
drawn on the 2-D grid, as in PlaceCell_Main_v21's ratemap figure.

Every picked cell also gets its lap raster:
  Linear / Circle  LapRaster_Linearized_v1, from the same compute_metrics output
                   as the maps above (PlaceCell_Main_v21's settings)
  Open             LapRaster_OpenField_v3_AdptBinFlds, with compute_metrics rerun
                   under that script's ratemap settings (RATEMAP_TYPE for every
                   map, field detection included); they apply to that run only

Output (FIGURES_DIR):
  <Arena>/<Best|Worst|Mean>/<rank>_<animal>_<session>_<unit>.png              one row per cell
  <Arena>/<Best|Worst|Mean>/<rank>_<animal>_<session>_<unit>_lap_raster.png   its lap raster
  <Arena>/<Arena>_<Best|Worst|Mean>.png                                      the cells stacked
  BestWorstMedian_SIR_selection.xlsx                                         the cells picked

Analysis settings (bin size, smoothing, filters, COORD_UNITS, ...) come from
PlaceCell_Main_v21; only the directories below are set here.
"""

import contextlib
import os
import re

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import PlaceCell_Main_v21 as pc
import LapRaster_Linearized_v1 as lap_lin

# LapRaster_OpenField_v3 sets its ratemap settings on pc when imported: keep
# them for its own lap rasters and restore PlaceCell_Main_v21's for the rest.
_LAP_OPEN_KEYS = ('SI_RATEMAP_TYPE', 'COHERENCE_STABILITY_RATEMAP_TYPE',
                  'FIRING_RATE_RATEMAP_TYPE', 'PLACE_FIELD_RATEMAP_TYPE',
                  'SKAGGS_ALPHA', 'MIN_FIELD_SIZE_BINS')
_pc_main = {k: getattr(pc, k) for k in _LAP_OPEN_KEYS}
import LapRaster_OpenField_v3_AdptBinFlds as lap_open   # noqa: E402
_pc_lap_open = {k: getattr(pc, k) for k in _LAP_OPEN_KEYS}
for _k, _v in _pc_main.items():
    setattr(pc, _k, _v)


# ── Directories (edit these per run) ──────────────────────────────────────────

# Session folders (root_folder of the PlaceCell_Main_v21 run): the workbook's
# 'session' column is relative to this folder.
root_folder = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final'

# Workbook written by PlaceCell_Main_v21 (output_excel); cells are picked from its 'Full' sheet.
input_excel = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final\All_TT_PlaceChar_SiSkaggsAdptBin_Final_OnlyPC.xlsx'

# Figures go here, one subfolder per arena.
FIGURES_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final\Results\BestWorstMedian_SIR'

# ── Settings ──────────────────────────────────────────────────────────────────

N_PER_GROUP = 5                          # cells per group (Best / Worst / Mean)
ARENAS      = ('Circle', 'Linear', 'Open')
SIR_COLUMN  = 'sir'

# Half maps: Gaussian-smoothed (PlaceCell_Main_v21._split_half_maps builds
# them with COHERENCE_STABILITY_RATEMAP_TYPE).
pc.COHERENCE_STABILITY_RATEMAP_TYPE = 'gaussian_smooth'

CMAP        = 'jet'
TRAJ_COLOR  = '0.6'
SPIKE_COLOR = 'red'


# ── Cell selection ────────────────────────────────────────────────────────────

def _arena_of(session: str) -> str | None:
    """'Circle' | 'Linear' | 'Open' from a session path such as
    'Fa1059\\Linear\\Day10\\1_0', or None."""
    parts = [p.lower() for p in re.split(r'[\\/]+', session) if p]
    for arena in ARENAS:
        if arena.lower() in parts:
            return arena
    return None


def _select_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Best / Worst / Mean (closest to the median SIR) cells per arena, one row
    per picked cell with columns arena, group, rank, median_sir and the
    workbook columns."""
    df = df[df['place_cell'] == True].copy()   # noqa: E712
    df[SIR_COLUMN] = pd.to_numeric(df[SIR_COLUMN], errors='coerce')
    df = df.dropna(subset=[SIR_COLUMN])
    df['arena'] = df['session'].astype(str).map(_arena_of)

    picked = []
    for arena in ARENAS:
        d = df[df['arena'] == arena]
        if d.empty:
            print(f'[WARN] No place cells for arena {arena}')
            continue
        median_sir = float(d[SIR_COLUMN].median())

        best  = d.sort_values(SIR_COLUMN, ascending=False).head(N_PER_GROUP)
        worst = d.sort_values(SIR_COLUMN, ascending=True).head(N_PER_GROUP)
        rest  = d.drop(index=best.index.union(worst.index))
        mean  = (rest.assign(_dist=(rest[SIR_COLUMN] - median_sir).abs())
                     .sort_values('_dist').head(N_PER_GROUP).drop(columns='_dist'))

        for group, sel in (('Best', best), ('Worst', worst), ('Mean', mean)):
            sel = sel.assign(group=group, rank=np.arange(1, len(sel) + 1),
                             median_sir=round(median_sir, 4))
            picked.append(sel)
        print(f'{arena}: {len(d)} place cells, median SIR = {median_sir:.4f}')

    lead = ['arena', 'group', 'rank', 'session', 'unit', SIR_COLUMN, 'median_sir']
    out = pd.concat(picked, ignore_index=True)
    return out[lead + [c for c in out.columns if c not in lead]]


# ── Maps ──────────────────────────────────────────────────────────────────────

def _find_tracking_csv(dirpath: str) -> str | None:
    """Same tracking-file selection as PlaceCell_Main_v21's batch scan."""
    tracking = [f for f in os.listdir(dirpath) if f.lower().endswith(('.csv', '.xlsx'))]
    if pc.COORD_UNITS == 'cm':
        tracking = [f for f in tracking if f.lower().endswith('_cm.csv')]
    else:
        tracking = [f for f in tracking if not f.lower().endswith('_cm.csv')]
    return os.path.join(dirpath, tracking[0]) if len(tracking) == 1 else None


def _load_cell(session: str, unit: str) -> tuple[dict, dict]:
    """compute_metrics (metrics, ctx) of one cell with the current pc settings."""
    dirpath  = os.path.join(root_folder, session)
    ntt_path = os.path.join(dirpath, unit)
    csv_path = _find_tracking_csv(dirpath)
    if csv_path is None:
        raise FileNotFoundError(f'no unique tracking file in {dirpath}')
    if not os.path.isfile(ntt_path):
        raise FileNotFoundError(ntt_path)

    metrics, ctx = pc.compute_metrics(csv_path, ntt_path, pc.arena_width_cm, pc.target_bin_cm)
    if not ctx or not ctx['valid_mask'].any():
        raise ValueError('no valid bins')
    return metrics, ctx


def _cell_maps(ctx: dict) -> dict:
    """Trajectory, spikes and the full / first-half / second-half
    Gaussian-smoothed ratemaps of one cell (masked outside each map's valid bins)."""
    halves = pc._split_half_maps(ctx, pc.target_bin_cm)
    return dict(
        x_all=ctx['x_cm_all'], y_all=ctx['y_cm_all'],
        x_spk=ctx['x_cm'][ctx['spike_frame']], y_spk=ctx['y_cm'][ctx['spike_frame']],
        extent=[0, ctx['n_bins_x'] * pc.target_bin_cm, 0, ctx['n_bins_y'] * pc.target_bin_cm],
        maps=[('Gaussian-smoothed',  np.ma.masked_where(~ctx['valid_mask'],     ctx['fr_smooth'])),
              ('First half',         np.ma.masked_where(~halves['valid_first'],  halves['fr_first'])),
              ('Second half',        np.ma.masked_where(~halves['valid_second'], halves['fr_second']))],
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def _draw_row(fig, axes, cell: pd.Series, maps: dict) -> None:
    """One cell: trajectory + spikes, then the three ratemaps, each scaled to
    its own peak."""
    ax_traj = axes[0]
    ax_traj.plot(maps['x_all'], maps['y_all'], color=TRAJ_COLOR, linewidth=0.5)
    ax_traj.plot(maps['x_spk'], maps['y_spk'], '.', color=SPIKE_COLOR, markersize=3)
    ax_traj.set_xlim(maps['extent'][0], maps['extent'][1])
    ax_traj.set_ylim(maps['extent'][2], maps['extent'][3])
    ax_traj.set_aspect('equal', adjustable='box')
    ax_traj.set_title(f'Trajectory + spikes (n = {len(maps["x_spk"])})', fontsize=9)
    ax_traj.set_xlabel('x (cm)')
    ax_traj.set_ylabel(f'{cell["group"]} {cell["rank"]}\n'
                       f'{cell["session"]}\n{cell["unit"]}\n'
                       f'SIR = {cell[SIR_COLUMN]:.3f}\n\ny (cm)', fontsize=8)

    for ax, (title, rate) in zip(axes[1:], maps['maps']):
        peak = float(rate.max()) if rate.count() else 0.0
        im = ax.imshow(rate.T, origin='lower', extent=maps['extent'], cmap=CMAP,
                       interpolation='nearest', vmin=0, vmax=peak if peak > 0 else 1.0)
        ax.set_title(f'{title}\npeak = {peak:.2f} Hz', fontsize=9)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        fig.colorbar(im, ax=ax, label='firing rate (Hz)', shrink=0.8)


def _save_figure(cells: list[tuple[pd.Series, dict]], title: str, save_path: str) -> None:
    # One subfigure per cell, its height following the map's y/x aspect, so
    # horizontal Linear tracks (0 / 180) are not squashed into the row height
    # of vertical ones (90 / 270) or of the Open / Circle arenas.
    heights = [float(np.clip(m['extent'][3] / m['extent'][1], 0.35, 1.0)) for _, m in cells]
    fig = Figure(figsize=(18, 4.4 * sum(heights) + 0.5), layout='constrained')
    FigureCanvasAgg(fig)
    subfigs = fig.subfigures(len(cells), 1, height_ratios=heights, squeeze=False)[:, 0]
    for subfig, (cell, maps) in zip(subfigs, cells):
        _draw_row(subfig, subfig.subplots(1, 4), cell, maps)
    fig.suptitle(title, fontsize=11)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f'  [SAVED] {save_path}')


def _safe_name(text: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', '_', text).strip('_')


# ── Lap rasters ───────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _lap_open_settings():
    """LapRaster_OpenField_v3's pc settings for the duration of the block."""
    saved = {k: getattr(pc, k) for k in _LAP_OPEN_KEYS}
    for k, v in _pc_lap_open.items():
        setattr(pc, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(pc, k, v)


def _save_lap_raster(arena: str, cell: pd.Series, metrics: dict, ctx: dict,
                     save_path: str) -> None:
    """Lap raster of one picked cell: Linear / Circle from the maps' own
    compute_metrics output, Open with compute_metrics rerun under
    LapRaster_OpenField_v3's settings."""
    session, unit = cell['session'], cell['unit']
    dirpath = os.path.join(root_folder, session)
    if arena == 'Open':
        with _lap_open_settings():
            metrics, ctx = _load_cell(session, unit)
            rows = lap_open.lap_raster_cell(metrics, ctx, dirpath, session, unit, save_path)
    else:
        rows = lap_lin.lap_raster_cell(metrics, ctx, dirpath, session, unit, save_path)
    if rows is None:
        print(f'  [SKIP] lap raster: no place field detected for {session} | {unit}')
    else:
        print(f'  [SAVED] {save_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    df_full  = pd.read_excel(input_excel, sheet_name='Full')
    selected = _select_cells(df_full)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    selection_path = os.path.join(FIGURES_DIR, 'BestWorstMedian_SIR_selection.xlsx')
    selected.to_excel(selection_path, index=False)
    print(f'[SAVED] {selection_path}\n')

    for (arena, group), sel in selected.groupby(['arena', 'group'], sort=False):
        arena, group = str(arena), str(group)
        print(f'{arena} – {group}')
        drawn = []
        for _, cell in sel.iterrows():
            try:
                metrics, ctx = _load_cell(cell['session'], cell['unit'])
                maps = _cell_maps(ctx)
            except Exception as e:
                print(f'  ERROR {cell["session"]} | {cell["unit"]}: {e}')
                continue
            drawn.append((cell, maps))
            name = _safe_name(f'{cell["rank"]}_{cell["session"]}_{os.path.splitext(cell["unit"])[0]}')
            _save_figure([(cell, maps)],
                         f'{arena} – {group} {cell["rank"]}',
                         os.path.join(FIGURES_DIR, arena, group, f'{name}.png'))
            try:
                _save_lap_raster(arena, cell, metrics, ctx,
                                 os.path.join(FIGURES_DIR, arena, group, f'{name}_lap_raster.png'))
            except Exception as e:
                print(f'  ERROR lap raster {cell["session"]} | {cell["unit"]}: {e}')

        if drawn:
            median_note = f' (closest to median SIR = {sel["median_sir"].iloc[0]:.3f})' if group == 'Mean' else ''
            _save_figure(drawn, f'{arena} – {group} {len(drawn)} place cells by SIR{median_note}',
                         os.path.join(FIGURES_DIR, arena, f'{arena}_{group}.png'))
        print()

    print(f'Done. Figures in {FIGURES_DIR}')
