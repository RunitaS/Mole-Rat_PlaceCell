# -*- coding: utf-8 -*-
"""
Example place cells by spatial information rate (SIR): best / worst / median – V1

For each arena (Circle, Linear, Open), picks from the 'Full' sheet of a
PlaceCell_Main_v22 workbook (place_cell == True rows only):
  Best    3 cells with the highest SIR
  Worst   3 cells with the lowest SIR
  Mean    3 cells whose SIR is closest to that arena's MEDIAN SIR (SIR is
          right-skewed, so the median is the typical cell); cells already
          picked as Best / Worst are excluded
and draws each cell as one row of five panels, left to right:
  1. trajectory (gray) with spike locations (red)
  2. Gaussian-smoothed ratemap, full session
  3. Gaussian-smoothed ratemap, first half of the session
  4. Gaussian-smoothed ratemap, second half of the session
  5. the cell's scores, read from its row in the workbook (the values the
     PlaceCell_Main_v22 run wrote, not recomputed here): peak and mean firing
     rate, spatial information (SIR, bits/spike), coherence, sparsity and the
     split-half stability score with its p value. Stability is coloured
       green      p < 0.05 and r > 0
       red        p < 0.05 and r < 0
       dark grey  p >= 0.05 (or not computed)
Every ratemap panel is scaled to ITS OWN peak (colour scale 0 .. that map's
max, peak given in the panel title); no panel shares a colour scale.

Maps are rebuilt with PlaceCell_Main_v22.compute_metrics (same cleaning,
smoothing, immobility filter, spike assignment and valid bins as the run that
wrote the workbook). The half maps are PlaceCell_Main_v22._split_half_maps:
both halves on the full-session grid, each within its own valid bins -- the
maps the split-half stability score is computed from. Circle sessions are
drawn on the 2-D grid, as in PlaceCell_Main_v22's ratemap figure.

Every picked cell also gets its lap raster, in every arena from
LapRaster_AllArenas_v3 (lap_raster_cell): compute_metrics is rerun on that
script's PlaceCell_Main module under its ratemap settings (RATEMAP_TYPE for every
map, Skaggs alpha, minimum field size; field detection included), as its own
batch run does. Those settings apply to that run only. LapRaster_AllArenas_v3
does not need to be run first; nothing is read from its output.

Output (FIGURES_DIR):
  <Arena>/<Best|Worst|Mean>/<rank>_<animal>_<session>_<unit>.png              one row per cell
  <Arena>/<Best|Worst|Mean>/<rank>_<animal>_<session>_<unit>_lap_raster.png   its lap raster
  <Arena>/<Arena>_<Best|Worst|Mean>.png                                      the cells stacked
  BestWorstMedian_SIR_selection.xlsx                                         the cells picked
Each map figure (one-cell and stacked) is also saved as <name>_woBox.png: the
same figure with the trajectory panel and the three ratemaps drawn without
x / y scales and frame.

Analysis settings (bin size, smoothing, filters, COORD_UNITS, ...) come from
PlaceCell_Main_v22; only the directories below are set here. (Was
PlaceCell_Main_v21; v22 only adds the within-field firing rate to its batch
output, so compute_metrics and the maps here are unchanged.)
"""

import contextlib
import os
import re

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import PlaceCell_Main_v22 as pc

# LapRaster_AllArenas_v3 sets its ratemap settings on its PlaceCell_Main module
# (lap.pc) when imported: keep them for its lap rasters and restore this
# script's for the rest, in case lap.pc is the same module as pc.
_LAP_KEYS = ('SI_RATEMAP_TYPE', 'COHERENCE_STABILITY_RATEMAP_TYPE',
             'FIRING_RATE_RATEMAP_TYPE', 'PLACE_FIELD_RATEMAP_TYPE',
             'SKAGGS_ALPHA', 'MIN_FIELD_SIZE_BINS')
_pc_main = {k: getattr(pc, k) for k in _LAP_KEYS}
import LapRaster_AllArenas_v3 as lap   # noqa: E402
_pc_lap = {k: getattr(lap.pc, k) for k in _LAP_KEYS}
for _k, _v in _pc_main.items():
    setattr(pc, _k, _v)


# ── Directories (edit these per run) ──────────────────────────────────────────

# Session folders (root_folder of the PlaceCell_Main_v22 run): the workbook's
# 'session' column is relative to this folder.
root_folder = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final'

# Workbook written by PlaceCell_Main_v22 (output_excel); cells are picked from its 'Full' sheet.
input_excel = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final\All_TT_PlaceChar_SiSkaggsAdptBin_Final_OnlyPC.xlsx'

# Figures go here, one subfolder per arena.
FIGURES_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final\Results\BestWorstMedian_SIR\Vals_v3'

# ── Settings ──────────────────────────────────────────────────────────────────

N_PER_GROUP = 5                          # cells per group (Best / Worst / Mean)
ARENAS      = ('Circle', 'Linear', 'Open')
SIR_COLUMN  = 'sir'

# Half maps: Gaussian-smoothed (PlaceCell_Main_v22._split_half_maps builds
# them with COHERENCE_STABILITY_RATEMAP_TYPE).
pc.COHERENCE_STABILITY_RATEMAP_TYPE = 'gaussian_smooth'

CMAP        = 'jet'
# Every map figure is also saved with this suffix, its trajectory panel and
# three ratemaps drawn without x / y scales and without the frame.
WO_BOX_SUFFIX = '_woBox'
TRAJ_COLOR  = '0.6'
SPIKE_COLOR = 'red'

# Scores panel: workbook columns written by PlaceCell_Main_v22.
STABILITY_ALPHA      = 0.05
STAB_POS_SIG_COLOR   = 'green'      # p < alpha, r > 0
STAB_NEG_SIG_COLOR   = 'red'        # p < alpha, r < 0
STAB_NOT_SIG_COLOR   = 'dimgray'    # p >= alpha or not computed
METRIC_TEXT_COLOR    = 'black'


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
    """Same tracking-file selection as PlaceCell_Main_v22's batch scan."""
    tracking = [f for f in os.listdir(dirpath) if f.lower().endswith(('.csv', '.xlsx'))]
    if pc.COORD_UNITS == 'cm':
        tracking = [f for f in tracking if f.lower().endswith('_cm.csv')]
    else:
        tracking = [f for f in tracking if not f.lower().endswith('_cm.csv')]
    return os.path.join(dirpath, tracking[0]) if len(tracking) == 1 else None


def _load_cell(session: str, unit: str, mod=pc) -> tuple[dict, dict]:
    """compute_metrics (metrics, ctx) of one cell with the current settings of
    mod (pc for the maps, lap.pc for the lap raster)."""
    dirpath  = os.path.join(root_folder, session)
    ntt_path = os.path.join(dirpath, unit)
    csv_path = _find_tracking_csv(dirpath)
    if csv_path is None:
        raise FileNotFoundError(f'no unique tracking file in {dirpath}')
    if not os.path.isfile(ntt_path):
        raise FileNotFoundError(ntt_path)

    metrics, ctx = mod.compute_metrics(csv_path, ntt_path, mod.arena_width_cm, mod.target_bin_cm)
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

def _draw_row(fig, axes, cell: pd.Series, maps: dict, box: bool = True) -> None:
    """One cell: trajectory + spikes, the three ratemaps (each scaled to its
    own peak), then the scores panel. box=False draws the ratemaps and the
    trajectory panel without x / y scales and without the frame."""
    ax_traj = axes[0]
    ax_traj.plot(maps['x_all'], maps['y_all'], color=TRAJ_COLOR, linewidth=0.5)
    ax_traj.plot(maps['x_spk'], maps['y_spk'], '.', color=SPIKE_COLOR, markersize=3)
    ax_traj.set_xlim(maps['extent'][0], maps['extent'][1])
    ax_traj.set_ylim(maps['extent'][2], maps['extent'][3])
    ax_traj.set_aspect('equal', adjustable='box')
    ax_traj.set_title(f'Trajectory + spikes (n = {len(maps["x_spk"])})', fontsize=9)
    cell_id = f'{cell["group"]} {cell["rank"]}\n{cell["session"]}\n{cell["unit"]}'
    if box:
        ax_traj.set_xlabel('x (cm)')
        ax_traj.set_ylabel(f'{cell_id}\n\ny (cm)', fontsize=8)
    else:
        # No x / y ticks, tick labels, axis labels or frame; the cell label
        # stays (set_axis_off would hide it too).
        ax_traj.set_xticks([])
        ax_traj.set_yticks([])
        for spine in ax_traj.spines.values():
            spine.set_visible(False)
        ax_traj.set_ylabel(cell_id, fontsize=8)

    for ax, (title, rate) in zip(axes[1:], maps['maps']):
        peak = float(rate.max()) if rate.count() else 0.0
        im = ax.imshow(rate.T, origin='lower', extent=maps['extent'], cmap=CMAP,
                       interpolation='nearest', vmin=0, vmax=peak if peak > 0 else 1.0)
        ax.set_title(f'{title}\npeak = {peak:.2f} Hz', fontsize=9)
        if box:
            ax.set_xlabel('x (cm)')
            ax.set_ylabel('y (cm)')
        else:
            # No x / y ticks, tick labels, axis labels or frame; the colour
            # bar and the title stay.
            ax.set_axis_off()
        fig.colorbar(im, ax=ax, label='firing rate (Hz)', shrink=0.8)

    _draw_scores(axes[4], cell)


def _num(value) -> float:
    """Workbook value as float (NaN when missing or not numeric)."""
    v = pd.to_numeric(value, errors='coerce')
    return float(v) if pd.notna(v) else float('nan')


def _fmt(value: float, fmt: str, unit: str = '') -> str:
    return 'n/a' if np.isnan(value) else f'{value:{fmt}}{unit}'


def _stability_color(r: float, p: float) -> str:
    if np.isnan(r) or np.isnan(p) or p >= STABILITY_ALPHA:
        return STAB_NOT_SIG_COLOR
    if r > 0:
        return STAB_POS_SIG_COLOR
    if r < 0:
        return STAB_NEG_SIG_COLOR
    return STAB_NOT_SIG_COLOR


def _draw_scores(ax, cell: pd.Series) -> None:
    """Text panel with the cell's workbook scores; stability coloured by its
    sign and significance."""
    ax.axis('off')
    r = _num(cell.get('stability_score'))
    p = _num(cell.get('stability_p_value'))
    lines = [
        (f'Peak FR = {_fmt(_num(cell.get("peak_fr")), ".2f", " Hz")}',            METRIC_TEXT_COLOR),
        (f'Mean FR = {_fmt(_num(cell.get("mean_fr")), ".2f", " Hz")}',            METRIC_TEXT_COLOR),
        (f'SI = {_fmt(_num(cell.get(SIR_COLUMN)), ".3f", " bits/spike")}',        METRIC_TEXT_COLOR),
        (f'Coherence = {_fmt(_num(cell.get("coherence")), ".3f")}',               METRIC_TEXT_COLOR),
        (f'Sparsity = {_fmt(_num(cell.get("sparsity")), ".3f")}',                 METRIC_TEXT_COLOR),
        (f'Stability r = {_fmt(r, ".3f")}',                                       _stability_color(r, p)),
        (f'   (p = {_fmt(p, ".3g")})',                                            _stability_color(r, p)),
    ]
    # Fixed point spacing from the top of the panel, so short (horizontal
    # Linear) rows do not squeeze the lines together.
    for i, (text, color) in enumerate(lines):
        ax.annotate(text, xy=(0.0, 1.0), xycoords='axes fraction',
                    xytext=(0, -13 * i), textcoords='offset points',
                    fontsize=9, color=color, va='top', ha='left')


def _save_figure(cells: list[tuple[pd.Series, dict]], title: str, save_path: str,
                 box: bool = True) -> None:
    # One subfigure per cell, its height following the map's y/x aspect, so
    # horizontal Linear tracks (0 / 180) are not squashed into the row height
    # of vertical ones (90 / 270) or of the Open / Circle arenas.
    heights = [float(np.clip(m['extent'][3] / m['extent'][1], 0.35, 1.0)) for _, m in cells]
    fig = Figure(figsize=(21, 4.4 * sum(heights) + 0.5), layout='constrained')
    FigureCanvasAgg(fig)
    subfigs = fig.subfigures(len(cells), 1, height_ratios=heights, squeeze=False)[:, 0]
    for subfig, (cell, maps) in zip(subfigs, cells):
        _draw_row(subfig, subfig.subplots(1, 5, width_ratios=[1, 1, 1, 1, 0.6]), cell, maps, box=box)
    fig.suptitle(title, fontsize=11)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f'  [SAVED] {save_path}')


def _wo_box_path(save_path: str) -> str:
    """'<name>.png' -> '<name>_woBox.png'."""
    stem, ext = os.path.splitext(save_path)
    return f'{stem}{WO_BOX_SUFFIX}{ext}'


def _save_map_figures(cells: list[tuple[pd.Series, dict]], title: str, save_path: str,
                      boxed: bool = True, wo_box: bool = True) -> None:
    """The map figure with x / y scales and frame on the ratemaps (save_path)
    and / or without them (save_path + WO_BOX_SUFFIX)."""
    if boxed:
        _save_figure(cells, title, save_path, box=True)
    if wo_box:
        _save_figure(cells, title, _wo_box_path(save_path), box=False)


def _safe_name(text: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', '_', text).strip('_')


def _cell_name(cell: pd.Series) -> str:
    """File name stem of one picked cell: <rank>_<session>_<unit>."""
    return _safe_name(f'{cell["rank"]}_{cell["session"]}_{os.path.splitext(cell["unit"])[0]}')


# ── Lap rasters ───────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _lap_settings():
    """LapRaster_AllArenas_v3's settings on lap.pc for the duration of the block."""
    saved = {k: getattr(lap.pc, k) for k in _LAP_KEYS}
    for k, v in _pc_lap.items():
        setattr(lap.pc, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(lap.pc, k, v)


def _save_lap_raster(cell: pd.Series, save_path: str) -> None:
    """Lap raster of one picked cell (any arena) with LapRaster_AllArenas_v3:
    compute_metrics rerun on lap.pc under v3's settings, as its own batch run
    does, so fields, passes and epochs match LapRaster_AllArenas_v3's figure."""
    session, unit = cell['session'], cell['unit']
    dirpath = os.path.join(root_folder, session)
    with _lap_settings():
        metrics, ctx = _load_cell(session, unit, mod=lap.pc)
        rows = lap.lap_raster_cell(metrics, ctx, dirpath, session, unit, save_path)
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
            name = _cell_name(cell)
            _save_map_figures([(cell, maps)],
                              f'{arena} – {group} {cell["rank"]}',
                              os.path.join(FIGURES_DIR, arena, group, f'{name}.png'))
            try:
                _save_lap_raster(cell, os.path.join(FIGURES_DIR, arena, group, f'{name}_lap_raster.png'))
            except Exception as e:
                print(f'  ERROR lap raster {cell["session"]} | {cell["unit"]}: {e}')

        if drawn:
            median_note = f' (closest to median SIR = {sel["median_sir"].iloc[0]:.3f})' if group == 'Mean' else ''
            _save_map_figures(drawn, f'{arena} – {group} {len(drawn)} place cells by SIR{median_note}',
                              os.path.join(FIGURES_DIR, arena, f'{arena}_{group}.png'))
        print()

    print(f'Done. Figures in {FIGURES_DIR}')
