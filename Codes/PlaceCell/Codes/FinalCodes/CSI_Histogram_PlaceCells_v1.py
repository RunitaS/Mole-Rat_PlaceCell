# -*- coding: utf-8 -*-
"""
Complex Spike Index (CSI, the original definition: % of ISIs in [3–20 ms] whose
2nd spike is smaller -- NOT csi_modified) histograms of all units:
  1. <name>_hist_CSI_PlaceCells        – place cells from PlaceCell_Main_v21.py in green
  2. <name>_hist_CSI_PlaceCells_Burst  – place cell without burst peak: BURST_PC_COLOR;
                                         not a place cell, burst peak: BURST_COLOR;
                                         place cell with burst peak: half / half

The histogram uses the bins of Codes/SpikeChar/Fig1def_spike_quality_v2_MTclean.py
('CSI' panel): rejected units as one gray bar at the bottom, accepted units in
one colour (BAR_COLOR) on top, multi-trigger rug, CSI = 10% threshold, mean and
median. Inside each bin the units keep the MTclean script's order (stacked
low → high peak amplitude), and a marked unit is coloured at its own
unit-height slot in that order.

Inputs:
  QUALITY_EXCEL     – <name>_MTclean.xlsx written by Fig1def_spike_quality_v2_MTclean.py
  PLACE_CELL_EXCEL  – workbook written by PlaceCell_Main_v21.py ('Full' sheet,
                      place_cell == True)
  BURST_EXCEL       – BURST_DETECTION.xlsx ('Burst Detection' sheet, BURST_COL == True)
Units are matched on (session, unit); sessions are compared from the animal-ID
folder onwards (e.g. Fa1059\\Linear\\Day10\\1_0), case-insensitive.
"""

import os
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib # type: ignore
matplotlib.use('Agg')
import matplotlib.pyplot as plt # type: ignore
from matplotlib.colors import to_hex # type: ignore
from matplotlib.legend_handler import HandlerTuple # type: ignore

# ── Configuration ──────────────────────────────────────────────────────────────

QUALITY_EXCEL    = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data\QUALITY_METRICS_MTclean.xlsx'
QUALITY_SHEET    = 'Quality Metrics (MT clean)'
PLACE_CELL_EXCEL = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final\All_TT_PlaceChar_SiSkaggsAdptBin_Final_OnlyPC.xlsx'
PLACE_CELL_SHEET = 'Full'
BURST_EXCEL      = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final\Results\QualityMetrics\burst_plots\BURST_DETECTION.xlsx'
BURST_SHEET      = 'Burst Detection'
BURST_COL        = 'has_burst_peak'   # burst power >= 5% and an interior KDE peak

PLOT_DIR = os.path.join(os.path.dirname(QUALITY_EXCEL), 'quality_plots_MTclean')
os.makedirs(PLOT_DIR, exist_ok=True)

CSI_COL      = 'csi'
CSI_THR_PCT  = 10.0
N_BINS       = 30          # same as draw_histogram in Fig1def_spike_quality_v2_MTclean.py

ANIMAL_NAMES = ('Fa1059', 'Fa23BD', 'Fa8477', 'Fa5834')

# ── Figure palette (identical to Fig1def_spike_quality_v2_MTclean.py) ─────────
FIG_CMAP = 'magma'
MAGMA_1, MAGMA_2, MAGMA_3, MAGMA_4 = [to_hex(matplotlib.colormaps[FIG_CMAP](v))
                                      for v in np.linspace(0, 1, 6)[1:-1]]
REJECT_COLOR   = '#e1e0d9'
INK_PRIMARY    = '#0b0b0b'
INK_SECONDARY  = '#52514e'
INK_MUTED      = '#898781'
GRID_COLOR     = '#e1e0d9'
SURFACE        = '#fcfcfb'

REJECTED_COLOR = REJECT_COLOR
MT_COLOR       = MAGMA_3
COLOR_THR_BAD  = MAGMA_3

# Accepted units: one light magma colour for all bars (no amplitude colour code)
BAR_COLOR = to_hex(matplotlib.colormaps[FIG_CMAP](0.60))

# Place cells: their block is filled green (full bin width) instead of
# BAR_COLOR; green is outside the magma palette, so it stands out
PC_BAND_COLOR = '#1a9850'

# Plot 2 (place cells x burst peak), lime green / lime yellow from viridis:
# place cell only = green, burst peak only = yellow, both = half green (left) /
# half yellow (right)
BURST_PC_COLOR = to_hex(matplotlib.colormaps['viridis'](0.80))
BURST_COLOR    = to_hex(matplotlib.colormaps['viridis'](0.20))


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path_base, exts=('png', 'svg')):
    """Save fig as <path_base>.<ext> for each ext, then close it."""
    for ext in exts:
        fig.savefig(f'{path_base}.{ext}', dpi=300, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def _to_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ('true', '1', 'yes')
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    return bool(v)


def _unit_key(session, unit) -> tuple[str, str]:
    """(session from the animal-ID folder onwards, unit), lower-case."""
    parts = os.path.normpath(str(session)).split(os.sep)
    for i, part in enumerate(parts):
        if part in ANIMAL_NAMES:
            parts = parts[i:]
            break
    return os.path.join(*parts).lower(), str(unit).strip().lower()


# ── Load quality metrics and place cells ──────────────────────────────────────

df = pd.read_excel(QUALITY_EXCEL, sheet_name=QUALITY_SHEET, engine='openpyxl')
session_col = df.columns[0]
df['accepted']      = df['accepted'].apply(_to_bool)
df['multi_trigger'] = df['multi_trigger'].apply(_to_bool)

pc_df = pd.read_excel(PLACE_CELL_EXCEL, sheet_name=PLACE_CELL_SHEET, engine='openpyxl')
pc_df = pc_df[pc_df['place_cell'].apply(_to_bool)]
pc_keys = {_unit_key(s, u) for s, u in zip(pc_df['session'], pc_df['unit'])}

df_keys = [_unit_key(s, u) for s, u in zip(df[session_col], df['unit'])]
df['place_cell'] = [k in pc_keys for k in df_keys]
unmatched = pc_keys - set(df_keys)
print(f'Place cells in {os.path.basename(PLACE_CELL_EXCEL)}: {len(pc_keys)}, '
      f'matched in quality table: {len(pc_keys) - len(unmatched)}')
for s, u in sorted(unmatched):
    print(f'  NOT FOUND in quality table: {s}  |  {u}')

burst_df = pd.read_excel(BURST_EXCEL, sheet_name=BURST_SHEET, engine='openpyxl')
burst_df = burst_df[burst_df[BURST_COL].apply(_to_bool)]
burst_keys = {_unit_key(s, u) for s, u in zip(burst_df['session'], burst_df['unit'])}
df['burst_peak'] = [k in burst_keys for k in df_keys]
unmatched = burst_keys - set(df_keys)
print(f'Units with {BURST_COL} in {os.path.basename(BURST_EXCEL)}: {len(burst_keys)}, '
      f'matched in quality table: {len(burst_keys) - len(unmatched)}')
for s, u in sorted(unmatched):
    print(f'  NOT FOUND in quality table: {s}  |  {u}')

# Multi-trigger units successfully re-computed on clean spikes (the MTclean
# script fills n_spikes_clean only for those)
mt_flags = (df['multi_trigger'] & df['n_spikes_clean'].notna()).values

# Peak amplitude only sets the stacking order within each bin (as in
# Fig1def_spike_quality_v2_MTclean.py), not the colour
peak_amps = pd.to_numeric(df['peak_amp_uV'], errors='coerce').to_numpy(dtype=float)


# ── Histogram (bins / stacking of the MTclean script + marked units) ──────────

def _block_positions(vals, amp, bin_edges):
    """Bin index and stack rank of every unit, stacked low → high amplitude
    within its bin (the order draw_histogram uses for the accepted blocks).
    Returns (bin index, rank) in the original order of `vals`."""
    b     = np.clip(np.digitize(vals, bin_edges) - 1, 0, len(bin_edges) - 2)
    order = np.lexsort((amp, b))
    b_s   = b[order]
    rank  = np.empty(len(vals), dtype=int)
    rank[order] = np.arange(len(b_s)) - np.searchsorted(b_s, b_s)
    return b, rank


def _draw_marked_blocks(ax, bins, bottoms, colors, bin_centers, bin_widths):
    """Fill the unit-height blocks at (bins, bottoms) across the full bin width,
    split side by side into len(colors) equal parts (left → right). Blocks
    stacked directly on top of each other are drawn as one run, so no seams
    appear between ~1 px blocks."""
    if len(bins) == 0:
        return
    order = np.lexsort((bottoms, bins))
    bins, bottoms = bins[order], bottoms[order]
    new_run = np.r_[True, (np.diff(bins) != 0) | (np.diff(bottoms) != 1)]
    run_id  = np.cumsum(new_run) - 1
    run_bin = bins[new_run]
    run_bot = bottoms[new_run]
    run_len = np.bincount(run_id)
    part_w  = bin_widths[run_bin] / len(colors)
    left    = bin_centers[run_bin] - bin_widths[run_bin] / 2
    for i, col in enumerate(colors):
        # Below the SURFACE bin outline (zorder 3.5), which keeps the gap between bins
        ax.bar(left + (i + 0.5) * part_w, run_len, width=part_w, bottom=run_bot,
               color=col, linewidth=0, zorder=3.4)


def draw_csi_histogram(ax, values, accepted, is_mt, amp, xlabel, marks,
                       thresholds=(), thr_colors=(), thr_labels=(), summary_lines=(),
                       n_bins=N_BINS):
    """
    Same bins and stacking as draw_histogram (Fig1def_spike_quality_v2_MTclean.py)
    with amp given, but accepted units are all BAR_COLOR; `amp` only sets each
    unit's position within its bin (low → high). The blocks of the units in
    each of `marks` are then filled with that mark's colour(s) at that position.
    marks: list of (unit mask, colours, legend label); with two colours the
           block is split half / half (left / right).
    Rejected units form one gray bar per bin; a marked rejected unit is filled
    at a unit-height slot inside that bar, slots ordered by amplitude like the
    accepted blocks.
    Returns the number of units drawn for each mark.
    """
    values = np.asarray(values, dtype=float)
    valid  = ~np.isnan(values)
    vals   = values[valid]
    acc    = np.asarray(accepted, dtype=bool)[valid]
    mt     = np.asarray(is_mt, dtype=bool)[valid]
    amp    = np.asarray(amp, dtype=float)[valid]

    bin_edges   = np.histogram_bin_edges(vals, bins=n_bins)
    bin_widths  = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + bin_widths / 2
    n_rej, _ = np.histogram(vals[~acc], bins=bin_edges)
    n_acc, _ = np.histogram(vals[acc],  bins=bin_edges)

    ax.bar(bin_centers, n_rej, width=bin_widths, color=REJECTED_COLOR, edgecolor=SURFACE,
           linewidth=1.0, zorder=3, label=f'Rejected (n={int(n_rej.sum())})')

    ax.bar(bin_centers, n_acc, width=bin_widths, bottom=n_rej, color=BAR_COLOR,
           linewidth=0, zorder=3)
    # Outline drawn above the marked blocks keeps the gap between bins
    ax.bar(bin_centers, n_acc, width=bin_widths, bottom=n_rej, fill=False,
           edgecolor=SURFACE, linewidth=1.0, zorder=3.5)

    b_acc, rank_acc = _block_positions(vals[acc], amp[acc], bin_edges)

    # Marked units: their own block is filled with the mark's colour(s), so
    # each keeps its amplitude-ordered position.
    b_rej, rank_rej = _block_positions(vals[~acc], amp[~acc], bin_edges)
    n_marked = []
    for mask, colors, _ in marks:
        m = np.asarray(mask, dtype=bool)[valid]
        m_acc, m_rej = m[acc], m[~acc]
        bins    = np.concatenate([b_acc[m_acc], b_rej[m_rej]])
        bottoms = np.concatenate([n_rej[b_acc[m_acc]] + rank_acc[m_acc], rank_rej[m_rej]])
        _draw_marked_blocks(ax, bins, bottoms, colors, bin_centers, bin_widths)
        n_marked.append(int(m.sum()))

    if mt.any():
        # Rug in its own strip below the bars so it never hides low bins
        top = max(float((n_rej + n_acc).max()), 1.0)
        ax.set_ylim(-0.10 * top, 1.05 * top)
        ax.axhline(0, color=INK_MUTED, linewidth=0.8, zorder=4)
        ax.plot(vals[mt], np.full(mt.sum(), -0.05 * top), linestyle='none', marker='|',
                markersize=7, markeredgewidth=0.8, alpha=0.6, color=MT_COLOR, zorder=6,
                label=f'Multi-trigger, MT spikes excluded (n={int(mt.sum())})')

    for xv, col, lbl in zip(thresholds, thr_colors, thr_labels):
        ax.axvline(xv, color=col, linewidth=1.5, linestyle='--', zorder=4, label=lbl)

    mean_v, median_v = float(np.mean(vals)), float(np.median(vals))
    ax.axvline(mean_v,   color=INK_PRIMARY, linewidth=1.5, linestyle='-', zorder=5,
               label=f'Mean   {mean_v:.3f}')
    ax.axvline(median_v, color=INK_PRIMARY, linewidth=1.5, linestyle=':', zorder=5,
               label=f'Median {median_v:.3f}')
    ax.set_xlabel(xlabel, color=INK_PRIMARY)
    ax.set_ylabel('Number of units', color=INK_PRIMARY)

    handles, labels = ax.get_legend_handles_labels()
    bars  = [i for i, l in enumerate(labels) if l.startswith('Rejected')]
    order = bars + [i for i in range(len(labels)) if i not in bars]
    handles, labels = [handles[i] for i in order], [labels[i] for i in order]
    handles.insert(1, plt.Rectangle((0, 0), 1, 1, color=BAR_COLOR))
    labels.insert(1, f'Accepted (n={int(n_acc.sum())})')
    for i, ((_, colors, label), n) in enumerate(zip(marks, n_marked)):
        # Several colours: patches side by side, like the split blocks
        handles.insert(2 + i, tuple(plt.Rectangle((0, 0), 1, 1, color=c) for c in colors))
        labels.insert(2 + i, f'{label} (n={n})')
    ax.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK_PRIMARY,
              loc='upper left', bbox_to_anchor=(1.0, 1.0),
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})

    summary = '\n'.join([f'n = {len(vals)}',
                         f'Mean ± SD = {mean_v:.3f} ± {float(np.std(vals)):.3f}',
                         f'Median = {median_v:.3f}', *summary_lines])
    ax.text(1.02, 0.0, summary, transform=ax.transAxes, fontsize=8, color=INK_SECONDARY,
            ha='left', va='bottom')
    return n_marked


def _group_summary(name, mask):
    """'<name>: median = .., k/n > CSI_THR_PCT%' for the units in mask with a CSI."""
    v = csi_vals[np.asarray(mask, dtype=bool) & ~np.isnan(csi_vals)]
    if len(v) == 0:
        return f'{name}: none'
    return (f'{name}: median = {float(np.median(v)):.3f}, '
            f'{int((v > CSI_THR_PCT).sum())}/{len(v)} > {CSI_THR_PCT:g}%')


def plot_csi(marks, title, out_name, summary_lines=()):
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    _style_axes(ax)
    n_marked = draw_csi_histogram(
        ax, csi_vals, df['accepted'].values, mt_flags, peak_amps,
        'CSI  (% of ISIs in [3–20 ms] with smaller 2nd spike)', marks,
        (CSI_THR_PCT,), (COLOR_THR_BAD,), (f'CSI = {CSI_THR_PCT:g}%',),
        summary_lines=summary_lines)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc='left')
    out_base = os.path.join(PLOT_DIR, f'{excel_base}_{out_name}')
    _save(fig, out_base)
    for (_, _, label), n in zip(marks, n_marked):
        print(f'  {label}: {n}')
    print(f'Saved: {out_base}.png / .svg')


csi_vals   = pd.to_numeric(df[CSI_COL], errors='coerce').to_numpy(dtype=float)
excel_base = os.path.splitext(os.path.basename(QUALITY_EXCEL))[0]
is_pc      = df['place_cell'].to_numpy(dtype=bool)
is_burst   = df['burst_peak'].to_numpy(dtype=bool)
for name, mask in (('place cell', is_pc), ('burst-peak unit', is_burst)):
    n_no_csi = int((mask & np.isnan(csi_vals)).sum())
    if n_no_csi:
        print(f'{n_no_csi} {name}(s) have no CSI value and are not in the histogram')

# 1. Place cells only
print('\nPlot 1: place cells')
plot_csi([(is_pc, (PC_BAND_COLOR,), 'Place cell')],
         'Complex Spike Index (CSI), place cells marked in green',
         'hist_CSI_PlaceCells',
         summary_lines=[_group_summary('Place cells', is_pc)])

# 2. Place cells x burst peak
print('\nPlot 2: place cells x burst peak')
plot_csi([(is_pc & ~is_burst, (BURST_PC_COLOR,), 'Place cell, no burst peak'),
          (~is_pc & is_burst, (BURST_COLOR,), 'Not a place cell, burst peak'),
          (is_pc & is_burst,  (BURST_PC_COLOR, BURST_COLOR), 'Place cell with burst peak')],
         'Complex Spike Index (CSI), place cells and burst-peak units',
         'hist_CSI_PlaceCells_Burst',
         summary_lines=[_group_summary('Place cells', is_pc),
                        _group_summary('Burst-peak units', is_burst),
                        _group_summary('Place cells with burst peak', is_pc & is_burst)])
