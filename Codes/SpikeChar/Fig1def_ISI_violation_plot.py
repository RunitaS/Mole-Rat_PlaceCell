# -*- coding: utf-8 -*-
"""
ISI-violation histogram only, drawn from the saved
    QUALITY_METRICS_MTclean_v2.xlsx  (sheet 'Quality Metrics (MT clean)')
written by Fig1def_spike_quality_v2_MTclean.py, so the .ntt files are not re-read.

Same look as the ISI panel of Fig1def_spike_quality_v2_MTclean.py (stacked rejected /
accepted bars, accepted units coloured by peak amplitude), with two axis breaks:
  x: the data end near 1.4 %, the 'Marginal < 5 %' line is far to the right; the empty
     stretch in between is squeezed into a '≈' break so the data fill the axis.
  y: ~96 % of units have exactly 0 % violations; that one bar is cut ('≈' on the y-axis)
     so the bins with non-zero violations are readable.

Output: <PLOT_DIR>/QUALITY_METRICS_MTclean_hist_ISI.png / .svg (same name and folder as
the ISI histogram of the main script, which it replaces).
"""

import os
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib # type: ignore
matplotlib.use('Agg')
import matplotlib.pyplot as plt # type: ignore
from matplotlib.colors import to_hex, ListedColormap, TwoSlopeNorm # type: ignore
from matplotlib.cm import ScalarMappable # type: ignore
from matplotlib.ticker import MaxNLocator, FixedLocator, FormatStrFormatter # type: ignore

# ── Configuration (same paths / thresholds as Fig1def_spike_quality_v2_MTclean.py) ──

OUTPUT_EXCEL = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data\QUALITY_METRICS.xlsx'
CLEAN_EXCEL  = os.path.splitext(OUTPUT_EXCEL)[0] + '_MTclean_v2.xlsx'
CLEAN_SHEET  = 'Quality Metrics (MT clean)'
PLOT_DIR     = os.path.join(os.path.dirname(OUTPUT_EXCEL), 'quality_plots_MTclean_v2')
os.makedirs(PLOT_DIR, exist_ok=True)

ISI_GOOD_PCT, ISI_MARGIN_PCT = 0.1, 5.0
N_BINS = 30

# Axis breaks
BREAK_GAP_FRAC = 0.05   # drawn width of a break, as a fraction of the segment before it
Y_BREAK_RATIO  = 4.0    # break the y-axis when the tallest bin is > this × the 2nd tallest
Y_UPPER_FRAC   = 0.18   # share of the axis height kept for the top of the tallest bin

# Palette
FIG_CMAP = 'magma'
MAGMA_1, MAGMA_2, MAGMA_3, MAGMA_4 = [to_hex(matplotlib.colormaps[FIG_CMAP](v))
                                      for v in np.linspace(0, 1, 6)[1:-1]]
INK_PRIMARY    = '#0b0b0b'
INK_SECONDARY  = '#52514e'
INK_MUTED      = '#898781'
GRID_COLOR     = '#e1e0d9'
SURFACE        = '#fcfcfb'
REJECTED_COLOR     = '#e1e0d9'
COLOR_THR_GOOD     = MAGMA_1
COLOR_THR_MARGINAL = MAGMA_4

AMP_CMAP_RANGE = (0.20, 0.80)
AMP_CMAP = ListedColormap(matplotlib.colormaps[FIG_CMAP](np.linspace(*AMP_CMAP_RANGE, 256)), name='magma_amp')
AMP_CMAP.set_bad(INK_MUTED)
AMP_CLIP_PCT = (2, 98)


def _to_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ('true', '1', 'yes')
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    return bool(v)


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def _break_scale(a, b, gap, k=1.0):
    """Forward / inverse of a piecewise-linear scale: unchanged up to `a`, the empty
    stretch a → b squeezed to `gap` display units, slope `k` beyond `b`."""
    s = gap / (b - a)

    def fwd(x):
        x = np.asarray(x, dtype=float)
        return np.where(x <= a, x, np.where(x < b, a + (x - a) * s, a + gap + (x - b) * k))

    def inv(y):
        y = np.asarray(y, dtype=float)
        return np.where(y <= a, y, np.where(y < a + gap, a + (y - a) / s, b + (y - a - gap) / k))
    return fwd, inv


def _break_mark(ax, pos, axis):
    """'≈' on the x (bottom) or y (left) spine at data position `pos`."""
    if axis == 'x':
        kw = dict(transform=ax.get_xaxis_transform(), rotation=90, x=pos, y=0)
    else:
        kw = dict(transform=ax.get_yaxis_transform(), rotation=0, x=0, y=pos)
    ax.text(s='≈', ha='center', va='center', fontsize=13, color=INK_SECONDARY, zorder=8,
            clip_on=False, bbox=dict(facecolor=SURFACE, edgecolor='none', pad=0.3), **kw)


def _apply_axis_breaks(ax, bin_edges, counts, thresholds):
    """
    x: thresholds lying further beyond the data than the data range is wide are kept,
       but the empty stretch before them is squeezed into a '≈' break.
    y: when one bin is > Y_BREAK_RATIO × taller than all others, the y-axis is cut
       between the 2nd-tallest bin and the top of the tallest one.
    """
    lo, hi = float(bin_edges[0]), float(bin_edges[-1])
    xt = MaxNLocator(5).tick_values(lo, hi)
    a  = float(xt[xt >= hi - 1e-12][0])                 # round end of the data segment
    far = [t for t in thresholds if t - a > a - lo]
    if far:
        pad  = 0.04 * (a - lo)
        xmin = lo - pad
        b    = min(far) - pad
        gap  = BREAK_GAP_FRAC * (a - xmin)
        ax.set_xscale('function', functions=_break_scale(a, b, gap))
        ax.set_xlim(xmin, max(far) + pad)
        ax.xaxis.set_major_locator(FixedLocator(list(xt[(xt >= lo - 1e-12) & (xt <= a + 1e-12)]) + far))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.axvspan(a, b, color=SURFACE, linewidth=0, zorder=2.5)
        _break_mark(ax, (a + b) / 2, 'x')

    counts = np.asarray(counts)
    srt    = np.sort(counts)
    top, second = int(srt[-1]), int(srt[-2]) if len(srt) > 1 else 0
    if second > 0 and top > Y_BREAK_RATIO * second:
        yt   = MaxNLocator(5, integer=True).tick_values(0, second * 1.15)
        c1   = float(yt[-1])
        ytop = top * 1.06                                 # room for the count label
        c2   = top * 0.92
        up_t = [float(t) for t in MaxNLocator(2, integer=True).tick_values(c2, top) if c2 <= t <= top]
        up_t = up_t[-1:] or [float(top)]                  # one tick only: the segment is short
        gap  = BREAK_GAP_FRAC * c1
        k    = Y_UPPER_FRAC * (c1 + gap) / (1 - Y_UPPER_FRAC) / (ytop - c2)
        ax.set_yscale('function', functions=_break_scale(c1, c2, gap, k))
        ax.set_ylim(0, ytop)
        ax.yaxis.set_major_locator(FixedLocator([float(t) for t in yt if t >= 0] + up_t))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.yaxis.set_minor_locator(FixedLocator([]))
        ax.axhspan(c1, c2, color=SURFACE, linewidth=0, zorder=3.7)   # cuts the tall bar
        _break_mark(ax, (c1 + c2) / 2, 'y')
        i_top = int(np.argmax(counts))
        ax.text(bin_edges[i_top + 1], top, f'{top} units', ha='left', va='top',
                fontsize=8, color=INK_SECONDARY, zorder=7,
                bbox=dict(facecolor=SURFACE, edgecolor='none', pad=1.0))


# ── Data ───────────────────────────────────────────────────────────────────────

df = pd.read_excel(CLEAN_EXCEL, sheet_name=CLEAN_SHEET)
accepted  = df['accepted'].apply(_to_bool).values.astype(bool)
isi       = pd.to_numeric(df['isi_violation_pct'], errors='coerce').values
peak_amps = pd.to_numeric(df['peak_amp_uV'], errors='coerce').values

# Peak-amplitude colour scale (accepted units; same as the main script)
acc_amps = peak_amps[accepted]
amp_lo, amp_hi = np.nanpercentile(acc_amps, AMP_CLIP_PCT)
amp_med  = float(np.nanmedian(acc_amps))
amp_norm = TwoSlopeNorm(vmin=amp_lo, vcenter=amp_med, vmax=amp_hi)

valid = ~np.isnan(isi)
vals, acc, amp = isi[valid], accepted[valid], peak_amps[valid]

# ── Plot ───────────────────────────────────────────────────────────────────────

thresholds = (ISI_GOOD_PCT, ISI_MARGIN_PCT)
thr_colors = (COLOR_THR_GOOD, COLOR_THR_MARGINAL)
thr_labels = (f'Good < {ISI_GOOD_PCT}%', f'Marginal < {ISI_MARGIN_PCT}%')

fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
_style_axes(ax)

bin_edges   = np.histogram_bin_edges(vals, bins=N_BINS)
bin_widths  = np.diff(bin_edges)
bin_centers = bin_edges[:-1] + bin_widths / 2
n_rej, _ = np.histogram(vals[~acc], bins=bin_edges)
n_acc, _ = np.histogram(vals[acc],  bins=bin_edges)

ax.bar(bin_centers, n_rej, width=bin_widths, color=REJECTED_COLOR, edgecolor=SURFACE,
       linewidth=1.0, zorder=3, label=f'Rejected (n={int(n_rej.sum())})')
# Accepted: one unit-height block per unit, coloured by peak amplitude, low → high within a bin
b     = np.clip(np.digitize(vals[acc], bin_edges) - 1, 0, len(bin_widths) - 1)
order = np.lexsort((amp[acc], b))
b_s, amp_s = b[order], amp[acc][order]
rank  = np.arange(len(b_s)) - np.searchsorted(b_s, b_s)
block_colors = AMP_CMAP(amp_norm(amp_s))
ax.bar(bin_centers[b_s], 1.0, width=bin_widths[b_s], bottom=n_rej[b_s] + rank,
       color=block_colors, edgecolor=block_colors, linewidth=0.3, zorder=3)
ax.bar(bin_centers, n_acc, width=bin_widths, bottom=n_rej, fill=False,
       edgecolor=SURFACE, linewidth=1.0, zorder=3.5)

for xv, col, lbl in zip(thresholds, thr_colors, thr_labels):
    ax.axvline(xv, color=col, linewidth=1.5, linestyle='--', zorder=4, label=lbl)
mean_v, median_v = float(np.mean(vals)), float(np.median(vals))
ax.axvline(mean_v,   color=INK_PRIMARY, linewidth=1.5, linestyle='-', zorder=5, label=f'Mean   {mean_v:.3f}')
ax.axvline(median_v, color=INK_PRIMARY, linewidth=1.5, linestyle=':', zorder=5, label=f'Median {median_v:.3f}')

ax.set_xlabel('ISI violations  (%)', color=INK_PRIMARY)
ax.set_ylabel('Number of units', color=INK_PRIMARY)
ax.set_title('ISI Violation Distribution', color=INK_PRIMARY, fontsize=11, loc='left')
_apply_axis_breaks(ax, bin_edges, n_rej + n_acc, thresholds)

# Legend (bars first) + summary
handles, labels = ax.get_legend_handles_labels()
bars  = [i for i, l in enumerate(labels) if l.startswith(('Rejected', 'Accepted'))]
order = bars + [i for i in range(len(labels)) if i not in bars]
handles, labels = [handles[i] for i in order], [labels[i] for i in order]
handles.insert(1, plt.Rectangle((0, 0), 1, 1, color=AMP_CMAP(0.5)))
labels.insert(1, f'Accepted (n={int(n_acc.sum())}), colour = peak amplitude')
ax.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK_PRIMARY,
          loc='upper left', bbox_to_anchor=(1.0, 1.0))
n_zero = int((vals == 0).sum())
summary = (f'n = {len(vals)}\n'
           f'Mean ± SD = {mean_v:.3f} ± {float(np.std(vals)):.3f}\n'
           f'Median = {median_v:.3f}\n'
           f'Exactly 0 = {n_zero} ({100 * n_zero / len(vals):.1f} %)\n'
           f'Max = {float(vals.max()):.3f}')
ax.text(1.02, 0.0, summary, transform=ax.transAxes, fontsize=8, color=INK_SECONDARY,
        ha='left', va='bottom')

# Peak-amplitude colour bar
cbar = fig.colorbar(ScalarMappable(norm=amp_norm, cmap=AMP_CMAP), ax=ax, orientation='horizontal',
                    location='bottom', extend='both', pad=0.18, fraction=0.05, aspect=45)
cbar.set_label(f'Peak amplitude  (µV; centre = median {amp_med:.0f} µV, '
               f'ends = percentiles {AMP_CLIP_PCT[0]} / {AMP_CLIP_PCT[1]})', color=INK_PRIMARY, fontsize=9)
cbar.set_ticks([amp_lo, (amp_lo + amp_med) / 2, amp_med, (amp_med + amp_hi) / 2, amp_hi])
cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
cbar.ax.axvline(amp_med, color=SURFACE, linewidth=1.2)
cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
cbar.outline.set_edgecolor(INK_MUTED)

excel_base = os.path.splitext(os.path.basename(OUTPUT_EXCEL))[0]
out_base   = os.path.join(PLOT_DIR, f'{excel_base}_MTclean_hist_ISI')
for ext in ('png', 'svg'):
    fig.savefig(f'{out_base}.{ext}', dpi=300, bbox_inches='tight', facecolor=SURFACE)
plt.close(fig)

print(f'Units with ISI value: {len(vals)}  |  exactly 0 %: {n_zero}  |  > {ISI_GOOD_PCT} %: '
      f'{int((vals > ISI_GOOD_PCT).sum())}  |  >= {ISI_MARGIN_PCT} %: {int((vals >= ISI_MARGIN_PCT).sum())}  |  '
      f'max: {vals.max():.3f} %')
print(f'Saved: {out_base}.png / .svg')
