# -*- coding: utf-8 -*-
"""
Plot all single-unit quality metrics from the QUALITY_METRICS.xlsx workbook
written by Fig1def_spike_quality_v2.py, after re-computing the waveform-shape
metrics of multi-trigger units on their clean spikes only.

For every row flagged multi_trigger == True, the .ntt file is located at
    ROOT_FOLDER / <first column of the sheet (session)> / <unit>
and the following metrics are re-computed on the spikes that have NO
secondary trigger (same detection rule and parameters as
Fig1def_spike_quality_v2.py; cluster 0 is excluded as before):

  spike_duration_ms     – onset of depolarization → end of AHP (best channel, mean waveform)
  template_correlation  – mean Pearson r of each spike vs. mean template
  pt_time_ms            – peak-to-trough time (best channel, mean waveform)
  pt_ratio              – |peak| / |trough| (best channel, mean waveform)
  peak_amp_uV           – peak amplitude (best channel, mean waveform)

The re-computed values replace the originals for those units in every plot.
Outputs (next to OUTPUT_EXCEL):
  <name>_MTclean.xlsx            – corrected metrics table + before/after table for MT units
  quality_plots_MTclean/         – one histogram per metric, an all-metrics grid,
                                   a before/after comparison and per-unit MT waveforms
"""

import os
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib # type: ignore
matplotlib.use('Agg')
import matplotlib.pyplot as plt # type: ignore
from matplotlib.colors import to_hex, ListedColormap, Normalize, TwoSlopeNorm # type: ignore
from matplotlib.cm import ScalarMappable # type: ignore

# ── Configuration ──────────────────────────────────────────────────────────────

# Root folder containing the session subfolders with the .ntt spike files
ROOT_FOLDER  = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data'
OUTPUT_EXCEL = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data\QUALITY_METRICS.xlsx'
SHEET_NAME   = 'Quality Metrics'

CLEAN_EXCEL = os.path.splitext(OUTPUT_EXCEL)[0] + '_MTclean.xlsx'
PLOT_DIR    = os.path.join(os.path.dirname(OUTPUT_EXCEL), 'quality_plots_MTclean')
os.makedirs(PLOT_DIR, exist_ok=True)

N_SAMPLES            = 32        # samples per waveform (NTT tetrode standard)
WAVEFORM_DURATION_MS = 1.0       # 32 samples @ 32 kHz (NTT header SamplingFrequey 32000)
SAMPLE_INTERVAL_MS   = WAVEFORM_DURATION_MS / N_SAMPLES

# Must match Fig1def_spike_quality_v2.py so the same spikes are excluded
SPIKE_DURATION_THRESHOLD_K   = 2.0
MULTI_TRIGGER_OFFSET_SAMPLES = 4
MULTI_TRIGGER_AMP_FRAC       = 0.5
MULTI_TRIGGER_NOISE_K        = 3.0

# Thresholds drawn on the histograms (same as Fig1def_spike_quality_v2.py)
SNR_LOW_BAD, SNR_LOW_OK, SNR_HIGH_GOOD = 1.5, 2.5, 4.0
ISI_GOOD_PCT, ISI_MARGIN_PCT           = 0.1, 5.0

# ── Figure palette (magma colormap + neutral inks, as in Fig1def_spike_quality_v2.py) ──
FIG_CMAP = 'magma'
MAGMA_1, MAGMA_2, MAGMA_3, MAGMA_4 = [to_hex(matplotlib.colormaps[FIG_CMAP](v))
                                      for v in np.linspace(0, 1, 6)[1:-1]]
REJECT_COLOR   = '#e1e0d9'
INK_PRIMARY    = '#0b0b0b'
INK_SECONDARY  = '#52514e'
INK_MUTED      = '#898781'
GRID_COLOR     = '#e1e0d9'
SURFACE        = '#fcfcfb'

ACCEPTED_COLOR     = MAGMA_2
REJECTED_COLOR     = REJECT_COLOR
MT_COLOR           = MAGMA_3     # multi-trigger units (rug marks, "clean" values)
COLOR_THR_GOOD     = MAGMA_1
COLOR_THR_BAD      = MAGMA_3
COLOR_THR_MARGINAL = MAGMA_4

# Accepted units in the histograms are coloured by peak amplitude (low → high along magma).
# Magma is trimmed symmetrically so its very light top end is avoided and the colormap's
# centre stays magma(0.5), which is placed on the median peak amplitude.
AMP_CMAP_RANGE = (0.20, 0.80)
AMP_CMAP = ListedColormap(matplotlib.colormaps[FIG_CMAP](np.linspace(*AMP_CMAP_RANGE, 256)), name='magma_amp')
AMP_CMAP.set_bad(INK_MUTED)
AMP_CLIP_PCT = (2, 98)           # colour scale ends = these percentiles of accepted peak amplitudes


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


def _legend_outside(ax, handles=None, labels=None):
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK_PRIMARY,
              loc='upper left', bbox_to_anchor=(1.0, 1.0))


def _to_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ('true', '1', 'yes')
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    return bool(v)


# ── Neuralynx .ntt loading (copied from Fig1def_spike_quality_v2.py) ──────────

NTT_HEADER_BYTES   = 16 * 1024
DEFAULT_ADBITVOLTS = 0.000000195

NTT_DTYPE = np.dtype([
    ('timestamp',   '<u8'),
    ('sc_number',   '<u4'),
    ('cell_number', '<u4'),
    ('params',      '<u4', (8,)),
    ('waveforms',   '<i2', (32, 4)),
])


def _parse_adbitvolts(path: str) -> float:
    try:
        with open(path, 'rb') as fh:
            header_raw = fh.read(NTT_HEADER_BYTES)
        header_txt = header_raw.decode('latin-1', errors='replace')
        for line in header_txt.splitlines():
            if 'ADBitVolts' in line:
                for p in line.split():
                    try:
                        val = float(p)
                        if 0 < val < 1:
                            return val
                    except ValueError:
                        pass
    except Exception:
        pass
    return DEFAULT_ADBITVOLTS


def load_ntt_waveforms(ntt_path: str):
    """Waveforms (µV, (n, 32, 4)) of all non-cluster-0 spikes."""
    uv_scale   = _parse_adbitvolts(ntt_path) * 1e6
    spike_data = np.memmap(ntt_path, dtype=NTT_DTYPE, mode='r', offset=NTT_HEADER_BYTES)
    spike_data = spike_data[spike_data['cell_number'] != 0]  # type: ignore[index]
    return spike_data['waveforms'].astype(np.float64) * uv_scale  # type: ignore[operator]


# ── Metric functions (identical definitions to Fig1def_spike_quality_v2.py) ───

def compute_template_correlation(waveforms: np.ndarray) -> float:
    """Mean Pearson r between each spike waveform and the mean template, averaged across channels."""
    if waveforms.shape[0] < 2:
        return float('nan')
    template = waveforms.mean(axis=0)
    t = template - template.mean(axis=0)
    w = waveforms - waveforms.mean(axis=1, keepdims=True)

    numer    = np.sum(t[np.newaxis] * w, axis=1)
    t_sq_sum = np.sum(t ** 2, axis=0)
    w_sq_sum = np.sum(w ** 2, axis=1)
    denom    = np.sqrt(t_sq_sum[np.newaxis] * w_sq_sum)

    with np.errstate(invalid='ignore', divide='ignore'):
        corr = np.where(denom > 0, numer / denom, np.nan)
    return float(np.nanmean(corr))


def compute_spike_duration(wf: np.ndarray, peak_idx: int, trough_idx: int,
                           k: float = SPIKE_DURATION_THRESHOLD_K) -> dict:
    """Onset of depolarization → end of AHP, from baseline-band crossings around peak/trough."""
    n = wf.shape[0]
    baseline      = wf[:5]
    baseline_mean = float(np.mean(baseline))
    mad           = np.median(np.abs(baseline - np.median(baseline)))
    sigma         = float(mad / 0.6745) if mad > 0 else float(np.std(baseline))
    if sigma == 0:
        sigma = float(np.finfo(float).eps)
    threshold = k * sigma

    first_ext = min(peak_idx, trough_idx)
    last_ext  = max(peak_idx, trough_idx)

    start_idx = 0
    for i in range(first_ext, -1, -1):
        if abs(wf[i] - baseline_mean) < threshold:
            start_idx = i
            break

    end_idx = n - 1
    for i in range(last_ext, n):
        if abs(wf[i] - baseline_mean) < threshold:
            end_idx = i
            break

    return {'start_idx': start_idx, 'end_idx': end_idx,
            'spike_duration_ms': round((end_idx - start_idx) * SAMPLE_INTERVAL_MS, 5)}


def compute_pt(waveforms: np.ndarray) -> dict:
    """Peak/trough metrics on the channel whose mean waveform reaches the highest peak."""
    mean_wf = waveforms.mean(axis=0)
    best_ch = int(np.argmax(mean_wf.max(axis=0)))
    wf      = mean_wf[:, best_ch]

    peak_idx   = int(np.argmax(wf))
    trough_idx = int(np.argmin(wf))
    peak_val   = float(wf[peak_idx])
    trough_val = float(wf[trough_idx])
    pt_ratio   = abs(peak_val) / abs(trough_val) if trough_val != 0 else float('nan')
    dur        = compute_spike_duration(wf, peak_idx, trough_idx)

    return {
        'best_channel':      best_ch,
        'peak_idx':          peak_idx,
        'trough_idx':        trough_idx,
        'peak_amp_uV':       round(peak_val, 4),
        'trough_amp_uV':     round(trough_val, 4),
        'pt_time_ms':        round(abs(peak_idx - trough_idx) * SAMPLE_INTERVAL_MS, 5),
        'pt_ratio':          round(pt_ratio, 4) if not np.isnan(pt_ratio) else float('nan'),
        'spike_start_idx':   dur['start_idx'],
        'spike_end_idx':     dur['end_idx'],
        'spike_duration_ms': dur['spike_duration_ms'],
        'mean_waveform':     wf,
    }


def detect_multi_triggers(waveforms: np.ndarray, best_ch: int, peak_idx: int,
                          offset: int = MULTI_TRIGGER_OFFSET_SAMPLES,
                          amp_frac: float = MULTI_TRIGGER_AMP_FRAC,
                          noise_k: float = MULTI_TRIGGER_NOISE_K) -> np.ndarray:
    """(n_spikes,) count of secondary peaks after the main peak on `best_ch`."""
    wf = waveforms[:, :, best_ch]
    n_spk, n_smp = wf.shape
    start = peak_idx + max(int(offset), 1)
    if n_spk == 0 or start >= n_smp:
        return np.zeros(n_spk, dtype=int)

    baseline = wf[:, :5].flatten()
    sigma    = float(np.median(np.abs(baseline - np.median(baseline))) / 0.6745)

    main_amp = wf[:, peak_idx]
    prev = wf[:, start - 1:n_smp - 1]
    cur  = wf[:, start:]
    nxt  = np.concatenate([wf[:, start + 1:], np.full((n_spk, 1), -np.inf)], axis=1)

    thr     = np.maximum(amp_frac * main_amp, noise_k * sigma)
    is_peak = (cur > prev) & (cur >= nxt) & (cur > thr[:, None]) & (main_amp > 0)[:, None]
    return is_peak.sum(axis=1)


RECOMPUTED_KEYS = ['spike_duration_ms', 'template_correlation', 'pt_time_ms', 'pt_ratio', 'peak_amp_uV']


def recompute_without_multi_triggers(ntt_path: str):
    """
    Metrics in RECOMPUTED_KEYS on the spikes without secondary triggers.
    Best channel / peak index for the detection come from the mean of ALL
    spikes, exactly as in Fig1def_spike_quality_v2.py.
    Returns (metrics dict, pt_all, pt_clean).
    """
    waveforms = load_ntt_waveforms(ntt_path)
    pt_all    = compute_pt(waveforms)
    counts    = detect_multi_triggers(waveforms, pt_all['best_channel'], pt_all['peak_idx'])
    clean     = waveforms[counts == 0]

    out = {'n_spikes_clean': int(clean.shape[0]),
           'n_mt_spikes_excluded': int((counts > 0).sum())}
    if clean.shape[0] == 0:
        out.update({k: float('nan') for k in RECOMPUTED_KEYS})
        return out, pt_all, None

    pt_clean = compute_pt(clean)
    out.update({
        'spike_duration_ms':    pt_clean['spike_duration_ms'],
        'template_correlation': round(compute_template_correlation(clean), 4),
        'pt_time_ms':           pt_clean['pt_time_ms'],
        'pt_ratio':             pt_clean['pt_ratio'],
        'peak_amp_uV':          pt_clean['peak_amp_uV'],
        'pt_best_channel_clean': pt_clean['best_channel'],
    })
    return out, pt_all, pt_clean


def plot_mt_waveforms(pt_all: dict, pt_clean: dict, out_base: str, title: str, n_excl: int, n_clean: int):
    """Mean waveform of all spikes vs. clean spikes (each on its own best channel)."""
    time_axis = np.arange(N_SAMPLES) * SAMPLE_INTERVAL_MS
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
    _style_axes(ax)
    ax.plot(time_axis, pt_all['mean_waveform'], color=INK_MUTED, lw=1.6, ls='--',
            label=f'All spikes (ch {pt_all["best_channel"]})')
    ax.plot(time_axis, pt_clean['mean_waveform'], color=MT_COLOR, lw=1.8,
            label=f'MT spikes excluded (ch {pt_clean["best_channel"]})')
    wf = pt_clean['mean_waveform']
    s, e = pt_clean['spike_start_idx'], pt_clean['spike_end_idx']
    ax.axvspan(s * SAMPLE_INTERVAL_MS, e * SAMPLE_INTERVAL_MS, color=MAGMA_4, alpha=0.18, lw=0, zorder=1)
    ax.scatter([pt_clean['peak_idx'] * SAMPLE_INTERVAL_MS], [wf[pt_clean['peak_idx']]],
               facecolor=MAGMA_3, edgecolor=SURFACE, s=45, zorder=5)
    ax.scatter([pt_clean['trough_idx'] * SAMPLE_INTERVAL_MS], [wf[pt_clean['trough_idx']]],
               facecolor=MAGMA_1, edgecolor=SURFACE, s=45, zorder=5)
    ax.set_xlabel('Time (ms)', color=INK_PRIMARY)
    ax.set_ylabel('Amplitude (µV)', color=INK_PRIMARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=10, loc='left')
    _legend_outside(ax)
    note = (f'Excluded {n_excl} MT spikes, kept {n_clean}\n'
            f'P-T time = {pt_clean["pt_time_ms"]:.3f} ms\n'
            f'P/T ratio = {pt_clean["pt_ratio"]:.3f}\n'
            f'Peak amp = {pt_clean["peak_amp_uV"]:.1f} µV\n'
            f'Spike duration = {pt_clean["spike_duration_ms"]:.3f} ms')
    ax.text(1.02, 0.0, note, transform=ax.transAxes, fontsize=8, color=INK_SECONDARY,
            ha='left', va='bottom')
    _save(fig, out_base, exts=('png',))


# ── Load workbook and re-compute multi-trigger units ──────────────────────────

df = pd.read_excel(OUTPUT_EXCEL, sheet_name=SHEET_NAME, engine='openpyxl')
session_col = df.columns[0]
df['accepted']      = df['accepted'].apply(_to_bool)
df['multi_trigger'] = df['multi_trigger'].apply(_to_bool)
df['n_spikes_clean']       = np.nan
df['n_mt_spikes_excluded'] = np.nan
df['pt_best_channel_clean'] = np.nan

mt_rows   = df.index[df['multi_trigger']].tolist()
mt_before = df.loc[mt_rows, [session_col, 'unit'] + RECOMPUTED_KEYS].copy()
print(f'{len(mt_rows)} multi-trigger unit(s) found in {os.path.basename(OUTPUT_EXCEL)}')

mt_wf_dir = os.path.join(PLOT_DIR, 'MT_waveforms')
os.makedirs(mt_wf_dir, exist_ok=True)

recomputed_ok = []
for idx in mt_rows:
    session = str(df.at[idx, session_col])
    unit    = str(df.at[idx, 'unit'])
    ntt_path = os.path.normpath(os.path.join(ROOT_FOLDER, session, unit))
    print(f'Re-computing: {session}  |  {unit}')
    if not os.path.isfile(ntt_path):
        print(f'  ERROR: file not found: {ntt_path}')
        continue
    try:
        metrics, pt_all, pt_clean = recompute_without_multi_triggers(ntt_path)
    except Exception as exc:
        print(f'  ERROR: {exc}')
        continue
    for k, v in metrics.items():
        df.at[idx, k] = v
    recomputed_ok.append(idx)
    print(f'  excluded {metrics["n_mt_spikes_excluded"]} MT spikes, kept {metrics["n_spikes_clean"]}')
    if pt_clean is not None:
        label = f'{session}_{os.path.splitext(unit)[0]}'.replace(os.sep, '_').replace('/', '_')
        plot_mt_waveforms(pt_all, pt_clean, os.path.join(mt_wf_dir, f'{label}_MTclean'),
                          f'{session} | {unit}',
                          metrics['n_mt_spikes_excluded'], metrics['n_spikes_clean'])

# ── Save corrected table + before/after table ─────────────────────────────────

mt_after   = df.loc[mt_rows, RECOMPUTED_KEYS + ['n_spikes_clean', 'n_mt_spikes_excluded']]
comparison = mt_before.copy()
for k in RECOMPUTED_KEYS:
    comparison = comparison.rename(columns={k: f'{k}_all'})
    comparison[f'{k}_clean'] = mt_after[k]
comparison['n_spikes_all']         = df.loc[mt_rows, 'n_spikes']
comparison['n_spikes_clean']       = mt_after['n_spikes_clean']
comparison['n_mt_spikes_excluded'] = mt_after['n_mt_spikes_excluded']
comparison['accepted']             = df.loc[mt_rows, 'accepted']

with pd.ExcelWriter(CLEAN_EXCEL, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Quality Metrics (MT clean)', index=False)
    comparison.to_excel(writer, sheet_name='MT units before-after', index=False)
print(f'\nCorrected metrics saved: {CLEAN_EXCEL}')

# ── Histograms of all quality metrics ─────────────────────────────────────────

def draw_histogram(ax, values, accepted, is_mt, xlabel, thresholds=(), thr_colors=(), thr_labels=(),
                   n_bins=30, legend=True, amp=None, amp_norm=None):
    """
    Stacked histogram (rejected in gray at the bottom, accepted in colour on top),
    with multi-trigger units (values re-computed on clean spikes) marked as a rug
    below the bars.
    If `amp` is given, every accepted unit is drawn as its own unit-height block,
    coloured by its peak amplitude (AMP_CMAP / amp_norm) and stacked low → high
    amplitude within each bin.
    """
    values   = np.asarray(values, dtype=float)
    valid    = ~np.isnan(values)
    vals     = values[valid]
    acc      = np.asarray(accepted, dtype=bool)[valid]
    mt       = np.asarray(is_mt, dtype=bool)[valid]
    if amp is not None:
        amp = np.asarray(amp, dtype=float)[valid]
        if amp_norm is None:
            amp_norm = Normalize(np.nanmin(amp), np.nanmax(amp))

    if len(vals) == 0:
        ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color=INK_MUTED)
        ax.set_xlabel(xlabel, color=INK_PRIMARY)
        return

    bin_edges   = np.histogram_bin_edges(vals, bins=n_bins)
    bin_widths  = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + bin_widths / 2
    n_rej, _ = np.histogram(vals[~acc], bins=bin_edges)
    n_acc, _ = np.histogram(vals[acc],  bins=bin_edges)

    ax.bar(bin_centers, n_rej, width=bin_widths, color=REJECTED_COLOR, edgecolor=SURFACE,
           linewidth=1.0, zorder=3, label=f'Rejected (n={int(n_rej.sum())})')
    if amp is None:
        ax.bar(bin_centers, n_acc, width=bin_widths, bottom=n_rej, color=ACCEPTED_COLOR,
               edgecolor=SURFACE, linewidth=1.0, zorder=3, label=f'Accepted (n={int(n_acc.sum())})')
    else:
        b   = np.clip(np.digitize(vals[acc], bin_edges) - 1, 0, len(bin_widths) - 1)
        order = np.lexsort((amp[acc], b))
        b_s, amp_s = b[order], amp[acc][order]
        rank = np.arange(len(b_s)) - np.searchsorted(b_s, b_s)   # position within its bin
        block_colors = AMP_CMAP(amp_norm(amp_s))
        # Edges in the block's own colour hide hairline seams between stacked blocks
        ax.bar(bin_centers[b_s], 1.0, width=bin_widths[b_s], bottom=n_rej[b_s] + rank,
               color=block_colors, edgecolor=block_colors, linewidth=0.3, zorder=3)
        # Outline restores the gap between bins drawn by the SURFACE edges elsewhere
        ax.bar(bin_centers, n_acc, width=bin_widths, bottom=n_rej, fill=False,
               edgecolor=SURFACE, linewidth=1.0, zorder=3.5)
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

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        # Bars first (matplotlib lists lines before bars)
        bars  = [i for i, l in enumerate(labels) if l.startswith(('Rejected', 'Accepted'))]
        order = bars + [i for i in range(len(labels)) if i not in bars]
        handles, labels = [handles[i] for i in order], [labels[i] for i in order]
        if amp is not None:
            # Proxy patch for the per-unit blocks (legend entry 2, after "Rejected")
            handles.insert(1, plt.Rectangle((0, 0), 1, 1, color=AMP_CMAP(0.5)))
            labels.insert(1, f'Accepted (n={int(n_acc.sum())}), colour = peak amplitude')
        _legend_outside(ax, handles, labels)
        summary = (f'n = {len(vals)}\n'
                   f'Mean ± SD = {mean_v:.3f} ± {float(np.std(vals)):.3f}\n'
                   f'Median = {median_v:.3f}')
        ax.text(1.02, 0.0, summary, transform=ax.transAxes, fontsize=8, color=INK_SECONDARY,
                ha='left', va='bottom')


histogram_specs = [
    # (file suffix, df column, title, xlabel, thresholds, threshold colours, threshold labels)
    ('NSpikes',  'n_spikes',             'Number of Spikes',             'Spikes per unit',
     (), (), ()),
    ('SNR',      'snr',                  'SNR Distribution',             'SNR  (A_peak / σ_noise)',
     (SNR_LOW_BAD, SNR_LOW_OK, SNR_HIGH_GOOD), (COLOR_THR_BAD, COLOR_THR_MARGINAL, COLOR_THR_GOOD),
     (f'Too low {SNR_LOW_BAD}', f'Marginal {SNR_LOW_OK}', f'High {SNR_HIGH_GOOD}')),
    ('ISI',      'isi_violation_pct',    'ISI Violation Distribution',   'ISI violations  (%)',
     (ISI_GOOD_PCT, ISI_MARGIN_PCT), (COLOR_THR_GOOD, COLOR_THR_MARGINAL),
     (f'Good < {ISI_GOOD_PCT}%', f'Marginal < {ISI_MARGIN_PCT}%')),
    ('TmplCorr', 'template_correlation', 'Waveform Template Correlation', 'Mean Pearson r  (spike vs template)',
     (0.90,), (COLOR_THR_GOOD,), ('r = 0.90',)),
    ('PeakAmp',  'peak_amp_uV',          'Peak Amplitude',               'Peak amplitude  (µV)',
     (), (), ()),
    ('PTTime',   'pt_time_ms',           'Peak-to-Trough Time',          'PT time  (ms)',
     (), (), ()),
    ('PTRatio',  'pt_ratio',             'Peak-to-Trough Ratio',         '|peak| / |trough|  (a.u.)',
     (1.0,), (INK_MUTED,), ('ratio = 1.0',)),
    ('SpikeDur', 'spike_duration_ms',    'Spike Duration',               'Spike duration  (ms)',
     (), (), ()),
    ('CSI',      'csi',                  'Complex Spike Index (CSI)',    'CSI  (% of ISIs in [3–20 ms] with smaller 2nd spike)',
     (10.0,), (COLOR_THR_BAD,), ('CSI = 10%',)),
    ('CSImod',   'csi_modified',         'CSI modified (no amplitude criterion)', 'CSI modified  (% of ISIs in [3–20 ms])',
     (10.0,), (COLOR_THR_BAD,), ('CSI = 10%',)),
]

accepted_flags = df['accepted'].values
mt_flags       = df.index.isin(recomputed_ok)
excel_base     = os.path.splitext(os.path.basename(OUTPUT_EXCEL))[0]

# Peak-amplitude colour scale shared by all histograms (accepted units only):
# AMP_CLIP_PCT[0] → median → AMP_CLIP_PCT[1] map onto the lower and upper halves of AMP_CMAP
peak_amps = pd.to_numeric(df['peak_amp_uV'], errors='coerce').values
acc_amps  = peak_amps[accepted_flags.astype(bool)]
amp_lo, amp_hi = np.nanpercentile(acc_amps, AMP_CLIP_PCT)
amp_med   = float(np.nanmedian(acc_amps))
amp_norm  = TwoSlopeNorm(vmin=amp_lo, vcenter=amp_med, vmax=amp_hi)
AMP_LABEL = (f'Peak amplitude  (µV; centre = median {amp_med:.0f} µV, '
             f'ends = percentiles {AMP_CLIP_PCT[0]} / {AMP_CLIP_PCT[1]})')


def _amp_colorbar(cbar):
    cbar.set_label(AMP_LABEL, color=INK_PRIMARY, fontsize=9)
    cbar.set_ticks([amp_lo, (amp_lo + amp_med) / 2, amp_med, (amp_med + amp_hi) / 2, amp_hi])
    cbar.ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))
    cbar.ax.axvline(amp_med, color=SURFACE, linewidth=1.2)
    cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    cbar.outline.set_edgecolor(INK_MUTED)


for suffix, col, title, xlabel, thr, thr_c, thr_l in histogram_specs:
    if col not in df.columns:
        print(f'Column "{col}" not in sheet, skipped')
        continue
    color_by_amp = col != 'peak_amp_uV'
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    _style_axes(ax)
    draw_histogram(ax, pd.to_numeric(df[col], errors='coerce').values, accepted_flags, mt_flags,
                   xlabel, thr, thr_c, thr_l,
                   amp=peak_amps if color_by_amp else None, amp_norm=amp_norm)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc='left')
    if color_by_amp:
        _amp_colorbar(fig.colorbar(ScalarMappable(norm=amp_norm, cmap=AMP_CMAP), ax=ax,
                                   orientation='horizontal', location='bottom', extend='both',
                                   pad=0.18, fraction=0.05, aspect=45))
    _save(fig, os.path.join(PLOT_DIR, f'{excel_base}_MTclean_hist_{suffix}'))
    print(f'Histogram saved: {suffix}')

# All-metrics grid (shared legend at the bottom)
specs_present = [s for s in histogram_specs if s[1] in df.columns]
n_cols = 3
n_rows = int(np.ceil(len(specs_present) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.6 * n_rows), facecolor=SURFACE)
axes = np.atleast_1d(axes).ravel()
for ax, (suffix, col, title, xlabel, thr, thr_c, thr_l) in zip(axes, specs_present):
    _style_axes(ax)
    draw_histogram(ax, pd.to_numeric(df[col], errors='coerce').values, accepted_flags, mt_flags,
                   xlabel, thr, thr_c, thr_l, legend=False,
                   amp=peak_amps if col != 'peak_amp_uV' else None, amp_norm=amp_norm)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc='left')
for ax in axes[len(specs_present):]:
    ax.set_visible(False)
grid_handles = [plt.Rectangle((0, 0), 1, 1, color=AMP_CMAP(0.5)),
                plt.Rectangle((0, 0), 1, 1, color=REJECTED_COLOR),
                plt.Line2D([], [], linestyle='none', marker='|', markersize=7,
                           markeredgewidth=0.8, color=MT_COLOR),
                plt.Line2D([], [], color=INK_PRIMARY, lw=1.5),
                plt.Line2D([], [], color=INK_PRIMARY, lw=1.5, ls=':')]
fig.legend(grid_handles, ['Accepted (colour = peak amplitude; solid in the peak-amplitude panel)',
                          'Rejected', 'Multi-trigger (MT spikes excluded)', 'Mean', 'Median'],
           loc='lower center', ncol=5, frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
fig.tight_layout(rect=(0, 0.11, 1, 1))
grid_cax = fig.add_axes((0.35, 0.065, 0.30, 0.012))
_amp_colorbar(fig.colorbar(ScalarMappable(norm=amp_norm, cmap=AMP_CMAP), cax=grid_cax,
                           orientation='horizontal', extend='both'))
grid_cax.xaxis.set_label_position('top')
_save(fig, os.path.join(PLOT_DIR, f'{excel_base}_MTclean_all_metrics'))
print('All-metrics grid saved')

# ── Before/after comparison for multi-trigger units ───────────────────────────

if recomputed_ok:
    compare_specs = [('spike_duration_ms', 'Spike duration (ms)'),
                     ('template_correlation', 'Template correlation (r)'),
                     ('pt_time_ms', 'PT time (ms)'),
                     ('pt_ratio', '|peak| / |trough|'),
                     ('peak_amp_uV', 'Peak amplitude (µV)')]
    fig, axes = plt.subplots(1, len(compare_specs), figsize=(3.0 * len(compare_specs), 4.0),
                             facecolor=SURFACE)
    for ax, (key, ylabel) in zip(axes, compare_specs):
        _style_axes(ax)
        before = pd.to_numeric(mt_before.loc[recomputed_ok, key], errors='coerce').values
        after  = pd.to_numeric(df.loc[recomputed_ok, key], errors='coerce').values
        for b, a in zip(before, after):
            ax.plot([0, 1], [b, a], color=INK_MUTED, lw=0.8, alpha=0.6, zorder=2)
        ax.scatter(np.zeros(len(before)), before, s=22, color=INK_SECONDARY, zorder=3)
        ax.scatter(np.ones(len(after)),   after,  s=22, color=MT_COLOR,      zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['All spikes', 'MT excluded'], color=INK_PRIMARY)
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylabel(ylabel, color=INK_PRIMARY)
    fig.suptitle(f'Multi-trigger units (n = {len(recomputed_ok)}): metrics before vs. after '
                 f'excluding multi-trigger spikes', color=INK_PRIMARY, fontsize=11, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, os.path.join(PLOT_DIR, f'{excel_base}_MTclean_before_after'))
    print('Before/after comparison saved')

print(f'\nDone. Plots in {PLOT_DIR}')
print(f'Multi-trigger units re-computed: {len(recomputed_ok)} of {len(mt_rows)}')
