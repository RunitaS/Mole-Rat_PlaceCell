# -*- coding: utf-8 -*-
"""
Single-unit quality metrics for Neuralynx .ntt files.
Outputs a colour-coded Excel workbook with per-unit metrics and individual histograms.

Metrics computed
────────────────
  SNR                  – A_peak / σ_noise  (MAD-based: σ = MAD/0.6745 on first-5-sample baseline)
  ISI violation %      – fraction of ISIs < 2 ms (good: < 1 %)
  Template correlation – mean Pearson r of each spike vs. mean waveform
  Peak/trough time+amp – peak and trough time (ms) and amplitude (µV) on best channel (mean waveform)
  PT time (ms)         – peak-to-trough time on best channel (mean waveform)
  PT ratio             – |peak| / |trough| on best channel (mean waveform)
  Spike duration (ms)  – time from onset of depolarization to end of the after-hyperpolarization
                         on best channel (mean waveform); onset/offset are the baseline threshold
                         crossings that bracket the peak and trough
  Presence ratio       – fraction of 1-s bins that contain ≥ 1 spike
  CSI                  – complex spike index: % of all ISIs in [3, 20] ms
                         where the 2nd spike amplitude < 1st (Bhatt et al. 2020)
  CSI modified         – same as CSI but WITHOUT the decreasing-amplitude criterion:
                         % of all ISIs in [3, 20] ms

Acceptance criteria (unit is "accepted" only if ALL three hold)
─────────────────────────────────────────────────────────────
  ISI violation %       < 1 %
  SNR                   > 2.5
  Template correlation  > 0.5

Multi-trigger (bursting) review
──────────────────────────────
  For every spike, local maxima on the best channel that occur ≥ MULTI_TRIGGER_OFFSET_SAMPLES
  after the main peak and exceed both MULTI_TRIGGER_AMP_FRAC × that spike's main peak and
  MULTI_TRIGGER_NOISE_K × baseline σ are counted as secondary triggers. Units with more than
  MULTI_TRIGGER_MIN_PEAKS such peaks are held back and shown in a GUI after the scan, where
  each can be manually accepted, rejected, or accepted with its multi-trigger spikes moved
  to cluster 0 (the copy in the accepted tree is rewritten so those spikes have
  cell_number = 0; the original .ntt is never modified). The manual decision overrides the
  automatic one (for copying and for the accepted/rejected colouring in every histogram).
  If USE_SAVED_MT_LABELS is True and SAVED_MT_LABELS_FILE exists (.csv or .xlsx with
  session / unit / manual_review columns, e.g. a previous QUALITY_METRICS.xlsx or
  <name>_MT_labels.csv), units labelled there take the saved decision and only the
  remaining ones are shown in the GUI. Every run writes the decisions to
  <OUTPUT_EXCEL name>_MT_labels.csv.

Accepted units' .ntt files, plus every .csv and .ncs file found alongside them,
are copied into a sibling '<ROOT_FOLDER>_Accepted' tree that mirrors the
session subfolder structure. In every histogram below, rejected units are
drawn as a gray segment stacked at the bottom of each bin, with accepted
units in color on top.

References
──────────
Lee D et al. (2018) Exp Neurobiol 27:593-604.
Ludwig KA et al. (2011) J Neural Eng 8:014001.
Bhatt DL et al. / Bhatt et al. 2020 — Cell 2000 (S0092-8674(00)81828-0)
"""

import os
import shutil
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib # type: ignore
matplotlib.use('Agg')
import matplotlib.pyplot as plt # type: ignore
from matplotlib.colors import to_hex # type: ignore
from openpyxl import load_workbook # type: ignore
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side # type: ignore
from openpyxl.utils import get_column_letter # type: ignore
from openpyxl.drawing.image import Image as XLImage # type: ignore

# ── Configuration ──────────────────────────────────────────────────────────────

ROOT_FOLDER  = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data'
OUTPUT_EXCEL = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data\QUALITY_METRICS.xlsx'

# Sibling folder that receives accepted units' .ntt files plus every .csv/.ncs
# file found in the same session folder (mirrors ROOT_FOLDER's subfolder tree).
ACCEPTED_ROOT_FOLDER = ROOT_FOLDER.rstrip('\\/') + '_SpkQltyFilt'
# True: copy accepted units (+ session .csv/.ncs files) into ACCEPTED_ROOT_FOLDER.
# False: compute metrics/figures only, no files are copied.
SAVE_ACCEPTED_COPY = True

# True: take the manually curated multi-trigger labels (accepted / rejected /
# accepted, MT spikes -> cl 0) from SAVED_MT_LABELS_FILE instead of re-reviewing
# them in the GUI. Units not found in the file are still shown in the GUI.
# False: review every multi-trigger unit in the GUI.
USE_SAVED_MT_LABELS = True
SAVED_MT_LABELS_FILE = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data\QUALITY_METRICS.xlsx'

# ROOT_FOLDER  = r'C:/Runita/NMR/analysis/SurgeryPaperSpikeLFP/8477/L_Iso_CSI_Test'
# OUTPUT_EXCEL = r'C:/Runita/NMR/analysis/SurgeryPaperSpikeLFP/8477/L_Iso_CSI_Test/QUALITY_METRICS.xlsx'

N_SAMPLES            = 32        # samples per waveform (NTT tetrode standard)
WAVEFORM_DURATION_MS = 1.0       # 32 samples @ 32 kHz (NTT header SamplingFrequency 32000)
SAMPLE_INTERVAL_MS   = WAVEFORM_DURATION_MS / N_SAMPLES

# ── Figure palette (magma colormap + neutral inks, as in PlaceCell_CompareRuns.py) ──
# Data colours are equidistant on the magma colormap: n colours sit at
# k/(n+1), k = 1..n (here 0.2, 0.4, 0.6, 0.8), so the gaps between colours
# and to both ends of the map are equal and the near-black / near-white ends
# are never used.
FIG_CMAP = 'magma'
MAGMA_1, MAGMA_2, MAGMA_3, MAGMA_4 = [to_hex(matplotlib.colormaps[FIG_CMAP](v))
                                      for v in np.linspace(0, 1, 6)[1:-1]]
REJECT_COLOR   = '#e1e0d9'
INK_PRIMARY    = '#0b0b0b'
INK_SECONDARY  = '#52514e'
INK_MUTED      = '#898781'
GRID_COLOR     = '#e1e0d9'
SURFACE        = '#fcfcfb'


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


# ── Waveform plot colours ──────────────────────────────────────────────────────
COLOR_WAVEFORM = INK_SECONDARY
COLOR_PEAK     = MAGMA_3
COLOR_TROUGH   = MAGMA_1
COLOR_DURATION = MAGMA_4

# ── Histogram threshold-line colours (same meaning in every histogram) ────────
COLOR_THR_GOOD     = MAGMA_1
COLOR_THR_BAD      = MAGMA_3
COLOR_THR_MARGINAL = MAGMA_4

# ── SNR quality thresholds ─────────────────────────────────────────────────────
SNR_LOW_BAD   = 1.5
SNR_LOW_OK    = 2.5
SNR_HIGH_GOOD = 4.0

# ── ISI violation thresholds (%) ──────────────────────────────────────────────
ISI_GOOD_PCT   = 0.1   # < 1 %  → GREEN
ISI_MARGIN_PCT = 5.0   # 1–5 %  → YELLOW, > 5 % → RED

# ── Unit acceptance thresholds (ISI + SNR + template correlation) ─────────────
TEMPLATE_CORR_ACCEPT_MIN = 0.5   # template correlation must exceed this
ACCEPTED_COLOR = MAGMA_2         # color used for accepted units in stacked plots
REJECTED_COLOR = REJECT_COLOR    # gray used for rejected units in stacked plots

# ── Spike duration onset/offset threshold ──────────────────────────────────────
# Onset (depolarization begin) / offset (hyperpolarization end) are taken as the
# baseline threshold crossings bracketing the trough/peak, where the threshold is
# this many MAD-based baseline sigmas away from the baseline mean.
SPIKE_DURATION_THRESHOLD_K = 2.0

# ── Multi-trigger (bursting) detection → manual review GUI ────────────────────
MULTI_TRIGGER_OFFSET_SAMPLES = 4     # secondary peaks searched from main peak + this many samples
MULTI_TRIGGER_AMP_FRAC       = 0.5   # secondary peak must exceed this fraction of the spike's main peak
MULTI_TRIGGER_NOISE_K        = 3.0   # ...and this many baseline MAD-sigmas (suppresses noise wiggles)
MULTI_TRIGGER_MIN_PEAKS      = 10    # unit flagged for review if total secondary peaks > this
REVIEW_MAX_SPIKES_SHOWN      = 300   # per category (clean / multi-trigger) drawn in the GUI

# ── openpyxl fill colours ─────────────────────────────────────────────────────
FILL_RED    = PatternFill(fill_type='solid', fgColor='FFFF9999')
FILL_YELLOW = PatternFill(fill_type='solid', fgColor='FFFFFF99')
FILL_GREEN  = PatternFill(fill_type='solid', fgColor='FF99FF99')
FILL_PURPLE = PatternFill(fill_type='solid', fgColor='FFD9B3FF')
FILL_HEADER = PatternFill(fill_type='solid', fgColor='FF2E4057')

FONT_HEADER = Font(bold=True, color='FFFFFFFF', name='Calibri', size=10)
FONT_BODY   = Font(name='Calibri', size=10)
ALIGN_CTR   = Alignment(horizontal='center', vertical='center')
ALIGN_LEFT  = Alignment(horizontal='left',   vertical='center')

THIN_BORDER = Border(
    left=Side(style='thin'),  right=Side(style='thin'),
    top=Side(style='thin'),   bottom=Side(style='thin'),
)

# ── Neuralynx dtype constants ─────────────────────────────────────────────────
NTT_HEADER_BYTES   = 16 * 1024
NCS_HEADER_BYTES   = 16 * 1024
DEFAULT_ADBITVOLTS = 0.000000195

NTT_DTYPE = np.dtype([
    ('timestamp',   '<u8'),
    ('sc_number',   '<u4'),
    ('cell_number', '<u4'),
    ('params',      '<u4', (8,)),
    ('waveforms',   '<i2', (32, 4)),
])

NCS_RECORD_DTYPE = np.dtype([
    ('timestamp',   '<u8'),
    ('channel_nr',  '<u4'),
    ('sample_freq', '<u4'),
    ('n_valid',     '<u4'),
    ('samples',     '<i2', (512,)),
])

# ── All metric keys (used for consistent NaN rows on error) ───────────────────
ALL_METRIC_KEYS = [
    'n_spikes',
    'peak_to_peak_ch0_uV', 'peak_to_peak_ch1_uV',
    'peak_to_peak_ch2_uV', 'peak_to_peak_ch3_uV',
    'best_channel', 'a_peak_uV', 'sigma_noise_uV', 'snr', 'snr_db',
    'isi_violation_pct', 'template_correlation',
    'pt_best_channel', 'peak_time_ms', 'peak_amp_uV',
    'trough_time_ms', 'trough_amp_uV', 'pt_time_ms', 'pt_ratio',
    'spike_start_time_ms', 'spike_end_time_ms', 'spike_duration_ms',
    'csi', 'csi_modified',
    'n_multi_trigger_peaks',
]

COLUMN_ORDER = ['session', 'unit'] + ALL_METRIC_KEYS + ['multi_trigger', 'manual_review', 'accepted']


# ── Helpers: file I/O ──────────────────────────────────────────────────────────

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


def load_ntt_waveforms(ntt_path: str, exclude_cluster0: bool = True):
    """
    Load waveforms/timestamps from an .ntt file.

    exclude_cluster0 : if True (default), spikes with cell_number == 0
        (unclustered / noise cluster) are dropped, so metrics are computed
        only on the sorted main cluster.
    """
    adbitvolts = _parse_adbitvolts(ntt_path)
    uv_scale   = adbitvolts * 1e6
    spike_data = np.memmap(ntt_path, dtype=NTT_DTYPE, mode='r',
                           offset=NTT_HEADER_BYTES)
    cell_number = spike_data['cell_number']  # type: ignore[index]
    if exclude_cluster0:
        keep = cell_number != 0
        spike_data = spike_data[keep]
    waveforms  = spike_data['waveforms'].astype(np.float64) * uv_scale  # type: ignore[operator]
    timestamps = spike_data['timestamp'].astype(np.float64)
    return waveforms, timestamps, adbitvolts


# ── SNR: MAD-based noise estimator ───────────────────────────────────────────

def compute_snr_mad(waveforms: np.ndarray) -> dict:
    """
    SNR for tetrode (.ntt) data using the mean-waveform peak amplitude and a
    MAD-based noise estimate from the pre-spike baseline.

    waveforms : (n_spikes, n_samples, n_channels)
    Returns dict with best_channel, a_peak_uV, sigma_noise_uV, snr, snr_db.
    """
    # Transpose to (n_channels, n_samples, n_spikes) for indexing clarity
    wf = np.transpose(waveforms, (2, 1, 0))          # (4, 32, n_spikes)

    mean_wf = wf.mean(axis=2)                         # (4, 32)
    max_idx  = np.unravel_index(np.argmax(np.abs(mean_wf)), mean_wf.shape)
    best_ch  = int(max_idx[0])

    a_peak = float(np.max(np.abs(mean_wf[best_ch, :])))

    baseline = wf[best_ch, :5, :].flatten()           # first 5 samples as noise proxy
    mad = np.median(np.abs(baseline - np.median(baseline)))
    sigma_noise = float(mad / 0.6745)

    if sigma_noise > 0:
        snr    = a_peak / sigma_noise
        snr_db = 20.0 * np.log10(snr) if snr > 0 else float('nan')
    else:
        snr = snr_db = float('nan')

    return {
        'best_channel':   best_ch,
        'a_peak_uV':      round(a_peak,      4),
        'sigma_noise_uV': round(sigma_noise, 4),
        'snr':            round(snr,    4) if not np.isnan(snr)    else float('nan'),
        'snr_db':         round(snr_db, 4) if not np.isnan(snr_db) else float('nan'),
    }


# ── Quality label helpers ─────────────────────────────────────────────────────

def snr_quality(snr_val) -> tuple:
    if snr_val is None or (isinstance(snr_val, float) and np.isnan(snr_val)):
        return ('N/A', PatternFill())
    if snr_val < SNR_LOW_BAD:
        return ('Too low', FILL_RED)
    if snr_val < SNR_LOW_OK:
        return ('Marginal', FILL_YELLOW)
    if snr_val <= SNR_HIGH_GOOD:
        return ('Good', FILL_GREEN)
    return ('Very high', FILL_PURPLE)


def isi_quality(pct) -> tuple:
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return ('N/A', PatternFill())
    if pct < ISI_GOOD_PCT:
        return ('Good', FILL_GREEN)
    if pct < ISI_MARGIN_PCT:
        return ('Marginal', FILL_YELLOW)
    return ('High', FILL_RED)


def is_unit_accepted(isi_pct, snr_val, template_corr) -> bool:
    """A unit is accepted only if ISI violation < 1 %, SNR > 2.5, and
    template correlation > 0.5 all hold (NaN/None in any metric → rejected)."""
    for v in (isi_pct, snr_val, template_corr):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return False
    return (isi_pct < ISI_GOOD_PCT) and (snr_val > SNR_LOW_OK) and (template_corr > TEMPLATE_CORR_ACCEPT_MIN)


# ── Individual quality-metric functions ───────────────────────────────────────

def compute_isi_violations(spike_ts_us: np.ndarray, ref_period_us: float = 1000.0) -> float:
    """Return % of ISIs that violate the refractory period (< ref_period_us)."""
    if len(spike_ts_us) < 2:
        return float('nan')
    isis = np.diff(np.sort(spike_ts_us))
    return float(np.sum(isis < ref_period_us) / len(isis) * 100.0)


def compute_template_correlation(waveforms: np.ndarray) -> float:
    """Mean Pearson r between each spike waveform and the mean template, averaged across channels."""
    n_spikes = waveforms.shape[0]
    if n_spikes < 2:
        return float('nan')
    template = waveforms.mean(axis=0)               # (32, 4)
    t = template - template.mean(axis=0)            # demean along time
    w = waveforms - waveforms.mean(axis=1, keepdims=True)  # (n, 32, 4)

    numer    = np.sum(t[np.newaxis] * w, axis=1)   # (n, 4)
    t_sq_sum = np.sum(t ** 2, axis=0)              # (4,)
    w_sq_sum = np.sum(w ** 2, axis=1)              # (n, 4)
    denom    = np.sqrt(t_sq_sum[np.newaxis] * w_sq_sum)

    with np.errstate(invalid='ignore', divide='ignore'):
        corr = np.where(denom > 0, numer / denom, np.nan)

    return float(np.nanmean(corr))


def compute_spike_duration(wf: np.ndarray, peak_idx: int, trough_idx: int,
                            k: float = SPIKE_DURATION_THRESHOLD_K) -> dict:
    """
    Spike duration: time from the onset of depolarization to the end of the
    after-hyperpolarization, on a single channel's mean waveform.

    The baseline (first 5 samples, same convention as the SNR estimator) gives
    a MAD-based sigma. Onset is the last baseline-level sample before the
    earlier of the peak/trough (depolarization begins where the trace first
    departs the baseline band); offset is the first baseline-level sample
    after the later of the peak/trough (hyperpolarization ends where the
    trace returns to the baseline band).

    wf : (32,) mean waveform on the channel used for peak/trough detection.
    """
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

    duration_ms = (end_idx - start_idx) * SAMPLE_INTERVAL_MS

    return {
        'start_idx':          start_idx,
        'start_time_ms':      round(start_idx * SAMPLE_INTERVAL_MS, 5),
        'end_idx':             end_idx,
        'end_time_ms':         round(end_idx * SAMPLE_INTERVAL_MS, 5),
        'spike_duration_ms':   round(duration_ms, 5),
    }


def compute_pt(waveforms: np.ndarray) -> dict:
    """
    waveforms : (n_spikes, 32, 4) microvolts

    Selects the channel whose MEAN waveform reaches the highest peak
    (max amplitude, not peak-to-peak spread), then measures peak/trough
    time, amplitude ratio, and spike duration on that channel's mean
    waveform.
    """
    mean_wf = waveforms.mean(axis=0)          # (32, 4)

    peak_per_channel = mean_wf.max(axis=0)    # (4,)
    best_ch = int(np.argmax(peak_per_channel))
    wf = mean_wf[:, best_ch]                  # (32,)

    peak_idx   = int(np.argmax(wf))
    trough_idx = int(np.argmin(wf))
    peak_val   = float(wf[peak_idx])
    trough_val = float(wf[trough_idx])

    pt_time_ms = abs(peak_idx - trough_idx) * SAMPLE_INTERVAL_MS
    pt_ratio   = abs(peak_val) / abs(trough_val) if trough_val != 0 else float('nan')

    duration_result = compute_spike_duration(wf, peak_idx, trough_idx)

    return {
        'best_channel':   best_ch,
        'peak_idx':       peak_idx,
        'peak_time_ms':   round(peak_idx * SAMPLE_INTERVAL_MS, 5),
        'peak_amp_uV':    round(peak_val, 4),
        'trough_idx':     trough_idx,
        'trough_time_ms': round(trough_idx * SAMPLE_INTERVAL_MS, 5),
        'trough_amp_uV':  round(trough_val, 4),
        'pt_time_ms':     round(pt_time_ms, 5),
        'pt_ratio':       round(pt_ratio, 4) if not np.isnan(pt_ratio) else float('nan'),
        'spike_start_idx':      duration_result['start_idx'],
        'spike_start_time_ms':  duration_result['start_time_ms'],
        'spike_end_idx':        duration_result['end_idx'],
        'spike_end_time_ms':    duration_result['end_time_ms'],
        'spike_duration_ms':    duration_result['spike_duration_ms'],
        'mean_waveform':  wf,   # used for plotting only, dropped before Excel export
    }


# ── Per-file waveform figure ─────────────────────────────────────────────────

def plot_waveform(wf: np.ndarray, result: dict, out_path: str, title: str):
    time_axis = np.arange(N_SAMPLES) * SAMPLE_INTERVAL_MS

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
    _style_axes(ax)

    ax.plot(time_axis, wf, color=COLOR_WAVEFORM, linewidth=1.8, zorder=3)
    ax.scatter([result['peak_time_ms']], [result['peak_amp_uV']],
               facecolor=COLOR_PEAK, edgecolor=SURFACE, linewidths=1.0, s=50, zorder=5,
               label=f"Peak  {result['peak_amp_uV']:.1f} µV")
    ax.scatter([result['trough_time_ms']], [result['trough_amp_uV']],
               facecolor=COLOR_TROUGH, edgecolor=SURFACE, linewidths=1.0, s=50, zorder=5,
               label=f"Trough  {result['trough_amp_uV']:.1f} µV")

    start_t = result['spike_start_time_ms']
    end_t   = result['spike_end_time_ms']
    ax.scatter([start_t, end_t], [wf[result['spike_start_idx']], wf[result['spike_end_idx']]],
               color=COLOR_DURATION, marker='|', s=180, linewidths=2.0, zorder=6,
               label='Spike start/end')
    ax.axvspan(start_t, end_t, color=COLOR_DURATION, alpha=0.18, linewidth=0, zorder=1)

    ax.set_xlabel('Time (ms)', color=INK_PRIMARY)
    ax.set_ylabel('Amplitude (µV)', color=INK_PRIMARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc='left')
    _legend_outside(ax)

    note = (f"Ch {result['best_channel']}\n"
            f"P-T time = {result['pt_time_ms']:.3f} ms\n"
            f"P/T ratio = {result['pt_ratio']:.3f}\n"
            f"Spike duration = {result['spike_duration_ms']:.3f} ms")
    ax.text(1.02, 0.0, note, transform=ax.transAxes, fontsize=8, color=INK_SECONDARY,
            ha='left', va='bottom')

    # PNG only: one figure per unit, embedded in the Excel workbook
    _save(fig, os.path.splitext(out_path)[0], exts=('png',))



def compute_csi(waveforms: np.ndarray, spike_ts_us: np.ndarray,
                min_isi_us: float = 3000.0, max_isi_us: float = 20000.0) -> float:
    """
    Complex Spike Index: % of ALL consecutive ISIs that fall in [3, 20] ms
    AND whose second spike has a smaller amplitude than the first.
    (Bhatt et al. 2020 / Ranck 1973)
    """
    if len(spike_ts_us) < 2:
        return float('nan')
    sort_idx  = np.argsort(spike_ts_us)
    ts_sorted = spike_ts_us[sort_idx]
    wf_sorted = waveforms[sort_idx]

    isis     = np.diff(ts_sorted)
    in_range = (isis >= min_isi_us) & (isis <= max_isi_us)

    if not np.any(in_range):
        return 0.0

    mean_pp = (wf_sorted.max(axis=1) - wf_sorted.min(axis=1)).mean(axis=1)  # (n_spikes,)
    first_pp  = mean_pp[:-1][in_range]
    second_pp = mean_pp[1:][in_range]

    n_complex = int(np.sum(second_pp < first_pp))
    return round(float(n_complex / len(isis) * 100.0), 4)


def compute_csi_modified(spike_ts_us: np.ndarray,
                         min_isi_us: float = 3000.0, max_isi_us: float = 20000.0) -> float:
    """
    CSI modified: identical to compute_csi but without the decreasing-amplitude
    criterion, i.e. % of ALL consecutive ISIs that fall in [3, 20] ms.
    """
    if len(spike_ts_us) < 2:
        return float('nan')
    isis     = np.diff(np.sort(spike_ts_us))
    in_range = (isis >= min_isi_us) & (isis <= max_isi_us)
    return round(float(np.sum(in_range) / len(isis) * 100.0), 4)


def detect_multi_triggers(waveforms: np.ndarray, best_ch: int, peak_idx: int,
                          offset: int = MULTI_TRIGGER_OFFSET_SAMPLES,
                          amp_frac: float = MULTI_TRIGGER_AMP_FRAC,
                          noise_k: float = MULTI_TRIGGER_NOISE_K) -> np.ndarray:
    """
    Count secondary peaks per spike on `best_ch`, searching from
    `peak_idx + offset` to the end of the 32-sample snippet. A sample is a
    secondary peak if it is a local maximum (the last sample counts if still
    rising, i.e. a truncated second spike) and exceeds both amp_frac × the
    spike's own main-peak amplitude and noise_k × the baseline MAD-sigma.

    Returns (n_spikes,) int array of secondary-peak counts.
    """
    wf = waveforms[:, :, best_ch]                       # (n, 32)
    n_spk, n_smp = wf.shape
    start = peak_idx + max(int(offset), 1)
    if n_spk == 0 or start >= n_smp:
        return np.zeros(n_spk, dtype=int)

    baseline = wf[:, :5].flatten()
    sigma    = float(np.median(np.abs(baseline - np.median(baseline))) / 0.6745)

    main_amp = wf[:, peak_idx]                          # (n,)
    prev = wf[:, start - 1:n_smp - 1]
    cur  = wf[:, start:]
    nxt  = np.concatenate([wf[:, start + 1:], np.full((n_spk, 1), -np.inf)], axis=1)

    thr     = np.maximum(amp_frac * main_amp, noise_k * sigma)
    is_peak = (cur > prev) & (cur >= nxt) & (cur > thr[:, None]) & (main_amp > 0)[:, None]
    return is_peak.sum(axis=1)


def write_cleaned_ntt(src_path: str, dst_path: str, best_ch: int, peak_idx: int) -> int:
    """
    Copy `src_path` to `dst_path`, then set cell_number = 0 in the copy for
    every non-cluster-0 spike that has ≥ 1 secondary trigger (the same spikes
    drawn in red in the review GUI). Returns the number of spikes moved.
    """
    shutil.copy2(src_path, dst_path)
    uv_scale = _parse_adbitvolts(src_path) * 1e6
    data = np.memmap(dst_path, dtype=NTT_DTYPE, mode='r+', offset=NTT_HEADER_BYTES)
    kept = np.flatnonzero(data['cell_number'] != 0)  # type: ignore[index]
    waveforms = data['waveforms'][kept].astype(np.float64) * uv_scale  # type: ignore[index]
    moved = kept[detect_multi_triggers(waveforms, best_ch, peak_idx) > 0]
    data['cell_number'][moved] = 0  # type: ignore[index]
    data.flush()
    del data
    return int(len(moved))


# ── Manual review GUI for multi-trigger units ────────────────────────────────

def review_multi_trigger_units(candidates: list) -> dict:
    """
    Tk window listing every flagged unit. For the selected unit, all four
    channels are drawn: clean spikes in light gray, spikes with secondary
    triggers in red, mean waveform in black; the dashed line on the best
    channel marks where the secondary-peak search starts.

    Keys: A = accept, R = reject, M = accept but move the red (multi-trigger)
    spikes to cluster 0 (all three advance), ←/→ = previous/next.
    Units left undecided keep their automatic decision.

    Returns {candidate_index: 'accept' / 'reject' / 'clean'}.
    """
    try:
        import tkinter as tk
        from matplotlib.figure import Figure # type: ignore
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # type: ignore
    except ImportError as exc:
        print(f'Manual review GUI unavailable ({exc}); keeping automatic decisions.')
        return {}

    decisions = {}
    state     = {'i': 0}
    rng       = np.random.default_rng(0)
    time_axis = np.arange(N_SAMPLES) * SAMPLE_INTERVAL_MS

    root = tk.Tk()
    root.title('Multi-trigger unit review  —  A: accept   R: reject   '
               'M: accept, multi-trigger spikes → cluster 0   ←/→: navigate')
    root.geometry('1500x900')

    left = tk.Frame(root)
    left.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)
    listbox = tk.Listbox(left, width=55, font=('Consolas', 9), exportselection=False)
    sb = tk.Scrollbar(left, orient=tk.VERTICAL, command=listbox.yview)
    listbox.config(yscrollcommand=sb.set)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.pack(side=tk.LEFT, fill=tk.Y)

    right = tk.Frame(root)
    right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    fig    = Figure(figsize=(11, 7.5), facecolor=SURFACE)
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    btns = tk.Frame(right)
    btns.pack(fill=tk.X, pady=4)
    status_var = tk.StringVar()

    DECISION_TAG    = {'accept': 'ACCEPT', 'reject': 'REJECT', 'clean': 'ACC-MT0'}
    DECISION_COLOUR = {'accept': 'forest green', 'reject': 'red3', 'clean': 'dark orange3'}
    DECISION_TEXT   = {'accept': 'ACCEPTED', 'reject': 'REJECTED',
                       'clean': 'ACCEPTED, multi-trigger spikes → cluster 0'}

    def list_label(k):
        c = candidates[k]
        if k in decisions:
            tag = DECISION_TAG[decisions[k]]
        else:
            tag = 'auto:acc' if c['auto_accepted'] else 'auto:rej'
        return f'[{tag:>8s}] {c["session"]} | {c["unit"]}'

    def refresh_item(k):
        listbox.delete(k)
        listbox.insert(k, list_label(k))
        listbox.itemconfig(k, fg=DECISION_COLOUR[decisions[k]] if k in decisions else 'black')

    def draw(k):
        c = candidates[k]
        fig.clear()
        try:
            waveforms, _, _ = load_ntt_waveforms(c['ntt_path'])
        except Exception as exc:
            fig.text(0.5, 0.5, f'Could not load file:\n{exc}', ha='center', va='center')
            canvas.draw()
            return
        counts  = detect_multi_triggers(waveforms, c['best_ch'], c['peak_idx'])
        idx_mt  = np.flatnonzero(counts > 0)
        idx_ok  = np.flatnonzero(counts == 0)
        if len(idx_mt) > REVIEW_MAX_SPIKES_SHOWN:
            idx_mt = rng.choice(idx_mt, REVIEW_MAX_SPIKES_SHOWN, replace=False)
        if len(idx_ok) > REVIEW_MAX_SPIKES_SHOWN:
            idx_ok = rng.choice(idx_ok, REVIEW_MAX_SPIKES_SHOWN, replace=False)

        axes = fig.subplots(2, 2, sharex=True, sharey=True)
        for ch, ax in enumerate(axes.flat):
            _style_axes(ax)
            if len(idx_ok):
                ax.plot(time_axis, waveforms[idx_ok, :, ch].T, color=INK_MUTED, lw=0.5, alpha=0.3)
            if len(idx_mt):
                ax.plot(time_axis, waveforms[idx_mt, :, ch].T, color=MAGMA_3, lw=0.6, alpha=0.35)
            ax.plot(time_axis, waveforms[:, :, ch].mean(axis=0), color=INK_PRIMARY, lw=2.0)
            is_best = ch == c['best_ch']
            if is_best:
                ax.axvline((c['peak_idx'] + MULTI_TRIGGER_OFFSET_SAMPLES) * SAMPLE_INTERVAL_MS,
                           color=MAGMA_1, lw=1.2, ls='--')
            ax.set_title(f'Ch {ch}' + ('  (best)' if is_best else ''), color=INK_PRIMARY,
                         fontsize=10, loc='left', fontweight='bold' if is_best else 'normal')
            if ch >= 2:
                ax.set_xlabel('Time (ms)', color=INK_PRIMARY)
            if ch % 2 == 0:
                ax.set_ylabel('Amplitude (µV)', color=INK_PRIMARY)

        def fmt(v):
            return 'nan' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.3f}'
        fig.suptitle(
            f'{c["session"]} | {c["unit"]}   ({k + 1}/{len(candidates)})\n'
            f'n spikes = {waveforms.shape[0]}   |   secondary peaks = {int(counts.sum())} '
            f'in {int((counts > 0).sum())} spikes (red)   |   '
            f'SNR = {fmt(c["snr"])}   ISI viol = {fmt(c["isi"])} %   TmplCorr = {fmt(c["tmpl_corr"])}   |   '
            f'auto: {"ACCEPTED" if c["auto_accepted"] else "REJECTED"}',
            fontsize=10, color=INK_PRIMARY)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        canvas.draw()

        n_done = len(decisions)
        current = DECISION_TEXT[decisions[k]] if k in decisions else 'undecided'
        status_var.set(f'This unit: {current}     |     reviewed {n_done}/{len(candidates)}')

    def select(k):
        k = max(0, min(k, len(candidates) - 1))
        state['i'] = k
        listbox.selection_clear(0, tk.END)
        listbox.selection_set(k)
        listbox.see(k)
        draw(k)

    def decide(value):
        k = state['i']
        decisions[k] = value
        refresh_item(k)
        if k < len(candidates) - 1:
            select(k + 1)
        else:
            select(k)

    def on_list_select(_event):
        sel = listbox.curselection()
        if sel and sel[0] != state['i']:
            select(sel[0])

    tk.Button(btns, text='◀ Prev', width=10, command=lambda: select(state['i'] - 1)).pack(side=tk.LEFT, padx=4)
    tk.Button(btns, text='Accept (A)', width=14, bg='#99FF99',
              command=lambda: decide('accept')).pack(side=tk.LEFT, padx=4)
    tk.Button(btns, text='Accept, red → cluster 0 (M)', width=26, bg='#FFCC80',
              command=lambda: decide('clean')).pack(side=tk.LEFT, padx=4)
    tk.Button(btns, text='Reject (R)', width=14, bg='#FF9999',
              command=lambda: decide('reject')).pack(side=tk.LEFT, padx=4)
    tk.Button(btns, text='Next ▶', width=10, command=lambda: select(state['i'] + 1)).pack(side=tk.LEFT, padx=4)
    tk.Label(btns, textvariable=status_var, font=('Calibri', 11)).pack(side=tk.LEFT, padx=16)
    tk.Button(btns, text='Done — apply decisions', width=24,
              command=root.destroy).pack(side=tk.RIGHT, padx=4)

    for k in range(len(candidates)):
        listbox.insert(tk.END, list_label(k))
    listbox.bind('<<ListboxSelect>>', on_list_select)
    root.bind('<KeyPress-a>', lambda _e: decide('accept'))
    root.bind('<KeyPress-A>', lambda _e: decide('accept'))
    root.bind('<KeyPress-r>', lambda _e: decide('reject'))
    root.bind('<KeyPress-R>', lambda _e: decide('reject'))
    root.bind('<KeyPress-m>', lambda _e: decide('clean'))
    root.bind('<KeyPress-M>', lambda _e: decide('clean'))
    root.bind('<Left>',  lambda _e: select(state['i'] - 1))
    root.bind('<Right>', lambda _e: select(state['i'] + 1))
    root.protocol('WM_DELETE_WINDOW', root.destroy)

    select(0)
    root.focus_force()
    root.mainloop()
    return decisions


# ── Saved (manually curated) multi-trigger labels ────────────────────────────

def _label_key(session, unit) -> tuple:
    return (os.path.normpath(str(session)).lower(), str(unit).strip().lower())


def _parse_mt_label(value):
    """Map a manual_review entry to 'accept' / 'reject' / 'clean' (None = no decision)."""
    s = str(value).strip().lower()
    if not s or s in ('nan', 'none', 'not reviewed'):
        return None
    if s.startswith('rej'):
        return 'reject'
    if s == 'clean' or 'cl 0' in s or 'cluster 0' in s or 'moved' in s or 'mt0' in s:
        return 'clean'
    if s.startswith('acc'):
        return 'accept'
    return None


def load_saved_mt_labels(path: str) -> dict:
    """
    Read curated multi-trigger labels from a .csv or .xlsx (first sheet) with
    'session', 'unit' and 'manual_review' columns (as written by this script).
    Returns {(session, unit): 'accept' / 'reject' / 'clean'}.
    """
    if not os.path.isfile(path):
        print(f'Saved multi-trigger labels not found: {path}\n'
              f'  -> all multi-trigger units will be reviewed in the GUI.')
        return {}
    if path.lower().endswith('.csv'):
        lab = pd.read_csv(path)
    else:
        lab = pd.read_excel(path, sheet_name=0, engine='openpyxl')
    if 'manual_review' not in lab.columns:
        print(f'Saved labels file has no "manual_review" column: {path}\n'
              f'  -> all multi-trigger units will be reviewed in the GUI.')
        return {}
    session_col = 'session' if 'session' in lab.columns else lab.columns[0]
    unit_col    = 'unit' if 'unit' in lab.columns else lab.columns[1]
    labels = {}
    for session, unit, value in zip(lab[session_col], lab[unit_col], lab['manual_review']):
        decision = _parse_mt_label(value)
        if decision is not None:
            labels[_label_key(session, unit)] = decision
    print(f'Loaded {len(labels)} saved multi-trigger label(s) from {path}')
    return labels


# ── Master metric computation ─────────────────────────────────────────────────

def compute_metrics(ntt_path: str) -> dict:
    waveforms, spike_ts, _ = load_ntt_waveforms(ntt_path)
    n_spikes = waveforms.shape[0]

    nan_row = {k: float('nan') for k in ALL_METRIC_KEYS}
    nan_row['n_spikes'] = 0
    if n_spikes == 0:
        return nan_row

    # ── Per-channel peak-to-peak (mean across spikes) ─────────────────────────
    pp_per_spike        = waveforms.max(axis=1) - waveforms.min(axis=1)  # (n, 4)
    mean_pp_per_channel = pp_per_spike.mean(axis=0)                      # (4,)

    # ── SNR via MAD-based noise estimator ────────────────────────────────────
    snr_metrics = compute_snr_mad(waveforms)

    # ── Waveform shape metrics (peak-to-trough) ───────────────────────────────
    pt_result = compute_pt(waveforms)

    return {
        'n_spikes':             n_spikes,
        'peak_to_peak_ch0_uV':  round(float(mean_pp_per_channel[0]), 4),
        'peak_to_peak_ch1_uV':  round(float(mean_pp_per_channel[1]), 4),
        'peak_to_peak_ch2_uV':  round(float(mean_pp_per_channel[2]), 4),
        'peak_to_peak_ch3_uV':  round(float(mean_pp_per_channel[3]), 4),
        **snr_metrics,
        'isi_violation_pct':    round(compute_isi_violations(spike_ts), 4),
        'template_correlation': round(compute_template_correlation(waveforms), 4),
        'pt_best_channel':      pt_result['best_channel'],
        'peak_time_ms':         pt_result['peak_time_ms'],
        'peak_amp_uV':          pt_result['peak_amp_uV'],
        'trough_time_ms':       pt_result['trough_time_ms'],
        'trough_amp_uV':        pt_result['trough_amp_uV'],
        'pt_time_ms':           pt_result['pt_time_ms'],
        'pt_ratio':             pt_result['pt_ratio'],
        'spike_start_time_ms':  pt_result['spike_start_time_ms'],
        'spike_end_time_ms':    pt_result['spike_end_time_ms'],
        'spike_duration_ms':    pt_result['spike_duration_ms'],
        'csi':                  compute_csi(waveforms, spike_ts),
        'csi_modified':         compute_csi_modified(spike_ts),
        'n_multi_trigger_peaks': int(detect_multi_triggers(waveforms, pt_result['best_channel'],
                                                           pt_result['peak_idx']).sum()),
        '_pt_plot':            pt_result,   # not an Excel column; used for per-unit waveform plots
    }


# ── Batch scan ────────────────────────────────────────────────────────────────

WAVEFORM_PLOT_DIR = os.path.join(os.path.dirname(OUTPUT_EXCEL), 'waveform_plots')
os.makedirs(WAVEFORM_PLOT_DIR, exist_ok=True)
MT_LABELS_CSV = os.path.splitext(OUTPUT_EXCEL)[0] + '_MT_labels.csv'

# Read before the scan, so a labels file that is also OUTPUT_EXCEL is read before it is overwritten
saved_mt_labels = load_saved_mt_labels(SAVED_MT_LABELS_FILE) if USE_SAVED_MT_LABELS else {}

records = []
waveform_plot_entries = []   # [(label, png_path), ...] for Excel embedding
n_units_copied = 0
n_support_files_copied = 0
review_candidates = []   # multi-trigger units held back for the manual review GUI
for dirpath, _, filenames in os.walk(ROOT_FOLDER):
    ntt_files = sorted(f for f in filenames if f.lower().endswith('.ntt'))
    if not ntt_files:
        continue

    session_name = os.path.relpath(dirpath, ROOT_FOLDER)
    dest_dir     = os.path.join(ACCEPTED_ROOT_FOLDER, session_name)

    # Copy every .csv and .ncs file in this session folder, regardless of
    # any individual unit's acceptance (they describe the recording session,
    # not a single cluster).
    support_files = sorted(f for f in filenames if f.lower().endswith(('.csv', '.ncs')))
    if SAVE_ACCEPTED_COPY and support_files:
        os.makedirs(dest_dir, exist_ok=True)
        for support_file in support_files:
            shutil.copy2(os.path.join(dirpath, support_file), os.path.join(dest_dir, support_file))
            n_support_files_copied += 1

    for ntt_file in ntt_files:
        ntt_path = os.path.join(dirpath, ntt_file)
        print(f'Processing: {session_name}  |  {ntt_file}')
        multi_trigger = False
        try:
            metrics  = compute_metrics(ntt_path)
            pt_plot  = metrics.pop('_pt_plot', None)
            accepted = is_unit_accepted(metrics.get('isi_violation_pct'),
                                        metrics.get('snr'),
                                        metrics.get('template_correlation'))
            multi_trigger = (pt_plot is not None and
                             metrics.get('n_multi_trigger_peaks', 0) > MULTI_TRIGGER_MIN_PEAKS)
            if multi_trigger and pt_plot is not None:
                # Copying is deferred until after manual review
                print(f'  -> multi-trigger: {metrics["n_multi_trigger_peaks"]} secondary peaks, '
                      f'queued for manual review')
                review_candidates.append({
                    'record_idx':    len(records),
                    'session':       session_name,
                    'unit':          ntt_file,
                    'ntt_path':      ntt_path,
                    'dest_dir':      dest_dir,
                    'best_ch':       pt_plot['best_channel'],
                    'peak_idx':      pt_plot['peak_idx'],
                    'auto_accepted': accepted,
                    'snr':           metrics.get('snr'),
                    'isi':           metrics.get('isi_violation_pct'),
                    'tmpl_corr':     metrics.get('template_correlation'),
                })
            elif accepted and SAVE_ACCEPTED_COPY:
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(ntt_path, os.path.join(dest_dir, ntt_file))
                n_units_copied += 1
            if pt_plot is not None:
                unit_label = f'{session_name}_{os.path.splitext(ntt_file)[0]}'.replace(os.sep, '_')
                png_path   = os.path.join(WAVEFORM_PLOT_DIR, f'{unit_label}_waveform.png')
                plot_waveform(pt_plot['mean_waveform'], pt_plot, png_path,
                             title=f'{session_name} | {ntt_file}')
                waveform_plot_entries.append((f'{session_name} | {ntt_file}', png_path))
        except Exception as exc:
            print(f'  ERROR: {exc}')
            metrics  = {k: None for k in ALL_METRIC_KEYS}
            accepted = False
        metrics['session']  = session_name # type: ignore
        metrics['unit']     = ntt_file # type: ignore
        metrics['multi_trigger'] = multi_trigger # type: ignore
        metrics['manual_review'] = '' # type: ignore
        metrics['accepted'] = accepted # type: ignore
        records.append(metrics)

# ── Manual review of multi-trigger (bursting) units ───────────────────────────
# The manual decision overrides the automatic one, so manually accepted units
# are copied to ACCEPTED_ROOT_FOLDER and coloured as accepted in all histograms.

n_manual_accepted = n_manual_rejected = n_manual_cleaned = 0
n_from_saved = 0
if review_candidates:
    decisions = {}
    for k, cand in enumerate(review_candidates):
        saved = saved_mt_labels.get(_label_key(cand['session'], cand['unit']))
        if saved is not None:
            decisions[k] = saved
    n_from_saved = len(decisions)
    if USE_SAVED_MT_LABELS:
        print(f'\n{n_from_saved} of {len(review_candidates)} multi-trigger unit(s) '
              f'labelled from saved file')
    to_review = [k for k in range(len(review_candidates)) if k not in decisions]
    if to_review:
        print(f'\n{len(to_review)} unit(s) with > {MULTI_TRIGGER_MIN_PEAKS} secondary peaks '
              f'-> opening manual review GUI...')
        gui_decisions = review_multi_trigger_units([review_candidates[k] for k in to_review])
        decisions.update({to_review[j]: d for j, d in gui_decisions.items()})
    for k, cand in enumerate(review_candidates):
        rec = records[cand['record_idx']]
        decision = decisions.get(k)
        if decision is None:
            rec['manual_review'] = 'not reviewed'
        else:
            rec['accepted'] = decision in ('accept', 'clean')
            rec['manual_review'] = {'accept': 'accepted', 'reject': 'rejected',
                                    'clean': 'accepted, MT spikes -> cl 0'}[decision]
            if decision == 'accept':
                n_manual_accepted += 1
            elif decision == 'clean':
                n_manual_cleaned += 1
            else:
                n_manual_rejected += 1
        if rec['accepted'] and SAVE_ACCEPTED_COPY:
            os.makedirs(cand['dest_dir'], exist_ok=True)
            dst_path = os.path.join(cand['dest_dir'], cand['unit'])
            if decision == 'clean':
                n_moved = write_cleaned_ntt(cand['ntt_path'], dst_path,
                                            cand['best_ch'], cand['peak_idx'])
                rec['manual_review'] = f'accepted, {n_moved} MT spikes -> cl 0'
                print(f'  {cand["session"]} | {cand["unit"]}: {n_moved} multi-trigger '
                      f'spike(s) moved to cluster 0 in the accepted copy')
            else:
                shutil.copy2(cand['ntt_path'], dst_path)
            n_units_copied += 1
    print(f'Manual review: {n_manual_accepted} accepted, '
          f'{n_manual_cleaned} accepted with multi-trigger spikes -> cluster 0, '
          f'{n_manual_rejected} rejected, '
          f'{len(review_candidates) - len(decisions)} not reviewed (automatic decision kept); '
          f'{n_from_saved} taken from saved labels')

    # Curated labels for re-use (point SAVED_MT_LABELS_FILE here on a later run)
    pd.DataFrame([{'session':       cand['session'],
                   'unit':          cand['unit'],
                   'manual_review': records[cand['record_idx']]['manual_review'],
                   'accepted':      records[cand['record_idx']]['accepted']}
                  for cand in review_candidates]).to_csv(MT_LABELS_CSV, index=False)
    print(f'Multi-trigger labels saved: {MT_LABELS_CSV}')

if SAVE_ACCEPTED_COPY:
    print(f'\nAccepted units copied: {n_units_copied}  |  Support files (.csv/.ncs) copied: {n_support_files_copied}')
    print(f'Accepted-unit tree: {ACCEPTED_ROOT_FOLDER}')
else:
    print('\nSAVE_ACCEPTED_COPY = False -> no files copied.')

# ── Build DataFrame ───────────────────────────────────────────────────────────

df = pd.DataFrame(records, columns=COLUMN_ORDER)
df.to_excel(OUTPUT_EXCEL, index=False, engine='openpyxl')

# ── Post-process: colour coding via openpyxl ──────────────────────────────────

wb = load_workbook(OUTPUT_EXCEL)
ws = wb.active
assert ws is not None
ws.title = 'Quality Metrics'

COL_WIDTHS = {
    'session':              28, 'unit':                 18,
    'n_spikes':             10,
    'peak_to_peak_ch0_uV':  18, 'peak_to_peak_ch1_uV':  18,
    'peak_to_peak_ch2_uV':  18, 'peak_to_peak_ch3_uV':  18,
    'best_channel':         14, 'a_peak_uV':            14, 'sigma_noise_uV':       16,
    'snr':                  10, 'snr_db':               10,
    'isi_violation_pct':    18, 'template_correlation':  22,
    'pt_best_channel':      14, 'peak_time_ms':          14, 'peak_amp_uV':          14,
    'trough_time_ms':       14, 'trough_amp_uV':         14, 'pt_time_ms':           14,
    'pt_ratio':             12,
    'spike_start_time_ms':  16, 'spike_end_time_ms':     16, 'spike_duration_ms':    16,
    'csi':                  10, 'csi_modified':         14,
    'n_multi_trigger_peaks': 20, 'multi_trigger':        14, 'manual_review':        30,
}
for col_idx, col_name in enumerate(COLUMN_ORDER, start=1):
    ws.column_dimensions[get_column_letter(col_idx)].width = COL_WIDTHS.get(col_name, 14)

for cell in ws[1]:
    cell.fill = FILL_HEADER; cell.font = FONT_HEADER
    cell.alignment = ALIGN_CTR; cell.border = THIN_BORDER
ws.row_dimensions[1].height = 22
ws.freeze_panes = 'A2'

snr_col     = COLUMN_ORDER.index('snr')               + 1
snr_db_col  = COLUMN_ORDER.index('snr_db')            + 1
isi_col     = COLUMN_ORDER.index('isi_violation_pct') + 1

# Two appended quality-label columns (right after the data)
SNR_QUALITY_COL = len(COLUMN_ORDER) + 1
ISI_QUALITY_COL = len(COLUMN_ORDER) + 2

for row_idx, row in enumerate(ws.iter_rows(min_row=2, max_row=ws.max_row), start=2):
    # Base styling
    for cell in row:
        cell.font      = FONT_BODY
        cell.border    = THIN_BORDER
        cell.alignment = ALIGN_CTR if cell.column > 1 else ALIGN_LEFT # type: ignore

    # SNR colour
    try:
        snr_val = float(ws.cell(row=row_idx, column=snr_col).value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        snr_val = float('nan')
    snr_lbl, snr_fill = snr_quality(snr_val)
    ws.cell(row=row_idx, column=snr_col).fill    = snr_fill
    ws.cell(row=row_idx, column=snr_db_col).fill = snr_fill

    c = ws.cell(row=row_idx, column=SNR_QUALITY_COL, value=snr_lbl)  # type: ignore[arg-type]
    c.fill = snr_fill; c.font = FONT_BODY; c.alignment = ALIGN_CTR; c.border = THIN_BORDER

    # ISI colour
    try:
        isi_val = float(ws.cell(row=row_idx, column=isi_col).value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        isi_val = float('nan')
    isi_lbl, isi_fill = isi_quality(isi_val)
    ws.cell(row=row_idx, column=isi_col).fill = isi_fill

    c = ws.cell(row=row_idx, column=ISI_QUALITY_COL, value=isi_lbl)  # type: ignore[arg-type]
    c.fill = isi_fill; c.font = FONT_BODY; c.alignment = ALIGN_CTR; c.border = THIN_BORDER

# Quality label column headers
for col_idx, label in [(SNR_QUALITY_COL, 'SNR Quality'), (ISI_QUALITY_COL, 'ISI Quality')]:
    c = ws.cell(row=1, column=col_idx, value=label)
    c.fill = FILL_HEADER; c.font = FONT_HEADER; c.alignment = ALIGN_CTR; c.border = THIN_BORDER
    ws.column_dimensions[get_column_letter(col_idx)].width = 14

# ── Legend sheet ──────────────────────────────────────────────────────────────

leg = wb.create_sheet('Legend')
leg.column_dimensions['A'].width = 16
leg.column_dimensions['B'].width = 36
leg.column_dimensions['C'].width = 24

legend_rows = [
    ('Colour',  'Meaning',                                  'Threshold'),
    ('GREEN',   'SNR good / ISI low (< 1 %)',               f'SNR {SNR_LOW_OK}–{SNR_HIGH_GOOD}  |  ISI < {ISI_GOOD_PCT} %'),
    ('YELLOW',  'SNR marginal / ISI borderline (1–5 %)',    f'SNR {SNR_LOW_BAD}–{SNR_LOW_OK}  |  ISI {ISI_GOOD_PCT}–{ISI_MARGIN_PCT} %'),
    ('RED',     'SNR too low / ISI high (> 5 %)',           f'SNR < {SNR_LOW_BAD}  |  ISI > {ISI_MARGIN_PCT} %'),
    ('PURPLE',  'SNR very high – check for artefacts',      f'SNR > {SNR_HIGH_GOOD}'),
]
fills_legend = [FILL_HEADER, FILL_GREEN, FILL_YELLOW, FILL_RED, FILL_PURPLE]
for r_idx, (row_data, fill) in enumerate(zip(legend_rows, fills_legend), start=1):
    for c_idx, val in enumerate(row_data, start=1):
        cell = leg.cell(row=r_idx, column=c_idx, value=val)  # type: ignore[arg-type]
        cell.fill = fill
        cell.font = FONT_HEADER if r_idx == 1 else FONT_BODY
        cell.alignment = ALIGN_CTR; cell.border = THIN_BORDER
    leg.row_dimensions[r_idx].height = 20

wb.save(OUTPUT_EXCEL)
print('Excel data saved.')


# ── Histogram helper ──────────────────────────────────────────────────────────

def make_histogram(values, accepted, title, xlabel,
                   thresholds=(), threshold_colors=(), threshold_labels=(),
                   bar_color=ACCEPTED_COLOR, n_bins=30):
    """
    Stacked histogram: for each bin, rejected units (accepted == False) are
    drawn as a gray segment at the bottom, accepted units in `bar_color` on
    top, so discarded cells are visually separated from the accepted ones
    within the same bin.
    """
    values   = np.asarray(values,   dtype=float)
    accepted = np.asarray([bool(a) if a is not None else False for a in accepted], dtype=bool)
    valid    = ~np.isnan(values)
    vals     = values[valid]
    acc      = accepted[valid] if len(accepted) == len(values) else np.zeros(vals.shape, dtype=bool)

    # Legend + statistics sit outside the axes on the right so they never
    # overlap the bars.
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    _style_axes(ax)

    if len(vals) > 0:
        bin_edges    = np.histogram_bin_edges(vals, bins=n_bins)
        bin_widths   = np.diff(bin_edges)
        bin_centers  = bin_edges[:-1] + bin_widths / 2
        n_rejected, _ = np.histogram(vals[~acc], bins=bin_edges)
        n_accepted, _ = np.histogram(vals[acc],  bins=bin_edges)

        ax.bar(bin_centers, n_rejected, width=bin_widths, color=REJECTED_COLOR,
               edgecolor=SURFACE, linewidth=1.0, zorder=3,
               label=f'Rejected (n={int(n_rejected.sum())})')
        ax.bar(bin_centers, n_accepted, width=bin_widths, bottom=n_rejected,
               color=bar_color, edgecolor=SURFACE, linewidth=1.0, zorder=3,
               label=f'Accepted (n={int(n_accepted.sum())})')

        for xv, col, lbl in zip(thresholds, threshold_colors, threshold_labels):
            ax.axvline(xv, color=col, linewidth=1.5, linestyle='--', zorder=4, label=lbl)

        mean_v   = float(np.mean(vals))
        median_v = float(np.median(vals))
        ax.axvline(mean_v,   color=INK_PRIMARY, linewidth=1.5, linestyle='-',  zorder=5,
                   label=f'Mean   {mean_v:.3f}')
        ax.axvline(median_v, color=INK_PRIMARY, linewidth=1.5, linestyle=':', zorder=5,
                   label=f'Median {median_v:.3f}')

        handles, labels = ax.get_legend_handles_labels()
        # Accepted, Rejected first, then threshold / mean / median lines
        bars  = [i for l in ('Accepted', 'Rejected') for i, s in enumerate(labels) if s.startswith(l)]
        order = bars + [i for i in range(len(labels)) if i not in bars]
        _legend_outside(ax, [handles[i] for i in order], [labels[i] for i in order])

        summary = (f'n = {len(vals)}\n'
                   f'Mean ± SD = {mean_v:.3f} ± {float(np.std(vals)):.3f}\n'
                   f'Median = {median_v:.3f}')
        ax.text(1.02, 0.0, summary, transform=ax.transAxes, fontsize=8, color=INK_SECONDARY,
                ha='left', va='bottom')
    else:
        ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color=INK_MUTED)

    ax.set_xlabel(xlabel, color=INK_PRIMARY)
    ax.set_ylabel('Number of units', color=INK_PRIMARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc='left')
    return fig


# ── Build and save all histograms ─────────────────────────────────────────────

base = os.path.splitext(OUTPUT_EXCEL)[0]

histogram_specs = [
    # (sheet_name,  png_suffix,        df_col,               title,                            xlabel,                          thresholds,                     thr_colors,                               thr_labels,                                              bar_color)
    ('Hist_SNR',     '_hist_SNR',       'snr',                'SNR Distribution',               'SNR  (A_peak / σ_noise)',       (SNR_LOW_BAD, SNR_LOW_OK, SNR_HIGH_GOOD), (COLOR_THR_BAD, COLOR_THR_MARGINAL, COLOR_THR_GOOD), (f'Too low {SNR_LOW_BAD}', f'Marginal {SNR_LOW_OK}', f'High {SNR_HIGH_GOOD}'), ACCEPTED_COLOR),
    ('Hist_ISI',     '_hist_ISI',       'isi_violation_pct',  'ISI Violation Distribution',     'ISI violations  (%)',           (ISI_GOOD_PCT, ISI_MARGIN_PCT),  (COLOR_THR_GOOD, COLOR_THR_MARGINAL),    (f'Good < {ISI_GOOD_PCT}%', f'Marginal < {ISI_MARGIN_PCT}%'),                 ACCEPTED_COLOR),
    ('Hist_TmplCorr','_hist_TmplCorr',  'template_correlation','Waveform Template Correlation', 'Mean Pearson r  (spike vs template)', (0.90,),                    (COLOR_THR_GOOD,),                      ('r = 0.90',),                                                                   ACCEPTED_COLOR),
    ('Hist_PTTime',  '_hist_PTTime',    'pt_time_ms',         'Peak-to-Trough Time',            'PT time  (ms)',                 (),                              (),                                     (),                                                                              ACCEPTED_COLOR),
    ('Hist_SpikeDur','_hist_SpikeDur',  'spike_duration_ms',  'Spike Duration',                 'Spike duration  (ms)',          (),                              (),                                     (),                                                                              ACCEPTED_COLOR),
    ('Hist_PTRatio', '_hist_PTRatio',   'pt_ratio',           'Peak-to-Trough Ratio',           '|peak| / |trough|  (a.u.)',     (1.0,),                          (INK_MUTED,),                           ('ratio = 1.0',),                                                                ACCEPTED_COLOR),
    ('Hist_CSI',     '_hist_CSI',       'csi',                'Complex Spike Index (CSI)',       'CSI  (% of ISIs in [3–20 ms] with smaller 2nd spike)', (10.0,), (COLOR_THR_BAD,),                       ('CSI = 10%',),                                                                  ACCEPTED_COLOR),
    ('Hist_CSImod',  '_hist_CSImod',    'csi_modified',       'CSI modified (no amplitude criterion)', 'CSI modified  (% of ISIs in [3–20 ms])', (10.0,), (COLOR_THR_BAD,),                ('CSI = 10%',),                                                                  ACCEPTED_COLOR),
]

png_paths = {}
accepted_flags = df['accepted'].values if 'accepted' in df.columns else np.zeros(len(df), dtype=bool)
for sheet_name, suffix, col, title, xlabel, thresholds, thr_colors, thr_labels, bar_color in histogram_specs:
    vals = df[col].values if col in df.columns else []
    fig  = make_histogram(vals, accepted_flags, title, xlabel,
                          thresholds=thresholds,
                          threshold_colors=thr_colors,
                          threshold_labels=thr_labels,
                          bar_color=bar_color)
    png  = base + suffix + '.png'
    _save(fig, base + suffix)    # .png (embedded in Excel) + .svg
    png_paths[sheet_name] = png
    print(f'Histogram saved: {png}')

# ── Embed all histograms as separate sheets ───────────────────────────────────

wb2 = load_workbook(OUTPUT_EXCEL)

for sheet_name, suffix, col, title, *_ in histogram_specs:
    hs = wb2.create_sheet(sheet_name)
    hs.sheet_view.showGridLines = False
    tc = hs.cell(row=1, column=1, value=title)
    tc.font      = Font(bold=True, size=13, color='FF2E4057', name='Calibri')
    tc.alignment = Alignment(horizontal='left', vertical='center')
    hs.row_dimensions[1].height = 24
    img = XLImage(png_paths[sheet_name])
    img.anchor = 'A3'
    hs.add_image(img)

# ── Embed per-unit waveform plots (one sheet, stacked rows) ───────────────────

if waveform_plot_entries:
    ROWS_PER_PLOT = 34   # label row + ~image height in default-height rows
    wf_sheet = wb2.create_sheet('Waveforms')
    wf_sheet.sheet_view.showGridLines = False
    wf_sheet.column_dimensions['A'].width = 16

    row_cursor = 1
    for label, png_path in waveform_plot_entries:
        tc = wf_sheet.cell(row=row_cursor, column=1, value=label)
        tc.font      = Font(bold=True, size=11, color='FF2E4057', name='Calibri')
        tc.alignment = Alignment(horizontal='left', vertical='center')
        img = XLImage(png_path)
        img.anchor = f'A{row_cursor + 1}'
        wf_sheet.add_image(img)
        row_cursor += ROWS_PER_PLOT

wb2.save(OUTPUT_EXCEL)

# ── Summary ───────────────────────────────────────────────────────────────────

print(f'\nDone. Results saved to {OUTPUT_EXCEL}')
print(f'Total units processed: {len(df)}')

for col, label in [('snr', 'SNR quality'), ('isi_violation_pct', 'ISI quality')]:
    if df[col].notna().any():
        qual_fn = snr_quality if col == 'snr' else isi_quality
        summary = df[col].apply(lambda v: qual_fn(v)[0]).value_counts()
        print(f'\n{label}:')
        for lbl, n in summary.items():
            print(f'  {lbl:12s}: {n}')

for col in ['isi_violation_pct', 'template_correlation', 'pt_time_ms',
            'pt_ratio', 'spike_duration_ms', 'csi', 'csi_modified']:
    vals = df[col].dropna()
    if len(vals):
        print(f'\n{col}: mean={vals.mean():.4f}  median={vals.median():.4f}  SD={vals.std():.4f}  n={len(vals)}')

# ── Final accepted/rejected tally (printed last, once everything else is done) ─
n_accepted = int(df['accepted'].sum())
n_rejected = int((~df['accepted']).sum())
print(f'\n{"="*50}')
print(f'Cells accepted (ISI<1%, SNR>2.5, TmplCorr>0.5): {n_accepted}')
print(f'Cells rejected: {n_rejected}')
print(f'Multi-trigger units reviewed manually: {n_manual_accepted} accepted, '
      f'{n_manual_cleaned} accepted with MT spikes -> cluster 0, '
      f'{n_manual_rejected} rejected (of {len(review_candidates)} flagged)')
print(f'{"="*50}')
