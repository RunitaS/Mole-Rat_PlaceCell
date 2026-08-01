# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 22:50:40 2023

PSD batch processing and plotting, with targeted Dual-Criteria Epoch Rejection.

LFP Cleaning Steps in PSD_Batch_Avg_Downsampled_zeroPhaseHighPass.py
Unit conversion — raw ADC counts converted to true microvolts by multiplying by ADBitVolts and 1e6 (line 139), done before downsampling.

Downsampling — signal.resample_poly resamples from 32 kHz (fs) to 250 Hz (fs_down) using polyphase filtering (line 140).

50 Hz line-noise notch filter — clean_lfp() applies a zero-phase IIR notch (signal.iirnotch, Q=35) via filtfilt to remove 50 Hz line noise, while no high-pass filter is applied (deliberately, to preserve physiological 1–3 Hz delta rhythms — noted in the docstring and the file's revision comments at the top).

Dual-criteria epoch rejection (in compute_psd_clean_epochs()), applied to 4-second Welch windows:

Criterion 1 — broadband peak-to-peak amplitude: rejects epochs whose p2p amplitude exceeds median + 5×MAD (catches fast transient spikes/pops).
Criterion 2 — 1–3 Hz band power: a 4th-order Butterworth bandpass (sosfiltfilt) isolates 1–3 Hz power; epochs exceeding median + 5×MAD are rejected (catches large mechanical/movement baseline sways).
Only epochs passing both criteria are kept for PSD averaging.
Per-epoch detrending — linear detrend (signal.detrend) applied to each kept epoch before FFT (line 98).

Windowing — Hann window applied per epoch before the FFT, with proper power normalization (lines 93–101).

Post-hoc notch-dip interpolation for plotting only — interp_notch_psd() log-interpolates over the residual 50±3 Hz notch dip in the averaged PSD curve for display — this is cosmetic, applied after all real cleaning/averaging, not to the time-domain signal.

Snippet selection for raw-trace figures — find_clean_snippets() reuses the same p2p + 1–3 Hz power metrics (normalized and summed) to pick the "calmest" 1-second windows to display as example raw LFP traces.
@author: shirdhankar
"""

#%% import libraries
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os
from scipy import signal # type: ignore

#%% Settings

# Root directories — one per animal
# Make sure you arrange the .ncs files as separate animal in it's respective arena. Don't mix the animals IDs or the arenas.
# Compare peaks and powers in different anmals and arenas.
ANIMALS = {
    'Fa8477': r'C:/Runita/NMR/analysis/SurgeryPaperSpikeLFP/LFP/Main/8477_single',
    # 'FaDDE42': r'C:/Runita/NMR/analysis/SurgeryPaperSpikeLFP/LFP/Main/DDE42',
    # 'Fa23BD': r'C:/Runita/NMR/analysis/SurgeryPaperSpikeLFP/LFP/23BD',
}

OUTPUT_DIR = r'C:/Runita/NMR/analysis/SurgeryPaperSpikeLFP/LFP/Figs/trash_v2'  # saved plots go here

fs      = 32000  # original sampling rate (Hz)
fs_down = 250    # target sampling rate after downsampling
nperseg = int(4 * fs_down)  # 4-second Welch segments (1000 samples at 250 Hz)
ADBitVolts = 0.000003051757812500000169  # V per ADC count (Neuralynx header)

# neuralynx .ncs file format
ncs_dtype = np.dtype([
    ('timestamp'  , '<u8'),
    ('sc_number'  , '<u4'),
    ('cell_number', '<u4'),
    ('params'     , '<u4'),
    ('samples'    , '<i2', (512,)),
])

#%% Data cleaning pipeline

def clean_lfp(x, fs):
    """
    Applies biologically safe 50Hz IIR notch filtering.
    Removed high-pass filtering so physiological 1-3Hz Delta is preserved.
    """
    # 50 Hz line noise notch (Q=35 matches MATLAB bw = wo/35)
    b_n, a_n = signal.iirnotch(w0=50, Q=35, fs=fs)
    x = signal.filtfilt(b_n, a_n, x)

    return x


def compute_psd_clean_epochs(lfp, fs, nperseg, mad_thresh=5.0):
    """
    Welch-style PSD with DUAL epoch rejection.
    Rejects epochs based on broadband noise AND specific 1-3Hz static wander.
    """
    n_total = len(lfp) // nperseg
    if n_total == 0:
        raise ValueError('LFP too short for even one epoch')

    epochs = lfp[:n_total * nperseg].reshape(n_total, nperseg)

    # CRITERION 1: Broadband peak-to-peak amplitude (catches fast spikes/pops)
    p2p_amp = np.ptp(epochs, axis=1)
    med_p2p = np.median(p2p_amp)
    thresh_p2p = med_p2p + mad_thresh * (np.median(np.abs(p2p_amp - med_p2p)) / 0.6745)

    # CRITERION 2: 1-3 Hz Band Power (catches the massive mechanical baseline sways)
    sos_low = signal.butter(4, [1.0, 3.0], btype='bandpass', fs=fs, output='sos')
    low_power = np.mean(signal.sosfiltfilt(sos_low, epochs, axis=1) ** 2, axis=1)
    med_low = np.median(low_power)
    thresh_low = med_low + mad_thresh * (np.median(np.abs(low_power - med_low)) / 0.6745)

    # Keep epochs that are safe from BOTH types of noise
    clean_mask = (p2p_amp <= thresh_p2p) & (low_power <= thresh_low)

    n_clean = int(clean_mask.sum())
    if n_clean == 0:
        raise ValueError('All epochs rejected — loosen mad_thresh or check data for heavy noise')

    # One-sided PSD averaged over clean epochs (matches Welch 'hann', density)
    win = signal.windows.hann(nperseg) # type: ignore
    win_power = np.sum(win ** 2)
    psds = []
    
    for ep in epochs[clean_mask]:
        ep_d = signal.detrend(ep, type='linear')
        fft_ep = np.fft.rfft(ep_d * win)
        psd_ep = (np.abs(fft_ep) ** 2) / (fs * win_power)
        psd_ep[1:-1] *= 2  # one-sided correction (exclude DC and Nyquist)
        psds.append(psd_ep)

    f = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    psd = np.mean(psds, axis=0)
    
    return f, psd, n_clean, n_total


def interp_notch_psd(f, Pxx, notch_hz=50, bw_hz=3):
    """Interpolate over any residual notch-filter dip in a PSD vector."""
    mask = np.abs(f - notch_hz) <= bw_hz
    good = ~mask
    out = Pxx.copy()
    log_pxx = np.log(np.maximum(Pxx, 1e-30))
    out[mask] = np.exp(np.interp(f[mask], f[good], log_pxx[good]))
    return out


#%% Helper — process one animal folder → (freq_vec, mean_psd, sem_psd, n_files)

def process_animal(folder):
    ncs_files = []
    for root, dirs, files in os.walk(folder):
        for fname in files:
            if fname.endswith('.ncs'):
                ncs_files.append(os.path.join(root, fname))

    print(f'Found {len(ncs_files)} .ncs files in {folder}')

    psds     = []
    freq_vec = None

    for fpath in ncs_files:
        try:
            data = np.memmap(fpath, dtype=ncs_dtype, mode='r', offset=16*1024)
            
            # Conversion happens BEFORE downsampling (Corrected to 1e6 for µV)
            lfp  = np.concatenate(data['samples']).astype(np.float64) * ADBitVolts * 1e6
            lfp  = signal.resample_poly(lfp, fs_down, fs)
            lfp  = clean_lfp(lfp, fs_down)

            f, Pxx, n_clean, n_total = compute_psd_clean_epochs(
                lfp, fs_down, nperseg, mad_thresh=5.0
            )

            if freq_vec is None:
                freq_vec = f

            psds.append(Pxx)
            print(f'  OK: {os.path.relpath(fpath, folder)}'
                  f'  [{n_clean}/{n_total} epochs kept]')

        except Exception as e:
            print(f'  SKIP: {os.path.relpath(fpath, folder)} — {e}')

    print(f'  -> {len(psds)} files processed\n')

    if not psds:
        raise ValueError(f'No files processed successfully in {folder}')

    psds     = np.array(psds)
    mean_psd = np.mean(psds, axis=0)
    sem_psd  = np.std(psds,  axis=0) / np.sqrt(psds.shape[0])
    return freq_vec, mean_psd, sem_psd, psds.shape[0]


#%% Process each animal

results = {}
for label, folder in ANIMALS.items():
    print(f'=== {label} ===')
    results[label] = process_animal(folder)

#%% Plot — both animals overlaid

STYLES = {
    'Fa8477': ('#1A56DB', '#93C5FD'),   # blue
}

plt.rcParams.update({
    'font.family'      : 'Arial',
    'font.size'        : 12,
    'axes.linewidth'   : 1.0,
    'xtick.direction'  : 'out',
    'ytick.direction'  : 'out',
    'xtick.major.size' : 5,
    'ytick.major.size' : 5,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'legend.frameon'   : False,
    'figure.facecolor' : 'white',
    'axes.facecolor'   : 'white',
})

fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=1000)

y_min_global, y_max_global = np.inf, -np.inf

for label, (freq_vec, mean_psd, sem_psd, n_files) in results.items():
    line_col, fill_col = STYLES.get(label, ('#333333', '#AAAAAA'))
    freq_mask = (freq_vec >= 1) & (freq_vec <= 100)

    mean_psd_plot = interp_notch_psd(freq_vec, mean_psd, notch_hz=50, bw_hz=3)
    fill_hi = interp_notch_psd(freq_vec, mean_psd + sem_psd, notch_hz=50, bw_hz=3)
    fill_lo = interp_notch_psd(freq_vec, np.maximum(mean_psd - sem_psd, 1e-12), notch_hz=50, bw_hz=3)

    ax.fill_between(
        freq_vec,
        fill_lo,
        fill_hi,
        alpha=0.25, color=fill_col, zorder=2
    )
    ax.semilogy(freq_vec, mean_psd_plot, color=line_col, linewidth=2.0, zorder=3)

    peak_idx  = np.argmax(mean_psd_plot[freq_mask])
    peak_freq = freq_vec[freq_mask][peak_idx]
    peak_psd  = mean_psd_plot[freq_mask][peak_idx]
    ax.axvline(peak_freq, color=line_col, linewidth=1.0, linestyle='--', alpha=0.55, zorder=2)
    ax.plot(peak_freq, peak_psd, 'o', color=line_col, markersize=7, zorder=5)
    ax.text(peak_freq * 1.12, peak_psd, f'{peak_freq:.1f} Hz',
            color=line_col, fontsize=9, va='center', zorder=5)

    y_min_global = min(y_min_global, fill_lo[freq_mask].min())
    y_max_global = max(y_max_global, fill_hi[freq_mask].max())

# Scales and limits
ax.set_xscale('log')
ax.set_xlim([1, 100]) # type: ignore
ax.set_ylim(max(y_min_global * 0.5, 1e-12), y_max_global * 2)

ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
ax.xaxis.set_major_formatter(plt.ScalarFormatter()) # type: ignore
ax.tick_params(axis='x', which='minor', bottom=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('Frequency (Hz)', fontsize=13, labelpad=6)
ax.set_ylabel('PSD (µV²/Hz)',   fontsize=13, labelpad=6)
ax.set_title('Power Spectral Density', fontsize=13, pad=14, fontweight='semibold')

plt.tight_layout()

os.makedirs(OUTPUT_DIR, exist_ok=True)
svg_path = os.path.join(OUTPUT_DIR, 'PSD_Batch_Avg.svg')
png_path = os.path.join(OUTPUT_DIR, 'PSD_Batch_Avg.png')
fig.savefig(svg_path, format='svg', bbox_inches='tight')
fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
print(f'SVG saved to: {svg_path}')
print(f'PNG saved to: {png_path}')

plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#%% Raw LFP snippet plots — 2 × 200 ms clean examples per .ncs file
# ══════════════════════════════════════════════════════════════════════════════

SNIPPET_DUR_MS    = 1000  
SNIPPET_N         = 2     

def _nice_scale(value: float) -> float:
    exp  = np.floor(np.log10(max(abs(value), 1e-9)))
    frac = value / 10 ** exp
    nice = 1 if frac <= 1 else 2 if frac <= 2 else 5 if frac <= 5 else 10
    return nice * 10 ** exp


def find_clean_snippets(lfp: np.ndarray, fs: int,
                        n: int = SNIPPET_N,
                        dur_ms: float = SNIPPET_DUR_MS) -> tuple:
    n_samp = int(dur_ms / 1000.0 * fs)
    n_seg  = len(lfp) // n_samp
    if n_seg < n:
        raise ValueError(f'Signal too short: need ≥ {n} × {dur_ms} ms windows')

    segs  = lfp[: n_seg * n_samp].reshape(n_seg, n_samp)
    
    # Calculate both P2P and 1-3Hz power for snippet selection
    p2p_amp = np.ptp(segs, axis=1)
    
    sos_low = signal.butter(4, [1.0, 3.0], btype='bandpass', fs=fs, output='sos')
    low_power = np.mean(signal.sosfiltfilt(sos_low, segs, axis=1) ** 2, axis=1)

    # Normalize metrics so we can sum them to find the absolute "calmest" baseline
    norm_p2p = p2p_amp / np.median(p2p_amp)
    norm_low = low_power / np.median(low_power)
    combined_noise = norm_p2p + norm_low

    order   = np.argsort(combined_noise)
    min_gap = 5                                                
    chosen  = []
    for idx in order:
        if all(abs(idx - c) >= min_gap for c in chosen):
            chosen.append(int(idx))
        if len(chosen) == n:
            break

    if len(chosen) < n:               
        chosen = [int(i) for i in order[:n]]

    return [c * n_samp for c in chosen], n_samp


def plot_lfp_snippets(lfp: np.ndarray, fs: int,
                      starts: list, n_samp: int,
                      title: str, out_path: str,
                      color: str = '#1A56DB') -> None:
    n   = len(starts)
    fig = plt.figure(figsize=(6.5, 2.6 * n), facecolor='white')
    fig.subplots_adjust(hspace=0.55)

    t_ms = np.linspace(0, n_samp / fs * 1000.0, n_samp, endpoint=False)

    for i, s0 in enumerate(starts):
        ax = fig.add_subplot(n, 1, i + 1)
        snippet = lfp[s0 : s0 + n_samp]

        ax.plot(t_ms, snippet,
                color=color, linewidth=1.0, alpha=0.95,
                solid_capstyle='round', solid_joinstyle='round', zorder=3)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([]);  ax.set_yticks([])
        ax.set_facecolor('white')
        ax.axhline(0, color='#E5E7EB', linewidth=0.6, zorder=1)

        y_range  = float(snippet.max() - snippet.min())
        v_bar    = _nice_scale(y_range * 0.25)   
        t_bar_ms = 200.0                          

        x0 = t_ms[-1] * 0.05
        y0 = float(snippet.min()) - y_range * 0.18

        ax.plot([x0, x0 + t_bar_ms], [y0, y0],
                color='#1F2937', linewidth=1.6,
                solid_capstyle='butt', zorder=4)
        ax.plot([x0, x0], [y0, y0 + v_bar],
                color='#1F2937', linewidth=1.6,
                solid_capstyle='butt', zorder=4)

        ax.text(x0 + t_bar_ms / 2, y0 - y_range * 0.07,
                f'{t_bar_ms:.0f} ms',
                ha='center', va='top', fontsize=8,
                color='#1F2937', fontfamily='Arial')
        ax.text(x0 - t_ms[-1] * 0.025, y0 + v_bar / 2,
                f'{v_bar:.0f} µV',
                ha='right', va='center', fontsize=8,
                color='#1F2937', fontfamily='Arial', rotation=90)

        ax.set_xlim(-t_ms[-1] * 0.02, t_ms[-1] * 1.02)

        ax.text(-0.06, 1.05, chr(ord('A') + i),
                transform=ax.transAxes,
                fontsize=11, fontweight='bold', color='#111827',
                va='bottom', ha='left', fontfamily='Arial')

    fig.suptitle(title, fontsize=10.5, fontweight='semibold',
                 color='#111827', y=1.01)

    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    svg_path = os.path.splitext(out_path)[0] + '.svg'
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'   snippet → {os.path.basename(out_path)}  +  .svg')


#%% Generate per-file LFP snippet plots 

os.makedirs(OUTPUT_DIR, exist_ok=True)

for label, folder in ANIMALS.items():
    snippet_color = STYLES.get(label, ('#1A56DB', None))[0]

    ncs_files = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.endswith('.ncs'):
                ncs_files.append(os.path.join(root, fname))

    for fpath in ncs_files:
        rel = os.path.relpath(fpath, folder)
        print(f'[{label}] snippet: {rel}')
        try:
            raw       = np.memmap(fpath, dtype=ncs_dtype, mode='r', offset=16 * 1024)
            
            lfp       = np.concatenate(raw['samples']).astype(np.float64) * ADBitVolts * 1e6
            lfp       = np.asarray(signal.resample_poly(lfp, fs_down, fs))
            lfp       = np.asarray(clean_lfp(lfp, fs_down))

            starts, n_samp = find_clean_snippets(
                lfp, fs_down, n=SNIPPET_N, dur_ms=SNIPPET_DUR_MS)

            safe      = rel.replace(os.sep, '_').replace('.ncs', '')
            out_path  = os.path.join(OUTPUT_DIR, f'LFP_snippet_{label}_{safe}.png')

            plot_lfp_snippets(
                lfp, fs_down, starts, n_samp,
                title=f'{label}  ·  {os.path.basename(fpath)}',
                out_path=out_path,
                color=snippet_color,
            )

        except Exception as exc:
            print(f'  SKIP {rel}: {exc}')