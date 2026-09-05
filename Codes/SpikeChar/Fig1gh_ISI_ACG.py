import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import os

mpl.rcParams['font.family'] = 'Arial'

# ── Neuralynx .ntt dtype ────────────────────────────────────────────────────
NTT_HEADER_BYTES   = 16 * 1024
DEFAULT_ADBITVOLTS = 0.000000195

NTT_DTYPE = np.dtype([
    ('timestamp',   '<u8'),
    ('sc_number',   '<u4'),
    ('cell_number', '<u4'),
    ('params',      '<u4', (8,)),
    ('waveforms',   '<i2', (32, 4)),
])

# ── Configuration ────────────────────────────────────────────────────────────
NTT_FOLDER  = r'C:/Runita/NMR/Data/ClusterPlot'
SAVE_FOLDER = r'C:/Runita/NMR/analysis/SurgeryPaperSpikeLFP/8477/ClusterProj_Wavfm_ISI_ACG/Figures/v5_ISI_Peaks'   # <-- output folder for all figures
NTT_FILES   = sorted(glob.glob(os.path.join(NTT_FOLDER, '*.ntt')))
MAX_SPIKES  = 500    # max waveforms to overlay per neuron in waveform plots

os.makedirs(SAVE_FOLDER, exist_ok=True)

if not NTT_FILES:
    raise FileNotFoundError(f"No .ntt files found in: {NTT_FOLDER}")

print(f'Found {len(NTT_FILES)} neuron(s):')
for f in NTT_FILES:
    print(f'  {os.path.basename(f)}')

# ── Shared colour palette – Office chart colours (matches reference figures) ──
_PALETTE  = ['#4472C4', '#ED7D31', '#70AD47', '#C00000', '#7030A0',
             '#00B0F0', '#FFC000', '#375623', '#833C00', '#31849B']
n_neurons = len(NTT_FILES)
colors    = [_PALETTE[i % len(_PALETTE)] for i in range(n_neurons)]

# ── Helpers ──────────────────────────────────────────────────────────────────
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
    adbitvolts = _parse_adbitvolts(ntt_path)
    uv_scale   = adbitvolts * 1e6
    spike_data = np.memmap(ntt_path, dtype=NTT_DTYPE, mode='r',
                           offset=NTT_HEADER_BYTES)
    waveforms  = spike_data['waveforms'].astype(np.float64) * uv_scale
    timestamps = spike_data['timestamp'].astype(np.float64)
    return waveforms, timestamps, adbitvolts


def peak_amplitudes(waveforms: np.ndarray) -> np.ndarray:
    """Peak amplitude (µV) per channel per spike — largest absolute sample."""
    peak_idx = np.argmax(np.abs(waveforms), axis=1)   # (n_spikes, 4)
    n_spikes = waveforms.shape[0]
    n_ch     = waveforms.shape[2]
    spk_idx  = np.arange(n_spikes)[:, None]
    ch_idx   = np.arange(n_ch)[None, :]
    return waveforms[spk_idx, peak_idx, ch_idx]        # (n_spikes, 4)


# ── Load all neurons ─────────────────────────────────────────────────────────
neurons    = []
all_peaks  = []
neuron_ids = []

for neuron_idx, fpath in enumerate(NTT_FILES):
    waveforms, timestamps, adbitvolts = load_ntt_waveforms(fpath)

    peaks = peak_amplitudes(waveforms)
    neurons.append({
        'name'       : os.path.splitext(os.path.basename(fpath))[0],
        'waveforms'  : waveforms,
        'peaks'      : peaks,
        'timestamps' : timestamps,   # in µs (Neuralynx native)
    })
    all_peaks.append(peaks)
    neuron_ids.extend([neuron_idx] * len(peaks))
    print(f'  [{neuron_idx}] {os.path.basename(fpath):30s}  '
          f'{len(peaks):>6,} spikes  (ADBitVolts={adbitvolts:.3e})')

all_peaks  = np.vstack(all_peaks)   # (total_spikes, 4)
neuron_ids = np.array(neuron_ids)   # (total_spikes,)

# ── Per-neuron figure: waveforms (top) | ISI + ACG side-by-side (bottom) ─────
N_CHANNELS    = 4
T_AXIS        = np.linspace(0, 5, 32)
ACG_MAX_LAG   = 1000.0
ACG_BIN_SIZE  = 1.0
acg_bin_edges = np.arange(-ACG_MAX_LAG, ACG_MAX_LAG + ACG_BIN_SIZE, ACG_BIN_SIZE)

for neuron, color in zip(neurons, colors):
    ts_ms = np.sort(neuron['timestamps']) / 1000.0
    isi   = np.diff(ts_ms)
    isi   = isi[isi > 0]

    lags = []
    for t in ts_ms:
        lo   = np.searchsorted(ts_ms, t - ACG_MAX_LAG, side='left')
        hi   = np.searchsorted(ts_ms, t + ACG_MAX_LAG, side='right')
        diff = ts_ms[lo:hi] - t
        lags.append(diff[diff != 0])
    all_lags = np.concatenate(lags) if lags else np.array([])

    fig = plt.figure(figsize=(16, 8))
    gs  = fig.add_gridspec(2, N_CHANNELS, hspace=0.5, wspace=0.35)
    fig.suptitle(neuron['name'], fontsize=15, fontweight='bold', y=1.01)

    # ── Row 0: waveforms (one panel per channel) ──
    wf_all = neuron['waveforms']
    n_plot = min(MAX_SPIKES, len(wf_all))
    idx    = np.random.choice(len(wf_all), n_plot, replace=False)
    ax_wf  = []
    for ch in range(N_CHANNELS):
        ax = fig.add_subplot(gs[0, ch], sharey=ax_wf[0] if ch > 0 else None)
        ax_wf.append(ax)
        wf_sub  = wf_all[idx, :, ch]
        mean_wf = wf_sub.mean(axis=0)
        std_wf  = wf_sub.std(axis=0)
        ax.fill_between(T_AXIS, mean_wf - std_wf, mean_wf + std_wf,
                        color=color, alpha=0.3)
        ax.plot(T_AXIS, mean_wf, color=color, linewidth=2)
        ax.set_title(f'Channel {ch + 1}', fontsize=13)
        ax.set_xlabel('Time (ms)', fontsize=11)
        if ch == 0:
            ax.set_ylabel('Amplitude (µV)', fontsize=11)
        ax.tick_params(axis='both', labelsize=9)
        ax.spines[['top', 'right']].set_visible(False)

    # ── Row 1 left: ISI ──
    ax_isi = fig.add_subplot(gs[1, :2])
    if len(isi) >= 2:
        isi_bins = np.logspace(np.log10(max(isi.min(), 0.1)), np.log10(isi.max()), 80).tolist()
        counts, bin_edges, _ = ax_isi.hist(isi, bins=isi_bins, color=color, edgecolor='none', alpha=0.85)
        ax_isi.axvline(1, color='k', linestyle='--', linewidth=1.2, alpha=0.6)

        # Mark the ISI bin with the highest count within 0-10 ms
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mask_0_10   = bin_centers <= 10
        if mask_0_10.any():
            peak_center = bin_centers[mask_0_10][np.argmax(counts[mask_0_10])]
            ax_isi.axvline(peak_center, color='red', linestyle='-', linewidth=1.5, alpha=0.85,
                            label=f'Peak (0-10 ms): {peak_center:.2f} ms')

        # Reference lines at 200 ms and 250 ms
        for ref_ms in (200, 250):
            ax_isi.axvline(ref_ms, color='red', linestyle='-', linewidth=1.5, alpha=0.85)

        ax_isi.legend(fontsize=8, loc='upper right', frameon=False)
    ax_isi.set_xscale('log')
    ax_isi.set_xlabel('Inter-Spike Interval (ms)', fontsize=12)
    ax_isi.set_ylabel('Count', fontsize=12)
    ax_isi.set_title('ISI', fontsize=13, fontweight='bold')
    ax_isi.tick_params(axis='both', labelsize=10)
    ax_isi.spines[['top', 'right']].set_visible(False)

    # ── Row 1 right: ACG ──
    ax_acg = fig.add_subplot(gs[1, 2:])
    if len(all_lags):
        ax_acg.hist(all_lags, bins=acg_bin_edges, color=color, edgecolor='none', alpha=0.85) # type: ignore
    ax_acg.set_xlim(-ACG_MAX_LAG, ACG_MAX_LAG)
    ax_acg.axvline(0, color='k', linewidth=0.8, alpha=0.4)
    ax_acg.set_xlabel('Time lag (ms)', fontsize=12)
    ax_acg.set_ylabel('Count', fontsize=12)
    ax_acg.set_title('Autocorrelogram', fontsize=13, fontweight='bold')
    ax_acg.tick_params(axis='both', labelsize=10)
    ax_acg.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    for ext in ('png', 'svg'):
        save_path = os.path.join(SAVE_FOLDER, f'{neuron["name"]}.{ext}')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.close(fig)

# ── Plot 3: one figure per channel-pair cluster scatter ───────────────────────
CH_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

for cx, cy in CH_PAIRS:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor('#f9f9f9')
    lim_x = all_peaks[:, cx].max() + 50
    lim_y = all_peaks[:, cy].max() + 50

    for i in range(n_neurons):
        mask = neuron_ids == i
        ax.scatter(
            all_peaks[mask, cx],
            all_peaks[mask, cy],
            color=colors[i],
            marker='o',
            s=8,
            alpha=0.6,
            linewidths=0,
            label=neurons[i]['name'],
        )

    ax.set_xlim(0, lim_x)
    ax.set_ylim(0, lim_y)
    ax.set_xlabel(f'Ch {cx + 1} amplitude (µV)', fontsize=13, labelpad=8)
    ax.set_ylabel(f'Ch {cy + 1} amplitude (µV)', fontsize=13, labelpad=8)
    ax.set_title(f'Ch {cx + 1} vs Ch {cy + 1}', fontsize=14, fontweight='bold', pad=12)
    ax.tick_params(axis='both', labelsize=11, length=4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(1.2)
    plt.tight_layout()
    base_name = f'clusters_ch{cx + 1}_vs_ch{cy + 1}'
    for ext in ('png', 'svg'):
        save_path = os.path.join(SAVE_FOLDER, f'{base_name}.{ext}')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.close(fig)

plt.show()
