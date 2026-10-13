# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 22:50:40 2023

PSD batch processing and plotting, with targeted Dual-Criteria Epoch Rejection.

Updated:
1. Removed global high-pass filtering to preserve continuous biological Delta rhythms.
2. Implemented dual-criteria epoch rejection: discards 4-second windows that exceed 
   MAD thresholds for either broadband peak-to-peak amplitude OR 1-3 Hz band power.
3. Removed non-linear median filtering to prevent odd-harmonic injection.
4. Removed time-domain spectral interpolation to preserve phase integrity.
5. Corrected unit conversion from ADC counts to true microvolts (1e6).

@author: shirdhankar
"""

#%% import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import os
from scipy import signal
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr
from fooof import FOOOF

#%% Settings

# Root directories — one per animal.
# Add/remove/rename animals here (and give each one a color in ANIMAL_COLORS) —
# filenames and legends below are derived automatically from this dict's keys.
ANIMALS = {
    'Fa8477': r'X:/NMR_group_data/Runita/Data/Ephys_Data/AllSortedData/Tetrode/Fa8477',
    'Fa5834': r'C:/Runita/NMR/analysis/SurgeryPaperSpikeLFP/LFP/Main/DDE42',
    'Fa23BD': r'C:/Runita/NMR/analysis/AllSort_Results/LFP/23BDTest',
    'Fa1059' : r'C:/Runita/NMR/analysis/AllSort_Results/LFP/1059Test',
}

OUTPUT_DIR = r'C:/Runita/NMR/analysis/AllSort_Results/LFP'  # saved plots go here

# Per-animal colors — first 4 colors of matplotlib's Pastel1 palette.
ANIMAL_COLORS = {
    'Fa8477': '#FBB4AE',  # Pastel1[0] red
    'Fa1059': '#B3CDE3',  # Pastel1[1] blue
    'Fa23BD': '#CCEBC5',  # Pastel1[2] green
    'Fa5834': '#DECBE4',  # Pastel1[3] purple
}


def _lighten(hex_color, amount=0.55):
    """Blend a hex color toward white, for use as a shaded fill color."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


# {label: (line_color, fill_color)} — auto-built from ANIMALS, one entry per animal.
STYLES = {
    label: (ANIMAL_COLORS[label], _lighten(ANIMAL_COLORS[label]))
    for label in ANIMALS
}

fs      = 32000  # original sampling rate (Hz)
fs_down = 500    # target sampling rate after downsampling
nperseg = int(4 * fs_down)  # 4-second Welch segments (1000 samples at 500 Hz)
ADBitVolts = 0.000003051757812500000169  # V per ADC count (Neuralynx header)

# neuralynx .ncs file format
ncs_dtype = np.dtype([
    ('timestamp'  , '<u8'),
    ('sc_number'  , '<u4'),
    ('cell_number', '<u4'),
    ('params'     , '<u4'),
    ('samples'    , '<i2', (512,)),
])

# ---- ACG / ED_min theta-fit analysis ----
# Same parameters as ACG_theta_continuity_TT_ThetaVsSpeed_v4_FerretOLS.py
# (Dunn et al. 2022 peak-range method). Runs on its own 1 kHz branch of each
# file, independent of the 500 Hz PSD branch above.
FS_ACG              = 1000          # Hz -- reference sinusoid bank is defined on a 1 kHz time base
NPERSEG_ACG         = int(1.0 * FS_ACG)  # 1 s autocorrelogram epochs
LINE_HARMONICS      = [50.0, 100.0, 150.0, 200.0]
NOTCH_Q             = 30.0
LOWPASS_CUTOFF_HZ   = 100.0
LOWPASS_ORDER       = 4
DELTA_BAND          = (1.0, 3.0)    # Hz -- epoch rejected if delta power > theta power
THETA_BAND          = (3.0, 7.0)    # Hz
ACG_MAD_THRESH      = 5.0           # peak-to-peak robust-z artifact threshold
ACG_FREQ_RANGE      = (3.0, 7.0)    # Hz -- reference sinusoid bank
ACG_FREQ_RES        = 0.1           # Hz
ACG_PEAK_PROMINENCE = 0.05
ACG_ED_MIN_THRESH   = 1.0           # epochs with ED_min above this count as ED_min-rejected


#%% Data cleaning pipeline (PSD branch)

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


#%% ACG / ED_min quantification (ACG branch)
# Ported from ACG_theta_continuity_TT_ThetaVsSpeed_v4_FerretOLS.py. Each 1 s
# epoch's normalised autocorrelogram is matched to the closest reference
# sinusoid autocorrelogram; ED_min is the normalised Euclidean distance to it
# (0 = perfect sinusoid). No tracking data here, so unlike the reference there
# is no moving/immobile gating -- every artifact-clean epoch is used.

def notch_filter(x, fs_hz, freqs, Q=30.0):
    """Zero-phase IIR notch at each frequency in `freqs` (skips freqs >= Nyquist)."""
    y = np.asarray(x, dtype=np.float64)
    for f0 in freqs:
        if 0 < f0 < fs_hz / 2.0:
            b, a = signal.iirnotch(f0, Q, fs_hz)
            y = signal.filtfilt(b, a, y)
    return y


def lowpass_filter(x, fs_hz, cutoff_hz, order=4):
    """Zero-phase Butterworth low-pass filter."""
    b, a = signal.butter(order, cutoff_hz / (fs_hz / 2.0), btype='low')
    return signal.filtfilt(b, a, np.asarray(x, dtype=np.float64))


def _robust_high_outliers(x, thresh, ref_mask=None):
    """High outliers by robust (MAD) z-score; median/MAD taken from x[ref_mask] if given."""
    x = np.asarray(x, dtype=np.float64)
    ref = x if ref_mask is None else x[np.asarray(ref_mask, dtype=bool)]
    if ref.size == 0:
        ref = x
    med = np.median(ref)
    mad = np.median(np.abs(ref - med))
    if mad == 0:
        mad = 1e-20
    return 0.6745 * (x - med) / mad > thresh


def zero_crossings(x):
    """Fractional (linearly-interpolated) sample indices where `x` crosses zero."""
    x = np.asarray(x, dtype=np.float64)
    s = np.sign(x)
    s[s == 0] = 1.0
    idx = np.where(np.diff(s) != 0)[0]
    x1, x2 = x[idx], x[idx + 1]
    return idx + (-x1 / (x2 - x1))


def _find_extrema(x, prominence, kind='max'):
    """Local maxima/minima above a fixed prominence (MATLAB findMinMax 'fixed')."""
    idx, _ = find_peaks(x if kind == 'max' else -np.asarray(x), prominence=prominence)
    return idx, np.asarray(x)[idx]


def create_sine_ref_xcorrs(ref_freqs, datsize, fs=FS_ACG, peak_prominence=ACG_PEAK_PROMINENCE):
    """Reference bank of sinusoid autocorrelograms + peak/trough search windows
    (translation of create_sine_ref_xcorrs.m). See the reference script for details."""
    ref_freqs = np.asarray(ref_freqs, dtype=np.float64)
    n_freqs = ref_freqs.size
    reft = np.arange(datsize) / fs
    center = datsize - 1

    refXC = np.empty((2 * datsize - 1, n_freqs))
    refRange = np.empty(n_freqs)
    refP1range = np.empty((2, n_freqs), dtype=int)
    refT1range = np.empty((2, n_freqs), dtype=int)
    refT2range = np.empty((2, n_freqs), dtype=int)

    for n, freq in enumerate(ref_freqs):
        refsig = np.sin(2 * np.pi * freq * reft)
        refxc_raw = signal.correlate(refsig, refsig, mode='full', method='auto')
        refxc = refxc_raw / np.max(refxc_raw)
        refXC[:, n] = refxc

        max_idx, max_val = _find_extrema(refxc, peak_prominence, kind='max')
        min_idx, min_val = _find_extrema(refxc, peak_prominence, kind='min')

        center_matches = np.where(max_idx == center)[0]
        center_row = int(center_matches[0]) if center_matches.size else \
            int(np.argmin(np.abs(max_idx - center)))
        if center_row + 1 >= len(max_idx):
            raise ValueError(f'{freq:.3f} Hz reference sine: no side peak after zero lag.')
        peak1_idx = max_idx[center_row + 1]
        peak1_val = max_val[center_row + 1]

        before = min_idx < peak1_idx
        after = min_idx > peak1_idx
        if not before.any() or not after.any():
            raise ValueError(f'{freq:.3f} Hz reference sine: could not bracket troughs.')
        trough1_idx, trough1_val = min_idx[before][-1], min_val[before][-1]
        trough2_idx, trough2_val = min_idx[after][0], min_val[after][0]

        refRange[n] = peak1_val - np.mean([trough1_val, trough2_val])

        crossings = zero_crossings(refxc)
        belowt1 = crossings[crossings < trough1_idx]
        abovet2 = crossings[crossings > trough2_idx]
        abovep1 = crossings[crossings > peak1_idx]
        belowp1 = crossings[crossings < peak1_idx]
        if not (belowp1.size and abovep1.size and belowt1.size and abovet2.size):
            raise ValueError(f'{freq:.3f} Hz reference sine: no bounding zero-crossings.')

        refP1range[:, n] = [int(round(belowp1[-1])), int(round(abovep1[0]))]
        refT1range[:, n] = [int(round(belowt1[-1])), int(round(belowp1[-1]))]
        refT2range[:, n] = [int(round(abovep1[0])), int(round(abovet2[0]))]

    refED = np.linalg.norm(refXC, axis=0)
    return refXC, refED, refRange, refP1range, refT1range, refT2range


# Epoch length is fixed, so the reference bank is built once and shared by every file
REF_FREQS = np.arange(ACG_FREQ_RANGE[0], ACG_FREQ_RANGE[1] + ACG_FREQ_RES / 2, ACG_FREQ_RES)
REF_BANK  = create_sine_ref_xcorrs(REF_FREQS, NPERSEG_ACG)


def _normalised_autocorr(epochs):
    """Full ('-L+1 .. L-1' lag) autocorrelogram of every row of `epochs`,
    normalised to 1 at zero lag -- FFT-batched equivalent of
    signal.correlate(ep, ep, 'full') / max."""
    L = epochs.shape[1]
    nfft = 1 << int(np.ceil(np.log2(2 * L - 1)))
    spec = np.fft.rfft(epochs, n=nfft, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, n=nfft, axis=1)
    xc = np.concatenate([ac[:, nfft - (L - 1):], ac[:, :L]], axis=1)
    return xc / xc.max(axis=1, keepdims=True)


def quantify_edmin_epochs(epochs, ed_min_thresh=ACG_ED_MIN_THRESH):
    """ED_min, matched frequency and normalised peak range per epoch -- a
    vectorised version of quantify_xcorr_epochs from the reference script
    (same matching, peak/trough windows and normalisation).

    `epochs` : (n_epochs, NPERSEG_ACG) array at FS_ACG.
    """
    refXC, refED, refRange, refP1, refT1, refT2 = REF_BANK
    xc = _normalised_autocorr(epochs)

    # ||ref - xc||^2 = ||ref||^2 + ||xc||^2 - 2 ref.xc, for every epoch x reference at once
    sq_dist = (np.sum(xc ** 2, axis=1)[:, None] + (refED ** 2)[None, :] - 2 * xc @ refXC)
    norm_ed = np.sqrt(np.maximum(sq_dist, 0)) / refED
    mi = np.argmin(norm_ed, axis=1)
    ed_min = norm_ed[np.arange(len(mi)), mi]

    # Peak/trough windows depend on the matched reference -> group epochs by it
    peakrangenorm = np.empty(len(mi))
    for k in np.unique(mi):
        rows = np.where(mi == k)[0]
        pk = xc[rows, refP1[0, k]:refP1[1, k] + 1].max(axis=1)
        t1 = xc[rows, refT1[0, k]:refT1[1, k] + 1].min(axis=1)
        t2 = xc[rows, refT2[0, k]:refT2[1, k] + 1].min(axis=1)
        peakrangenorm[rows] = (pk - (t1 + t2) / 2) / refRange[k]

    skipped_edmin = (ed_min > ed_min_thresh) if ed_min_thresh is not None \
        else np.zeros(len(mi), dtype=bool)
    return pd.DataFrame({
        'ED_min': ed_min,
        'freq': REF_FREQS[mi],
        'peakrangenorm': peakrangenorm,
        'skipped_edmin': skipped_edmin,
    })


def reject_acg_epochs(epochs, fs_hz):
    """Keep-mask for 1 s ACG epochs, as in the reference script: reject epochs
    whose delta power exceeds theta power, then peak-to-peak MAD outliers
    (MAD reference taken from the delta/theta survivors only)."""
    f, Pxx = signal.welch(epochs, fs=fs_hz, window='hann', nperseg=epochs.shape[1],
                          noverlap=0, detrend='constant', axis=1)
    df = f[1] - f[0]
    low_pow   = Pxx[:, (f >= DELTA_BAND[0]) & (f <= DELTA_BAND[1])].sum(axis=1) * df
    theta_pow = Pxx[:, (f >= THETA_BAND[0]) & (f <= THETA_BAND[1])].sum(axis=1) * df
    keep_dt = low_pow <= theta_pow

    p2p = np.ptp(epochs, axis=1)
    keep_art = ~_robust_high_outliers(p2p, ACG_MAD_THRESH, ref_mask=keep_dt)
    return keep_dt & keep_art & (p2p > 0)   # p2p > 0 drops flat (dead-channel) epochs


def compute_edmin_for_file(lfp_uv_native):
    """1 kHz ACG branch for one file: resample -> notch -> detrend -> low-pass,
    1 s epochs, artifact rejection, ED_min quantification.

    Returns (per-epoch DataFrame for the clean epochs, n_total epochs).
    """
    x = signal.resample_poly(lfp_uv_native, FS_ACG, fs)
    x = notch_filter(x, FS_ACG, LINE_HARMONICS, NOTCH_Q)
    x = signal.detrend(x, type='linear')
    x = lowpass_filter(x, FS_ACG, LOWPASS_CUTOFF_HZ, LOWPASS_ORDER)

    n_total = len(x) // NPERSEG_ACG
    if n_total == 0:
        raise ValueError('Trace is shorter than one ACG epoch.')
    epochs = x[:n_total * NPERSEG_ACG].reshape(n_total, NPERSEG_ACG)

    keep = reject_acg_epochs(epochs, FS_ACG)
    if not keep.any():
        raise ValueError('No clean ACG epochs after artifact rejection.')
    return quantify_edmin_epochs(epochs[keep]), n_total


#%% Helper — process one animal folder → (freq_vec, mean_psd, sem_psd, n_files)
# Each .ncs file is read, downsampled, and notch-filtered ONCE; that single
# cleaned trace is reused for both the PSD accumulation and the snippet figure
# (previously each file was re-read and re-filtered a second time for snippets).

def process_animal(label, folder):
    ncs_files = []
    for root, dirs, files in os.walk(folder):
        for fname in files:
            if fname.endswith('.ncs'):
                ncs_files.append(os.path.join(root, fname))

    print(f'Found {len(ncs_files)} .ncs files in {folder}')

    # line_col = STYLES[label][0]  # only needed for the LFP snippet figures below
    psds      = []
    freq_vec  = None
    edmin_dfs = []   # per-file ED_min epoch tables (ACG branch)
    rej_rows  = []   # per-file epoch-rejection percentages

    for fpath in ncs_files:
        rel = os.path.relpath(fpath, folder)
        try:
            data = np.memmap(fpath, dtype=ncs_dtype, mode='r', offset=16*1024)

            # Conversion happens BEFORE downsampling (Corrected to 1e6 for µV)
            lfp_raw = np.concatenate(data['samples']).astype(np.float64) * ADBitVolts * 1e6
            lfp = np.asarray(signal.resample_poly(lfp_raw, fs_down, fs))
            lfp = np.asarray(clean_lfp(lfp, fs_down))

            f, Pxx, n_clean, n_total = compute_psd_clean_epochs(
                lfp, fs_down, nperseg, mad_thresh=5.0
            )

            if freq_vec is None:
                freq_vec = f

            # --- NORMALIZATION BLOCK ---
            # Calculate total power between 1 and 100 Hz
            df = f[1] - f[0] # Frequency resolution
            valid_idx = (f >= 1.0) & (f <= 100.0)
            total_power = np.sum(Pxx[valid_idx]) * df

            # Divide PSD by total power to get relative power
            Pxx_norm = Pxx / total_power
            # ---------------------------

            psds.append(Pxx_norm) # Append the normalized array
            print(f'  OK: {rel}  [{n_clean}/{n_total} epochs kept]')

            # --- ACG / ED_min branch (1 kHz, 1 s epochs) from the same raw trace ---
            rej = {'animal': label, 'file': rel,
                   'pct_psd_rejected': 100 * (n_total - n_clean) / n_total,
                   'pct_acg_rejected': np.nan, 'pct_edmin_rejected': np.nan}
            try:
                edmin_df, n_acg_total = compute_edmin_for_file(lfp_raw)
                edmin_df['animal'] = label
                edmin_df['file'] = rel
                edmin_dfs.append(edmin_df)
                n_acg_clean = len(edmin_df)
                n_edmin_rej = int(edmin_df['skipped_edmin'].sum())
                rej['pct_acg_rejected'] = 100 * (n_acg_total - n_acg_clean) / n_acg_total
                rej['pct_edmin_rejected'] = 100 * n_edmin_rej / n_acg_clean
                print(f'      ED_min: {n_acg_clean}/{n_acg_total} ACG epochs clean; '
                      f'ED_min > {ACG_ED_MIN_THRESH:g} rejected {n_edmin_rej} '
                      f'({rej["pct_edmin_rejected"]:.1f}%)')
            except Exception as e:
                print(f'      ED_min SKIP: {e}')
            rej_rows.append(rej)

            # --- LFP snippet figure (reuses the cleaned trace above) ---
            # Disabled for now — not needed this run, but kept for future use.
            # starts, n_samp = find_clean_snippets(
            #     lfp, fs_down, n=SNIPPET_N, dur_ms=SNIPPET_DUR_MS)
            # safe     = rel.replace(os.sep, '_').replace('.ncs', '')
            # out_path = os.path.join(OUTPUT_DIR, f'LFP_snippet_{label}_{safe}.png')
            # plot_lfp_snippets(
            #     lfp, fs_down, starts, n_samp,
            #     title=f'{label}  ·  {os.path.basename(fpath)}',
            #     out_path=out_path,
            #     color=STYLES[label][0],
            # )

        except Exception as e:
            print(f'  SKIP: {rel} — {e}')

    print(f'  -> {len(psds)} files processed\n')

    if not psds:
        raise ValueError(f'No files processed successfully in {folder}')

    psds     = np.array(psds)
    mean_psd = np.mean(psds, axis=0)
    sem_psd  = np.std(psds,  axis=0) / np.sqrt(psds.shape[0])
    edmin_all = pd.concat(edmin_dfs, ignore_index=True) if edmin_dfs else pd.DataFrame()
    return (freq_vec, mean_psd, sem_psd, psds.shape[0]), edmin_all, pd.DataFrame(rej_rows)


#%% Process each animal (LFP snippet figures currently disabled, see process_animal)

os.makedirs(OUTPUT_DIR, exist_ok=True)

results       = {}   # label -> (freq_vec, mean_psd, sem_psd, n_files)
edmin_results = {}   # label -> per-epoch ED_min DataFrame
rejection     = {}   # label -> per-file rejection-percentage DataFrame
for label, folder in ANIMALS.items():
    print(f'=== {label} ===')
    results[label], edmin_results[label], rejection[label] = process_animal(label, folder)

#%% Average % of epochs rejected per animal (mean ± SD across .ncs files)

REJ_COLS = {
    'pct_psd_rejected'  : 'PSD dual-criteria (4 s epochs)',
    'pct_acg_rejected'  : 'ACG delta/theta + MAD (1 s epochs)',
    'pct_edmin_rejected': f'ED_min > {ACG_ED_MIN_THRESH:g} (of ACG-clean epochs)',
}
rejection_df = pd.concat(rejection.values(), ignore_index=True)
print('\n=== Average % of epochs rejected per animal (mean ± SD across files) ===')
for label, sub in rejection_df.groupby('animal', sort=False):
    print(f'{label}  ({len(sub)} files)')
    for col, desc in REJ_COLS.items():
        print(f'    {desc:<42s}: {sub[col].mean():5.1f} ± {sub[col].std():4.1f} %')

rejection_path = os.path.join(OUTPUT_DIR, 'epoch_rejection_per_file.csv')
rejection_df.to_csv(rejection_path, index=False)
print(f'Saved per-file rejection percentages -> {rejection_path}\n')

#%% Plot — combined overlay + one figure per animal

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

#%% ED_min vs. normalised peak range — points coloured by animal (STYLES / ANIMAL_COLORS)

def plot_edmin_vs_peakrangenorm(df, title='ED_min vs. normalised peak range', ax=None,
                                figsize=(6, 5), point_size=10, alpha=0.3, seed=0):
    """Scatter of ED_min vs. peakrangenorm, each point coloured by its animal's
    STYLES line color, with a pooled linear fit, Pearson/Spearman stats and the
    ACG_ED_MIN_THRESH cutoff -- same layout as plot_edmin_vs_peakrangenorm in
    ACG_theta_continuity_TT_ThetaVsSpeed_v4_FerretOLS.py.

    Not filtered on 'skipped_edmin' (that flag is itself the ED_min threshold).

    Returns (fig, ax, stats_dict).
    """
    sub = df.dropna(subset=['ED_min', 'peakrangenorm'])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if len(sub) < 2:
        ax.set_title('Not enough epochs for correlation')
        return fig, ax, {'pearson_r': np.nan, 'pearson_p': np.nan,
                         'spearman_r': np.nan, 'spearman_p': np.nan, 'n': len(sub)}

    x = sub['ED_min'].to_numpy()
    y = sub['peakrangenorm'].to_numpy()
    colors = sub['animal'].map(lambda a: STYLES[a][0]).to_numpy()

    r_p, p_p = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)

    # Random draw order so no animal's cloud is buried under the one plotted last
    order = np.random.default_rng(seed).permutation(len(sub))
    ax.scatter(x[order], y[order], s=point_size, c=colors[order], alpha=alpha,
               edgecolor='none', zorder=2)

    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.array([x.min(), x.max()])
    fit_line, = ax.plot(x_line, slope * x_line + intercept, color='black', linewidth=1.5,
                        zorder=4, label='linear fit')
    thr_line = ax.axvline(ACG_ED_MIN_THRESH, color='#C44E52', linestyle='--', linewidth=1,
                          zorder=3, label=f'ED_min threshold ({ACG_ED_MIN_THRESH:g})')

    animal_handles = [
        Line2D([], [], marker='o', linestyle='none', markersize=5,
               markerfacecolor=STYLES[a][0], markeredgecolor='none',
               label=f'{a} (n={int((sub["animal"] == a).sum())})')
        for a in ANIMALS if (sub['animal'] == a).any()
    ]
    ax.legend(handles=animal_handles + [fit_line, thr_line], loc='upper right',
              fontsize=7, frameon=False)

    ax.set_xlabel('ED_min (distance to closest-matching reference sinusoid)')
    ax.set_ylabel('Autocorr. peak range (norm.)')
    ax.set_title(title)

    stats_text = (f'Pearson r = {r_p:.3f} (p = {p_p:.2e})\n'
                  f'Spearman rho = {r_s:.3f} (p = {p_s:.2e})\n'
                  f'n = {len(sub)}')
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=8, color='0.1',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='0.7'))

    fig.tight_layout()
    return fig, ax, {'pearson_r': r_p, 'pearson_p': p_p,
                     'spearman_r': r_s, 'spearman_p': p_s, 'n': len(sub)}


edmin_df = pd.concat([d for d in edmin_results.values() if not d.empty], ignore_index=True)

# All animals pooled, one colour per animal
fig_edmin, _ax_edmin, edmin_stats = plot_edmin_vs_peakrangenorm(edmin_df)
edmin_path = os.path.join(OUTPUT_DIR, 'edmin_vs_peakrangenorm_allAnimals.png')
fig_edmin.savefig(edmin_path, dpi=200, bbox_inches='tight')
plt.show()
print(f'Saved figure -> {edmin_path}  '
      f"(Pearson r={edmin_stats['pearson_r']:.3f}, "
      f"Spearman rho={edmin_stats['spearman_r']:.3f}, n={edmin_stats['n']})")
edmin_stats_rows = [{'animal': 'ALL', **edmin_stats}]

# One figure per animal, in that animal's colour
for label in ANIMALS:
    sub = edmin_df[edmin_df['animal'] == label]
    if sub.empty:
        continue
    fig_a, _ax_a, stats_a = plot_edmin_vs_peakrangenorm(
        sub, title=f'{label}: ED_min vs. normalised peak range')
    path_a = os.path.join(OUTPUT_DIR, f'edmin_vs_peakrangenorm_{label}.png')
    fig_a.savefig(path_a, dpi=200, bbox_inches='tight')
    plt.show()
    print(f'Saved figure -> {path_a}')
    edmin_stats_rows.append({'animal': label, **stats_a})

edmin_stats_path = os.path.join(OUTPUT_DIR, 'edmin_vs_peakrangenorm_stats.csv')
pd.DataFrame(edmin_stats_rows).to_csv(edmin_stats_path, index=False)
print(f'Saved ED_min vs. peakrangenorm correlation stats -> {edmin_stats_path}')

