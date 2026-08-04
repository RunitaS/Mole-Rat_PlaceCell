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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from scipy import signal
from fooof import FOOOF

#%% Settings

# Root directories — one per animal.
# Add/remove/rename animals ONLY here — plot colors, filenames, and legends
# below are all derived automatically from this dict's keys.
ANIMALS = {
    'Fa8477': r'X:/NMR_group_data/Runita/Data/Ephys_Data/AllSortedData/Tetrode/Fa8477',
    #'FaDDE42': r'C:/Runita/NMR/analysis/SurgeryPaperSpikeLFP/LFP/Main/DDE42',
    'Fa23BD': r'C:/Runita/NMR/analysis/AllSort_Results/LFP/23BDTest',
    'Fa1059' : r'C:/Runita/NMR/analysis/AllSort_Results/LFP/1059Test',
}

OUTPUT_DIR = r'C:/Runita/NMR/analysis/AllSort_Results/LFP'  # saved plots go here

# Color palette cycled across however many animals are in ANIMALS (repeats past 8).
_PALETTE = [
    '#1A56DB',  # blue
    '#4DAF4A',  # green
    '#E41A1C',  # red
    '#FF7F0E',  # orange
    '#9467BD',  # purple
    '#17BECF',  # cyan
    '#BCBD22',  # olive
    '#E377C2',  # pink
]


def _lighten(hex_color, amount=0.55):
    """Blend a hex color toward white, for use as a shaded fill color."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


# {label: (line_color, fill_color)} — auto-built from ANIMALS, one entry per animal.
STYLES = {
    label: (_PALETTE[i % len(_PALETTE)], _lighten(_PALETTE[i % len(_PALETTE)]))
    for i, label in enumerate(ANIMALS)
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
    psds     = []
    freq_vec = None

    for fpath in ncs_files:
        rel = os.path.relpath(fpath, folder)
        try:
            data = np.memmap(fpath, dtype=ncs_dtype, mode='r', offset=16*1024)

            # Conversion happens BEFORE downsampling (Corrected to 1e6 for µV)
            lfp = np.concatenate(data['samples']).astype(np.float64) * ADBitVolts * 1e6
            lfp = np.asarray(signal.resample_poly(lfp, fs_down, fs))
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
    return freq_vec, mean_psd, sem_psd, psds.shape[0]


#%% Process each animal (LFP snippet figures currently disabled, see process_animal)

os.makedirs(OUTPUT_DIR, exist_ok=True)

results = {}
for label, folder in ANIMALS.items():
    print(f'=== {label} ===')
    results[label] = process_animal(label, folder)

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

