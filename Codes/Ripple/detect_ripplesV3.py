"""
Sharp-Wave Ripple (SWR) Detection from Neuralynx .ncs files
Algorithm: bz_FindRipples (Buzsaki lab, buzcode repository)
Translated from MATLAB to Python — no external ripple_detection package needed.

Run with: python detect_ripples.py
"""

import os
import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.signal import firwin, fftconvolve
from scipy.ndimage import uniform_filter1d

# Route stderr into stdout so errors always appear in the same console window
sys.stderr = sys.stdout

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
NCS_FOLDER = r'C:/Runita/NMR/analysis/SWR/CSC'
OUTPUT_DIR = r'C:/Runita/NMR/analysis/SWR/ripple_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# bz_FindRipples parameters
PASSBAND            = (100, 250)   # Hz — ripple band
LOW_THRESH          = 2.0          # std dev — threshold for ripple start/end
HIGH_THRESH         = 5.0          # std dev — threshold for ripple peak
MIN_INTER_RIPPLE_MS = 30           # ms  — merge gap between events
MAX_DURATION_MS     = 100          # ms  — discard events longer than this
MIN_DURATION_MS     = 20           # ms  — discard events shorter than this


# ─────────────────────────────────────────────────────────────
# HELPER: parse ADBitVolts from Neuralynx text header
# ─────────────────────────────────────────────────────────────
def parse_ad_bit_volts(fpath):
    with open(fpath, 'rb') as f:
        header = f.read(16 * 1024).decode('latin-1', errors='replace')
    for line in header.splitlines():
        if '-ADBitVolts' in line:
            try:
                return float(line.strip().split()[-1])
            except ValueError:
                pass
    raise ValueError(f"ADBitVolts not found in header of {fpath}")


# ─────────────────────────────────────────────────────────────
# bz_FindRipples core functions
# ─────────────────────────────────────────────────────────────
def _bandpass_filter(x, lowcut, highcut, fs):
    """
    Zero-phase FIR bandpass filter via FFT convolution.
    Uses firwin (Hamming window) + fftconvolve(mode='same').
    A symmetric FIR filter applied with mode='same' is inherently zero-phase
    (the (numtaps-1)/2 group delay is absorbed by the 'same' centering).
    This avoids all IIR sosfilt/sosfiltfilt instabilities on large arrays.
    """
    if highcut >= fs / 2:
        raise ValueError(
            f"highcut ({highcut} Hz) >= Nyquist ({fs/2:.1f} Hz). "
            f"Check your sampling rate ({fs:.1f} Hz)."
        )
    # Number of taps: ~4 * fs / transition_hz, forced odd for symmetry.
    # Transition bandwidth capped at half the passband width to avoid overlap.
    trans_hz = min(lowcut * 0.5, (highcut - lowcut) * 0.3, 50.0)
    numtaps  = int(4.0 * fs / trans_hz)
    numtaps  = numtaps if numtaps % 2 == 1 else numtaps + 1
    numtaps  = max(numtaps, 101)
    taps = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs)
    return fftconvolve(x.astype(np.float64), taps, mode='same')


def _smooth_squared(x, window_len=11):
    """
    Square x, then apply a zero-phase rectangular moving average.
    Uses uniform_filter1d (O(N), fast) instead of np.convolve (O(N*M), slow).
    mode='nearest' replicates edge values, avoiding zero-boundary artifacts.
    """
    return uniform_filter1d(x ** 2, size=window_len, mode='nearest')


def _unity(A, sd=None, keep=None):
    """
    Normalize A to zero mean, unit std (MATLAB unity).
    keep: boolean mask — if given, statistics use only those samples.
    sd:   override the computed std with a previously stored value.
    """
    if keep is not None and np.any(keep):
        mean_A = np.mean(A[keep])
        std_A  = np.std(A[keep])
    else:
        mean_A = np.mean(A)
        std_A  = np.std(A)
    if sd is not None:
        std_A = sd
    # Guard against divide-by-zero on flat/silent signals
    if std_A == 0:
        std_A = 1.0
    return (A - mean_A) / std_A, std_A


def _empty_result(sd):
    return {
        'timestamps':        np.empty((0, 2)),
        'peaks':             np.array([]),
        'peak_normed_power': np.array([]),
        'stdev':             sd,
        'noise': {
            'times':             np.empty((0, 2)),
            'peaks':             np.array([]),
            'peak_normed_power': np.array([]),
        },
    }


def find_ripples(lfp, timestamps, fs,
                 passband=(100, 250),
                 low_thresh=2.0, high_thresh=5.0,
                 min_inter_ripple_ms=30, max_duration_ms=100, min_duration_ms=20,
                 sd=None, restrict=None, noise=None):
    """
    Detect hippocampal ripples using the bz_FindRipples algorithm.

    Steps
    -----
    1. Bandpass-filter the LFP to the ripple band.
    2. Square and smooth with an 11-sample rectangular window.
    3. Normalize to zero mean / unit std → Normalized Squared Signal (NSS).
    4. Threshold NSS at low_thresh to find candidate start/stop indices.
    5. Merge events whose gap is < min_inter_ripple_ms.
    6. Discard events whose peak NSS < high_thresh.
    7. Locate the trough of the filtered signal within each event (peak time).
    8. Discard events outside [min_duration_ms, max_duration_ms].
    9. Optionally reject events also present on a noise channel.

    Parameters
    ----------
    lfp        : 1-D float array — raw LFP (any voltage unit, single channel)
    timestamps : 1-D float array — sample times in seconds, same length as lfp
    fs         : float           — sampling frequency (Hz)
    passband   : (low, high)     — ripple band in Hz
    low_thresh : float           — NSS threshold for event onset/offset (× std)
    high_thresh: float           — NSS threshold for peak acceptance (× std)
    min_inter_ripple_ms : float  — events closer than this are merged (ms)
    max_duration_ms     : float  — maximum allowed ripple duration (ms)
    min_duration_ms     : float  — minimum allowed ripple duration (ms)
    sd         : float or None   — reuse a previously computed std
    restrict   : Nx2 array or None — time intervals used for normalization
    noise      : 1-D array or None — noise LFP channel; events coincident with
                                     noise above high_thresh are rejected
    """
    print(f"  [1/9] Bandpass filtering ({passband[0]}–{passband[1]} Hz)…", flush=True)
    lfp_f64 = lfp.astype(np.float64)
    print(f"        Signal: {len(lfp_f64)} samples | "
          f"range=[{lfp_f64.min():.2f}, {lfp_f64.max():.2f}] | "
          f"NaN={np.any(np.isnan(lfp_f64))} | Inf={np.any(np.isinf(lfp_f64))}", flush=True)
    print(f"        Designing FIR filter and running FFT convolution…", flush=True)
    try:
        signal = _bandpass_filter(lfp_f64, passband[0], passband[1], fs)
    except Exception:
        print("        Filtering FAILED:", flush=True)
        traceback.print_exc()
        raise
    print(f"        Filter done. Output range=[{signal.min():.6f}, {signal.max():.6f}]",
          flush=True)

    print(f"  [2/9] Squaring and smoothing…", flush=True)
    smoothed = _smooth_squared(signal, window_len=11)

    print(f"  [3/9] Normalizing → NSS…", flush=True)
    keep = None
    if restrict is not None:
        keep = np.zeros(len(timestamps), dtype=bool)
        for interval in np.atleast_2d(restrict):
            keep |= (timestamps >= interval[0]) & (timestamps <= interval[1])

    nss, sd_out = _unity(smoothed, sd=sd, keep=keep)
    print(f"        stdev = {sd_out:.6f}", flush=True)

    print(f"  [4/9] Thresholding at {low_thresh} × std…", flush=True)
    above  = (nss > low_thresh).astype(np.int8)
    starts = np.where(np.diff(above) > 0)[0]
    stops  = np.where(np.diff(above) < 0)[0]

    # Handle incomplete events at signal boundaries
    if len(stops) == len(starts) - 1:
        starts = starts[:-1]
    if len(stops) - 1 == len(starts):
        stops = stops[1:]
    if len(starts) > 0 and len(stops) > 0 and starts[0] > stops[0]:
        stops  = stops[1:]
        starts = starts[:-1]

    if len(starts) == 0 or len(stops) == 0:
        print("        No events found after thresholding.", flush=True)
        return _empty_result(sd_out)

    first_pass = np.column_stack([starts, stops])
    print(f"        {len(first_pass)} events after thresholding.", flush=True)

    print(f"  [5/9] Merging events closer than {min_inter_ripple_ms} ms…", flush=True)
    min_gap     = int(np.round(min_inter_ripple_ms / 1000.0 * fs))
    second_pass = []
    current     = first_pass[0].copy()
    for i in range(1, len(first_pass)):
        if first_pass[i, 0] - current[1] < min_gap:
            current[1] = first_pass[i, 1]
        else:
            second_pass.append(current.copy())
            current = first_pass[i].copy()
    second_pass.append(current.copy())
    second_pass = np.array(second_pass)
    print(f"        {len(second_pass)} events after merge.", flush=True)

    print(f"  [6/9] Peak thresholding at {high_thresh} × std…", flush=True)
    third_pass = []
    peak_power = []
    for seg in second_pass:
        chunk   = nss[seg[0]: seg[1] + 1]
        max_val = chunk.max()
        if max_val > high_thresh:
            third_pass.append(seg)
            peak_power.append(max_val)

    if len(third_pass) == 0:
        print("        No events survived peak thresholding.", flush=True)
        return _empty_result(sd_out)

    third_pass = np.array(third_pass)
    peak_power = np.array(peak_power)
    print(f"        {len(third_pass)} events after peak thresholding.", flush=True)

    print(f"  [7/9] Finding signal troughs (peak times)…", flush=True)
    peak_idx = np.zeros(len(third_pass), dtype=int)
    for i, seg in enumerate(third_pass):
        chunk       = signal[seg[0]: seg[1] + 1]
        peak_idx[i] = np.argmin(chunk) + seg[0]

    print(f"  [8/9] Duration filter ({min_duration_ms}–{max_duration_ms} ms)…", flush=True)
    rips = np.column_stack([
        timestamps[third_pass[:, 0]],
        timestamps[peak_idx],
        timestamps[third_pass[:, 1]],
        peak_power,
    ])
    durations = rips[:, 2] - rips[:, 0]
    keep_mask = (durations >= min_duration_ms / 1000.0) & \
                (durations <= max_duration_ms / 1000.0)
    rips = rips[keep_mask]
    print(f"        {len(rips)} events after duration filter.", flush=True)

    if len(rips) == 0:
        return _empty_result(sd_out)

    print(f"  [9/9] Noise rejection…", flush=True)
    bad = np.empty((0, 4))
    if noise is not None:
        noise_sig  = _bandpass_filter(noise, passband[0], passband[1], fs)
        noise_nss, _ = _unity(_smooth_squared(noise_sig, window_len=11), sd=sd_out)
        excluded = np.zeros(len(rips), dtype=bool)
        for i, r in enumerate(rips):
            mask = (timestamps >= r[0]) & (timestamps <= r[2])
            if np.any(noise_nss[mask] > high_thresh):
                excluded[i] = True
        bad  = rips[excluded]
        rips = rips[~excluded]
        print(f"        {len(rips)} events after noise rejection.", flush=True)
    else:
        print(f"        No noise channel provided — skipped.", flush=True)

    return {
        'timestamps':        rips[:, [0, 2]],
        'peaks':             rips[:, 1],
        'peak_normed_power': rips[:, 3],
        'stdev':             sd_out,
        'noise': {
            'times':             bad[:, [0, 2]] if len(bad) else np.empty((0, 2)),
            'peaks':             bad[:, 1]      if len(bad) else np.array([]),
            'peak_normed_power': bad[:, 3]      if len(bad) else np.array([]),
        },
    }


# ─────────────────────────────────────────────────────────────
# LOAD .ncs FILES
# ─────────────────────────────────────────────────────────────
ncs_dtype = np.dtype([
    ('timestamp',   '<u8'),
    ('sc_number',   '<u4'),
    ('cell_number', '<u4'),
    ('params',      '<u4'),
    ('samples',     '<i2', (512,)),
])

ncs_files = sorted(f for f in os.listdir(NCS_FOLDER) if f.endswith('.ncs'))
if not ncs_files:
    raise FileNotFoundError(f"No .ncs files found in {NCS_FOLDER}")

print(f"Found {len(ncs_files)} .ncs file(s): {ncs_files}", flush=True)

lfp_data, lfp_t = [], []
for fname in ncs_files:
    fpath        = os.path.join(NCS_FOLDER, fname)
    ad_bit_volts = parse_ad_bit_volts(fpath)
    raw          = np.memmap(fpath, dtype=ncs_dtype, mode='r', offset=16 * 1024)
    samples      = raw['samples'].ravel().astype(np.float64) * ad_bit_volts * 1e6
    lfp_data.append(samples)
    lfp_t.append(raw['timestamp'])
    print(f"  {fname}: {len(samples)} samples, "
          f"{raw['timestamp'][0]/1e6:.2f}–{raw['timestamp'][-1]/1e6:.2f} s, "
          f"ADBitVolts={ad_bit_volts:.6e}", flush=True)

# ─────────────────────────────────────────────────────────────
# BUILD TIME VECTOR
# ─────────────────────────────────────────────────────────────
t_records = lfp_t[0].astype(np.float64) * 1e-6
dt_record = np.median(np.diff(t_records))
FS        = 512.0 / dt_record
print(f"\nSampling frequency: {FS:.1f} Hz", flush=True)

if FS < 2 * PASSBAND[1]:
    raise ValueError(
        f"Sampling rate {FS:.1f} Hz is too low to detect {PASSBAND[1]} Hz ripples. "
        f"Need at least {2 * PASSBAND[1]} Hz."
    )

min_len  = min(len(d) for d in lfp_data)
lfp_data = [d[:min_len] for d in lfp_data]
time     = t_records[0] + np.arange(min_len) / FS

lfp_array = np.column_stack(lfp_data)
print(f"LFP array shape: {lfp_array.shape}", flush=True)
print(f"Signal range ch0: [{lfp_array[:,0].min():.2f}, {lfp_array[:,0].max():.2f}] µV  "
      f"std={lfp_array[:,0].std():.2f} µV", flush=True)

# ─────────────────────────────────────────────────────────────
# DETECT RIPPLES
# ─────────────────────────────────────────────────────────────
print(f"\nDetecting ripples (bz_FindRipples, passband {PASSBAND[0]}–{PASSBAND[1]} Hz)…",
      flush=True)
ripples = find_ripples(
    lfp_array[:, 0], time, FS,
    passband=PASSBAND,
    low_thresh=LOW_THRESH,
    high_thresh=HIGH_THRESH,
    min_inter_ripple_ms=MIN_INTER_RIPPLE_MS,
    max_duration_ms=MAX_DURATION_MS,
    min_duration_ms=MIN_DURATION_MS,
)

n_ripples = len(ripples['timestamps'])
print(f"\nDetected {n_ripples} ripples  (stdev used = {ripples['stdev']:.6f})", flush=True)

# Save CSV
df = pd.DataFrame({
    'start_time':        ripples['timestamps'][:, 0] if n_ripples else [],
    'end_time':          ripples['timestamps'][:, 1] if n_ripples else [],
    'peak_time':         ripples['peaks'],
    'peak_normed_power': ripples['peak_normed_power'],
    'duration_ms':       (ripples['timestamps'][:, 1] - ripples['timestamps'][:, 0]) * 1000
                         if n_ripples else [],
})
csv_path = os.path.join(OUTPUT_DIR, 'ripples.csv')
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}", flush=True)

# ─────────────────────────────────────────────────────────────
# FILTERED SIGNAL FOR PLOTTING
# ─────────────────────────────────────────────────────────────
filtered_ch0 = _bandpass_filter(lfp_array[:, 0], PASSBAND[0], PASSBAND[1], FS)

# ─────────────────────────────────────────────────────────────
# FIGURE 1: Full-session overview
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
fig.suptitle('Sharp-Wave Ripple Detection — bz_FindRipples algorithm', fontsize=14)

axes[0].plot(time, lfp_array[:, 0], color='steelblue', lw=0.3, label='Raw LFP (ch0)')
axes[0].set_ylabel('Amplitude (µV)')
axes[0].legend(loc='upper right', fontsize=8)

axes[1].plot(time, filtered_ch0, color='gray', lw=0.3, alpha=0.8,
             label=f'Filtered ({PASSBAND[0]}–{PASSBAND[1]} Hz)')
axes[1].set_ylabel('Amplitude (µV)')
axes[1].legend(loc='upper right', fontsize=8)

axes[2].plot(time, lfp_array[:, 0], color='steelblue', lw=0.3)
for i in range(n_ripples):
    t0, t1 = ripples['timestamps'][i]
    axes[2].axvspan(t0, t1, color='red', alpha=0.3, label='Ripple' if i == 0 else '')
    axes[2].axvline(ripples['peaks'][i], color='k', lw=0.5, alpha=0.4)
axes[2].set_ylabel('Raw LFP (µV)')
axes[2].set_xlabel('Time (s)')
if n_ripples > 0:
    axes[2].legend(loc='upper right', fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'ripples_overview.png'), dpi=150)
print("Saved: ripples_overview.png", flush=True)

# ─────────────────────────────────────────────────────────────
# FIGURE 2: Individual ripple examples (up to 12)
# ─────────────────────────────────────────────────────────────
WINDOW_S = 0.25
n_plot   = min(12, n_ripples)

if n_plot > 0:
    ncols = min(4, n_plot)
    nrows = int(np.ceil(n_plot / ncols))
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes2 = np.atleast_1d(axes2).ravel()
    fig2.suptitle(f'Individual Ripple Examples (n={n_plot})', fontsize=13)

    for i in range(n_plot):
        t0, t1   = ripples['timestamps'][i]
        t_center = (t0 + t1) / 2
        ta, tb   = t_center - WINDOW_S / 2, t_center + WINDOW_S / 2
        mask     = (time >= ta) & (time <= tb)
        t_ms     = (time[mask] - t_center) * 1000

        ax = axes2[i]
        ax.plot(t_ms, lfp_array[mask, 0],  color='steelblue',  lw=0.8, alpha=0.7, label='Raw')
        ax.plot(t_ms, filtered_ch0[mask],   color='darkorange', lw=1.2,            label='Filtered')
        ax.axvspan((t0 - t_center) * 1000, (t1 - t_center) * 1000,
                   color='red', alpha=0.15, label='Ripple')
        ax.axvline((ripples['peaks'][i] - t_center) * 1000,
                   color='k', lw=0.8, ls='--', label='Peak')
        ax.set_title(f'Ripple {i+1}  ({(t1-t0)*1000:.0f} ms)', fontsize=9)
        ax.set_xlabel('Time (ms)')
        if i % ncols == 0:
            ax.set_ylabel('Amplitude (µV)')
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(n_plot, len(axes2)):
        axes2[j].set_visible(False)

    plt.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, 'ripple_examples.png'), dpi=150)
    print("Saved: ripple_examples.png", flush=True)

# ─────────────────────────────────────────────────────────────
# Protect the print statement in case no ripples are found
# ─────────────────────────────────────────────────────────────
if n_ripples > 0:
    print(f"first ripple: t0={ripples['timestamps'][0,0]:.3f}, t1={ripples['timestamps'][0,1]:.3f}")
else:
    print("No ripples detected. Skipping snippet generation.")

# ─────────────────────────────────────────────────────────────
# FIGURE: Individual ripple snippets (one PNG per event, ±100 ms)
# ─────────────────────────────────────────────────────────────
SNIPPET_PAD_S = 0.100   # 100 ms padding before / after detected event
snippets_dir  = os.path.join(OUTPUT_DIR, 'ripple_snippets')
os.makedirs(snippets_dir, exist_ok=True)

saved_count = 0

# KEY FIX: Create the figure and axis ONCE, outside the loop using pyplot
fig_s, ax_s = plt.subplots(figsize=(6, 3))

for i in range(n_ripples):
    try:
        t0, t1 = ripples['timestamps'][i]
        ta     = max(t0 - SNIPPET_PAD_S, time[0])   # clamp to signal bounds
        tb     = min(t1 + SNIPPET_PAD_S, time[-1])

        mask = (time >= ta) & (time <= tb)
        
        if not np.any(mask):
            print(f"  [snippet] Ripple {i+1}: mask is empty, skipping.", flush=True)
            continue

        n_samples = np.sum(mask)
        if n_samples < 5:
            print(f"  [snippet] Ripple {i+1}: only {n_samples} samples in window, skipping.", flush=True)
            continue

        t_ms = (time[mask] - t0) * 1000  # 0 ms = ripple start

        # Clear the axis for the new data instead of creating a new figure
        ax_s.clear()

        ax_s.plot(t_ms, lfp_array[mask, 0],  color='steelblue',  lw=0.8, alpha=0.7, label='Raw LFP')
        ax_s.plot(t_ms, filtered_ch0[mask],   color='darkorange', lw=1.2,            label='Filtered')
        ax_s.axvspan(0.0, (t1 - t0) * 1000, color='red', alpha=0.15, label='Ripple')
        ax_s.axvline((ripples['peaks'][i] - t0) * 1000, color='k', lw=0.8, ls='--', label='Peak')
        ax_s.axvline((ta - t0) * 1000, color='gray', lw=0.5, ls=':')   # actual window edge
        ax_s.axvline((tb - t0) * 1000, color='gray', lw=0.5, ls=':')

        ax_s.set_xlabel('Time relative to ripple start (ms)')
        ax_s.set_ylabel('Amplitude (µV)')
        ax_s.set_title(
            f'Ripple {i+1}  |  t={t0:.3f} s  |  dur={(t1-t0)*1000:.0f} ms'
            f'  |  peak NSS={ripples["peak_normed_power"][i]:.1f}',
            fontsize=9,
        )
        ax_s.legend(fontsize=7, loc='upper right')
        
        # tight_layout now works safely because plt.subplots() manages the renderer
        fig_s.tight_layout()

        snippet_path = os.path.join(snippets_dir, f'ripple_{i+1:04d}.png')
        fig_s.savefig(snippet_path, dpi=150)
        saved_count += 1

    except Exception as e:
        print(f"  [snippet] Ripple {i+1}: ERROR — {e}", flush=True)
        traceback.print_exc()

# Close the single figure safely after the loop is done
plt.close(fig_s)

print(f"Saved {saved_count}/{n_ripples} ripple snippets → {snippets_dir}", flush=True)

# ─────────────────────────────────────────────────────────────
# FIGURE 3: Summary statistics
# ─────────────────────────────────────────────────────────────
if n_ripples > 0:
    durations_ms = (ripples['timestamps'][:, 1] - ripples['timestamps'][:, 0]) * 1000
    session_len  = time[-1] - time[0]

    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))
    fig3.suptitle('Ripple Statistics — bz_FindRipples', fontsize=13)

    axes3[0].hist(durations_ms, bins=20, color='steelblue', edgecolor='white')
    axes3[0].set_xlabel('Duration (ms)')
    axes3[0].set_ylabel('Count')
    axes3[0].set_title(f'Duration\nmean={durations_ms.mean():.1f} ms, '
                       f'median={np.median(durations_ms):.1f} ms')

    axes3[1].hist(ripples['peak_normed_power'], bins=20, color='darkorange', edgecolor='white')
    axes3[1].set_xlabel('Peak normalized power (× std)')
    axes3[1].set_ylabel('Count')
    axes3[1].set_title('Peak NSS distribution')

    axes3[2].bar(['Ripples'], [n_ripples / session_len], color='steelblue')
    axes3[2].set_ylabel('Rate (events / s)')
    axes3[2].set_title(f'Session: {session_len:.1f} s  |  N = {n_ripples}')

    plt.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, 'ripple_stats.png'), dpi=150)
    print("Saved: ripple_stats.png", flush=True)

plt.show()
print("\nDone.", flush=True)
