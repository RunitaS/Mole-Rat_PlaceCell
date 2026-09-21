"""Theta-phase locking test for hippocampal single units.

For every session folder under ROOT_DIR (a folder holding one or more .ntt
spike files and exactly one .ncs LFP file), the .ncs is band-pass filtered in
the theta band and its instantaneous phase is estimated with the Hilbert
transform. Each spike in each .ntt is assigned the LFP phase at its
timestamp (circular interpolation), binned every PHASE_BIN_SIZE_DEG degrees
for a polar-histogram plot, and tested for non-uniformity with a Rayleigh
test on the raw (unbinned) per-spike phases.

In addition (ported from this lab's AditiPrecessionUtils.py), for every unit
the full phase-locking battery -- MRL / preferred phase / Rayleigh test (the
polar-plot statistics), phase peak/valley (findPhaseValley), and the Theta
Modulation Index with its shuffle significance (calcTMI) -- is run twice,
once per spike-phase-estimation method, and reported as two independent
sets of results/plots:
  - "Hilbert": phase = instantaneous Hilbert-transform phase of the
    theta-filtered LFP (as before);
  - "Interp": phase = linear interpolation between consecutive theta-
    filtered-LFP peaks (getPhase), the phase basis AditiPrecessionUtils uses.
Each method's TMI significance is assessed via its own shuffling test: 1000
surrogate spike trains are built by circularly time-shifting the real spike
train (shift magnitude drawn randomly, but always >= 20 s, so real
theta-phase alignment is destroyed while the spike train's own temporal
structure is preserved), each is re-scored for TMI using that method's own
phase assignment, and the real TMI is deemed significant if it exceeds the
shuffle-derived null distribution at alpha = 0.05.

The intrinsic oscillation frequency (precessionFreq) and the ISI /
autocorrelogram do not depend on LFP phase, so they are computed once per
unit and shared by both methods' plots.

Plots are saved to "polar plots" subfolders next to each session's .ntt
files -- two figures per unit (one per phase method, suffixed _Hilbert /
_Interp), each with that method's theta polar plot, the ISI histogram, and
the spike-train autocorrelogram side by side (layout follows this lab's
clusterprojection_wvfrms_smoothACGv3.py). A single summary table (one row
per unit, with _Hilbert/_Interp-suffixed columns for every phase-locking
metric) is written to one sheet of OUTPUT_EXCEL.

The tracking .csv in each session folder is not used here -- theta-
modulation testing only needs spike times and LFP phase, not position.

NCS/NTT reading uses np.memmap against the raw Neuralynx binary record
layout (16 KB text header, then fixed-size binary records) rather than the
MATLAB Nlx2Mat*/mex toolbox, so this has no external Neuralynx dependency.
"""

import os

import numpy as np
import pandas as pd
from scipy import signal

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# %% ==================== Configuration ====================

ROOT_DIR     = r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True'  # EDIT ME
OUTPUT_EXCEL = os.path.join(ROOT_DIR, 'ThetaModulation_Analyzed.xlsx')

TARGET_FS           = 500          # Hz, LFP is polyphase-resampled to this rate before filtering
THETA_BAND           = (3.0, 7.0)  # Hz
FILTER_ORDER         = 4            # Butterworth order (zero-phase filtfilt -> effectively 8th order)
PHASE_BIN_SIZE_DEG   = 6            # degrees per polar-histogram bin (360 must be divisible by this)
ALPHA                = 0.05         # Rayleigh-test / TMI-shuffle significance threshold
MIN_SPIKES           = 8            # min spikes (with LFP coverage) required to run stats/plot
GAP_FACTOR           = 2.5          # timestamp gap > GAP_FACTOR * median dt starts a new continuous segment

N_TMI_SHUFFLES       = 1000         # number of surrogate spike trains for the TMI shuffle test
TMI_SHUFFLE_MIN_SHIFT_SEC = 20.0    # minimum circular time-shift magnitude ("shuffle by 20 sec")
RANDOM_SEED          = 0            # seed for the TMI shuffle test's RNG, for reproducibility

NLX_HEADER_BYTES = 16 * 1024  # standard Neuralynx (Cheetah) ASCII header size

# Neuralynx .ncs record: 8-byte timestamp (us) + channel/sample-rate/valid-count
# (4 bytes each) + 512 int16 samples. Byte layout matches the lab's existing
# FOOOF.py ncs_dtype; fields are renamed here to their real meaning since
# num_valid_samples/sample_freq are needed to correctly handle the last
# record of a file and per-record sample rate.
ncs_dtype = np.dtype([
    ('timestamp',          '<u8'),
    ('channel_number',     '<u4'),
    ('sample_freq',        '<u4'),
    ('num_valid_samples',  '<u4'),
    ('samples',            '<i2', (512,)),
])

# Neuralynx .ntt record: 8-byte timestamp (us) + sc_number/cell_number (4
# bytes each) + 8 uint32 feature params + 32-sample x 4-channel int16
# waveform. Matches the lab's existing ntt_dtype (PlaceCell pipeline).
ntt_dtype = np.dtype([
    ('timestamp',   '<u8'),
    ('sc_number',   '<u4'),
    ('cell_number', '<u4'),
    ('params',      '<u4', (8,)),
    ('waveforms',   '<i2', (32, 4)),
])


# %% ==================== NCS / NTT readers ====================

def load_ncs(fpath):
    """Read a Neuralynx .ncs file into a flat per-sample signal + timestamp trace.

    Unlike a naive `np.concatenate(data['samples'])`, this respects
    `num_valid_samples` (the last record of a file is often partially
    filled) and each record's own `sample_freq`, and returns absolute
    per-sample UNIX timestamps (us) rather than assuming records are
    gap-free -- gap handling happens later in `compute_theta_phase`.

    Returns
    -------
    sig   : ndarray, raw AD counts (float64). Phase estimation is scale
            invariant, so no ADBitVolts conversion is applied.
    ts_us : ndarray, absolute per-sample UNIX timestamp (us), same length as sig.
    fs    : float, nominal sampling rate (Hz), median across records.
    """
    raw = np.memmap(fpath, dtype=ncs_dtype, mode='r', offset=NLX_HEADER_BYTES)
    ts_records = raw['timestamp'].astype(np.float64)
    fs_records = raw['sample_freq'].astype(np.float64)
    n_valid = raw['num_valid_samples'].astype(np.int64)
    samples = raw['samples']
    block = samples.shape[1]

    fs = float(np.median(fs_records))
    dt_us_per_record = 1e6 / fs_records
    ts_full = ts_records[:, None] + np.arange(block)[None, :] * dt_us_per_record[:, None]
    valid_mask = np.arange(block)[None, :] < n_valid[:, None]

    ts_us = ts_full[valid_mask]
    sig = np.asarray(samples, dtype=np.float64)[valid_mask]
    return sig, ts_us, fs


def load_ntt_spike_times(fpath):
    """Read spike timestamps (us) and CellNumbers from a Neuralynx .ntt file."""
    raw = np.memmap(fpath, dtype=ntt_dtype, mode='r', offset=NLX_HEADER_BYTES)
    spike_ts_us = raw['timestamp'].astype(np.float64)
    cell_numbers = raw['cell_number'].astype(np.int64)
    return spike_ts_us, cell_numbers


# %% ==================== Theta phase estimation (Hilbert) ====================

def find_contiguous_segments(ts_us, gap_factor=GAP_FACTOR):
    """Split a timestamp vector into (start_idx, end_idx) runs, breaking at
    any gap bigger than `gap_factor` times the median sample spacing.

    Used on the raw LFP (to keep filtering/Hilbert away from real Neuralynx
    recording gaps), on the concatenated per-segment phase trace (to keep
    spike-phase interpolation from bridging those same gaps), and on the
    concatenated per-segment theta-peak-time trace (same reason, for the
    interpolation-based phase method).
    """
    if len(ts_us) < 2:
        return [(0, len(ts_us) - 1)] if len(ts_us) else []
    dt = np.diff(ts_us)
    nominal = np.median(dt)
    gap_idx = np.where(dt > gap_factor * nominal)[0]
    starts = np.r_[0, gap_idx + 1]
    ends = np.r_[gap_idx, len(ts_us) - 1]
    return list(zip(starts.tolist(), ends.tolist()))


def _filtered_theta_segments(sig, ts_us, fs, theta_band=THETA_BAND, target_fs=TARGET_FS):
    """Per contiguous recording segment: polyphase-resample to target_fs and
    zero-phase Butterworth band-pass filter in theta_band.

    Shared by both the Hilbert-phase method (`compute_theta_phase`) and the
    peak-interpolation method (`compute_theta_peak_times`), so both phase
    estimates are derived from the same filtered LFP.

    Returns a list of (seg_filt, seg_ts_dec) tuples, one per usable segment.
    """
    nyq = target_fs / 2.0
    if theta_band[1] >= nyq:
        raise ValueError(
            f'Theta band upper edge ({theta_band[1]} Hz) exceeds Nyquist '
            f'({nyq} Hz) at target_fs={target_fs}. Increase target_fs.')
    sos = signal.butter(FILTER_ORDER, [theta_band[0] / nyq, theta_band[1] / nyq],
                         btype='bandpass', output='sos')

    segments = []
    for start, end in find_contiguous_segments(ts_us):
        seg_sig = sig[start:end + 1]
        seg_ts = ts_us[start:end + 1]
        min_len = max(3 * int(fs / theta_band[0]), 100)
        if len(seg_sig) < min_len:
            continue  # segment too short relative to a theta cycle to filter meaningfully

        seg_dec = signal.resample_poly(seg_sig, target_fs, fs)
        n_dec = len(seg_dec)
        if n_dec < 3 * (target_fs / theta_band[0]):
            continue
        seg_ts_dec = seg_ts[0] + np.arange(n_dec) * (1e6 / target_fs)

        seg_filt = signal.sosfiltfilt(sos, seg_dec)
        segments.append((seg_filt, seg_ts_dec))

    return segments


def compute_theta_phase(sig, ts_us, fs, theta_band=THETA_BAND, target_fs=TARGET_FS):
    """Per-segment Hilbert-transform instantaneous theta phase.

    Returns
    -------
    phase_deg    : ndarray, instantaneous theta phase in [0, 360) degrees.
    phase_ts_us  : ndarray, matching absolute UNIX timestamp (us) for each
                   phase sample (same length as phase_deg).
    """
    phase_chunks = []
    ts_chunks = []
    for seg_filt, seg_ts_dec in _filtered_theta_segments(sig, ts_us, fs, theta_band, target_fs):
        analytic = signal.hilbert(seg_filt)
        seg_phase_deg = np.degrees(np.angle(analytic)) % 360
        phase_chunks.append(seg_phase_deg)
        ts_chunks.append(seg_ts_dec)

    if not phase_chunks:
        return np.array([]), np.array([])
    return np.concatenate(phase_chunks), np.concatenate(ts_chunks)


def assign_spike_phase(spike_ts_us, phase_ts_us, phase_deg):
    """Assign each spike the interpolated LFP theta phase at its timestamp.

    Interpolates cos/sin of phase separately (not the angle itself) so the
    0/360-degree wraparound doesn't corrupt the interpolated value, then
    re-derives the angle. Spikes falling inside an LFP recording gap (i.e.
    outside every contiguous phase segment) are excluded.

    Returns
    -------
    spike_phase_deg : ndarray (len = len(spike_ts_us)), NaN where excluded.
    keep            : bool ndarray, True where a valid phase was assigned.
    """
    n = len(spike_ts_us)
    spike_phase_deg = np.full(n, np.nan)
    keep = np.zeros(n, dtype=bool)
    if len(phase_ts_us) == 0:
        return spike_phase_deg, keep

    cos_phase = np.cos(np.radians(phase_deg))
    sin_phase = np.sin(np.radians(phase_deg))

    for start, end in find_contiguous_segments(phase_ts_us):
        ts_seg = phase_ts_us[start:end + 1]
        if len(ts_seg) < 2:
            continue
        in_range = (spike_ts_us >= ts_seg[0]) & (spike_ts_us <= ts_seg[-1])
        if not np.any(in_range):
            continue
        cos_i = np.interp(spike_ts_us[in_range], ts_seg, cos_phase[start:end + 1])
        sin_i = np.interp(spike_ts_us[in_range], ts_seg, sin_phase[start:end + 1])
        spike_phase_deg[in_range] = np.degrees(np.arctan2(sin_i, cos_i)) % 360
        keep[in_range] = True

    return spike_phase_deg, keep


# %% ============ Theta phase estimation (peak-interpolation method) ============
# Port of AditiPrecessionUtils.getPhase: rather than the Hilbert-transform
# instantaneous phase used above, each spike's phase is linearly interpolated
# between the two consecutive peaks of the theta-filtered LFP that bracket it
# (0 deg at a peak, 360 deg at the next peak). This is the phase basis Aditi's
# pipeline uses for calcTMI / findPhaseValley / precessionFreq-style metrics,
# so it is kept separate from the Hilbert phase used for the Rayleigh/MRL
# polar-plot statistics above.

def compute_theta_peak_times(sig, ts_us, fs, theta_band=THETA_BAND, target_fs=TARGET_FS):
    """Timestamps (us) of every peak of the theta-filtered LFP, per segment."""
    peak_chunks = []
    for seg_filt, seg_ts_dec in _filtered_theta_segments(sig, ts_us, fs, theta_band, target_fs):
        idx, _ = signal.find_peaks(seg_filt)
        if len(idx):
            peak_chunks.append(seg_ts_dec[idx])

    if not peak_chunks:
        return np.array([])
    return np.concatenate(peak_chunks)


def assign_spike_interp_phase(spike_ts_us, peak_ts_us):
    """Vectorized AditiPrecessionUtils.getPhase: phase by linear interpolation
    between the theta peak immediately before and immediately after each spike.

    Returns
    -------
    phase_deg : ndarray (len = len(spike_ts_us)), NaN outside peak coverage.
    keep      : bool ndarray, True where a valid phase was assigned.
    """
    n = len(spike_ts_us)
    phase_deg = np.full(n, np.nan)
    keep = np.zeros(n, dtype=bool)
    if len(peak_ts_us) < 2:
        return phase_deg, keep

    for start, end in find_contiguous_segments(peak_ts_us):
        seg_peaks = peak_ts_us[start:end + 1]
        if len(seg_peaks) < 2:
            continue
        in_range = (spike_ts_us >= seg_peaks[0]) & (spike_ts_us <= seg_peaks[-1])
        if not np.any(in_range):
            continue

        spk = spike_ts_us[in_range]
        # smallest peak >= spike (equal-inclusive) and largest peak <= spike (equal-inclusive)
        after_idx = np.clip(np.searchsorted(seg_peaks, spk, side='left'), 0, len(seg_peaks) - 1)
        before_idx = np.clip(np.searchsorted(seg_peaks, spk, side='right') - 1, 0, len(seg_peaks) - 1)
        pk_after = seg_peaks[after_idx]
        pk_before = seg_peaks[before_idx]

        interpk_int = pk_after - pk_before
        local_phase = np.zeros(len(spk))
        nonzero = interpk_int > 0
        local_phase[nonzero] = ((spk[nonzero] - pk_before[nonzero]) / interpk_int[nonzero]) * 360.0
        # spikes exactly coincident with a peak (interpk_int == 0) get phase 0, by convention

        phase_deg[in_range] = local_phase
        keep[in_range] = True

    return phase_deg, keep


# %% ==================== Gaussian smoothing (AditiPrecessionUtils.gaussSmooth) ====================

def _gauss_kernel(n=11, sigma=1.0):
    """AditiPrecessionUtils.gauss: an n-point 1D Gaussian smoothing kernel."""
    r = np.arange(-(n // 2), n // 2 + 1)
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(r.astype(float) ** 2) / (2 * sigma ** 2))


def _gauss_smooth_1d(y, n=11, sigma=1.0):
    """AditiPrecessionUtils.gaussSmooth: edge-padded Gaussian convolution
    smoothing of a 1D signal (used for the ACG, TMI histogram, and intrinsic-
    frequency histogram, matching the original lab pipeline's smoothing)."""
    g = _gauss_kernel(n, sigma)
    y = np.asarray(y, dtype=float)
    padded = np.pad(y, (len(g),), mode='edge')
    smoothed = np.convolve(padded, g, 'same')
    return smoothed[len(g):len(smoothed) - len(g)]


# %% ==================== Circular statistics (Rayleigh test) ====================
# Direct ports of this lab's circ_r.m / circ_mean.m / circ_rtest.m (unweighted,
# unbinned case) so p-values/MRLs match the MATLAB pipeline exactly. The
# Rayleigh test is run on raw per-spike phases, not the binned histogram
# counts, for maximal statistical power; binning (PHASE_BIN_SIZE_DEG) is only
# used for the polar-plot visualization.

def circ_r(alpha_rad):
    return float(np.abs(np.mean(np.exp(1j * alpha_rad))))


def circ_mean(alpha_rad):
    return float(np.angle(np.mean(np.exp(1j * alpha_rad))))


def circ_rtest(alpha_rad):
    n = len(alpha_rad)
    r = circ_r(alpha_rad)
    R = n * r
    z = R ** 2 / n
    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))
    return float(pval), float(z)


# %% ==================== TMI, phase peak/valley (AditiPrecessionUtils) ====================

def calc_tmi(spike_phase_deg, num_cycles=5, bin_width_deg=36, return_hist=False):
    """Port of AditiPrecessionUtils.calcTMI: tiles spike phases across
    num_cycles repeated 360-degree cycles, bins/smooths the histogram, keeps
    the middle two cycles, max-normalizes, and defines
    TMI = 1 - trough of the normalized histogram (near 1 = strongly
    theta-modulated firing, near 0 = phase-uniform firing).
    """
    phases = spike_phase_deg[~np.isnan(spike_phase_deg)]
    tiled = np.concatenate([phases + j * 360 for j in range(num_cycles)])
    edges = np.arange(0, num_cycles * 360, bin_width_deg)
    counts, edges = np.histogram(tiled, bins=edges)

    smcounts = _gauss_smooth_1d(counts.astype(float), n=7, sigma=0.5)
    edges2 = edges[:-1]
    mask = (edges2 >= 360) & (edges2 <= 1080)
    edges2 = edges2[mask] - 360
    counts2 = smcounts[mask]
    normcounts = counts2 / np.max(counts2)
    tmi = float(1 - np.min(normcounts))

    if return_hist:
        bincentres = edges2 + bin_width_deg / 2
        return tmi, edges2, bincentres, normcounts
    return tmi


def find_phase_peak_valley(spike_phase_deg):
    """Phase peak and valley (AditiPrecessionUtils.findPhaseValley), located
    on the same smoothed/normalized phase histogram calc_tmi uses, so the
    reported peak/valley phases are consistent with the TMI score."""
    tmi, _edges2, bincentres, normcounts = calc_tmi(spike_phase_deg, return_hist=True)
    peak_phase_deg = float(bincentres[np.argmax(normcounts)] % 360)
    valley_phase_deg = float(bincentres[np.argmin(normcounts)] % 360)
    return peak_phase_deg, valley_phase_deg, tmi


# %% ==================== ISI, autocorrelogram, intrinsic frequency ====================

def compute_isi_ms(spike_ts_us):
    """Inter-spike intervals (ms) and their coefficient of variation
    (AditiPrecessionUtils.getISI)."""
    ts_sorted = np.sort(spike_ts_us)
    isi_ms = np.diff(ts_sorted) / 1000.0  # us -> ms
    isi_ms = isi_ms[isi_ms > 0]
    cv = float(np.std(isi_ms) / np.mean(isi_ms)) if len(isi_ms) else np.nan
    return isi_ms, cv


def _lag_histogram_ms(spike_ts_us, max_lag_ms, bin_size_ms):
    """Symmetric pairwise time-lag histogram (ms) within +-max_lag_ms of every
    spike, the shared building block behind both TempAutocorr and
    precessionFreq in AditiPrecessionUtils (vectorized here via searchsorted
    rather than their O(n^2) double loop)."""
    ts_ms = np.sort(spike_ts_us) / 1000.0
    edges = np.arange(-max_lag_ms, max_lag_ms + bin_size_ms, bin_size_ms)
    lags = []
    for t in ts_ms:
        lo = np.searchsorted(ts_ms, t - max_lag_ms, side='left')
        hi = np.searchsorted(ts_ms, t + max_lag_ms, side='right')
        diff = ts_ms[lo:hi] - t
        lags.append(diff[diff != 0])
    all_lags = np.concatenate(lags) if lags else np.array([])
    counts, _ = np.histogram(all_lags, bins=edges)
    return edges, counts.astype(float)


def temporal_autocorr_ms(spike_ts_us, max_lag_ms=1000.0, bin_size_ms=20.0):
    """AditiPrecessionUtils.TempAutocorr: +-1000 ms, 20 ms bins, Gaussian smoothed."""
    edges, counts = _lag_histogram_ms(spike_ts_us, max_lag_ms, bin_size_ms)
    smoothed = _gauss_smooth_1d(counts, n=7, sigma=1.0)
    return edges, counts, smoothed


def intrinsic_frequency_hz(spike_ts_us, zero_halfwidth_ms=50.0, max_lag_ms=500.0, bin_size_ms=2.0):
    """AditiPrecessionUtils.precessionFreq: fine-binned (+-500 ms, 2 ms bins)
    autocorrelogram with the central +-50 ms zero-lag peak removed, Gaussian
    smoothed, and the cell's intrinsic oscillation frequency estimated as
    1000 / (lag of the first side peak in ms)."""
    edges, counts = _lag_histogram_ms(spike_ts_us, max_lag_ms, bin_size_ms)
    bin_centres = edges[:-1] + bin_size_ms / 2
    counts[np.abs(bin_centres) < zero_halfwidth_ms] = 0.0
    smoothed = _gauss_smooth_1d(counts, n=11, sigma=2.0)

    peak_idx, _ = signal.find_peaks(smoothed)
    if len(peak_idx) == 0:
        return np.nan
    first_peak_lag_ms = float(np.min(np.abs(bin_centres[peak_idx])))
    if first_peak_lag_ms == 0:
        return np.nan
    return 1000.0 / first_peak_lag_ms


# %% ==================== TMI significance via spike-train shuffling ====================

def shuffle_tmi_significance(spike_ts_us, epoch_ts_us, observed_tmi, phase_assign_fn,
                              n_shuffles=N_TMI_SHUFFLES,
                              min_shift_sec=TMI_SHUFFLE_MIN_SHIFT_SEC,
                              rng=None):
    """Null distribution for the TMI via circular time-shifting of the spike
    train ("shuffle by 20 sec"): each of n_shuffles surrogates circularly
    shifts every spike time by the same random offset (magnitude always >=
    min_shift_sec, direction/exact magnitude random) within the LFP epoch
    [epoch_ts_us[0], epoch_ts_us[-1]], re-derives phase for the shifted
    spikes via phase_assign_fn -- bound to whichever method's phase
    trace/peak times is being tested (assign_spike_phase for Hilbert,
    assign_spike_interp_phase for the interpolation method) -- and re-scores
    TMI. Shifting (rather than e.g. randomizing each spike phase
    independently) preserves the spike train's own temporal structure (ISIs,
    bursting) while destroying its true alignment to theta phase -- the
    standard control for phase-locking significance.

    Returns
    -------
    pval           : fraction of shuffle TMIs >= observed_tmi (add-one
                     smoothed so p is never exactly 0); NaN if the epoch is
                     too short to shuffle or too few shuffles yielded enough
                     spikes.
    shuffle_tmis   : ndarray of the valid (non-NaN) shuffle TMI scores.
    """
    rng = rng if rng is not None else np.random.default_rng(RANDOM_SEED)
    if len(epoch_ts_us) < 2:
        return np.nan, np.array([])

    epoch_start = epoch_ts_us[0]
    epoch_dur_us = epoch_ts_us[-1] - epoch_ts_us[0]
    min_shift_us = min_shift_sec * 1e6
    if epoch_dur_us <= 2 * min_shift_us:
        return np.nan, np.array([])

    shuffle_tmis = np.full(n_shuffles, np.nan)
    for i in range(n_shuffles):
        shift_us = rng.uniform(min_shift_us, epoch_dur_us - min_shift_us)
        shifted_ts_us = epoch_start + (spike_ts_us - epoch_start + shift_us) % epoch_dur_us
        shuf_phase_deg, shuf_keep = phase_assign_fn(shifted_ts_us)
        shuf_phase_deg = shuf_phase_deg[shuf_keep]
        if len(shuf_phase_deg) >= MIN_SPIKES:
            shuffle_tmis[i] = calc_tmi(shuf_phase_deg)

    valid = shuffle_tmis[~np.isnan(shuffle_tmis)]
    if len(valid) == 0:
        return np.nan, valid
    pval = float((np.sum(valid >= observed_tmi) + 1) / (len(valid) + 1))
    return pval, valid


# %% ==================== Full per-method phase-locking battery ====================

def compute_phase_locking_metrics(spike_ts_us, phase_assign_fn, epoch_ts_us, rng):
    """Run the complete phase-locking battery for one phase-estimation method:
    MRL / preferred phase / Rayleigh test (the polar-plot statistics), phase
    peak/valley, and TMI with its shuffle significance.

    Parameters
    ----------
    phase_assign_fn : maps an arbitrary spike-time array to (phase_deg, keep)
                       for this method -- assign_spike_phase for Hilbert,
                       assign_spike_interp_phase for the interpolation method.
    epoch_ts_us      : the LFP epoch (phase_ts_us or peak_ts_us) that method's
                       phase is defined over, used by the shuffle test.

    Returns
    -------
    metrics    : dict of the battery's summary values (NaN/False if too few
                 spikes had phase coverage).
    phase_deg  : ndarray of this method's per-spike phase (deg), phase-
                 covered spikes only -- for plotting the polar histogram.
    """
    phase_deg, keep = phase_assign_fn(spike_ts_us)
    spike_ts_kept = spike_ts_us[keep]
    phase_deg = phase_deg[keep]

    metrics = dict(SpikesUsed=len(phase_deg), SpikesExcluded_NoLFP=int(np.sum(~keep)))

    if len(phase_deg) < MIN_SPIKES:
        metrics.update(MRL=np.nan, PreferredPhase_deg=np.nan, Rayleigh_p=np.nan,
                        SignificantThetaModulation=False, PhasePeak_deg=np.nan,
                        PhaseValley_deg=np.nan, TMI=np.nan, TMI_shuffle_p=np.nan,
                        TMI_Significant=False)
        return metrics, phase_deg

    alpha_rad = np.radians(phase_deg)
    mrl = circ_r(alpha_rad)
    pref_phase_deg = np.degrees(circ_mean(alpha_rad)) % 360
    rayleigh_p, _ = circ_rtest(alpha_rad)
    is_sig = rayleigh_p < ALPHA

    peak_phase_deg, valley_phase_deg, tmi = find_phase_peak_valley(phase_deg)
    tmi_pval, _shuffle_dist = shuffle_tmi_significance(
        spike_ts_kept, epoch_ts_us, tmi, phase_assign_fn, rng=rng)
    tmi_sig = bool(np.isfinite(tmi_pval) and tmi_pval < ALPHA)

    metrics.update(MRL=mrl, PreferredPhase_deg=pref_phase_deg, Rayleigh_p=rayleigh_p,
                    SignificantThetaModulation=bool(is_sig), PhasePeak_deg=peak_phase_deg,
                    PhaseValley_deg=valley_phase_deg, TMI=tmi, TMI_shuffle_p=tmi_pval,
                    TMI_Significant=tmi_sig)
    return metrics, phase_deg


# %% ==================== Plotting ====================

def plot_unit_summary(spike_phase_deg, mrl, pref_phase_deg, rayleigh_p, is_sig, bin_size_deg,
                       isi_ms, cv_isi, acg_edges, acg_counts, acg_smoothed,
                       intrinsic_freq_hz, tmi, tmi_pval, tmi_sig,
                       peak_phase_deg, valley_phase_deg,
                       title_str, save_path):
    """One figure per unit: theta polar plot | ISI histogram | autocorrelogram,
    side by side (layout follows clusterprojection_wvfrms_smoothACGv3.py)."""
    fig = Figure(figsize=(15, 5), dpi=150)
    FigureCanvasAgg(fig)

    # --- polar plot (Hilbert-phase MRL / Rayleigh test) ---
    ax_polar = fig.add_subplot(1, 3, 1, projection='polar')
    edges_deg = np.arange(0, 360 + bin_size_deg, bin_size_deg)
    edges_rad = np.radians(edges_deg)
    counts, _ = np.histogram(np.radians(spike_phase_deg), bins=edges_rad)
    width = np.radians(bin_size_deg)

    if is_sig:
        face_color, vec_color, vec_width = '#B3B3B3', 'red', 4
    else:
        face_color, vec_color, vec_width = '#D9D9D9', '#4D4D4D', 2

    ax_polar.bar(edges_rad[:-1], counts, width=width, align='edge',
                 facecolor=face_color, edgecolor='black', linewidth=0.5)
    max_count = counts.max() if counts.max() > 0 else 1
    ax_polar.plot([0, np.radians(pref_phase_deg)], [0, mrl * max_count],
                  color=vec_color, linewidth=vec_width)
    ax_polar.set_theta_zero_location('E')
    ax_polar.set_theta_direction(1)
    ax_polar.set_xticks(np.radians([0, 90, 180, 270]))
    sig_str = 'SIGNIFICANT' if is_sig else 'not significant'
    ax_polar.set_title(
        f"n={len(spike_phase_deg)} spikes | MRL={mrl:.3f} | pref={pref_phase_deg:.0f} deg\n"
        f"Rayleigh p={rayleigh_p:.3g} ({sig_str})", fontsize=9)

    # --- ISI histogram ---
    ax_isi = fig.add_subplot(1, 3, 2)
    if len(isi_ms) >= 2:
        isi_bins = np.logspace(np.log10(max(isi_ms.min(), 0.1)), np.log10(isi_ms.max()), 60)
        ax_isi.hist(isi_ms, bins=isi_bins, color='#4472C4', edgecolor='none', alpha=0.85)
    ax_isi.set_xscale('log')
    ax_isi.set_xlabel('ISI (ms)', fontsize=11)
    ax_isi.set_ylabel('Count', fontsize=11)
    cv_str = f'{cv_isi:.2f}' if np.isfinite(cv_isi) else 'n/a'
    ax_isi.set_title(f'ISI (CV={cv_str})', fontsize=12, fontweight='bold')
    ax_isi.tick_params(axis='both', labelsize=9)
    ax_isi.spines[['top', 'right']].set_visible(False)

    # --- autocorrelogram + intrinsic frequency / TMI / peak-valley ---
    ax_acg = fig.add_subplot(1, 3, 3)
    bin_size_ms = acg_edges[1] - acg_edges[0]
    ax_acg.bar(acg_edges[:-1], acg_counts, width=bin_size_ms, align='edge',
               color='#4472C4', edgecolor='none', alpha=0.5)
    ax_acg.plot(acg_edges[:-1] + bin_size_ms / 2, acg_smoothed, color='black', linewidth=1.5)
    ax_acg.axvline(0, color='k', linewidth=0.8, alpha=0.4)
    ax_acg.set_xlabel('Lag (ms)', fontsize=11)
    ax_acg.set_ylabel('Count', fontsize=11)
    freq_str = f'{intrinsic_freq_hz:.2f} Hz' if np.isfinite(intrinsic_freq_hz) else 'n/a'
    tmi_p_str = f'{tmi_pval:.3g}' if np.isfinite(tmi_pval) else 'n/a'
    tmi_sig_str = 'SIGNIFICANT' if tmi_sig else 'not significant'
    ax_acg.set_title(
        f'ACG | intrinsic f={freq_str}\n'
        f'TMI={tmi:.2f} (p={tmi_p_str}, {tmi_sig_str})\n'
        f'phase peak={peak_phase_deg:.0f} deg, valley={valley_phase_deg:.0f} deg',
        fontsize=9)
    ax_acg.tick_params(axis='both', labelsize=9)
    ax_acg.spines[['top', 'right']].set_visible(False)

    fig.suptitle(title_str, fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(save_path, dpi=300)


# %% ==================== Folder discovery ====================

def find_session_folders(root_dir):
    """Every folder under root_dir that directly contains at least one .ntt file."""
    folders = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        if any(fn.lower().endswith('.ntt') for fn in filenames):
            folders.append(dirpath)
    return sorted(folders)


# %% ==================== Main batch loop ====================

def main():
    if 360 % PHASE_BIN_SIZE_DEG != 0:
        raise ValueError('PHASE_BIN_SIZE_DEG must divide 360 evenly.')

    rng = np.random.default_rng(RANDOM_SEED)

    folders = find_session_folders(ROOT_DIR)
    if not folders:
        print(f'No .ntt files found anywhere under {ROOT_DIR}')
        return pd.DataFrame()

    rows = []

    for folder in folders:
        print(f'Processing folder: {folder}')

        ncs_files = sorted(fn for fn in os.listdir(folder) if fn.lower().endswith('.ncs'))
        if not ncs_files:
            print('  No .ncs file found - skipping folder.')
            continue
        if len(ncs_files) > 1:
            print(f'  Multiple .ncs files found; using {ncs_files[0]}')
        ncs_path = os.path.join(folder, ncs_files[0])

        try:
            sig, ts_us, fs = load_ncs(ncs_path)
            phase_deg, phase_ts_us = compute_theta_phase(sig, ts_us, fs)
            peak_ts_us = compute_theta_peak_times(sig, ts_us, fs)
        except Exception as e:
            print(f'  Failed to compute theta phase: {e}')
            continue

        if phase_deg.size == 0:
            print('  No usable LFP segments - skipping folder.')
            continue

        polar_dir = os.path.join(folder, 'polar plots')
        os.makedirs(polar_dir, exist_ok=True)

        session_label = '_'.join(os.path.normpath(folder).split(os.sep)[-3:])
        ntt_files = sorted(fn for fn in os.listdir(folder) if fn.lower().endswith('.ntt'))

        for ntt_name in ntt_files:
            ntt_path = os.path.join(folder, ntt_name)
            unit_name = os.path.splitext(ntt_name)[0]

            try:
                spike_ts_us, cell_numbers = load_ntt_spike_times(ntt_path)
            except Exception as e:
                print(f'  Failed to read {ntt_name}: {e}')
                continue

            unique_cells = np.unique(cell_numbers)
            if unique_cells.size > 1:
                print(f'  WARNING: {ntt_name} contains multiple CellNumbers '
                      f'{unique_cells.tolist()} - this pipeline assumes one isolated '
                      f'unit per .ntt file, so all of its spikes are pooled together.')

            n_total = len(spike_ts_us)

            # --- run the full phase-locking battery once per phase-estimation method ---
            hilbert_assign = lambda ts: assign_spike_phase(ts, phase_ts_us, phase_deg)
            interp_assign = lambda ts: assign_spike_interp_phase(ts, peak_ts_us)

            hilbert_metrics, hilbert_phase_used = compute_phase_locking_metrics(
                spike_ts_us, hilbert_assign, phase_ts_us, rng)
            interp_metrics, interp_phase_used = compute_phase_locking_metrics(
                spike_ts_us, interp_assign, peak_ts_us, rng)

            # --- ISI / autocorrelogram / intrinsic frequency: use every spike, ---
            # --- irrespective of LFP phase coverage (these don't need LFP), and ---
            # --- are shared between the two phase-estimation methods' plots ---
            isi_ms, cv_isi = compute_isi_ms(spike_ts_us)
            acg_edges, acg_counts, acg_smoothed = temporal_autocorr_ms(spike_ts_us)
            intrinsic_freq = intrinsic_frequency_hz(spike_ts_us)

            for method, metrics, phase_used in (
                    ('Hilbert', hilbert_metrics, hilbert_phase_used),
                    ('Interp', interp_metrics, interp_phase_used)):
                if metrics['SpikesUsed'] < MIN_SPIKES:
                    print(f'  {unit_name} ({method}): only {metrics["SpikesUsed"]} spikes with '
                          f'LFP coverage (< MIN_SPIKES={MIN_SPIKES}) - skipping plot.')
                    continue
                save_path = os.path.join(polar_dir, f'{unit_name}_ThetaSummary_{method}.jpg')
                title_str = f'{session_label} | {unit_name} | {method} phase'
                plot_unit_summary(
                    phase_used, metrics['MRL'], metrics['PreferredPhase_deg'],
                    metrics['Rayleigh_p'], metrics['SignificantThetaModulation'],
                    PHASE_BIN_SIZE_DEG, isi_ms, cv_isi, acg_edges, acg_counts, acg_smoothed,
                    intrinsic_freq, metrics['TMI'], metrics['TMI_shuffle_p'],
                    metrics['TMI_Significant'], metrics['PhasePeak_deg'],
                    metrics['PhaseValley_deg'], title_str, save_path)

            row = dict(Session=session_label, FolderPath=folder, Unit=unit_name,
                       TotalSpikes=n_total, ISI_CV=cv_isi, IntrinsicFreq_Hz=intrinsic_freq)
            for method, metrics in (('Hilbert', hilbert_metrics), ('Interp', interp_metrics)):
                for key, val in metrics.items():
                    row[f'{key}_{method}'] = val
            rows.append(row)

    metric_cols = ['SpikesUsed', 'SpikesExcluded_NoLFP', 'MRL', 'PreferredPhase_deg',
                   'Rayleigh_p', 'SignificantThetaModulation', 'PhasePeak_deg',
                   'PhaseValley_deg', 'TMI', 'TMI_shuffle_p', 'TMI_Significant']
    columns = ['Session', 'FolderPath', 'Unit', 'TotalSpikes', 'ISI_CV', 'IntrinsicFreq_Hz']
    columns += [f'{c}_Hilbert' for c in metric_cols] + [f'{c}_Interp' for c in metric_cols]
    df = pd.DataFrame(rows, columns=columns)
    df.to_excel(OUTPUT_EXCEL, sheet_name='AllUnits', index=False)
    print(f'Saved summary of {len(df)} units to {OUTPUT_EXCEL}')
    return df


if __name__ == '__main__':
    main()
