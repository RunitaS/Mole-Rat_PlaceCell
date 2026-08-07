"""
Extract, preprocess and visualise Neuropixels LFP across depth with SpikeInterface.

Pipeline (per recording)
------------------------
1. Preprocessing     : phase shift -> remove bad channels -> bandpass (1-500 Hz) -> common reference
2. Downsampling      : resample to 1000 Hz
   -> cache the downsampled LFP to a memmapped binary before the heavy steps, so
      motion estimation reads a flat 1 kHz binary instead of re-deriving the 30 kHz
      chain for every chunk (much faster, and avoids large-read OOM).
3. Motion correction : dredge_lfp (estimate motion on the LFP traces) -> interpolate_motion
4. Save preprocessed LFP to disk (binary) so downstream reads are fast.

Visualisation (per recording)
-----------------------------
1. Scan the WHOLE recording in short bins and flag noisy epochs (robust amplitude
   threshold). Noisy bins are excluded from everything below.
2. Theta-band (3-7 Hz) power across depth, averaged over the ENTIRE recording
   (noisy epochs excluded) -> one power-vs-depth profile.
3. Ten 5-second snippets spread throughout the recording (each chosen as the
   least-noisy clean window in its part of the recording). Each snippet is
   band-passed to theta and drawn as a depth stack of traces.

The trace/power aesthetic follows Figure 3 of Dunn et al. (ferret theta paper):
stacked filtered traces with the depth axis reversed (top channel of the sequence
on top), beside a power-versus-depth profile.

Depth sequence
--------------
Each recording gets its own ordered depth sequence, given as a list of (start, end)
depth ranges (in microns, as reported by the recording's channel geometry). Each
range selects the surviving channels whose depth falls within it, ordered so that the
channel nearest `start` is on top. Ranges are concatenated top-to-bottom. Examples:

    [(2670, 2310)]            ->  the channel nearest 2670 um on top, descending in
                                  depth to the channel nearest 2310 um at the bottom.
    [(2310, 2670)]            ->  2310 um on top, ascending to 2670 um at the bottom.
    [(2670, 2310), (1900, 1700)] -> first band stacked above the second.

The top channel of the sequence (nearest the first `start` depth) is the reference
for the theta-phase profile (0 ms / 0 deg). Depth is read from
`recording.get_channel_locations()[:, DEPTH_AXIS]` (default axis 1 = y along probe).

Note on DREDGE
--------------
`si.correct_motion(rec, preset="dredge")` is the AP/peak-based variant: it detects
and localises spikes, so it does not work on downsampled LFP. For LFP we use the
`dredge_lfp` method, which estimates motion directly from the traces via
`estimate_motion(...)`, then builds a corrected recording with `interpolate_motion(...)`.
"""

import os
import re
import sys
import math

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import spikeinterface.full as si
from spikeinterface.sortingcomponents.motion import estimate_motion, interpolate_motion
import NpxUtils

# np.trapz was renamed to np.trapezoid in NumPy 2.0 (and removed under the old name).
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Default number of parallel workers suggested at the interactive prompt.
# These are the CPU worker count for the chunked steps (preprocessing, resampling,
# saving to binary). The DREDGE motion estimation is separately routed to the GPU
# when one is available (see configure_compute()).
DEFAULT_CPU_N_JOBS = 8      # suggested n_jobs when no GPU is found
DEFAULT_GPU_N_JOBS = 12     # suggested n_jobs when a GPU is found
CHUNK_DURATION = "1s"       # chunk size for the parallel CPU steps

RESAMPLE_RATE = 1000        # Hz, target LFP sampling rate
THETA_BAND = (3.0, 7.0)     # Hz, band for the filtered traces and the power profile
SNIPPET_DURATION_S = 5.0    # seconds per snippet trace stack
N_SNIPPETS = 10             # number of snippets spread throughout each recording
DEPTH_AXIS = 1              # column of get_channel_locations() holding depth (1 = y along probe)

# --- whole-recording scan / noisy-epoch removal ---
NOISE_BIN_S = 1.0           # bin length for the noise scan and power estimate (s)
NOISE_MAD_THRESH = 5.0      # a bin is "noisy" if its amplitude > median + k*1.4826*MAD
SCAN_BLOCK_S = 60.0         # how many seconds to read at once during the scan (memory-bounded)

# If a processed recording already exists on disk, reuse it instead of re-running
# the (expensive) pipeline. Set True to force a fresh run.
REPROCESS = False

# Where the intermediate binary caches (_00_downsampled_LFP, _01_preprocessed_LFP)
# are written. None -> beside each recording folder (keeps existing caches valid).
# Point this at a local drive if disk space next to the raw data (e.g. Z:) is tight;
# each cache is a few hundred MB to a couple GB depending on channels/duration.
CACHE_DIR = None

# Folder where all generated figures are saved (created if missing).
OUTPUT_DIR = r"F:\Temp\Temp_npx_figs"

# Display figures interactively at the end. This BLOCKS the terminal until you close
# the windows (that's how matplotlib's interactive show works in a plain script).
# Figures are always saved to OUTPUT_DIR regardless, so set this False for a hands-off
# run that saves and exits immediately.
SHOW_FIGURES = True

# On the first snippet of each figure, label EVERY channel in the sequence (rather
# than a sparse subset) so you can verify the exact channel-to-trace mapping. This
# makes that panel taller; set False for a more compact figure with sparse labels.
LABEL_ALL_CHANNELS_ON_FIRST = True

# Theta phase profile: for each channel, the lag of its first theta peak relative to
# the first theta peak on the top-most channel (which defines 0 deg / 0 ms), averaged
# over the snippets (circular mean). If True, lags are reported signed in
# (-T/2, +T/2] so a channel that LEADS the top channel shows a negative lag; if False,
# the raw "first peak strictly after" lag in [0, T) is used instead.
PHASE_SIGNED = True
# Peaks must stand out by at least this many trace-SDs to count as a theta crest
# (rejects small noise ripples in the troughs that would corrupt the period/lag).
PHASE_PEAK_PROMINENCE_STD = 0.5

# ---------------------------------------------------------------------------
# Recordings to process / plot.
# Add one dict per recording; each produces its own figure.
# ---------------------------------------------------------------------------
FILES = [
    dict(
        label="FA1680378B  Day5_1Cntrl",
        base_session_folder=(
            r"Z:\NMR_group_data\Runita\Data\Ephys_Data\FA1680378B"
            r"\Day5_CntrlNoRotRot_Shank4BankA\NeuralData\1Cntrl"
        ),
        # Depth (um) ranges, ordered top-to-bottom. First value is the top of the plot
        # and the theta-phase reference. Here: 2670 um on top -> 2310 um at the bottom.
        depth_sequence=[(2670, 2310)],
        # Optional cell-layer markers: {name: depth_um}. Shaded on the power/phase axes.
        layer_boundaries=None,
    ),
    # --- add more recordings like this -------------------------------------
    # dict(
    #     label="FA1680378B  Day5_2Rot",
    #     base_session_folder=r"Z:\...\2Rot",
    #     depth_sequence=[(2310, 2670)],
    #     layer_boundaries={"pyr": 2500},
    # ),
]


# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------
def configure_plot_style(font_path=r"C:/Windows/Fonts/arial.ttf"):
    """Apply a consistent seaborn/matplotlib style; fall back gracefully if the font is missing."""
    font_name = "sans-serif"
    if os.path.isfile(font_path):
        from matplotlib import font_manager

        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.sans-serif"] = font_name

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11
    sns.set_theme(
        style="ticks",
        palette="colorblind",
        font_scale=1.2,
        font=font_name,
        rc={"axes.spines.right": False, "axes.spines.top": False},
    )


# ---------------------------------------------------------------------------
# Compute configuration (GPU detection + interactive n_jobs prompt)
# ---------------------------------------------------------------------------
def _ask_int(prompt, default, lo=None, hi=None):
    """Prompt for an integer; blank input returns `default`. Re-asks on bad input."""
    while True:
        raw = input(f"{prompt} [default {default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
        except ValueError:
            print("  Please enter a whole number.")
            continue
        if lo is not None and val < lo:
            print(f"  Must be >= {lo}.")
            continue
        if hi is not None and val > hi:
            print(f"  Must be <= {hi}.")
            continue
        return val


def detect_gpu():
    """Return (available, device_names). Requires torch; DREDGE needs it anyway."""
    try:
        import torch
    except ImportError:
        print("torch is not installed -> GPU cannot be used "
              "(and DREDGE motion estimation requires torch).")
        return False, []

    if torch.cuda.is_available():
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        return True, names
    return False, []


def configure_compute():
    """Detect the GPU, ask for the number of parallel jobs, and return (job_kwargs, device).

    Flow:
      1. Look for a CUDA GPU.
      2. If found, route DREDGE motion estimation to it and ask for the number of
         jobs (n_jobs) for the GPU-accelerated run.
      3. If not found, everything runs on CPU and we ask for the CPU n_jobs.

    Only the DREDGE cross-correlation runs on the GPU. The chunked steps
    (preprocessing, resampling, saving) are CPU-bound and parallelised by n_jobs.
    """
    print("\n" + "-" * 60)
    print("Compute configuration")
    print("-" * 60)

    gpu_available, gpu_names = detect_gpu()

    if gpu_available:
        print(f"GPU found: {len(gpu_names)} CUDA device(s) available.")
        for i, name in enumerate(gpu_names):
            print(f"  [{i}] {name}")
        gpu_index = 0
        if len(gpu_names) > 1:
            gpu_index = _ask_int(
                "Select GPU index", default=0, lo=0, hi=len(gpu_names) - 1
            )
        device = f"cuda:{gpu_index}"
        n_jobs = _ask_int(
            "Number of parallel jobs (n_jobs) for GPU-accelerated processing",
            default=DEFAULT_GPU_N_JOBS, lo=1,
        )
    else:
        print("No GPU found -> running everything on CPU.")
        device = "cpu"
        n_jobs = _ask_int(
            "Number of parallel jobs (n_jobs) for CPU processing",
            default=DEFAULT_CPU_N_JOBS, lo=1,
        )

    job_kwargs = dict(n_jobs=n_jobs, chunk_duration=CHUNK_DURATION, progress_bar=True)
    si.set_global_job_kwargs(**job_kwargs)
    print(f"\nUsing device='{device}', n_jobs={n_jobs}, chunk_duration='{CHUNK_DURATION}'.")
    print("-" * 60 + "\n")
    return job_kwargs, device


# ---------------------------------------------------------------------------
# 1. Load recording
# ---------------------------------------------------------------------------
def load_recording(base_session_folder):
    """Return the first Open Ephys recording folder found under the session folder."""
    if not os.path.isdir(base_session_folder):
        raise FileNotFoundError(f"Session folder does not exist: {base_session_folder}")

    recording_folders = sorted(NpxUtils.get_rec_folders(base_session_folder))
    print(f"Found {len(recording_folders)} recording folder(s).")
    if not recording_folders:
        raise FileNotFoundError("No recording folders found.")
    return recording_folders[0]


def load_raw_recording(recording):
    """Read the LFP stream and print a short summary."""
    print(f"\nLoading data from: {recording}")
    raw_rec = si.read_openephys(recording, stream_id="1")

    timestamps = raw_rec.get_times()
    print(f"Sampling frequency : {raw_rec.get_sampling_frequency()} Hz")
    print(f"Number of channels : {raw_rec.get_num_channels()}")
    print(f"Duration           : {timestamps[-1] - timestamps[0]:.2f} s")
    return raw_rec


# ---------------------------------------------------------------------------
# 2-4. Preprocessing, downsampling, DREDGE
# ---------------------------------------------------------------------------
def preprocess(raw_rec):
    """phase shift -> remove bad channels -> bandpass (1-500 Hz) -> common reference."""
    preprocessing_dict = {
        "phase_shift": {},
        "detect_and_remove_bad_channels": {},
        "bandpass_filter": {"freq_min": 1, "freq_max": 500, "dtype": "float32"},
        "common_reference": {"operator": "median", "reference": "global"},
    }
    print("\nStarting preprocessing...")
    preproc_rec = si.apply_preprocessing_pipeline(raw_rec, preprocessing_dict)
    print("Preprocessing complete.")
    return preproc_rec


def downsample(preproc_rec, resample_rate=RESAMPLE_RATE):
    """Resample the preprocessed recording to `resample_rate` Hz."""
    print(f"\nDownsampling to {resample_rate} Hz...")
    return si.resample(preproc_rec, resample_rate=resample_rate)


def correct_motion_lfp(downsampled_rec, device="cpu", rigid=True):
    """Estimate drift from the LFP (dredge_lfp) and return a motion-interpolated recording.

    dredge_lfp works directly on traces, so it needs its own preprocessing chain (bandpass,
    phase shift, extra downsampling, spatial derivative, average across the probe's two
    columns). Motion is estimated on that derived signal, then applied to the full-channel
    downsampled LFP with interpolate_motion.
    """
    print(f"\nEstimating LFP drift with dredge_lfp on device='{device}'...")

    # Signal tuned for motion estimation only (not the recording we keep).
    mrec = si.bandpass_filter(
        downsampled_rec,
        freq_min=0.5,
        freq_max=250,
        margin_ms=1500.0,
        filter_order=3,
        dtype="float32",
        add_reflect_padding=True,
    )
    mrec = si.phase_shift(mrec)
    mrec = si.resample(mrec, resample_rate=250, margin_ms=1000)
    mrec = si.directional_derivative(mrec, order=2, edge_order=1)
    mrec = si.average_across_direction(mrec)

    # `device` is forwarded to dredge_online_lfp -> xcorr_windows, where the
    # normalized cross-correlations run on the GPU when device is a CUDA device.
    motion = estimate_motion(
        mrec, method="dredge_lfp", rigid=rigid, device=device, progress_bar=True
    )

    print("Interpolating motion onto the downsampled LFP...")
    corrected_rec = interpolate_motion(recording=downsampled_rec, motion=motion)
    print("Motion correction complete.")
    return corrected_rec, motion


def _load_saved(folder):
    """Load a previously saved recording, tolerating SpikeInterface API differences."""
    try:
        return si.load(folder)
    except (AttributeError, TypeError):
        return si.load_extractor(folder)


def _cache_folders(cfg, recording_path):
    """Return (downsampled_cache, corrected_cache) folder paths for one recording."""
    if CACHE_DIR is None:
        # Beside the recording (keeps any existing _01_preprocessed_LFP caches valid).
        return (recording_path + "_00_downsampled_LFP",
                recording_path + "_01_preprocessed_LFP")
    root = os.path.join(CACHE_DIR, re.sub(r"[^A-Za-z0-9._-]+", "_", cfg["label"]).strip("_"))
    return root + "_00_downsampled_LFP", root + "_01_preprocessed_LFP"


def cache_recording(recording, folder, job_kwargs, reuse=True):
    """Write `recording` to a memmapped binary and return the on-disk recording.

    `recording.save(format="binary", ...)` streams the data to disk chunk-by-chunk
    (memory bounded by chunk_duration) and returns a flat, memmapped recording whose
    reads are cheap and do NOT recompute the upstream lazy chain. Reuses an existing
    cache when present.
    """
    if reuse and os.path.isdir(folder):
        print(f"  Reusing cache: {folder}")
        return _load_saved(folder)
    print(f"  Caching to binary: {folder}")
    return recording.save(folder=folder, format="binary", overwrite=True, **job_kwargs)


def get_processed_recording(cfg, job_kwargs, device="cpu", reprocess=REPROCESS):
    """Run (or reload) the full pipeline for one recording and return the final LFP recording."""
    recording = load_recording(cfg["base_session_folder"])
    ds_folder, corrected_folder = _cache_folders(cfg, recording)

    # Fast path: the fully processed (motion-corrected) cache already exists.
    if (not reprocess) and os.path.isdir(corrected_folder):
        print(f"\nReusing cached processed LFP: {corrected_folder}")
        return _load_saved(corrected_folder)

    raw_rec = load_raw_recording(recording)
    preproc_rec = preprocess(raw_rec)
    downsampled_rec = downsample(preproc_rec, RESAMPLE_RATE)

    # Cache the downsampled LFP to a memmapped binary BEFORE the heavy steps.
    # Without this, dredge_lfp reads the still-lazy chain in chunks and re-derives
    # phase_shift -> bad-channel removal -> bandpass -> CMR -> resample from the
    # 30 kHz parent for every chunk (slow, and a single large read can OOM because
    # CMR forces all channels through phase_shift's float64 FFT at once). Reading
    # from the flat 1 kHz binary instead makes motion estimation far faster.
    print("\nCaching downsampled LFP before motion correction...")
    downsampled_cached = cache_recording(
        downsampled_rec, ds_folder, job_kwargs, reuse=not reprocess
    )

    corrected_rec, _motion = correct_motion_lfp(downsampled_cached, device=device)

    print(f"\nSaving preprocessed LFP to: {corrected_folder}")
    corrected_rec = corrected_rec.save(
        folder=corrected_folder, format="binary", overwrite=True, **job_kwargs
    )
    print("Saved.")
    return corrected_rec


# ---------------------------------------------------------------------------
# Depth-sequence handling
# ---------------------------------------------------------------------------
def resolve_depth_sequence(recording, depth_sequence, depth_axis=DEPTH_AXIS):
    """Select and order channels by depth; return list of (depth_um, channel_id) top-to-bottom.

    For each (start, end) range, all surviving channels whose depth lies in
    [min(start,end), max(start,end)] are selected and ordered so the channel nearest
    `start` is on top (descending depth when start > end, ascending otherwise). Ranges
    are concatenated. Every entry corresponds to a real channel (no gaps), because we
    select from the channels that actually exist in the recording.
    """
    ids = np.asarray(recording.get_channel_ids())
    locs = recording.get_channel_locations()
    depths = np.asarray(locs)[:, depth_axis].astype(float)

    resolved = []
    for start, end in depth_sequence:
        lo, hi = (end, start) if start > end else (start, end)
        mask = (depths >= lo) & (depths <= hi)
        sel_ids = ids[mask]
        sel_depths = depths[mask]
        order = np.argsort(sel_depths, kind="stable")   # ascending depth
        if start > end:
            order = order[::-1]                          # top = nearest `start` (max depth)
        for i in order:
            resolved.append((float(sel_depths[i]), sel_ids[i]))
        print(f"  Depth band {start}->{end} um: {sel_ids.size} channels "
              f"(depths {sel_depths.min():.0f}-{sel_depths.max():.0f} um)."
              if sel_ids.size else
              f"  Depth band {start}->{end} um: no channels found in range!")
    return resolved


# ---------------------------------------------------------------------------
# Whole-recording scan: noisy-epoch detection + theta power per bin
# ---------------------------------------------------------------------------
def scan_recording(recording, bin_s=NOISE_BIN_S, band=THETA_BAND, block_s=SCAN_BLOCK_S):
    """Stream the whole recording in bins; return per-bin amplitude metric + theta power.

    Reads the (cached, memmapped) recording in blocks of `block_s` seconds so memory
    stays bounded, splits each block into `bin_s` bins, and for every bin computes:
      * a robust broadband amplitude metric (median across channels of per-channel SD),
        used to flag noisy epochs, and
      * the integrated theta-band power per channel (linear, uV^2), used for the
        depth power profile.
    """
    fs = recording.get_sampling_frequency()
    n_total = recording.get_num_samples()
    n_ch = recording.get_num_channels()
    n_bin = int(round(bin_s * fs))
    n_bins = n_total // n_bin
    if n_bins == 0:
        raise ValueError("Recording shorter than one noise bin.")

    metric = np.empty(n_bins, dtype=np.float64)          # per-bin amplitude
    band_lin = np.empty((n_bins, n_ch), dtype=np.float64)  # per-bin, per-channel theta power

    bins_per_block = max(1, int(round(block_s / bin_s)))
    print(f"  Scanning {n_bins} bins of {bin_s:g}s ({n_bins * bin_s / 60:.1f} min total)...")

    next_report = 0.1
    for b0 in range(0, n_bins, bins_per_block):
        b1 = min(n_bins, b0 + bins_per_block)
        nb = b1 - b0
        block = recording.get_traces(
            start_frame=b0 * n_bin, end_frame=b1 * n_bin, return_in_uV=True
        )  # (nb*n_bin, n_ch)
        block = block[: nb * n_bin].reshape(nb, n_bin, n_ch)  # (bins, samples, ch)

        # broadband amplitude per bin: median over channels of the per-channel SD
        metric[b0:b1] = np.median(block.std(axis=1), axis=1)

        # theta power per bin per channel (one periodogram per bin)
        freqs, pxx = signal.welch(block, fs=fs, nperseg=n_bin, axis=1)  # (bins, nfreq, ch)
        m = (freqs >= band[0]) & (freqs <= band[1])
        band_lin[b0:b1] = _trapz(pxx[:, m, :], freqs[m], axis=1)  # (bins, ch)

        if b1 / n_bins >= next_report:
            print(f"    ...{100 * b1 / n_bins:.0f}%")
            next_report += 0.1

    ids = list(recording.get_channel_ids())
    return dict(metric=metric, band_lin=band_lin, ids=ids, n_bin=n_bin, fs=fs, n_bins=n_bins)


def flag_noisy_bins(metric, k=NOISE_MAD_THRESH):
    """Return (good_mask, threshold): a bin is noisy if metric > median + k*1.4826*MAD."""
    med = np.median(metric)
    mad = np.median(np.abs(metric - med))
    thr = med + k * 1.4826 * mad
    good = metric <= thr
    # guard against dead/zero bins as well
    good &= metric > 0
    return good, thr


def pick_snippet_starts(metric, good, n_bin, fs, n_snippets, snippet_len_s):
    """Choose `n_snippets` snippet start times (s), spread across the recording.

    The recording is split into `n_snippets` equal parts; in each part the
    least-noisy fully-clean `snippet_len_s` window is chosen (falling back to the
    least-noisy window if none is fully clean).
    """
    n_bins = len(metric)
    snip_bins = max(1, int(round(snippet_len_s * fs / n_bin)))
    if n_bins <= snip_bins:
        return [0.0]

    max_start = n_bins - snip_bins  # last valid start bin
    # windowed mean metric over each candidate start bin
    csum = np.concatenate([[0.0], np.cumsum(metric)])
    win_mean = (csum[snip_bins:snip_bins + max_start + 1] - csum[:max_start + 1]) / snip_bins
    # windows that are fully clean
    gsum = np.concatenate([[0], np.cumsum(good.astype(int))])
    win_good = (gsum[snip_bins:snip_bins + max_start + 1] - gsum[:max_start + 1]) == snip_bins

    seg_edges = np.linspace(0, max_start + 1, n_snippets + 1).astype(int)
    starts_bins = []
    for i in range(n_snippets):
        lo = seg_edges[i]
        hi = max(seg_edges[i] + 1, seg_edges[i + 1])
        idx = np.arange(lo, min(hi, max_start + 1))
        if idx.size == 0:
            idx = np.array([min(lo, max_start)])
        cand = idx[win_good[idx]] if win_good[idx].any() else idx
        best = cand[np.argmin(win_mean[cand])]
        starts_bins.append(int(best))

    return [b * n_bin / fs for b in starts_bins]


# ---------------------------------------------------------------------------
# Trace extraction and filtering
# ---------------------------------------------------------------------------
def get_window_traces(recording, start_s, duration_s):
    """Return (traces, id_to_col, time_vector, fs) for a window of the recording.

    traces has shape (n_samples, n_all_channels) in microvolts. The window is
    clamped to the available data.
    """
    fs = recording.get_sampling_frequency()
    n_total = recording.get_num_samples()
    n_win = min(int(duration_s * fs), n_total)
    start = int(start_s * fs)
    start = max(0, min(start, n_total - n_win))

    traces = recording.get_traces(
        start_frame=start, end_frame=start + n_win, return_in_uV=True
    )
    ids = list(recording.get_channel_ids())
    id_to_col = {cid: i for i, cid in enumerate(ids)}
    t = np.arange(n_win) / fs
    return traces, id_to_col, t, fs


def bandpass_theta(traces, fs, band=THETA_BAND, order=4):
    """Zero-phase Butterworth band-pass filter applied along the time axis (axis 0)."""
    sos = signal.butter(order, band, btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, traces, axis=0)


def compute_phase_profile(snippets, resolved, fs, band=THETA_BAND,
                          peak_distance_s=0.1, edge_guard_s=0.2, signed=PHASE_SIGNED):
    """Theta phase (peak-lag) vs depth, relative to the top-most channel.

    Method (per snippet):
      * On the top-most present channel, find the first theta peak -> defines t=0 (0 deg).
      * On every other channel, find the first theta peak at or after t=0 and take the
        lag. The lag is turned into a phase, 2*pi * lag / T, where T is the reference
        channel's median theta period in that snippet.
    Phases are then combined across snippets with a circular mean (robust to
    cycle-to-cycle jitter and to wraparound). With `signed=True` the resulting phase is
    reported in (-pi, pi] (leads negative); otherwise mapped to [0, 2*pi).

    Returns dict with per-position: phase_deg, lag_ms, R (circular concentration 0-1),
    count (snippets contributing), plus T_mean (s) and ref_pos.
    """
    n = len(resolved)
    ref_pos = next((p for p, (_, cid) in enumerate(resolved) if cid is not None), None)
    if ref_pos is None:
        return None

    dist = max(1, int(peak_distance_s * fs))
    accum = np.zeros(n, dtype=complex)
    count = np.zeros(n, dtype=int)
    periods = []

    def _peak_times(tr, tvec):
        # prominence threshold keyed to the trace's own amplitude, so only genuine
        # theta crests are detected (not small ripples on the troughs).
        prom = PHASE_PEAK_PROMINENCE_STD * np.std(tr)
        idx, _ = signal.find_peaks(tr, distance=dist, prominence=max(prom, 1e-9))
        return tvec[idx]

    for snip in snippets:
        tvec = snip["t"]
        ref = snip["theta_by_pos"][ref_pos]
        if ref is None:
            continue
        ref_pk = _peak_times(ref, tvec)
        ref_pk = ref_pk[ref_pk >= edge_guard_s]
        if ref_pk.size < 2:
            continue
        t_ref = ref_pk[0]
        T = np.median(np.diff(ref_pk))
        if not (np.isfinite(T) and T > 0):
            continue
        periods.append(T)

        for p in range(n):
            tr = snip["theta_by_pos"][p]
            if tr is None:
                continue
            pk = _peak_times(tr, tvec)
            after = pk[pk >= t_ref]
            if after.size == 0:
                continue
            lag = after[0] - t_ref                      # first peak at/after reference
            accum[p] += np.exp(1j * 2 * np.pi * (lag % T) / T)
            count[p] += 1

    has = count > 0
    mean_phase = np.full(n, np.nan)
    R = np.full(n, np.nan)
    mean_phase[has] = np.angle(accum[has])              # (-pi, pi]
    R[has] = np.abs(accum[has]) / count[has]
    if not signed:
        mean_phase[has] = mean_phase[has] % (2 * np.pi)  # [0, 2*pi)

    T_mean = float(np.mean(periods)) if periods else 2.0 / (band[0] + band[1])
    phase_deg = np.degrees(mean_phase)
    lag_ms = mean_phase / (2 * np.pi) * T_mean * 1000.0
    return dict(phase_deg=phase_deg, lag_ms=lag_ms, R=R, count=count,
                T_mean=T_mean, ref_pos=ref_pos)


# ---------------------------------------------------------------------------
# Per-recording assembly
# ---------------------------------------------------------------------------
def build_result(cfg, recording, color):
    """Extract everything needed to draw one recording's figure."""
    print(f"\nBuilding depth profile for: {cfg['label']}")
    resolved = resolve_depth_sequence(recording, cfg["depth_sequence"])
    if not resolved:
        raise ValueError(f"No channels found for depth_sequence={cfg['depth_sequence']}.")
    positions = np.arange(len(resolved))  # 0 = top of the sequence (nearest first depth)

    # --- whole-recording scan + noisy-epoch removal ---
    scan = scan_recording(recording)
    good, thr = flag_noisy_bins(scan["metric"])
    frac_bad = 1.0 - good.mean()
    print(f"  Noisy epochs removed: {frac_bad * 100:.1f}% of {scan['n_bins']} bins "
          f"(threshold {thr:.1f} uV SD).")

    id_to_col = {cid: i for i, cid in enumerate(scan["ids"])}

    # theta power per channel over the ENTIRE recording (clean bins only)
    mean_lin = scan["band_lin"][good].mean(axis=0)  # (n_ch,) linear uV^2
    power_db = np.array([
        10.0 * np.log10(mean_lin[id_to_col[cid]] + 1e-12) if cid is not None else np.nan
        for _, cid in resolved
    ])

    # --- ten 5 s snippets spread through the recording (clean windows) ---
    starts = pick_snippet_starts(
        scan["metric"], good, scan["n_bin"], scan["fs"], N_SNIPPETS, SNIPPET_DURATION_S
    )
    snippets, pooled = [], []
    for st in starts:
        snip, snip_id_to_col, t, fs = get_window_traces(recording, st, SNIPPET_DURATION_S)
        theta = bandpass_theta(snip, fs)
        theta_by_pos = [
            (theta[:, snip_id_to_col[cid]] if cid is not None else None) for _, cid in resolved
        ]
        snippets.append(dict(start_s=st, t=t, theta_by_pos=theta_by_pos))
        pooled.extend([np.abs(tr) for tr in theta_by_pos if tr is not None])

    # common amplitude gain across ALL snippets so amplitudes are comparable
    scale = np.percentile(np.concatenate(pooled), 99) if pooled else 1.0
    gain = 0.42 / scale if scale > 0 else 1.0  # trace spans < half the 1.0 unit spacing

    # theta phase (peak-lag) vs depth, relative to the top-most channel
    phase = compute_phase_profile(snippets, resolved, fs, THETA_BAND)
    if phase is not None:
        finite = np.isfinite(phase["lag_ms"])
        if finite.any():
            ref_depth, ref_cid = resolved[phase["ref_pos"]]
            print(f"  Theta phase: reference = {ref_depth:.0f} um ({ref_cid}), lag range "
                  f"{np.nanmin(phase['lag_ms']):+.1f} to {np.nanmax(phase['lag_ms']):+.1f} ms "
                  f"(period {phase['T_mean'] * 1000:.0f} ms).")

    # optional layer markers (keyed by depth in um) -> nearest positions
    layers = None
    if cfg.get("layer_boundaries"):
        depths_arr = np.array([d for d, _ in resolved])
        layers = {name: int(np.argmin(np.abs(depths_arr - dval)))
                  for name, dval in cfg["layer_boundaries"].items()}

    return dict(
        label=cfg["label"],
        resolved=resolved,
        positions=positions,
        snippets=snippets,
        power_db=power_db,
        phase=phase,
        gain=gain,
        amp_scale=scale,
        color=color,
        layers=layers,
        frac_bad=frac_bad,
    )


# ---------------------------------------------------------------------------
# Plotting: Figure-3-style depth profiles (10 snippets + whole-recording power)
# ---------------------------------------------------------------------------
def _nice_number(x):
    """Round x up to 1, 2 or 5 x 10^k for a tidy scale-bar label."""
    if x <= 0:
        return 1.0
    exp = math.floor(math.log10(x))
    base = x / 10 ** exp
    for b in (1, 2, 5):
        if base <= b:
            return b * 10 ** exp
    return 10 ** (exp + 1)


def _depth_yticks(ax, resolved, positions, max_labels=12):
    """Label a sparse subset of positions with their depth (um)."""
    step = max(1, len(positions) // max_labels)
    idx = list(range(0, len(positions), step))
    ax.set_yticks(positions[idx])
    ax.set_yticklabels([f"{resolved[i][0]:.0f}" for i in idx])


def _depth_yticks_all(ax, resolved, positions, fontsize=6):
    """Label EVERY position with its depth (um) and channel id, to verify the mapping."""
    labels = [f"{depth:.0f} ({cid})" for depth, cid in resolved]
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.tick_params(axis="y", length=2, pad=1)


def _plot_snippet_stack(ax, res, snip, show_y, show_x, show_ampbar,
                        label_all=False, tick_fontsize=6):
    """Draw one snippet's depth stack of theta-filtered traces."""
    t, gain = snip["t"], res["gain"]
    positions, resolved = res["positions"], res["resolved"]
    n = len(positions)

    for pos, tr in zip(positions, snip["theta_by_pos"]):
        if tr is None:
            continue
        ax.plot(t, pos + gain * tr, color=res["color"], lw=0.5)

    ax.set_ylim(n - 0.5, -0.5)          # depth reversed: top of sequence on top
    ax.set_xlim(t[0], t[-1])
    ax.set_title(f"t = {snip['start_s']:.0f} s", fontsize=9)
    ax.spines["left"].set_visible(show_y)

    if show_y:
        if label_all:
            _depth_yticks_all(ax, resolved, positions, fontsize=tick_fontsize)
        else:
            _depth_yticks(ax, resolved, positions)
        ax.set_ylabel("Depth (\u00b5m)")
    else:
        ax.tick_params(labelleft=False)
        ax.set_yticks([])
    if show_x:
        ax.set_xlabel("Time (s)")
    else:
        ax.tick_params(labelbottom=False)

    if show_ampbar:
        amp = _nice_number(res["amp_scale"])
        x0 = t[-1] - 0.02 * (t[-1] - t[0])
        y0 = 0.5
        ax.plot([x0, x0], [y0, y0 + gain * amp], color="k", lw=1.5, clip_on=False)
        ax.text(x0 - 0.02 * (t[-1] - t[0]), y0 + gain * amp / 2,
                f"{amp:.0f} \u00b5V", ha="right", va="center", fontsize=8)


def _plot_power_profile(ax, res):
    """Whole-recording theta power vs depth."""
    ax.plot(res["power_db"], res["positions"], "-o", color=res["color"], ms=3, lw=1)
    ax.set_xlabel("Theta power (dB)")
    ax.set_title("Theta power\n(whole recording)", fontsize=9)
    ax.set_ylim(len(res["positions"]) - 0.5, -0.5)
    ax.tick_params(labelleft=False)

    if res["layers"]:
        for name, pos in res["layers"].items():
            ax.axhspan(pos - 0.45, pos + 0.45, color=[0.5, 0.5, 0.5], alpha=0.3)
            ax.text(ax.get_xlim()[1], pos, f" {name}", va="center", fontsize=7, clip_on=False)


def _plot_phase_profile(ax, res):
    """Theta peak-lag vs depth, relative to the top-most channel (0 ms = 0 deg)."""
    phase = res["phase"]
    positions = res["positions"]
    if phase is None:
        ax.text(0.5, 0.5, "no phase\n(no reference peak)", ha="center", va="center",
                fontsize=8, transform=ax.transAxes)
        ax.set_xticks([])
        return

    ax.plot(phase["lag_ms"], positions, "-o", color=res["color"], ms=3, lw=1)
    ax.axvline(0, color="k", lw=0.5)
    # mark the reference channel at 0
    ax.plot(0, phase["ref_pos"], marker="*", color="k", ms=9, clip_on=False, zorder=5)
    ax.set_xlabel("Theta lag (ms)")
    ax.set_title("Theta phase\n(vs top channel)", fontsize=9)
    ax.set_ylim(len(positions) - 0.5, -0.5)
    ax.tick_params(labelleft=False)

    # secondary axis: phase in degrees (linear in lag via the mean theta period)
    T = phase["T_mean"]
    if T and T > 0:
        secax = ax.secondary_xaxis(
            "top",
            functions=(lambda x: x / 1000.0 / T * 360.0,
                       lambda d: d / 360.0 * T * 1000.0),
        )
        secax.set_xlabel("phase (deg)", fontsize=8)
        secax.tick_params(labelsize=7)

    if res["layers"]:
        for _, pos in res["layers"].items():
            ax.axhspan(pos - 0.45, pos + 0.45, color=[0.5, 0.5, 0.5], alpha=0.3)


def plot_recording_figure(res, band=THETA_BAND, output_dir=OUTPUT_DIR):
    """One figure per recording: N snippet trace stacks + whole-recording power profile."""
    snippets = res["snippets"]
    n_snip = len(snippets)
    n_ch = len(res["positions"])
    ncols = 5
    nrows = int(np.ceil(n_snip / ncols))

    # When labelling every channel, give each snippet row enough height to fit the
    # ticks, and shrink the font as the channel count grows.
    if LABEL_ALL_CHANNELS_ON_FIRST:
        row_h = max(5.0, 0.10 * n_ch)
        tick_fs = int(np.clip(700.0 / max(n_ch, 1), 4, 8))
    else:
        row_h, tick_fs = 5.0, 6

    fig = plt.figure(figsize=(3.2 * ncols + 5.0, row_h * nrows))
    gs = gridspec.GridSpec(
        nrows, ncols + 2, width_ratios=[1] * ncols + [1.3, 1.3],
        wspace=0.2, hspace=0.28, figure=fig,
    )
    fig.suptitle(
        f"{res['label']}   |   theta {band[0]:g}-{band[1]:g} Hz   |   "
        f"{res['frac_bad'] * 100:.1f}% noisy epochs removed",
        fontsize=12,
    )

    ax0 = None
    for i, snip in enumerate(snippets):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c], sharey=ax0, sharex=ax0) if ax0 is not None \
            else fig.add_subplot(gs[r, c])
        if ax0 is None:
            ax0 = ax
        # Only the first snippet carries the y-axis; label every channel there so the
        # channel-to-trace mapping can be checked. The rest share this axis.
        _plot_snippet_stack(
            ax, res, snip,
            show_y=(i == 0),
            show_x=(r == nrows - 1),
            show_ampbar=(i == 0),
            label_all=(i == 0 and LABEL_ALL_CHANNELS_ON_FIRST),
            tick_fontsize=tick_fs,
        )

    ax_pow = fig.add_subplot(gs[:, ncols], sharey=ax0)
    _plot_power_profile(ax_pow, res)
    ax_pha = fig.add_subplot(gs[:, ncols + 1], sharey=ax0)
    _plot_phase_profile(ax_pha, res)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = "lfp_theta_depth_" + re.sub(r"[^A-Za-z0-9._-]+", "_", res["label"]).strip("_") + ".png"
    fig_path = os.path.join(output_dir, fname)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {fig_path}")
    if not SHOW_FIGURES:
        plt.close(fig)  # free memory; nothing will be displayed
    return fig_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    configure_plot_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if CACHE_DIR is not None:
        os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Figures will be saved to: {os.path.abspath(OUTPUT_DIR)}")

    # Ask about GPU/CPU and number of jobs before doing any heavy work.
    job_kwargs, device = configure_compute()

    palette = sns.color_palette("colorblind", n_colors=max(len(FILES), 1))

    for cfg, color in zip(FILES, palette):
        recording = get_processed_recording(cfg, job_kwargs=job_kwargs, device=device)
        res = build_result(cfg, recording, color)
        plot_recording_figure(res)

    if SHOW_FIGURES:
        print("\nDisplaying figures. Close the window(s) to exit.")
        plt.show()          # blocks until you close the figures
    plt.close("all")
    print("Done.")


if __name__ == "__main__":
    main()
    # Return control to the shell deterministically. This runs normal interpreter
    # shutdown (flushing output, atexit cleanup of worker pools). If a run ever still
    # hangs here because a background library left a process alive, replace this with
    # `os._exit(0)` to force-terminate immediately.
    sys.exit(0)