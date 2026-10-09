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

LFP cleaning (same protocol as the tetrode PSD pipeline, Fig4a_PSD.py)
-----------------------------------------------------------------------
Applied to the depth-sequence channels before any power / phase is computed:
  1. 50 Hz notch      : zero-phase IIR notch at LINE_HARMONICS (Q = NOTCH_Q). Blocks
                        are read with NOTCH_MARGIN_S padding that is trimmed off, so
                        filtfilt edge transients never enter the analysed data.
  2. Detrend          : a per-channel linear trend fit over the WHOLE recording
                        (streamed, fit_trend) is subtracted -- equivalent to
                        scipy.signal.detrend on the full trace.
  3. Epoching         : EPOCH_S (2 s) epochs, one Hann-window Welch periodogram each.
  4. Delta/theta      : an epoch is rejected if its delta (LOW_BAND) power exceeds its
     filter             theta (THETA_BAND) power.
  5. MAD rejection    : an epoch is rejected if it is a high outlier (robust z >
                        MAD_THRESH) on EITHER peak-to-peak amplitude OR delta power;
                        median/MAD are estimated only from epochs surviving (4).
Per-epoch metrics are the median across the depth-sequence channels, so an epoch is
kept or rejected for all channels together (required for the phase profile). There
is no running-speed gating (no tracking data here).

Averaging method across multiple clean snippets:
Both are averaged over multiple windows, not a single snippet —
but via different mechanisms:

Theta power vs. depth: computed from scan_recording, which splits the entire
recording into EPOCH_S windows, rejects epochs with clean_epochs (see above),
and averages theta power across all remaining clean epochs. This isn't snippet-based
at all — it uses the whole recording.

Theta phase/lag vs. depth: compute_phase_profile loops
over the N_SNIPPETS = 10 clean 5 s snippets (picked by pick_snippet_starts), 
computes a peak-lag phase estimate per snippet, and combines them via a circular 
mean across snippets (weighted by concentration R), not from a single snippet.

So neither calculation relies on just one snippet: power comes from whole-recording 
binning, and phase comes from a circular average over 10 clean snippets.

Visualisation (per recording)
-----------------------------
1. Scan the WHOLE (notched + detrended) recording in 2 s epochs and reject epochs
   with the delta/theta filter and dual-criteria MAD rejection (see "LFP cleaning").
   Rejected epochs are excluded from everything below.
2. Theta-band (3-7 Hz) power across depth, averaged over the ENTIRE recording
   (rejected epochs excluded) -> one power-vs-depth profile.
3. Ten 5-second snippets spread throughout the recording (each chosen as the
   lowest-amplitude fully-clean window in its part of the recording). Each snippet
   is notched, detrended, band-passed to theta and drawn as a depth stack of traces.

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

Multiple recordings and the cross-recording average
-----------------------------------------------------
`FILES` (below) is a list: add one dict per recording and each gets its own
per-recording figure (snippet stacks + whole-recording power profile + phase
profile), exactly as before. When 2+ recordings are listed, the script ALSO builds
one additional "average" figure across all of them (`compute_average_profile` /
`plot_average_figure`):

  * Recordings are aligned by ABSOLUTE DEPTH (um), not by channel rank, since
    different recordings can use different depth_sequence ranges / channel counts.
    Each recording's power/phase profile is linearly interpolated onto a common
    depth grid (step `AVERAGE_DEPTH_BIN_UM`); a recording contributes nan outside
    its own depth range (no extrapolation).
  * Power is averaged with a plain mean (in dB) across recordings that have data
    at a given depth.
  * Phase is a circular quantity, so recordings are interpolated on the unit
    circle (real/imag parts of R*exp(i*phase)) and combined with a circular mean,
    the same way compute_phase_profile already combines snippets within one
    recording. NOTE: each recording's phase is relative to the TOP channel of
    THAT recording's own depth_sequence, so averaging across recordings is only
    physically meaningful if those top-reference channels correspond to the same
    (or a comparable) anatomical landmark across recordings.
"""

!!! Remove aperiod component. Use FOOOF !!!

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

# --- whole-recording scan / epoch cleaning (same protocol as the tetrode PSD
# pipeline, Fig4a_PSD.py -- see "LFP cleaning" in the module docstring) ---
EPOCH_S = 2.0               # epoch length for cleaning + power (s); 2 s -> 0.5 Hz resolution
MAD_THRESH = 5.0            # dual-criteria epoch-rejection threshold (robust z)
LOW_BAND = (1.0, 3.0)       # delta band: MAD rejection criterion + delta/theta filter
APPLY_DELTA_THETA_FILTER = True  # reject epochs whose delta power exceeds their theta power
SCAN_BLOCK_S = 60.0         # how many seconds to read at once during the scan (memory-bounded)

# --- 50 Hz line-noise cleaning (European mains) ---
LINE_HARMONICS = [50.0, 100.0, 150.0, 200.0]
APPLY_TIME_NOTCH = True     # scipy IIR notch on the time series
NOTCH_Q = 30.0
NOTCH_MARGIN_S = 2.0        # padding read on each side of a block, trimmed after filtfilt

# --- detrending (after notch filter) ---
APPLY_TIME_DETREND = True   # remove slow drift, fit over the whole recording
DETREND_TYPE = "linear"     # 'linear' or 'constant' (as in scipy.signal.detrend)

# --- cross-recording average (see "Multiple recordings and the cross-recording
# average" in the module docstring) ---
AVERAGE_DEPTH_BIN_UM = 20.0  # step of the common depth (um) grid used to align recordings

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
# Add one dict per recording; each produces its own figure (power & phase vs
# depth, alongside the theta-filtered snippet stacks). When 2+ recordings are
# listed, an additional figure averaging power & phase across all of them
# (aligned by absolute depth) is also produced -- see the module docstring.
# ---------------------------------------------------------------------------
FILES = [
    dict(
        label="FA1680378B  Day5_1Cntrl",
        base_session_folder=(
            r"X:\NMR_group_data\Runita\Data\Ephys_Data\FA1680378B"
            r"\Day5_CntrlNoRotRot_Shank4BankA\NeuralData\1Cntrl"
        ),
        # Depth (um) ranges, ordered top-to-bottom. First value is the top of the plot
        # and the theta-phase reference. Here: 2670 um on top -> 2310 um at the bottom.
        depth_sequence=[(2670, 2310)],
        # Optional cell-layer markers: {name: depth_um}. Shaded on the power/phase axes.
        layer_boundaries=None,
    ),
        dict(
        label="FA1680378B  Day1_1Cntrl",
        base_session_folder=(
            r"X:\NMR_group_data\Runita\Data\Ephys_Data\FA1680378B"
            r"\Day1_CntrlCntrlCntrlZero_Shank1BankA\NeuralData\Cntrl1"
        ),
        # Depth (um) ranges, ordered top-to-bottom. First value is the top of the plot
        # and the theta-phase reference. Here: 2670 um on top -> 2310 um at the bottom.
        depth_sequence=[(2460,2190)],
        # Optional cell-layer markers: {name: depth_um}. Shaded on the power/phase axes.
        layer_boundaries=None,
    ),
            dict(
        label="FA1680378B  Day3_1Cntrl",
        base_session_folder=(
            r"X:\NMR_group_data\Runita\Data\Ephys_Data\FA1680378B"
            r"\Day3_CntrlRotNoRotCntrl_Shank2BankA\NeuralData\1Cntrl"
        ),
        # Depth (um) ranges, ordered top-to-bottom. First value is the top of the plot
        # and the theta-phase reference. Here: 2670 um on top -> 2310 um at the bottom.
        depth_sequence=[(2610,2250)],
        # Optional cell-layer markers: {name: depth_um}. Shaded on the power/phase axes.
        layer_boundaries=None,
    ),
                dict(
        label="FA1680378B  Day6_1Cntrl",
        base_session_folder=(
            r"X:\NMR_group_data\Runita\Data\Ephys_Data\FA1680378B"
            r"\Day6_CntrlCntrlCntrl_Shank4BankA\NeuralData\1Cntrl "
        ),
        # Depth (um) ranges, ordered top-to-bottom. First value is the top of the plot
        # and the theta-phase reference. Here: 2670 um on top -> 2310 um at the bottom.
        depth_sequence=[(2655,2370)],
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
# LFP cleaning (ported from the tetrode PSD pipeline, Fig4a_PSD.py)
# ---------------------------------------------------------------------------
def notch_filter(x, fs_hz, freqs, Q=30.0):
    """Zero-phase IIR notch at each frequency in `freqs` along axis 0 (skips freqs >= Nyquist)."""
    y = np.asarray(x, dtype=np.float64)
    nyq = fs_hz / 2.0
    for f0 in freqs:
        if f0 <= 0 or f0 >= nyq:
            continue
        b, a = signal.iirnotch(f0, Q, fs_hz)
        y = signal.filtfilt(b, a, y, axis=0)
    return y


def fit_trend(recording, channel_ids, dtype=DETREND_TYPE, block_s=SCAN_BLOCK_S):
    """Per-channel polynomial trend over the WHOLE recording, fit by streaming.

    Equivalent to fitting scipy.signal.detrend(type=dtype) on the full trace, but
    accumulated block-by-block so the recording is never loaded at once. The fit is
    done on the un-notched trace; the notch has unity gain at DC / slow drift, so the
    trend is the same. Returns (slope, intercept) per channel (uV/s, uV), with t in
    seconds from the first sample; slope is zero for dtype='constant'.
    """
    fs = recording.get_sampling_frequency()
    n_total = recording.get_num_samples()
    n_ch = len(channel_ids)
    n_blk = int(round(block_s * fs))

    st = stt = 0.0
    sx = np.zeros(n_ch)
    stx = np.zeros(n_ch)
    for s0 in range(0, n_total, n_blk):
        s1 = min(n_total, s0 + n_blk)
        x = recording.get_traces(
            start_frame=s0, end_frame=s1, channel_ids=channel_ids, return_in_uV=True
        ).astype(np.float64)
        t = np.arange(s0, s1) / fs
        st += t.sum()
        stt += (t * t).sum()
        sx += x.sum(axis=0)
        stx += t @ x

    n = float(n_total)
    if dtype == "linear":
        slope = (n * stx - st * sx) / (n * stt - st ** 2)
    else:
        slope = np.zeros(n_ch)
    intercept = (sx - slope * st) / n
    return slope, intercept


def read_clean_traces(recording, start_frame, end_frame, channel_ids, trend=None):
    """Read [start_frame, end_frame) for `channel_ids` in uV, notch-filtered and detrended.

    The notch runs on a window padded by NOTCH_MARGIN_S on each side (clamped to the
    recording) and the padding is trimmed afterwards, so filtfilt edge transients never
    land inside the requested window. `trend` = (slope, intercept) from fit_trend();
    subtracting the whole-recording trend matches one detrend of the full trace.
    Returns an array of shape (n_samples, len(channel_ids)).
    """
    fs = recording.get_sampling_frequency()
    n_total = recording.get_num_samples()
    pad = int(round(NOTCH_MARGIN_S * fs)) if APPLY_TIME_NOTCH else 0
    r0 = max(0, start_frame - pad)
    r1 = min(n_total, end_frame + pad)

    x = recording.get_traces(
        start_frame=r0, end_frame=r1, channel_ids=channel_ids, return_in_uV=True
    ).astype(np.float64)
    if APPLY_TIME_NOTCH:
        x = notch_filter(x, fs, LINE_HARMONICS, NOTCH_Q)
    x = x[start_frame - r0: end_frame - r0]

    if APPLY_TIME_DETREND and trend is not None:
        slope, intercept = trend
        t = np.arange(start_frame, end_frame) / fs
        x = x - (t[:, None] * slope[None, :] + intercept[None, :])
    return x


def _robust_high_outliers(x, thresh, ref_mask=None):
    """Boolean mask of samples that are high outliers by robust (MAD) z-score.

    If `ref_mask` is given, the median/MAD reference statistics are estimated
    from `x[ref_mask]` only (e.g. epochs that already passed the delta/theta
    filter), so epochs excluded upstream can't skew the robust threshold --
    z-scores are still returned for every element of `x`.
    """
    x   = np.asarray(x, dtype=np.float64)
    ref = x if ref_mask is None else x[np.asarray(ref_mask, dtype=bool)]
    if ref.size == 0:
        ref = x
    med = np.median(ref)
    mad = np.median(np.abs(ref - med))
    if mad == 0:
        mad = 1e-20
    robust_z = 0.6745 * (x - med) / mad
    return robust_z > thresh


# ---------------------------------------------------------------------------
# Whole-recording scan: per-epoch cleaning metrics + theta power
# ---------------------------------------------------------------------------
def scan_recording(recording, channel_ids, trend=None, epoch_s=EPOCH_S, band=THETA_BAND,
                   low_band=LOW_BAND, block_s=SCAN_BLOCK_S):
    """Stream the whole recording in epochs; return per-epoch cleaning metrics + theta power.

    Reads the depth-sequence channels in blocks of `block_s` seconds (memory bounded),
    notch-filtered and detrended via read_clean_traces, splits each block into
    `epoch_s` epochs and computes one Welch periodogram per epoch per channel (Hann
    window, epoch-long segment, no overlap, constant detrend -- as in the tetrode
    pipeline). For every epoch:
      * theta-band power per channel (linear, uV^2) -> depth power profile;
      * delta (`low_band`) and theta power, median across channels -> delta/theta
        filter and MAD criterion;
      * broadband peak-to-peak amplitude, median across channels -> MAD criterion
        and snippet ranking.
    Band power is sum(Pxx) * df over the band, matching the tetrode pipeline.
    """
    fs = recording.get_sampling_frequency()
    n_total = recording.get_num_samples()
    n_ch = len(channel_ids)
    n_ep = int(round(epoch_s * fs))
    n_epochs = n_total // n_ep
    if n_epochs == 0:
        raise ValueError("Recording shorter than one epoch.")

    p2p = np.empty(n_epochs, dtype=np.float64)
    delta_pow = np.empty(n_epochs, dtype=np.float64)
    theta_pow = np.empty(n_epochs, dtype=np.float64)
    band_lin = np.empty((n_epochs, n_ch), dtype=np.float64)  # per-epoch, per-channel theta power

    win = signal.get_window("hann", n_ep)
    epochs_per_block = max(1, int(round(block_s / epoch_s)))
    print(f"  Scanning {n_epochs} epochs of {epoch_s:g}s ({n_epochs * epoch_s / 60:.1f} min total), "
          f"notch={'on' if APPLY_TIME_NOTCH else 'off'}, "
          f"detrend={'on' if (APPLY_TIME_DETREND and trend is not None) else 'off'}...")

    next_report = 0.1
    for e0 in range(0, n_epochs, epochs_per_block):
        e1 = min(n_epochs, e0 + epochs_per_block)
        ne = e1 - e0
        block = read_clean_traces(recording, e0 * n_ep, e1 * n_ep, channel_ids, trend)
        block = block.reshape(ne, n_ep, n_ch)  # (epochs, samples, ch)

        p2p[e0:e1] = np.median(block.max(axis=1) - block.min(axis=1), axis=1)

        freqs, pxx = signal.welch(block, fs=fs, window=win, nperseg=n_ep, noverlap=0,
                                  detrend="constant", axis=1)  # (epochs, nfreq, ch)
        df = freqs[1] - freqs[0]
        m_low = (freqs >= low_band[0]) & (freqs <= low_band[1])
        m_theta = (freqs >= band[0]) & (freqs <= band[1])
        th = pxx[:, m_theta, :].sum(axis=1) * df  # (epochs, ch)
        dl = pxx[:, m_low, :].sum(axis=1) * df

        band_lin[e0:e1] = th
        theta_pow[e0:e1] = np.median(th, axis=1)
        delta_pow[e0:e1] = np.median(dl, axis=1)

        if e1 / n_epochs >= next_report:
            print(f"    ...{100 * e1 / n_epochs:.0f}%")
            next_report += 0.1

    return dict(p2p=p2p, delta_pow=delta_pow, theta_pow=theta_pow, band_lin=band_lin,
                ids=list(channel_ids), n_bin=n_ep, fs=fs, n_bins=n_epochs)


def clean_epochs(scan, mad_thresh=MAD_THRESH, apply_delta_theta_filter=APPLY_DELTA_THETA_FILTER):
    """Epoch rejection with the same stages and order as compute_psd_clean_epochs in
    the tetrode pipeline (minus running-speed gating -- no tracking here):

      1. Delta/theta filter -- an epoch is rejected if its delta (LOW_BAND) power
         exceeds its THETA_BAND power.
      2. Dual-criteria MAD outlier rejection -- an epoch is rejected if it is a high
         outlier (robust z > mad_thresh) on EITHER peak-to-peak amplitude OR delta
         power. Median/MAD are computed ONLY from epochs surviving (1).

    Dead (all-zero) epochs are always rejected. If no epoch survives, all epochs are
    kept (as the tetrode pipeline does without speed gating) and a warning is printed.
    Returns (keep_mask, stats_dict).
    """
    p2p, delta, theta = scan["p2p"], scan["delta_pow"], scan["theta_pow"]
    n_total = p2p.size

    keep = p2p > 0
    n_dead = int((~keep).sum())

    n_dt = 0
    if apply_delta_theta_filter:
        dt_fail = keep & (delta > theta)
        n_dt = int(dt_fail.sum())
        keep &= ~dt_fail

    reject = _robust_high_outliers(p2p, mad_thresh, ref_mask=keep) | \
             _robust_high_outliers(delta, mad_thresh, ref_mask=keep)
    n_mad = int((keep & reject).sum())
    keep &= ~reject

    if not keep.any():
        print("  WARNING: no epochs pass the delta/theta + MAD criteria -> using all epochs.")
        keep = np.ones(n_total, dtype=bool)

    stats = dict(n_total=n_total, n_dead=n_dead, n_delta_theta=n_dt,
                 n_mad=n_mad, n_kept=int(keep.sum()))
    return keep, stats


def pick_snippet_starts(metric, good, n_bin, fs, n_snippets, snippet_len_s):
    """Choose `n_snippets` snippet start times (s), spread across the recording.

    The recording is split into `n_snippets` equal parts; in each part the
    least-noisy fully-clean `snippet_len_s` window is chosen (falling back to the
    least-noisy window if none is fully clean).
    """
    n_bins = len(metric)
    # ceil: the checked window must cover the whole snippet (e.g. 5 s -> three 2 s epochs)
    snip_bins = max(1, int(np.ceil(snippet_len_s * fs / n_bin - 1e-9)))
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
def get_window_traces(recording, start_s, duration_s, channel_ids, trend=None):
    """Return (traces, time_vector, fs) for a window of the recording.

    traces has shape (n_samples, len(channel_ids)) in microvolts, notch-filtered and
    detrended exactly like the whole-recording scan (read_clean_traces). The window
    is clamped to the available data.
    """
    fs = recording.get_sampling_frequency()
    n_total = recording.get_num_samples()
    n_win = min(int(duration_s * fs), n_total)
    start = int(start_s * fs)
    start = max(0, min(start, n_total - n_win))

    traces = read_clean_traces(recording, start, start + n_win, channel_ids, trend)
    t = np.arange(n_win) / fs
    return traces, t, fs


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
# Cross-recording average (absolute-depth alignment, see module docstring)
# ---------------------------------------------------------------------------
def _interp_with_range(grid, x, y):
    """Linearly interpolate y(x) onto `grid`; nan outside [min(x), max(x)] (no extrapolation)."""
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return np.full(grid.shape, np.nan, dtype=float)
    out = np.interp(grid, x, y)
    out[(grid < x.min()) | (grid > x.max())] = np.nan
    return out


def compute_average_profile(results, depth_step=AVERAGE_DEPTH_BIN_UM, band=THETA_BAND):
    """Average theta power and phase across recordings, aligned by absolute depth (um).

    Each recording's power_db and phase (as R*exp(i*phase) on the unit circle, to
    respect circularity) is interpolated onto a common depth grid spanning the union
    of all recordings' depths, then combined:
      * power -> plain mean (dB) across recordings with data at that depth.
      * phase -> circular mean (vector sum of the interpolated unit-circle values),
        matching how compute_phase_profile combines snippets within one recording.

    Returns a dict with the common grid, the averaged profiles, and each recording's
    own depth-interpolated profile (for overlaying as faint per-recording lines).
    """
    all_depths = np.concatenate(
        [np.array([d for d, _ in r["resolved"]], dtype=float) for r in results]
    )
    d_min, d_max = np.nanmin(all_depths), np.nanmax(all_depths)
    n_grid = max(2, int(round((d_max - d_min) / depth_step)) + 1)
    grid = np.linspace(d_min, d_max, n_grid)

    n_rec = len(results)
    power_stack = np.full((n_rec, grid.size), np.nan)
    phase_deg_stack = np.full((n_rec, grid.size), np.nan)
    lag_ms_stack = np.full((n_rec, grid.size), np.nan)
    R_stack = np.full((n_rec, grid.size), np.nan)
    T_means = np.full(n_rec, np.nan)

    for i, r in enumerate(results):
        depths_pos = np.array([d for d, _ in r["resolved"]], dtype=float)
        order = np.argsort(depths_pos, kind="stable")
        depths_sorted = depths_pos[order]
        power_stack[i] = _interp_with_range(grid, depths_sorted, r["power_db"][order])

        phase = r["phase"]
        if phase is None:
            continue
        T = phase["T_mean"]
        T_means[i] = T
        z = phase["R"][order] * np.exp(1j * np.deg2rad(phase["phase_deg"][order]))
        z_re = _interp_with_range(grid, depths_sorted, z.real)
        z_im = _interp_with_range(grid, depths_sorted, z.imag)
        z_grid = z_re + 1j * z_im
        phase_deg_stack[i] = np.degrees(np.angle(z_grid))
        R_stack[i] = np.abs(z_grid)
        if T and T > 0:
            lag_ms_stack[i] = np.deg2rad(phase_deg_stack[i]) / (2 * np.pi) * T * 1000.0

    avg_power_db = np.nanmean(power_stack, axis=0)
    n_power = np.sum(np.isfinite(power_stack), axis=0)

    z_vecs = np.where(np.isfinite(phase_deg_stack) & np.isfinite(R_stack),
                       R_stack * np.exp(1j * np.deg2rad(phase_deg_stack)), 0.0)
    n_phase = np.sum(np.isfinite(phase_deg_stack), axis=0)
    z_sum = z_vecs.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        avg_phase_deg = np.where(n_phase > 0, np.degrees(np.angle(z_sum)), np.nan)
        avg_R = np.where(n_phase > 0, np.abs(z_sum) / np.maximum(n_phase, 1), np.nan)

    T_avg = float(np.nanmean(T_means)) if np.isfinite(T_means).any() else 2.0 / (band[0] + band[1])
    avg_lag_ms = np.deg2rad(avg_phase_deg) / (2 * np.pi) * T_avg * 1000.0

    return dict(
        grid_depth=grid,
        avg_power_db=avg_power_db, n_power=n_power,
        avg_phase_deg=avg_phase_deg, avg_lag_ms=avg_lag_ms, avg_R=avg_R, n_phase=n_phase,
        T_avg=T_avg,
        power_stack=power_stack, lag_ms_stack=lag_ms_stack,
    )


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

    channel_ids = [cid for _, cid in resolved]

    # --- whole-recording trend (for detrending), scan + epoch cleaning ---
    trend = None
    if APPLY_TIME_DETREND:
        print(f"  Fitting whole-recording {DETREND_TYPE} trend on {len(channel_ids)} channels...")
        trend = fit_trend(recording, channel_ids)
    scan = scan_recording(recording, channel_ids, trend)
    good, stats = clean_epochs(scan)
    frac_bad = 1.0 - good.mean()
    print(f"  Epochs kept: {stats['n_kept']}/{stats['n_total']} "
          f"({frac_bad * 100:.1f}% rejected: {stats['n_dead']} dead, "
          f"{stats['n_delta_theta']} delta>theta, {stats['n_mad']} MAD outliers).")

    id_to_col = {cid: i for i, cid in enumerate(scan["ids"])}

    # theta power per channel over the ENTIRE recording (clean bins only)
    mean_lin = scan["band_lin"][good].mean(axis=0)  # (n_ch,) linear uV^2
    power_db = np.array([
        10.0 * np.log10(mean_lin[id_to_col[cid]] + 1e-12) if cid is not None else np.nan
        for _, cid in resolved
    ])

    # --- ten 5 s snippets spread through the recording (clean windows) ---
    starts = pick_snippet_starts(
        scan["p2p"], good, scan["n_bin"], scan["fs"], N_SNIPPETS, SNIPPET_DURATION_S
    )
    snippets, pooled = [], []
    for st in starts:
        snip, t, fs = get_window_traces(recording, st, SNIPPET_DURATION_S, channel_ids, trend)
        theta = bandpass_theta(snip, fs)
        theta_by_pos = [theta[:, j] for j in range(len(resolved))]  # columns follow `resolved`
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
        f"{res['frac_bad'] * 100:.1f}% epochs rejected (delta/theta + MAD)",
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


def plot_average_figure(avg, results, band=THETA_BAND, output_dir=OUTPUT_DIR):
    """Overall figure: average theta power & phase vs absolute depth (um) across ALL
    recordings, with each recording's own depth-interpolated profile drawn faintly
    behind the (bold) average. See "Multiple recordings and the cross-recording
    average" in the module docstring for how recordings are aligned/combined.
    """
    grid = avg["grid_depth"]
    fig, (ax_pow, ax_pha) = plt.subplots(1, 2, figsize=(9, 6), sharey=True)

    for i, r in enumerate(results):
        ax_pow.plot(avg["power_stack"][i], grid, "-", color=r["color"], lw=1.0,
                    alpha=0.45, label=r["label"])
    ax_pow.plot(avg["avg_power_db"], grid, "-o", color="k", ms=4, lw=2.2, label="Average")
    ax_pow.set_xlabel("Theta power (dB)")
    ax_pow.set_ylabel("Depth (µm)")
    ax_pow.set_title(f"Theta power\n(average of {len(results)} recordings)", fontsize=10)
    ax_pow.legend(fontsize=7, loc="best", frameon=False)

    for i, r in enumerate(results):
        ax_pha.plot(avg["lag_ms_stack"][i], grid, "-", color=r["color"], lw=1.0, alpha=0.45)
    ax_pha.plot(avg["avg_lag_ms"], grid, "-o", color="k", ms=4, lw=2.2)
    ax_pha.axvline(0, color="k", lw=0.5)
    ax_pha.set_xlabel("Theta lag (ms)")
    ax_pha.set_title(f"Theta phase\n(average of {len(results)} recordings)", fontsize=10)

    T = avg["T_avg"]
    if T and T > 0:
        secax = ax_pha.secondary_xaxis(
            "top",
            functions=(lambda x: x / 1000.0 / T * 360.0,
                       lambda d: d / 360.0 * T * 1000.0),
        )
        secax.set_xlabel("phase (deg)", fontsize=8)
        secax.tick_params(labelsize=7)

    # Larger depth value on top, matching the per-recording figures' convention
    # (resolve_depth_sequence puts the channel nearest the first, larger `start`
    # depth at the top of the stack for a descending depth_sequence).
    ax_pow.set_ylim(grid.max(), grid.min())

    fig.suptitle(
        f"Average across recordings   |   theta {band[0]:g}-{band[1]:g} Hz   |   "
        f"aligned by absolute depth ({AVERAGE_DEPTH_BIN_UM:g} µm grid)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = os.path.join(output_dir, "lfp_theta_depth_AVERAGE_all_recordings.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved average figure: {fig_path}")
    if not SHOW_FIGURES:
        plt.close(fig)
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

    results = []
    for cfg, color in zip(FILES, palette):
        recording = get_processed_recording(cfg, job_kwargs=job_kwargs, device=device)
        res = build_result(cfg, recording, color)
        plot_recording_figure(res)
        results.append(res)

    if len(results) >= 2:
        print(f"\nComputing average power/phase across {len(results)} recordings "
              f"(aligned by absolute depth)...")
        avg = compute_average_profile(results)
        plot_average_figure(avg, results)
    else:
        print("\nOnly one recording processed -> skipping the cross-recording average figure.")

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