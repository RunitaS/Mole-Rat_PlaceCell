# -*- coding: utf-8 -*-
# ^ "Encoding declaration". Python reads this special comment on line 1/2 and decodes the
#   source file as UTF-8, which is needed because the file contains non-ASCII characters
#   (µ, Σ, ², the box-drawing ─ characters used in section headers, etc.).
#
# =============================================================================================
# READING GUIDE  (all comments in this "_Commented" copy were added as line-by-line annotations;
# the executable code itself is identical to PlaceCell_Main_v15.py)
# =============================================================================================
# What this script does, per sorted unit (.ntt file) in every session folder under root_folder:
#   1. Load the tracking file (time, x, y), remove tracker glitches, convert to cm, smooth x/y.
#   2. Remove immobile frames (speed < IMMOBILITY_SPEED_CMS) and the spikes fired during them,
#      so firing during rest (e.g. sharp-wave ripples) does not contaminate the spatial map.
#   3. Divide the arena into square bins (target_bin_cm x target_bin_cm) and build
#        occupancy map  = seconds spent in each bin
#        spike map      = number of spikes fired in each bin
#        rate map       = spike map / occupancy map  (Hz)  -> "raw" map, plus a Gaussian-smoothed copy.
#   4. Compute single-cell spatial metrics:
#        peak / mean firing rate, Skaggs spatial information (SIR, bits/spike), sparsity,
#        spatial coherence, split-half stability, theta modulation, speed modulation.
#   5. Significance by circular-shift shuffling ("permutation test"): the spike train is shifted
#      in time relative to the position track 1000 times, SI is recomputed each time to form a
#      null distribution, and the real SI must exceed that null's 95th percentile (p < 0.05).
#   6. Place cell  = n_spikes > 50  AND  1 Hz < peak rate < 25 Hz  AND  SI > 0.5 bits/spike
#                    AND  SI is shuffle-significant.
#   7. For place cells, place fields are isolated with a "threshold method" (bins >= 20 % of the
#      peak rate and above the mean rate, contiguous, >= 9 bins); the place-cell verdict can be
#      overruled if no valid field backs the peak, or if a field covers > 50 % of the arena.
#   8. Everything is written to an Excel workbook; figures are saved next to each .ntt file;
#      files of confirmed place cells are copied to Output_PlaceTrue.
#
# How this maps onto the reference paper (Zhang, Donoghue, Qasim & Jacobs 2025, bioRxiv
# doi:10.1101/2025.08.29.672705, "Evaluating Place Cell Detection Methods in Rats and Humans"):
#   * Spatial information (paper Eq. 5):  SI = Σ_x p(x) · (λ(x)/λ̄) · log2(λ(x)/λ̄)
#       p(x) = occupancy probability of bin x, λ(x) = firing rate in bin x, λ̄ = mean rate.
#     This is exactly what _compute_sir() computes (originally Skaggs et al. 1993).
#   * The paper (Table 1) shows rodent studies classify place cells either with a fixed SI
#     threshold (SI > 0.25 or SI > 0.5 bits/spike), a permutation test (p < 0.05 against
#     circularly shifted spike trains), or both. This script uses BOTH (SI > 0.5 AND p < 0.05),
#     the same combination as e.g. Duvelle et al. 2021 and Jin & Lee 2021 in that table.
#   * The paper's main finding: SI mostly reflects the CONTRAST between peak and average firing
#     (sharp, high-contrast fields), and is relatively INSENSITIVE to trial-to-trial consistency;
#     ANOVA-based methods (used in human studies) capture consistency instead. This script has no
#     ANOVA; its closest consistency measure is split-half stability (computed and reported, but
#     NOT part of the place-cell rule).
#   * Paper Fig. S2/S3: smoothing deflates SI and finer bins inflate SI, so FIXED thresholds depend
#     on bin size and smoothing, whereas permutation tests are more robust. Here SI is computed on
#     the RAW (unsmoothed) 2-cm map, so the 0.5 bits/spike threshold is specific to these settings;
#     the shuffle test is the more portable part of the criterion.
#   * The paper's "place field width" feature (bins above 20 % of peak) is the same 20 % rule used
#     by the field-detection "threshold method" below (METHOD2_RATE_THRESHOLD_FRAC = 0.20).
#
# Python syntax conventions used throughout (explained once here, referred to later):
#   * `name: type = value`      -> a variable with a type hint (hints are not enforced at runtime).
#   * `def f(a: int) -> float:` -> function with typed parameter and return-type hint.
#   * `X | None`                -> type hint meaning "X or None" (Python 3.10+ syntax).
#   * Leading underscore (_name)-> convention meaning "private / internal helper".
#   * ALL_CAPS names            -> convention for module-level constants (configuration).
#   * np.xxx / pd.xxx           -> functions from NumPy (arrays) / pandas (tables).
#   * Boolean mask indexing     -> arr[mask] keeps only elements where mask is True.
# =============================================================================================
"""
Batch place cell analysis – V14

Directories to edit: see "Directories" block right below this docstring.

Corrections applied:
Metric>    Before>    Now
SIR, and its bootstrap    occupancy-weighted> smoothed map>    raw
Sparsity>    smoothed>    raw
Mean firing rate>    smoothed>    raw
Peak firing rate>    raw>    raw (unchanged)
Coherence, and its bootstrap>    smoothed>    smoothed (unchanged)
Split-half stability>    raw>    smoothed

Also added minimum velocity filter to remove non-spatial firing during immobility (e.g. sharp-wave ripples). See compute_metrics step 2c.
"""
# ^ Module docstring: a triple-quoted string as the first statement of the file. Python stores it
#   in __doc__; it documents which ratemap (raw vs smoothed) each metric uses.
#   NOTE (review): the docstring says "V14" but this is the v15 script.

# ── Standard-library imports ─────────────────────────────────────────────────────────────────
# `import X` loads module X and makes it available under the name X.
# os: operating-system utilities -- file paths (os.path.join, os.walk), creating folders.
import os
# shutil: high-level file operations; used at the end to copy .ntt/tracking files (shutil.copy2).
import shutil
# threading: thread primitives; used for a Lock (serialised printing) and a Semaphore (GPU access).
import threading
# time: used for time.sleep() while waiting for the GPU to become less busy.
import time
# concurrent.futures: ThreadPoolExecutor, used to analyse several units in parallel.
import concurrent.futures
# types: provides SimpleNamespace, used as an empty placeholder object when CuPy/pynvml are missing.
import types
# random: Python's random-number generator; random.randint() picks the circular-shift offsets
#   for the shuffle (bootstrap) tests. NOTE (review): it is never seeded, so shuffle p-values
#   differ slightly from run to run (not bit-for-bit reproducible).
import random
# `from X import Y` imports only name Y from module X. datetime gives the current date/time,
#   used to timestamp the run-metadata CSV.
from datetime import datetime
# typing helpers: TYPE_CHECKING is False at runtime but True for static type checkers (Pylance);
#   NamedTuple builds small immutable record classes; TypedDict describes dicts with fixed keys.
from typing import TYPE_CHECKING, NamedTuple, TypedDict

# ── Third-party scientific imports ──────────────────────────────────────────────────────────
# `import numpy as np` imports NumPy under the short alias np (array maths; used everywhere).
import numpy as np
# pandas (alias pd): tables (DataFrames); reads tracking CSV/XLSX and writes the Excel output.
import pandas as pd
# scipy.ndimage.convolve: N-dimensional convolution (2-D Gaussian smoothing of rate maps on CPU).
# gaussian_filter1d: 1-D Gaussian smoothing (position traces, speed, instantaneous firing rate).
from scipy.ndimage import convolve, gaussian_filter1d
# pearsonr: Pearson correlation coefficient r and its parametric p-value.
# `t as _t_dist`: imports Student's t distribution renamed (aliased) to _t_dist; used to get the
#   p-value of the weighted regression slope.
from scipy.stats import pearsonr, t as _t_dist

# Thread-safe Matplotlib imports for parallel rendering
# Figure: matplotlib's figure object, created directly (object-oriented API) instead of via
#   pyplot. pyplot keeps global state and is NOT thread-safe; because units are processed in
#   parallel threads, every plot here is built on its own Figure object.
from matplotlib.figure import Figure
# FigureCanvasAgg: the "Agg" raster backend that renders a Figure to a PNG without any GUI window.
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ── Directories (edit these per run) ──────────────────────────────────────────
# r'...' is a "raw string": backslashes are NOT escape characters, convenient for Windows paths.

# root_folder: top-level folder searched recursively (os.walk) for session folders that contain
#   one tracking file plus one or more .ntt (Neuralynx tetrode spike) files.
root_folder  = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt'
# output_excel: path of the results workbook (sheets Full / First_Half / Second_Half /
#   PlaceFields / Speed_Summary). A *_run_metadata.csv is also written next to it.
output_excel = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\All_TT_PlaceChar_SirSparADptBin_Corrected.xlsx'

# Destination for .ntt + tracking files of confirmed place cells (folder pattern
# replicated from the animal-ID folder onwards, e.g. Fa1059/Open/<session>/...)
# Output_PlaceTrue: root folder into which confirmed place cells' files are copied at the end.
Output_PlaceTrue = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\PC_True_irSparADptBin_Corrected'


# Per-unit figure subfolders, created next to each .ntt file
# Name of the sub-folder that receives the speed-modulation figures (2x2 panel PNG per unit).
SPEED_MOD_SUBDIR = 'speed modulation_Sir_Spar_AdptBin_Corrected'
# Name of the sub-folder that receives the SI / coherence shuffle-histogram figures.
SHUFFLING_SUBDIR = 'shuffling_Sir_Spar_AdptBin_Corrected'
# Name of the sub-folder that receives the trajectory + rate-map figures.
RATEMAPS_SUBDIR  = 'ratemaps_Sir_Spar_AdptBin_Corrected'

# ── Which ratemap each metric uses ────────────────────────────────────────────
# RAW (unsmoothed) ratemap : SIR, sparsity, peak_fr, mean_fr, and the SIR
#                            bootstrap shuffles.
# SMOOTHED ratemap         : coherence, split-half stability, the coherence
#                            bootstrap shuffles, and place-field extraction.
#
# USE_SI_MIN_OCC = True restricts the SIR and sparsity sums to valid bins with
# >= SI_MIN_OCC_S of occupancy (applied identically to the real data and every
# bootstrap shuffle, see _compute_sir / _compute_sparsity). False: all valid bins.
# USE_SI_MIN_OCC (bool): if True, only bins visited for >= SI_MIN_OCC_S seconds enter the SI and
#   sparsity sums. Rationale: in a bin with very little occupancy one stray spike gives a huge,
#   unreliable rate, which inflates SI (SI rewards bins whose rate is far above the mean).
USE_SI_MIN_OCC = True
# SI_MIN_OCC_S (float, seconds): that minimum occupancy (0.5 s = 15 frames at 30 fps).
SI_MIN_OCC_S   = 0.5     # min occupancy (s) for a bin to enter the SIR and sparsity sums


# `class _Metrics(TypedDict, total=False):` declares the shape of the per-unit metrics dict for
#   static type checkers only (no effect at runtime). total=False means every key is optional.
#   Each line below is `key: type`; `int | None` means "an int, or None if not computed".
class _Metrics(TypedDict, total=False):
    # number of spikes kept for spatial analysis (matched to a moving tracking frame)
    n_spikes:        int | None
    # spikes discarded because no tracking frame lies within MAX_GAP_US (50 ms) of them
    n_discarded:     int | None
    # spikes discarded because they occurred while the animal was immobile
    n_spikes_immobile: int | None
    # highest firing rate (Hz) in any valid bin of the RAW rate map
    peak_fr:         float | None
    # occupancy-weighted mean firing rate (Hz) over valid bins of the RAW map
    mean_fr:         float | None
    # Skaggs spatial information rate, bits/spike ("SIR" = spatial information rate)
    sir:             float | None
    # Skaggs sparsity (0..1; small = firing confined to a small part of the arena)
    sparsity:        float | None
    # spatial coherence (Fisher-z of neighbour correlation, Muller & Kubie 1989 style)
    coherence:       float | None
    # Pearson r between first-half and second-half smoothed rate maps
    stability_score:    float | None
    # parametric p-value of that Pearson r
    stability_p_value:  float | None
    # number of bins valid in both halves that entered the stability correlation
    stability_n_bins:   int | None
    # mean SI of the 1000 circular-shift shuffles (null distribution mean)
    bootstrap_mean:  float | None
    # 95th percentile of the shuffled SI distribution (the significance cut-off)
    bootstrap_p95:   float | None
    # True if the real SI > bootstrap_p95, i.e. SI significant at p < 0.05 (one-sided)
    bootstrap_sig:   bool | None
    # same three quantities for spatial coherence, from the same shuffles
    coherence_bootstrap_mean: float | None
    coherence_bootstrap_p95:  float | None
    coherence_bootstrap_sig:  bool | None
    # True if the spike autocorrelogram has a dominant 3-7 Hz rhythm
    theta_modulated: bool | None
    # frequency (Hz) of the largest spectral peak inside 3-7 Hz
    theta_peak_freq: float | None
    # ---- binned speed-modulation fit (rate vs speed, speed-binned medians) ----
    # r of the weighted linear fit of binned firing rate vs running speed
    speed_score:     float | None
    # parametric p-value of that slope
    speed_p_value:   float | None
    # r squared of that fit
    speed_r2:        float | None
    # slope (Hz per cm/s) of the fit
    speed_beta:      float | None
    # intercept (Hz at 0 cm/s) of the fit
    speed_f0:        float | None
    # True if speed_p_value < 0.05 (parametric)
    speed_modulated: bool | None
    # mean, 2.5th and 97.5th percentiles of the shuffled speed scores
    speed_shuffle_mean: float | None
    speed_shuffle_lo:   float | None
    speed_shuffle_hi:   float | None
    # two-tailed shuffle p-value of speed_score
    speed_shuffle_p:    float | None
    # True if the shuffle test is significant (p < 0.05)
    speed_modulated_shuffle: bool | None
    # True if the shuffle was actually run (only when the parametric test passed first)
    speed_shuffle_ran:  bool | None
    # ---- time-domain speed score (Pearson r of speed vs rate time series) -------------
    speed_score_td:      float | None
    speed_p_value_td:    float | None
    speed_r2_td:          float | None
    speed_shuffle_mean_td: float | None
    speed_shuffle_lo_td:   float | None
    speed_shuffle_hi_td:   float | None
    speed_shuffle_p_td:    float | None
    speed_modulated_td:    bool | None
    speed_shuffle_ran_td:  bool | None
    # positively speed-modulated cell (score above the shuffle 97.5th percentile)
    p_speed:         bool | None
    # negatively speed-modulated cell (score below the shuffle 2.5th percentile)
    n_speed:         bool | None
    # average of speed_score and speed_score_td
    final_speed_score: float | None
    speed_cell:      bool | str | None   # True / False / 'not tested'
    # final place-cell verdict (True / False / None if it could not be evaluated)
    place_cell:      bool | None
    # session folder path relative to root_folder
    session:         str
    # .ntt file name of the unit
    unit:            str
    # position of this unit in the job list (used to restore the original order of results)
    job_order:       int

# ── GPU availability ──────────────────────────────────────────────────────────
# The 2-D Gaussian convolutions can optionally run on an NVIDIA GPU via CuPy (a NumPy-like
# library for CUDA). The blocks below detect whether that is possible.

# `if TYPE_CHECKING:` is only True for static type checkers; these imports exist purely so the
#   editor (Pylance) knows what cp / cp_convolve / pynvml are. They are never executed.
if TYPE_CHECKING:
    # cupy = GPU array library with the same API as NumPy
    import cupy as cp                                          # type: ignore
    # GPU version of scipy.ndimage.convolve
    from cupyx.scipy.ndimage import convolve as cp_convolve    # type: ignore
    # NVIDIA Management Library bindings (reads GPU utilisation)
    import pynvml                                              # type: ignore

# try / except: run the `try` block; if an exception (error) of a listed type occurs, jump to
#   the matching `except` block instead of crashing.
try:
    # attempt to import CuPy (raises ImportError if it is not installed)
    import cupy as cp                                          # type: ignore[import-untyped]
    # attempt to import the GPU convolution function
    from cupyx.scipy.ndimage import convolve as cp_convolve    # type: ignore[import-untyped]
    # Validate that CUDA JIT (nvrtc) actually works before committing to GPU mode
    # make a 3x3 array of zeros on the GPU (float64 = double precision)
    _t = cp.zeros((3, 3), dtype=cp.float64)
    # make a 3x3 averaging kernel (all ones divided by 9) on the GPU
    _k = cp.ones((3, 3), dtype=cp.float64) / 9.0
    # run one tiny convolution; if CUDA compilation is broken this raises an exception
    cp_convolve(_t, _k, mode='constant')
    # `del` removes the temporary names (frees the small GPU arrays)
    del _t, _k
    # module-level flag: True -> GPU path is used in _convolve_pair / _gaussian_smooth_circular
    _GPU = True
    # report the mode to the console
    print("CuPy detected – GPU (CUDA) acceleration enabled.")
# ImportError: CuPy is not installed at all
except ImportError:
    # SimpleNamespace() is an empty object; `cp` is given a harmless placeholder value
    cp          = types.SimpleNamespace()                      # type: ignore[assignment]
    # Assign dummy lambda to prevent "None cannot be called" Pylance errors
    # `lambda *args, **kwargs: None` is an anonymous function accepting any arguments and
    #   returning None (never actually called because _GPU is False).
    cp_convolve = lambda *args, **kwargs: None                 # type: ignore[assignment]
    # CPU mode
    _GPU = False
    print("CuPy not found – running on CPU (install cupy-cuda12x to enable GPU).")
# `except Exception as _gpu_err` catches any other error and binds it to the name _gpu_err
except Exception as _gpu_err:
    # CuPy imported but CUDA JIT unavailable (e.g. missing nvrtc*.dll)
    cp          = types.SimpleNamespace()                      # type: ignore[assignment]
    cp_convolve = lambda *args, **kwargs: None                 # type: ignore[assignment]
    _GPU = False
    # f"...{expr}..." is an f-string: the expression in braces is evaluated and inserted
    print(f"CuPy found but GPU JIT unavailable ({_gpu_err}) – falling back to CPU.")

# Second optional dependency: pynvml lets the script read how busy GPU 0 currently is.
try:
    import pynvml                                              # type: ignore[import-untyped]
    # initialise the NVIDIA management library
    pynvml.nvmlInit()
    # flag: GPU utilisation can be queried
    _NVML        = True
    # handle (reference) to GPU number 0
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
# any failure (not installed, no NVIDIA driver) -> utilisation monitoring disabled
except Exception:
    pynvml       = types.SimpleNamespace()                     # type: ignore[assignment]
    _nvml_handle = None
    _NVML        = False


# _gpu_util_pct() -> int : returns current GPU utilisation in percent (0 if unavailable).
def _gpu_util_pct() -> int:
    # `not _NVML` is True when monitoring is unavailable -> report 0 %
    if not _NVML:
        return 0
    try:
        # query utilisation; `.gpu` is the % of time the GPU was busy over the last sample period
        return int(pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle).gpu)
    except Exception:
        # any query error -> 0 %
        return 0


# ── Configuration ─────────────────────────────────────────────────────────────
# (input/output directories are set at the top of the file)

# fps: video tracking frame rate (frames per second). Used for frame duration (1/fps s), for
#   converting the 20-s shuffle margin into frames, and for the occupancy-time cap.
fps           = 30           # tracking frame rate (Hz)
# target_bin_cm: side length of each square spatial bin of the rate map (2 x 2 cm).
#   Paper (Fig. S3): finer bins -> higher SI values, so the SI threshold depends on this choice.
target_bin_cm  = 2.0          # bin size in cm
# arena_width_cm: real arena width; only used to convert pixels -> cm when COORD_UNITS == 'pixel'.
arena_width_cm = 80.0         # physical arena width in cm
# ── Valid-bin criteria (each can be switched on/off independently) ──────────
# A bin is valid only if it passes every enabled criterion. If both are False,
# any bin with non-zero occupancy is valid.
# use_min_occ_s: if True, a bin needs >= min_occ_s seconds of total occupancy to be "valid".
use_min_occ_s   = False        # True: exclude bins with < min_occ_s seconds occupancy
min_occ_s       = 0.5          #       occupancy threshold (s)
# use_min_visits: if True, a bin needs to have been entered at least min_visits separate times.
#   This guards against a bin that was crossed only once from producing a spurious "field".
use_min_visits  = True         # True: exclude bins visited < min_visits times
min_visits      = 2            #       a "visit" = one contiguous entry into the bin
# MAX_GAP_US: a spike is assigned to the nearest tracking frame only if that frame is within
#   50 000 µs (50 ms); otherwise the position is unknown and the spike is discarded.
#   (Neuralynx timestamps are in microseconds; 50_000 -- the underscore is just a digit separator.)
MAX_GAP_US      = 50_000       # max spike–position gap in µs (50 ms)
# N_BOOTSTRAP: number of circular-shift shuffles for the SI (and coherence) significance test.
#   The paper also uses 1,000 surrogates with a 95th-percentile criterion.
N_BOOTSTRAP    = 1000         # circular-shift shuffles for SIR significance
# SIR_SHUFFLE_MARGIN_S: each shuffle shifts spikes by at least 20 s (and at most session - 20 s),
#   so that a shuffle never leaves spikes almost aligned with their true positions.
SIR_SHUFFLE_MARGIN_S = 20.0   # min circular-shift offset (s) from either end for SIR shuffling (matches Fenton reference)

# STABILITY_MIN_BINS: the split-half correlation is only reported if at least this many bins are
#   valid in both halves (a correlation over 2-4 points is meaningless).
STABILITY_MIN_BINS = 5        # min jointly-occupied bins (>= min_occ_s in BOTH halves)
                               # required before a split-half stability r is reported

# Theta-modulation test parameters (see _compute_theta_modulation):
# half-width of the spike autocorrelogram: lags from -500 ms to +500 ms
AUTOCORR_WINDOW_MS  = 500.0   # autocorrelogram half-window (ms)
# autocorrelogram bin width; 5 ms bins -> "sampling rate" of 200 Hz for the FFT
AUTOCORR_BIN_MS     = 5.0     # autocorrelogram bin size (ms)
# a cell is theta-modulated if its 3-7 Hz peak power > 2 x the mean power of the whole spectrum
THETA_POWER_THRESH  = 2.0     # theta peak must exceed N× mean spectrum power

# Speed-modulation parameters (see _compute_speed_modulation):
# only samples with smoothed speed between 1 and 90 cm/s are used
SPEED_MIN_CMS       = 1.0     # lowest speed (cm/s) included in speed-modulation analysis
SPEED_MAX_CMS       = 90.0   # highest speed (cm/s) included in speed-modulation analysis
# width of the speed bins used for the binned fit (4 cm/s; previous value 2.0 kept in the comment)
SPEED_BIN_CMS       = 4.0 #2.0     # width of each speed bin (cm/s)
# Gaussian smoothing sigma (in seconds) applied to both the speed and firing-rate time series
SPEED_SMOOTH_S      = 0.3 #0.08    # Gaussian smoothing window (s) applied to instantaneous firing rate
# speed bins with fewer than 0.2 % of all samples are dropped from the fit (unreliable)
SPEED_MIN_BIN_FRAC  = 0.002   # discard speed bins holding < this fraction of samples

# number of circular-shift shuffles for the speed tests, and their 20-s minimum shift
SPEED_N_SHUFFLE      = 1000   # circular-shift shuffles for speed-modulation significance
SPEED_SHUFFLE_MARGIN_S = 20.0 # min circular-shift offset (s) from either end, matches SIR shuffling

# Tracking clean-up parameters:
# a frame-to-frame step implying > 90 cm/s is physically implausible for the animal -> artifact
POS_JUMP_THRESH_CMS  = 90.0   # frame-to-frame jumps implying a speed above this (cm/s) are tracking artifacts
                              # (= 0.00009 cm/µs; used by both _load_tracking and _smooth_tracking_position)
# Gaussian sigma (in frames) used to smooth x and y: 5 frames = 167 ms at 30 fps
POS_SMOOTH_SIGMA_SMP = 5.0   # Gaussian smoothing sigma (in samples) applied to x/y tracking position

# frames slower than 0.5 cm/s are treated as immobility and removed from all spatial analyses
IMMOBILITY_SPEED_CMS = 0.5    # tracking frames whose speed (from the smoothed position) is below this
                              # (cm/s) are removed from all spatial analyses, together with the spikes
                              # assigned to them -- removes non-spatial firing during immobility
                              # (e.g. sharp-wave ripples). See compute_metrics step 2c.

# take every Nth tracking frame for the speed analysis. NOTE (review): the value is 1, i.e. NO
#   downsampling; the comment below (30 -> 15 fps) describes the old value of 2 and is stale.
SPEED_MOD_DOWNSAMPLE_FACTOR = 1   # downsample tracking for speed-modulation analysis only:
                                   # 30 fps -> 15 fps by keeping every alternate frame. The
                                   # 33.33 ms native bin made the speed-vs-firing-rate cloud
                                   # too scattered; speed/rate are recomputed on this coarser
                                   # frame base (rest of the pipeline, e.g. SIR/ratemap, is
                                   # unaffected and still uses the native-rate tracking).

# GPU work waits while the GPU is >= 60 % busy (see _wait_for_gpu_slot)
MAX_GPU_UTIL_PCT = 60
# number of units analysed concurrently (threads in the ThreadPoolExecutor)
MAX_WORKERS      = 4

# 'pixel' or 'cm' – set interactively at startup (see __main__ below).
# 'pixel' : tracking file has 'x'/'y'/'time' columns in pixels, converted to cm.
# 'cm'    : tracking file is a .csv with time in column A, x (cm) in column D,
#           y (cm) in column E – used directly, no pixel→cm conversion.
# NOTE (review): despite the comment above, it is NOT set interactively; it is hard-coded here.
COORD_UNITS = 'cm'

# Semaphore(2): a counter that lets at most 2 threads use the GPU at the same time
#   (acquire() decrements it / blocks at 0; release() increments it).
_gpu_semaphore = threading.Semaphore(2)

# Hockeimer et al. 2025 (eLife 85599): ratemaps binned at 10 px (2.1 cm) per
# bin, smoothed with a Gaussian kernel of sigma = 1.5 bins. Stored here as a
# physical sigma in cm (1.5 * 2.1 cm) so it converts correctly to whatever
# bin size (target_bin_cm) this script is run with.
# GAUSSIAN_SIGMA_CM: Gaussian smoothing sigma in cm (3 cm -> 1.5 bins at 2-cm bins).
#   Paper (Fig. S2): smoothing changes SI; that is why SI here uses the raw map and the smoothed
#   map is used only for coherence, stability, field detection and display.
GAUSSIAN_SIGMA_CM = 3 #1.5 * 2.1

# (SIR / sparsity bin settings USE_SI_MIN_OCC / SI_MIN_OCC_S are set at the top of the file)

# ── Place-field detection ("threshold method") ────────────────────────────────
# Applied only to cells that pass the place-cell criteria below (see
# detect_place_fields_threshold_2d). A bin qualifies for a field if its
# Gaussian-smoothed firing rate is >= METHOD2_RATE_THRESHOLD_FRAC of the
# cell's peak (smoothed) rate AND above the cell's mean firing rate;
# 8-connected components of qualifying bins spanning >= MIN_FIELD_SIZE_BINS
# contiguous bins are reported as fields.
# MIN_FIELD_SIZE_BINS: minimum field size; 9 bins of 2x2 cm = 36 cm² (e.g. a 3x3 block).
MIN_FIELD_SIZE_BINS = 9               # a field must span >= 9 contiguous 8-connected bins
# METHOD2_RATE_THRESHOLD_FRAC: 20 % of peak -- same criterion as the paper's "place field width"
#   feature (number of bins exceeding 20 % of the peak rate).
METHOD2_RATE_THRESHOLD_FRAC = 0.20    # bins must be >= 20% of the cell's peak smoothed rate

# A detected field spanning more than this fraction of the total occupied
# (visited) area is treated as diffuse/non-spatial firing rather than a true
# place field -- a cell whose largest field crosses this covers too much of
# the arena to be considered spatially selective, so it is reclassified as
# NOT a place cell (see the 'pct_of_occupied_area > 50' override in _run_job).
PLACE_FIELD_MAX_AREA_PCT = 50.0

# Place-cell classification thresholds, applied to metrics['place_cell'] below.
# Kept as named constants (rather than inline literals) so a single source of
# truth drives both the classification and the run-metadata record of it.
# minimum spike count: SI estimates from few spikes are biased upward and noisy
PLACE_CELL_MIN_SPIKES  = 50    # n_spikes must exceed this
# peak rate window: > 1 Hz (enough firing to define a field) and < 25 Hz (excludes putative
#   fast-spiking interneurons, which fire at high rates throughout the arena)
PLACE_CELL_MIN_PEAK_FR = 1.0   # peak_fr (Hz) lower bound
PLACE_CELL_MAX_PEAK_FR = 25.0  # peak_fr (Hz) upper bound
# fixed SI threshold: 0.5 bits/spike (paper Table 1: Newman 2017, Grieves 2020, Duvelle 2021,
#   Harland 2021, Jin & Lee 2021 use 0.5; others use 0.25)
PLACE_CELL_MIN_SIR     = 0.5   # spatial information rate lower bound
# sparsity ceiling (defined but not applied; see PLACE_CELL_CRITERIA_DISABLED)
PLACE_CELL_MAX_SPARSITY = 0.9  # sparsity upper bound (currently disabled, see below)

# Criteria actually ANDed together in metrics['place_cell'] (see _build_row).
# Update this alongside the boolean expression if a criterion is toggled on/off.
# A tuple of strings: only documentation for the metadata CSV; it does not drive the logic.
PLACE_CELL_CRITERIA_ACTIVE = (
    # spike count criterion
    'n_spikes > PLACE_CELL_MIN_SPIKES',
    # peak firing-rate window
    'PLACE_CELL_MIN_PEAK_FR < peak_fr < PLACE_CELL_MAX_PEAK_FR',
    # fixed SI threshold
    'sir > PLACE_CELL_MIN_SIR',
    # SI shuffle (permutation) significance
    'bootstrap_sig is True',
)
# Criteria present in the code but currently commented out (not appliedl).
PLACE_CELL_CRITERIA_DISABLED = (
    # sparsity ceiling (disabled)
    'sparsity < PLACE_CELL_MAX_SPARSITY',
    # coherence shuffle significance (disabled)
    'coherence_bootstrap_sig is True',
)

# Names of every hardcoded analysis setting above, in the order they should
# appear in the run-metadata CSV. Add new config variables here as they're
# introduced so they get captured automatically.
# A list of strings; _save_run_metadata looks up each name in globals() to record its value.
RUN_CONFIG_VARS = [
    # input/output paths
    'root_folder', 'output_excel', 'Output_PlaceTrue',
    # figure sub-folder names
    'SPEED_MOD_SUBDIR', 'SHUFFLING_SUBDIR', 'RATEMAPS_SUBDIR',
    # frame rate, bin size, arena width
    'fps', 'target_bin_cm', 'arena_width_cm',
    # valid-bin criteria
    'use_min_occ_s', 'min_occ_s', 'use_min_visits', 'min_visits',
    # spike gating and SI shuffle settings
    'MAX_GAP_US', 'N_BOOTSTRAP', 'SIR_SHUFFLE_MARGIN_S',
    # stability
    'STABILITY_MIN_BINS',
    # theta
    'AUTOCORR_WINDOW_MS', 'AUTOCORR_BIN_MS', 'THETA_POWER_THRESH',
    # speed modulation
    'SPEED_MIN_CMS', 'SPEED_MAX_CMS', 'SPEED_BIN_CMS', 'SPEED_SMOOTH_S',
    'SPEED_MIN_BIN_FRAC', 'SPEED_N_SHUFFLE', 'SPEED_SHUFFLE_MARGIN_S',
    # tracking clean-up and immobility
    'POS_JUMP_THRESH_CMS', 'POS_SMOOTH_SIGMA_SMP', 'IMMOBILITY_SPEED_CMS',
    'SPEED_MOD_DOWNSAMPLE_FACTOR',
    # parallelism
    'MAX_GPU_UTIL_PCT', 'MAX_WORKERS',
    # coordinate units
    'COORD_UNITS',
    # smoothing
    'GAUSSIAN_SIGMA_CM',
    # SI occupancy restriction
    'USE_SI_MIN_OCC', 'SI_MIN_OCC_S',
    # field detection
    'MIN_FIELD_SIZE_BINS', 'METHOD2_RATE_THRESHOLD_FRAC', 'PLACE_FIELD_MAX_AREA_PCT',
    # place-cell thresholds
    'PLACE_CELL_MIN_SPIKES', 'PLACE_CELL_MIN_PEAK_FR', 'PLACE_CELL_MAX_PEAK_FR',
    'PLACE_CELL_MIN_SIR', 'PLACE_CELL_MAX_SPARSITY',
    'PLACE_CELL_CRITERIA_ACTIVE', 'PLACE_CELL_CRITERIA_DISABLED',
    # constants defined further down the file (they exist by the time the function is called)
    'ANIMAL_NAMES', 'DT_S_TOL_MS', 'RADIAL_RANGE_CLIP_PCTILE',
]


# _save_run_metadata(output_path) -> str : writes a CSV listing every setting, returns its path.
def _save_run_metadata(output_path: str) -> str:
    """Write every hardcoded setting in RUN_CONFIG_VARS, plus which script
    generated the run and when, to a CSV next to `output_path`. Called once
    at the start of a batch run so any figure/result produced downstream can
    be traced back to the exact settings that made it.
    """
    # a list of dicts; each dict becomes one row (parameter, value) of the CSV
    rows = [
        # __file__ is the path of this script; basename() keeps only the file name
        {'parameter': 'script_name', 'value': os.path.basename(__file__)},
        # current local time, ISO-8601 format, e.g. '2026-10-05T14:03:22'
        {'parameter': 'run_timestamp', 'value': datetime.now().isoformat(timespec='seconds')},
    ]
    # `+=` appends a list comprehension: for each config name, globals()[name] fetches the
    #   current value of the module-level variable with that name.
    rows += [{'parameter': name, 'value': globals()[name]} for name in RUN_CONFIG_VARS]

    # splitext() splits 'file.xlsx' into ('file', '.xlsx'); [0] keeps the part without extension
    meta_path = os.path.splitext(output_path)[0] + '_run_metadata.csv'
    # build a DataFrame from the rows and save it as CSV without the pandas row index column
    pd.DataFrame(rows).to_csv(meta_path, index=False)
    print(f'[SAVED] Run metadata: {meta_path}')
    # return the path so the caller could use it (the caller ignores it)
    return meta_path


# ntt_dtype: a NumPy "structured dtype" describing ONE record of a Neuralynx .ntt (tetrode) file.
#   Each record is one spike. '<' = little-endian byte order; 'u8' = unsigned 64-bit int;
#   'u4' = unsigned 32-bit; 'i2' = signed 16-bit. The file is read as an array of these records.
ntt_dtype = np.dtype([
    # spike time in microseconds (same clock as the tracking timestamps)
    ('timestamp',   '<u8'),
    # sub-channel (acquisition entity) number
    ('sc_number',   '<u4'),
    # cluster ID assigned by spike sorting; 0 = unsorted / noise cluster
    ('cell_number', '<u4'),
    # 8 feature parameters stored by the acquisition system (not used here)
    ('params',      '<u4', (8,)),
    # waveform: 32 samples x 4 tetrode channels (not used here)
    ('waveforms',   '<i2', (32, 4)),
])

# tuple of animal-ID folder names; used to rebuild the folder structure when copying place-cell
#   files. NOTE (review): the docstring below and a later comment say 'Fa5384' but this tuple
#   says 'Fa5834' -- check which spelling the real folder uses.
ANIMAL_NAMES = ('Fa1059', 'Fa23BD', 'Fa8477', 'Fa5834')


# _animal_relpath(dirpath) -> str | None : returns the tail of a path starting at the animal folder.
def _animal_relpath(dirpath: str) -> str | None:
    """Return the portion of `dirpath` starting at the animal-ID folder
    (Fa1059 / Fa23BD / Fa8477 / Fa5384), or None if no such folder is found.
    """
    # normpath() cleans the path (unifies separators); split(os.sep) breaks it into folder names
    parts = os.path.normpath(dirpath).split(os.sep)
    # enumerate() yields (index, value) pairs, so i is the position of each folder name
    for i, part in enumerate(parts):
        # `in` tests membership in the ANIMAL_NAMES tuple
        if part in ANIMAL_NAMES:
            # parts[i:] = all folders from the animal folder onward; `*` unpacks the list into
            #   separate arguments of os.path.join -> 'Fa1059\\Open\\session'
            return os.path.join(*parts[i:])
    # no animal folder found anywhere in the path
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

# _wait_for_gpu_slot(poll_interval=0.5): blocks (sleeps) while the GPU is busier than
#   MAX_GPU_UTIL_PCT, so this script does not saturate a GPU shared with other users/jobs.
#   `poll_interval: float = 0.5` is a parameter with a default value.
def _wait_for_gpu_slot(poll_interval: float = 0.5):
    # nothing to wait for if running on CPU or utilisation cannot be read
    if not _GPU or not _NVML:
        return
    # `while cond:` repeats the body as long as cond is True
    while _gpu_util_pct() >= MAX_GPU_UTIL_PCT:
        # pause this thread for poll_interval seconds before checking again
        time.sleep(poll_interval)


# _bin_visit_counts(flat_idx, n_bins) -> array of length n_bins with the number of separate
#   visits (entries) into each bin. flat_idx = the bin index of every tracking frame, in time order.
def _bin_visit_counts(flat_idx: np.ndarray, n_bins: int) -> np.ndarray:
    """Number of visits per bin: a visit starts whenever the animal enters a bin
    (first frame, or bin index differs from the previous frame)."""
    # np.asarray converts lists etc. to a NumPy array (no copy if it already is one)
    flat_idx = np.asarray(flat_idx)
    # `.size` = number of elements; with no frames every bin has 0 visits
    if flat_idx.size == 0:
        # np.zeros(n, dtype=np.int64) -> array of n integer zeros
        return np.zeros(n_bins, dtype=np.int64)
    # boolean array, True where a new visit starts; initialise all to True
    starts = np.ones(flat_idx.size, dtype=bool)
    # slicing: flat_idx[1:] = frames 1..end, flat_idx[:-1] = frames 0..end-1, so this compares
    #   every frame with the previous one; a visit starts where the bin index changes.
    #   (frame 0 keeps True: the first frame always starts a visit)
    starts[1:] = flat_idx[1:] != flat_idx[:-1]
    # flat_idx[starts] = bin index at each visit start; np.bincount counts how many times each
    #   integer occurs; minlength pads the result to n_bins entries.
    return np.bincount(flat_idx[starts], minlength=n_bins)


# _visits_2d(bx, by, n_bins_x, n_bins_y) -> 2-D array (n_bins_x x n_bins_y) of visit counts.
def _visits_2d(bx: np.ndarray, by: np.ndarray, n_bins_x: int, n_bins_y: int) -> np.ndarray:
    # converts each (x-bin, y-bin) pair into one flat index (row-major: bx * n_bins_y + by),
    #   counts visits on the flat array, then .reshape() folds it back into a 2-D grid
    return _bin_visit_counts(bx * n_bins_y + by, n_bins_x * n_bins_y).reshape(n_bins_x, n_bins_y)


# _valid_bin_mask(occ, visits) -> boolean map of bins that are "valid" (sampled well enough to
#   have a trustworthy firing rate). Invalid bins are excluded from every metric and plotted blank.
def _valid_bin_mask(occ: np.ndarray, visits: np.ndarray) -> np.ndarray:
    """Valid bins under the enabled criteria (use_min_occ_s / use_min_visits)."""
    # start with every bin the animal occupied at all (element-wise comparison -> boolean array)
    mask = occ > 0
    # optional occupancy criterion (currently off)
    if use_min_occ_s:
        # `&=` = in-place element-wise AND: keep only bins that also pass this criterion
        mask &= occ >= min_occ_s
    # optional visit-count criterion (currently on: >= 2 separate visits)
    if use_min_visits:
        mask &= visits >= min_visits
    return mask


# _gaussian_kernel(sigma_bins) -> normalised 2-D Gaussian weights for smoothing rate maps.
def _gaussian_kernel(sigma_bins: float) -> np.ndarray:
    """2D Gaussian kernel, sigma given in bins, truncated at 3 sigma."""
    # kernel half-width = ceil(3 sigma) bins (99.7 % of a Gaussian lies within ±3 sigma), min 1
    radius = max(1, int(np.ceil(3 * sigma_bins)))
    # integer offsets from -radius to +radius (np.arange stop value is exclusive, hence +1)
    ax = np.arange(-radius, radius + 1)
    # meshgrid makes 2-D coordinate grids; indexing='ij' -> first axis = x, second = y
    xx, yy = np.meshgrid(ax, ax, indexing='ij')
    # Gaussian formula exp(-(x² + y²) / (2 sigma²)); `**` is the power operator
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_bins ** 2))
    # normalise so the weights sum to 1 (smoothing then preserves the average rate)
    kernel /= kernel.sum()
    return kernel


# _convolve_pair(a, b, kernel) -> (a smoothed, b smoothed). Convolves two maps with the same
#   kernel; used to smooth the rate map and the valid-bin mask together (see _gaussian_smooth).
#   `-> tuple[np.ndarray, np.ndarray]` means it returns a pair of arrays.
def _convolve_pair(a: np.ndarray, b: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-padded 2D convolution of two same-shaped maps with one kernel (GPU if available)."""
    # GPU path
    if _GPU:
        # cp.asarray copies the NumPy arrays to GPU memory as float64
        a_gpu    = cp.asarray(a, dtype=cp.float64)
        b_gpu    = cp.asarray(b, dtype=cp.float64)
        kern_gpu = cp.asarray(kernel, dtype=cp.float64)
        # wait if the GPU is too busy
        _wait_for_gpu_slot()
        # take one of the 2 GPU "slots" (blocks if 2 threads are already using the GPU)
        _gpu_semaphore.acquire()
        # try/finally: the `finally` block runs even if an error occurs, so the slot is always freed
        try:
            # convolve on GPU; mode='constant', cval=0.0 treats everything outside the map as 0;
            #   cp.asnumpy copies the result back to normal (CPU) memory
            a_conv = cp.asnumpy(cp_convolve(a_gpu, kern_gpu, mode='constant', cval=0.0))
            b_conv = cp.asnumpy(cp_convolve(b_gpu, kern_gpu, mode='constant', cval=0.0))
        finally:
            # give the GPU slot back
            _gpu_semaphore.release()
    # CPU path: identical maths with scipy.ndimage.convolve
    else:
        a_conv = convolve(a, kernel, mode='constant', cval=0.0)
        b_conv = convolve(b, kernel, mode='constant', cval=0.0)
    # return both results as a tuple
    return a_conv, b_conv


# _gaussian_smooth(fr_map, valid_mask, bin_cm) -> smoothed rate map.
#   "Normalised convolution": unvisited bins are excluded, and each smoothed value is the
#   Gaussian-weighted average of only the VALID neighbouring bins. This prevents unvisited bins
#   (rate set to 0) and the arena edges from dragging smoothed rates down.
def _gaussian_smooth(fr_map: np.ndarray, valid_mask: np.ndarray, bin_cm: float) -> np.ndarray:
    """
    Apply Gaussian smoothing restricted to valid bins.
    Edge effects corrected by dividing the convolved rates by the convolved mask.
    """
    # convert sigma from cm to bins (3 cm / 2 cm = 1.5 bins)
    sigma_bins = GAUSSIAN_SIGMA_CM / bin_cm
    # build the 2-D kernel
    kernel = _gaussian_kernel(sigma_bins)

    # np.where(cond, A, B) -> element-wise "A where cond is True else B": invalid bins become 0
    fr_in   = np.where(valid_mask, fr_map, 0.0)
    # boolean mask -> 1.0 / 0.0 floats
    mask_in = valid_mask.astype(np.float64)

    # numerator: Σ w·rate over valid neighbours;  denominator: Σ w over valid neighbours
    smoothed_fr, smoothed_weights = _convolve_pair(fr_in, mask_in, kernel)

    # Correct edge effects by normalizing by gathered weights
    # output array of zeros with the same shape/dtype
    smoothed = np.zeros_like(smoothed_fr)
    # bins that received weight from at least one valid neighbour
    valid_weights = smoothed_weights > 0
    # weighted average = numerator / denominator (only where the denominator is > 0)
    smoothed[valid_weights] = smoothed_fr[valid_weights] / smoothed_weights[valid_weights]

    # `~` is element-wise NOT: invalid bins stay 0 (they are masked out in plots and metrics)
    smoothed[~valid_mask] = 0.0
    return smoothed


# _si_bin_mask(occ_map, valid_mask) -> which bins enter the SI and sparsity sums.
def _si_bin_mask(occ_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Bins entering the SIR / sparsity sums (see USE_SI_MIN_OCC)."""
    # with the option on: valid AND occupied >= 0.5 s
    if USE_SI_MIN_OCC:
        return valid_mask & (occ_map >= SI_MIN_OCC_S)
    # otherwise every valid bin
    return valid_mask


# _compute_sir(occ_map, fr_raw, valid_mask) -> Skaggs spatial information in bits/spike.
#   SI = Σ_i p_i · (r_i / r̄) · log2(r_i / r̄)
#     p_i = fraction of time spent in bin i,  r_i = firing rate in bin i,  r̄ = Σ p_i r_i (mean rate).
#   Meaning: how many bits about the animal's location each spike carries. A cell firing
#   uniformly everywhere has r_i = r̄ -> log2(1) = 0 -> SI = 0; a cell firing in one small spot
#   has large r_i/r̄ there -> high SI. As the reference paper shows (Fig. 6, S4), SI tracks
#   mostly the peak-to-average ratio, falls when baseline firing rises, and barely reacts to
#   trial-to-trial inconsistency.
def _compute_sir(occ_map: np.ndarray, fr_raw: np.ndarray, valid_mask: np.ndarray) -> float:
    """Skaggs spatial information (bits/spike): Σ pi (ri/r̄) log2(ri/r̄), on
    the RAW (unsmoothed) ratemap. pi is occupancy normalised over the bins in
    the sum (see _si_bin_mask). Shared by the real data and the bootstrap
    shuffles.
    """
    # which bins are summed over
    si_mask = _si_bin_mask(occ_map, valid_mask)
    # .any() is True if at least one element is True; no usable bins -> SI = 0
    if not si_mask.any():
        return 0.0

    # p_i: occupancy of the included bins divided by their total occupancy (sums to 1).
    #   Boolean indexing occ_map[si_mask] returns a 1-D ("flat") array of just those bins.
    pi_flat = occ_map[si_mask] / occ_map[si_mask].sum()
    # r_i: raw firing rates of the same bins, same order
    ri_flat = fr_raw[si_mask]
    # r̄ = Σ p_i r_i (occupancy-weighted mean rate = total spikes / total time over these bins)
    r_mean  = float(np.sum(pi_flat * ri_flat))
    # a silent cell carries no information (and we must not divide by 0)
    if r_mean <= 0:
        return 0.0

    # bins with zero rate contribute 0 (limit of x·log2(x) as x -> 0), so skip them to avoid log2(0)
    nonzero = ri_flat > 0
    # r_i / r̄ for the non-zero bins
    ratio   = ri_flat[nonzero] / r_mean
    # Σ p_i · (r_i/r̄) · log2(r_i/r̄) -> bits per spike
    return float(np.sum(pi_flat[nonzero] * ratio * np.log2(ratio)))


# _compute_sparsity(occ_map, fr_raw, valid_mask) -> Skaggs et al. 1996 sparsity.
#   sparsity = (Σ p_i r_i)² / Σ p_i r_i²   (range 0..1)
#   ≈ fraction of the arena in which the cell fires. 1 = fires equally everywhere;
#   small values (e.g. 0.1-0.3) = firing confined to a small region (typical place cell).
def _compute_sparsity(occ_map: np.ndarray, fr_raw: np.ndarray,
                      valid_mask: np.ndarray) -> float:
    """Skaggs sparsity: (Σ pi ri)² / Σ pi ri² on the RAW (unsmoothed)
    ratemap, over the same bins as the SIR sum (see _si_bin_mask). pi is
    occupancy normalised over the bins in the sum.
    """
    # same bins as SI
    sp_mask = _si_bin_mask(occ_map, valid_mask)
    if not sp_mask.any():
        return 0.0

    # p_i (occupancy probability), r_i (rate)
    pi_flat  = occ_map[sp_mask] / occ_map[sp_mask].sum()
    ri_flat  = fr_raw[sp_mask]
    # numerator term Σ p_i r_i  (= mean rate)
    spar_num = float(np.sum(pi_flat * ri_flat))
    # denominator Σ p_i r_i²   (= mean of squared rate)
    spar_den = float(np.sum(pi_flat * ri_flat ** 2))
    # conditional expression `A if cond else B`: guard against division by zero for a silent cell
    return float((spar_num ** 2) / spar_den) if spar_den > 0 else 0.0


# ── Theta modulation ──────────────────────────────────────────────────────────
# Hippocampal neurons often fire rhythmically, locked to the theta oscillation. Rhythmic firing
# shows up as regularly spaced peaks in the spike-time autocorrelogram; the FFT of the
# autocorrelogram then has a peak at the theta frequency.

# Function signature spread over several lines; the defaults come from the config constants.
#   Returns a tuple (theta_modulated: bool, peak_frequency_Hz: float).
def _compute_theta_modulation(
    # spike times in microseconds
    spike_ts_us: np.ndarray,
    # autocorrelogram half-width (500 ms)
    window_ms:   float = AUTOCORR_WINDOW_MS,
    # autocorrelogram bin width (5 ms)
    bin_ms:      float = AUTOCORR_BIN_MS,
    # power ratio needed to call the cell theta-modulated (2x)
    thresh:      float = THETA_POWER_THRESH,
) -> tuple[bool, float]:
    """FFT-based theta modulation test on the spike autocorrelogram.

    Mirrors the MATLAB logic:
      1. Build a ±window_ms autocorrelogram with bin_ms resolution.
      2. One-sided power spectrum via FFT.
      3. theta_modulated = peak_power(3–7 Hz) > thresh × mean(full spectrum).

    Returns (theta_modulated, dominant_freq_in_theta_band_hz).
    spike_ts_us must be in microseconds (raw .ntt timestamps).
    """
    # too few spikes for a meaningful autocorrelogram; float('nan') = "not a number" (missing value)
    if len(spike_ts_us) < 10:
        return False, float('nan')

    # sort the spike times and convert µs -> ms (1e-3 = 0.001)
    spike_ms = np.sort(spike_ts_us) * 1e-3          # µs → ms
    # number of autocorrelogram bins: 2 x 500 ms / 5 ms = 200 bins covering -500..+500 ms
    n_bins   = int(round(2.0 * window_ms / bin_ms))
    # histogram counts, start at 0
    autocorr = np.zeros(n_bins, dtype=np.float64)

    # `for i in range(N):` loops i = 0, 1, ..., N-1 (one iteration per reference spike)
    for i in range(len(spike_ms)):
        # time of the reference spike
        t_ref = spike_ms[i]
        # np.searchsorted finds where a value would be inserted in a sorted array -> index of the
        #   first spike >= t_ref - 500 ms ...
        lo    = int(np.searchsorted(spike_ms, t_ref - window_ms, side='left'))
        # ... and one past the last spike <= t_ref + 500 ms
        hi    = int(np.searchsorted(spike_ms, t_ref + window_ms, side='right'))
        # time lags of all spikes within ±500 ms relative to the reference spike
        diffs = spike_ms[lo:hi] - t_ref
        # drop the zero lag (the reference spike itself)
        diffs = diffs[diffs != 0.0]
        # `continue` skips to the next loop iteration if there are no neighbours
        if len(diffs) == 0:
            continue
        # convert each lag into a bin index: shift by +500 ms so lags start at 0, divide by 5 ms,
        #   .astype(int) truncates towards zero
        bin_idx = ((diffs + window_ms) / bin_ms).astype(int)
        # keep indices inside 0..n_bins-1 (a lag of exactly +500 ms would otherwise be index 200);
        #   out=bin_idx writes the result back into the same array
        np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)
        # np.add.at adds 1 at every index, correctly counting repeated indices (autocorr[bin_idx] += 1
        #   would count a repeated index only once)
        np.add.at(autocorr, bin_idx, 1)

    # One-sided power spectrum
    # autocorrelogram "sampling rate": one bin per 5 ms -> 200 Hz
    fs      = 1000.0 / bin_ms                       # sampling freq in Hz
    # number of samples (200)
    N       = len(autocorr)
    # frequency of each FFT output bin: k · fs/N -> resolution fs/N = 1 Hz
    f_axis  = np.arange(N, dtype=float) * (fs / N)
    # `//` is integer division; for real input only the first half (0 .. fs/2) is unique
    half    = N // 2
    # keep frequencies 0 .. 99 Hz
    f_axis  = f_axis[:half]

    # MEAN CENTER autocorr to remove DC offset before FFT
    # subtracting the mean removes the large 0-Hz (DC) component that would otherwise dominate
    autocorr_centered = autocorr - autocorr.mean()
    # np.fft.fft = discrete Fourier transform; np.abs(...)**2 = power at each frequency
    power   = np.abs(np.fft.fft(autocorr_centered)[:half]) ** 2

    # Peak in theta band (3–7 Hz)
    # boolean mask of the frequencies inside 3-7 Hz (with 1-Hz resolution: 3, 4, 5, 6, 7 Hz).
    #   NOTE (review): classic rat theta is ~6-10 Hz; 3-7 Hz is a lower band (presumably chosen
    #   for the mole-rat recordings) -- worth stating explicitly in the methods.
    theta_mask = (f_axis >= 3.0) & (f_axis <= 7.0)
    if not theta_mask.any():
        return False, float('nan')

    # power values in the theta band
    theta_power = power[theta_mask]
    # largest theta-band power
    peak_power  = float(theta_power.max())
    # np.argmax gives the position of the maximum -> look up its frequency
    peak_freq   = float(f_axis[theta_mask][np.argmax(theta_power)])

    # theta-modulated if the theta peak exceeds 2 x the average power over 0-99 Hz
    theta_modulated = peak_power > thresh * float(power.mean())
    # round the frequency to 2 decimals for the output table
    return theta_modulated, round(peak_freq, 2)


# ── Tracking position smoothing ───────────────────────────────────────────────

# _smooth_tracking_position(x_cm, y_cm, t_us, ...) -> (x_smooth, y_smooth).
#   Removes tracking glitches (impossible jumps), fills them by interpolation, then smooths.
#   Position noise matters because speed = distance / time is very sensitive to jitter, and
#   speed is used for the immobility filter and the speed-modulation analysis.
def _smooth_tracking_position(
    # x position (cm) of every frame
    x_cm:            np.ndarray,
    # y position (cm) of every frame
    y_cm:            np.ndarray,
    # timestamp (µs) of every frame
    t_us:            np.ndarray,
    # speed above which a step is an artifact (90 cm/s)
    jump_thresh_cms: float = POS_JUMP_THRESH_CMS,
    # Gaussian sigma in frames (5)
    sigma_samples:   float = POS_SMOOTH_SIGMA_SMP,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean and smooth the x/y tracking position before any downstream use.

    Iterative jump removal + Gaussian smoothing:
      1. Frame-to-frame steps between surviving ("good") samples that imply
         a speed above `jump_thresh_cms` are treated as tracking artifacts.
         This is re-tested against the shrinking good-sample set until no
         new artifacts are found, so runs of two or more consecutive bad
         frames (e.g. a tracker glitch that lingers for a few frames) are
         caught — not just single-frame jumps relative to the immediately
         preceding raw sample.
      2. Artifact frames are filled by linear interpolation (in time) between
         the nearest surviving good samples, rather than held at the last
         good position — holding would freeze position then "snap" back at
         the far edge of a dropout, itself implying an extra artificial
         speed spike right at the resumption point.
      3. The cleaned x/y traces are Gaussian-smoothed (`imgaussfilt` equivalent
         via `gaussian_filter1d`), sigma expressed in samples.
    """
    # number of frames
    n = len(x_cm)
    # fewer than 2 frames: nothing to clean; .copy() returns new arrays so the caller's are untouched
    if n < 2:
        return x_cm.copy(), y_cm.copy()

    # bad[i] = True marks frame i as an artifact; start with none marked
    bad = np.zeros(n, dtype=bool)
    # repeat at most n times (each pass marks >= 1 new frame or stops); `_` = unused loop variable
    for _ in range(n):
        # np.where(cond)[0] -> indices where cond is True, i.e. the frames still considered good
        good_idx = np.where(~bad)[0]
        # `break` leaves the loop early
        if len(good_idx) < 2:
            break
        # time between consecutive GOOD frames, µs -> s
        dt_good = np.diff(t_us[good_idx]) * 1e-6
        # `with np.errstate(...)`: temporarily silence NumPy warnings for invalid ops / division by 0
        with np.errstate(invalid='ignore', divide='ignore'):
            # np.hypot(dx, dy) = sqrt(dx² + dy²) = Euclidean step length; / dt -> speed (cm/s)
            step_speed = np.hypot(np.diff(x_cm[good_idx]), np.diff(y_cm[good_idx])) / dt_good
        # duplicate / out-of-order timestamps -> speed undefined -> treat as 0 (not a jump)
        step_speed[dt_good <= 0] = 0.0

        # which steps are impossibly fast
        newly_bad = step_speed > jump_thresh_cms
        # converged: no new artifacts found
        if not newly_bad.any():
            break
        # Mark the later sample of each offending pair as bad and re-test
        # against the remaining good set next round.
        # good_idx[1:] = the second frame of each step; [newly_bad] picks those of offending steps
        bad[good_idx[1:][newly_bad]] = True

    # final set of good frames
    good_idx = np.where(~bad)[0]
    # nothing to interpolate if all frames are good (or none are)
    if len(good_idx) == 0 or len(good_idx) == n:
        x_clean, y_clean = x_cm.copy(), y_cm.copy()
    else:
        # np.interp(new_t, known_t, known_values): linear interpolation in time -- bad frames get
        #   positions on the straight line between the surrounding good frames
        x_clean = np.interp(t_us, t_us[good_idx], x_cm[good_idx])
        y_clean = np.interp(t_us, t_us[good_idx], y_cm[good_idx])

    # 1-D Gaussian smoothing over frames (sigma = 5 frames); mode='nearest' pads the ends by
    #   repeating the edge values, so the first/last positions are not pulled towards 0
    x_smooth = gaussian_filter1d(x_clean, sigma=sigma_samples, mode='nearest')
    y_smooth = gaussian_filter1d(y_clean, sigma=sigma_samples, mode='nearest')
    return x_smooth, y_smooth


# ── Weighted least squares (speed-modulation regression) ──────────────────────

# class _WLSResult(NamedTuple): a small immutable record with named fields, so a result can be
#   read as res.slope, res.rvalue etc. (like scipy.stats.linregress's result).
class _WLSResult(NamedTuple):
    # fitted slope
    slope:     float
    # fitted intercept
    intercept: float
    # weighted Pearson correlation r
    rvalue:    float
    # two-tailed p-value for slope != 0
    pvalue:    float


# _weighted_linregress(x, y, w) -> _WLSResult. Weighted least-squares straight-line fit.
def _weighted_linregress(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> _WLSResult:
    """Weighted least-squares fit of y = slope*x + intercept, weighting each
    (x, y) point by `w`.

    Used in place of scipy.stats.linregress for the speed-modulation fit:
    each point there is a speed bin's mean firing rate, but bins hold very
    different sample counts (sparse high-speed bins vs. dense low-speed
    bins). An unweighted fit gives a 50-sample bin the same leverage as a
    5000-sample bin, biasing the slope/r/p-value toward whatever the noisy,
    sparse bins happen to show. Weighting by `w = n_per_bin` corrects that.

    rvalue is the weighted Pearson correlation; pvalue is a two-tailed t-test
    on the slope (df = n_points - 2), mirroring linregress's convention.
    """
    # make sure all inputs are float64 NumPy arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    # total weight
    w_sum  = w.sum()
    # weighted means of x and y
    x_mean = np.sum(w * x) / w_sum
    y_mean = np.sum(w * y) / w_sum
    # deviations from the weighted means (tuple assignment: two values in one line)
    dx, dy = x - x_mean, y - y_mean

    # weighted sums of squares / cross-products
    Sxx = np.sum(w * dx * dx)
    Syy = np.sum(w * dy * dy)
    Sxy = np.sum(w * dx * dy)

    # all x identical -> slope undefined; return a flat line with r = 0, p = 1
    if Sxx <= 0:
        return _WLSResult(slope=0.0, intercept=float(y_mean), rvalue=0.0, pvalue=1.0)

    # least-squares slope = Sxy / Sxx
    slope     = Sxy / Sxx
    # line passes through the weighted mean point
    intercept = y_mean - slope * x_mean
    # weighted Pearson r = Sxy / sqrt(Sxx · Syy) (0 if y is constant)
    rvalue    = Sxy / np.sqrt(Sxx * Syy) if Syy > 0 else 0.0
    # np.clip keeps r within [-1, 1] (guards against tiny floating-point overshoot)
    rvalue    = float(np.clip(rvalue, -1.0, 1.0))

    # number of points (speed bins) and degrees of freedom (2 parameters estimated)
    n   = len(x)
    dof = n - 2
    if dof > 0:
        # residuals of the fit
        resid = y - (slope * x + intercept)
        # weighted mean squared error
        mse   = np.sum(w * resid * resid) / dof
        # standard error of the slope
        se_slope = np.sqrt(mse / Sxx)
        if se_slope > 0:
            # t statistic for H0: slope = 0
            tstat  = slope / se_slope
            # two-tailed p = 2 · P(T > |t|); .sf = survival function = 1 - CDF
            pvalue = float(2.0 * _t_dist.sf(np.abs(tstat), dof))
        else:
            # perfect fit: p = 0 if there is a non-zero slope, else 1
            pvalue = 0.0 if slope != 0.0 else 1.0
    else:
        # only 2 points: p-value undefined
        pvalue = float('nan')

    # build and return the named-tuple result (keyword arguments name each field)
    return _WLSResult(slope=float(slope), intercept=float(intercept),
                       rvalue=rvalue, pvalue=pvalue)


# ── Shuffle significance (shared by binned + time-domain speed modulation) ────

# _ShuffleSummary: named tuple bundling the null-distribution summary of a shuffle test.
class _ShuffleSummary(NamedTuple):
    # mean of the shuffled scores
    mean:      float
    # 2.5th percentile
    lo:        float
    # 97.5th percentile
    hi:        float
    # two-tailed p-value
    p:         float
    # True if p < 0.05
    modulated: bool


# _shuffle_ci_and_pvalue(real_score, shuff_scores, min_valid=100) -> _ShuffleSummary or None.
#   The return annotation is a string ('...') -- a "forward reference" type hint.
def _shuffle_ci_and_pvalue(real_score: float, shuff_scores: np.ndarray,
                           min_valid: int = 100) -> '_ShuffleSummary | None':
    """Two-tailed 2.5/97.5 shuffle CI + signed-tail p-value from a null
    distribution of circular-shift shuffle scores.

    Used identically for BOTH the binned-regression speed score and the
    time-domain (Pearson r) speed score, so the two significance tests only
    ever differ in which score they're fed — never in how significance is
    computed from it.

    Returns None if fewer than `min_valid` finite shuffle scores survive.
    """
    # np.isfinite is False for NaN/inf; keep only shuffles that produced a real number
    valid = shuff_scores[np.isfinite(shuff_scores)]
    # not enough shuffles to define a null distribution
    if len(valid) < min_valid:
        return None

    # number of usable shuffles
    n    = len(valid)
    # mean of the null distribution
    mean = float(np.mean(valid))
    # lower and upper bounds of the central 95 % of the null distribution
    lo   = float(np.percentile(valid, 2.5))
    hi   = float(np.percentile(valid, 97.5))

    # Two-tailed p-value, built from the SAME signed-tail rule as lo/hi above
    # (rather than pooling |shuffle| >= |real| across both tails, which can
    # silently disagree with the lo/hi crossing test when the null
    # distribution is skewed): count how far into whichever tail the real
    # score sits on its own side, then double that one-sided fraction so a
    # 2.5% one-sided tail on either side reads as p = 0.05 two-sided. This
    # makes "p < 0.05" and "outside [lo, hi]" the same test by construction.
    # positive real score -> count shuffles at least as large (upper tail)
    if real_score >= 0:
        tail_count = int(np.sum(valid >= real_score))
    # negative real score -> count shuffles at least as small (lower tail)
    else:
        tail_count = int(np.sum(valid <= real_score))
    # (count + 1)/(n + 1) is the standard permutation p-value (the +1 counts the real data as one
    #   of the possible arrangements, so p is never exactly 0); x2 makes it two-tailed; capped at 1
    p = float(min(1.0, 2.0 * (tail_count + 1) / (n + 1)))

    # return the summary; bool(...) converts NumPy booleans to a plain Python bool
    return _ShuffleSummary(mean=mean, lo=lo, hi=hi, p=p, modulated=bool(p < 0.05))


# ── 1D Convolution Helper for NaN-safe Temporal Smoothing ─────────────────────

# _smooth_1d(arr, finite_mask, sigma) -> Gaussian-smoothed copy of a time series that may
#   contain NaN gaps. Same "normalised convolution" idea as _gaussian_smooth, in 1-D.
def _smooth_1d(arr: np.ndarray, finite_mask: np.ndarray, sigma: float) -> np.ndarray:
    """1D normalized Gaussian convolution handling NaN dropouts securely."""
    # replace missing samples with 0 so they add nothing to the weighted sum
    arr_in = np.where(finite_mask, arr, 0.0)
    # 1.0 where data exist, 0.0 where missing
    mask_in = finite_mask.astype(np.float64)

    # Gaussian-weighted sum of the data and of the weights (zero padding beyond both ends)
    smoothed_arr = gaussian_filter1d(arr_in, sigma=sigma, mode='constant', cval=0.0)
    smoothed_weights = gaussian_filter1d(mask_in, sigma=sigma, mode='constant', cval=0.0)

    # output initialised to NaN; np.full_like(a, v) = array shaped like a, filled with v
    out = np.full_like(arr, np.nan)
    # samples that received any weight
    valid_weights = smoothed_weights > 0
    # weighted average of the available neighbours only
    out[valid_weights] = smoothed_arr[valid_weights] / smoothed_weights[valid_weights]
    # samples that were missing stay missing (no invented values)
    out[~finite_mask] = np.nan
    return out


# ── Speed modulation ──────────────────────────────────────────────────────────
# "Speed cells" change their firing rate with running speed (Kropff et al. 2015; Iwase et al.
# 2020). Two complementary scores are computed:
#   (a) BINNED: median firing rate in each 4-cm/s speed bin, then a weighted straight-line fit
#       rate = beta·speed + f0; score = r of that fit.
#   (b) TIME-DOMAIN: Pearson r between the smoothed speed and smoothed firing-rate time series.
# Each is checked against a circular-shift shuffle null distribution (like the SI test).

# _compute_speed_modulation(...) -> dict of speed metrics (and saves a 2x2 figure if ntt_path given).
def _compute_speed_modulation(
    # position (cm) and time (µs) of every frame -- contiguous track (immobile frames NOT removed)
    x_cm:               np.ndarray,
    y_cm:               np.ndarray,
    t_us:               np.ndarray,
    # spike times (µs)
    spike_ts_us:        np.ndarray,
    # tracking frame rate (Hz), used to convert seconds into frames
    pos_sample_rate_hz: float,
    # speed range kept for the analysis (1-90 cm/s)
    min_speed_cms:      float = SPEED_MIN_CMS,
    max_speed_cms:      float = SPEED_MAX_CMS,
    # speed bin width (4 cm/s)
    speed_bin_cms:      float = SPEED_BIN_CMS,
    # Gaussian smoothing sigma in seconds (0.3 s)
    smooth_window_s:    float = SPEED_SMOOTH_S,
    # minimum fraction of samples per speed bin (0.2 %)
    min_bin_frac:       float = SPEED_MIN_BIN_FRAC,
    # .ntt path (to know where to save the figure); None -> no figure
    ntt_path:           str | None = None,
    # suffix added to the figure name ('full', 'first_half', ...)
    label:              str = '',
) -> dict:
    """Relates firing rate to running speed (ports MATLAB `speed_firing_runita`).

      1. Per-frame running speed (cm/s) from consecutive positions, using the
         actual inter-frame interval rather than an assumed fixed rate, then
         Gaussian-smoothed with the same kernel as the firing rate (step 2)
         so the two signals being compared are smoothed on equal footing.
      2. Instantaneous firing rate on the same frame base (spike count per
         inter-frame interval / interval duration), Gaussian-smoothed.
      3. Restrict to samples with min_speed_cms < smoothed speed < max_speed_cms.
      4. Bin firing rate by (smoothed) speed (speed_bin_cms-wide bins), drop
         bins with < min_bin_frac of the samples, and fit rate = beta*speed + f0
         to the binned medians (beta/f0 naming kept from the MATLAB source).

    x_cm, y_cm, t_us must be same-length position tracks (t_us in µs, sorted).
    spike_ts_us must be in the same time base (µs).
    Returns speed_score (r), speed_p_value, speed_beta (slope),
    speed_f0 (intercept), speed_modulated (p < 0.05), and a circular-shift
    shuffle confirmation (speed_shuffle_mean/lo/hi/p, speed_modulated_shuffle)
    analogous to the SIR location-shuffling bootstrap.
    """
    # result dict pre-filled with "missing" values (NaN / None / False); returned as-is if the
    #   analysis cannot run, and filled in step by step otherwise. A dict literal is {key: value, ...}.
    result = {'speed_score': float('nan'), 'speed_p_value': float('nan'),
              # r² of the binned fit
              'speed_r2': float('nan'),
              # slope and intercept of the binned fit
              'speed_beta': float('nan'), 'speed_f0': float('nan'),
              # parametric significance of the binned fit
              'speed_modulated': None,
              # binned-fit shuffle summary
              'speed_shuffle_mean': float('nan'), 'speed_shuffle_lo': float('nan'),
              'speed_shuffle_hi': float('nan'), 'speed_shuffle_p': float('nan'),
              'speed_modulated_shuffle': None,
              # Time-domain correlation (Iwase et al. 2020 "speed score"): direct
              # Pearson r between the smoothed speed and smoothed firing-rate time
              # series, sample-by-sample, with no speed-binning step. Computed
              # alongside — not instead of — the binned-regression result above.
              'speed_score_td': float('nan'), 'speed_p_value_td': float('nan'),
              'speed_r2_td': float('nan'),
              # time-domain shuffle summary
              'speed_shuffle_mean_td': float('nan'), 'speed_shuffle_lo_td': float('nan'),
              'speed_shuffle_hi_td': float('nan'), 'speed_shuffle_p_td': float('nan'),
              'speed_modulated_td': None,
              # p-Speed / n-Speed classification, matching Iwase et al.: real
              # speed_score_td above the 99th percentile (p-Speed) or below the
              # 1st percentile (n-Speed) of its own shuffle null distribution.
              # (NOTE (review): the code below actually uses the 97.5th / 2.5th percentiles.)
              'p_speed': None, 'n_speed': None,
              # Whether each method's initial (parametric) R²/p-value screen
              # passed and its circular-shift shuffle was therefore run at all
              # -- see the "initial significance gate" below.
              'speed_shuffle_ran': False, 'speed_shuffle_ran_td': False}

    # number of tracking frames
    n = len(t_us)
    # need >= 3 frames and >= 1 spike; `or` is logical OR
    if n < 3 or len(spike_ts_us) == 0:
        return result

    # ── Per-frame running speed (cm/s) ────────────────────────────────────────
    # There are n position samples, hence n-1 "inter-frame intervals" between
    # them. dt_s is the true duration of each of those intervals (in seconds),
    # computed from the actual timestamps rather than assuming a fixed frame
    # rate — the tracker's frame interval is not perfectly constant.
    # np.diff -> t[i+1] - t[i] (µs); x 1e-6 -> seconds
    dt_s = np.diff(t_us) * 1e-6
    with np.errstate(invalid='ignore', divide='ignore'):
        # Euclidean distance travelled between consecutive frames, divided by
        # how long that step took -> speed (cm/s) for each of the n-1 intervals.
        speed = np.hypot(np.diff(x_cm), np.diff(y_cm)) / dt_s
    # boolean-mask assignment: set speed to NaN wherever dt is 0 or negative
    speed[dt_s <= 0] = np.nan          # guard against zero/negative dt (duplicate or out-of-order timestamps)
    # np.append adds one element at the end -> length n, aligned with the n frames
    speed = np.append(speed, speed[-1])              # pad to length n (repeat last value for the final sample)
                                                    #padding at stop or start?
                                                    # Predictive signals will be masked if assinged to (n-1, n) rather than (n, n+1)

    # ── Gaussian smoothing of speed, matching the firing-rate smoothing ───────
    # Same kernel (sigma_samples, derived from smooth_window_s and the position
    # sample rate) and same NaN-safe convolution trick as the firing-rate
    # smoothing below.
    # sigma in frames = 0.3 s x 30 frames/s = 9 frames; max(..., 1e-6) prevents a zero sigma
    sigma_samples = max(smooth_window_s * pos_sample_rate_hz, 1e-6)
    # which speed samples are real numbers
    speed_finite = np.isfinite(speed)
    # NaN-safe Gaussian smoothing of the speed trace
    speed_smooth = _smooth_1d(speed, speed_finite, sigma_samples)

    # ── Instantaneous firing rate on the same frame base ──────────────────────
    # Goal: turn the spike train into one firing-rate value per inter-frame
    # interval, so it lines up 1-to-1 with the `speed` array above.
    #
    # searchsorted(t_us, spike_ts_us, side='right') - 1 finds, for every spike
    # timestamp, the index of the position frame immediately *before* it —
    # i.e. which of the n-1 intervals [t[i], t[i+1]) the spike falls into.
    # side='right' means a spike landing exactly on a frame timestamp is
    # assigned to the interval that *starts* at that timestamp.
    interval_idx = np.searchsorted(t_us, spike_ts_us, side='right') - 1
    # Clip so spikes before the first frame or after the last frame don't
    # produce an out-of-bounds index; they get folded into the first/last interval.
    interval_idx = np.clip(interval_idx, 0, n - 2)
    # Removed MAX_GAP_US cull here so spikes inside interpolated tracking gaps
    # are correctly divided by the full dt_s of that gap, preserving instantaneous rate
    # spike count per interval (bincount counts occurrences of each interval index)
    counts = np.bincount(interval_idx, minlength=n - 1).astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        # Instantaneous rate for interval i = (spikes in interval i) / (duration
        # of interval i). This is a raw, un-smoothed, per-frame firing rate (Hz).
        fr_inst = counts / dt_s
    # undefined where dt <= 0
    fr_inst[dt_s <= 0] = np.nan
    # pad to n samples, same convention as speed
    fr_inst = np.append(fr_inst, fr_inst[-1])         # pad to length n, same convention as `speed`

    # Gaussian smoothing of the instantaneous rate.
    # which rate samples are real numbers
    finite = np.isfinite(fr_inst)
    # smoothed firing rate (same sigma as speed)
    fr_smooth = _smooth_1d(fr_inst, finite, sigma_samples)

    # ── Restrict to the usable speed range ─────────────────────────────────────
    # Very low speeds (near-stationary, e.g. grooming/resting) and very high
    # speeds (tracking artifacts / jumps) are excluded so the fit isn't
    # dominated by outliers or immobility-related firing (e.g. sharp-wave
    # ripples during rest).
    # boolean mask of samples to analyse: 1 < speed < 90 cm/s and a valid rate
    in_range = ((speed_smooth > min_speed_cms) & (speed_smooth < max_speed_cms)
                & np.isfinite(fr_smooth))
    # summing a boolean array counts its True values
    if in_range.sum() < 3:
        return result

    # the analysed speed and rate samples (paired, same order)
    speed_valid = speed_smooth[in_range]
    rate_valid  = fr_smooth[in_range]
    # a constant signal has no correlation (np.std = standard deviation)
    if np.std(speed_valid) == 0 or np.std(rate_valid) == 0:
        return result

    # ── Combined 2×2 speed-analysis figure ──────────────────────────────────────
    # top-left: binned speed-modulation scatter/fit; bottom-left: its shuffle
    # histogram. top-right: time-domain speed-vs-rate correlation (time series);
    # bottom-right: its shuffle histogram. All four panels for this cell are
    # drawn into one figure and saved together, in a single 'speed modulation'
    # folder, rather than as four separate plots split across two folders.
    # chained assignment: all five names start as None (no figure)
    fig_combined = ax_bin = ax_bin_shuf = ax_td = ax_td_shuf = None
    # `is not None` tests identity with None (the correct way to check for "not given")
    if ntt_path is not None:
        # new figure, 14 x 10 inches
        fig_combined = Figure(figsize=(14, 10))
        # attach the Agg canvas so the figure can be rendered/saved
        FigureCanvasAgg(fig_combined)
        # add_subplot(rows, cols, index): index counts left->right, top->bottom (1 = top-left)
        ax_bin      = fig_combined.add_subplot(2, 2, 1)
        ax_bin_shuf = fig_combined.add_subplot(2, 2, 3)
        ax_td       = fig_combined.add_subplot(2, 2, 2)
        ax_td_shuf  = fig_combined.add_subplot(2, 2, 4)

    # ── Time-domain correlation (Iwase et al. 2020 "speed score") ──────────────
    # Direct Pearson correlation between the smoothed speed and smoothed firing
    # rate, sample-by-sample across time — no speed-binning step, unlike the
    # binned-regression fit below. This is the method the paper actually uses.
    # pearsonr returns (r, p); tuple unpacking assigns them to r_td and p_td
    r_td, p_td = pearsonr(speed_valid, rate_valid)
    # store r rounded to 4 decimals
    result['speed_score_td'] = round(float(r_td), 4)
    # r² = fraction of rate variance explained by speed
    result['speed_r2_td']    = round(float(r_td ** 2), 4)
    # NOTE: this parametric p-value assumes independent residuals, which is
    # violated here — Gaussian-smoothed adjacent samples are strongly
    # autocorrelated — so it will be anti-conservative (falsely low). The
    # shuffle-based speed_p_value_td / speed_modulated_td set further below,
    # from circularly-shifted null data, is the one to trust for significance.
    # It's still used below as a cheap initial screen: a fit that isn't even
    # nominally significant on its face doesn't get the expensive shuffle.
    # (speed_p_value_td therefore stays the PARAMETRIC p; the shuffle p is stored separately in
    #   speed_shuffle_p_td.)
    result['speed_p_value_td'] = round(float(p_td), 4)

    # ── top-right panel: time-domain speed-vs-rate correlation (time series) ───
    # Drawn unconditionally, mirroring the binned scatter/fit panel below (also
    # independent of significance), so this panel is present whether or not the
    # shuffle ends up running. The title is refined further down with the
    # shuffle-based label/p-value once (if) that shuffle completes.
    if ax_td is not None:
        # time axis in seconds from the start of the session
        time_s = (t_us - t_us[0]) * 1e-6
        # speed trace in blue (linewidth in points)
        ax_td.plot(time_s, speed_smooth, color='tab:blue', linewidth=0.6)
        ax_td.set_xlabel('time (s)')
        # y-label coloured like its trace
        ax_td.set_ylabel('speed (cm/s)', color='tab:blue')
        # colour the y tick labels to match
        ax_td.tick_params(axis='y', labelcolor='tab:blue')
        # twinx() creates a second y-axis sharing the same x-axis (right side)
        ax_td_twin = ax_td.twinx()
        # firing-rate trace in orange on the right axis
        ax_td_twin.plot(time_s, fr_smooth, color='tab:orange', linewidth=0.6)
        ax_td_twin.set_ylabel('firing rate (Hz)', color='tab:orange')
        ax_td_twin.tick_params(axis='y', labelcolor='tab:orange')
        # format spec :.3f = 3 decimals; :.3g = 3 significant digits
        ax_td.set_title(f"time-domain: r = {r_td:.3f} (parametric p = {p_td:.3g})")

    # ── Bin firing rate by speed ────────────────────────────────────────────────
    # Build speed_bin_cms-wide bin edges spanning [min_speed_cms, max_speed_cms]
    # (e.g. 2 cm/s wide bins from 2 to 90 cm/s), giving n_bins bins.
    # (with the current settings: edges 1, 5, 9, ..., 93 cm/s -> 23 bins of 4 cm/s)
    speed_bins = np.arange(min_speed_cms, max_speed_cms + speed_bin_cms, speed_bin_cms)
    # number of bins = number of edges - 1
    n_bins     = len(speed_bins) - 1
    # For every valid (speed, rate) sample, find which speed bin it falls in.
    # np.digitize returns 1-indexed bin numbers; subtract 1 to make it
    # 0-indexed, and clip to guard against samples exactly at/above the last
    # edge (which digitize would otherwise put in an out-of-range bin).
    bin_idx    = np.clip(np.digitize(speed_valid, speed_bins) - 1, 0, n_bins - 1)

    # bin_centres: the midpoint speed of each bin, used as the x-value in the
    # final regression (rate vs. speed).
    bin_centres = 0.5 * (speed_bins[:-1] + speed_bins[1:])
    # median rate per speed bin, NaN until filled
    binned_rate = np.full(n_bins, np.nan)
    # number of samples per speed bin
    n_per_bin   = np.zeros(n_bins, dtype=int)
    # empty Python list that will hold one boolean mask per bin
    bin_masks   = []   # per-bin boolean membership, reused (unchanged across shuffles) below
    # For each speed bin, take the median (not mean) of the (already
    # Gaussian-smoothed) firing rate of every sample that fell in it. This
    # collapses the noisy per-frame rate/speed cloud into one representative
    # rate per speed bin, using the median so a handful of outlier frames
    # (e.g. a burst landing in an otherwise low-rate bin) don't drag the
    # bin's representative value the way a mean would.
    for b in range(n_bins):
        # samples belonging to speed bin b
        sel = bin_idx == b
        # .append adds an item to the end of a list
        bin_masks.append(sel)
        # sample count of this bin
        n_per_bin[b] = int(sel.sum())
        # median rate of the bin (only if it has samples)
        if n_per_bin[b] > 0:
            binned_rate[b] = float(np.median(rate_valid[sel]))

    # total number of analysed samples
    total_pts = n_per_bin.sum()
    if total_pts == 0:
        return result
    # Bins that hold too few samples (fraction of total < min_bin_frac, e.g.
    # 0.2%) are unreliable estimates of rate at that speed, so they're
    # discarded (set to NaN) rather than allowed to bias the regression —
    # this matters most at the high-speed tail, which is sparsely sampled.
    binned_rate[(n_per_bin / total_pts) < min_bin_frac] = np.nan   # drop under-sampled bins

    # bins that will enter the fit
    fit_sel = np.isfinite(binned_rate)
    # a line through < 3 points has no meaningful r/p
    if fit_sel.sum() < 3:
        return result

    # Weighted linear regression of binned median firing rate against
    # bin-centre speed: rate = speed_beta * speed + speed_f0, each bin
    # weighted by its sample count (n_per_bin) so sparsely-sampled bins
    # (mostly at the high-speed tail) don't get the same leverage as
    # densely-sampled ones.
    # reg.rvalue is the speed score (correlation coefficient r), reg.pvalue
    # tests whether the slope is significantly different from zero
    # (p < 0.05 => speed_modulated).
    # weights = sample counts of the fitted bins
    weights_fit = n_per_bin[fit_sel].astype(np.float64)
    # run the weighted fit
    reg = _weighted_linregress(bin_centres[fit_sel], binned_rate[fit_sel], weights_fit)

    # store the fit results (rounded to 4 decimals)
    result['speed_score']     = round(float(reg.rvalue), 4)
    result['speed_p_value']   = round(float(reg.pvalue), 4)
    result['speed_r2']        = round(float(reg.rvalue ** 2), 4)
    result['speed_beta']      = round(float(reg.slope), 4)
    result['speed_f0']        = round(float(reg.intercept), 4)
    # parametric significance (NOTE (review): the binned medians are of smoothed, autocorrelated
    #   samples and there are only ~20 bins, so this p-value is only a screen, as stated below)
    result['speed_modulated'] = bool(reg.pvalue < 0.05)

    # ── Initial R²/p-value significance gate ────────────────────────────────────
    # Before spending SPEED_N_SHUFFLE circular-shift iterations on a cell, first
    # check whether its real (parametric) fit is even nominally significant:
    # binned_significant from the weighted-regression p-value (reg.pvalue, same
    # test as speed_modulated above), td_significant from the raw Pearson p_td.
    # A method whose own initial fit doesn't clear p < 0.05 doesn't get shuffled
    # at all -- its shuffle_* / modulated_* / p_speed / n_speed fields are left
    # at their NaN/None defaults rather than being computed from a null test
    # that was never going to matter. The two methods are gated independently:
    # one can pass its screen and run its shuffle while the other doesn't.
    binned_significant = result['speed_modulated']
    td_significant      = bool(p_td < 0.05)

    # ── Circular-shift shuffling (confirms speed_modulated) ────────────────────
    # Same idea as the SIR location-shuffling bootstrap (_run_bootstrap): the
    # spike-count-per-interval trace is circularly shifted by a random offset
    # of at least SPEED_SHUFFLE_MARGIN_S (20 s) from either end, decorrelating
    # firing from speed while preserving each trace's own statistics. The
    # speed bin edges/membership (bin_idx) and which bins pass min_bin_frac
    # (fit_sel) depend only on `speed`, not on firing, so they are identical
    # for every shuffle and are reused rather than recomputed N_SHUFFLE times.
    # number of inter-frame intervals (n - 1)
    n_intervals   = len(counts)
    # 20 s expressed in frames (600 at 30 fps)
    MARGIN_FRAMES = int(SPEED_SHUFFLE_MARGIN_S * pos_sample_rate_hz)
    # real binned speed score (r)
    real_score    = float(reg.rvalue)

    binned_shuf_drawn = False   # whether the bottom-left panel got real histogram content
    td_shuf_drawn     = False   # whether the bottom-right panel got real histogram content

    # run shuffles only if the session is longer than 2 x 20 s and at least one screen passed
    if n_intervals > 2 * MARGIN_FRAMES and (binned_significant or td_significant):
        # Record which method(s) actually cleared the initial screen and
        # therefore get their null distribution built below.
        result['speed_shuffle_ran']    = binned_significant
        result['speed_shuffle_ran_td'] = td_significant

        # integer indices of the fitted speed bins
        fit_idx         = np.where(fit_sel)[0]
        # their sample counts (weights), centres and membership masks -- fixed for all shuffles
        n_per_bin_fit    = n_per_bin[fit_idx].astype(np.float64)
        bin_centres_fit  = bin_centres[fit_idx]
        # list comprehension: [expr for item in iterable] -> new list
        bin_masks_fit    = [bin_masks[b] for b in fit_idx]   # same membership as the real fit, reused every shuffle
        # arrays that will hold the 1000 shuffled scores (NaN = shuffle produced no score)
        shuff_scores     = np.full(SPEED_N_SHUFFLE, np.nan)
        shuff_scores_td  = np.full(SPEED_N_SHUFFLE, np.nan)   # null distribution for the time-domain correlation

        # one iteration per shuffle
        for i in range(SPEED_N_SHUFFLE):
            # random shift between 20 s and (session - 20 s), inclusive, in frames
            rnd             = random.randint(MARGIN_FRAMES, n_intervals - MARGIN_FRAMES)
            # np.roll shifts the array circularly: elements pushed off the end reappear at the
            #   start. Spike counts are moved in time relative to speed; the spike train's own
            #   structure (bursts, rate, autocorrelation) is preserved.
            shifted_counts  = np.roll(counts, rnd)
            with np.errstate(invalid='ignore', divide='ignore'):
                # instantaneous rate of the shifted spike train (divided by the UNshifted dt)
                fr_inst_s = shifted_counts / dt_s
            fr_inst_s[dt_s <= 0] = np.nan
            # pad to n samples
            fr_inst_s = np.append(fr_inst_s, fr_inst_s[-1])

            # smooth with the same kernel as the real rate
            fr_smooth_s = _smooth_1d(fr_inst_s, finite, sigma_samples)

            # Per-bin median of this shuffle's rate (matches the real-fit
            # computation above) — can't use the old bincount-sum/n trick
            # since median isn't additive across samples.
            # shuffled rate at the same in-range samples as the real data
            rate_valid_s   = fr_smooth_s[in_range]

            # Time-domain null score: same shuffled rate, correlated directly
            # against the (unshuffled) smoothed speed — mirrors real_score's
            # r_td computation above, giving a matched null distribution for
            # the time-domain correlation. Only built if the time-domain fit
            # passed its own initial significance gate.
            if td_significant and np.std(rate_valid_s) > 0:
                # [0] keeps only r from pearsonr's (r, p) result
                shuff_scores_td[i] = pearsonr(speed_valid, rate_valid_s)[0]

            # Binned null score. Only built if the binned fit passed its own
            # initial significance gate.
            if binned_significant:
                # median shuffled rate in each fitted speed bin
                binned_rate_fit = np.array([np.median(rate_valid_s[m]) for m in bin_masks_fit])
                # skip a flat (constant) binned rate
                if np.std(binned_rate_fit) != 0:
                    # weighted fit on the shuffled data; keep its r
                    reg_s = _weighted_linregress(bin_centres_fit, binned_rate_fit, n_per_bin_fit)
                    shuff_scores[i] = reg_s.rvalue

        # ── bottom-left panel: binned-fit shuffle histogram ─────────────────────
        # finite shuffled binned scores (for plotting)
        valid_shuff = shuff_scores[np.isfinite(shuff_scores)]
        # null summary; None if the binned shuffle did not run (all NaN)
        summ = _shuffle_ci_and_pvalue(real_score, shuff_scores)
        if summ is not None:
            # unpack the named-tuple fields into local variables
            shuffle_mean, shuffle_lo, shuffle_hi, shuffle_p = summ.mean, summ.lo, summ.hi, summ.p

            # store them in the result
            result['speed_shuffle_mean']      = round(shuffle_mean, 4)
            result['speed_shuffle_lo']        = round(shuffle_lo, 4)
            result['speed_shuffle_hi']        = round(shuffle_hi, 4)
            result['speed_shuffle_p']         = round(shuffle_p, 4)
            # shuffle-confirmed binned speed modulation (p < 0.05, two-tailed)
            result['speed_modulated_shuffle'] = summ.modulated

            if ax_bin_shuf is not None:
                # histogram with 100 bins; .hist returns (counts, bin_edges, patches)
                hist_result_s = ax_bin_shuf.hist(valid_shuff, 100, color='black')
                # counts per histogram bar
                counts_s: np.ndarray = np.asarray(hist_result_s[0])
                # tallest bar (used to size the vertical marker line); 1.0 if empty
                max_count_s = float(counts_s.max()) if counts_s.max() > 0 else 1.0

                # vertical red dash-dot line ('r-.') at the real score
                ax_bin_shuf.plot([real_score, real_score], [0, max_count_s], 'r-.')
                # 2.5% / 97.5% shuffle CI bounds, marking the significance cutoff.
                # axvline draws a vertical line across the axes; color '0.5' = mid grey
                ax_bin_shuf.axvline(shuffle_lo, color='0.5', linestyle=':', linewidth=0.8)
                ax_bin_shuf.axvline(shuffle_hi, color='0.5', linestyle=':', linewidth=0.8)
                # three title lines
                sh_av = f"Shuffle mean r = {shuffle_mean:.3f}"
                sh_ci = f"95% CI = [{shuffle_lo:.3f}, {shuffle_hi:.3f}]"
                # adjacent string literals inside parentheses are joined automatically
                sh_p  = (f"Cell r = {real_score:.3f} "
                         f"({'p < 0.05' if result['speed_modulated_shuffle'] else 'ns'})")
                # '\n' = newline; multialignment centres each line
                ax_bin_shuf.set_title(sh_av + '\n' + sh_ci + '\n' + sh_p, multialignment='center')
                ax_bin_shuf.set_ylabel('count')
                ax_bin_shuf.set_xlabel('speed score (r)')
                # remember that this panel has content
                binned_shuf_drawn = True

        # ── Time-domain shuffle summary + p-Speed/n-Speed classification ───────
        # Uses the exact same _shuffle_ci_and_pvalue helper as the binned result
        # above — identical CI/p-value construction, only the input score differs.
        # finite shuffled time-domain scores
        valid_shuff_td = shuff_scores_td[np.isfinite(shuff_scores_td)]
        # null summary for the time-domain score
        summ_td = _shuffle_ci_and_pvalue(r_td, shuff_scores_td)
        if summ_td is not None:
            # parenthesised tuple unpacking across two lines
            shuffle_mean_td, shuffle_lo_td, shuffle_hi_td, shuffle_p_td = (
                summ_td.mean, summ_td.lo, summ_td.hi, summ_td.p)

            # store them
            result['speed_shuffle_mean_td'] = round(shuffle_mean_td, 4)
            result['speed_shuffle_lo_td']   = round(shuffle_lo_td, 4)
            result['speed_shuffle_hi_td']   = round(shuffle_hi_td, 4)
            result['speed_shuffle_p_td']    = round(shuffle_p_td, 4)
            result['speed_modulated_td']    = summ_td.modulated

            # p-Speed / n-Speed classification: uses the SAME two-tailed 2.5/97.5
            # shuffle CI as speed_modulated_td above (not the paper's one-sided
            # 1st/99th percentile) — a cell is p-Speed if its real time-domain
            # score sits above that CI, n-Speed if it sits below.
            result['p_speed'] = bool(r_td > shuffle_hi_td)
            result['n_speed'] = bool(r_td < shuffle_lo_td)

            # if / elif / else chain: choose a label and colour for the plot
            if result['p_speed']:
                _td_label, _td_color = 'p-Speed', 'green'
            elif result['n_speed']:
                _td_label, _td_color = 'n-Speed', 'red'
            else:
                _td_label, _td_color = 'non-speed', 'black'

            # Refine the top-right panel's title now that the shuffle-based
            # label/p-value are known (the panel itself was already drawn above).
            if ax_td is not None:
                # the call's arguments continue on the next line (inside the parentheses)
                ax_td.set_title(f"{_td_label}  (r = {r_td:.3f}, shuffle p = {shuffle_p_td:.3g})",
                                 color=_td_color, fontweight='bold')

            # ── bottom-right panel: time-domain shuffle histogram ───────────────
            if ax_td_shuf is not None:
                # histogram of the shuffled time-domain scores
                hist_result_td = ax_td_shuf.hist(valid_shuff_td, 100, color='black')
                counts_td: np.ndarray = np.asarray(hist_result_td[0])
                max_count_td = float(counts_td.max()) if counts_td.max() > 0 else 1.0

                # vertical line at the real time-domain score, coloured by classification
                ax_td_shuf.plot([r_td, r_td], [0, max_count_td], color=_td_color, linestyle='-.', linewidth=1.5)
                # 95 % shuffle interval bounds
                ax_td_shuf.axvline(shuffle_lo_td, color='0.5', linestyle=':', linewidth=0.8)
                ax_td_shuf.axvline(shuffle_hi_td, color='0.5', linestyle=':', linewidth=0.8)

                # title built from several f-string pieces
                title_td = (f"{_td_label}\n"
                            # real r
                            f"Cell r = {r_td:.3f}  "
                            # shuffle interval
                            f"(shuffle 95% CI = [{shuffle_lo_td:.3f}, {shuffle_hi_td:.3f}], "
                            # shuffle p
                            f"p = {shuffle_p_td:.3g})")
                ax_td_shuf.set_title(title_td, color=_td_color, fontweight='bold', multialignment='center')
                ax_td_shuf.set_ylabel('count')
                ax_td_shuf.set_xlabel('time-domain speed score (r)')
                td_shuf_drawn = True

    # Shuffle panels that never ran (initial screen not significant, or not
    # enough frames for the margin) get a placeholder instead of staying blank.
    if ax_bin_shuf is not None and not binned_shuf_drawn:
        # text at the panel centre; transform=transAxes means (0.5, 0.5) is in axes fractions
        ax_bin_shuf.text(0.5, 0.5, 'binned-fit shuffle not run\n(initial fit not significant)',
                          ha='center', va='center', transform=ax_bin_shuf.transAxes)
        # `;` puts two statements on one line: remove the x and y ticks
        ax_bin_shuf.set_xticks([]); ax_bin_shuf.set_yticks([])
    if ax_td_shuf is not None and not td_shuf_drawn:
        ax_td_shuf.text(0.5, 0.5, 'time-domain shuffle not run\n(initial correlation not significant)',
                         ha='center', va='center', transform=ax_td_shuf.transAxes)
        ax_td_shuf.set_xticks([]); ax_td_shuf.set_yticks([])

    # ── top-left panel: binned speed-modulation scatter/fit ─────────────────────
    if ax_bin is not None:
        # every analysed sample as a small light-grey dot (s = marker size, alpha = transparency)
        ax_bin.scatter(speed_valid, rate_valid, s=3, color='0.75', alpha=0.4,
                        label='raw samples')
        # binned medians as large black dots
        ax_bin.scatter(bin_centres[fit_sel], binned_rate[fit_sel], s=80, color='black',
                        label='binned median')

        # x range of the fitted line (lowest and highest fitted bin centre)
        fit_x = np.array([bin_centres[fit_sel].min(), bin_centres[fit_sel].max()])
        # y values of the fitted line at those x values
        fit_y = reg.slope * fit_x + reg.intercept
        # red solid line ('r-')
        ax_bin.plot(fit_x, fit_y, 'r-', label='fit')

        # statistics text box
        stats_txt = (f"binned: r = {reg.rvalue:.3f}, slope = {reg.slope:.3f}, "
                     # continued: intercept and p of the binned fit
                     f"intercept = {reg.intercept:.3f}, p = {reg.pvalue:.3g}\n"
                     # time-domain r and its shuffle p
                     f"time-domain: r = {r_td:.3f}, p(shuffle) = {result['speed_shuffle_p_td']}")
        # place the text at the top-left corner (axes coordinates) in a rounded white box
        ax_bin.text(0.02, 0.98, stats_txt, transform=ax_bin.transAxes,
                    va='top', ha='left', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax_bin.set_xlabel('speed (cm/s)')
        ax_bin.set_ylabel('firing rate (Hz)')
        # legend in the lower-right corner
        ax_bin.legend(loc='lower right', fontsize=8)

    # ── save the combined 2×2 figure ─────────────────────────────────────────────
    if fig_combined is not None:
        # automatically adjust spacing so titles/labels do not overlap
        fig_combined.tight_layout()

        # .ntt file name without folder and extension, e.g. 'TT1_01'
        ntt_name   = os.path.splitext(os.path.basename(ntt_path))[0]
        # <folder of the .ntt>/<SPEED_MOD_SUBDIR>
        save_dir   = os.path.join(os.path.dirname(ntt_path), SPEED_MOD_SUBDIR)
        # create the folder (and parents); exist_ok=True -> no error if it already exists
        os.makedirs(save_dir, exist_ok=True)
        # '_full' etc. if a label was given, else empty string
        lbl_suffix = f'_{label}' if label else ''
        save_path  = os.path.join(save_dir, f'{ntt_name}{lbl_suffix}_speed_modulation.png')
        # write the PNG at 150 dots per inch
        fig_combined.savefig(save_path, dpi=150)
        print(f'  [SAVED] {save_path}')

    # hand back all speed metrics
    return result


# ── Bootstrap helpers ─────────────────────────────────────────────────────────
# "Bootstrap" here really means a PERMUTATION (shuffle) test, as in the reference paper
# ("Permutation Testing" section): spikes are circularly shifted along the session relative to
# position 1000 times; each shift keeps the spike train's rate and temporal structure (bursts,
# theta rhythm) and the animal's path/occupancy, but breaks the link between spikes and places.
# The SI of each shuffle forms a cell-specific null distribution; the real SI is significant if it
# exceeds that null's 95th percentile (p < 0.05). Paper: this is more robust than a fixed SI
# threshold because it adapts to each cell's firing rate and to bin size / smoothing.

# _sir_and_coherence_from_spikes_locshuf(...) -> (SI, coherence) for ONE shuffle.
def _sir_and_coherence_from_spikes_locshuf(spike_frame_indices: np.ndarray, rnd: int,
                              # x/y bin index of every (moving) tracking frame
                              beh_bx: np.ndarray, beh_by: np.ndarray,
                              # real occupancy map and valid mask (unchanged by shuffling)
                              occ_map: np.ndarray, valid_mask: np.ndarray,
                              # grid size and bin size
                              n_bins_x: int, n_bins_y: int, bin_cm: float) -> tuple[float, float]:
    """Compute SIR and spatial coherence from the same location-shuffled spike
    train after circularly shifting the position time series by `rnd` frames.
    Spike-to-frame assignments are kept fixed; only the location at each frame
    changes. Equivalent to MATLAB: locs_rand = [locs(rnd:end); locs(1:rnd-1)]

    Both metrics are derived from the single shuffled rate map built here so
    that each bootstrap iteration only needs one shuffled spike train: SIR is
    computed on the RAW shuffled ratemap (matching the real-data SIR), while
    coherence is computed on the Gaussian-SMOOTHED shuffled ratemap (matching
    the real-data coherence).
    """
    # number of tracking frames
    n_frames   = len(beh_bx)
    # shift every spike's frame index by rnd frames; `%` (modulo) wraps indices past the end
    #   back to the start -> a circular shift
    shuf_frame = (spike_frame_indices + rnd) % n_frames

    # shuffled spike-count map
    spike_map = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    # add 1 at the (x-bin, y-bin) of each shuffled spike; np.add.at handles repeated bins
    np.add.at(spike_map, (beh_bx[shuf_frame], beh_by[shuf_frame]), 1.0)

    # shuffled raw rate map = spikes / occupancy, only in valid bins (others stay 0);
    #   np.divide(..., out=, where=) writes the quotient into fr_raw only where valid_mask is True
    fr_raw = np.zeros_like(spike_map)
    np.divide(spike_map, occ_map, out=fr_raw, where=valid_mask)

    # SI of the shuffled RAW map (same function and bins as the real SI)
    sir = _compute_sir(occ_map, fr_raw, valid_mask)

    # coherence of the shuffled SMOOTHED map (same processing as the real coherence)
    fr_smooth = _gaussian_smooth(fr_raw, valid_mask, bin_cm)
    coherence = _compute_coherence(fr_smooth, valid_mask, n_bins_x, n_bins_y)

    # return both values as a tuple
    return sir, coherence


# default "no result" values for the coherence shuffle; merged into other dicts with **
_NULL_COHERENCE_BOOTSTRAP = {'coherence_bootstrap_mean': float('nan'),
                             'coherence_bootstrap_p95':  float('nan'),
                             'coherence_bootstrap_sig':  None}


# _run_bootstrap(...) -> dict with the SI and coherence shuffle results; also saves a figure.
def _run_bootstrap(spike_frame_indices: np.ndarray, t: np.ndarray,
                   # frame bin indices
                   beh_bx: np.ndarray, beh_by: np.ndarray,
                   # occupancy and valid bins
                   occ_map: np.ndarray, valid_mask: np.ndarray,
                   # grid and bin size
                   n_bins_x: int, n_bins_y: int, bin_cm: float,
                   # the real (unshuffled) SI and coherence, plus the .ntt path for the figure
                   real_sir: float, real_coherence: float, ntt_path: str,
                   # figure-name suffix ('full', 'first_half', 'second_half')
                   label: str = '') -> dict:
    """Location-shuffling bootstrap (matches MATLAB calcSI_v3_locshuf).

    For each permutation, the position time series is circularly shifted by a
    random number of frames (at least 20 s from either end), while spike-to-frame
    assignments remain unchanged.  This decorrelates spikes from positions without
    altering the animal's occupancy statistics.

    SIR and spatial coherence share the same N_BOOTSTRAP shuffled spike trains
    (one shuffle per iteration feeds both metrics via
    `_sir_and_coherence_from_spikes_locshuf`) rather than running two separate
    bootstraps, halving the shuffle-generation cost.
    """
    # no spikes -> nothing to test
    if len(spike_frame_indices) == 0:
        return {'bootstrap_mean': float('nan'),
                'bootstrap_p95':  float('nan'),
                'bootstrap_sig':  False,
                # `**dict` unpacks another dict's key/value pairs into this one
                **_NULL_COHERENCE_BOOTSTRAP}

    # number of tracking frames (moving frames only)
    n_frames      = len(t)
    # 20 s -> 600 frames at 30 fps
    MARGIN_FRAMES = int(SIR_SHUFFLE_MARGIN_S * fps)   # frames excluded at either end (matches Fenton reference)

    # session too short (< 40 s) to allow shifts of at least 20 s from either end
    if n_frames <= 2 * MARGIN_FRAMES:
        return {'bootstrap_mean': float('nan'),
                'bootstrap_p95':  float('nan'),
                # None = "not tested" (as opposed to False = "tested, not significant")
                'bootstrap_sig':  None,
                **_NULL_COHERENCE_BOOTSTRAP}

    # arrays for the 1000 shuffled SI and coherence values
    sir_i = np.zeros(N_BOOTSTRAP, dtype=np.float64)
    coh_i = np.zeros(N_BOOTSTRAP, dtype=np.float64)
    for i in range(N_BOOTSTRAP):
        # random shift in [600, n_frames - 600] frames (randint includes both ends)
        rnd  = random.randint(MARGIN_FRAMES, n_frames - MARGIN_FRAMES)
        # compute both shuffled metrics and store them at position i (tuple assignment)
        sir_i[i], coh_i[i] = _sir_and_coherence_from_spikes_locshuf(
            # arguments of the call, continued over several lines
            spike_frame_indices, rnd,
            beh_bx, beh_by,
            occ_map, valid_mask,
            n_bins_x, n_bins_y, bin_cm)

    # mean of the null SI distribution (how much SI a cell gets by chance, given its spike count
    #   and the animal's path -- SI is positively biased for low spike counts / sparse sampling)
    bootstrap_mean = float(np.mean(sir_i))
    # 95th percentile of the null = significance threshold
    bootstrap_p95  = float(np.percentile(sir_i, 95))
    # significant (one-sided p < 0.05) if the real SI is above that threshold
    bootstrap_sig  = bool(real_sir > bootstrap_p95)

    # coherence may be NaN for some shuffles (too few valid bins); keep only finite values
    coh_valid = coh_i[np.isfinite(coh_i)]
    if len(coh_valid) > 0:
        # same summary for coherence
        coherence_bootstrap_mean = float(np.mean(coh_valid))
        coherence_bootstrap_p95  = float(np.percentile(coh_valid, 95))
        # significant if real coherence > 95th percentile; None if the real coherence is NaN
        coherence_bootstrap_sig  = (bool(real_coherence > coherence_bootstrap_p95)
                                     if np.isfinite(real_coherence) else None)
    else:
        coherence_bootstrap_mean = float('nan')
        coherence_bootstrap_p95  = float('nan')
        coherence_bootstrap_sig  = None

    # ── histogram plots (SIR left, coherence right) ─────────────────────────────
    # Thread-safe Object-Oriented Figure generation prevents race conditions
    # 10 x 4.5 inch figure
    fig = Figure(figsize=(10, 4.5))
    # attach a renderer (the variable itself is not used afterwards)
    canvas = FigureCanvasAgg(fig)
    # add_subplot(121) = 1 row, 2 columns, panel 1 (left)
    ax = fig.add_subplot(121)

    # histogram (100 bars) of the shuffled SI values
    hist_result = ax.hist(sir_i, 100, color='black')
    counts: np.ndarray = np.asarray(hist_result[0])
    # tallest bar, used to scale the box plot and marker line
    max_count = float(counts.max()) if counts.max() > 0 else 1.0

    # horizontal box plot of the null below the histogram; whis=[5, 95] puts the whiskers at the
    #   5th and 95th percentiles, so the right whisker end = the significance threshold
    box_plot = ax.boxplot(sir_i, whis=[5, 95], orientation='horizontal', showfliers=False, # type: ignore
                          positions=[-max_count / 10], widths=max_count / 15)
    # red dash-dot vertical line at the real SI
    ax.plot([real_sir, real_sir], [0, max_count], 'r-.')

    # x coordinate of the end of the upper whisker = 95th percentile of the null
    ci95 = float(box_plot['whiskers'][1].get_xdata()[1])
    # title lines
    bs_av = f"Bootstrap mean SIR = {bootstrap_mean:.3f} bits/spike"
    up_ci = f"Upper 95% CI = {ci95:.3f} bits/spike"
    bs_p  = (f"Cell SIR = {real_sir:.3f} bits/spike "
             f"({'p < 0.05' if bootstrap_sig else 'ns'})")
    ax.set_title(bs_av + '\n' + up_ci + '\n' + bs_p, multialignment='center')

    # y-axis ticks at 0, 25, 50, 75, 100 % of the tallest bar
    ax.set_yticks([0, max_count * 0.25, max_count * 0.5,
                   max_count * 0.75, max_count])
    # their labels: list comprehension over the tuple of tick values, each rounded to 1 decimal
    #   and converted to text with str()
    ax.set_yticklabels([str(round(v, 1))
                        for v in (0, max_count * 0.25, max_count * 0.5,
                                  max_count * 0.75, max_count)])
    ax.set_ylabel('count')
    ax.set_xlabel('spatial information rate (bits/spike)')
    # y range leaves room below 0 for the box plot
    ax.set_ylim(-max_count / 7, max_count * 1.1)

    # right panel: coherence null distribution
    ax2 = fig.add_subplot(122)
    if len(coh_valid) > 0:
        # same plotting recipe as the SI panel
        hist_result2 = ax2.hist(coh_valid, 100, color='black')
        counts2: np.ndarray = np.asarray(hist_result2[0])
        max_count2 = float(counts2.max()) if counts2.max() > 0 else 1.0

        box_plot2 = ax2.boxplot(coh_valid, whis=[5, 95], orientation='horizontal', showfliers=False, # type: ignore
                                positions=[-max_count2 / 10], widths=max_count2 / 15)
        # real coherence marker (only if it is a number)
        if np.isfinite(real_coherence):
            ax2.plot([real_coherence, real_coherence], [0, max_count2], 'r-.')

        # 95th percentile of the coherence null
        ci95_2 = float(box_plot2['whiskers'][1].get_xdata()[1])
        bs_av2 = f"Bootstrap mean coherence (z) = {coherence_bootstrap_mean:.3f}"
        up_ci2 = f"Upper 95% CI = {ci95_2:.3f}"
        bs_p2  = (f"Cell coherence (z) = {real_coherence:.3f} "
                 f"({'p < 0.05' if coherence_bootstrap_sig else 'ns'})")
        ax2.set_title(bs_av2 + '\n' + up_ci2 + '\n' + bs_p2, multialignment='center')

        # y ticks and labels as above
        ax2.set_yticks([0, max_count2 * 0.25, max_count2 * 0.5,
                        max_count2 * 0.75, max_count2])
        ax2.set_yticklabels([str(round(v, 1))
                            for v in (0, max_count2 * 0.25, max_count2 * 0.5,
                                      max_count2 * 0.75, max_count2)])
        ax2.set_ylim(-max_count2 / 7, max_count2 * 1.1)
    else:
        # placeholder text when no coherence shuffle values exist
        ax2.text(0.5, 0.5, 'coherence shuffle not available\n(insufficient valid bins)',
                 ha='center', va='center', transform=ax2.transAxes)
    ax2.set_ylabel('count')
    ax2.set_xlabel('spatial coherence (Fisher z)')
    fig.tight_layout()

    # save into <ntt folder>/<SHUFFLING_SUBDIR>/<unit>_<label>_bootstrapping.png
    ntt_name   = os.path.splitext(os.path.basename(ntt_path))[0]
    save_dir   = os.path.join(os.path.dirname(ntt_path), SHUFFLING_SUBDIR)
    os.makedirs(save_dir, exist_ok=True)
    lbl_suffix = f'_{label}' if label else ''
    save_path  = os.path.join(save_dir, f'{ntt_name}{lbl_suffix}_bootstrapping.png')
    fig.savefig(save_path, dpi=150)
    print(f'  [SAVED] {save_path}')

    # return the rounded results; NaN values are passed through unrounded
    return {'bootstrap_mean': round(bootstrap_mean, 4),
            'bootstrap_p95':  round(bootstrap_p95,  4),
            'bootstrap_sig':  bootstrap_sig,
            # conditional expression on two lines, inside parentheses
            'coherence_bootstrap_mean': (round(coherence_bootstrap_mean, 4)
                                         if not np.isnan(coherence_bootstrap_mean) else float('nan')),
            'coherence_bootstrap_p95':  (round(coherence_bootstrap_p95, 4)
                                         if not np.isnan(coherence_bootstrap_p95) else float('nan')),
            'coherence_bootstrap_sig':  coherence_bootstrap_sig}


# ── Tracking load/clean/convert ────────────────────────────────────────────────

# _load_tracking(csv_path, arena_width_cm, half=None) -> (x_cm, y_cm, t).
#   half = None (whole session), 'first' or 'second' (first/second half of the frames).
def _load_tracking(csv_path: str, arena_width_cm: float,
                   half: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load, clean, and pixel→cm-convert one tracking file (no position
    smoothing applied — see `_smooth_tracking_position`).

    Returns (x_cm, y_cm, t) with t in the same (µs) time base as spike
    timestamps. Arrays are empty if no valid tracking samples remain.
    """
    # read .xlsx with read_excel, anything else with read_csv (conditional expression in parentheses)
    data = (pd.read_excel(csv_path) if csv_path.lower().endswith('.xlsx')
            else pd.read_csv(csv_path))

    # 'cm' files: columns selected by POSITION
    if COORD_UNITS == 'cm':
        # Column A = timestamp, column D = x (cm), column E = y (cm)
        # .iloc[:, k] = all rows of column number k (0-based: 0 = A, 3 = D, 4 = E)
        t = np.asarray(data.iloc[:, 0], dtype=float)
        x = np.asarray(data.iloc[:, 3], dtype=float)
        y = np.asarray(data.iloc[:, 4], dtype=float)
    # 'pixel' files: columns selected by NAME
    else:
        x = np.asarray(data['x'],    dtype=float)
        y = np.asarray(data['y'],    dtype=float)
        t = np.asarray(data['time'], dtype=float)

    # np.isin(x, [1, -1]) is True where x equals 1 or -1 -- the tracker's "position lost" codes;
    #   `~` inverts it so those frames are dropped. NOTE (review): in 'cm' mode a genuine
    #   x position of exactly 1.0 cm would also be dropped (harmless but worth knowing).
    mask = ~np.isin(x, [1, -1])
    # apply the same mask to all three arrays (multiple assignment)
    x, y, t = x[mask], y[mask], t[mask]

    # ── Jump removal (POS_JUMP_THRESH_CMS, e.g. 90 cm/s = 0.00009 cm/µs) ───────
    # Per-frame speed is the forward difference frame i -> i+1. The last frame
    # has no successor, so it takes the speed (and dt validity) of the
    # second-last frame.
    if len(t) >= 2:
        if COORD_UNITS == 'cm':
            # positions already in cm
            units_per_cm = 1.0
        else:
            # x/y are still in pixels here: convert the cm/s threshold to px/s
            # using a robust (percentile) span estimate, so the glitch frames
            # this filter removes can't inflate the pixel->cm scale.
            # span between the 0.5th and 99.5th percentile = arena extent ignoring outliers
            x_span = np.percentile(x, 99.5) - np.percentile(x, 0.5)
            y_span = np.percentile(y, 99.5) - np.percentile(y, 0.5)
            # pixels per cm, assuming the larger span equals the arena width
            units_per_cm = max(x_span, y_span) / arena_width_cm
        # threshold in units per microsecond (90 cm/s x units/cm x 1e-6 s/µs)
        jump_thresh_per_us = POS_JUMP_THRESH_CMS * units_per_cm * 1e-6

        # time step between frames (µs)
        dt_step   = np.diff(t)
        # distance moved between frames
        dxy_step  = np.hypot(np.diff(x), np.diff(y))
        # steps with positive dt
        valid_step = dt_step > 0
        # speed per step (units per µs); 0 where dt is invalid
        speed_step = np.zeros_like(dxy_step)
        speed_step[valid_step] = dxy_step[valid_step] / dt_step[valid_step]

        # give every frame the speed of the step that STARTS at it; last frame repeats the previous
        speed    = np.append(speed_step, speed_step[-1])
        valid_dt = np.append(valid_step, valid_step[-1])

        # indices of frames with a valid dt AND a plausible speed
        keep = np.where(valid_dt & (speed < jump_thresh_per_us))[0]
        # keep only those frames
        x, y, t = x[keep], y[keep], t[keep]

    # np.argsort returns the indices that would sort t; reorder all arrays chronologically
    order = np.argsort(t)
    x, y, t = x[order], y[order], t[order]

    # optional split into first / second half by FRAME COUNT
    if half is not None:
        # midpoint frame index
        mid = len(t) // 2
        if half == 'first':
            # frames 0 .. mid-1
            x, y, t = x[:mid], y[:mid], t[:mid]
        elif half == 'second':
            # frames mid .. end
            x, y, t = x[mid:], y[mid:], t[mid:]

    # nothing left: return the (empty) arrays as float64
    if len(t) == 0:
        return x.astype(np.float64), y.astype(np.float64), t.astype(np.float64)

    # ── Pixel → cm conversion ──────────────────────────────────────────────────
    if COORD_UNITS == 'cm':
        # Coordinates are already in cm – just zero the origin for binning.
        # shift so the minimum x and y are 0 (bins then start at the arena's lower-left corner)
        #   NOTE (review): when half is 'first'/'second' the origin comes from that half's own
        #   minimum, so half-session maps are not bin-aligned with each other or with the full map
        #   (the split-half stability function deliberately avoids this, see below).
        x_cm = x - x.min()
        y_cm = y - y.min()
    else:
        # full extent in pixels
        x_span = x.max() - x.min()
        y_span = y.max() - y.min()
        # pixels per cm (larger extent = arena width)
        px_per_cm = max(x_span, y_span) / arena_width_cm

        # convert to cm with origin at 0
        x_cm = (x - x.min()) / px_per_cm
        y_cm = (y - y.min()) / px_per_cm

    return x_cm, y_cm, t


# _instantaneous_speed(x_cm, y_cm, t_us) -> speed (cm/s) of every frame.
def _instantaneous_speed(x_cm: np.ndarray, y_cm: np.ndarray, t_us: np.ndarray) -> np.ndarray:
    """Per-frame running speed (cm/s): frame i gets the speed of step i -> i+1;
    the last frame gets the second-last frame's speed."""
    # frame intervals in seconds
    dt_s = np.diff(t_us) * 1e-6
    with np.errstate(invalid='ignore', divide='ignore'):
        # distance / time
        speed = np.hypot(np.diff(x_cm), np.diff(y_cm)) / dt_s
    # undefined where dt <= 0
    speed[dt_s <= 0] = np.nan
    # pad to one value per frame
    return np.append(speed, speed[-1])


# _plot_and_save_speed(csv_path, arena_width_cm) -> None. Quality-control plot of speed before and
#   after position smoothing, with the immobility threshold drawn, saved next to the tracking file.
def _plot_and_save_speed(csv_path: str, arena_width_cm: float) -> None:
    """Plot pre- vs. post-smoothing running speed for one tracking file and
    save the comparison next to it.
    """
    # load and clean the tracking (no smoothing yet)
    x_raw, y_raw, t = _load_tracking(csv_path, arena_width_cm)
    if len(t) < 2:
        print(f'  [SKIP] Not enough valid tracking samples to plot speed: {csv_path}')
        return

    # smoothed positions
    x_smooth, y_smooth = _smooth_tracking_position(x_raw, y_raw, t)

    # speed from raw and from smoothed positions
    speed_raw    = _instantaneous_speed(x_raw,    y_raw,    t)
    speed_smooth = _instantaneous_speed(x_smooth, y_smooth, t)

    # time in seconds from session start
    time_s = (t - t[0]) * 1e-6

    # new figure with a single panel (add_subplot(111) = 1 row, 1 col, panel 1)
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    # raw speed in light grey, smoothed speed in black
    ax.plot(time_s, speed_raw,    linewidth=0.5, color='0.75', label='pre-smoothing (raw)')
    ax.plot(time_s, speed_smooth, linewidth=0.7, color='black', label='post-smoothing')
    # horizontal dashed red line at the immobility cut-off
    ax.axhline(IMMOBILITY_SPEED_CMS, color='red', linewidth=0.8, linestyle='--',
               label=f'immobility cut-off ({IMMOBILITY_SPEED_CMS} cm/s)')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('speed (cm/s)')
    # title = tracking file name without extension
    ax.set_title(os.path.splitext(os.path.basename(csv_path))[0])
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()

    # save as <tracking file name>_speed.png in the same folder
    csv_name  = os.path.splitext(os.path.basename(csv_path))[0]
    save_path = os.path.join(os.path.dirname(csv_path), f'{csv_name}_speed.png')
    fig.savefig(save_path, dpi=150)
    print(f'  [SAVED] {save_path}')


# tolerance (ms) for flagging irregular frame intervals in the dt_s plot
DT_S_TOL_MS = 1.0  # flag a frame interval as anomalous if it differs from the expected 1000/fps by more than this (ms)


# _plot_and_save_dt_s(csv_path, arena_width_cm, fps) -> None. Quality-control plot of the time
#   between tracking frames (should be ~33.33 ms at 30 fps; gaps reveal dropped frames).
def _plot_and_save_dt_s(csv_path: str, arena_width_cm: float, fps: float) -> None:
    """Plot the tracking inter-frame interval (dt_s) in the time domain for one
    tracking file and save it next to it, flagging intervals that deviate from
    the expected 1000/fps spacing (e.g. 33.33 ms at 30 fps).
    """
    # `_` discards x and y; only t is needed
    _, _, t = _load_tracking(csv_path, arena_width_cm)
    if len(t) < 2:
        print(f'  [SKIP] Not enough valid tracking samples to plot dt_s: {csv_path}')
        return

    # frame intervals, µs -> ms
    dt_ms  = np.diff(t) * 1e-3        # us -> ms
    time_s = (t[1:] - t[0]) * 1e-6    # time of each interval (end timestamp), relative to session start

    # expected interval in ms (33.33 at 30 fps)
    expected_ms  = 1000.0 / fps
    # intervals deviating by more than 1 ms
    is_anomalous = np.abs(dt_ms - expected_ms) > DT_S_TOL_MS
    # how many
    n_anom       = int(is_anomalous.sum())

    fig    = Figure()
    canvas = FigureCanvasAgg(fig)
    ax     = fig.add_subplot(111)
    # every interval as a small grey dot (no connecting line: linestyle='None')
    ax.plot(time_s, dt_ms, marker='.', markersize=2, linestyle='None', color='0.4', alpha=0.5, label='dt_s')
    # green dashed line at the expected interval
    ax.axhline(expected_ms, color='green', linewidth=0.8, linestyle='--',
               label=f'expected ({expected_ms:.2f} ms)')
    # anomalous intervals highlighted in red, drawn on top (zorder=3)
    if n_anom > 0:
        ax.scatter(time_s[is_anomalous], dt_ms[is_anomalous], color='red', s=10, zorder=3,
                   label=f'off by >{DT_S_TOL_MS:.1f} ms (n={n_anom})')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('inter-frame interval dt_s (ms)')
    ax.set_title(os.path.splitext(os.path.basename(csv_path))[0])
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()

    # save as <tracking file name>_dt_s.png
    csv_name  = os.path.splitext(os.path.basename(csv_path))[0]
    save_path = os.path.join(os.path.dirname(csv_path), f'{csv_name}_dt_s.png')
    fig.savefig(save_path, dpi=150)

    # percentage of anomalous intervals, printed with the min/max interval
    pct = 100 * n_anom / len(dt_ms)
    print(f'  [SAVED] {save_path}   ({n_anom}/{len(dt_ms)} intervals off by >{DT_S_TOL_MS:.1f} ms, '
          f'{pct:.2f}%; range {dt_ms.min():.2f}-{dt_ms.max():.2f} ms)')


# _plot_and_save_ratemap(...) -> None. Trajectory + spikes panel and smoothed rate-map panel for
#   one cell; place cells get extra field panel(s). `list[dict] | None = None` = optional list of dicts.
def _plot_and_save_ratemap(fr_map: np.ndarray, valid_mask: np.ndarray,
                            # grid size and bin size (cm)
                            n_bins_x: int, n_bins_y: int, target_bin_cm: float,
                            # .ntt path and the metrics dict (for the title)
                            ntt_path: str, metrics: dict,
                            # full (unfiltered) trajectory, for the grey path
                            x_cm_all: np.ndarray, y_cm_all: np.ndarray,
                            # moving-only trajectory (spike_frame indexes into these)
                            x_cm: np.ndarray, y_cm: np.ndarray,
                            # frame index of every kept spike
                            spike_frame: np.ndarray,
                            # detected fields (place cells only)
                            fields_threshold: list[dict] | None = None,
                            # '2D' or 'Circle'
                            arena_type: str | None = None,
                            # angular ratemap data for Circle arenas
                            ctx_circular: dict | None = None) -> None:
    """Plot the raw trajectory (with spike locations overlaid) beside the
    fixed-bin, Gaussian-smoothed firing-rate map for one cell, and save both
    next to its .ntt file, in a 'ratemaps' folder (mirrors the 'speed
    modulation' folder convention above).

    `fields_threshold` is the list of threshold-method fields detected for
    this cell, passed only for confirmed (final-verdict) place cells; it is
    None for every other cell, in which case no extra panel is drawn. When
    given, one extra panel (2-D grid arenas: field boundaries contoured on
    the rate map) or two extra polar panels (Circle/ring arenas: angular
    tuning curve + ring rate map, via `ctx_circular`) are appended to this
    same figure instead of being saved as a separate file.
    """
    # np.ma.masked_where makes a "masked array": invalid bins are hidden (drawn blank) in imshow
    display_map = np.ma.masked_where(~valid_mask, fr_map)

    # whether to draw field panels
    show_fields    = fields_threshold is not None
    # nested conditional expression: 2 extra panels for Circle, 1 for 2-D, 0 if no fields
    n_field_panels = (2 if arena_type == 'Circle' else 1) if show_fields else 0
    # total panel count
    n_panels       = 2 + n_field_panels

    # figure width scales with the number of panels (6 inches each)
    fig = Figure(figsize=(6 * n_panels, 6.5))
    FigureCanvasAgg(fig)

    # ── left panel: trajectory + spike locations ────────────────────────────
    ax_traj = fig.add_subplot(1, n_panels, 1)
    # Changed to plot x_cm_all, y_cm_all so missing immobility gaps aren't straight-lined
    # full path as a thin grey line
    ax_traj.plot(x_cm_all, y_cm_all, color='0.6', linewidth=0.5, label='trajectory')
    # Filtered tracking arrays preserve accurate spike alignment
    # position of each spike: fancy indexing x_cm[spike_frame] picks the frame of every spike
    ax_traj.plot(x_cm[spike_frame], y_cm[spike_frame], '.', color='red',
                 markersize=3, label='spikes')
    ax_traj.set_xlabel('x (cm)')
    ax_traj.set_ylabel('y (cm)')
    # equal scaling of x and y so the arena is not distorted
    ax_traj.set_aspect('equal', adjustable='box')
    # legend placed just above the panel (anchored at its top-left), 2 columns
    ax_traj.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=2, fontsize=8)

    # ── second panel: firing-rate map ────────────────────────────────────────
    ax_rate = fig.add_subplot(1, n_panels, 2)
    # axis limits in cm: [x_min, x_max, y_min, y_max]
    extent = [0, n_bins_x * target_bin_cm, 0, n_bins_y * target_bin_cm]
    # imshow draws a 2-D array as an image. .T (transpose) because the map is indexed [x, y]
    #   while imshow expects [row = y, column = x]; origin='lower' puts y = 0 at the bottom;
    #   cmap='jet' colour scale; interpolation='nearest' shows crisp square bins
    im = ax_rate.imshow(display_map.T, origin='lower', extent=extent,
                         cmap='jet', interpolation='nearest')
    # colour bar for the rate map
    fig.colorbar(im, ax=ax_rate, label='firing rate (Hz)')
    ax_rate.set_xlabel('x (cm)')
    ax_rate.set_ylabel('y (cm)')

    # ── extra panel(s): detected place field(s), confirmed place cells only ──
    if show_fields:
        # `assert` stops with an error if the condition is False; here it also tells the type
        #   checker that fields_threshold is not None below
        assert fields_threshold is not None
        if arena_type == 'Circle' and ctx_circular is not None:
            # two polar (angle/radius) axes
            ax_line = fig.add_subplot(1, n_panels, 3, projection='polar')
            ax_ring = fig.add_subplot(1, n_panels, 4, projection='polar')
            # polar tuning curve with field bins marked
            _plot_ratemap_polar(ax_line, ctx_circular['fr_smooth'], ctx_circular['valid_mask'],
                                 fields_threshold, ctx_circular['n_bins_theta'],
                                 ctx_circular['bin_width_rad'],
                                 f'Threshold method, angular ({len(fields_threshold)} field(s))')
            # ring-shaped heat map with field outlines; returns the colour mesh for the colour bar
            im_ring = _plot_ratemap_ring(ax_ring, ctx_circular['fr_smooth'], ctx_circular['valid_mask'],
                                          fields_threshold, ctx_circular['n_bins_theta'],
                                          ctx_circular['bin_width_rad'], ctx_circular['r_min'],
                                          ctx_circular['r_max'], 'Binned rate map (ring)')
            fig.colorbar(im_ring, ax=ax_ring, label='Hz', fraction=0.046, pad=0.1)
        else:
            # 2-D arena: rate map with black field contours
            ax_field = fig.add_subplot(1, n_panels, 3)
            im_field = _plot_ratemap_with_field_boundaries_2d(
                ax_field, fr_map, valid_mask, fields_threshold, n_bins_x, n_bins_y,
                f'Threshold method ({len(fields_threshold)} field(s))')
            fig.colorbar(im_field, ax=ax_field, label='Hz')

    # figure title: unit name + peak/mean rate + SI; dict.get(key) returns None if key is absent
    ntt_name = os.path.splitext(os.path.basename(ntt_path))[0]
    peak_fr  = metrics.get('peak_fr')
    mean_fr  = metrics.get('mean_fr')
    sir      = metrics.get('sir')
    title = ntt_name
    if peak_fr is not None and mean_fr is not None and sir is not None:
        # `+=` on a string appends text
        title += f'\npeak = {peak_fr:.2f} Hz, mean = {mean_fr:.2f} Hz, SIR = {sir:.3f}'
    # suptitle = title above all panels
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    # save into <ntt folder>/<RATEMAPS_SUBDIR>/<unit>_ratemap.png
    save_dir  = os.path.join(os.path.dirname(ntt_path), RATEMAPS_SUBDIR)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{ntt_name}_ratemap.png')
    fig.savefig(save_path, dpi=150)
    print(f'  [SAVED] {save_path}')


# _compute_coherence(fr_map, valid_mask, n_bins_x, n_bins_y) -> spatial coherence (Fisher z).
#   Muller & Kubie (1989): correlation between each bin's rate and the mean rate of its neighbours.
#   High coherence = firing changes smoothly across space (one compact field); low = salt-and-pepper.
#   NOTE (review): Muller & Kubie computed coherence on the UNSMOOTHED map. Here it is computed on
#   the SMOOTHED map, where neighbours are correlated by construction, so absolute values are
#   inflated. The shuffle test is still fair (shuffles are smoothed identically), but values are
#   not directly comparable with papers that use raw maps.
def _compute_coherence(fr_map: np.ndarray, valid_mask: np.ndarray,
                       n_bins_x: int, n_bins_y: int) -> float:
    """Spatial coherence: correlation of each bin's rate (from `fr_map`) with the
    mean rate of its (up to 8) occupied neighbours, Fisher Z-transformed.
    """
    # np.argwhere -> array of (x, y) index pairs of every valid bin
    valid_idx    = np.argwhere(valid_mask)
    # lists collecting each bin's own rate and its neighbour-mean rate
    fr_bin_vals  = []
    fr_nbr_means = []
    # loop over valid bins; each row of valid_idx is unpacked into bx, by
    for bx, by in valid_idx:
        # list comprehension with two nested `for`s (the 3x3 neighbourhood) and conditions:
        nbr_vals = [
            # the neighbour's rate ...
            fr_map[bx + dx, by + dy]
            # ... for every offset dx, dy in {-1, 0, 1}
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            # excluding the centre bin itself
            if not (dx == 0 and dy == 0)
            # inside the grid in x
            and 0 <= bx + dx < n_bins_x
            # inside the grid in y
            and 0 <= by + dy < n_bins_y
            # and the neighbour is a valid (visited) bin
            and valid_mask[bx + dx, by + dy]
        ]
        # an empty list is "falsy"; only bins with >= 1 valid neighbour are used
        if nbr_vals:
            fr_bin_vals.append(fr_map[bx, by])
            fr_nbr_means.append(float(np.mean(nbr_vals)))

    # need at least 3 pairs for a correlation
    if len(fr_bin_vals) > 2:
        # Pearson r between bin rates and neighbour means (p-value discarded with `_`)
        r_coef, _ = pearsonr(fr_bin_vals, fr_nbr_means)
        # clip r away from ±1 so the Fisher transform stays finite
        r_coef    = float(np.clip(r_coef, -0.9999, 0.9999)) # type: ignore
        # Fisher z = 0.5 · ln((1 + r) / (1 - r)) = arctanh(r); makes r approximately normal
        return float(0.5 * np.log((1 + r_coef) / (1 - r_coef)))
    # not enough data
    return float('nan')


# _split_half_smoothed_maps(ctx, bin_cm) -> dict with the first-half and second-half smoothed
#   rate maps (and their valid masks), both on the full-session grid.
def _split_half_smoothed_maps(ctx: dict, bin_cm: float) -> dict:
    """First/second-half occupancy-normalised ratemaps on the SAME full-session
    grid (ctx['beh_bx'] / ctx['beh_by']), each Gaussian-smoothed within its own
    valid bins. Shared by _compute_split_half_stability and
    _plot_and_save_split_half_ratemap so the plot shows exactly the maps the
    stability score is computed from."""
    # unpack what is needed from the context dict built by compute_metrics
    t           = ctx['t']
    beh_bx      = ctx['beh_bx']
    beh_by      = ctx['beh_by']
    spike_frame = ctx['spike_frame']
    dt_frames   = ctx['dt_frames']
    n_bins_x    = ctx['n_bins_x']
    n_bins_y    = ctx['n_bins_y']

    # split point = middle frame (moving frames only)
    mid = len(t) // 2

    # occupancy maps (seconds) of each half
    occ_first  = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    occ_second = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    # add each frame's duration to its bin; [:mid] = first half, [mid:] = second half
    np.add.at(occ_first,  (beh_bx[:mid], beh_by[:mid]), dt_frames[:mid])
    np.add.at(occ_second, (beh_bx[mid:], beh_by[mid:]), dt_frames[mid:])

    # spikes whose frame is in the first half; the rest are second half
    spk_first  = spike_frame < mid
    spk_second = ~spk_first

    # spike-count maps of each half
    spike_first  = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    spike_second = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    # +1 at the bin of every spike of that half
    np.add.at(spike_first,  (beh_bx[spike_frame[spk_first]],  beh_by[spike_frame[spk_first]]),  1.0)
    np.add.at(spike_second, (beh_bx[spike_frame[spk_second]], beh_by[spike_frame[spk_second]]), 1.0)

    # valid bins of each half, using that half's own occupancy and visit counts
    valid_first  = _valid_bin_mask(occ_first,  _visits_2d(beh_bx[:mid], beh_by[:mid], n_bins_x, n_bins_y))
    valid_second = _valid_bin_mask(occ_second, _visits_2d(beh_bx[mid:], beh_by[mid:], n_bins_x, n_bins_y))

    # raw rate maps of each half (spikes / seconds) in their valid bins
    fr_first_raw  = np.zeros_like(occ_first)
    fr_second_raw = np.zeros_like(occ_second)
    fr_first_raw[valid_first]   = spike_first[valid_first]   / occ_first[valid_first]
    fr_second_raw[valid_second] = spike_second[valid_second] / occ_second[valid_second]

    # dict(key=value, ...) builds a dict with keyword syntax; the maps are smoothed here
    return dict(valid_first=valid_first, valid_second=valid_second,
                fr_first_smooth=_gaussian_smooth(fr_first_raw,  valid_first,  bin_cm),
                fr_second_smooth=_gaussian_smooth(fr_second_raw, valid_second, bin_cm))


# _compute_split_half_stability(ctx, bin_cm, min_valid_bins=5) -> dict(stability_score, p, n_bins).
#   Stability = Pearson r between the first-half and second-half rate maps. A true place cell
#   fires in the same place throughout the session -> r close to 1. This is the "Stability
#   (Split-Half)" criterion of Royer et al. 2010 in the paper's Table 1, and a session-level
#   cousin of the paper's even-odd correlation (which the paper found tracks ANOVA, i.e. the
#   trial-consistency dimension that SI largely misses).
#   NOTE (review): the parametric p-value treats every bin as independent, but smoothing makes
#   neighbouring bins correlated, so this p is optimistic (too small). A shuffle would be safer.
def _compute_split_half_stability(ctx: dict, bin_cm: float,
                                   min_valid_bins: int = STABILITY_MIN_BINS) -> dict:
    """Split-half spatial stability: Pearson correlation of the Gaussian-
    SMOOTHED firing-rate map between the first and second half of the
    session (each half smoothed within its own valid bins, see
    _split_half_smoothed_maps).

    Uses the SAME spatial grid as the full-session ratemap (ctx['beh_bx'] /
    ctx['beh_by'], built from the full session's extent) for both halves, so
    corresponding array entries refer to the same physical bin in both --
    unlike the independent first/second-half rows above (which each derive
    their own pixel->cm scaling and grid from only that half's tracking
    samples and are therefore NOT bin-aligned with each other). The session
    is split at the midpoint frame (matching the first/second-half
    convention used elsewhere in this script).

    NaN-bin handling: a bin that is not valid in either half has no
    reliable firing-rate estimate for that half, so such bins are dropped
    from the correlation rather than imputed to 0 Hz (which would spuriously
    push the correlation toward zero or negative) or imputed via the other
    half's rate (which would inflate it). Only bins valid in BOTH halves are
    correlated -- the standard split-half approach in the place-cell
    literature. If fewer than `min_valid_bins` bins survive this, the score
    is left as NaN rather than reported from too few points.
    """
    # default result (NaN / 0) returned if stability cannot be computed
    result = {'stability_score': float('nan'), 'stability_p_value': float('nan'),
              'stability_n_bins': 0}
    # an empty dict is falsy -> `not ctx` is True when compute_metrics failed
    if not ctx or len(ctx['t']) < 2:
        return result

    # first/second-half smoothed maps
    halves       = _split_half_smoothed_maps(ctx, bin_cm)
    # bins valid in BOTH halves
    common_valid = halves['valid_first'] & halves['valid_second']

    # how many such bins
    n_common = int(common_valid.sum())
    result['stability_n_bins'] = n_common
    # too few bins for a meaningful correlation
    if n_common < min_valid_bins:
        return result

    # rates of the common bins in each half (1-D arrays in the same bin order)
    fr_first  = halves['fr_first_smooth'][common_valid]
    fr_second = halves['fr_second_smooth'][common_valid]

    # correlation undefined if either map is flat
    if np.std(fr_first) == 0 or np.std(fr_second) == 0:
        return result

    # Pearson correlation between the two halves
    r, p = pearsonr(fr_first, fr_second)
    result['stability_score']   = round(float(r), 4)
    result['stability_p_value'] = round(float(p), 4)
    return result


# _plot_and_save_split_half_ratemap(ctx, stability, ntt_path, target_bin_cm) -> None.
#   Figure: full-session map on top, first-half and second-half maps below, stability r at bottom.
def _plot_and_save_split_half_ratemap(ctx: dict, stability: dict,
                                       ntt_path: str, target_bin_cm: float) -> None:
    """Combined figure for a confirmed place cell: full-session ratemap
    (top, spanning both columns), first-half ratemap (bottom-left) and
    second-half ratemap (bottom-right), with the split-half stability score
    annotated at the bottom.

    First/second-half maps come from `_split_half_smoothed_maps` -- the SAME
    full-session grid and smoothed maps `_compute_split_half_stability`
    correlates (not the independently-binned first/second-half rows computed
    elsewhere), so this plot is bin-aligned with, and visually explains, the
    reported stability score.
    """
    # grid size, full-session smoothed map and valid mask
    n_bins_x    = ctx['n_bins_x']
    n_bins_y    = ctx['n_bins_y']
    fr_full     = ctx['fr_smooth']
    valid_full  = ctx['valid_mask']

    if len(ctx['t']) < 2:
        return

    # half-session maps (same function as the stability score)
    halves           = _split_half_smoothed_maps(ctx, target_bin_cm)
    valid_first      = halves['valid_first']
    valid_second     = halves['valid_second']
    fr_first_smooth  = halves['fr_first_smooth']
    fr_second_smooth = halves['fr_second_smooth']

    # masked arrays hide invalid bins in the images
    disp_full   = np.ma.masked_where(~valid_full,   fr_full)
    disp_first  = np.ma.masked_where(~valid_first,  fr_first_smooth)
    disp_second = np.ma.masked_where(~valid_second, fr_second_smooth)

    # axis extent in cm
    extent = [0, n_bins_x * target_bin_cm, 0, n_bins_y * target_bin_cm]
    # common colour-scale maximum across the three maps (so colours are comparable);
    #   .count() = number of unmasked elements (0 -> use 0.0 instead of .max())
    vmax = max((disp_full.max()   if disp_full.count()   else 0.0),
               (disp_first.max()  if disp_first.count()  else 0.0),
               (disp_second.max() if disp_second.count() else 0.0))
    # guard against an all-zero map
    vmax = float(vmax) if vmax > 0 else 1.0

    fig = Figure(figsize=(10, 10))
    FigureCanvasAgg(fig)
    # 4-column grid: the full-session map occupies the two middle columns
    # (centered over the figure), each half-session map takes two columns.
    gs = fig.add_gridspec(2, 4)

    # gs[row, col_start:col_end] selects a block of grid cells (slice end is exclusive)
    ax_full   = fig.add_subplot(gs[0, 1:3])
    ax_first  = fig.add_subplot(gs[1, 0:2])
    ax_second = fig.add_subplot(gs[1, 2:4])

    # full-session map with a fixed colour range 0..vmax
    im = ax_full.imshow(disp_full.T, origin='lower', extent=extent,
                         cmap='jet', interpolation='nearest', vmin=0, vmax=vmax)
    ax_full.set_title('Full session')
    ax_full.set_xlabel('x (cm)')
    ax_full.set_ylabel('y (cm)')
    # Colorbar in an inset axes so it doesn't shrink/shift the centered map.
    # inset_axes([x0, y0, width, height]) in axes fractions -> a thin bar right of the map
    cax = ax_full.inset_axes([1.03, 0.0, 0.04, 1.0])
    fig.colorbar(im, cax=cax, label='firing rate (Hz)')

    # first-half map, same colour range
    ax_first.imshow(disp_first.T, origin='lower', extent=extent,
                     cmap='jet', interpolation='nearest', vmin=0, vmax=vmax)
    ax_first.set_title('First half')
    ax_first.set_xlabel('x (cm)')
    ax_first.set_ylabel('y (cm)')

    # second-half map, same colour range
    ax_second.imshow(disp_second.T, origin='lower', extent=extent,
                      cmap='jet', interpolation='nearest', vmin=0, vmax=vmax)
    ax_second.set_title('Second half')
    ax_second.set_xlabel('x (cm)')
    ax_second.set_ylabel('y (cm)')

    # unit name as the overall title
    ntt_name = os.path.splitext(os.path.basename(ntt_path))[0]
    fig.suptitle(ntt_name, fontsize=11)

    # stability values for the caption
    r      = stability.get('stability_score')
    p      = stability.get('stability_p_value')
    n_bins = stability.get('stability_n_bins')
    # isinstance(r, float) checks the type; np.isnan(r) checks for NaN
    if r is not None and not (isinstance(r, float) and np.isnan(r)):
        stab_text = f'Split-half stability: r = {r:.3f}, p = {p:.3g}, n_bins = {n_bins}'
    else:
        stab_text = 'Split-half stability: n/a'
    # fig.text places text in figure coordinates (0.5 = centre, 0.02 = near the bottom)
    fig.text(0.5, 0.02, stab_text, ha='center', va='bottom',
              fontsize=11, fontweight='bold')

    # rect=(left, bottom, right, top) leaves space for the caption and title
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))

    # save into the ratemaps folder
    save_dir  = os.path.join(os.path.dirname(ntt_path), RATEMAPS_SUBDIR)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{ntt_name}_split_half_ratemap.png')
    fig.savefig(save_path, dpi=150)
    print(f'  [SAVED] {save_path}')


# ── Core metric computation ───────────────────────────────────────────────────

# compute_metrics(csv_path, ntt_path, arena_width_cm, target_bin_cm, half=None) -> (metrics, ctx)
#   metrics = dict of single-cell spatial metrics; ctx = dict of intermediate arrays (positions,
#   bins, occupancy, rate maps) reused later by the shuffle test, stability, fields and plots.
def compute_metrics(csv_path: str, ntt_path: str,
                    arena_width_cm: float, target_bin_cm: float,
                    half: str | None = None) -> tuple:

    # ── 1-2. Load, clean, and convert tracking ─────────────────────────────────
    # positions (cm) and timestamps (µs), optionally only one half of the session
    x_cm, y_cm, t = _load_tracking(csv_path, arena_width_cm, half=half)

    # no tracking left: return zero metrics and an empty ctx
    if len(t) == 0:
         return ({'n_spikes': 0, 'n_discarded': 0, 'peak_fr': 0.0, 'mean_fr': 0.0, 'sir': 0.0}, {})

    # ── 2b. Smooth tracking position (jump removal + Gaussian smoothing) ──────
    x_cm, y_cm = _smooth_tracking_position(x_cm, y_cm, t)

    # Per-frame occupancy time, computed on the contiguous (pre-immobility-
    # filter) tracking so that removing immobile epochs below doesn't turn the
    # resulting gaps into extra occupancy for the frame after each gap.
    # np.empty allocates an uninitialised array (every element is set below)
    dt_frames        = np.empty(len(t), dtype=np.float64)
    # first frame has no predecessor: credit one nominal frame duration (1/30 s)
    dt_frames[0]     = 1.0 / fps
    # true interval before each subsequent frame (s)
    raw_dt           = np.diff(t) * 1e-6

    # Cap dt_frames to avoid artificial occupancy hotspots when tracking drops
    # E.g., if a gap is > ~2 frames, only credit the standard frame rate to prevent inflation
    # cap = 2 frame durations (66.7 ms)
    max_frame_s      = 2.0 / fps
    # occupancy credited to frame i = time since frame i-1, capped (element-wise minimum)
    dt_frames[1:]    = np.minimum(raw_dt, max_frame_s)

    # ── 2c. Immobility filter (removes e.g. sharp-wave-ripple firing) ─────────
    # Speed from the smoothed position (frame i -> i+1; the last frame takes the
    # second-last frame's speed). Frames below IMMOBILITY_SPEED_CMS are dropped
    # from every spatial analysis below, and so are the spikes assigned to them
    # (step 4). The unfiltered track is kept in ctx for speed modulation, which
    # needs a contiguous time series and applies its own SPEED_MIN_CMS cut.
    # keep references to the full (unfiltered) track
    x_cm_all, y_cm_all, t_all = x_cm, y_cm, t
    if len(t) >= 2:
        # moving = NOT (speed < 0.5 cm/s). Written this way so frames with NaN speed count as
        #   moving (NaN < 0.5 is False -> ~False = True), i.e. they are kept
        moving = ~(_instantaneous_speed(x_cm, y_cm, t) < IMMOBILITY_SPEED_CMS)   # NaN speed kept
    else:
        # a single frame: treat as moving
        moving = np.ones(len(t), dtype=bool)

    # animal never moved: nothing to analyse
    if not moving.any():
        return ({'n_spikes': 0, 'n_discarded': 0, 'peak_fr': 0.0, 'mean_fr': 0.0, 'sir': 0.0}, {})

    # ── 3. Bin tracking positions ─────────────────────────────────────────────
    # Replaced np.ceil alone with max(1, ...) to guard against perfectly linear 0-dim sets
    # number of bins along x and y = arena extent / bin size, rounded up (at least 1)
    #   (x_cm / y_cm still include the immobile frames here, so the grid covers the whole
    #   tracked area, not only the places the animal ran through)
    n_bins_x = max(1, int(np.ceil(x_cm.max() / target_bin_cm)))
    n_bins_y = max(1, int(np.ceil(y_cm.max() / target_bin_cm)))

    # bin index of every frame: position / bin size, truncated to an integer, clipped into range
    #   (the clip puts x == x_max into the last bin instead of one past it)
    beh_bx = np.clip((x_cm / target_bin_cm).astype(int), 0, n_bins_x - 1)
    beh_by = np.clip((y_cm / target_bin_cm).astype(int), 0, n_bins_y - 1)

    # ── 4. Load spikes & nearest-timestamp assignment (50 ms gate) ────────────
    # np.memmap maps the binary file into memory without loading it all; offset=16*1024 skips the
    #   16-KB Neuralynx text header; dtype=ntt_dtype interprets the rest as an array of spike records
    spike_data = np.memmap(ntt_path, dtype=ntt_dtype, mode='r', offset=16 * 1024)
    # keep only spikes of sorted clusters (cell_number 0 = unsorted / noise)
    #   NOTE: all non-zero clusters in this .ntt are pooled, so each .ntt is assumed to contain ONE unit
    spike_data = spike_data[spike_data['cell_number'] != 0]  # drop unsorted/discarded cluster 0
    # spike timestamps (µs) as sorted float64
    spike_ts   = np.sort(spike_data['timestamp'].astype(np.float64))

    # half-session analysis: keep only spikes within the half's time range (± 50 ms)
    if half is not None:
        t_lo = t[0]  - MAX_GAP_US
        t_hi = t[-1] + MAX_GAP_US
        spike_ts = spike_ts[(spike_ts >= t_lo) & (spike_ts <= t_hi)]

    # Spikes are matched against the full (unfiltered) track first, so a spike
    # fired during an immobile epoch is assigned to its immobile frame and then
    # removed -- rather than being re-assigned to the nearest moving frame.
    # for every spike: index of the first frame with t >= spike time
    idx   = np.searchsorted(t, spike_ts, side='left')
    # candidate frames just before (idx-1) and at/after (idx) the spike, clipped into range
    idx_l = np.clip(idx - 1, 0, len(t) - 1)
    idx_r = np.clip(idx,     0, len(t) - 1)
    # time distance to each candidate
    dist_l   = np.abs(spike_ts - t[idx_l])
    dist_r   = np.abs(spike_ts - t[idx_r])
    # nearest frame (left one on ties)
    nearest  = np.where(dist_l <= dist_r, idx_l, idx_r)
    # distance to that nearest frame
    min_dist = np.minimum(dist_l, dist_r)

    # spike has a tracking frame within 50 ms
    valid_spike   = min_dist <= MAX_GAP_US
    # ... and that frame is a moving frame (moving[nearest] looks up each spike's frame)
    moving_spike  = valid_spike & moving[nearest]
    # spikes without a nearby frame
    n_discarded   = int((~valid_spike).sum())
    # spikes during immobility
    n_immobile    = int((valid_spike & ~moving[nearest]).sum())
    # spikes used for all spatial metrics
    n_spikes      = int(moving_spike.sum())
    spike_ts_all  = spike_ts[valid_spike]          # 50 ms-gated, before immobility filter

    # Re-index onto the moving-only frames.
    # np.cumsum(moving) counts moving frames up to each position; - 1 gives every moving frame its
    #   new index in the filtered arrays (e.g. moving = [T, F, T] -> new index [0, -, 1])
    new_frame_idx = np.cumsum(moving) - 1
    # new (filtered) frame index of every kept spike
    spike_frame   = new_frame_idx[nearest[moving_spike]]
    # drop immobile frames from all per-frame arrays
    x_cm, y_cm, t = x_cm[moving], y_cm[moving], t[moving]
    beh_bx, beh_by = beh_bx[moving], beh_by[moving]
    dt_frames     = dt_frames[moving]

    # spatial bin of every kept spike
    sp_bx = beh_bx[spike_frame]
    sp_by = beh_by[spike_frame]

    # ── 5. Build occupancy and spike-count maps ────────────────────────────────
    # 2-D arrays shaped (n_bins_x, n_bins_y)
    occ_map   = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    spike_map = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)

    # occupancy: add each frame's duration (s) to its bin -> seconds spent per bin
    np.add.at(occ_map,   (beh_bx, beh_by), dt_frames)
    # spike counts: add 1 per spike to its bin
    np.add.at(spike_map, (sp_bx,  sp_by),  1.0)

    # number of separate visits to each bin, and the resulting valid-bin mask (>= 2 visits)
    visit_map  = _visits_2d(beh_bx, beh_by, n_bins_x, n_bins_y)
    valid_mask = _valid_bin_mask(occ_map, visit_map)

    # ── 6. Non-smoothed firing rate map ──────────────────────────────────────
    # rate (Hz) = spike count / seconds, in valid bins only (others stay 0)
    fr_raw = np.zeros_like(occ_map)
    fr_raw[valid_mask] = spike_map[valid_mask] / occ_map[valid_mask]

    # ── 7. Smoothed firing rate map ───────────────────────────────────────────
    # Gaussian-smoothed copy (sigma 3 cm), used for coherence, fields, stability and display
    fr_smooth = _gaussian_smooth(fr_raw, valid_mask, target_bin_cm)

    # ── 8. Compute metrics ────────────────────────────────────────────────────
    # spike_ts / spike_frame / t / x_cm / y_cm / beh_* / dt_frames: moving
    # frames only (immobility-filtered). *_all: full track and 50 ms-gated
    # spikes before the immobility filter -- used only for speed modulation.
    # dict(...) bundles every intermediate array under a name
    ctx = dict(spike_ts=spike_ts[moving_spike], spike_frame=spike_frame, t=t,
               x_cm=x_cm, y_cm=y_cm,
               x_cm_all=x_cm_all, y_cm_all=y_cm_all, t_all=t_all,
               spike_ts_all=spike_ts_all,
               beh_bx=beh_bx, beh_by=beh_by,
               occ_map=occ_map, valid_mask=valid_mask,
               dt_frames=dt_frames,
               n_bins_x=n_bins_x, n_bins_y=n_bins_y,
               fr_raw=fr_raw, fr_smooth=fr_smooth)

    # no valid bin at all: return zero metrics (but keep ctx)
    if not valid_mask.any():
        return ({'n_spikes': n_spikes, 'n_discarded': n_discarded,
                 'n_spikes_immobile': n_immobile,
                 'peak_fr': 0.0, 'mean_fr': 0.0, 'sir': 0.0,
                 'sparsity': 0.0, 'coherence': float('nan')}, ctx)

    # Peak / mean firing rate, SIR, sparsity: RAW (unsmoothed) ratemap.
    # total time in valid bins
    total_occ_s = occ_map[valid_mask].sum()
    # occupancy probability p_i of each valid bin
    pi_flat     = occ_map[valid_mask] / total_occ_s
    # rate r_i of each valid bin
    ri_flat     = fr_raw[valid_mask]
    # occupancy-weighted mean rate Σ p_i r_i (= spikes in valid bins / time in valid bins)
    r_mean      = float(np.sum(pi_flat * ri_flat))

    # peak rate = highest single-bin RAW rate.
    #   NOTE (review): a raw single-bin maximum is noisy: a bin with ~0.1 s occupancy and one spike
    #   reads 10 Hz (only >= 2 visits is required, no minimum occupancy). This value gates the
    #   1-25 Hz place-cell window, so one poorly sampled bin can push a cell outside it.
    peak_fr = float(fr_raw[valid_mask].max())
    # mean rate (the paper's "average firing rate" feature)
    mean_fr = r_mean

    # Skaggs spatial information (bits/spike) on the raw map
    sir = _compute_sir(occ_map, fr_raw, valid_mask)

    # Sparsity = (Σ pi ri)² / Σ pi ri²   (Skaggs et al. 1996), same bins as SIR
    sparsity = _compute_sparsity(occ_map, fr_raw, valid_mask)

    # Spatial coherence: SMOOTHED map vs 8-neighbour mean (Fisher Z)
    coherence = _compute_coherence(fr_smooth, valid_mask, n_bins_x, n_bins_y)

    # assemble the metrics dict (values rounded to 4 decimals; NaN coherence kept as NaN)
    metrics = {
        'n_spikes':    n_spikes,
        'n_discarded': n_discarded,
        'n_spikes_immobile': n_immobile,
        'peak_fr':     round(peak_fr,   4),
        'mean_fr':     round(mean_fr,   4),
        'sir':         round(sir,       4),
        'sparsity':    round(sparsity,  4),
        'coherence':   round(coherence, 4) if not np.isnan(coherence) else float('nan'),
    }

    # Ratemap plot is deferred to _run_job, which draws it only after the
    # cell's final place-cell verdict and its detected fields (if any) are
    # known, so confirmed place cells can get the field-boundary panel(s)
    # merged into this same figure (see _plot_and_save_ratemap).

    # return both dicts as a tuple
    return metrics, ctx


# ── Place-field isolation – "threshold method" (2-D grid) ──────────────────────
# Run only on cells that pass the place-cell criteria (see _run_job). Reuses the
# fr_smooth / occ_map / valid_mask already built by compute_metrics for the full
# session (ctx['fr_smooth'] etc.) rather than rebuilding the ratemap.
# A "place field" is a contiguous patch of elevated firing. The paper's feature definitions
# match this approach: "Place Field Width = number of bins exceeding 20 % of the peak" and
# "Number of Place Fields = distinct contiguous regions meeting minimum width and firing criteria".

# _connected_components_8(qualifies, n_bins_x, n_bins_y) -> list of regions; each region is a list
#   of (x, y) bins that touch each other (8-connectivity: diagonal neighbours count as touching).
def _connected_components_8(qualifies: np.ndarray, n_bins_x: int, n_bins_y: int) -> list:
    """8-connected component labelling of the bins where `qualifies` is True."""
    # non-qualifying bins are marked "visited" up front so they are never added to a region
    visited = ~qualifies
    # result list
    components = []
    # scan every bin of the grid
    for i in range(n_bins_x):
        for j in range(n_bins_y):
            # already in a region, or not qualifying -> skip
            if visited[i, j]:
                continue
            # start a new region at this bin (depth-first "flood fill")
            region = []
            # stack (list used last-in-first-out) of bins still to expand
            stack = [(i, j)]
            visited[i, j] = True
            # while the stack is non-empty
            while stack:
                # .pop() removes and returns the last item
                bx, by = stack.pop()
                region.append((bx, by))
                # look at the 8 neighbours
                for ddx in (-1, 0, 1):
                    for ddy in (-1, 0, 1):
                        # skip the bin itself
                        if ddx == 0 and ddy == 0:
                            continue
                        # neighbour coordinates
                        nx, ny = bx + ddx, by + ddy
                        # inside the grid and not yet visited (i.e. qualifying and unclaimed)
                        if 0 <= nx < n_bins_x and 0 <= ny < n_bins_y and not visited[nx, ny]:
                            visited[nx, ny] = True
                            # expand it later
                            stack.append((nx, ny))
            # region complete
            components.append(region)
    return components


# detect_place_fields_threshold_2d(base_metrics, ctx, target_bin_cm) -> (fields, peak_field_verified)
#   fields = list of dicts describing each field; peak_field_verified = bool (see docstring).
def detect_place_fields_threshold_2d(base_metrics: dict, ctx: dict, target_bin_cm: float) -> tuple[list[dict], bool]:
    """"Threshold method" place-field isolation: a bin only qualifies for a
    field if its Gaussian-smoothed rate is >= METHOD2_RATE_THRESHOLD_FRAC of
    the cell's peak (smoothed) rate AND above the cell's mean firing rate;
    8-connected components of qualifying bins spanning >= MIN_FIELD_SIZE_BINS
    contiguous bins (no discontinuity) are reported as fields (peak bin +
    rate-weighted centre of mass).

    Returns (fields, peak_field_verified). `peak_field_verified` is False
    when the cell's own peak-firing bin (the RAW-map global max that gated
    it into place-cell status via PLACE_CELL_MIN/MAX_PEAK_FR) does not fall
    inside any of the detected fields above, AND no other detected field's
    own raw-map peak falls in that same [PLACE_CELL_MIN_PEAK_FR,
    PLACE_CELL_MAX_PEAK_FR] window either -- i.e. the place-cell verdict
    isn't backed by any field that actually spans >= MIN_FIELD_SIZE_BINS
    contiguous bins at a real (unsmoothed) rate in range. Callers should
    overrule the place-cell verdict when this is False.
    """
    # maps and grid size from compute_metrics
    valid_mask  = ctx['valid_mask']
    fr_smooth   = ctx['fr_smooth']
    fr_raw      = ctx['fr_raw']
    n_bins_x    = ctx['n_bins_x']
    n_bins_y    = ctx['n_bins_y']

    # total number of valid (visited) bins = the "occupied area"
    total_valid_bins = int(valid_mask.sum())
    if total_valid_bins == 0:
        return [], False

    # peak of the SMOOTHED map (fields are defined on the smoothed map)
    peak_fr = float(fr_smooth[valid_mask].max())
    # occupancy-weighted mean rate from the raw map (dict.get with a default of 0.0)
    mean_fr = float(base_metrics.get('mean_fr', 0.0))
    # 20 % of the smoothed peak
    rate_threshold = METHOD2_RATE_THRESHOLD_FRAC * peak_fr
    # minimum field size (9 bins)
    min_size_bins  = MIN_FIELD_SIZE_BINS

    # qualifying bins: valid AND >= 20 % of peak AND above the mean rate
    qualifies = valid_mask & (fr_smooth >= rate_threshold) & (fr_smooth > mean_fr)

    # geometric centre of the grid (in bin units), used to flag the most central field
    centre_bin = (n_bins_x / 2.0, n_bins_y / 2.0)
    # running minimum distance (starts at infinity)
    best_centre_dist  = np.inf
    # index of the most central field so far (None = none yet)
    centre_field_idx  = None
    fields  = []
    regions = []   # bin coords per field, parallel to `fields` -- reused below
                   # to test whether the cell's raw-map peak bin lies inside
                   # a detected field.

    # examine every contiguous patch of qualifying bins
    for region in _connected_components_8(qualifies, n_bins_x, n_bins_y):
        # too small to be a field -> skip
        if len(region) < min_size_bins:
            continue

        # x and y bin indices of the region's bins (list comprehensions over (x, y) tuples)
        bxs   = np.array([b[0] for b in region])
        bys   = np.array([b[1] for b in region])
        # smoothed rate of each bin of the region (paired fancy indexing)
        rates = fr_smooth[bxs, bys]

        # bin with the highest smoothed rate inside this field
        peak_local_idx = int(np.argmax(rates))
        peak_bin = (int(bxs[peak_local_idx]), int(bys[peak_local_idx]))
        peak_val = float(rates[peak_local_idx])

        # Raw (unsmoothed) rate at that SAME bin -- lets you directly see how
        # much the Gaussian smoothing inflated/deflated the reported peak.
        peak_val_raw_same_bin = float(fr_raw[peak_bin[0], peak_bin[1]])

        # Raw (unsmoothed) rate at this field's own highest-raw-rate bin,
        # which may sit at a different bin than peak_bin above -- reused
        # below in the peak_field_verified fallback check (see
        # detect_place_fields_threshold_2d's docstring).
        raw_rates          = fr_raw[bxs, bys]
        raw_peak_local_idx = int(np.argmax(raw_rates))
        field_raw_max_bin  = (int(bxs[raw_peak_local_idx]), int(bys[raw_peak_local_idx]))
        field_raw_max_val  = float(raw_rates[raw_peak_local_idx])

        # rate-weighted centre of mass: Σ(rate · x) / Σ rate (and the same for y)
        total_rate = float(rates.sum())
        com_x = float(np.sum(rates * bxs) / total_rate)
        com_y = float(np.sum(rates * bys) / total_rate)

        # field size in bins, cm², and as % of all valid bins
        n_bins_field = len(region)
        area_cm2     = n_bins_field * (target_bin_cm ** 2)
        pct_area     = 100.0 * n_bins_field / total_valid_bins
        # text list of the field's bins, e.g. '3-4;3-5;4-4'; str.join glues the pieces with ';'
        #   (generator expression inside join; sorted() orders the tuples)
        bin_coords_str = ';'.join(f'{bx}-{by}' for bx, by in sorted(region))

        # distance from the field's centre of mass to the grid centre
        dist_to_centre = float(np.hypot(com_x - centre_bin[0], com_y - centre_bin[1]))
        # remember the most central field (len(fields) = the index this field is about to get)
        if dist_to_centre < best_centre_dist:
            best_centre_dist = dist_to_centre
            centre_field_idx = len(fields)

        # one dict per field with every descriptor (becomes one row of the PlaceFields sheet)
        fields.append({
            # 1-based field number
            'field_number':          len(fields) + 1,
            # bin of the smoothed peak and its smoothed / raw rate
            'peak_bin_x':            peak_bin[0],
            'peak_bin_y':            peak_bin[1],
            'peak_fr_hz':            round(peak_val, 4),
            'peak_fr_hz_raw':        round(peak_val_raw_same_bin, 4),
            # bin and value of the highest RAW rate in the field
            'field_raw_max_bin_x':   field_raw_max_bin[0],
            'field_raw_max_bin_y':   field_raw_max_bin[1],
            'field_raw_max_fr_hz':   round(field_raw_max_val, 4),
            # centre of mass in bins and in cm
            'com_bin_x':             round(com_x, 3),
            'com_bin_y':             round(com_y, 3),
            'com_cm_x':              round(com_x * target_bin_cm, 2),
            'com_cm_y':              round(com_y * target_bin_cm, 2),
            # size
            'n_bins':                n_bins_field,
            'area_cm2':              round(area_cm2, 2),
            'pct_of_occupied_area':  round(pct_area, 2),
            # bounding box in bins
            'bbox_x_min':            int(bxs.min()),
            'bbox_x_max':            int(bxs.max()),
            'bbox_y_min':            int(bys.min()),
            'bbox_y_max':            int(bys.max()),
            # bounding box in cm (+1 so the max edge is the far side of the last bin)
            'bbox_cm_x_min':         round(bxs.min() * target_bin_cm, 2),
            'bbox_cm_x_max':         round((bxs.max() + 1) * target_bin_cm, 2),
            'bbox_cm_y_min':         round(bys.min() * target_bin_cm, 2),
            'bbox_cm_y_max':         round((bys.max() + 1) * target_bin_cm, 2),
            # list of bins (used to redraw the field outline)
            'bin_coords':            bin_coords_str,
            # detection parameters, recorded for traceability
            'total_occupied_bins':   total_valid_bins,
            'min_field_size_bins':   min_size_bins,
            'rate_threshold_hz':     round(rate_threshold, 4),
            'mean_fr_threshold_hz':  round(mean_fr, 4),
            # set to True for the most central field after the loop
            'is_centre_field':       False,
        })
        regions.append(region)

    # flag the most central field
    if centre_field_idx is not None:
        fields[centre_field_idx]['is_centre_field'] = True

    # ── Verify the place-cell peak bin is actually backed by a field ───────
    # metrics['peak_fr'] (the RAW-map global max that gated this cell into
    # place-cell status via PLACE_CELL_MIN/MAX_PEAK_FR, see compute_metrics)
    # is independent of the smoothed-map, >=MIN_FIELD_SIZE_BINS field search
    # above -- so that peak bin need not fall inside any detected field
    # (e.g. n_fields == 0, or the peak sits in an isolated bin elsewhere).
    # When it doesn't, fall back to asking whether some OTHER detected
    # field's own raw-map peak (field_raw_max_fr_hz above) still lands
    # inside the place-cell peak-rate window; if none does, nothing actually
    # backs the place-cell verdict and the caller should overrule it.
    # raw map with invalid bins set to -infinity so they can never be the maximum
    masked_raw       = np.where(valid_mask, fr_raw, -np.inf)
    # np.argmax on a 2-D array returns a FLAT index ...
    global_peak_flat = int(np.argmax(masked_raw))
    # ... np.unravel_index converts it back to (x, y); tuple(...) of a generator makes an (int, int)
    global_peak_bin  = tuple(int(v) for v in np.unravel_index(global_peak_flat, masked_raw.shape))

    # any(...) is True if at least one region contains the global peak bin
    if any(global_peak_bin in region for region in regions):
        peak_field_verified = True
    else:
        # fallback: does any field's own raw peak lie inside the 1-25 Hz window?
        #   (chained comparison a < b < c means a < b and b < c)
        peak_field_verified = any(
            PLACE_CELL_MIN_PEAK_FR < f['field_raw_max_fr_hz'] < PLACE_CELL_MAX_PEAK_FR
            for f in fields
        )

    return fields, peak_field_verified


# _field_mask_from_bin_coords('3-4;3-5', n_bins_x, n_bins_y) -> boolean 2-D mask of the field.
def _field_mask_from_bin_coords(bin_coords: str, n_bins_x: int, n_bins_y: int) -> np.ndarray:
    # all False to start
    mask = np.zeros((n_bins_x, n_bins_y), dtype=bool)
    # empty string -> empty mask
    if not bin_coords:
        return mask
    # 'a-b;c-d' -> ['a-b', 'c-d']
    for pair in bin_coords.split(';'):
        # 'a-b' -> ['a', 'b']
        bx_str, by_str = pair.split('-')
        # convert to int and mark that bin
        mask[int(bx_str), int(by_str)] = True
    return mask


# _plot_ratemap_with_field_boundaries_2d(ax, ...) -> image handle. Smoothed rate map with each
#   field outlined in black. `ax` has no type hint (any matplotlib Axes).
def _plot_ratemap_with_field_boundaries_2d(ax, fr_smooth: np.ndarray, valid_mask: np.ndarray,
                                            fields: list[dict], n_bins_x: int, n_bins_y: int, title: str):
    # hide invalid bins
    display_map = np.ma.masked_where(~valid_mask, fr_smooth)
    # draw the map in bin units (no extent given)
    im = ax.imshow(display_map.T, origin='lower', cmap='jet', interpolation='nearest')

    for field in fields:
        # rebuild the field's mask from its bin list (.get with '' as default)
        field_mask = _field_mask_from_bin_coords(field.get('bin_coords', ''), n_bins_x, n_bins_y)
        if not field_mask.any():
            continue
        # contour at level 0.5 of a 0/1 mask = the field's boundary line
        ax.contour(field_mask.T.astype(float), levels=[0.5], colors='black', linewidths=1.5)

    ax.set_title(title)
    ax.set_xlabel('x bin')
    ax.set_ylabel('y bin')
    # returned so the caller can add a colour bar
    return im


# ── Place-field isolation – "threshold method" (Circle / angular ratemap) ──────
# The Circle (ring) track's animal position is fundamentally 1-D (its bearing
# theta around the ring), not 2-D -- binning (x, y) on the square Cartesian
# grid above and 8-connected-component searching it does not respect that
# topology: the ring's curvature makes the square grid sample the track
# unevenly, so a single true field can get sliced into several disconnected
# 2-D components (spurious "multifield" cells) purely from binning artifacts,
# independent of any real firing discontinuity. For sessions whose folder
# path contains 'Circle' (see _is_circle_arena), field detection instead
# builds a fresh 1-D angular ratemap -- every tracking sample and spike is
# assigned to a bin by its bearing (theta) around the track's fitted centre
# only (radial distance from centre is discarded), collapsing the ratemap to
# a single 1-D array around the ring, smoothed with a wrap-around 1-D
# Gaussian kernel -- and finds contiguous runs of qualifying bins on that
# array, wrapping around the theta=0/2*pi seam. This reuses the tracking/
# spike data already loaded by compute_metrics (ctx['x_cm'], ctx['y_cm'],
# ctx['spike_frame'], ctx['dt_frames']) rather than reloading anything; only
# the binning/smoothing/field-search differ from the 2-D grid method above.
# NOTE (review): only FIELD DETECTION uses the angular map. SI, sparsity, coherence, the shuffle
#   test and the place-cell verdict for Circle sessions still come from the 2-D grid map.

# percentile trimmed from each end of the radius distribution when estimating the track width
RADIAL_RANGE_CLIP_PCTILE = 0.5   # trim this many percentiles off each end of the
                                  # radial-distance distribution before estimating
                                  # the track width, so a handful of tracking-jitter
                                  # outliers can't stretch/blur the estimate


# _is_circle_arena(dirpath) -> True if the folder path contains 'circle' (any capitalisation).
def _is_circle_arena(dirpath: str) -> bool:
    """True if the session folder path indicates a circular-alley track
    (path contains 'Circle', case-insensitive)."""
    # .lower() makes the test case-insensitive; `in` on strings tests for a substring
    return 'circle' in dirpath.lower()


# _fit_ring_centre(x_cm, y_cm) -> (cx, cy). Least-squares circle fit (Kasa method).
#   A circle (x-cx)² + (y-cy)² = R² can be rewritten as  x² + y² = 2cx·x + 2cy·y + (R² - cx² - cy²),
#   which is LINEAR in the unknowns (2cx, 2cy, c) -> solvable with ordinary least squares.
def _fit_ring_centre(x_cm: np.ndarray, y_cm: np.ndarray) -> tuple:
    """Algebraic (Kasa) circle fit to the tracked positions, returning
    (cx, cy) -- the track's centre in the same cm frame as x_cm/y_cm. Assumes
    the great majority of samples lie on (or near) the ring, which holds for
    a circular-alley track."""
    # design matrix: columns x, y, 1 (np.column_stack places 1-D arrays side by side)
    A = np.column_stack([x_cm, y_cm, np.ones_like(x_cm)])
    # right-hand side x² + y²
    b = x_cm ** 2 + y_cm ** 2
    # least-squares solution of A·sol = b; `sol, *_ =` keeps the first return value and discards
    #   the rest (residuals, rank, singular values) into the throw-away list `_`
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    # sol[0] = 2cx, sol[1] = 2cy
    cx = sol[0] / 2.0
    cy = sol[1] / 2.0
    return cx, cy


# _circular_mean_rad(angles_rad, weights) -> weighted circular mean angle in [0, 2π).
#   Angles cannot be averaged directly (the mean of 350° and 10° is 0°, not 180°), so each angle is
#   turned into a unit vector, the vectors are summed, and the angle of the sum is taken.
def _circular_mean_rad(angles_rad: np.ndarray, weights: np.ndarray) -> float:
    # weighted sum of the sine (y) components
    s = float(np.sum(weights * np.sin(angles_rad)))
    # weighted sum of the cosine (x) components
    c = float(np.sum(weights * np.cos(angles_rad)))
    # arctan2(y, x) = angle of the resultant vector in (-π, π]; % 2π maps it into [0, 2π)
    return float(np.arctan2(s, c)) % (2.0 * np.pi)


# _gaussian_kernel_1d(sigma_bins) -> normalised 1-D Gaussian weights (±3 sigma).
def _gaussian_kernel_1d(sigma_bins: float) -> np.ndarray:
    """1D Gaussian kernel, sigma given in bins, truncated at 3 sigma."""
    # half-width in bins
    radius = max(1, int(np.ceil(3 * sigma_bins)))
    # offsets -radius..+radius
    ax = np.arange(-radius, radius + 1)
    # Gaussian weights
    kernel = np.exp(-(ax ** 2) / (2 * sigma_bins ** 2))
    # normalise to sum 1
    kernel /= kernel.sum()
    return kernel


# _gaussian_smooth_circular(fr_map, valid_mask, bin_cm) -> smoothed 1-D angular rate map, with
#   wrap-around at the ends (the last bin is the neighbour of the first bin).
def _gaussian_smooth_circular(fr_map: np.ndarray, valid_mask: np.ndarray, bin_cm: float) -> np.ndarray:
    """Gaussian smoothing on a 1D angular rate map. The axis is circular --
    the track is a closed loop -- so it is padded by wrapping rather than
    zero-filling, which would otherwise create a spurious rate dip at the
    arbitrary theta=0/2*pi seam."""
    # sigma in bins (each angular bin is ~target_bin_cm of arc)
    sigma_bins = GAUSSIAN_SIGMA_CM / bin_cm
    kernel = _gaussian_kernel_1d(sigma_bins)
    # kernel radius (number of bins of padding needed on each side)
    kr = kernel.shape[0] // 2

    # invalid bins -> 0 rate and 0 weight (normalised convolution, as in _gaussian_smooth)
    fr_in   = np.where(valid_mask, fr_map, 0.0)
    mask_in = valid_mask.astype(np.float64)

    # nested helper function: pad an array by kr bins on both ends with WRAPPED values
    #   (end values copied before the start and vice versa). `xp` is the array module
    #   (numpy or cupy), so the same helper works on CPU and GPU arrays.
    def _pad(arr, xp):
        return xp.pad(arr, (kr, kr), mode='wrap')

    # number of angular bins
    n_theta = fr_map.shape[0]

    # GPU path (same structure as _convolve_pair)
    if _GPU:
        fr_gpu   = _pad(cp.asarray(fr_in,   dtype=cp.float64), cp)
        mask_gpu = _pad(cp.asarray(mask_in, dtype=cp.float64), cp)
        kern_gpu = cp.asarray(kernel, dtype=cp.float64)
        _wait_for_gpu_slot()
        _gpu_semaphore.acquire()
        try:
            # convolve the padded rates
            smoothed_fr_full = cp.asnumpy(
                cp_convolve(fr_gpu, kern_gpu, mode='constant', cval=0.0)
            )
            # convolve the padded weights
            smoothed_weights_full = cp.asnumpy(
                cp_convolve(mask_gpu, kern_gpu, mode='constant', cval=0.0)
            )
        finally:
            _gpu_semaphore.release()
    # CPU path
    else:
        fr_pad   = _pad(fr_in,   np)
        mask_pad = _pad(mask_in, np)
        smoothed_fr_full      = convolve(fr_pad,   kernel, mode='constant', cval=0.0)
        smoothed_weights_full = convolve(mask_pad, kernel, mode='constant', cval=0.0)

    # cut the padding off again -> back to n_theta bins
    smoothed_fr      = smoothed_fr_full[kr:kr + n_theta]
    smoothed_weights = smoothed_weights_full[kr:kr + n_theta]

    # weighted average of the valid neighbours; invalid bins set to 0
    smoothed = np.zeros_like(smoothed_fr)
    valid_weights = smoothed_weights > 0
    smoothed[valid_weights] = smoothed_fr[valid_weights] / smoothed_weights[valid_weights]
    smoothed[~valid_mask] = 0.0
    return smoothed


# _build_ratemap_circular(ctx, target_bin_cm) -> dict with the 1-D angular occupancy, rate maps
#   and ring geometry (centre, radii, bin width).
def _build_ratemap_circular(ctx: dict, target_bin_cm: float) -> dict:
    """1-D angular ratemap for the Circle (ring) track, built from the same
    tracking/spike data already loaded by compute_metrics (ctx['x_cm'],
    ctx['y_cm'], ctx['spike_frame'], ctx['dt_frames']) -- see module note
    above. Every sample's bearing (theta) around the track's fitted centre
    is binned; radial distance from centre is discarded. Also reports its
    own occupancy-weighted mean_fr (rather than reusing the 2-D grid's),
    so the field-detection mean-rate threshold is self-consistent with the
    map it's applied to. Uses the immobility-filtered (moving-only) frames
    and spikes in ctx, same as the 2-D ratemap."""
    # moving-only positions, spike frames and frame durations
    x_cm        = ctx['x_cm']
    y_cm        = ctx['y_cm']
    spike_frame = ctx['spike_frame']
    dt_frames   = ctx['dt_frames']

    # ring centre
    cx, cy = _fit_ring_centre(x_cm, y_cm)
    # radial distance of every frame from the centre
    r_cm        = np.hypot(x_cm - cx, y_cm - cy)
    # bearing (angle) of every frame around the centre
    theta_rad   = np.arctan2(y_cm - cy, x_cm - cx)             # (-pi, pi]
    # np.mod maps negative angles into [0, 2π)
    theta_0_2pi = np.mod(theta_rad, 2.0 * np.pi)               # [0, 2*pi)

    # inner and outer radius of the track (0.5th and 99.5th percentiles of the radius)
    r_min = float(np.percentile(r_cm, RADIAL_RANGE_CLIP_PCTILE))
    r_max = float(np.percentile(r_cm, 100.0 - RADIAL_RANGE_CLIP_PCTILE))
    # track width (cm), never negative
    track_width_cm = max(r_max - r_min, 0.0)

    # typical radius (median)
    r_mean = float(np.median(r_cm))
    # angular bin width so that each bin spans ~target_bin_cm of arc (arc = radius x angle)
    bin_width_rad = target_bin_cm / max(r_mean, 1e-6)          # arc length ~= target_bin_cm
    # whole number of bins around the full circle
    n_bins_theta = max(1, int(round(2.0 * np.pi / bin_width_rad)))
    # recompute the width so n_bins_theta bins exactly fill 2π
    bin_width_rad = 2.0 * np.pi / n_bins_theta                 # re-close evenly around the ring

    # angular bin of every frame, and of every spike
    beh_bt = np.clip((theta_0_2pi / bin_width_rad).astype(int), 0, n_bins_theta - 1)
    sp_bt  = beh_bt[spike_frame]

    # 1-D occupancy (s) and spike-count maps
    occ_map   = np.zeros(n_bins_theta, dtype=np.float64)
    spike_map = np.zeros(n_bins_theta, dtype=np.float64)
    np.add.at(occ_map,   beh_bt, dt_frames)
    np.add.at(spike_map, sp_bt,  1.0)

    # valid bins (same criteria as the 2-D map), raw and smoothed angular rate maps
    valid_mask = _valid_bin_mask(occ_map, _bin_visit_counts(beh_bt, n_bins_theta))
    fr_raw = np.zeros_like(occ_map)
    fr_raw[valid_mask] = spike_map[valid_mask] / occ_map[valid_mask]
    fr_smooth = _gaussian_smooth_circular(fr_raw, valid_mask, target_bin_cm)

    # Mean rate from the RAW angular map (matches the 2-D mean_fr); field
    # detection itself runs on fr_smooth.
    mean_fr = 0.0
    if valid_mask.any():
        # Σ p_i r_i over valid angular bins
        total_occ = occ_map[valid_mask].sum()
        pi_flat   = occ_map[valid_mask] / total_occ
        mean_fr   = float(np.sum(pi_flat * fr_raw[valid_mask]))

    # bundle everything for field detection and plotting
    return dict(beh_bt=beh_bt, occ_map=occ_map, valid_mask=valid_mask,
                fr_raw=fr_raw, fr_smooth=fr_smooth, mean_fr=mean_fr,
                n_bins_theta=n_bins_theta, bin_width_rad=bin_width_rad,
                cx=cx, cy=cy, r_min=r_min, r_max=r_max, track_width_cm=track_width_cm)


# _circular_runs(qualifies, n_bins_theta) -> list of runs (lists of consecutive bin indices),
#   treating the last bin and the first bin as adjacent.
def _circular_runs(qualifies: np.ndarray, n_bins_theta: int) -> list:
    """Contiguous-run labelling of the angular bins where `qualifies` is
    True, wrapping around the theta=0/2*pi seam (bin 0 borders bin
    n_bins_theta-1) since the track is a closed loop -- the circular
    analogue of `_connected_components_8`. A field is never artificially
    split just because it straddles that arbitrary seam."""
    # no qualifying bins -> no runs
    if not qualifies.any():
        return []
    # every bin qualifies -> one run covering the whole ring
    if qualifies.all():
        return [list(range(n_bins_theta))]

    # sorted indices of qualifying bins
    idx = np.where(qualifies)[0]
    runs = []
    # start the first run with the first qualifying bin
    current = [int(idx[0])]
    # walk through the remaining qualifying bins
    for b in idx[1:]:
        b = int(b)
        # current[-1] = last bin of the current run; consecutive -> extend the run
        if b == current[-1] + 1:
            current.append(b)
        # gap -> close the current run and start a new one
        else:
            runs.append(current)
            current = [b]
    # close the final run
    runs.append(current)

    # if the first run starts at bin 0 and the last run ends at the last bin, they are one field
    #   crossing the 0/2π seam -> merge them (last run + first run, in ring order)
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == n_bins_theta - 1:
        merged = runs[-1] + runs[0]
        # runs[1:-1] = all runs except the first and last
        runs = runs[1:-1] + [merged]

    return runs


# detect_place_fields_threshold_circular(ctx_circular, target_bin_cm) -> (fields, peak_field_verified)
#   Angular counterpart of detect_place_fields_threshold_2d.
def detect_place_fields_threshold_circular(ctx_circular: dict, target_bin_cm: float) -> tuple[list[dict], bool]:
    """"Threshold method" place-field isolation on the angular ratemap: a bin
    only qualifies for a field if its Gaussian-smoothed rate is >=
    METHOD2_RATE_THRESHOLD_FRAC of the cell's peak (smoothed) rate AND above
    the cell's mean firing rate (both computed from this same angular map,
    see _build_ratemap_circular); contiguous runs of qualifying bins spanning
    >= MIN_FIELD_SIZE_BINS bins (no discontinuity, wrapping around the theta
    seam) are reported as fields (peak bin + rate-weighted circular centre
    of mass, the angular analogue of detect_place_fields_threshold_2d).

    Returns (fields, peak_field_verified) -- see
    detect_place_fields_threshold_2d's docstring for what
    `peak_field_verified` means and why.
    """
    # unpack the angular map
    valid_mask     = ctx_circular['valid_mask']
    fr_smooth      = ctx_circular['fr_smooth']
    fr_raw         = ctx_circular['fr_raw']
    n_bins_theta   = ctx_circular['n_bins_theta']
    bin_width_rad  = ctx_circular['bin_width_rad']
    track_width_cm = ctx_circular['track_width_cm']

    total_valid_bins = int(valid_mask.sum())
    if total_valid_bins == 0:
        return [], False

    # same thresholds as the 2-D method, from the angular map
    peak_fr = float(fr_smooth[valid_mask].max())
    mean_fr = float(ctx_circular.get('mean_fr', 0.0))
    rate_threshold = METHOD2_RATE_THRESHOLD_FRAC * peak_fr
    # NOTE (review): 9 bins here means 9 x 2 cm = 18 cm of ARC, whereas in 2-D it means 9 bins of
    #   area (36 cm²) -- so the minimum field size is not equivalent between arena types.
    min_size_bins  = MIN_FIELD_SIZE_BINS

    # qualifying angular bins
    qualifies = valid_mask & (fr_smooth >= rate_threshold) & (fr_smooth > mean_fr)
    # centre angle of every bin ((i + 0.5) x width)
    theta_centers_all = (np.arange(n_bins_theta) + 0.5) * bin_width_rad

    # Legacy "centre field" heuristic (nearest field to a reference bearing)
    # kept for output-schema continuity with the 2-D method; on a ring track
    # there is no single geometric centre bin, so this just flags the field
    # closest to the arbitrary theta=0 reference bearing.
    best_centre_dist = np.inf
    centre_field_idx = None
    fields  = []
    regions = []   # bin (theta) indices per field, parallel to `fields` --
                   # reused below to test whether the cell's raw-map peak
                   # bin lies inside a detected field.

    # each contiguous run of qualifying bins (seam-aware)
    for region in _circular_runs(qualifies, n_bins_theta):
        # too short -> skip
        if len(region) < min_size_bins:
            continue

        # the run's bin indices as an array, and their smoothed rates
        bts   = np.array(region)
        rates = fr_smooth[bts]

        # smoothed peak bin, value and angle
        peak_local_idx = int(np.argmax(rates))
        peak_bt        = int(bts[peak_local_idx])
        peak_val       = float(rates[peak_local_idx])
        peak_theta_rad = theta_centers_all[peak_bt]

        # Raw (unsmoothed) rate at that SAME bin -- lets you directly see how
        # much the Gaussian smoothing inflated/deflated the reported peak.
        peak_val_raw_same_bin = float(fr_raw[peak_bt])

        # Raw (unsmoothed) rate at this field's own highest-raw-rate bin,
        # which may sit at a different bin than peak_bt above -- reused
        # below in the peak_field_verified fallback check (see
        # detect_place_fields_threshold_2d's docstring).
        raw_rates          = fr_raw[bts]
        raw_peak_local_idx = int(np.argmax(raw_rates))
        field_raw_max_bt   = int(bts[raw_peak_local_idx])
        field_raw_max_val  = float(raw_rates[raw_peak_local_idx])

        # rate-weighted circular mean angle of the field (its angular centre of mass)
        theta_centers = theta_centers_all[bts]
        com_theta_rad = _circular_mean_rad(theta_centers, rates)

        # Angular span/bbox relative to the field's own circular mean, so a
        # field straddling the theta=0/2*pi seam still gets a small, correct
        # angular width instead of an apparent near-full-circle bbox.
        # signed angular offset of each bin from the centre, wrapped into [-π, π)
        offsets_rad = np.mod(theta_centers - com_theta_rad + np.pi, 2.0 * np.pi) - np.pi
        # .tolist() converts the array to a Python list; True if the field contains both bin 0
        #   and the last bin, i.e. it crosses the seam
        wraps_seam  = bool((0 in bts.tolist()) and (n_bins_theta - 1 in bts.tolist()))

        # size: bins, arc length (cm), area (arc x track width), % of valid bins, % of circumference
        n_bins_field  = len(region)
        arc_length_cm = n_bins_field * target_bin_cm
        area_cm2      = arc_length_cm * track_width_cm
        pct_area      = 100.0 * n_bins_field / total_valid_bins
        pct_circumference = 100.0 * n_bins_field / n_bins_theta
        # bins as text, e.g. '0;1;2;57;58'
        bin_coords_str = ';'.join(str(bt) for bt in sorted(region))

        # signed angle of the centre of mass from the reference bearing 0, wrapped into [-π, π)
        theta_diff_from_ref = ((com_theta_rad + np.pi) % (2.0 * np.pi)) - np.pi  # ref bearing = 0 rad
        # converted to bins
        dtheta_bins = abs(theta_diff_from_ref) / bin_width_rad
        # remember the field closest to bearing 0
        if dtheta_bins < best_centre_dist:
            best_centre_dist = dtheta_bins
            centre_field_idx = len(fields)

        # field descriptor dict (one PlaceFields row); np.degrees converts radians to degrees
        fields.append({
            'field_number':               len(fields) + 1,
            # smoothed peak bin, angle and rates
            'peak_bin_theta':             peak_bt,
            'peak_theta_deg':             round(np.degrees(peak_theta_rad) % 360.0, 2),
            'peak_fr_hz':                 round(peak_val, 4),
            'peak_fr_hz_raw':             round(peak_val_raw_same_bin, 4),
            # highest raw rate in the field
            'field_raw_max_bin_theta':    field_raw_max_bt,
            'field_raw_max_fr_hz':        round(field_raw_max_val, 4),
            # angular centre of mass (degrees)
            'com_theta_deg':              round(np.degrees(com_theta_rad) % 360.0, 2),
            # size descriptors
            'n_bins':                     n_bins_field,
            'arc_length_cm':              round(arc_length_cm, 2),
            'track_width_cm':             round(track_width_cm, 2),
            'area_cm2':                   round(area_cm2, 2),
            'pct_of_occupied_area':       round(pct_area, 2),
            'pct_of_track_circumference': round(pct_circumference, 2),
            # angular extent and its start/end angle
            'theta_span_deg':             round(float(offsets_rad.max() - offsets_rad.min()) * 180.0 / np.pi, 2),
            'bbox_theta_min_deg':         round(np.degrees(com_theta_rad + offsets_rad.min()) % 360.0, 2),
            'bbox_theta_max_deg':         round(np.degrees(com_theta_rad + offsets_rad.max()) % 360.0, 2),
            'wraps_theta_seam':           wraps_seam,
            # bins and detection parameters
            'bin_coords':                 bin_coords_str,
            'total_occupied_bins':        total_valid_bins,
            'min_field_size_bins':        min_size_bins,
            'rate_threshold_hz':          round(rate_threshold, 4),
            'mean_fr_threshold_hz':       round(mean_fr, 4),
            'is_centre_field':            False,
        })
        regions.append(region)

    # flag the field nearest bearing 0
    if centre_field_idx is not None:
        fields[centre_field_idx]['is_centre_field'] = True

    # ── Verify the place-cell peak bin is actually backed by a field ───────
    # See detect_place_fields_threshold_2d's matching block for the full
    # rationale -- identical logic, just over theta bins instead of (x, y).
    # NOTE (review): here the "global peak" is the peak of the ANGULAR raw map, which is not the
    #   same bin/value as metrics['peak_fr'] (2-D raw map) that gated the cell.
    masked_raw       = np.where(valid_mask, fr_raw, -np.inf)
    global_peak_bt   = int(np.argmax(masked_raw))

    # peak inside a field -> verified; otherwise fall back to any field's raw peak in 1-25 Hz
    if any(global_peak_bt in region for region in regions):
        peak_field_verified = True
    else:
        peak_field_verified = any(
            PLACE_CELL_MIN_PEAK_FR < f['field_raw_max_fr_hz'] < PLACE_CELL_MAX_PEAK_FR
            for f in fields
        )

    return fields, peak_field_verified


# colours cycled through for the field markers in the polar plot
_FIELD_COLORS_CIRC = ['tab:red', 'tab:green', 'tab:purple', 'tab:orange',
                       'tab:brown', 'tab:pink', 'tab:cyan', 'tab:olive']


# _plot_ratemap_polar(ax, ...) -> None. Angular rate map drawn as a polar curve.
def _plot_ratemap_polar(ax, fr_smooth: np.ndarray, valid_mask: np.ndarray,
                         fields: list[dict], n_bins_theta: int,
                         bin_width_rad: float, title: str):
    """Plots the 1-D angular rate map as a polar tuning curve, so the ring
    track renders as an actual ring: theta=0 and theta=2*pi coincide in
    physical space, so a field that straddles that seam still appears as one
    unbroken arc -- no wraparound artifact, unlike a flat bar/line plot."""
    # bin-centre angles
    theta_centers = (np.arange(n_bins_theta) + 0.5) * bin_width_rad
    # rates, with invalid bins as NaN (NaN leaves a gap in the line)
    rates = np.where(valid_mask, fr_smooth, np.nan)

    # repeat the first point at the end (+2π) so the curve closes on itself
    theta_plot = np.concatenate([theta_centers, theta_centers[:1] + 2.0 * np.pi])
    rates_plot = np.concatenate([rates, rates[:1]])

    # the tuning curve, and a translucent fill under it (NaN replaced by 0 for the fill)
    ax.plot(theta_plot, rates_plot, color='tab:blue', linewidth=1.5)
    ax.fill(theta_plot, np.nan_to_num(rates_plot), color='tab:blue', alpha=0.15)

    # enumerate gives the field number i (0, 1, ...) and the field dict
    for i, field in enumerate(fields):
        bin_coords = field.get('bin_coords', '')
        if not bin_coords:
            continue
        # '0;1;2' -> [0, 1, 2]
        bts = [int(b) for b in bin_coords.split(';')]
        # `%` cycles through the colour list if there are more than 8 fields
        color = _FIELD_COLORS_CIRC[i % len(_FIELD_COLORS_CIRC)]
        # mark the field's bins with coloured dots, drawn above the curve (zorder=5)
        ax.plot(theta_centers[bts], rates[bts], 'o', color=color, markersize=3,
                zorder=5, label=f"Field {field['field_number']}")

    ax.set_title(title)
    # angle 0 points East (right), angles increase counter-clockwise (direction 1)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    # legend only if there is at least one field (an empty list is falsy)
    if fields:
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=7)


# _plot_ratemap_ring(ax, ...) -> colour mesh. Angular rate map drawn as a coloured ring (annulus).
def _plot_ratemap_ring(ax, fr_smooth: np.ndarray, valid_mask: np.ndarray,
                        fields: list[dict], n_bins_theta: int, bin_width_rad: float,
                        r_min: float, r_max: float, title: str):
    """Plots the 1-D angular rate map as a binned heatmap in an actual ring
    shape (an annulus spanning the track's fitted radial extent, r_min to
    r_max), the angular analogue of the 2-D grid's imshow rate map. Each
    angular bin is one wedge of the ring, colour-coded by its smoothed rate;
    the polar r-axis is forced to start at 0 so the empty disc inside r_min
    renders as the ring's hole. Detected fields are outlined in black, the
    same way field boundaries are contoured on the 2-D grid."""
    # wedge boundaries in angle (n+1 edges) and in radius (inner, outer)
    theta_edges = np.arange(n_bins_theta + 1) * bin_width_rad
    r_edges = np.array([r_min, r_max])
    # 2-D grids of the cell corners for pcolormesh
    theta_mesh, r_mesh = np.meshgrid(theta_edges, r_edges)

    # one row of colour values (shape 1 x n_bins_theta), invalid bins masked
    rates = np.ma.masked_where(~valid_mask, fr_smooth).reshape(1, n_bins_theta)
    # pcolormesh fills each (theta, r) cell with its colour; shading='flat' = one colour per cell
    mesh = ax.pcolormesh(theta_mesh, r_mesh, rates, cmap='jet', shading='flat')

    # bin-centre angles plus a closing point at +2π
    theta_centers = (np.arange(n_bins_theta) + 0.5) * bin_width_rad
    theta_ext = np.concatenate([theta_centers, theta_centers[:1] + 2.0 * np.pi])
    # small radial padding so the contour line sits just inside/outside the ring
    track_width = max(r_max - r_min, 1e-6)
    pad = min(0.15 * track_width, r_min) if r_min > 0 else 0.15 * track_width
    # four radii: below the ring, inner edge, outer edge, above the ring
    r_centers = np.array([r_min - pad, r_min, r_max, r_max + pad])

    for field in fields:
        bin_coords = field.get('bin_coords', '')
        if not bin_coords:
            continue
        bts = [int(b) for b in bin_coords.split(';')]
        # 1 in the field's bins, 0 elsewhere
        field_row = np.zeros(n_bins_theta)
        field_row[bts] = 1.0
        # 4 x n mask: 0 outside the ring radially, field_row on the ring's two edges
        mask_2d = np.vstack([np.zeros(n_bins_theta), field_row, field_row, np.zeros(n_bins_theta)])
        # append the first column at the end so the contour closes across the seam
        mask_ext = np.concatenate([mask_2d, mask_2d[:, :1]], axis=1)
        # coordinate grids matching mask_ext
        theta_c, r_c = np.meshgrid(theta_ext, r_centers)
        # outline where the mask crosses 0.5
        ax.contour(theta_c, r_c, mask_ext, levels=[0.5], colors='black', linewidths=1.5)

    ax.set_title(title)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    # radial axis from 0 (shows the ring's hole) to just beyond the outer edge
    ax.set_ylim(0, r_max * 1.1)
    # hide radius tick labels
    ax.set_yticklabels([])
    # returned for the colour bar
    return mesh




# ── Per-job wrapper (called from thread pool) ─────────────────────────────────

# a Lock lets only one thread at a time run the code inside `with _print_lock:`, so console
#   lines printed by parallel threads do not interleave
_print_lock = threading.Lock()

# "no bootstrap result" placeholder (all None), used when the shuffle could not run
_NULL_BOOTSTRAP = {'bootstrap_mean': None, 'bootstrap_p95': None, 'bootstrap_sig': None,
                   'coherence_bootstrap_mean': None, 'coherence_bootstrap_p95': None,
                   'coherence_bootstrap_sig': None}

# _run_job(args) -> (full_row, first_row, second_row, field_rows): the complete analysis of ONE
#   unit. Executed in a worker thread; `args` is a single tuple so it can be passed via submit().
def _run_job(args):
    # unpack the job tuple: running number, total count, original order, folder, tracking file, .ntt name
    unit_idx, total_units, job_order, dirpath, csv_path, ntt_file = args
    # session label = folder path relative to root_folder
    session_name = os.path.relpath(dirpath, root_folder)
    # full path of the .ntt file
    ntt_path     = os.path.join(dirpath, ntt_file)
    # progress percentage
    pct          = 100 * unit_idx / total_units

    # print a progress line (one thread at a time)
    with _print_lock:
        # :.1f = one decimal place
        print(f'[{unit_idx}/{total_units}  {pct:.1f}%]  {session_name}  |  {ntt_file}  '
              f'(GPU {_gpu_util_pct()}%)')

    # template row returned when compute_metrics fails: every column present, values None.
    #   `_err_row: dict = {...}` is an annotated assignment (type hint `dict`).
    _err_row: dict = {
        # spike counts and basic rates
        'n_spikes': None, 'n_discarded': None, 'n_spikes_immobile': None,
        'peak_fr':  None, 'mean_fr':     None, 'sir': None,
        'sparsity': None, 'coherence':   None,
        # stability
        'stability_score': None, 'stability_p_value': None, 'stability_n_bins': None,
        # SI and coherence shuffle results
        'bootstrap_mean': None, 'bootstrap_p95': None, 'bootstrap_sig': None,
        'coherence_bootstrap_mean': None, 'coherence_bootstrap_p95': None,
        'coherence_bootstrap_sig': None,
        # theta
        'theta_modulated': None, 'theta_peak_freq': None,
        # binned speed results
        'speed_score': None, 'speed_p_value': None, 'speed_r2': None,
        'speed_beta': None, 'speed_f0': None, 'speed_modulated': None,
        'speed_shuffle_mean': None, 'speed_shuffle_lo': None,
        'speed_shuffle_hi': None, 'speed_shuffle_p': None,
        'speed_modulated_shuffle': None, 'speed_shuffle_ran': None,
        # time-domain speed results
        'speed_score_td': None, 'speed_p_value_td': None, 'speed_r2_td': None,
        'speed_shuffle_mean_td': None, 'speed_shuffle_lo_td': None,
        'speed_shuffle_hi_td': None, 'speed_shuffle_p_td': None,
        'speed_modulated_td': None, 'speed_shuffle_ran_td': None,
        'p_speed': None, 'n_speed': None,
        # final speed classification
        'final_speed_score': float('nan'), 'speed_cell': 'not tested',
        # identifiers and verdicts
        'session': session_name, 'unit': ntt_file,
        'job_order': job_order, 'place_cell': None, 'n_fields_detected': None,
    }

    # nested function (defined inside _run_job, so it can use ntt_path, csv_path, _err_row, ...):
    #   computes one result row for half = None (full session), 'first' or 'second'.
    def _build_row(half: str | None, label: str) -> tuple[dict, dict]:
        try:
            # rate maps and basic metrics
            metrics, ctx = compute_metrics(csv_path, ntt_path, arena_width_cm, target_bin_cm, half=half)
        # any error: report it and return a copy (dict(...)) of the error template with an empty ctx
        except Exception as e:
            with _print_lock:
                print(f'  ERROR in {ntt_file} [{label}]: {e}')
            return dict(_err_row), {}

        # empty ctx (no tracking / no movement) -> no shuffle test
        if not ctx:
            bootst = _NULL_BOOTSTRAP
        else:
            try:
                # SI + coherence circular-shift shuffle test (1000 shuffles) and its figure
                bootst = _run_bootstrap(
                    # spike frames, frame times, frame bins
                    ctx['spike_frame'], ctx['t'], ctx['beh_bx'], ctx['beh_by'],
                    # occupancy, valid mask, grid size
                    ctx['occ_map'], ctx['valid_mask'], ctx['n_bins_x'], ctx['n_bins_y'],
                    # bin size
                    target_bin_cm,
                    # real SI and coherence (defaults if missing)
                    metrics.get('sir', 0.0), metrics.get('coherence', float('nan')),  # type: ignore
                    # where to save the figure, and its label
                    ntt_path, label=label,
                )
            except Exception as e:
                with _print_lock:
                    print(f'  BOOTSTRAP ERROR in {ntt_file} [{label}]: {e}')
                bootst = _NULL_BOOTSTRAP

        # copy the shuffle results into the metrics dict
        metrics['bootstrap_mean'] = bootst.get('bootstrap_mean')
        metrics['bootstrap_p95']  = bootst.get('bootstrap_p95')
        metrics['bootstrap_sig']  = bootst.get('bootstrap_sig')
        metrics['coherence_bootstrap_mean'] = bootst.get('coherence_bootstrap_mean')
        metrics['coherence_bootstrap_p95']  = bootst.get('coherence_bootstrap_p95')
        metrics['coherence_bootstrap_sig']  = bootst.get('coherence_bootstrap_sig')

        # Theta modulation
        # moving-only spike times (None if ctx is empty)
        spike_ts = ctx.get('spike_ts') if ctx else None
        if spike_ts is not None and len(spike_ts) >= 10:
            try:
                # autocorrelogram FFT test
                theta_mod, theta_freq = _compute_theta_modulation(spike_ts)
                metrics['theta_modulated'] = theta_mod
                metrics['theta_peak_freq'] = theta_freq
            except Exception as e:
                with _print_lock:
                    print(f'  THETA ERROR in {ntt_file} [{label}]: {e}')
                metrics['theta_modulated'] = None
                metrics['theta_peak_freq'] = None
        else:
            # too few spikes -> not tested
            metrics['theta_modulated'] = None
            metrics['theta_peak_freq'] = None

        # Speed modulation — full session only (not run on first/second half splits).
        # Uses the contiguous, pre-immobility-filter track and spikes (ctx['*_all']):
        # dropping immobile frames would leave gaps that corrupt frame-to-frame
        # speed, and _compute_speed_modulation already excludes speeds < SPEED_MIN_CMS.
        if half is None and ctx and ctx.get('spike_ts_all') is not None and len(ctx['spike_ts_all']) > 0:
            try:
                # Downsample tracking to 15 fps (keep every alternate frame) before
                # estimating speed and binning spikes for the speed-modulation fit —
                # see SPEED_MOD_DOWNSAMPLE_FACTOR.
                # arr[::k] = every k-th element (slice with step k). With k = 1 this is the whole
                #   array, i.e. NO downsampling (the comment above is stale, see the constant).
                x_cm_ds = ctx['x_cm_all'][::SPEED_MOD_DOWNSAMPLE_FACTOR]
                y_cm_ds = ctx['y_cm_all'][::SPEED_MOD_DOWNSAMPLE_FACTOR]
                t_ds    = ctx['t_all'][::SPEED_MOD_DOWNSAMPLE_FACTOR]
                # effective frame rate after downsampling
                pos_fps_ds = fps / SPEED_MOD_DOWNSAMPLE_FACTOR
                # run both speed analyses (and save the figure)
                speed_res = _compute_speed_modulation(
                    x_cm_ds, y_cm_ds, t_ds, ctx['spike_ts_all'], pos_fps_ds,
                    ntt_path=ntt_path, label=label,
                )
                # copy every speed result into metrics (binned fit ...)
                metrics['speed_score']     = speed_res['speed_score']
                metrics['speed_p_value']   = speed_res['speed_p_value']
                metrics['speed_r2']        = speed_res['speed_r2']
                metrics['speed_beta']      = speed_res['speed_beta']
                metrics['speed_f0']        = speed_res['speed_f0']
                metrics['speed_modulated'] = speed_res['speed_modulated']
                # ... binned shuffle ...
                metrics['speed_shuffle_mean']      = speed_res['speed_shuffle_mean']
                metrics['speed_shuffle_lo']        = speed_res['speed_shuffle_lo']
                metrics['speed_shuffle_hi']        = speed_res['speed_shuffle_hi']
                metrics['speed_shuffle_p']         = speed_res['speed_shuffle_p']
                metrics['speed_modulated_shuffle'] = speed_res['speed_modulated_shuffle']
                metrics['speed_shuffle_ran']       = speed_res['speed_shuffle_ran']
                # ... time-domain score and its shuffle ...
                metrics['speed_score_td']          = speed_res['speed_score_td']
                metrics['speed_p_value_td']        = speed_res['speed_p_value_td']
                metrics['speed_r2_td']             = speed_res['speed_r2_td']
                metrics['speed_shuffle_mean_td']   = speed_res['speed_shuffle_mean_td']
                metrics['speed_shuffle_lo_td']     = speed_res['speed_shuffle_lo_td']
                metrics['speed_shuffle_hi_td']     = speed_res['speed_shuffle_hi_td']
                metrics['speed_shuffle_p_td']      = speed_res['speed_shuffle_p_td']
                metrics['speed_modulated_td']      = speed_res['speed_modulated_td']
                metrics['speed_shuffle_ran_td']    = speed_res['speed_shuffle_ran_td']
                # ... and the positive / negative speed-cell flags
                metrics['p_speed']                 = speed_res['p_speed']
                metrics['n_speed']                 = speed_res['n_speed']
            # speed analysis crashed: report and set every speed field to None
            except Exception as e:
                with _print_lock:
                    print(f'  SPEED ERROR in {ntt_file} [{label}]: {e}')
                # binned fit -> None
                metrics['speed_score']     = None
                metrics['speed_p_value']   = None
                metrics['speed_r2']        = None
                metrics['speed_beta']      = None
                metrics['speed_f0']        = None
                metrics['speed_modulated'] = None
                # binned shuffle -> None
                metrics['speed_shuffle_mean']      = None
                metrics['speed_shuffle_lo']        = None
                metrics['speed_shuffle_hi']        = None
                metrics['speed_shuffle_p']         = None
                metrics['speed_modulated_shuffle'] = None
                metrics['speed_shuffle_ran']       = None
                # time-domain -> None
                metrics['speed_score_td']          = None
                metrics['speed_p_value_td']        = None
                metrics['speed_r2_td']             = None
                metrics['speed_shuffle_mean_td']   = None
                metrics['speed_shuffle_lo_td']     = None
                metrics['speed_shuffle_hi_td']     = None
                metrics['speed_shuffle_p_td']      = None
                metrics['speed_modulated_td']      = None
                metrics['speed_shuffle_ran_td']    = None
                # speed-cell flags -> None
                metrics['p_speed']                 = None
                metrics['n_speed']                 = None
        # half-session rows (or no spikes): speed is not computed -> every speed field None
        else:
            # binned fit -> None
            metrics['speed_score']     = None
            metrics['speed_p_value']   = None
            metrics['speed_r2']        = None
            metrics['speed_beta']      = None
            metrics['speed_f0']        = None
            metrics['speed_modulated'] = None
            # binned shuffle -> None
            metrics['speed_shuffle_mean']      = None
            metrics['speed_shuffle_lo']        = None
            metrics['speed_shuffle_hi']        = None
            metrics['speed_shuffle_p']         = None
            metrics['speed_modulated_shuffle'] = None
            metrics['speed_shuffle_ran']       = None
            # time-domain -> None
            metrics['speed_score_td']          = None
            metrics['speed_p_value_td']        = None
            metrics['speed_r2_td']             = None
            metrics['speed_shuffle_mean_td']   = None
            metrics['speed_shuffle_lo_td']     = None
            metrics['speed_shuffle_hi_td']     = None
            metrics['speed_shuffle_p_td']      = None
            metrics['speed_modulated_td']      = None
            metrics['speed_shuffle_ran_td']    = None
            # speed-cell flags -> None
            metrics['p_speed']                 = None
            metrics['n_speed']                 = None

        # ── Final speed score / speed-cell classification ──────────────────────
        # final_speed_score: mean of the binned (speed_score) and time-domain
        # (speed_score_td) scores; NaN if either is missing.
        # speed_cell, evaluated in this order:
        #   False        - binned and time-domain scores have opposite signs, or
        #                  either shuffle test ran and failed
        #   True         - both shuffle tests ran and passed
        #   'not tested' - otherwise (a shuffle never ran, or speed wasn't computed)
        # convert None to NaN so both scores are floats
        _s_bin = float('nan') if metrics['speed_score']    is None else float(metrics['speed_score'])
        _s_td  = float('nan') if metrics['speed_score_td'] is None else float(metrics['speed_score_td'])
        # True only if both scores are real numbers
        _both_scores = bool(np.isfinite(_s_bin) and np.isfinite(_s_td))
        if _both_scores:
            # average of the two speed scores
            metrics['final_speed_score'] = round((_s_bin + _s_td) / 2.0, 4)
        else:
            metrics['final_speed_score'] = float('nan')
        # shuffle-test outcomes (True / False / None = not run)
        _mod_bin = metrics['speed_modulated_shuffle']
        _mod_td  = metrics['speed_modulated_td']
        # opposite signs (product of signs < 0) -> the two methods disagree -> not a speed cell
        if _both_scores and np.sign(_s_bin) * np.sign(_s_td) < 0:
            metrics['speed_cell'] = False
        # `is False` checks for the actual value False (None does not match)
        elif _mod_bin is False or _mod_td is False:
            metrics['speed_cell'] = False
        # both shuffle tests passed
        elif _mod_bin is True and _mod_td is True:
            metrics['speed_cell'] = True
        # anything else (a shuffle never ran)
        else:
            metrics['speed_cell'] = 'not tested'

        # identifiers
        metrics['session']   = session_name
        metrics['unit']      = ntt_file
        metrics['job_order'] = job_order

        # values needed for the place-cell rule
        n_spikes      = metrics.get('n_spikes')
        sir           = metrics.get('sir')
        peak_fr       = metrics.get('peak_fr')
        sparsity      = metrics.get('sparsity')
        boot_sig      = metrics.get('bootstrap_sig')
        coh_boot_sig  = metrics.get('coherence_bootstrap_sig')

        # The rule is evaluated only if every input exists.
        #   NOTE (review): sparsity and coh_boot_sig must be non-None here even though those two
        #   criteria are switched off below. coherence_bootstrap_sig is None whenever the real
        #   coherence is NaN or no coherence shuffle was valid, and in that case the cell gets
        #   place_cell = None (excluded) even if it passes every ACTIVE criterion.
        if ((n_spikes is not None) and (sir is not None) and (peak_fr is not None)
                and (sparsity is not None) and (boot_sig is not None) and (coh_boot_sig is not None)):
            # place cell = all active criteria true (`and` chains them; the result is a bool)
            metrics['place_cell'] = (
                # more than 50 spikes
                int(n_spikes)    >  PLACE_CELL_MIN_SPIKES    and
                # peak rate above 1 Hz ...
                float(peak_fr)   >  PLACE_CELL_MIN_PEAK_FR  and
                # ... and below 25 Hz
                float(peak_fr)   <  PLACE_CELL_MAX_PEAK_FR  and
                # SI above 0.5 bits/spike (fixed threshold)
                float(sir)       >  PLACE_CELL_MIN_SIR      and
                #float(sparsity)  <  PLACE_CELL_MAX_SPARSITY  and
                # SI above the 95th percentile of its shuffle null (permutation test)
                boot_sig is True                             #and
                #coh_boot_sig is True
            )
        else:
            # cannot be evaluated
            metrics['place_cell'] = None

        # return this row and its ctx
        return metrics, ctx

    # build the three rows (each runs compute_metrics + its own 1000-shuffle test + theta);
    #   only the full-session ctx is kept (the half-session ctx is discarded with `_`)
    full_row,  full_ctx = _build_row(None,     'full')
    first_row, _        = _build_row('first',  'first_half')
    second_row, _       = _build_row('second', 'second_half')

    # Split-half spatial stability (Pearson r of the smoothed ratemap between the
    # first and second half of the SESSION) -- computed once here on the
    # full-session grid, not from the independently-binned first/second-half
    # rows above (see _compute_split_half_stability docstring). Reported only
    # on the full-session row; not a per-half quantity.
    try:
        stability = _compute_split_half_stability(full_ctx, target_bin_cm)
    except Exception as e:
        with _print_lock:
            print(f'  STABILITY ERROR in {ntt_file}: {e}')
        stability = {'stability_score': None, 'stability_p_value': None, 'stability_n_bins': None}

    # store stability in the full-session row
    full_row['stability_score']   = stability.get('stability_score')
    full_row['stability_p_value'] = stability.get('stability_p_value')
    full_row['stability_n_bins']  = stability.get('stability_n_bins')

    # half-session rows: stability not applicable
    first_row['stability_score']   = None
    first_row['stability_p_value'] = None
    first_row['stability_n_bins']  = None
    second_row['stability_score']   = None
    second_row['stability_p_value'] = None
    second_row['stability_n_bins']  = None

    # ── Place-field isolation — confirmed place cells only, full session ──────
    # Circle-track sessions (folder path contains 'Circle') use the 1-D
    # angular method instead of the 2-D grid: the 2-D 8-connected-component
    # search doesn't respect the ring's topology and can slice one true field
    # into several spurious pieces purely from binning artifacts (see the
    # "Place-field isolation – "threshold method" (Circle / angular ratemap)"
    # section above). That method needs its own ratemap (built fresh here
    # from full_ctx's already-loaded tracking/spike data), since the 2-D
    # grid ratemap in full_ctx isn't usable for angular binning.
    # field rows for the PlaceFields sheet, the detected fields, angular-map data, arena type
    field_rows: list[dict] = []
    fields_threshold: list[dict] | None = None
    ctx_circular: dict | None = None
    arena_type: str | None = None
    # only for cells currently classified as place cells, with a usable rate map
    if (full_row.get('place_cell') is True and full_ctx
            and full_ctx.get('valid_mask') is not None and full_ctx['valid_mask'].any()):
        try:
            # ring track: angular map + angular field detection
            if _is_circle_arena(dirpath):
                ctx_circular = _build_ratemap_circular(full_ctx, target_bin_cm)
                fields_threshold, peak_field_verified = detect_place_fields_threshold_circular(
                    ctx_circular, target_bin_cm)
                arena_type = 'Circle'
            # open field / linear track: 2-D field detection
            else:
                fields_threshold, peak_field_verified = detect_place_fields_threshold_2d(
                    full_row, full_ctx, target_bin_cm)
                arena_type = '2D'

            # The cell's peak-firing bin (metrics['peak_fr'], the RAW-map
            # global max that gated it into place-cell status) must belong
            # to one of the fields just detected, or -- failing that -- some
            # OTHER detected field's own raw-map peak must still land in the
            # place-cell peak-rate window (see detect_place_fields_threshold_2d
            # / _circular). If neither holds, nothing actually backs the
            # place-cell verdict: overrule it on every sheet (not just
            # 'Full'), and drop this cell from the PlaceFields output like
            # any other non-place-cell.
            if not peak_field_verified:
                # overrule the verdict in all three rows
                full_row['place_cell']        = False
                first_row['place_cell']       = False
                second_row['place_cell']      = False
                full_row['n_fields_detected'] = None
            else:
                # number of fields found
                full_row['n_fields_detected'] = len(fields_threshold)

                # A field covering more than PLACE_FIELD_MAX_AREA_PCT of the
                # occupied arena is too diffuse to be a real place field --
                # overrule the earlier place_cell = True verdict for this cell.
                # any() over a generator: True if at least one field covers > 50 % of the arena.
                #   NOTE (review): unlike the override above, this one only changes full_row --
                #   first_row/second_row keep their own verdict.
                #   NOTE (review): the field rows are still added to PlaceFields below even when this
                #   override fires, so that sheet can contain fields of cells marked place_cell = False.
                if any(f.get('pct_of_occupied_area', 0.0) > PLACE_FIELD_MAX_AREA_PCT
                       for f in fields_threshold):
                    full_row['place_cell'] = False

                # one PlaceFields row per field: identifiers + arena type + all field descriptors
                #   (**f unpacks the field dict into the new dict)
                for f in fields_threshold:
                    field_rows.append({'session': session_name, 'unit': ntt_file,
                                        'arena_type': arena_type, **f})
        except Exception as e:
            with _print_lock:
                print(f'  FIELD DETECTION ERROR in {ntt_file}: {e}')
            full_row['n_fields_detected'] = None
    # not a place cell: no field detection
    else:
        full_row['n_fields_detected'] = None

    # field counts are reported only on the full-session row
    first_row['n_fields_detected']  = None
    second_row['n_fields_detected'] = None

    # Ratemap plot — full session only, every cell with valid occupancy.
    # Confirmed place cells (final verdict, i.e. after the field-area
    # override above) get their detected place-field panel(s) merged into
    # this same figure; every other cell gets just the trajectory + rate-map
    # panels (see _plot_and_save_ratemap).
    if full_ctx and full_ctx.get('valid_mask') is not None and full_ctx['valid_mask'].any():
        try:
            # final verdict
            is_place_cell = full_row.get('place_cell') is True
            _plot_and_save_ratemap(
                # smoothed map, mask, grid size, bin size
                full_ctx['fr_smooth'], full_ctx['valid_mask'],
                full_ctx['n_bins_x'], full_ctx['n_bins_y'], target_bin_cm,
                # .ntt path and metrics (title)
                ntt_path, full_row,
                # full trajectory
                full_ctx['x_cm_all'], full_ctx['y_cm_all'],
                # moving-only trajectory
                full_ctx['x_cm'], full_ctx['y_cm'],
                # spike frames
                full_ctx['spike_frame'],
                # field panels only for confirmed place cells (None otherwise)
                fields_threshold=(fields_threshold if is_place_cell else None),
                arena_type=(arena_type if is_place_cell else None),
                ctx_circular=(ctx_circular if is_place_cell else None))
        except Exception as e:
            with _print_lock:
                print(f'  RATEMAP PLOT ERROR for {ntt_file}: {e}')

    # Combined full/first-half/second-half ratemap + stability-score figure —
    # confirmed place cells only (final verdict, i.e. after the field-area
    # override above), matching the gating used for field-detection plots.
    if (full_row.get('place_cell') is True and full_ctx
            and full_ctx.get('valid_mask') is not None and full_ctx['valid_mask'].any()):
        try:
            _plot_and_save_split_half_ratemap(full_ctx, stability, ntt_path, target_bin_cm)
        except Exception as e:
            with _print_lock:
                print(f'  SPLIT-HALF RATEMAP PLOT ERROR for {ntt_file}: {e}')

    # the job's result: three table rows plus the list of field rows
    return (full_row, first_row, second_row, field_rows)


# ── Batch scan ────────────────────────────────────────────────────────────────

# `if __name__ == "__main__":` -- this block runs only when the file is executed as a script
#   (python PlaceCell_Main_v15.py), not when it is imported as a module by another script.
if __name__ == "__main__":
    # report which coordinate convention is active
    print(f"Using '{COORD_UNITS}' tracking coordinates.\n")

    # write the settings CSV first, so every run is traceable
    _save_run_metadata(output_excel)

    # list of (folder, tracking file, .ntt file) jobs, and folder -> tracking-file lookup
    all_jobs   = []
    dir_to_csv = {}
    # the output workbook's own name, so it is never mistaken for a tracking file
    output_excel_basename = os.path.basename(output_excel).lower()
    # os.walk yields (folder path, sub-folder names, file names) for every folder under root_folder;
    #   `_` ignores the sub-folder list
    for dirpath, _, filenames in os.walk(root_folder):
        # candidate tracking files: .csv or .xlsx, excluding the output workbook
        #   (str.endswith accepts a tuple of suffixes)
        tracking_files_all = [f for f in filenames
                              if f.lower().endswith(('.csv', '.xlsx'))
                              and f.lower() != output_excel_basename]
        # '_cm.csv' files are the cm-converted tracking files; everything else
        # (.xlsx, or a plain .csv without that suffix) is pixel-based tracking.
        if COORD_UNITS == 'cm':
            tracking_files = [f for f in tracking_files_all if f.lower().endswith('_cm.csv')]
        else:
            tracking_files = [f for f in tracking_files_all if not f.lower().endswith('_cm.csv')]
        # spike files in this folder
        ntt_files      = [f for f in filenames if f.lower().endswith('.ntt')]
        # a session folder must contain exactly ONE tracking file and at least one .ntt
        #   (folders with 0 or 2+ tracking files are silently skipped)
        if len(tracking_files) == 1 and len(ntt_files) > 0:
            csv_path = os.path.join(dirpath, tracking_files[0])
            dir_to_csv[dirpath] = csv_path
            # one job per .ntt, in alphabetical order
            for ntt_file in sorted(ntt_files):
                all_jobs.append((dirpath, csv_path, ntt_file))

    # number of units found
    total_units = len(all_jobs)
    print(f'Found {total_units} unit(s) across all sessions.\n')

    # ── Speed-vs-time plots (one per tracking file) ─────────────────────────────
    print(f'Generating speed and dt_s plots for {len(dir_to_csv)} tracking file(s)...')
    # .values() iterates over the dict's values (the tracking-file paths)
    for csv_path in dir_to_csv.values():
        # each plot in its own try so one failure does not stop the other
        try:
            _plot_and_save_speed(csv_path, arena_width_cm)
        except Exception as e:
            print(f'  SPEED PLOT ERROR for {csv_path}: {e}')
        try:
            _plot_and_save_dt_s(csv_path, arena_width_cm, fps)
        except Exception as e:
            print(f'  DT_S PLOT ERROR for {csv_path}: {e}')
    # blank line
    print()

    # job argument tuples: (1-based running number, total, 0-based order, folder, tracking, .ntt);
    #   enumerate(..., start=1) numbers the jobs from 1
    job_args = [
        (idx, total_units, idx - 1, dirpath, csv_path, ntt_file)
        for idx, (dirpath, csv_path, ntt_file) in enumerate(all_jobs, start=1)
    ]

    # ── Parallel execution ────────────────────────────────────────────────────────

    # collected job results
    results = []
    # thread pool with MAX_WORKERS (4) threads; `with` shuts the pool down cleanly afterwards
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # submit every job; dict comprehension maps each Future (pending result) to its args
        futures = {executor.submit(_run_job, args): args for args in job_args}
        # as_completed yields futures in the order they FINISH (not the order submitted)
        for future in concurrent.futures.as_completed(futures):
            # .result() returns _run_job's return value (or re-raises its exception)
            results.append(future.result())

    # restore the original job order; key=lambda r: ... sorts by the full row's job_order
    results.sort(key=lambda r: r[0]['job_order'])

    # ── Save to Excel ─────────────────────────────────────────────────────────────

    # column order of the Full / First_Half / Second_Half sheets
    column_order = ['session', 'unit', 'n_spikes', 'n_discarded', 'n_spikes_immobile',
                    # rate-map metrics
                    'peak_fr', 'mean_fr', 'sir', 'sparsity', 'coherence',
                    # stability
                    'stability_score', 'stability_p_value', 'stability_n_bins',
                    # SI and coherence shuffle tests
                    'bootstrap_mean', 'bootstrap_p95', 'bootstrap_sig',
                    'coherence_bootstrap_mean', 'coherence_bootstrap_p95', 'coherence_bootstrap_sig',
                    # theta
                    'theta_modulated', 'theta_peak_freq',
                    # binned speed analysis
                    'speed_score', 'speed_p_value', 'speed_r2', 'speed_beta', 'speed_f0', 'speed_modulated',
                    'speed_shuffle_mean', 'speed_shuffle_lo', 'speed_shuffle_hi',
                    'speed_shuffle_p', 'speed_modulated_shuffle', 'speed_shuffle_ran',
                    # time-domain speed analysis
                    'speed_score_td', 'speed_p_value_td', 'speed_r2_td',
                    'speed_shuffle_mean_td', 'speed_shuffle_lo_td', 'speed_shuffle_hi_td',
                    'speed_shuffle_p_td', 'speed_modulated_td', 'speed_shuffle_ran_td',
                    'p_speed', 'n_speed',
                    # final classifications
                    'final_speed_score', 'speed_cell',
                    'place_cell', 'n_fields_detected']

    # Shared columns first, then the 2-D grid (Open/Linear) field columns,
    # then the angular (Circle) field columns -- a given row only populates
    # whichever set matches its arena_type (see _run_job), the other set is
    # left blank/NaN.
    field_columns = ['session', 'unit', 'arena_type', 'field_number',
                     # peak_fr_hz_raw = raw (unsmoothed) rate at the SAME bin
                     # as peak_fr_hz -- shows how much the Gaussian smoothing
                     # inflated/deflated the reported peak at that location.
                     'peak_fr_hz', 'peak_fr_hz_raw',
                     # size and detection parameters
                     'n_bins', 'area_cm2', 'pct_of_occupied_area',
                     'total_occupied_bins', 'min_field_size_bins',
                     'rate_threshold_hz', 'mean_fr_threshold_hz', 'is_centre_field',
                     # field_raw_max_fr_hz = the highest RAW rate anywhere in
                     # this field (its own bin, field_raw_max_bin_x/y or
                     # _theta, may differ from the smoothed peak bin above).
                     'field_raw_max_fr_hz',
                     # 2-D grid (Open field / Linear track)
                     'peak_bin_x', 'peak_bin_y', 'field_raw_max_bin_x', 'field_raw_max_bin_y',
                     'com_bin_x', 'com_bin_y', 'com_cm_x', 'com_cm_y',
                     'bbox_x_min', 'bbox_x_max', 'bbox_y_min', 'bbox_y_max',
                     'bbox_cm_x_min', 'bbox_cm_x_max', 'bbox_cm_y_min', 'bbox_cm_y_max',
                     # Angular (Circle track)
                     'peak_bin_theta', 'peak_theta_deg', 'field_raw_max_bin_theta', 'com_theta_deg',
                     'arc_length_cm', 'track_width_cm', 'pct_of_track_circumference',
                     'theta_span_deg', 'bbox_theta_min_deg', 'bbox_theta_max_deg', 'wraps_theta_seam',
                     # bin list
                     'bin_coords']

    # one DataFrame per sheet: r[0] = full row, r[1] = first half, r[2] = second half;
    #   columns= selects/orders the columns (keys not listed, e.g. job_order, are dropped)
    df_full   = pd.DataFrame([r[0] for r in results], columns=column_order)
    df_first  = pd.DataFrame([r[1] for r in results], columns=column_order)
    df_second = pd.DataFrame([r[2] for r in results], columns=column_order)

    # flatten the per-unit field lists into one list (nested comprehension: for r ..., for f in r[3])
    all_field_rows = [f for r in results for f in r[3]]
    df_fields = pd.DataFrame(all_field_rows, columns=field_columns)

    # Speed-cell counts (Full session), using speed_cell: passed BOTH the
    # binned and time-domain shuffle tests with same-sign scores.
    # element-wise comparisons give boolean Series (`== True` is needed because the column mixes
    #   True/False/'not tested'/None; `# noqa: E712` silences the linter's style warning about it)
    _is_speed    = df_full['speed_cell'] == True    # noqa: E712  ('not tested' compares False)
    _is_place    = df_full['place_cell'] == True    # noqa: E712
    _is_nonplace = df_full['place_cell'] == False   # noqa: E712
    # counts: summing a boolean Series counts True values; `&` = element-wise AND
    n_speed_cells        = int(_is_speed.sum())
    n_place_speed_mod    = int((_is_place & _is_speed).sum())
    n_nonplace_speed_mod = int((_is_nonplace & _is_speed).sum())
    n_speed_not_tested   = int((df_full['speed_cell'] == 'not tested').sum())

    # summary table: one column of labels, one column of counts
    df_speed_summary = pd.DataFrame({
        'metric': ['Total units processed',
                   'Place cells',
                   'Speed cells (all, passed binned AND inst. tests)',
                   'Place cells also speed-modulated',
                   'Non-place cells speed-modulated',
                   'Speed not tested'],
        # counts in the same order as the labels
        'count':  [len(df_full),
                   int(_is_place.sum()),
                   n_speed_cells,
                   n_place_speed_mod,
                   n_nonplace_speed_mod,
                   n_speed_not_tested],
    })

    # write all sheets into one workbook (openpyxl engine); `with` saves and closes the file at the end
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # index=False omits pandas' row-number column
        df_full.to_excel(writer,   sheet_name='Full',        index=False)
        df_first.to_excel(writer,  sheet_name='First_Half',  index=False)
        df_second.to_excel(writer, sheet_name='Second_Half', index=False)
        df_fields.to_excel(writer, sheet_name='PlaceFields', index=False)
        df_speed_summary.to_excel(writer, sheet_name='Speed_Summary', index=False)

    # console summary
    print(f'\nDone. Results saved to {output_excel}')
    print(f'Total units processed              : {len(df_full)}')
    # summing the place_cell column counts True values (None/False add nothing)
    print(f'Place cells found                  : {df_full["place_cell"].sum()}')
    # adjacent f-strings inside print( ) are joined into one message
    print(f'Speed cells (binned AND inst. tests): {n_speed_cells}  '
          f'(shuffle-confirmed, {SPEED_N_SHUFFLE} shuffles, '
          f'{SPEED_SHUFFLE_MARGIN_S:.0f}s window)')
    print(f'Place cells also speed-modulated    : {n_place_speed_mod}')
    print(f'Non-place cells speed-modulated     : {n_nonplace_speed_mod}')
    print(f'Speed not tested                    : {n_speed_not_tested}')
    # pd.to_numeric(..., errors='coerce') turns non-numbers (None) into NaN
    n_fields_col = pd.to_numeric(df_full['n_fields_detected'], errors='coerce')
    print(f'Place fields detected                : {len(df_fields)}  '
          f'(threshold method, across {int((n_fields_col > 0).sum())} place cell(s))')

    # ── Copy .ntt + tracking files for confirmed place cells ───────────────────
    # Replicates the folder structure from the animal-ID folder onwards
    # (Fa1059 / Fa23BD / Fa8477 / Fa5384) under Output_PlaceTrue.

    # rows of the Full sheet whose final verdict is True (boolean-Series row selection)
    place_rows = df_full[df_full['place_cell'] == True]  # noqa: E712
    # set of folders whose tracking/.nev/.ncs files were already copied (copy them only once)
    copied_tracking_dirs = set()

    # .iterrows() yields (index, row) pairs; the index is ignored with `_`
    for _, row in place_rows.iterrows():
        session = row['session']
        # relpath gives '.' when the session folder IS root_folder
        dirpath = root_folder if session == '.' else os.path.join(root_folder, session)
        ntt_path = os.path.join(dirpath, row['unit'])

        # path tail starting at the animal-ID folder
        rel = _animal_relpath(dirpath)
        if rel is None:
            print(f'  [SKIP] Could not locate animal-ID folder in path: {dirpath}')
            continue

        # same sub-folder structure under Output_PlaceTrue
        dest_dir = os.path.join(Output_PlaceTrue, rel)
        os.makedirs(dest_dir, exist_ok=True)

        # copy the .ntt (copy2 also preserves file timestamps)
        if os.path.isfile(ntt_path):
            shutil.copy2(ntt_path, dest_dir)
        else:
            print(f'  [SKIP] .ntt file not found: {ntt_path}')

        # once per session folder: copy the tracking file and the raw .nev/.ncs files
        if dirpath not in copied_tracking_dirs:
            # dict.get returns None if the folder is not in the lookup
            csv_path = dir_to_csv.get(dirpath)
            if csv_path and os.path.isfile(csv_path):
                shutil.copy2(csv_path, dest_dir)
            # Also copy all .nev and .ncs files from the session folder
            # os.listdir lists the folder's entries (.nev = Neuralynx events, .ncs = continuous LFP)
            for fname in os.listdir(dirpath):
                if fname.lower().endswith(('.nev', '.ncs')):
                    src = os.path.join(dirpath, fname)
                    if os.path.isfile(src):
                        shutil.copy2(src, dest_dir)
            # mark the folder as done
            copied_tracking_dirs.add(dirpath)

    print(f'Place-cell files copied to : {Output_PlaceTrue}')
