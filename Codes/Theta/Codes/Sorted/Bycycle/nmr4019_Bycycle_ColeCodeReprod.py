"""
FA23BD-LFP — adapted to the folder-walk / .ncs data structure.

Frequency range: 1-40 Hz
Frequency resolution: 0.5 Hz (2 s epochs)
Aperiodic fit: knee (aperiodic_params_ = offset, knee, exponent)

This script keeps the notebook's analysis (mean PSDs, FOOOF, theta-property and
fit-quality plots, the master summary figure) but swaps the data-loading and
PSD-generation front end to the reference pipeline you provided:

    * animals are defined by a root folder (ANIMALS dict);
    * every .ncs under that folder is read with load_ncs();
    * PSDs are computed per file with dual-criteria epoch rejection, 50 Hz
      notch + spectral interpolation, and 1-100 Hz relative-power normalization
      (compute_psd_clean_epochs / clean_line_noise_psd / process_animal).

Because the reference structure is *flat* (one PSD per .ncs file) rather than the
notebook's animal -> date -> session -> tetrode -> channel hierarchy, the
downstream code was rewired to iterate over per-file PSD matrices. Per-file
metadata (date / session / tetrode / channel) is derived by
parse_metadata_from_path(), which you can edit to match your own naming.

PART 2 (fBOSC / IEI / speed, gated by y/n checkpoints) is appended unchanged from
the original notebook: those analyses read their OWN data (fBOSC episodesTables
pickles, tracking files) and are not produced by this .ncs PSD pipeline.

PART 3 (bycycle, gated by its own y/n checkpoint) adds cycle-by-cycle theta
analysis in the time domain, adapted from the bycycle-tools tutorial
"1. Theta oscillation cycle feature distributions"
(https://github.com/bycycle-tools/bycycle.git). It runs on the same cleaned
per-file LFP traces PART 2's fBOSC section reads from PROCESSED_PSDS_PKL.
"""

import os
import re
import sys
import math
import pickle

import numpy as np
import pandas as pd

# bycycle (burst/cycle.py) does in-place numpy assignment (is_burst[0] = False) on
# the array returned by Series.to_numpy(). Newer pandas hands back a read-only view
# from to_numpy() under copy-on-write, and pandas >=3 made CoW mandatory -- setting
# pd.options.mode.copy_on_write = False no longer disables it -- so force to_numpy()
# to always return a writable copy instead.
_pd_series_to_numpy = pd.Series.to_numpy
def _writable_series_to_numpy(self, *args, **kwargs):
    kwargs.setdefault('copy', True)
    return _pd_series_to_numpy(self, *args, **kwargs)
pd.Series.to_numpy = _writable_series_to_numpy

from scipy import signal, stats, interpolate, ndimage
from scipy.signal import savgol_filter

# bycycle 0.1.0's own filter helper (bycycle/filt.py: bandpass_filter/lowpass_filter,
# used internally by find_extrema -> compute_features for peak/trough localization)
# calls scipy.signal.firwin(..., nyq=f_nyq) -- a kwarg scipy removed in favor of
# fs=. Translate nyq -> fs=2*nyq so bycycle's cycle detection still works against
# a modern scipy, without touching the installed package.
_scipy_firwin = signal.firwin
def _firwin_nyq_compat(*args, **kwargs):
    if 'nyq' in kwargs:
        kwargs['fs'] = 2 * kwargs.pop('nyq')
    return _scipy_firwin(*args, **kwargs)
signal.firwin = _firwin_nyq_compat

import seaborn as sns
import matplotlib.pyplot as plt

# The script calls plt.show() ~16 times to display figures as they're produced.
# Its default (block=True) halts execution until each window is closed by hand.
# Make every plt.show() non-blocking instead, so figures pop up without pausing
# the run; plt.ion() + a short event-loop pump keep the GUI responsive so the
# windows actually draw. NOTE: pump via canvas.start_event_loop() directly, NOT
# plt.pause() -- plt.pause() internally calls the module-level show(block=False),
# which (since plt.show is reassigned below) would re-enter _non_blocking_show
# and recurse infinitely (RecursionError in matplotlib's Tk backend).
plt.ion()
_plt_show = plt.show
def _non_blocking_show(*args, **kwargs):
    kwargs.setdefault('block', False)
    _plt_show(*args, **kwargs)
    fig = plt.gcf()
    if fig.canvas.figure.stale:
        fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.001)
plt.show = _non_blocking_show

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from fooof import FOOOF, FOOOFGroup
from fooof.utils import interpolate_spectrum

from neurodsp.filt import filter_signal
from neurodsp.plts import plot_time_series
# The installed bycycle here is a pre-1.0 build -- the same vintage of API the
# Cole/Voytek paper's own script (2_Compute_cyclebycycle_features.py) uses
# (bycycle.features.compute_features), NOT the newer class-based `Bycycle`/
# `BycycleGroup` API (bycycle >=1.0), which this build does not provide. Only
# compute_features itself is imported -- burst-threshold logic (is_burst) is
# fully vendored in recompute_is_burst() below instead of relying on
# compute_features' burst_detection_kwargs / bycycle.burst._min_consecutive_cycles,
# since exactly which of those a given 0.1.x point-release accepts has proven
# inconsistent even across environments on this machine.
from bycycle.features import compute_features as bycycle_compute_features

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# compute_bycycle_features() deliberately never passes burst_detection_kwargs/
# threshold_kwargs to bycycle_compute_features (see import comment above) --
# bycycle warns about that on every single call, which would otherwise print
# once per file across the whole batch. is_burst is always overwritten
# immediately via recompute_is_burst(), so the warning doesn't reflect
# anything left unhandled. The message text differs between bycycle builds
# ("...parameters are provided" vs "...thresholds are provided") and both
# start with a newline + indentation (it's a triple-quoted string), so match
# leading whitespace explicitly -- warnings.filterwarnings anchors at the
# start of the string, and a bare "No burst..." pattern won't match past that
# leading whitespace.
warnings.filterwarnings("ignore",
                        message=r"\s*No burst detection (parameters|thresholds) are provided")


# %% ==================== Configuration (from reference code) ====================

# ---- Shared frequency range for FOOOF (Part 1) and fBOSC (Part 2) ----
# Single source of truth: both the FOOOF spectral/peak fit (FOOOF_RANGE, below)
# and the fBOSC oscillation-detection scan (FBOSC_F_ARRAY, via FBOSC_REGION_F_MAX,
# below) are derived from these two numbers, so the two analyses always agree on
# what frequency band is being analyzed. Change ONLY these two lines to change
# the band used everywhere.
FREQ_RANGE_LOW  = 1.0     # Hz; lower bound of the frequency range analyzed by FOOOF and fBOSC
FREQ_RANGE_HIGH = 40.0   # Hz; upper bound of the frequency range analyzed by FOOOF and fBOSC (this is what gets reported/detected against)

# Root directories -- one per animal.
# Add/remove/rename animals ONLY here -- plot colors, filenames, and legends
# below are all derived automatically from this dict's keys.
ANIMALS = {
    'Fa8477':  r'F:/Temp/Theta/fBOSC_Res/8477_Test',
    # 'FaDDE42': r'C:/Runita/NMR/analysis/SurgeryPaperSpikeLFP/LFP/Main/DDE42',
    'Fa23BD': r'F:/Temp/Theta/fBOSC_Res/23BD_Test',
    'Fa1059': r'F:/Temp/Theta/fBOSC_Res/1059_Test',
}


OUTPUT_DIR = r'F:/Temp/Theta/Bycycle_Res/Bycycle_Cole_v1'  # saved plots go here
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')            # summary figures

# fBOSCpy (oscillation-episode detection, used by PART 2 below) lives in a
# sibling folder and isn't installed as a package -- add it to sys.path so
# `from fBOSCpy_wrapper_v2 import fBOSCpy_wrapper_v2` resolves. PART 2 now
# runs fBOSC itself on the cleaned LFP traces this script produces, instead
# of reading pre-computed output from myCodes/LFP/fBOSCpy/run_fBOSC_batch.py.
FBOSCPY_DIR = r'F:/Temp/Theta/fBOSC_Res/v9_Res0.1/PostProcFalse'
if FBOSCPY_DIR not in sys.path:
    sys.path.insert(0, FBOSCPY_DIR)
from fBOSCpy_wrapper_v2 import fBOSCpy_wrapper_v2

# # Root directories -- one per animal.
# # Add/remove/rename animals ONLY here -- plot colors, filenames, and legends
# # below are all derived automatically from this dict's keys.
# ANIMALS = {
#     #'Fa8477':  r'X:/NMR_group_data/Runita/Data/Ephys_Data/AllSortedData/Tetrode/Fa8477',
#     # 'FaDDE42': r'C:/Runita/NMR/analysis/SurgeryPaperSpikeLFP/LFP/Main/DDE42',
#     'Fa23BD': r'C:/Runita/NMR/analysis/AllSort_Results/LFP/23BDTest',
#     'Fa1059': r'C:/Runita/NMR/analysis/AllSort_Results/LFP/1059Test',
# }


# OUTPUT_DIR = r'C:/Runita/NMR/analysis/AllSort_Results/LFP/BandByBand/v1'  # saved plots go here
# FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')            # summary figures

# ---- Saved-artifact paths (Part 1 outputs, under OUTPUT_DIR) ----
# PROCESSED_PSDS_PKL and FOOOF_RESULTS_PKL live under INPUT_PKL_DIR, which
# doubles as a cache: if answering "n" to the FOOOF checkpoint below, these
# are loaded from here instead of recomputed, and only recreated if missing.
INPUT_PKL_DIR         = os.path.join(OUTPUT_DIR, 'input_pkl')
PROCESSED_PSDS_PKL   = os.path.join(INPUT_PKL_DIR, 'processed_psds.pkl')  # per-file cleaned PSDs (pre-FOOOF)
FOOOF_RESULTS_PKL    = os.path.join(INPUT_PKL_DIR, 'fooof_results.pkl')   # full FOOOF fit output: {'fooof_results': [...], 'expanded_fooof_df': DataFrame}
LOW_QUALITY_FITS_TXT = os.path.join(OUTPUT_DIR, 'low_quality_fooof_fits.txt')

# ---- fBOSC (oscillation-episode detection, Part 2 below) ----
# Part 2 runs fBOSC itself, directly on the cleaned per-file LFP traces this
# script saves into PROCESSED_PSDS_PKL (see process_animal() / the 'lfp' entry
# added to each file's dict) -- it no longer reads pre-computed output from
# myCodes/LFP/fBOSCpy/run_fBOSC_batch.py.
# NOTE: fBOSC fits its own background spectrum internally (fBOSCUtils.fBOSC_getThresholds)
# and does NOT read FOOOF_RESULTS_PKL above -- that pkl is saved purely so Part 1's FOOOF
# fit results are available on disk (e.g. for later comparison/QC), not consumed by fBOSC.
# Frequencies scanned for oscillations. fBOSC_getThresholds() locks the FOOOF
# background/aperiodic fit range to exactly (F_array[0], F_array[-1]) -- so
# whatever sits at the TOP of F_array is always an unconstrained fit edge,
# where the 'knee' aperiodic model tends to undershoot true background power
# and understate the power threshold there, systematically over-detecting
# episodes at the top frequency regardless of whether the PSD actually peaks
# there. Fix: scan past the region we actually trust/report (FBOSC_REGION_F_MAX)
# by FBOSC_F_BUFFER_HZ, so that edge artifact lands in a buffer zone that gets
# discarded downstream (see `filtered_episodes` in the fBOSC section below)
# instead of contaminating the region of interest. Kept well clear of the
# 50/100/150/200 Hz line-noise harmonics (see LINE_HARMONICS).
FBOSC_REGION_F_MAX = FREQ_RANGE_HIGH   # Hz; upper bound of the region we actually report on (shared with FOOOF, see FREQ_RANGE_HIGH above)
FBOSC_F_BUFFER_HZ  = 8.0    # Hz; extra scan range above FBOSC_REGION_F_MAX, for the FOOOF fit only
FBOSC_F_ARRAY      = np.arange(FREQ_RANGE_LOW, FBOSC_REGION_F_MAX + FBOSC_F_BUFFER_HZ + 0.5, 0.1)
FBOSC_POSTPROC     = False                     # run FWHM episode post-processing (recommended)
# Episode duration threshold: dt_cycles(F) = max(FBOSC_MIN_NCYCLES, FBOSC_MIN_DURATION_S * F)
# -- requires an episode to last at least FBOSC_MIN_NCYCLES cycles AND at least
# FBOSC_MIN_DURATION_S seconds, whichever is stricter at that frequency. A fixed
# cycle count alone becomes an ever-shorter absolute-time requirement as F grows
# (3 cycles at 100 Hz is only 30 ms), letting brief noise/artifact transients
# pass the duration criterion purely because F is high. Below the crossover
# frequency (FBOSC_MIN_NCYCLES / FBOSC_MIN_DURATION_S = 3/0.15 = 20 Hz here) this
# reduces to the classic fixed-cycle rule, so theta (THETA_BAND, well under 20 Hz)
# is unaffected; only the higher end of FBOSC_F_ARRAY gets a stricter floor.
FBOSC_MIN_NCYCLES   = 3      # cycles; classic BOSC/eBOSC default
FBOSC_MIN_DURATION_S = 0.15  # s; absolute floor (~shortest credible oscillatory burst)
FBOSC_OUTPUT_DIR   = os.path.join(OUTPUT_DIR, 'fBOSC')                      # this script's own fBOSC outputs
FBOSC_FIGURE_DIR   = os.path.join(FBOSC_OUTPUT_DIR, 'figures')              # all Part 2 (fBOSC/IEI/speed) figures go here
FBOSC_FIT_DIR       = os.path.join(FBOSC_FIGURE_DIR, 'individual_fits')      # per-file fBOSC background-fit figures, <animal>/<file>_fbosc_fit.png
FBOSC_SAVE_FIT_FIGS = True   # save one background/aperiodic-fit figure per file (see _style_fbosc_fit_ax)
FBOSC_EPISODES_PKL = os.path.join(FBOSC_OUTPUT_DIR, 'fBOSC_episodesTable.pkl')  # cached combined episode table
FBOSC_USE_CACHE    = True   # reuse FBOSC_EPISODES_PKL if present instead of re-running the wavelet transform on every file
                             # NOTE: delete FBOSC_EPISODES_PKL (or set this False) after changing
                             # FBOSC_F_ARRAY / FBOSC_REGION_F_MAX / FBOSC_F_BUFFER_HZ -- a cached
                             # table was detected with the OLD frequency range and still carries
                             # the old fit's edge artifact.

# ---- bycycle (cycle-by-cycle theta analysis in the time domain, Part 3) ----
# Started from the bycycle-tools tutorial "1. Theta oscillation cycle feature
# distributions" (https://github.com/bycycle-tools/bycycle.git) and adapted to
# run per-file on this pipeline's own cleaned LFP traces (PROCESSED_PSDS_PKL's
# lfp_store, i.e. the same notch-filtered/detrended/downsampled traces PART 2's
# fBOSC section uses) instead of the tutorial's bundled ca1.npy/ec3.npy example.
BYCYCLE_F_THETA        = (3.0, 7.0)  # Hz; fallback theta band scanned for cycles -- used
                                      # as-is when BYCYCLE_USE_FOOOF_THETA is False, and as
                                      # the per-animal fallback (see compute_fooof_theta_bands)
                                      # when it's True but an animal has no qualifying FOOOF fit.
# FOOOF (PART 1) fits, per file, whether a genuine oscillatory peak rises above
# the aperiodic background and where it's centred -- bycycle (time domain)
# can't see that, since averaging over cycles is exactly what a cycle-by-cycle
# method avoids. So rather than guessing one fixed band for every animal:
#   * BYCYCLE_USE_FOOOF_THETA centres each animal's BYCYCLE_F_THETA on that
#     animal's own FOOOF-measured theta peak (compute_fooof_theta_bands: median
#     centre freq +/- half the median bandwidth, across that animal's
#     good-quality fits) instead of the fixed band above;
#   * BYCYCLE_GATE_ON_FOOOF_QUALITY skips bycycle entirely for files where
#     FOOOF found no significant theta peak or the fit is below the same
#     r_squared/error quality bar export_low_quality_fits() flags -- no point
#     characterizing cycle shape where there's no evidence of a real rhythm.
BYCYCLE_USE_FOOOF_THETA       = True
BYCYCLE_GATE_ON_FOOOF_QUALITY = True
# Paper-matched preprocessing (Cole_2019_theta/config.py + compute_shape()):
# lowpass at cf_low=25 Hz (0.5 s kernel) THEN highpass at cf_high=1 Hz (2 s
# kernel), applied in that order, before cycles are ever detected. Restored to
# the paper's 25 Hz/1 Hz pair here (a prior tuning pass had raised the lowpass
# to 40 Hz and left the highpass step out entirely -- see BYCYCLE_APPLY_HIGHPASS
# below, which fills that gap).
BYCYCLE_LOWPASS_HZ     = 25.0  # Hz; lowpass cutoff applied before cycle detection (paper: cf_low=25)
BYCYCLE_FILTER_SECONDS = 0.5   # s; lowpass filter length (paper: N_seconds_low=0.5)
BYCYCLE_APPLY_HIGHPASS  = True  # paper always runs this step (not optional there); kept as a
                                 # toggle here only for consistency with this script's other steps
BYCYCLE_HIGHPASS_HZ         = 1.0  # Hz; highpass cutoff, removes slow drift (paper: cf_high=1)
BYCYCLE_HIGHPASS_SECONDS    = 2.0  # s; highpass filter length (paper: N_seconds_high=2)
BYCYCLE_CENTER_EXTREMA = 'trough'     # cycles centered on trough. Peak trough yields similar results. Trough has slightly higher theta cycles.
# process_animal() already runs one global linear detrend per file (see
# APPLY_TIME_DETREND) before saving into lfp_store, but that single fit spans
# the whole (multi-minute) trace and won't remove slower local drift within a
# file -- which corrupts bycycle's amplitude/period-consistency checks and
# causes real theta cycles to be dropped as non-bursting. So detrend again,
# specifically on the trace handed to bycycle, using DETREND_TYPE.
BYCYCLE_APPLY_DETREND  = False
# Outlier samples (movement/chewing artifacts, transient spikes) in the raw
# trace inflate the amp_consistency/period_consistency reference statistics
# for neighboring cycles, pushing genuine theta cycles below threshold. A
# Hampel filter (rolling median + MAD, see hampel_filter()) flags and
# linearly interpolates those samples before bycycle ever sees them.
# BYCYCLE_HAMPEL_WINDOW_S is the TOTAL rolling window length in seconds --
# sized to ~1-2 theta cycles so it's loose enough not to flag genuine cycle
# extrema as outliers; BYCYCLE_HAMPEL_SIGMAS is the robust-sigma (MAD-scaled)
# threshold.
BYCYCLE_APPLY_HAMPEL   = False
BYCYCLE_HAMPEL_WINDOW_S = 0.3   # s; total window (~1-2 cycles at theta freq)
BYCYCLE_HAMPEL_SIGMAS   = 5.0   # robust-sigma threshold
# Savitzky-Golay smoothing, applied to the Hampel-filtered signal (i.e. AFTER
# outlier interpolation, so a polynomial isn't fit across a raw spike) to
# smooth out fine jaggedness/interpolation artifacts left behind by Hampel,
# before the existing lowpass filter shapes the theta band. Unlike the
# lowpass filter (a hard frequency cutoff), Savitzky-Golay is a local
# polynomial fit -- it smooths without the phase distortion/ringing an IIR
# filter can introduce, which matters here since bycycle relies on precise
# extrema/zero-crossing locations. BYCYCLE_SAVGOL_WINDOW_S must stay well
# below one theta cycle's period (e.g. <1/4 of it) or it starts flattening
# genuine peaks/troughs instead of noise.
BYCYCLE_APPLY_SAVGOL     = False
BYCYCLE_SAVGOL_WINDOW_S  = 0.02  # s; ~20 ms, well under 1/4 of a theta cycle (~140-330 ms)
BYCYCLE_SAVGOL_POLYORDER = 3
# Burst-detection thresholds (tutorial defaults) -- a cycle counts as an
# oscillatory burst only if it and its neighbors clear all four criteria.
BYCYCLE_THRESHOLDS = {
    'amp_fraction_threshold':       0.0, # 0.2 -> 0.0: the bycycle-tools theta tutorial (Cole &
                                          # Voytek) uses 0, i.e. no percentile-amplitude cutoff on
                                          # top of the four thresholds below. 0.2 required every
                                          # cycle's flank amplitude to clear the file's own 20th
                                          # percentile -- an extra, undocumented cut that penalizes
                                          # exactly the amplitude modulation (running speed/arousal)
                                          # real theta cycles show.
    'amp_consistency_threshold':    0.6, #0.6, #decreasing this increases theta cycles.
    'period_consistency_threshold': 0.75, #0.75,#decreasing this increases theta cycles.
    'monotonicity_threshold':       0.6, #0.8, #decreasing this increases theta cycles.
    'min_n_cycles':                 3,
}
# Paper robustness sweep (Cole_2019_theta/config.py `burst_kwargs_list`): the
# paper does not commit to one burst-threshold choice -- it recomputes is_burst
# under 4 parameter sets (their index 0 is also the set used for the *primary*
# figures) and reports results are consistent across all of them. BYCYCLE_THRESHOLDS
# above is this pipeline's own (separately tuned, see comments above) primary
# set; these 4 are additionally recomputed per cycle -- with zero extra bycycle
# fitting cost, since amp_fraction/amp_consistency/period_consistency/monotonicity
# are already computed once per cycle regardless of threshold -- and stored as
# is_burst_paper0..is_burst_paper3 for direct comparison/robustness reporting.
BYCYCLE_PAPER_BURST_SETS = [
    {'amp_fraction_threshold': 0, 'amp_consistency_threshold': 0.40,
     'period_consistency_threshold': 0.55, 'monotonicity_threshold': 0.80, 'min_n_cycles': 3},
    {'amp_fraction_threshold': 0, 'amp_consistency_threshold': 0.60,
     'period_consistency_threshold': 0.60, 'monotonicity_threshold': 0.90, 'min_n_cycles': 3},
    {'amp_fraction_threshold': 0, 'amp_consistency_threshold': 0.35,
     'period_consistency_threshold': 0.50, 'monotonicity_threshold': 0.75, 'min_n_cycles': 3},
    {'amp_fraction_threshold': 0, 'amp_consistency_threshold': 0.65,
     'period_consistency_threshold': 0.65, 'monotonicity_threshold': 0.95, 'min_n_cycles': 3},
]
BYCYCLE_OUTPUT_DIR   = os.path.join(OUTPUT_DIR, 'bycycle')          # this script's own bycycle outputs
BYCYCLE_FIGURE_DIR   = os.path.join(BYCYCLE_OUTPUT_DIR, 'figures')  # all Part 3 figures go here
BYCYCLE_FEATURES_PKL = os.path.join(BYCYCLE_OUTPUT_DIR, 'bycycle_features.pkl')  # cached combined cycle-feature table
BYCYCLE_USE_CACHE    = True   # reuse BYCYCLE_FEATURES_PKL if present instead of re-running cycle detection on every file
                               # NOTE: delete BYCYCLE_FEATURES_PKL (or set this False) after changing
                               # BYCYCLE_F_THETA / BYCYCLE_LOWPASS_HZ / BYCYCLE_THRESHOLDS /
                               # BYCYCLE_USE_FOOOF_THETA / BYCYCLE_GATE_ON_FOOOF_QUALITY -- a cached
                               # table was detected with the OLD settings.

# ---- Acquisition / PSD ----
fs      = 32000  # original sampling rate (Hz)
fs_down = 500    # target sampling rate after downsampling
nperseg = int(2 * fs_down)  # 2 s epochs (1000 samples at 1000 Hz) -> df = 0.5 Hz
ADBitVolts = 0.000003051757812500000169  # V per ADC count (Neuralynx header)
MAD_THRESH = 5.0          # dual-criteria epoch-rejection threshold (robust z)
LOW_BAND   = (1.0, 3.0)   # delta band: 1-3 Hz rejection criterion + delta/theta filter
NORM_BAND  = (1.0, 100.0) # band used for relative-power normalization

# Delta/theta epoch filter -- applied to velocity-passed epochs BEFORE MAD
# filtering: an epoch is rejected if its LOW_BAND (delta, 1-3 Hz) power
# exceeds its THETA_BAND (3-7 Hz, defined below) power. THETA_BAND is
# resolved at call time, so its definition later in this file still applies.
APPLY_DELTA_THETA_FILTER = True

# ---- Velocity / running-speed epoch gating ----
# Every .ncs file's session folder holds one tracking .csv with a UNIX
# timestamp (us, column A) on the same clock as the .ncs timestamps, plus x/y
# position in cm (columns D, E). Epochs are additionally gated on running
# speed: an epoch is only kept if the animal's median smoothed speed during it
# falls within [SPEED_MIN_CMS, SPEED_MAX_CMS].
SPEED_MIN_CMS       = 1    # lower running-speed bound for epoch acceptance (cm/s)
SPEED_MAX_CMS       = 90.0   # upper running-speed bound for epoch acceptance (cm/s)
SPEED_SMOOTH_WINDOW = 11     # Savitzky-Golay window (samples, odd; ~0.37 s at 30 Hz)
SPEED_SMOOTH_POLY   = 3

# ---- 50 Hz line-noise cleaning (European mains) ----
LINE_FREQ             = 50.0
LINE_HARMONICS        = [50.0, 100.0, 150.0, 200.0]  # harmonics below Nyquist (250 Hz)
APPLY_TIME_NOTCH      = True   # scipy IIR notch on the time series
NOTCH_Q               = 30.0
APPLY_SPECTRAL_INTERP = True   # FOOOF interpolate_spectrum on the PSD
INTERP_HALFWIDTH      = 2.0

# ---- Detrending (after notch filter) ----
APPLY_TIME_DETREND    = True   # remove slow drift from the full trace
DETREND_TYPE          = 'linear'  # 'linear' or 'constant' (see scipy.signal.detrend)

# ---- FOOOF / specparam ----
FOOOF_RANGE    = [FREQ_RANGE_LOW, FREQ_RANGE_HIGH]   # fit range (Hz) -- shared with fBOSC, see FREQ_RANGE_LOW/HIGH above
FOOOF_SETTINGS = dict(
    peak_width_limits=[1.0, 8.0],   
    max_n_peaks=6,
    min_peak_height=0.1,
    peak_threshold=2.0,
    aperiodic_mode='knee',    # 'knee' lets the aperiodic fit bend -- needed across 1-40 Hz, see FOOOF_fBOSC_Readme.txt
)

# ---- Theta extraction / property plotting (from notebook) ----
THETA_BAND = (3.0, 7.0)        # Hz window used to pull the theta peak from FOOOF

# Neuralynx .ncs record format (512 int16 samples per record, 16 kB header skipped)
ncs_dtype = np.dtype([
    ('timestamp'  , '<u8'),
    ('sc_number'  , '<u4'),
    ('cell_number', '<u4'),
    ('params'     , '<u4'),
    ('samples'    , '<i2', (512,)),
])

# ---- Plot palette (auto-cycled across however many animals are in ANIMALS) ----
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


# {label: (line_color, fill_color)} -- auto-built from ANIMALS, one entry per animal.
STYLES = {
    label: (_PALETTE[i % len(_PALETTE)], _lighten(_PALETTE[i % len(_PALETTE)]))
    for i, label in enumerate(ANIMALS)
}

# convenience: a simple list of line colors in ANIMALS order (used by notebook plots)
animals = list(ANIMALS.keys())
ANIMAL_COLORS = [STYLES[a][0] for a in animals]


# %% ==================== Signal-processing helpers (from reference) ============

def load_ncs(fpath):
    """Read a Neuralynx .ncs file.

    Returns
    -------
    lfp             : ndarray  raw trace in microvolts (uV)
    start_timestamp : int      UNIX timestamp (us) of the first sample -- same
                                clock as the tracking .csv, used to align
                                epochs to running speed.
    """
    data = np.memmap(fpath, dtype=ncs_dtype, mode='r', offset=16 * 1024)
    lfp = np.concatenate(data['samples']).astype(np.float64) * ADBitVolts * 1e6
    start_timestamp = int(data['timestamp'][0])
    return lfp, start_timestamp


def notch_filter(x, fs_hz, freqs, Q=30.0):
    """Zero-phase IIR notch at each frequency in `freqs` (skips freqs >= Nyquist)."""
    y = np.asarray(x, dtype=np.float64)
    nyq = fs_hz / 2.0
    for f0 in freqs:
        if f0 <= 0 or f0 >= nyq:
            continue
        b, a = signal.iirnotch(f0, Q, fs_hz)
        y = signal.filtfilt(b, a, y)
    return y


def detrend_signal(x, dtype='linear'):
    """Remove a polynomial trend (default: linear) from the full LFP trace,
    to clear slow drift that a notch filter alone doesn't address."""
    return signal.detrend(np.asarray(x, dtype=np.float64), type=dtype)


def hampel_filter(x, window_size, n_sigmas=4.0):
    """Hampel identifier: flag outlier samples by robust (MAD) z-score against
    a ROLLING (not global) median, then replace them via linear interpolation
    from the nearest surviving (non-outlier) neighbors on either side.

    Windowed rather than global so the reference statistics track local
    signal level -- e.g. a chewing-artifact burst in one part of the trace
    doesn't get judged against a quiet stretch elsewhere.

    Parameters
    ----------
    x           : array-like
    window_size : int    half-window in samples; the rolling window centered
                  on each sample spans `2 * window_size + 1` samples
    n_sigmas    : float   outlier threshold, in robust-sigma units (MAD * 1.4826
                  approximates a Gaussian sigma)

    Returns
    -------
    x_clean      : ndarray  same shape as `x`, outliers replaced by interpolation
    outlier_mask : ndarray of bool  True where `x` was flagged as an outlier
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    win = 2 * window_size + 1

    rolling_median = ndimage.median_filter(x, size=win, mode='reflect')
    abs_dev        = np.abs(x - rolling_median)
    rolling_mad    = ndimage.median_filter(abs_dev, size=win, mode='reflect')

    # A rolling MAD of (near-)zero would make the threshold near-zero too,
    # flagging almost every sample in an otherwise-quiet stretch -- floor it
    # against the trace's global MAD (scaled down, since local MAD is
    # normally smaller than global) rather than against a bare epsilon.
    global_mad  = np.median(np.abs(x - np.median(x)))
    floor       = max(1e-3 * global_mad, 1e-20)
    rolling_mad = np.maximum(rolling_mad, floor)

    outlier_mask = abs_dev > (n_sigmas * 1.4826 * rolling_mad)

    x_clean = x.copy()
    good = ~outlier_mask
    if outlier_mask.any() and good.sum() >= 2:
        idx = np.arange(n)
        x_clean[outlier_mask] = np.interp(idx[outlier_mask], idx[good], x[good])

    return x_clean, outlier_mask


def _robust_high_outliers(x, thresh, ref_mask=None):
    """Boolean mask of samples that are high outliers by robust (MAD) z-score.

    If `ref_mask` is given, the median/MAD reference statistics are estimated
    from `x[ref_mask]` only (e.g. epochs that already passed the velocity and
    delta/theta filters), so epochs excluded upstream can't skew the robust
    threshold -- z-scores are still returned for every element of `x`.
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


def compute_psd_clean_epochs(lfp, fs_hz, nperseg, mad_thresh=5.0,
                             low_band=(1.0, 3.0), theta_band=None,
                             apply_delta_theta_filter=True, speed_keep=None):
    """Welch PSD averaged over 4 s epochs, with epoch rejection applied in
    three stages, in order:

      1. Running-speed gating -- `speed_keep` (one bool per epoch), computed
         upstream from tracking data.
      2. Delta/theta filter -- within the epochs kept by (1), an epoch is
         rejected if its `low_band` (delta, e.g. 1-3 Hz) power exceeds its
         `theta_band` (e.g. 3-7 Hz) power.
      3. Dual-criteria MAD outlier rejection -- an epoch is rejected if it is
         a high outlier (robust MAD z > mad_thresh) on EITHER broadband
         peak-to-peak amplitude OR `low_band` power. The median/MAD reference
         statistics are computed ONLY from the epochs that survive (1) and
         (2), so epochs already excluded can't skew the robust threshold.

    `theta_band` defaults to the module-level THETA_BAND (3-7 Hz).
    """
    theta_band = theta_band or THETA_BAND
    lfp = np.asarray(lfp, dtype=np.float64)
    n_total = len(lfp) // nperseg
    if n_total == 0:
        raise ValueError('Trace is shorter than one epoch.')

    epochs = lfp[:n_total * nperseg].reshape(n_total, nperseg)
    p2p = epochs.max(axis=1) - epochs.min(axis=1)

    win = signal.get_window('hann', nperseg)
    band_pow  = np.empty(n_total)  # delta (low_band) power, per epoch
    theta_pow = np.empty(n_total)  # theta_band power, per epoch
    psd_stack = None
    f = None
    for i in range(n_total):
        f, Pi = signal.welch(epochs[i], fs=fs_hz, window=win,
                             nperseg=nperseg, noverlap=0, detrend='constant')
        if psd_stack is None:
            psd_stack = np.empty((n_total, Pi.size))
        psd_stack[i] = Pi
        df  = f[1] - f[0]
        idx_low   = (f >= low_band[0])   & (f <= low_band[1])
        idx_theta = (f >= theta_band[0]) & (f <= theta_band[1])
        band_pow[i]  = np.sum(Pi[idx_low]) * df
        theta_pow[i] = np.sum(Pi[idx_theta]) * df

    # 1) velocity gating
    keep = np.asarray(speed_keep, dtype=bool) if speed_keep is not None \
        else np.ones(n_total, dtype=bool)

    # 2) delta/theta filter, restricted to the epochs already kept above
    if apply_delta_theta_filter:
        keep = keep & (band_pow <= theta_pow)

    # 3) MAD outlier rejection, referenced to the epochs kept by (1)-(2)
    reject = _robust_high_outliers(p2p, mad_thresh, ref_mask=keep) | \
             _robust_high_outliers(band_pow, mad_thresh, ref_mask=keep)
    keep = keep & ~reject

    if speed_keep is not None:
        n_clean = int(keep.sum())
        if n_clean == 0:
            raise ValueError(
                'No epochs pass the running-speed '
                f'({SPEED_MIN_CMS}-{SPEED_MAX_CMS} cm/s), delta/theta, and '
                'MAD artifact-rejection criteria.')
    else:
        n_clean = int(keep.sum())
        if n_clean == 0:
            keep = np.ones(n_total, dtype=bool)
            n_clean = n_total

    Pxx = psd_stack[keep].mean(axis=0)
    return f, Pxx, n_clean, n_total


def find_position_file(ncs_path):
    """Return the tracking .csv that lives alongside `ncs_path` (same folder).

    Every session folder holds exactly one tracking .csv shared by all its
    .ncs files. Returns None if none is found; if more than one is present,
    the first (alphabetically) is used.
    """
    folder = os.path.dirname(ncs_path)
    candidates = sorted(
        fn for fn in os.listdir(folder)
        if fn.lower().endswith('.csv'))
    if not candidates:
        return None
    if len(candidates) > 1:
        print(f'    Multiple .csv tracking files in {folder}, '
              f'using: {candidates[0]}')
    return os.path.join(folder, candidates[0])


def compute_velocity_from_position(csv_path):
    """Compute smoothed running speed (cm/s) from a tracking .csv.

    Column layout (positional): UNIX timestamp in us (col A) on the same
    clock as the .ncs LFP timestamps, x in cm (col D), y in cm (col E).
    Speed at sample n is the frame-to-frame displacement divided by the
    actual elapsed time between consecutive rows (from the column-A
    timestamps, so irregular/dropped tracking frames don't bias it),
    converted to cm/s, then smoothed with a Savitzky-Golay filter (preserves
    genuine speed transients better than a moving average while removing
    frame-to-frame tracking jitter).

    Returns
    -------
    time_us : ndarray   absolute UNIX timestamp of each tracking sample (us)
                        -- same clock as the .ncs LFP start timestamp, so
                        epochs can be aligned to speed without assuming a
                        shared t=0
    speed   : ndarray   smoothed running speed (cm/s), same length as time_us
    """
    df = pd.read_csv(csv_path, usecols=[0, 3, 4])
    df.columns = ['time_us', 'x', 'y']
    df = df.sort_values('time_us').reset_index(drop=True)

    time_us = df['time_us'].to_numpy(dtype=np.float64)

    dt_us = np.diff(time_us)
    dx = np.diff(df['x'].to_numpy(dtype=np.float64))
    dy = np.diff(df['y'].to_numpy(dtype=np.float64))
    dist = np.sqrt(dx**2 + dy**2)
    speed_step = np.divide(dist, dt_us, out=np.zeros_like(dist), where=dt_us > 0) * 1e6  # cm/s
    speed_raw = np.concatenate(([0.0], speed_step))

    n = len(speed_raw)
    window = min(SPEED_SMOOTH_WINDOW, n if n % 2 == 1 else n - 1)
    if window < SPEED_SMOOTH_POLY + 2:
        speed_smooth = speed_raw.copy()
    else:
        speed_smooth = savgol_filter(speed_raw, window, SPEED_SMOOTH_POLY)
    speed_smooth = np.clip(speed_smooth, 0, None)

    return time_us, speed_smooth


def compute_epoch_speed_keep(time_us, speed, n_total, nperseg, fs_hz, lfp_start_us,
                             speed_min=SPEED_MIN_CMS, speed_max=SPEED_MAX_CMS):
    """Boolean mask (len n_total): True where an epoch's median running speed
    falls within [speed_min, speed_max] cm/s.

    Epoch i spans absolute UNIX time [lfp_start_us + i*epoch_dur_us,
    lfp_start_us + (i+1)*epoch_dur_us) -- lfp_start_us is the .ncs file's
    first-sample timestamp, on the same UNIX clock as the tracking .csv, so
    epochs are aligned to actual recording time rather than an assumed
    shared t=0.
    """
    epoch_dur_us = 1e6 * nperseg / fs_hz
    keep = np.zeros(n_total, dtype=bool)
    for i in range(n_total):
        t0 = lfp_start_us + i * epoch_dur_us
        t1 = lfp_start_us + (i + 1) * epoch_dur_us
        in_epoch = (time_us >= t0) & (time_us < t1)
        if in_epoch.any():
            med_speed = np.median(speed[in_epoch])
        else:
            nearest = np.argmin(np.abs(time_us - (t0 + t1) / 2))
            med_speed = speed[nearest]
        keep[i] = (med_speed >= speed_min) & (med_speed <= speed_max)
    return keep


def clean_line_noise_psd(f, Pxx, harmonics, halfwidth=2.0):
    """Interpolate the PSD across each mains harmonic (FOOOF interpolate_spectrum)."""
    nyq    = f[-1]
    ranges = [[h - halfwidth, h + halfwidth]
              for h in harmonics if (h + halfwidth) < nyq]
    if not ranges:
        return f, Pxx
    f_i, P_i = interpolate_spectrum(f, Pxx, ranges)
    return f_i, P_i


# %% ==================== Per-animal processing (from reference) ================

def process_animal(label, folder):
    """Process every .ncs under `folder`.

    Returns
    -------
    freq_vec, mean_psd, sem_psd, n_files, psds_norm, file_names, lfp_store
        lfp_store: {rel_file_name: {'lfp': cleaned trace (post notch/detrend,
        pre-epoch-averaging, fs_down Hz), 'fs': fs_down, 'start_us': .ncs
        first-sample UNIX timestamp}} -- the raw material PART 2's fBOSC
        section needs; saved into PROCESSED_PSDS_PKL below so fBOSC can run
        directly off this script's own output.
    """
    ncs_files = []
    for root, _dirs, files in os.walk(folder):
        for fname in files:
            if fname.endswith('.ncs'):
                ncs_files.append(os.path.join(root, fname))
    print(f'Found {len(ncs_files)} .ncs files in {folder}')

    psds       = []
    file_names = []
    freq_vec   = None
    lfp_store  = {}   # {rel_file_name: {'lfp':..., 'fs':..., 'start_us':...}} -- for fBOSC (Part 2)
    velocity_cache = {}   # {csv_path: (time_us, speed)} -- shared across .ncs in a session

    for fpath in ncs_files:
        rel = os.path.relpath(fpath, folder)
        try:
            lfp, lfp_start_us = load_ncs(fpath)
            lfp = signal.resample_poly(lfp, fs_down, fs)
            if APPLY_TIME_NOTCH:
                lfp = notch_filter(lfp, fs_down, LINE_HARMONICS, NOTCH_Q)
            if APPLY_TIME_DETREND:
                lfp = detrend_signal(lfp, dtype=DETREND_TYPE)

            pos_path = find_position_file(fpath)
            speed_keep = None
            if pos_path is None:
                print(f'    No tracking .csv found next to {rel} '
                      f'-- speed filter skipped for this file')
            else:
                if pos_path not in velocity_cache:
                    velocity_cache[pos_path] = compute_velocity_from_position(pos_path)
                time_us, speed = velocity_cache[pos_path]
                n_total_est = len(lfp) // nperseg
                speed_keep = compute_epoch_speed_keep(
                    time_us, speed, n_total_est, nperseg, fs_down, lfp_start_us)

            f, Pxx, n_clean, n_total = compute_psd_clean_epochs(
                lfp, fs_down, nperseg, mad_thresh=MAD_THRESH, low_band=LOW_BAND,
                apply_delta_theta_filter=APPLY_DELTA_THETA_FILTER,
                speed_keep=speed_keep)

            if APPLY_SPECTRAL_INTERP:
                f, Pxx = clean_line_noise_psd(
                    f, Pxx, LINE_HARMONICS, INTERP_HALFWIDTH)

            if freq_vec is None:
                freq_vec = f

            df        = f[1] - f[0]
            valid_idx = (f >= NORM_BAND[0]) & (f <= NORM_BAND[1])
            total_power = np.sum(Pxx[valid_idx]) * df
            Pxx_norm    = Pxx / total_power

            psds.append(Pxx_norm)
            file_names.append(rel)
            lfp_store[rel] = {'lfp': lfp, 'fs': fs_down, 'start_us': lfp_start_us}
            if speed_keep is not None:
                print(f'  OK: {rel}  [{n_clean}/{n_total} epochs kept; '
                      f'{int(speed_keep.sum())}/{n_total} met '
                      f'{SPEED_MIN_CMS}-{SPEED_MAX_CMS} cm/s speed criterion]')
            else:
                print(f'  OK: {rel}  [{n_clean}/{n_total} epochs kept]')

        except Exception as e:
            print(f'  SKIP: {rel} -- {e}')

    print(f'  -> {len(psds)} files processed\n')
    if not psds:
        raise ValueError(f'No files processed successfully in {folder}')

    psds     = np.array(psds)
    mean_psd = np.mean(psds, axis=0)
    sem_psd  = np.std(psds, axis=0) / np.sqrt(psds.shape[0])
    return freq_vec, mean_psd, sem_psd, psds.shape[0], psds, file_names, lfp_store


# %% ==================== Flat-structure metadata hook ==========================

def parse_metadata_from_path(rel_path):
    """Map a relative .ncs path to (date, session, tetrode, channel) metadata.

    The reference pipeline is flat (one PSD per file), so there is no built-in
    session/tetrode hierarchy. EDIT THIS to match your folder/filename layout if
    you want session- or arena-level grouping downstream (e.g. plot_arena_comparison).

    Default heuristic:
      * tetrode/channel parsed from a Neuralynx-style 'CSC<t>ch<c>' token if present;
      * date/session taken from the first two parent folders of the relative path;
      * anything unknown falls back to the filename stem or 'NA'.
    """
    fname = os.path.basename(rel_path)
    parts = os.path.normpath(rel_path).split(os.sep)

    m = re.search(r'CSC(\d+)(?:ch(\d+))?', fname, re.IGNORECASE)
    tetrode = m.group(1) if m else 'NA'
    channel = m.group(2) if (m and m.group(2)) else os.path.splitext(fname)[0]

    date    = parts[0] if len(parts) >= 2 else 'NA'
    session = parts[1] if len(parts) >= 3 else (parts[0] if len(parts) >= 2 else 'NA')
    return {'date': date, 'session': session, 'tetrode': tetrode, 'channel': channel}


# %% ==================== FOOOF over per-file PSD matrices =======================

def build_fooof_results(results, fooof_settings=None, fooof_range=None,
                        save_fits=False, save_dir=None, fit_xlim=(1, 20)):
    """Fit FOOOF to every per-file PSD (via FOOOFGroup) and return a results list.

    Each entry mirrors the notebook's fooof_results dicts so that the original
    fooof_results_to_df() works unchanged:
        animal, date, session, tetrode, channel,
        aperiodic_params, peak_params, r_squared, error

    If `save_fits` is True, also saves one model-fit figure (original spectrum,
    full model, aperiodic fit) per file under
    `save_dir/<animal>/<sanitized file stem>_fooof_fit.png` (default save_dir:
    FIGURE_DIR/individual_fits, i.e. under OUTPUT_DIR).
    """
    fooof_settings = fooof_settings or FOOOF_SETTINGS
    fooof_range    = fooof_range or FOOOF_RANGE

    if save_fits:
        save_dir = save_dir or os.path.join(FIGURE_DIR, 'individual_fits')

    fooof_results = []
    for animal, (freqs, _mean, _sem, _n, psds_norm, file_names, _lfp_store) in results.items():
        print(f"  FOOOFGroup: {animal}  ({psds_norm.shape[0]} PSDs)")
        fg = FOOOFGroup(**fooof_settings)
        fg.fit(freqs, psds_norm, fooof_range)

        if save_fits:
            animal_dir = os.path.join(save_dir, animal)
            os.makedirs(animal_dir, exist_ok=True)

        for i in range(psds_norm.shape[0]):
            # regenerate=True so the modeled spectrum/aperiodic fit are
            # available for plotting (regenerate=False only keeps params).
            fm = fg.get_fooof(i, regenerate=save_fits)
            meta = parse_metadata_from_path(file_names[i])
            fooof_results.append({
                'animal':           animal,
                'file':             file_names[i],
                **meta,
                'aperiodic_params': fm.aperiodic_params_,
                'peak_params':      fm.peak_params_,
                'r_squared':        fm.r_squared_,
                'error':            fm.error_,
            })

            if save_fits:
                fig, ax = plt.subplots(figsize=(5, 3))
                _style_fooof_fit_ax(ax, fm, xlim=fit_xlim,
                                    title=f"{animal}: {os.path.basename(file_names[i])}")
                stem = re.sub(r'[\\/]+', '_', os.path.splitext(file_names[i])[0])
                fig.savefig(os.path.join(animal_dir, f'{stem}_fooof_fit.png'),
                           dpi=200, bbox_inches='tight')
                plt.close(fig)

        if save_fits:
            print(f"    Saved {psds_norm.shape[0]} fit figures -> {animal_dir}")

    return fooof_results


def extract_theta_peak(peak_params, theta_band=None):
    """Return (cf, pw, bw) of the strongest FOOOF peak whose centre frequency
    falls within theta_band, or (nan, nan, nan) if none does."""
    theta_band = theta_band or THETA_BAND
    if len(peak_params) > 0:
        cfs = peak_params[:, 0]
        in_theta = (cfs >= theta_band[0]) & (cfs <= theta_band[1])
        theta_peaks = peak_params[in_theta]
        if len(theta_peaks) > 0:
            strongest = theta_peaks[np.argmax(theta_peaks[:, 1])]
            return tuple(strongest)
    return np.nan, np.nan, np.nan


def theta_range_from_peak(cf, bw):
    """Upper/lower theta bound from a FOOOF peak's centre freq + bandwidth
    (FOOOF's peak_params bandwidth is the full width, so +/- bw/2 around cf)."""
    if np.isnan(cf) or np.isnan(bw):
        return np.nan, np.nan
    return cf - bw / 2, cf + bw / 2


def fooof_results_to_df(fooof_results, theta_band):
    """Convert fooof_results list of dicts to a flat dataframe.

    Extracts theta peak (CF, PW, BW) from peak_params within theta_band, plus
    the resulting theta_low/theta_high frequency range. One row per LFP file.
    """
    rows = []
    for r in fooof_results:
        ap = r['aperiodic_params']
        if len(ap) == 2:
            offset, exponent = ap
            knee = np.nan
        else:
            offset, knee, exponent = ap

        theta_cf, theta_pw, theta_bw = extract_theta_peak(r['peak_params'], theta_band)
        theta_low, theta_high = theta_range_from_peak(theta_cf, theta_bw)

        rows.append({
            'animal':     r['animal'],
            'file':       r.get('file', ''),
            'date':       r['date'],
            'session':    r['session'],
            'tetrode':    r['tetrode'],
            'channel':    r['channel'],
            'offset':     offset,
            'knee':       knee,
            'exponent':   exponent,
            'theta_cf':   theta_cf,
            'theta_pw':   theta_pw,
            'theta_bw':   theta_bw,
            'theta_low':  theta_low,
            'theta_high': theta_high,
            'has_theta':  not np.isnan(theta_cf),
            'r_squared':  r['r_squared'],
            'error':      r['error'],
        })

    df = pd.DataFrame(rows)
    print(f"Total units: {len(df)}")
    print(f"Units with theta peak: {df['has_theta'].sum()} ({100 * df['has_theta'].mean():.1f}%)")
    print(f"Animals: {df['animal'].unique()}")
    return df


# ---- FOOOF fit-quality thresholds for flagging poor fits ----
R_SQUARED_MIN = 0.98   # flag files with r_squared below this
ERROR_MAX     = 0.4    # flag files with error above this


def export_low_quality_fits(df, out_path, r2_min=R_SQUARED_MIN, error_max=ERROR_MAX):
    """Write the list of files whose FOOOF fit has r_squared < r2_min OR
    error > error_max to a .txt file (one file path per line)."""
    flagged = df[(df['r_squared'] < r2_min) | (df['error'] > error_max)]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as fh:
        for _, row in flagged.iterrows():
            fh.write(f"{row['file']}\n")
    print(f"Flagged {len(flagged)}/{len(df)} files "
          f"(r_squared < {r2_min} or error > {error_max}) -> {out_path}")
    return flagged


# %% ==================== Property-plotting config + functions ===================

THETA_PROPS = {
    'theta_cf': {'xlabel': 'Centre Frequency (Hz)',   'xlim': (3.5, 6.5)},
    'theta_pw': {'xlabel': 'Power (a.u.)',            'xlim': (0,   1.0)},
    'theta_bw': {'xlabel': 'Peak Bandwidth (Hz)',     'xlim': (0,   3.5)},
    'exponent': {'xlabel': 'Aperiodic Exponent',      'xlim': (0,   5.0)},
    'offset':   {'xlabel': 'Aperiodic Offset',        'xlim': (-3,  3.0)},
    'r_squared':{'xlabel': 'R\u00b2',                 'xlim': (0,   1.0)},
    'error':    {'xlabel': 'FOOOF Error',             'xlim': (0,   0.5)},
}
THETA_SPECIFIC = {'theta_cf', 'theta_pw', 'theta_bw'}


def plot_theta_properties(df, props=None, save=True, save_dir=None):
    """Histograms of theta / aperiodic properties, coloured by animal."""
    if props is None:
        selected = list(THETA_PROPS.keys())
    elif isinstance(props, str):
        selected = [props]
    else:
        selected = list(props)

    unknown = [p for p in selected if p not in THETA_PROPS]
    if unknown:
        raise ValueError(f"Unknown property/ies: {unknown}. Choose from {list(THETA_PROPS.keys())}")

    df_theta = df[df['has_theta']]

    n_plots = len(selected)
    n_cols  = min(n_plots, 3)
    n_rows  = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.5 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax, key in zip(axes_flat, selected):
        meta    = THETA_PROPS[key]
        data_df = df_theta if key in THETA_SPECIFIC else df

        if 'animal' in data_df.columns:
            animals_here = sorted(data_df['animal'].unique())
            for i, animal in enumerate(animals_here):
                vals = data_df.loc[data_df['animal'] == animal, key].dropna()
                ax.hist(vals, bins=20, range=meta['xlim'],
                        alpha=0.6, color=ANIMAL_COLORS[i % len(ANIMAL_COLORS)],
                        edgecolor='white', lw=0.5, label=str(animal))
            ax.legend(fontsize=7)
        else:
            vals = data_df[key].dropna()
            ax.hist(vals, bins=20, range=meta['xlim'],
                    color='#AAAAAA', edgecolor='#555555', lw=0.6)
            ax.axvline(vals.median(), color='steelblue', lw=1.5, ls='--',
                       label=f'median = {vals.median():.2f}')
            ax.legend(fontsize=7)

        n = data_df[key].notna().sum()
        ax.set_title(f'n = {n}', fontsize=8)
        ax.set_xlabel(meta['xlabel'])
        ax.set_xlim(meta['xlim'])
        ax.set_ylabel('No. of units')
        ax.spines[['top', 'right']].set_visible(False)

    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    fig.suptitle('Theta & aperiodic properties', fontsize=11)
    plt.tight_layout()

    if save:
        out_dir = save_dir or FIGURE_DIR
        os.makedirs(out_dir, exist_ok=True)
        tag = '_'.join(selected)
        for ext in ('png', 'svg'):
            fig.savefig(os.path.join(out_dir, f'theta_properties_{tag}.{ext}'),
                        bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close(fig)


def plot_fit_quality(df, group_col="animal", metrics="both", mode="box",
                     jitter=0.08, figsize=None, save=True, save_path=None,
                     show_points=True, point_color="k", point_size=3,
                     point_alpha=0.35, box_kwargs=None):
    """Plot FOOOF fit quality (R^2 / error) grouped by `group_col`."""
    if group_col not in df.columns:
        raise ValueError(f"group_col='{group_col}' not in df.columns")

    if isinstance(metrics, str):
        key = metrics.lower()
        if key in ("both", "all"):
            metrics_list = ["r_squared", "error"]
        elif key in ("r2", "r_squared", "rsquared"):
            metrics_list = ["r_squared"]
        elif key in ("err", "error"):
            metrics_list = ["error"]
        else:
            metrics_list = [metrics]
    else:
        metrics_list = list(metrics)

    for m in metrics_list:
        if m not in df.columns:
            raise ValueError(f"metric='{m}' not in df.columns")

    plot_df = df[[group_col, *metrics_list]].copy()
    order = sorted(plot_df[group_col].dropna().unique())

    try:
        palette_list = ANIMAL_COLORS
    except NameError:
        palette_list = sns.color_palette("colorblind", n_colors=len(order))
    color_map = {g: palette_list[i % len(palette_list)] for i, g in enumerate(order)}

    n_metrics = len(metrics_list)
    if figsize is None:
        figsize = (5 * n_metrics, 4)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, squeeze=False)
    axes = axes[0]
    box_kwargs = box_kwargs or {}

    for ax, metric in zip(axes, metrics_list):
        sub = plot_df[[group_col, metric]].dropna()
        if mode == "box":
            sns.boxplot(data=sub, x=group_col, y=metric, order=order,
                        palette=color_map, ax=ax, fliersize=0, **box_kwargs)
            if show_points:
                sns.stripplot(data=sub, x=group_col, y=metric, order=order,
                              ax=ax, color=point_color, size=point_size,
                              alpha=point_alpha, jitter=jitter)
        elif mode == "square":
            for xi, g in enumerate(order):
                vals = sub.loc[sub[group_col] == g, metric].to_numpy()
                if vals.size == 0:
                    continue
                if show_points:
                    xs = xi + np.random.uniform(-jitter, jitter, size=vals.size)
                    ax.scatter(xs, vals, s=12, alpha=0.35, color=color_map[g])
                mean = np.nanmean(vals)
                sem = np.nanstd(vals) / np.sqrt(np.sum(~np.isnan(vals)))
                ax.errorbar([xi], [mean], yerr=[sem], fmt="s",
                            color="black", mfc="white", mec="black",
                            ms=8, capsize=4, lw=1.5, zorder=5)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order)
        else:
            raise ValueError("mode must be 'box' or 'square'")

        for t in ax.get_xticklabels():
            t.set_rotation(45)
            t.set_horizontalalignment('right')
        ax.set_title(metric)
        ax.set_xlabel(group_col)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0.5, None)
    plt.tight_layout()

    if save:
        if save_path is None:
            save_path = os.path.join(FIGURE_DIR, 'fit_quality.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# %% ==================== Sample-fit + summary-figure helpers ====================

def get_sample_psd(results, animal=None, index=0):
    """Return (freqs, psd) for one file. Defaults to the first file of the
    first animal. Replaces the notebook's get_psd_from_store()."""
    if animal is None:
        animal = next(iter(results))
    freqs, _mean, _sem, _n, psds_norm, _files, _lfp_store = results[animal]
    return freqs, psds_norm[index]


AX_LABEL_FONTSIZE   = 10
TICK_LABEL_FONTSIZE = 10


def _fmt(ax):
    ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)


def plot_mean_psds_all_animals_on_ax(ax, freqs, master_psds_dict, animals_list,
                                     file_counts=None, xlim=(2, 20)):
    for i, animal in enumerate(animals_list):
        mean, sem = master_psds_dict[animal]
        color = ANIMAL_COLORS[i % len(ANIMAL_COLORS)]
        if file_counts is not None:
            label = f"{animal} ({file_counts.get(animal, '?')} files)"
        else:
            label = str(animal)
        ax.plot(freqs, mean, lw=1.7, color=color, label=label)
        ax.fill_between(freqs, mean - sem, mean + sem, color=color, alpha=0.25)
    ax.set_xlim(xlim)
    ax.set_xlabel("Frequency (Hz)", fontsize=AX_LABEL_FONTSIZE)
    ax.set_ylabel("Normalized PSD", fontsize=AX_LABEL_FONTSIZE)
    ax.set_title("A  Mean Power Spectra", y=1.03, fontsize=10, pad=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.axvspan(3, 7, color='gray', alpha=0.12, zorder=0)  # theta band highlight
    _fmt(ax)


def _style_fooof_fit_ax(ax, fm, xlim=(1, 20), title="Sample FOOOF fit",
                        theta_band=None):
    """Plot an already-fit FOOOF model (original spectrum, full model, aperiodic
    fit) onto `ax` with the shared color/label styling used across the script.

    Also extracts the theta peak (strongest peak within theta_band) from `fm`
    and shades/labels its [cf - bw/2, cf + bw/2] range on the axis.
    """
    fm.plot(ax=ax, add_legend=False)

    line_styles = [
        ("Original PSD", "#333333", "-",  1.6),
        ("Full Model",   "#1263E6", "--", 1.4),
        ("Aperiodic",    "#EA080C", "--", 1.4),
    ]
    for line, (label, color, ls, lw) in zip(ax.lines, line_styles):
        line.set_color(color)
        line.set_label(label)
        line.set_linestyle(ls)
        line.set_linewidth(lw)
        line.set_alpha(0.9)

    ax.set_xlim(xlim)
    ax.text(0.5, 1.11, title, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=10)
    ax.text(0.5, 1.01, f"R\u00b2={fm.r_squared_:.3f}, error={fm.error_:.3f}",
            transform=ax.transAxes, ha='center', va='bottom', fontsize=8)

    theta_cf, _theta_pw, theta_bw = extract_theta_peak(fm.peak_params_, theta_band)
    theta_low, theta_high = theta_range_from_peak(theta_cf, theta_bw)
    if not np.isnan(theta_low):
        ax.axvspan(theta_low, theta_high, color='green', alpha=0.15, zorder=0)
        ax.text(0.5, 0.99, f"Theta range: {theta_low:.2f}-{theta_high:.2f} Hz",
                transform=ax.transAxes, ha='center', va='top', fontsize=7.5,
                color='#1a7a1a')

    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel("Frequency (Hz)", fontsize=AX_LABEL_FONTSIZE)
    ax.set_ylabel("Power", fontsize=AX_LABEL_FONTSIZE)
    ax.grid(False)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    _fmt(ax)


def _style_fbosc_fit_ax(ax, freqs, fbosc_static, xlim=(1, 20),
                        title="fBOSC background fit", theta_band=None):
    """Plot fBOSC's own per-file background/aperiodic fit -- the FOOOF fit
    `fBOSC_getThresholds()` (fBOSCUtils.py) runs internally on the file's mean
    wavelet power spectrum to set its statistical power threshold -- onto `ax`
    with the same line styling as `_style_fooof_fit_ax`.

    Unlike `_style_fooof_fit_ax`, there is no live FOOOF model object here --
    `fbosc_static` is the `fBOSC["static"]` dict returned via
    `fBOSCpy_wrapper_v2(..., return_diagnostics=True)`, whose 'bg_log10_pow' /
    'fooofed_spectrum' / 'ap_fit' entries are already the log10-power arrays
    a FOOOF model's .plot() would draw, so they're plotted directly. Also
    draws fBOSC's power threshold (the 95th-percentile line episodes must
    cross to be detected) -- not part of a plain FOOOF plot, but the reason
    this fit matters for fBOSC specifically.
    """
    freqs = np.asarray(freqs)
    line_styles = [
        ('bg_log10_pow',      "Original PSD", "#333333", "-",  1.6),
        ('fooofed_spectrum',  "Full Model",   "#1263E6", "--", 1.4),
        ('ap_fit',            "Aperiodic",    "#EA080C", "--", 1.4),
    ]
    for key, label, color, ls, lw in line_styles:
        ax.plot(freqs, np.asarray(fbosc_static[key]), color=color, ls=ls,
                lw=lw, alpha=0.9, label=label)
    ax.plot(freqs, np.log10(np.asarray(fbosc_static['pt'])),
            color='#555555', ls=':', lw=1.3, alpha=0.9,
            label='Power threshold (95th pct)')

    ax.set_xlim(xlim)
    ax.text(0.5, 1.11, title, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=10)
    ax.text(0.5, 1.01, f"R²={fbosc_static['r_squared']:.3f}, "
            f"error={fbosc_static['error']:.3f}",
            transform=ax.transAxes, ha='center', va='bottom', fontsize=8)

    peak_params = np.asarray(fbosc_static.get('peak_params', []))
    theta_cf, _theta_pw, theta_bw = extract_theta_peak(peak_params, theta_band)
    theta_low, theta_high = theta_range_from_peak(theta_cf, theta_bw)
    if not np.isnan(theta_low):
        ax.axvspan(theta_low, theta_high, color='green', alpha=0.15, zorder=0)
        ax.text(0.5, 0.99, f"Theta range: {theta_low:.2f}-{theta_high:.2f} Hz",
                transform=ax.transAxes, ha='center', va='top', fontsize=7.5,
                color='#1a7a1a')

    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel("Frequency (Hz)", fontsize=AX_LABEL_FONTSIZE)
    ax.set_ylabel("log(Power)", fontsize=AX_LABEL_FONTSIZE)
    ax.grid(False)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    _fmt(ax)


def plot_sample_psd_and_fooof_on_ax(ax, freqs, psd, fooof_kwargs=None,
                                    freq_range=(1, 20), xlim=(1, 20)):
    if fooof_kwargs is None:
        fooof_kwargs = dict(**FOOOF_SETTINGS, verbose=False)
    fm = FOOOF(**fooof_kwargs)
    fm.fit(freqs, psd, list(freq_range))
    _style_fooof_fit_ax(ax, fm, xlim=xlim, title="B  Sample FOOOF fit")
    return fm


def plot_theta_prop_hist_on_ax(ax, df, prop, xlim=None, xlabel=None, title=None):
    df_theta = df[df["has_theta"]].copy()
    animals_here = sorted(df_theta["animal"].unique())
    palette = {a: ANIMAL_COLORS[i % len(ANIMAL_COLORS)]
               for i, a in enumerate(animals_here)}

    all_vals = df_theta[prop].dropna()
    if xlim is not None:
        all_vals = all_vals.clip(*xlim)
    if all_vals.empty:
        return
    bins = np.linspace(all_vals.min(), all_vals.max(), 21)
    bin_width = bins[1] - bins[0]

    for a in animals_here:
        vals = df_theta.loc[df_theta["animal"] == a, prop].dropna()
        counts, _ = np.histogram(vals, bins=bins)
        if counts.sum() == 0:
            continue
        proportion = counts / counts.sum()
        ax.bar(bins[:-1], proportion, width=bin_width,
               alpha=0.55, color=palette[a], edgecolor="white",
               linewidth=0.4, align="edge", label=a)

    ax.set_ylabel("Proportion", fontsize=AX_LABEL_FONTSIZE)
    ax.set_title(title or prop, fontsize=10, pad=8)
    ax.set_xlabel(xlabel or prop, fontsize=AX_LABEL_FONTSIZE)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.spines[['top', 'right']].set_visible(False)
    _fmt(ax)


def plot_master_summary(results, master_psds_dict, expanded_fooof_df,
                        file_counts, freqs, save=True):
    """Composite summary figure (adapted from notebook cell 61)."""
    fig = plt.figure(figsize=(12, 5.5))
    gs = gridspec.GridSpec(2, 4, figure=fig,
                           width_ratios=[2.8, 1.4, 0.72, 0.72],
                           height_ratios=[1, 1], hspace=0.65, wspace=0.55,
                           top=0.87, bottom=0.11, left=0.07, right=0.97)
    ax_A  = fig.add_subplot(gs[0:2, 0])
    ax_B  = fig.add_subplot(gs[0, 1])
    ax_C1 = fig.add_subplot(gs[0, 2])
    ax_C2 = fig.add_subplot(gs[0, 3])
    ax_D  = fig.add_subplot(gs[1, 1])
    ax_E  = fig.add_subplot(gs[1, 2])
    ax_F  = fig.add_subplot(gs[1, 3])
    axes = dict(A=ax_A, B=ax_B, C1=ax_C1, C2=ax_C2, D=ax_D, E=ax_E, F=ax_F)

    fig.suptitle("Characterising LFP Power Spectra using FOOOF", fontsize=12, y=0.97)

    # A: mean PSDs
    plot_mean_psds_all_animals_on_ax(axes["A"], freqs, master_psds_dict,
                                     list(master_psds_dict.keys()),
                                     file_counts=file_counts)

    # B: sample FOOOF fit (first file of first animal)
    s_freqs, s_psd = get_sample_psd(results)
    plot_sample_psd_and_fooof_on_ax(axes["B"], freqs=s_freqs, psd=s_psd,
                                    freq_range=tuple(FOOOF_RANGE), xlim=(1, 20))

    # C1 / C2: fit quality
    def _quality_ax(ax, col, ylim, title):
        order_animals = sorted(expanded_fooof_df["animal"].unique())
        palette = {a: ANIMAL_COLORS[i % len(ANIMAL_COLORS)]
                   for i, a in enumerate(order_animals)}
        long_df = expanded_fooof_df[["animal", col]].dropna()
        sns.boxplot(data=long_df, x="animal", y=col, ax=ax,
                    palette=palette, fliersize=0, linewidth=0.8)
        sns.stripplot(data=long_df, x="animal", y=col, ax=ax,
                      palette=palette, size=2.5, alpha=0.35, jitter=True)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=AX_LABEL_FONTSIZE, pad=4)
        ax.set_xlabel("")
        ax.set_ylabel(title, fontsize=AX_LABEL_FONTSIZE)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', rotation=30, labelsize=TICK_LABEL_FONTSIZE)
        ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)

    _quality_ax(axes["C1"], "r_squared", (0.9, 1.0), "R\u00b2")
    _quality_ax(axes["C2"], "error", (0, 0.1), "Error")

    # D / E / F: theta histograms
    plot_theta_prop_hist_on_ax(axes["D"], expanded_fooof_df, prop="theta_cf",
                               xlim=THETA_PROPS["theta_cf"]["xlim"],
                               xlabel=THETA_PROPS["theta_cf"]["xlabel"],
                               title="D  Centre frequency")
    plot_theta_prop_hist_on_ax(axes["F"], expanded_fooof_df, prop="theta_pw",
                               xlim=THETA_PROPS["theta_pw"]["xlim"],
                               xlabel=THETA_PROPS["theta_pw"]["xlabel"],
                               title="E  Power")
    plot_theta_prop_hist_on_ax(axes["E"], expanded_fooof_df, prop="theta_bw",
                               xlim=THETA_PROPS["theta_bw"]["xlim"],
                               xlabel=THETA_PROPS["theta_bw"]["xlabel"],
                               title="F  Bandwidth")
    axes["D"].legend(frameon=False, fontsize=7, title="animal",
                     title_fontsize=7, loc="upper right")

    if save:
        os.makedirs(FIGURE_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURE_DIR, "lfp_psd_fooof_summary.png"),
                    dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(FIGURE_DIR, "lfp_psd_fooof_summary.svg"),
                    bbox_inches="tight")
    plt.show()
    return fig, axes


# %% ==================== MAIN PIPELINE (PSD -> FOOOF -> plots) ==================

def _ask_yes_no(question):
    """Prompt the user with a yes/no question; returns True for yes."""
    try:
        resp = input(f"{question} (y/n): ").strip().lower()
    except EOFError:
        return False
    return resp in ("y", "yes")


os.makedirs(INPUT_PKL_DIR, exist_ok=True)

# ---------- CHECKPOINT: FOOOF analysis ----------
_run_fooof = _ask_yes_no("Would you like to perform the FOOOF analysis?")

if _run_fooof:
    # 1) Generate PSDs for every animal via the reference folder-walk pipeline.
    results = {}
    for label, folder in ANIMALS.items():
        print(f"=== Processing {label} ===")
        results[label] = process_animal(label, folder)

    # Cache the frequency vector and per-animal summaries the notebook code expects.
    freqs_store      = results[next(iter(results))][0]
    master_psds_dict = {a: (r[1], r[2]) for a, r in results.items()}   # {animal: (mean, sem)}
    file_counts      = {a: r[3] for a, r in results.items()}           # {animal: n_files}

    # Persist the processed PSDs -- 'lfp_store' (per-file cleaned LFP trace + fs)
    # is what PART 2's fBOSC section reads below, so fBOSC runs entirely off this
    # script's own output rather than a separate run_fBOSC_batch.py pass.
    with open(PROCESSED_PSDS_PKL, 'wb') as fh:
        pickle.dump({a: {'freqs': r[0], 'mean': r[1], 'sem': r[2],
                         'psds': r[4], 'files': r[5], 'lfp_store': r[6]}
                    for a, r in results.items()}, fh)
    print(f"Saved processed PSDs -> {PROCESSED_PSDS_PKL}")

    # 2) Multi-animal mean +/- SEM PSD plot.
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_mean_psds_all_animals_on_ax(ax, freqs_store, master_psds_dict, animals,
                                     file_counts=file_counts)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURE_DIR, "mean_psds_all_animals.svg"),
                dpi=300, bbox_inches="tight")
    plt.show()

    # 3) FOOOF summary on each animal's averaged PSD (quick report).
    for animal, (freqs, mean, sem, n, psds_norm, files, _lfp_store) in results.items():
        fm = FOOOF(**FOOOF_SETTINGS)
        fm.fit(freqs, mean, FOOOF_RANGE)
        print(f"[{animal}] mean-PSD FOOOF: R\u00b2={fm.r_squared_:.3f}, "
              f"error={fm.error_:.3f}, aperiodic={fm.aperiodic_params_}")

    # 4) FOOOF on every individual PSD -> df_fooof (flat, one row per file).
    #    Also saves a model-fit figure (original spectrum, full model, aperiodic
    #    fit) for every file under FIGURE_DIR/individual_fits/<animal>/ (under OUTPUT_DIR).
    fooof_results = build_fooof_results(
        results, save_fits=True,
        save_dir=os.path.join(FIGURE_DIR, 'individual_fits'))
    df_fooof = pd.DataFrame(fooof_results)

    # 5) Expand into per-property dataframe and make the property/quality plots.
    expanded_fooof_df = fooof_results_to_df(fooof_results, theta_band=THETA_BAND)

    # Persist the full FOOOF fit output (previously only the pre-FOOOF PSDs were
    # pickled -- aperiodic_params, peak_params, r_squared, error, and the derived
    # per-property table never made it to disk).
    with open(FOOOF_RESULTS_PKL, 'wb') as fh:
        pickle.dump({'fooof_results': fooof_results,
                    'expanded_fooof_df': expanded_fooof_df}, fh)
    print(f"Saved FOOOF fit results -> {FOOOF_RESULTS_PKL}")

    # Flag/export files with a poor FOOOF fit (r_squared < 0.98 or error > 0.4).
    export_low_quality_fits(expanded_fooof_df, LOW_QUALITY_FITS_TXT)

    plot_theta_properties(expanded_fooof_df, props=list(THETA_SPECIFIC), save=True)
    plot_fit_quality(expanded_fooof_df, group_col="animal", metrics="both",
                     mode="box", save=True)

    # 6) Sample fit + composite summary figure.
    fig, ax = plt.subplots(figsize=(5, 3))
    s_freqs, s_psd = get_sample_psd(results)
    plot_sample_psd_and_fooof_on_ax(ax, freqs=s_freqs, psd=s_psd,
                                    freq_range=tuple(FOOOF_RANGE), xlim=(1, 20))
    fig.savefig(os.path.join(FIGURE_DIR, "sample_fooof.svg"),
                dpi=300, bbox_inches="tight")
    plt.show()

    plot_master_summary(results, master_psds_dict, expanded_fooof_df,
                        file_counts, freqs_store, save=True)

else:
    print("Skipping FOOOF analysis -- loading cached pkls from INPUT_PKL_DIR "
          f"({INPUT_PKL_DIR}) instead, recreating only whatever is missing.")

    # ---- processed_psds.pkl: recreate only if missing from INPUT_PKL_DIR ----
    if os.path.exists(PROCESSED_PSDS_PKL):
        print(f"Loading cached processed PSDs -> {PROCESSED_PSDS_PKL}")
        with open(PROCESSED_PSDS_PKL, 'rb') as fh:
            _processed_psds = pickle.load(fh)
        results = {a: (d['freqs'], d['mean'], d['sem'], len(d['files']), d['psds'], d['files'], d['lfp_store'])
                   for a, d in _processed_psds.items()}
    else:
        print(f"No cached processed PSDs found at {PROCESSED_PSDS_PKL} -- recreating.")
        results = {}
        for label, folder in ANIMALS.items():
            print(f"=== Processing {label} ===")
            results[label] = process_animal(label, folder)
        with open(PROCESSED_PSDS_PKL, 'wb') as fh:
            pickle.dump({a: {'freqs': r[0], 'mean': r[1], 'sem': r[2],
                             'psds': r[4], 'files': r[5], 'lfp_store': r[6]}
                        for a, r in results.items()}, fh)
        print(f"Saved processed PSDs -> {PROCESSED_PSDS_PKL}")

    freqs_store      = results[next(iter(results))][0]
    master_psds_dict = {a: (r[1], r[2]) for a, r in results.items()}
    file_counts      = {a: r[3] for a, r in results.items()}

    # ---- fooof_results.pkl: recreate only if missing from INPUT_PKL_DIR ----
    if os.path.exists(FOOOF_RESULTS_PKL):
        print(f"Loading cached FOOOF results -> {FOOOF_RESULTS_PKL}")
        with open(FOOOF_RESULTS_PKL, 'rb') as fh:
            _fooof_cache = pickle.load(fh)
        fooof_results     = _fooof_cache['fooof_results']
        expanded_fooof_df = _fooof_cache['expanded_fooof_df']
    else:
        print(f"No cached FOOOF results found at {FOOOF_RESULTS_PKL} -- recreating.")
        fooof_results = build_fooof_results(
            results, save_fits=True,
            save_dir=os.path.join(FIGURE_DIR, 'individual_fits'))
        expanded_fooof_df = fooof_results_to_df(fooof_results, theta_band=THETA_BAND)
        with open(FOOOF_RESULTS_PKL, 'wb') as fh:
            pickle.dump({'fooof_results': fooof_results,
                        'expanded_fooof_df': expanded_fooof_df}, fh)
        print(f"Saved FOOOF fit results -> {FOOOF_RESULTS_PKL}")

    df_fooof = pd.DataFrame(fooof_results)


# %% ==================== LFP signal / theta-episode visualization ==============
# Debugging aid for fBOSC episode detection: plots the raw LFP trace with
# significant theta episodes overlaid in red (everything else black), so you
# can SEE where/why theta is or isn't being called, instead of only reading
# summary statistics. Works on any (lfp, fs, episodes_df) combination -- not
# tied to the ANIMALS pipeline above -- so it can also be called standalone,
# e.g. in an interactive session, on a single file's trace + episode table.
#
# episodes_df / candidate_df are expected to have 'Onset'/'Offset' columns in
# the same units as fBOSC's raw output (MICROSECONDS -- see time_units below;
# the pipeline above passes `timestamps_us` into fBOSCpy_wrapper_v2, so
# fBOSC_episodesTable's Onset/Offset are microseconds until PART 2's speed
# section divides them by 1e6). Pass time_units='s' if you've already
# converted, or time_units='samples' if you're working with raw sample indices.

def _theta_time_mask(n_samples, fs, episodes_df, theta_band=None,
                     onset_col='Onset', offset_col='Offset',
                     freq_col='FrequencyMean', time_units='us'):
    """Boolean mask (len n_samples), True where the sample falls inside an
    episode of `episodes_df` whose FrequencyMean is within theta_band.

    `episodes_df` is expected to already be fBOSC's OUTPUT (i.e. only
    episodes that already passed whatever statistical/duration thresholds
    that particular table represents) -- this only adds the theta-band
    frequency filter and rasterizes [Onset, Offset) onto the sample grid.
    """
    mask = np.zeros(n_samples, dtype=bool)
    if episodes_df is None or len(episodes_df) == 0:
        return mask
    df = episodes_df
    if theta_band is not None and freq_col in df.columns:
        df = df[(df[freq_col] >= theta_band[0]) & (df[freq_col] <= theta_band[1])]
    for onset, offset in zip(df[onset_col].to_numpy(), df[offset_col].to_numpy()):
        if time_units == 'us':
            i0, i1 = onset / 1e6 * fs, offset / 1e6 * fs
        elif time_units == 's':
            i0, i1 = onset * fs, offset * fs
        elif time_units == 'samples':
            i0, i1 = onset, offset
        else:
            raise ValueError("time_units must be 'us', 's', or 'samples'")
        i0 = max(int(round(i0)), 0)
        i1 = min(int(round(i1)), n_samples)
        if i1 > i0:
            mask[i0:i1] = True
    return mask


def _category_mask(n_samples, fs, episodes_df, candidate_df=None, theta_band=None,
                   time_units='us'):
    """Per-sample category: 2 = significant theta (episodes_df), 1 = candidate
    only (in candidate_df but not episodes_df -- e.g. detected pre-postproc
    but shrunk/dropped by FWHM refinement or shoulder removal), 0 = neither
    (never crossed the power threshold in the first place)."""
    sig_mask = _theta_time_mask(n_samples, fs, episodes_df, theta_band, time_units=time_units)
    cat = sig_mask.astype(np.int8) * 2
    if candidate_df is not None:
        cand_mask = _theta_time_mask(n_samples, fs, candidate_df, theta_band, time_units=time_units)
        cat[(cand_mask) & (~sig_mask)] = 1
    return cat


_CAT_COLORS = {0: '#222222', 1: '#FF7F0E', 2: '#E41A1C'}
_CAT_LABELS = {0: 'non-theta', 1: 'candidate (dropped)', 2: 'significant theta'}


def _plot_categorical_line(ax, t, x, cat, lw=0.8):
    """Plot (t, x) as line segments colored by integer category `cat`
    (see _CAT_COLORS), connecting adjacent segments so the line has no gaps."""
    change_idx = np.flatnonzero(np.diff(cat)) + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [len(cat)]))
    seen = set()
    for s, e in zip(starts, ends):
        s2 = max(s - 1, 0)
        c = int(cat[s])
        label = _CAT_LABELS[c] if c not in seen else None
        seen.add(c)
        ax.plot(t[s2:e], x[s2:e], color=_CAT_COLORS[c], lw=lw, label=label)


def plot_theta_overview(lfp, fs, episodes_df, candidate_df=None, theta_band=None,
                        bin_s=0.05, figsize=(16, 3), save_path=None, title=None,
                        time_units='us'):
    """Static bird's-eye-view figure of the ENTIRE trace: min/max envelope per
    `bin_s`-second bin, colored red where the bin overlaps a significant theta
    episode, orange if it only overlaps a `candidate_df` episode (pass fBOSC's
    PRE-postproc episode table here to see what gets dropped and where), and
    black otherwise. Use this to spot, at a glance, where in the session theta
    detections are landing (or missing) -- then zoom in with plot_theta_window
    / interactive_theta_viewer.
    """
    theta_band = theta_band or THETA_BAND
    lfp = np.asarray(lfp, dtype=np.float64)
    n = len(lfp)
    cat = _category_mask(n, fs, episodes_df, candidate_df, theta_band, time_units)

    bin_n = max(1, int(round(bin_s * fs)))
    n_bins = n // bin_n
    if n_bins == 0:
        raise ValueError('Trace shorter than one bin_s bin.')
    trimmed = lfp[:n_bins * bin_n].reshape(n_bins, bin_n)
    cat_trim = cat[:n_bins * bin_n].reshape(n_bins, bin_n)
    lo = trimmed.min(axis=1)
    hi = trimmed.max(axis=1)
    bin_cat = cat_trim.max(axis=1)  # priority: significant > candidate > none
    t = (np.arange(n_bins) * bin_n) / fs

    fig, ax = plt.subplots(figsize=figsize)
    for c in (0, 1, 2):
        sel = bin_cat == c
        if not sel.any():
            continue
        ax.vlines(t[sel], lo[sel], hi[sel], colors=_CAT_COLORS[c],
                  linewidth=1.0, label=_CAT_LABELS[c])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('LFP (µV)')
    ax.set_xlim(t[0], t[-1] + bin_s)
    pct_theta = 100 * (bin_cat == 2).mean()
    ax.set_title(title or f'Full-signal theta overview  ({pct_theta:.1f}% significant theta)')
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved -> {save_path}')
    else:
        plt.show()
    return fig, ax


def plot_theta_window(lfp, fs, episodes_df, t_start, t_end, candidate_df=None,
                      theta_band=None, figsize=(14, 3.5), save_path=None,
                      title=None, time_units='us'):
    """Full-resolution figure of the trace between t_start/t_end (seconds),
    with significant theta episodes in red, candidate-only (dropped) episodes
    in orange (if candidate_df given), and everything else in black."""
    theta_band = theta_band or THETA_BAND
    lfp = np.asarray(lfp, dtype=np.float64)
    n = len(lfp)
    i0 = max(0, int(round(t_start * fs)))
    i1 = min(n, int(round(t_end * fs)))
    if i1 <= i0:
        raise ValueError(f'Empty window: t_start={t_start}, t_end={t_end}')

    cat_full = _category_mask(n, fs, episodes_df, candidate_df, theta_band, time_units)
    t = np.arange(i0, i1) / fs
    x = lfp[i0:i1]
    cat = cat_full[i0:i1]

    fig, ax = plt.subplots(figsize=figsize)
    _plot_categorical_line(ax, t, x, cat)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('LFP (µV)')
    ax.set_title(title or f'{t_start:.1f}-{t_end:.1f} s')
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved -> {save_path}')
    else:
        plt.show()
    return fig, ax


def interactive_theta_viewer(lfp, fs, episodes_df, candidate_df=None, theta_band=None,
                             window_s=10.0, title=None, time_units='us'):
    """Scroll through the whole trace in a fixed-width window via a slider
    (plus <</>> buttons and left/right arrow keys to step by 1/4-window), with
    significant theta episodes in red, candidate-only/dropped episodes in
    orange (if candidate_df given), and everything else in black.

    Requires an interactive matplotlib backend (e.g. `%matplotlib qt` in a
    notebook, or plain `python script.py` with a Tk/Qt backend installed) --
    with a static/inline backend the window will just show the first frame.
    """
    from matplotlib.widgets import Slider, Button

    theta_band = theta_band or THETA_BAND
    lfp = np.asarray(lfp, dtype=np.float64)
    n = len(lfp)
    dur = n / fs
    cat_full = _category_mask(n, fs, episodes_df, candidate_df, theta_band, time_units)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    plt.subplots_adjust(bottom=0.25)
    state = {'t0': 0.0}

    def draw(t0):
        t0 = float(np.clip(t0, 0, max(0.0, dur - window_s)))
        state['t0'] = t0
        i0 = int(round(t0 * fs))
        i1 = min(n, int(round((t0 + window_s) * fs)))
        t = np.arange(i0, i1) / fs
        x = lfp[i0:i1]
        cat = cat_full[i0:i1]
        ax.clear()
        _plot_categorical_line(ax, t, x, cat)
        ax.set_xlim(t0, t0 + window_s)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('LFP (µV)')
        ax.set_title(title or 'Theta episodes (red = significant, orange = dropped candidate)',
                    fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc='upper right')
        ax.spines[['top', 'right']].set_visible(False)
        fig.canvas.draw_idle()

    ax_slider = fig.add_axes([0.15, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Start (s)', 0, max(0.001, dur - window_s), valinit=0)
    slider.on_changed(draw)

    ax_prev = fig.add_axes([0.78, 0.09, 0.05, 0.05])
    ax_next = fig.add_axes([0.85, 0.09, 0.05, 0.05])
    b_prev = Button(ax_prev, '<<')
    b_next = Button(ax_next, '>>')

    def step(delta):
        slider.set_val(float(np.clip(state['t0'] + delta, 0, max(0.0, dur - window_s))))

    b_prev.on_clicked(lambda evt: step(-window_s))
    b_next.on_clicked(lambda evt: step(window_s))

    def on_key(event):
        if event.key == 'right':
            step(window_s * 0.25)
        elif event.key == 'left':
            step(-window_s * 0.25)
    fig.canvas.mpl_connect('key_press_event', on_key)

    draw(0)
    # Block here (bypassing the module-wide non-blocking plt.show() patch, see
    # top of file) so the window -- and its slider/buttons -- stays open for
    # interactive exploration until the user closes it by hand.
    _plt_show(block=True)
    return fig, ax, slider


# %% ####################################################################
# %% PART 2 — fBOSC / IEI / speed analyses (gated by y/n checkpoints)
# %% --------------------------------------------------------------------
# %% NOTE: Carried over from the original notebook, rewired to run off this
# %% pipeline's own outputs instead of the original FileManager-based
# %% objects: fBOSC episodes come from PART 1's PROCESSED_PSDS_PKL (via
# %% filtered_episodes, built from fBOSC_episodesTable), and the speed
# %% section's tracking-file lookup (build_animal_session_index(), cell 102)
# %% walks ANIMALS[...] + parse_metadata_from_path() instead of calling
# %% file_manager.get_files()/list_indexed_metadata(). day_arena_map is
# %% still optional (left as None) everywhere it's used. Answer "n" at any
# %% checkpoint below to skip that section.
# %% ####################################################################


# _ask_yes_no is defined earlier, before the MAIN PIPELINE / FOOOF checkpoint.

# ====================================================================
# fBOSC analysis
# (notebook cell [66])
# ====================================================================

# ---------- CHECKPOINT: fBOSC analysis ----------
_run_fbosc = _ask_yes_no("Would you like to perform the fBOSC analysis?")
if not _run_fbosc:
    print("Skipping fBOSC analysis section.")
else:
    # fBOSC runs IN-LINE here, directly on the cleaned per-file LFP traces
    # PART 1 saved into PROCESSED_PSDS_PKL (process_animal()'s lfp_store) --
    # no separate run_fBOSC_batch.py pass or on-disk group_summary.csv needed.

    # ================================================================
    # Run fBOSC on every file's cleaned LFP trace + build episodesTable
    # (notebook cell [67]/[68], rewired to this script's own PSD pipeline)
    # ================================================================

    # ---- cell [68] ----
    if not os.path.exists(PROCESSED_PSDS_PKL):
        raise FileNotFoundError(
            f"No processed PSDs found: {PROCESSED_PSDS_PKL} does not exist.\n"
            "Run PART 1 of this script (the FOOOF/PSD pipeline above) first "
            "-- it saves the cleaned LFP traces fBOSC needs.")

    if FBOSC_USE_CACHE and os.path.exists(FBOSC_EPISODES_PKL):
        print(f"Loading cached fBOSC episode table -> {FBOSC_EPISODES_PKL}")
        fBOSC_episodesTable = pd.read_pickle(FBOSC_EPISODES_PKL)
    else:
        with open(PROCESSED_PSDS_PKL, 'rb') as fh:
            processed_psds = pickle.load(fh)

        episode_tables = []
        for animal, animal_data in processed_psds.items():
            lfp_store = animal_data.get('lfp_store', {})
            print(f"=== fBOSC: {animal} ({len(lfp_store)} files) ===")
            for rel, file_data in lfp_store.items():
                lfp     = file_data['lfp']
                fs_file = file_data['fs']
                try:
                    # fBOSC's Onset/Offset columns are copied straight from
                    # the timestamp vector passed in below; using
                    # microseconds here matches what calculate_total_theta_time()
                    # (cell 72, unchanged from the original notebook) expects.
                    timestamps_us = np.arange(len(lfp), dtype=np.float64) / fs_file * 1e6
                    meta = parse_metadata_from_path(rel)

                    _ep_raw, ep_post, fbosc_diag = fBOSCpy_wrapper_v2(
                        animal, meta['date'], meta['session'], meta['tetrode'], meta['channel'],
                        lfp, timestamps_us,
                        F_array=FBOSC_F_ARRAY, Fs=fs_file,
                        postproc=FBOSC_POSTPROC,
                        min_ncycles=FBOSC_MIN_NCYCLES, min_duration_s=FBOSC_MIN_DURATION_S,
                        plot_histogram=False,
                        results_root=None, return_diagnostics=True)

                    # Per-file fBOSC background/aperiodic-fit figure (the fit
                    # fBOSC_getThresholds ran internally to set this file's power
                    # threshold) -- saved regardless of whether any episodes were
                    # detected below, since it's a QC plot of the fit itself.
                    if FBOSC_SAVE_FIT_FIGS:
                        animal_fit_dir = os.path.join(FBOSC_FIT_DIR, animal)
                        os.makedirs(animal_fit_dir, exist_ok=True)
                        fig, ax = plt.subplots(figsize=(5, 3))
                        _style_fbosc_fit_ax(ax, fbosc_diag['cfg_fBOSC']['F'],
                                            fbosc_diag['fBOSC']['static'],
                                            title=f"{animal}: {os.path.basename(rel)}")
                        stem = re.sub(r'[\\/]+', '_', os.path.splitext(rel)[0])
                        fig.savefig(os.path.join(animal_fit_dir, f'{stem}_fbosc_fit.png'),
                                   dpi=200, bbox_inches='tight')
                        plt.close(fig)

                    episode_table = pd.DataFrame.from_dict(ep_post)
                    if episode_table.empty:
                        print(f"  SKIP: {rel} -- 0 episodes detected")
                        continue

                    session_duration_sec = len(lfp) / fs_file
                    episode_table['animal']    = animal
                    episode_table['day']       = meta['date']
                    episode_table['session']   = meta['session']
                    episode_table['tetrode']   = meta['tetrode']
                    episode_table['channel']   = meta['channel']
                    episode_table['session_duration_sec'] = session_duration_sec
                    episode_table['proportion_time'] = episode_table['DurationS'] / session_duration_sec
                    episode_table['FrequencyMean_bin'] = (np.floor(episode_table['FrequencyMean'] / 1) * 1)
                    episode_tables.append(episode_table)
                    print(f"  OK: {rel}  [{len(episode_table)} episodes]")
                except Exception as e:
                    print(f"  SKIP: {rel} -- {e}")

        if not episode_tables:
            raise ValueError(
                "fBOSC detected zero episodes across every file in "
                f"{PROCESSED_PSDS_PKL}. Check FBOSC_F_ARRAY and the input LFP traces.")

        fBOSC_episodesTable = pd.concat(episode_tables, ignore_index=True)

        os.makedirs(FBOSC_OUTPUT_DIR, exist_ok=True)
        fBOSC_episodesTable.to_pickle(FBOSC_EPISODES_PKL)
        print(f"Saved fBOSC episode table -> {FBOSC_EPISODES_PKL}")

    # ---- cell [69] ----
    fBOSC_episodesTable = fBOSC_episodesTable.drop(columns=['RowID', 'ColID', 'Trial', 'Channel', 'SNR', 'SNRMean'], errors='ignore')

    #reorder the columns to have animal, day, session, tetrode, channel at the front
    cols = fBOSC_episodesTable.columns.tolist()
    new_order = ['animal', 'day', 'session', 'tetrode', 'channel'] + [col for col in cols if col not in ['animal', 'day', 'session', 'tetrode', 'channel']]

    fBOSC_episodesTable = fBOSC_episodesTable[new_order]
    fBOSC_episodesTable

    # ---- cell [70] ----
    os.makedirs(FBOSC_FIGURE_DIR, exist_ok=True)
    fig, ax = plt.subplots()
    ax.hist(fBOSC_episodesTable['FrequencyMean'], bins=40)
    fig.savefig(os.path.join(FBOSC_FIGURE_DIR, 'fbosc_frequency_histogram_all.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    # ---- cell [71] ----
    # Drop episodes whose FrequencyMean falls in the FBOSC_F_BUFFER_HZ buffer
    # zone above FBOSC_REGION_F_MAX -- that band was only scanned to give the
    # FOOOF background fit real curvature data past our reporting boundary; its
    # own top edge (FBOSC_F_ARRAY[-1]) still carries the fit's edge-undershoot
    # artifact. Cutting at FBOSC_REGION_F_MAX rather than FBOSC_F_ARRAY[-1] is
    # what keeps that artifact out of the reported episodes.
    filtered_episodes = fBOSC_episodesTable[fBOSC_episodesTable['FrequencyMean'] < FBOSC_REGION_F_MAX]
    fig, ax = plt.subplots()
    ax.hist(filtered_episodes['FrequencyMean'], bins=20)
    fig.savefig(os.path.join(FBOSC_FIGURE_DIR, 'fbosc_frequency_histogram_filtered_lt20hz.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    # ================================================================
    # Signal visualization: theta episodes overlaid on the raw LFP trace
    # (see plot_theta_overview / plot_theta_window / interactive_theta_viewer,
    # defined above PART 2) -- pick a file below (defaults to the first file
    # of the first animal; edit VIZ_ANIMAL / VIZ_REL to look at a different
    # one -- printed options list every session/tetrode/channel available).
    # ================================================================

    # ---------- CHECKPOINT: signal visualization ----------
    _run_theta_viz = _ask_yes_no(
        "Would you like to visualize the LFP signal with theta episodes highlighted?")
    if not _run_theta_viz:
        print("Skipping signal visualization section.")
    else:
        with open(PROCESSED_PSDS_PKL, 'rb') as fh:
            _viz_processed_psds = pickle.load(fh)

        VIZ_ANIMAL = animals[0]              # edit to pick a different animal
        _viz_lfp_store = _viz_processed_psds[VIZ_ANIMAL]['lfp_store']
        VIZ_REL = next(iter(_viz_lfp_store))  # edit to pick a different file
        print(f"Visualizing: {VIZ_ANIMAL} / {VIZ_REL}")
        print(f"  (other files for {VIZ_ANIMAL}: {list(_viz_lfp_store.keys())})")

        _viz_lfp  = _viz_lfp_store[VIZ_REL]['lfp']
        _viz_fs   = _viz_lfp_store[VIZ_REL]['fs']
        _viz_meta = parse_metadata_from_path(VIZ_REL)

        # Final, statistically-significant + duration-passed episodes for this
        # file (already computed above, cached in fBOSC_episodesTable).
        _viz_sig = fBOSC_episodesTable[
            (fBOSC_episodesTable['animal']  == VIZ_ANIMAL) &
            (fBOSC_episodesTable['day']     == _viz_meta['date']) &
            (fBOSC_episodesTable['session'] == _viz_meta['session']) &
            (fBOSC_episodesTable['tetrode'] == _viz_meta['tetrode']) &
            (fBOSC_episodesTable['channel'] == _viz_meta['channel'])
        ]

        # The red/black plot above needs nothing but what's already loaded
        # (PROCESSED_PSDS_PKL + fBOSC_episodesTable -- both come straight from
        # cache if FBOSC_USE_CACHE was used, no rerun). The ORANGE "dropped
        # candidate" overlay is the one exception: fBOSC's PRE-postproc
        # episodes (crossed the power threshold + initial duration check,
        # before FWHM onset/offset refinement + shoulder removal can
        # shrink/drop them) are never saved to disk -- only the final ep_post
        # table is cached -- so showing them means re-running fBOSC's wavelet
        # transform on this ONE file (seconds, not the full batch). Skip it
        # if you just want the fast red/black view of the cached results.
        _viz_show_candidate = _ask_yes_no(
            "Also show dropped/candidate episodes in orange? "
            "(reruns fBOSC on this one file; 'n' uses only the cached episode table, no rerun)")
        if _viz_show_candidate:
            _viz_timestamps_us = np.arange(len(_viz_lfp), dtype=np.float64) / _viz_fs * 1e6
            _viz_ep_raw, _viz_ep_post, _viz_diag = fBOSCpy_wrapper_v2(
                VIZ_ANIMAL, _viz_meta['date'], _viz_meta['session'],
                _viz_meta['tetrode'], _viz_meta['channel'],
                _viz_lfp, _viz_timestamps_us,
                F_array=FBOSC_F_ARRAY, Fs=_viz_fs, postproc=FBOSC_POSTPROC,
                min_ncycles=FBOSC_MIN_NCYCLES, min_duration_s=FBOSC_MIN_DURATION_S,
                plot_histogram=False, results_root=None, return_diagnostics=True)
            _viz_candidate = pd.DataFrame.from_dict(_viz_ep_raw)
        else:
            _viz_candidate = None

        plot_theta_overview(
            _viz_lfp, _viz_fs, _viz_sig, candidate_df=_viz_candidate,
            theta_band=THETA_BAND, title=f"{VIZ_ANIMAL}: {VIZ_REL}",
            save_path=os.path.join(FBOSC_FIGURE_DIR, 'theta_overview_example.png'))

        _run_interactive_viz = _ask_yes_no(
            "Open the interactive sliding-window viewer for this file? "
            "(needs an interactive matplotlib backend, e.g. %matplotlib qt)")
        if _run_interactive_viz:
            interactive_theta_viewer(
                _viz_lfp, _viz_fs, _viz_sig, candidate_df=_viz_candidate,
                theta_band=THETA_BAND, window_s=10.0,
                title=f"{VIZ_ANIMAL}: {VIZ_REL}")

    # ---- cell [72] ----
    def calculate_total_theta_time(df):
        """
        Given a dataframe with 'OnsetS' and 'OffsetS' columns for theta episodes,
        calculate the total time covered by these episodes, accounting for overlaps.
        """
        intervals = df[['Onset', 'Offset']].values
        intervals = intervals/1e6
        intervals = intervals[intervals[:, 1] > intervals[:, 0]]  # filter out invalid intervals

        if len(intervals) == 0:
            return 0.0

        # Sort intervals by onset time
        intervals = intervals[np.argsort(intervals[:, 0])]

        merged_intervals = []
        current_start, current_end = intervals[0]

        for start, end in intervals[1:]:
            if start <= current_end:  # overlap
                current_end = max(current_end, end)
            else:
                merged_intervals.append((current_start, current_end))
                current_start, current_end = start, end

        merged_intervals.append((current_start, current_end))  # add the last interval

        total_time = sum(end - start for start, end in merged_intervals)
        return total_time

    # ---- cell [73] ----
    # Now we can apply this function to calculate the total theta time per session
    theta_time_per_session = (
        filtered_episodes[
            (filtered_episodes['FrequencyMean'] >= THETA_BAND[0]) &
            (filtered_episodes['FrequencyMean'] <= THETA_BAND[1])
        ]
        .groupby(['animal', 'day', 'session', 'tetrode', 'channel'])
        .apply(calculate_total_theta_time)
        .reset_index(name='total_theta_time_sec') )

    #add total number of episodes (total and theta episodes) per session
    total_episodes_per_session = fBOSC_episodesTable.groupby(['animal', 'day', 'session', 'tetrode', 'channel'])['DurationS'].count().reset_index(name='total_episodes')
    theta_episodes_per_session = filtered_episodes[
        (filtered_episodes['FrequencyMean'] >= THETA_BAND[0]) &
        (filtered_episodes['FrequencyMean'] <= THETA_BAND[1])
    ].groupby(['animal', 'day', 'session', 'tetrode', 'channel'])['DurationS'].count().reset_index(name='theta_episodes')

    # Now we can calculate the proportion of time in theta episodes per session
    session_durations = fBOSC_episodesTable.groupby(['animal', 'day', 'session'])['session_duration_sec'].first().reset_index()
    theta_proportions = pd.merge(theta_time_per_session, session_durations, on=['animal', 'day', 'session'])
    theta_proportions['proportion_time'] = theta_proportions['total_theta_time_sec'] / theta_proportions['session_duration_sec']

    #create pepisode_df with columns: animal, day, session, tetrode, channel, total_episodes, theta_episodes, session_duration, total_theta_time_sec, proportion_time
    pepisode_df = pd.merge(theta_proportions, total_episodes_per_session, 
                            on=['animal', 'day', 'session', 'tetrode', 'channel'])
    pepisode_df = pd.merge(pepisode_df, theta_episodes_per_session, 
                            on=['animal', 'day', 'session', 'tetrode', 'channel'])
    pepisode_df = pepisode_df[['animal', 'day', 'session', 'tetrode', 'channel', 
                                'total_episodes', 'theta_episodes', 
                                'session_duration_sec', 'total_theta_time_sec', 
                                'proportion_time']]
    pepisode_df

    # ================================================================
    # different kind of plots
    # (notebook cell [74])
    # ================================================================

    # ---- cell [75] ----
    #box plot of proportion time
    # Pool all tetrode+channel proportion_time values across days/sessions and plot a single pooled boxplot
    df_pooled = theta_proportions.copy()
    df_pooled['pooled_group'] = 'all'  # single group for pooling
    pooled = df_pooled['proportion_time']

    print(f"Pooled N = {len(pooled)}, mean = {pooled.mean():.4f}, median = {pooled.median():.4f}, std = {pooled.std():.4f}")

    plt.figure(figsize=(3, 5))
    sns.boxplot(data=df_pooled, x='pooled_group', y='proportion_time', color=sns.color_palette("colorblind")[0])
    # overlay individual points for visibility
    sns.swarmplot(data=df_pooled, x='pooled_group', y='proportion_time', color='k', size=3, alpha=0.6)
    plt.xlabel('')
    plt.ylabel('Proportion of Time in Theta Episodes')
    plt.title('Proportion of Time in Theta Episodes (all days/sessions/tetrodes)')
    plt.ylim(0, df_pooled['proportion_time'].max() * 1.1)
    #plt.tight_layout()
    plt.gcf().savefig(os.path.join(FBOSC_FIGURE_DIR, 'fbosc_proportion_time_theta_pooled.png'),
                       bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

    # ---- cell [76] ----
    # Calculate fraction of episodes that are in the theta range (THETA_BAND) for each session,
    # then plot as a bar plot with error bars = SEM across sessions grouped by day.

    # Define theta range from the same THETA_BAND used by the FOOOF analysis (line ~169)
    theta_episodes = filtered_episodes[
        (filtered_episodes['FrequencyMean'] >= THETA_BAND[0]) &
        (filtered_episodes['FrequencyMean'] <= THETA_BAND[1])
    ]

    # Count total episodes and theta episodes per (day, session)
    total_counts = filtered_episodes.groupby(['day', 'session']).size().rename('total_episodes')
    theta_counts = theta_episodes.groupby(['day', 'session']).size().rename('theta_episodes')

    # Combine into a single DataFrame and compute fraction per session
    frac_df = pd.concat([total_counts, theta_counts], axis=1).fillna(0)
    frac_df['theta_fraction'] = frac_df['theta_episodes'] / frac_df['total_episodes']
    frac_df = frac_df.reset_index()

    # Aggregate by day: compute mean and SEM across sessions
    day_stats = frac_df.groupby('day')['theta_fraction'].agg(['mean', 'count', 'std']).rename(
        columns={'mean': 'mean_fraction', 'std': 'std_fraction'}
    )
    day_stats['sem'] = day_stats['std_fraction'] / np.sqrt(day_stats['count'])

    # Plot with matplotlib so we can pass explicit SEM as error bars (avoids seaborn's errorbar format issues)
    plt.figure(figsize=(10, 5))
    x = np.arange(len(day_stats))
    plt.bar(x, day_stats['mean_fraction'], yerr=day_stats['sem'], capsize=6, color=sns.color_palette("colorblind", len(day_stats)))
    plt.xticks(x, day_stats.index, rotation=45, ha='right')
    plt.xlabel('Day')
    plt.ylabel(f'Fraction of Theta Episodes ({THETA_BAND[0]:g}-{THETA_BAND[1]:g} Hz)')
    plt.title(f'Fraction of Oscillatory Episodes in Theta Range ({THETA_BAND[0]:g}-{THETA_BAND[1]:g} Hz) by Day\n(error bars = SEM across sessions)')
    plt.ylim(0, 1.05 * day_stats['mean_fraction'].max())
    plt.tight_layout()
    plt.gcf().savefig(os.path.join(FBOSC_FIGURE_DIR, 'fbosc_theta_fraction_by_day.png'),
                       bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

    # ---- cell [77] ----
    # first divide the episodes into bins based on their FrequencyMean, with bean width of 0.2 Hz, then calculate the fraction of episodes in each bin compared to the total number of episodes in that session, and plot as a scatter plot with error bars = SEM across all sessions.

    # Define frequency bins (0-20 Hz with 0.2 Hz width)
    bin_width = 0.8
    bins = np.arange(0, 20 + bin_width, bin_width)
    filtered_episodes['Frequency_bin'] = pd.cut(filtered_episodes['FrequencyMean'], bins=bins, right=False)
    # Count total episodes and episodes in each frequency bin per (day, session)
    total_counts = filtered_episodes.groupby(['day', 'session']).size().rename('total_episodes')
    # Call .size() to get counts (avoid leaving .size as the function object)
    bin_counts = filtered_episodes.groupby(['day', 'session', 'Frequency_bin']).size().rename('bin_episodes').reset_index()
    # Merge total counts to compute fraction per bin
    bin_counts = bin_counts.merge(total_counts.reset_index(), on=['day', 'session'])
    bin_counts['fraction'] = bin_counts['bin_episodes'] / bin_counts['total_episodes']
    # Aggregate by frequency bin: compute mean and SEM across sessions
    bin_stats = bin_counts.groupby('Frequency_bin')['fraction'].agg(['mean', 'count', 'std']).rename(
        columns={'mean': 'mean_fraction', 'std': 'std_fraction'}
    )
    bin_stats['sem'] = bin_stats['std_fraction'] / np.sqrt(bin_stats['count'])
    # Plot with matplotlib
    plt.figure(figsize=(12, 6))
    x = [interval.mid for interval in bin_stats.index]  # Get the mid-point of each bin interval
    plt.errorbar(x, bin_stats['mean_fraction'], yerr=bin_stats['sem'], fmt='o', ecolor='gray', capsize=4)
    plt.xlabel('Mean Frequency of Oscillatory Episode (Hz)')
    plt.ylabel('Fraction of Episodes in Bin')
    plt.title('Fraction of Oscillatory Episodes by Frequency Bin\n(error bars = SEM across sessions for each bin)')
    plt.xlim(0, 20)
    plt.ylim(0, bin_stats['mean_fraction'].max() * 1.1)
    plt.tight_layout()
    plt.gcf().savefig(os.path.join(FBOSC_FIGURE_DIR, 'fbosc_episode_fraction_by_frequency_bin.png'),
                       bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

    # ---- cell [78] ----
    # ...existing code...

    # Define frequency bins (0-20 Hz with 0.8 Hz width)
    bin_width = 1
    bins = np.arange(0, 20 + bin_width, bin_width)
    filtered_episodes['Frequency_bin'] = pd.cut(filtered_episodes['FrequencyMean'], bins=bins, right=False)

    # Calculate bin stats separately for each animal
    animals = filtered_episodes['animal'].unique()
    bin_stats_by_animal = {}

    for animal in animals:
        animal_df = filtered_episodes[filtered_episodes['animal'] == animal]
        # Count total episodes and episodes in each frequency bin per (day, session)
        total_counts = animal_df.groupby(['day', 'session']).size().rename('total_episodes')
        bin_counts = animal_df.groupby(['day', 'session', 'Frequency_bin']).size().rename('bin_episodes').reset_index()
        bin_counts = bin_counts.merge(total_counts.reset_index(), on=['day', 'session'])
        bin_counts['fraction'] = bin_counts['bin_episodes'] / bin_counts['total_episodes']
        # Aggregate by frequency bin: compute mean and SEM across sessions
        bin_stats = bin_counts.groupby('Frequency_bin')['fraction'].agg(['mean', 'count', 'std']).rename(
            columns={'mean': 'mean_fraction', 'std': 'std_fraction'}
        )
        bin_stats['sem'] = bin_stats['std_fraction'] / np.sqrt(bin_stats['count'])
        bin_stats_by_animal[animal] = bin_stats

    # Plot with matplotlib, coloring by animal
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("colorblind", len(animals))
    for i, animal in enumerate(animals):
        bin_stats = bin_stats_by_animal[animal]
        x = [interval.mid for interval in bin_stats.index]
        plt.errorbar(x, bin_stats['mean_fraction'], yerr=bin_stats['sem'], fmt='o-', 
                     ecolor=colors[i], color=colors[i], capsize=4, label=animal)

    plt.xlabel('Mean Frequency of Oscillatory Episode (Hz)')
    plt.ylabel('Fraction of Episodes in Bin')
    plt.title('Fraction of Oscillatory Episodes by Frequency Bin\n(error bars = SEM across sessions for each bin, per animal)')
    plt.xlim(0, 20)
    plt.ylim(0, max([b['mean_fraction'].max() for b in bin_stats_by_animal.values()]) * 1.1)
    plt.legend(title='Animal')
    plt.tight_layout()
    plt.gcf().savefig(os.path.join(FBOSC_FIGURE_DIR, 'fbosc_episode_fraction_by_frequency_bin_per_animal.png'),
                       bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    # ...existing code...

    # ================================================================
    # episodes properties plot
    # (notebook cell [79])
    # ================================================================

    # ---- cell [80] ----
    def _save_or_show(fig, save, save_dir, fname):
        if save:
            out_dir = save_dir or FBOSC_FIGURE_DIR
            os.makedirs(out_dir, exist_ok=True)
            for ext in ('png', 'svg'):
                fig.savefig(os.path.join(out_dir, f'{fname}.{ext}'),
                            bbox_inches='tight', dpi=300)
            print(f"Saved: {fname} -> {out_dir}")
        else:
            plt.show()
        plt.close(fig)

    # ---- cell [81] ----
    filtered_episodes['animal'].unique()

    # ---- cell [82] ----
    def plot_episode_properties(episodes_df, pepisode_df=None,
                                 day_arena_map=None, save=True, save_dir=None):

        props = [
            {'col': 'FrequencyMean', 'xlabel': 'Episode frequency (Hz)', 'xlim': (1,  20)},
            {'col': 'DurationC',     'xlabel': 'Duration (cycles)',       'xlim': (0,  30)},
            {'col': 'DurationS',     'xlabel': 'Duration (s)',            'xlim': (0,   5)},
            {'col': 'PowerMean',     'xlabel': 'Episode power (a.u.)',    'xlim': (0, None)},
        ]

        show_pepisode = pepisode_df is not None
        arena_colors  = {'1D_linear': '#4C72B0', '2D_open': '#DD8452'}
        animals       = sorted(episodes_df['animal'].unique())
        n_cols        = len(props) + (2 if show_pepisode else 0)

        # ── helper: get per-animal arena map ─────────────────────────────────────
        def _get_arena_map(animal):
            if day_arena_map is None:
                return None
            first_val = next(iter(day_arena_map.values()))
            if isinstance(first_val, dict):
                return day_arena_map.get(animal, None)
            return day_arena_map   # flat map, same for all animals

        # ── detect day column name (handle 'day' vs 'date') ──────────────────────
        def _day_col(df):
            for c in ['day', 'date']:
                if c in df.columns:
                    return c
            raise ValueError(f"No 'day' or 'date' column found. Columns: {df.columns.tolist()}")

        ep_day_col  = _day_col(episodes_df)
        pep_day_col = _day_col(pepisode_df) if show_pepisode else None

        # ── add arena_type column to both dataframes ──────────────────────────────
        episodes_df = episodes_df.copy()
        episodes_df['arena_type'] = ''

        if show_pepisode:
            pepisode_df = pepisode_df.copy()
            pepisode_df['arena_type'] = ''
            pepisode_df['pepisode_pct'] = pepisode_df['proportion_time'] * 100

        for animal in animals:
            amap = _get_arena_map(animal)
            if not amap:
                continue

            # ── map episodes_df ───────────────────────────────────────────────────
            ep_mask = episodes_df['animal'] == animal
            mapped  = episodes_df.loc[ep_mask, ep_day_col].map(amap)

            # if all NaN, try stripping whitespace from both keys and values
            if mapped.isna().all():
                amap_stripped = {k.strip(): v for k, v in amap.items()}
                mapped = episodes_df.loc[ep_mask, ep_day_col].str.strip().map(amap_stripped)

            # if mapped.isna().all():
            #     print(f"  Warning: no arena matches for animal '{animal}' in episodes_df.")
            #     print(f"    Day values:  {episodes_df.loc[ep_mask, ep_day_col].unique()[:3]}")
            #     print(f"    Map keys:    {list(amap.keys())[:3]}")
            else:
                episodes_df.loc[ep_mask, 'arena_type'] = mapped

            # ── map pepisode_df ───────────────────────────────────────────────────
            if show_pepisode:
                pep_mask   = pepisode_df['animal'] == animal
                pep_mapped = pepisode_df.loc[pep_mask, pep_day_col].map(amap)

                if pep_mapped.isna().all():
                    amap_stripped = {k.strip(): v for k, v in amap.items()}
                    pep_mapped = (pepisode_df.loc[pep_mask, pep_day_col]
                                  .str.strip().map(amap_stripped))

                # if pep_mapped.isna().all():
                #     print(f"  Warning: no arena matches for animal '{animal}' in pepisode_df.")
                #     print(f"    Day values:  {pepisode_df.loc[pep_mask, pep_day_col].unique()[:3]}")
                else:
                    pepisode_df.loc[pep_mask, 'arena_type'] = pep_mapped

        # ── print mapping summary before plotting ─────────────────────────────────
        # print("\nArena mapping summary:")
        # for animal in animals:
        #     ep_arenas  = episodes_df[episodes_df['animal'] == animal]['arena_type'].value_counts(dropna=False)
        #     print(f"  {animal} episodes_df:  {ep_arenas.to_dict()}")
        #     if show_pepisode:
        #         pep_arenas = pepisode_df[pepisode_df['animal'] == animal]['arena_type'].value_counts(dropna=False)
        #         print(f"  {animal} pepisode_df: {pep_arenas.to_dict()}")

        # ── plot ──────────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(len(animals), n_cols,
                                  figsize=(3.5 * n_cols, 3.2 * len(animals)),
                                  squeeze=False)

        for row, animal in enumerate(animals):
            adf            = episodes_df[episodes_df['animal'] == animal]
            arenas_present = sorted(adf['arena_type'].dropna().unique())
            use_color      = len(arenas_present) > 0

            # ── episode property histograms ───────────────────────────────────────
            for col, prop in enumerate(props):
                ax   = axes[row][col]
                xlim = prop['xlim']

                if use_color:
                    for arena in arenas_present:
                        color = arena_colors.get(arena, '#888888')
                        vals  = adf[adf['arena_type'] == arena][prop['col']].dropna()
                        if len(vals) == 0:
                            continue
                        rng = (xlim[0], xlim[1]) if xlim[1] else None
                        ax.hist(vals, bins=20, range=rng,
                                alpha=0.6, color=color, edgecolor='white',
                                lw=0.5, label=arena)
                    if row == 0 and col == 0:
                        ax.legend(fontsize=7)
                else:
                    vals = adf[prop['col']].dropna()
                    if len(vals):
                        rng = (xlim[0], xlim[1]) if xlim[1] else None
                        ax.hist(vals, bins=20, range=rng,
                                color='#AAAAAA', edgecolor='#555555', lw=0.6)
                        ax.axvline(vals.median(), color='steelblue', lw=1.5, linestyle='--')

                if xlim[1]:
                    ax.set_xlim(xlim)
                ax.spines[['top', 'right']].set_visible(False)
                if row == 0:
                    ax.set_title(prop['xlabel'], fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"{animal}\nn={len(adf)} episodes", fontsize=8)

            # ── p-episode histogram ───────────────────────────────────────────────
            if show_pepisode:
                pdf        = pepisode_df[pepisode_df['animal'] == animal]
                arenas_pep = sorted(pdf['arena_type'].dropna().unique())
                use_col_pep = len(arenas_pep) > 0

                ax = axes[row][-2]
                if use_col_pep:
                    for arena in arenas_pep:
                        color = arena_colors.get(arena, '#888888')
                        vals  = pdf[pdf['arena_type'] == arena]['pepisode_pct'].dropna()
                        if len(vals) == 0:
                            continue
                        ax.hist(vals, bins=15, range=(0, 100),
                                alpha=0.6, color=color, edgecolor='white',
                                lw=0.5, label=arena)
                    if row == 0:
                        ax.legend(fontsize=7)
                else:
                    vals = pdf['pepisode_pct'].dropna()
                    if len(vals):
                        ax.hist(vals, bins=15, range=(0, 100),
                                color='#AAAAAA', edgecolor='#555555', lw=0.6)
                        ax.axvline(vals.median(), color='steelblue', lw=1.5, linestyle='--')

                ax.set_xlim([0, 100])
                ax.set_xlabel('P-episode (%)', fontsize=8)
                ax.spines[['top', 'right']].set_visible(False)
                if row == 0:
                    ax.set_title('P-episode\n(% time in theta)', fontsize=9)

                # ── theta vs total episodes stacked bar ───────────────────────────
                ax       = axes[row][-1]
                sessions = (pdf.sort_values(['day', 'session'])
                               [['day', 'session', 'arena_type']]
                               .drop_duplicates())

                for si, (_, s_row) in enumerate(sessions.iterrows()):
                    sess_df = pdf[(pdf[pep_day_col] == s_row['day']) &
                                  (pdf['session']   == s_row['session'])]
                    if len(sess_df) == 0:
                        continue
                    n_total = sess_df['total_episodes'].mean()
                    n_theta = sess_df['theta_episodes'].mean()
                    arena   = s_row['arena_type']
                    color   = arena_colors.get(arena, '#888888')

                    ax.bar(si, n_theta,           color=color, alpha=0.8,
                           edgecolor='white', lw=0.5,
                           label=arena if si == 0 else '')
                    ax.bar(si, n_total - n_theta, bottom=n_theta,
                           color=color, alpha=0.2, edgecolor='white', lw=0.5)

                ax.set_xlabel('Session index', fontsize=8)
                ax.set_ylabel('N episodes', fontsize=8)
                ax.set_xticks(range(len(sessions)))
                ax.set_xticklabels(
                    [f"{r.session}" for _, r in sessions.iterrows()],
                    fontsize=6, rotation=45, ha='right')
                ax.spines[['top', 'right']].set_visible(False)
                if row == 0:
                    ax.set_title('Theta (dark) vs\ntotal episodes (light)', fontsize=9)
                    ax.legend(fontsize=7)

        fig.suptitle('fBOSC episode properties per animal', fontsize=11)
        plt.tight_layout()
        _save_or_show(fig, save, save_dir, 'fbosc_episode_properties')
        return fig

    # ---- cell [83] ----
    # No per-day arena mapping is available from the fBOSC batch output --
    # day_arena_map is optional everywhere it's used below and simply falls
    # back to an ungrouped ('all') coloring when None.
    day_arena_map = None
    plot_episode_properties(filtered_episodes, pepisode_df=pepisode_df, day_arena_map=day_arena_map, save=True, save_dir=FBOSC_FIGURE_DIR)

    # ---- cell [84] ----
    pepisode_df

    # ---- cell [85] ----
    def plot_duration_frequency_relationship(episodes_df, day_arena_map=None,
                                              save=True, save_dir=None):
        animals      = sorted(episodes_df['animal'].unique())
        arena_colors = {'1D_linear': '#4C72B0', '2D_open': '#DD8452', 'all': '#888888'}

        def _get_arena_map(animal):
            if day_arena_map is None:
                return None
            first_val = next(iter(day_arena_map.values()))
            if isinstance(first_val, dict):
                return day_arena_map.get(animal, None)
            return day_arena_map

        def _day_col(df):
            for c in ['day', 'date']:
                if c in df.columns:
                    return c
            raise ValueError(f"No 'day' or 'date' column. Columns: {df.columns.tolist()}")

        # ── add arena_type ────────────────────────────────────────────────────────
        episodes_df = episodes_df.copy()
        ep_day_col  = _day_col(episodes_df)
        episodes_df['arena_type'] = ''

        for animal in animals:
            amap = _get_arena_map(animal)
            if not amap:
                continue
            mask   = episodes_df['animal'] == animal
            mapped = episodes_df.loc[mask, ep_day_col].map(amap)
            if mapped.isna().all():
                amap_stripped = {k.strip(): v for k, v in amap.items()}
                mapped = episodes_df.loc[mask, ep_day_col].str.strip().map(amap_stripped)
            if mapped.isna().all():
                print(f"  Warning: no arena matches for '{animal}' in episodes_df.")
            else:
                episodes_df.loc[mask, 'arena_type'] = mapped.fillna('')

        fig, axes = plt.subplots(len(animals), 2,
                                  figsize=(10, 3.5 * len(animals)),
                                  squeeze=False)

        for row, animal in enumerate(animals):
            adf            = episodes_df[episodes_df['animal'] == animal]
            arenas_present = sorted(adf['arena_type'][adf['arena_type'] != ''].unique())
            use_color      = len(arenas_present) > 0

            for col, (ycol, ylabel) in enumerate([
                ('DurationS', 'Duration (s)  ← frequency-dependent'),
                ('DurationC', 'Duration (cycles)  ← frequency-independent'),
            ]):
                ax = axes[row][col]

                plot_groups = arenas_present if use_color else ['_all']

                for arena in plot_groups:
                    if arena == '_all':
                        sub   = adf
                        color = '#888888'
                        label = 'all'
                    else:
                        sub   = adf[adf['arena_type'] == arena]
                        color = arena_colors.get(arena, '#888888')
                        label = arena

                    x = sub['FrequencyMean'].dropna()
                    y = sub[ycol].dropna()
                    idx = x.index.intersection(y.index)
                    if len(idx) == 0:
                        continue

                    ax.scatter(x[idx], y[idx], color=color,
                               alpha=0.2, s=8, label=label)

                    if len(idx) > 2:
                        slope, intercept, r, p, _ = stats.linregress(x[idx], y[idx])
                        xfit = np.linspace(x[idx].min(), x[idx].max(), 100)
                        ax.plot(xfit, slope * xfit + intercept,
                                color=color, lw=1.5, linestyle='--',
                                label=f'{label} r={r:.2f}, p={p:.3f}')

                ax.set_xlabel('Episode frequency (Hz)', fontsize=8)
                ax.set_ylabel(ylabel, fontsize=8)
                ax.spines[['top', 'right']].set_visible(False)
                ax.legend(fontsize=6)

                if row == 0:
                    ax.set_title(
                        'DurationS vs Frequency\n(should slope down)'
                        if col == 0 else
                        'DurationC vs Frequency\n(should be flat)',
                        fontsize=8)
                if col == 0:
                    ax.set_ylabel(f"{animal}\n{ylabel}", fontsize=8)

        fig.suptitle('Duration–frequency relationship', fontsize=10)
        plt.tight_layout()
        _save_or_show(fig, save, save_dir, 'fbosc_duration_frequency')
        return fig


    def plot_episode_scatter(episodes_df, day_arena_map=None,
                              save=True, save_dir=None):
        animals      = sorted(episodes_df['animal'].unique())
        arena_colors = {'1D_linear': '#4C72B0', '2D_open': '#DD8452', 'all': '#888888'}

        def _get_arena_map(animal):
            if day_arena_map is None:
                return None
            first_val = next(iter(day_arena_map.values()))
            if isinstance(first_val, dict):
                return day_arena_map.get(animal, None)
            return day_arena_map

        def _day_col(df):
            for c in ['day', 'date']:
                if c in df.columns:
                    return c
            raise ValueError(f"No 'day' or 'date' column. Columns: {df.columns.tolist()}")

        # ── add arena_type ────────────────────────────────────────────────────────
        episodes_df = episodes_df.copy()
        ep_day_col  = _day_col(episodes_df)
        episodes_df['arena_type'] = ''

        for animal in animals:
            amap = _get_arena_map(animal)
            if not amap:
                continue
            mask   = episodes_df['animal'] == animal
            mapped = episodes_df.loc[mask, ep_day_col].map(amap)
            if mapped.isna().all():
                amap_stripped = {k.strip(): v for k, v in amap.items()}
                mapped = episodes_df.loc[mask, ep_day_col].str.strip().map(amap_stripped)
            if mapped.isna().all():
                print(f"  Warning: no arena matches for '{animal}' in episodes_df.")
            else:
                episodes_df.loc[mask, 'arena_type'] = mapped.fillna('')

        pairs = [
            ('DurationC',     'PowerMean',  'Duration (cycles)', 'Power (a.u.)'),
            ('FrequencyMean', 'PowerMean',  'Frequency (Hz)',    'Power (a.u.)'),
            ('FrequencyMean', 'DurationC',  'Frequency (Hz)',    'Duration (cycles)'),
        ]

        fig, axes = plt.subplots(len(animals), len(pairs),
                                  figsize=(4.5 * len(pairs), 3.5 * len(animals)),
                                  squeeze=False)

        for row, animal in enumerate(animals):
            adf            = episodes_df[episodes_df['animal'] == animal]
            arenas_present = sorted(adf['arena_type'][adf['arena_type'] != ''].unique())
            use_color      = len(arenas_present) > 0

            for col, (xcol, ycol, xlabel, ylabel) in enumerate(pairs):
                ax = axes[row][col]

                plot_groups = arenas_present if use_color else ['_all']

                for arena in plot_groups:
                    if arena == '_all':
                        sub   = adf
                        color = '#888888'
                        label = 'all'
                    else:
                        sub   = adf[adf['arena_type'] == arena]
                        color = arena_colors.get(arena, '#888888')
                        label = arena

                    x = sub[xcol].dropna()
                    y = sub[ycol].dropna()
                    idx = x.index.intersection(y.index)
                    if len(idx) == 0:
                        continue

                    ax.scatter(x[idx], y[idx], color=color,
                               alpha=0.2, s=6, label=label)

                    if len(idx) > 2:
                        r, p = stats.spearmanr(x[idx], y[idx])
                        slope, intercept, *_ = stats.linregress(x[idx], y[idx])
                        xfit = np.linspace(x[idx].min(), x[idx].max(), 100)
                        ax.plot(xfit, slope * xfit + intercept,
                                color=color, lw=1.5, linestyle='--',
                                label=f'{label} ρ={r:.2f}, p={p:.3f}')

                ax.set_xlabel(xlabel, fontsize=8)
                ax.set_ylabel(ylabel, fontsize=8)
                ax.spines[['top', 'right']].set_visible(False)
                ax.legend(fontsize=6)

                if row == 0:
                    ax.set_title(f'{xlabel} vs\n{ylabel}', fontsize=8)
                if col == 0:
                    ax.set_ylabel(f"{animal}\n{ylabel}", fontsize=8)

        fig.suptitle('Episode property relationships', fontsize=11)
        plt.tight_layout()
        _save_or_show(fig, save, save_dir, 'fbosc_episode_scatter')
        return fig

    # ---- cell [86] ----
    plot_duration_frequency_relationship(filtered_episodes, day_arena_map=None, save=True, save_dir=FBOSC_FIGURE_DIR)
    plot_episode_scatter(filtered_episodes, day_arena_map=day_arena_map, save=True, save_dir=FBOSC_FIGURE_DIR)

    # ================================================================
    # Theta Continuity or IEI analysis
    # (notebook cell [87])
    # ================================================================

    # ---------- CHECKPOINT: Theta Continuity / IEI analysis ----------
    _run_iei = _ask_yes_no("Would you like to perform the Theta Continuity / IEI analysis?")
    if not _run_iei:
        print("Skipping Theta Continuity / IEI analysis section.")
    else:

        # ---- cell [88] ----
        def plot_theta_continuity(episodes_df, day_arena_map=None,
                                   save=True, save_dir=None):
            """
            Transient vs continuous structure.
            Panels: duration dist | IEI dist | duration CDF | IEI CDF |
                    duration vs IEI scatter | n episodes per session
            """
            arena_colors = {'1D_linear': '#4C72B0', '2D_open': '#DD8452', 'all': '#888888'}
            animals      = sorted(episodes_df['animal'].unique())

            def _get_arena_map(animal):
                if day_arena_map is None:
                    return None
                first_val = next(iter(day_arena_map.values()))
                if isinstance(first_val, dict):
                    return day_arena_map.get(animal, None)
                return day_arena_map

            def _day_col(df):
                for c in ['day', 'date']:
                    if c in df.columns:
                        return c
                raise ValueError(f"No 'day' or 'date' column. Columns: {df.columns.tolist()}")

            # ── add arena_type ────────────────────────────────────────────────────────
            episodes_df = episodes_df.copy()
            ep_day_col  = _day_col(episodes_df)
            episodes_df['arena_type'] = ''

            for animal in animals:
                amap = _get_arena_map(animal)
                if not amap:
                    continue
                mask   = episodes_df['animal'] == animal
                mapped = episodes_df.loc[mask, ep_day_col].map(amap)
                if mapped.isna().all():
                    amap_stripped = {k.strip(): v for k, v in amap.items()}
                    mapped = episodes_df.loc[mask, ep_day_col].str.strip().map(amap_stripped)
                if mapped.isna().all():
                    print(f"  Warning: no arena matches for '{animal}' in episodes_df.")
                else:
                    episodes_df.loc[mask, 'arena_type'] = mapped.fillna('')

            fig, axes = plt.subplots(len(animals), 6,
                                      figsize=(22, 3.5 * len(animals)), squeeze=False)

            for row, animal in enumerate(animals):
                adf            = episodes_df[episodes_df['animal'] == animal]
                arenas_present = sorted(adf['arena_type'][adf['arena_type'] != ''].unique())
                use_color      = len(arenas_present) > 0
                plot_groups    = arenas_present if use_color else ['_all']

                def _color(arena):
                    return arena_colors.get(arena, '#888888') if arena != '_all' else '#888888'

                def _label(arena):
                    return arena if arena != '_all' else 'all'

                def _sub(arena):
                    return adf if arena == '_all' else adf[adf['arena_type'] == arena]

                def _hist(ax, col, xlabel, log_x=False, bins=30):
                    for arena in plot_groups:
                        vals = _sub(arena)[col].dropna()
                        vals = vals[vals > 0]
                        if len(vals) == 0:
                            continue
                        data = np.log10(vals) if log_x else vals
                        ax.hist(data, bins=bins, alpha=0.6, color=_color(arena),
                                edgecolor='white', lw=0.4, label=_label(arena))
                    ax.set_xlabel(f'{xlabel} (log₁₀)' if log_x else xlabel, fontsize=8)
                    ax.set_ylabel('Count', fontsize=8)
                    ax.spines[['top', 'right']].set_visible(False)
                    if row == 0:
                        ax.legend(fontsize=7)

                def _cdf(ax, col, xlabel):
                    for arena in plot_groups:
                        vals = np.sort(_sub(arena)[col].dropna())
                        vals = vals[vals > 0]
                        if len(vals) == 0:
                            continue
                        cdf = np.arange(1, len(vals) + 1) / len(vals)
                        ax.plot(vals, cdf, lw=1.5,
                                color=_color(arena), label=_label(arena))
                    ax.set_xscale('log')
                    ax.set_xlabel(xlabel, fontsize=8)
                    ax.set_ylabel('Cumulative fraction', fontsize=8)
                    ax.spines[['top', 'right']].set_visible(False)
                    if row == 0:
                        ax.legend(fontsize=7)

                # ── A: duration histogram ─────────────────────────────────────────────
                _hist(axes[row][0], 'DurationC', 'Duration (cycles)', log_x=True)
                axes[row][0].set_ylabel(f'{animal}\nCount', fontsize=8)
                if row == 0:
                    axes[row][0].set_title('Episode duration\ndistribution', fontsize=8)

                # ── B: IEI histogram ──────────────────────────────────────────────────
                _hist(axes[row][1], 'iei_s', 'IEI (s)', log_x=True)
                if row == 0:
                    axes[row][1].set_title('Inter-episode interval\ndistribution', fontsize=8)

                # ── C: duration CDF ───────────────────────────────────────────────────
                _cdf(axes[row][2], 'DurationC', 'Duration (cycles)')
                if row == 0:
                    axes[row][2].set_title('Duration CDF\n(log x)', fontsize=8)

                # ── D: IEI CDF ────────────────────────────────────────────────────────
                _cdf(axes[row][3], 'iei_s', 'IEI (s)')
                if row == 0:
                    axes[row][3].set_title('IEI CDF\n(log x)', fontsize=8)

                # ── E: duration vs following IEI scatter ──────────────────────────────
                ax = axes[row][4]
                for arena in plot_groups:
                    sub = _sub(arena).sort_values('Onset')
                    dur = sub['DurationC'].values[:-1]
                    iei = sub['iei_s'].values[1:]
                    ok  = (~np.isnan(dur)) & (~np.isnan(iei)) & (iei > 0) & (dur > 0)
                    if ok.sum() == 0:
                        continue
                    ax.scatter(dur[ok], iei[ok], color=_color(arena),
                               alpha=0.15, s=6, label=_label(arena))
                    if ok.sum() > 2:
                        r, p = stats.spearmanr(dur[ok], iei[ok])
                        ax.text(0.05, 0.88 - list(plot_groups).index(arena) * 0.12,
                                f'ρ={r:.2f}', transform=ax.transAxes,
                                fontsize=7, color=_color(arena))
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel('Episode duration (cycles)', fontsize=8)
                ax.set_ylabel('Following IEI (s)', fontsize=8)
                ax.spines[['top', 'right']].set_visible(False)
                if row == 0:
                    ax.set_title('Duration vs\nfollowing IEI', fontsize=8)
                    ax.legend(fontsize=7)

                # ── F: n episodes per session ─────────────────────────────────────────
                ax = axes[row][5]
                n_ep = (adf.groupby([ep_day_col, 'session', 'arena_type'])
                           .size()
                           .reset_index(name='n'))
                for arena in plot_groups:
                    vals = (n_ep if arena == '_all'
                            else n_ep[n_ep['arena_type'] == arena])['n']
                    if len(vals) == 0:
                        continue
                    ax.hist(vals, bins=15, alpha=0.6, color=_color(arena),
                            edgecolor='white', lw=0.4, label=_label(arena))
                ax.set_xlabel('N episodes / session', fontsize=8)
                ax.set_ylabel('Count', fontsize=8)
                ax.spines[['top', 'right']].set_visible(False)
                if row == 0:
                    ax.set_title('Episodes per\nsession', fontsize=8)
                    ax.legend(fontsize=7)

            fig.suptitle('Theta continuity — transient vs sustained', fontsize=11)
            plt.tight_layout()
            _save_or_show(fig, save, save_dir, 'theta_continuity')
            return fig

        # ---- cell [89] ----
        filtered_episodes

        # ---- cell [90] ----
        # calculate IEI and add to filtered_episodes, IEI: gap between offset of episode i and onset of episode i+1
        out = filtered_episodes.sort_values('Onset').reset_index(drop=True)
        filtered_episodes['iei_s'] = out['Onset'] - out['Offset'].shift(1)

        #divide values in Onset, Offset and iei_s columns with 1e6 to convert them to seconds
        filtered_episodes['Onset'] = filtered_episodes['Onset'] / 1e6
        filtered_episodes['Offset'] = filtered_episodes['Offset'] / 1e6
        filtered_episodes['iei_s'] = filtered_episodes['iei_s'] / 1e6

        # ---- cell [91] ----
        plot_theta_continuity(filtered_episodes, day_arena_map=day_arena_map, save=True, save_dir=FBOSC_FIGURE_DIR)

        # ---- cell [92] ----
        # ── diagnostic first ──────────────────────────────────────────────────────────
        short_iei = filtered_episodes[filtered_episodes['iei_s'] < 0.1].copy()
        print(f"Episodes with IEI < 0.1s: {len(short_iei)} ({100*len(short_iei)/len(filtered_episodes):.1f}%)")
        print(f"\nIEI < 0.01s:  {(filtered_episodes['iei_s'] < 0.01).sum()}")
        print(f"IEI < 0.001s: {(filtered_episodes['iei_s'] < 0.001).sum()}")
        print(f"\nSample short-IEI rows:")
        print(short_iei[['animal', 'day', 'session', 'tetrode', 'channel',
                          'Onset', 'Offset', 'DurationS', 'iei_s']].head(10).to_string())

        # ---- cell [93] ----
        short_iei

        # ---- cell [94] ----
        def merge_split_episodes(episodes_df, min_iei_s=0.167):
            """
            Merge consecutive episodes separated by less than min_iei_s.
            Default 0.167s = one cycle at 6 Hz — the minimum physiologically
            plausible gap between two distinct theta episodes.

            Merging logic:
                - offset  → taken from the later episode (extends the bout)
                - DurationS → recomputed from merged onset/offset
                - DurationC → summed across merged episodes
                - FrequencyMean, PowerMean, SNRMean → weighted mean by DurationS

            Parameters
            ----------
            episodes_df : pd.DataFrame   from load_all_episodes / parse_episodes_table
            min_iei_s   : float          merge threshold in seconds

            Returns
            -------
            pd.DataFrame — same columns as input, with short-IEI episodes merged
            """
            group_cols  = ['animal', 'day', 'session', 'tetrode', 'channel']
            merged_all  = []
            n_before    = len(episodes_df)

            # weighted-average columns — weighted by DurationS
            wavg_cols = ['FrequencyMean', 'PowerMean', 'SNRMean']
            # summed columns
            sum_cols  = ['DurationC']
            # take from last episode in merge group
            last_cols = ['offset_s', 'Offset']
            # take from first episode in merge group
            first_cols = ['onset_s', 'Onset', 'Trial']

            for keys, grp in episodes_df.groupby(group_cols):
                grp     = grp.sort_values('onset_s').reset_index(drop=True)
                merged  = []
                current = grp.iloc[0].to_dict()
                current['_dur_weight'] = current['DurationS']   # track for weighted avg

                for i in range(1, len(grp)):
                    row = grp.iloc[i]
                    gap = row['onset_s'] - current['offset_s']

                    if 0 < gap < min_iei_s:
                        # ── extend current episode ────────────────────────────────────
                        w_old = current['_dur_weight']
                        w_new = row['DurationS']
                        w_tot = w_old + w_new

                        # weighted mean for spectral properties
                        for c in wavg_cols:
                            if c in current and c in row:
                                current[c] = (current[c] * w_old + row[c] * w_new) / w_tot

                        # sum cycle count
                        for c in sum_cols:
                            if c in current and c in row:
                                current[c] = current[c] + row[c]

                        # extend to end of later episode
                        for c in last_cols:
                            if c in row:
                                current[c] = row[c]

                        # update duration in seconds from onset/offset
                        current['DurationS']    = current['offset_s'] - current['onset_s']
                        current['_dur_weight']  = w_tot

                    else:
                        merged.append({k: v for k, v in current.items()
                                        if k != '_dur_weight'})
                        current = row.to_dict()
                        current['_dur_weight'] = current['DurationS']

                merged.append({k: v for k, v in current.items() if k != '_dur_weight'})

                # ── recompute IEI after merging ───────────────────────────────────────
                merged_df          = pd.DataFrame(merged)
                merged_df          = merged_df.sort_values('onset_s').reset_index(drop=True)
                merged_df['iei_s'] = (merged_df['onset_s']
                                      - merged_df['offset_s'].shift(1))

                merged_all.append(merged_df)

            result   = pd.concat(merged_all, ignore_index=True)
            n_after  = len(result)
            n_merged = n_before - n_after

            print(f"Merge threshold:  {min_iei_s*1000:.0f} ms  "
                  f"(1 cycle at {1/min_iei_s:.0f} Hz)")
            print(f"Before: {n_before:,} episodes")
            print(f"After:  {n_after:,} episodes  ({n_merged:,} merged, "
                  f"{100*n_merged/n_before:.1f}%)")

            # ── verify: short IEIs should now be gone ────────────────────────────────
            remaining = (result['iei_s'] < min_iei_s).sum()
            print(f"IEI < threshold remaining: {remaining} "
                  f"({'✓ clean' if remaining == 0 else '⚠ check groupby cols'})")

            return result

        # ---- cell [95] ----
        def test_frequency_continuity_across_iei(episodes_df, target_iei_range=(0.8, 1.5)):
            """
            If theta is continuous but just dipping below threshold, the episodes
            flanking the gap should have very similar FrequencyMean.
            If theta genuinely stops and restarts, frequencies should be more variable.

            Compares frequency difference for:
                - pairs flanking target IEI range  (the suspicious gaps)
                - pairs flanking longer IEIs       (genuine gaps, control)
            """
            max_iei_s = episodes_df['session_duration_sec'].median()

            results = []
            for keys, grp in episodes_df.groupby(
                    ['animal', 'day', 'session', 'tetrode', 'channel']):   # <-- must include tetrode+channel
                grp = grp.sort_values('Onset').reset_index(drop=True)

                for i in range(1, len(grp)):
                    iei = grp.loc[i, 'iei_s']
                    if np.isnan(iei) or iei <= 0 or iei > 100:  # <-- filter impossible IEIs
                        continue
                    f_before = grp.loc[i-1, 'FrequencyMean']
                    f_after  = grp.loc[i,   'FrequencyMean']
                    freq_diff = abs(f_after - f_before)

                    if target_iei_range[0] <= iei <= target_iei_range[1]:
                        group = 'suspicious_gap'
                    elif iei > 3.0:
                        group = 'long_gap'
                    else:
                        continue

                    results.append({'iei_s': iei, 'freq_diff_hz': freq_diff, 'group': group})

            res_df = pd.DataFrame(results)
            if len(res_df) == 0:
                print("No pairs found in target range.")
                return

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # frequency difference distribution
            for grp_name, color in [('suspicious_gap', '#DD8452'), ('long_gap', '#4C72B0')]:
                vals = res_df[res_df['group'] == grp_name]['freq_diff_hz']
                if len(vals):
                    axes[0].hist(vals, bins=30, alpha=0.6, color=color,
                                 edgecolor='white', lw=0.4, label=f'{grp_name} (n={len(vals)})')

            axes[0].set_xlabel('|FrequencyMean difference| (Hz)', fontsize=9)
            axes[0].set_ylabel('Count', fontsize=9)
            axes[0].set_title('Frequency continuity across IEI\n'
                               'Smaller diff = more likely same oscillator', fontsize=8)
            axes[0].legend(fontsize=8)
            axes[0].spines[['top', 'right']].set_visible(False)

            # stats
            s_vals = res_df[res_df['group'] == 'suspicious_gap']['freq_diff_hz'].dropna()
            l_vals = res_df[res_df['group'] == 'long_gap']['freq_diff_hz'].dropna()
            if len(s_vals) > 1 and len(l_vals) > 1:
                stat, p = stats.mannwhitneyu(s_vals, l_vals, alternative='less')
                axes[0].text(0.6, 0.85,
                             f'suspicious < long?\np={p:.3f} (Mann-Whitney)',
                             transform=axes[0].transAxes, fontsize=8)

            # IEI vs freq diff scatter
            axes[1].scatter(res_df['iei_s'], res_df['freq_diff_hz'],
                            alpha=0.2, s=8, color='#888888')
            axes[1].set_xlabel('IEI duration (s)', fontsize=9)
            axes[1].set_ylabel('|Frequency difference| (Hz)', fontsize=9)
            axes[1].set_title('Does longer gap = more frequency drift?', fontsize=8)
            axes[1].spines[['top', 'right']].set_visible(False)
            r, p = stats.spearmanr(res_df['iei_s'].dropna(), res_df['freq_diff_hz'].dropna())
            axes[1].text(0.6, 0.85, f'ρ={r:.2f}, p={p:.3f}',
                         transform=axes[1].transAxes, fontsize=8)

            plt.tight_layout()
            os.makedirs(FBOSC_FIGURE_DIR, exist_ok=True)
            fig.savefig(os.path.join(FBOSC_FIGURE_DIR, 'fbosc_freq_continuity_across_iei.png'),
                        bbox_inches='tight', dpi=300)
            plt.show()
            plt.close(fig)

            print(f"\nSuspicious gaps: median freq diff = {s_vals.median():.2f} Hz")
            print(f"Long gaps:        median freq diff = {l_vals.median():.2f} Hz")
            return fig, res_df

        # ---- cell [96] ----
        fig, res_df = test_frequency_continuity_across_iei(
            filtered_episodes, target_iei_range=(0.8, 1.5))

    # ================================================================
    # speed - episode correlation
    # (notebook cell [97])
    # ================================================================

    # ---- cell [102] ----
    def build_animal_session_index(animal):
        """Walk ANIMALS[animal]'s .ncs files and index them by (day, session)
        as parsed by parse_metadata_from_path(), replacing the original
        notebook's FileManager.list_indexed_metadata()/get_files() lookups
        (this pipeline never builds a FileManager -- see PART 2's header note).

        Returns
        -------
        day_session_map   : dict  {day: [session1, session2, ...]}
        tracking_path_map : dict  {(day, session): tracking .csv path or None}
            One tracking .csv per session folder (find_position_file(), same
            file PART 1 uses for its own speed-gating).
        """
        folder = ANIMALS[animal]
        day_sessions = {}
        tracking_path_map = {}
        for root, _dirs, files in os.walk(folder):
            for fname in files:
                if not fname.endswith('.ncs'):
                    continue
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, folder)
                meta = parse_metadata_from_path(rel)
                day, session = meta['date'], meta['session']
                day_sessions.setdefault(day, set()).add(session)
                key = (day, session)
                if key not in tracking_path_map:
                    tracking_path_map[key] = find_position_file(fpath)
        day_session_map = {d: sorted(s) for d, s in day_sessions.items()}
        return day_session_map, tracking_path_map


    def load_tracking_for_all_sessions(tracking_path_map, animal, day_session_map,
                                        px_per_cm=None, smooth_method='savgol',
                                        savgol_window=11, savgol_poly=3,
                                        gaussian_sigma_s=0.2):
        """
        Load and compute speed for all sessions, using this pipeline's own
        folder-walk tracking-file index (tracking_path_map, from
        build_animal_session_index()) instead of a FileManager object.

        Parameters
        ----------
        tracking_path_map : dict  {(day, session): tracking .csv path or None}
        animal         : str
        day_session_map: dict  {day: [session1, session2, ...]}
        px_per_cm      : float or None  pixel-to-cm conversion
        smooth_method  : str   'savgol' | 'gaussian' | 'median' | 'none'

        Returns
        -------
        tracking_dict : dict
            {(animal, day, session): pd.DataFrame with [time_s, x, y, speed_raw, speed_smooth]}
        interp_dict   : dict
            {(animal, day, session): scipy interpolator(time_s) → speed_smooth}
        """
        tracking_dict = {}

        for day, sessions in day_session_map.items():
            for session in sessions:

                # ── get the tracking file via this pipeline's own index ───────────
                tracking_path = tracking_path_map.get((day, session))
                if not tracking_path:
                    print(f"  No tracking file for {animal} | {day} | {session}")
                    continue

                try:
                    df = load_tracking_and_compute_speed(
                        tracking_path,
                        px_per_cm=px_per_cm,
                        smooth_method=smooth_method,
                        savgol_window=savgol_window,
                        savgol_poly=savgol_poly,
                        gaussian_sigma_s=gaussian_sigma_s,
                    )
                    tracking_dict[(animal, day, session)] = df
                    print(f"  Loaded: {animal} | {day} | {session} | "
                          f"{os.path.basename(tracking_path)} | "
                          f"{len(df)} frames | {df['time_s'].iloc[-1]:.1f}s")

                except Exception as e:
                    print(f"  Error loading {tracking_path}: {e}")
                    continue

        print(f"\nTracking loaded for {len(tracking_dict)} / "
              f"{sum(len(s) for s in day_session_map.values())} sessions.")

        interp_dict = build_speed_interpolators(tracking_dict)
        return tracking_dict, interp_dict


    def attach_speed_to_episodes(episodes_df, interp_dict, tracking_dict,
                                  n_samples=50):
        """
        For each episode, sample speed across the episode window
        and the preceding IEI window.

        Timestamp alignment:
            - fBOSC Onset/Offset are in microseconds (absolute hardware timestamps)
            - Tracking time_s is zeroed to first tracking frame (seconds from 0)
            - We convert Onset/Offset to seconds, then zero to first tracking frame
              by subtracting the LFP session start (min Onset / 1e6)
        """

        def _day_col(df):
            for c in ['day', 'date']:
                if c in df.columns:
                    return c
            raise ValueError(f"No 'day'/'date' column. Columns: {df.columns.tolist()}")

        ep_day_col  = _day_col(episodes_df)
        episodes_df = episodes_df.copy()

        ep_means     = []
        ep_medians   = []
        iei_means    = []
        iei_medians  = []
        missing_keys = set()

        # ── pre-compute per-session LFP start time in seconds ─────────────────────
        # Onset is absolute microseconds → convert to seconds
        # tracking time_s is zeroed to first tracking frame
        # so we zero LFP timestamps to the same reference:
        #   onset_in_tracking_s = (Onset_us / 1e6) - lfp_session_start_s
        # where lfp_session_start_s = min(Onset) / 1e6 for that session

        session_lfp_start = {}   # {(animal, day, session): min Onset in seconds}

        for (animal, day, session), grp in episodes_df.groupby(
                ['animal', ep_day_col, 'session']):
            key = (animal, day, session)
            lfp_start_s = grp['Onset'].min() / 1e6
            session_lfp_start[key] = lfp_start_s

            # sanity check against tracking
            tracking = tracking_dict.get(key)
            if tracking is not None:
                track_dur   = tracking['time_s'].max() - tracking['time_s'].min()
                lfp_dur     = (grp['Offset'].max() - grp['Onset'].min()) / 1e6
                onset_range = ((grp['Onset'].min() - grp['Onset'].min()) / 1e6,
                               (grp['Onset'].max() - grp['Onset'].min()) / 1e6)
                print(f"  {key}:")
                print(f"    LFP duration:      {lfp_dur:.1f}s")
                print(f"    Tracking duration: {track_dur:.1f}s")
                print(f"    Episode onset range (zeroed): "
                      f"{onset_range[0]:.1f}s → {onset_range[1]:.1f}s")
                print(f"    Tracking time range: "
                      f"{tracking['time_s'].min():.1f}s → {tracking['time_s'].max():.1f}s")
                if onset_range[1] > tracking['time_s'].max() + 10:
                    print(f"    ⚠ Episode onsets exceed tracking duration — check alignment")
            else:
                if key not in missing_keys:
                    print(f"  Warning: no tracking for {key}")

        # ── compute IEI per episode (gap to previous episode within same group) ───
        # IEI in seconds = (Onset[i] - Offset[i-1]) / 1e6
        # stored per episode, NaN for first episode in group
        iei_s_list = []
        for (animal, day, session), grp in episodes_df.groupby(
                ['animal', ep_day_col, 'session']):
            grp_sorted = grp.sort_values('Onset')
            iei_vals   = (grp_sorted['Onset'].values[1:] -
                          grp_sorted['Offset'].values[:-1]) / 1e6
            # prepend NaN for first episode
            iei_full = np.concatenate([[np.nan], iei_vals])
            iei_s_list.append(
                pd.Series(iei_full, index=grp_sorted.index))

        episodes_df['iei_s'] = pd.concat(iei_s_list).reindex(episodes_df.index)

        # ── attach speed per episode ──────────────────────────────────────────────
        for _, row in episodes_df.iterrows():
            key        = (row['animal'], row[ep_day_col], row['session'])
            interp     = interp_dict.get(key)
            lfp_start  = session_lfp_start.get(key, np.nan)

            if interp is None:
                if key not in missing_keys:
                    print(f"  Warning: no speed interpolator for {key}")
                    missing_keys.add(key)
                ep_means.append(np.nan);    ep_medians.append(np.nan)
                iei_means.append(np.nan);   iei_medians.append(np.nan)
                continue

            if np.isnan(row['Onset']) or np.isnan(row['Offset']):
                ep_means.append(np.nan);    ep_medians.append(np.nan)
                iei_means.append(np.nan);   iei_medians.append(np.nan)
                continue

            # convert to seconds, zeroed to session LFP start
            onset_s  = row['Onset']  / 1e6 - lfp_start
            offset_s = row['Offset'] / 1e6 - lfp_start

            # ── speed during episode ──────────────────────────────────────────────
            t_ep  = np.linspace(onset_s, offset_s, n_samples)
            sp_ep = interp(t_ep)
            ep_means.append(np.nanmean(sp_ep))
            ep_medians.append(np.nanmedian(sp_ep))

            # ── speed during preceding IEI ────────────────────────────────────────
            iei = row.get('iei_s', np.nan)
            if np.isnan(iei) or iei <= 0:
                iei_means.append(np.nan)
                iei_medians.append(np.nan)
            else:
                iei_start = onset_s - iei
                t_iei     = np.linspace(iei_start, onset_s, n_samples)
                sp_iei    = interp(t_iei)
                iei_means.append(np.nanmean(sp_iei))
                iei_medians.append(np.nanmedian(sp_iei))

        episodes_df['speed_ep_mean']    = ep_means
        episodes_df['speed_ep_median']  = ep_medians
        episodes_df['speed_iei_mean']   = iei_means
        episodes_df['speed_iei_median'] = iei_medians

        n_ok = pd.Series(ep_means).notna().sum()
        print(f"\nSpeed attached to {n_ok:,} / {len(episodes_df):,} episodes.")

        if n_ok == 0:
            print("⚠ ALL speed values are NaN — likely timestamp misalignment.")
        elif n_ok < len(episodes_df) * 0.5:
            print(f"⚠ Only {100*n_ok/len(episodes_df):.0f}% of episodes got speed.")

        return episodes_df


    def plot_speed_theta(episodes_df, pepisode_df, interp_dict,
                         day_arena_map=None, speed_threshold_cms=5.0,
                         save=True, save_dir=None):
        """
        Speed vs theta relationship. One row per animal.

        Panel A: Speed during episode vs during IEI — violin + dots
        Panel B: Episode DurationC vs speed — scatter + Spearman
        Panel C: IEI duration vs speed during IEI — scatter + Spearman
        Panel D: P(theta | speed bin) — movement gating curve
        Panel E: Speed aligned to episode onset — peri-event average
        Panel F: P-episode vs mean session speed — session scatter
        """
        arena_colors  = {'1D_linear': '#4C72B0', '2D_open': '#DD8452', 'all': '#888888'}
        animals       = sorted(episodes_df['animal'].unique())
        animal_colors = plt.cm.tab10(np.linspace(0, 1, len(animals)))
        acmap         = dict(zip(animals, animal_colors))

        def _get_arena_map(animal):
            if day_arena_map is None:
                return None
            first_val = next(iter(day_arena_map.values()))
            if isinstance(first_val, dict):
                return day_arena_map.get(animal, None)
            return day_arena_map

        def _day_col(df):
            for c in ['day', 'date']:
                if c in df.columns:
                    return c
            raise ValueError(f"No 'day'/'date' column. Columns: {df.columns.tolist()}")

        ep_day_col = _day_col(episodes_df)

        # ── add arena_type ────────────────────────────────────────────────────────
        episodes_df = episodes_df.copy()
        episodes_df['arena_type'] = ''
        for animal in animals:
            amap = _get_arena_map(animal)
            if not amap:
                continue
            mask   = episodes_df['animal'] == animal
            mapped = episodes_df.loc[mask, ep_day_col].map(amap)
            if mapped.isna().all():
                amap_s = {k.strip(): v for k, v in amap.items()}
                mapped = episodes_df.loc[mask, ep_day_col].str.strip().map(amap_s)
            episodes_df.loc[mask, 'arena_type'] = mapped.fillna('')

        if pepisode_df is not None:
            pep_day_col = _day_col(pepisode_df)
            pepisode_df = pepisode_df.copy()
            pepisode_df['arena_type'] = ''
            for animal in animals:
                amap = _get_arena_map(animal)
                if not amap:
                    continue
                mask   = pepisode_df['animal'] == animal
                mapped = pepisode_df.loc[mask, pep_day_col].map(amap)
                if mapped.isna().all():
                    amap_s = {k.strip(): v for k, v in amap.items()}
                    mapped = pepisode_df.loc[mask, pep_day_col].str.strip().map(amap_s)
                pepisode_df.loc[mask, 'arena_type'] = mapped.fillna('')

        has_speed = episodes_df['speed_ep_mean'].notna().any()
        if not has_speed:
            print("No speed data. Run attach_speed_to_episodes first.")
            return None

        # ── pre-compute session LFP start times for onset alignment ──────────────
        session_lfp_start = {}
        for (animal, day, session), grp in episodes_df.groupby(
                ['animal', ep_day_col, 'session']):
            session_lfp_start[(animal, day, session)] = grp['Onset'].min() / 1e6

        fig, axes = plt.subplots(len(animals), 6,
                                  figsize=(24, 4.5 * len(animals)), squeeze=False)

        for row, animal in enumerate(animals):
            adf            = episodes_df[(episodes_df['animal'] == animal) &
                                          episodes_df['speed_ep_mean'].notna()]
            arenas_present = sorted(adf['arena_type'][adf['arena_type'] != ''].unique())
            use_color      = len(arenas_present) > 0
            plot_groups    = arenas_present if use_color else ['_all']

            def _color(arena): return arena_colors.get(arena, '#888888')
            def _label(arena): return arena if arena != '_all' else 'all'
            def _sub(arena):   return adf if arena == '_all' \
                                      else adf[adf['arena_type'] == arena]

            # ── A: speed during theta vs IEI violin ───────────────────────────────
            ax     = axes[row][0]
            paired = adf[['speed_ep_median', 'speed_iei_median']].dropna()
            for xi, (col, label, c) in enumerate([
                ('speed_ep_median',  'During\ntheta', '#4C72B0'),
                ('speed_iei_median', 'During\nIEI',   '#DD8452'),
            ]):
                vals = paired[col]
                if len(vals) < 2:
                    continue
                parts = ax.violinplot(vals, positions=[xi], widths=0.5,
                                       showmedians=True, showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(c); pc.set_alpha(0.4)
                parts['cmedians'].set_color(c)
                parts['cmedians'].set_linewidth(2)
                jit = np.random.uniform(-0.07, 0.07, size=len(vals))
                ax.scatter(xi + jit, vals, color=c, alpha=0.25, s=6, zorder=3)

            if len(paired) > 5:
                _, p = stats.wilcoxon(paired['speed_ep_median'],
                                       paired['speed_iei_median'],
                                       nan_policy='omit')
                ax.set_title(f'Speed: theta vs IEI\np={p:.3f}', fontsize=8,
                             color='red' if p < 0.05 else 'gray')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['During\ntheta', 'During\nIEI'], fontsize=8)
            ax.set_ylabel(f'{animal}\nSpeed (cm/s)', fontsize=8)
            ax.spines[['top', 'right']].set_visible(False)

            # ── B: DurationC vs speed scatter ─────────────────────────────────────
            ax = axes[row][1]
            for arena in plot_groups:
                sub  = _sub(arena)
                x, y = sub['speed_ep_median'], sub['DurationC']
                idx  = x.index.intersection(y.index)
                if len(idx) == 0:
                    continue
                ax.scatter(x[idx], y[idx], color=_color(arena),
                           alpha=0.2, s=8, label=_label(arena))
                if len(idx) > 2:
                    r, p = stats.spearmanr(x[idx], y[idx])
                    ax.text(0.05,
                            0.88 - list(plot_groups).index(arena) * 0.12,
                            f'ρ={r:.2f}, p={p:.3f}',
                            transform=ax.transAxes, fontsize=7,
                            color=_color(arena))
            ax.set_xlabel('Speed during episode (cm/s)', fontsize=8)
            ax.set_ylabel('Duration (cycles)', fontsize=8)
            ax.set_yscale('log')
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(fontsize=7)
            if row == 0:
                ax.set_title('Speed →\nepisode duration', fontsize=8)

            # ── C: IEI vs speed during IEI ────────────────────────────────────────
            ax = axes[row][2]
            for arena in plot_groups:
                sub  = _sub(arena)
                sub  = sub[(sub['iei_s'] > 0) & sub['speed_iei_median'].notna()]
                x, y = sub['speed_iei_median'], sub['iei_s']
                idx  = x.index.intersection(y.index)
                if len(idx) == 0:
                    continue
                ax.scatter(x[idx], y[idx], color=_color(arena),
                           alpha=0.2, s=8, label=_label(arena))
                if len(idx) > 2:
                    r, p = stats.spearmanr(x[idx], y[idx])
                    ax.text(0.05,
                            0.88 - list(plot_groups).index(arena) * 0.12,
                            f'ρ={r:.2f}, p={p:.3f}',
                            transform=ax.transAxes, fontsize=7,
                            color=_color(arena))
            ax.set_xlabel('Speed during IEI (cm/s)', fontsize=8)
            ax.set_ylabel('IEI duration (s)', fontsize=8)
            ax.set_yscale('log')
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(fontsize=7)
            if row == 0:
                ax.set_title('Speed during gap\nvs IEI length', fontsize=8)

            # ── D: P(theta | speed bin) ───────────────────────────────────────────
            ax         = axes[row][3]
            speed_bins = np.arange(0, 65, 5)
            bin_c      = (speed_bins[:-1] + speed_bins[1:]) / 2

            for arena in plot_groups:
                sub   = _sub(arena)
                t_ep  = np.zeros(len(speed_bins) - 1)
                t_iei = np.zeros(len(speed_bins) - 1)

                for _, ep in sub.iterrows():
                    for spd, dur, is_ep in [
                        (ep['speed_ep_median'],  ep['DurationS'],             True),
                        (ep['speed_iei_median'], ep.get('iei_s', np.nan),    False),
                    ]:
                        if np.isnan(spd) or np.isnan(dur) or dur <= 0:
                            continue
                        b = np.digitize(spd, speed_bins) - 1
                        if 0 <= b < len(t_ep):
                            if is_ep:
                                t_ep[b]  += dur
                            else:
                                t_iei[b] += dur

                total = t_ep + t_iei
                p_ep  = np.where(total > 0, 100 * t_ep / total, np.nan)
                ax.plot(bin_c, p_ep, 'o-', lw=1.5, markersize=5,
                        color=_color(arena), label=_label(arena))

            ax.axvline(speed_threshold_cms, color='gray', lw=1,
                       linestyle=':', label=f'{speed_threshold_cms} cm/s')
            ax.set_xlabel('Speed (cm/s)', fontsize=8)
            ax.set_ylabel('% time in theta', fontsize=8)
            ax.set_ylim([0, 100])
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(fontsize=7)
            if row == 0:
                ax.set_title('P(theta) vs\nspeed bin', fontsize=8)

            # ── E: speed aligned to episode onset ─────────────────────────────────
            ax       = axes[row][4]
            window_s = 3.0
            t_axis   = np.linspace(-window_s, window_s, 120)

            for arena in plot_groups:
                sub    = _sub(arena)
                trials = []

                for _, ep in sub.iterrows():
                    key       = (animal, ep[ep_day_col], ep['session'])
                    interp    = interp_dict.get(key)
                    lfp_start = session_lfp_start.get(key, np.nan)

                    if interp is None or np.isnan(lfp_start):
                        continue

                    # convert Onset from microseconds to seconds,
                    # zeroed to session LFP start → same frame as tracking
                    onset_s = ep['Onset'] / 1e6 - lfp_start
                    sp      = interp(onset_s + t_axis)

                    if not np.all(np.isnan(sp)):
                        trials.append(sp)

                if trials:
                    mat  = np.array(trials)
                    mean = np.nanmean(mat, axis=0)
                    sem  = np.nanstd(mat, axis=0) / np.sqrt(
                        np.sum(~np.isnan(mat), axis=0).clip(1))
                    ax.plot(t_axis, mean, lw=1.5, color=_color(arena),
                            label=f'{_label(arena)} (n={len(trials)})')
                    ax.fill_between(t_axis, mean - sem, mean + sem,
                                    alpha=0.25, color=_color(arena))
                else:
                    ax.text(0.1, 0.5 - list(plot_groups).index(arena) * 0.15,
                            f'{_label(arena)}: no trials',
                            transform=ax.transAxes, fontsize=7,
                            color=_color(arena))

            ax.axvline(0, color='black', lw=1, linestyle='--', label='Onset')
            ax.set_xlabel('Time from onset (s)', fontsize=8)
            ax.set_ylabel('Speed (cm/s)', fontsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(fontsize=7)
            if row == 0:
                ax.set_title('Speed aligned to\nepisode onset', fontsize=8)

            # ── F: p-episode vs mean session speed ────────────────────────────────
            ax = axes[row][5]
            if pepisode_df is not None:
                pdf = pepisode_df[pepisode_df['animal'] == animal]

                for arena in plot_groups:
                    sub = pdf if arena == '_all' \
                          else pdf[pdf['arena_type'] == arena]
                    xs, ys = [], []

                    for _, r in sub.iterrows():
                        key       = (animal, r[pep_day_col], r['session'])
                        interp    = interp_dict.get(key)
                        lfp_start = session_lfp_start.get(key, np.nan)

                        if interp is None or np.isnan(lfp_start):
                            continue

                        # sample speed across full session duration
                        sess_dur = r.get('session_duration_sec', np.nan)
                        if np.isnan(sess_dur):
                            ep_sub = episodes_df[
                                (episodes_df['animal']    == animal) &
                                (episodes_df[ep_day_col]  == r[pep_day_col]) &
                                (episodes_df['session']   == r['session'])
                            ]
                            sess_dur = ep_sub['session_duration_sec'].iloc[0] \
                                       if len(ep_sub) else np.nan
                        if np.isnan(sess_dur):
                            continue

                        t_samp = np.linspace(0, sess_dur, 500)
                        sp     = np.nanmedian(interp(t_samp))
                        if not np.isnan(sp):
                            xs.append(sp)
                            ys.append(r['proportion_time'] * 100)

                    if len(xs) == 0:
                        continue
                    ax.scatter(xs, ys, color=_color(arena), alpha=0.7,
                               s=40, label=_label(arena), zorder=3)
                    if len(xs) > 2:
                        r_val, p = stats.spearmanr(xs, ys)
                        ax.text(0.05,
                                0.88 - list(plot_groups).index(arena) * 0.12,
                                f'ρ={r_val:.2f}, p={p:.3f}',
                                transform=ax.transAxes, fontsize=7,
                                color=_color(arena))

            ax.set_xlabel('Median session speed (cm/s)', fontsize=8)
            ax.set_ylabel('P-episode (%)', fontsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(fontsize=7)
            if row == 0:
                ax.set_title('P-episode vs\nsession speed', fontsize=8)

        fig.suptitle('Theta episodes vs animal speed', fontsize=11)
        plt.tight_layout()
        _save_or_show(fig, save, save_dir, 'theta_speed')
        return fig

    def plot_theta_bout_summary(episodes_df, pepisode_df, interp_dict,
                                 day_arena_map=None, save=True, save_dir=None):
        """
        Summary figure for the transient vs continuous narrative.

        Panel A: Example session — episode bars on speed trace
        Panel B: P-episode vs mean session speed
        Panel C: Episode DurationC vs mean speed
        Panel D: Speed during theta vs outside theta per animal
        """
        arena_colors  = {'1D_linear': '#4C72B0', '2D_open': '#DD8452', 'all': '#888888'}
        animals       = sorted(episodes_df['animal'].unique())
        animal_colors = plt.cm.tab10(np.linspace(0, 1, len(animals)))
        acmap         = dict(zip(animals, animal_colors))

        def _get_arena_map(animal):
            if day_arena_map is None:
                return None
            first_val = next(iter(day_arena_map.values()))
            if isinstance(first_val, dict):
                return day_arena_map.get(animal, None)
            return day_arena_map

        def _day_col(df):
            for c in ['day', 'date']:
                if c in df.columns:
                    return c
            raise ValueError(f"No 'day'/'date' column. Columns: {df.columns.tolist()}")

        # ── add arena_type ────────────────────────────────────────────────────────
        ep_day_col  = _day_col(episodes_df)
        episodes_df = episodes_df.copy()
        episodes_df['arena_type'] = ''
        for animal in animals:
            amap = _get_arena_map(animal)
            if not amap:
                continue
            mask   = episodes_df['animal'] == animal
            mapped = episodes_df.loc[mask, ep_day_col].map(amap)
            if mapped.isna().all():
                amap_s = {k.strip(): v for k, v in amap.items()}
                mapped = episodes_df.loc[mask, ep_day_col].str.strip().map(amap_s)
            episodes_df.loc[mask, 'arena_type'] = mapped.fillna('')

        if pepisode_df is not None:
            pep_day_col = _day_col(pepisode_df)
            pepisode_df = pepisode_df.copy()
            pepisode_df['arena_type'] = ''
            for animal in animals:
                amap = _get_arena_map(animal)
                if not amap:
                    continue
                mask   = pepisode_df['animal'] == animal
                mapped = pepisode_df.loc[mask, pep_day_col].map(amap)
                if mapped.isna().all():
                    amap_s = {k.strip(): v for k, v in amap.items()}
                    mapped = pepisode_df.loc[mask, pep_day_col].str.strip().map(amap_s)
                pepisode_df.loc[mask, 'arena_type'] = mapped.fillna('')

        # ── session-level summaries ───────────────────────────────────────────────
        sess_summary = (episodes_df
                        .groupby(['animal', ep_day_col, 'session', 'arena_type'])
                        .agg(
                            mean_speed_theta =('speed_ep_median',  'median'),
                            mean_speed_iei   =('speed_iei_median', 'median'),
                            mean_dur_c       =('DurationC',        'median'),
                            n_episodes       =('DurationC',        'count'),
                            session_dur      =('session_duration_sec', 'first'),
                        ).reset_index())

        if pepisode_df is not None:
            pep_sess = (pepisode_df
                        .groupby(['animal', pep_day_col, 'session', 'arena_type'])
                        ['proportion_time'].mean()
                        .reset_index()
                        .rename(columns={pep_day_col: ep_day_col}))
            pep_sess['pepisode_pct'] = pep_sess['proportion_time'] * 100
            sess_summary = sess_summary.merge(
                pep_sess[['animal', ep_day_col, 'session', 'pepisode_pct']],
                on=['animal', ep_day_col, 'session'], how='left')

        # add mean session speed from interpolators
        sess_speeds = {}
        for _, row in sess_summary.iterrows():
            key    = (row['animal'], row[ep_day_col], row['session'])
            interp = interp_dict.get(key)
            if interp and not np.isnan(row.get('session_dur', np.nan)):
                t_samp = np.linspace(0, row['session_dur'], 500)
                sess_speeds[key] = np.nanmedian(interp(t_samp))
            else:
                sess_speeds[key] = np.nan

        sess_summary['sess_speed'] = sess_summary.apply(
            lambda r: sess_speeds.get((r['animal'], r[ep_day_col], r['session']), np.nan),
            axis=1)

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))

        # ── A: example session — speed trace + episode bars ───────────────────────
        ax = axes[0]
        best_key = max(interp_dict.keys(),
                       key=lambda k: len(episodes_df[
                           (episodes_df['animal']   == k[0]) &
                           (episodes_df[ep_day_col] == k[1]) &
                           (episodes_df['session']  == k[2])
                       ]))
        animal_ex, day_ex, sess_ex = best_key
        ex      = episodes_df[
            (episodes_df['animal']   == animal_ex) &
            (episodes_df[ep_day_col] == day_ex)    &
            (episodes_df['session']  == sess_ex)
        ]
        interp_ex = interp_dict.get(best_key)

        if interp_ex is not None and 'Onset' in ex.columns:
            sess_dur = ex['session_duration_sec'].iloc[0]
            t_trace  = np.linspace(0, sess_dur, int(sess_dur * 25))  # 25 Hz for display
            sp_trace = interp_ex(t_trace)
            ax.plot(t_trace, sp_trace, lw=0.6, color='#444444', alpha=0.8)

            # shade episodes
            for _, ep in ex.iterrows():
                if np.isnan(ep.get('Onset', np.nan)):
                    continue
                arena = ep.get('arena_type', '')
                color = arena_colors.get(arena, '#4C72B0')
                ax.axvspan(ep['Onset'], ep['Offset'],
                           alpha=0.3, color=color, lw=0)

            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Speed (cm/s)', fontsize=8)
            ax.set_title(f'Example session\n{animal_ex} | {sess_ex}', fontsize=8)
        else:
            ax.text(0.2, 0.5, 'Need Onset + speed data',
                    transform=ax.transAxes, fontsize=8, color='gray')
        ax.spines[['top', 'right']].set_visible(False)

        # ── B: p-episode vs session speed ─────────────────────────────────────────
        ax = axes[1]
        if 'pepisode_pct' in sess_summary.columns:
            for animal in animals:
                sub = sess_summary[sess_summary['animal'] == animal]
                ax.scatter(sub['sess_speed'], sub['pepisode_pct'],
                           color=acmap[animal], alpha=0.7, s=40,
                           label=animal, zorder=3)
            x   = sess_summary['sess_speed'].dropna()
            y   = sess_summary['pepisode_pct'].dropna()
            idx = x.index.intersection(y.index)
            if len(idx) > 2:
                r, p = stats.spearmanr(x[idx], y[idx])
                xfit = np.linspace(x[idx].min(), x[idx].max(), 100)
                sl, ic, *_ = stats.linregress(x[idx], y[idx])
                ax.plot(xfit, sl * xfit + ic, color='black', lw=1.5,
                        linestyle='--', label=f'ρ={r:.2f}, p={p:.3f}')
            ax.set_xlabel('Median session speed (cm/s)', fontsize=8)
            ax.set_ylabel('P-episode (%)', fontsize=8)
            ax.set_title('Movement gating:\np-episode vs speed', fontsize=8)
            ax.legend(fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)

        # ── C: DurationC vs session speed ─────────────────────────────────────────
        ax = axes[2]
        for animal in animals:
            sub = sess_summary[sess_summary['animal'] == animal]
            ax.scatter(sub['sess_speed'], sub['mean_dur_c'],
                       color=acmap[animal], alpha=0.7, s=40,
                       label=animal, zorder=3)
        x   = sess_summary['sess_speed'].dropna()
        y   = sess_summary['mean_dur_c'].dropna()
        idx = x.index.intersection(y.index)
        if len(idx) > 2:
            r, p = stats.spearmanr(x[idx], y[idx])
            sl, ic, *_ = stats.linregress(x[idx], y[idx])
            xfit = np.linspace(x[idx].min(), x[idx].max(), 100)
            ax.plot(xfit, sl * xfit + ic, color='black', lw=1.5,
                    linestyle='--', label=f'ρ={r:.2f}, p={p:.3f}')
        ax.set_xlabel('Median session speed (cm/s)', fontsize=8)
        ax.set_ylabel('Median episode duration (cycles)', fontsize=8)
        ax.set_title('Does speed predict\nepisode length?', fontsize=8)
        ax.legend(fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)

        # ── D: speed during theta vs IEI per animal ───────────────────────────────
        ax      = axes[3]
        x_pos   = np.arange(len(animals))
        width   = 0.35

        for xi, animal in enumerate(animals):
            sub = sess_summary[sess_summary['animal'] == animal]
            for offset, col, color, label in [
                (-width/2, 'mean_speed_theta', '#4C72B0', 'During theta'),
                ( width/2, 'mean_speed_iei',   '#DD8452', 'During IEI'),
            ]:
                vals = sub[col].dropna()
                if len(vals) == 0:
                    continue
                mean = vals.mean()
                sem  = vals.sem()
                ax.bar(xi + offset, mean, width=width, color=color,
                       alpha=0.7, label=label if xi == 0 else '')
                ax.errorbar(xi + offset, mean, yerr=sem,
                            fmt='none', color='black', capsize=3, lw=1.2)
                jit = np.random.uniform(-0.05, 0.05, size=len(vals))
                ax.scatter(xi + offset + jit, vals,
                           color=color, alpha=0.5, s=15, zorder=3)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(animals, fontsize=8, rotation=15, ha='right')
        ax.set_ylabel('Speed (cm/s)', fontsize=8)
        ax.set_title('Speed: during theta\nvs during IEI', fontsize=8)
        ax.legend(fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)

        fig.suptitle('Theta bout structure and movement relationship', fontsize=11)
        plt.tight_layout()
        _save_or_show(fig, save, save_dir, 'theta_bout_summary')
        return fig

    def plot_speed_smoothing_check(tracking_df, title='', n_seconds=60,
                                    save=True, save_dir=None):
        """
        Quick diagnostic: raw vs smoothed speed for the first n_seconds.
        Always run this first to verify smoothing looks right.
        """
        sub = tracking_df[tracking_df['time_s'] <= n_seconds]
        fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)

        axes[0].plot(sub['time_s'], sub['x'], lw=0.8, label='x', color='#4C72B0')
        axes[0].plot(sub['time_s'], sub['y'], lw=0.8, label='y', color='#DD8452')
        axes[0].set_ylabel('Position (px)')
        axes[0].legend(fontsize=8)
        axes[0].spines[['top', 'right']].set_visible(False)

        axes[1].plot(sub['time_s'], sub['speed_raw'],    lw=0.6, alpha=0.5,
                     color='gray', label='raw')
        axes[1].plot(sub['time_s'], sub['speed_smooth'], lw=1.2,
                     color='#4C72B0', label='smoothed')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Speed (cm/s)')
        axes[1].legend(fontsize=8)
        axes[1].spines[['top', 'right']].set_visible(False)

        fig.suptitle(f'Speed smoothing check — {title}', fontsize=10)
        plt.tight_layout()
        _save_or_show(fig, save, save_dir, f'speed_check_{title}')
        return fig

    def load_tracking_and_compute_speed(tracking_path, px_per_cm=None,
                                         smooth_method='savgol',
                                         savgol_window=11, savgol_poly=3,
                                         gaussian_sigma_s=0.2):
        """
        Load raw tracking CSV/Excel with columns [time, x, y].
        Compute speed with smoothing to handle noisy pose estimates.

        Parameters
        ----------
        tracking_path  : str    path to csv or xlsx
        px_per_cm      : float  pixel-to-cm conversion. If None, speed in px/s.
        smooth_method  : str    'savgol' | 'gaussian' | 'median' | 'none'
        savgol_window  : int    Savitzky-Golay window (must be odd)
        savgol_poly    : int    Savitzky-Golay polynomial order
        gaussian_sigma_s : float  gaussian sigma in seconds (for gaussian smoothing)

        Returns
        -------
        pd.DataFrame with columns: time_s, x, y, speed_raw, speed_smooth
        """
        ext = os.path.splitext(tracking_path)[1].lower()
        if ext in ('.xlsx', '.xls'):
            df = pd.read_excel(tracking_path)
        else:
            df = pd.read_csv(tracking_path)

        # normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        rename = {}
        for c in df.columns:
            if 'time' in c:   rename[c] = 'time_s'
            elif c == 'x':    rename[c] = 'x'
            elif c == 'y':    rename[c] = 'y'
        df = df.rename(columns=rename)

        # zero time to session start
        df['time_s'] = df['time_s'] - df['time_s'].iloc[0]
        df['time_s'] = df['time_s']/1e6 if df['time_s'].max() > 1e5 else df['time_s']  # convert us → s if needed
        df = df.sort_values('time_s').reset_index(drop=True)

        # ── detect and remove jump artifacts before computing speed ───────────────
        # large jumps in position (> 3 std) are likely tracking errors — interpolate
        dx = df['x'].diff()
        dy = df['y'].diff()
        jump_thresh = 3 * np.sqrt(dx**2 + dy**2).std()
        is_jump = np.sqrt(dx**2 + dy**2) > jump_thresh
        df.loc[is_jump, ['x', 'y']] = np.nan
        df[['x', 'y']] = df[['x', 'y']].interpolate(method='linear')

        # ── raw speed ─────────────────────────────────────────────────────────────
        dt = df['time_s'].diff()
        dx = df['x'].diff()
        dy = df['y'].diff()
        speed_px_s = np.sqrt(dx**2 + dy**2) / dt
        speed_px_s = speed_px_s.fillna(0)  # Fixed SettingWithCopyWarning

        # convert units
        scale = 1.0 / px_per_cm if px_per_cm else 1.0
        df['speed_raw'] = speed_px_s * scale   # cm/s if px_per_cm given

        # ── smoothing ─────────────────────────────────────────────────────────────
        if smooth_method == 'savgol':
            # Savitzky-Golay: preserves peaks, good default for tracking
            window = savgol_window if savgol_window % 2 == 1 else savgol_window + 1
            df['speed_smooth'] = savgol_filter(
                df['speed_raw'].fillna(0), window, savgol_poly)

        elif smooth_method == 'gaussian':
            # Gaussian: heavier smoothing, good for very noisy data
            fps = 1.0 / dt.median()
            sigma_frames = gaussian_sigma_s * fps
            df['speed_smooth'] = ndimage.gaussian_filter1d(
                df['speed_raw'].fillna(0), sigma=sigma_frames)

        elif smooth_method == 'median':
            # Median filter: best for removing spike artifacts
            df['speed_smooth'] = df['speed_raw'].rolling(
                window=savgol_window, center=True, min_periods=1).median()

        else:
            df['speed_smooth'] = df['speed_raw']

        # clip negative speeds (smoothing artifact)
        df['speed_smooth'] = df['speed_smooth'].clip(lower=0)

        print(f"Tracking loaded: {len(df)} frames | "
              f"duration={df['time_s'].iloc[-1]:.1f}s | "
              f"mean speed={df['speed_smooth'].mean():.1f} "
              f"{'cm/s' if px_per_cm else 'px/s'}")
        return df


    def build_speed_interpolators(tracking_dict):
        """
        Build per-session speed interpolators from tracking dataframes.

        Parameters
        ----------
        tracking_dict : dict
            {(animal, day, session): pd.DataFrame from load_tracking_and_compute_speed}

        Returns
        -------
        interp_dict : dict
            {(animal, day, session): scipy interpolator(time_s) → speed}
        """
        interp_dict = {}
        for key, df in tracking_dict.items():
            interp_dict[key] = interpolate.interp1d(
                df['time_s'], df['speed_smooth'],
                bounds_error=False, fill_value=np.nan
            )
        return interp_dict

    # ---- cell [103] ----
    # _speed_animal replaces the original notebook's hardcoded 'FA23BD' (which
    # didn't match this pipeline's ANIMALS casing, e.g. 'Fa23BD', and would
    # have silently filtered to zero rows) -- defaults to the first animal in
    # ANIMALS; edit the index if you want a different one.
    _speed_animal = animals[0]
    day_session_map, tracking_path_map = build_animal_session_index(_speed_animal)

    filtered_episodes_fa23bd = filtered_episodes[filtered_episodes['animal'] == _speed_animal]

    pepisode_df_fa23bd = pepisode_df[pepisode_df['animal'] == _speed_animal]

    # ── 1. load tracking via this pipeline's own folder index and build interpolators ──
    tracking_dict, interp_dict = load_tracking_for_all_sessions(
        tracking_path_map = tracking_path_map,
        animal         = _speed_animal,
        day_session_map= day_session_map,
        px_per_cm      = None,
        smooth_method  = 'savgol',
        savgol_window  = 11,
    )

    # ── 2. always check smoothing on one example ──────────────────────────────────
    example_key = list(tracking_dict.keys())[0]
    plot_speed_smoothing_check(tracking_dict[example_key],
                                title=str(example_key), n_seconds=60,
                                save=True, save_dir=FBOSC_FIGURE_DIR)

    # ── 3. attach speed to episodes ───────────────────────────────────────────────
    episodes_df = attach_speed_to_episodes(filtered_episodes_fa23bd, interp_dict, tracking_dict=tracking_dict)

    # ---- cell [104] ----
    plot_speed_theta(episodes_df, pepisode_df_fa23bd, interp_dict,
                      day_arena_map=day_arena_map,
                      speed_threshold_cms=5.0,
                      save=True, save_dir=FBOSC_FIGURE_DIR)

    # ---- cell [105] ----
    plot_theta_bout_summary(episodes_df, pepisode_df_fa23bd, interp_dict,
                             day_arena_map=day_arena_map,
                             save=True, save_dir=FBOSC_FIGURE_DIR)

    # ---- cell [106] ----
    filtered_episodes_fa23bd


# %% ####################################################################
# %% PART 3 — bycycle: cycle-by-cycle theta analysis in the time domain
# %% --------------------------------------------------------------------
# %% Started from the bycycle-tools tutorial "1. Theta oscillation cycle
# %% feature distributions" (https://github.com/bycycle-tools/bycycle.git),
# %% with preprocessing/thresholds matched to the paper's own analysis code
# %% (voytekresearch/Cole_2019_theta: config.py + 2_Compute_cyclebycycle_
# %% features.py) -- lowpass 25 Hz then highpass 1 Hz before cycle detection,
# %% trough-centered cycles, and the paper's 4-set burst-threshold robustness
# %% sweep (BYCYCLE_PAPER_BURST_SETS) recomputed alongside this pipeline's own
# %% tuned BYCYCLE_THRESHOLDS. Theta band is (3, 7) Hz here (BYCYCLE_F_THETA),
# %% not the paper's rat-derived (4, 10) Hz, to match this species' theta.
# %% Runs on the cleaned per-file LFP traces PART 1 saved into
# %% PROCESSED_PSDS_PKL (process_animal()'s lfp_store) -- the same traces
# %% PART 2's fBOSC section reads -- instead of the tutorial's bundled
# %% ca1.npy/ec3.npy example recordings or the paper's own .mat files. Each
# %% file is filtered and fit individually via bycycle_compute_features (the
# %% same pre-1.0 bycycle.features.compute_features the paper itself used --
# %% see compute_bycycle_features docstring for why not BycycleGroup); the
# %% per-file cycle tables are tagged with parse_metadata_from_path() metadata
# %% and concatenated into one long-form dataframe, mirroring how PART 2
# %% builds fBOSC_episodesTable. Answer "n" at the checkpoint below to skip
# %% this section.
# %% ####################################################################

# _ask_yes_no is defined earlier, before the MAIN PIPELINE / FOOOF checkpoint.

BYCYCLE_PROPS = {
    'volt_amp':   {'xlabel': 'Cycle amplitude (µV)',
                   'xlim': None},
    'period_ms':  {'xlabel': 'Cycle period (ms)',
                   'xlim': (0, 400)},
    'time_rdsym': {'xlabel': 'Rise-decay asymmetry\n(fraction of cycle in rise period)',
                   'xlim': (0, 1)},
    'time_ptsym': {'xlabel': 'Peak-trough asymmetry\n(fraction of cycle in peak period)',
                   'xlim': (0, 1)},
}


def compute_fooof_theta_bands(fooof_df, r2_min=None, error_max=None, default_band=None):
    """Per-animal theta band derived from FOOOF's fitted theta peaks.

    Each animal's band is centred on that animal's OWN FOOOF-measured theta
    oscillation -- median centre frequency +/- half the median bandwidth
    (theta_range_from_peak), taken across that animal's good-quality fits --
    rather than one fixed band guessed for every animal. "Good-quality" means
    the same bar export_low_quality_fits() uses: a theta peak was found
    (has_theta) and r_squared/error clear r2_min/error_max.

    Falls back to `default_band` (BYCYCLE_F_THETA) for any animal with no
    qualifying fits.

    Returns
    -------
    dict  {animal: (f_low, f_high)}
    """
    r2_min       = r2_min if r2_min is not None else R_SQUARED_MIN
    error_max    = error_max if error_max is not None else ERROR_MAX
    default_band = default_band or BYCYCLE_F_THETA

    good = fooof_df[fooof_df['has_theta'] &
                    (fooof_df['r_squared'] >= r2_min) &
                    (fooof_df['error'] <= error_max)]

    bands = {}
    for animal in fooof_df['animal'].unique():
        sub = good[good['animal'] == animal]
        if len(sub) == 0:
            print(f"  [{animal}] no good-quality FOOOF theta peaks -- "
                  f"falling back to default BYCYCLE_F_THETA={default_band}")
            bands[animal] = default_band
            continue
        cf = sub['theta_cf'].median()
        bw = sub['theta_bw'].median()
        lo, hi = theta_range_from_peak(cf, bw)
        bands[animal] = (lo, hi)
        print(f"  [{animal}] FOOOF-derived theta band: {lo:.2f}-{hi:.2f} Hz "
              f"(median cf={cf:.2f}, median bw={bw:.2f}, n={len(sub)} good fits)")
    return bands


def _bycycle_center_extrema_alt(center_extrema):
    """Map between this script's 'trough'/'peak' config value and bycycle 0.1.x's
    abbreviated 'T'/'P' codes (confusingly, that old build's docstring P/T column
    description is reversed -- 'T' is what actually gives trough-centered cycles,
    matching the paper's own extrema_center='T')."""
    mapping = {'trough': 'T', 'peak': 'P', 'T': 'trough', 'P': 'peak'}
    if center_extrema not in mapping:
        raise ValueError(f"Unknown center_extrema {center_extrema!r}; expected "
                         "'trough'/'peak' (or 'T'/'P').")
    return mapping[center_extrema]


def _resolve_bycycle_center_extrema(center_extrema, f_theta_probe):
    """Different bycycle installs across the machines/environments this script has
    run on disagree on the center_extrema convention compute_features wants: current
    bycycle (bycycle/features/features.py, per github.com/bycycle-tools/bycycle)
    requires the literal words 'peak'/'trough' and raises ValueError otherwise;
    bycycle 0.1.x requires the abbreviated codes 'P'/'T' and raises ValueError on
    anything else. Rather than guess which is installed, probe once (cheap -- a
    couple seconds of synthetic sine wave, no real data involved) on a value this
    script's own config always uses ('trough'/'peak' -- so it costs nothing on the
    now-current bycycle), falling back to the alternate convention only if that
    raises a center_extrema-specific ValueError. Resolved once per
    compute_bycycle_features() call and reused for every file."""
    probe_fs = 500.0
    probe_t = np.arange(0, 3.0, 1 / probe_fs)
    probe_sig = np.sin(2 * np.pi * float(np.mean(f_theta_probe)) * probe_t)
    for code in (center_extrema, _bycycle_center_extrema_alt(center_extrema)):
        try:
            bycycle_compute_features(probe_sig, probe_fs, f_theta_probe, center_extrema=code)
            return code
        except ValueError as e:
            if 'center_extrema' not in str(e) and 'extrema' not in str(e).lower():
                raise
    raise ValueError(
        f"Installed bycycle accepts neither {center_extrema!r} nor "
        f"{_bycycle_center_extrema_alt(center_extrema)!r} as center_extrema.")


def _bycycle_burst_kwargs(thresholds):
    """Translate this script's BYCYCLE_THRESHOLDS/BYCYCLE_PAPER_BURST_SETS key
    names (amp_fraction_threshold, amp_consistency_threshold,
    period_consistency_threshold, monotonicity_threshold, min_n_cycles) to the
    bycycle 0.1.x compute_features/detect_bursts_cycles keyword names
    (amplitude_fraction_threshold, amplitude_consistency_threshold,
    period_consistency_threshold, monotonicity_threshold, N_cycles_min)."""
    return dict(
        amplitude_fraction_threshold=thresholds['amp_fraction_threshold'],
        amplitude_consistency_threshold=thresholds['amp_consistency_threshold'],
        period_consistency_threshold=thresholds['period_consistency_threshold'],
        monotonicity_threshold=thresholds['monotonicity_threshold'],
        N_cycles_min=thresholds['min_n_cycles'],
    )


def _min_consecutive_cycles(is_burst, N_cycles_min=3):
    """Zero out any run of consecutive True values shorter than N_cycles_min.
    Vendored from bycycle.burst._min_consecutive_cycles (same logic, operating
    directly on a boolean array) so this script doesn't depend on that private
    helper existing/matching signature across bycycle 0.1.x point-releases."""
    is_burst = np.asarray(is_burst, dtype=bool).copy()
    run_len = 0
    for i, c in enumerate(is_burst):
        if c:
            run_len += 1
        else:
            if run_len < N_cycles_min:
                is_burst[i - run_len:i] = False
            run_len = 0
    if run_len < N_cycles_min:
        is_burst[len(is_burst) - run_len:] = False
    return is_burst


def recompute_is_burst(df_orig, amplitude_fraction_threshold=0,
                       amplitude_consistency_threshold=0,
                       period_consistency_threshold=0,
                       monotonicity_threshold=0, N_cycles_min=3):
    """Determine which cycles are bursting under a given set of burst
    thresholds, reusing the amp_fraction/amp_consistency/period_consistency/
    monotonicity columns bycycle_compute_features() already computed once per
    cycle -- so no bycycle refitting is needed. Identical logic to bycycle's
    own detect_bursts_cycles threshold step (and the Cole_2019_theta paper
    script's own recompute_is_burst()), reimplemented locally -- see the
    module-level bycycle import comment -- so it's used for EVERY threshold
    set, including this pipeline's own primary BYCYCLE_THRESHOLDS, not just
    the paper's BYCYCLE_PAPER_BURST_SETS robustness sweep."""
    df = df_orig
    cycle_good_amp            = df['amp_fraction'] > amplitude_fraction_threshold
    cycle_good_amp_consist    = df['amp_consistency'] > amplitude_consistency_threshold
    cycle_good_period_consist = df['period_consistency'] > period_consistency_threshold
    cycle_good_monotonicity   = df['monotonicity'] > monotonicity_threshold
    is_burst = (cycle_good_amp & cycle_good_amp_consist & cycle_good_period_consist
               & cycle_good_monotonicity).to_numpy(copy=True)
    if len(is_burst):
        is_burst[0]  = False
        is_burst[-1] = False
    is_burst = _min_consecutive_cycles(is_burst, N_cycles_min=N_cycles_min)
    return pd.Series(is_burst, index=df.index)


def compute_bycycle_features(processed_psds, f_theta=None, lowpass_hz=None,
                             filter_seconds=None, apply_highpass=None,
                             highpass_hz=None, highpass_seconds=None,
                             center_extrema=None,
                             thresholds=None, apply_detrend=None,
                             apply_hampel=None, hampel_window_s=None,
                             hampel_sigmas=None, apply_savgol=None,
                             savgol_window_s=None, savgol_polyorder=None,
                             fooof_df=None, use_fooof_theta=None,
                             gate_on_fooof_quality=None):
    """Run bycycle cycle-by-cycle feature extraction on every file's cleaned
    LFP trace in `processed_psds` (the dict loaded from PROCESSED_PSDS_PKL).

    `fooof_df` (expanded_fooof_df from PART 1, one row per file with
    animal/file/has_theta/r_squared/error/theta_cf/theta_bw) wires bycycle to
    FOOOF's frequency-domain fits:
      * if `use_fooof_theta` (default BYCYCLE_USE_FOOOF_THETA), each animal's
        f_theta band comes from compute_fooof_theta_bands() instead of the
        fixed `f_theta`/BYCYCLE_F_THETA;
      * if `gate_on_fooof_quality` (default BYCYCLE_GATE_ON_FOOOF_QUALITY),
        files whose FOOOF fit found no theta peak or falls below the
        R_SQUARED_MIN/ERROR_MAX quality bar are skipped entirely -- bycycle
        only characterizes cycle shape where FOOOF found real evidence of an
        oscillation to characterize.
    Both require `fooof_df`; leave both False to reproduce the old
    fixed-band, ungated behaviour without passing it.

    Mirrors the Cole_2019_theta paper script's own preprocessing (compute_shape():
    lowpass at cf_low=25 Hz THEN highpass at cf_high=1 Hz, both remove_edge_
    artifacts=False, then compute_features) via bycycle_compute_features (the
    same pre-1.0 bycycle.features.compute_features the paper itself used), plus
    three extra passes this pipeline adds, run in order right before the
    lowpass filter:
      1. scipy.signal.detrend (see BYCYCLE_APPLY_DETREND), on top of
         process_animal()'s one-shot whole-trace detrend, to strip local
         drift that otherwise fails bycycle's amplitude/period-consistency
         checks and drops real theta cycles. (Diagnosed via
         diagnose_bycycle_preprocessing() to make close to no difference
         here, since it's the same linear-detrend operation
         process_animal() already ran once -- left in as an off-by-default
         option, see BYCYCLE_APPLY_DETREND.)
      2. hampel_filter() (see BYCYCLE_APPLY_HAMPEL), to flag and
         linearly-interpolate outlier samples (movement/chewing artifacts,
         transient spikes) that would otherwise skew those same
         amplitude/period-consistency reference statistics for neighboring
         genuine theta cycles.
      3. scipy.signal.savgol_filter (see BYCYCLE_APPLY_SAVGOL), run on the
         Hampel-filtered signal to smooth fine jaggedness/interpolation
         artifacts left behind by step 2, before the lowpass filter shapes
         the theta band.
    Every file is filtered and fit individually (not batched through
    BycycleGroup) since files/sessions here have differing recording lengths
    and BycycleGroup's multiprocessing Pool isn't safe to spawn from this
    script without a `__main__` guard.
    Alongside this pipeline's own BYCYCLE_THRESHOLDS burst decision (column
    'is_burst'), the paper's 4-set threshold robustness sweep
    (BYCYCLE_PAPER_BURST_SETS) is also recomputed per cycle -- via
    recompute_is_burst(), reusing the amp_fraction/amp_consistency/
    period_consistency/monotonicity columns already computed once, at no extra
    fitting cost -- and stored as columns 'is_burst_paper0'..'is_burst_paper3'.
    Every cycle's row is tagged with animal/day/session/tetrode/channel
    metadata (parse_metadata_from_path) plus its source file. bycycle 0.1.x's
    compute_features always returns cyclepoint sample-index columns (e.g.
    sample_trough/sample_last_peak/sample_next_peak for trough-centered
    cycles), which plot_bycycle_theta_overview / interactive_bycycle_theta_viewer
    use to locate bursting cycles on the raw trace.

    Returns
    -------
    df_bycycle : pd.DataFrame  one row per detected cycle, every bycycle
                 feature column (including cyclepoint sample indices, and
                 is_burst_paper0..3) plus the metadata columns above
    """
    default_f_theta        = f_theta or BYCYCLE_F_THETA
    lowpass_hz       = lowpass_hz if lowpass_hz is not None else BYCYCLE_LOWPASS_HZ
    filter_seconds   = filter_seconds if filter_seconds is not None else BYCYCLE_FILTER_SECONDS
    apply_highpass   = apply_highpass if apply_highpass is not None else BYCYCLE_APPLY_HIGHPASS
    highpass_hz      = highpass_hz if highpass_hz is not None else BYCYCLE_HIGHPASS_HZ
    highpass_seconds = highpass_seconds if highpass_seconds is not None else BYCYCLE_HIGHPASS_SECONDS
    center_extrema   = center_extrema or BYCYCLE_CENTER_EXTREMA
    thresholds       = thresholds or BYCYCLE_THRESHOLDS
    apply_detrend    = apply_detrend if apply_detrend is not None else BYCYCLE_APPLY_DETREND
    apply_hampel     = apply_hampel if apply_hampel is not None else BYCYCLE_APPLY_HAMPEL
    hampel_window_s  = hampel_window_s if hampel_window_s is not None else BYCYCLE_HAMPEL_WINDOW_S
    hampel_sigmas    = hampel_sigmas if hampel_sigmas is not None else BYCYCLE_HAMPEL_SIGMAS
    apply_savgol     = apply_savgol if apply_savgol is not None else BYCYCLE_APPLY_SAVGOL
    savgol_window_s  = savgol_window_s if savgol_window_s is not None else BYCYCLE_SAVGOL_WINDOW_S
    savgol_polyorder = savgol_polyorder if savgol_polyorder is not None else BYCYCLE_SAVGOL_POLYORDER
    use_fooof_theta       = use_fooof_theta if use_fooof_theta is not None else BYCYCLE_USE_FOOOF_THETA
    gate_on_fooof_quality = gate_on_fooof_quality if gate_on_fooof_quality is not None else BYCYCLE_GATE_ON_FOOOF_QUALITY

    if (use_fooof_theta or gate_on_fooof_quality) and fooof_df is None:
        raise ValueError(
            "use_fooof_theta/gate_on_fooof_quality (or their BYCYCLE_USE_FOOOF_THETA/"
            "BYCYCLE_GATE_ON_FOOOF_QUALITY defaults) require fooof_df (pass "
            "expanded_fooof_df from PART 1) -- or set both False to skip the FOOOF "
            "wiring entirely.")

    animal_theta_bands = (compute_fooof_theta_bands(fooof_df, default_band=default_f_theta)
                          if use_fooof_theta else {})

    good_files = None
    if gate_on_fooof_quality:
        good = fooof_df[fooof_df['has_theta'] &
                        (fooof_df['r_squared'] >= R_SQUARED_MIN) &
                        (fooof_df['error'] <= ERROR_MAX)]
        good_files = set(zip(good['animal'], good['file']))

    extrema_code = _resolve_bycycle_center_extrema(center_extrema, default_f_theta)
    print(f"bycycle: resolved center_extrema={extrema_code!r} for the installed bycycle build")

    cycle_tables = []
    for animal, animal_data in processed_psds.items():
        lfp_store = animal_data.get('lfp_store', {})
        f_theta = animal_theta_bands.get(animal, default_f_theta) if use_fooof_theta else default_f_theta
        print(f"=== bycycle: {animal} ({len(lfp_store)} files)  f_theta={f_theta} ===")
        for rel, file_data in lfp_store.items():
            if gate_on_fooof_quality and (animal, rel) not in good_files:
                print(f"  SKIP: {animal}/{rel} -- no significant FOOOF theta peak, or "
                      f"fit below quality bar (r_squared>={R_SQUARED_MIN}, error<={ERROR_MAX})")
                continue
            fs_file = file_data['fs']
            sig = detrend_signal(file_data['lfp'], dtype=DETREND_TYPE) if apply_detrend \
                else file_data['lfp']
            if apply_hampel:
                half_window = max(1, int(round(fs_file * hampel_window_s / 2)))
                sig, outlier_mask = hampel_filter(sig, half_window, hampel_sigmas)
                n_outliers = int(outlier_mask.sum())
                if n_outliers:
                    print(f"    Hampel: interpolated {n_outliers}/{len(sig)} "
                          f"outlier samples ({100 * n_outliers / len(sig):.2f}%)")
            if apply_savgol:
                savgol_win = max(savgol_polyorder + 2, int(round(fs_file * savgol_window_s)))
                if savgol_win % 2 == 0:
                    savgol_win += 1
                sig = savgol_filter(sig, savgol_win, savgol_polyorder)
            lp = filter_signal(sig, fs_file, 'lowpass', lowpass_hz,
                               n_seconds=filter_seconds, remove_edges=False)
            if apply_highpass:
                # Paper order (compute_shape()): lowpass THEN highpass -- the
                # highpass removes drift left over after the lowpass step.
                lp = filter_signal(lp, fs_file, 'highpass', highpass_hz,
                                   n_seconds=highpass_seconds, remove_edges=False)

            # burst_detection_kwargs/threshold_kwargs deliberately NOT passed here:
            # the amp_fraction/amp_consistency/period_consistency/monotonicity columns
            # compute_features computes don't depend on the threshold values at all
            # (only its own internal is_burst does, and that gets overwritten
            # immediately below) -- so every threshold set, including this pipeline's
            # own BYCYCLE_THRESHOLDS, is applied uniformly via recompute_is_burst()
            # instead. This also avoids relying on compute_features accepting any
            # particular threshold-kwarg name, which differs across bycycle builds
            # (burst_detection_kwargs vs. threshold_kwargs). extrema_code was
            # resolved once above, before this loop.
            df_cycles = bycycle_compute_features(
                np.asarray(lp), fs_file, f_theta, center_extrema=extrema_code)

            if df_cycles is None or len(df_cycles) == 0:
                print(f"  SKIP: {animal}/{rel} -- 0 cycles detected")
                continue

            df_cycles['is_burst'] = recompute_is_burst(df_cycles, **_bycycle_burst_kwargs(thresholds))
            for i_set, burst_set in enumerate(BYCYCLE_PAPER_BURST_SETS):
                df_cycles[f'is_burst_paper{i_set}'] = recompute_is_burst(
                    df_cycles, **_bycycle_burst_kwargs(burst_set))

            meta = parse_metadata_from_path(rel)
            df_cycles = df_cycles.copy()
            df_cycles['animal']  = animal
            df_cycles['file']    = rel
            df_cycles['day']     = meta['date']
            df_cycles['session'] = meta['session']
            df_cycles['tetrode'] = meta['tetrode']
            df_cycles['channel'] = meta['channel']
            n_bursts = int(df_cycles['is_burst'].sum()) if 'is_burst' in df_cycles else len(df_cycles)
            print(f"  OK: {animal}/{rel}  [{len(df_cycles)} cycles, {n_bursts} in bursts]")
            cycle_tables.append(df_cycles)

    if not cycle_tables:
        raise ValueError(
            "bycycle detected zero cycles across every file in PROCESSED_PSDS_PKL "
            "(or every file was skipped by gate_on_fooof_quality). Check "
            "BYCYCLE_F_THETA/animal_theta_bands, BYCYCLE_GATE_ON_FOOOF_QUALITY, "
            "and the input LFP traces.")

    df_bycycle = pd.concat(cycle_tables, ignore_index=True)
    cols  = df_bycycle.columns.tolist()
    front = ['animal', 'day', 'session', 'tetrode', 'channel', 'file']
    df_bycycle = df_bycycle[front + [c for c in cols if c not in front]]
    return df_bycycle


def _windowed_mean_variance(x, fs_hz, window_s):
    """Variance of the per-window MEAN across non-overlapping `window_s`-second
    windows -- a proxy for slow baseline drift that global variance can't
    isolate. A signal with no drift has a flat local mean across windows
    (this stays near 0); a signal that still wanders keeps a large value here
    even after a global linear detrend has already zeroed out its single
    best-fit trend (see diagnose_bycycle_preprocessing's docstring)."""
    win = int(round(fs_hz * window_s))
    n_win = len(x) // win
    if n_win < 2:
        return np.nan
    means = x[:n_win * win].reshape(n_win, win).mean(axis=1)
    return float(np.var(means))


def diagnose_bycycle_preprocessing(processed_psds, apply_detrend=None, apply_hampel=None,
                                   hampel_window_s=None, hampel_sigmas=None,
                                   drift_window_s=5.0, plot_window_s=30.0,
                                   save_dir=None, save=True):
    """Quantify + plot what compute_bycycle_features()'s detrend and Hampel
    steps actually do to each file's trace, so their effectiveness can be
    checked directly instead of assumed.

    IMPORTANT: BYCYCLE_APPLY_DETREND runs scipy.signal.detrend(type='linear')
    on a trace process_animal() ALREADY ran the exact same operation on once
    (see APPLY_TIME_DETREND). A single linear detrend removes the ONE
    best-fit line for the whole trace; running it again on an already
    globally-detrended signal is close to a no-op (that signal's best-fit
    line is already ~0), so it does nothing for drift that isn't one straight
    line across the whole file -- which is most real drift. Check the
    'drift_*' columns this function returns: if drift_detrend isn't much
    lower than drift_raw, that confirms the second pass isn't helping. Fix:
    a piecewise/breakpoint linear detrend (scipy.signal.detrend's `bp` arg,
    with breakpoints every N seconds) or a highpass filter (e.g.
    neurodsp.filt.filter_signal(..., 'highpass', 0.5)) -- either removes
    drift on shorter timescales than one straight line can.

    For every file (raw = lfp_store's trace, i.e. process_animal()'s
    notch-filtered + one global linear detrend, BEFORE any bycycle-specific
    step), runs the same steps compute_bycycle_features() runs (local-pass
    detrend, then Hampel outlier interpolation) and computes at each stage:
      * global variance (np.var)                    -- overall signal spread
      * drift variance (_windowed_mean_variance)     -- local-mean wander
    plus the Hampel outlier count/fraction.

    Saves one PNG per file under `save_dir` (default
    BYCYCLE_FIGURE_DIR/preprocessing_qc/<animal>/<file>_preproc_qc.png):
    raw-vs-after-detrend and after-detrend-vs-after-hampel trace overlays
    (first `plot_window_s` seconds only, for legibility -- the variance/drift
    numbers themselves are always computed over the FULL trace), plus a bar
    chart of the variance/drift numbers.

    Returns
    -------
    df_qc : pd.DataFrame  one row per file: animal, file, var_raw/var_detrend/
            var_hampel, drift_raw/drift_detrend/drift_hampel, n_outliers,
            frac_outliers. Get dataset-wide effectiveness with e.g.
            df_qc[['var_raw','var_detrend','var_hampel']].mean().
    """
    apply_detrend   = apply_detrend if apply_detrend is not None else BYCYCLE_APPLY_DETREND
    apply_hampel    = apply_hampel if apply_hampel is not None else BYCYCLE_APPLY_HAMPEL
    hampel_window_s = hampel_window_s if hampel_window_s is not None else BYCYCLE_HAMPEL_WINDOW_S
    hampel_sigmas   = hampel_sigmas if hampel_sigmas is not None else BYCYCLE_HAMPEL_SIGMAS

    out_dir = save_dir or os.path.join(BYCYCLE_FIGURE_DIR, 'preprocessing_qc')
    if save:
        os.makedirs(out_dir, exist_ok=True)

    rows = []
    for animal, animal_data in processed_psds.items():
        lfp_store = animal_data.get('lfp_store', {})
        print(f"=== bycycle preprocessing QC: {animal} ({len(lfp_store)} files) ===")
        for rel, file_data in lfp_store.items():
            fs_file = file_data['fs']
            raw = np.asarray(file_data['lfp'], dtype=np.float64)

            detrended = detrend_signal(raw, dtype=DETREND_TYPE) if apply_detrend else raw

            if apply_hampel:
                half_window = max(1, int(round(fs_file * hampel_window_s / 2)))
                hampeled, outlier_mask = hampel_filter(detrended, half_window, hampel_sigmas)
            else:
                hampeled, outlier_mask = detrended, np.zeros(len(detrended), dtype=bool)

            var_raw       = float(np.var(raw))
            var_detrend   = float(np.var(detrended))
            var_hampel    = float(np.var(hampeled))
            drift_raw     = _windowed_mean_variance(raw, fs_file, drift_window_s)
            drift_detrend = _windowed_mean_variance(detrended, fs_file, drift_window_s)
            drift_hampel  = _windowed_mean_variance(hampeled, fs_file, drift_window_s)
            n_outliers    = int(outlier_mask.sum())

            rows.append({
                'animal': animal, 'file': rel,
                'var_raw': var_raw, 'var_detrend': var_detrend, 'var_hampel': var_hampel,
                'drift_raw': drift_raw, 'drift_detrend': drift_detrend, 'drift_hampel': drift_hampel,
                'n_outliers': n_outliers, 'frac_outliers': n_outliers / len(hampeled),
            })

            print(f"  {animal}/{rel}: var {var_raw:.2f} -> {var_detrend:.2f} -> {var_hampel:.2f} uV^2 | "
                  f"drift {drift_raw:.2f} -> {drift_detrend:.2f} -> {drift_hampel:.2f} uV^2 | "
                  f"{n_outliers} outliers ({100 * n_outliers / len(hampeled):.2f}%)")

            if save:
                n_plot = min(len(raw), int(round(plot_window_s * fs_file)))
                t = np.arange(n_plot) / fs_file

                fig, axes = plt.subplots(3, 1, figsize=(14, 9))

                axes[0].plot(t, raw[:n_plot], color='#888888', lw=0.7,
                            label='raw (notch + global detrend)')
                axes[0].plot(t, detrended[:n_plot], color='#1A56DB', lw=0.7, alpha=0.85,
                            label='after bycycle detrend')
                axes[0].set_title(f'Detrend: raw vs. after (first {plot_window_s:g}s)', fontsize=9)
                axes[0].legend(fontsize=7, loc='upper right')
                axes[0].set_ylabel('µV')

                axes[1].plot(t, detrended[:n_plot], color='#1A56DB', lw=0.7,
                            label='before Hampel (after detrend)')
                axes[1].plot(t, hampeled[:n_plot], color='#4DAF4A', lw=0.7, alpha=0.85,
                            label='after Hampel')
                mask_plot = outlier_mask[:n_plot]
                if mask_plot.any():
                    axes[1].scatter(t[mask_plot], detrended[:n_plot][mask_plot],
                                    color='#E41A1C', s=10, zorder=5,
                                    label=f'flagged outliers (n={n_outliers} in full file)')
                axes[1].set_title(f'Hampel: before vs. after (first {plot_window_s:g}s)', fontsize=9)
                axes[1].legend(fontsize=7, loc='upper right')
                axes[1].set_ylabel('µV')
                axes[1].set_xlabel('Time (s)')

                bar_labels = ['raw', 'after\ndetrend', 'after\nhampel']
                var_vals   = [var_raw, var_detrend, var_hampel]
                drift_vals = [drift_raw, drift_detrend, drift_hampel]
                x = np.arange(3)
                w = 0.35
                ax2  = axes[2]
                ax2b = ax2.twinx()
                ax2.bar(x - w / 2, var_vals, width=w, color='#1A56DB', label='global variance (full file)')
                ax2b.bar(x + w / 2, drift_vals, width=w, color='#E41A1C',
                        label=f'drift variance ({drift_window_s:g}s windows, full file)')
                ax2.set_xticks(x)
                ax2.set_xticklabels(bar_labels, fontsize=8)
                ax2.set_ylabel('Global variance (µV²)', color='#1A56DB')
                ax2b.set_ylabel('Drift variance (µV²)', color='#E41A1C')
                ax2.set_title('Variance summary (full file)', fontsize=9)
                h1, l1 = ax2.get_legend_handles_labels()
                h2, l2 = ax2b.get_legend_handles_labels()
                ax2.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper right')

                for ax in axes[:2]:
                    ax.spines[['top', 'right']].set_visible(False)

                fig.suptitle(f'{animal}: {rel}', fontsize=10)
                plt.tight_layout()

                stem = re.sub(r'[\\/]+', '_', os.path.splitext(rel)[0])
                animal_dir = os.path.join(out_dir, animal)
                os.makedirs(animal_dir, exist_ok=True)
                fig.savefig(os.path.join(animal_dir, f'{stem}_preproc_qc.png'),
                           dpi=200, bbox_inches='tight')
                plt.close(fig)

    df_qc = pd.DataFrame(rows)
    if save:
        print(f"\nSaved {len(df_qc)} QC figures -> {out_dir}")
    print("\nMean across all files:")
    print(df_qc[['var_raw', 'var_detrend', 'var_hampel',
                 'drift_raw', 'drift_detrend', 'drift_hampel',
                 'frac_outliers']].mean().to_string())
    return df_qc


def plot_bycycle_sample_traces(processed_psds, animals_list=None, window_s=2.0,
                               save=True, save_dir=None):
    """Plot a short window of each animal's raw LFP trace (first file) --
    mirrors the tutorial's CA1/EC3 time-series panel (plot_time_series)."""
    animals_list = animals_list or list(processed_psds.keys())

    fig, axes = plt.subplots(len(animals_list), 1,
                             figsize=(15, 3 * len(animals_list)), squeeze=False)
    axes = axes[:, 0]

    for i, (ax, animal) in enumerate(zip(axes, animals_list)):
        lfp_store = processed_psds[animal]['lfp_store']
        rel = next(iter(lfp_store))
        lfp, fs_file = lfp_store[rel]['lfp'], lfp_store[rel]['fs']
        n = int(window_s * fs_file)
        times = np.arange(n) / fs_file
        plot_time_series(times, lfp[:n], ax=ax,
                         colors=ANIMAL_COLORS[i % len(ANIMAL_COLORS)],
                         xlim=(0, window_s), xlabel='Time (s)',
                         ylabel=f'{animal} Voltage (µV)')
        ax.set_title(os.path.basename(rel), fontsize=8)

    fig.suptitle('bycycle: sample LFP trace (first file per animal)', fontsize=10)
    plt.tight_layout()
    if save:
        out_dir = save_dir or BYCYCLE_FIGURE_DIR
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, 'bycycle_sample_traces.svg'),
                   dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)
    return fig


# ── bycycle analogs of PART 2's plot_theta_overview / interactive_theta_viewer
# (defined above PART 2, around _CAT_COLORS/_plot_categorical_line) -- same
# red-burst/black-non-burst rendering, driven by bycycle's is_burst column and
# per-cycle boundary sample indices instead of fBOSC's episode Onset/Offset. ──

_BYCYCLE_SAMPLE_BOUND_COLS = [
    ('sample_last_trough', 'sample_next_trough'),  # center_extrema='peak'
    ('sample_last_peak',   'sample_next_peak'),    # center_extrema='trough'
]


def _bycycle_sample_bound_cols(df):
    """Return whichever (start, end) cyclepoint sample-column pair is present
    in `df` -- the names depend on BYCYCLE_CENTER_EXTREMA ('peak' vs 'trough')."""
    for c0, c1 in _BYCYCLE_SAMPLE_BOUND_COLS:
        if c0 in df.columns and c1 in df.columns:
            return c0, c1
    raise KeyError(
        "No cycle-boundary sample columns found in df_bycycle -- expected one "
        f"of {_BYCYCLE_SAMPLE_BOUND_COLS}. Was Bycycle fit with "
        f"return_samples=True? Available columns: {list(df.columns)}")


def _bycycle_burst_mask(n_samples, df_file_cycles):
    """Per-sample category array (0/2, reusing _CAT_COLORS/_CAT_LABELS from
    PART 2) marking samples inside a bycycle-detected bursting theta cycle
    (2 = red / 'significant theta') vs. everything else (0 = black /
    'non-theta'). `df_file_cycles` must already be filtered to one file's rows."""
    cat = np.zeros(n_samples, dtype=np.int8)
    if df_file_cycles is None or len(df_file_cycles) == 0:
        return cat
    bursts = df_file_cycles[df_file_cycles['is_burst'].astype(bool)]
    if len(bursts) == 0:
        return cat
    c0, c1 = _bycycle_sample_bound_cols(df_file_cycles)
    for start, end in zip(bursts[c0].to_numpy(), bursts[c1].to_numpy()):
        i0 = max(int(start), 0)
        i1 = min(int(end), n_samples)
        if i1 > i0:
            cat[i0:i1] = 2
    return cat


def plot_bycycle_theta_overview(lfp, fs, df_file_cycles, bin_s=0.05, figsize=(16, 3),
                                save_path=None, title=None):
    """Static bird's-eye-view figure of the ENTIRE trace: min/max envelope per
    `bin_s`-second bin, colored red where the bin overlaps a bycycle-detected
    bursting theta cycle, black otherwise. bycycle analog of plot_theta_overview
    (PART 2) -- use this to save one full-trace figure per recording."""
    lfp = np.asarray(lfp, dtype=np.float64)
    n = len(lfp)
    cat = _bycycle_burst_mask(n, df_file_cycles)

    bin_n = max(1, int(round(bin_s * fs)))
    n_bins = n // bin_n
    if n_bins == 0:
        raise ValueError('Trace shorter than one bin_s bin.')
    trimmed  = lfp[:n_bins * bin_n].reshape(n_bins, bin_n)
    cat_trim = cat[:n_bins * bin_n].reshape(n_bins, bin_n)
    lo = trimmed.min(axis=1)
    hi = trimmed.max(axis=1)
    bin_cat = cat_trim.max(axis=1)  # priority: bursting > non-bursting
    t = (np.arange(n_bins) * bin_n) / fs

    fig, ax = plt.subplots(figsize=figsize)
    for c in (0, 2):
        sel = bin_cat == c
        if not sel.any():
            continue
        ax.vlines(t[sel], lo[sel], hi[sel], colors=_CAT_COLORS[c],
                  linewidth=1.0, label=_CAT_LABELS[c])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('LFP (µV)')
    ax.set_xlim(t[0], t[-1] + bin_s)
    pct_burst = 100 * (bin_cat == 2).mean()
    ax.set_title(title or f'Full-signal bycycle theta overview  ({pct_burst:.1f}% bursting)')
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved -> {save_path}')
    else:
        plt.show()
    return fig, ax


def interactive_bycycle_theta_viewer(lfp, fs, df_file_cycles, window_s=10.0, title=None):
    """Scroll through the whole trace in a fixed-width window via a slider
    (plus <</>> buttons and left/right arrow keys to step by 1/4-window), with
    bycycle-detected bursting theta cycles in red and everything else in black.
    bycycle analog of interactive_theta_viewer (PART 2).

    Requires an interactive matplotlib backend (e.g. `%matplotlib qt` in a
    notebook, or plain `python script.py` with a Tk/Qt backend installed) --
    with a static/inline backend the window will just show the first frame.
    """
    from matplotlib.widgets import Slider, Button

    lfp = np.asarray(lfp, dtype=np.float64)
    n = len(lfp)
    dur = n / fs
    cat_full = _bycycle_burst_mask(n, df_file_cycles)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    plt.subplots_adjust(bottom=0.25)
    state = {'t0': 0.0}

    def draw(t0):
        t0 = float(np.clip(t0, 0, max(0.0, dur - window_s)))
        state['t0'] = t0
        i0 = int(round(t0 * fs))
        i1 = min(n, int(round((t0 + window_s) * fs)))
        t = np.arange(i0, i1) / fs
        x = lfp[i0:i1]
        cat = cat_full[i0:i1]
        ax.clear()
        _plot_categorical_line(ax, t, x, cat)
        ax.set_xlim(t0, t0 + window_s)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('LFP (µV)')
        ax.set_title(title or 'bycycle theta cycles (red = bursting)', fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc='upper right')
        ax.spines[['top', 'right']].set_visible(False)
        fig.canvas.draw_idle()

    ax_slider = fig.add_axes([0.15, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Start (s)', 0, max(0.001, dur - window_s), valinit=0)
    slider.on_changed(draw)

    ax_prev = fig.add_axes([0.78, 0.09, 0.05, 0.05])
    ax_next = fig.add_axes([0.85, 0.09, 0.05, 0.05])
    b_prev = Button(ax_prev, '<<')
    b_next = Button(ax_next, '>>')

    def step(delta):
        slider.set_val(float(np.clip(state['t0'] + delta, 0, max(0.0, dur - window_s))))

    b_prev.on_clicked(lambda evt: step(-window_s))
    b_next.on_clicked(lambda evt: step(window_s))

    def on_key(event):
        if event.key == 'right':
            step(window_s * 0.25)
        elif event.key == 'left':
            step(-window_s * 0.25)
    fig.canvas.mpl_connect('key_press_event', on_key)

    draw(0)
    # Block here (bypassing the module-wide non-blocking plt.show() patch, see
    # top of file) so the window -- and its slider/buttons -- stays open for
    # interactive exploration until the user closes it by hand.
    _plt_show(block=True)
    return fig, ax, slider


def plot_bycycle_feature_distributions(df_bycycle_cycles, fs, props=None,
                                       save=True, save_dir=None):
    """2x2 histogram panel of cycle amplitude / period / rise-decay symmetry /
    peak-trough symmetry, one color per animal, restricted to bursting cycles
    -- generalizes the tutorial's CA1-vs-EC3 comparison to however many
    animals are in ANIMALS."""
    df = df_bycycle_cycles.copy()
    df['period_ms'] = df['period'] / fs * 1000

    selected     = props or list(BYCYCLE_PROPS.keys())
    animals_here = sorted(df['animal'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes_flat = axes.flatten()

    for ax, key in zip(axes_flat, selected):
        meta     = BYCYCLE_PROPS[key]
        vals_all = df[key].dropna()
        xlim     = meta['xlim'] or (vals_all.min(), vals_all.max())
        bins     = np.linspace(xlim[0], xlim[1], 31)

        for i, animal in enumerate(animals_here):
            vals = df.loc[df['animal'] == animal, key].dropna()
            ax.hist(vals, bins=bins, alpha=0.55, density=True,
                   color=ANIMAL_COLORS[i % len(ANIMAL_COLORS)],
                   edgecolor='white', lw=0.5, label=animal)

        ax.set_xlabel(meta['xlabel'])
        ax.set_xlim(xlim)
        ax.set_ylabel('Density')
        ax.legend(fontsize=8, frameon=False)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Theta cycle feature distributions (bycycle, bursting cycles only)',
                 fontsize=11)
    plt.tight_layout()

    if save:
        out_dir = save_dir or BYCYCLE_FIGURE_DIR
        os.makedirs(out_dir, exist_ok=True)
        for ext in ('png', 'svg'):
            fig.savefig(os.path.join(out_dir, f'bycycle_feature_distributions.{ext}'),
                       dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)
    return fig


# ---------- CHECKPOINT: bycycle analysis ----------
_run_bycycle = _ask_yes_no(
    "Would you like to perform the bycycle (theta cycle-feature) analysis?")

if not _run_bycycle:
    print("Skipping bycycle analysis section.")
else:
    if not os.path.exists(PROCESSED_PSDS_PKL):
        raise FileNotFoundError(
            f"No processed PSDs found: {PROCESSED_PSDS_PKL} does not exist.\n"
            "Run PART 1 of this script (the FOOOF/PSD pipeline above) first "
            "-- it saves the cleaned LFP traces bycycle needs.")

    with open(PROCESSED_PSDS_PKL, 'rb') as fh:
        processed_psds = pickle.load(fh)

    # ---------- CHECKPOINT: bycycle preprocessing QC ----------
    _run_bycycle_qc = _ask_yes_no(
        "Would you like to run bycycle preprocessing QC first (per-file "
        "detrend/Hampel before-vs-after plots + variance summary)?")
    if _run_bycycle_qc:
        df_bycycle_qc = diagnose_bycycle_preprocessing(processed_psds)

    if BYCYCLE_USE_CACHE and os.path.exists(BYCYCLE_FEATURES_PKL):
        print(f"Loading cached bycycle feature table -> {BYCYCLE_FEATURES_PKL}")
        df_bycycle = pd.read_pickle(BYCYCLE_FEATURES_PKL)
        try:
            _bycycle_sample_bound_cols(df_bycycle)
        except KeyError:
            print("Cached bycycle table predates return_samples=True (no cyclepoint "
                 "sample columns) -- recomputing so burst locations can be plotted.")
            df_bycycle = compute_bycycle_features(processed_psds, fooof_df=expanded_fooof_df)
            df_bycycle.to_pickle(BYCYCLE_FEATURES_PKL)
            print(f"Saved bycycle feature table -> {BYCYCLE_FEATURES_PKL}")
    else:
        df_bycycle = compute_bycycle_features(processed_psds, fooof_df=expanded_fooof_df)
        os.makedirs(BYCYCLE_OUTPUT_DIR, exist_ok=True)
        df_bycycle.to_pickle(BYCYCLE_FEATURES_PKL)
        print(f"Saved bycycle feature table -> {BYCYCLE_FEATURES_PKL}")

    # Limit distribution plots to oscillatory bursts, as in the tutorial
    # (df_ca1_cycles = df_ca1[df_ca1['is_burst']]).
    df_bycycle_cycles = df_bycycle[df_bycycle['is_burst']]
    print(f"{len(df_bycycle)} cycles detected total, "
         f"{len(df_bycycle_cycles)} "
         f"({100 * len(df_bycycle_cycles) / len(df_bycycle):.1f}%) in oscillatory bursts")

    os.makedirs(BYCYCLE_FIGURE_DIR, exist_ok=True)
    plot_bycycle_sample_traces(processed_psds, save=True)
    plot_bycycle_feature_distributions(df_bycycle_cycles, fs=fs_down, save=True)

    print(df_bycycle_cycles.groupby('animal')[
        ['volt_amp', 'period', 'time_rdsym', 'time_ptsym']].mean())

    # Full-trace theta-burst overview (red = bycycle-detected bursting cycle,
    # black = everything else), saved for every file that was analyzed.
    BYCYCLE_OVERVIEW_DIR = os.path.join(BYCYCLE_FIGURE_DIR, 'theta_overview')
    os.makedirs(BYCYCLE_OVERVIEW_DIR, exist_ok=True)
    _n_overview_saved = 0
    for animal, animal_data in processed_psds.items():
        lfp_store = animal_data.get('lfp_store', {})
        for rel, file_data in lfp_store.items():
            df_file = df_bycycle[(df_bycycle['animal'] == animal) &
                                 (df_bycycle['file'] == rel)]
            if len(df_file) == 0:
                continue  # e.g. 0 cycles detected for this file
            safe_name = re.sub(r'[\\/]+', '_', os.path.splitext(rel)[0])
            plot_bycycle_theta_overview(
                file_data['lfp'], file_data['fs'], df_file,
                title=f"{animal}: {rel}",
                save_path=os.path.join(BYCYCLE_OVERVIEW_DIR, f"{animal}__{safe_name}.png"))
            _n_overview_saved += 1
    print(f"Saved {_n_overview_saved} bycycle theta overview figures -> {BYCYCLE_OVERVIEW_DIR}")

    # ---------- CHECKPOINT: interactive bycycle viewer ----------
    _run_bycycle_interactive = _ask_yes_no(
        "Open the interactive sliding-window bycycle viewer, for one random "
        "file? (needs an interactive matplotlib backend, e.g. %matplotlib qt)")
    if _run_bycycle_interactive:
        # Opening several of these Tk-slider windows back-to-back (one per
        # animal) via the non-blocking plt.show() patch above triggers a
        # Windows-specific RecursionError in matplotlib's Tk foreground-window
        # restore code, so show only one random file instead.
        import random
        animal = random.choice(list(processed_psds.keys()))
        lfp_store = processed_psds[animal].get('lfp_store', {})
        rel = random.choice(list(lfp_store.keys()))
        df_file = df_bycycle[(df_bycycle['animal'] == animal) &
                             (df_bycycle['file'] == rel)]
        interactive_bycycle_theta_viewer(
            lfp_store[rel]['lfp'], lfp_store[rel]['fs'], df_file,
            window_s=10.0, title=f"{animal}: {rel}")