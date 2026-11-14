# -*- coding: utf-8 -*-
"""
Theta modulation of place cells, and the theta phase of their peak-bin, firing-onset and
firing-offset spikes. Built from ThetaMod_PhasePrec_v31_OnsetOffsetPure.py (LFP cleaning, Hilbert
phase, ACG theta epochs, Rayleigh and TMI tests, figure style) and
ThetaMod_PhasePrec_v30_FiringNormPassIndex.py (place fields and their traversals). There is no
phase-precession fit and no pass index.

Per unit (one .ntt file = one already-isolated unit):

  1. Theta phase. The LFP is loaded onto a timestamp-anchored grid (ncs_repair.py), cleaned as in
     Figc_ACG_ScalingCorrected_v8_NoiseCleaning.py (clean_lfp: resampled to 1 kHz, 50/100/150/200 Hz
     notched, wavelet denoised, detrended, outliers filled, 100 Hz low-passed), bandpass filtered
     (LFP_FILTER_BAND) and Hilbert transformed: 0 deg at a theta peak, 180 deg at a trough. Only
     spikes in ACG theta epochs (Dunn et al. 2022; USE_ACG_THETA_EPOCHS) enter any phase below: 1 s
     epochs that are not within SATURATION_MARGIN_S of ADC clipping (+/-InputRange from each file's
     header, as Figc_ACG_ScalingCorrected_v11_SaturationReject.py), pass the noise rejection (delta >
     theta power, peak-to-peak MAD outlier) and whose
     closest reference sinusoid, matched over a 1-20 Hz bank, is in 3-7 Hz (v8's broadband fitting;
     no ED_min threshold). Noise and out-of-band epochs are thrown out.
  2. Theta modulation, on the ACG theta-epoch spikes with an LFP phase (>= MIN_SPIKES_FOR_TMI):
       Rayleigh+  Rayleigh test p < ALPHA                                      (Rayleigh_Significant)
       TMI+       TMI (Frank et al. 2001), burst-preserving shuffle p < ALPHA  (TMI_Significant)
       Both+      Rayleigh+ AND TMI+ = theta-modulated                         (ThetaModulated)
     Units with too few spikes are negative on all three.
  3. Place fields and the raw peak bin (needs the session's tracking file). Rate map and fields as
     in v30: PlaceCell_Main_v20's threshold method on Climer's rate map (sigma = SMTH_WIDTH / 2),
     from all of the unit's spikes while moving (>= RATEMAP_IMMOBILITY_SPEED_CM_S); a 1-D angular
     map on the Circle track. Multi-peak fields are not split. Peak bin = the valid bin (visited >=
     FIELD_MIN_VISITS times) with the highest rate in the RAW map (spikes / occupancy, unsmoothed).
     Main field = the field holding the raw peak bin; if the raw peak bin is outside every field, the
     field holding the peak bin of the smoothed map (main_field_source); none if that is outside every
     field too. Every field-based analysis (the laps, onset and offset) uses only the main field.
  4. Analysed spikes = ACG theta-epoch spikes with an LFP phase, fired at smoothed speed >= MIN_SPEED_CM_S.
       Peak-bin phase  circular mean phase of every analysed spike fired in the raw peak bin over the
                       whole session (bin from the raw position, as in the raw map).
       Laps            traversals of the main field, as v30's field_traversals: one contiguous run of
                       moving, jitter-smoothed position samples inside the field; runs <= MERGE_GAP_S
                       apart merged; runs shorter than LAP_MIN_DURATION_S, or cut by the start / end
                       of tracking, dropped. Laps with >= MIN_SPIKES_PER_LAP analysed spikes are kept.
       Onset phase     phase of the first analysed spike of each kept lap; the unit's value is the
                       circular mean over its kept laps.
       Offset phase    phase of the last analysed spike of each kept lap, likewise.
       Phase span      (v3) units with >= MIN_PEAK_BIN_SPIKES_FOR_SPAN peak-bin spikes: their phases are
                       cut at the largest circular gap, span = 360 deg - that gap, and each phase is
                       re-measured from the first spike after the gap (0 deg = start of the span, so
                       the span does not depend on where on the theta cycle it sits). One Gaussian-
                       kernel density of the re-measured phases per unit (SPAN_KDE_SIGMA_DEG).
     A unit with no main field gets a peak-bin phase (and span) but no onset / offset.

Comparisons (one value per unit; units treated as independent, as in v31 and the 'KW' mode of
ArenaSessionType_StatsComparison_v18.py), at that script's three levels:
  1. Session types within each arena (ARENA_SESSION_TYPES)          RESULTS_FOLDER\\SessionType\\<arena>
  2. Arena types, session types pooled                               RESULTS_FOLDER\\ArenaType
  3. Arena types, only the animals recorded in all compared arenas   RESULTS_FOLDER\\AnimalControlledArenaType
     (ANIMAL_CONTROLLED_COMPARISONS), the comparisons side by side in one image, each with its own
     statistics (they are not tested against each other)
  Theta modulation: % of ALL units that are Rayleigh+, TMI+ and Both+ per group. The bars overlap: a
    Both+ unit counts in all three. Tested on Both+ only: chi-square test of independence, pairwise
    chi-square (Yates for 2 x 2), Holm-Bonferroni within the comparison. AllUnits_ThetaModProportion.png
    pools every unit. <prefix>_ThetaModProportion_Tested.png: % of TESTED units (>= MIN_SPIKES_FOR_TMI
    theta-epoch spikes) that are Both+, the same tests ('*_TMTested_Omni' / '_Pair' sheets).
    Across animals (arenas and session types pooled within each animal): the same two plots and tests,
    RESULTS_FOLDER\\ANIMAL_DIR\\Animal_ThetaModProportion(_Tested).png, 'Animal_*' sheets.
  Peak-bin, onset and offset phase: once for all units, once for theta-modulated (Both+) units only.
    Mardia-Watson-Wheeler (permutation p) and Watson-Williams, omnibus + Holm-corrected pairwise.
    Points coloured by theta trough range (THETA_TROUGH_RANGE_DEG) / theta peak range (the rest).
  Theta-modulated vs non-theta-modulated units: within each group (and all units pooled), the same
    three phases of the two kinds of units, one two-sample Watson-Williams test (circular t-test) per
    group, Holm-Bonferroni over the groups of the comparison; <prefix>_PhaseAt<metric>_TMvsNonTM.png,
    '*_<metric>_TMvsNon' sheets. NON_TM_INCLUDES_UNTESTED: whether units too sparse to test count as
    non-theta-modulated.
  Peak-bin phase span (v3), all units and theta-modulated units: per group the units' densities and
    their mean +/- SEM, and a box plot of the spans; <prefix>_PeakBinPhaseSpan_<cell set>.png,
    '*_PkSpan_*' sheets. Density curves: permutation test of the units' group labels (statistic
    T = sum_g n_g * integral (group mean density - grand mean)^2), omnibus + pairwise; spans:
    Kruskal-Wallis + pairwise Mann-Whitney U; Holm per test. Spans grow with the number of spikes.

v3 changes from v2: main-field fallback to the smoothed-map peak bin; the peak-bin phase span analysis;
ADC-saturated epochs thrown out (ACG_REJECT_SATURATED_EPOCHS); proportions compared across animals.

Output: <session>\\SESSION_OUTPUT_FOLDER\\ per-unit figures (PolarPlot, PhaseHistogram, TMIShuffle,
PeakOnsetOffset); the PolarPlot / PhaseHistogram / TMIShuffle figures are also copied to
RESULTS_FOLDER\\<class folder> (THETA_MOD_CLASS_FOLDERS: ThetaModulated, TMI, Rayleigh,
NonThetaModulated). RESULTS_FOLDER: OUTPUT_EXCEL_NAME (README, Units = one row per unit, Summary, and the
statistics sheets 'All_*', 'Ses<arena>_*', 'Arena_*', 'AC<A><B>_*'), the same statistics in one
long-format CSV, the ACG epoch QC plot / table and a metadata CSV.

Requires: numpy, scipy, pandas, matplotlib, pillow, openpyxl, pywavelets, and ncs_repair.py (NCS_REPAIR_DIR).
"""

from __future__ import annotations

import functools
import re
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
from scipy import signal, stats
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter, gaussian_filter1d, distance_transform_edt, label as nd_label

# Timestamp-anchored .ncs loading: ncs_repair.py corrects the partial-record and sample-loss timing
# error of the raw .ncs files (see its header).
NCS_REPAIR_DIR = r'C:\Runita\NMR\Mole-Rat_PlaceCell\Codes\Theta\Codes\ncsDropCorr'
if NCS_REPAIR_DIR not in sys.path:
    sys.path.insert(0, NCS_REPAIR_DIR)
from ncs_repair import load_repaired_uv, record_lengths, find_runs  # noqa: E402
import PlotLayout_Utils as plu  # noqa: E402  shared y-range of side-by-side plots, % labels on bars


# ============================================================================
# Configuration -- EDIT THESE
# ============================================================================
ROOT_FOLDER = Path(r"X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final")
# All pooled results (Excel/CSV files, comparison plots) go here
RESULTS_FOLDER = ROOT_FOLDER / 'Results' / 'ThetaMod' / 'ThetaMod_Pure_v3'
OUTPUT_EXCEL_NAME = 'ThetaMod_Pure_v3.xlsx'          # written to RESULTS_FOLDER
STATS_CSV_NAME = 'ThetaMod_Pure_v3_AllStats.csv'     # every statistics table, long format, in RESULTS_FOLDER
SESSION_OUTPUT_FOLDER = 'ThetaMod_Pure_v3'           # per-unit figures, in each session folder
# ThetaModClass -> RESULTS_FOLDER sub-folder that gets a copy of the unit's Rayleigh / TMI figures
# (PolarPlot, PhaseHistogram, TMIShuffle). Units with too few spikes have no such figures.
THETA_MOD_CLASS_FOLDERS = {
    'Rayleigh+ & TMI+': 'ThetaModulated',
    'TMI+ only': 'TMI',
    'Rayleigh+ only': 'Rayleigh',
    'neither': 'NonThetaModulated',
}
THETA_MOD_PLOT_SUFFIXES = ('PolarPlot', 'PhaseHistogram', 'TMIShuffle')

TRACKING_TIME_UNIT = 'us'     # 'us', 'ms', or 's' -- units of the tracking timestamp column

# --- Spike/LFP channel-matching toggle ---
# True: match each .ntt tetrode file to the .ncs LFP file sharing its embedded channel number (e.g.
# TT1.ntt -> CSC1.ncs), via match_ncs_to_ntt(). False: reference every tetrode in the session to the
# first .ncs file found (sorted naturally), loaded and filtered once for the whole session.
MATCH_SPIKE_LFP_FILE = True


# --- ACG theta epochs (Dunn et al. 2022; protocol of Figc_ACG_ScalingCorrected_v8_NoiseCleaning.py,
# see acg_theta_epochs) ---
# True: only spikes falling in 1 s theta epochs are used for every theta phase. An epoch is a theta
# epoch if it survives the noise rejection (delta/theta power, peak-to-peak MAD outlier) AND the
# reference sinusoid whose autocorrelogram is closest to its own (ED_min), matched over the whole
# ACG_REF_BANK, lies in ACG_FREQ_RANGE. There is no threshold on the ED_min value itself. Noise and
# out-of-band (non-theta) epochs are thrown out. The rate map / place fields are still built from
# all spikes. False: use the entire signal.
USE_ACG_THETA_EPOCHS = True
ACG_EPOCH_SEC = 1.0
ACG_FS = 1000.0                  # Hz, LFP is resampled to this rate (clean_lfp: Hilbert phase + ACG)
ACG_FREQ_RANGE = (3.0, 7.0)      # Hz, theta band: an epoch is accepted only if its freq_est falls here
ACG_REF_BANK = (1.0, 20.0)       # Hz, span of the reference sinusoid bank every epoch is matched against.
                                 # References below ~1.75 Hz have no side peak within a 1 s epoch: they
                                 # still take part in the ED_min match, but give no peak range
ACG_FREQ_RES = 0.1               # Hz, bank step
NCS_NATIVE_FS = 32000            # Hz, grid of the repaired .ncs trace (ncs_repair.py)
ACG_REJECT_PARTIAL_RECORD_EPOCHS = False   # True: also drop every epoch containing a short .ncs record
# ADC saturation (as Figc_ACG_ScalingCorrected_v11_SaturationReject.py). The amplifier clips at
# +/-InputRange, read from each file's own .ncs header (-InputRange, uV: 1500 or 1000 depending on the
# recording); clipped samples sit flat at the rail (+/-32767 counts). They are found on the raw,
# repaired native-rate trace, before any filtering: a sample is clipped if |value| >= InputRange -
# SATURATION_TOL_STEPS ADC steps (one step = ADBitVolts; 1.5 flags exactly the rail and is safe against
# rounding). Each clipped run is widened by SATURATION_MARGIN_S on both sides, since filtering spreads
# the clip edges, and every epoch overlapping a widened run is thrown out ('rejected_saturated'), right
# after the record-integrity check and before the delta/theta and MAD rejection, so its spikes enter no
# theta phase. Files without -InputRange in the header get no check.
ACG_REJECT_SATURATED_EPOCHS = True
SATURATION_TOL_STEPS = 1.5       # ADC steps below +/-InputRange that still count as clipped
SATURATION_MARGIN_S = 0.2        # s, either side of each clipped run
# LFP cleaning (clean_lfp, shared by the Hilbert phase and the ACG epochs): resample to ACG_FS ->
# mains notch -> wavelet denoise -> linear detrend -> outlier fill (-> median filter -> Savitzky-Golay,
# if ACG_APPLY_SMOOTHING) -> low-pass. Sample-based windows assume ACG_FS = 1000 Hz.
ACG_LINE_HARMONICS = [50.0, 100.0, 150.0, 200.0]   # mains notch
ACG_NOTCH_Q = 30.0
ACG_WAVELET_NAME = 'sym4'            # wdenoise default
ACG_WAVELET_LEVEL = 4                # wdenoise(x, 4)
ACG_OUTLIER_WINDOW = 400             # 0.4 s: filloutliers(x, 'linear', 'movmedian', 100) at 250 Hz
ACG_OUTLIER_THRESH = 3.0             # local scaled MADs (filloutliers default)
ACG_APPLY_SMOOTHING = False          # medfilt1(x, 5) + smoothdata(x, 'sgolay', 15) at 250 Hz; off, as in v8
ACG_MEDFILT_KERNEL = 21
ACG_SGOLAY_WINDOW = 61
ACG_SGOLAY_POLY = 2
ACG_LOWPASS_CUTOFF_HZ = 100.0
ACG_LOWPASS_ORDER = 4
# Noise rejection, per epoch
ACG_REJECT_DELTA_OVER_THETA = True   # drop epochs whose 1-3 Hz power exceeds 3-7 Hz power
ACG_DELTA_BAND = (1.0, 3.0)
ACG_REJECT_MAD_ARTIFACTS = True      # drop epochs with outlier peak-to-peak amplitude
ACG_MAD_THRESH = 5.0
ACG_PEAK_PROMINENCE = 0.05           # reference-ACG peak/trough prominence (findMinMax(refxc, 0.05, 'fixed'));
                                     # only used for the QC plot's normalised peak range
# ACG QC plot (freq_est vs. normalised ACG peak range of every fitted epoch, coloured by animal,
# non-theta epochs highlighted) and its per-epoch table, both written to RESULTS_FOLDER
ACG_QC_PLOT_NAME = 'ACG_FreqEst_vs_PeakRange.png'
ACG_EPOCHS_CSV_NAME = 'ACG_epochs.csv'
# Per-animal colors — first 4 colors of matplotlib's Pastel1 palette.
ANIMAL_COLORS = {
    'Fa8477': '#FBB4AE',  # Pastel1[0] red
    'Fa1059': '#B3CDE3',  # Pastel1[1] blue
    'Fa23BD': '#CCEBC5',  # Pastel1[2] green
    'Fa5834': '#DECBE4',  # Pastel1[3] purple
}

# --- Theta phase and theta modulation (Steps 1-2) ---
LFP_FILTER_BAND = (3.0, 7.0)        # Hz, theta band of the Hilbert phase
PHASE_BIN_SIZE_DEG = 12             # degrees per polar-histogram bin (360 must be divisible by this);
                                     # matches calc_tmi's bin_width_deg for consistency
ALPHA = 0.05                        # Rayleigh-test / TMI-shuffle / comparison significance threshold
MIN_SPIKES_FOR_TMI = 50             # min spikes required to run the Rayleigh / TMI tests
N_TMI_SHUFFLES = 1000               # number of surrogate random-phase draws for the TMI shuffle test
TMI_BURST_MAX_GAP_SEC = 0.05        # consecutive spikes closer than this (Frank et al. 2001: 50 ms) AND
TMI_BURST_PHASE_BIN_DEG = 36        # in the same real-phase bin this wide are kept together as one
                                     # burst and given the same shuffled phase (calc_tmi's own bin width)
RANDOM_SEED = 0                     # seed for the TMI shuffle test's RNG, for reproducibility

LFP_MATCH_JITTER_FACTOR = 1.25      # a spike is matched to an LFP sample only if the nearest sample is
                                     # within 1/(1.25*lfp_fs) s (one sample interval, less jitter margin)

NCS_SAMPLES_PER_RECORD = 512
HEADER_BYTES = 16 * 1024
DEFAULT_ADBITVOLTS = 0.000000195

# --- Place fields (v30: "threshold method" of PlaceCell_Main_v20 on Climer's rate map) ---
BINSIDE = 2.0                       # spatial bin size (cm)
SMTH_WIDTH = 12.0                   # rate-map Gaussian smoothing width (= 2 * sigma; sigma = 6 cm)
FIELD_RATE_THRESHOLD_FRAC = 0.20    # bin qualifies if smoothed rate >= this * peak AND > the cell's mean
                                    # (raw, occupancy-weighted) rate
MIN_FIELD_SIZE_BINS = 9             # >= this many 8-connected bins (angular contiguous bins on the Circle)
FIELD_MIN_VISITS = 2                # valid bin = occupied and entered >= this many times; the raw peak
                                    # bin is the highest valid bin of the raw map
RATEMAP_IMMOBILITY_SPEED_CM_S = 0.5 # samples (and their spikes) below this speed are left out of the rate
                                    # map / field detection; laps run on samples at or above it
RING_RADIAL_CLIP_PCTILE = 0.5       # Circle: track width = radial range between these percentiles

# --- Laps and spike filter (v30 field_traversals) ---
POS_SMOOTH_SEC = 0.2                # Gaussian sigma (s) applied to x/y before speed and lap detection;
                                    # removes frame-to-frame tracking jitter
MIN_SPEED_CM_S = 1.0                # analysed spikes are fired at smoothed speed >= this
MIN_SPIKES_PER_LAP = 5              # only laps with >= this many analysed spikes give an onset / offset spike
MERGE_GAP_S = 0.5                   # field exits this short (s, real time) do not end a lap
LAP_MIN_DURATION_S = 0.1            # laps shorter than this (s) are dropped
MIN_PEAK_BIN_SPIKES = 1             # a unit needs >= this many analysed peak-bin spikes for a peak-bin phase
MIN_LAPS_FOR_ONOFF = 1              # ... and >= this many kept laps for an onset / offset phase

# --- Theta-phase span of the peak-bin spikes (see circular_phase_span / run_peakbin_span) ---
# Per unit: the theta phases of its analysed raw-peak-bin spikes are cut at their largest circular gap;
# span = 360 deg - that gap, and every phase is re-measured from the first spike after the gap (so the
# span starts at 0 deg, whatever theta phase it sits at). One Gaussian-kernel density of these
# re-measured phases per unit; densities and spans compared across groups.
MIN_PEAK_BIN_SPIKES_FOR_SPAN = 10   # a unit needs >= this many analysed peak-bin spikes for a span / density
SPAN_KDE_SIGMA_DEG = 15.0           # Gaussian kernel SD (deg) of each unit's density (on [0, span], reflected at both ends)
SPAN_GRID_STEP_DEG = 1.0            # density evaluated every this many deg over 0-360
N_DENSITY_PERMUTATIONS = 10000      # unit-label permutations for the density-curve test

# --- Groups compared ---
# Data layout: ROOT_FOLDER/<animal>/<arena>/DayN/<session>, e.g. .../Fa23BD/Open/Day1/1Cntrl -- the
# arena is read off whichever path component (case-insensitively) matches one of these labels.
ARENA_LABELS = ('Circle', 'Linear', 'Open')
# Session type is read off the session folder name (see detect_session_type), e.g. '1_Cntrl' ->
# 'Cntrl', '3_180' -> '180', '2NoRot' -> 'NoRot'. Listed in plotting order.
ARENA_SESSION_TYPES = {
    'Open': ('Cntrl', 'Rotate', 'Zero'),
    'Linear': ('0', '90', '180', '270'),
    'Circle': ('Stnd', 'NoRot', 'Rot'),
}
SESSION_TYPE_DIR = 'SessionType'       # in RESULTS_FOLDER, one sub-folder per arena
ARENA_TYPE_DIR = 'ArenaType'
# Animal-controlled arena comparisons (level 3): {name: (animals, arenas)}. Each compares the given
# arenas using only the units of the given animals, which were recorded in all of those arenas
# (matched case-insensitively against the animal folder). The plots of all comparisons are saved side
# by side (left to right in this order) in ANIMAL_CONTROLLED_DIR.
ANIMAL_CONTROLLED_DIR = 'AnimalControlledArenaType'
ANIMAL_CONTROLLED_COMPARISONS = {
    'Circle_vs_Open': (['Fa8477', 'Fa5834'], ['Circle', 'Open']),
    'Linear_vs_Open': (['Fa1059', 'Fa23BD'], ['Linear', 'Open']),
}
PANEL_GAP_IN = 0.3   # white gap between the side-by-side panels (inches)
# Per-animal comparison of the theta-modulated proportions (arenas and session types pooled within each
# animal): RESULTS_FOLDER\ANIMAL_DIR\Animal_ThetaModProportion(_Tested).png. Animals in ANIMAL_COLORS
# order, then any other animal found, alphabetically.
ANIMAL_DIR = 'Animal'

# --- Theta-modulation criteria plotted per group: (column, label, bar hatch). Only the last one
# (Rayleigh+ AND TMI+, the theta-modulation criterion) is tested across groups. ---
THETA_MOD_CRITERIA = (
    ('Rayleigh_Significant', 'Rayleigh+', '///'),
    ('TMI_Significant', 'TMI+', '...'),
    ('ThetaModulated', 'Rayleigh+ & TMI+', None),
)

# --- Spike phases compared across groups: (column, plot label, file tag, sheet tag) ---
SPIKE_PHASE_METRICS = (
    ('PeakBin_phase_deg', 'the raw-map peak bin', 'PeakBin', 'PkBin'),
    ('Onset_phase_deg', 'firing onset (first spike of each lap)', 'Onset', 'Onset'),
    ('Offset_phase_deg', 'firing offset (last spike of each lap)', 'Offset', 'Offset'),
)
# Units each phase comparison is run on: (selecting column or None = all units, file tag, sheet tag, label)
CELL_SETS = (
    (None, 'AllUnits', 'All', 'all units'),
    ('ThetaModulated', 'ThetaMod', 'TM', 'theta-modulated units (Rayleigh+ & TMI+)'),
)
# Theta-modulated vs non-theta-modulated units, within each group (Watson-Williams; see compare_tm_vs_non).
# True: non-theta-modulated = every unit that is not Rayleigh+ & TMI+, including those with too few spikes
# (< MIN_SPIKES_FOR_TMI) to be tested, as in the proportion plots. False: only tested units.
NON_TM_INCLUDES_UNTESTED = True
# Theta trough range on the circle (0 deg = theta peak); the rest of the cycle is the theta peak range.
THETA_TROUGH_RANGE_DEG = (70.0, 250.0)
PHASE_RANGE_COLORS = {'trough': '#8C2981', 'peak': '#FCFDBF'}   # magma purple (0.4) / light yellow (1.0)
N_MWW_PERMUTATIONS = 10000          # label permutations for the Mardia-Watson-Wheeler test

# Per-unit figure: spikes marked on the trajectory and phase histograms
PEAK_BIN_SPIKE_COLOR = 'tab:green'
FIRING_ONSET_COLOR = 'tab:red'
FIRING_OFFSET_COLOR = 'tab:blue'


# ============================================================================
# Neuralynx file I/O
# ============================================================================

def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', path.name)]


def _read_header_text(path: Path) -> str:
    with open(path, 'rb') as fh:
        raw = fh.read(HEADER_BYTES)
    return raw.decode('latin-1', errors='replace')


def _header_field(header_text: str, name: str, default=None, cast=float):
    for line in header_text.splitlines():
        line = line.strip()
        if line.startswith(f'-{name}'):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return cast(parts[1])
                except ValueError:
                    return default
    return default


def load_ncs(path: Path):
    """Load a Neuralynx .ncs file onto a uniform, timestamp-anchored NCS_NATIVE_FS grid
    (ncs_repair.load_repaired_uv, as in Figc_ACG_ScalingCorrected_v8_NoiseCleaning.py): only each
    record's first NumValidSamples samples are data, each is placed by its own record's absolute
    (PTP-synced) timestamp, and missing samples are linearly interpolated. A gap longer than 1 s
    raises ValueError.

    Returns (samples_uV, start_s, fs_hz, records): sample i is at start_s + i / fs_hz; records holds
    each record's first grid index ('start') and NumValidSamples ('nvalid'), for the short-record
    check of acg_theta_epochs, this file's header values 'ad_bit_volts' (V/count) and
    'input_range_uv' (uV, None if absent), and 'saturated', the clipped-sample mask of the raw
    native-rate trace (saturated_samples), for the saturation check of acg_theta_epochs.
    """
    header_text = _read_header_text(path)
    adbitvolts = _header_field(header_text, 'ADBitVolts', default=DEFAULT_ADBITVOLTS)
    input_range_uv = _header_field(header_text, 'InputRange', default=None)
    samples_uV, start_us = load_repaired_uv(path, adbitvolts)

    ncs_dtype = np.dtype([
        ('TimeStamp', '<u8'),
        ('ChannelNumber', '<u4'),
        ('SampleFreq', '<u4'),
        ('NumValidSamples', '<u4'),
        ('Samples', '<i2', (NCS_SAMPLES_PER_RECORD,)),
    ])
    data = np.memmap(path, dtype=ncs_dtype, mode='r', offset=HEADER_BYTES)
    start = np.round((np.asarray(data['TimeStamp']).astype(np.int64) - start_us)
                     * NCS_NATIVE_FS / 1e6).astype(np.int64)
    records = {'start': start, 'nvalid': record_lengths(data['NumValidSamples']),
               'ad_bit_volts': adbitvolts, 'input_range_uv': input_range_uv,
               'saturated': saturated_samples(samples_uV, input_range_uv, adbitvolts)}
    return samples_uV, start_us / 1e6, float(NCS_NATIVE_FS), records


def saturated_samples(lfp_uv, input_range_uv, ad_bit_volts, tol_steps=SATURATION_TOL_STEPS):
    """Boolean mask of clipped samples of the raw, native-rate trace (uV): those within
    `tol_steps` ADC steps of +/-`input_range_uv` (saturated_samples of
    Figc_ACG_ScalingCorrected_v11_SaturationReject.py). All False if the file has no InputRange."""
    if input_range_uv is None:
        return np.zeros(len(lfp_uv), dtype=bool)
    rail_uv = input_range_uv - tol_steps * ad_bit_volts * 1e6
    return np.abs(lfp_uv) >= rail_uv


def epoch_saturation(sat_native, n_total, nperseg, fs_hz, margin_s=SATURATION_MARGIN_S):
    """Clipped-sample count and near-clipping flag of each (fs_hz) epoch (epoch_saturation of
    Figc_ACG_ScalingCorrected_v11_SaturationReject.py). `sat_native` = saturated_samples() of the
    native-rate trace. Epoch i covers native samples first..last, mapped as in acg_theta_epochs'
    short-record check; each clipped run is widened by `margin_s` on both sides, and an epoch is
    near clipping if it overlaps any widened run.
    Returns (n_clipped, near_clip)."""
    step = NCS_NATIVE_FS / fs_hz
    i = np.arange(n_total)
    first = np.round(i * nperseg * step).astype(np.int64)
    last = np.minimum(np.round(((i + 1) * nperseg - 1) * step).astype(np.int64), len(sat_native) - 1)
    sat_cum = np.concatenate(([0], np.cumsum(sat_native, dtype=np.int64)))
    n_clipped = sat_cum[last + 1] - sat_cum[first]

    # widened runs [run_lo, run_hi] (sorted by start); an epoch overlaps one if the last run starting
    # at or before its end reaches (running max of ends) its start
    run_starts, run_lengths = find_runs(sat_native)
    margin = int(round(margin_s * NCS_NATIVE_FS))
    near_clip = np.zeros(n_total, dtype=bool)
    if len(run_starts):
        run_lo = run_starts - margin
        run_hi_max = np.maximum.accumulate(run_starts + run_lengths - 1 + margin)
        j = np.searchsorted(run_lo, last, side='right') - 1
        near_clip = (j >= 0) & (run_hi_max[np.maximum(j, 0)] >= first)
    return n_clipped, near_clip


def load_ntt_spike_times(path: Path):
    """Load a Neuralynx .ntt file. Returns dict {cell_number: spike_times_s}.

    cell_number 0 (unsorted) is always dropped;
    only sorted, non-zero clusters are returned as units.
    """
    ntt_dtype = np.dtype([
        ('timestamp', '<u8'),
        ('sc_number', '<u4'),
        ('cell_number', '<u4'),
        ('params', '<u4', (8,)),
        ('waveforms', '<i2', (32, 4)),
    ])
    records = np.memmap(path, dtype=ntt_dtype, mode='r', offset=HEADER_BYTES)
    timestamps_s = records['timestamp'].astype(np.float64) / 1e6
    cell_numbers = records['cell_number'].astype(np.int64)

    units = {}
    unique_cells = np.unique(cell_numbers)
    unique_cells = unique_cells[unique_cells != 0]
    for cell in unique_cells:
        units[int(cell)] = np.sort(timestamps_s[cell_numbers == cell])
    return units


# ============================================================================
# Tracking file I/O
# ============================================================================

def _find_tracking_file(folder: Path) -> Path:
    candidates = sorted(folder.glob('*.csv'))
    if not candidates:
        raise FileNotFoundError(f'No .csv tracking file found in {folder}')
    return candidates[0]


def load_tracking(path: Path, time_unit: str = 'us'):
    """Load tracking coordinates already in cm.

    Column order is fixed: col 0 = timestamp, col 1 = x (pixels, unused),
    col 2 = y (pixels, unused), col 3 = x (cm), col 4 = y (cm).

    Returns (pos_ts_s, pos_xy_cm) sorted by ascending timestamp, with NaN and
    duplicate-timestamp rows removed.
    """
    probe = pd.read_csv(path, header=None, nrows=1)

    has_header = False
    for val in probe.iloc[0, :5]:
        try:
            float(val)
        except (TypeError, ValueError):
            has_header = True
            break

    df = pd.read_csv(path, header=0 if has_header else None)

    data = df.iloc[:, [0, 3, 4]].to_numpy(dtype=np.float64)
    time_scale = {'us': 1e-6, 'ms': 1e-3, 's': 1.0}[time_unit]
    pos_ts = data[:, 0] * time_scale
    pos_xy = data[:, 1:3]

    valid = ~np.any(np.isnan(data), axis=1)
    pos_ts, pos_xy = pos_ts[valid], pos_xy[valid]

    order = np.argsort(pos_ts, kind='stable')
    pos_ts, pos_xy = pos_ts[order], pos_xy[order]
    keep = np.concatenate(([True], np.diff(pos_ts) > 0))
    return pos_ts[keep], pos_xy[keep]


# ============================================================================
# Rate map, position and LFP helpers
# ============================================================================

def _fill_nan_nearest(arr: np.ndarray) -> np.ndarray:
    mask = np.isnan(arr)
    if not mask.any():
        return arr
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


def spk_pos(pos_ts, pos_xy, spk_ts):
    """Nearest tracked position at each spike time."""
    idx = np.searchsorted(pos_ts, spk_ts)
    idx = np.clip(idx, 1, len(pos_ts) - 1)
    left, right = idx - 1, idx
    use_left = np.abs(spk_ts - pos_ts[left]) <= np.abs(pos_ts[right] - spk_ts)
    idx = np.where(use_left, left, right)
    return pos_xy[idx], idx


def rate_map(pos_ts, pos_xy, spk_xy, binside, smth_width, sample_mask=None, return_counts=False):
    """Occupancy-normalized 2D rate map (rate_map.m: unvisited bins filled with the
    nearest visited value, Gaussian sigma = smth_width / 2, unvisited bins then 0).
    Returns (map, occupancy, x_edges, y_edges). `sample_mask` (bool per position
    sample) restricts occupancy to those samples; the grid always spans all of
    pos_xy. With return_counts=True, the raw spike-count map is appended."""
    mins = np.floor(pos_xy.min(axis=0) / binside) * binside
    maxs = np.ceil(pos_xy.max(axis=0) / binside) * binside
    x_edges = np.arange(mins[0], maxs[0] + binside, binside)
    y_edges = np.arange(mins[1], maxs[1] + binside, binside)

    dt = float(np.mean(np.diff(pos_ts)))
    occ_xy = pos_xy if sample_mask is None else pos_xy[sample_mask]
    occupancy, _, _ = np.histogram2d(occ_xy[:, 0], occ_xy[:, 1], bins=[x_edges, y_edges])
    occupancy = occupancy * dt

    spk_counts, _, _ = np.histogram2d(spk_xy[:, 0], spk_xy[:, 1], bins=[x_edges, y_edges])
    with np.errstate(invalid='ignore', divide='ignore'):
        rmap = spk_counts / occupancy
    rmap[occupancy == 0] = np.nan

    rmap = _fill_nan_nearest(rmap)
    sigma = smth_width / binside / 2.0
    rmap = gaussian_filter(rmap, sigma=sigma, truncate=3.0, mode='nearest')
    rmap[occupancy == 0] = 0.0

    if return_counts:
        return rmap, occupancy, x_edges, y_edges, spk_counts
    return rmap, occupancy, x_edges, y_edges


def smooth_positions(pos_ts, pos_xy, sigma_sec=POS_SMOOTH_SEC):
    """Gaussian-smoothed positions and speed (cm/s). Raw tracking jitter (~0.1-0.2 cm
    per frame even when the animal sits still) otherwise inflates path length."""
    dt = float(np.median(np.diff(pos_ts)))
    sm = gaussian_filter1d(pos_xy, sigma_sec / dt, axis=0, mode='nearest') if sigma_sec > 0 else pos_xy
    speed = np.hypot(*np.gradient(sm, pos_ts, axis=0).T)
    return sm, speed


def bandpass_filter(data, low, high, fs, order=4):
    """Zero-phase Butterworth bandpass, always via second-order sections.

    SOS form avoids the b/a transfer-function representation's numerical
    instability (spurious pole at z=1) for narrow or near-DC bands. Wn is
    clamped into the valid open interval (0, 1) relative to Nyquist so a
    degenerate band raises a clear error instead of a cryptic scipy exception.
    """
    nyq = fs / 2.0
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 1 - 1e-6)
    if low_n >= high_n:
        raise ValueError(
            f'invalid filter band after clamping to Nyquist: '
            f'requested ({low:.4g}, {high:.4g}), fs={fs:.4g} -> '
            f'normalized ({low_n:.4g}, {high_n:.4g})'
        )
    sos = signal.butter(order, [low_n, high_n], btype='band', output='sos')
    filtered = signal.sosfiltfilt(sos, data)
    if np.any(np.isnan(filtered)):
        filtered = signal.sosfilt(sos, data)
    return filtered


# ============================================================================
# Place fields: "threshold method" of PlaceCell_Main_v20 (detect_place_fields_threshold_2d /
# _circular) on Climer's rate map, as in v30 (multi-peak fields not split), plus the raw peak bin
# ============================================================================

def _visit_counts(flat_idx, n_bins):
    """Visits per bin: a visit starts whenever the bin index differs from the
    previous sample's (PlaceCell_Main_v20 _bin_visit_counts)."""
    flat_idx = np.asarray(flat_idx)
    if flat_idx.size == 0:
        return np.zeros(n_bins, dtype=np.int64)
    starts = np.ones(flat_idx.size, dtype=bool)
    starts[1:] = flat_idx[1:] != flat_idx[:-1]
    return np.bincount(flat_idx[starts], minlength=n_bins)


def _fit_ring_centre(x, y):
    """Algebraic (Kasa) circle fit -> (cx, cy) (PlaceCell_Main_v20 _fit_ring_centre)."""
    A = np.column_stack([x, y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    return sol[0] / 2.0, sol[1] / 2.0


def _circular_runs(qualifies):
    """Contiguous runs of True bins on a circular axis, a run touching both ends
    merged across the 0/2*pi seam (PlaceCell_Main_v20 _circular_runs)."""
    n = len(qualifies)
    if not qualifies.any():
        return []
    if qualifies.all():
        return [list(range(n))]
    idx = np.flatnonzero(qualifies)
    runs, current = [], [int(idx[0])]
    for b in idx[1:]:
        if b == current[-1] + 1:
            current.append(int(b))
        else:
            runs.append(current)
            current = [int(b)]
    runs.append(current)
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == n - 1:
        runs = runs[1:-1] + [runs[-1] + runs[0]]
    return runs


def _number_fields(fields):
    """Sort by peak (smoothed) rate, highest first, and number 1..n."""
    fields.sort(key=lambda f: -f['peak_rate_Hz'])
    for k, f in enumerate(fields, start=1):
        f['field_number'] = k
    return fields


def _field_record_2d(mask, rmap, raw, valid, binside):
    """Field dict (peak rates, size) of the 2-D bins in `mask`."""
    n_bins = int(mask.sum())
    area = n_bins * binside ** 2
    return dict(mask_2d=mask, peak_rate_Hz=float(rmap[mask].max()),
                peak_rate_raw_Hz=float(np.nanmax(raw[mask])),
                n_bins=n_bins, area_cm2=area, pct_of_occupied_area=100.0 * n_bins / int(valid.sum()))


def _detect_fields_2d(rmap, valid, mean_fr, record):
    """Threshold method on the 2-D map: valid bins with smoothed rate >=
    FIELD_RATE_THRESHOLD_FRAC * peak AND > the mean raw rate, 8-connected
    components of >= MIN_FIELD_SIZE_BINS bins. record(mask) -> field dict."""
    peak = float(rmap[valid].max())
    qualifies = valid & (rmap >= FIELD_RATE_THRESHOLD_FRAC * peak) & (rmap > mean_fr)
    labels, n_comp = nd_label(qualifies, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(labels.ravel(), minlength=n_comp + 1)
    return _number_fields([record(labels == c) for c in range(1, n_comp + 1) if sizes[c] >= MIN_FIELD_SIZE_BINS])


def _detect_fields_circular(pos_ts, pos_xy, moving, spk_xy_moving, binside, smth_width,
                            x_edges, y_edges, occupancy_2d):
    """Threshold method on the 1-D angular map of the Circle track
    (PlaceCell_Main_v20 _build_ratemap_circular / detect_place_fields_threshold_circular):
    bearing around the fitted ring centre binned at ~binside cm of arc, radius discarded,
    smoothed Climer-style (nearest-fill of unvisited bins, Gaussian sigma = smth_width / 2,
    wrapping around the ring); qualifying runs of >= MIN_FIELD_SIZE_BINS bins, wrapping
    across the 0/360 deg seam. Each field also gets a 2-D display mask (occupied 2-D
    bins whose centre bearing falls in the field). Returns (fields, ring) with ring =
    dict(cx, cy, r_med, theta_bin, n_bins, occ, spk, valid, raw, smooth): theta_bin(xy) ->
    angular bin, and the angular maps (raw = spikes / occupancy, 0 where unoccupied)."""
    xy = pos_xy[moving]
    cx, cy = _fit_ring_centre(xy[:, 0], xy[:, 1])
    r = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    r_med = float(np.median(r))
    n_t = max(1, int(round(2 * np.pi * r_med / binside)))
    bin_w = 2 * np.pi / n_t

    def _theta_bin(p):
        th = np.mod(np.arctan2(p[:, 1] - cy, p[:, 0] - cx), 2 * np.pi)
        return np.clip((th / bin_w).astype(int), 0, n_t - 1)

    dt = float(np.mean(np.diff(pos_ts)))
    bt = _theta_bin(xy)
    occ = np.bincount(bt, minlength=n_t) * dt
    spk = np.bincount(_theta_bin(spk_xy_moving), minlength=n_t).astype(float)
    valid = (occ > 0) & (_visit_counts(bt, n_t) >= FIELD_MIN_VISITS)
    if not valid.any():
        raise ValueError('no valid bins on the angular map')

    with np.errstate(invalid='ignore', divide='ignore'):
        raw = np.where(occ > 0, spk / occ, np.nan)
    filled = _fill_nan_nearest(np.tile(raw, 3))[n_t:2 * n_t]      # circular nearest fill
    smooth = gaussian_filter1d(filled, (smth_width / 2.0) / binside, mode='wrap', truncate=3.0)
    smooth[occ == 0] = 0.0

    peak = float(smooth[valid].max())
    mean_fr = float(spk[valid].sum() / occ[valid].sum())
    qualifies = valid & (smooth >= FIELD_RATE_THRESHOLD_FRAC * peak) & (smooth > mean_fr)
    track_w = float(np.percentile(r, 100 - RING_RADIAL_CLIP_PCTILE) - np.percentile(r, RING_RADIAL_CLIP_PCTILE))

    xc, yc = 0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (y_edges[1:] + y_edges[:-1])
    gx, gy = np.meshgrid(xc, yc, indexing='ij')
    grid_tbin = _theta_bin(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)

    def record(run):
        run = np.asarray(run)
        area = len(run) * bin_w * r_med * track_w
        return dict(theta_bins=run, mask_2d=np.isin(grid_tbin, run) & (occupancy_2d > 0),
                    peak_rate_Hz=float(smooth[run].max()), peak_rate_raw_Hz=float(np.nanmax(raw[run])),
                    n_bins=len(run), area_cm2=area, pct_of_occupied_area=100.0 * len(run) / int(valid.sum()))

    fields = _number_fields([record(run) for run in _circular_runs(qualifies) if len(run) >= MIN_FIELD_SIZE_BINS])
    ring = dict(cx=cx, cy=cy, r_med=r_med, theta_bin=_theta_bin, n_bins=n_t,
                occ=occ, spk=spk, valid=valid, raw=np.nan_to_num(raw), smooth=smooth)
    return fields, ring


def detect_place_fields(pos_ts, pos_xy, spk_ts, arena, binside, smth_width) -> dict[str, Any]:
    """Climer rate map, the place fields of PlaceCell_Main_v20's threshold method on it (2-D,
    or angular on the Circle track) and the raw peak bin. Immobile samples (smoothed speed <
    RATEMAP_IMMOBILITY_SPEED_CM_S) and their spikes are left out, as in v30.
    Raw peak bin = the valid bin (occupied, entered >= FIELD_MIN_VISITS times) with the
    highest spikes / occupancy, unsmoothed (angular bin on the Circle track). Main field = the
    field holding it, or None.
    Returns a dict with rate_map, occupancy, raw_map_2d (display), x/y_edges, fields,
    main_field, raw_peak_bin (flat bin index, as bin_id), raw_peak_rate_Hz, raw_peak_xy (cm),
    raw_peak_theta_deg (Circle; else NaN), detection, and bin_id(xy) -> flat bin index /
    field_label(xy) -> field_number (0 = outside every field) for positions.
    If the raw peak bin is outside every field, main_field falls back to the field holding the peak
    bin of the smoothed map (valid bins only); main_field_source says which ('raw peak bin',
    'smoothed peak bin', or '' for none). raw_peak_in_field, smooth_peak_bin and smooth_peak_xy are
    also returned."""
    _, speed = smooth_positions(pos_ts, pos_xy)
    moving = speed >= RATEMAP_IMMOBILITY_SPEED_CM_S
    spk_xy, spk_idx = spk_pos(pos_ts, pos_xy, spk_ts)
    spk_xy_moving = spk_xy[moving[spk_idx]]

    rmap, occupancy, x_edges, y_edges, spk_counts = rate_map(
        pos_ts, pos_xy, spk_xy_moving, binside, smth_width, sample_mask=moving, return_counts=True)
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    xc, yc = 0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (y_edges[1:] + y_edges[:-1])
    with np.errstate(invalid='ignore', divide='ignore'):
        raw_2d = np.where(occupancy > 0, spk_counts / occupancy, np.nan)

    def _xy_bins(p):
        return (np.clip(np.digitize(p[:, 0], x_edges) - 1, 0, nx - 1),
                np.clip(np.digitize(p[:, 1], y_edges) - 1, 0, ny - 1))

    if arena == 'Circle':
        fields, ring = _detect_fields_circular(
            pos_ts, pos_xy, moving, spk_xy_moving, binside, smth_width, x_edges, y_edges, occupancy)
        detection = 'Circle (1-D angular)'
        raw, valid = ring['raw'], ring['valid']
        bin_id = ring['theta_bin']
        labels = np.zeros(ring['n_bins'], dtype=int)
        for f in fields:
            labels[f['theta_bins']] = f['field_number']
    else:
        bx, by = _xy_bins(pos_xy[moving])
        valid_2d = (occupancy > 0) & (_visit_counts(bx * ny + by, nx * ny).reshape(nx, ny) >= FIELD_MIN_VISITS)
        if not valid_2d.any():
            raise ValueError('no valid bins on the rate map')
        mean_fr = float(spk_counts[valid_2d].sum() / occupancy[valid_2d].sum())
        fields = _detect_fields_2d(rmap, valid_2d, mean_fr,
                                   lambda mask: _field_record_2d(mask, rmap, raw_2d, valid_2d, binside))
        detection = '2-D grid'
        raw, valid = np.nan_to_num(raw_2d).ravel(), valid_2d.ravel()     # flat index = bx * ny + by

        def bin_id(p):
            bx_p, by_p = _xy_bins(p)
            return bx_p * ny + by_p

        labels = np.zeros(nx * ny, dtype=int)
        for f in fields:
            labels[f['mask_2d'].ravel()] = f['field_number']

    def field_label(p):
        return labels[bin_id(p)]

    def _bin_xy(b):
        """(x, y) cm of flat bin b and its bearing (deg; Circle track, else NaN)."""
        if arena == 'Circle':
            theta_c = (b + 0.5) * 2 * np.pi / ring['n_bins']
            return ((float(ring['cx'] + ring['r_med'] * np.cos(theta_c)),
                     float(ring['cy'] + ring['r_med'] * np.sin(theta_c))), float(np.rad2deg(theta_c)))
        bx_b, by_b = divmod(b, ny)
        return (float(xc[bx_b]), float(yc[by_b])), np.nan

    raw_peak_bin = int(np.argmax(np.where(valid, raw, -np.inf)))
    raw_peak_rate = float(raw[raw_peak_bin])
    if raw_peak_rate <= 0:
        raise ValueError('no spikes in the valid bins of the raw rate map')
    raw_peak_xy, raw_peak_theta_deg = _bin_xy(raw_peak_bin)
    smooth = ring['smooth'] if arena == 'Circle' else rmap.ravel()
    smooth_peak_bin = int(np.argmax(np.where(valid, smooth, -np.inf)))
    smooth_peak_xy, _ = _bin_xy(smooth_peak_bin)

    # Main field: the one holding the raw peak bin; if that bin is outside every field, the one holding
    # the peak bin of the smoothed map; else none.
    main_field_source = ''
    main_field = next((f for f in fields if f['field_number'] == labels[raw_peak_bin]), None)
    if main_field is not None:
        main_field_source = 'raw peak bin'
    else:
        main_field = next((f for f in fields if f['field_number'] == labels[smooth_peak_bin]), None)
        if main_field is not None:
            main_field_source = 'smoothed peak bin'

    return dict(rate_map=rmap, occupancy=occupancy, raw_map_2d=raw_2d, x_edges=x_edges, y_edges=y_edges,
                fields=fields, main_field=main_field, main_field_source=main_field_source,
                raw_peak_bin=raw_peak_bin, raw_peak_rate_Hz=raw_peak_rate,
                raw_peak_xy=raw_peak_xy, raw_peak_theta_deg=raw_peak_theta_deg,
                smooth_peak_bin=smooth_peak_bin, smooth_peak_xy=smooth_peak_xy,
                raw_peak_in_field=bool(labels[raw_peak_bin] > 0),
                detection=detection, bin_id=bin_id, field_label=field_label)


# ============================================================================
# Laps through the main field (v30 field_traversals), and the peak-bin / onset / offset spikes
# ============================================================================

def _pass_trace(pos_ts, pos_xy, maps):
    """Field-independent part of the laps, computed once per unit and cached in `maps`:
    jitter-smoothed positions, moving (>= RATEMAP_IMMOBILITY_SPEED_CM_S) and running
    (>= MIN_SPEED_CM_S) masks."""
    if 'trace' not in maps:
        pos_sm, speed = smooth_positions(pos_ts, pos_xy)
        maps['trace'] = dict(pos_sm=pos_sm, moving=speed >= RATEMAP_IMMOBILITY_SPEED_CM_S,
                             running=speed >= MIN_SPEED_CM_S)
    return maps['trace']


def field_traversals(pos_ts, pos_xy, maps, field):
    """Every lap of one place field (v30 field_traversals, as a pass of
    LapRaster_OpenField_v3_AdptBinFlds.py). On the moving-only (smoothed speed >=
    RATEMAP_IMMOBILITY_SPEED_CM_S) sequence of jitter-smoothed position samples (_pass_trace), a
    lap is one contiguous run of samples inside `field` (entry -> exit through any of its bins;
    pausing inside it does not end it); runs <= MERGE_GAP_S apart in real time are merged, and
    those shorter than LAP_MIN_DURATION_S dropped. Laps touching the first / last moving sample
    (entry or exit not observed, so firing onset or offset may be missing) are dropped too.
    Returns dict(first, last: the laps' first / last sample in the moving-only sequence,
    n_session_edge: laps dropped for touching the start / end of tracking)."""
    trace = _pass_trace(pos_ts, pos_xy, maps)
    pos_sm, moving = trace['pos_sm'], trace['moving']
    idx = np.flatnonzero(moving)
    in_field = maps['field_label'](pos_sm[idx]) == field['field_number']
    t_moving = pos_ts[idx]
    dt = np.r_[np.diff(pos_ts), np.median(np.diff(pos_ts))]
    cum_dt = np.r_[0.0, np.cumsum(dt[moving])]

    edges = np.diff(np.r_[0, in_field.astype(np.int8), 0])
    runs: list[list[int]] = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1):
        if runs and t_moving[a] - t_moving[runs[-1][1]] <= MERGE_GAP_S:
            runs[-1][1] = int(b)
        else:
            runs.append([int(a), int(b)])

    first, last = [], []
    n_session_edge = 0
    for a, b in runs:
        if cum_dt[b + 1] - cum_dt[a] < LAP_MIN_DURATION_S:
            continue
        if a == 0 or b == in_field.size - 1:
            n_session_edge += 1
            continue
        first.append(a)
        last.append(b)
    return dict(first=np.array(first, dtype=np.int64), last=np.array(last, dtype=np.int64),
                n_session_edge=n_session_edge)


def compute_peak_onset_offset(pos_ts, pos_xy, maps, spk_ts, lfp_ts, lfp_fs, lfp_phase_unwrapped):
    """Peak-bin, firing-onset and firing-offset spikes of one unit and their theta phases.
    `spk_ts` = the unit's ACG theta-epoch spikes within the tracking / LFP overlap. Analysed
    spikes = those with an LFP phase (assign_spike_phase) fired at smoothed speed >= MIN_SPEED_CM_S
    (speed at the nearest position sample).
      peakbin  every analysed spike whose raw position (nearest sample, as in the raw map) lies in
               the raw peak bin, over the whole session.
      onset /  on the laps of the main field (field_traversals), each analysed spike is assigned to
      offset   the lap holding its nearest moving position sample; laps with >= MIN_SPIKES_PER_LAP
               analysed spikes are kept, and their first / last analysed spike is the onset / offset
               spike. Main field = the field holding the raw peak bin, else the one holding the
               smoothed-map peak bin (detect_place_fields); none: no onset / offset spikes.
    Unit values: peak-bin phase = circular mean of its peak-bin spikes (>= MIN_PEAK_BIN_SPIKES);
    onset / offset phase = circular mean over its kept laps (>= MIN_LAPS_FOR_ONOFF); peak-bin phase
    span (circular_phase_span) of its peak-bin spikes (>= MIN_PEAK_BIN_SPIKES_FOR_SPAN), with the
    re-measured phases as a ';'-joined string (PeakBin_SpanPhases_deg) for the population densities.
    Returns dict with bool masks over the sorted spikes (analysed, in_lap, peakbin, onset, offset),
    spk_ts, spk_xy, theta (rad), span_shifted (re-measured peak-bin phases, deg, or None), lap counts
    and the per-unit summary values (summary)."""
    trace = _pass_trace(pos_ts, pos_xy, maps)
    spk_ts = np.sort(spk_ts)
    n = len(spk_ts)
    spk_xy, spk_idx = spk_pos(pos_ts, pos_xy, spk_ts)
    theta = assign_spike_phase(spk_ts, lfp_ts, lfp_phase_unwrapped, lfp_fs)
    analysed = np.isfinite(theta) & trace['running'][spk_idx]
    peakbin = analysed & (maps['bin_id'](spk_xy) == maps['raw_peak_bin'])

    out = dict(spk_ts=spk_ts, spk_xy=spk_xy, theta=theta, analysed=analysed, peakbin=peakbin,
               in_lap=np.zeros(n, dtype=bool), onset=np.zeros(n, dtype=bool), offset=np.zeros(n, dtype=bool),
               n_laps=0, n_laps_session_edge=0, n_laps_kept=0)
    field = maps['main_field']
    if field is not None:
        trav = field_traversals(pos_ts, pos_xy, maps, field)
        n_laps = len(trav['first'])
        spk_frame = (np.cumsum(trace['moving']) - 1)[spk_idx]     # nearest sample in the moving-only sequence
        spk_lap = np.searchsorted(trav['first'], spk_frame, side='right') - 1
        in_lap = analysed & trace['moving'][spk_idx] & (spk_lap >= 0)
        in_lap[in_lap] = spk_frame[in_lap] <= trav['last'][spk_lap[in_lap]]
        n_per_lap = np.bincount(spk_lap[in_lap], minlength=n_laps)
        kept = np.flatnonzero(n_per_lap >= MIN_SPIKES_PER_LAP)
        for j in kept:
            sel = np.flatnonzero(in_lap & (spk_lap == j))
            out['onset'][sel[0]] = True
            out['offset'][sel[-1]] = True
        out.update(in_lap=in_lap, n_laps=n_laps, n_laps_session_edge=trav['n_session_edge'],
                   n_laps_kept=len(kept))

    summary = {'n_spikes_analysed': int(analysed.sum()), 'n_spikes_in_laps': int(out['in_lap'].sum()),
               'n_laps': out['n_laps'], 'n_laps_session_edge': out['n_laps_session_edge'],
               'n_laps_kept': out['n_laps_kept']}
    for kind, prefix, count_col, min_n in (('peakbin', 'PeakBin', 'PeakBin_n_spikes', MIN_PEAK_BIN_SPIKES),
                                           ('onset', 'Onset', 'Onset_n_laps', MIN_LAPS_FOR_ONOFF),
                                           ('offset', 'Offset', 'Offset_n_laps', MIN_LAPS_FOR_ONOFF)):
        ph = theta[out[kind]]
        ok = len(ph) >= max(min_n, 1)
        mean_deg = float(np.mod(np.rad2deg(circ_mean(ph)), 360)) if ok else np.nan
        summary.update({count_col: len(ph), f'{prefix}_phase_deg': mean_deg,
                        f'{prefix}_MRL': circ_r(ph) if ok else np.nan,
                        f'{prefix}_range': phase_range_label(mean_deg)})

    # Theta-phase span of the peak-bin spikes (>= MIN_PEAK_BIN_SPIKES_FOR_SPAN of them)
    pk_deg = np.rad2deg(theta[peakbin])
    out['span_shifted'] = None
    summary.update(PeakBin_PhaseSpan_deg=np.nan, PeakBin_PhaseSpan_start_deg=np.nan, PeakBin_SpanPhases_deg='')
    if len(pk_deg) >= MIN_PEAK_BIN_SPIKES_FOR_SPAN:
        shifted, start, span = circular_phase_span(pk_deg)
        out['span_shifted'] = shifted
        summary.update(PeakBin_PhaseSpan_deg=span, PeakBin_PhaseSpan_start_deg=start,
                       PeakBin_SpanPhases_deg=';'.join(f'{v:.2f}' for v in np.sort(shifted)))
    out['summary'] = summary
    return out


def circular_phase_span(phase_deg):
    """Span of a set of theta phases on the circle, independent of where on the theta cycle it lies:
    the phases are cut at their largest circular gap, span = 360 deg - that gap, and every phase is
    re-measured (deg) from the first phase after the gap, so the re-measured phases run from 0 to span.
    E.g. 350 and 10 deg -> span 20 deg, re-measured 0 and 20 deg.
    Returns (shifted_deg in the input order, start_deg, span_deg)."""
    ph = np.mod(np.asarray(phase_deg, dtype=float), 360)
    if len(ph) == 0:
        return ph, np.nan, np.nan
    srt = np.sort(ph)
    if len(srt) == 1:
        return np.zeros(1), float(srt[0]), 0.0
    gaps = np.diff(np.r_[srt, srt[0] + 360])          # gap after each sorted phase (the last wraps)
    k = int(np.argmax(gaps))
    start = float(srt[(k + 1) % len(srt)])
    return np.mod(ph - start, 360), start, float(360 - gaps[k])


def _span_grid():
    return np.arange(0, 360 + SPAN_GRID_STEP_DEG / 2, SPAN_GRID_STEP_DEG)


def span_density(shifted_deg, sigma=SPAN_KDE_SIGMA_DEG):
    """Gaussian kernel density (1/deg, area 1) of one unit's re-measured peak-bin phases
    (circular_phase_span) on _span_grid(), supported on [0, span] (span = the largest re-measured
    phase): kernels are reflected at both ends of the span, so neither end loses density (a uniform
    spread gives a flat density), and the density is 0 beyond the span. A zero span (all phases
    equal) gives a kernel reflected at 0 only."""
    grid = _span_grid()
    s = np.asarray(shifted_deg, dtype=float)[:, None]
    span = float(s.max())

    def k(x):
        return np.exp(-0.5 * ((grid - x) / sigma) ** 2)

    if span < SPAN_GRID_STEP_DEG:
        d = (k(s) + k(-s)).sum(axis=0)
    else:
        d = (k(s) + k(-s) + k(2 * span - s)).sum(axis=0)
        d[grid > span] = 0.0
    return d / trapezoid(d, grid)


# ============================================================================
# ACG theta epoch detection (Dunn et al. 2022, Nat Commun 13:6997, Supp. Fig. 5;
# protocol of Figc_ACG_ScalingCorrected_v8_NoiseCleaning.py).
#
# The repaired LFP is cleaned (clean_lfp, shared with the Hilbert phase path:
# resample to ACG_FS -> notch -> wavelet denoise -> detrend -> outlier fill ->
# low-pass) and split into ACG_EPOCH_SEC epochs. Noise epochs are thrown out:
# delta power above theta power, then robust (MAD) peak-to-peak outliers. Each
# remaining epoch's autocorrelogram is compared with a bank of reference-sinusoid
# autocorrelograms spanning the broad ACG_REF_BANK; freq_est = the frequency of
# the closest one (ED_min = normalised Euclidean distance to it). The epoch is a
# theta epoch only if freq_est lies in ACG_FREQ_RANGE; out-of-band epochs are
# thrown out. There is no threshold on the ED_min value itself.
# ============================================================================

def _acg_notch(x, fs_hz, freqs, q):
    y = np.asarray(x, dtype=np.float64)
    for f0 in freqs:
        if 0 < f0 < fs_hz / 2.0:
            b, a = signal.iirnotch(f0, q, fs_hz)
            y = signal.filtfilt(b, a, y)
    return y


def _denoise_wavelet(x):
    """Soft-threshold the ACG_WAVELET_NAME detail coefficients, with the noise level estimated
    from the finest level. BayesShrink (Chang et al. 2000), standing in for wdenoise's default
    empirical Bayes threshold; a level with no signal above noise is zeroed."""
    coeffs = pywt.wavedec(x, ACG_WAVELET_NAME, level=ACG_WAVELET_LEVEL)
    noise_sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    denoised = [coeffs[0]]
    for detail in coeffs[1:]:
        signal_var = max(np.mean(detail ** 2) - noise_sigma ** 2, 0)
        thresh = noise_sigma ** 2 / np.sqrt(signal_var) if signal_var > 0 else np.abs(detail).max()
        denoised.append(pywt.threshold(detail, thresh, 'soft'))
    return pywt.waverec(denoised, ACG_WAVELET_NAME)[:len(x)]


def _fill_outliers(x, window=ACG_OUTLIER_WINDOW, thresh=ACG_OUTLIER_THRESH, chunk=20000):
    """MATLAB's filloutliers(x, 'linear', 'movmedian', window): a sample more than `thresh`
    scaled MADs from its moving median is replaced by linear interpolation. Windows are k/2
    back and k/2 - 1 forward, shrinking at the ends. Returns the filled trace and the outlier mask."""
    n = len(x)
    n_before = window // 2
    n_after = window - n_before - 1
    med, mad = np.empty(n), np.empty(n)
    windows = sliding_window_view(x, window)
    for j in range(0, len(windows), chunk):
        block = windows[j:j + chunk]
        m = np.median(block, axis=1)
        med[j + n_before:j + n_before + len(block)] = m
        mad[j + n_before:j + n_before + len(block)] = np.median(np.abs(block - m[:, None]), axis=1)
    for k in [*range(n_before), *range(n - n_after, n)]:
        block = x[max(0, k - n_before):min(n, k + n_after + 1)]
        med[k] = np.median(block)
        mad[k] = np.median(np.abs(block - med[k]))
    is_outlier = np.abs(x - med) > thresh * 1.482602218505602 * mad

    filled = x.copy()
    filled[is_outlier] = np.interp(np.flatnonzero(is_outlier), np.flatnonzero(~is_outlier), x[~is_outlier])
    return filled, is_outlier


def clean_lfp(lfp_sig, start_s, lfp_fs):
    """ACG-pipeline LFP cleaning (clean_lfp of Figc_ACG_ScalingCorrected_v8_NoiseCleaning.py),
    applied before both the Hilbert phase and the ACG epoch detection: the repaired native-rate
    trace (load_ncs) -> ACG_FS -> ACG_LINE_HARMONICS notch -> wavelet denoise -> linear detrend ->
    outlier fill (-> median filter -> Savitzky-Golay, if ACG_APPLY_SMOOTHING) -> low-pass.
    Returns (x, ts, ACG_FS, outlier_frac); ts = start_s + i / ACG_FS, on the recording's own
    (timestamp-anchored) clock; outlier_frac = fraction of samples outlier-filled."""
    x = signal.resample_poly(lfp_sig, int(ACG_FS), int(round(lfp_fs)))
    x = _acg_notch(x, ACG_FS, ACG_LINE_HARMONICS, ACG_NOTCH_Q)
    x = _denoise_wavelet(x)
    x = signal.detrend(x, type='linear')
    x, is_outlier = _fill_outliers(x)
    if ACG_APPLY_SMOOTHING:
        x = signal.medfilt(x, ACG_MEDFILT_KERNEL)
        x = signal.savgol_filter(x, ACG_SGOLAY_WINDOW, ACG_SGOLAY_POLY)
    b, a = signal.butter(ACG_LOWPASS_ORDER, ACG_LOWPASS_CUTOFF_HZ / (ACG_FS / 2.0), btype='low')
    x = signal.filtfilt(b, a, x)
    ts = start_s + np.arange(len(x)) / ACG_FS
    return x, ts, ACG_FS, float(is_outlier.mean())


def _zero_crossings(x):
    """Fractional (linearly interpolated) sample indices where x crosses zero."""
    s = np.sign(x)
    s[s == 0] = 1.0
    idx = np.flatnonzero(np.diff(s) != 0)
    return idx - x[idx] / (x[idx + 1] - x[idx])


def _acg_reference_bank(ref_freqs, datsize, fs):
    """Reference sinusoid autocorrelograms and their peak-range search windows, as in
    create_sine_ref_xcorrs.m (create_sine_ref_xcorrs of Figc_ACG_ScalingCorrected_v8_NoiseCleaning.py).

    Returns (refXC, refED, refRange, p1, t1, t2): refXC is each reference's normalised
    autocorrelogram and refED its L2 norm; refRange is each reference's own peak range (first
    side peak minus mean of its flanking troughs); p1/t1/t2 are (2, n_freqs) [lo, hi] sample-index
    windows bracketing that side peak and the troughs before and after it, used to find the
    matching peak/troughs in a data autocorrelogram. References too slow to show a side peak and
    its flanking troughs within one epoch (below ~1.75 Hz for 1 s epochs) keep refXC / refED, so
    they still take part in the ED_min match, but get refRange = NaN and windows of -1.
    """
    reft = np.arange(datsize) / fs
    center = datsize - 1
    n_freqs = len(ref_freqs)
    refXC = np.empty((2 * datsize - 1, n_freqs))
    refRange = np.full(n_freqs, np.nan)
    p1, t1, t2 = (np.full((2, n_freqs), -1, dtype=int) for _ in range(3))
    for n, freq in enumerate(ref_freqs):
        refsig = np.sin(2 * np.pi * freq * reft)
        xc = signal.correlate(refsig, refsig, mode='full', method='auto')
        refxc = xc / np.max(xc)
        refXC[:, n] = refxc

        max_idx, _ = signal.find_peaks(refxc, prominence=ACG_PEAK_PROMINENCE)
        min_idx, _ = signal.find_peaks(-refxc, prominence=ACG_PEAK_PROMINENCE)
        side = max_idx[max_idx > center]
        if not side.size:
            continue   # no side peak after the zero-lag peak within the epoch
        peak1 = side[0]
        before, after = min_idx[min_idx < peak1], min_idx[min_idx > peak1]
        if not (before.size and after.size):
            continue   # can't bracket troughs around the first side peak
        trough1, trough2 = before[-1], after[0]

        zc = _zero_crossings(refxc)
        below_t1, above_t2 = zc[zc < trough1], zc[zc > trough2]
        below_p1, above_p1 = zc[zc < peak1], zc[zc > peak1]
        if not (below_t1.size and above_t2.size and below_p1.size and above_p1.size):
            continue   # no bounding zero-crossings for the search windows
        refRange[n] = refxc[peak1] - (refxc[trough1] + refxc[trough2]) / 2
        lo_p1, hi_p1 = int(round(below_p1[-1])), int(round(above_p1[0]))
        p1[:, n] = [lo_p1, hi_p1]
        t1[:, n] = [int(round(below_t1[-1])), lo_p1]
        t2[:, n] = [hi_p1, int(round(above_t2[0]))]
    return refXC, np.linalg.norm(refXC, axis=0), refRange, p1, t1, t2


def acg_theta_epochs(lfp_clean, records=None):
    """Per-epoch ACG theta-epoch detection for one LFP trace, already cleaned by clean_lfp.

    Epoch i is the i-th block of ACG_EPOCH_SEC * ACG_FS consecutive samples; use
    times_in_ok_epochs to match absolute times to epochs by the samples' timestamps.
    `records` (from load_ncs) is needed for ACG_REJECT_PARTIAL_RECORD_EPOCHS and
    ACG_REJECT_SATURATED_EPOCHS.

    Returns a dict of per-epoch arrays:
      ok             True for theta epochs (status 'accepted')
      status         'accepted', or the first rejection: 'rejected_partial' (short .ncs record),
                     'rejected_saturated' (clipped at +/-InputRange, or within SATURATION_MARGIN_S
                     of clipping), 'rejected_deltatheta' (delta power > theta power), 'rejected_artifact'
                     (peak-to-peak MAD outlier), 'rejected_flat' (no autocorrelogram), or
                     'rejected_outofband' (freq_est outside ACG_FREQ_RANGE: non-theta)
      ED_min         normalised distance to the closest reference sinusoid (NaN if not fitted)
      freq_est       frequency of that reference (Hz)
      peakrangenorm  autocorrelogram peak range normalised by the matched reference's own peak
                     range (1 = clean sinusoid, 0 = flat; NaN if the matched reference is too slow
                     to have one) -- QC plot only, it does not decide anything
      n_clipped      number of clipped native-rate samples in the epoch (0 without a saturation check)
    """
    x = lfp_clean
    nperseg = int(round(ACG_FS * ACG_EPOCH_SEC))
    n_total = len(x) // nperseg
    status = np.full(n_total, 'accepted', dtype=object)
    ed_min, freq_est, peakrangenorm = (np.full(n_total, np.nan) for _ in range(3))
    n_clipped = np.zeros(n_total, dtype=np.int64)
    if n_total == 0:
        return dict(ok=np.zeros(0, dtype=bool), status=status, ED_min=ed_min, freq_est=freq_est,
                    peakrangenorm=peakrangenorm, n_clipped=n_clipped)
    epochs = x[:n_total * nperseg].reshape(n_total, nperseg)

    keep = np.ones(n_total, dtype=bool)
    if ACG_REJECT_PARTIAL_RECORD_EPOCHS and records is not None:
        # map each epoch's first / last sample back to the native-rate grid and its records
        step = NCS_NATIVE_FS / ACG_FS
        i = np.arange(n_total)
        first = np.round(i * nperseg * step).astype(np.int64)
        last = np.round(((i + 1) * nperseg - 1) * step).astype(np.int64)
        short_cum = np.concatenate(([0], np.cumsum(records['nvalid'] < NCS_SAMPLES_PER_RECORD)))
        r0 = np.searchsorted(records['start'], first, side='right') - 1
        r1 = np.searchsorted(records['start'], last, side='right') - 1
        partial = (short_cum[r1 + 1] - short_cum[r0]) > 0
        status[partial] = 'rejected_partial'
        keep &= ~partial
    if records is not None and records.get('input_range_uv') is not None:
        n_clipped, near_clip = epoch_saturation(records['saturated'], n_total, nperseg, ACG_FS)
        is_sat = n_clipped > 0
        print(f'  saturation (InputRange {records["input_range_uv"]:g} uV): {int(is_sat.sum())}/{n_total} '
              f'epochs contain clipped samples, {int((near_clip & ~is_sat).sum())} more within '
              f'{SATURATION_MARGIN_S:g} s of clipping; {100 * records["saturated"].mean():.3f}% of samples clipped')
        if ACG_REJECT_SATURATED_EPOCHS:
            status[keep & near_clip] = 'rejected_saturated'
            keep &= ~near_clip
    elif records is not None:
        print('  saturation: no -InputRange in header -- check skipped')
    if ACG_REJECT_DELTA_OVER_THETA:
        win = signal.get_window('hann', nperseg)
        for i in np.flatnonzero(keep):
            f, pxx = signal.welch(epochs[i], fs=ACG_FS, window=win, nperseg=nperseg,
                                  noverlap=0, detrend='constant')
            low = pxx[(f >= ACG_DELTA_BAND[0]) & (f <= ACG_DELTA_BAND[1])].sum()
            theta = pxx[(f >= ACG_FREQ_RANGE[0]) & (f <= ACG_FREQ_RANGE[1])].sum()
            if low > theta:
                keep[i] = False
                status[i] = 'rejected_deltatheta'
    if ACG_REJECT_MAD_ARTIFACTS and keep.any():
        # median / MAD from the epochs still kept, so epochs rejected above can't skew the threshold
        p2p = epochs.max(axis=1) - epochs.min(axis=1)
        med = np.median(p2p[keep])
        mad = np.median(np.abs(p2p[keep] - med)) or 1e-20
        artifact = keep & (0.6745 * (p2p - med) / mad > ACG_MAD_THRESH)
        status[artifact] = 'rejected_artifact'
        keep &= ~artifact

    # round so 0.1 Hz steps compare exactly with the ACG_FREQ_RANGE edges
    ref_freqs = np.round(np.arange(ACG_REF_BANK[0], ACG_REF_BANK[1] + ACG_FREQ_RES / 2, ACG_FREQ_RES), 6)
    refXC, refED, refRange, p1, t1, t2 = _acg_reference_bank(ref_freqs, nperseg, ACG_FS)

    for i in np.flatnonzero(keep):
        xc = signal.correlate(epochs[i], epochs[i], mode='full', method='auto')
        peak = np.max(xc)
        if not np.isfinite(peak) or peak <= 0:
            status[i] = 'rejected_flat'
            continue
        xc = xc / peak
        ed = np.sqrt(np.sum((refXC - xc[:, None]) ** 2, axis=0)) / refED
        mi = int(np.argmin(ed))
        ed_min[i], freq_est[i] = ed[mi], ref_freqs[mi]
        if not ACG_FREQ_RANGE[0] <= freq_est[i] <= ACG_FREQ_RANGE[1]:
            status[i] = 'rejected_outofband'
        if np.isfinite(refRange[mi]):
            pk = xc[p1[0, mi]:p1[1, mi] + 1].max()
            tr1 = xc[t1[0, mi]:t1[1, mi] + 1].min()
            tr2 = xc[t2[0, mi]:t2[1, mi] + 1].min()
            peakrangenorm[i] = (pk - (tr1 + tr2) / 2) / refRange[mi]

    return dict(ok=status == 'accepted', status=status, ED_min=ed_min, freq_est=freq_est,
                peakrangenorm=peakrangenorm, n_clipped=n_clipped)


def times_in_ok_epochs(t_s, lfp_ts, lfp_fs, epoch_ok):
    """Boolean mask over absolute times `t_s` (s, on the same PTP-synced clock as
    `lfp_ts`) that fall inside an ACG theta epoch. Epoch i is the i-th block of
    ACG_EPOCH_SEC * ACG_FS LFP samples, so each time is matched to its nearest LFP
    sample by timestamp and takes that sample's epoch, which stays correct across
    recording gaps. Times with no LFP sample within one inter-sample interval (see
    spikes_with_lfp_sample) belong to no epoch."""
    t_s = np.asarray(t_s, dtype=np.float64)
    out = np.zeros(len(t_s), dtype=bool)
    if len(t_s) == 0 or len(lfp_ts) == 0:
        return out
    idx = np.searchsorted(lfp_ts, t_s)
    left = np.clip(idx - 1, 0, len(lfp_ts) - 1)
    right = np.clip(idx, 0, len(lfp_ts) - 1)
    nearest = np.where(np.abs(t_s - lfp_ts[left]) <= np.abs(lfp_ts[right] - t_s), left, right)
    epoch = nearest // int(round(ACG_FS * ACG_EPOCH_SEC))
    inb = (epoch < len(epoch_ok)) & spikes_with_lfp_sample(t_s, lfp_ts, lfp_fs)
    out[inb] = epoch_ok[epoch[inb]]
    return out


# ============================================================================
# Theta phase by direct Hilbert transform.
#
# Phase = angle of the analytic signal of the bandpass-filtered trace
# (np.angle(signal.hilbert(x))): 0 at a peak, +/-pi at a trough. The raw
# Hilbert phase is used as is -- the Generalized Phase correction (Davis,
# Muller et al. 2020) for low/negative-frequency epochs is deliberately NOT
# applied here.
# ============================================================================

def hilbert_phase(x, fs):
    """Instantaneous phase (rad, wrapped to (-pi, pi], 0 at a peak) and
    instantaneous frequency (in units of fs, per sample) of the already
    bandpass-filtered real signal `x`, from its Hilbert analytic signal.
    Returns (ph, wt); ph is all NaN if the analytic signal is."""
    x = np.asarray(x, dtype=np.float64)
    npts = x.shape[0]
    dt = 1.0 / fs

    def _inst_freq(xo):
        wt = np.zeros(npts)
        wt[:-1] = np.angle(xo[1:] * np.conj(xo[:-1])) / (2 * np.pi * dt)
        return wt

    # analytic signal representation (scipy.signal.hilbert already implements
    # the single-sided FFT approach of Marple 1999 used by the MATLAB original)
    xo = signal.hilbert(x)
    ph = np.angle(xo)
    md = np.abs(xo)
    wt_raw = _inst_freq(xo)

    # rectify rotation direction so instantaneous frequency is positive
    finite_wt = wt_raw[np.isfinite(wt_raw)]
    sign_if = np.sign(np.mean(finite_wt)) if finite_wt.size else 1.0
    if sign_if == -1:
        xo = md * np.exp(1j * (sign_if * ph))
        ph = np.angle(xo)
        md = np.abs(xo)
        wt_raw = _inst_freq(xo)

    if np.all(np.isnan(ph)):
        return np.full(npts, np.nan), wt_raw
    return ph, wt_raw


def phase_at(t, phase_unwrapped, query_t):
    """Unwrapped phase (rad) of a trace sampled at times `t`, linearly
    interpolated at `query_t`. Interpolating the unwrapped phase (not the
    wrapped angle) keeps the 2*pi wraparound from corrupting values between
    samples. NaN outside [t[0], t[-1]]."""
    return np.interp(np.asarray(query_t, dtype=np.float64), t, phase_unwrapped,
                     left=np.nan, right=np.nan)


# ============================================================================
# Circular statistics
# ============================================================================

def circ_r(alpha_rad):
    """Mean resultant vector length (circ_r.m, unweighted/unbinned case)."""
    return float(np.abs(np.mean(np.exp(1j * alpha_rad))))


def circ_mean(alpha_rad):
    """Circular mean (circ_mean.m)."""
    return float(np.angle(np.mean(np.exp(1j * alpha_rad))))


def circ_rtest(alpha_rad):
    """Rayleigh test for non-uniformity of circular data (circ_rtest.m)."""
    n = len(alpha_rad)
    r = circ_r(alpha_rad)
    R = n * r
    z = R ** 2 / n
    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))
    return float(pval), float(z)


def circ_kappa(r):
    """Von Mises concentration estimate from a mean resultant length r
    (circ_kappa.m approximation, no small-sample correction)."""
    if r < 0.53:
        return 2 * r + r ** 3 + 5 * r ** 5 / 6
    if r < 0.85:
        return -0.4 + 1.39 * r + 0.43 / (1 - r)
    if r >= 1:
        return np.inf
    return 1 / (r ** 3 - 4 * r ** 2 + 3 * r)


def watson_williams_test(groups_rad):
    """Watson-Williams k-sample test for equal mean direction (circ_wwtest.m).
    Parametric: assumes von Mises samples sharing one concentration (the
    approximation is poor when the pooled mean resultant length is < ~0.45).
    Returns (F, p, kappa)."""
    ns = np.array([len(g) for g in groups_rad])
    k, n_total = len(groups_rad), int(ns.sum())
    R_i = np.array([np.abs(np.sum(np.exp(1j * g))) for g in groups_rad])
    R = np.abs(np.sum(np.exp(1j * np.concatenate(groups_rad))))
    kappa = circ_kappa(R_i.sum() / n_total)
    denom = n_total - R_i.sum()
    if k < 2 or n_total <= k or kappa <= 0 or denom <= 0:
        return np.nan, np.nan, kappa
    beta = 1 + 3 / (8 * kappa)
    F = beta * (n_total - k) / (k - 1) * (R_i.sum() - R) / denom
    return float(F), float(stats.f.sf(F, k - 1, n_total - k)), kappa


def _mww_statistic(cos_sc, sin_sc, labels, n_groups):
    W = 0.0
    for g in range(n_groups):
        sel = labels == g
        W += (cos_sc[sel].sum() ** 2 + sin_sc[sel].sum() ** 2) / sel.sum()
    return 2 * W


def mardia_watson_wheeler_test(groups_rad, rng, n_perm=N_MWW_PERMUTATIONS):
    """Mardia-Watson-Wheeler uniform-scores k-sample test: nonparametric test
    that the circular distributions are identical (sensitive to differences in
    location or spread). The pooled angles are ranked and mapped to uniform
    scores 2*pi*rank/N; W = 2 * sum_i (C_i^2 + S_i^2) / n_i. Returns
    (W, p_chi2, p_perm): p_chi2 from the asymptotic chi^2(2(k-1)) distribution
    (needs ~10+ units per group), p_perm from n_perm group-label permutations
    (valid at any sample size)."""
    k = len(groups_rad)
    pooled = np.mod(np.concatenate(groups_rad), 2 * np.pi)
    labels = np.repeat(np.arange(k), [len(g) for g in groups_rad])
    scores = 2 * np.pi * stats.rankdata(pooled) / len(pooled)
    cos_sc, sin_sc = np.cos(scores), np.sin(scores)

    W = _mww_statistic(cos_sc, sin_sc, labels, k)
    p_chi2 = float(stats.chi2.sf(W, 2 * (k - 1)))
    perm_W = np.array([_mww_statistic(cos_sc, sin_sc, rng.permutation(labels), k)
                       for _ in range(n_perm)])
    p_perm = float((np.sum(perm_W >= W - 1e-12) + 1) / (n_perm + 1))
    return float(W), p_chi2, p_perm


def holm_adjust(pvals):
    """Holm-Bonferroni step-down adjusted p-values (NaNs left as NaN)."""
    p = np.asarray(pvals, dtype=float)
    adj = np.full(len(p), np.nan)
    finite_idx = np.flatnonzero(np.isfinite(p))
    m = len(finite_idx)
    running = 0.0
    for rank, idx in enumerate(finite_idx[np.argsort(p[finite_idx])]):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj


def phase_range_label(phase_deg):
    """'trough' if the phase (deg, any cycle) falls in THETA_TROUGH_RANGE_DEG on the
    circle, else 'peak'; None for NaN."""
    if not np.isfinite(phase_deg):
        return None
    lo, hi = THETA_TROUGH_RANGE_DEG
    return 'trough' if lo <= np.mod(phase_deg, 360) < hi else 'peak'


# ============================================================================
# Steps 1 & 2: theta phase-locking, Theta Modulation Index, shuffle test
# ============================================================================

def _gauss_kernel(n=11, sigma=1.0):
    """An n-point 1D Gaussian smoothing kernel (AditiPrecessionUtils.gauss)."""
    r = np.arange(-(n // 2), n // 2 + 1)
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(r.astype(float) ** 2) / (2 * sigma ** 2))


def _gauss_smooth_1d(y, n=11, sigma=1.0):
    """Edge-padded Gaussian convolution smoothing (AditiPrecessionUtils.gaussSmooth)."""
    g = _gauss_kernel(n, sigma)
    y = np.asarray(y, dtype=float)
    padded = np.pad(y, (len(g),), mode='edge')
    smoothed = np.convolve(padded, g, 'same')
    return smoothed[len(g):len(smoothed) - len(g)]


def calc_tmi(spike_phase_deg, num_cycles=5, bin_width_deg=36, return_hist=False):
    """Theta Modulation Index (AditiPrecessionUtils.calcTMI): tiles spike
    phases across num_cycles repeated 360-degree cycles, bins/smooths the
    histogram, keeps the middle two cycles, max-normalizes, and defines
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


def spikes_with_lfp_sample(spk_ts, lfp_ts, lfp_fs, jitter_factor=LFP_MATCH_JITTER_FACTOR):
    """Boolean mask: True where the LFP sample closest in time to the spike is
    within 1/(jitter_factor*lfp_fs) s of it (i.e. one inter-sample interval, less
    a margin for timestamp jitter). False for spikes in LFP gaps / outside the
    recording, which must be skipped."""
    spk_ts = np.asarray(spk_ts, dtype=np.float64)
    if len(lfp_ts) == 0 or len(spk_ts) == 0:
        return np.zeros(len(spk_ts), dtype=bool)
    idx = np.searchsorted(lfp_ts, spk_ts)
    left = np.clip(idx - 1, 0, len(lfp_ts) - 1)
    right = np.clip(idx, 0, len(lfp_ts) - 1)
    nearest_dt = np.minimum(np.abs(spk_ts - lfp_ts[left]), np.abs(lfp_ts[right] - spk_ts))
    return nearest_dt <= 1.0 / (jitter_factor * lfp_fs)


def assign_spike_phase(spk_ts, lfp_ts, lfp_phase_unwrapped, lfp_fs):
    """Hilbert theta phase (rad, [0, 2*pi), 0 at an LFP theta peak) at each
    spike time, interpolated from the unwrapped per-sample Hilbert phase of
    the filtered LFP (see hilbert_phase / phase_at). NaN for spikes outside
    the LFP's time range, or with no LFP sample within one inter-sample
    interval (see spikes_with_lfp_sample)."""
    phase = np.mod(phase_at(lfp_ts, lfp_phase_unwrapped, spk_ts), 2 * np.pi)
    phase[~spikes_with_lfp_sample(spk_ts, lfp_ts, lfp_fs)] = np.nan
    return phase


def _burst_groups(spk_ts, phase_deg, max_gap_sec=TMI_BURST_MAX_GAP_SEC,
                  phase_bin_deg=TMI_BURST_PHASE_BIN_DEG):
    """Group spikes into bursts for the shuffle test below: consecutive
    spikes (in time order) are merged into the same group if they are
    < max_gap_sec apart AND fall in the same real-data phase bin (Frank et
    al. 2001's rule for keeping burst spikes assigned to the same phase
    under shuffling). Returns a group-id array aligned to spk_ts's original
    order (arbitrary integers, not necessarily contiguous)."""
    order = np.argsort(spk_ts)
    ts_sorted = spk_ts[order]
    bin_idx_sorted = np.floor(phase_deg[order] / phase_bin_deg).astype(np.int64)

    same_burst = (np.diff(ts_sorted) < max_gap_sec) & (np.diff(bin_idx_sorted) == 0)
    group_id_sorted = np.concatenate(([0], np.cumsum(~same_burst)))

    group_id = np.empty_like(group_id_sorted)
    group_id[order] = group_id_sorted
    return group_id


def shuffle_tmi_significance(spk_ts, phase_deg, observed_tmi, rng, n_shuffles=N_TMI_SHUFFLES):
    """Null distribution for the TMI (Frank et al. 2001): each of n_shuffles
    surrogates assigns every spike an independent random phase in [0, 360),
    except that consecutive spikes forming a burst (see _burst_groups) are
    kept together and given the same shuffled phase, preserving burst
    temporal structure while destroying true theta-phase alignment.

    Returns
    -------
    pval         : fraction of shuffle TMIs >= observed_tmi (add-one
                   smoothed so p is never exactly 0).
    shuffle_tmis : ndarray of the n_shuffles shuffle TMI scores.
    """
    group_id = _burst_groups(spk_ts, phase_deg)
    n_groups = int(group_id.max()) + 1

    shuffle_tmis = np.empty(n_shuffles)
    for i in range(n_shuffles):
        group_phases = rng.uniform(0.0, 360.0, size=n_groups)
        shuffle_tmis[i] = calc_tmi(group_phases[group_id])

    pval = float((np.sum(shuffle_tmis >= observed_tmi) + 1) / (n_shuffles + 1))
    return pval, shuffle_tmis


def theta_mod_class(rayleigh_sig, tmi_sig, tested):
    """Exclusive theta-modulation class of one unit (ThetaModClass column)."""
    if not tested:
        return 'too few spikes'
    if rayleigh_sig and tmi_sig:
        return 'Rayleigh+ & TMI+'
    if rayleigh_sig:
        return 'Rayleigh+ only'
    return 'TMI+ only' if tmi_sig else 'neither'


def compute_theta_modulation(spk_ts, lfp_ts, lfp_phase_unwrapped, lfp_fs, rng):
    """Steps 1 & 2 for one unit: Hilbert theta-phase statistics (MRL, preferred
    phase, Rayleigh test) plus the Theta Modulation Index and its shuffle-test
    significance. Spikes with no defined phase (outside the LFP's time range or
    in an LFP gap) are dropped. ThetaModulated = Rayleigh+ AND TMI+.

    Returns (metrics dict, phase_deg array of per-spike theta phase, for
    plotting, shuffle_tmis array of the TMI shuffle-test null distribution --
    None if too few spikes to run it).
    """
    phase_rad = assign_spike_phase(spk_ts, lfp_ts, lfp_phase_unwrapped, lfp_fs)
    has_phase = np.isfinite(phase_rad)
    spk_ts, phase_rad = spk_ts[has_phase], phase_rad[has_phase]
    phase_deg = np.degrees(phase_rad)
    n = len(phase_deg)

    metrics = dict(n_spikes_theta=n)
    if n < MIN_SPIKES_FOR_TMI:
        metrics.update(MRL=np.nan, PreferredPhase_deg=np.nan, Rayleigh_p=np.nan,
                       Rayleigh_Significant=False, PhasePeak_deg=np.nan,
                       PhaseValley_deg=np.nan, TMI=np.nan, TMI_shuffle_p=np.nan,
                       TMI_Significant=False, ThetaModulated=False,
                       ThetaModClass=theta_mod_class(False, False, False))
        return metrics, phase_deg, None

    mrl = circ_r(phase_rad)
    pref_phase_deg = np.degrees(circ_mean(phase_rad)) % 360
    rayleigh_p, _ = circ_rtest(phase_rad)

    peak_phase_deg, valley_phase_deg, tmi = find_phase_peak_valley(phase_deg)
    tmi_pval, shuffle_tmis = shuffle_tmi_significance(spk_ts, phase_deg, tmi, rng)
    tmi_sig = bool(np.isfinite(tmi_pval) and tmi_pval < ALPHA)
    rayleigh_sig = bool(rayleigh_p < ALPHA)

    metrics.update(MRL=mrl, PreferredPhase_deg=pref_phase_deg, Rayleigh_p=rayleigh_p,
                   Rayleigh_Significant=rayleigh_sig,
                   PhasePeak_deg=peak_phase_deg, PhaseValley_deg=valley_phase_deg,
                   TMI=tmi, TMI_shuffle_p=tmi_pval, TMI_Significant=tmi_sig,
                   ThetaModulated=tmi_sig and rayleigh_sig,
                   ThetaModClass=theta_mod_class(rayleigh_sig, tmi_sig, True))
    return metrics, phase_deg, shuffle_tmis


# ============================================================================
# Per-unit plots
# ============================================================================

def _draw_polar_theta(ax, phase_deg, mrl, pref_phase_deg, rayleigh_p, is_sig, title=None,
                      bin_size_deg=PHASE_BIN_SIZE_DEG):
    """Draw the Step 1 polar histogram (spike counts vs. Hilbert theta phase,
    with the MRL/preferred-phase vector) on a polar ax. title, if given,
    heads the panel title above the spike-count / Rayleigh details."""
    edges_deg = np.arange(0, 360 + bin_size_deg, bin_size_deg)
    edges_rad = np.radians(edges_deg)
    counts, _ = np.histogram(np.radians(phase_deg), bins=edges_rad)
    width = np.radians(bin_size_deg)

    if is_sig:
        face_color, vec_color, vec_width = '#B3B3B3', 'red', 4
    else:
        face_color, vec_color, vec_width = '#D9D9D9', '#4D4D4D', 2

    ax.bar(edges_rad[:-1], counts, width=width, align='edge',
           facecolor=face_color, edgecolor='black', linewidth=0.5)
    max_count = counts.max() if counts.max() > 0 else 1
    ax.plot([0, np.radians(pref_phase_deg)], [0, mrl * max_count],
            color=vec_color, linewidth=vec_width)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    sig_str = 'SIGNIFICANT' if is_sig else 'not significant'
    ax.set_title(
        (f"{title}\n" if title else '') +
        f"n={len(phase_deg)} spikes | MRL={mrl:.3f} | pref={pref_phase_deg:.0f} deg\n"
        f"Rayleigh p={rayleigh_p:.3g} ({sig_str})", fontsize=9, pad=14)


def plot_polar_theta(phase_deg, mrl, pref_phase_deg, rayleigh_p, is_sig, title, out_path: Path,
                     bin_size_deg=PHASE_BIN_SIZE_DEG):
    """Step 1: polar histogram of spike counts vs. Hilbert theta phase, with the
    MRL/preferred-phase vector (red if the Rayleigh test is significant)."""
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection='polar')
    _draw_polar_theta(ax, phase_deg, mrl, pref_phase_deg, rayleigh_p, is_sig, title, bin_size_deg)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_phase_histogram(phase_deg, is_sig, title, out_path: Path,
                         bin_size_deg=PHASE_BIN_SIZE_DEG):
    """Linear histogram of spike counts vs. Hilbert theta phase
    bin (same phase data and bin width as plot_polar_theta, shown over two
    repeated 360-degree cycles for readability)."""
    edges_deg = np.arange(0, 360 + bin_size_deg, bin_size_deg)
    counts, _ = np.histogram(phase_deg, bins=edges_deg)
    centers_deg = edges_deg[:-1] + bin_size_deg / 2

    centers_dup = np.concatenate([centers_deg, centers_deg + 360])
    counts_dup = np.concatenate([counts, counts])

    face_color = '#B3B3B3' if is_sig else '#D9D9D9'

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(centers_dup, counts_dup, width=bin_size_deg, align='center',
           facecolor=face_color, edgecolor='black', linewidth=0.5)
    ax.set_xlim(0, 720)
    ax.set_xticks(np.arange(0, 720 + 1, 180))
    ax.set_xlabel('Theta phase (deg)')
    ax.set_ylabel('Spike count')
    sig_str = 'SIGNIFICANT' if is_sig else 'not significant'
    ax.set_title(f'{title}\nn={len(phase_deg)} spikes ({sig_str})', fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _tmi_verdict(tmi, tmi_pval, is_sig):
    """(colour, 'TMI=... | shuffle p=... | PASS/FAIL ...') for the TMI plots."""
    color = 'red' if is_sig else '#4D4D4D'
    verdict = 'PASS (TMI+)' if is_sig else 'FAIL (TMI-)'
    return color, f'TMI={tmi:.3f} | shuffle p={tmi_pval:.3g} | {verdict}'


def _draw_tmi_histogram(ax_h, phase_deg, tmi, is_sig):
    """Draw the smoothed/max-normalized phase histogram calc_tmi scores
    (TMI = 1 - its trough), with the trough-to-peak gap marked."""
    _tmi, edges2, bincentres, normcounts = calc_tmi(phase_deg, return_hist=True)
    bin_width = 2 * (bincentres[0] - edges2[0])
    color = 'red' if is_sig else '#4D4D4D'

    ax_h.bar(bincentres, normcounts, width=bin_width, align='center',
             facecolor='#B3B3B3', edgecolor='black', linewidth=0.5)
    i_min = int(np.argmin(normcounts))
    ax_h.axhline(1.0, color='0.5', linestyle=':', linewidth=1)
    ax_h.annotate('', xy=(bincentres[i_min], normcounts[i_min]), xytext=(bincentres[i_min], 1.0),
                  arrowprops=dict(arrowstyle='<->', color=color, linewidth=1.5))
    ax_h.text(bincentres[i_min], (1.0 + normcounts[i_min]) / 2, f'  TMI={tmi:.3f}',
              color=color, va='center', fontsize=9)
    ax_h.set_xlim(edges2[0], edges2[-1] + bin_width)
    ax_h.set_xticks(np.arange(0, 720 + 1, 180))
    ax_h.set_ylim(0, 1.05)
    ax_h.set_xlabel('Theta phase (deg)')
    ax_h.set_ylabel('Smoothed, max-normalized count')
    ax_h.set_title(f'TMI histogram (n={len(phase_deg)} spikes)', fontsize=9)


def _draw_tmi_null(ax_s, tmi, is_sig, shuffle_tmis):
    """Draw the TMI shuffle-test null distribution with the observed TMI and
    the null's (1 - ALPHA) percentile."""
    color = 'red' if is_sig else '#4D4D4D'
    thresh = float(np.percentile(shuffle_tmis, 100 * (1 - ALPHA)))
    ax_s.hist(shuffle_tmis, bins=40, color='#D9D9D9', edgecolor='black', linewidth=0.5)
    ax_s.axvline(thresh, color='0.3', linestyle='--', linewidth=1.5,
                 label=f'shuffle {100 * (1 - ALPHA):g}th pct = {thresh:.3f}')
    ax_s.axvline(tmi, color=color, linewidth=2.5, label=f'observed TMI = {tmi:.3f}')
    ax_s.set_xlabel('TMI')
    ax_s.set_ylabel('Shuffle count')
    ax_s.set_title(f'TMI shuffle null ({len(shuffle_tmis)} burst-preserving shuffles)', fontsize=9)
    ax_s.legend(fontsize=8, loc='upper left')


def plot_tmi_shuffle(phase_deg, tmi, tmi_pval, is_sig, shuffle_tmis, title, out_path: Path):
    """Step 2: left, the smoothed/max-normalized phase histogram calc_tmi scores
    (TMI = 1 - its trough); right, the TMI shuffle-test null distribution with
    the observed TMI and the null's (1 - ALPHA) percentile."""
    color, verdict_line = _tmi_verdict(tmi, tmi_pval, is_sig)
    fig, (ax_h, ax_s) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f'{title}\n{verdict_line}', fontsize=10, color=color)
    _draw_tmi_histogram(ax_h, phase_deg, tmi, is_sig)
    _draw_tmi_null(ax_s, tmi, is_sig, shuffle_tmis)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_peak_onset_offset(maps, res, title, out_path: Path, bin_size_deg=PHASE_BIN_SIZE_DEG):
    """Per-unit figure of Step 4: left, the raw rate map (main field outlined black, other
    fields dashed white, raw peak bin = star); middle, the jitter-smoothed trajectory with the
    analysed spikes (black), peak-bin spikes, and the onset / offset spike of every kept lap;
    right, the theta phase histograms of those three spike sets over two cycles, with their
    circular means (dashed), on the theta trough / peak range bands; far right, the peak-bin phase
    span: the peak-bin spike phases re-measured from the start of the span (circular_phase_span),
    histogram + the unit's density (span_density). A fallback main field (from the smoothed-map peak
    bin, marked x) is named in the map title."""
    s = res['summary']
    fig, (ax_map, ax_traj, ax_ph, ax_span) = plt.subplots(1, 4, figsize=(24, 5.5))
    x_edges, y_edges = maps['x_edges'], maps['y_edges']
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    xc, yc = 0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (y_edges[1:] + y_edges[:-1])

    im = ax_map.imshow(np.ma.masked_invalid(maps['raw_map_2d']).T, origin='lower', extent=extent,
                       cmap='jet', interpolation='nearest')
    fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04, label='Firing rate (Hz)')
    for f in maps['fields']:
        if f['mask_2d'].any():
            is_main = f is maps['main_field']
            ax_map.contour(xc, yc, f['mask_2d'].T.astype(float), levels=[0.5],
                           colors='black' if is_main else 'white', linewidths=1.8 if is_main else 1.0,
                           linestyles='solid' if is_main else 'dashed')
    ax_map.plot(*maps['raw_peak_xy'], marker='*', color='white', markeredgecolor='black', markersize=16,
                linestyle='none')
    main = maps['main_field']
    if maps['main_field_source'] == 'smoothed peak bin':
        ax_map.plot(*maps['smooth_peak_xy'], marker='X', color='white', markeredgecolor='black', markersize=12,
                    linestyle='none')
        field_str = (f'raw peak outside fields; main field = F{main["field_number"]} of {len(maps["fields"])} '
                     f'(holds smoothed peak, x)')
    elif main is not None:
        field_str = f'main field = F{main["field_number"]} of {len(maps["fields"])}'
    else:
        field_str = f'raw and smoothed peak outside all {len(maps["fields"])} field(s)'
    ax_map.set_title(f'Raw rate map ({maps["detection"]})\npeak bin {maps["raw_peak_rate_Hz"]:.2f} Hz (star); '
                     + field_str, fontsize=9)
    ax_map.set_xlabel('x (cm)')
    ax_map.set_ylabel('y (cm)')

    pos_sm = maps['trace']['pos_sm']
    xy = res['spk_xy']
    ax_traj.plot(pos_sm[:, 0], pos_sm[:, 1], color='0.75', linewidth=0.4, zorder=1)
    ax_traj.plot(xy[res['analysed'], 0], xy[res['analysed'], 1], '.', color='black', markersize=2,
                 label=f'analysed spikes (n={s["n_spikes_analysed"]})', zorder=2)
    for kind, color, name, n_str in (('peakbin', PEAK_BIN_SPIKE_COLOR, 'peak-bin', f'{s["PeakBin_n_spikes"]} spikes'),
                                     ('onset', FIRING_ONSET_COLOR, 'onset', f'{s["Onset_n_laps"]} laps'),
                                     ('offset', FIRING_OFFSET_COLOR, 'offset', f'{s["Offset_n_laps"]} laps')):
        sel = res[kind]
        ax_traj.plot(xy[sel, 0], xy[sel, 1], 'o', color=color, markersize=5, markeredgecolor='black',
                     markeredgewidth=0.4, label=f'{name} ({n_str})', zorder=3)
    ax_traj.set_xlim(extent[0], extent[1])
    ax_traj.set_ylim(extent[2], extent[3])
    ax_traj.set_aspect('equal')
    ax_traj.legend(fontsize=7, loc='upper right')
    ax_traj.set_title(f'Trajectory + spikes | laps of the main field: {s["n_laps_kept"]} kept '
                      f'(>= {MIN_SPIKES_PER_LAP} spikes) of {s["n_laps"]}', fontsize=9)
    ax_traj.set_xlabel('x (cm)')
    ax_traj.set_ylabel('y (cm)')

    lo, hi = THETA_TROUGH_RANGE_DEG
    for off in (0, 360):
        ax_ph.axvspan(lo + off, hi + off, color=PHASE_RANGE_COLORS['trough'], alpha=0.08, lw=0, zorder=0)
        ax_ph.axvspan(hi + off, 360 + lo + off, color=PHASE_RANGE_COLORS['peak'], alpha=0.25, lw=0, zorder=0)
    ax_ph.axvspan(0, lo, color=PHASE_RANGE_COLORS['peak'], alpha=0.25, lw=0, zorder=0)
    edges = np.arange(0, 720 + bin_size_deg, bin_size_deg)
    for kind, prefix, color, name in (('peakbin', 'PeakBin', PEAK_BIN_SPIKE_COLOR, 'peak-bin'),
                                      ('onset', 'Onset', FIRING_ONSET_COLOR, 'onset'),
                                      ('offset', 'Offset', FIRING_OFFSET_COLOR, 'offset')):
        ph = np.rad2deg(res['theta'][res[kind]])
        if not len(ph):
            continue
        mean_deg = s[f'{prefix}_phase_deg']
        ax_ph.hist(np.r_[ph, ph + 360], bins=edges, histtype='step', color=color, linewidth=1.8,
                   label=f'{name}: mean {mean_deg:.0f}°, MRL {s[f"{prefix}_MRL"]:.2f}' if np.isfinite(mean_deg)
                   else f'{name}: n < minimum')
        if np.isfinite(mean_deg):
            for off in (0, 360):
                ax_ph.axvline(mean_deg + off, color=color, linestyle='--', linewidth=1.2)
    ax_ph.set_xlim(0, 720)
    ax_ph.set_xticks(np.arange(0, 721, 90))
    ax_ph.set_xlabel('Theta phase (deg; 0 = theta peak)')
    ax_ph.set_ylabel('Spike count')
    ax_ph.legend(fontsize=7, loc='upper right')
    ax_ph.set_title(f'Theta phase (purple band = trough range {lo:g}–{hi:g}°)', fontsize=9)

    shifted = res['span_shifted']
    if shifted is not None:
        span = s['PeakBin_PhaseSpan_deg']
        ax_span.hist(shifted, bins=np.arange(0, 360 + bin_size_deg, bin_size_deg), density=True,
                     color='#D9D9D9', edgecolor='black', linewidth=0.5)
        ax_span.plot(_span_grid(), span_density(shifted), color=PEAK_BIN_SPIKE_COLOR, linewidth=2)
        ax_span.plot(shifted, np.zeros_like(shifted), '|', color='black', markersize=10)
        ax_span.axvline(span, color='0.3', linestyle='--', linewidth=1.2)
        ax_span.set_xlim(0, 360)
        ax_span.set_title(f'Peak-bin phase span: {span:.0f}° (n={len(shifted)} spikes), starting at '
                          f'{s["PeakBin_PhaseSpan_start_deg"]:.0f}°\nKDE SD {SPAN_KDE_SIGMA_DEG:g}°', fontsize=9)
    else:
        ax_span.text(0.5, 0.5, f'{s["PeakBin_n_spikes"]} peak-bin spikes\n'
                     f'(< {MIN_PEAK_BIN_SPIKES_FOR_SPAN}: no phase span)', ha='center', va='center',
                     transform=ax_span.transAxes, fontsize=10)
        ax_span.set_title('Peak-bin phase span', fontsize=9)
    ax_span.set_xticks(np.arange(0, 361, 90))
    ax_span.set_xlabel('Theta phase from start of span (deg)')
    ax_span.set_ylabel('Density (1/deg)')

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Session discovery, animal / arena / session type
# ============================================================================

def find_session_folders(root: Path) -> list[Path]:
    """Recursively find folders directly containing both .ncs and .ntt files."""
    ncs_parents = {p.parent for p in root.rglob('*.ncs')}
    return sorted(folder for folder in ncs_parents if any(folder.glob('*.ntt')))


def _extract_channel_number(path: Path):
    """Numeric channel/tetrode id embedded in a filename, e.g. 'TT3.ntt' -> 3,
    'CSC12.ncs' -> 12. Returns None if the stem has no digits."""
    m = re.search(r'(\d+)', path.stem)
    return int(m.group(1)) if m else None


def match_ncs_to_ntt(ntt_path: Path, ncs_files: list[Path]) -> Path:
    """Find the .ncs file whose embedded number matches the .ntt file's
    (e.g. TT1.ntt -> CSC1.ncs), so each tetrode's spikes are referenced to
    their own LFP channel instead of a single session-wide channel."""
    tt_num = _extract_channel_number(ntt_path)
    if tt_num is not None:
        for ncs_path in ncs_files:
            if _extract_channel_number(ncs_path) == tt_num:
                return ncs_path
    raise FileNotFoundError(
        f'No .ncs file matching {ntt_path.name} (tetrode number {tt_num}) found among: '
        f'{[p.name for p in ncs_files]}')


def detect_animal_id(folder_path: Path) -> str:
    """Animal ID: the first path component under ROOT_FOLDER, per the layout
    ROOT_FOLDER/<animal>/<arena>/DayN/<session>. Falls back to the folder four
    levels up if folder_path isn't under ROOT_FOLDER."""
    try:
        return folder_path.relative_to(ROOT_FOLDER).parts[0]
    except (ValueError, IndexError):
        return folder_path.parts[-4] if len(folder_path.parts) >= 4 else 'UnknownAnimal'


def detect_arena(folder_path: Path) -> str | None:
    """Arena (Circle/Linear/Open) read off a case-insensitive match to
    ARENA_LABELS among folder_path's components, per the data layout
    ROOT_FOLDER/<animal>/<arena>/DayN/<session>. Returns None if no
    component matches (e.g. a differently organized dataset)."""
    parts_lower = {part.lower() for part in folder_path.parts}
    for label in ARENA_LABELS:
        if label.lower() in parts_lower:
            return label
    return None


def detect_session_type(folder_path: Path, arena: str | None) -> str | None:
    """Session type named in the session folder (folder_path's last component),
    per SessionType_StatsComparison_v6.py: Open 'Cntrl'/'Rotate'/'Zero' anywhere
    in the name (e.g. '1_Cntrl', '2Rotate'); Linear the angle after a final
    underscore (e.g. '3_180' -> '180'); Circle 'Stnd'/'NoRot'/'Rot' (e.g.
    '2NoRot'). Returns None if the arena is unknown or no type is recognized."""
    if arena is None:
        return None
    label = folder_path.name.strip().lower()
    if arena == 'Open':
        for t in ARENA_SESSION_TYPES['Open']:
            if t.lower() in label:
                return t
    elif arena == 'Linear':
        m = re.search(r'_(270|180|90|0)$', label)
        if m:
            return m.group(1)
    elif arena == 'Circle':
        # 'norot' must be tested before 'rot', which it contains.
        if 'norot' in label:
            return 'NoRot'
        if 'stnd' in label:
            return 'Stnd'
        if 'rot' in label:
            return 'Rot'
    return None


# ============================================================================
# Group comparisons: figure style (as v31 / ArenaSessionType_StatsComparison_v18.py)
# ============================================================================

PAL_MAGENTA = '#FA7FFA'
PAL_CYAN = '#7FFAFA'
PAL_BLUE = '#7FB3E5'
PAL_DKBLUE = '#0066CC'
PAL_GRAY = '#7F7F7F'
PAL_BLACK = '#000000'
FIG_FONT = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']

FS_LABEL, FS_TITLE, FS_TICK, FS_LEGEND, FS_STATS = 14, 15, 11, 11, 9
FIG_STATS_LINE_IN = 0.2     # panel height per line of stats text (inches)
FIG_LEGEND_ROW_IN = 0.3     # panel height per row of legend entries (inches)
FIG_DPI = 500
FIG_AXIS_LW = 0.8           # spines, ticks, dashed lines
FIG_BOX_LW = 1.0            # box outlines, whiskers, caps; significance lines
FIG_MEDIAN_LW = 1.5         # box median
FIG_GROUP_W = 1.1           # x-axis width per group (inches)
FIG_AXIS_PAD_W = 1.4        # extra width for the y-axis label/ticks
FIG_STATS_CHARS_PER_IN = 15  # approx. characters of stats text per inch of width

FIG_RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': FIG_FONT,
    'pdf.fonttype': 42,          # keep text editable in Illustrator
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'text.color': PAL_BLACK,
    'axes.labelcolor': PAL_BLACK,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.edgecolor': PAL_BLACK,
    'axes.linewidth': FIG_AXIS_LW,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.color': PAL_BLACK,
    'ytick.color': PAL_BLACK,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': FIG_AXIS_LW,
    'ytick.major.width': FIG_AXIS_LW,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.grid': False,
    'legend.frameon': False,
}

# magenta = circular track, cyan = linear track, blue = open arena
ARENA_COLORS = {
    'Open': {'face': PAL_BLUE, 'edge': PAL_BLACK},
    'Linear': {'face': PAL_CYAN, 'edge': PAL_BLACK},
    'Circle': {'face': PAL_MAGENTA, 'edge': PAL_BLACK},
}
# Per-session-type colors, as in SessionType_StatsComparison_v6.py: shades of the arena's hue.
SESSION_COLORS = {
    # Open: Cntrl light blue, Rotate dark blue, Zero gray
    'Cntrl': {'face': PAL_BLUE, 'edge': PAL_BLACK},
    'Rotate': {'face': PAL_DKBLUE, 'edge': PAL_BLACK},
    'Zero': {'face': '#BFBFBF', 'edge': PAL_BLACK},
    # Linear: 0 darkest cyan -> 270 lightest
    '0': {'face': '#008080', 'edge': PAL_BLACK},
    '90': {'face': '#00B3B3', 'edge': PAL_BLACK},
    '180': {'face': '#33D6D6', 'edge': PAL_BLACK},
    '270': {'face': PAL_CYAN, 'edge': PAL_BLACK},
    # Circle: Stnd lightest magenta, NoRot lighter, Rot dark
    'Stnd': {'face': '#FDC4FD', 'edge': PAL_BLACK},
    'NoRot': {'face': PAL_MAGENTA, 'edge': PAL_BLACK},
    'Rot': {'face': '#C000C0', 'edge': PAL_BLACK},
}
# Per-animal comparison (ANIMAL_DIR): the animals' ANIMAL_COLORS
ANIMAL_GROUP_COLORS = {a: {'face': c, 'edge': PAL_BLACK} for a, c in ANIMAL_COLORS.items()}
GROUP_COLORS = {**ARENA_COLORS, **SESSION_COLORS, **ANIMAL_GROUP_COLORS}
_DEFAULT_COLOR = {'face': '#BFBFBF', 'edge': PAL_BLACK}
# Pooled 'All units' plots: theta-modulated (Rayleigh+ & TMI+) units in viridis lime green, the rest gray
_ALL_UNITS_GROUP = 'All units'
PAL_TM_LIME = to_hex(matplotlib.colormaps['viridis'](0.85))   # '#9bd93c'


def _face(g, tm: bool = False) -> str:
    """Face colour of group g; for theta-modulated units (tm) of the pooled 'All units' group, lime."""
    if tm and g == _ALL_UNITS_GROUP:
        return PAL_TM_LIME
    return GROUP_COLORS.get(g, _DEFAULT_COLOR)['face']

# Grouping column -> names of the omnibus tables' group-count / n-per-group columns
GROUP_COUNT_COLUMNS = {'Arena': ('N_arenas', 'N_per_arena'),
                       'SessionType': ('N_session_types', 'N_per_session_type'),
                       'Animal': ('N_animals', 'N_per_animal'),
                       'All': ('N_groups', 'N_per_group')}
_TEST_STAT_NAMES = {'Mardia-Watson-Wheeler': 'W', 'Watson-Williams': 'F',
                    'Chi-square': 'χ²', 'Pairwise chi-square': 'χ²', 'Pairwise chi-square (Yates)': 'χ²',
                    'Density permutation': 'T', 'Kruskal-Wallis': 'H', 'Mann-Whitney U': 'U'}


def _arena_fig_style(func):
    """Run a plotting function under the FIG_RC style."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with plt.rc_context(FIG_RC):
            return func(*args, **kwargs)
    return wrapper


def _new_fig(figsize, legend_rows=0, stats_lines=None):
    """Figure + main axes. If there is a legend and/or stats text, a second
    borderless axes sits below the plot to hold them (legend on top, stats
    underneath), so neither overlaps the data. The panel is sized to fit."""
    w, h = figsize
    panel_h = legend_rows * FIG_LEGEND_ROW_IN + len(stats_lines or []) * FIG_STATS_LINE_IN
    if panel_h > 0:
        panel_h += 0.2
        fig, (ax, ax_info) = plt.subplots(2, 1, figsize=(w, h + panel_h),
                                          gridspec_kw={'height_ratios': [h, panel_h]})
        ax_info.axis('off')
        return fig, ax, ax_info
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, None


def _bar_fig_width(n_groups: int, per_group: float = FIG_GROUP_W) -> float:
    """Fixed width per group, so boxes / bars keep their proportions whatever the number of groups."""
    return n_groups * per_group + FIG_AXIS_PAD_W


def _wrap_stats(stats_lines, fig_w: float) -> list:
    width = max(40, int(fig_w * FIG_STATS_CHARS_PER_IN))
    return [w for line in (stats_lines or []) for w in (textwrap.wrap(line, width) or [''])]


def _draw_stats(ax_info, stats_lines):
    """Stats text at the bottom of the panel below the plot (under any legend)."""
    if ax_info is None or not stats_lines:
        return
    ax_info.text(0.5, 0.0, '\n'.join(stats_lines), transform=ax_info.transAxes,
                 fontsize=FS_STATS, va='bottom', ha='center', linespacing=1.3)


def _fmt_p(p) -> str:
    if p is None or pd.isna(p):
        return 'n/a'
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def _stars(p) -> str:
    if p is None or pd.isna(p):
        return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < ALPHA else 'n.s.'


def _label(group: str) -> str:
    """Display label for plots: Linear session types are angles."""
    return f'{group}°' if str(group).isdigit() else str(group)


def _test_lines(omnibus_df: pd.DataFrame, pairwise_df: pd.DataFrame, group_col: str) -> list:
    """Omnibus test(s) (statistic, dof, p) and post-hoc pairwise tests
    (statistic, raw p, Holm-corrected p) as text lines. Pairs are read from the
    pairwise table's '<group_col>1' / '<group_col>2' columns."""
    lines = []
    for _, r in omnibus_df.iterrows():
        if not np.isfinite(r['Statistic']):
            lines.append(f'{r["Test"]}: insufficient data')
            continue
        dof = r.get('dof')
        dof_str = f', dof = {int(dof)}' if dof is not None and pd.notna(dof) else ''
        p = r['p_value']
        lines.append(f'{r["Test"]}: {_TEST_STAT_NAMES.get(r["Test"], "stat")} = {r["Statistic"]:.3f}'
                     f'{dof_str}, p = {_fmt_p(p)} {_stars(p)}')
        note = r.get('Note')
        if isinstance(note, str) and note:
            lines.append(note)
    for test in pairwise_df['Test'].unique() if len(pairwise_df) else []:
        stat_name = _TEST_STAT_NAMES.get(test, 'stat')
        lines.append(f'Post-hoc ({test}, Holm-Bonferroni corrected):')
        for _, r in pairwise_df[pairwise_df['Test'] == test].iterrows():
            stat = r['Statistic']
            stat_str = 'n/a' if not np.isfinite(stat) else f'{stat:.3f}'
            lines.append(f'{_label(r[f"{group_col}1"])} vs {_label(r[f"{group_col}2"])}: '
                         f'{stat_name} = {stat_str}, '
                         f'p = {_fmt_p(r["p_value"])}, p(Holm) = {_fmt_p(r["p_holm"])} {_stars(r["p_holm"])}')
    return lines


def _set_title(ax, text):
    # ~7 bold title characters per inch of figure width
    ax.set_title(textwrap.fill(text, max(20, int(ax.figure.get_figwidth() * 7))),
                 fontsize=FS_TITLE - 2, pad=12)


def _style_axes(ax):
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.spines[['top', 'right']].set_visible(False)


def _group_boxplot(ax, data, positions, groups, widths=0.55, hatches=None, tm_flags=None):
    """Group-colored boxes with black outlines, black median, no fliers (optional hatch per box;
    tm_flags: per box, True = theta-modulated units, see _face)."""
    bp = ax.boxplot(data, positions=positions, showfliers=False, patch_artist=True, widths=widths,
                    manage_ticks=False,
                    boxprops=dict(linewidth=FIG_BOX_LW, color=PAL_BLACK),
                    whiskerprops=dict(linewidth=FIG_BOX_LW, color=PAL_BLACK),
                    capprops=dict(linewidth=FIG_BOX_LW, color=PAL_BLACK),
                    medianprops=dict(linewidth=FIG_MEDIAN_LW, color=PAL_BLACK),
                    zorder=2)
    for patch, g, hatch, tm in zip(bp['boxes'], groups, hatches or [None] * len(groups),
                                   tm_flags or [False] * len(groups)):
        patch.set_facecolor(_face(g, tm))
        patch.set_edgecolor(PAL_BLACK)
        patch.set_hatch(hatch)
    return bp


def _draw_sig_brackets(ax, pos_by_group: dict, pairwise_df, y_base: float, y_step: float, y_text: float,
                       group_col: str):
    """For every pairwise comparison with Holm-corrected p < ALPHA, draw a
    horizontal line above the two groups (read from the '<group_col>1' /
    '<group_col>2' columns) with the stars (* p<0.05, ** p<0.01, *** p<0.001)
    centered on it; narrower spans are stacked lowest. Returns the y of the
    highest line drawn, or None if no comparison was significant."""
    if pairwise_df is None or not len(pairwise_df):
        return None
    sig = []
    for _, r in pairwise_df.iterrows():
        p = r.get('p_holm')
        if p is None or pd.isna(p) or p >= ALPHA:
            continue
        g1, g2 = r[f'{group_col}1'], r[f'{group_col}2']
        if g1 not in pos_by_group or g2 not in pos_by_group:
            continue
        x1, x2 = sorted((pos_by_group[g1], pos_by_group[g2]))
        sig.append((x2 - x1, x1, x2, _stars(p)))
    y = y_base
    for _span, x1, x2, stars_str in sorted(sig):
        ax.plot([x1, x2], [y, y], color=PAL_BLACK, linewidth=FIG_BOX_LW,
                solid_capstyle='butt', clip_on=False, zorder=5)
        ax.text((x1 + x2) / 2, y + y_text, stars_str, ha='center', va='center',
                fontsize=FS_LABEL + 3, fontweight='bold', clip_on=False, zorder=5)
        y += y_step
    return y - y_step if sig else None


def _save_fig(fig, out_path):
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Theta-modulated proportions per group
# ============================================================================

def theta_mod_counts(df: pd.DataFrame, group_col: str, group_order) -> pd.DataFrame:
    """Per group present in `group_order`: number of units, units with enough spikes for the
    tests (n_spikes_theta >= MIN_SPIKES_FOR_TMI), and number / % of ALL units passing each
    THETA_MOD_CRITERIA criterion (units with too few spikes count as negative), plus the
    exclusive split (Rayleigh+ only, TMI+ only, neither) and the % of TESTED units that are
    Rayleigh+ & TMI+ (pct_of_tested_<col>; NaN for a group with no tested unit)."""
    test_col = THETA_MOD_CRITERIA[-1][0]
    rows = []
    for g in group_order:
        sub = df[df[group_col] == g]
        if sub.empty:
            continue
        n = len(sub)
        row = {group_col: g, 'n_units': n,
               'n_units_tested': int((sub['n_spikes_theta'] >= MIN_SPIKES_FOR_TMI).sum())}
        for col, _name, _hatch in THETA_MOD_CRITERIA:
            k = int((sub[col] == True).sum())                              # noqa: E712
            row[f'n_{col}'], row[f'pct_{col}'] = k, 100.0 * k / n
        ray, tmi = (sub['Rayleigh_Significant'] == True), (sub['TMI_Significant'] == True)  # noqa: E712
        row.update(n_Rayleigh_only=int((ray & ~tmi).sum()), n_TMI_only=int((tmi & ~ray).sum()),
                   n_neither=int((~ray & ~tmi).sum()))
        n_tested = row['n_units_tested']
        row[f'pct_of_tested_{test_col}'] = 100.0 * row[f'n_{test_col}'] / n_tested if n_tested else np.nan
        rows.append(row)
    columns = [group_col, 'n_units', 'n_units_tested',
               *(f'{p}_{col}' for col, *_ in THETA_MOD_CRITERIA for p in ('n', 'pct')),
               'n_Rayleigh_only', 'n_TMI_only', 'n_neither', f'pct_of_tested_{test_col}']
    return pd.DataFrame(rows, columns=columns)


def compare_proportion(counts_df: pd.DataFrame, col: str, metric: str, group_col: str,
                       total_col: str = 'n_units'):
    """Chi-square test of independence (yes / no x group) on the n_<col> of counts_df's groups
    out of their `total_col` units, then pairwise chi-square tests (Yates-corrected for 2 x 2
    tables), Holm-Bonferroni corrected, as _chi_square_tests of ArenaSessionType_StatsComparison_v18.py.
    Units are treated as independent; the omnibus 'Note' flags expected counts < 5. Needs >= 2
    groups (each with total_col > 0) and both outcomes present. Returns (omnibus_df, pairwise_df)."""
    n_col, n_per_col = GROUP_COUNT_COLUMNS[group_col]
    om_cols = ['Metric', 'Test', n_col, 'Statistic', 'dof', 'p_value', 'Note', n_per_col]
    pw_cols = ['Metric', f'{group_col}1', f'{group_col}2', 'N1', 'N2', 'Test', 'Statistic', 'dof',
               'p_value', 'p_holm']
    if len(counts_df) < 2:
        return pd.DataFrame(columns=om_cols), pd.DataFrame(columns=pw_cols)

    groups = counts_df[group_col].tolist()
    yes = counts_df[f'n_{col}'].to_numpy(int)
    table = np.column_stack([yes, counts_df[total_col].to_numpy(int) - yes]).astype(float)
    per_group = ', '.join(f'{g}={int(n)}' for g, n in zip(groups, table.sum(axis=1)))
    if (table.sum(axis=0) > 0).sum() < 2:
        omni = dict(Statistic=np.nan, dof=np.nan, p_value=np.nan, Note='only one outcome present: no test')
        return (pd.DataFrame([{'Metric': metric, 'Test': 'Chi-square', n_col: len(groups), **omni,
                               n_per_col: per_group}], columns=om_cols), pd.DataFrame(columns=pw_cols))

    chi2, p, dof, expected = stats.chi2_contingency(table)
    n_low = int((expected < 5).sum())
    note = (f'{n_low} of {expected.size} expected counts < 5: chi-square approximation may be unreliable'
            if n_low else '')
    omnibus_df = pd.DataFrame([{'Metric': metric, 'Test': 'Chi-square', n_col: len(groups), 'Statistic': chi2,
                                'dof': dof, 'p_value': p, 'Note': note, n_per_col: per_group}], columns=om_cols)
    rows = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            sub = table[[i, j]]
            sub = sub[:, sub.sum(axis=0) > 0]
            c2, pp, d = (stats.chi2_contingency(sub)[:3] if sub.shape[1] == 2 else (np.nan, np.nan, np.nan))
            rows.append({'Metric': metric, f'{group_col}1': groups[i], f'{group_col}2': groups[j],
                         'N1': int(table[i].sum()), 'N2': int(table[j].sum()),
                         'Test': 'Pairwise chi-square (Yates)', 'Statistic': c2, 'dof': d, 'p_value': pp})
    for row, p_adj in zip(rows, holm_adjust([r['p_value'] for r in rows])):
        row['p_holm'] = p_adj
    return omnibus_df, pd.DataFrame(rows, columns=pw_cols)


@_arena_fig_style
def plot_theta_mod_proportion(counts_df: pd.DataFrame, omnibus_df: pd.DataFrame, pairwise_df: pd.DataFrame,
                              out_path: Path, group_col: str, title_prefix: str = '', pooled: str = ''):
    """% of all units that are Rayleigh+, TMI+ and Rayleigh+ & TMI+ per group, three bars side by
    side in the group colour (hatch = criterion), '%' and 'n units' above each bar; stars over pairs of
    Rayleigh+ & TMI+ bars whose Holm-corrected pairwise chi-square p < ALPHA. Counts and tests in a
    panel underneath."""
    groups = counts_df[group_col].tolist()
    n_all = counts_df['n_units'].to_numpy()
    positions = np.arange(1, len(groups) + 1)
    k = len(THETA_MOD_CRITERIA)
    width = 0.8 / k
    offsets = (np.arange(k) - (k - 1) / 2) * width

    test_col, test_name = THETA_MOD_CRITERIA[-1][0], THETA_MOD_CRITERIA[-1][1]
    stats_lines = [f'% of all units; one value per unit; {pooled}. Units with < {MIN_SPIKES_FOR_TMI} '
                   f'theta-epoch spikes count as negative. Bars overlap: a {test_name} unit is also '
                   f'counted as Rayleigh+ and TMI+.']
    stats_lines += [f'{_label(r[group_col])} (n={r["n_units"]} units, {r["n_units_tested"]} tested): '
                    + ' | '.join(f'{name} {r[f"n_{col}"]} ({r[f"pct_{col}"]:.1f} %)'
                                 for col, name, _ in THETA_MOD_CRITERIA)
                    for _, r in counts_df.iterrows()]
    if len(omnibus_df):
        stats_lines.append(f'Tests on {test_name} only (units treated as independent):')
    stats_lines += _test_lines(omnibus_df, pairwise_df, group_col)
    fig_w = _bar_fig_width(len(groups), per_group=1.8 * FIG_GROUP_W)
    stats_lines = _wrap_stats(stats_lines, fig_w)
    fig, ax, ax_info = _new_fig((fig_w, 5.5), legend_rows=1, stats_lines=stats_lines)

    pct_max = 0.0
    for (col, _name, hatch), off in zip(THETA_MOD_CRITERIA, offsets):
        faces = [_face(g, tm=col == test_col) for g in groups]
        pct = counts_df[f'pct_{col}'].to_numpy(float)
        pct_max = max(pct_max, float(pct.max()))
        ax.bar(positions + off, pct, width=0.92 * width, color=faces, edgecolor=PAL_BLACK, hatch=hatch,
               linewidth=FIG_BOX_LW, zorder=2)
        for x, p_, n_k in zip(positions + off, pct, counts_df[f'n_{col}']):
            ax.text(x, p_ + 1.5, f'{p_:.1f}%\n{n_k}', ha='center', va='bottom', fontsize=FS_TICK - 3)
    top = _draw_sig_brackets(ax, dict(zip(groups, positions + offsets[-1])), pairwise_df,
                             y_base=pct_max + 15, y_step=8, y_text=3, group_col=group_col)
    ax.set_ylim(0, max(100.0, (top if top is not None else pct_max + 11) + 8))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{_label(g)}\n(n={n})' for g, n in zip(groups, n_all)])
    ax.set_xlim(0.4, len(groups) + 0.6)
    ax.set_ylabel('Units (% of all units)', fontsize=FS_LABEL, labelpad=8)
    _set_title(ax, f'{title_prefix}Theta-modulated units')
    _style_axes(ax)

    tm_legend_face = PAL_TM_LIME if groups == [_ALL_UNITS_GROUP] else '#BFBFBF'
    ax_info.legend(handles=[Patch(facecolor='white' if hatch else tm_legend_face, edgecolor=PAL_BLACK, hatch=hatch,
                                  label=name) for _col, name, hatch in THETA_MOD_CRITERIA],
                   loc='upper center', ncol=k, fontsize=FS_LEGEND - 2)
    _draw_stats(ax_info, stats_lines)
    fig.tight_layout()
    _save_fig(fig, out_path)


@_arena_fig_style
def plot_theta_mod_tested_proportion(counts_df: pd.DataFrame, omnibus_df: pd.DataFrame, pairwise_df: pd.DataFrame,
                                     out_path: Path, group_col: str, title_prefix: str = '', pooled: str = ''):
    """% of TESTED units (>= MIN_SPIKES_FOR_TMI theta-epoch spikes) that are Rayleigh+ & TMI+ per
    group, one bar per group in the group colour, '%' and 'k/n' above each bar; stars over pairs whose
    Holm-corrected pairwise chi-square p < ALPHA. Counts and tests in a panel underneath.
    `counts_df` holds only groups with >= 1 tested unit."""
    test_col, test_name = THETA_MOD_CRITERIA[-1][0], THETA_MOD_CRITERIA[-1][1]
    groups = counts_df[group_col].tolist()
    n_tested = counts_df['n_units_tested'].to_numpy()
    n_yes = counts_df[f'n_{test_col}'].to_numpy()
    pct = counts_df[f'pct_of_tested_{test_col}'].to_numpy(float)
    positions = np.arange(1, len(groups) + 1)

    stats_lines = [f'% of tested units (>= {MIN_SPIKES_FOR_TMI} theta-epoch spikes); units with fewer spikes are '
                   f'left out; one value per unit; {pooled}.']
    stats_lines += [f'{_label(g)}: {test_name} {k}/{n} tested units ({p:.1f} %; {n_all} units in total)'
                    for g, k, n, p, n_all in zip(groups, n_yes, n_tested, pct, counts_df['n_units'])]
    if len(omnibus_df):
        stats_lines.append('Tests (units treated as independent):')
    stats_lines += _test_lines(omnibus_df, pairwise_df, group_col)
    fig_w = _bar_fig_width(len(groups))
    stats_lines = _wrap_stats(stats_lines, fig_w)
    fig, ax, ax_info = _new_fig((fig_w, 5.5), stats_lines=stats_lines)

    ax.bar(positions, pct, width=0.55, color=[_face(g, tm=True) for g in groups],
           edgecolor=PAL_BLACK, linewidth=FIG_BOX_LW, zorder=2)
    for x, p_, k, n in zip(positions, pct, n_yes, n_tested):
        ax.text(x, p_ + 1.5, f'{p_:.1f}%\n{k}/{n}', ha='center', va='bottom', fontsize=FS_TICK - 3)
    pct_max = float(pct.max())
    top = _draw_sig_brackets(ax, dict(zip(groups, positions)), pairwise_df,
                             y_base=pct_max + 15, y_step=8, y_text=3, group_col=group_col)
    ax.set_ylim(0, max(100.0, (top if top is not None else pct_max + 11) + 8))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{_label(g)}\n(n={n})' for g, n in zip(groups, n_tested)])
    ax.set_xlim(0.4, len(groups) + 0.6)
    ax.set_ylabel('Units (% of tested units)', fontsize=FS_LABEL, labelpad=8)
    _set_title(ax, f'{title_prefix}Theta-modulated units ({test_name}) of tested units')
    _style_axes(ax)

    _draw_stats(ax_info, stats_lines)
    fig.tight_layout()
    _save_fig(fig, out_path)


def run_theta_mod_proportion(df: pd.DataFrame, group_col: str, group_order, out_path: Path,
                             pooled: str, title_prefix: str = '') -> dict[str, pd.DataFrame]:
    """Counts of every THETA_MOD_CRITERIA criterion per group, the tests on the last one
    (compare_proportion) and the plot (plot_theta_mod_proportion, saved to out_path; plotted even
    for a single group). Then the same tests on the % of TESTED units that are Rayleigh+ & TMI+
    (groups with no tested unit left out) and its plot (plot_theta_mod_tested_proportion, saved to
    <out_path stem>_Tested.png). Returns {table key: DataFrame}; empty if no group holds a unit."""
    counts_df = theta_mod_counts(df, group_col, group_order)
    if counts_df.empty:
        return {}
    test_col, test_name = THETA_MOD_CRITERIA[-1][0], THETA_MOD_CRITERIA[-1][1]
    omnibus_df, pairwise_df = compare_proportion(counts_df, test_col, f'{test_name} units (of all units)',
                                                 group_col)
    plot_theta_mod_proportion(counts_df, omnibus_df, pairwise_df, out_path, group_col,
                              title_prefix=title_prefix, pooled=pooled)
    print(f'{test_name} units: ' + '; '.join(
        f'{_label(r[group_col])} {r[f"n_{test_col}"]}/{r["n_units"]} ({r[f"pct_{test_col}"]:.1f} %)'
        for _, r in counts_df.iterrows()) + f' -- plot saved: {out_path.name}')
    tables = {'ThetaMod_Counts': counts_df, 'ThetaMod_Omni': omnibus_df, 'ThetaMod_Pair': pairwise_df}

    tested_df = counts_df[counts_df['n_units_tested'] > 0].reset_index(drop=True)
    if tested_df.empty:
        print(f'No tested units (>= {MIN_SPIKES_FOR_TMI} theta-epoch spikes): % of tested units skipped.')
        return tables
    omni_t, pair_t = compare_proportion(tested_df, test_col, f'{test_name} units (of tested units)', group_col,
                                        total_col='n_units_tested')
    tested_path = out_path.with_name(f'{out_path.stem}_Tested.png')
    plot_theta_mod_tested_proportion(tested_df, omni_t, pair_t, tested_path, group_col,
                                     title_prefix=title_prefix, pooled=pooled)
    print(f'{test_name} units of tested units: ' + '; '.join(
        f'{_label(r[group_col])} {r[f"n_{test_col}"]}/{r["n_units_tested"]} '
        f'({r[f"pct_of_tested_{test_col}"]:.1f} %)' for _, r in tested_df.iterrows())
        + f' -- plot saved: {tested_path.name}')
    tables.update(TMTested_Omni=omni_t, TMTested_Pair=pair_t)
    return tables


# ============================================================================
# Peak-bin / onset / offset phase per group (circular tests)
# ============================================================================

def compare_groups_phase(df_pt: pd.DataFrame, rng, col: str, label: str, group_col: str, group_order):
    """Theta phase in `col` (deg, one value per unit; described by `label`) across the groups
    of `group_order` in df_pt[group_col], with circular statistics (a linear Kruskal-Wallis would
    treat 359 and 1 deg as far apart). Primary: Mardia-Watson-Wheeler uniform-scores test
    (nonparametric, any difference in the circular distributions), permutation p. Also reported:
    Watson-Williams test (equal mean direction, von Mises assumption). Omnibus over all groups,
    then pairwise with Holm correction per test. Returns (descriptive_df, omnibus_df, pairwise_df)."""
    present = [g for g in group_order if (df_pt[group_col] == g).sum() > 0]
    groups = {g: np.deg2rad(df_pt.loc[df_pt[group_col] == g, col].dropna().to_numpy(dtype=float))
              for g in present}
    groups = {g: v for g, v in groups.items() if len(v) > 0}
    n_col, n_per_col = GROUP_COUNT_COLUMNS[group_col]

    desc_rows = []
    for g, v in groups.items():
        n_trough = int(sum(phase_range_label(x) == 'trough' for x in np.rad2deg(v)))
        rayleigh_p, _ = circ_rtest(v)
        desc_rows.append({
            group_col: g, 'N': len(v), 'CircMean_deg': float(np.mod(np.rad2deg(circ_mean(v)), 360)),
            'MRL': circ_r(v), 'Rayleigh_p': rayleigh_p,
            'N_trough_range': n_trough, 'N_peak_range': len(v) - n_trough,
            'Pct_trough_range': 100.0 * n_trough / len(v)})
    desc_df = pd.DataFrame(desc_rows, columns=[group_col, 'N', 'CircMean_deg', 'MRL', 'Rayleigh_p',
                                               'N_trough_range', 'N_peak_range', 'Pct_trough_range'])

    def run_tests(group_list):
        W, p_chi2, p_perm = mardia_watson_wheeler_test(group_list, rng)
        F, p_ww, kappa = watson_williams_test(group_list)
        return [dict(Test='Mardia-Watson-Wheeler', Statistic=W, p_value=p_perm, p_chi2_asymptotic=p_chi2),
                dict(Test='Watson-Williams', Statistic=F, p_value=p_ww, kappa=kappa)]

    omnibus_rows = []
    if len(groups) >= 2:
        for res in run_tests(list(groups.values())):
            omnibus_rows.append({'Metric': f'Theta phase at {label} (deg)', n_col: len(groups),
                                 n_per_col: ', '.join(f'{g}={len(v)}' for g, v in groups.items()),
                                 **res})
    omnibus_df = pd.DataFrame(omnibus_rows, columns=['Metric', 'Test', n_col, 'Statistic', 'p_value',
                                                     'p_chi2_asymptotic', 'kappa', n_per_col])

    pairwise_rows = []
    names = list(groups.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            g1, g2 = names[i], names[j]
            for res in run_tests([groups[g1], groups[g2]]):
                pairwise_rows.append({'Metric': f'Theta phase at {label} (deg)', f'{group_col}1': g1,
                                      f'{group_col}2': g2, 'N1': len(groups[g1]), 'N2': len(groups[g2]), **res})
    pairwise_df = pd.DataFrame(pairwise_rows, columns=['Metric', f'{group_col}1', f'{group_col}2', 'N1', 'N2',
                                                       'Test', 'Statistic', 'p_value', 'p_holm',
                                                       'p_chi2_asymptotic', 'kappa'])
    for test in pairwise_df['Test'].unique():
        sel = pairwise_df['Test'] == test
        pairwise_df.loc[sel, 'p_holm'] = holm_adjust(pairwise_df.loc[sel, 'p_value'].to_numpy())
    desc_df.insert(0, 'Metric', f'Theta phase at {label} (deg)')
    return desc_df, omnibus_df, pairwise_df


def _draw_phase_boxes(ax, positions, phases, color_groups, hatches=None, box_w=0.55, jitter_w=0.12,
                      tm_flags=None):
    """Circular box plots + jittered points of `phases` (deg, one array per box) at `positions`, on
    a single-cycle 0-360 deg axis with the theta trough / peak range bands. Box colour = the group's
    (color_groups), optional hatch per box. Each box is computed on its phases unwrapped to within
    +/-180 deg of their circular mean (so a cluster straddling 0/360 deg is not split), then drawn at
    -360/0/+360 deg offsets clipped to 0-360 so such a box wraps around. Points are purple in the
    theta trough range, light yellow in the theta peak range."""
    lo, hi = THETA_TROUGH_RANGE_DEG
    ax.axhspan(lo, hi, color=PHASE_RANGE_COLORS['trough'], alpha=0.08, lw=0, zorder=0)
    for y0, y1 in ((0, lo), (hi, 360)):
        ax.axhspan(y0, y1, color=PHASE_RANGE_COLORS['peak'], alpha=0.12, lw=0, zorder=0)

    clip_rect = Rectangle((0, 0), max(positions) + 1, 360, transform=ax.transData)
    hatches = hatches or [None] * len(phases)
    tm_flags = tm_flags or [False] * len(phases)
    for pos, g, hatch, tm, ph in zip(positions, color_groups, hatches, tm_flags, phases):
        if not len(ph):
            continue
        mu = np.mod(np.rad2deg(circ_mean(np.deg2rad(ph))), 360)
        unwrapped = mu + (np.mod(ph - mu + 180, 360) - 180)
        for off in (-360, 0, 360):
            bp = _group_boxplot(ax, [unwrapped + off], [pos], [g], widths=box_w, hatches=[hatch], tm_flags=[tm])
            for artist in (*bp['boxes'], *bp['whiskers'], *bp['caps'], *bp['medians']):
                artist.set_clip_path(clip_rect)

    rng = np.random.default_rng(0)
    for pos, ph in zip(positions, phases):
        jitter = rng.uniform(-jitter_w, jitter_w, size=len(ph))
        faces = [PHASE_RANGE_COLORS[phase_range_label(p)] for p in ph]
        ax.scatter(pos + jitter, ph, s=22, facecolors=faces, edgecolors=PAL_BLACK,
                   linewidths=0.5, zorder=3)
    ax.set_ylim(0, 360)
    ax.set_yticks(np.arange(0, 361, 90))


def _phase_range_handles():
    """Legend entries for the theta trough / peak range point colours."""
    lo, hi = THETA_TROUGH_RANGE_DEG
    return [Line2D([0], [0], marker='o', linestyle='None', markersize=8,
                   markerfacecolor=PHASE_RANGE_COLORS['trough'], markeredgecolor=PAL_BLACK,
                   markeredgewidth=0.5, label=f'theta trough range ({lo:g}–{hi:g}°)'),
            Line2D([0], [0], marker='o', linestyle='None', markersize=8,
                   markerfacecolor=PHASE_RANGE_COLORS['peak'], markeredgecolor=PAL_BLACK,
                   markeredgewidth=0.5, label=f'theta peak range (0–{lo:g}°, {hi:g}–360°)')]


@_arena_fig_style
def plot_phase_by_group(df_pt: pd.DataFrame, desc_df: pd.DataFrame, omnibus_df: pd.DataFrame,
                        pairwise_df: pd.DataFrame, out_path: Path, col: str, label: str, group_col: str,
                        group_order, title_prefix: str = '', population: str = ''):
    """Box plot + jittered points per group of the theta phase in `col`, on a single-cycle 0-360
    deg axis (v31 plot_fi0_phase_by_arena). Points are purple in the theta trough range, light
    yellow in the theta peak range (faint matching background bands). Because phase is circular,
    each group's box is computed on its phases unwrapped to within +/-180 deg of the group's
    circular mean (so a cluster straddling 0/360 deg is not split), then drawn at -360/0/+360 deg
    offsets clipped to 0-360 so such a box wraps around. Stars: pairs whose Holm-corrected
    Mardia-Watson-Wheeler p < ALPHA."""
    present = [g for g in group_order if df_pt.loc[df_pt[group_col] == g, col].notna().any()]
    if not present:
        return
    phases = [np.mod(df_pt.loc[df_pt[group_col] == g, col].dropna().to_numpy(dtype=float), 360)
              for g in present]
    positions = np.arange(1, len(present) + 1)

    stats_lines = [population] if population else []
    for _, r in desc_df.iterrows():
        stats_lines.append(f'{_label(r[group_col])} (n={int(r["N"])}): circular mean {r["CircMean_deg"]:.1f}°, '
                           f'MRL {r["MRL"]:.2f} | trough range {int(r["N_trough_range"])}/'
                           f'{int(r["N"])} ({r["Pct_trough_range"]:.1f} %)')
    stats_lines += _test_lines(omnibus_df, pairwise_df, group_col)
    stats_lines.append('Mardia-Watson-Wheeler p from label permutations; '
                       'Watson-Williams assumes von Mises phases with a common concentration')
    fig_w = _bar_fig_width(len(present))
    stats_lines = _wrap_stats(stats_lines, fig_w)
    fig, ax, ax_info = _new_fig((fig_w, 7.0), legend_rows=2, stats_lines=stats_lines)

    _draw_phase_boxes(ax, positions, phases, present)
    top = _draw_sig_brackets(ax, dict(zip(present, positions)),
                             pairwise_df[pairwise_df['Test'] == 'Mardia-Watson-Wheeler'],
                             y_base=360 + 0.05 * 360, y_step=0.08 * 360, y_text=0.025 * 360,
                             group_col=group_col)
    if top is not None:
        ax.set_ylim(0, top + 0.06 * 360)
    ax_info.legend(handles=_phase_range_handles(), loc='upper center', fontsize=FS_LEGEND - 2, ncol=1)
    _draw_stats(ax_info, stats_lines)

    ax.set_xticks(positions)
    ax.set_xticklabels([f'{_label(g)}\n(n={len(ph)})' for g, ph in zip(present, phases)])
    ax.set_xlim(0.4, len(present) + 0.6)
    ax.set_ylabel(textwrap.fill(f'Theta phase at {label} (deg)', 30), fontsize=FS_LABEL, labelpad=8)
    _set_title(ax, f'{title_prefix}Theta phase at {label}')
    _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, out_path)


# ============================================================================
# Theta-modulated vs non-theta-modulated units, within each group (Watson-Williams)
# ============================================================================

def _non_tm_mask(df: pd.DataFrame) -> pd.Series:
    """Non-theta-modulated units of df (ThetaModulated not True); without the units too sparse for
    the Rayleigh / TMI tests unless NON_TM_INCLUDES_UNTESTED."""
    non = df['ThetaModulated'] != True                                     # noqa: E712
    if not NON_TM_INCLUDES_UNTESTED:
        non &= df['n_spikes_theta'] >= MIN_SPIKES_FOR_TMI
    return non


def compare_tm_vs_non(df: pd.DataFrame, col: str, label: str, group_col: str, group_order) -> pd.DataFrame:
    """Within each group of `group_order`: the phase in `col` (deg, one value per unit) of
    theta-modulated vs non-theta-modulated units, one two-sample Watson-Williams test (circular
    analogue of the t-test: equal mean direction, von Mises phases with a common concentration).
    p_holm: Holm-Bonferroni over the groups of this comparison. One row per group with a value."""
    rows = []
    for g in group_order:
        sub = df[(df[group_col] == g) & df[col].notna()]
        tm = np.deg2rad(sub.loc[sub['ThetaModulated'] == True, col].to_numpy(float))   # noqa: E712
        non = np.deg2rad(sub.loc[_non_tm_mask(sub), col].to_numpy(float))
        if not len(tm) and not len(non):
            continue
        F, p, kappa = watson_williams_test([tm, non]) if len(tm) and len(non) else (np.nan, np.nan, np.nan)
        rows.append({'Metric': f'Theta phase at {label} (deg)', group_col: g,
                     'N_TM': len(tm), 'N_nonTM': len(non),
                     'CircMean_TM_deg': float(np.mod(np.rad2deg(circ_mean(tm)), 360)) if len(tm) else np.nan,
                     'MRL_TM': circ_r(tm) if len(tm) else np.nan,
                     'CircMean_nonTM_deg': float(np.mod(np.rad2deg(circ_mean(non)), 360)) if len(non) else np.nan,
                     'MRL_nonTM': circ_r(non) if len(non) else np.nan,
                     'Test': 'Watson-Williams', 'Statistic': F,
                     'dof': f'1, {len(tm) + len(non) - 2}' if np.isfinite(F) else '', 'p_value': p, 'kappa': kappa})
    out = pd.DataFrame(rows, columns=['Metric', group_col, 'N_TM', 'N_nonTM', 'CircMean_TM_deg', 'MRL_TM',
                                      'CircMean_nonTM_deg', 'MRL_nonTM', 'Test', 'Statistic', 'dof', 'p_value',
                                      'p_holm', 'kappa'])
    out['p_holm'] = holm_adjust(out['p_value'].to_numpy(float))
    return out


@_arena_fig_style
def plot_tm_vs_non(df: pd.DataFrame, table: pd.DataFrame, out_path: Path, col: str, label: str, group_col: str,
                   title_prefix: str = '', pooled: str = ''):
    """Per group: circular box plots of the phase in `col` of theta-modulated (plain box) and
    non-theta-modulated (hatched box) units side by side, in the group colour (_draw_phase_boxes);
    star over a group's pair if its Holm-corrected Watson-Williams p < ALPHA."""
    groups = table[group_col].tolist()
    centres = np.arange(1, len(groups) + 1)
    positions, phases, color_groups, hatches, tm_flags = [], [], [], [], []
    for c, g in zip(centres, groups):
        sub = df[(df[group_col] == g) & df[col].notna()]
        for dx, sel, hatch, tm in ((-0.2, sub['ThetaModulated'] == True, None, True),   # noqa: E712
                                   (0.2, _non_tm_mask(sub), '///', False)):
            positions.append(c + dx)
            phases.append(np.mod(sub.loc[sel, col].to_numpy(float), 360))
            color_groups.append(g)
            hatches.append(hatch)
            tm_flags.append(tm)

    non_label = 'not theta-modulated' + (' (incl. too few spikes to test)' if NON_TM_INCLUDES_UNTESTED else '')
    stats_lines = [f'Theta-modulated (Rayleigh+ & TMI+) vs {non_label} units, one value per unit; {pooled}. '
                   f'Two-sample Watson-Williams test per {"group" if len(groups) > 1 else "comparison"}'
                   + (', Holm-Bonferroni corrected over the groups' if len(groups) > 1 else '')]
    for _, r in table.iterrows():
        test = ('insufficient data' if not np.isfinite(r['Statistic']) else
                f'F = {r["Statistic"]:.3f}, p = {_fmt_p(r["p_value"])}, p(Holm) = {_fmt_p(r["p_holm"])} '
                f'{_stars(r["p_holm"])}')
        stats_lines.append(f'{_label(r[group_col])}: TM n={r["N_TM"]}, mean {r["CircMean_TM_deg"]:.1f}°, '
                           f'MRL {r["MRL_TM"]:.2f} | non-TM n={r["N_nonTM"]}, mean {r["CircMean_nonTM_deg"]:.1f}°, '
                           f'MRL {r["MRL_nonTM"]:.2f} | {test}')
    stats_lines.append('Watson-Williams assumes von Mises phases with a common concentration')
    fig_w = max(_bar_fig_width(len(groups), per_group=1.6 * FIG_GROUP_W), 4.5)
    stats_lines = _wrap_stats(stats_lines, fig_w)
    fig, ax, ax_info = _new_fig((fig_w, 7.0), legend_rows=2, stats_lines=stats_lines)

    _draw_phase_boxes(ax, positions, phases, color_groups, hatches, box_w=0.32, jitter_w=0.07, tm_flags=tm_flags)
    pairs = pd.DataFrame({'Pair1': [f'{g}|TM' for g in table[group_col]],
                          'Pair2': [f'{g}|non' for g in table[group_col]], 'p_holm': table['p_holm']})
    pos_by_key = {f'{g}|{kind}': c + dx for c, g in zip(centres, groups) for kind, dx in (('TM', -0.2), ('non', 0.2))}
    top = _draw_sig_brackets(ax, pos_by_key, pairs, y_base=360 + 0.05 * 360, y_step=0.08 * 360,
                             y_text=0.025 * 360, group_col='Pair')
    if top is not None:
        ax.set_ylim(0, top + 0.06 * 360)
    handles = _phase_range_handles() + [
        Patch(facecolor=PAL_TM_LIME if groups == [_ALL_UNITS_GROUP] else '#BFBFBF', edgecolor=PAL_BLACK,
              label='theta-modulated'),
        Patch(facecolor='#BFBFBF', edgecolor=PAL_BLACK, hatch='///', label='not theta-modulated')]
    ax_info.legend(handles=handles, loc='upper center', fontsize=FS_LEGEND - 2, ncol=2)
    _draw_stats(ax_info, stats_lines)

    ax.set_xticks(centres)
    ax.set_xticklabels([f'{_label(g)}\n(TM {r["N_TM"]} / non {r["N_nonTM"]})'
                        for g, (_, r) in zip(groups, table.iterrows())])
    ax.set_xlim(0.4, len(groups) + 0.6)
    ax.set_ylabel(textwrap.fill(f'Theta phase at {label} (deg)', 30), fontsize=FS_LABEL, labelpad=8)
    _set_title(ax, f'{title_prefix}Theta phase at {label}: theta-modulated vs not')
    _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, out_path)


def run_tm_vs_non(df: pd.DataFrame, group_col: str, group_order, out_dir: Path, file_prefix: str,
                  pooled: str, title_prefix: str = '') -> dict[str, pd.DataFrame]:
    """Theta-modulated vs non-theta-modulated units within each group, for every SPIKE_PHASE_METRICS
    phase (compare_tm_vs_non), plotted to out_dir/<file_prefix>_PhaseAt<metric>_TMvsNonTM.png.
    Returns {'<sheet tag>_TMvsNon': table}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for col, label, file_tag, sheet_tag in SPIKE_PHASE_METRICS:
        table = compare_tm_vs_non(df, col, label, group_col, group_order)
        if table.empty:
            continue
        tables[f'{sheet_tag}_TMvsNon'] = table
        plot_path = out_dir / f'{file_prefix}_PhaseAt{file_tag}_TMvsNonTM.png'
        plot_tm_vs_non(df, table, plot_path, col, label, group_col, title_prefix, pooled)
        print(f'{file_tag} phase, theta-modulated vs not: ' + '; '.join(
            f'{_label(r[group_col])} p(Holm) = {_fmt_p(r["p_holm"])}' for _, r in table.iterrows())
              + f' -- plot saved: {plot_path.name}')
    return tables


# ============================================================================
# Theta-phase span of the peak-bin spikes per group: one density per unit, compared across groups
# ============================================================================

def _parse_span_phases(s) -> np.ndarray:
    """Re-measured peak-bin phases (deg) from the ';'-joined PeakBin_SpanPhases_deg string."""
    return np.array([float(v) for v in str(s).split(';') if v != ''], dtype=float)


def _density_between_stat(D: np.ndarray, labels: np.ndarray, k: int) -> float:
    """T = sum_g n_g * integral (mean density of group g - grand mean density)^2 dphase, over the
    units' density curves D (n_units x n_grid); the functional analogue of a between-group sum of squares."""
    onehot = np.eye(k)[labels]
    counts = onehot.sum(axis=0)
    means = (onehot.T @ D) / counts[:, None]
    return float((counts[:, None] * (means - D.mean(axis=0)) ** 2).sum() * SPAN_GRID_STEP_DEG)


def density_permutation_test(curves: list, rng, n_perm=N_DENSITY_PERMUTATIONS):
    """Do the groups' units have the same mean density curve? `curves`: one (n_units x n_grid)
    array per group, one density per unit. Statistic T (_density_between_stat); p from n_perm
    permutations of the units' group labels (each unit keeps its whole curve). Returns (T, p)."""
    D = np.vstack(curves)
    labels = np.repeat(np.arange(len(curves)), [len(c) for c in curves])
    k = len(curves)
    T = _density_between_stat(D, labels, k)
    perm_T = np.array([_density_between_stat(D, rng.permutation(labels), k) for _ in range(n_perm)])
    return T, float((np.sum(perm_T >= T - 1e-12) + 1) / (n_perm + 1))


def compare_peakbin_span(curves: dict, spans: dict, n_spikes: dict, group_col: str, rng):
    """Peak-bin phase span across the groups of `curves` / `spans` / `n_spikes` ({group: per-unit
    densities (n x n_grid) / spans (deg) / peak-bin spike counts}), one unit = one sample:
      density curves  omnibus + pairwise density_permutation_test ('Density permutation'); pairwise
                      rows also give the L1 distance between the two mean curves (0 = same, 2 = disjoint)
      span (deg)      Kruskal-Wallis omnibus + pairwise two-sided Mann-Whitney U
    Holm-Bonferroni over the pairs, per test. Returns (descriptive_df, omnibus_df, pairwise_df);
    the tests need >= 2 groups."""
    metric = 'Theta-phase span of peak-bin spikes'
    n_col, n_per_col = GROUP_COUNT_COLUMNS[group_col]
    grid = _span_grid()
    desc_rows = []
    for g, D in curves.items():
        sp, ns = spans[g], n_spikes[g]
        mean_curve = D.mean(axis=0)
        desc_rows.append({'Metric': metric, group_col: g, 'N_units': len(sp),
                          'Span_mean_deg': float(np.mean(sp)),
                          'Span_SEM_deg': float(np.std(sp, ddof=1) / np.sqrt(len(sp))) if len(sp) > 1 else np.nan,
                          'Span_median_deg': float(np.median(sp)),
                          'Span_Q1_deg': float(np.percentile(sp, 25)), 'Span_Q3_deg': float(np.percentile(sp, 75)),
                          'PeakBinSpikes_median': float(np.median(ns)),
                          'MeanDensity_mode_deg': float(grid[np.argmax(mean_curve)]),
                          'MeanDensity_peak_per_deg': float(mean_curve.max())})
    desc_df = pd.DataFrame(desc_rows)

    om_cols = ['Metric', 'Test', n_col, 'Statistic', 'dof', 'p_value', 'Note', n_per_col]
    pw_cols = ['Metric', f'{group_col}1', f'{group_col}2', 'N1', 'N2', 'Test', 'Statistic', 'dof', 'p_value',
               'p_holm', 'L1_distance']
    names = list(curves)
    if len(names) < 2:
        return desc_df, pd.DataFrame(columns=om_cols), pd.DataFrame(columns=pw_cols)

    def kruskal(groups_):
        try:
            return stats.kruskal(*groups_)
        except ValueError:                       # e.g. every value identical
            return np.nan, np.nan

    per_group = ', '.join(f'{g}={len(spans[g])}' for g in names)
    T, p_T = density_permutation_test([curves[g] for g in names], rng)
    H, p_H = kruskal([spans[g] for g in names])
    omnibus_df = pd.DataFrame([
        {'Metric': f'{metric}: density', 'Test': 'Density permutation', n_col: len(names), 'Statistic': T,
         'dof': np.nan, 'p_value': p_T, 'Note': '', n_per_col: per_group},
        {'Metric': f'{metric}: span (deg)', 'Test': 'Kruskal-Wallis', n_col: len(names), 'Statistic': H,
         'dof': len(names) - 1, 'p_value': p_H, 'Note': '', n_per_col: per_group}], columns=om_cols)

    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            g1, g2 = names[i], names[j]
            base = {f'{group_col}1': g1, f'{group_col}2': g2, 'N1': len(spans[g1]), 'N2': len(spans[g2])}
            T_p, p_p = density_permutation_test([curves[g1], curves[g2]], rng)
            l1 = float(np.abs(curves[g1].mean(axis=0) - curves[g2].mean(axis=0)).sum() * SPAN_GRID_STEP_DEG)
            rows.append({'Metric': f'{metric}: density', **base, 'Test': 'Density permutation', 'Statistic': T_p,
                         'dof': np.nan, 'p_value': p_p, 'L1_distance': l1})
            U, p_u = stats.mannwhitneyu(spans[g1], spans[g2], alternative='two-sided')
            rows.append({'Metric': f'{metric}: span (deg)', **base, 'Test': 'Mann-Whitney U', 'Statistic': float(U),
                         'dof': np.nan, 'p_value': float(p_u), 'L1_distance': np.nan})
    pairwise_df = pd.DataFrame(rows, columns=pw_cols)
    for test in pairwise_df['Test'].unique():
        sel = pairwise_df['Test'] == test
        pairwise_df.loc[sel, 'p_holm'] = holm_adjust(pairwise_df.loc[sel, 'p_value'].to_numpy(float))
    return desc_df, omnibus_df, pairwise_df


_SPAN_LINESTYLES = ('-', '--', '-.', ':')


@_arena_fig_style
def plot_peakbin_span(curves: dict, spans: dict, n_spikes: dict, desc_df: pd.DataFrame, omnibus_df: pd.DataFrame,
                      pairwise_df: pd.DataFrame, out_path: Path, group_col: str, title_prefix: str = '',
                      population: str = '', tm: bool = False):
    """Left: each unit's density of its peak-bin phases re-measured from the start of its span (thin,
    faint) and the group's mean density +/- SEM (thick, group colour, black outline; line style per
    group). Right: box plot + points of the units' spans (white diamond = mean); stars over pairs whose
    Holm-corrected Mann-Whitney p < ALPHA. Group counts, the span-vs-spike-count correlation and the
    tests in a panel underneath."""
    groups = list(curves)
    grid = _span_grid()
    all_sp = np.concatenate([spans[g] for g in groups])
    all_ns = np.concatenate([n_spikes[g] for g in groups])

    stats_lines = [f'{population}; units with >= {MIN_PEAK_BIN_SPIKES_FOR_SPAN} analysed spikes in the raw peak bin. '
                   'Span = 360° minus the largest gap between the unit\'s peak-bin spike phases; each phase is '
                   're-measured from the first spike after that gap (0° = start of the span, whatever its theta '
                   f'phase). One density per unit on [0, its span]: Gaussian kernel SD {SPAN_KDE_SIGMA_DEG:g}°, '
                   'reflected at both ends of the span, area 1.']
    for _, r in desc_df.iterrows():
        stats_lines.append(f'{_label(r[group_col])} (n={int(r["N_units"])} units, median '
                           f'{r["PeakBinSpikes_median"]:.0f} peak-bin spikes): span mean {r["Span_mean_deg"]:.1f}° '
                           f'± {r["Span_SEM_deg"]:.1f}° SEM, median {r["Span_median_deg"]:.1f}° '
                           f'[{r["Span_Q1_deg"]:.1f}–{r["Span_Q3_deg"]:.1f}]; mean density peaks at '
                           f'{r["MeanDensity_mode_deg"]:.0f}°')
    if len(all_sp) >= 3 and np.ptp(all_sp) > 0 and np.ptp(all_ns) > 0:
        rho, p_rho = stats.spearmanr(all_sp, all_ns)
        stats_lines.append(f'Span vs number of peak-bin spikes (units pooled): Spearman rho = {rho:.2f}, '
                           f'p = {_fmt_p(p_rho)} (n = {len(all_sp)}); the span of a sample grows with its size')
    stats_lines += _test_lines(omnibus_df, pairwise_df, group_col)
    if len(omnibus_df):
        stats_lines.append(f'Density permutation: T = Σ n_g ∫ (group mean density − grand mean)² dφ, p from '
                           f'{N_DENSITY_PERMUTATIONS} permutations of the units\' group labels; L1 distances between '
                           'mean curves in the workbook. Units treated as independent.')

    w_den, w_box = 6.0, _bar_fig_width(len(groups))
    fig_w = w_den + w_box + 0.5
    stats_lines = _wrap_stats(stats_lines, fig_w)
    h = 5.5
    panel_h = len(stats_lines) * FIG_STATS_LINE_IN + 0.2
    fig = plt.figure(figsize=(fig_w, h + panel_h))
    gs = fig.add_gridspec(2, 2, height_ratios=[h, panel_h], width_ratios=[w_den, w_box])
    ax_d, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.axis('off')

    handles = []
    for i, g in enumerate(groups):
        D = curves[g]
        face = _face(g, tm)
        ls = _SPAN_LINESTYLES[i % len(_SPAN_LINESTYLES)]
        for d in D:
            ax_d.plot(grid, d, color=face, linewidth=0.6, alpha=0.35, zorder=1)
        m = D.mean(axis=0)
        if len(D) > 1:
            sem = D.std(axis=0, ddof=1) / np.sqrt(len(D))
            ax_d.fill_between(grid, m - sem, m + sem, color=face, alpha=0.3, lw=0, zorder=2)
        ax_d.plot(grid, m, color=face, linewidth=2.2, linestyle=ls, zorder=3,
                  path_effects=[pe.Stroke(linewidth=3.6, foreground=PAL_BLACK), pe.Normal()])
        handles.append(Line2D([0], [0], color=face, linewidth=2.2, linestyle=ls,
                              path_effects=[pe.Stroke(linewidth=3.6, foreground=PAL_BLACK), pe.Normal()],
                              label=f'{_label(g)} (n={len(D)})'))
    ax_d.set_xlim(0, min(360.0, float(all_sp.max()) + 3 * SPAN_KDE_SIGMA_DEG))
    ax_d.set_ylim(bottom=0)
    ax_d.set_xticks(np.arange(0, 361, 45))
    ax_d.set_xlabel('Theta phase from start of span (deg)', fontsize=FS_LABEL - 2)
    ax_d.set_ylabel('Density (1/deg)', fontsize=FS_LABEL - 2)
    ax_d.set_title('Density per unit (thin) and group mean ± SEM', fontsize=FS_TITLE - 4)
    _style_axes(ax_d)

    positions = np.arange(1, len(groups) + 1)
    _group_boxplot(ax_b, [spans[g] for g in groups], positions, groups, tm_flags=[tm] * len(groups))
    rng = np.random.default_rng(0)
    for pos, g in zip(positions, groups):
        ax_b.scatter(pos + rng.uniform(-0.12, 0.12, size=len(spans[g])), spans[g], s=18, facecolors='#BFBFBF',
                     edgecolors=PAL_BLACK, linewidths=0.5, zorder=3)
        ax_b.plot(pos, np.mean(spans[g]), marker='D', color='white', markeredgecolor=PAL_BLACK, markersize=7,
                  zorder=4, linestyle='none')
    y_max = float(all_sp.max())
    top = _draw_sig_brackets(ax_b, dict(zip(groups, positions)),
                             pairwise_df[pairwise_df['Test'] == 'Mann-Whitney U'] if len(pairwise_df) else None,
                             y_base=y_max + 0.06 * 360, y_step=0.08 * 360, y_text=0.025 * 360, group_col=group_col)
    ax_b.set_ylim(0, max(360.0, (top if top is not None else y_max) + 0.06 * 360))
    ax_b.set_yticks(np.arange(0, 361, 90))
    ax_b.set_xticks(positions)
    ax_b.set_xticklabels([f'{_label(g)}\n(n={len(spans[g])})' for g in groups])
    ax_b.set_xlim(0.4, len(groups) + 0.6)
    ax_b.set_ylabel('Phase span (deg)', fontsize=FS_LABEL - 2)
    ax_b.set_title('Span per unit (diamond = mean)', fontsize=FS_TITLE - 4)
    _style_axes(ax_b)

    ax_d.legend(handles=handles, loc='upper right', fontsize=FS_LEGEND - 2, handlelength=3)
    _draw_stats(ax_info, stats_lines)
    fig.suptitle(textwrap.fill(f'{title_prefix}Theta-phase span of the raw-peak-bin spikes', int(fig_w * 7)),
                 fontsize=FS_TITLE - 2, fontweight='bold')
    fig.tight_layout()
    _save_fig(fig, out_path)


def run_peakbin_span(df: pd.DataFrame, group_col: str, group_order, out_dir: Path, file_prefix: str,
                     pooled: str, title_prefix: str = '') -> dict[str, pd.DataFrame]:
    """Peak-bin phase span per group (compare_peakbin_span) for each CELL_SETS selection, among units
    with a span (>= MIN_PEAK_BIN_SPIKES_FOR_SPAN peak-bin spikes); plotted (plot_peakbin_span) to
    out_dir/<file_prefix>_PeakBinPhaseSpan_<cell set>.png, also for a single group (tests need 2).
    Returns {'PkSpan_<set>_Desc' / '_Omni' / '_Pair': table}, empty tables left out."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for set_col, set_file, set_sheet, set_label in CELL_SETS:
        sel = df['PeakBin_PhaseSpan_deg'].notna()
        if set_col is not None:
            sel &= df[set_col] == True                                      # noqa: E712
        sub = df.loc[sel]
        present = [g for g in group_order if (sub[group_col] == g).any()]
        if not present:
            print(f'No {set_label} with a peak-bin phase span -- span comparison skipped.')
            continue
        curves, spans, n_spikes = {}, {}, {}
        for g in present:
            sg = sub[sub[group_col] == g]
            curves[g] = np.vstack([span_density(_parse_span_phases(s)) for s in sg['PeakBin_SpanPhases_deg']])
            spans[g] = sg['PeakBin_PhaseSpan_deg'].to_numpy(float)
            n_spikes[g] = sg['PeakBin_n_spikes'].to_numpy(float)
        desc_df, omnibus_df, pairwise_df = compare_peakbin_span(curves, spans, n_spikes, group_col,
                                                                np.random.default_rng(RANDOM_SEED))
        tables.update({f'PkSpan_{set_sheet}_Desc': desc_df, f'PkSpan_{set_sheet}_Omni': omnibus_df,
                       f'PkSpan_{set_sheet}_Pair': pairwise_df})
        plot_path = out_dir / f'{file_prefix}_PeakBinPhaseSpan_{set_file}.png'
        plot_peakbin_span(curves, spans, n_spikes, desc_df, omnibus_df, pairwise_df, plot_path, group_col,
                          title_prefix, population=f'{set_label[0].upper()}{set_label[1:]} (one density per unit); '
                                                   f'{pooled}', tm=set_col == 'ThetaModulated')
        omni_str = '; '.join(f'{r["Test"]} p = {_fmt_p(r["p_value"])}' for _, r in omnibus_df.iterrows())
        print(f'Peak-bin phase span, {set_label} ({", ".join(f"{_label(g)} {len(spans[g])}" for g in present)} units)'
              + (f': {omni_str}' if omni_str else '') + f' -- plot saved: {plot_path.name}')
    return {k: t for k, t in tables.items() if not t.empty}


# ============================================================================
# One comparison (the groups of one level), and the animal-controlled level
# ============================================================================

def run_group_comparisons(df: pd.DataFrame, group_col: str, group_order, out_dir: Path,
                          file_prefix: str, group_word: str, title_prefix: str = '') -> dict[str, pd.DataFrame]:
    """Every comparison for the units in `df`, grouped by df[group_col] in `group_order` order,
    animals pooled within each group:
      - % Rayleigh+ / TMI+ / Rayleigh+ & TMI+ units, tests on the last (run_theta_mod_proportion),
        out_dir/<file_prefix>_ThetaModProportion.png, and the same tests on the % of tested units that
        are Rayleigh+ & TMI+, out_dir/<file_prefix>_ThetaModProportion_Tested.png
      - theta phase at the raw peak bin, firing onset and firing offset (SPIKE_PHASE_METRICS), for
        each CELL_SETS selection (all units / theta-modulated units), circular tests,
        out_dir/<file_prefix>_PhaseAt<metric>_<cell set>.png (skipped with < 2 groups holding data)
      - theta-modulated vs non-theta-modulated units within each group (run_tm_vs_non)
      - theta-phase span of the peak-bin spikes, one density per unit (run_peakbin_span),
        out_dir/<file_prefix>_PeakBinPhaseSpan_<cell set>.png
    Returns {table key: DataFrame} (empty tables left out)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pooled = f'animals pooled within {group_word}'
    tables = run_theta_mod_proportion(df, group_col, group_order, out_dir / f'{file_prefix}_ThetaModProportion.png',
                                      pooled, title_prefix)

    for col, label, file_tag, sheet_tag in SPIKE_PHASE_METRICS:
        for set_col, set_file, set_sheet, set_label in CELL_SETS:
            sel = df[col].notna()
            if set_col is not None:
                sel &= df[set_col] == True                                  # noqa: E712
            df_pt = df.loc[sel]
            present = [g for g in group_order if (df_pt[group_col] == g).any()]
            if len(present) < 2:
                print(f'Fewer than 2 {group_word}s with a {file_tag} phase among {set_label} '
                      f'({", ".join(map(_label, present)) or "none"}) -- comparison skipped.')
                continue
            desc_df, omnibus_df, pairwise_df = compare_groups_phase(
                df_pt, np.random.default_rng(RANDOM_SEED), col, label, group_col, group_order)
            tables[f'{sheet_tag}_{set_sheet}_Desc'] = desc_df
            tables[f'{sheet_tag}_{set_sheet}_Omni'] = omnibus_df
            tables[f'{sheet_tag}_{set_sheet}_Pair'] = pairwise_df
            plot_path = out_dir / f'{file_prefix}_PhaseAt{file_tag}_{set_file}.png'
            plot_phase_by_group(df_pt, desc_df, omnibus_df, pairwise_df, plot_path, col, label, group_col,
                                group_order, title_prefix=title_prefix,
                                population=f'{set_label[0].upper()}{set_label[1:]} (one value per unit); {pooled}')
            print(f'{file_tag} phase, {set_label} ({", ".join(map(_label, present))}): plot saved: {plot_path.name}')

    tables.update(run_tm_vs_non(df, group_col, group_order, out_dir, file_prefix, pooled, title_prefix))
    tables.update(run_peakbin_span(df, group_col, group_order, out_dir, file_prefix, pooled, title_prefix))
    return {k: t for k, t in tables.items() if not t.empty}


def _stitch_side_by_side(paths: list, out_path: Path):
    """Save the images `paths` side by side (top-aligned, PANEL_GAP_IN apart,
    on white) as one image. A missing image (None) leaves a blank space as
    wide as the widest one, so every comparison keeps its position."""
    imgs = []
    for p in paths:
        if p is None:
            imgs.append(None)
            continue
        with Image.open(p) as im:
            imgs.append(im.convert('RGB'))
    present = [im for im in imgs if im is not None]
    blank_w = max(im.width for im in present)
    widths = [im.width if im is not None else blank_w for im in imgs]
    gap = int(round(PANEL_GAP_IN * FIG_DPI))
    canvas = Image.new('RGB', (sum(widths) + gap * (len(imgs) - 1),
                               max(im.height for im in present)), 'white')
    x = 0
    for im, w in zip(imgs, widths):
        if im is not None:
            canvas.paste(im, (x, 0))
        x += w + gap
    canvas.save(out_path, dpi=(FIG_DPI, FIG_DPI))


def _combine_side_by_side(src_dirs: list[Path], out_dir: Path):
    """For every .png found in any of `src_dirs` (by name), save the versions from all
    `src_dirs` side by side, in that order, to the same name in `out_dir`."""
    names = sorted({p.name for d in src_dirs if d.is_dir() for p in d.glob('*.png')})
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        _stitch_side_by_side([d / name if (d / name).is_file() else None for d in src_dirs], out_dir / name)
    print(f'\n{len(names)} side-by-side plots ({" | ".join(d.name for d in src_dirs)}) saved to {out_dir}')


def run_animal_controlled(df: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    """Level 3: for each entry of ANIMAL_CONTROLLED_COMPARISONS, compare its arenas using only the
    units of its animals (run_group_comparisons). Each plot is saved to out_dir with the comparisons
    side by side, each with its own statistics -- the comparisons are not tested against each other.
    Returns {'AC<arenas>_<table key>': table}, each table with an 'Animals' column."""
    all_tables = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        src_dirs = []
        # Same y-range for the versions of every plot (plu.shared_y_axes): the lowest lower and
        # highest upper limit of the comparisons.
        with plu.shared_y_axes([Path(tmp_dir) / n for n in ANIMAL_CONTROLLED_COMPARISONS]):
            for name, (animals, arenas) in ANIMAL_CONTROLLED_COMPARISONS.items():
                order = [a for a in ARENA_LABELS if a in arenas]
                wanted = {a.lower() for a in animals}
                sel = df[df['Animal'].astype(str).str.lower().isin(wanted) & df['Arena'].isin(order)]
                print(f'\n=== Animal-controlled {name}: units per animal and arena ===')
                if sel.empty:
                    print('No units; skipped.')
                    continue
                print(pd.crosstab(sel['Animal'], sel['Arena'], margins=True))
                for arena in order:
                    missing = [a for a in animals if not ((sel['Animal'].astype(str).str.lower() == a.lower())
                                                          & (sel['Arena'] == arena)).any()]
                    if missing:
                        print(f'WARNING: no units of {missing} in the {arena} arena.')
                src = Path(tmp_dir) / name
                src_dirs.append(src)
                tables = run_group_comparisons(sel, 'Arena', order, src, 'AnimalControlled', 'arena',
                                               title_prefix=f'{" + ".join(animals)} only: ')
                tag = 'AC' + ''.join(order)
                for key, table in tables.items():
                    table.insert(0, 'Animals', ', '.join(animals))
                    all_tables[f'{tag}_{key}'] = table
        if src_dirs:
            _combine_side_by_side(src_dirs, out_dir)
    return all_tables


# ============================================================================
# ACG theta-epoch QC plot
# ============================================================================

PAL_NONTHETA = '#C44E52'


@_arena_fig_style
def plot_acg_epochs(acg_df: pd.DataFrame, out_path: Path):
    """Matched reference-sinusoid frequency (freq_est) vs. normalised ACG peak range for every
    epoch that got an ACG fit (passed the noise rejection), pooled across all sessions -- the
    frequency vs. peak-range view of Figc_ACG_ScalingCorrected_v8_NoiseCleaning.py. Each point is
    coloured by its animal; non-theta epochs (freq_est outside ACG_FREQ_RANGE) get a red outline;
    dashed lines mark the theta band. Epochs matched to a reference too slow to have a peak range
    are counted but can't be drawn. The panel underneath gives, per animal and pooled, the
    percentage of ALL epochs kept as theta, thrown out as non-theta and thrown out as noise.
    """
    if acg_df.empty:
        print('No ACG epochs -- ACG QC plot skipped.')
        return None
    fitted = acg_df[acg_df['Status'].isin(['accepted', 'rejected_outofband'])]
    animals = sorted(acg_df['Animal'].unique())
    cmap = plt.get_cmap('tab10' if len(animals) <= 10 else 'tab20')
    colors = {a: ANIMAL_COLORS.get(a, cmap(i % cmap.N)) for i, a in enumerate(animals)}
    band = f'{ACG_FREQ_RANGE[0]:g}-{ACG_FREQ_RANGE[1]:g} Hz'

    def _status_line(label, sub):
        n = len(sub)
        n_acc = int((sub['Status'] == 'accepted').sum())
        n_oob = int((sub['Status'] == 'rejected_outofband').sum())
        n_sat = int((sub['Status'] == 'rejected_saturated').sum())
        n_noise = n - n_acc - n_oob - n_sat
        return (f'{label}: theta {100 * n_acc / n:.1f}% ({n_acc}/{n}) | '
                f'non-theta {100 * n_oob / n:.1f}% ({n_oob}/{n}) | '
                f'saturated {100 * n_sat / n:.1f}% ({n_sat}/{n}) | '
                f'noise {100 * n_noise / n:.1f}% ({n_noise}/{n})')

    stats_lines = [f'All epochs. Theta = freq_est (bank {ACG_REF_BANK[0]:g}-{ACG_REF_BANK[1]:g} Hz) in {band}; '
                   f'saturated = clipped at +/-InputRange or within {SATURATION_MARGIN_S:g} s of clipping; '
                   f'noise = delta > theta power, peak-to-peak MAD outlier or other rejection']
    stats_lines += [_status_line(a, acg_df[acg_df['Animal'] == a]) for a in animals]
    if len(animals) > 1:
        stats_lines.append(_status_line('All animals', acg_df))
    stats_lines = _wrap_stats(stats_lines, 7.0)

    fig, ax, ax_info = _new_fig((7.0, 5.5), stats_lines=stats_lines)
    for a in animals:
        sub = fitted[fitted['Animal'] == a]
        acc = sub[sub['Status'] == 'accepted']
        oob = sub[sub['Status'] == 'rejected_outofband']
        ax.scatter(acc['freq_est'], acc['peakrangenorm'], s=10, color=colors[a], alpha=0.4,
                   edgecolor='none', zorder=2)
        ax.scatter(oob['freq_est'], oob['peakrangenorm'], s=18, color=colors[a], alpha=0.8,
                   edgecolor=PAL_NONTHETA, linewidth=0.8, zorder=3)
    for edge in ACG_FREQ_RANGE:
        ax.axvline(edge, color=PAL_BLACK, ls='--', lw=FIG_AXIS_LW, zorder=1)
    ax.set_xlim(ACG_REF_BANK)
    n_no_pr = int(fitted['peakrangenorm'].isna().sum())
    if n_no_pr:
        ax.text(0.98, 0.02, f'{n_no_pr} fitted epochs have no peak range (not drawn)',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=FS_STATS, zorder=5)

    handles = [Line2D([], [], ls='', marker='o', markersize=6, color=colors[a],
                      label=f'{a} (n={int((fitted["Animal"] == a).sum())})') for a in animals]
    handles += [Line2D([], [], ls='', marker='o', markersize=6, markerfacecolor='white',
                       markeredgecolor=PAL_NONTHETA, markeredgewidth=1.2,
                       label=f'non-theta (outside {band})'),
                Line2D([], [], color=PAL_BLACK, ls='--', lw=FIG_AXIS_LW, label=f'theta band ({band})')]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=FS_LEGEND - 2)

    ax.set_xlabel('Matched sinusoid frequency, freq_est (Hz)', fontsize=FS_LABEL - 2)
    ax.set_ylabel('ACG peak range (norm.)', fontsize=FS_LABEL - 2)
    ax.tick_params(labelsize=FS_TICK)
    _set_title(ax, 'ACG theta epochs: frequency vs. peak range')
    _draw_stats(ax_info, stats_lines)
    _save_fig(fig, out_path)
    return out_path


# ============================================================================
# Per-session processing
# ============================================================================

PLOT_SUFFIXES = ('PolarPlot', 'PhaseHistogram', 'TMIShuffle', 'PeakOnsetOffset')


def process_session(data_folder: Path, rng, acg_records: list | None = None) -> list[dict]:
    """Run Steps 1-4 on every unit in one session; returns one row per unit.
    If `acg_records` is given (and USE_ACG_THETA_EPOCHS), one dict per ACG
    epoch of each LFP file used (Animal, Session, lfp_file, epoch_index, ED_min,
    freq_est, peakrangenorm, Status) is appended to it, for the pooled ACG QC plot."""
    output_dir = data_folder / SESSION_OUTPUT_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)

    ncs_files = sorted(data_folder.glob('*.ncs'), key=_natural_key)

    # Tracking is only needed for Steps 3-4; a missing tracking file does not block Steps 1-2.
    pos_ts = pos_xy = None
    try:
        tracking_path = _find_tracking_file(data_folder)
        print(f'Using tracking file: {tracking_path.name}')
        pos_ts, pos_xy = load_tracking(tracking_path, TRACKING_TIME_UNIT)
    except FileNotFoundError as exc:
        print(f'  {exc} -- place fields / peak-bin / onset / offset phases skipped for this session.')

    session_label = '_'.join(data_folder.parts[-3:])
    animal_id = detect_animal_id(data_folder)
    arena = detect_arena(data_folder)
    session_type = detect_session_type(data_folder, arena)
    ntt_files = sorted(data_folder.glob('*.ntt'), key=_natural_key)

    # Cache per matched .ncs file (channel loading, theta filtering, ACG epochs)
    # so tetrodes that happen to share one don't reload/refilter it more than once.
    lfp_cache: dict[Path, dict] = {}

    rows = []
    for ntt_path in ntt_files:
        print(f'Processing: {ntt_path.name}')

        try:
            if MATCH_SPIKE_LFP_FILE:
                theta_ncs = match_ncs_to_ntt(ntt_path, ncs_files)
            elif ncs_files:
                theta_ncs = ncs_files[0]
            else:
                raise FileNotFoundError(f'No .ncs files found in {data_folder}')
        except FileNotFoundError as exc:
            print(f'  {exc} -- skipping this tetrode.')
            continue

        if theta_ncs not in lfp_cache:
            print(f'  Using LFP file: {theta_ncs.name}')
            try:
                lfp_sig, lfp_start, lfp_fs, ncs_records = load_ncs(theta_ncs)
            except ValueError as exc:
                print(f'  {theta_ncs.name}: {exc} -- skipping every tetrode referenced to it.')
                lfp_cache[theta_ncs] = None
                continue
            # Clean (resample to ACG_FS, notch, wavelet denoise, detrend, outlier fill, low-pass)
            # before the Hilbert phase; everything downstream then works on this 1 kHz trace.
            lfp_sig, lfp_ts, lfp_fs, outlier_frac = clean_lfp(lfp_sig, lfp_start, lfp_fs)
            print(f'  noise cleaning: {100 * outlier_frac:.2f}% samples outlier-filled')
            filtered_lfp = bandpass_filter(lfp_sig, LFP_FILTER_BAND[0], LFP_FILTER_BAND[1], lfp_fs)
            lfp_phase_unwrapped = np.unwrap(hilbert_phase(filtered_lfp, lfp_fs)[0])
            del filtered_lfp

            t_start = t_stop = None
            if pos_ts is not None:
                t_start = max(pos_ts.min(), lfp_ts.min())
                t_stop = min(pos_ts.max(), lfp_ts.max())

            acg_epoch_ok = None
            if USE_ACG_THETA_EPOCHS:
                acg = acg_theta_epochs(lfp_sig, ncs_records)
                acg_epoch_ok = acg['ok']
                rejected = pd.Series(acg['status'][~acg_epoch_ok]).value_counts()
                print(f'  ACG theta epochs (freq_est in {ACG_FREQ_RANGE[0]:g}-{ACG_FREQ_RANGE[1]:g} Hz): '
                      f'{int(acg_epoch_ok.sum())}/{len(acg_epoch_ok)}; thrown out: '
                      + (', '.join(f'{n} {s.removeprefix("rejected_")}' for s, n in rejected.items()) or 'none'))
                if acg_records is not None:
                    acg_records.extend(
                        dict(Animal=animal_id, Session=session_label, FolderPath=str(data_folder),
                             lfp_file=theta_ncs.name, epoch_index=i, ED_min=acg['ED_min'][i],
                             freq_est=acg['freq_est'][i], peakrangenorm=acg['peakrangenorm'][i],
                             Status=acg['status'][i], n_clipped=int(acg['n_clipped'][i]),
                             InputRange_uV=ncs_records['input_range_uv'])
                        for i in range(len(acg_epoch_ok)))

            lfp_cache[theta_ncs] = dict(lfp_ts=lfp_ts, lfp_fs=lfp_fs, lfp_phase_unwrapped=lfp_phase_unwrapped,
                                        t_start=t_start, t_stop=t_stop, acg_epoch_ok=acg_epoch_ok)

        lfp_data = lfp_cache[theta_ncs]
        if lfp_data is None:
            print(f'  {theta_ncs.name} could not be loaded -- skipping this tetrode.')
            continue
        lfp_ts, lfp_fs = lfp_data['lfp_ts'], lfp_data['lfp_fs']
        lfp_phase_unwrapped = lfp_data['lfp_phase_unwrapped']
        t_start, t_stop = lfp_data['t_start'], lfp_data['t_stop']
        acg_epoch_ok = lfp_data['acg_epoch_ok']

        units = load_ntt_spike_times(ntt_path)
        for cell_number, spk_ts in units.items():
            unit_label = f'{ntt_path.stem}_cell{cell_number}' if len(units) > 1 else ntt_path.stem
            plot_label = f'{animal_id} | {session_label} | {unit_label}'
            file_prefix = f'{animal_id}_{unit_label}'
            row: dict[str, Any] = dict(Animal=animal_id, Session=session_label, FolderPath=str(data_folder),
                                       Arena=arena, SessionType=session_type, Unit=unit_label,
                                       ntt_file=ntt_path.name, lfp_file=theta_ncs.name, cell_number=cell_number,
                                       n_spikes_total=len(spk_ts))

            # Remove this unit's plots from any earlier run, so a unit that no longer gets one leaves no stale plot.
            for suffix in PLOT_SUFFIXES:
                (output_dir / f'{file_prefix}_{suffix}.png').unlink(missing_ok=True)
            class_prefix = f'{animal_id}_{session_label}_{unit_label}'
            for class_dir_name in THETA_MOD_CLASS_FOLDERS.values():
                for suffix in THETA_MOD_PLOT_SUFFIXES:
                    (RESULTS_FOLDER / class_dir_name / f'{class_prefix}_{suffix}.png').unlink(missing_ok=True)

            # ---- ACG theta-epoch spike selection (every phase uses only these spikes) ----
            spk_ts_all = spk_ts
            if USE_ACG_THETA_EPOCHS:
                spk_ts = spk_ts_all[times_in_ok_epochs(spk_ts_all, lfp_ts, lfp_fs, acg_epoch_ok)]
                row['n_spikes_ACGtheta'] = len(spk_ts)
                row['ACGtheta_epoch_frac'] = float(acg_epoch_ok.mean()) if len(acg_epoch_ok) else np.nan

            # ---- Steps 1 & 2: Rayleigh test, TMI shuffle test ----
            metrics, phase_deg, shuffle_tmis = compute_theta_modulation(spk_ts, lfp_ts, lfp_phase_unwrapped,
                                                                        lfp_fs, rng)
            row.update(metrics)
            if metrics['n_spikes_theta'] >= MIN_SPIKES_FOR_TMI:
                plot_polar_theta(phase_deg, metrics['MRL'], metrics['PreferredPhase_deg'], metrics['Rayleigh_p'],
                                 metrics['Rayleigh_Significant'], plot_label, output_dir / f'{file_prefix}_PolarPlot.png')
                plot_phase_histogram(phase_deg, metrics['Rayleigh_Significant'], plot_label,
                                     output_dir / f'{file_prefix}_PhaseHistogram.png')
                plot_tmi_shuffle(phase_deg, metrics['TMI'], metrics['TMI_shuffle_p'], metrics['TMI_Significant'],
                                 shuffle_tmis, plot_label, output_dir / f'{file_prefix}_TMIShuffle.png')
                # Also copy them into RESULTS_FOLDER/<theta-modulation class folder> (next to the Excel file).
                class_dir = RESULTS_FOLDER / THETA_MOD_CLASS_FOLDERS[metrics['ThetaModClass']]
                class_dir.mkdir(parents=True, exist_ok=True)
                for suffix in THETA_MOD_PLOT_SUFFIXES:
                    shutil.copy2(output_dir / f'{file_prefix}_{suffix}.png', class_dir / f'{class_prefix}_{suffix}.png')
                print(f'  {unit_label}: Rayleigh p={metrics["Rayleigh_p"]:.3g}, TMI={metrics["TMI"]:.3f} '
                      f'(p={metrics["TMI_shuffle_p"]:.3g}) -> {metrics["ThetaModClass"]}')
            else:
                print(f'  {unit_label}: only {metrics["n_spikes_theta"]} spikes with LFP coverage '
                      f'(< MIN_SPIKES_FOR_TMI={MIN_SPIKES_FOR_TMI}): Rayleigh / TMI tests skipped')

            # ---- Steps 3 & 4: place fields, raw peak bin, peak-bin / onset / offset phases ----
            _process_place(row, pos_ts, pos_xy, spk_ts, spk_ts_all, t_start, t_stop, arena, lfp_ts, lfp_fs,
                           lfp_phase_unwrapped, output_dir / f'{file_prefix}_PeakOnsetOffset.png', plot_label)
            rows.append(row)

    return rows


def _process_place(row, pos_ts, pos_xy, spk_ts, spk_ts_all, t_start, t_stop, arena, lfp_ts, lfp_fs,
                   lfp_phase_unwrapped, plot_path: Path, plot_label: str):
    """Steps 3-4 for one unit: place fields and raw peak bin from all of its spikes (within the
    tracking / LFP overlap), then the peak-bin / onset / offset phases of its ACG theta-epoch spikes
    `spk_ts`; adds the unit's columns to `row` and saves its PeakOnsetOffset figure. A step that
    cannot run leaves PlaceSkippedReason saying why."""
    row['PlaceSkippedReason'] = ''
    if pos_ts is None:
        row['PlaceSkippedReason'] = 'no tracking file found for this session'
        return
    spk_ts_all_overlap = spk_ts_all[(spk_ts_all >= t_start) & (spk_ts_all <= t_stop)]
    spk_ts_overlap = spk_ts[(spk_ts >= t_start) & (spk_ts <= t_stop)]
    try:
        maps = detect_place_fields(pos_ts, pos_xy, spk_ts_all_overlap, arena, BINSIDE, SMTH_WIDTH)
    except Exception as exc:
        row['PlaceSkippedReason'] = f'ERROR in rate map / place-field detection ({exc})'
        print(f'  place-field detection ERROR ({exc})')
        return

    main = maps['main_field']
    row.update(n_fields=len(maps['fields']), FieldDetection=maps['detection'],
               raw_peak_rate_Hz=maps['raw_peak_rate_Hz'], raw_peak_x_cm=maps['raw_peak_xy'][0],
               raw_peak_y_cm=maps['raw_peak_xy'][1], raw_peak_theta_deg=maps['raw_peak_theta_deg'],
               raw_peak_in_field=maps['raw_peak_in_field'], main_field_source=maps['main_field_source'])
    if main is not None:
        row.update(main_field=main['field_number'], field_n_bins=main['n_bins'], field_area_cm2=main['area_cm2'],
                   field_pct_of_occupied_area=main['pct_of_occupied_area'],
                   field_peak_rate_Hz=main['peak_rate_Hz'], field_peak_rate_raw_Hz=main['peak_rate_raw_Hz'])
    elif not maps['fields']:
        row['PlaceSkippedReason'] = 'no place field detected: no onset / offset'
    else:
        row['PlaceSkippedReason'] = 'raw and smoothed peak bins outside every place field: no onset / offset'

    res = compute_peak_onset_offset(pos_ts, pos_xy, maps, spk_ts_overlap, lfp_ts, lfp_fs, lfp_phase_unwrapped)
    row.update(res['summary'])
    s = res['summary']
    field_str = ('no main field' if main is None else
                 f'main field F{main["field_number"]} (holds the {maps["main_field_source"]})')
    print(f'  {len(maps["fields"])} field(s), {field_str}; '
          f'peak bin {s["PeakBin_n_spikes"]} spikes ({s["PeakBin_phase_deg"]:.0f} deg, '
          f'span {s["PeakBin_PhaseSpan_deg"]:.0f} deg); '
          f'{s["n_laps_kept"]}/{s["n_laps"]} laps kept, onset {s["Onset_phase_deg"]:.0f} / '
          f'offset {s["Offset_phase_deg"]:.0f} deg')
    plot_peak_onset_offset(maps, res, plot_label, plot_path)


# ============================================================================
# Population: summary, workbook, comparisons
# ============================================================================

UNIT_COLUMNS = [
    'Animal', 'Session', 'FolderPath', 'Arena', 'SessionType', 'Unit', 'ntt_file', 'lfp_file', 'cell_number',
    'n_spikes_total', 'n_spikes_ACGtheta', 'ACGtheta_epoch_frac',
    # Steps 1-2
    'n_spikes_theta', 'MRL', 'PreferredPhase_deg', 'Rayleigh_p', 'Rayleigh_Significant',
    'PhasePeak_deg', 'PhaseValley_deg', 'TMI', 'TMI_shuffle_p', 'TMI_Significant', 'ThetaModulated', 'ThetaModClass',
    # Step 3
    'n_fields', 'FieldDetection', 'raw_peak_rate_Hz', 'raw_peak_x_cm', 'raw_peak_y_cm', 'raw_peak_theta_deg',
    'raw_peak_in_field', 'main_field_source', 'main_field', 'field_n_bins', 'field_area_cm2',
    'field_pct_of_occupied_area', 'field_peak_rate_Hz', 'field_peak_rate_raw_Hz',
    # Step 4
    'n_spikes_analysed', 'n_spikes_in_laps', 'n_laps', 'n_laps_session_edge', 'n_laps_kept',
    'PeakBin_n_spikes', 'PeakBin_phase_deg', 'PeakBin_MRL', 'PeakBin_range',
    'PeakBin_PhaseSpan_deg', 'PeakBin_PhaseSpan_start_deg', 'PeakBin_SpanPhases_deg',
    'Onset_n_laps', 'Onset_phase_deg', 'Onset_MRL', 'Onset_range',
    'Offset_n_laps', 'Offset_phase_deg', 'Offset_MRL', 'Offset_range',
    'PlaceSkippedReason',
]


def build_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Counts / percentages for the 'Summary' sheet, one row per unit in df."""
    def pct(n, d):
        return (100.0 * n / d) if d > 0 else np.nan

    total = len(df)
    tm = df['ThetaModulated'] == True                                      # noqa: E712
    rows = []
    for name, mask, denom, denom_label in (
            (f'Units with >= {MIN_SPIKES_FOR_TMI} theta-epoch spikes (tested)',
             df['n_spikes_theta'] >= MIN_SPIKES_FOR_TMI, total, 'total units'),
            ('Rayleigh+ units', df['Rayleigh_Significant'] == True, total, 'total units'),               # noqa: E712
            ('TMI+ units', df['TMI_Significant'] == True, total, 'total units'),                         # noqa: E712
            ('Rayleigh+ & TMI+ (theta-modulated) units', tm, total, 'total units'),
            *((f'ThetaModClass = {c}', df['ThetaModClass'] == c, total, 'total units')
              for c in ('Rayleigh+ & TMI+', 'Rayleigh+ only', 'TMI+ only', 'neither', 'too few spikes')),
            ('Units with a raw peak bin', df['raw_peak_rate_Hz'].notna(), total, 'total units'),
            ('Units whose raw peak bin is inside a place field', df['raw_peak_in_field'] == True,       # noqa: E712
             total, 'total units'),
            ('Units whose main field holds the raw peak bin', df['main_field_source'] == 'raw peak bin',
             total, 'total units'),
            ('Units whose main field holds the smoothed peak bin (raw peak bin outside every field)',
             df['main_field_source'] == 'smoothed peak bin', total, 'total units'),
            ('Units with a peak-bin phase', df['PeakBin_phase_deg'].notna(), total, 'total units'),
            (f'Units with a peak-bin phase span (>= {MIN_PEAK_BIN_SPIKES_FOR_SPAN} peak-bin spikes)',
             df['PeakBin_PhaseSpan_deg'].notna(), total, 'total units'),
            ('Units with an onset / offset phase', df['Onset_phase_deg'].notna(), total, 'total units'),
            ('Theta-modulated units with a peak-bin phase', tm & df['PeakBin_phase_deg'].notna(),
             int(tm.sum()), 'theta-modulated units'),
            ('Theta-modulated units with an onset / offset phase', tm & df['Onset_phase_deg'].notna(),
             int(tm.sum()), 'theta-modulated units')):
        n = int(mask.sum())
        rows.append(dict(Metric=name, Count=n, Denominator=denom, DenominatorLabel=denom_label, Percent=pct(n, denom)))
    return pd.DataFrame(rows, columns=['Metric', 'Count', 'Denominator', 'DenominatorLabel', 'Percent'])


def save_metadata_csv(csv_path: Path) -> Path:
    """Write the code name and every hard-coded analysis setting to a
    Parameter,Value CSV (next to the results Excel file).

    The configuration constants (the ALL_CAPS module-level assignments) are collected
    automatically, so newly added settings are recorded without editing this function;
    the constants hard-coded inside functions are listed explicitly.
    """
    code_path = Path(__file__).resolve()
    rows = [
        ('CodeName', code_path.name),
        ('CodePath', str(code_path)),
        ('RunTimestamp', pd.Timestamp.now().isoformat(timespec='seconds')),
    ]
    module_globals = globals()
    config_names = [n for n in module_globals
                    if n.isupper() and not n.startswith(('_', 'PAL_', 'FS_', 'FIG_'))   # skip figure style
                    and n not in ('ARENA_COLORS', 'SESSION_COLORS', 'GROUP_COLORS', 'GROUP_COUNT_COLUMNS',
                                  'UNIT_COLUMNS', 'README_NOTES', 'PLOT_SUFFIXES')]
    rows += [(n, repr(module_globals[n]) if not isinstance(module_globals[n], (str, Path))
              else str(module_globals[n])) for n in config_names]
    # constants hard-coded inside functions
    rows += [
        ('BANDPASS_FILTER_ORDER', 4),                         # bandpass_filter default
        ('TMI_NUM_CYCLES', 5),                                # calc_tmi default
        ('TMI_HIST_SMOOTH_KERNEL_N', 7),                      # calc_tmi histogram smoothing
        ('TMI_HIST_SMOOTH_SIGMA', 0.5),
        ('ACG_WELCH_WINDOW', 'hann'),
        ('ACG_WAVELET_THRESHOLD', 'BayesShrink, soft (noise sigma from finest level)'),
        ('ACG_THETA_EPOCH_CRITERION', 'passes delta/theta + peak-to-peak MAD noise rejection AND ED_min reference '
                                      'sinusoid (matched over ACG_REF_BANK) within ACG_FREQ_RANGE; no ED_min '
                                      'value threshold'),
        ('NCS_LOADER', 'ncs_repair.load_repaired_uv: valid samples on a timestamp-anchored grid, gaps up to 1 s '
                       'interpolated, files with a longer gap skipped'),
        ('ADBitVolts', 'per-file, from .ncs header (-ADBitVolts); DEFAULT_ADBITVOLTS only if absent'),
        ('InputRange', 'per-file, from .ncs header (-InputRange, uV); no saturation check if absent'),
        ('SATURATION_DETECTION', 'raw repaired native-rate trace, before filtering; |uV| >= InputRange - '
                                 'SATURATION_TOL_STEPS * ADBitVolts; each clipped run widened by SATURATION_MARGIN_S '
                                 'both sides; epoch rejected if it overlaps a widened run (before delta/theta + MAD)'),
        ('THETA_MODULATED_CRITERION', 'Rayleigh p < ALPHA AND TMI burst-preserving shuffle p < ALPHA'),
        ('PROPORTION_TESTS', 'chi-square test of independence + pairwise chi-square (Yates for 2 x 2), Holm'),
        ('PHASE_TESTS', 'Mardia-Watson-Wheeler (permutation p) + Watson-Williams, Holm'),
        ('MAIN_FIELD_RULE', 'field holding the raw peak bin; if none, the field holding the smoothed-map peak bin'),
        ('PEAKBIN_SPAN', 'span = 360 deg - largest circular gap between peak-bin spike phases; phases re-measured '
                         'from the first spike after the gap'),
        ('PEAKBIN_SPAN_TESTS', 'density curves: permutation of unit labels, T = sum n_g * integral (group mean - '
                               'grand mean)^2; span: Kruskal-Wallis + pairwise Mann-Whitney U; Holm per test'),
    ]
    pd.DataFrame(rows, columns=['Parameter', 'Value']).to_csv(csv_path, index=False)
    return csv_path


README_NOTES = [
    'Units sheet: ONE ROW PER UNIT (one .ntt cluster in one session). Theta phase = Hilbert phase of the cleaned, '
    f'{LFP_FILTER_BAND[0]:g}-{LFP_FILTER_BAND[1]:g} Hz filtered LFP, 0 deg = theta peak, 180 deg = trough; only spikes in '
    f'ACG theta epochs enter any phase: {ACG_EPOCH_SEC:g} s epochs not within {SATURATION_MARGIN_S:g} s of ADC clipping '
    f'(|uV| >= the file\'s -InputRange minus {SATURATION_TOL_STEPS:g} ADC steps; reject: {ACG_REJECT_SATURATED_EPOCHS}) '
    'that pass the noise rejection (delta > theta power, '
    f'peak-to-peak MAD outlier) and whose closest reference sinusoid ({ACG_REF_BANK[0]:g}-{ACG_REF_BANK[1]:g} Hz bank) lies '
    f'in {ACG_FREQ_RANGE[0]:g}-{ACG_FREQ_RANGE[1]:g} Hz (Figc_ACG_ScalingCorrected_v8_NoiseCleaning.py; no ED_min threshold). '
    'n_spikes_ACGtheta / ACGtheta_epoch_frac: the unit\'s spikes in theta epochs / fraction of its LFP\'s epochs that are theta.',
    f'Theta modulation (units with >= {MIN_SPIKES_FOR_TMI} theta-epoch spikes): Rayleigh_Significant = Rayleigh p < {ALPHA:g}; '
    f'TMI_Significant = burst-preserving TMI shuffle p < {ALPHA:g} (Frank et al. 2001); ThetaModulated = both. Units with too '
    'few spikes are negative on all three. ThetaModClass is the exclusive split.',
    f'Raw peak bin: the valid bin (visited >= {FIELD_MIN_VISITS} times) with the highest spikes / occupancy, unsmoothed, from all '
    f'spikes while moving (>= {RATEMAP_IMMOBILITY_SPEED_CM_S:g} cm/s); angular bin on the Circle track. Place fields: v30 '
    'threshold method on Climer\'s rate map; main_field = the field holding the raw peak bin, or, if the raw peak bin is '
    'outside every field (raw_peak_in_field = False), the field holding the peak bin of the smoothed map; '
    'main_field_source says which (blank: no main field).',
    f'Analysed spikes: theta-epoch spikes with an LFP phase fired at >= {MIN_SPEED_CM_S:g} cm/s. PeakBin_* = circular mean / MRL '
    'of every analysed spike in the raw peak bin, whole session.',
    f'Peak-bin phase span (units with >= {MIN_PEAK_BIN_SPIKES_FOR_SPAN} analysed peak-bin spikes): PeakBin_PhaseSpan_deg = 360 deg '
    'minus the largest circular gap between the peak-bin spike phases; PeakBin_PhaseSpan_start_deg = theta phase of the first '
    'spike after that gap; PeakBin_SpanPhases_deg = every peak-bin spike phase re-measured from that start (0 to span). One '
    f'Gaussian-kernel density (SD {SPAN_KDE_SIGMA_DEG:g} deg) per unit on [0, its span], reflected at both ends. '
    'PkSpan_<All/TM>_Desc / _Omni / _Pair: '
    f'density curves compared by permuting units\' group labels ({N_DENSITY_PERMUTATIONS} permutations; T = sum n_g * '
    'integral of (group mean density - grand mean)^2; L1_distance between pairwise mean curves), spans by Kruskal-Wallis + '
    'pairwise Mann-Whitney U, Holm per test. The span grows with the number of spikes, so compare PeakBinSpikes_median too.',
    f'Laps = traversals of the main field (v30 field_traversals); laps with >= {MIN_SPIKES_PER_LAP} analysed spikes are kept. '
    'Onset_* / Offset_* = circular mean / MRL over kept laps of the first / last analysed spike of each lap. *_range = theta '
    f'trough range ({THETA_TROUGH_RANGE_DEG[0]:g}-{THETA_TROUGH_RANGE_DEG[1]:g} deg) or theta peak range.',
    'Statistics sheets: All_* = every unit pooled; Animal_ThetaMod_* / Animal_TMTested_* = the theta-modulated '
    'proportions compared across animals, arenas and session types pooled within each animal; Ses<arena>_*= session types within an arena; Arena_* = arena types, session '
    'types pooled; AC<arenas>_* = arena types, only the animals in the Animals column. ThetaMod_Counts: n / % of all units '
    'Rayleigh+, TMI+, both; ThetaMod_Omni / _Pair: chi-square + pairwise chi-square (Yates), Holm, on Rayleigh+ & TMI+ only. '
    f'TMTested_Omni / _Pair: the same tests on the % of TESTED units (>= {MIN_SPIKES_FOR_TMI} theta-epoch spikes) that are '
    'Rayleigh+ & TMI+ (pct_of_tested_ThetaModulated in ThetaMod_Counts); groups with no tested unit left out. '
    'PkBin / Onset / Offset _All / _TM _Desc / _Omni / _Pair: phase of all units / theta-modulated units, Mardia-Watson-Wheeler '
    '(permutation p) and Watson-Williams, Holm per test. PkBin / Onset / Offset _TMvsNon: theta-modulated vs '
    'non-theta-modulated units within each group, two-sample Watson-Williams test, p_holm = Holm over the groups of that '
    f'sheet (non-theta-modulated includes untested units: {NON_TM_INCLUDES_UNTESTED}). Every test treats units as '
    'independent; a unit recorded in several sessions counts once per session.',
]


def finish_population(df: pd.DataFrame, excel_path: Path):
    """Everything after the per-unit step: the comparisons at every level, the summary workbook and the
    long-format statistics CSV. `df` = one row per unit (process_session)."""
    df = df.reindex(columns=UNIT_COLUMNS)
    summary_df = build_summary_stats(df)
    unknown_type = df.loc[df['Arena'].notna() & df['SessionType'].isna(), 'FolderPath'].unique()
    if len(unknown_type):
        print(f'\nWARNING: {len(unknown_type)} session folder(s) with no recognizable session type; '
              f'their units are left out of the session-type comparisons (kept in the arena comparisons):')
        for p in sorted(unknown_type):
            print(f'    {p}')
    all_tables: dict[str, pd.DataFrame] = {}

    # ---- Every unit pooled ----
    print('\n=== All units pooled ===')
    df_all = df.assign(All='All units')
    pooled_all = 'all arenas, session types and animals pooled'
    tables = run_theta_mod_proportion(df_all, 'All', ['All units'],
                                      excel_path.parent / 'AllUnits_ThetaModProportion.png', pooled_all)
    tables.update(run_tm_vs_non(df_all, 'All', ['All units'], excel_path.parent, 'AllUnits', pooled_all))
    tables.update(run_peakbin_span(df_all, 'All', ['All units'], excel_path.parent, 'AllUnits', pooled_all))
    all_tables.update({f'All_{k}': t for k, t in tables.items()})

    # ---- Animals: % theta-modulated units per animal, arenas and session types pooled ----
    print('\n=== Animal comparison (all arenas and session types pooled within each animal) ===')
    df_animal = df.loc[df['Animal'].notna()]
    present = set(df_animal['Animal'])
    animal_order = ([a for a in ANIMAL_COLORS if a in present]
                    + sorted(present - set(ANIMAL_COLORS), key=str))
    animal_dir = excel_path.parent / ANIMAL_DIR
    animal_dir.mkdir(parents=True, exist_ok=True)
    print(pd.crosstab(df_animal['Animal'], df_animal['Arena'], margins=True))
    tables = run_theta_mod_proportion(df_animal, 'Animal', animal_order,
                                      animal_dir / 'Animal_ThetaModProportion.png',
                                      'arenas and session types pooled within each animal')
    all_tables.update({f'Animal_{k}': t for k, t in tables.items()})

    # ---- 1) Session types within each arena ----
    for arena in ARENA_LABELS:
        df_arena = df.loc[(df['Arena'] == arena) & df['SessionType'].isin(ARENA_SESSION_TYPES[arena])]
        if df_arena.empty:
            continue
        print(f'\n=== {arena} arena: session-type comparison '
              f'({", ".join(_label(t) for t in ARENA_SESSION_TYPES[arena])}) ===')
        tables = run_group_comparisons(df_arena, 'SessionType', ARENA_SESSION_TYPES[arena],
                                       excel_path.parent / SESSION_TYPE_DIR / arena, f'SessionType_{arena}',
                                       'session type', title_prefix=f'{arena} arena: ')
        all_tables.update({f'Ses{arena}_{k}': t for k, t in tables.items()})

    # ---- 2) Arena types, session types pooled ----
    print('\n=== Arena comparison (all session types pooled within each arena) ===')
    tables = run_group_comparisons(df.loc[df['Arena'].notna()], 'Arena', ARENA_LABELS,
                                   excel_path.parent / ARENA_TYPE_DIR, 'ArenaType', 'arena')
    all_tables.update({f'Arena_{k}': t for k, t in tables.items()})

    # ---- 3) Arena types, only the animals recorded in all compared arenas ----
    all_tables.update(run_animal_controlled(df, excel_path.parent / ANIMAL_CONTROLLED_DIR))

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        pd.DataFrame({'Note': README_NOTES}).to_excel(writer, sheet_name='README', index=False)
        df.to_excel(writer, sheet_name='Units', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        for sheet_name, table in all_tables.items():
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    print(f'\nDone. {len(df)} unit(s) processed. Results saved to {excel_path}')

    # 'table' names the source sheet; columns that don't apply to a table are left blank.
    parts = [t.assign(table=name)[['table', *t.columns]] for name, t in all_tables.items() if len(t)]
    if parts:
        csv_path = excel_path.parent / STATS_CSV_NAME
        pd.concat(parts, ignore_index=True).to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f'All statistics also saved to {csv_path}')

    metadata_path = save_metadata_csv(excel_path.with_name(excel_path.stem + '_metadata.csv'))
    print(f'Run metadata saved to {metadata_path}')


def main():
    if 360 % PHASE_BIN_SIZE_DEG != 0:
        raise ValueError('PHASE_BIN_SIZE_DEG must divide 360 evenly.')

    rng = np.random.default_rng(RANDOM_SEED)
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    session_folders = find_session_folders(ROOT_FOLDER)
    if not session_folders:
        raise FileNotFoundError(f'No folders with both .ncs and .ntt files found under {ROOT_FOLDER}')

    all_rows = []
    acg_records: list[dict] = []
    for data_folder in session_folders:
        print(f'\n=== Session: {data_folder} ===')
        try:
            all_rows.extend(process_session(data_folder, rng, acg_records))
        except Exception as exc:
            print(f'ERROR processing {data_folder}: {exc}')
            continue

    # ---- ACG QC: per-epoch table + freq_est vs. peak range plot, all sessions pooled ----
    if acg_records:
        acg_df = pd.DataFrame(acg_records)
        acg_csv = RESULTS_FOLDER / ACG_EPOCHS_CSV_NAME
        acg_df.to_csv(acg_csv, index=False)
        print(f'\nACG per-epoch table saved to {acg_csv}')
        acg_png = plot_acg_epochs(acg_df, RESULTS_FOLDER / ACG_QC_PLOT_NAME)
        if acg_png is not None:
            print(f'ACG frequency vs. peak range plot saved to {acg_png}')

    finish_population(pd.DataFrame(all_rows), RESULTS_FOLDER / OUTPUT_EXCEL_NAME)


if __name__ == '__main__':
    main()
