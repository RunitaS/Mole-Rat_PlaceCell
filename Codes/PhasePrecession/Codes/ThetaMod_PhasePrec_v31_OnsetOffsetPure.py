# -*- coding: utf-8 -*-
"""
Theta modulation + phase precession from firing dynamics alone, with NO place field, pass
through a field, or any other spatial selectivity. Spike-train classification follows
Harris KD, Henze DA, Hirase H, Leinekugel X, Dragoi G, Czurko A, Buzsaki G (2002), Nature
417:738-741, "Spike train dynamics predicts theta-related phase precession in hippocampal
pyramidal cells". Run sequentially per unit (one .ntt file = one already-isolated unit):

  Step 1 (visualization only): polar plot of spike counts vs. theta phase, phase estimated by
    direct Hilbert transform of the cleaned (clean_lfp: resampled to 1 kHz, 50/100/150/200 Hz
    notched, detrended) and bandpass-filtered LFP (0 deg at a peak, as Harris et al.).
  Step 2: Theta Modulation Index (TMI; Frank et al. 2001) with its burst-preserving shuffle test,
    and the Rayleigh test (column ThetaModulated). Reported only: by default it does NOT gate
    Step 3 (REQUIRE_THETA_MOD_FOR_FIT), because a cell sweeping 180 deg of theta has a broad phase
    histogram and a theta-modulation gate would remove exactly the cells this analysis looks for.
  Step 3: spike-train dynamics of the unit (spike_train_dynamics, analyse_firing), then:
    a. Instantaneous rate (Harris Fig 1d): spikes within +/-360 deg of theta of each spike, the
       spike itself included ('No. of spikes in 2 cycles'). RatePhase_ANOVA_p: circular ANOVA of
       phase across rate groups (statistic sum cos(theta_i - mean_g(i)), within-unit rotation null).
    b. Onset / offset spikes (Harris Fig 2): a spike is ONSET if the previous ONOFF_WINDOW_CYCLES
       theta cycles hold <= ONOFF_LOW_MAX_SPIKES spikes and the following ones >=
       ONOFF_HIGH_MIN_SPIKES, OFFSET the other way round. Columns OnOff_*; Mardia-Watson-Wheeler
       test against the within-unit rotation null.
    c. Firing episodes and firing index: an episode runs from the first spike of an onset block to
       the last spike of the offset block that follows it. Firing index = -1 at that onset, +1 at
       that offset, linear in theta cycles (unwrapped LFP phase), replacing the pass index of v30.
       Phase vs firing index gets the same circular-linear fit as before (Kempter et al. 2012;
       fit_R gate against the within-unit rotation null, z-test on burst events), so a negative slope means
       phase advances from firing onset to offset. Harris et al. did not fit a line; it is added
       because onset and offset alone cannot give the direction of a shift close to 180 deg.
    d. Rate and rate derivative (Harris Fig 3): regression of spikes per cycle on cycle number over
       the central cycle and RATE_DERIV_HALF_WINDOW_CYCLES on each side (intercept = rate, slope =
       derivative).

  Population (all units; run_onoff_population, 'OnOff_*' sheets, OnOff_*.png in RESULTS_FOLDER):
    the same quantities pooled across units as in the paper, by arena and animal, plus:
    * Within-unit tests, on phase relative to each unit's preferred phase. Pooled spikes come from
      cells with different preferred phases and bursts, so a pooled test alone can reflect which
      cells contribute which spikes. Harris' own pooled, free-label test is reported alongside.
    * Every phase-vs-dynamics p comes from rotating each unit's phase sequence against its spike train
      (phase_rotator). Phases drift over seconds and onset / offset / high-rate spikes cluster in time,
      so shuffling spikes, blocks or bursts breaks structure the data really has: with the LFP time-
      shifted against the spikes (no true relation), those tests gave p < 0.05 in 10-100% of 40 shifts.
    * Position used ONLY as a control, never to classify. Ignoring position does not decouple firing
      from it: a place cell's firing onset still tends to happen at field entry. So everything is
      repeated on spikes / episodes during which the animal moved < ONOFF_STATIONARY_PATH_CM (Harris
      Fig 1c: head stationary < 3 cm), and with two tests that keep more data, since these animals are rarely
      still for ~3.4 s: onset vs offset compared only within one unit in one ONOFF_SAME_PLACE_BIN_CM
      bin (phase centred per bin), and episodes whose onset and offset lie < ONOFF_SAME_PLACE_CM apart.
      If precession is carried by position it should vanish in these subsets.

Deliberate differences from Harris et al., each because of how this data differs or to remove an artefact:
  * Windows are centred on each spike's own phase rather than aligned to fixed LFP cycles. With
    fixed cycles a late-phase spike's burst partner lands in the next cycle and an early-phase spike's
    in the previous one, which produces a phase-vs-derivative relation in the direction under test.
  * Spike-count thresholds are kept in spikes per cycle (the SDI model's unit), not Hz: at this data's
    ~4.7 Hz theta, 8 cycles last ~1.7 s and <= 4 -> >= 16 spikes means about 2.3 -> 9.4 Hz (not 4 -> 16).
  * Only the classified spike must lie in an ACG theta-positive epoch; its window counts all spikes. These
    epochs often last 1-2 s, shorter than the +/-8 cycle window.
  * SLOPE_BNDS is symmetric. v30's (-1, 0.25) searched four times further on the precessing side.

ROOT_FOLDER is searched recursively; every folder that directly contains at least one .ncs and at
least one .ntt file is treated as a session. Data layout expected per session folder:
    *.ncs   Neuralynx continuous (LFP) file. MATCH_SPIKE_LFP_FILE selects per-tetrode matching
            (match_ncs_to_ntt) or the session's first .ncs file for every unit.
    *.ntt   Neuralynx tetrode spike files, one file per already-isolated unit.
    tracking .csv file, auto-detected as the first .csv in the folder (col A = timestamp, cols D, E
            = x, y in cm). Used only for the stationary control; a session without it runs everything
            else. Its timestamps must be on the Neuralynx clock (TRACKING_TIME_UNIT = their units).

If cell sessions are laid out as <root>/<animal>/<arena>/DayN/<session>, with <arena> one of
ARENA_LABELS, significant fits are compared across session types within each arena and then across
arenas on r^2, fit_R, |slope| and phase range (Kruskal-Wallis + Holm-corrected Mann-Whitney U), TMI
and theta-modulated proportions as before, and the fitted phase at firing index -1 / 0 / +1 and
offset - onset phase with circular tests (Mardia-Watson-Wheeler, Watson-Williams).

Output: <session>/ThetaMod_PhasePrecession_v31_OnsetOffset/ per-unit figures; RESULTS_FOLDER: summary
workbook (one row per unit), population figures, per-class plot copies, and OnOff_spikes.csv.gz (per-spike
table, so the population step can be rerun without recomputing the units). v30 and earlier hold the
place-field, traversal and pass-index code that this version removes.

Requires: numpy, scipy, pandas, matplotlib, openpyxl (for writing .xlsx).
"""

#!Correct phase slope testing! Most precessing/recessing cells classified as phase locked.
#!Criteria is too strict.

from __future__ import annotations

import functools
import re
import shutil
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
from scipy import signal, stats
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import minimize_scalar
from scipy.special import erf



# ============================================================================
# Configuration -- EDIT THESE
# ============================================================================
ROOT_FOLDER = Path(r"X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final")
# Theta-modulation criterion of Step 2 (column ThetaModulated).
# False: TMI shuffle test only (p < ALPHA); True: TMI shuffle test AND Rayleigh test (both p < ALPHA).
REQUIRE_RAYLEIGH_FOR_THETA_MOD = True
# True: Step 3's per-unit firing-index fit only for ThetaModulated units, as in v30. False: for all units,
# as Harris et al. (every pyramidal cell); ThetaModulated is a subgroup of the population tables either way.
REQUIRE_THETA_MOD_FOR_FIT = True
# Output names follow the toggles, so one setting's run never overwrites another's results
_THETA_TAG = '_TMIandRayleigh' if REQUIRE_RAYLEIGH_FOR_THETA_MOD else ''
_FIT_TAG = '_ThetaModOnly' if REQUIRE_THETA_MOD_FOR_FIT else ''
# All pooled results (Excel/CSV files, comparison plots, per-class plot copies) go here
RESULTS_FOLDER = ROOT_FOLDER / 'Results' / f'PhasePrec_Results_v31_OnsetOffset{_THETA_TAG}{_FIT_TAG}'
OUTPUT_EXCEL_NAME = f'theta_phase_v31_OnsetOffset{_THETA_TAG}{_FIT_TAG}.xlsx'   # written to RESULTS_FOLDER
SESSION_OUTPUT_FOLDER = f'ThetaMod_PhasePrecession_v31_OnsetOffset{_THETA_TAG}{_FIT_TAG}'   # in each session folder
SPIKE_TABLE_NAME = 'OnOff_spikes.csv.gz'    # per-spike table for the population step, in RESULTS_FOLDER
# PrecessionClass -> folder (in RESULTS_FOLDER, beside the Excel file) receiving a copy of each cell's plot
# (no_phase_relation cells share the PhaseLocked folder)
CLASS_PLOT_FOLDERS = {
    'phase_precessing': 'PhasePrecessing_Plots_FitRShuffle',
    'phase_succeeding': 'PhaseSucceeding_Plots_FitRShuffle',
    'phase_locked': 'PhaseLocked_Plots_FitRShuffle',
    'no_phase_relation': 'PhaseLocked_Plots_FitRShuffle',
}
# fit_R_Significant -> subfolder of each CLASS_PLOT_FOLDERS folder (used only when the fit_R shuffle was run)
FIT_R_SUBFOLDERS = {True: 'fit R pass', False: 'fit R fail'}
# Fits whose wrapped phase line has > MAX_FIT_LINES segments (biologically implausible)
MULTILINES_FOLDER = 'MultiLinesFit'

TRACKING_TIME_UNIT = 'us'     # 'us', 'ms', or 's' -- units of the tracking timestamp column

# --- Spike/LFP channel-matching toggle ---
# True: match each .ntt tetrode file to the .ncs LFP file sharing its embedded channel number (e.g.
# TT1.ntt -> CSC1.ncs), via match_ncs_to_ntt(). False: reference every tetrode in the session to the
# first .ncs file found (sorted naturally), loaded and filtered once for the whole session.
MATCH_SPIKE_LFP_FILE = False

# --- ACG theta-positive epoch criterion (Dunn et al. 2022; see acg_theta_epoch_mask) ---
# True: only spikes falling in 1 s LFP epochs whose autocorrelogram matches a 3-7 Hz sinusoid well
# (ED_min < ACG_ED_MIN_THRESH) are analysed. Spike-count windows still count all of the unit's spikes
# (see spike_train_dynamics). False: use the entire signal.
USE_ACG_THETA_EPOCHS = True
ACG_ED_MIN_THRESH = 1.0          # keep epochs with ED_min strictly below this
ACG_EPOCH_SEC = 1.0
ACG_FS = 1000.0                  # Hz, LFP is resampled to this rate (clean_lfp: Hilbert phase + ACG)
ACG_FREQ_RANGE = (3.0, 7.0)      # Hz, reference sinusoid bank
ACG_FREQ_RES = 0.1               # Hz, bank step
ACG_LINE_HARMONICS = [50.0, 100.0, 150.0, 200.0]   # mains notch (clean_lfp: Hilbert phase + ACG)
ACG_NOTCH_Q = 30.0
ACG_LOWPASS_CUTOFF_HZ = 100.0
ACG_LOWPASS_ORDER = 4
ACG_REJECT_DELTA_OVER_THETA = True   # drop epochs whose 1-3 Hz power exceeds 3-7 Hz power
ACG_DELTA_BAND = (1.0, 3.0)
ACG_REJECT_MAD_ARTIFACTS = True      # drop epochs with outlier peak-to-peak amplitude
ACG_MAD_THRESH = 5.0
ACG_PEAK_PROMINENCE = 0.05           # reference-ACG peak/trough prominence (findMinMax(refxc, 0.05, 'fixed'));
                                     # only used for the ED_min QC plot's normalised peak range
# ED_min QC plot (ED_min vs. normalised ACG peak range of every fitted epoch, coloured by animal,
# ED_min-rejected epochs highlighted) and its per-epoch table, both written to RESULTS_FOLDER
EDMIN_PLOT_NAME = 'ACG_EDmin_vs_PeakRange.png'
EDMIN_EPOCHS_CSV_NAME = 'ACG_EDmin_epochs.csv'
# Per-animal colors — first 4 colors of matplotlib's Pastel1 palette.
ANIMAL_COLORS = {
    'Fa8477': '#FBB4AE',  # Pastel1[0] red
    'Fa1059': '#B3CDE3',  # Pastel1[1] blue
    'Fa23BD': '#CCEBC5',  # Pastel1[2] green
    'Fa5834': '#DECBE4',  # Pastel1[3] purple
}

# --- Step 3: spike-train dynamics (Harris et al. 2002) ---
LFP_FILTER_BAND = (3.0, 7.0)  # Hz, theta band used for both the theta-modulation test and LFP phase
# Onset spike: previous ONOFF_WINDOW_CYCLES cycles hold <= ONOFF_LOW_MAX_SPIKES spikes and the following ones
# >= ONOFF_HIGH_MIN_SPIKES; offset spike: the reverse. Counts exclude the spike itself. Harris et al.: 8, 4, 16.
ONOFF_WINDOW_CYCLES = 8
ONOFF_LOW_MAX_SPIKES = 4
ONOFF_HIGH_MIN_SPIKES = 16
# Other (low, high) thresholds at which the population tests are repeated. The primary result stays Harris' own;
# these exist because this data fires sparsely, and were fixed before any phase was looked at.
ONOFF_SENSITIVITY_THRESHOLDS = ((4, 12), (4, 8), (2, 8))
ONOFF_MIN_SPIKES_PER_UNIT = 5     # onset AND offset spikes a unit needs for its own offset - onset / MWW test
ONOFF_N_PERMUTATIONS = 1000       # circular ANOVA nulls (Harris et al.: 1000) and onset / offset block permutations
ONOFF_MAX_LFP_GAP_S = 0.1         # a spike whose window spans a longer LFP gap has no valid spike counts
INST_RATE_WINDOW_CYCLES = 1       # instantaneous rate: spikes within +/- this many cycles (spike included)
INST_RATE_MAX_GROUP = 10          # Fig 1d rate groups 1, 2, ... this and above pooled (Harris et al.: 10)
RATE_DERIV_HALF_WINDOW_CYCLES = 3 # rate / derivative regression over 2 * this + 1 cycles (Harris et al.: 3)
# Harris Fig 3 regions (spikes per cycle, spikes per cycle^2)
RATE_LOW_MAX = 0.25
RATE_HIGH_MIN = 1.0
RATE_DERIV_MIN = 1.0
RATE_DERIV_MIN_RELAXED = 0.25     # second accelerating / decelerating split: few spikes of this sparse data reach +/-1
FIRING_EPISODE_INDEX_BINS = 8     # bins of firing index for the model-free phase vs firing index table / plot
N_POOLED_FIT_SHUFFLES = 200       # population fit: within-unit phase rotations (phase_rotator)
ROTATION_MIN_FRAC = 0.1           # phase_rotator: each unit's phases move by 10-90% of its spikes
N_POOLED_FIT_BOOTSTRAPS = 200     # population fit: units resampled with replacement, for the slope's 95% CI
# --- Position, as a control only (never used to classify a spike or define an episode) ---
ONOFF_STATIONARY_PATH_CM = 3.0    # stationary = moved less than this over the window / episode (Harris Fig 1c: < 3 cm)
ONOFF_SAME_PLACE_BIN_CM = 10.0    # same-place test: onset / offset labels permuted only among blocks of one unit in one bin this wide
ONOFF_SAME_PLACE_CM = 5.0         # same-place episode: firing onset and offset less than this far apart (net displacement)
POS_SMOOTH_SEC = 0.2              # Gaussian sigma (s) applied to x/y before path length and speed;
                                  # removes frame-to-frame tracking jitter (~20% of raw arc length in this data)
TRACKING_IMMOBILITY_SPEED_CM_S = 0.5  # path length accumulates only at or above this smoothed speed

# --- Step 3: circular-linear fit of phase vs firing index ---
BURST_COLLAPSE_FOR_ZTEST = True  # Kempter z-test on one event per burst (spikes < TMI_BURST_MAX_GAP_SEC apart:
                              # circular-mean phase, mean firing index) so bursts are not counted as independent
SLOPE_BNDS = (-1.0, 1.0)      # (low, high) bound on slope (cycles/unit); x spans 2 units per episode, so this is
                              # +/-720 deg/episode. Symmetric, because the slope's sign is the question here
MIN_SPIKES_FOR_FIT = 30       # skip circular-linear fit if fewer spikes than this
SLOPE_THRESH_DEG_PER_EPISODE = 15.0   # significant fits need |slope| (deg/episode) above this to be
                              # precessing (negative) / succeeding (positive); else phase locked
MAX_FIT_LINES = 3            # a fit whose wrapped phase line has more segments than this is
                              # marked non-precessing and gets no further analysis
# --- Step 3 gate: fit_R shuffle test (see shuffle_fit_r_significance) ---
# Before the Kempter z-test is used to classify a cell, its fit_R must beat a null in
# which the phase-firing index pairing is broken: whole bursts (spikes < TMI_BURST_MAX_GAP_SEC
# apart, as in the TMI shuffle but with no phase-bin condition) swap phases with each other,
# keeping the cell's phase distribution (its theta locking) and burst structure intact.
# Cells failing the gate are classed no_phase_relation.
USE_FIT_R_GATE = True
N_FIT_R_SHUFFLES = 1000             # surrogates (v31: within-unit phase rotations); each refits anglereg

# --- Step 1/2: theta phase-locking / modulation parameters ---
PHASE_BIN_SIZE_DEG = 36             # degrees per polar-histogram bin (360 must be divisible by this);
                                     # matches calc_tmi's bin_width_deg for consistency
ALPHA = 0.05                        # Rayleigh-test / TMI-shuffle significance threshold
MIN_SPIKES_FOR_TMI = 50              # min spikes required to run the polar-plot / TMI battery
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

# --- Arena comparison (Circle/Linear/Open) parameters ---
# Data layout: ROOT_FOLDER/<animal>/<arena>/DayN/<session>, e.g.
# .../Fa23BD/Open/Day1/1Cntrl -- the arena is read off whichever path
# component (case-insensitively) matches one of these labels.
ARENA_LABELS = ('Circle', 'Linear', 'Open')

# --- Session-type comparison within each arena ---
# Session type is read off the session folder name (the last path component), as
# in SessionType_StatsComparison_v6.py (see detect_session_type), e.g. '1_Cntrl' ->
# 'Cntrl', '3_180' -> '180', '2NoRot' -> 'NoRot'. Listed in plotting order.
ARENA_SESSION_TYPES = {
    'Open': ('Cntrl', 'Rotate', 'Zero'),
    'Linear': ('0', '90', '180', '270'),
    'Circle': ('Stnd', 'NoRot', 'Rot'),
}
SESSION_TYPE_PLOT_FOLDER = 'SessionTypeComparison'   # in RESULTS_FOLDER, one sub-folder per arena

# --- Theta phase at firing index 0 (episode midpoint), compared across arenas ---
# Phase at firing index 0 = the fitted line 2*pi*s*x + b evaluated at x = 0, i.e. the
# intercept b, wrapped to [0, 360) deg. On the circle the precession range is
# [70, 250) deg -- 70-250 and 430-610 deg on the double-cycle 0-720 plot axis; the
# rest, 250-430 deg (= 250-360 + 0-70) and 610-720 deg, is the procession range.
FI0_PRECESSION_RANGE_DEG = (70.0, 250.0)
FI0_PHASE_CLASSES = ('phase_precessing', 'phase_succeeding', 'phase_locked', 'no_phase_relation')  # cells included
N_FI0_PERMUTATIONS = 10000          # label permutations for the Mardia-Watson-Wheeler test
FI0_RANGE_COLORS = {'precession': '#8C2981', 'procession': '#FCFDBF'}   # magma light purple (0.4) / light yellow (1.0)
# Firing-index points at which the fitted line 2*pi*s*x + b is evaluated, each extracted,
# plotted and compared across arenas exactly as for firing index 0 (same cells, same
# precession/procession ranges, same circular tests):
# (firing index x, column key -> phase_at_<key>_deg, plot label, file tag, sheet tag)
FIRING_INDEX_PHASE_POINTS = (
    (-1.0, 'fim1', 'firing onset (firing index -1)', 'FiringIndexMinus1', 'FIm1'),
    (0.0, 'fi0', 'episode midpoint (firing index 0)', 'FiringIndex0', 'FI0'),
    (1.0, 'fip1', 'firing offset (firing index +1)', 'FiringIndexPlus1', 'FIp1'),
)
ONSET, OFFSET = 1, -1               # spike labels of analyse_firing (0 = neither)
FIRING_ONSET_COLOR = 'tab:red'      # onset spikes, as Harris Fig 2
FIRING_OFFSET_COLOR = 'tab:blue'    # offset spikes, as Harris Fig 2


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
    """Load a Neuralynx .ncs file. Returns (samples_uV, timestamps_s, fs_hz).

    Only each record's first NumValidSamples samples are kept: the rest of a partial
    record holds stale buffer values, and the next record's timestamp starts right after
    the valid ones. Each sample's time is its record's absolute (PTP-synced) timestamp
    plus its offset within the record.
    """
    header_text = _read_header_text(path)
    adbitvolts = _header_field(header_text, 'ADBitVolts', default=DEFAULT_ADBITVOLTS)

    ncs_dtype = np.dtype([
        ('TimeStamp', '<u8'),
        ('ChannelNumber', '<u4'),
        ('SampleFreq', '<u4'),
        ('NumValidSamples', '<u4'),
        ('Samples', '<i2', (NCS_SAMPLES_PER_RECORD,)),
    ])
    records = np.fromfile(path, dtype=ncs_dtype, offset=HEADER_BYTES)
    if len(records) == 0:
        raise ValueError(f'No records found in {path}')

    fs = float(records['SampleFreq'][0])
    if fs <= 0:
        fs = _header_field(header_text, 'SamplingFrequency', default=32000.0)

    sample_period_us = 1e6 / fs
    offsets_us = np.arange(NCS_SAMPLES_PER_RECORD) * sample_period_us
    valid = np.arange(NCS_SAMPLES_PER_RECORD)[None, :] < records['NumValidSamples'][:, None]
    timestamps_us = (records['TimeStamp'][:, None].astype(np.float64) + offsets_us[None, :])[valid]
    samples_uV = (records['Samples'].astype(np.float64) * adbitvolts * 1e6)[valid]

    return samples_uV, timestamps_us / 1e6, fs


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
# Position and LFP helpers
# ============================================================================

def spk_pos(pos_ts, pos_xy, spk_ts):
    """Nearest tracked position at each spike time."""
    idx = np.searchsorted(pos_ts, spk_ts)
    idx = np.clip(idx, 1, len(pos_ts) - 1)
    left, right = idx - 1, idx
    use_left = np.abs(spk_ts - pos_ts[left]) <= np.abs(pos_ts[right] - spk_ts)
    idx = np.where(use_left, left, right)
    return pos_xy[idx], idx


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
    degenerate auto-selected band (e.g. from a very sparse/diffuse field
    estimate) raises a clear error instead of a cryptic scipy exception.
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
# ACG theta-positive epoch detection (Dunn et al. 2022, Nat Commun 13:6997,
# Supp. Fig. 5; ported from ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean_v3.py).
#
# The LFP is resampled to ACG_FS, line-noise notched, detrended (clean_lfp, shared
# with the Hilbert phase path) and low-passed,
# split into ACG_EPOCH_SEC epochs, and each epoch's autocorrelogram is compared
# with a bank of reference-sinusoid autocorrelograms spanning ACG_FREQ_RANGE.
# ED_min = the normalised Euclidean distance to the closest reference sinusoid
# (0 = perfect sinusoid). An epoch is "theta-positive" if ED_min < ACG_ED_MIN_THRESH.
# Optionally (same as the ACG script) epochs are first rejected if delta power
# exceeds theta power or if their peak-to-peak amplitude is a robust MAD outlier.
# ============================================================================

def _acg_notch(x, fs_hz, freqs, q):
    y = np.asarray(x, dtype=np.float64)
    for f0 in freqs:
        if 0 < f0 < fs_hz / 2.0:
            b, a = signal.iirnotch(f0, q, fs_hz)
            y = signal.filtfilt(b, a, y)
    return y


def clean_lfp(lfp_sig, lfp_ts, lfp_fs):
    """ACG-pipeline LFP cleaning, applied before both the Hilbert phase and the
    ACG epoch detection: resample to ACG_FS, notch the ACG_LINE_HARMONICS mains
    harmonics, remove the linear trend. Returns (x, ts, ACG_FS); ts gives each
    resampled sample's time, interpolated from the original sample timestamps
    so the recording's own clock is kept."""
    up, down = int(ACG_FS), int(round(lfp_fs))
    x = signal.resample_poly(lfp_sig, up, down)
    x = _acg_notch(x, ACG_FS, ACG_LINE_HARMONICS, ACG_NOTCH_Q)
    x = signal.detrend(x, type='linear')
    ts = np.interp(np.arange(len(x)) * (down / up), np.arange(len(lfp_ts)), lfp_ts)
    return x, ts, ACG_FS


def _acg_reference_bank(ref_freqs, datsize, fs):
    """Normalised autocorrelogram of each reference sinusoid -> (refXC, refED)."""
    reft = np.arange(datsize) / fs
    refXC = np.empty((2 * datsize - 1, len(ref_freqs)))
    for n, freq in enumerate(ref_freqs):
        refsig = np.sin(2 * np.pi * freq * reft)
        xc = signal.correlate(refsig, refsig, mode='full', method='auto')
        refXC[:, n] = xc / np.max(xc)
    return refXC, np.linalg.norm(refXC, axis=0)


def _zero_crossings(x):
    """Fractional (linearly interpolated) sample indices where x crosses zero."""
    s = np.sign(x)
    s[s == 0] = 1.0
    idx = np.flatnonzero(np.diff(s) != 0)
    return idx - x[idx] / (x[idx + 1] - x[idx])


def _acg_reference_windows(refXC, datsize):
    """Peak-range search windows of each reference autocorrelogram, as in
    create_sine_ref_xcorrs.m (ported from
    ACG_theta_continuity_TT_ThetaVsSpeed_v4_FerretOLS.py).

    Returns (refRange, p1, t1, t2): refRange is each reference's own peak range
    (first side peak minus mean of its flanking troughs); p1/t1/t2 are (2, n_freqs)
    [lo, hi] sample-index windows bracketing that side peak and the troughs before
    and after it, used to find the matching peak/troughs in a data autocorrelogram.
    """
    center = datsize - 1
    n_freqs = refXC.shape[1]
    refRange = np.empty(n_freqs)
    p1, t1, t2 = (np.empty((2, n_freqs), dtype=int) for _ in range(3))
    for n in range(n_freqs):
        refxc = refXC[:, n]
        max_idx, _ = signal.find_peaks(refxc, prominence=ACG_PEAK_PROMINENCE)
        min_idx, _ = signal.find_peaks(-refxc, prominence=ACG_PEAK_PROMINENCE)
        peak1 = max_idx[max_idx > center][0]
        trough1 = min_idx[min_idx < peak1][-1]
        trough2 = min_idx[min_idx > peak1][0]
        refRange[n] = refxc[peak1] - (refxc[trough1] + refxc[trough2]) / 2

        zc = _zero_crossings(refxc)
        below_p1 = int(round(zc[zc < peak1][-1]))
        above_p1 = int(round(zc[zc > peak1][0]))
        p1[:, n] = [below_p1, above_p1]
        t1[:, n] = [int(round(zc[zc < trough1][-1])), below_p1]
        t2[:, n] = [above_p1, int(round(zc[zc > trough2][0]))]
    return refRange, p1, t1, t2


def acg_theta_epoch_mask(lfp_clean, return_peakrange=False):
    """Per-epoch theta-positive mask for one LFP trace, already cleaned by
    clean_lfp (resampled to ACG_FS, notched, detrended).

    Epoch i is the i-th block of ACG_EPOCH_SEC * ACG_FS consecutive samples; use
    times_in_ok_epochs to match absolute times to epochs by the samples' timestamps.
    Returns (ok, ed_min): boolean mask and ED_min
    per epoch (NaN where the epoch was rejected before/without an ACG fit).
    With return_peakrange=True, returns (ok, ed_min, peakrangenorm), adding each
    fitted epoch's autocorrelogram peak range normalised by its matched reference
    sinusoid's own peak range (Dunn et al. 2022 Supp. Fig. 5; 1 = clean sinusoid,
    0 = flat) -- used only for the ED_min QC plot, the mask is unchanged.
    """
    x = lfp_clean
    b, a = signal.butter(ACG_LOWPASS_ORDER, ACG_LOWPASS_CUTOFF_HZ / (ACG_FS / 2.0), btype='low')
    x = signal.filtfilt(b, a, x)

    nperseg = int(round(ACG_FS * ACG_EPOCH_SEC))
    n_total = len(x) // nperseg
    if n_total == 0:
        empty = (np.zeros(0, dtype=bool), np.zeros(0))
        return (*empty, np.zeros(0)) if return_peakrange else empty
    epochs = x[:n_total * nperseg].reshape(n_total, nperseg)

    keep = np.ones(n_total, dtype=bool)
    if ACG_REJECT_DELTA_OVER_THETA:
        win = signal.get_window('hann', nperseg)
        for i in range(n_total):
            f, pxx = signal.welch(epochs[i], fs=ACG_FS, window=win, nperseg=nperseg,
                                  noverlap=0, detrend='constant')
            low = pxx[(f >= ACG_DELTA_BAND[0]) & (f <= ACG_DELTA_BAND[1])].sum()
            theta = pxx[(f >= ACG_FREQ_RANGE[0]) & (f <= ACG_FREQ_RANGE[1])].sum()
            keep[i] = low <= theta
    if ACG_REJECT_MAD_ARTIFACTS and keep.any():
        p2p = epochs.max(axis=1) - epochs.min(axis=1)
        med = np.median(p2p[keep])
        mad = np.median(np.abs(p2p[keep] - med)) or 1e-20
        keep &= ~(0.6745 * (p2p - med) / mad > ACG_MAD_THRESH)

    ref_freqs = np.arange(ACG_FREQ_RANGE[0], ACG_FREQ_RANGE[1] + ACG_FREQ_RES / 2, ACG_FREQ_RES)
    refXC, refED = _acg_reference_bank(ref_freqs, nperseg, ACG_FS)
    if return_peakrange:
        refRange, p1, t1, t2 = _acg_reference_windows(refXC, nperseg)

    ed_min = np.full(n_total, np.nan)
    peakrangenorm = np.full(n_total, np.nan)
    for i in np.flatnonzero(keep):
        xc = signal.correlate(epochs[i], epochs[i], mode='full', method='auto')
        peak = np.max(xc)
        if not np.isfinite(peak) or peak <= 0:
            continue
        xc = xc / peak
        ed = np.sqrt(np.sum((refXC - xc[:, None]) ** 2, axis=0)) / refED
        mi = int(np.argmin(ed))
        ed_min[i] = ed[mi]
        if return_peakrange:
            pk = xc[p1[0, mi]:p1[1, mi] + 1].max()
            tr1 = xc[t1[0, mi]:t1[1, mi] + 1].min()
            tr2 = xc[t2[0, mi]:t2[1, mi] + 1].min()
            peakrangenorm[i] = (pk - (tr1 + tr2) / 2) / refRange[mi]

    ok = np.isfinite(ed_min) & (ed_min < ACG_ED_MIN_THRESH)
    return (ok, ed_min, peakrangenorm) if return_peakrange else (ok, ed_min)


def times_in_ok_epochs(t_s, lfp_ts, lfp_fs, epoch_ok):
    """Boolean mask over absolute times `t_s` (s, on the same PTP-synced clock as
    `lfp_ts`) that fall inside a theta-positive epoch. Epoch i is the i-th block of
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
# Linear-circular regression (Kempter et al. 2012 / anglereg.m)
# ============================================================================

def anglereg(x: np.ndarray, theta: np.ndarray, bnds=None):
    """Linear-circular regression (Kempter et al. 2012). Returns
    (slope_cycles_per_unit, intercept_rad, fit_R).

    Finds the slope s (cycles per unit x) and intercept b maximizing the
    mean resultant length of theta (not to be confusd with hippocampal theta, 
    this is the angle theta) - 2*pi*s*x. fit_R is that maximized mean
    resultant length of the residuals, R(s*) in [0, 1]: the goodness of fit
    (1 = all spikes on the fitted line, 0 = residual phases uniformly spread).
    This objective is highly multimodal in s -- it has an aliased local 
    maximum roughly every 1/range(x) cycles -- so a coarse grid search over 
    the plausible slope range locates the correct basin before a local bounded 
    refinement.
    (The previous approach here tried only two Nelder-Mead starts, seeded
    from a naive linear-regression guess and its reciprocal-negative, both
    unbounded; empirically that missed the true optimum for anything but
    near-zero slopes, returning essentially arbitrary slope estimates for
    real steep precession -- see ThetaModPrecFixes.md.)

    bnds, if given, is (lo, hi) in cycles per unit x and is searched
    exhaustively via the same grid+refine strategy (not a single bounded
    local optimization, which is equally vulnerable to local optima within
    a wide bnds range).

    The grid search itself is vectorized over all grid points at once
    (chunked to bound peak memory) rather than calling neg_resultant in a
    Python-level loop, since this function runs once per real fit plus once
    per unit -- the unvectorized loop dominated Step 3's runtime.
    """
    theta = np.mod(theta, 2 * np.pi)
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    def neg_resultant(s):
        return -np.sqrt((np.sum(np.cos(theta - 2 * np.pi * s * x)) / n) ** 2 +
                        (np.sum(np.sin(theta - 2 * np.pi * s * x)) / n) ** 2)

    if bnds is None:
        x_range = np.ptp(x)
        if x_range <= 0:
            x_range = 1.0
        bnds = (-4.0 / x_range, 4.0 / x_range)
    lo, hi = bnds

    grid_n = max(500, int(200 * (hi - lo) * max(np.ptp(x), 1.0)))
    grid = np.linspace(lo, hi, grid_n)

    # costs[j] = neg_resultant(grid[j]) for every j at once, via an (n, chunk)
    # broadcast of phase residuals instead of grid_n separate Python calls.
    # Chunked over the grid axis so a large spike count doesn't blow up
    # memory (bounded to roughly 20M float64 elements, ~160MB, per chunk).
    max_chunk_elems = 20_000_000
    chunk_size = max(1, min(grid_n, max_chunk_elems // max(n, 1)))
    costs = np.empty(grid_n, dtype=np.float64)
    for start in range(0, grid_n, chunk_size):
        stop = min(start + chunk_size, grid_n)
        s_chunk = grid[start:stop]
        phase = theta[:, None] - 2 * np.pi * s_chunk[None, :] * x[:, None]
        cos_sum = np.cos(phase).sum(axis=0) / n
        sin_sum = np.sin(phase).sum(axis=0) / n
        costs[start:stop] = -np.sqrt(cos_sum ** 2 + sin_sum ** 2)

    s0 = grid[np.argmin(costs)]
    step = (hi - lo) / grid_n

    res = minimize_scalar(neg_resultant, bounds=(max(lo, s0 - 3 * step), min(hi, s0 + 3 * step)),
                           method='bounded', options={'xatol': 1e-10})
    s = float(res.x)

    b = np.arctan2(np.sum(np.sin(theta - 2 * np.pi * s * x)),
                   np.sum(np.cos(theta - 2 * np.pi * s * x)))
    fit_R = -neg_resultant(s)
    return s, b, fit_R

def kempter_lincirc(x, theta, s=None, b=None, slope_bnds=None):
    """Linear-circular correlation (Kempter et al. 2012). Returns (rho, p, s, b, fit_R),
    fit_R being the mean resultant length of the residuals about the fitted line
    (goodness of fit; see anglereg)."""
    x = np.asarray(x, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    good = ~np.isnan(x) & ~np.isnan(theta)
    x, theta = x[good], theta[good]

    if len(x) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    if s is None:
        s, b, fit_R = anglereg(x, theta, slope_bnds)
    else:
        fit_R = np.abs(np.mean(np.exp(1j * (theta - 2 * np.pi * s * x))))

    n = len(x)
    # Kempter et al. 2012 Eq. 3-4, as in CINPLA phase_precession.cl_corr(return_pval=True):
    # scale x by |s| (not s) so the sign of rho comes from the data, and centre each
    # variable on its own circular mean. (A previous version centred both on the fitted
    # line, 2*pi*s*mean(x) and +b; since s and b are fit to align the residuals, that
    # biased num > 0 even for independent data and gave ~35% false positives at p<0.05.)
    phi = np.mod(2 * np.pi * np.abs(s) * x, 2 * np.pi)
    theta_w = np.mod(theta, 2 * np.pi)

    def circ_mean(a):
        return np.arctan2(np.sum(np.sin(a)), np.sum(np.cos(a)))

    def rayleigh_p(a):
        Rn = np.hypot(np.sum(np.cos(a)), np.sum(np.sin(a)))
        return np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - Rn ** 2)) - (1 + 2 * n))

    sin_phi = np.sin(phi - circ_mean(phi))
    sin_theta = np.sin(theta_w - circ_mean(theta_w))
    den = np.sqrt(np.sum(sin_phi ** 2) * np.sum(sin_theta ** 2))
    if den == 0 or s == 0:
        return 0.0, 1.0, s, b, fit_R

    if rayleigh_p(phi) > 0.5 or rayleigh_p(theta_w) > 0.5:
        # Near-uniform variable -> circular mean ill-defined; use the centring-free
        # numerator R(phi - theta) - R(phi + theta) (Jammalamadaka & SenGupta 2001
        # Eq. 8.2.4; Kempter et al. 2012 p.122; CINPLA corr_cc_uniform).
        R_minus = np.abs(np.sum(np.exp(1j * (phi - theta_w))))
        R_plus = np.abs(np.sum(np.exp(1j * (phi + theta_w))))
        rho = (R_minus - R_plus) / (2 * den)
    else:
        rho = np.sum(sin_phi * sin_theta) / den

    l20, l02, l22 = np.mean(sin_phi ** 2), np.mean(sin_theta ** 2), np.mean(sin_phi ** 2 * sin_theta ** 2)
    z = rho * np.sqrt(n * l20 * l02 / l22) if l22 > 0 else 0.0
    p = 1 - erf(np.abs(z) / np.sqrt(2))

    return rho, p, s, b, fit_R


"""
Circular means when phases span a full cycle: the circular mean of a near-uniform distribution is
ill-defined, which can drive the Eq. 3 numerator toward zero for genuinely precessing cells.
Kempter et al. (2012, p.122) handle this by switching to the centring-free numerator
R(phi - theta) - R(phi + theta) (Jammalamadaka & SenGupta 2001, Eq. 8.2.4), which kempter_lincirc
does when a Rayleigh test finds either variable near-uniform (p > 0.5, as in CINPLA cl_corr).
Centring both variables on the fitted line (2*pi*s*mean(x), +b) instead is NOT valid: s and b are
fit to align the residuals, so the numerator is positive even for independent data and p-values
are badly anti-conservative (~35% false positives at p < 0.05 on pure noise).
"""

r"""
BUG FIX:
The surrounding statistical logic in your pipeline is mathematically sound:Asymptotic z-test: 
Your test statistic $z = \rho \sqrt{n \frac{\lambda_{20} \lambda_{02}}{\lambda_{22}}}$ and 
standard normal conversion $p = 1 - \text{erf}(\vert{}z\vert{} / \sqrt{2})$ are correct implementations 
of the asymptotic test.Optimization: The chunked grid-search followed by minimize_scalar in anglereg 
is a highly robust way to avoid the local minima that frequently trap standard gradient descent in 
circular-linear optimization.Slope Conversion: Multiplying by $4\pi$ (np.rad2deg(4 * np.pi * s)) correctly 
accounts for the fact that $x \in [-1, 1]$ means one episode equals 2 units, yielding $720s$ degrees per episode.
"""

# def kempter_lincirc(x, theta, s=None, b=None, slope_bnds=None):
#     """Linear-circular correlation (Kempter et al. 2012). Returns (rho, p, s, b)."""
#     x = np.asarray(x, dtype=np.float64)
#     theta = np.asarray(theta, dtype=np.float64)
#     good = ~np.isnan(x) & ~np.isnan(theta)
#     x, theta = x[good], theta[good]

#     if len(x) == 0:
#         return np.nan, np.nan, np.nan, np.nan

#     if s is None:
#         s, b = anglereg(x, theta, slope_bnds)

#     n = len(x)
#     # 2*pi factor: s is in cycles per unit x (matching anglereg's own
#     # optimizer, theta - 2*pi*s*x), so the fitted phase is 2*pi*s*x, not
#     # s*x -- this was previously omitted, which decoupled rho/p from the
#     # fitted slope for anything but near-zero s.
#     phi = np.mod(2 * np.pi * s * x, 2 * np.pi)
#     theta_w = np.mod(theta, 2 * np.pi)
#     phi_bar = np.angle(np.sum(np.exp(1j * phi)) / n)
#     theta_bar = np.angle(np.sum(np.exp(1j * theta_w)) / n)

#     num = np.sum(np.sin(theta_w - theta_bar) * np.sin(phi - phi_bar))
#     den = np.sqrt(np.sum(np.sin(theta_w - theta_bar) ** 2) * np.sum(np.sin(phi - phi_bar) ** 2))
#     rho = (np.abs(num / den) if den > 0 else 0.0) * np.sign(s)

#     def lam(i, j):
#         return np.sum((np.sin(phi - phi_bar) ** i) * (np.sin(theta_w - theta_bar) ** j)) / n

#     l20, l02, l22 = lam(2, 0), lam(0, 2), lam(2, 2)
#     z = rho * np.sqrt(n * l20 * l02 / l22) if l22 > 0 else 0.0
#     p = 1 - erf(np.abs(z) / np.sqrt(2))

#     return rho, p, s, b


# ============================================================================
# Step 3: phase-precession computation for one unit, on its firing index
# ============================================================================

def shuffle_fit_r_significance(spk_ts, spk_firing_index, spk_theta_phase, observed_fit_r, rng,
                               slope_bnds=None, n_shuffles=N_FIT_R_SHUFFLES, null_draw=None):
    """Null distribution for fit_R (max over slopes of the residual MRL, see anglereg)
    under 'theta phase is independent of firing index'.

    v31: if `null_draw` is given (null_draw(rng) -> one surrogate of spk_theta_phase), it
    supplies the surrogates instead: compute_firing_fit passes the within-unit phase rotation
    (phase_rotator), because the burst permutation below is anti-conservative on these data.

    Spikes are grouped into bursts by the TMI shuffle's time rule (consecutive spikes
    < TMI_BURST_MAX_GAP_SEC apart) but WITHOUT its same-phase-bin condition: a burst
    straddling a bin edge would otherwise be split into groups that are permuted
    independently although they share position and phase, which narrows the null
    (on simulated phase-locked, non-precessing bursty cells the phase-bin rule gave
    ~12% false positives at ALPHA = 0.05, the time-only rule ~5%). Each surrogate permutes the bursts' circular-mean phases across bursts, each spike keeping
    its offset from its own burst's mean, and leaves every spike's firing index in place.
    This keeps the cell's phase distribution (so its theta locking -- fit_R at slope 0 is
    just the phase MRL) and its burst structure, and destroys only the phase-position
    pairing. (The TMI shuffle's uniform random phases would also remove theta locking,
    making the null far too lenient for a TMI-significant cell.) Each surrogate is refit
    with anglereg over the same slope_bnds, so the slope search is part of the null.

    Returns
    -------
    pval          : fraction of shuffle fit_Rs >= observed_fit_r (add-one smoothed).
    shuffle_fit_r : ndarray of the n_shuffles shuffle fit_R values.
    """
    theta = np.mod(spk_theta_phase, 2 * np.pi)
    # phase_bin_deg=360 puts every phase in one bin, i.e. bursts are grouped by time only
    _, group_id = np.unique(_burst_groups(spk_ts, np.rad2deg(theta), phase_bin_deg=360.0),
                            return_inverse=True)
    n_groups = int(group_id.max()) + 1
    group_mean = np.arctan2(np.bincount(group_id, np.sin(theta), n_groups),
                            np.bincount(group_id, np.cos(theta), n_groups))
    offsets = theta - group_mean[group_id]

    shuffle_fit_r = np.empty(n_shuffles)
    for i in range(n_shuffles):
        theta_shuf = null_draw(rng) if null_draw is not None else group_mean[rng.permutation(n_groups)][group_id] + offsets
        shuffle_fit_r[i] = anglereg(spk_firing_index, theta_shuf, slope_bnds)[2]

    pval = float((np.sum(shuffle_fit_r >= observed_fit_r) + 1) / (n_shuffles + 1))
    return pval, shuffle_fit_r


class TooFewSpikes(Exception):
    """Too few spikes left for the circular-linear fit after the firing-index gates."""


def burst_events(spk_ts, firing_index, theta):
    """Collapse each burst (consecutive spikes < TMI_BURST_MAX_GAP_SEC apart, time rule
    only, as in shuffle_fit_r_significance) to one event: mean firing index (linear: -1 and +1
    are opposite ends of an episode, not neighbours across a wrap) and circular-mean theta
    phase. Returns (event_firing_index, event_theta)."""
    theta = np.mod(theta, 2 * np.pi)
    _, gid = np.unique(_burst_groups(spk_ts, np.rad2deg(theta), phase_bin_deg=360.0), return_inverse=True)
    n = int(gid.max()) + 1
    event_firing_index = np.bincount(gid, firing_index, n) / np.bincount(gid, minlength=n)
    return event_firing_index, np.arctan2(np.bincount(gid, np.sin(theta), n), np.bincount(gid, np.cos(theta), n))



def _precession_fit(spk_ts, spk_firing_index, spk_theta_phase, rng, slope_bnds,
                    occ_firing_index, occ_theta_phase, occ_dt, null_draw=None):
    """Circular-linear fit, fit_R shuffle gate, classification and density map for one
    set of analysed spikes (the theta-positive spikes of a unit's firing episodes).
    occ_firing_index / occ_theta_phase (rad) are the density-map occupancy's LFP samples,
    each worth occ_dt s. null_draw: surrogate phases for the fit_R gate (see
    shuffle_fit_r_significance). Returns a dict of the fit statistics, keyed as in
    compute_firing_index's result."""
    # Slope/intercept/fit_R from all analysed spikes; the asymptotic z-test (rho, p) on one
    # event per burst at that slope, since burst spikes share phase and firing index and are
    # not independent samples (counting them inflates z by ~sqrt(burst size)).
    rho, p, s, b, fit_R = kempter_lincirc(spk_firing_index, spk_theta_phase, slope_bnds=slope_bnds)
    n_events = len(spk_ts)
    if BURST_COLLAPSE_FOR_ZTEST and np.isfinite(s):
        ev_x, ev_theta = burst_events(spk_ts, spk_firing_index, spk_theta_phase)
        n_events = len(ev_x)
        rho, p, *_ = kempter_lincirc(ev_x, ev_theta, s=s, b=b)
    # s is cycles per unit of firing index; a full episode spans 2 units (-1 to
    # +1, firing onset to offset), so total phase change over one full episode is
    # 2*s cycles = rad2deg(2*(2*pi*s)) degrees -- previously this used
    # rad2deg(2*pi*s), i.e. degrees per 1 unit of firing index (half an episode),
    # under-reporting the true per-episode slope (and the classification bounds
    # below) by a factor of 2.
    slope_deg_per_episode = np.rad2deg(4 * np.pi * s) if not np.isnan(s) else np.nan

    # Gate: fit_R must beat the within-unit phase-rotation null (null_draw; else the burst-preserving
    # phase permutation, see shuffle_fit_r_significance) before the z-test below is considered.
    fit_r_shuffle_p, shuffle_fit_r = np.nan, None
    fit_r_significant = True
    if USE_FIT_R_GATE and np.isfinite(fit_R):
        fit_r_shuffle_p, shuffle_fit_r = shuffle_fit_r_significance(
            spk_ts, spk_firing_index, spk_theta_phase, fit_R, rng, slope_bnds=slope_bnds, null_draw=null_draw)
        fit_r_significant = bool(fit_r_shuffle_p < ALPHA)

    # Significance: fit_R gate passed AND Kempter et al. (2012) asymptotic z-test on
    # the circular-linear correlation (p from kempter_lincirc).
    is_significant_precession = bool(fit_r_significant and np.isfinite(p) and p < ALPHA)

    # Number of wrapped line segments the fitted phase line (2*pi*s*x + b, drawn
    # mod 2*pi as in plot_unit_summary) makes over the spikes' firing-index extent.
    # More than MAX_FIT_LINES is an implausibly steep fit: the cell is marked
    # non-precessing.
    n_fit_lines = np.nan
    fit_too_steep = False
    if len(spk_firing_index) > 0 and np.isfinite(s) and np.isfinite(b):
        ph_ends = 2 * np.pi * s * np.array([np.min(spk_firing_index), np.max(spk_firing_index)]) + b
        n_fit_lines = int(np.floor(ph_ends.max() / (2 * np.pi)) - np.floor(ph_ends.min() / (2 * np.pi)) + 1)
        fit_too_steep = bool(n_fit_lines > MAX_FIT_LINES)
    if fit_too_steep:
        is_significant_precession = False
    # Significant fit (z-test) AND slope beyond +/-SLOPE_THRESH_DEG_PER_EPISODE:
    # negative -> phase precessing (spike phase advances to earlier phase over
    # the episode); positive -> phase succeeding (phase moves later).
    # Significant with |slope| within the threshold: phase locked.
    # Non-significant fit: no phase relation.
    is_precessing = bool(is_significant_precession and slope_deg_per_episode < -SLOPE_THRESH_DEG_PER_EPISODE)
    is_recessing = bool(is_significant_precession and slope_deg_per_episode > SLOPE_THRESH_DEG_PER_EPISODE)
    is_phase_locked = bool(is_significant_precession and not (is_precessing or is_recessing))
    if not np.isfinite(p):
        precession_class = None
    elif fit_too_steep:
        precession_class = 'non_precessing'
    elif is_precessing:
        precession_class = 'phase_precessing'
    elif is_recessing:
        precession_class = 'phase_succeeding'
    elif is_phase_locked:
        precession_class = 'phase_locked'
    else:
        precession_class = 'no_phase_relation'

    # r^2 of the circular-linear fit (rho is Kempter et al.'s circular-linear
    # correlation coefficient, the circular analogue of a linear r).
    r_squared = rho ** 2 if not np.isnan(rho) else np.nan
    # Phase range: total degrees of phase actually swept by this cell's
    # spikes, i.e. the fit slope times the observed extent of firing index
    # (not the full -1..1 range, which the cell may not have fully covered).
    firing_index_range = float(np.ptp(spk_firing_index)) if len(spk_firing_index) > 0 else np.nan
    phase_range_deg = (abs(slope_deg_per_episode) * firing_index_range
                        if not (np.isnan(slope_deg_per_episode) or np.isnan(firing_index_range))
                        else np.nan)

    # Density map: spike density over (firing index, LFP phase), occupancy-normalized.
    fi_edges = np.linspace(-1, 1, 41)
    ph_edges = np.linspace(0, 2 * np.pi, 101)
    occ_density, _, _ = np.histogram2d(occ_firing_index, np.mod(occ_theta_phase, 2 * np.pi),
                                        bins=[fi_edges, ph_edges])
    occ_density *= occ_dt
    spk_density, _, _ = np.histogram2d(spk_firing_index, np.mod(spk_theta_phase, 2 * np.pi),
                                        bins=[fi_edges, ph_edges])
    # Theta phase at firing index 0 (episode midpoint): the fitted line
    # 2*pi*s*x + b at x = 0, i.e. the intercept b, wrapped to [0, 360) deg. The same line is
    # also evaluated at the other FIRING_INDEX_PHASE_POINTS (x = -1 firing onset, +1 firing offset).
    phase_at_points = {
        f'phase_at_{key}_deg': (float(np.mod(np.rad2deg(2 * np.pi * s * x + b), 360))
                                if np.isfinite(s) and np.isfinite(b) else np.nan)
        for x, key, *_ in FIRING_INDEX_PHASE_POINTS}

    with np.errstate(invalid='ignore', divide='ignore'):
        density = spk_density / occ_density
    density[~np.isfinite(density)] = 0.0
    density = gaussian_filter(density, sigma=1.5, truncate=2.0, mode='nearest')

    return {
        'spk_ts': spk_ts, 'spk_firing_index': spk_firing_index, 'spk_theta_phase': spk_theta_phase,
        'rho': rho, 'p': p, 's': s, 'b': b, 'fit_R': fit_R,
        'fit_R_shuffle_p': fit_r_shuffle_p, 'fit_R_Significant': fit_r_significant,
        'shuffle_fit_r': shuffle_fit_r,
        'slope_deg_per_episode': slope_deg_per_episode,
        'r_squared': r_squared, 'phase_range_deg': phase_range_deg,
        **phase_at_points,
        'is_precessing': is_precessing, 'is_recessing': is_recessing,
        'is_phase_locked': is_phase_locked, 'precession_class': precession_class,
        'n_fit_lines': n_fit_lines, 'fit_too_steep': fit_too_steep,
        'density': density, 'fi_edges': fi_edges, 'ph_edges': ph_edges,
        'n_spikes': len(spk_ts), 'n_events': n_events,
    }


def lfp_cycle_clock(lfp_ts, lfp_phase_unwrapped):
    """Theta-cycle clock of one LFP trace: continuous cycle number per sample (unwrapped phase / 2*pi, made
    non-decreasing because the Hilbert phase briefly runs backwards at about 1% of samples) and the cycle
    numbers of LFP gaps longer than ONOFF_MAX_LFP_GAP_S, across which spike counts are not defined."""
    cycle = np.maximum.accumulate(lfp_phase_unwrapped) / (2 * np.pi)
    return dict(ts=lfp_ts, cycle=cycle, gap_cycles=cycle[np.flatnonzero(np.diff(lfp_ts) > ONOFF_MAX_LFP_GAP_S)])


def spike_train_dynamics(spk_ts_all, clock, lfp_phase_unwrapped, lfp_fs, acg_epoch_ok=None):
    """Per-spike firing dynamics of one unit, position not involved. `spk_ts_all` = ALL the unit's spikes: a
    spike-count window counts every spike, whatever its LFP epoch, but only spikes in an ACG theta-positive epoch
    are later analysed (in_theta), since these epochs are often shorter than the window. Spikes with no LFP sample
    are dropped. Every window is centred on the spike's own phase, not aligned to fixed LFP cycles (see header).

    Returns a dict of arrays over the remaining spikes, in time order:
      ts, cycle (continuous cycle number), phase (rad, [0, 2*pi), 0 = LFP peak), in_theta;
      window_ok: the +/-ONOFF_WINDOW_CYCLES window lies inside the LFP and spans no LFP gap;
      n_prev / n_next: spikes in the previous / following ONOFF_WINDOW_CYCLES cycles, this spike excluded;
      inst_count: spikes within +/-INST_RATE_WINDOW_CYCLES, this spike included (Harris Fig 1d);
      rate / deriv: intercept (spikes per cycle) and slope (spikes per cycle^2) of the regression of spikes
      per cycle on cycle number, over the central cycle and RATE_DERIV_HALF_WINDOW_CYCLES each side (Fig 3)."""
    ts = np.sort(np.asarray(spk_ts_all, dtype=np.float64))
    ts = ts[spikes_with_lfp_sample(ts, clock['ts'], lfp_fs)]
    phase = phase_at(clock['ts'], lfp_phase_unwrapped, ts)
    ts, phase = ts[np.isfinite(phase)], np.mod(phase[np.isfinite(phase)], 2 * np.pi)
    cycle = np.interp(ts, clock['ts'], clock['cycle'])
    in_theta = (times_in_ok_epochs(ts, clock['ts'], lfp_fs, acg_epoch_ok) if acg_epoch_ok is not None
                else np.ones(len(ts), dtype=bool))

    w = ONOFF_WINDOW_CYCLES
    idx = np.arange(len(ts))
    n_prev = idx - np.searchsorted(cycle, cycle - w, side='left')
    n_next = np.searchsorted(cycle, cycle + w, side='right') - idx - 1
    n_gaps = (np.searchsorted(clock['gap_cycles'], cycle + w) - np.searchsorted(clock['gap_cycles'], cycle - w))
    window_ok = (cycle - w >= clock['cycle'][0]) & (cycle + w <= clock['cycle'][-1]) & (n_gaps == 0)

    inst_count = (np.searchsorted(cycle, cycle + INST_RATE_WINDOW_CYCLES, side='right')
                  - np.searchsorted(cycle, cycle - INST_RATE_WINDOW_CYCLES, side='left'))
    k = np.arange(-RATE_DERIV_HALF_WINDOW_CYCLES, RATE_DERIV_HALF_WINDOW_CYCLES + 1)
    counts = np.stack([np.searchsorted(cycle, cycle + j + 0.5, side='left')
                       - np.searchsorted(cycle, cycle + j - 0.5, side='left') for j in k], axis=1)
    return dict(ts=ts, cycle=cycle, phase=phase, in_theta=in_theta, window_ok=window_ok, n_prev=n_prev, n_next=n_next,
                inst_count=inst_count, rate=counts.mean(axis=1), deriv=counts @ k / np.sum(k ** 2))


def tracking_trace(pos_ts, pos_xy):
    """Field-free position trace for the stationary control: jitter-smoothed positions, speed (cm/s) and the
    path length travelled (cm), accumulating only at speed >= TRACKING_IMMOBILITY_SPEED_CM_S (a pause collapses
    to a point instead of contributing jitter distance)."""
    pos_sm, speed = smooth_positions(pos_ts, pos_xy)
    step = np.r_[0.0, np.hypot(*np.diff(pos_sm, axis=0).T)] * (speed >= TRACKING_IMMOBILITY_SPEED_CM_S)
    return dict(ts=pos_ts, pos_sm=pos_sm, speed=speed, path=np.cumsum(step))


def add_position_control(dyn, clock, trace):
    """Add the position-control columns to `dyn` (NaN without tracking, or outside it): x_cm, y_cm, speed_cm_s and
    path_cum_cm at each spike, and window_path_cm = path travelled over its +/-ONOFF_WINDOW_CYCLES window."""
    ts, n = dyn['ts'], len(dyn['ts'])
    for key in ('x_cm', 'y_cm', 'speed_cm_s', 'path_cum_cm', 'window_path_cm'):
        dyn[key] = np.full(n, np.nan)
    if trace is None or not n:
        return dyn
    inside = (ts >= trace['ts'][0]) & (ts <= trace['ts'][-1])
    t = ts[inside]
    dyn['x_cm'][inside] = np.interp(t, trace['ts'], trace['pos_sm'][:, 0])
    dyn['y_cm'][inside] = np.interp(t, trace['ts'], trace['pos_sm'][:, 1])
    dyn['speed_cm_s'][inside] = np.interp(t, trace['ts'], trace['speed'])
    dyn['path_cum_cm'][inside] = np.interp(t, trace['ts'], trace['path'])
    t_lo = np.interp(dyn['cycle'] - ONOFF_WINDOW_CYCLES, clock['cycle'], clock['ts'])
    t_hi = np.interp(dyn['cycle'] + ONOFF_WINDOW_CYCLES, clock['cycle'], clock['ts'])
    in_tracking = (t_lo >= trace['ts'][0]) & (t_hi <= trace['ts'][-1])
    dyn['window_path_cm'][in_tracking] = (np.interp(t_hi[in_tracking], trace['ts'], trace['path'])
                                          - np.interp(t_lo[in_tracking], trace['ts'], trace['path']))
    return dyn


def analyse_firing(cycle, n_prev, n_next, in_theta, window_ok, low=ONOFF_LOW_MAX_SPIKES, high=ONOFF_HIGH_MIN_SPIKES):
    """Onset / offset spikes, blocks, episodes and firing index of one unit (arrays as in spike_train_dynamics).
    Only spikes in a theta-positive epoch with a valid window can be labelled. A block = consecutive labelled spikes
    with the same label and no silence >= ONOFF_WINDOW_CYCLES between them. An episode = an onset block followed
    directly by an offset block, with no silence >= ONOFF_WINDOW_CYCLES and no LFP gap between its first and
    last spike. Firing index: -1 at the first onset spike, +1 at the last offset spike, linear in cycles.
    Returns dict(label: ONSET / OFFSET / 0, block: id or -1, episodes: [(first, last)] spike indices,
    firing_index: NaN outside episodes, episode_id: index into episodes or -1)."""
    n = len(cycle)
    usable = in_theta & window_ok
    label = np.zeros(n, dtype=np.int8)
    label[usable & (n_prev <= low) & (n_next >= high)] = ONSET
    label[usable & (n_prev >= high) & (n_next <= low)] = OFFSET
    block = np.full(n, -1, dtype=np.int64)
    firing_index = np.full(n, np.nan)
    episode_id = np.full(n, -1, dtype=np.int64)
    episodes = []
    sel = np.flatnonzero(label != 0)
    if len(sel):
        new_block = np.r_[True, (np.diff(label[sel]) != 0) | (np.diff(cycle[sel]) >= ONOFF_WINDOW_CYCLES)]
        block[sel] = np.cumsum(new_block) - 1
        starts = np.flatnonzero(new_block)
        stops = np.r_[starts[1:], len(sel)] - 1
        for j in range(len(starts) - 1):
            if label[sel[starts[j]]] != ONSET or label[sel[starts[j + 1]]] != OFFSET:
                continue
            first, last = sel[starts[j]], sel[stops[j + 1]]
            if np.diff(cycle[first:last + 1]).max() >= ONOFF_WINDOW_CYCLES or not window_ok[first:last + 1].all():
                continue
            firing_index[first:last + 1] = 2 * (cycle[first:last + 1] - cycle[first]) / (cycle[last] - cycle[first]) - 1
            episode_id[first:last + 1] = len(episodes)
            episodes.append((int(first), int(last)))
    return dict(label=label, block=block, episodes=episodes, firing_index=firing_index, episode_id=episode_id)


def phase_rotator(unit):
    """Null for 'phase is independent of firing dynamics, given the unit': within each unit, the sequence of its phases
    is rotated (circularly shifted, in spike order) against its spike train by a random offset of ROTATION_MIN_FRAC to
    1 - ROTATION_MIN_FRAC of its spikes, and the labels / firing index / rate stay where they are. This keeps every unit's
    phase distribution (its theta locking), its bursts and the slow drift of its phases over seconds, and the clustering in
    time of its onset, offset and high-rate spikes; only their pairing is broken. Shuffling bursts instead keeps neither
    slow structure: with the LFP time-shifted against the spikes (no true relation), burst-permutation p < 0.05 came out in
    80-100% of shifts for the firing-index fit and the rate ANOVA. `unit`: sorted (spikes of a unit contiguous, in time
    order). Returns draw(rng, theta) -> theta rotated within units."""
    unit = np.asarray(unit)
    bounds = np.r_[0, np.flatnonzero(np.diff(unit)) + 1, len(unit)]
    sizes = np.diff(bounds)
    start = np.repeat(bounds[:-1], sizes)
    size = np.repeat(sizes, sizes)
    pos = np.arange(len(unit)) - start

    def draw(rng, theta):
        k = np.floor(rng.uniform(ROTATION_MIN_FRAC, 1 - ROTATION_MIN_FRAC, len(sizes)) * sizes).astype(np.int64)
        return theta[start + (pos + np.repeat(k, sizes)) % size]
    return draw


def _null_p(observed, null):
    """Add-one permutation p of `observed` against the surrogate statistics `null` (larger = more extreme)."""
    null = np.asarray(null, dtype=float)
    return float((np.sum(null >= observed - 1e-12) + 1) / (len(null) + 1))


def onset_offset_unit_stats(phase, label, block, rng, usable):
    """Onset vs offset phase of one unit (Excel OnOff_* columns). OnOff_MWW_p: Mardia-Watson-Wheeler statistic on uniform
    scores against the unit's phases rotated among its `usable` (theta-positive) spikes (phase_rotator); needs
    ONOFF_MIN_SPIKES_PER_UNIT spikes of each label."""
    on, off = label == ONSET, label == OFFSET
    out = {'OnOff_n_onset': int(on.sum()), 'OnOff_n_offset': int(off.sum()),
           'OnOff_n_onset_blocks': len(np.unique(block[on])), 'OnOff_n_offset_blocks': len(np.unique(block[off]))}
    for name, sel in (('onset', on), ('offset', off)):
        out[f'OnOff_{name}_phase_deg'] = float(np.mod(np.rad2deg(circ_mean(phase[sel])), 360)) if sel.any() else np.nan
        out[f'OnOff_{name}_MRL'] = circ_r(phase[sel]) if sel.any() else np.nan
    out['OnOff_offset_minus_onset_deg'] = (float(np.mod(out['OnOff_offset_phase_deg'] - out['OnOff_onset_phase_deg'] + 180,
                                                        360) - 180) if on.any() and off.any() else np.nan)
    out['OnOff_MWW_W'], out['OnOff_MWW_p'] = np.nan, np.nan
    if on.sum() >= ONOFF_MIN_SPIKES_PER_UNIT and off.sum() >= ONOFF_MIN_SPIKES_PER_UNIT:
        base = np.flatnonzero(usable)
        lab = label[base]
        is_off = (lab[lab != 0] == OFFSET).astype(np.int64)

        def mww(th):
            scores = 2 * np.pi * stats.rankdata(th[lab != 0]) / int((lab != 0).sum())
            return _mww_statistic(np.cos(scores), np.sin(scores), is_off, 2)
        draw = phase_rotator(np.zeros(len(base), dtype=np.int64))
        W = mww(phase[base])
        out['OnOff_MWW_W'] = float(W)
        out['OnOff_MWW_p'] = _null_p(W, [mww(draw(rng, phase[base])) for _ in range(ONOFF_N_PERMUTATIONS)])
    return out


def _rate_groups(inst_count):
    """Harris Fig 1d rate group of each spike, as 0-based index: inst_count 1, 2, ..., INST_RATE_MAX_GROUP and above."""
    return np.clip(inst_count, 1, INST_RATE_MAX_GROUP) - 1


def _anova_statistic(theta, groups):
    """Harris et al. Methods' circular ANOVA statistic sum_i cos(theta_i - mean_g(i)) = sum_g R_g."""
    n = INST_RATE_MAX_GROUP
    return float(np.hypot(np.bincount(groups, np.cos(theta), n), np.bincount(groups, np.sin(theta), n)).sum())


def rate_phase_anova_within(unit, theta, groups, rng, n_perm=ONOFF_N_PERMUTATIONS):
    """Circular ANOVA of phase across rate groups, against the within-unit phase rotation (phase_rotator). `unit` sorted."""
    if len(np.unique(groups)) < 2:
        return np.nan
    draw = phase_rotator(unit)
    return _null_p(_anova_statistic(theta, groups), [_anova_statistic(draw(rng, theta), groups) for _ in range(n_perm)])


def compute_firing_fit(dyn, analysis, clock, lfp_phase_unwrapped, lfp_fs, rng, slope_bnds=None, lfp_theta_mask=None):
    """Phase precession of one unit on its firing index: the circular-linear fit, shuffle gate and classification of
    _precession_fit on the theta-positive spikes of its firing episodes. The density map's occupancy = the LFP samples
    of the same episodes (theta-positive ones, if lfp_theta_mask is given). Raises TooFewSpikes."""
    use = np.isfinite(analysis['firing_index']) & dyn['in_theta']
    if use.sum() < MIN_SPIKES_FOR_FIT:
        raise TooFewSpikes(f'only {int(use.sum())} spikes in {len(analysis["episodes"])} firing episodes '
                           f'< MIN_SPIKES_FOR_FIT={MIN_SPIKES_FOR_FIT}')
    lfp_ts = clock['ts']
    lfp_phase = np.mod(lfp_phase_unwrapped + np.pi, 2 * np.pi) - np.pi
    occ_fi, occ_phase = [], []
    for first, last in analysis['episodes']:
        lo, hi = np.searchsorted(lfp_ts, [dyn['ts'][first], dyn['ts'][last]], side='left')
        ok = np.ones(hi - lo, dtype=bool) if lfp_theta_mask is None else lfp_theta_mask[lo:hi]
        c0, c1 = dyn['cycle'][first], dyn['cycle'][last]
        occ_fi.append(2 * (clock['cycle'][lo:hi][ok] - c0) / (c1 - c0) - 1)
        occ_phase.append(lfp_phase[lo:hi][ok])
    # fit_R gate null: the unit's phases rotated among all its theta-positive spikes, then read at the fitted spikes
    theta_idx = np.flatnonzero(dyn['in_theta'])
    use_in_theta = np.searchsorted(theta_idx, np.flatnonzero(use))
    rotate = phase_rotator(np.zeros(len(theta_idx), dtype=np.int64))

    def null_draw(r):
        return rotate(r, dyn['phase'][theta_idx])[use_in_theta]
    fit = _precession_fit(dyn['ts'][use], analysis['firing_index'][use], dyn['phase'][use], rng, slope_bnds,
                          np.concatenate(occ_fi), np.concatenate(occ_phase), float(np.mean(np.diff(lfp_ts))),
                          null_draw=null_draw)
    fit['spk_label'] = analysis['label'][use]
    return fit


# ============================================================================
# Population: Harris et al.'s analyses across units, plus within-unit tests and the stationary control
# ============================================================================

def circ_mean_ci_deg(theta):
    """Circular mean (deg, [0, 360)), half-width of its 95% CI (deg; Fisher 1993, as Harris et al.: 1.96 * sqrt(delta / n),
    delta = (1 - rho2) / (2 R^2); 180 if undefined) and mean resultant length."""
    n = len(theta)
    if not n:
        return np.nan, np.nan, np.nan
    m = np.mean(np.exp(1j * theta))
    r, mu = float(np.abs(m)), float(np.angle(m))
    if n < 2 or r == 0:
        return float(np.mod(np.rad2deg(mu), 360)), 180.0, r
    delta = (1 - np.mean(np.cos(2 * (theta - mu)))) / (2 * r ** 2)
    return float(np.mod(np.rad2deg(mu), 360)), float(min(180.0, np.rad2deg(1.96 * np.sqrt(delta / n)))), r


def _wrap_deg(d):
    return float(np.mod(d + 180, 360) - 180)


def label_population(sp: pd.DataFrame, low: int, high: int) -> dict:
    """analyse_firing per unit over the per-spike table `sp` (sorted by unit, then time), at thresholds (low, high).
    Returns arrays over sp: label, block and episode_id (ids unique across units), firing_index, and episode_path /
    episode_displacement (cm, per spike in an episode; NaN without tracking or outside one), plus the 'episodes'
    DataFrame with one row per episode."""
    n = len(sp)
    label = np.zeros(n, dtype=np.int8)
    block = np.full(n, -1, dtype=np.int64)
    firing_index = np.full(n, np.nan)
    episode_id = np.full(n, -1, dtype=np.int64)
    episode_path = np.full(n, np.nan)
    episode_displacement = np.full(n, np.nan)
    cycle, phase = sp['cycle'].to_numpy(), sp['phase'].to_numpy()
    arrays = {c: sp[c].to_numpy() for c in ('n_prev', 'n_next', 'in_theta', 'window_ok', 'path_cum_cm', 'x_cm', 'y_cm')}
    unit, ts = sp['unit_id'].to_numpy(), sp['ts'].to_numpy()
    bounds = np.r_[0, np.flatnonzero(np.diff(unit)) + 1, n]
    n_block = n_ep = 0
    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        res = analyse_firing(cycle[a:b], arrays['n_prev'][a:b], arrays['n_next'][a:b], arrays['in_theta'][a:b],
                             arrays['window_ok'][a:b], low, high)
        label[a:b] = res['label']
        has_block = res['block'] >= 0
        block[a:b][has_block] = res['block'][has_block] + n_block
        n_block += int(res['block'].max()) + 1 if has_block.any() else 0
        firing_index[a:b] = res['firing_index']
        has_ep = res['episode_id'] >= 0
        episode_id[a:b][has_ep] = res['episode_id'][has_ep] + n_ep
        for j, (f, l) in enumerate(res['episodes']):
            path = arrays['path_cum_cm'][a + l] - arrays['path_cum_cm'][a + f]
            displacement = np.hypot(arrays['x_cm'][a + l] - arrays['x_cm'][a + f], arrays['y_cm'][a + l] - arrays['y_cm'][a + f])
            episode_path[a + f:a + l + 1] = path
            episode_displacement[a + f:a + l + 1] = displacement
            sub = np.arange(a + f, a + l + 1)
            on_ph, off_ph = phase[sub[label[sub] == ONSET]], phase[sub[label[sub] == OFFSET]]
            rows.append(dict(
                unit_id=unit[a], episode=n_ep + j, n_cycles=cycle[a + l] - cycle[a + f], duration_s=ts[a + l] - ts[a + f],
                n_spikes=l - f + 1, onset_phase_deg=np.mod(np.rad2deg(circ_mean(on_ph)), 360),
                offset_phase_deg=np.mod(np.rad2deg(circ_mean(off_ph)), 360), path_cm=path, displacement_cm=displacement))
        n_ep += len(res['episodes'])
    episodes = pd.DataFrame(rows, columns=['unit_id', 'episode', 'n_cycles', 'duration_s', 'n_spikes', 'onset_phase_deg',
                                           'offset_phase_deg', 'path_cm', 'displacement_cm'])
    if len(episodes):
        episodes['offset_minus_onset_deg'] = np.mod(episodes['offset_phase_deg'] - episodes['onset_phase_deg'] + 180, 360) - 180
    return dict(label=label, block=block, firing_index=firing_index, episode_id=episode_id, episode_path=episode_path,
                episode_displacement=episode_displacement, episodes=episodes)


def onset_offset_population_row(sp, lab, mask, rng, same_place=False):
    """Harris Fig 2 for the spikes in `mask`. Pooled absolute phases with the paper's free-label Mardia-Watson-Wheeler test
    (reported for comparison only: with the LFP time-shifted it gave p < 0.05 in 15% of shifts), then the within-unit test:
    within each stratum (a unit) holding both labels, phase is centred on the stratum's mean and the distance between the
    offset and onset mean vectors is tested against the unit's phases rotated among its `mask` spikes (phase_rotator); and
    one offset - onset per unit (Rayleigh test across units). same_place=True: strata are one unit in one
    ONOFF_SAME_PLACE_BIN_CM spatial bin (by each block's mean position), so onset and offset are compared at the same place
    and a phase that depends on place cannot appear as an onset / offset difference."""
    label = lab['label']
    on, off = mask & (label == ONSET), mask & (label == OFFSET)
    phase = sp['phase'].to_numpy()
    on_mean, on_ci, on_r = circ_mean_ci_deg(phase[on])
    off_mean, off_ci, off_r = circ_mean_ci_deg(phase[off])
    row = dict(n_onset=int(on.sum()), n_offset=int(off.sum()), onset_phase_deg=on_mean, onset_ci_deg=on_ci, onset_MRL=on_r,
               offset_phase_deg=off_mean, offset_ci_deg=off_ci, offset_MRL=off_r,
               offset_minus_onset_deg=_wrap_deg(off_mean - on_mean) if on.any() and off.any() else np.nan,
               MWW_W=np.nan, MWW_p_free_labels=np.nan, n_units_both=0, n_strata=0, n_blocks=0,
               centred_offset_minus_onset_deg=np.nan, within_unit_rotation_p=np.nan, n_units_tested=0,
               unit_offset_minus_onset_deg=np.nan, unit_shift_MRL=np.nan, unit_shift_Rayleigh_p=np.nan)
    if not (on.any() and off.any()):
        return row
    row['MWW_W'], _, row['MWW_p_free_labels'] = mardia_watson_wheeler_test([phase[on], phase[off]], rng, n_perm=ONOFF_N_PERMUTATIONS)

    # Strata (units, or unit x spatial bin) holding both labels; the null rotates phases among each unit's `mask` spikes
    sel = np.flatnonzero(on | off)
    unit, centred, blk = sp['unit_id'].to_numpy()[sel], sp['phase_centred'].to_numpy()[sel], lab['block'][sel]
    _, ginv = np.unique(blk, return_inverse=True)
    n_blk = int(ginv.max()) + 1
    z = np.bincount(ginv, np.cos(centred), n_blk) + 1j * np.bincount(ginv, np.sin(centred), n_blk)
    size = np.bincount(ginv, minlength=n_blk).astype(float)
    blk_unit = np.zeros(n_blk, dtype=np.int64)
    blk_unit[ginv] = unit
    blk_off = np.zeros(n_blk, dtype=bool)
    blk_off[ginv] = label[sel] == OFFSET
    if same_place:
        bx = np.bincount(ginv, sp['x_cm'].to_numpy()[sel], n_blk) / size
        by = np.bincount(ginv, sp['y_cm'].to_numpy()[sel], n_blk) / size
        placed = np.isfinite(bx) & np.isfinite(by)
        _, stratum = np.unique(np.c_[blk_unit[placed], np.floor(bx[placed] / ONOFF_SAME_PLACE_BIN_CM),
                                     np.floor(by[placed] / ONOFF_SAME_PLACE_BIN_CM)], axis=0, return_inverse=True)
        blk_stratum = np.full(n_blk, -1, dtype=np.int64)
        blk_stratum[placed] = stratum.ravel()
    else:
        blk_stratum = blk_unit
    n_labels = pd.DataFrame(dict(u=blk_stratum, o=blk_off))[blk_stratum >= 0].groupby('u')['o'].agg(['sum', 'size'])
    both = n_labels.index[(n_labels['sum'] > 0) & (n_labels['sum'] < n_labels['size'])].to_numpy()
    keep = np.isin(blk_stratum, both)
    row['n_units_both'] = len(np.unique(blk_unit[keep]))
    row['n_strata'] = int(len(both))
    row['n_blocks'] = int(keep.sum())
    if keep.any():
        spk_keep = keep[ginv]                                   # labelled spikes of strata holding both labels
        stratum, spk_off = blk_stratum[ginv][spk_keep], blk_off[ginv][spk_keep]
        _, stratum = np.unique(stratum, return_inverse=True)
        n_str = int(stratum.max()) + 1
        base = np.flatnonzero(mask)                              # rotation base: the unit's spikes in `mask`
        in_base = np.searchsorted(base, sel[spk_keep])
        draw = phase_rotator(sp['unit_id'].to_numpy()[base])
        base_theta = sp['phase_centred'].to_numpy()[base]

        def statistic(th, signed=False):
            zs = np.exp(1j * th)
            centre = np.angle(np.bincount(stratum, zs.real, n_str) + 1j * np.bincount(stratum, zs.imag, n_str))
            zc = zs * np.exp(-1j * centre[stratum])
            if signed:
                return _wrap_deg(np.rad2deg(np.angle(zc[spk_off].sum()) - np.angle(zc[~spk_off].sum())))
            return np.abs(zc[spk_off].mean() - zc[~spk_off].mean())
        observed = statistic(base_theta[in_base])
        row['within_unit_rotation_p'] = _null_p(observed, [statistic(draw(rng, base_theta)[in_base])
                                                           for _ in range(ONOFF_N_PERMUTATIONS)])
        row['centred_offset_minus_onset_deg'] = statistic(base_theta[in_base], signed=True)

    # One offset - onset per unit
    shifts = []
    for u in np.unique(unit):
        in_u = unit == u
        lab_u, ph_u = label[sel][in_u], centred[in_u]
        if (lab_u == ONSET).sum() >= ONOFF_MIN_SPIKES_PER_UNIT and (lab_u == OFFSET).sum() >= ONOFF_MIN_SPIKES_PER_UNIT:
            shifts.append(circ_mean(ph_u[lab_u == OFFSET]) - circ_mean(ph_u[lab_u == ONSET]))
    row['n_units_tested'] = len(shifts)
    if shifts:
        shifts = np.asarray(shifts)
        row['unit_offset_minus_onset_deg'] = _wrap_deg(np.rad2deg(circ_mean(shifts)))
        row['unit_shift_MRL'] = circ_r(shifts)
        row['unit_shift_Rayleigh_p'] = circ_rtest(shifts)[0]
    return row


def rate_phase_population(sp, mask, rng):
    """Harris Fig 1d for the spikes in `mask`. Returns (bin rows, row with the ANOVA p): free labels on absolute phases,
    as the paper (comparison only: p < 0.05 in 45% of LFP time shifts), and the within-unit phase rotation on phase
    relative to each unit's preferred phase."""
    sub = sp.loc[mask]
    if not len(sub):
        return [], dict(n_spikes=0)
    phase, centred = sub['phase'].to_numpy(), sub['phase_centred'].to_numpy()
    unit, ts = sub['unit_id'].to_numpy(), sub['ts'].to_numpy()
    groups = _rate_groups(sub['inst_count'].to_numpy())
    rows = []
    for g in np.unique(groups):
        in_g = groups == g
        m, ci, r = circ_mean_ci_deg(phase[in_g])
        mc, cic, _ = circ_mean_ci_deg(centred[in_g])
        rows.append(dict(spikes_in_2_cycles=int(g) + 1, n_spikes=int(in_g.sum()), n_units=len(np.unique(unit[in_g])), phase_deg=m,
                         ci_deg=ci, MRL=r, centred_phase_deg=_wrap_deg(mc), centred_ci_deg=cic))
    observed = _anova_statistic(phase, groups)
    null = np.array([_anova_statistic(phase, rng.permutation(groups)) for _ in range(ONOFF_N_PERMUTATIONS)])
    return rows, dict(n_spikes=len(sub), n_units=len(np.unique(unit)),
                      circ_ANOVA_p_free_labels=float((np.sum(null >= observed - 1e-9) + 1) / (ONOFF_N_PERMUTATIONS + 1)),
                      circ_ANOVA_p_within_unit=rate_phase_anova_within(unit, centred, groups, rng))


def rate_derivative_population(sp, mask, rng):
    """Harris Fig 3 regions for the spikes in `mask`: low rate, then at high rate accelerating, steady and decelerating.
    The accelerating vs decelerating difference (distance between mean vectors of relative phase) is tested against
    the within-unit phase rotation. The paper's text reads 'derivative <1' for the decelerating region; given its
    92 deg and mauve region, that is taken to be <= -1."""
    sub = sp.loc[mask]
    rate, deriv = sub['rate'].to_numpy(), sub['deriv'].to_numpy()
    phase, centred = sub['phase'].to_numpy(), sub['phase_centred'].to_numpy()
    high = rate > RATE_HIGH_MIN
    unit, ts = sub['unit_id'].to_numpy(), sub['ts'].to_numpy()

    def region_row(name, sel, p=np.nan):
        m, ci, r = circ_mean_ci_deg(phase[sel])
        mc, cic, _ = circ_mean_ci_deg(centred[sel])
        return dict(Region=name, n_spikes=int(sel.sum()), n_units=len(np.unique(unit[sel])), phase_deg=m, ci_deg=ci, MRL=r,
                    centred_phase_deg=_wrap_deg(mc) if sel.any() else np.nan, centred_ci_deg=cic, accel_vs_decel_within_unit_p=p)

    rows = [region_row(f'rate <= {RATE_LOW_MAX:g}', rate <= RATE_LOW_MAX)]
    for d_min in (RATE_DERIV_MIN, RATE_DERIV_MIN_RELAXED):
        acc, dec = high & (deriv >= d_min), high & (deriv <= -d_min)
        p = np.nan
        if acc.any() and dec.any():
            def statistic(th):
                return np.abs(np.mean(np.exp(1j * th[dec])) - np.mean(np.exp(1j * th[acc])))
            draw = phase_rotator(unit)
            p = _null_p(statistic(centred), [statistic(draw(rng, centred)) for _ in range(ONOFF_N_PERMUTATIONS)])
        rows += [region_row(f'rate > {RATE_HIGH_MIN:g}, accelerating (deriv >= {d_min:g})', acc, p),
                 region_row(f'rate > {RATE_HIGH_MIN:g}, steady (|deriv| < {d_min:g})', high & (np.abs(deriv) < d_min), p),
                 region_row(f'rate > {RATE_HIGH_MIN:g}, decelerating (deriv <= -{d_min:g})', dec, p)]
    return rows


def _thirds_shift_deg(x, theta):
    """Model-free direction of phase across an episode: circular mean phase in the early, middle and late third of firing
    index, and the unwrapped shift = (middle - early) + (late - middle), each step wrapped to [-180, 180). Returns
    (early, middle, late, total); a shift near 180 deg gets its sign from the route through the middle third."""
    edges = (-1 / 3, 1 / 3)
    means = [circ_mean(theta[s]) if s.any() else np.nan for s in (x < edges[0], (x >= edges[0]) & (x < edges[1]), x >= edges[1])]
    e, m, l = np.rad2deg(means)
    return float(e), float(m), float(l), _wrap_deg(m - e) + _wrap_deg(l - m)


def firing_fit_population(sp, x, mask, rng, n_boot=N_POOLED_FIT_BOOTSTRAPS):
    """Phase relative to each unit's preferred phase vs firing index, over all spikes in `mask`'s episodes: a line fit across
    units, with p for fit_R against the within-unit phase rotation (each unit's phases rotated among its `mask` spikes, then
    read at its episode spikes) and a 95% CI on the slope from resampling units (the spikes of one unit are not independent);
    and the model-free unwrapped shift of _thirds_shift_deg with the same null (two-sided) and CI, because the line's slope
    is often multimodal on so few spikes. Returns (fit row, firing-index bin rows)."""
    base = np.flatnonzero(mask)
    sel = np.flatnonzero(mask & np.isfinite(x))
    row = dict(n_spikes=len(sel), n_units=0, slope_deg_per_episode=np.nan, slope_ci_lo=np.nan, slope_ci_hi=np.nan, fit_R=np.nan,
               fit_R_within_unit_p=np.nan, early_deg=np.nan, middle_deg=np.nan, late_deg=np.nan, unwrapped_shift_deg=np.nan,
               shift_ci_lo=np.nan, shift_ci_hi=np.nan, shift_within_unit_p=np.nan)
    if len(sel) < MIN_SPIKES_FOR_FIT:
        return row, []
    unit = sp['unit_id'].to_numpy()[sel]
    theta, xs = sp['phase_centred'].to_numpy()[sel], x[sel]
    row['n_units'] = len(np.unique(unit))
    s, _, fit_r = anglereg(xs, theta, SLOPE_BNDS)
    draw = phase_rotator(sp['unit_id'].to_numpy()[base])
    base_theta, in_base = sp['phase_centred'].to_numpy()[base], np.searchsorted(base, sel)
    early, middle, late, shift = _thirds_shift_deg(xs, theta)
    null, null_shift = np.empty(N_POOLED_FIT_SHUFFLES), np.empty(N_POOLED_FIT_SHUFFLES)
    for i in range(N_POOLED_FIT_SHUFFLES):
        th = draw(rng, base_theta)[in_base]
        null[i] = anglereg(xs, th, SLOPE_BNDS)[2]
        null_shift[i] = _thirds_shift_deg(xs, th)[3]
    row.update(slope_deg_per_episode=720.0 * s, fit_R=fit_r,
               fit_R_within_unit_p=float((np.sum(null >= fit_r) + 1) / (N_POOLED_FIT_SHUFFLES + 1)),
               early_deg=early, middle_deg=middle, late_deg=late, unwrapped_shift_deg=shift,
               shift_within_unit_p=float((np.sum(np.abs(null_shift) >= abs(shift)) + 1) / (N_POOLED_FIT_SHUFFLES + 1)))
    idx_by_unit = [np.flatnonzero(unit == u) for u in np.unique(unit)]
    if n_boot and len(idx_by_unit) >= 2:
        boots, boot_shift = [], []
        for _ in range(n_boot):
            take = np.concatenate([idx_by_unit[i] for i in rng.integers(len(idx_by_unit), size=len(idx_by_unit))])
            boots.append(720.0 * anglereg(xs[take], theta[take], SLOPE_BNDS)[0])
            boot_shift.append(_thirds_shift_deg(xs[take], theta[take])[3])
        row['slope_ci_lo'], row['slope_ci_hi'] = np.percentile(boots, [2.5, 97.5])
        row['shift_ci_lo'], row['shift_ci_hi'] = np.nanpercentile(boot_shift, [2.5, 97.5])

    edges = np.linspace(-1, 1, FIRING_EPISODE_INDEX_BINS + 1)
    bins = np.clip(np.digitize(xs, edges) - 1, 0, FIRING_EPISODE_INDEX_BINS - 1)
    bin_rows = []
    for j in range(FIRING_EPISODE_INDEX_BINS):
        m, ci, r = circ_mean_ci_deg(theta[bins == j])
        bin_rows.append(dict(firing_index_lo=edges[j], firing_index_hi=edges[j + 1], n_spikes=int((bins == j).sum()),
                             centred_phase_deg=_wrap_deg(m) if np.isfinite(m) else np.nan, ci_deg=ci, MRL=r))
    return row, bin_rows


def unit_slope_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Direction of the per-unit fits: how many units have a negative vs positive slope, binomial and Wilcoxon signed-rank
    tests against no preferred direction, and the PrecessionClass counts. By arena and animal, and for all fitted units."""
    fitted = df.loc[df['PrecessionTested'] == True]                               # noqa: E712
    groups = [('All', fitted)]
    groups += [(a, fitted[fitted['Arena'] == a]) for a in ARENA_LABELS]
    groups += [(a, fitted[fitted['Animal'] == a]) for a in sorted(fitted['Animal'].dropna().unique())]
    groups += [('ThetaModulated', fitted[fitted['ThetaModulated'] == True])]      # noqa: E712
    rows = []
    for name, sub in groups:
        s = sub['slope_deg_per_episode'].dropna().to_numpy(dtype=float)
        n_neg, n_pos = int((s < 0).sum()), int((s > 0).sum())
        rows.append(dict(
            Group=name, n_units=len(s), n_neg_slope=n_neg, n_pos_slope=n_pos,
            binomial_p=stats.binomtest(n_neg, n_neg + n_pos).pvalue if n_neg + n_pos else np.nan,
            median_slope_deg_per_episode=float(np.median(s)) if len(s) else np.nan,
            wilcoxon_p=stats.wilcoxon(s).pvalue if len(s) >= 5 else np.nan,
            **{f'n_{c}': int((sub['PrecessionClass'] == c).sum()) for c in FI0_PHASE_CLASSES}))
    return pd.DataFrame(rows)


def run_onoff_population(sp: pd.DataFrame, units: pd.DataFrame, out_dir: Path, rng) -> dict[str, pd.DataFrame]:
    """Every population analysis of one per-spike table (from process_session; sorted by unit, then time). `units` has one
    row per unit: unit_id, Animal, Arena, ThetaModulated, and the fit columns. Returns {sheet name: table} and saves the figures."""
    sp = sp.sort_values(['unit_id', 'ts'], kind='stable').reset_index(drop=True)
    theta_sp = sp['in_theta'].to_numpy()
    pref = sp.loc[theta_sp].groupby('unit_id')['phase'].agg(circ_mean)
    sp['phase_centred'] = sp['phase'] - sp['unit_id'].map(pref)
    meta = units.drop_duplicates('unit_id').set_index('unit_id')
    arena = sp['unit_id'].map(meta['Arena']).to_numpy(dtype=object)
    animal = sp['unit_id'].map(meta['Animal']).to_numpy(dtype=object)
    theta_mod = sp['unit_id'].map(meta['ThetaModulated']).fillna(False).to_numpy(dtype=bool)
    usable = theta_sp & sp['window_ok'].to_numpy()

    def groups_of(base):
        out = [('All', base)]
        out += [(a, base & (arena == a)) for a in ARENA_LABELS if (base & (arena == a)).any()]
        out += [(a, base & (animal == a)) for a in sorted(set(animal)) if (base & (animal == a)).any()]
        out.append(('ThetaModulated', base & theta_mod))
        return out

    window_path = sp['window_path_cm'].to_numpy()
    primary = (ONOFF_LOW_MAX_SPIKES, ONOFF_HIGH_MIN_SPIKES)
    fig2_rows, fi_rows, fi_bins, episode_tables, threshold_rows = [], [], [], [], []
    panels_fi, panels_fig2 = {}, {}
    for low, high in (primary, *ONOFF_SENSITIVITY_THRESHOLDS):
        is_primary = (low, high) == primary
        lab = label_population(sp, low, high)
        tag = dict(low_max_spikes=low, high_min_spikes=high)
        threshold_rows.append({**tag, 'n_analysed_spikes': int(usable.sum()), 'n_onset': int((lab['label'] == ONSET).sum()),
                               'n_offset': int((lab['label'] == OFFSET).sum()), 'n_episodes': len(lab['episodes']),
                               'n_units_with_episodes': lab['episodes']['unit_id'].nunique() if len(lab['episodes']) else 0})
        if is_primary:
            episode_tables.append(lab['episodes'])
        # The stationary control applies to the classified spike's own window (Fig 2) and to the episode (fit)
        groups = groups_of(usable) if is_primary else [('All', usable)]
        spike_groups = groups + [(f'Stationary (window path < {ONOFF_STATIONARY_PATH_CM:g} cm)',
                                  usable & (window_path < ONOFF_STATIONARY_PATH_CM)),
                                 (f'Moving (window path >= {ONOFF_STATIONARY_PATH_CM:g} cm)',
                                  usable & (window_path >= ONOFF_STATIONARY_PATH_CM))]
        for name, mask in spike_groups:
            print(f'  onset / offset: <= {low} -> >= {high} spikes, {name}')
            fig2_rows.append({**tag, 'Group': name, **onset_offset_population_row(sp, lab, mask, rng)})
        fig2_rows.append({**tag, 'Group': f'Same place (unit x {ONOFF_SAME_PLACE_BIN_CM:g} cm bin)',
                          **onset_offset_population_row(sp, lab, usable, rng, same_place=True)})
        ep_groups = groups + [(f'Stationary episodes (path < {ONOFF_STATIONARY_PATH_CM:g} cm)',
                               usable & (lab['episode_path'] < ONOFF_STATIONARY_PATH_CM)),
                              (f'Same-place episodes (onset-offset < {ONOFF_SAME_PLACE_CM:g} cm)',
                               usable & (lab['episode_displacement'] < ONOFF_SAME_PLACE_CM)),
                              (f'Displaced episodes (onset-offset >= {ONOFF_SAME_PLACE_CM:g} cm)',
                               usable & (lab['episode_displacement'] >= ONOFF_SAME_PLACE_CM))]
        for name, mask in ep_groups:
            print(f'  firing-index fit: <= {low} -> >= {high} spikes, {name}')
            row, bins = firing_fit_population(sp, lab['firing_index'], mask, rng, n_boot=N_POOLED_FIT_BOOTSTRAPS if is_primary else 0)
            fi_rows.append({**tag, 'Group': name, **row})
            fi_bins += [{**tag, 'Group': name, **b} for b in bins]
            if is_primary:
                panels_fi[name] = (mask, row)
        if is_primary:
            lab_primary = lab
            panels_fig2 = [r for r in fig2_rows if (r['low_max_spikes'], r['high_min_spikes']) == primary]

    print('  phase vs instantaneous rate; rate x derivative')
    rate_bins, rate_tests, region_rows = [], [], []
    for name, mask in groups_of(usable) + [(f'Stationary (window path < {ONOFF_STATIONARY_PATH_CM:g} cm)',
                                            usable & (window_path < ONOFF_STATIONARY_PATH_CM))]:
        bins, test = rate_phase_population(sp, mask, rng)
        rate_bins += [{'Group': name, **b} for b in bins]
        rate_tests.append({'Group': name, **test})
        region_rows += [{'Group': name, **r} for r in rate_derivative_population(sp, mask, rng)]

    episodes = pd.concat(episode_tables, ignore_index=True)
    episodes = episodes.merge(units.drop_duplicates('unit_id')[['unit_id', 'Animal', 'Arena', 'Session', 'Unit']], on='unit_id', how='left')
    tables = {'OnOff_Thresholds': pd.DataFrame(threshold_rows), 'OnOff_OnsetOffset': pd.DataFrame(fig2_rows),
              'OnOff_FiringIndexFit': pd.DataFrame(fi_rows), 'OnOff_FiringIndexBins': pd.DataFrame(fi_bins),
              'OnOff_UnitSlopes': unit_slope_rows(units), 'OnOff_RateBins': pd.DataFrame(rate_bins),
              'OnOff_RateANOVA': pd.DataFrame(rate_tests), 'OnOff_RateDerivRegions': pd.DataFrame(region_rows),
              'OnOff_Episodes': episodes}

    arenas_present = [a for a in ARENA_LABELS if (arena == a).any()]
    plot_phase_vs_rate(tables['OnOff_RateBins'], ['All', *arenas_present], out_dir / 'OnOff_PhaseVsRate.png')
    plot_onset_offset_groups(panels_fig2, out_dir / 'OnOff_OnsetVsOffset.png')
    plot_rate_derivative_maps(sp, {n: m for n, m in groups_of(usable) if n in ('All', *arenas_present)},
                              out_dir / 'OnOff_PhaseRateDerivative.png')
    plot_firing_index_population(sp, lab_primary['firing_index'], panels_fi, tables['OnOff_FiringIndexBins'], primary,
                                 ['All', *arenas_present, *[n for n in panels_fi if 'episodes' in n]],
                                 out_dir / 'OnOff_FiringIndex_Population.png')
    return tables


# ============================================================================
# Steps 1 & 2: theta phase-locking, Theta Modulation Index, shuffle test
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


def mardia_watson_wheeler_test(groups_rad, rng, n_perm=N_FI0_PERMUTATIONS):
    """Mardia-Watson-Wheeler uniform-scores k-sample test: nonparametric test
    that the circular distributions are identical (sensitive to differences in
    location or spread). The pooled angles are ranked and mapped to uniform
    scores 2*pi*rank/N; W = 2 * sum_i (C_i^2 + S_i^2) / n_i. Returns
    (W, p_chi2, p_perm): p_chi2 from the asymptotic chi^2(2(k-1)) distribution
    (needs ~10+ cells per arena), p_perm from n_perm group-label permutations
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
    """'precession' if the phase (deg, any cycle) falls in FI0_PRECESSION_RANGE_DEG
    on the circle, else 'procession'; None for NaN."""
    if not np.isfinite(phase_deg):
        return None
    lo, hi = FI0_PRECESSION_RANGE_DEG
    return 'precession' if lo <= np.mod(phase_deg, 360) < hi else 'procession'


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


def compute_theta_modulation(spk_ts, lfp_ts, lfp_phase_unwrapped, lfp_fs, rng):
    """Steps 1 & 2 for one unit: Hilbert theta-phase polar-plot statistics
    (MRL, preferred phase, Rayleigh test) plus the Theta Modulation Index and
    its shuffle-test significance. Spikes with no defined phase (outside the
    LFP's time range or in an LFP gap) are dropped.

    Returns (metrics dict, phase_deg array of per-spike theta phase, for
    plotting the Step 1 polar histogram, shuffle_tmis array of the TMI
    shuffle-test null distribution -- None if too few spikes to run it).
    """
    phase_rad = assign_spike_phase(spk_ts, lfp_ts, lfp_phase_unwrapped, lfp_fs)
    has_phase = np.isfinite(phase_rad)
    spk_ts, phase_rad = spk_ts[has_phase], phase_rad[has_phase]
    phase_deg = np.degrees(phase_rad)
    n = len(phase_deg)

    metrics = dict(n_spikes_theta=n)
    if n < MIN_SPIKES_FOR_TMI:
        metrics.update(MRL=np.nan, PreferredPhase_deg=np.nan, Rayleigh_p=np.nan,
                        SignificantThetaModulation=False, PhasePeak_deg=np.nan,
                        PhaseValley_deg=np.nan, TMI=np.nan, TMI_shuffle_p=np.nan,
                        TMI_Significant=False, ThetaModulated=False)
        return metrics, phase_deg, None

    mrl = circ_r(phase_rad)
    pref_phase_deg = np.degrees(circ_mean(phase_rad)) % 360
    rayleigh_p, _ = circ_rtest(phase_rad)

    peak_phase_deg, valley_phase_deg, tmi = find_phase_peak_valley(phase_deg)
    tmi_pval, shuffle_tmis = shuffle_tmi_significance(spk_ts, phase_deg, tmi, rng)
    tmi_sig = bool(np.isfinite(tmi_pval) and tmi_pval < ALPHA)
    rayleigh_sig = bool(rayleigh_p < ALPHA)

    # ThetaModulated: the criterion gating Step 3 (TMI shuffle, AND Rayleigh if REQUIRE_RAYLEIGH_FOR_THETA_MOD)
    metrics.update(MRL=mrl, PreferredPhase_deg=pref_phase_deg, Rayleigh_p=rayleigh_p,
                    SignificantThetaModulation=rayleigh_sig,
                    PhasePeak_deg=peak_phase_deg, PhaseValley_deg=valley_phase_deg,
                    TMI=tmi, TMI_shuffle_p=tmi_pval, TMI_Significant=tmi_sig,
                    ThetaModulated=tmi_sig and (rayleigh_sig or not REQUIRE_RAYLEIGH_FOR_THETA_MOD))
    return metrics, phase_deg, shuffle_tmis


# ============================================================================
# Plotting
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
    """Step 1 (visualization only): polar histogram of spike counts vs.
    Hilbert theta phase, with the MRL/preferred-phase vector."""
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
    verdict = 'PASS (theta-modulated)' if is_sig else 'FAIL (not theta-modulated)'
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
    """Step 2 (visualization): left, the smoothed/max-normalized phase
    histogram calc_tmi scores (TMI = 1 - its trough); right, the TMI
    shuffle-test null distribution with the observed TMI and the null's
    (1 - ALPHA) percentile. The cell passes to Step 3 iff p < ALPHA."""
    color, verdict_line = _tmi_verdict(tmi, tmi_pval, is_sig)
    fig, (ax_h, ax_s) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f'{title}\n{verdict_line}', fontsize=10, color=color)
    _draw_tmi_histogram(ax_h, phase_deg, tmi, is_sig)
    _draw_tmi_null(ax_s, tmi, is_sig, shuffle_tmis)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _scatter_phase_dup(ax, firing_index, theta, **kw):
    """Scatter (firing index, theta phase) with the phase duplicated over 0-720 deg."""
    ph = np.rad2deg(np.mod(theta, 2 * np.pi))
    ax.scatter(np.r_[firing_index, firing_index], np.r_[ph, ph + 360], **kw)


def _draw_firing_index_phase(ax, results, label=None):
    """Draw LFP phase (duplicated over 0-720 deg) vs firing index for each spike, with the fitted circular-linear
    line (Step 3). `label`, if given, is put on its own line above the fit statistics in the title. Onset and
    offset spikes in their Harris Fig 2 colours, all other spikes black."""
    # background gradient: theta peak (0/360/720 deg) -> trough (180/540 deg), light magma
    ph_bg = np.linspace(0, 720, 721)
    peakness = 0.5 *(1 + np.cos(np.deg2rad(ph_bg)))  # 1 at peak, 0 at trough
    mag = plt.get_cmap('magma')(0.35 + 0.65 * peakness)[:, :3]  # skip the darkest blacks
    bg_alpha = 0.18  # magma opacity: lower = lighter/more transparent
    bg = bg_alpha * mag + (1 - bg_alpha) * 1.0  # blend with white for a lighter hue
    ax.imshow(bg[:, None, :], origin='lower', aspect='auto', extent=[-1, 1, 0, 720],
              zorder=0, interpolation='bilinear')
    x, theta, spk_label = results['spk_firing_index'], results['spk_theta_phase'], results['spk_label']
    _scatter_phase_dup(ax, x[spk_label == 0], theta[spk_label == 0], s=8, alpha=0.6, color='k', zorder=2)
    for lab, color, name in ((ONSET, FIRING_ONSET_COLOR, 'onset'), (OFFSET, FIRING_OFFSET_COLOR, 'offset')):
        _scatter_phase_dup(ax, x[spk_label == lab], theta[spk_label == lab], s=22, color=color, edgecolors='k',
                           linewidths=0.4, zorder=3, label=f'{name} spikes (n={int((spk_label == lab).sum())})')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.85)
    s, b = results['s'], results['b']
    if not np.isnan(s):
        xg = np.linspace(-1, 1, 500)
        phi = np.mod(2 * np.pi * s * xg + b, 2 * np.pi)
        phi[np.abs(np.diff(phi, prepend=phi[0])) > np.pi] = np.nan
        ax.plot(xg, np.rad2deg(phi), 'r', linewidth=2)
        ax.plot(xg, np.rad2deg(phi) + 360, 'r', linewidth=2)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 720)
    ax.set_xlabel('Firing index (-1 onset, +1 offset)')
    ax.set_ylabel('LFP phase (deg)')
    ax.set_title((f'{label}\n' if label else '') +
                 f'rho={results["rho"]:.2f}  p={results["p"]:.3g}  '
                 f'slope={results["slope_deg_per_episode"]:.1f} deg/episode  '
                 f'class={results["precession_class"]}', fontsize=9)


def _draw_density(fig, ax, results, title):
    """Occupancy-normalized spike density over (firing index, LFP phase), the phase axis duplicated over 0-720 deg."""
    ph_centers = np.rad2deg(0.5 * (results['ph_edges'][:-1] + results['ph_edges'][1:]))
    fi_centers = 0.5 * (results['fi_edges'][:-1] + results['fi_edges'][1:])
    dens = results['density']
    dens_dup = np.concatenate([dens, dens], axis=1)
    ph_dup = np.concatenate([ph_centers, ph_centers + 360])
    im = ax.pcolormesh(fi_centers, ph_dup, dens_dup.T, cmap='jet', shading='auto')
    fig.colorbar(im, ax=ax, label='Rate (Hz)')
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 720)
    ax.set_xlabel('Firing index (-1 onset, +1 offset)')
    ax.set_ylabel('LFP phase (deg)')
    ax.set_title(title)


def _fit_text_lines(results):
    """Text-panel lines with the circular-linear fit statistics of one spike set."""
    return [
        '-- Kempter z-test (on burst events) --' if BURST_COLLAPSE_FOR_ZTEST
        else '-- Kempter (circular-linear, asymptotic z-test) --',
        f"rho = {results['rho']:.3f}   (r^2 = {results['r_squared']:.3f})",
        f"p (asymptotic) = {results['p']:.4g}",
        f"fit R = {results['fit_R']:.3f}  (shuffle p = {results['fit_R_shuffle_p']:.3g}, "
        f"{'PASS' if results['fit_R_Significant'] else 'FAIL'})",
        f"slope ={results['slope_deg_per_episode']:.2f} deg/episode;  phase range = {results['phase_range_deg']:.1f} deg",
        *(f"phase @ firing index {x:g} ={results[f'phase_at_{key}_deg']:.1f} deg "
          f"({phase_range_label(results[f'phase_at_{key}_deg'])} range)"
          for x, key, *_ in FIRING_INDEX_PHASE_POINTS),
        f"class = {results['precession_class']}",
    ]


def _draw_text_panel(ax, text_lines):
    ax.axis('off')
    for i, line in enumerate(text_lines):
        is_bold = line.startswith(('class = ', 'EPISODES'))
        ax.text(0.0, 0.99 - i * 0.07, line, fontsize=9, fontweight='bold' if is_bold else 'normal')


def _draw_fit_r_null(ax_s, fit_r, is_sig, shuffle_fit_r):
    """Draw the fit_R shuffle-test null distribution (shuffle_fit_r_significance)
    with the observed fit_R and the null's (1 - ALPHA) percentile."""
    color = 'red' if is_sig else '#4D4D4D'
    thresh = float(np.percentile(shuffle_fit_r, 100 * (1 - ALPHA)))
    ax_s.hist(shuffle_fit_r, bins=40, color='#D9D9D9', edgecolor='black', linewidth=0.5)
    ax_s.axvline(thresh, color='0.3', linestyle='--', linewidth=1.5,
                 label=f'shuffle {100 * (1 - ALPHA):g}th pct = {thresh:.3f}')
    ax_s.axvline(fit_r, color=color, linewidth=2.5, label=f'observed fit R = {fit_r:.3f}')
    ax_s.set_xlabel('fit R (residual MRL)')
    ax_s.set_ylabel('Shuffle count')
    ax_s.set_title(f'fit R null ({len(shuffle_fit_r)} within-unit phase rotations)',
                   fontsize=9)
    ax_s.legend(fontsize=8, loc='upper left')


# ============================================================================
# Batch main
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
# Arena comparison (Circle vs Linear vs Open) for significantly precessing/
# recessing cells, animals pooled within each arena
# ============================================================================

ARENA_COMPARISON_METRICS = {
    'r_squared': 'R² (ρ²)',
    'fit_R': 'Fit R (residual MRL)',
    'abs_slope_deg_per_episode': 'Slope magnitude (deg/episode)',
    'phase_range_deg': 'Phase range (deg)',
}
TMI_COMPARISON_METRICS = {
    'TMI': 'Theta Modulation Index (TMI)',
}
# Theta-modulated (ThetaModulated) cells per recording session; see theta_mod_cells_by_session
THETA_MOD_CELL_METRICS = {
    'pct_theta_modulated': 'Theta-modulated cells (% of units per session)',
    'n_theta_modulated': 'Theta-modulated cells per session (n)',
}

# ---- Figure style for the arena-comparison plots, matching ArenaType_StatsComparison_v7.py
# (the Figure 3 place-cell plots): Arial, bold axis labels, small plain tick labels, thin
# black left/bottom axes with short outward ticks, arena-colored boxes with black outlines,
# black-edged points, significance stars, and a stats panel underneath. Applied only to the
# arena-comparison plots (via plt.rc_context in _arena_fig_style), so the per-unit
# diagnostic plots keep matplotlib's defaults.
PAL_MAGENTA = '#FA7FFA'
PAL_CYAN = '#7FFAFA'
PAL_BLUE = '#7FB3E5'
PAL_GRAY = '#7F7F7F'
PAL_GREEN = '#00C000'
PAL_YELLOW = '#FFD700'
PAL_BLACK = '#000000'
FIG_FONT = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']

FS_LABEL, FS_TITLE, FS_TICK, FS_LEGEND, FS_STATS = 14, 15, 11, 11, 9
FIG_STATS_LINE_IN = 0.2     # panel height per line of stats text (inches)
FIG_LEGEND_ROW_IN = 0.3     # panel height per row of legend entries (inches)
FIG_DPI = 500
FIG_AXIS_LW = 0.8           # spines, ticks, dashed lines
FIG_BOX_LW = 1.0            # box outlines, whiskers, caps; significance lines
FIG_MEDIAN_LW = 1.5         # box median
FIG_GROUP_W = 1.1           # x-axis width per arena (inches)
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
# Per-session-type colors, as in SessionType_StatsComparison_v6.py: shades of the
# arena's hue (blue = open arena, cyan = linear track, magenta = circular track).
PAL_DKBLUE = '#0066CC'
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
# Arena, session-type and animal labels don't overlap, so one lookup serves every comparison.
GROUP_COLORS = {**ARENA_COLORS, **SESSION_COLORS,
                **{a: {'face': c, 'edge': PAL_BLACK} for a, c in ANIMAL_COLORS.items()}}
_DEFAULT_COLOR = {'face': PAL_GRAY, 'edge': PAL_BLACK}
# Grouping column -> names of the omnibus tables' group-count / n-per-group columns
GROUP_COUNT_COLUMNS = {'Arena': ('N_arenas', 'N_per_arena'),
                       'SessionType': ('N_session_types', 'N_per_session_type'),
                       'Animal': ('N_animals', 'N_per_animal')}
_TEST_STAT_NAMES = {'Kruskal-Wallis': 'H', 'Mann-Whitney U': 'U',
                    'Mardia-Watson-Wheeler': 'W', 'Watson-Williams': 'F',
                    'Chi-square': 'χ²', 'Fisher exact': 'OR'}


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


def _bar_fig_width(n_groups: int) -> float:
    """Fixed width per arena, so boxes keep their proportions whatever the number of arenas."""
    return n_groups * FIG_GROUP_W + FIG_AXIS_PAD_W


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


def _desc_str(vals) -> str:
    """'mean ± SD | median [Q1, Q3]' for an array of values."""
    vals = np.asarray(vals, dtype=float)
    if not len(vals):
        return 'n/a'
    sd = f'{np.std(vals, ddof=1):.3g}' if len(vals) > 1 else 'n/a'
    return (f'mean {np.mean(vals):.3g} ± {sd} SD | median {np.median(vals):.3g} '
            f'[Q1 {np.percentile(vals, 25):.3g}, Q3 {np.percentile(vals, 75):.3g}]')


def _label(group: str) -> str:
    """Display label for plots: Linear session types are angles."""
    return f'{group}°' if str(group).isdigit() else str(group)


def _arena_test_lines(omnibus_df: pd.DataFrame, pairwise_df: pd.DataFrame,
                      group_col: str = 'Arena') -> list:
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
    for test in pairwise_df['Test'].unique() if len(pairwise_df) else []:
        stat_name = _TEST_STAT_NAMES.get(test, 'stat')
        lines.append(f'Post-hoc ({test}, Holm-Bonferroni corrected):')
        for _, r in pairwise_df[pairwise_df['Test'] == test].iterrows():
            stat = r['Statistic']
            stat_str = 'n/a' if not np.isfinite(stat) else f'{stat:{".1f" if stat_name == "U" else ".3f"}}'
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


def _arena_boxplot(ax, data, positions, arenas):
    """Arena- (or session-type-) colored boxes with black outlines, black median, no fliers."""
    bp = ax.boxplot(data, positions=positions, showfliers=False, patch_artist=True, widths=0.55,
                    manage_ticks=False,
                    boxprops=dict(linewidth=FIG_BOX_LW, color=PAL_BLACK),
                    whiskerprops=dict(linewidth=FIG_BOX_LW, color=PAL_BLACK),
                    capprops=dict(linewidth=FIG_BOX_LW, color=PAL_BLACK),
                    medianprops=dict(linewidth=FIG_MEDIAN_LW, color=PAL_BLACK),
                    zorder=2)
    for patch, arena in zip(bp['boxes'], arenas):
        patch.set_facecolor(GROUP_COLORS.get(arena, _DEFAULT_COLOR)['face'])
        patch.set_edgecolor(PAL_BLACK)
    return bp


def _draw_sig_brackets(ax, pos_by_arena: dict, pairwise_df, y_base: float, y_step: float, y_text: float,
                       group_col: str = 'Arena'):
    """For every pairwise comparison with Holm-corrected p < ALPHA, draw a
    horizontal line above the two groups (arenas or session types, read from
    the '<group_col>1' / '<group_col>2' columns) with the stars (* p<0.05,
    ** p<0.01, *** p<0.001) centered on it; narrower spans are stacked lowest.
    Returns the y of the highest line drawn, or None if no comparison was
    significant."""
    if pairwise_df is None or not len(pairwise_df):
        return None
    sig = []
    for _, r in pairwise_df.iterrows():
        p = r.get('p_holm')
        if p is None or pd.isna(p) or p >= ALPHA:
            continue
        g1, g2 = r[f'{group_col}1'], r[f'{group_col}2']
        if g1 not in pos_by_arena or g2 not in pos_by_arena:
            continue
        x1, x2 = sorted((pos_by_arena[g1], pos_by_arena[g2]))
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


def compare_arenas(df_sig: pd.DataFrame, metrics: dict = ARENA_COMPARISON_METRICS,
                   group_col: str = 'Arena', group_order=ARENA_LABELS):
    """Kruskal-Wallis omnibus test + pairwise Mann-Whitney U post-hoc tests
    (Holm-Bonferroni corrected within each metric), one per metric, across
    whichever groups of `group_order` are present in df_sig[group_col]
    (arenas by default; session types within one arena with
    group_col='SessionType'). Returns (omnibus_df, pairwise_df)."""
    arenas_present = [a for a in group_order if (df_sig[group_col] == a).sum() > 0]
    n_col, n_per_col = GROUP_COUNT_COLUMNS[group_col]

    omnibus_rows, pairwise_rows = [], []
    for col, label in metrics.items():
        groups = {a: df_sig.loc[df_sig[group_col] == a, col].dropna().to_numpy() for a in arenas_present}
        groups = {a: v for a, v in groups.items() if len(v) > 0}

        if len(groups) >= 2:
            kw_stat, kw_p = stats.kruskal(*groups.values())
        else:
            kw_stat, kw_p = np.nan, np.nan
        omnibus_rows.append({
            'Metric': label, 'Test': 'Kruskal-Wallis', n_col: len(groups), 'Statistic': kw_stat,
            'dof': len(groups) - 1 if len(groups) >= 2 else np.nan, 'p_value': kw_p,
            n_per_col: ', '.join(f'{a}={len(v)}' for a, v in groups.items())})

        metric_rows = []
        arena_names = list(groups.keys())
        for i in range(len(arena_names)):
            for j in range(i + 1, len(arena_names)):
                a1, a2 = arena_names[i], arena_names[j]
                v1, v2 = groups[a1], groups[a2]
                u_stat, p_pair = stats.mannwhitneyu(v1, v2, alternative='two-sided')
                metric_rows.append({'Metric': label, f'{group_col}1': a1, f'{group_col}2': a2,
                                    'N1': len(v1), 'N2': len(v2), 'Test': 'Mann-Whitney U',
                                    'Statistic': u_stat, 'p_value': p_pair})
        # Holm-Bonferroni correction within each metric
        for row, p_adj in zip(metric_rows, holm_adjust([r['p_value'] for r in metric_rows])):
            row['p_holm'] = p_adj
        pairwise_rows += metric_rows

    omnibus_df = pd.DataFrame(omnibus_rows,
                               columns=['Metric', 'Test', n_col, 'Statistic', 'dof', 'p_value', n_per_col])
    pairwise_df = pd.DataFrame(pairwise_rows,
                                columns=['Metric', f'{group_col}1', f'{group_col}2', 'N1', 'N2', 'Test',
                                         'Statistic', 'p_value', 'p_holm'])
    return omnibus_df, pairwise_df


@_arena_fig_style
def plot_arena_comparison(df_sig: pd.DataFrame, omnibus_df: pd.DataFrame, pairwise_df: pd.DataFrame,
                           out_prefix: Path, metrics: dict = ARENA_COMPARISON_METRICS,
                           population: str = '', group_col: str = 'Arena', group_order=ARENA_LABELS,
                           title_prefix: str = '') -> list[Path]:
    """One box plot + jittered points figure per metric (saved as
    <out_prefix>_<metric>.png, or <out_prefix>.png for a single metric), with
    stars over group pairs (arenas, or session types with group_col=
    'SessionType') whose Holm-corrected Mann-Whitney p < ALPHA, and
    descriptives + test results in a panel underneath. Returns the saved paths."""
    rng = np.random.default_rng(0)
    saved = []
    for col, label in metrics.items():
        present = [a for a in group_order if df_sig.loc[df_sig[group_col] == a, col].notna().any()]
        if not present:
            continue
        data = [df_sig.loc[df_sig[group_col] == a, col].dropna().to_numpy(dtype=float) for a in present]
        positions = np.arange(1, len(present) + 1)
        post = pairwise_df[pairwise_df['Metric'] == label]

        stats_lines = [f'Population: {population}'] if population else []
        stats_lines += [f'{_label(a)} (n={len(v)}): {_desc_str(v)}' for a, v in zip(present, data)]
        stats_lines += _arena_test_lines(omnibus_df[omnibus_df['Metric'] == label], post, group_col)
        fig_w = _bar_fig_width(len(present))
        stats_lines = _wrap_stats(stats_lines, fig_w)
        fig, ax, ax_info = _new_fig((fig_w, 5.5), stats_lines=stats_lines)

        _arena_boxplot(ax, data, positions, present)
        for pos, arena, vals in zip(positions, present, data):
            c = GROUP_COLORS.get(arena, _DEFAULT_COLOR)
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, s=22, facecolors=c['face'],
                       edgecolors=c['edge'], linewidths=0.5, zorder=3)

        # Significance lines above significantly different pairs of arenas.
        y_lo, y_hi = ax.get_ylim()
        span = y_hi - y_lo
        top = _draw_sig_brackets(ax, dict(zip(present, positions)), post,
                                 y_base=max(v.max() for v in data) + 0.06 * span,
                                 y_step=0.10 * span, y_text=0.03 * span, group_col=group_col)
        if top is not None:
            ax.set_ylim(y_lo, max(y_hi, top + 0.08 * span))
        _draw_stats(ax_info, stats_lines)

        ax.set_xticks(positions)
        ax.set_xticklabels([f'{_label(a)}\n(n={len(v)})' for a, v in zip(present, data)])
        ax.set_xlim(0.4, len(present) + 0.6)
        ax.set_ylabel(textwrap.fill(label, 30), fontsize=FS_LABEL, labelpad=8)
        _set_title(ax, f'{title_prefix}{label}')
        _style_axes(ax)
        fig.tight_layout()

        out_path = (out_prefix.with_name(f'{out_prefix.name}.png') if len(metrics) == 1
                    else out_prefix.with_name(f'{out_prefix.name}_{col}.png'))
        _save_fig(fig, out_path)
        saved.append(out_path)
    return saved


def compare_arenas_fi0_phase(df_fi0: pd.DataFrame, rng, col: str = 'phase_at_fi0_deg',
                             label: str = 'episode midpoint (firing index 0)',
                             group_col: str = 'Arena', group_order=ARENA_LABELS):
    """Theta phase at firing index 0 (or at the firing index whose phase is in `col`,
    described by `label`) across arenas (or, with group_col='SessionType',
    across session types), with circular statistics
    (a linear Kruskal-Wallis would treat 359 and 1 deg as far apart).
    Primary: Mardia-Watson-Wheeler uniform-scores test (nonparametric, tests
    for any difference in the circular distributions), permutation p. Also
    reported: Watson-Williams test (equal mean direction, von Mises
    assumption). Omnibus over all groups, then pairwise with Holm correction
    per test. Returns (descriptive_df, omnibus_df, pairwise_df)."""
    arenas_present = [a for a in group_order if (df_fi0[group_col] == a).sum() > 0]
    groups = {a: np.deg2rad(df_fi0.loc[df_fi0[group_col] == a, col].dropna().to_numpy())
              for a in arenas_present}
    groups = {a: v for a, v in groups.items() if len(v) > 0}
    n_col, n_per_col = GROUP_COUNT_COLUMNS[group_col]

    desc_rows = []
    for a, v in groups.items():
        n_prec = int(sum(phase_range_label(x) == 'precession' for x in np.rad2deg(v)))
        rayleigh_p, _ = circ_rtest(v)
        desc_rows.append({
            group_col: a, 'N': len(v), 'CircMean_deg': float(np.mod(np.rad2deg(circ_mean(v)), 360)),
            'MRL': circ_r(v), 'Rayleigh_p': rayleigh_p,
            'N_precession_range': n_prec, 'N_procession_range': len(v) - n_prec,
            'Pct_precession_range': 100.0 * n_prec / len(v)})
    desc_df = pd.DataFrame(desc_rows, columns=[group_col, 'N', 'CircMean_deg', 'MRL', 'Rayleigh_p',
                                               'N_precession_range', 'N_procession_range',
                                               'Pct_precession_range'])

    def run_tests(group_list):
        W, p_chi2, p_perm = mardia_watson_wheeler_test(group_list, rng)
        F, p_ww, kappa = watson_williams_test(group_list)
        return [dict(Test='Mardia-Watson-Wheeler', Statistic=W, p_value=p_perm, p_chi2_asymptotic=p_chi2),
                dict(Test='Watson-Williams', Statistic=F, p_value=p_ww, kappa=kappa)]

    omnibus_rows = []
    if len(groups) >= 2:
        for res in run_tests(list(groups.values())):
            omnibus_rows.append({'Metric': f'Theta phase at {label} (deg)', n_col: len(groups),
                                 n_per_col: ', '.join(f'{a}={len(v)}' for a, v in groups.items()),
                                 **res})
    omnibus_df = pd.DataFrame(omnibus_rows, columns=['Metric', 'Test', n_col, 'Statistic', 'p_value',
                                                     'p_chi2_asymptotic', 'kappa', n_per_col])

    pairwise_rows = []
    arena_names = list(groups.keys())
    for i in range(len(arena_names)):
        for j in range(i + 1, len(arena_names)):
            a1, a2 = arena_names[i], arena_names[j]
            for res in run_tests([groups[a1], groups[a2]]):
                pairwise_rows.append({f'{group_col}1': a1, f'{group_col}2': a2,
                                      'N1': len(groups[a1]), 'N2': len(groups[a2]), **res})
    pairwise_df = pd.DataFrame(pairwise_rows, columns=[f'{group_col}1', f'{group_col}2', 'N1', 'N2', 'Test',
                                                       'Statistic', 'p_value', 'p_holm', 'p_chi2_asymptotic',
                                                       'kappa'])
    for test in pairwise_df['Test'].unique():
        sel = pairwise_df['Test'] == test
        pairwise_df.loc[sel, 'p_holm'] = holm_adjust(pairwise_df.loc[sel, 'p_value'].to_numpy())
    return desc_df, omnibus_df, pairwise_df


@_arena_fig_style
def plot_fi0_phase_by_arena(df_fi0: pd.DataFrame, desc_df: pd.DataFrame, omnibus_df: pd.DataFrame,
                             pairwise_df: pd.DataFrame, out_path: Path, col: str = 'phase_at_fi0_deg',
                             label: str = 'episode midpoint (firing index 0)', group_col: str = 'Arena',
                             group_order=ARENA_LABELS, title_prefix: str = '',
                             pooled_within: str = 'arena'):
    """Box plot + jittered points per arena (or per session type, with
    group_col='SessionType') of the theta phase at the episode midpoint
    (or at the firing index whose phase is in `col`, described by `label`),
    on a single-cycle 0-360 deg axis, in the same style as plot_arena_comparison.
    Points are purple in the precession range, yellow in the procession range
    (faint matching background bands). Because phase is circular, each arena's
    box is computed on its phases unwrapped to within +/-180 deg of the arena's
    circular mean (so a cluster straddling 0/360 deg is not split), then drawn
    at -360/0/+360 deg offsets clipped to 0-360 so such a box wraps around.
    Stars: pairs whose Holm-corrected Mardia-Watson-Wheeler p < ALPHA."""
    present = [a for a in group_order if df_fi0.loc[df_fi0[group_col] == a, col].notna().any()]
    if not present:
        return
    phases = [np.mod(df_fi0.loc[df_fi0[group_col] == a, col].dropna().to_numpy(dtype=float), 360)
              for a in present]
    positions = np.arange(1, len(present) + 1)
    lo, hi = FI0_PRECESSION_RANGE_DEG

    stats_lines = [f'Units (one value per unit): {", ".join(FI0_PHASE_CLASSES)}; '
                   f'animals pooled within {pooled_within}']
    for _, r in desc_df.iterrows():
        stats_lines.append(f'{_label(r[group_col])} (n={int(r["N"])}): circular mean {r["CircMean_deg"]:.1f}°, '
                           f'MRL {r["MRL"]:.2f} | precession range {int(r["N_precession_range"])}/'
                           f'{int(r["N"])} ({r["Pct_precession_range"]:.1f} %)')
    stats_lines += _arena_test_lines(omnibus_df, pairwise_df, group_col)
    stats_lines.append('Mardia-Watson-Wheeler p from label permutations; '
                       'Watson-Williams assumes von Mises phases with a common concentration')
    fig_w = _bar_fig_width(len(present))
    stats_lines = _wrap_stats(stats_lines, fig_w)
    fig, ax, ax_info = _new_fig((fig_w, 7.0), legend_rows=2, stats_lines=stats_lines)

    ax.axhspan(lo, hi, color=FI0_RANGE_COLORS['precession'], alpha=0.08, lw=0, zorder=0)
    for y0, y1 in ((0, lo), (hi, 360)):
        ax.axhspan(y0, y1, color=FI0_RANGE_COLORS['procession'], alpha=0.12, lw=0, zorder=0)

    # boxes are repeated every 360 deg and clipped to the 0-360 phase axis, so a
    # box straddling 0/360 deg wraps around instead of being cut off
    clip_rect = Rectangle((0, 0), len(present) + 1, 360, transform=ax.transData)
    for pos, arena, ph in zip(positions, present, phases):
        mu = np.mod(np.rad2deg(circ_mean(np.deg2rad(ph))), 360)
        unwrapped = mu + (np.mod(ph - mu + 180, 360) - 180)
        for off in (-360, 0, 360):
            bp = _arena_boxplot(ax, [unwrapped + off], [pos], [arena])
            for artist in (*bp['boxes'], *bp['whiskers'], *bp['caps'], *bp['medians']):
                artist.set_clip_path(clip_rect)

    rng = np.random.default_rng(0)
    for pos, ph in zip(positions, phases):
        jitter = rng.uniform(-0.12, 0.12, size=len(ph))
        faces = [FI0_RANGE_COLORS[phase_range_label(p)] for p in ph]
        ax.scatter(pos + jitter, ph, s=22, facecolors=faces, edgecolors=PAL_BLACK,
                   linewidths=0.5, zorder=3)

    ax.set_ylim(0, 360)
    top = _draw_sig_brackets(ax, dict(zip(present, positions)),
                             pairwise_df[pairwise_df['Test'] == 'Mardia-Watson-Wheeler'],
                             y_base=360 + 0.05 * 360, y_step=0.08 * 360, y_text=0.025 * 360,
                             group_col=group_col)
    if top is not None:
        ax.set_ylim(0, top + 0.06 * 360)
    ax.set_yticks(np.arange(0, 361, 90))

    handles = [Line2D([0], [0], marker='o', linestyle='None', markersize=8,
                      markerfacecolor=FI0_RANGE_COLORS['precession'], markeredgecolor=PAL_BLACK,
                      markeredgewidth=0.5,
                      label=f'precession range ({lo:g}–{hi:g}°)'),
               Line2D([0], [0], marker='o', linestyle='None', markersize=8,
                      markerfacecolor=FI0_RANGE_COLORS['procession'], markeredgecolor=PAL_BLACK,
                      markeredgewidth=0.5,
                      label=f'procession range (0–{lo:g}°, {hi:g}–360°)')]
    ax_info.legend(handles=handles, loc='upper center', fontsize=FS_LEGEND - 2, ncol=1)
    _draw_stats(ax_info, stats_lines)

    ax.set_xticks(positions)
    ax.set_xticklabels([f'{_label(a)}\n(n={len(ph)})' for a, ph in zip(present, phases)])
    ax.set_xlim(0.4, len(present) + 0.6)
    ax.set_ylabel(textwrap.fill(f'Theta phase at {label} (deg)', 30), fontsize=FS_LABEL, labelpad=8)
    _set_title(ax, f'{title_prefix}Theta phase at {label}')
    _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, out_path)


def theta_mod_cells_by_session(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """One row per recording session (FolderPath): number of units recorded,
    number with enough spikes for the TMI test (n_spikes_theta >= MIN_SPIKES_FOR_TMI),
    number theta-modulated (ThetaModulated) and their percentage of ALL units
    recorded in the session (units too sparse for the TMI test count as not
    theta-modulated, as in the Summary sheet). Cells are counted once each,
    not once per anything else."""
    cells = df.drop_duplicates(['FolderPath', 'Unit'])
    cells = cells.assign(_tm=cells['ThetaModulated'] == True,                  # noqa: E712
                         _tested=cells['n_spikes_theta'] >= MIN_SPIKES_FOR_TMI)
    out = (cells.groupby('FolderPath', sort=False)
           .agg(Session=('Session', 'first'), **{group_col: (group_col, 'first')},
                n_units=('Unit', 'size'), n_units_TMI_tested=('_tested', 'sum'),
                n_theta_modulated=('_tm', 'sum'))
           .reset_index())
    out['pct_theta_modulated'] = 100.0 * out['n_theta_modulated'] / out['n_units']
    return out[['Session', 'FolderPath', group_col, 'n_units', 'n_units_TMI_tested',
                'n_theta_modulated', 'pct_theta_modulated']]


def compare_theta_mod_proportion(df: pd.DataFrame, group_col: str, group_order):
    """v27: theta-modulated (ThetaModulated) units pooled per group (arena or session type),
    counted once per unit. Returns (counts_df, omnibus_df, pairwise_df): counts_df has, per group,
    n_units (all units recorded), n_units_TMI_tested (n_spikes_theta >= MIN_SPIKES_FOR_TMI),
    n_theta_modulated, n_not_theta_modulated (of all units) and their percentages of all / tested
    units; omnibus = chi-square test of independence (theta-modulated vs not x group), pairwise =
    Fisher exact tests (statistic = odds ratio), Holm-corrected. Tests need >= 2 groups."""
    cells = df.drop_duplicates(['FolderPath', 'Unit'])
    rows = []
    for g in group_order:
        sub = cells[cells[group_col] == g]
        if sub.empty:
            continue
        n_all = len(sub)
        n_tested = int((sub['n_spikes_theta'] >= MIN_SPIKES_FOR_TMI).sum())
        n_tm = int((sub['ThetaModulated'] == True).sum())                      # noqa: E712
        rows.append({group_col: g, 'n_units': n_all, 'n_units_TMI_tested': n_tested,
                     'n_theta_modulated': n_tm, 'n_not_theta_modulated': n_all - n_tm,
                     'pct_of_units': 100.0 * n_tm / n_all,
                     'pct_of_TMI_tested': 100.0 * n_tm / n_tested if n_tested else np.nan})
    counts_df = pd.DataFrame(rows, columns=[group_col, 'n_units', 'n_units_TMI_tested', 'n_theta_modulated',
                                            'n_not_theta_modulated', 'pct_of_units', 'pct_of_TMI_tested'])
    n_col, n_per_col = GROUP_COUNT_COLUMNS[group_col]
    omnibus_rows, pairwise_rows = [], []
    if len(counts_df) >= 2:
        table = counts_df[['n_theta_modulated', 'n_not_theta_modulated']].to_numpy().T
        # chi2_contingency fails on an all-zero row (no / every unit theta-modulated in every group)
        chi2, p, dof = (stats.chi2_contingency(table)[:3] if (table.sum(axis=1) > 0).all()
                        else (np.nan, np.nan, len(counts_df) - 1))
        omnibus_rows.append({'Metric': 'Theta-modulated units (of all units)', 'Test': 'Chi-square',
                             n_col: len(counts_df), 'Statistic': chi2, 'dof': dof, 'p_value': p,
                             n_per_col: ', '.join(f'{r[group_col]}={r["n_units"]}' for _, r in counts_df.iterrows())})
        recs = counts_df.to_dict('records')
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                r1, r2 = recs[i], recs[j]
                odds, p_pair = stats.fisher_exact([[r1['n_theta_modulated'], r1['n_not_theta_modulated']],
                                                   [r2['n_theta_modulated'], r2['n_not_theta_modulated']]])
                pairwise_rows.append({'Metric': 'Theta-modulated units (of all units)',
                                      f'{group_col}1': r1[group_col], f'{group_col}2': r2[group_col],
                                      'N1': r1['n_units'], 'N2': r2['n_units'], 'Test': 'Fisher exact',
                                      'Statistic': odds, 'p_value': p_pair})
        for row, p_adj in zip(pairwise_rows, holm_adjust([r['p_value'] for r in pairwise_rows])):
            row['p_holm'] = p_adj
    omnibus_df = pd.DataFrame(omnibus_rows,
                              columns=['Metric', 'Test', n_col, 'Statistic', 'dof', 'p_value', n_per_col])
    pairwise_df = pd.DataFrame(pairwise_rows,
                               columns=['Metric', f'{group_col}1', f'{group_col}2', 'N1', 'N2', 'Test',
                                        'Statistic', 'p_value', 'p_holm'])
    return counts_df, omnibus_df, pairwise_df


@_arena_fig_style
def plot_theta_mod_proportion(counts_df: pd.DataFrame, omnibus_df: pd.DataFrame, pairwise_df: pd.DataFrame,
                              out_path: Path, group_col: str = 'Arena', title_prefix: str = '',
                              pooled: str = 'animals pooled within arena'):
    """v27: left, % of all units that are theta-modulated per group (bars in the group colours,
    'n theta-modulated / n units' above each, stars over pairs with Holm-corrected Fisher
    p < ALPHA); right, absolute numbers as stacked bars (theta-modulated filled, not
    theta-modulated hatched). Counts and tests in a panel underneath."""
    groups = counts_df[group_col].tolist()
    positions = np.arange(1, len(groups) + 1)
    faces = [GROUP_COLORS.get(g, _DEFAULT_COLOR)['face'] for g in groups]
    n_tm, n_not = counts_df['n_theta_modulated'].to_numpy(), counts_df['n_not_theta_modulated'].to_numpy()
    n_all, n_tested = counts_df['n_units'].to_numpy(), counts_df['n_units_TMI_tested'].to_numpy()
    pct = counts_df['pct_of_units'].to_numpy()

    crit = f'TMI shuffle{" + Rayleigh" if REQUIRE_RAYLEIGH_FOR_THETA_MOD else ""} p < {ALPHA:g}'
    stats_lines = [f'Theta-modulated = {crit}; one value per unit; {pooled}']
    stats_lines += [f'{_label(r[group_col])}: {r["n_theta_modulated"]}/{r["n_units"]} units '
                    f'({r["pct_of_units"]:.1f} %) | {r["n_theta_modulated"]}/{r["n_units_TMI_tested"]} '
                    f'TMI-tested (>= {MIN_SPIKES_FOR_TMI} spikes; {r["pct_of_TMI_tested"]:.1f} %)'
                    for _, r in counts_df.iterrows()]
    if len(groups) > 1:
        stats_lines.append(f'All: {n_tm.sum()}/{n_all.sum()} units ({100.0 * n_tm.sum() / n_all.sum():.1f} %) | '
                           f'{n_tm.sum()}/{n_tested.sum()} TMI-tested')
    stats_lines += _arena_test_lines(omnibus_df, pairwise_df, group_col)
    fig_w = 2 * _bar_fig_width(len(groups)) + 0.8
    stats_lines = _wrap_stats(stats_lines, fig_w)
    panel_h = FIG_LEGEND_ROW_IN + len(stats_lines) * FIG_STATS_LINE_IN + 0.2
    fig = plt.figure(figsize=(fig_w, 5.5 + panel_h))
    gs = fig.add_gridspec(2, 2, height_ratios=[5.5, panel_h])
    ax_p, ax_n = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.axis('off')

    ax_p.bar(positions, pct, width=0.6, color=faces, edgecolor=PAL_BLACK, linewidth=FIG_BOX_LW, zorder=2)
    for x, p_, a, b in zip(positions, pct, n_tm, n_all):
        ax_p.text(x, p_ + 2, f'{p_:.1f}%\n{a}/{b}', ha='center', va='bottom', fontsize=FS_TICK)
    top = _draw_sig_brackets(ax_p, dict(zip(groups, positions)), pairwise_df,
                             y_base=pct.max() + 18, y_step=10, y_text=3, group_col=group_col)
    ax_p.set_ylim(0, max(100.0, (top if top is not None else pct.max() + 14) + 8))
    ax_p.set_ylabel('Theta-modulated units (% of all units)', fontsize=FS_LABEL - 2, labelpad=8)
    _set_title(ax_p, f'{title_prefix}Theta-modulated proportion')

    ax_n.bar(positions, n_tm, width=0.6, color=faces, edgecolor=PAL_BLACK, linewidth=FIG_BOX_LW, zorder=2)
    ax_n.bar(positions, n_not, width=0.6, bottom=n_tm, color='white', edgecolor=PAL_BLACK, hatch='///',
             linewidth=FIG_BOX_LW, zorder=2)
    y_top = n_all.max() * 1.12 + 1
    for x, a, b in zip(positions, n_tm, n_not):
        if a >= 0.04 * y_top:
            ax_n.text(x, a / 2, str(a), ha='center', va='center', fontsize=FS_TICK, fontweight='bold', zorder=3)
        elif a:     # too short to hold its label: written beside the bar
            ax_n.text(x + 0.33, 0, str(a), ha='left', va='bottom', fontsize=FS_TICK, fontweight='bold', zorder=3)
        ax_n.text(x, a + b, str(a + b), ha='center', va='bottom', fontsize=FS_TICK, zorder=3)
    ax_n.set_ylim(0, y_top)
    ax_n.set_ylabel('Units (n)', fontsize=FS_LABEL - 2, labelpad=8)
    _set_title(ax_n, f'{title_prefix}Theta-modulated units (n)')

    for ax in (ax_p, ax_n):
        ax.set_xticks(positions)
        ax.set_xticklabels([f'{_label(g)}\n(n={n})' for g, n in zip(groups, n_all)])
        ax.set_xlim(0.4, len(groups) + 0.6)
        _style_axes(ax)
    ax_info.legend(handles=[Patch(facecolor=PAL_GRAY, edgecolor=PAL_BLACK, label='theta-modulated'),
                            Patch(facecolor='white', edgecolor=PAL_BLACK, hatch='///',
                                  label='not theta-modulated (incl. too few spikes for TMI)')],
                   loc='upper center', ncol=2, fontsize=FS_LEGEND - 2)
    _draw_stats(ax_info, stats_lines)
    fig.tight_layout()
    _save_fig(fig, out_path)


def run_theta_mod_proportion(df: pd.DataFrame, group_col: str, group_order, out_path: Path,
                             pooled: str, title_prefix: str = '') -> dict[str, pd.DataFrame]:
    """Proportion and number of theta-modulated units pooled per group: tests
    (compare_theta_mod_proportion) and plot (plot_theta_mod_proportion, saved to
    out_path; `pooled` says what the plot's footer reports as pooled). Returns
    {table key: DataFrame}; empty if no group holds a unit."""
    counts_df, omnibus_df, pairwise_df = compare_theta_mod_proportion(df, group_col, group_order)
    if counts_df.empty:
        return {}
    plot_theta_mod_proportion(counts_df, omnibus_df, pairwise_df, out_path, group_col=group_col,
                              title_prefix=title_prefix, pooled=pooled)
    print('Theta-modulated units: ' + '; '.join(
        f'{_label(r[group_col])} {r["n_theta_modulated"]}/{r["n_units"]} ({r["pct_of_units"]:.1f} %)'
        for _, r in counts_df.iterrows()) + f' -- plot saved: {out_path.name}')
    return {'ThetaModCells_Pooled': counts_df, 'ThetaModCells_PoolOmni': omnibus_df,
            'ThetaModCells_PoolPair': pairwise_df}


def run_group_comparisons(df: pd.DataFrame, group_col: str, group_order, out_dir: Path,
                          file_prefix: str, group_word: str, title_prefix: str = '') -> dict[str, pd.DataFrame]:
    """Every between-group comparison for the cells in `df`, grouped by
    df[group_col] in `group_order` order (the arenas, or one arena's session
    types), animals pooled within each group:
      - r^2, fit_R, |slope| and phase range of significantly precessing/recessing
        cells (Kruskal-Wallis + Mann-Whitney U), plotted to
        out_dir/<file_prefix>_PhasePrecession_<metric>.png
      - TMI of significantly theta-modulated cells, out_dir/<file_prefix>_TMI.png
      - % and number of theta-modulated cells per recording session
        (theta_mod_cells_by_session; Kruskal-Wallis + Mann-Whitney U, one value
        per session), out_dir/<file_prefix>_ThetaModCells_<metric>.png
      - v27: % and absolute number of theta-modulated units pooled per group
        (compare_theta_mod_proportion; chi-square + pairwise Fisher exact), plotted
        even for a single group, out_dir/<file_prefix>_ThetaModProportion.png
      - theta phase at each FIRING_INDEX_PHASE_POINTS firing index of the
        FI0_PHASE_CLASSES cells (circular tests),
        out_dir/<file_prefix>_PhaseAt<file tag>.png
    A comparison with fewer than 2 groups holding data is skipped. Returns
    {table key: DataFrame} with keys 'Omnibus', 'Pairwise', 'TMI_Omnibus',
    'TMI_Pairwise', 'ThetaModCells_BySession' / '_Omnibus' / '_Pairwise' and
    'PhaseAt<sheet tag>_By' / '_Omnibus' / '_Pairwise'
    (empty tables left out); _comparison_sheet_name gives their sheet names."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pooled = f'animals pooled within {group_word}'
    tables = {}

    def groups_with_data(sub):
        return [g for g in group_order if (sub[group_col] == g).any()]

    def skip_msg(present, cells, what):
        print(f'Fewer than 2 {group_word}s with {cells} found '
              f'({", ".join(map(_label, present)) or "none"}) -- skipping {what} comparison.')

    # ---- r^2 / fit_R / |slope| / phase range (significantly precessing/recessing cells) ----
    sig_mask = (df['is_precessing'] == True) | (df['is_recessing'] == True)  # noqa: E712
    df_sig = df.loc[sig_mask].copy()
    df_sig['abs_slope_deg_per_episode'] = df_sig['slope_deg_per_episode'].abs()
    present = groups_with_data(df_sig)
    if len(present) >= 2:
        omnibus_df, pairwise_df = compare_arenas(df_sig, group_col=group_col, group_order=group_order)
        tables['Omnibus'], tables['Pairwise'] = omnibus_df, pairwise_df
        plot_paths = plot_arena_comparison(
            df_sig, omnibus_df, pairwise_df, out_dir / f'{file_prefix}_PhasePrecession',
            population=f'significantly precessing/recessing units (one value per unit); {pooled}',
            group_col=group_col, group_order=group_order, title_prefix=title_prefix)
        print(f'Phase-precession comparison ({", ".join(map(_label, present))}) plots saved: '
              f'{", ".join(p.name for p in plot_paths)}')
    else:
        skip_msg(present, 'significantly precessing/recessing cells', 'phase-precession')

    # ---- TMI (significantly theta-modulated cells) ----
    df_tmi_sig = df.loc[df['ThetaModulated'] == True].drop_duplicates(['FolderPath', 'Unit']).copy()  # noqa: E712
    present = groups_with_data(df_tmi_sig)
    if len(present) >= 2:
        omnibus_df, pairwise_df = compare_arenas(df_tmi_sig, TMI_COMPARISON_METRICS,
                                                 group_col=group_col, group_order=group_order)
        tables['TMI_Omnibus'], tables['TMI_Pairwise'] = omnibus_df, pairwise_df
        plot_paths = plot_arena_comparison(
            df_tmi_sig, omnibus_df, pairwise_df, out_dir / f'{file_prefix}_TMI', TMI_COMPARISON_METRICS,
            population=(f'significantly theta-modulated cells '
                        f'(TMI shuffle{" + Rayleigh" if REQUIRE_RAYLEIGH_FOR_THETA_MOD else ""}); {pooled}'),
            group_col=group_col, group_order=group_order, title_prefix=title_prefix)
        print(f'TMI comparison ({", ".join(map(_label, present))}) plot saved: '
              f'{", ".join(p.name for p in plot_paths)}')
    else:
        skip_msg(present, 'significantly theta-modulated cells', 'TMI')

    # ---- % and number of theta-modulated cells (one value per recording session) ----
    df_tm_ses = theta_mod_cells_by_session(df, group_col)
    tables['ThetaModCells_BySession'] = df_tm_ses
    present = groups_with_data(df_tm_ses)
    if len(present) >= 2:
        omnibus_df, pairwise_df = compare_arenas(df_tm_ses, THETA_MOD_CELL_METRICS,
                                                 group_col=group_col, group_order=group_order)
        tables['ThetaModCells_Omnibus'], tables['ThetaModCells_Pairwise'] = omnibus_df, pairwise_df
        pooled_counts = []
        for g in present:
            sub = df_tm_ses[df_tm_ses[group_col] == g]
            n_tm, n_all = int(sub['n_theta_modulated'].sum()), int(sub['n_units'].sum())
            pooled_counts.append(f'{_label(g)} {n_tm}/{n_all} units ({100.0 * n_tm / n_all:.1f} %) '
                                 f'in {len(sub)} sessions')
        plot_paths = plot_arena_comparison(
            df_tm_ses, omnibus_df, pairwise_df, out_dir / f'{file_prefix}_ThetaModCells',
            THETA_MOD_CELL_METRICS,
            population=(f'recording sessions (one value per session); theta-modulated = TMI shuffle'
                        f'{" + Rayleigh" if REQUIRE_RAYLEIGH_FOR_THETA_MOD else ""} p < {ALPHA:g}, '
                        f'% of all units recorded in the session; {pooled}. '
                        f'Totals: {"; ".join(pooled_counts)}'),
            group_col=group_col, group_order=group_order, title_prefix=title_prefix)
        print(f'Theta-modulated cell count/proportion comparison ({", ".join(map(_label, present))}) '
              f'plots saved: {", ".join(p.name for p in plot_paths)}')
    else:
        skip_msg(present, 'recording sessions', 'theta-modulated cell count/proportion')

    # ---- v27: proportion and absolute number of theta-modulated units, pooled per group ----
    tables.update(run_theta_mod_proportion(df, group_col, group_order,
                                           out_dir / f'{file_prefix}_ThetaModProportion.png',
                                           pooled, title_prefix))

    # ---- Theta phase at firing index -1 (firing onset) / 0 (episode midpoint) / +1 (firing offset) ----
    for x, key, label, file_tag, sheet_tag in FIRING_INDEX_PHASE_POINTS:
        col = f'phase_at_{key}_deg'
        df_pt = df.loc[df['PrecessionClass'].isin(FI0_PHASE_CLASSES) & df[col].notna()].copy()
        present = groups_with_data(df_pt)
        if len(present) >= 2:
            desc_df, omnibus_df, pairwise_df = compare_arenas_fi0_phase(
                df_pt, np.random.default_rng(RANDOM_SEED), col=col, label=label,
                group_col=group_col, group_order=group_order)
            tables[f'PhaseAt{sheet_tag}_By'] = desc_df
            tables[f'PhaseAt{sheet_tag}_Omnibus'] = omnibus_df
            tables[f'PhaseAt{sheet_tag}_Pairwise'] = pairwise_df
            pt_plot_path = out_dir / f'{file_prefix}_PhaseAt{file_tag}.png'
            plot_fi0_phase_by_arena(df_pt, desc_df, omnibus_df, pairwise_df, pt_plot_path,
                                    col=col, label=label, group_col=group_col, group_order=group_order,
                                    title_prefix=title_prefix, pooled_within=group_word)
            print(f'Phase-at-firing-index-{x:g} comparison ({", ".join(map(_label, present))}) '
                  f'plot saved: {pt_plot_path.name}')
        else:
            skip_msg(present, f'{"/".join(FI0_PHASE_CLASSES)} cells', f'phase-at-firing-index-{x:g}')

    # ---- Offset - onset phase of each unit with enough onset and offset spikes (circular tests, tables only) ----
    df_shift = df.loc[df['OnOff_offset_minus_onset_deg'].notna() & (df['OnOff_n_onset'] >= ONOFF_MIN_SPIKES_PER_UNIT)
                      & (df['OnOff_n_offset'] >= ONOFF_MIN_SPIKES_PER_UNIT)].copy()
    df_shift['offset_minus_onset_360'] = np.mod(df_shift['OnOff_offset_minus_onset_deg'], 360)
    present = groups_with_data(df_shift)
    if len(present) >= 2:
        desc_df, omnibus_df, pairwise_df = compare_arenas_fi0_phase(
            df_shift, np.random.default_rng(RANDOM_SEED), col='offset_minus_onset_360', label='offset - onset',
            group_col=group_col, group_order=group_order)
        tables['OnOffShift_By'], tables['OnOffShift_Omnibus'], tables['OnOffShift_Pairwise'] = desc_df, omnibus_df, pairwise_df
    else:
        skip_msg(present, f'>= {ONOFF_MIN_SPIKES_PER_UNIT} onset and offset spikes', 'offset - onset phase')

    return {k: t for k, t in tables.items() if not t.empty}


def _comparison_sheet_name(group_col: str, key: str) -> str:
    """Excel sheet name (<= 31 characters) for a run_group_comparisons table.
    The arena comparison keeps v18's sheet names ('ArenaComparison_Omnibus',
    'ArenaComparison_TMI_Pairwise', 'PhaseAtFI0_ByArena', 'ThetaModCells_Pairwise',
    ...); session-type tables get a 'SesType_' prefix ('SesType_Omnibus',
    'SesType_TMI_Pairwise', 'SesType_PhaseAtFI0_BySesType', ...) and per-animal
    tables an 'Animal_' prefix ('Animal_ThetaModCells_Pooled', ...)."""
    if group_col == 'Animal':
        return f'Animal_{key}'
    if key.endswith('_By'):
        key += 'Arena' if group_col == 'Arena' else 'SesType'
    if group_col != 'Arena':
        return f'SesType_{key}'
    # 'ArenaComparison_ThetaModCells_...' would exceed Excel's 31-character limit
    return key if key.startswith(('PhaseAt', 'ThetaModCells', 'OnOffShift')) else f'ArenaComparison_{key}'


PAL_EDMIN_REJECT = '#C44E52'


@_arena_fig_style
def plot_edmin_epochs(edmin_df: pd.DataFrame, out_path: Path):
    """ED_min vs. normalised ACG peak range for every epoch that got an ACG fit
    (i.e. passed the delta/theta + MAD artifact screening), pooled across all
    sessions -- style of plot_edmin_vs_peakrangenorm in
    ACG_theta_continuity_TT_ThetaVsSpeed_v4_FerretOLS.py. Each point is coloured
    by its animal; epochs rejected by the ED_min gate (ED_min >= ACG_ED_MIN_THRESH)
    get a red outline and sit in a red-shaded region right of the dashed threshold.
    Pearson r / Spearman rho and a least-squares line are computed over all fitted
    epochs (not filtered on the threshold, which would truncate the high-ED_min
    epochs). The panel underneath gives, per animal and pooled, the percentage of
    fitted epochs accepted and rejected.
    """
    fitted = edmin_df[edmin_df['Status'] != 'rejected_prescreen']
    if fitted.empty:
        print('No epochs with an ACG fit -- ED_min plot skipped.')
        return None
    thr = ACG_ED_MIN_THRESH
    animals = sorted(fitted['Animal'].unique())
    cmap = plt.get_cmap('tab10' if len(animals) <= 10 else 'tab20')
    colors = {a: ANIMAL_COLORS.get(a, cmap(i % cmap.N)) for i, a in enumerate(animals)}

    def _acc_rej_line(label, sub):
        n = len(sub)
        n_acc = int((sub['Status'] == 'accepted').sum())
        n_rej = n - n_acc
        return (f'{label}: accepted {100 * n_acc / n:.1f}% ({n_acc}/{n}) | '
                f'rejected {100 * n_rej / n:.1f}% ({n_rej}/{n})')

    stats_lines = [f'Epochs with an ACG fit (passed delta/theta + artifact screening); '
                   f'rejected = ED_min ≥ {thr:g}']
    stats_lines += [_acc_rej_line(a, fitted[fitted['Animal'] == a]) for a in animals]
    if len(animals) > 1:
        stats_lines.append(_acc_rej_line('All animals', fitted))
    stats_lines = _wrap_stats(stats_lines, 7.0)

    fig, ax, ax_info = _new_fig((7.0, 5.5), stats_lines=stats_lines)
    for a in animals:
        sub = fitted[fitted['Animal'] == a]
        acc = sub[sub['Status'] == 'accepted']
        rej = sub[sub['Status'] == 'rejected_EDmin']
        ax.scatter(acc['ED_min'], acc['peakrangenorm'], s=10, color=colors[a], alpha=0.4,
                   edgecolor='none', zorder=2)
        ax.scatter(rej['ED_min'], rej['peakrangenorm'], s=18, color=colors[a], alpha=0.8,
                   edgecolor=PAL_EDMIN_REJECT, linewidth=0.8, zorder=3)

    xy = fitted.dropna(subset=['ED_min', 'peakrangenorm'])
    x, y = xy['ED_min'].to_numpy(float), xy['peakrangenorm'].to_numpy(float)
    has_fit = len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0
    if has_fit:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.array([x.min(), x.max()])
        ax.plot(x_line, slope * x_line + intercept, color=PAL_BLACK, lw=FIG_BOX_LW * 1.5, zorder=4)
        r_p, p_p = stats.pearsonr(x, y)
        r_s, p_s = stats.spearmanr(x, y)
        ax.text(0.02, 0.02, f'Pearson r = {r_p:.3f} (p = {p_p:.2e})\n'
                            f'Spearman ρ = {r_s:.3f} (p = {p_s:.2e})\nn = {len(x)} epochs',
                transform=ax.transAxes, ha='left', va='bottom', fontsize=FS_STATS,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='0.7'), zorder=5)

    x_lo, x_hi = ax.get_xlim()
    ax.axvspan(thr, max(x_hi, thr), color=PAL_EDMIN_REJECT, alpha=0.08, lw=0, zorder=0)
    ax.axvline(thr, color=PAL_EDMIN_REJECT, ls='--', lw=FIG_AXIS_LW, zorder=1)
    ax.set_xlim(x_lo, max(x_hi, thr))

    handles = [Line2D([], [], ls='', marker='o', markersize=6, color=colors[a],
                      label=f'{a} (n={int((fitted["Animal"] == a).sum())})') for a in animals]
    handles += [Line2D([], [], ls='', marker='o', markersize=6, markerfacecolor='white',
                       markeredgecolor=PAL_EDMIN_REJECT, markeredgewidth=1.2,
                       label=f'ED_min rejected (≥ {thr:g})'),
                Line2D([], [], color=PAL_EDMIN_REJECT, ls='--', lw=FIG_AXIS_LW,
                       label=f'ED_min threshold ({thr:g})')]
    if has_fit:
        handles.append(Line2D([], [], color=PAL_BLACK, lw=FIG_BOX_LW * 1.5, label='linear fit'))
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=FS_LEGEND - 2)

    ax.set_xlabel('ED_min (distance to closest reference sinusoid)', fontsize=FS_LABEL - 2)
    ax.set_ylabel('ACG peak range (norm.)', fontsize=FS_LABEL - 2)
    ax.tick_params(labelsize=FS_TICK)
    _set_title(ax, 'ACG theta epochs: ED_min vs. peak range')
    _draw_stats(ax_info, stats_lines)
    _save_fig(fig, out_path)
    return out_path



def plot_fit_r_shuffle(results, title, out_path: Path):
    """Step 3 gate (visualization): left, LFP phase vs firing index with the fitted line; right, the fit_R
    shuffle-test null distribution with the observed fit_R and the null's (1 - ALPHA) percentile. The cell's
    z-test is considered only if p < ALPHA."""
    is_sig = results['fit_R_Significant']
    color = 'red' if is_sig else '#4D4D4D'
    verdict = 'PASS' if is_sig else 'FAIL'
    fig, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f'{title}\nfit R={results["fit_R"]:.3f} | shuffle p={results["fit_R_shuffle_p"]:.3g} '
                 f'| {verdict}', fontsize=10, color=color)
    _draw_firing_index_phase(ax_p, results)
    _draw_fit_r_null(ax_s, results['fit_R'], is_sig, results['shuffle_fit_r'])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _draw_spike_train_raster(ax, dyn, analysis):
    """Harris Fig 2a-c: spike phase (duplicated over 0-720 deg) against theta cycle around the unit's longest firing
    episode (or, with no episode, its first labelled spike), onset red, offset blue, others black; vertical lines
    every cycle."""
    label, cycle = analysis['label'], dyn['cycle']
    if analysis['episodes']:
        first, last = max(analysis['episodes'], key=lambda e: cycle[e[1]] - cycle[e[0]])
        what = 'longest firing episode'
    elif (label != 0).any():
        first = last = int(np.flatnonzero(label != 0)[0])
        what = 'first labelled spike (no complete episode)'
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'no onset or offset spikes', ha='center', va='center', fontsize=10)
        return
    c0, c1 = cycle[first] - ONOFF_WINDOW_CYCLES, cycle[last] + ONOFF_WINDOW_CYCLES
    sel = (cycle >= c0) & (cycle <= c1) & np.isfinite(dyn['phase'])
    x, ph, lab = cycle[sel] - c0, np.rad2deg(dyn['phase'][sel]), label[sel]
    for c in np.arange(np.ceil(c0), c1):
        ax.axvline(c - c0, color='0.85', linewidth=0.5, zorder=0)
    for value, color, size in ((0, 'k', 12), (ONSET, FIRING_ONSET_COLOR, 26), (OFFSET, FIRING_OFFSET_COLOR, 26)):
        ax.scatter(np.r_[x[lab == value], x[lab == value]], np.r_[ph[lab == value], ph[lab == value] + 360], s=size,
                   color=color, zorder=2)
    ax.set_xlim(0, c1 - c0)
    ax.set_ylim(0, 720)
    ax.set_xlabel('Theta cycles')
    ax.set_ylabel('Spike phase (deg)')
    ax.set_title(f'Spike train around the {what}', fontsize=9)


def _draw_onset_offset_hist(ax, dyn, analysis, st):
    """Onset and offset spikes' theta-phase histograms (duplicated over 0-720 deg) with circular means."""
    edges = np.arange(0, 720 + PHASE_BIN_SIZE_DEG, PHASE_BIN_SIZE_DEG)
    for lab, name, color in ((ONSET, 'onset', FIRING_ONSET_COLOR), (OFFSET, 'offset', FIRING_OFFSET_COLOR)):
        ph = np.rad2deg(dyn['phase'][analysis['label'] == lab])
        if not len(ph):
            continue
        ax.hist(np.r_[ph, ph + 360], bins=edges, histtype='stepfilled', alpha=0.45, color=color, edgecolor='k', linewidth=0.8,
                label=f'{name} (n={len(ph)}): mean {st[f"OnOff_{name}_phase_deg"]:.0f}°, MRL {st[f"OnOff_{name}_MRL"]:.2f}')
        for off in (0, 360):
            ax.axvline(st[f'OnOff_{name}_phase_deg'] + off, color=color, linewidth=2, linestyle='--')
    ax.set_xlim(0, 720)
    ax.set_xticks(np.arange(0, 721, 90))
    ax.set_xlabel('LFP theta phase (deg)')
    ax.set_ylabel('Spikes')
    ax.set_title(f'Offset - onset = {st["OnOff_offset_minus_onset_deg"]:.0f}°; MWW p (rotation null) = '
                 f'{st["OnOff_MWW_p"]:.3g}', fontsize=9)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc='upper right')


def _draw_phase_vs_rate(ax, dyn, anova_p):
    """Harris Fig 1d for one unit: circular mean phase and its 95% CI per rate group."""
    usable = dyn['in_theta']
    groups = _rate_groups(dyn['inst_count'][usable])
    phase = dyn['phase'][usable]
    for g in np.unique(groups):
        m, ci, _ = circ_mean_ci_deg(phase[groups == g])
        ax.errorbar(g + 1, m, yerr=ci, fmt='o', color='k', markersize=4, capsize=2, linewidth=0.8)
        ax.text(g + 1, 345, f'{int((groups == g).sum())}', ha='center', fontsize=6, color='0.35')
    ax.set_xlim(0, INST_RATE_MAX_GROUP + 1)
    ax.set_ylim(0, 360)
    ax.set_yticks(np.arange(0, 361, 90))
    ax.set_xlabel(f'No. of spikes in 2 cycles (n spikes on top; last group: >= {INST_RATE_MAX_GROUP})')
    ax.set_ylabel('Mean phase (deg)')
    ax.set_title(f'Phase vs instantaneous rate; circular ANOVA p = {anova_p:.3g}', fontsize=9)


def plot_unit_summary(dyn, analysis, st, results, skip_reason, title, out_path: Path, theta=None):
    """Step 3 summary of one unit, 3 columns. Top row: spike train around its longest episode, onset / offset phase
    histograms, phase vs instantaneous rate. Bottom row: phase vs firing index with the fit, density map and
    statistics (or, with no fit, skip_reason). theta (optional): dict(phase_deg=..., metrics=..., shuffle_tmis=...) from
    compute_theta_modulation; if given, a first row is added with Steps 1-2's polar plot, TMI histogram and TMI null."""
    n_cols = 3
    if theta is None:
        fig, axes = plt.subplots(2, n_cols, figsize=(17, 10))
    else:
        fig = plt.figure(figsize=(17, 15.5))
        gs = fig.add_gridspec(3, n_cols)
        m = theta['metrics']
        _draw_polar_theta(fig.add_subplot(gs[0, 0], projection='polar'), theta['phase_deg'], m['MRL'],
                          m['PreferredPhase_deg'], m['Rayleigh_p'], m['SignificantThetaModulation'])
        ax_h = fig.add_subplot(gs[0, 1])
        _draw_tmi_histogram(ax_h, theta['phase_deg'], m['TMI'], m['TMI_Significant'])
        color, verdict_line = _tmi_verdict(m['TMI'], m['TMI_shuffle_p'], m['TMI_Significant'])
        ax_h.set_title(f'TMI histogram (n={len(theta["phase_deg"])} spikes)\n{verdict_line}', fontsize=9, color=color)
        _draw_tmi_null(fig.add_subplot(gs[0, 2]), m['TMI'], m['TMI_Significant'], theta['shuffle_tmis'])
        axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in (1, 2)])
    fig.suptitle(title, fontsize=12, fontweight='bold')

    _draw_spike_train_raster(axes[0, 0], dyn, analysis)
    _draw_onset_offset_hist(axes[0, 1], dyn, analysis, st)
    _draw_phase_vs_rate(axes[0, 2], dyn, st['RatePhase_ANOVA_p'])

    header = [f"EPISODES: {len(analysis['episodes'])}, median {st['episode_median_cycles']:.1f} cycles, "
              f"{st['episode_median_path_cm']:.1f} cm path",
              f"onset / offset spikes = {st['OnOff_n_onset']} / {st['OnOff_n_offset']} in "
              f"{st['OnOff_n_onset_blocks']} / {st['OnOff_n_offset_blocks']} blocks",
              f"theta-positive spikes = {int(dyn['in_theta'].sum())}; mean rate = {st['rate_spikes_per_cycle']:.3f} spikes/cycle"]
    if results is None:
        for ax in axes[1, :2]:
            ax.axis('off')
            ax.text(0.5, 0.5, 'firing-index fit skipped', ha='center', va='center', fontsize=10)
        _draw_text_panel(axes[1, 2], header + ['skipped:', *textwrap.wrap(skip_reason or '', 60)])
    else:
        _draw_firing_index_phase(axes[1, 0], results)
        _draw_density(fig, axes[1, 1], results, 'Density map (firing episodes)')
        _draw_text_panel(axes[1, 2], header + [f"fitted spikes = {results['n_spikes']}; burst events = {results['n_events']}",
                                               *_fit_text_lines(results)])
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _errorbar_circular(ax, y, mean, half, color, **kw):
    """Horizontal 95% CI of a circular mean on a 0-360 deg axis, wrapped so a CI crossing 0/360 deg is drawn in two parts."""
    if not np.isfinite(mean):
        return
    half = min(half, 180.0)
    lo, hi = mean - half, mean + half
    for a, b in ((max(lo, 0), min(hi, 360)), (lo + 360, 360) if lo < 0 else (0, 0), (0, hi - 360) if hi > 360 else (0, 0)):
        if b > a:
            ax.plot([a, b], [y, y], color=color, linewidth=FIG_BOX_LW, solid_capstyle='butt', zorder=2)
    ax.plot(mean, y, 'o', color=color, markersize=7, markeredgecolor='white', markeredgewidth=0.8, zorder=3, **kw)


@_arena_fig_style
def plot_onset_offset_groups(rows, out_path: Path):
    """Harris Fig 2d: per group, circular mean phase and 95% CI of onset and offset spikes (left) and offset - onset
    phase relative to each unit's preferred phase with the within-unit rotation p (right)."""
    rows = [r for r in rows if r['n_onset'] and r['n_offset']]
    if not rows:
        return
    ypos = np.arange(len(rows))[::-1]
    fig, (ax, ax_d) = plt.subplots(1, 2, figsize=(11.5, 0.5 * len(rows) + 1.8), sharey=True, gridspec_kw=dict(width_ratios=[1.4, 1]))
    for y, r in zip(ypos, rows):
        _errorbar_circular(ax, y + 0.15, r['onset_phase_deg'], r['onset_ci_deg'], FIRING_ONSET_COLOR)
        _errorbar_circular(ax, y - 0.15, r['offset_phase_deg'], r['offset_ci_deg'], FIRING_OFFSET_COLOR)
        d = r['centred_offset_minus_onset_deg']
        if np.isfinite(d):
            ax_d.plot(d, y, 'o', color=PAL_BLACK, markersize=7, zorder=3)
            ax_d.text(188, y, f'p = {_fmt_p(r["within_unit_rotation_p"])}\n{r["n_units_both"]} units, {r["n_blocks"]} blocks',
                      va='center', fontsize=FS_STATS, clip_on=False)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f'{r["Group"]} ({r["n_onset"]} / {r["n_offset"]} spikes)' for r in rows], fontsize=FS_TICK - 1)
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 90))
    ax.set_xlabel('Theta phase (deg; 0 = LFP peak)', fontsize=FS_LABEL - 2)
    ax.plot([], [], 'o', color=FIRING_ONSET_COLOR, label='Onset')
    ax.plot([], [], 'o', color=FIRING_OFFSET_COLOR, label='Offset')
    ax.legend(fontsize=FS_LEGEND - 1, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2)
    ax_d.axvline(0, color='0.6', linestyle='--', linewidth=FIG_AXIS_LW, zorder=1)
    ax_d.set_xlim(-180, 180)
    ax_d.set_xticks(np.arange(-180, 181, 90))
    ax_d.set_xlabel('Offset - onset, within units (deg)', fontsize=FS_LABEL - 2)
    for a in (ax, ax_d):
        _style_axes(a)
    fig.tight_layout()
    _save_fig(fig, out_path)


@_arena_fig_style
def plot_phase_vs_rate(bins: pd.DataFrame, groups, out_path: Path):
    """Harris Fig 1d, one column per group: top, circular mean phase and 95% CI vs spikes in 2 cycles; bottom, the same
    relative to each unit's preferred phase, which removes differences between cells' preferred phases."""
    groups = [g for g in groups if (bins['Group'] == g).any()]
    if not groups:
        return
    fig, axes = plt.subplots(2, len(groups), figsize=(3.3 * len(groups), 6.2), sharex=True, squeeze=False)
    for j, g in enumerate(groups):
        sub = bins[bins['Group'] == g]
        axes[0, j].errorbar(sub['spikes_in_2_cycles'], sub['phase_deg'], yerr=sub['ci_deg'], fmt='o', color=PAL_BLACK,
                            markersize=4, capsize=2, linewidth=FIG_AXIS_LW)
        axes[1, j].errorbar(sub['spikes_in_2_cycles'], sub['centred_phase_deg'], yerr=sub['centred_ci_deg'], fmt='o', color=PAL_BLACK,
                            markersize=4, capsize=2, linewidth=FIG_AXIS_LW)
        axes[0, j].set_ylim(0, 360)
        axes[0, j].set_yticks(np.arange(0, 361, 90))
        axes[1, j].set_ylim(-180, 180)
        axes[1, j].set_yticks(np.arange(-180, 181, 90))
        axes[1, j].axhline(0, color='0.6', linestyle='--', linewidth=FIG_AXIS_LW, zorder=0)
        axes[0, j].set_title(f'{g} (n={int(sub["n_spikes"].sum())} spikes)', fontsize=FS_TITLE - 4)
        axes[1, j].set_xlabel(f'No. of spikes in 2 cycles\n(>= {INST_RATE_MAX_GROUP} pooled)', fontsize=FS_LABEL - 4)
        for a in axes[:, j]:
            _style_axes(a)
    axes[0, 0].set_ylabel('Mean phase (deg)', fontsize=FS_LABEL - 3)
    axes[1, 0].set_ylabel('Relative to unit (deg)', fontsize=FS_LABEL - 3)
    fig.tight_layout()
    _save_fig(fig, out_path)


@_arena_fig_style
def plot_rate_derivative_maps(sp, group_masks: dict, out_path: Path):
    """Harris Fig 3b, one column per group: mean phase as a function of instantaneous rate and its derivative (cyclic colour
    scale; black where there are too few spikes, full brightness from one smoothed spike per bin up), absolute (top) and
    relative to each unit's preferred phase (bottom). Local circular means use Gaussian smoothing of 1.5 bins; the main
    text does not give the paper's."""
    r_edges, d_edges = np.linspace(0, 8, 33), np.linspace(-2, 2, 33)
    rate, deriv = sp['rate'].to_numpy(), sp['deriv'].to_numpy()
    cmap = plt.get_cmap('hsv')       # cyclic, for phase only, on black as in the paper
    fig, axes = plt.subplots(2, len(group_masks), figsize=(3.4 * len(group_masks) + 1, 6.4), sharex=True, sharey=True, squeeze=False)
    for j, (name, mask) in enumerate(group_masks.items()):
        for i, col in enumerate(('phase', 'phase_centred')):
            th = sp[col].to_numpy()[mask]
            kw = dict(bins=[r_edges, d_edges])
            count, _, _ = np.histogram2d(rate[mask], deriv[mask], **kw)
            cos_map, _, _ = np.histogram2d(rate[mask], deriv[mask], weights=np.cos(th), **kw)
            sin_map, _, _ = np.histogram2d(rate[mask], deriv[mask], weights=np.sin(th), **kw)
            count, cos_map, sin_map = (gaussian_filter(m, 1.5, mode='constant') for m in (count, cos_map, sin_map))
            phase_map = np.mod(np.arctan2(sin_map, cos_map), 2 * np.pi) / (2 * np.pi)
            rgba = cmap(phase_map)
            rgba[..., 3] = np.clip(count, 0, 1)
            ax = axes[i, j]
            ax.set_facecolor('black')
            ax.imshow(np.transpose(rgba, (1, 0, 2)), origin='lower', aspect='auto', interpolation='nearest',
                      extent=[r_edges[0], r_edges[-1], d_edges[0], d_edges[-1]])
            ax.axhline(0, color='white', linestyle='--', linewidth=FIG_AXIS_LW)
            _style_axes(ax)
        axes[0, j].set_title(f'{name} (n={int(mask.sum())} spikes)', fontsize=FS_TITLE - 4)
        axes[1, j].set_xlabel('Spikes per cycle', fontsize=FS_LABEL - 3)
    axes[0, 0].set_ylabel('Derivative (spikes per cycle²)\nabsolute phase', fontsize=FS_LABEL - 4)
    axes[1, 0].set_ylabel('Derivative (spikes per cycle²)\nrelative to unit', fontsize=FS_LABEL - 4)
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    cax = fig.add_axes([0.935, 0.25, 0.012, 0.5])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 360), cmap=cmap), cax=cax, ticks=np.arange(0, 361, 90))
    cb.set_label('Phase (deg)', fontsize=FS_STATS + 1)
    _save_fig(fig, out_path)


@_arena_fig_style
def plot_firing_index_population(sp, firing_index, panels: dict, bins: pd.DataFrame, thresholds, names, out_path: Path):
    """Phase relative to each unit's preferred phase vs firing index, per group: spike density normalised within each firing-index
    bin (so episodes' uneven spike counts do not hide the phase trend), duplicated over two cycles, with circular means
    and 95% CIs per firing-index bin. Panel title: model-free early-to-late shift with its unit-resampled CI and within-unit p,
    and the line fit's slope."""
    names = [n for n in names if n in panels and panels[n][1]['n_spikes'] >= MIN_SPIKES_FOR_FIT]
    if not names:
        return
    n_cols = min(4, len(names))
    n_rows = int(np.ceil(len(names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.5 * n_rows), squeeze=False, sharex=True, sharey=True)
    x_edges, y_edges = np.linspace(-1, 1, 25), np.linspace(-180, 180, 37)
    theta = np.rad2deg(np.mod(sp['phase_centred'].to_numpy() + np.pi, 2 * np.pi) - np.pi)
    sub_bins = bins[(bins['low_max_spikes'] == thresholds[0]) & (bins['high_min_spikes'] == thresholds[1])]
    for ax in axes.flat[len(names):]:
        ax.axis('off')
    for ax, name in zip(axes.flat, names):
        mask, row = panels[name]
        use = mask & np.isfinite(firing_index)
        h, _, _ = np.histogram2d(firing_index[use], theta[use], bins=[x_edges, y_edges])
        h = gaussian_filter(h, 1.0, mode=('nearest', 'wrap'))
        with np.errstate(invalid='ignore', divide='ignore'):
            h = np.nan_to_num(h / h.sum(axis=1, keepdims=True))
        ax.pcolormesh(x_edges, np.r_[y_edges[:-1], y_edges + 360], np.concatenate([h, h], axis=1).T, cmap='viridis', shading='flat')
        b = sub_bins[sub_bins['Group'] == name]
        xc = 0.5 * (b['firing_index_lo'] + b['firing_index_hi'])
        for off in (0, 360):
            ax.errorbar(xc, b['centred_phase_deg'] + off, yerr=b['ci_deg'], fmt='o', color='white', markeredgecolor=PAL_BLACK,
                        markeredgewidth=0.5, ecolor='white', markersize=4, capsize=0, linewidth=FIG_AXIS_LW)
        ax.set_ylim(-180, 540)
        ax.set_xlim(-1, 1)
        ax.set_yticks(np.arange(-180, 541, 180))
        ax.set_title(f'{name}\nearly-to-late shift {row["unwrapped_shift_deg"]:.0f}° [{row["shift_ci_lo"]:.0f}, {row["shift_ci_hi"]:.0f}], '
                     f'p = {_fmt_p(row["shift_within_unit_p"])}\nline {row["slope_deg_per_episode"]:.0f}°/episode, '
                     f'{row["n_spikes"]} spikes, {row["n_units"]} units', fontsize=FS_STATS)
        _style_axes(ax)
    for ax in axes[-1, :]:
        ax.set_xlabel('Firing index (-1 onset, +1 offset)', fontsize=FS_LABEL - 4)
    for ax in axes[:, 0]:
        ax.set_ylabel('Phase relative to unit (deg)', fontsize=FS_LABEL - 4)
    fig.tight_layout()
    _save_fig(fig, out_path)


def process_session(data_folder: Path, rng, edmin_records: list | None = None, spike_tables: list | None = None) -> list[dict]:
    """Run Steps 1-3 on every unit in one session; returns one row per unit. Each unit's per-spike table (with its unit_id,
    also in the row) is appended to `spike_tables`, for the population step.
    If `edmin_records` is given (and USE_ACG_THETA_EPOCHS), one dict per ACG
    epoch of each LFP file used (Animal, Session, lfp_file, epoch_index, ED_min,
    peakrangenorm, Status) is appended to it, for the pooled ED_min QC plot."""
    output_dir = data_folder / SESSION_OUTPUT_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)

    ncs_files = sorted(data_folder.glob('*.ncs'), key=_natural_key)

    # Tracking serves only the stationary control; a session without it runs everything else.
    trace = None
    try:
        tracking_path = _find_tracking_file(data_folder)
        print(f'Using tracking file: {tracking_path.name}')
        trace = tracking_trace(*load_tracking(tracking_path, TRACKING_TIME_UNIT))
    except FileNotFoundError as exc:
        print(f'  {exc} -- no stationary control for this session.')

    session_label = '_'.join(data_folder.parts[-3:])
    animal_id = detect_animal_id(data_folder)
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
            lfp_sig, lfp_ts, lfp_fs = load_ncs(theta_ncs)
            # Clean (resample to ACG_FS, mains notch, detrend) before the Hilbert
            # phase; everything downstream then works on this 1 kHz trace.
            lfp_sig, lfp_ts, lfp_fs = clean_lfp(lfp_sig, lfp_ts, lfp_fs)
            filtered_lfp = bandpass_filter(lfp_sig, LFP_FILTER_BAND[0], LFP_FILTER_BAND[1], lfp_fs)
            lfp_phase_unwrapped = np.unwrap(hilbert_phase(filtered_lfp, lfp_fs)[0])
            del filtered_lfp

            acg_epoch_ok = None
            lfp_theta_mask = None
            if USE_ACG_THETA_EPOCHS:
                acg_epoch_ok, acg_ed_min, acg_peakrange = acg_theta_epoch_mask(
                    lfp_sig, return_peakrange=True)
                lfp_theta_mask = times_in_ok_epochs(lfp_ts, lfp_ts, lfp_fs, acg_epoch_ok)
                n_fit = int(np.isfinite(acg_ed_min).sum())
                print(f'  ACG theta-positive epochs (ED_min < {ACG_ED_MIN_THRESH:g}): '
                      f'{int(acg_epoch_ok.sum())}/{len(acg_epoch_ok)} '
                      f'({n_fit} passed delta/theta + artifact screening)')
                if edmin_records is not None:
                    status = np.where(acg_epoch_ok, 'accepted',
                                      np.where(np.isfinite(acg_ed_min), 'rejected_EDmin',
                                               'rejected_prescreen'))
                    edmin_records.extend(
                        dict(Animal=animal_id, Session=session_label, FolderPath=str(data_folder),
                             lfp_file=theta_ncs.name, epoch_index=i, ED_min=acg_ed_min[i],
                             peakrangenorm=acg_peakrange[i], Status=status[i])
                        for i in range(len(acg_epoch_ok)))

            lfp_cache[theta_ncs] = dict(lfp_ts=lfp_ts, lfp_fs=lfp_fs, lfp_phase_unwrapped=lfp_phase_unwrapped,
                                        clock=lfp_cycle_clock(lfp_ts, lfp_phase_unwrapped),
                                        acg_epoch_ok=acg_epoch_ok, lfp_theta_mask=lfp_theta_mask)

        lfp_data = lfp_cache[theta_ncs]
        lfp_ts, lfp_fs = lfp_data['lfp_ts'], lfp_data['lfp_fs']
        lfp_phase_unwrapped = lfp_data['lfp_phase_unwrapped']
        acg_epoch_ok, lfp_theta_mask = lfp_data['acg_epoch_ok'], lfp_data['lfp_theta_mask']
        clock = lfp_data['clock']

        units = load_ntt_spike_times(ntt_path)
        for cell_number, spk_ts in units.items():
            unit_label = f'{ntt_path.stem}_cell{cell_number}' if len(units) > 1 else ntt_path.stem
            plot_label = f'{animal_id} | {unit_label}'
            file_prefix = f'{animal_id}_{unit_label}'
            row: dict[str, Any] = dict(Animal=animal_id, Session=session_label, FolderPath=str(data_folder), Unit=unit_label,
                       ntt_file=ntt_path.name, lfp_file=theta_ncs.name, cell_number=cell_number,
                       n_spikes_total=len(spk_ts))

            # Remove this unit's plots from any earlier v31 run, so a cell that no longer
            # passes a step (e.g. now fails the TMI shuffle test) leaves no stale plot.
            for suffix in ('PolarPlot', 'PhaseHistogram', 'TMIShuffle', 'OnsetOffset', 'FitRShuffle'):
                (output_dir / f'{file_prefix}_{suffix}.png').unlink(missing_ok=True)
            stale_dirs = [RESULTS_FOLDER / MULTILINES_FOLDER,
                          *(RESULTS_FOLDER / d / sub for d in set(CLASS_PLOT_FOLDERS.values())
                            for sub in ('', *FIT_R_SUBFOLDERS.values()))]
            class_prefix = f'{animal_id}_{session_label}_{unit_label}'
            for stale_dir in stale_dirs:
                for suffix in ('OnsetOffset', 'FitRShuffle'):
                    (stale_dir / f'{class_prefix}_{suffix}.png').unlink(missing_ok=True)

            # ---- ACG theta-positive spike selection (Steps 1-2 use only these spikes) ----
            spk_ts_all = spk_ts
            if USE_ACG_THETA_EPOCHS:
                spk_ts = spk_ts_all[times_in_ok_epochs(spk_ts_all, lfp_ts, lfp_fs, acg_epoch_ok)]
                row['n_spikes_ACGtheta'] = len(spk_ts)
                row['ACGtheta_epoch_frac'] = (float(acg_epoch_ok.mean()) if len(acg_epoch_ok)
                                               else np.nan)

            # ---- Steps 1 & 2: theta phase-locking polar plot + TMI shuffle test ----
            metrics, phase_deg, shuffle_tmis = compute_theta_modulation(
                spk_ts, lfp_ts, lfp_phase_unwrapped, lfp_fs, rng)
            row.update(metrics)

            if metrics['n_spikes_theta'] >= MIN_SPIKES_FOR_TMI:
                polar_path = output_dir / f'{file_prefix}_PolarPlot.png'
                plot_polar_theta(phase_deg, metrics['MRL'], metrics['PreferredPhase_deg'],
                                  metrics['Rayleigh_p'], metrics['SignificantThetaModulation'],
                                  plot_label, polar_path)

                hist_path = output_dir / f'{file_prefix}_PhaseHistogram.png'
                plot_phase_histogram(phase_deg, metrics['SignificantThetaModulation'],
                                      plot_label, hist_path)

                tmi_shuffle_path = output_dir / f'{file_prefix}_TMIShuffle.png'
                plot_tmi_shuffle(phase_deg, metrics['TMI'], metrics['TMI_shuffle_p'],
                                  metrics['TMI_Significant'], shuffle_tmis, plot_label, tmi_shuffle_path)
                print(f'  {unit_label}: TMI={metrics["TMI"]:.3f} '
                      f'(p={metrics["TMI_shuffle_p"]:.3g}), Rayleigh p={metrics["Rayleigh_p"]:.3g} -> '
                      f'{"theta-modulated" if metrics["ThetaModulated"] else "not theta-modulated"}')
            else:
                print(f'  {unit_label}: only {metrics["n_spikes_theta"]} spikes with LFP coverage '
                      f'(< MIN_SPIKES_FOR_TMI={MIN_SPIKES_FOR_TMI}), skipping polar plot / TMI test')

            # ---- Step 3: firing dynamics, then phase vs firing index ----
            row['PrecessionTested'] = False
            row['PrecessionSkippedReason'] = ''
            row['unit_id'] = len(spike_tables) if spike_tables is not None else np.nan
            dyn = spike_train_dynamics(spk_ts_all, clock, lfp_phase_unwrapped, lfp_fs, acg_epoch_ok)
            add_position_control(dyn, clock, trace)
            if spike_tables is not None:
                spike_tables.append(pd.DataFrame({**dyn, 'unit_id': row['unit_id']}))
            _process_dynamics(row, dyn, clock, lfp_phase_unwrapped, lfp_fs, lfp_theta_mask, rng, metrics, output_dir,
                              file_prefix, plot_label, class_prefix, unit_label,
                              dict(phase_deg=phase_deg, metrics=metrics, shuffle_tmis=shuffle_tmis))
            rows.append(row)

    return rows


def _process_dynamics(row, dyn, clock, lfp_phase_unwrapped, lfp_fs, lfp_theta_mask, rng, metrics, output_dir, file_prefix,
                      plot_label, class_prefix, unit_label, theta):
    """Step 3 for one unit: onset / offset spikes, firing episodes, phase vs instantaneous rate, firing-index fit, figures
    (<prefix>_OnsetOffset / _FitRShuffle, plus the class-folder copy) and the unit's Excel columns, added to `row`."""
    analysis = analyse_firing(dyn['cycle'], dyn['n_prev'], dyn['n_next'], dyn['in_theta'], dyn['window_ok'])
    usable = dyn['in_theta']
    st = onset_offset_unit_stats(dyn['phase'], analysis['label'], analysis['block'], rng, usable)
    span_cycles = float(np.ptp(clock['cycle']))
    episodes = analysis['episodes']
    st.update(
        rate_spikes_per_cycle=len(dyn['ts']) / span_cycles if span_cycles > 0 else np.nan,
        lfp_cycles_per_s=span_cycles / float(np.ptp(clock['ts'])) if len(clock['ts']) > 1 else np.nan,
        n_episodes=len(episodes), n_spikes_in_episodes=int((np.isfinite(analysis['firing_index']) & usable).sum()),
        episode_median_cycles=float(np.median([dyn['cycle'][l] - dyn['cycle'][f] for f, l in episodes])) if episodes else np.nan,
        episode_median_path_cm=(float(np.nanmedian([dyn['path_cum_cm'][l] - dyn['path_cum_cm'][f] for f, l in episodes]))
                                if episodes and np.isfinite(dyn['path_cum_cm']).any() else np.nan),
        RatePhase_ANOVA_p=(rate_phase_anova_within(np.zeros(int(usable.sum()), dtype=np.int64), dyn['phase'][usable],
                                                   _rate_groups(dyn['inst_count'][usable]), rng) if usable.sum() >= MIN_SPIKES_FOR_TMI
                           else np.nan))
    row.update(st)
    has_labels = st['OnOff_n_onset'] + st['OnOff_n_offset'] > 0
    print(f'  {unit_label}: {st["OnOff_n_onset"]} onset / {st["OnOff_n_offset"]} offset spikes, {len(episodes)} firing episodes')

    if REQUIRE_THETA_MOD_FOR_FIT and not metrics['ThetaModulated']:
        failed = [name for name, ok in (('TMI shuffle test', metrics['TMI_Significant']),
                                        ('Rayleigh test', metrics['SignificantThetaModulation']
                                         or not REQUIRE_RAYLEIGH_FOR_THETA_MOD)) if not ok]
        row['PrecessionSkippedReason'] = f'not significantly theta-modulated ({" and ".join(failed)})'
        results = None
    else:
        try:
            results = compute_firing_fit(dyn, analysis, clock, lfp_phase_unwrapped, lfp_fs, rng,
                                         slope_bnds=SLOPE_BNDS, lfp_theta_mask=lfp_theta_mask)
        except TooFewSpikes as exc:
            row['PrecessionSkippedReason'] = str(exc)
            results = None
        except Exception as exc:
            row['PrecessionSkippedReason'] = f'ERROR ({exc})'
            print(f'  {unit_label}: firing-index fit ERROR ({exc})')
            results = None

    png_path = output_dir / f'{file_prefix}_OnsetOffset.png'
    if has_labels:
        plot_unit_summary(dyn, analysis, st, results, row['PrecessionSkippedReason'], plot_label, png_path)
    if results is None:
        print(f'  {unit_label}: firing-index fit skipped -- {row["PrecessionSkippedReason"]}')
        return

    # fit_R shuffle-test figure, for every unit the fit_R gate was run on.
    fit_r_png = None
    if results['shuffle_fit_r'] is not None:
        fit_r_png = output_dir / f'{file_prefix}_FitRShuffle.png'
        plot_fit_r_shuffle(results, plot_label, fit_r_png)
    # Also save the plot into RESULTS_FOLDER/<class folder> (next to the Excel file);
    # the CLASS_PLOT_FOLDERS versions get an extra top row with the polar plot, TMI
    # histogram and TMI shuffle null.
    class_dir_name = MULTILINES_FOLDER if results['fit_too_steep'] else CLASS_PLOT_FOLDERS.get(results['precession_class'])
    if class_dir_name:
        class_dir = RESULTS_FOLDER / class_dir_name
        if class_dir_name != MULTILINES_FOLDER and results['shuffle_fit_r'] is not None:
            class_dir = class_dir / FIT_R_SUBFOLDERS[results['fit_R_Significant']]
        class_dir.mkdir(parents=True, exist_ok=True)
        class_png = class_dir / f'{class_prefix}_OnsetOffset.png'
        if class_dir_name == MULTILINES_FOLDER or metrics['n_spikes_theta'] < MIN_SPIKES_FOR_TMI:
            shutil.copy2(png_path, class_png)
        else:
            plot_unit_summary(dyn, analysis, st, results, '', plot_label, class_png, theta=theta)
        if fit_r_png is not None:
            shutil.copy2(fit_r_png, class_dir / f'{class_prefix}_FitRShuffle.png')

    row['PrecessionTested'] = True
    row.update({'FiringIndex_n_spikes': results['n_spikes'], 'FiringIndex_n_burst_events': results['n_events'],
                **_fit_columns(results)})
    print(f'  {unit_label}: PRECESSION rho={results["rho"]:.3f}  p={results["p"]:.3g}  '
          f'fitR={results["fit_R"]:.3f} (shuffle p={results["fit_R_shuffle_p"]:.3g})  '
          f'slope={results["slope_deg_per_episode"]:.1f} deg/episode  class={results["precession_class"]}')


def _fit_columns(res):
    """Excel columns of one circular-linear fit (_precession_fit result)."""
    return {
        'rho': res['rho'],
        'precession_p': res['p'],
        'slope_deg_per_episode': res['slope_deg_per_episode'],
        'r_squared': res['r_squared'], 'fit_R': res['fit_R'],
        'fit_R_shuffle_p': res['fit_R_shuffle_p'],
        'fit_R_Significant': res['fit_R_Significant'],
        'phase_range_deg': res['phase_range_deg'],
        **{c: v for _, key, *_ in FIRING_INDEX_PHASE_POINTS
           for c, v in ((f'phase_at_{key}_deg', res[f'phase_at_{key}_deg']),
                        (f'phase_at_{key}_range', phase_range_label(res[f'phase_at_{key}_deg'])))},
        'is_precessing': res['is_precessing'],
        'is_recessing': res['is_recessing'],
        'is_phase_locked': res['is_phase_locked'],
        'PrecessionClass': res['precession_class'],
        'n_fit_lines': res['n_fit_lines'],
        'MultiLinesFit': res['fit_too_steep'],
    }


def build_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Population-level summary counts/percentages for the 'Summary' sheet, one row per unit in df: theta-modulation criteria,
    units with onset / offset spikes and firing episodes, units with a firing-index fit, and its PrecessionClass outcomes."""
    def pct(n, d):
        return (100.0 * n / d) if d > 0 else np.nan

    total = len(df)
    tested = df['PrecessionTested'] == True                                  # noqa: E712
    n_tested = int(tested.sum())
    rows = []
    for name, mask, denom, denom_label in (
            ('SignificantThetaModulation (Rayleigh) units', df['SignificantThetaModulation'] == True, total, 'total units'),  # noqa: E712
            ('TMI_Significant (TMI shuffle) units', df['TMI_Significant'] == True, total, 'total units'),                      # noqa: E712
            (('ThetaModulated units (TMI shuffle AND Rayleigh)' if REQUIRE_RAYLEIGH_FOR_THETA_MOD
              else 'ThetaModulated units (TMI shuffle)'), df['ThetaModulated'] == True, total, 'total units'),                 # noqa: E712
            (f'Units with >= {ONOFF_MIN_SPIKES_PER_UNIT} onset AND offset spikes',
             (df['OnOff_n_onset'] >= ONOFF_MIN_SPIKES_PER_UNIT) & (df['OnOff_n_offset'] >= ONOFF_MIN_SPIKES_PER_UNIT), total, 'total units'),
            ('Units with >= 1 firing episode', df['n_episodes'] >= 1, total, 'total units'),
            ('Units with a firing-index fit (PrecessionTested)', tested, total, 'total units'),
            *((f'{cls} units (of fitted units)', df['PrecessionClass'] == cls, n_tested, 'fitted units')
              for cls in FI0_PHASE_CLASSES)):
        n = int(mask.sum())
        rows.append(dict(Metric=name, Count=n, Denominator=denom, DenominatorLabel=denom_label, Percent=pct(n, denom)))
    return pd.DataFrame(rows, columns=['Metric', 'Count', 'Denominator', 'DenominatorLabel', 'Percent'])


def save_metadata_csv(csv_path: Path) -> Path:
    """Write the code name and every hard-coded analysis setting to a
    Parameter,Value CSV (next to the results Excel file).

    The configuration constants (the ALL_CAPS assignments between the imports
    and the Neuralynx I/O section) are collected automatically from the module
    namespace, so newly added settings are recorded without editing this
    function; the constants hard-coded inside functions are listed explicitly.
    """
    code_path = Path(__file__).resolve()
    rows = [
        ('CodeName', code_path.name),
        ('CodePath', str(code_path)),
        ('RunTimestamp', pd.Timestamp.now().isoformat(timespec='seconds')),
    ]
    module_globals = globals()
    config_names = [n for n in module_globals
                    if n.isupper() and not n.startswith(('_', 'PAL_', 'FS_', 'FIG_'))  # skip figure style
                    and n not in ('ARENA_COMPARISON_METRICS', 'TMI_COMPARISON_METRICS',
                                  'THETA_MOD_CELL_METRICS', 'ARENA_COLORS',
                                  'SESSION_COLORS', 'GROUP_COLORS', 'GROUP_COUNT_COLUMNS',
                                  'FI0_RANGE_COLORS')]
    rows += [(n, repr(module_globals[n]) if not isinstance(module_globals[n], (str, Path))
              else str(module_globals[n])) for n in config_names]
    # constants hard-coded inside functions
    rows += [
        ('FIRING_DENSITY_INDEX_BINS', 40),                    # _precession_fit: linspace(-1, 1, 41)
        ('FIRING_DENSITY_PHASE_BINS', 100),                   # _precession_fit: linspace(0, 2*pi, 101)
        ('FIRING_DENSITY_GAUSS_SIGMA_BINS', 1.5),             # _precession_fit density smoothing
        ('FIRING_DENSITY_GAUSS_TRUNCATE', 2.0),
        ('RATE_DERIV_MAP_GAUSS_SIGMA_BINS', 1.5),             # plot_rate_derivative_maps
        ('BANDPASS_FILTER_ORDER', 4),                         # bandpass_filter default
        ('TMI_NUM_CYCLES', 5),                                # calc_tmi default
        ('TMI_HIST_SMOOTH_KERNEL_N', 7),                      # calc_tmi histogram smoothing
        ('TMI_HIST_SMOOTH_SIGMA', 0.5),
        ('ACG_WELCH_WINDOW', 'hann'),
        ('PRECESSION_SIGNIFICANCE_TEST',
         ('fit_R within-unit phase-rotation null p < ALPHA (if USE_FIT_R_GATE) AND '
          'Kempter et al. (2012) asymptotic z-test p < ALPHA (on burst events if BURST_COLLAPSE_FOR_ZTEST); '
          'per unit: theta-positive spikes of firing episodes (onset block to following offset block), firing index '
          '-1 at firing onset to +1 at firing offset, linear in theta cycles; no position used')),
    ]
    pd.DataFrame(rows, columns=['Parameter', 'Value']).to_csv(csv_path, index=False)
    return csv_path


README_NOTES = [
    'ThetaPhase sheet: ONE ROW PER UNIT. No place field, traversal or position enters any classification: position is used only '
    'for the stationary control. Unit column names are Harris et al. 2002 (Nature 417:738) analyses as noted below; the header of '
    'the code lists every deliberate difference from that paper.',
    f'Onset / offset spikes (OnOff_*; Harris Fig 2): onset = previous {ONOFF_WINDOW_CYCLES} theta cycles hold <= {ONOFF_LOW_MAX_SPIKES} '
    f'spikes and the following {ONOFF_WINDOW_CYCLES} >= {ONOFF_HIGH_MIN_SPIKES} spikes; offset = the reverse. Windows are centred on '
    'the spike\'s own phase; the spike must lie in an ACG theta-positive epoch but its window counts all spikes. '
    'A block = consecutive spikes with one label. OnOff_MWW_p = Mardia-Watson-Wheeler test against the within-unit rotation null '
    f'(needs >= {ONOFF_MIN_SPIKES_PER_UNIT} onset and offset spikes). OnOff_offset_minus_onset_deg wrapped to [-180, 180).',
    'Every p for phase vs firing dynamics (OnOff_MWW_p, RatePhase_ANOVA_p, fit_R_shuffle_p, the within-unit p of the OnOff_* sheets) '
    'comes from rotating each unit\'s phase sequence against its spike train (10-90% of its spikes). It keeps phase locking, bursts and the '
    'slow drift of phase, and the clustering in time of onset / offset / high-rate spikes. With the LFP time-shifted against the spikes '
    '(no true relation; 40 shifts) the older burst / block / spike shuffles gave p < 0.05 in 10-100% of shifts, so the free-label p columns '
    '(as in the paper) are for comparison only.',
    'Firing episode = an onset block and the offset block right after it, no silence >= the window between them. Firing index: -1 at its '
    'first onset spike, +1 at its last offset spike, linear in theta cycles. slope_deg_per_episode, rho, fit_R, PrecessionClass and '
    'phase_at_* are the v30 circular-linear fit and classification on this axis instead of the pass index: negative slope = phase advances '
    'from firing onset to offset. PrecessionTested = False: PrecessionSkippedReason says why.',
    f'RatePhase_ANOVA_p (Harris Fig 1d): circular ANOVA of phase across instantaneous-rate groups (spikes within +/-'
    f'{INST_RATE_WINDOW_CYCLES} cycle, spike included; {INST_RATE_MAX_GROUP} and above pooled) against the rotation null.',
    'OnOff_Thresholds: yield at the primary (Harris) spike-count thresholds and the sensitivity thresholds.',
    'OnOff_OnsetOffset (Harris Fig 2d), by group and thresholds: onset / offset circular means and 95% CIs of pooled spikes and the free-label '
    'Mardia-Watson-Wheeler p (as in the paper); then the within-unit test, on phase relative to the unit\'s preferred phase: '
    'centred_offset_minus_onset_deg and within_unit_rotation_p (only units with both labels, phase centred per unit), and one '
    'value per unit (unit_offset_minus_onset_deg, Rayleigh test of those values). Weigh the within-unit p, not the free-label p.',
    f'Stationary / Moving groups (position control): a spike is stationary if the animal moved < {ONOFF_STATIONARY_PATH_CM:g} cm over its '
    f'+/-{ONOFF_WINDOW_CYCLES} cycle window (window_path_cm), an episode if it moved < {ONOFF_STATIONARY_PATH_CM:g} cm between its onset and '
    f'offset. Same place (OnOff_OnsetOffset): onset vs offset only within one unit in one {ONOFF_SAME_PLACE_BIN_CM:g} cm bin, phase centred '
    f'per bin, so place-dependent phase cancels; n_strata = such bins '
    f'holding both labels. Same-place / Displaced episodes (fit tables): firing onset and offset < / >= {ONOFF_SAME_PLACE_CM:g} cm apart. '
    'Tracking cm/pixel outliers affect only these position groups.',
    'OnOff_FiringIndexFit: phase relative to the unit\'s preferred phase vs firing index over all episodes of a group: slope with a 95% CI '
    'from resampling units (at the primary thresholds) and fit_R tested against the within-unit rotation null; '
    'early_deg / middle_deg / late_deg = circular mean phase in each third of firing index; unwrapped_shift_deg = (middle - early) + (late - middle), '
    'each step wrapped to [-180, 180): negative = phase advances from onset to offset, with its own CI and two-sided within-unit p. '
    'Read this shift for the direction: the line slope is often multimodal on so few spikes (see its CI). '
    'OnOff_FiringIndexBins: the same data as circular means per firing-index bin, with no line assumed. '
    'OnOff_UnitSlopes: signs of the per-unit slopes. OnOff_Episodes: every episode at the primary thresholds.',
    f'OnOff_RateBins / OnOff_RateANOVA (Harris Fig 1d); OnOff_RateDerivRegions (Harris Fig 3): regression of spikes per cycle on cycle number '
    f'over {2 * RATE_DERIV_HALF_WINDOW_CYCLES + 1} cycles around the spike (intercept = rate, slope = derivative).',
    'Arena / session-type comparisons use one value per unit. PhaseAt* compare the fitted phase at firing index -1 / 0 / +1; OnOffShift_* compare '
    'OnOff_offset_minus_onset_deg. ThetaModCells_* are as in v30 (per recording session, or pooled per group). '
    'Animal_ThetaModCells_* sheets: the pooled proportions per animal, with all arenas pooled (Arena = All) and within '
    'the Open and Linear arenas.',
]


def main():
    if 360 % PHASE_BIN_SIZE_DEG != 0:
        raise ValueError('PHASE_BIN_SIZE_DEG must divide 360 evenly.')

    rng = np.random.default_rng(RANDOM_SEED)
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    session_folders = find_session_folders(ROOT_FOLDER)
    if not session_folders:
        raise FileNotFoundError(f'No folders with both .ncs and .ntt files found under {ROOT_FOLDER}')

    all_rows = []
    edmin_records: list[dict] = []
    spike_tables: list[pd.DataFrame] = []
    for data_folder in session_folders:
        print(f'\n=== Session: {data_folder} ===')
        try:
            all_rows.extend(process_session(data_folder, rng, edmin_records, spike_tables))
        except Exception as exc:
            print(f'ERROR processing {data_folder}: {exc}')
            continue

    # ---- ACG ED_min QC: per-epoch table + ED_min vs. peak range plot, all sessions pooled ----
    if edmin_records:
        edmin_df = pd.DataFrame(edmin_records)
        edmin_csv = RESULTS_FOLDER / EDMIN_EPOCHS_CSV_NAME
        edmin_df.to_csv(edmin_csv, index=False)
        print(f'\nACG per-epoch ED_min table saved to {edmin_csv}')
        edmin_png = plot_edmin_epochs(edmin_df, RESULTS_FOLDER / EDMIN_PLOT_NAME)
        if edmin_png is not None:
            print(f'ED_min vs. peak range plot saved to {edmin_png}')

    spikes = pd.concat(spike_tables, ignore_index=True)
    spikes.to_csv(RESULTS_FOLDER / SPIKE_TABLE_NAME, index=False)
    print(f'\nPer-spike table saved to {RESULTS_FOLDER / SPIKE_TABLE_NAME}')
    finish_population(pd.DataFrame(all_rows), spikes, RESULTS_FOLDER / OUTPUT_EXCEL_NAME)


def finish_population(df: pd.DataFrame, spikes: pd.DataFrame, excel_path: Path):
    """Everything after the per-unit step: arena / session-type comparisons, the population analyses and the summary workbook.
    `df` = one row per unit (process_session), `spikes` = the concatenated per-spike tables."""
    columns = ['unit_id', 'Animal', 'Session', 'FolderPath', 'Unit', 'ntt_file', 'lfp_file', 'cell_number', 'n_spikes_total',
               'n_spikes_ACGtheta', 'ACGtheta_epoch_frac', 'lfp_cycles_per_s', 'rate_spikes_per_cycle',
               'n_spikes_theta', 'MRL', 'PreferredPhase_deg', 'Rayleigh_p',
               'SignificantThetaModulation', 'PhasePeak_deg', 'PhaseValley_deg', 'TMI',
               'TMI_shuffle_p', 'TMI_Significant', 'ThetaModulated',
               # Harris Fig 1d and Fig 2, per unit
               'RatePhase_ANOVA_p', 'OnOff_n_onset', 'OnOff_n_offset', 'OnOff_n_onset_blocks', 'OnOff_n_offset_blocks',
               'OnOff_onset_phase_deg', 'OnOff_onset_MRL', 'OnOff_offset_phase_deg', 'OnOff_offset_MRL',
               'OnOff_offset_minus_onset_deg', 'OnOff_MWW_W', 'OnOff_MWW_p',
               # firing episodes and the fit on their firing index
               'n_episodes', 'n_spikes_in_episodes', 'episode_median_cycles', 'episode_median_path_cm',
               'PrecessionTested', 'FiringIndex_n_spikes', 'FiringIndex_n_burst_events',
               'rho', 'r_squared', 'fit_R', 'fit_R_shuffle_p', 'fit_R_Significant', 'precession_p', 'slope_deg_per_episode', 'phase_range_deg',
               *(f'phase_at_{key}_{suffix}' for _, key, *_ in FIRING_INDEX_PHASE_POINTS
                 for suffix in ('deg', 'range')),
               'is_precessing', 'is_recessing', 'is_phase_locked', 'PrecessionClass', 'n_fit_lines', 'MultiLinesFit',
               'PrecessionSkippedReason']
    df = df.reindex(columns=columns)
    summary_df = build_summary_stats(df)

    df['Arena'] = df['FolderPath'].apply(lambda p: detect_arena(Path(p)))
    df['SessionType'] = [detect_session_type(Path(p), a) for p, a in zip(df['FolderPath'], df['Arena'])]
    unknown_type = df.loc[df['Arena'].notna() & df['SessionType'].isna(), 'FolderPath'].unique()
    if len(unknown_type):
        print(f'\nWARNING: {len(unknown_type)} session folder(s) with no recognizable session type; '
              f'their cells are left out of the session-type comparisons (kept in the arena comparisons):')
        for p in sorted(unknown_type):
            print(f'    {p}')

    # ---- 1) Session-type comparison within each arena (e.g. Cntrl vs Rotate vs Zero) ----
    # sheet name -> per-arena tables, stacked into one sheet with an 'Arena' column
    session_type_tables: dict[str, list[pd.DataFrame]] = {}
    for arena in ARENA_LABELS:
        df_arena = df.loc[(df['Arena'] == arena) & df['SessionType'].notna()]
        if df_arena.empty:
            continue
        print(f'\n=== {arena} arena: session-type comparison '
              f'({", ".join(_label(t) for t in ARENA_SESSION_TYPES[arena])}) ===')
        tables = run_group_comparisons(
            df_arena, 'SessionType', ARENA_SESSION_TYPES[arena],
            out_dir=excel_path.parent / SESSION_TYPE_PLOT_FOLDER / arena, file_prefix=f'SessionType_{arena}',
            group_word='session type', title_prefix=f'{arena}: ')
        for key, table in tables.items():
            table.insert(0, 'Arena', arena)
            session_type_tables.setdefault(_comparison_sheet_name('SessionType', key), []).append(table)

    # ---- 2) Arena comparison (Circle vs Linear vs Open), all session types pooled ----
    print('\n=== Arena comparison (all session types pooled within each arena) ===')
    arena_tables = run_group_comparisons(
        df.loc[df['Arena'].notna()], 'Arena', ARENA_LABELS, out_dir=excel_path.parent,
        file_prefix='ArenaComparison', group_word='arena')

    # ---- 3) Theta-modulated proportion per animal: all arenas pooled, then within Open (the only arena
    # recorded in every animal) and Linear (Fa1059 and Fa23BD) ----
    # sheet name -> tables per scope, stacked into one sheet with an 'Arena' column ('All' = arenas pooled)
    animal_tables: dict[str, list[pd.DataFrame]] = {}
    for scope in ('All', 'Open', 'Linear'):
        df_scope = df if scope == 'All' else df.loc[df['Arena'] == scope]
        print(f'\n=== Theta-modulated units per animal ({scope}) ===')
        tables = run_theta_mod_proportion(
            df_scope, 'Animal', sorted(df_scope['Animal'].unique()),
            excel_path.parent / ('AnimalComparison_ThetaModProportion.png' if scope == 'All'
                              else f'AnimalComparison_{scope}_ThetaModProportion.png'),
            pooled=('arenas and session types pooled within animal' if scope == 'All'
                    else f'{scope} arena only; session types pooled within animal'),
            title_prefix='' if scope == 'All' else f'{scope}: ')
        for key, table in tables.items():
            if not table.empty:
                table.insert(0, 'Arena', scope)
                animal_tables.setdefault(_comparison_sheet_name('Animal', key), []).append(table)

    # ---- 4) Harris et al.'s analyses across units, within-unit tests, stationary control ----
    print('\n=== Population: onset / offset, rate, rate derivative, firing index ===')
    population_tables = run_onoff_population(spikes, df, excel_path.parent, np.random.default_rng(RANDOM_SEED + 1))

    readme_df = pd.DataFrame({'Note': README_NOTES})
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        readme_df.to_excel(writer, sheet_name='README', index=False)
        df.to_excel(writer, sheet_name='ThetaPhase', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        for sheet_name, table in population_tables.items():
            table.to_excel(writer, sheet_name=sheet_name, index=False)
        for sheet_name, tables in session_type_tables.items():
            pd.concat(tables, ignore_index=True).to_excel(writer, sheet_name=sheet_name, index=False)
        for key, table in arena_tables.items():
            table.to_excel(writer, sheet_name=_comparison_sheet_name('Arena', key), index=False)
        for sheet_name, tables in animal_tables.items():
            pd.concat(tables, ignore_index=True).to_excel(writer, sheet_name=sheet_name, index=False)
    print(f'\nDone. {len(df)} unit(s) processed. Summary saved to {excel_path}')

    metadata_path = save_metadata_csv(excel_path.with_name(excel_path.stem + '_metadata.csv'))
    print(f'Run metadata saved to {metadata_path}')


if __name__ == '__main__':
    main()
