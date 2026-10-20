# -*- coding: utf-8 -*-
"""
Combined theta-modulation + phase-precession pipeline, run sequentially per
unit (one .ntt file = one already-isolated unit):

  Step 1 (visualization only): polar plot of spike counts vs. theta phase,
    phase estimated by direct Hilbert transform of the cleaned (clean_lfp:
    resampled to 1 kHz, 50/100/150/200 Hz notched, detrended) and
    bandpass-filtered LFP
    (angle of the analytic signal, 0 deg at a peak; see hilbert_phase -- no
    Generalized Phase correction). The same Hilbert phase is used for
    Step 3's spatial pass-index phase.
  Step 2: Theta Modulation Index (TMI) for that unit (Frank et al. 2001:
    TMI = 1 - the minimum of the smoothed, normalized theta-phase
    histogram), tested for significance via that paper's shuffling
    procedure -- each spike is assigned an independent random phase in
    [0, 360), except that consecutive spikes < 50 ms apart and in the same
    real-data phase bin are kept together and given the same shuffled
    phase, preserving burst structure -- (as well as the classic Rayleigh
    test / mean resultant length on the same phases, reported alongside
    for QC). Each unit's TMI histogram and shuffle null distribution are
    saved as <session>/ThetaMod_PhasePrecession_LFPshuffle/<unit>_TMIShuffle.png.
  Step 3: only for units found to be significantly theta-modulated in Step 2
    (TMI shuffle test), run the Pass Index phase-precession analysis
    (Climer, Newman & Hasselmo 2013, Eur J Neurosci 38:2526-2541, with
    Kempter et al. 2012, J Neurosci Methods 207:113-124 for the
    circular-linear regression) to test whether the cell's spatial firing
    phase-precesses against theta.

ROOT_FOLDER is searched recursively; every folder that directly contains at
least one .ncs and at least one .ntt file is treated as a session. Data
layout expected per session folder:
    *.ncs   Neuralynx continuous (LFP) file, one per tetrode/channel. By
            default (MATCH_SPIKE_LFP_FILE = True) each .ntt file is matched
            to the .ncs file sharing its embedded channel number (e.g.
            TT1.ntt -> CSC1.ncs, TT12.ntt -> CSC12.ncs) via
            match_ncs_to_ntt(), rather than every unit in the session being
            referenced to a single session-wide LFP channel. Set
            MATCH_SPIKE_LFP_FILE = False to skip this per-tetrode matching
            (it can be slow) and reference every unit to the session's first
            .ncs file instead.
    *.ntt   Neuralynx tetrode spike files, one file per already-isolated
            unit. Every .ntt file in the folder is processed.
    tracking .csv file, auto-detected as the first .csv in the folder (only
            needed for Step 3). Column order is fixed: col A = timestamp,
            col B = x (pixels, unused), col C = y (pixels, unused), col D =
            x (cm), col E = y (cm). Columns D/E are used directly; no
            pixel-to-cm conversion is performed. A session with no tracking
            file still runs Steps 1-2; Step 3 is skipped for it.

Assumption: tracking timestamps are on the same absolute clock as the
Neuralynx acquisition system. Set TRACKING_TIME_UNIT below to match the
units the timestamp column is actually stored in.

Output: all pooled results (summary Excel, metadata CSV, comparison plots and
per-class PassIndex plot copies) are written to RESULTS_FOLDER =
ROOT_FOLDER/PhasePrec_Results. One row per unit is appended to a single summary
table written to RESULTS_FOLDER/theta_phase.xlsx, covering Step 1's polar-plot statistics
(MRL, preferred phase, Rayleigh p), Step 2's TMI and its shuffle p-value,
and Step 3's phase-precession fit (rho, p, slope, PrecessionClass) where it ran.
Step 3's significance is two gates, not a fixed slope-magnitude window: first
fit_R (residual MRL about the fitted line) must beat a burst-preserving
phase-permutation null (shuffle_fit_r_significance, USE_FIT_R_GATE; p in
fit_R_shuffle_p), then Kempter et al.'s (2012) asymptotic z-test on the
circular-linear correlation must give p < ALPHA. A significant fit with slope below
-SLOPE_THRESH_DEG_PER_PASS (22 deg/pass) is labeled phase_precessing
(is_precessing); above +22 deg/pass, phase_succeeding (is_recessing); a
significant fit with |slope| <= 22 deg/pass is labeled phase_locked
(is_phase_locked); a non-significant fit is labeled no_phase_relation. Each
cell's PassIndex plot is also copied to RESULTS_FOLDER/PhasePrecessing_Plots,
PhaseSucceeding_Plots or PhaseLocked_Plots (beside the Excel file; no_phase_relation
cells go to PhaseLocked_Plots too), in a 'fit R pass' or 'fit R fail' subfolder
according to the fit_R shuffle test (CLASS_PLOT_FOLDERS, FIT_R_SUBFOLDERS). Every cell the fit_R gate is run on also gets a <unit>_FitRShuffle.png
(phase vs pass index with the fit, and the fit_R shuffle null), saved next to its
other figures in <session>/ThetaMod_PhasePrecession_LFPshuffle and in its class folder.

Per-unit plots (polar plot from Step 1, and the 6-panel Pass Index summary
from Step 3, when run) are saved to
<session_folder>/ThetaPhasePrecession_Combined/.

If ROOT_FOLDER's cell sessions are laid out as
<root>/<animal>/<arena>/DayN/<session>, with <arena> one of ARENA_LABELS
('Circle', 'Linear', 'Open'), the significantly precessing/recessing
cells (pooled across animals within each arena) are additionally compared
across arenas on r^2 (rho^2 of the circular-linear fit), fit_R (mean
resultant length of the phase residuals about the fitted line, i.e. the
goodness of fit), |slope| (deg/pass), and phase range (deg actually swept, |slope| times the
observed pass-index extent), via Kruskal-Wallis + pairwise Mann-Whitney U
tests (Holm-Bonferroni corrected within each metric). One plot per metric is
saved to RESULTS_FOLDER/ArenaComparison_PhasePrecession_<metric>.png (figure
style matches ArenaType_StatsComparison_v7.py: arena-colored box plots, stars
over significantly different pairs, stats panel underneath, 500 dpi);
stats tables added as 'ArenaComparison_Omnibus' / 'ArenaComparison_Pairwise'
sheets in the summary workbook. The same arena comparison (Kruskal-Wallis +
pairwise Mann-Whitney U) is run on TMI for all significantly theta-modulated
cells (TMI_Significant), pooled across animals within each arena, saved to
RESULTS_FOLDER/ArenaComparison_TMI.png with 'ArenaComparison_TMI_Omnibus' /
'ArenaComparison_TMI_Pairwise' sheets.

Theta phase at pass index 0 (field center) is also extracted per cell
(phase_at_pi0_deg: the fitted phase line at pass index 0, i.e. its intercept,
in [0, 360)) and labeled precession range (70-250 deg) or procession range
(250-70 deg) in phase_at_pi0_range. For the PI0_PHASE_CLASSES cells it is
compared across arenas with circular tests -- Mardia-Watson-Wheeler
(nonparametric, permutation p) and Watson-Williams, omnibus + Holm-corrected
pairwise -- plotted to RESULTS_FOLDER/ArenaComparison_PhaseAtPassIndex0.png
(magma purple points = precession range, magma yellow = procession range) with
'PhaseAtPI0_ByArena' / 'PhaseAtPI0_Omnibus' / 'PhaseAtPI0_Pairwise' sheets.
The same extraction, plot and statistics are repeated at pass index -1 (field
entry; phase_at_pim1_deg/_range, ArenaComparison_PhaseAtPassIndexMinus1.png,
'PhaseAtPIm1_*' sheets) and +1 (field exit; phase_at_pip1_deg/_range,
ArenaComparison_PhaseAtPassIndexPlus1.png, 'PhaseAtPIp1_*' sheets) -- see
PASS_INDEX_PHASE_POINTS.

Before the arena comparison, every one of the comparisons above (r^2 / fit_R / |slope| /
phase range, TMI, phase at pass index -1/0/+1) is first run across SESSION
TYPES within each arena (Open: Cntrl/Rotate/Zero; Linear: 0/90/180/270 deg;
Circle: Stnd/NoRot/Rot -- see ARENA_SESSION_TYPES), with the session type read
off the session folder name as in SessionType_StatsComparison_v6.py
(detect_session_type) and boxes colored with that script's per-session-type
shades of the arena color. Animals are pooled within each session type, and
every cell is treated as independent (a unit recorded in several sessions of
one day counts once per session). Plots go to
RESULTS_FOLDER/SessionTypeComparison/<arena>/SessionType_<arena>_*.png; the stats
tables of all arenas are stacked (with an 'Arena' column) into 'SesType_*'
sheets of the summary workbook. The arena comparison then pools all session
types within each arena.

v24 changes to Step 3 (pass index; compared against jrclimer/Pass_Index):
  * Positions are Gaussian-smoothed (POS_SMOOTH_SEC) before arc length / field index,
    and arc length accumulates only while speed >= MIN_SPEED_CM_S; spikes fired
    below that speed are excluded (they otherwise pile up as vertical stripes at a
    frozen pass index).
  * Field index along the path is bilinearly interpolated from the field-index map
    (was a nearest-bin staircase).
  * Place fields are isolated with PlaceCell_Main_v20's threshold method (bins >= 20%
    of peak AND above the mean rate, visited >= 2 times, 8-connected components of
    >= 9 bins; 1-D angular map with wrap-around on the Circle track), on Climer's
    rate map (sigma 6 cm, unvisited bins nearest-filled). See detect_place_fields.
  * Every field is analysed separately: its own spatial filter band (FILTER_LOW/
    HIGH_PER_RADIUS / r, r from the field's area; Butterworth order 3 as in
    pass_index_parser.m), and only passes whose field-index maximum lies inside the
    field and reaches PASS_MIN_FIELD_PEAK_FRAC of the field's peak. The Hilbert pass
    index is amplitude-blind, so without this gate out-of-field wiggles of the
    filtered trace count as full -1 -> +1 passes (4-15x more passes than entries).
    Field 1 = highest peak rate. One Excel row and one figure (<unit>_F<k>_...) per field.
  * Kempter z-test run on one event per burst (BURST_COLLAPSE_FOR_ZTEST).
  * Density-map occupancy restricted to the same running / kept-pass samples.

v25 changes (complete field traversals):
  * The whole circular-linear analysis (Kempter fit, burst-collapsed z-test, fit_R
    shuffle gate, classification, phase at pass index -1/0/+1, density map) is repeated
    on spikes from COMPLETE traversals only: stretches of a kept pass whose pass index
    starts near -1, crosses 0 and ends near +1 (COMPLETE_PASS_EDGE_TOL). Stretches where
    the pass-index phase turns back across the -1/+1 boundary, or that are cut off by the
    start/end of the recording, are dropped.
  * The PassIndex figure gets a third row below the all-spike row: phase vs pass index,
    density map and fit statistics for complete traversals. Excel columns n_passes_complete
    and CP_* hold the complete-traversal results. Classification, class folders and the
    arena / session-type comparisons still use the all-spike fit.
  Per-unit figures go to <session>/ThetaMod_PhasePrecession_v25.

v26 changes (multi-peak fields split into sub-fields):
  * Right after field detection, every place field is tested for multiple peaks with the
    KDE method of PlaceCell_MultiPeak_KDE_v2.py (see split_multipeak_fields): a weighted
    Gaussian KDE of the field's bins (weights = MP_KDE_RATEMAP_TYPE rate map), local maxima
    >= MP_PEAK_REL_HEIGHT_FRAC of the field's highest KDE peak counted as peaks, and a field
    with >= 2 peaks cut at the troughs between them by marker-based watershed on the KDE
    (1-D angular map with wrap-around on the Circle track).
  * The sub-fields then REPLACE their parent field for everything downstream: each has its
    own spatial filter band (from its own radius), its own pass gate (pass field-index
    maximum inside the sub-field, >= PASS_MIN_FIELD_PEAK_FRAC of the sub-field's peak),
    figures and Excel row. Treating a multi-peak field as one field makes the field index
    rise and fall once per peak along a single run, which turns into a jittery pass index.
  * Sub-fields of < MIN_SUBFIELD_SIZE_BINS bins are not analysed (their bins belong to no
    field); they still get an Excel row with PrecessionSkippedReason set.
  * Field IDs: '<k>' for a single-peak field, '<k>.<j>' for sub-field j of field k (j = 1 is
    the highest KDE peak). Figures are <unit>_F<id>_PassIndex.png.
  Per-unit figures go to <session>/ThetaMod_PhasePrecession_v26.

Requires: numpy, scipy, pandas, matplotlib, openpyxl (for writing .xlsx).
"""

#!Correct phase slope testing! Most precessing/recessing cells classified as phase locked.
#!Criteria is too strict.

from __future__ import annotations

import copy
import functools
import heapq
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
from matplotlib.patches import Rectangle
from scipy import signal, stats
from scipy.ndimage import gaussian_filter, gaussian_filter1d, distance_transform_edt, label as nd_label
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize_scalar
from scipy.special import erf

# Variable to optimize: MIN_SPEED_CM_S, PASS_MIN_FIELD_PEAK_FRAC, FILTER_HIGH_PER_RADIUS, POS_SMOOTH_SEC.
!Add theta modulation passed proportion and absolute cell number. 
!The plot in place cell char gives supiciusly high number of theta modulated cells.

!For onset, peak, and offset plots, color code the type of phase relationship for each data point.
# ============================================================================
# Configuration -- EDIT THESE
# ============================================================================
#ROOT_FOLDER = Path(r"X:\NMR_group_data\Runita\Analysis\Thesis\Data_v2_Accepted")

ROOT_FOLDER = Path(r"X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\OldVers\PC_True_SiAdptBin_CI99_Bin2") 
# All pooled results (Excel/CSV files, comparison plots, per-class plot copies) go here
RESULTS_FOLDER = ROOT_FOLDER / 'Results' / 'PhasePrec_Results_v26_MultiPeakSplit'
OUTPUT_EXCEL_NAME = 'theta_phase_v26_MultiPeakSplit.xlsx'   # written to RESULTS_FOLDER
# PrecessionClass -> folder (in RESULTS_FOLDER, beside the Excel file) receiving a copy of each cell's PassIndex plot
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
# True (default): match each .ntt tetrode file to the .ncs LFP file sharing its
# embedded channel number (e.g. TT1.ntt -> CSC1.ncs), via match_ncs_to_ntt() --
# more accurate (each tetrode's spikes are referenced to their own LFP channel),
# but loading/filtering a separate LFP file per matched channel can be slow.
# False: skip matching and reference every tetrode in the session to the first
# .ncs file found (sorted naturally, e.g. CSC1 before CSC2/CSC10), loaded and
# filtered once for the whole session -- much faster, at the cost of using one
# session-wide LFP channel instead of each tetrode's own.
MATCH_SPIKE_LFP_FILE = False

# --- ACG theta-positive epoch criterion (Dunn et al. 2022; see acg_theta_epoch_mask) ---
# True: only spikes falling in 1 s LFP epochs whose autocorrelogram matches a 3-7 Hz
# sinusoid well (ED_min < ACG_ED_MIN_THRESH) are used for Steps 1-3 (polar plot, TMI,
# phase precession). The rate map / place field is still built from all spikes, so the
# field definition is unaffected. False: use the entire signal, as before.
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

# --- Step 3: Pass Index phase-precession parameters ---
METHOD = 'place'              # 'place' (recommended for place cells) or 'grid'
BINSIDE = 2.0                 # spatial bin size (position units, e.g. cm). 'auto' = 4
SMTH_WIDTH = 12.0             # rate-map Gaussian smoothing width (= 2 * sigma; sigma = 6 cm).
                              # 'auto' = 3 * BINSIDE
FILTER_BAND = 'auto'          # (low, high) cycles/unit-distance for the spatial filter
LFP_FILTER_BAND = (3.0, 7.0) # Hz, theta band used for both the theta-modulation test and LFP phase
# --- v24 pass-index fixes (see compute_pass_index) ---
POS_SMOOTH_SEC = 0.2          # Gaussian sigma (s) applied to x/y before arc length, speed and field index;
                              # removes frame-to-frame tracking jitter (~20% of raw arc length in this data)
MIN_SPEED_CM_S = 1.0          # arc length accumulates, and spikes are analysed, only while smoothed speed >= this
FILTER_LOW_PER_RADIUS = 1.0 / 6.0   # 'auto' band low edge = this / r cycles/cm (pass_index_parser.m: 1/(6r)),
FILTER_HIGH_PER_RADIUS = 1.0  # 'auto' band high edge = this / r cycles/cm (pass_index_parser.m: 3/r);
                              # r = radius of the circle with the same area as the analysed place field
SPATIAL_FILTER_ORDER = 3      # Butterworth order of the field-index filter (pass_index_parser.m: butter(3, Wn))
PASS_MIN_FIELD_PEAK_FRAC = 0.5  # a pass (one -1 -> +1 cycle of the pass index) is assigned to field k only if
                              # its field-index maximum lies inside field k AND reaches this fraction of field
                              # k's own peak field index; other "passes" are amplitude-blind Hilbert noise
# --- v25: complete field traversals (see compute_pass_index / _precession_fit) ---
# The whole circular-linear analysis is repeated on spikes from COMPLETE traversals only:
# a contiguous stretch of the arc-sampled trace inside one kept pass whose pass index
# starts near -1 (field entry), crosses 0 (field centre) and ends near +1 (field exit).
# Stretches where the pass-index phase turns back across the -1 / +1 boundary (entered
# from +1, or left again through -1) or that are cut off by the start/end of the
# recording are rejected, along with their spikes.
COMPLETE_PASS_EDGE_TOL = 0.1  # pass index must start <= -1 + this and end >= +1 - this
# --- Place-field definition: "threshold method" of PlaceCell_Main_v20 (see detect_place_fields) ---
# Applied to the pass-index rate map, which keeps Climer's rate_map.m smoothing (Gaussian
# sigma = SMTH_WIDTH / 2 = 6 cm, unvisited bins filled with the nearest visited value)
# instead of PlaceCell_Main_v20's sigma = 3 cm normalised convolution. Every field of a
# multi-field cell is analysed separately (one Excel row and one figure per field);
# Field 1 = highest peak (smoothed) rate, then descending.
FIELD_RATE_THRESHOLD_FRAC = 0.20   # bin qualifies if smoothed rate >= this * peak AND > the cell's mean
                                   # (raw, occupancy-weighted) rate (METHOD2_RATE_THRESHOLD_FRAC)
MIN_FIELD_SIZE_BINS = 9            # >= this many 8-connected bins (angular contiguous bins on the Circle)
FIELD_MIN_VISITS = 2               # valid bin = occupied and entered >= this many times (use_min_visits)
RATEMAP_IMMOBILITY_SPEED_CM_S = 0.5  # frames (and their spikes) below this speed are left out of the rate
                                   # map / field detection (IMMOBILITY_SPEED_CMS)
RING_RADIAL_CLIP_PCTILE = 0.5      # Circle: track width = radial range between these percentiles
                                   # (RADIAL_RANGE_CLIP_PCTILE)
# --- v26: multi-peak fields split into sub-fields (KDE method of PlaceCell_MultiPeak_KDE_v2;
# see split_multipeak_fields). Keep the MP_ settings equal to that script's run. ---
SPLIT_MULTIPEAK_FIELDS = True      # False: every field analysed whole, as in v25
MIN_SUBFIELD_SIZE_BINS = 9         # sub-fields with fewer bins are not analysed for phase precession
MP_PEAK_REL_HEIGHT_FRAC = 0.75     # a KDE peak counts if its height is >= this fraction of the field's
                                   # highest KDE peak (PEAK_REL_HEIGHT_FRAC)
MP_MIN_TROUGH_DIP_FRAC = 0.0       # ...and the KDE dips below (1 - this) x its height on the way to every
                                   # higher peak (MIN_TROUGH_DIP_FRAC)
MP_KDE_BW_METHOD = 0.5             # gaussian_kde bw_method ('scott', 'silverman' or a scalar factor)
MP_MIN_KDE_POINTS = 3              # fields with fewer non-zero-rate bins get no KDE (single peak)
# Rate map the KDE is fitted on (weights of the field bins):
#   'raw'                     : spikes / occupancy, unsmoothed
#   'gaussian_smooth'         : the Climer rate map the fields are detected on (sigma = SMTH_WIDTH / 2)
#   'Yartsev_adaptive_binned' : adaptive disc grows until it holds MP_YARTSEV_DISC_OCC_S of occupancy
#   'Skaggs_adaptive_binned'  : adaptive disc grows until r_cm * occ_s * sqrt(spikes) >= MP_SKAGGS_ALPHA
MP_KDE_RATEMAP_TYPE = 'Skaggs_adaptive_binned'
MP_ADAPTIVE_MAX_RSQ = 200          # disc stops growing at r^2 = this - 1 (bins^2)
MP_ADAPTIVE_MIN_OCC_S = 0.5        # bins whose final disc holds < this occupancy (s) get rate 0
MP_YARTSEV_DISC_OCC_S = 1.0
MP_SKAGGS_ALPHA = 200.0
_MP_KDE_RATEMAP_TYPES = ('raw', 'gaussian_smooth', 'Yartsev_adaptive_binned', 'Skaggs_adaptive_binned')
if MP_KDE_RATEMAP_TYPE not in _MP_KDE_RATEMAP_TYPES:
    raise ValueError(f'MP_KDE_RATEMAP_TYPE must be one of {_MP_KDE_RATEMAP_TYPES}, got {MP_KDE_RATEMAP_TYPE!r}')
BURST_COLLAPSE_FOR_ZTEST = True  # Kempter z-test on one event per burst (spikes < TMI_BURST_MAX_GAP_SEC apart:
                              # circular-mean phase, mean pass index) so bursts are not counted as independent
SLOPE_BNDS = (-1.0, 0.25)     # (low, high) bound on precession slope (cycles/unit); x spans 2
                              # units per pass, so this is -720..+180 deg/pass. None = +-4 cycles/pass
MIN_SPIKES_FOR_FIT = 50       # skip circular-linear fit if fewer spikes than this
SLOPE_THRESH_DEG_PER_PASS = 15.0   # significant fits need |slope| (deg/pass) above this to be
                              # precessing (negative) / succeeding (positive); else phase locked
MAX_FIT_LINES = 3            # a fit whose wrapped phase line has more segments than this is
                              # marked non-precessing and gets no further analysis
# --- Step 3 gate: fit_R shuffle test (see shuffle_fit_r_significance) ---
# Before the Kempter z-test is used to classify a cell, its fit_R must beat a null in
# which the phase-position pairing is broken: whole bursts (spikes < TMI_BURST_MAX_GAP_SEC
# apart, as in the TMI shuffle but with no phase-bin condition) swap phases with each other,
# keeping the cell's phase distribution (its theta locking) and burst structure intact.
# Cells failing the gate are classed no_phase_relation.
USE_FIT_R_GATE = True
N_FIT_R_SHUFFLES = 1000             # burst-preserving phase permutations; each refits anglereg

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

# --- Theta phase at pass index 0 (field center), compared across arenas ---
# Phase at pass index 0 = the fitted line 2*pi*s*x + b evaluated at x = 0, i.e. the
# intercept b, wrapped to [0, 360) deg. On the circle the precession range is
# [70, 250) deg -- 70-250 and 430-610 deg on the double-cycle 0-720 plot axis; the
# rest, 250-430 deg (= 250-360 + 0-70) and 610-720 deg, is the procession range.
PI0_PRECESSION_RANGE_DEG = (70.0, 250.0)
PI0_PHASE_CLASSES = ('phase_precessing', 'phase_succeeding', 'phase_locked', 'no_phase_relation')  # cells included
N_PI0_PERMUTATIONS = 10000          # label permutations for the Mardia-Watson-Wheeler test
PI0_RANGE_COLORS = {'precession': '#8C2981', 'procession': '#FCFDBF'}   # magma light purple (0.4) / light yellow (1.0)
# Pass-index points at which the fitted line 2*pi*s*x + b is evaluated, each extracted,
# plotted and compared across arenas exactly as for pass index 0 (same cells, same
# precession/procession ranges, same circular tests):
# (pass index x, column key -> phase_at_<key>_deg, plot label, file tag, sheet tag)
PASS_INDEX_PHASE_POINTS = (
    (-1.0, 'pim1', 'field entry (pass index -1)', 'PassIndexMinus1', 'PIm1'),
    (0.0, 'pi0', 'field center (pass index 0)', 'PassIndex0', 'PI0'),
    (1.0, 'pip1', 'field exit (pass index +1)', 'PassIndexPlus1', 'PIp1'),
)


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

    Assumes continuously-sampled, full 512-sample records (standard for an
    uninterrupted Neuralynx CSC recording), matching the rest of this repo's
    .ncs readers.
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
    timestamps_us = (records['TimeStamp'][:, None].astype(np.float64) + offsets_us[None, :]).ravel()
    samples_uV = (records['Samples'].astype(np.float64) * adbitvolts * 1e6).ravel()

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
# Rate map / field index
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


def field_index_map(rmap, occupancy, method):
    """Normalize the rate map into a 0-1 field index map."""
    fi_map = rmap.copy()
    fi_map[occupancy == 0] = np.nan
    valid = ~np.isnan(fi_map)

    if method == 'grid':
        vals = fi_map[valid]
        order = np.argsort(vals)
        ranked = np.empty_like(vals)
        ranked[order] = np.linspace(0, 1, len(vals))
        fi_map[valid] = ranked
    else:  # 'place'
        vmin, vmax = np.nanmin(fi_map), np.nanmax(fi_map)
        rng = vmax - vmin
        fi_map[valid] = (fi_map[valid] - vmin) / rng if rng > 0 else 0.0

    return fi_map


def smooth_positions(pos_ts, pos_xy, sigma_sec=POS_SMOOTH_SEC):
    """Gaussian-smoothed positions and speed (cm/s). Raw tracking jitter (~0.1-0.2 cm
    per frame even when the animal sits still) otherwise inflates arc length and makes
    the field index flicker across bin edges, which the pass-index filter turns into
    spurious passes."""
    dt = float(np.median(np.diff(pos_ts)))
    sm = gaussian_filter1d(pos_xy, sigma_sec / dt, axis=0, mode='nearest') if sigma_sec > 0 else pos_xy
    speed = np.hypot(*np.gradient(sm, pos_ts, axis=0).T)
    return sm, speed


def field_index_per_position(pos_xy, fi_map, x_edges, y_edges):
    """Field index at every position sample, bilinearly interpolated between bin
    centres (unvisited bins filled with the nearest visited value first). The
    previous nearest-bin lookup gave a 2 cm staircase whose steps leak through the
    pass-index band-pass filter."""
    xc, yc = 0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (y_edges[1:] + y_edges[:-1])
    interp = RegularGridInterpolator((xc, yc), _fill_nan_nearest(fi_map), bounds_error=False, fill_value=None)
    return interp(np.column_stack([np.clip(pos_xy[:, 0], xc[0], xc[-1]),
                                   np.clip(pos_xy[:, 1], yc[0], yc[-1])]))


def sample_along_arc(pos_ts, pos_xy, field_index, running):
    """Resample field_index at evenly spaced steps along the arc length (path
    distance) traversed, as in pass_index_parser.m sample_along_arc, except that
    arc length only accumulates while `running` (stationary periods collapse to a
    point instead of contributing jitter distance)."""
    step = np.concatenate(([0.0], np.hypot(*np.diff(pos_xy, axis=0).T))) * running
    arc = np.cumsum(step)
    moving = np.concatenate(([True], np.diff(arc) > 0))
    arc_m, ts_m = arc[moving], pos_ts[moving]
    if len(arc_m) < 2 or arc_m[-1] <= 0:
        raise ValueError('no running epochs (speed >= MIN_SPEED_CM_S) in tracking')

    cc = np.linspace(0, arc_m[-1], len(ts_m))
    ts2 = np.interp(cc, arc_m, ts_m)
    resampled = np.interp(ts2, pos_ts, field_index)
    return cc, ts2, resampled


def _interp_nearest_extrap(x_ref, y_ref, x_query):
    idx = np.searchsorted(x_ref, x_query)
    idx = np.clip(idx, 1, len(x_ref) - 1)
    left, right = idx - 1, idx
    use_left = np.abs(x_query - x_ref[left]) <= np.abs(x_ref[right] - x_query)
    idx = np.where(use_left, left, right)
    return y_ref[idx]


def bandpass_filter(data, low, high, fs, order=4, warn_clamp=False):
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
    if warn_clamp and high / nyq >= 1:
        print(f'    WARNING: filter high edge {high:.4g} >= Nyquist {nyq:.4g}; clamped (no low-pass applied)')
    sos = signal.butter(order, [low_n, high_n], btype='band', output='sos')
    filtered = signal.sosfiltfilt(sos, data)
    if np.any(np.isnan(filtered)):
        filtered = signal.sosfilt(sos, data)
    return filtered


def auto_filter_band(method, field_radius_cm):
    """Spatial filter band (cycles/cm). 'place': from the analysed field's radius r
    (radius of the circle with the field's area; n_dims == 2 case of
    pass_index_parser.m), FILTER_LOW/HIGH_PER_RADIUS / r."""
    if method == 'grid':
        return (1.0 / (2 * 170.0), 1.0 / (26.7 / 8.0))
    r = max(float(field_radius_cm), 1e-6)
    return (FILTER_LOW_PER_RADIUS / r, FILTER_HIGH_PER_RADIUS / r)


# ============================================================================
# Place-field isolation: "threshold method" of PlaceCell_Main_v20
# (detect_place_fields_threshold_2d / _circular), on Climer's rate map
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


def _field_record_2d(mask, rmap, raw, valid, binside, xc, yc):
    """Field dict (peak rates, size, rate-weighted centre of mass) of the 2-D bins in `mask`."""
    n_bins = int(mask.sum())
    rates = rmap[mask]
    bx, by = np.nonzero(mask)
    area = n_bins * binside ** 2
    return dict(
        mask_2d=mask, peak_rate_Hz=float(rates.max()),
        peak_rate_raw_Hz=float(np.nanmax(raw[mask])),
        n_bins=n_bins, area_cm2=area, radius_cm=float(np.sqrt(area / np.pi)),
        pct_of_occupied_area=100.0 * n_bins / int(valid.sum()),
        com_x_cm=float(np.sum(rates * xc[bx]) / rates.sum()),
        com_y_cm=float(np.sum(rates * yc[by]) / rates.sum()),
        com_theta_deg=np.nan, arc_length_cm=np.nan, track_width_cm=np.nan)


def _detect_fields_2d(rmap, valid, mean_fr, record):
    """Threshold method on the 2-D map: valid bins with smoothed rate >=
    FIELD_RATE_THRESHOLD_FRAC * peak AND > the mean raw rate, 8-connected
    components of >= MIN_FIELD_SIZE_BINS bins. record(mask) -> field dict."""
    peak = float(rmap[valid].max())
    qualifies = valid & (rmap >= FIELD_RATE_THRESHOLD_FRAC * peak) & (rmap > mean_fr)
    labels, n_comp = nd_label(qualifies, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(labels.ravel(), minlength=n_comp + 1)
    fields = [record(labels == c) for c in range(1, n_comp + 1) if sizes[c] >= MIN_FIELD_SIZE_BINS]
    return _number_fields(fields), peak


def _detect_fields_circular(pos_ts, pos_xy, moving, spk_xy_moving, binside, smth_width,
                            x_edges, y_edges, occupancy_2d):
    """Threshold method on the 1-D angular map of the Circle track
    (PlaceCell_Main_v20 _build_ratemap_circular / detect_place_fields_threshold_circular):
    bearing around the fitted ring centre binned at ~binside cm of arc, radius discarded,
    smoothed Climer-style (nearest-fill of unvisited bins, Gaussian sigma = smth_width / 2,
    wrapping around the ring); qualifying runs of >= MIN_FIELD_SIZE_BINS bins, wrapping
    across the 0/360 deg seam. Each field also gets a 2-D display mask (occupied 2-D
    bins whose centre bearing falls in the field). Returns (fields, mean_fr, peak,
    ring) with ring = dict(cx, cy, theta_bin, n_bins, record, occ, spk, valid, raw,
    smooth): theta_bin(xy) -> angular bin, record(run) -> field dict of angular bins
    `run`, and the angular maps (for the multi-peak KDE)."""
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
    theta_c = (np.arange(n_t) + 0.5) * bin_w

    def record(run):
        run = np.asarray(run)
        rates = smooth[run]
        arc = len(run) * bin_w * r_med
        area = arc * track_w
        com = np.mod(np.arctan2(np.sum(rates * np.sin(theta_c[run])), np.sum(rates * np.cos(theta_c[run]))),
                     2 * np.pi)
        return dict(
            theta_bins=run, mask_2d=np.isin(grid_tbin, run) & (occupancy_2d > 0),
            peak_rate_Hz=float(rates.max()), peak_rate_raw_Hz=float(np.nanmax(raw[run])),
            n_bins=len(run), area_cm2=area, radius_cm=float(np.sqrt(area / np.pi)),
            pct_of_occupied_area=100.0 * len(run) / int(valid.sum()),
            com_x_cm=float(cx + r_med * np.cos(com)), com_y_cm=float(cy + r_med * np.sin(com)),
            com_theta_deg=float(np.rad2deg(com)), arc_length_cm=arc, track_width_cm=track_w)

    fields = _number_fields([record(run) for run in _circular_runs(qualifies)
                             if len(run) >= MIN_FIELD_SIZE_BINS])
    ring = dict(cx=cx, cy=cy, theta_bin=_theta_bin, n_bins=n_t, record=record,
                occ=occ, spk=spk, valid=valid, raw=np.nan_to_num(raw), smooth=smooth)
    return fields, mean_fr, peak, ring


def detect_place_fields(pos_ts, pos_xy, spk_ts, arena, binside, smth_width, method=METHOD) -> dict[str, Any]:
    """Climer rate map + field-index map, and the place fields of PlaceCell_Main_v20's
    threshold method on it (2-D, or angular on the Circle track). Immobile frames
    (< RATEMAP_IMMOBILITY_SPEED_CM_S, smoothed speed) and their spikes are left out,
    as in PlaceCell_Main_v20. Right after detection, multi-peak fields are split into
    their sub-fields (split_multipeak_fields), which replace them from here on.
    Returns a dict with rate_map, occupancy, x/y_edges, field_index_map, fields (the
    fields / sub-fields to analyse, field_number = 1..n unique label, field_id =
    '<k>' or '<k>.<j>'), dropped_subfields (sub-fields < MIN_SUBFIELD_SIZE_BINS bins),
    n_detected_fields, mean_fr, and field_label(xy) -> field_number (0 = outside every
    analysed field) for positions."""
    _, speed = smooth_positions(pos_ts, pos_xy)
    moving = speed >= RATEMAP_IMMOBILITY_SPEED_CM_S
    spk_xy, spk_idx = spk_pos(pos_ts, pos_xy, spk_ts)
    spk_xy_moving = spk_xy[moving[spk_idx]]

    rmap, occupancy, x_edges, y_edges, spk_counts = rate_map(
        pos_ts, pos_xy, spk_xy_moving, binside, smth_width, sample_mask=moving, return_counts=True)
    fi_map = field_index_map(rmap, occupancy, method)

    nx, ny = len(x_edges) - 1, len(y_edges) - 1

    def _xy_bins(p):
        return (np.clip(np.digitize(p[:, 0], x_edges) - 1, 0, nx - 1),
                np.clip(np.digitize(p[:, 1], y_edges) - 1, 0, ny - 1))

    if arena == 'Circle':
        fields, mean_fr, peak, ring = _detect_fields_circular(
            pos_ts, pos_xy, moving, spk_xy_moving, binside, smth_width, x_edges, y_edges, occupancy)
        detection = 'Circle (1-D angular)'
        kde_map = _kde_input_map_circular(ring, binside) if fields and SPLIT_MULTIPEAK_FIELDS else None

        def analyse(f):
            return f['theta_bins'], _analyse_field_circular(f['theta_bins'], kde_map)

        sub_record = ring['record']
    else:
        bx, by = _xy_bins(pos_xy[moving])
        valid = (occupancy > 0) & (_visit_counts(bx * ny + by, nx * ny).reshape(nx, ny) >= FIELD_MIN_VISITS)
        xc, yc = 0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (y_edges[1:] + y_edges[:-1])
        with np.errstate(invalid='ignore', divide='ignore'):
            raw = np.where(occupancy > 0, spk_counts / occupancy, np.nan)
        mean_fr = float(spk_counts[valid].sum() / occupancy[valid].sum())

        def record(mask):
            return _field_record_2d(mask, rmap, raw, valid, binside, xc, yc)

        fields, peak = _detect_fields_2d(rmap, valid, mean_fr, record)
        detection = '2-D grid'
        kde_map = (_kde_input_map_2d(rmap, raw, spk_counts, occupancy, valid, fields, binside)
                   if fields and SPLIT_MULTIPEAK_FIELDS else None)

        def analyse(f):
            nodes = np.argwhere(f['mask_2d'])
            return nodes, _analyse_field_2d([tuple(n) for n in nodes], kde_map)

        def sub_record(nodes):
            mask = np.zeros((nx, ny), dtype=bool)
            mask[nodes[:, 0], nodes[:, 1]] = True
            return record(mask)

    n_detected = len(fields)
    fields, dropped = split_multipeak_fields(fields, analyse, sub_record)
    for f in fields + dropped:
        f['peak_field_index'] = float(np.nanmax(fi_map[f['mask_2d']])) if f['mask_2d'].any() else np.nan

    # Position -> analysed field / sub-field (0 outside, and in sub-fields too small to analyse)
    if arena == 'Circle':
        theta_labels = np.zeros(ring['n_bins'], dtype=int)
        for f in fields:
            theta_labels[f['theta_bins']] = f['field_number']

        def field_label(p):
            return theta_labels[ring['theta_bin'](p)]
    else:
        label_map = np.zeros((nx, ny), dtype=int)
        for f in fields:
            label_map[f['mask_2d']] = f['field_number']

        def field_label(p):
            return label_map[_xy_bins(p)]

    return dict(rate_map=rmap, occupancy=occupancy, x_edges=x_edges, y_edges=y_edges,
                field_index_map=fi_map, fields=fields, dropped_subfields=dropped,
                n_detected_fields=n_detected, mean_fr=mean_fr, peak_rate=peak,
                detection=detection, field_label=field_label)


# ============================================================================
# v26: multi-peak fields -> sub-fields (KDE method of PlaceCell_MultiPeak_KDE_v2.py)
#
# Per field: weighted Gaussian KDE of the field's bin centres (weights = firing rate in
# the MP_KDE_RATEMAP_TYPE map), local maxima inside the field (8-neighbourhood; left/right
# on the Circle's angular map; the KDE is also evaluated on a 1-bin ring around the field
# so an edge bin only counts if the KDE really falls off outside it), peaks >=
# MP_PEAK_REL_HEIGHT_FRAC of the highest counted, and >= 2 peaks -> the field is cut at
# the troughs by marker-based watershed on the KDE.
# ============================================================================

def _adaptive_rate(spk_in, valid, spk_disc, occ_disc, disc_r_bins, binside):
    """Yartsev or Skaggs adaptive-disc rate (MP_KDE_RATEMAP_TYPE) of every bin from the
    spikes / occupancy inside its candidate discs (disc index 0..n_discs-1 along axis 0,
    bins flattened along axis 1); the first disc meeting the criterion is used, else the
    largest. Bins whose final disc holds < MP_ADAPTIVE_MIN_OCC_S of occupancy get rate 0."""
    rate = np.zeros(valid.shape, dtype=np.float64)
    if not valid.any() or spk_in[valid].sum() <= 0:
        return rate
    if MP_KDE_RATEMAP_TYPE == 'Yartsev_adaptive_binned':
        satisfied = occ_disc >= MP_YARTSEV_DISC_OCC_S
    else:   # Skaggs & McNaughton 1998, r = disc radius + half a bin, in cm
        r_cm = (disc_r_bins + 0.5) * binside
        satisfied = r_cm[:, None] * occ_disc * np.sqrt(spk_disc) >= MP_SKAGGS_ALPHA
    k = np.where(satisfied.any(axis=0), satisfied.argmax(axis=0), satisfied.shape[0] - 1)
    cols = np.arange(valid.size)
    spk_fin = spk_disc[k, cols].reshape(valid.shape)
    occ_fin = occ_disc[k, cols].reshape(valid.shape)
    ok = valid & (occ_fin >= MP_ADAPTIVE_MIN_OCC_S)
    rate[ok] = spk_fin[ok] / occ_fin[ok]
    return rate


def _kde_input_map_2d(rmap, raw, spk_counts, occupancy, valid, fields, binside):
    """MP_KDE_RATEMAP_TYPE rate map on the 2-D grid. Only the field bins are used as KDE
    weights, so the adaptive maps are computed on the fields' bounding box plus the
    largest disc radius (exact for every field bin); bins outside it are 0."""
    if MP_KDE_RATEMAP_TYPE == 'raw':
        return np.where(valid, np.nan_to_num(raw), 0.0)
    if MP_KDE_RATEMAP_TYPE == 'gaussian_smooth':
        return rmap
    pad = int(np.floor(np.sqrt(MP_ADAPTIVE_MAX_RSQ - 1)))
    ix, iy = np.nonzero(np.any([f['mask_2d'] for f in fields], axis=0))
    box = (slice(max(ix.min() - pad, 0), ix.max() + pad + 1),
           slice(max(iy.min() - pad, 0), iy.max() + pad + 1))
    v = valid[box]
    spk_in = np.where(v, spk_counts[box], 0.0)
    occ_in = np.where(v, occupancy[box], 0.0)
    nbx, nby = v.shape

    # Disc offsets sorted by r^2; disc k (r^2 <= k) holds the first n_enclosed[k] offsets.
    ax = np.arange(-pad, pad + 1)
    dx, dy = (a.ravel() for a in np.meshgrid(ax, ax, indexing='ij'))
    d2 = dx ** 2 + dy ** 2
    keep = d2 <= MP_ADAPTIVE_MAX_RSQ - 1
    order = np.argsort(d2[keep], kind='stable')
    dx, dy, d2 = dx[keep][order], dy[keep][order], d2[keep][order]
    bx, by = np.meshgrid(np.arange(nbx), np.arange(nby), indexing='ij')
    gather = (bx.ravel()[None, :] + pad + dx[:, None]) * (nby + 2 * pad) + (by.ravel()[None, :] + pad + dy[:, None])
    n_enclosed = np.searchsorted(d2, np.arange(MP_ADAPTIVE_MAX_RSQ), side='right')
    spk_disc = np.cumsum(np.pad(spk_in, pad).ravel()[gather], axis=0)[n_enclosed - 1]
    occ_disc = np.cumsum(np.pad(occ_in, pad).ravel()[gather], axis=0)[n_enclosed - 1]

    out = np.zeros(valid.shape, dtype=np.float64)
    out[box] = _adaptive_rate(spk_in, v, spk_disc, occ_disc, np.sqrt(d2[n_enclosed - 1]), binside)
    return out


def _kde_input_map_circular(ring, binside):
    """MP_KDE_RATEMAP_TYPE rate map on the Circle track's 1-D angular map; the adaptive
    disc becomes an interval of +-r bins along the track (wrapping around the seam),
    growing in whole bins up to floor(sqrt(MP_ADAPTIVE_MAX_RSQ - 1))."""
    valid = ring['valid']
    if MP_KDE_RATEMAP_TYPE == 'raw':
        return np.where(valid, ring['raw'], 0.0)
    if MP_KDE_RATEMAP_TYPE == 'gaussian_smooth':
        return ring['smooth']
    spk_in = np.where(valid, ring['spk'], 0.0)
    occ_in = np.where(valid, ring['occ'], 0.0)
    n = len(occ_in)
    r_max = min(int(np.floor(np.sqrt(MP_ADAPTIVE_MAX_RSQ - 1))), (n - 1) // 2)
    spk_disc = np.empty((r_max + 1, n))
    occ_disc = np.empty((r_max + 1, n))
    spk_disc[0], occ_disc[0] = spk_in, occ_in
    for r in range(1, r_max + 1):
        spk_disc[r] = spk_disc[r - 1] + np.roll(spk_in, r) + np.roll(spk_in, -r)
        occ_disc[r] = occ_disc[r - 1] + np.roll(occ_in, r) + np.roll(occ_in, -r)
    return _adaptive_rate(spk_in, valid, spk_disc, occ_disc, np.arange(r_max + 1, dtype=float), binside)


def _watershed(values, adj, seeds):
    """Marker-based watershed (priority flood, highest value first) over the field nodes
    0..len(values)-1. adj[i] lists node i's neighbours (indices >= len(values) are
    outside the field and ignored); seeds[k] gets label k+1. Each node takes the label
    of the basin that reaches it first while descending the surface, so basins meet at
    the troughs."""
    n_field = len(values)
    labels = np.zeros(n_field, dtype=int)
    heap = []
    for k, s in enumerate(seeds, start=1):
        labels[s] = k
        heapq.heappush(heap, (-values[s], s))
    while heap:
        _, i = heapq.heappop(heap)
        for j in adj[i]:
            if j < n_field and labels[j] == 0:
                labels[j] = labels[i]
                heapq.heappush(heap, (-values[j], j))
    return labels


def _merge_levels(rate, adj):
    """Union-find over the field bins in descending KDE order (0-D topological
    persistence): for every bin, the KDE level at which the region it tops merges into a
    region with a higher top, i.e. the trough that has to be crossed to reach a higher
    peak. The field's highest bin never merges (inf)."""
    n_field = len(rate)
    parent = np.full(n_field, -1)
    top = np.arange(n_field)
    merge = np.full(n_field, np.inf)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in np.argsort(-rate, kind='stable'):
        parent[i] = i
        for j in adj[i]:
            if j >= n_field or parent[j] < 0:
                continue
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            # Higher top survives; exact ties go to the lower bin index (same order as the peaks).
            ti, tj = top[ri], top[rj]
            hi, lo = (ri, rj) if (rate[ti], -ti) >= (rate[tj], -tj) else (rj, ri)
            merge[top[lo]] = rate[i]
            parent[lo] = hi
    return merge


def _kde_multipeak(pts_ext, n_field, adj, w):
    """KDE peak search + trough cut for one field (PlaceCell_MultiPeak_KDE_v2._kde_multipeak).
    pts_ext: (d, n_ext) bin-centre coordinates in bins, the field's n_field bins first,
    then a 1-bin ring around it; adj: neighbours (indices into pts_ext) of each field bin;
    w: KDE input rate of each field bin. Returns dict(rate, labels, counted, ref_hz,
    kde_note); labels[i] = sub-field (1..len(counted)) of field bin i, counted = peak bins
    in descending KDE height."""
    d = pts_ext.shape[0]
    pos = w > 0

    def single(note):
        peak = int(np.argmax(w))
        return dict(rate=np.full(n_field, np.nan), labels=np.ones(n_field, dtype=int),
                    counted=[peak], ref_hz=np.nan, kde_note=note)

    if pos.sum() < MP_MIN_KDE_POINTS:
        return single(f'{int(pos.sum())} non-zero-rate bin(s) < MP_MIN_KDE_POINTS: no KDE, single peak')

    # Collinear non-zero-rate bins (a 1-bin-wide strip) make the 2-D covariance singular:
    # fit a 1-D KDE along the field's long (first principal) axis instead.
    X = pts_ext[:, :n_field][:, pos]
    evals, evecs = np.linalg.eigh(np.atleast_2d(np.cov(X, aweights=w[pos])))
    if evals.max() <= 0:
        return single('all non-zero-rate bins at one location: no KDE, single peak')
    proj = evecs[:, -1][None, :] if d > 1 and evals.min() <= 1e-6 * evals.max() else np.eye(d)
    kde = stats.gaussian_kde(proj @ X, bw_method=MP_KDE_BW_METHOD, weights=w[pos])

    # sum_i r_i K(x - x_i), K integrating to 1 over one bin -> rate in Hz
    rate_ext = kde(proj @ pts_ext) * float(w.sum())
    rate = rate_ext[:n_field]

    # Local maxima inside the field (>= every neighbour, ring bins included); adjacent
    # maxima can only be exact ties -- keep one of them.
    cand = sorted((i for i in range(n_field) if all(rate_ext[i] >= rate_ext[j] for j in adj[i])),
                  key=lambda i: -rate[i])
    local_max = []
    for i in cand:
        if not any(i in adj[p] for p in local_max):
            local_max.append(i)
    if not local_max:          # KDE maximum just outside a non-convex field
        local_max = [int(np.argmax(rate))]

    # A peak counts if it is tall enough AND a real trough separates it from every higher peak.
    merge = _merge_levels(rate, adj)
    ref_hz = float(rate[local_max[0]])
    counted = [local_max[0]] + [i for i in local_max[1:]
                                if rate[i] >= MP_PEAK_REL_HEIGHT_FRAC * ref_hz
                                and merge[i] < (1.0 - MP_MIN_TROUGH_DIP_FRAC) * rate[i]]
    return dict(rate=rate, labels=_watershed(rate, adj, counted), counted=counted,
                ref_hz=ref_hz, kde_note='')


def _analyse_field_2d(nodes, kde_map):
    """KDE multi-peak analysis of one 2-D field (nodes = list of (bx, by) bins)."""
    index = {n: k for k, n in enumerate(nodes)}
    ext_nodes = list(nodes)
    adj = []
    for bx, by in nodes:
        nbrs = []
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                if ddx == 0 and ddy == 0:
                    continue
                nb = (bx + ddx, by + ddy)
                k = index.get(nb)
                if k is None:                     # ring bin just outside the field
                    k = len(ext_nodes)
                    index[nb] = k
                    ext_nodes.append(nb)
                nbrs.append(k)
        adj.append(nbrs)
    pts_ext = np.asarray(ext_nodes, dtype=float).T + 0.5
    idx = np.asarray(nodes)
    return _kde_multipeak(pts_ext, len(nodes), adj, kde_map[idx[:, 0], idx[:, 1]])


def _analyse_field_circular(ordered, kde_map):
    """KDE multi-peak analysis of one Circle field along the track: `ordered` = its
    angular bins in track order (as _circular_runs returns them, seam-straddling runs
    included), analysed as positions 0..L-1 plus one ring bin at each end."""
    L = len(ordered)
    pos_ext = np.concatenate([np.arange(L), [-1, L]]).astype(float) + 0.5
    adj = [[i - 1 if i > 0 else L, i + 1 if i < L - 1 else L + 1] for i in range(L)]
    return _kde_multipeak(pos_ext[None, :], L, adj, kde_map[np.asarray(ordered)])


def split_multipeak_fields(fields, analyse, sub_record):
    """Replace every multi-peak field by its sub-fields (right after field detection).
    analyse(field) -> (bins, KDE result) with bins in the order the KDE result's labels
    refer to; sub_record(bins subset) -> field dict of those bins. Sub-fields of <
    MIN_SUBFIELD_SIZE_BINS bins are returned separately (not analysed). Every returned
    field gets parent_field, sub_field (NaN if not split), n_subfields, multi_peak,
    field_id ('<k>' / '<k>.<j>'), kde_peak_Hz / kde_peak_rel (its KDE peak, and that
    relative to the parent field's highest), kde_note, and a new field_number: 1..n
    over the analysed fields / sub-fields (0 for the dropped ones), in parent-field order.
    Returns (analysed, dropped)."""
    units, dropped = [], []
    for f in fields:
        k = f['field_number']
        bins, res = analyse(f) if SPLIT_MULTIPEAK_FIELDS else (None, None)
        n_peaks = len(res['counted']) if res is not None else 1
        info = dict(parent_field=k, n_subfields=n_peaks, multi_peak=n_peaks >= 2,
                    kde_note=res['kde_note'] if res is not None else 'multi-peak splitting disabled')
        if n_peaks < 2:
            ref = res['ref_hz'] if res is not None else np.nan
            units.append({**f, **info, 'sub_field': np.nan, 'field_id': str(k),
                          'kde_peak_Hz': ref, 'kde_peak_rel': 1.0 if np.isfinite(ref) else np.nan})
            continue
        for j, c in enumerate(res['counted'], start=1):
            sub = {**sub_record(bins[res['labels'] == j]), **info, 'sub_field': j, 'field_id': f'{k}.{j}',
                   'kde_peak_Hz': float(res['rate'][c]), 'kde_peak_rel': float(res['rate'][c] / res['ref_hz'])}
            (units if sub['n_bins'] >= MIN_SUBFIELD_SIZE_BINS else dropped).append(sub)
    for i, u in enumerate(units, start=1):
        u['field_number'] = i
    for d in dropped:
        d['field_number'] = 0
    return units, dropped


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

    Epoch i spans [t0 + i*ACG_EPOCH_SEC, t0 + (i+1)*ACG_EPOCH_SEC) where t0 is
    the first LFP sample's time. Returns (ok, ed_min): boolean mask and ED_min
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


def times_in_ok_epochs(t_s, t0_s, epoch_ok):
    """Boolean mask over times `t_s` (s) that fall inside a theta-positive epoch."""
    idx = np.floor((np.asarray(t_s) - t0_s) / ACG_EPOCH_SEC).astype(np.int64)
    inb = (idx >= 0) & (idx < len(epoch_ok))
    out = np.zeros(len(idx), dtype=bool)
    out[inb] = epoch_ok[idx[inb]]
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
    this is the angle theta, this theta value is an angular tranformation of 
    linear position in place field) - 2*pi*s*x. fit_R is that maximized mean
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
accounts for the fact that $x \in [-1, 1]$ means one pass equals 2 units, yielding $720s$ degrees per pass.
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
# Step 3: Pass index phase-precession computation for one unit
# ============================================================================

def shuffle_fit_r_significance(spk_ts, spk_pass_index, spk_theta_phase, observed_fit_r, rng,
                               slope_bnds=None, n_shuffles=N_FIT_R_SHUFFLES):
    """Null distribution for fit_R (max over slopes of the residual MRL, see anglereg)
    under 'theta phase is independent of pass index'.

    Spikes are grouped into bursts by the TMI shuffle's time rule (consecutive spikes
    < TMI_BURST_MAX_GAP_SEC apart) but WITHOUT its same-phase-bin condition: a burst
    straddling a bin edge would otherwise be split into groups that are permuted
    independently although they share position and phase, which narrows the null
    (on simulated phase-locked, non-precessing bursty cells the phase-bin rule gave
    ~12% false positives at ALPHA = 0.05, the time-only rule ~5%). Each surrogate permutes the bursts' circular-mean phases across bursts, each spike keeping
    its offset from its own burst's mean, and leaves every spike's pass index in place.
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
        theta_shuf = group_mean[rng.permutation(n_groups)][group_id] + offsets
        shuffle_fit_r[i] = anglereg(spk_pass_index, theta_shuf, slope_bnds)[2]

    pval = float((np.sum(shuffle_fit_r >= observed_fit_r) + 1) / (n_shuffles + 1))
    return pval, shuffle_fit_r


class TooFewSpikes(Exception):
    """Too few spikes left for the circular-linear fit after the pass-index gates."""


def burst_events(spk_ts, pass_index, theta):
    """Collapse each burst (consecutive spikes < TMI_BURST_MAX_GAP_SEC apart, time rule
    only, as in shuffle_fit_r_significance) to one event: circular-mean pass index (as
    an angle pass_index*pi, so a burst straddling +1/-1 is averaged correctly) and
    circular-mean theta phase. Returns (event_pass_index, event_theta)."""
    theta = np.mod(theta, 2 * np.pi)
    _, gid = np.unique(_burst_groups(spk_ts, np.rad2deg(theta), phase_bin_deg=360.0), return_inverse=True)
    n = int(gid.max()) + 1

    def _cmean(a):
        return np.arctan2(np.bincount(gid, np.sin(a), n), np.bincount(gid, np.cos(a), n))

    return _cmean(pass_index * np.pi) / np.pi, _cmean(theta)


def _pass_trace(pos_ts, pos_xy, maps):
    """Field-independent part of the pass index, computed once per cell and cached
    in `maps`: jitter-smoothed positions, running mask, the field index (Climer's
    whole-map field-index map, bilinear) resampled along running arc length, and the
    field number (maps['field_label']) at every resampled sample."""
    if 'trace' not in maps:
        pos_sm, speed = smooth_positions(pos_ts, pos_xy)
        running = speed >= MIN_SPEED_CM_S
        field_index = field_index_per_position(pos_sm, maps['field_index_map'],
                                               maps['x_edges'], maps['y_edges'])
        cc, ts2, resampled = sample_along_arc(pos_ts, pos_sm, field_index, running)
        xy2 = np.column_stack([np.interp(ts2, pos_ts, pos_sm[:, 0]), np.interp(ts2, pos_ts, pos_sm[:, 1])])
        maps['trace'] = dict(pos_sm=pos_sm, running=running, cc=cc, ts2=ts2, resampled=resampled,
                             arc_label=maps['field_label'](xy2))
    return maps['trace']


def _precession_fit(spk_ts, spk_pass_index, spk_theta_phase, rng, slope_bnds,
                    occ_pass_index, occ_theta_phase, occ_dt):
    """Circular-linear fit, fit_R shuffle gate, classification and density map for one
    set of analysed spikes (all spikes in kept passes, or complete traversals only).
    occ_pass_index / occ_theta_phase (rad) are the density-map occupancy's LFP samples,
    each worth occ_dt s. Returns a dict of the fit statistics, keyed as in
    compute_pass_index's result."""
    # Slope/intercept/fit_R from all analysed spikes; the asymptotic z-test (rho, p) on one
    # event per burst at that slope, since burst spikes share phase and pass index and are
    # not independent samples (counting them inflates z by ~sqrt(burst size)).
    rho, p, s, b, fit_R = kempter_lincirc(spk_pass_index, spk_theta_phase, slope_bnds=slope_bnds)
    n_events = len(spk_ts)
    if BURST_COLLAPSE_FOR_ZTEST and np.isfinite(s):
        ev_x, ev_theta = burst_events(spk_ts, spk_pass_index, spk_theta_phase)
        n_events = len(ev_x)
        rho, p, *_ = kempter_lincirc(ev_x, ev_theta, s=s, b=b)
    # s is cycles per unit of pass index; a full pass spans 2 units (-1 to
    # +1, entry to exit), so total phase change over one full pass is
    # 2*s cycles = rad2deg(2*(2*pi*s)) degrees -- previously this used
    # rad2deg(2*pi*s), i.e. degrees per 1 unit of pass index (half a pass),
    # under-reporting the true per-pass slope (and the classification bounds
    # below) by a factor of 2.
    slope_deg_per_pass = np.rad2deg(4 * np.pi * s) if not np.isnan(s) else np.nan

    # Gate: fit_R must beat the burst-preserving phase-permutation null
    # (shuffle_fit_r_significance) before the z-test below is considered.
    fit_r_shuffle_p, shuffle_fit_r = np.nan, None
    fit_r_significant = True
    if USE_FIT_R_GATE and np.isfinite(fit_R):
        fit_r_shuffle_p, shuffle_fit_r = shuffle_fit_r_significance(
            spk_ts, spk_pass_index, spk_theta_phase, fit_R, rng, slope_bnds=slope_bnds)
        fit_r_significant = bool(fit_r_shuffle_p < ALPHA)

    # Significance: fit_R gate passed AND Kempter et al. (2012) asymptotic z-test on
    # the circular-linear correlation (p from kempter_lincirc).
    is_significant_precession = bool(fit_r_significant and np.isfinite(p) and p < ALPHA)

    # Number of wrapped line segments the fitted phase line (2*pi*s*x + b, drawn
    # mod 2*pi as in plot_unit_summary) makes over the spikes' pass-index extent.
    # More than MAX_FIT_LINES is an implausibly steep fit: the cell is marked
    # non-precessing.
    n_fit_lines = np.nan
    fit_too_steep = False
    if len(spk_pass_index) > 0 and np.isfinite(s) and np.isfinite(b):
        ph_ends = 2 * np.pi * s * np.array([np.min(spk_pass_index), np.max(spk_pass_index)]) + b
        n_fit_lines = int(np.floor(ph_ends.max() / (2 * np.pi)) - np.floor(ph_ends.min() / (2 * np.pi)) + 1)
        fit_too_steep = bool(n_fit_lines > MAX_FIT_LINES)
    if fit_too_steep:
        is_significant_precession = False
    # Significant fit (z-test) AND slope beyond +/-SLOPE_THRESH_DEG_PER_PASS:
    # negative -> phase precessing (spike phase advances to earlier phase over
    # the field pass); positive -> phase succeeding (phase moves later).
    # Significant with |slope| within the threshold: phase locked.
    # Non-significant fit: no phase relation.
    is_precessing = bool(is_significant_precession and slope_deg_per_pass < -SLOPE_THRESH_DEG_PER_PASS)
    is_recessing = bool(is_significant_precession and slope_deg_per_pass > SLOPE_THRESH_DEG_PER_PASS)
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
    # spikes, i.e. the fit slope times the observed extent of pass index
    # (not the full -1..1 range, which the cell may not have fully covered).
    pass_index_range = float(np.ptp(spk_pass_index)) if len(spk_pass_index) > 0 else np.nan
    phase_range_deg = (abs(slope_deg_per_pass) * pass_index_range
                        if not (np.isnan(slope_deg_per_pass) or np.isnan(pass_index_range))
                        else np.nan)

    # Density map: spike density over (pass index, LFP phase), occupancy-normalized.
    pi_edges = np.linspace(-1, 1, 41)
    ph_edges = np.linspace(0, 2 * np.pi, 101)
    occ_density, _, _ = np.histogram2d(occ_pass_index, np.mod(occ_theta_phase, 2 * np.pi),
                                        bins=[pi_edges, ph_edges])
    occ_density *= occ_dt
    spk_density, _, _ = np.histogram2d(spk_pass_index, np.mod(spk_theta_phase, 2 * np.pi),
                                        bins=[pi_edges, ph_edges])
    # Theta phase at pass index 0 (field center): the fitted line 2*pi*s*x + b at
    # x = 0, i.e. the intercept b, wrapped to [0, 360) deg. The same line is also
    # evaluated at the other PASS_INDEX_PHASE_POINTS (x = -1 field entry, +1 field exit).
    phase_at_points = {
        f'phase_at_{key}_deg': (float(np.mod(np.rad2deg(2 * np.pi * s * x + b), 360))
                                if np.isfinite(s) and np.isfinite(b) else np.nan)
        for x, key, *_ in PASS_INDEX_PHASE_POINTS}

    with np.errstate(invalid='ignore', divide='ignore'):
        density = spk_density / occ_density
    density[~np.isfinite(density)] = 0.0
    density = gaussian_filter(density, sigma=1.5, truncate=2.0, mode='nearest')

    return {
        'spk_pass_index': spk_pass_index, 'spk_theta_phase': spk_theta_phase,
        'rho': rho, 'p': p, 's': s, 'b': b, 'fit_R': fit_R,
        'fit_R_shuffle_p': fit_r_shuffle_p, 'fit_R_Significant': fit_r_significant,
        'shuffle_fit_r': shuffle_fit_r,
        'slope_deg_per_pass': slope_deg_per_pass,
        'r_squared': r_squared, 'phase_range_deg': phase_range_deg,
        **phase_at_points,
        'is_precessing': is_precessing, 'is_recessing': is_recessing,
        'is_phase_locked': is_phase_locked, 'precession_class': precession_class,
        'n_fit_lines': n_fit_lines, 'fit_too_steep': fit_too_steep,
        'density': density, 'pi_edges': pi_edges, 'ph_edges': ph_edges,
        'n_spikes': len(spk_ts), 'n_events': n_events,
    }


def compute_pass_index(pos_ts, pos_xy, maps, field, phase_spk_ts, lfp_ts, lfp_sig, lfp_fs, rng,
                        method='place', filter_band='auto', lfp_filter_band=(3.0, 7.0),
                        slope_bnds=None, lfp_theta_mask=None, lfp_phase_unwrapped=None):
    """Pass-index phase precession for ONE place field (`field`, an entry of
    maps['fields'] from detect_place_fields). `phase_spk_ts` (ACG theta-positive
    spikes) are the spikes entering the pass-index / theta-phase analysis and
    shuffles; the rate map / fields in `maps` come from all spikes.
    `lfp_theta_mask` (bool per LFP sample), if given, restricts the density-map
    occupancy to theta-positive samples. `lfp_phase_unwrapped` (per LFP sample), if
    given, is the precomputed unwrapped Hilbert theta phase of `lfp_sig` filtered to
    `lfp_filter_band`; otherwise it is computed here."""
    rmap, occupancy = maps['rate_map'], maps['occupancy']
    x_edges, y_edges, fi_map = maps['x_edges'], maps['y_edges'], maps['field_index_map']
    trace = _pass_trace(pos_ts, pos_xy, maps)
    pos_sm, running = trace['pos_sm'], trace['running']
    cc, ts2, resampled, arc_label = trace['cc'], trace['ts2'], trace['resampled'], trace['arc_label']
    k = field['field_number']

    spk_ts = phase_spk_ts
    spk_xy, _ = spk_pos(pos_ts, pos_xy, spk_ts)
    n_spikes_input = len(spk_ts)

    if filter_band == 'auto':
        filter_band = auto_filter_band(method, field['radius_cm'])
    fs_arc = 1.0 / np.mean(np.diff(cc))
    filtered_field_index = bandpass_filter(resampled, filter_band[0], filter_band[1], fs_arc,
                                           order=SPATIAL_FILTER_ORDER, warn_clamp=True)

    # Spatial pass-index phase: Hilbert phase of the filtered field-index trace
    # (0 at a field-index peak, wrapped to [-1, 1) x pi).
    def _wrap_pm_pi(u):
        return np.mod(u + np.pi, 2 * np.pi) - np.pi

    field_phase_unwrapped = np.unwrap(hilbert_phase(filtered_field_index, fs_arc)[0])

    # Passes: one per cycle of the unwrapped phase (-1 -> +1). The Hilbert phase is
    # amplitude-blind, so every small out-of-field wiggle of the filtered trace is also
    # a full "pass" (4-15x more passes than real field entries in this data). A pass
    # belongs to this field only if the sample where its field index peaks lies inside
    # the field and reaches PASS_MIN_FIELD_PEAK_FRAC of the field's peak field index.
    pass_offset = int(np.floor((field_phase_unwrapped.min() + np.pi) / (2 * np.pi)))

    def _pass_id(ph_unwrapped):
        return np.floor((ph_unwrapped + np.pi) / (2 * np.pi)) - pass_offset

    trace_pass_id = _pass_id(field_phase_unwrapped).astype(np.int64)
    n_passes = int(trace_pass_id.max()) + 1
    order = np.lexsort((resampled, trace_pass_id))          # by pass, then field index
    last_of_pass = np.r_[np.flatnonzero(np.diff(trace_pass_id[order])), len(order) - 1]
    peak_idx = order[last_of_pass]                           # argmax of field index per pass
    pass_peak_fi = np.full(n_passes, -np.inf)
    pass_peak_label = np.zeros(n_passes, dtype=int)
    pass_peak_fi[trace_pass_id[peak_idx]] = resampled[peak_idx]
    pass_peak_label[trace_pass_id[peak_idx]] = arc_label[peak_idx]
    good_pass = (pass_peak_label == k) & (pass_peak_fi >= PASS_MIN_FIELD_PEAK_FRAC * field['peak_field_index'])

    def _in_good_pass(ph_unwrapped):
        pid = _pass_id(ph_unwrapped)
        ok = np.isfinite(pid) & (pid >= 0) & (pid < n_passes)
        ok[ok] = good_pass[pid[ok].astype(np.int64)]
        return ok

    # Complete traversals (v25): split the arc-sampled trace into runs of constant pass id
    # (each run is bounded by the phase crossing the -1/+1 boundary, in either direction).
    # A run is a complete traversal if it lies in a kept pass and its pass index starts
    # near -1 (entered from the previous pass) and ends near +1 (left into the next one);
    # it then necessarily crosses 0 (field centre) in between.
    trace_pi = _wrap_pm_pi(field_phase_unwrapped) / np.pi
    run_start = np.r_[0, np.flatnonzero(np.diff(trace_pass_id)) + 1]
    run_stop = np.r_[run_start[1:], len(trace_pass_id)]
    run_pid = trace_pass_id[run_start]
    complete_run = (good_pass[run_pid] & (trace_pi[run_start] <= -1 + COMPLETE_PASS_EDGE_TOL)
                    & (trace_pi[run_stop - 1] >= 1 - COMPLETE_PASS_EDGE_TOL))
    sample_run = np.repeat(np.arange(len(run_start)), run_stop - run_start)

    def _in_complete_pass(t, ph_unwrapped):
        """Times t (with interpolated unwrapped field phase) inside a complete traversal:
        the nearest trace sample's run is complete and has the same pass id."""
        pid = _pass_id(ph_unwrapped)
        r = _interp_nearest_extrap(ts2, sample_run, t)
        return np.isfinite(pid) & complete_run[r] & (pid == run_pid[r])

    spk_field_phase = phase_at(ts2, field_phase_unwrapped, spk_ts)
    spk_pass_index = _wrap_pm_pi(spk_field_phase) / np.pi
    spk_running = _interp_nearest_extrap(pos_ts, running, spk_ts).astype(bool)
    spk_in_pass = _in_good_pass(spk_field_phase)
    spk_in_complete = _in_complete_pass(spk_ts, spk_field_phase)

    if lfp_phase_unwrapped is None:
        filtered_lfp = bandpass_filter(lfp_sig, lfp_filter_band[0], lfp_filter_band[1], lfp_fs)
        lfp_phase_unwrapped = np.unwrap(hilbert_phase(filtered_lfp, lfp_fs)[0])
    lfp_phase = _wrap_pm_pi(lfp_phase_unwrapped)
    spk_theta_phase = _wrap_pm_pi(phase_at(lfp_ts, lfp_phase_unwrapped, spk_ts))

    # Spikes outside the position / LFP time range, or in an LFP gap, have no defined phase: drop them.
    spk_theta_phase[~spikes_with_lfp_sample(spk_ts, lfp_ts, lfp_fs)] = np.nan
    has_phase = np.isfinite(spk_pass_index) & np.isfinite(spk_theta_phase)
    n_spikes_running = int(np.sum(has_phase & spk_running))
    spk_valid = has_phase & spk_running & spk_in_pass
    spk_xy_excluded = spk_xy[~spk_valid]
    spk_complete = spk_in_complete[spk_valid]        # complete-traversal subset of the analysed spikes
    spk_ts, spk_xy = spk_ts[spk_valid], spk_xy[spk_valid]
    spk_pass_index, spk_theta_phase = spk_pass_index[spk_valid], spk_theta_phase[spk_valid]
    if len(spk_ts) < MIN_SPIKES_FOR_FIT:
        raise TooFewSpikes(
            f'field {field["field_id"]}: only {len(spk_ts)} spikes in running (>= {MIN_SPEED_CM_S:g} cm/s) passes '
            f'through the field ({int(good_pass.sum())} passes) < MIN_SPIKES_FOR_FIT={MIN_SPIKES_FOR_FIT}')

    # Density-map occupancy uses the same sample selection as the spikes: LFP samples inside
    # the tracking span, while running, inside a kept pass (and theta-positive, if masked);
    # for the complete-traversal map, additionally inside a complete traversal.
    # The unwrapped field phase is interpolated, then wrapped (interpolating the wrapped
    # trace smeared values across the +1/-1 jump).
    lfp_field_phase = phase_at(ts2, field_phase_unwrapped, lfp_ts)
    lfp_pass_index = _wrap_pm_pi(lfp_field_phase) / np.pi
    dt_lfp = float(np.mean(np.diff(lfp_ts)))
    occ_sel = (np.isfinite(lfp_pass_index) & np.isfinite(lfp_phase) & _in_good_pass(lfp_field_phase)
               & _interp_nearest_extrap(pos_ts, running, lfp_ts).astype(bool))
    if lfp_theta_mask is not None:
        occ_sel &= lfp_theta_mask
    occ_complete = occ_sel & _in_complete_pass(lfp_ts, lfp_field_phase)

    # The complete-traversal fit_R shuffle draws from a copy of rng, so the all-spike
    # results (and every later unit's shuffles) stay identical to v24.
    rng_complete = copy.deepcopy(rng)
    fit = _precession_fit(spk_ts, spk_pass_index, spk_theta_phase, rng, slope_bnds,
                          lfp_pass_index[occ_sel], lfp_phase[occ_sel], dt_lfp)

    # Same analysis on complete traversals only (pass index -1 -> 0 -> +1).
    n_passes_complete = int(complete_run.sum())
    n_complete = int(spk_complete.sum())
    complete, complete_skip_reason = None, None
    if n_complete >= MIN_SPIKES_FOR_FIT:
        complete = _precession_fit(spk_ts[spk_complete], spk_pass_index[spk_complete],
                                   spk_theta_phase[spk_complete], rng_complete, slope_bnds,
                                   lfp_pass_index[occ_complete], lfp_phase[occ_complete], dt_lfp)
        complete['spk_xy'] = spk_xy[spk_complete]
    else:
        complete_skip_reason = (f'only {n_complete} spikes in {n_passes_complete} complete traversals '
                                f'< MIN_SPIKES_FOR_FIT={MIN_SPIKES_FOR_FIT}')

    return {
        'rate_map': rmap, 'occupancy': occupancy, 'x_edges': x_edges, 'y_edges': y_edges,
        'field_index_map': fi_map, 'field': field, 'fields': maps['fields'],
        'dropped_subfields': maps['dropped_subfields'], 'n_detected_fields': maps['n_detected_fields'],
        'field_detection': maps['detection'],
        'spk_xy': spk_xy,
        **fit,
        'n_spikes_input': n_spikes_input, 'n_spikes_running': n_spikes_running,
        'n_passes_total': n_passes, 'n_passes_kept': int(good_pass.sum()),
        'n_passes_complete': n_passes_complete, 'n_spikes_complete': n_complete,
        'complete': complete, 'complete_skip_reason': complete_skip_reason,
        'filter_band': tuple(float(f) for f in filter_band),
        'spk_xy_excluded': spk_xy_excluded, 'pos_sm': pos_sm,
    }


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


def mardia_watson_wheeler_test(groups_rad, rng, n_perm=N_PI0_PERMUTATIONS):
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
    """'precession' if the phase (deg, any cycle) falls in PI0_PRECESSION_RANGE_DEG
    on the circle, else 'procession'; None for NaN."""
    if not np.isfinite(phase_deg):
        return None
    lo, hi = PI0_PRECESSION_RANGE_DEG
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
                        TMI_Significant=False)
        return metrics, phase_deg, None

    mrl = circ_r(phase_rad)
    pref_phase_deg = np.degrees(circ_mean(phase_rad)) % 360
    rayleigh_p, _ = circ_rtest(phase_rad)

    peak_phase_deg, valley_phase_deg, tmi = find_phase_peak_valley(phase_deg)
    tmi_pval, shuffle_tmis = shuffle_tmi_significance(spk_ts, phase_deg, tmi, rng)
    tmi_sig = bool(np.isfinite(tmi_pval) and tmi_pval < ALPHA)

    metrics.update(MRL=mrl, PreferredPhase_deg=pref_phase_deg, Rayleigh_p=rayleigh_p,
                    SignificantThetaModulation=bool(rayleigh_p < ALPHA),
                    PhasePeak_deg=peak_phase_deg, PhaseValley_deg=valley_phase_deg,
                    TMI=tmi, TMI_shuffle_p=tmi_pval, TMI_Significant=tmi_sig)
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


def _draw_pass_index_phase(ax, results, label=None):
    """Draw LFP phase (duplicated over 0-720 deg) vs pass index for each spike,
    with the fitted circular-linear line (Step 3). `label`, if given, is put on
    its own line above the fit statistics in the title."""
    # background gradient: theta peak (0/360/720 deg) -> trough (180/540 deg), light magma
    ph_bg = np.linspace(0, 720, 721)
    peakness = 0.5 *(1 + np.cos(np.deg2rad(ph_bg)))  # 1 at peak, 0 at trough
    mag = plt.get_cmap('magma')(0.35 + 0.65 * peakness)[:, :3]  # skip the darkest blacks
    bg_alpha = 0.18  # magma opacity: lower = lighter/more transparent
    bg = bg_alpha * mag + (1 - bg_alpha) * 1.0  # blend with white for a lighter hue
    ax.imshow(bg[:, None, :], origin='lower', aspect='auto', extent=[-1, 1, 0, 720],
              zorder=0, interpolation='bilinear')
    pi_dup =np.concatenate([results['spk_pass_index'], results['spk_pass_index']])
    phase_dup = np.concatenate([np.rad2deg(np.mod(results['spk_theta_phase'], 2 * np.pi)),
                                 np.rad2deg(np.mod(results['spk_theta_phase'], 2 * np.pi)) + 360])
    ax.scatter(pi_dup, phase_dup, s=8, alpha=0.6, color='k', zorder=2)
    s, b = results['s'], results['b']
    if not np.isnan(s):
        xg = np.linspace(-1, 1, 500)
        phi = np.mod(2 * np.pi * s * xg + b, 2 * np.pi)
        phi[np.abs(np.diff(phi, prepend=phi[0])) > np.pi] = np.nan
        ax.plot(xg, np.rad2deg(phi), 'r', linewidth=2)
        ax.plot(xg, np.rad2deg(phi) + 360, 'r', linewidth=2)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 720)
    ax.set_xlabel('Pass index')
    ax.set_ylabel('LFP phase (deg)')
    ax.set_title((f'{label}\n' if label else '') +
                 f'rho={results["rho"]:.2f}  p={results["p"]:.3g}  '
                 f'slope={results["slope_deg_per_pass"]:.1f} deg/pass  '
                 f'class={results["precession_class"]}', fontsize=9)


def _draw_density(fig, ax, results, title):
    """Occupancy-normalized spike density over (pass index, LFP phase), the phase
    axis duplicated over 0-720 deg."""
    ph_centers = np.rad2deg(0.5 * (results['ph_edges'][:-1] + results['ph_edges'][1:]))
    pi_centers = 0.5 * (results['pi_edges'][:-1] + results['pi_edges'][1:])
    dens = results['density']
    dens_dup = np.concatenate([dens, dens], axis=1)
    ph_dup = np.concatenate([ph_centers, ph_centers + 360])
    im = ax.pcolormesh(pi_centers, ph_dup, dens_dup.T, cmap='jet', shading='auto')
    fig.colorbar(im, ax=ax, label='Rate (Hz)')
    ax.set_xlabel('Pass index')
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
        f"slope ={results['slope_deg_per_pass']:.2f} deg/pass;  phase range = {results['phase_range_deg']:.1f} deg",
        *(f"phase @ pass index {x:g} ={results[f'phase_at_{key}_deg']:.1f} deg "
          f"({phase_range_label(results[f'phase_at_{key}_deg'])} range)"
          for x, key, *_ in PASS_INDEX_PHASE_POINTS),
        f"class = {results['precession_class']}",
    ]


def _draw_text_panel(ax, text_lines):
    ax.axis('off')
    for i, line in enumerate(text_lines):
        is_bold = line.startswith(('class = ', 'FIELD ', 'COMPLETE '))
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
    ax_s.set_title(f'fit R shuffle null ({len(shuffle_fit_r)} burst-preserving phase permutations)',
                   fontsize=9)
    ax_s.legend(fontsize=8, loc='upper left')


def plot_fit_r_shuffle(results, title, out_path: Path):
    """Step 3 gate (visualization): left, LFP phase vs pass index with the
    fitted line; right, the fit_R shuffle-test null distribution with the
    observed fit_R and the null's (1 - ALPHA) percentile. The cell's z-test is
    considered only if p < ALPHA."""
    is_sig = results['fit_R_Significant']
    color = 'red' if is_sig else '#4D4D4D'
    verdict = 'PASS' if is_sig else 'FAIL'
    fig, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f'{title}\nfit R={results["fit_R"]:.3f} | shuffle p={results["fit_R_shuffle_p"]:.3g} '
                 f'| {verdict}', fontsize=10, color=color)
    _draw_pass_index_phase(ax_p, results)
    _draw_fit_r_null(ax_s, results['fit_R'], is_sig, results['shuffle_fit_r'])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_unit_summary(pos_xy, results, title, out_path: Path, theta=None):
    """Step 3: 9-panel Pass Index phase-precession summary figure. Rows: trajectory /
    rate map / field-index map; phase vs pass index, density map and fit statistics for
    all spikes in the field's kept passes; the same three panels for complete traversals
    only (pass index -1 -> 0 -> +1; results['complete']).

    theta (optional): dict(phase_deg=..., metrics=..., shuffle_tmis=...) from
    compute_theta_modulation. If given, a top row is added with Steps 1-2's
    polar plot, TMI histogram and TMI shuffle null (same details as the
    standalone PolarPlot / TMIShuffle figures)."""
    if theta is None:
        fig, axes = plt.subplots(3, 3, figsize=(16, 15))
    else:
        fig = plt.figure(figsize=(16, 20.5))
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1])
        m = theta['metrics']
        _draw_polar_theta(fig.add_subplot(gs[0, 0], projection='polar'), theta['phase_deg'],
                           m['MRL'], m['PreferredPhase_deg'], m['Rayleigh_p'],
                           m['SignificantThetaModulation'])
        ax_h = fig.add_subplot(gs[0, 1])
        _draw_tmi_histogram(ax_h, theta['phase_deg'], m['TMI'], m['TMI_Significant'])
        color, verdict_line = _tmi_verdict(m['TMI'], m['TMI_shuffle_p'], m['TMI_Significant'])
        ax_h.set_title(f'TMI histogram (n={len(theta["phase_deg"])} spikes)\n{verdict_line}',
                       fontsize=9, color=color)
        _draw_tmi_null(fig.add_subplot(gs[0, 2]), m['TMI'], m['TMI_Significant'], theta['shuffle_tmis'])
        axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in (1, 2, 3)])
    fig.suptitle(title, fontsize=12, fontweight='bold')

    ax = axes[0, 0]
    pos_plot = results.get('pos_sm', pos_xy)
    ax.plot(pos_plot[:, 0], pos_plot[:, 1], color='0.8', linewidth=0.5, zorder=1)
    if len(results.get('spk_xy_excluded', [])):
        ax.scatter(results['spk_xy_excluded'][:, 0], results['spk_xy_excluded'][:, 1],
                   color='0.55', s=4, alpha=0.5, zorder=2, label='excluded (slow / outside kept passes)')
        ax.legend(fontsize=7, loc='lower left')
    sc = ax.scatter(results['spk_xy'][:, 0], results['spk_xy'][:, 1],
                     c=results['spk_pass_index'], cmap='hsv', s=10, vmin=-1, vmax=1, zorder=3)
    fig.colorbar(sc, ax=ax, label='Pass index')
    ax.set_title('Trajectory + analysed spikes (color = pass index)', fontsize=10)
    ax.set_aspect('equal')

    xe, ye = results['x_edges'], results['y_edges']
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    this_k = results['field']['field_number']

    def _outline_fields(ax, color):
        """This field / sub-field solid, the cell's other analysed ones dashed, sub-fields
        too small to analyse dotted; each labelled F<field id>."""
        shown = [(f, f['field_number'] == this_k) for f in results['fields']]
        shown += [(f, None) for f in results.get('dropped_subfields', [])]
        for f, this in shown:
            if not f['mask_2d'].any() or min(f['mask_2d'].shape) < 2:
                continue
            ax.contour(xc, yc, f['mask_2d'].T.astype(float), levels=[0.5], colors=color,
                       linewidths=2.0 if this else 1.0,
                       linestyles='solid' if this else 'dashed' if this is False else 'dotted')
            ax.text(f['com_x_cm'], f['com_y_cm'], f"F{f['field_id']}", color=color, ha='center',
                    va='center', fontsize=9, fontweight='bold' if this else 'normal')

    ax = axes[0, 1]
    im = ax.imshow(results['rate_map'].T, origin='lower', cmap='jet', extent=[xe[0], xe[-1], ye[0], ye[-1]])
    _outline_fields(ax, 'white')
    fig.colorbar(im, ax=ax, label='Rate (Hz)')
    ax.set_title(f'Rate map (fields: {results["field_detection"]})', fontsize=10)
    ax.set_aspect('equal')

    ax = axes[0, 2]
    im = ax.imshow(results['field_index_map'].T, origin='lower', cmap='hot', vmin=0, vmax=1,
                    extent=[xe[0], xe[-1], ye[0], ye[-1]])
    _outline_fields(ax, 'cyan')
    fig.colorbar(im, ax=ax, label='Field index')
    ax.set_title('Field index map')
    ax.set_aspect('equal')

    _draw_pass_index_phase(axes[1, 0], results, label='All spikes in kept passes')
    _draw_density(fig, axes[1, 1], results, 'Density map (all spikes in kept passes)')

    fld = results['field']
    _draw_text_panel(axes[1, 2], [
        f"FIELD {fld['field_id']}: peak {fld['peak_rate_Hz']:.2f} Hz, "
        f"{fld['area_cm2']:.0f} cm^2 ({fld['pct_of_occupied_area']:.0f}% of arena), r = {fld['radius_cm']:.1f} cm",
        (f"sub-field {fld['sub_field']} of {fld['n_subfields']} of multi-peak field {fld['parent_field']} "
         f"(KDE peak = {fld['kde_peak_rel']:.0%} of field max)" if fld['multi_peak']
         else f"single-peak field ({fld['kde_note'] or 'not split'})"),
        f"spikes: {results['n_spikes_input']} theta -> {results['n_spikes_running']} running "
        f"-> {results['n_spikes']} in this field's passes",
        f"passes kept = {results['n_passes_kept']} / {results['n_passes_total']};  "
        f"burst events = {results['n_events']}",
        *_fit_text_lines(results),
    ])

    # Complete traversals only (pass index -1 -> 0 -> +1)
    comp = results.get('complete')
    header = [
        'COMPLETE TRAVERSALS (pass index -1 -> 0 -> +1)',
        f"complete traversals = {results['n_passes_complete']} of {results['n_passes_kept']} kept passes",
        f"spikes: {results['n_spikes_complete']} of {results['n_spikes']} in this field's passes"
        + (f";  burst events = {comp['n_events']}" if comp is not None else ''),
    ]
    if comp is not None:
        _draw_pass_index_phase(axes[2, 0], comp, label='Complete traversals only (-1 -> 0 -> +1)')
        _draw_density(fig, axes[2, 1], comp, 'Density map (complete traversals only)')
        _draw_text_panel(axes[2, 2], header + _fit_text_lines(comp))
    else:
        for ax in axes[2, :2]:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Complete-traversal analysis skipped', ha='center', va='center', fontsize=10)
        _draw_text_panel(axes[2, 2], header + ['skipped:', *textwrap.wrap(results['complete_skip_reason'], 60)])

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


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
    'abs_slope_deg_per_pass': 'Slope magnitude (deg/pass)',
    'phase_range_deg': 'Phase range (deg)',
}
TMI_COMPARISON_METRICS = {
    'TMI': 'Theta Modulation Index (TMI)',
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
# Arena and session-type labels don't overlap, so one lookup serves both comparisons.
GROUP_COLORS = {**ARENA_COLORS, **SESSION_COLORS}
_DEFAULT_COLOR = {'face': PAL_GRAY, 'edge': PAL_BLACK}
# Grouping column -> names of the omnibus tables' group-count / n-per-group columns
GROUP_COUNT_COLUMNS = {'Arena': ('N_arenas', 'N_per_arena'),
                       'SessionType': ('N_session_types', 'N_per_session_type')}
_TEST_STAT_NAMES = {'Kruskal-Wallis': 'H', 'Mann-Whitney U': 'U',
                    'Mardia-Watson-Wheeler': 'W', 'Watson-Williams': 'F'}


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


def compare_arenas_pi0_phase(df_pi0: pd.DataFrame, rng, col: str = 'phase_at_pi0_deg',
                             label: str = 'field center (pass index 0)',
                             group_col: str = 'Arena', group_order=ARENA_LABELS):
    """Theta phase at pass index 0 (or at the pass index whose phase is in `col`,
    described by `label`) across arenas (or, with group_col='SessionType',
    across session types), with circular statistics
    (a linear Kruskal-Wallis would treat 359 and 1 deg as far apart).
    Primary: Mardia-Watson-Wheeler uniform-scores test (nonparametric, tests
    for any difference in the circular distributions), permutation p. Also
    reported: Watson-Williams test (equal mean direction, von Mises
    assumption). Omnibus over all groups, then pairwise with Holm correction
    per test. Returns (descriptive_df, omnibus_df, pairwise_df)."""
    arenas_present = [a for a in group_order if (df_pi0[group_col] == a).sum() > 0]
    groups = {a: np.deg2rad(df_pi0.loc[df_pi0[group_col] == a, col].dropna().to_numpy())
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
def plot_pi0_phase_by_arena(df_pi0: pd.DataFrame, desc_df: pd.DataFrame, omnibus_df: pd.DataFrame,
                             pairwise_df: pd.DataFrame, out_path: Path, col: str = 'phase_at_pi0_deg',
                             label: str = 'field center (pass index 0)', group_col: str = 'Arena',
                             group_order=ARENA_LABELS, title_prefix: str = '',
                             pooled_within: str = 'arena'):
    """Box plot + jittered points per arena (or per session type, with
    group_col='SessionType') of the theta phase at field center
    (or at the pass index whose phase is in `col`, described by `label`),
    on a single-cycle 0-360 deg axis, in the same style as plot_arena_comparison.
    Points are purple in the precession range, yellow in the procession range
    (faint matching background bands). Because phase is circular, each arena's
    box is computed on its phases unwrapped to within +/-180 deg of the arena's
    circular mean (so a cluster straddling 0/360 deg is not split), then drawn
    at -360/0/+360 deg offsets clipped to 0-360 so such a box wraps around.
    Stars: pairs whose Holm-corrected Mardia-Watson-Wheeler p < ALPHA."""
    present = [a for a in group_order if df_pi0.loc[df_pi0[group_col] == a, col].notna().any()]
    if not present:
        return
    phases = [np.mod(df_pi0.loc[df_pi0[group_col] == a, col].dropna().to_numpy(dtype=float), 360)
              for a in present]
    positions = np.arange(1, len(present) + 1)
    lo, hi = PI0_PRECESSION_RANGE_DEG

    stats_lines = [f'Place fields (one value per field): {", ".join(PI0_PHASE_CLASSES)}; '
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

    ax.axhspan(lo, hi, color=PI0_RANGE_COLORS['precession'], alpha=0.08, lw=0, zorder=0)
    for y0, y1 in ((0, lo), (hi, 360)):
        ax.axhspan(y0, y1, color=PI0_RANGE_COLORS['procession'], alpha=0.12, lw=0, zorder=0)

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
        faces = [PI0_RANGE_COLORS[phase_range_label(p)] for p in ph]
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
                      markerfacecolor=PI0_RANGE_COLORS['precession'], markeredgecolor=PAL_BLACK,
                      markeredgewidth=0.5,
                      label=f'precession range ({lo:g}–{hi:g}°)'),
               Line2D([0], [0], marker='o', linestyle='None', markersize=8,
                      markerfacecolor=PI0_RANGE_COLORS['procession'], markeredgecolor=PAL_BLACK,
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


def run_group_comparisons(df: pd.DataFrame, group_col: str, group_order, out_dir: Path,
                          file_prefix: str, group_word: str, title_prefix: str = '') -> dict[str, pd.DataFrame]:
    """Every between-group comparison for the cells in `df`, grouped by
    df[group_col] in `group_order` order (the arenas, or one arena's session
    types), animals pooled within each group:
      - r^2, fit_R, |slope| and phase range of significantly precessing/recessing
        cells (Kruskal-Wallis + Mann-Whitney U), plotted to
        out_dir/<file_prefix>_PhasePrecession_<metric>.png
      - TMI of significantly theta-modulated cells, out_dir/<file_prefix>_TMI.png
      - theta phase at each PASS_INDEX_PHASE_POINTS pass index of the
        PI0_PHASE_CLASSES cells (circular tests),
        out_dir/<file_prefix>_PhaseAt<file tag>.png
    A comparison with fewer than 2 groups holding data is skipped. Returns
    {table key: DataFrame} with keys 'Omnibus', 'Pairwise', 'TMI_Omnibus',
    'TMI_Pairwise' and 'PhaseAt<sheet tag>_By' / '_Omnibus' / '_Pairwise'
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
    df_sig['abs_slope_deg_per_pass'] = df_sig['slope_deg_per_pass'].abs()
    present = groups_with_data(df_sig)
    if len(present) >= 2:
        omnibus_df, pairwise_df = compare_arenas(df_sig, group_col=group_col, group_order=group_order)
        tables['Omnibus'], tables['Pairwise'] = omnibus_df, pairwise_df
        plot_paths = plot_arena_comparison(
            df_sig, omnibus_df, pairwise_df, out_dir / f'{file_prefix}_PhasePrecession',
            population=f'significantly precessing/recessing place fields (one value per field); {pooled}',
            group_col=group_col, group_order=group_order, title_prefix=title_prefix)
        print(f'Phase-precession comparison ({", ".join(map(_label, present))}) plots saved: '
              f'{", ".join(p.name for p in plot_paths)}')
    else:
        skip_msg(present, 'significantly precessing/recessing cells', 'phase-precession')

    # ---- TMI (significantly theta-modulated cells; one row per cell, not per field) ----
    df_tmi_sig = df.loc[df['TMI_Significant'] == True].drop_duplicates(['FolderPath', 'Unit']).copy()  # noqa: E712
    present = groups_with_data(df_tmi_sig)
    if len(present) >= 2:
        omnibus_df, pairwise_df = compare_arenas(df_tmi_sig, TMI_COMPARISON_METRICS,
                                                 group_col=group_col, group_order=group_order)
        tables['TMI_Omnibus'], tables['TMI_Pairwise'] = omnibus_df, pairwise_df
        plot_paths = plot_arena_comparison(
            df_tmi_sig, omnibus_df, pairwise_df, out_dir / f'{file_prefix}_TMI', TMI_COMPARISON_METRICS,
            population=f'significantly theta-modulated cells (TMI shuffle); {pooled}',
            group_col=group_col, group_order=group_order, title_prefix=title_prefix)
        print(f'TMI comparison ({", ".join(map(_label, present))}) plot saved: '
              f'{", ".join(p.name for p in plot_paths)}')
    else:
        skip_msg(present, 'significantly theta-modulated cells', 'TMI')

    # ---- Theta phase at pass index -1 / 0 (field center) / +1 ----
    for x, key, label, file_tag, sheet_tag in PASS_INDEX_PHASE_POINTS:
        col = f'phase_at_{key}_deg'
        df_pt = df.loc[df['PrecessionClass'].isin(PI0_PHASE_CLASSES) & df[col].notna()].copy()
        present = groups_with_data(df_pt)
        if len(present) >= 2:
            desc_df, omnibus_df, pairwise_df = compare_arenas_pi0_phase(
                df_pt, np.random.default_rng(RANDOM_SEED), col=col, label=label,
                group_col=group_col, group_order=group_order)
            tables[f'PhaseAt{sheet_tag}_By'] = desc_df
            tables[f'PhaseAt{sheet_tag}_Omnibus'] = omnibus_df
            tables[f'PhaseAt{sheet_tag}_Pairwise'] = pairwise_df
            pt_plot_path = out_dir / f'{file_prefix}_PhaseAt{file_tag}.png'
            plot_pi0_phase_by_arena(df_pt, desc_df, omnibus_df, pairwise_df, pt_plot_path,
                                    col=col, label=label, group_col=group_col, group_order=group_order,
                                    title_prefix=title_prefix, pooled_within=group_word)
            print(f'Phase-at-pass-index-{x:g} comparison ({", ".join(map(_label, present))}) '
                  f'plot saved: {pt_plot_path.name}')
        else:
            skip_msg(present, f'{"/".join(PI0_PHASE_CLASSES)} cells', f'phase-at-pass-index-{x:g}')

    return {k: t for k, t in tables.items() if not t.empty}


def _comparison_sheet_name(group_col: str, key: str) -> str:
    """Excel sheet name (<= 31 characters) for a run_group_comparisons table.
    The arena comparison keeps v18's sheet names ('ArenaComparison_Omnibus',
    'ArenaComparison_TMI_Pairwise', 'PhaseAtPI0_ByArena', ...); session-type
    tables get a 'SesType_' prefix ('SesType_Omnibus', 'SesType_TMI_Pairwise',
    'SesType_PhaseAtPI0_BySesType', ...)."""
    if key.endswith('_By'):
        key += 'Arena' if group_col == 'Arena' else 'SesType'
    if group_col != 'Arena':
        return f'SesType_{key}'
    return key if key.startswith('PhaseAt') else f'ArenaComparison_{key}'


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


def process_session(data_folder: Path, rng, edmin_records: list | None = None) -> list[dict]:
    """Run Steps 1-3 on every unit in one session; returns one row per analysed place
    field (a multi-field unit gives one row per field, its Steps 1-2 columns repeated),
    or one row for a unit with no field analysed.
    If `edmin_records` is given (and USE_ACG_THETA_EPOCHS), one dict per ACG
    epoch of each LFP file used (Animal, Session, lfp_file, epoch_index, ED_min,
    peakrangenorm, Status) is appended to it, for the pooled ED_min QC plot."""
    output_dir = data_folder / 'ThetaMod_PhasePrecession_v26'
    output_dir.mkdir(parents=True, exist_ok=True)

    ncs_files = sorted(data_folder.glob('*.ncs'), key=_natural_key)

    # Tracking is only needed for Step 3; a missing tracking file should not
    # block Steps 1-2 for this session.
    pos_ts = pos_xy = None
    try:
        tracking_path = _find_tracking_file(data_folder)
        print(f'Using tracking file: {tracking_path.name}')
        pos_ts, pos_xy = load_tracking(tracking_path, TRACKING_TIME_UNIT)
    except FileNotFoundError as exc:
        print(f'  {exc} -- Step 3 (phase precession) will be skipped for this session.')

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

            t_start = t_stop = None
            if pos_ts is not None:
                t_start = max(pos_ts.min(), lfp_ts.min())
                t_stop = min(pos_ts.max(), lfp_ts.max())

            acg_epoch_ok = None
            lfp_theta_mask = None
            if USE_ACG_THETA_EPOCHS:
                acg_epoch_ok, acg_ed_min, acg_peakrange = acg_theta_epoch_mask(
                    lfp_sig, return_peakrange=True)
                lfp_theta_mask = times_in_ok_epochs(lfp_ts, lfp_ts[0], acg_epoch_ok)
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

            lfp_cache[theta_ncs] = dict(lfp_sig=lfp_sig, lfp_ts=lfp_ts, lfp_fs=lfp_fs,
                                         lfp_phase_unwrapped=lfp_phase_unwrapped,
                                         t_start=t_start, t_stop=t_stop,
                                         acg_epoch_ok=acg_epoch_ok, lfp_theta_mask=lfp_theta_mask)

        lfp_data = lfp_cache[theta_ncs]
        lfp_sig, lfp_ts, lfp_fs = lfp_data['lfp_sig'], lfp_data['lfp_ts'], lfp_data['lfp_fs']
        lfp_phase_unwrapped = lfp_data['lfp_phase_unwrapped']
        t_start, t_stop = lfp_data['t_start'], lfp_data['t_stop']
        acg_epoch_ok, lfp_theta_mask = lfp_data['acg_epoch_ok'], lfp_data['lfp_theta_mask']

        units = load_ntt_spike_times(ntt_path)
        for cell_number, spk_ts in units.items():
            unit_label = f'{ntt_path.stem}_cell{cell_number}' if len(units) > 1 else ntt_path.stem
            plot_label = f'{animal_id} | {unit_label}'
            file_prefix = f'{animal_id}_{unit_label}'
            row: dict[str, Any] = dict(Animal=animal_id, Session=session_label, FolderPath=str(data_folder), Unit=unit_label,
                       ntt_file=ntt_path.name, lfp_file=theta_ncs.name, cell_number=cell_number,
                       n_spikes_total=len(spk_ts))

            # Remove this unit's plots from any earlier run, so a cell that no longer
            # passes a step (e.g. now fails the TMI shuffle test) leaves no stale plot.
            # Per-field plots are <prefix>_F<k>_<suffix>.png; the unnumbered names are older runs'.
            for suffix in ('PolarPlot', 'PhaseHistogram', 'TMIShuffle', 'PassIndex', 'FitRShuffle'):
                (output_dir / f'{file_prefix}_{suffix}.png').unlink(missing_ok=True)
            for suffix in ('PassIndex', 'FitRShuffle'):
                for stale in output_dir.glob(f'{file_prefix}_F[0-9]*_{suffix}.png'):
                    stale.unlink(missing_ok=True)
            stale_dirs = [RESULTS_FOLDER / MULTILINES_FOLDER,
                          *(RESULTS_FOLDER / d / sub for d in set(CLASS_PLOT_FOLDERS.values())
                            for sub in ('', *FIT_R_SUBFOLDERS.values()))]
            class_prefix = f'{animal_id}_{session_label}_{unit_label}'
            for stale_dir in stale_dirs:
                for suffix in ('PassIndex', 'FitRShuffle'):
                    (stale_dir / f'{class_prefix}_{suffix}.png').unlink(missing_ok=True)
                    for stale in stale_dir.glob(f'{class_prefix}_F[0-9]*_{suffix}.png'):
                        stale.unlink(missing_ok=True)

            # ---- ACG theta-positive spike selection (Steps 1-3 use only these spikes) ----
            spk_ts_all = spk_ts
            if USE_ACG_THETA_EPOCHS:
                spk_ts = spk_ts_all[times_in_ok_epochs(spk_ts_all, lfp_ts[0], acg_epoch_ok)]
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
                      f'(p={metrics["TMI_shuffle_p"]:.3g}, '
                      f'{"theta-modulated" if metrics["TMI_Significant"] else "not theta-modulated"})')
            else:
                print(f'  {unit_label}: only {metrics["n_spikes_theta"]} spikes with LFP coverage '
                      f'(< MIN_SPIKES_FOR_TMI={MIN_SPIKES_FOR_TMI}), skipping polar plot / TMI test')

            # ---- Step 3: phase precession, gated on Step 2's TMI significance ----
            # One output row per detected place field (Field 1 = highest peak rate); a unit
            # with no field analysed (TMI fail, no tracking, no field, ...) keeps one row.
            row['PrecessionTested'] = False
            row['PrecessionSkippedReason'] = ''
            field_rows: list[dict] = []
            if not metrics['TMI_Significant']:
                row['PrecessionSkippedReason'] = 'not significantly theta-modulated (TMI shuffle test)'
            elif pos_ts is None:
                row['PrecessionSkippedReason'] = 'no tracking file found for this session'
            else:
                # spk_ts is already ACG-theta-filtered (if enabled): these are the
                # spikes used for phase; all spikes in the window define the field.
                spk_ts_overlap = spk_ts[(spk_ts >= t_start) & (spk_ts <= t_stop)]
                spk_ts_all_overlap = spk_ts_all[(spk_ts_all >= t_start) & (spk_ts_all <= t_stop)]
                if len(spk_ts_overlap) < MIN_SPIKES_FOR_FIT:
                    row['PrecessionSkippedReason'] = (
                        f'only {len(spk_ts_overlap)} spikes in tracking/LFP overlap window '
                        f'(< MIN_SPIKES_FOR_FIT={MIN_SPIKES_FOR_FIT})')
                else:
                    try:
                        maps = detect_place_fields(pos_ts, pos_xy, spk_ts_all_overlap, detect_arena(data_folder),
                                                   BINSIDE, SMTH_WIDTH, METHOD)
                    except Exception as exc:
                        maps = None
                        row['PrecessionSkippedReason'] = f'ERROR in place-field detection ({exc})'
                        print(f'  {unit_label}: place-field detection ERROR ({exc})')
                    if maps is not None and maps['n_detected_fields'] == 0:
                        row['n_fields'] = 0
                        row['FieldDetection'] = maps['detection']
                        row['PrecessionSkippedReason'] = 'no place field detected (threshold method)'
                        print(f'  {unit_label}: no place field detected -- precession skipped')
                    elif maps is not None:
                        n_fields = maps['n_detected_fields']
                        dropped = maps['dropped_subfields']
                        n_multi = len({f['parent_field'] for f in maps['fields'] + dropped if f['multi_peak']})
                        print(f'  {unit_label}: {n_fields} place field(s) ({maps["detection"]}), '
                              f'{n_multi} multi-peak -> {len(maps["fields"])} field(s)/sub-field(s) analysed'
                              + (f', {len(dropped)} sub-field(s) < {MIN_SUBFIELD_SIZE_BINS} bins dropped'
                                 if dropped else ''))
                        for field in maps['fields']:
                            row_f = _process_field(
                                row, field, n_fields, maps, pos_ts, pos_xy, spk_ts_overlap,
                                lfp_ts, lfp_sig, lfp_fs, rng, lfp_theta_mask, lfp_phase_unwrapped,
                                output_dir, file_prefix, plot_label, class_prefix, unit_label,
                                dict(phase_deg=phase_deg, metrics=metrics, shuffle_tmis=shuffle_tmis))
                            field_rows.append(row_f)
                        # Sub-fields too small for phase precession: listed, not analysed
                        for sub in dropped:
                            field_rows.append({
                                **row, **_field_columns(sub, n_fields, maps['detection']),
                                'PrecessionSkippedReason': (
                                    f'sub-field of {sub["n_bins"]} bins < MIN_SUBFIELD_SIZE_BINS='
                                    f'{MIN_SUBFIELD_SIZE_BINS}: not analysed')})
                        field_rows.sort(key=lambda r: (r['ParentField'], np.nan_to_num(r['SubField'])))

            rows.extend(field_rows if field_rows else [row])

    return rows


def _field_columns(field, n_fields, detection):
    """Excel columns describing one place field or sub-field (one row per field / sub-field)."""
    return {
        'Field': field['field_id'], 'ParentField': field['parent_field'], 'SubField': field['sub_field'],
        'n_fields': n_fields, 'MultiField': n_fields > 1,
        'MultiPeakField': field['multi_peak'], 'n_subfields': field['n_subfields'],
        'field_kde_peak_Hz': field['kde_peak_Hz'], 'field_kde_peak_rel': field['kde_peak_rel'],
        'field_kde_note': field['kde_note'],
        'FieldDetection': detection,
        'field_peak_rate_Hz': field['peak_rate_Hz'], 'field_peak_rate_raw_Hz': field['peak_rate_raw_Hz'],
        'field_n_bins': field['n_bins'], 'field_area_cm2': field['area_cm2'],
        'field_pct_of_occupied_area': field['pct_of_occupied_area'], 'field_radius_cm': field['radius_cm'],
        'field_com_x_cm': field['com_x_cm'], 'field_com_y_cm': field['com_y_cm'],
        'field_com_theta_deg': field['com_theta_deg'], 'field_arc_length_cm': field['arc_length_cm'],
        'field_track_width_cm': field['track_width_cm'],
    }


def _process_field(row, field, n_fields, maps, pos_ts, pos_xy, spk_ts_overlap,
                   lfp_ts, lfp_sig, lfp_fs, rng, lfp_theta_mask, lfp_phase_unwrapped,
                   output_dir, file_prefix, plot_label, class_prefix, unit_label, theta):
    """Step 3 for one place field (or sub-field) of one unit: pass index + phase-precession
    fit, figures (<prefix>_F<id>_PassIndex / _FitRShuffle, plus the class-folder copy), and
    the Excel row (the unit's Steps 1-2 columns + this field's columns + its fit)."""
    k = field['field_id']
    row_f = {**row, **_field_columns(field, n_fields, maps['detection'])}
    if field['multi_peak']:
        field_label = (f'{plot_label} | Sub-field {k} (sub-field {field["sub_field"]} of '
                       f'{field["n_subfields"]} of field {field["parent_field"]} of {n_fields})')
    else:
        field_label = f'{plot_label} | Field {k} of {n_fields}'
    try:
        results = compute_pass_index(
            pos_ts, pos_xy, maps, field, spk_ts_overlap, lfp_ts, lfp_sig, lfp_fs, rng,
            method=METHOD, filter_band=FILTER_BAND, lfp_filter_band=LFP_FILTER_BAND,
            slope_bnds=SLOPE_BNDS, lfp_theta_mask=lfp_theta_mask,
            lfp_phase_unwrapped=lfp_phase_unwrapped)
    except TooFewSpikes as exc:
        row_f['PrecessionSkippedReason'] = str(exc)
        print(f'  {unit_label} field {k}: precession skipped -- {exc}')
        return row_f
    except Exception as exc:
        row_f['PrecessionSkippedReason'] = f'ERROR ({exc})'
        print(f'  {unit_label} field {k}: phase-precession ERROR ({exc})')
        return row_f

    tag = f'F{k}'
    png_path = output_dir / f'{file_prefix}_{tag}_PassIndex.png'
    plot_unit_summary(pos_xy, results, field_label, png_path)
    # fit_R shuffle-test figure, for every field the fit_R gate was run on.
    fit_r_png = None
    if results['shuffle_fit_r'] is not None:
        fit_r_png = output_dir / f'{file_prefix}_{tag}_FitRShuffle.png'
        plot_fit_r_shuffle(results, field_label, fit_r_png)
    # Also save the plot into RESULTS_FOLDER/<class folder> (next to the Excel file);
    # the CLASS_PLOT_FOLDERS versions get an extra top row with the polar plot, TMI
    # histogram and TMI shuffle null.
    if results['fit_too_steep']:
        class_dir_name = MULTILINES_FOLDER
    else:
        class_dir_name = CLASS_PLOT_FOLDERS.get(results['precession_class'])
    if class_dir_name:
        class_dir = RESULTS_FOLDER / class_dir_name
        if class_dir_name != MULTILINES_FOLDER and results['shuffle_fit_r'] is not None:
            class_dir = class_dir / FIT_R_SUBFOLDERS[results['fit_R_Significant']]
        class_dir.mkdir(parents=True, exist_ok=True)
        class_png = class_dir / f'{class_prefix}_{tag}_PassIndex.png'
        if class_dir_name == MULTILINES_FOLDER:
            shutil.copy2(png_path, class_png)
        else:
            plot_unit_summary(pos_xy, results, field_label, class_png, theta=theta)
        if fit_r_png is not None:
            shutil.copy2(fit_r_png, class_dir / f'{class_prefix}_{tag}_FitRShuffle.png')

    row_f['PrecessionTested'] = True
    row_f.update({
        'PassIndex_n_spikes': results['n_spikes'],
        'PassIndex_n_burst_events': results['n_events'],
        'PassIndex_n_spikes_running': results['n_spikes_running'],
        'n_passes_total': results['n_passes_total'],
        'n_passes_kept': results['n_passes_kept'],
        'spatial_filter_band': '%.4f-%.4f' % results['filter_band'],
        **_fit_columns(results),
    })
    print(f'  {unit_label} field {k}: PRECESSION rho={results["rho"]:.3f}  '
          f'p={results["p"]:.3g}  '
          f'fitR={results["fit_R"]:.3f} (shuffle p={results["fit_R_shuffle_p"]:.3g})  '
          f'slope={results["slope_deg_per_pass"]:.1f} deg/pass  '
          f'class={results["precession_class"]}')

    # Complete traversals only (pass index -1 -> 0 -> +1): same fit, CP_ columns
    comp = results['complete']
    row_f['n_passes_complete'] = results['n_passes_complete']
    row_f['CP_n_spikes'] = results['n_spikes_complete']
    if comp is None:
        row_f['CP_SkippedReason'] = results['complete_skip_reason']
        print(f'  {unit_label} field {k}: complete traversals skipped -- {results["complete_skip_reason"]}')
    else:
        row_f['CP_n_burst_events'] = comp['n_events']
        row_f.update({f'CP_{c}': v for c, v in _fit_columns(comp).items()})
        print(f'  {unit_label} field {k}: COMPLETE ({results["n_passes_complete"]} traversals, '
              f'{comp["n_spikes"]} spikes) rho={comp["rho"]:.3f}  p={comp["p"]:.3g}  '
              f'fitR={comp["fit_R"]:.3f} (shuffle p={comp["fit_R_shuffle_p"]:.3g})  '
              f'slope={comp["slope_deg_per_pass"]:.1f} deg/pass  class={comp["precession_class"]}')
    return row_f


def _fit_columns(res):
    """Excel columns of one circular-linear fit (_precession_fit result)."""
    return {
        'rho': res['rho'],
        'precession_p': res['p'],
        'slope_deg_per_pass': res['slope_deg_per_pass'],
        'r_squared': res['r_squared'], 'fit_R': res['fit_R'],
        'fit_R_shuffle_p': res['fit_R_shuffle_p'],
        'fit_R_Significant': res['fit_R_Significant'],
        'phase_range_deg': res['phase_range_deg'],
        **{c: v for _, key, *_ in PASS_INDEX_PHASE_POINTS
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
    """Population-level summary counts/percentages for the 'Summary' sheet.

    Each row's Count/Denominator/Percent together cover one pair of
    (number, percentage) items: total cells; SignificantThetaModulation;
    TMI_Significant (theta-modulated); PrecessionTested; and the
    PrecessionClass outcomes (phase_precessing / phase_recessing /
    phase_locked / no_phase_relation, from the asymptotic z-test p in kempter_lincirc) counted two ways -- out of all
    theta-modulated (TMI_Significant) cells, and out of only the subset that
    was actually precession-tested (some theta-modulated cells are skipped,
    e.g. no tracking file or too few overlapping spikes).
    """
    def pct(n, d):
        return (100.0 * n / d) if d > 0 else np.nan

    # df has one row per place field: cell-level counts use one row per unit, and the
    # precession outcomes are counted per field (the Field rows) and per cell (cells
    # with >= 1 field of that class).
    unit_key = ['FolderPath', 'Unit']
    cells = df.drop_duplicates(unit_key)
    total_cells = len(cells)
    n_sig_theta = int((cells['SignificantThetaModulation'] == True).sum())   # noqa: E712
    n_tmi_sig = int((cells['TMI_Significant'] == True).sum())                # noqa: E712
    tested = df['PrecessionTested'] == True                                  # noqa: E712
    n_fields_tested = int(tested.sum())
    n_cells_tested = int(df.loc[tested, unit_key].drop_duplicates().shape[0])
    n_multi = int((cells['n_fields'] > 1).sum())

    def n_cells_with(mask):
        return int(df.loc[mask, unit_key].drop_duplicates().shape[0])

    multipeak = df['MultiPeakField'] == True                                  # noqa: E712
    n_multipeak_fields = int(df.loc[multipeak, unit_key + ['ParentField']].drop_duplicates().shape[0])
    n_subfields = int(multipeak.sum())
    n_small = int((multipeak & (df['field_n_bins'] < MIN_SUBFIELD_SIZE_BINS)).sum())
    rows = [
        dict(Metric='Cells with > 1 place field (each field analysed separately)',
             Count=n_multi, Denominator=total_cells, DenominatorLabel='total cells',
             Percent=pct(n_multi, total_cells)),
        dict(Metric='Cells with >= 1 multi-peak field (split into sub-fields)',
             Count=n_cells_with(multipeak), Denominator=total_cells, DenominatorLabel='total cells',
             Percent=pct(n_cells_with(multipeak), total_cells)),
        dict(Metric='Multi-peak fields (each split into its sub-fields)',
             Count=n_multipeak_fields, Denominator=np.nan, DenominatorLabel='', Percent=np.nan),
        dict(Metric=f'Sub-fields < {MIN_SUBFIELD_SIZE_BINS} bins (not analysed)',
             Count=n_small, Denominator=n_subfields, DenominatorLabel='sub-fields',
             Percent=pct(n_small, n_subfields)),
        dict(Metric='Place fields / sub-fields precession-tested',
             Count=n_fields_tested, Denominator=np.nan, DenominatorLabel='',
             Percent=np.nan),
    ]
    for cls, col in (('phase_precessing', 'is_precessing'), ('phase_succeeding', 'is_recessing'),
                     ('phase_locked', 'is_phase_locked'), ('no_phase_relation', None)):
        mask = (df['PrecessionClass'] == cls) if col is None else (df[col] == True)  # noqa: E712
        n_f = int(mask.sum())
        n_c = n_cells_with(mask)
        rows += [
            dict(Metric=f'{cls} fields (of precession-tested fields)', Count=n_f,
                 Denominator=n_fields_tested, DenominatorLabel='precession-tested fields',
                 Percent=pct(n_f, n_fields_tested)),
            dict(Metric=f'cells with >= 1 {cls} field (of precession-tested cells)', Count=n_c,
                 Denominator=n_cells_tested, DenominatorLabel='precession-tested cells',
                 Percent=pct(n_c, n_cells_tested)),
        ]
    n_precessing = n_cells_with(df['is_precessing'] == True)                 # noqa: E712
    cell_rows = [
        dict(Metric='SignificantThetaModulation (Rayleigh) cells',
             Count=n_sig_theta, Denominator=total_cells, DenominatorLabel='total cells',
             Percent=pct(n_sig_theta, total_cells)),
        dict(Metric='TMI_Significant (theta-modulated) cells',
             Count=n_tmi_sig, Denominator=total_cells, DenominatorLabel='total cells',
             Percent=pct(n_tmi_sig, total_cells)),
        dict(Metric='PrecessionTested cells (>= 1 field tested)',
             Count=n_cells_tested, Denominator=total_cells, DenominatorLabel='total cells',
             Percent=pct(n_cells_tested, total_cells)),
        dict(Metric='cells with >= 1 phase_precessing field (of theta-modulated cells)',
             Count=n_precessing, Denominator=n_tmi_sig, DenominatorLabel='theta-modulated cells',
             Percent=pct(n_precessing, n_tmi_sig)),
    ]
    return pd.DataFrame(cell_rows + rows, columns=['Metric', 'Count', 'Denominator', 'DenominatorLabel', 'Percent'])


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
                    and n not in ('ARENA_COMPARISON_METRICS', 'TMI_COMPARISON_METRICS', 'ARENA_COLORS',
                                  'SESSION_COLORS', 'GROUP_COLORS', 'GROUP_COUNT_COLUMNS',
                                  'PI0_RANGE_COLORS')]
    rows += [(n, repr(module_globals[n]) if not isinstance(module_globals[n], (str, Path))
              else str(module_globals[n])) for n in config_names]
    # constants hard-coded inside functions
    rows += [
        ('PASS_DENSITY_PASS_INDEX_BINS', 40),                 # compute_pass_index: linspace(-1, 1, 41)
        ('PASS_DENSITY_PHASE_BINS', 100),                     # compute_pass_index: linspace(0, 2*pi, 101)
        ('PASS_DENSITY_GAUSS_SIGMA_BINS', 1.5),               # compute_pass_index density smoothing
        ('PASS_DENSITY_GAUSS_TRUNCATE', 2.0),
        ('RATEMAP_GAUSS_TRUNCATE', 3.0),                      # rate-map smoothing
        ('BANDPASS_FILTER_ORDER', 4),                         # bandpass_filter default
        ('TMI_NUM_CYCLES', 5),                                # calc_tmi default
        ('TMI_HIST_SMOOTH_KERNEL_N', 7),                      # calc_tmi histogram smoothing
        ('TMI_HIST_SMOOTH_SIGMA', 0.5),
        ('ACG_WELCH_WINDOW', 'hann'),
        ('PRECESSION_SIGNIFICANCE_TEST',
         ('fit_R burst-preserving phase-permutation shuffle p < ALPHA (if USE_FIT_R_GATE) AND '
          'Kempter et al. (2012) asymptotic z-test p < ALPHA (on burst events if BURST_COLLAPSE_FOR_ZTEST); '
          'per place field: spikes restricted to speed >= MIN_SPEED_CM_S and to passes whose field-index '
          'peak lies inside the field and reaches PASS_MIN_FIELD_PEAK_FRAC of its peak field index')),
    ]
    pd.DataFrame(rows, columns=['Parameter', 'Value']).to_csv(csv_path, index=False)
    return csv_path


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
    for data_folder in session_folders:
        print(f'\n=== Session: {data_folder} ===')
        try:
            all_rows.extend(process_session(data_folder, rng, edmin_records))
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

    columns = ['Session', 'FolderPath', 'Unit', 'Field', 'ParentField', 'SubField', 'n_fields', 'MultiField',
               'MultiPeakField', 'n_subfields', 'field_kde_peak_Hz', 'field_kde_peak_rel', 'field_kde_note',
               'FieldDetection',
               'field_peak_rate_Hz', 'field_peak_rate_raw_Hz', 'field_n_bins', 'field_area_cm2',
               'field_pct_of_occupied_area', 'field_radius_cm', 'field_com_x_cm', 'field_com_y_cm',
               'field_com_theta_deg', 'field_arc_length_cm', 'field_track_width_cm',
               'ntt_file', 'lfp_file', 'cell_number', 'n_spikes_total',
               'n_spikes_theta', 'MRL', 'PreferredPhase_deg', 'Rayleigh_p',
               'SignificantThetaModulation', 'PhasePeak_deg', 'PhaseValley_deg', 'TMI',
               'TMI_shuffle_p', 'TMI_Significant', 'PrecessionTested', 'PassIndex_n_spikes',
               'PassIndex_n_burst_events', 'PassIndex_n_spikes_running', 'n_passes_total', 'n_passes_kept',
               'spatial_filter_band',
               'rho', 'r_squared', 'fit_R', 'fit_R_shuffle_p', 'fit_R_Significant', 'precession_p', 'slope_deg_per_pass', 'phase_range_deg',
               *(f'phase_at_{key}_{suffix}' for _, key, *_ in PASS_INDEX_PHASE_POINTS
                 for suffix in ('deg', 'range')),
               'is_precessing', 'is_recessing', 'is_phase_locked', 'PrecessionClass', 'n_fit_lines',
               'PrecessionSkippedReason',
               # complete traversals only (pass index -1 -> 0 -> +1)
               'n_passes_complete', 'CP_n_spikes', 'CP_n_burst_events',
               *(f'CP_{c}' for c in ('rho', 'r_squared', 'fit_R', 'fit_R_shuffle_p', 'fit_R_Significant',
                                     'precession_p', 'slope_deg_per_pass', 'phase_range_deg',
                                     *(f'phase_at_{key}_{suffix}' for _, key, *_ in PASS_INDEX_PHASE_POINTS
                                       for suffix in ('deg', 'range')),
                                     'is_precessing', 'is_recessing', 'is_phase_locked', 'PrecessionClass',
                                     'n_fit_lines')),
               'CP_SkippedReason']
    df = pd.DataFrame(all_rows, columns=columns)
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
            out_dir=RESULTS_FOLDER / SESSION_TYPE_PLOT_FOLDER / arena, file_prefix=f'SessionType_{arena}',
            group_word='session type', title_prefix=f'{arena}: ')
        for key, table in tables.items():
            table.insert(0, 'Arena', arena)
            session_type_tables.setdefault(_comparison_sheet_name('SessionType', key), []).append(table)

    # ---- 2) Arena comparison (Circle vs Linear vs Open), all session types pooled ----
    print('\n=== Arena comparison (all session types pooled within each arena) ===')
    arena_tables = run_group_comparisons(
        df.loc[df['Arena'].notna()], 'Arena', ARENA_LABELS, out_dir=RESULTS_FOLDER,
        file_prefix='ArenaComparison', group_word='arena')

    readme_df = pd.DataFrame({'Note': [
        'ThetaPhase sheet: ONE ROW PER PLACE FIELD. A cell with several place fields (MultiField = True, '
        'n_fields > 1) has one row per field; its unit-level columns (spike counts, MRL, Rayleigh, TMI) are '
        'repeated on each of its rows. Cells with no field analysed (not theta-modulated, no tracking, '
        'no field detected) have a single row with Field empty.',
        'Field numbering: Field 1 = the field with the highest peak (smoothed) firing rate, then 2, 3, ... in '
        'descending order of peak rate. Per-field figures are named <animal>_<unit>_F<Field>_PassIndex.png '
        '(and _F<Field>_FitRShuffle.png); the figure title gives "Field k of n".',
        f'Place fields: threshold method of PlaceCell_Main_v20 -- bins visited >= {FIELD_MIN_VISITS} times with '
        f'smoothed rate >= {FIELD_RATE_THRESHOLD_FRAC:g} x peak AND > the mean rate, 8-connected components '
        f'of >= {MIN_FIELD_SIZE_BINS} bins (Circle track: 1-D angular map, contiguous runs wrapping around); '
        f'frames slower than {RATEMAP_IMMOBILITY_SPEED_CM_S:g} cm/s excluded. Rate map: Climer pass-index '
        f'rate map ({BINSIDE:g} cm bins, Gaussian sigma = {SMTH_WIDTH / 2:g} cm, unvisited bins filled with the '
        f'nearest visited value).',
        f'Multi-peak fields (v26): right after detection every field is tested for multiple peaks with the '
        f'KDE method of PlaceCell_MultiPeak_KDE_v2 (KDE of the field bins weighted by the {MP_KDE_RATEMAP_TYPE} '
        f'map, bw_method={MP_KDE_BW_METHOD}; peaks >= {MP_PEAK_REL_HEIGHT_FRAC:g} x the field\'s highest KDE peak). '
        'A field with >= 2 peaks (MultiPeakField = True) is cut at the troughs by watershed on the KDE, and its '
        'sub-fields REPLACE it in everything below (one row each; Field = "<ParentField>.<SubField>", '
        'sub-field 1 = highest KDE peak; field_* columns describe the sub-field). Sub-fields of '
        f'< {MIN_SUBFIELD_SIZE_BINS} bins are not analysed (row kept, PrecessionSkippedReason set). n_fields = '
        'number of fields detected before splitting. Arena / session-type precession comparisons use one value '
        'per analysed field or sub-field.',
        'Each field is analysed separately: spatial filter band from that field\'s radius (field_radius_cm), '
        'and only passes whose field-index peak lies inside that field (and reaches '
        f'{PASS_MIN_FIELD_PEAK_FRAC:g} x its peak field index) contribute spikes.',
        'Summary sheet: theta-modulation counts are per cell; precession outcomes are given per field and as '
        '"cells with >= 1 field" of each class. Arena / session-type TMI comparisons use one value per cell; '
        'precession comparisons (r^2, fit R, slope, phase range, phase at pass index) use one value per field.',
        'CP_* columns: the same circular-linear analysis repeated on spikes from complete traversals only '
        f'(pass index starts <= {-1 + COMPLETE_PASS_EDGE_TOL:g}, crosses 0 and ends >= {1 - COMPLETE_PASS_EDGE_TOL:g} '
        'within one kept pass); n_passes_complete = number of such traversals. CP_SkippedReason is set when '
        'fewer than MIN_SPIKES_FOR_FIT spikes fall in complete traversals. PrecessionClass, the class plot '
        'folders and all arena / session-type comparisons use the all-spike fit.',
    ]})
    excel_path = RESULTS_FOLDER / OUTPUT_EXCEL_NAME
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        readme_df.to_excel(writer, sheet_name='README', index=False)
        df.to_excel(writer, sheet_name='ThetaPhase', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        for sheet_name, tables in session_type_tables.items():
            pd.concat(tables, ignore_index=True).to_excel(writer, sheet_name=sheet_name, index=False)
        for key, table in arena_tables.items():
            table.to_excel(writer, sheet_name=_comparison_sheet_name('Arena', key), index=False)
    n_units = df[['FolderPath', 'Unit']].drop_duplicates().shape[0]
    print(f'\nDone. {n_units} unit(s) processed, {int(df["Field"].notna().sum())} place field row(s). '
          f'Summary saved to {excel_path}')

    metadata_path = save_metadata_csv(excel_path.with_name(excel_path.stem + '_metadata.csv'))
    print(f'Run metadata saved to {metadata_path}')


if __name__ == '__main__':
    main()
