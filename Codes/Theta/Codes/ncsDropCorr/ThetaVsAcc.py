"""
Theta frequency/power vs. running ACCELERATION.

ACCELERATION VARIANT of ThetaVsSpeed_v8_Ferret_Mobility.py (Mobility_v3). The
LFP pipeline, theta frequency/power estimate, speed calculation, saturation
rejection, duplicate-file handling and power normalizations are unchanged; the
predictor of every fit is acceleration instead of speed:

  * Acceleration (cm/s^2) is derived from the speed values the speed script
    already computes: the BIN_AGG_METHOD running speed of each BINSIZE
    (0.25 s) time bin, differentiated with a central difference,
    a[i] = (v[i+1] - v[i-1]) / (2 * BINSIZE)   (np.gradient; one-sided at the
    first/last bin). It is taken on the full, gap-free time-bin grid -- before
    any bin is dropped -- so the two neighbours are always the true adjacent
    0.25 s bins; a bin whose neighbour has no speed gets NaN and is dropped.
    Acceleration is SIGNED: speeding up > 0, slowing down < 0.
  * Immobility exclusion unchanged: time bins with speed < MIN_SPEED_CMS
    (4 cm/s) are kept in the parquet (Mobile = False) but left out of every
    fit; on the mixed-model figures their acceleration bins are drawn in gray.
  * Acceleration bins are ACC_BIN_WIDTH (4 cm/s^2) wide in every analysis,
    centred on 0: [-2, 2) is the 0 cm/s^2 bin, then [2, 6) -> 4, [-6, -2) -> -4 ...
  * Mixed model: response ~ Acceleration + (1 | Session), Acceleration = the
    acceleration-bin centre (one value per acceleration bin visited per session).
  * Speed-controlled mixed model (run_speed_controlled_model): response ~
    Acceleration + Speed + (1 | Session), fit on 2-D (speed bin x acceleration
    bin) medians per session -- 4 cm/s speed bins anchored at 0, ACC_BIN_WIDTH
    acceleration bins centred on 0 -- with the bin centres as predictors.
  * Dunn Fig. 3 per-session OLS: NO percentile cut (the speed script dropped
    each session's speeds above the 90th percentile); every mobile acceleration
    bin is used.
  * Individual-file fits: unchanged rule -- no percentile cut, an acceleration
    bin is kept only with >= INDIV_MIN_TIME_PER_BIN_S (5 s) of data.
  * PowerdB baseline unchanged (4-7 cm/s running SPEED), so Power/PowerdB/
    PowerRel values per time bin are the same as in the speed script.
  * Results are written to Results/ThetaVsAcc.
  * Acceleration and deceleration analysed separately (run_acc_decel_split):
    every analysis above is repeated on the time bins with a >= 2 cm/s^2
    (-> Results/ThetaVsAcc/Acc) and with a < -2 cm/s^2 (-> .../Decel), the
    near-zero bin [-2, 2) in neither; predictor = |a| in both, so a positive
    Decel slope means theta rises with harder braking. Tables there are .csv.

Speed-script description (pipeline unchanged):

Recursively finds every Neuralynx .ncs LFP file under ROOT_DIR, pairs it with
the tracking .csv in its own folder, cleans the LFP the same way as
ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean_v3.py (resample -> notch ->
detrend -> low-pass), then band-pass filters to theta and estimates
instantaneous frequency/power using the peak-trough interpolation method of
Dunn et al. 2022 (Nature Communications 13:5905, the ferret theta paper) --
a port of the paper's own implementation
(calculate_peak_trough_signal_parameters.m / findMinMax.m /
clean_peaks_and_troughs.m, Toolbox/Signal-processing/,
https://github.com/slsdunn/theta-paper-code), binned against running speed
computed from the cleaned tracking position.

Before the mixed-model statistics, the per-time-bin (BINSIZE) Frequency/Power
values are further aggregated into 4 cm/s-wide speed bins per session
(representative Frequency/Power per bin -- see aggregate_by_speed_bin), and
the random-intercept LMM is fit with each bin's center speed as the predictor,
i.e. one value per speed bin actually visited per session rather than
thousands of highly autocorrelated time samples. The representative value used for every bin (both the per-time-bin
aggregation and the per-speed-bin aggregation) is controlled by the
BIN_AGG_METHOD config option -- 'median' or 'mean'.

Theta power is analysed three ways (see add_normalized_power), each with its
own mixed-model / per-session-OLS figures:
  * Power    -- no normalization, squared envelope in uV^2;
  * PowerdB  -- decibel baseline normalization, 10*log10(P / B), B = the
                session's representative power while running at
                DB_BASELINE_SPEED_CMS (Cohen 2014, Analyzing Neural Time Series
                Data, ch. 18);
  * PowerRel -- relative power, theta power as % of the session's total
                REL_POWER_BAND power -- the time-domain (Parseval) equivalent
                of the PSD normalization in Fig4a_PSD_v2_Simul.py.

MOBILITY VARIANT (copy of ThetaVsSpeed_v8_Ferret_Final.py). Same pipeline,
except:
  * Immobility excluded: per-time-bin (BINSIZE) rows with running speed below
    MIN_SPEED_CMS (4 cm/s) are left out of every fit. They are still computed
    and kept in the parquet (column Mobile = False), and the mixed-model
    figures draw their speed bins in gray (EXCLUDED_COLOR), not in the fit.
  * dB baseline moved from 2-5 cm/s to DB_BASELINE_SPEED_CMS = 4-7 cm/s (the
    old window lies mostly below the 4 cm/s cut).
  * Dunn Fig. 3 per-session OLS: speeds < 4 cm/s are dropped first, then the
    per-session 90th-percentile cut is taken on the remaining (mobile) speeds.
  * Speed bins are SPEED_BIN_WIDTH_CMS = 4 cm/s wide in every analysis (mixed
    model, Dunn Fig. 3 per-session OLS, individual-file plots), anchored at 0:
    [0,4) is immobility and the first mobile bin is a full [4,8) cm/s.
    (Before: 2 cm/s for the mixed model and 5 cm/s for Fig. 3, whose first
    mobile bin held only 4-5 cm/s.)
  * Relative power denominator unchanged: whole-session 1-100 Hz power.
  * Copied .ncs files dropped (DROP_DUPLICATE_LFP): within a recording folder,
    files with identical record timestamps / valid-sample counts / samples are
    copies of one signal; only the first in natural name order is analysed.
    The files are read from the list made once by Make_NcsFileList.py
    (NCS_FILE_LIST_CSV) -- no directory walk or file comparison per run.
  * Results are written to Results/ThetaVsSpeed/Mobility_v3 (v2 = 2 / 5 cm/s
    speed bins, PowerdB in the individual-file plots).
  * Individual-file correlations (run_individual_correlations): the Dunn
    Fig. 3 per-session OLS is drawn for every analysed .ncs file on its own
    (Frequency and PowerRel -- relative power -- vs speed)
    -> IndivdualCorrel/<session>.png. Each
    file's slope is classed as significantly positive / n.s. positive / n.s.
    negative / significantly negative (Benjamini-Hochberg FDR across files,
    per measure) or not fitted, and the % of all files in each class is
    plotted overall and per animal (IndividualCorrel_PercentPositive.png).
    The individual fits drop no upper tail of speeds (the group per-session
    OLS keeps the 90th-percentile cut); instead a speed bin is kept only with
    >= INDIV_MIN_TIME_PER_BIN_S (5 s) of data.
    Files significant for BOTH Frequency and PowerRel are counted by the
    signs of their two slopes (both positive = red, both negative = blue,
    opposite = violet, the RGB mix of the two), overall and per animal
    -> IndividualCorrel_BothSignificant.png/.xlsx. Re-plot this figure alone
    from the saved per-file workbook with IndividualCorrel_BothSignificant.py.
  * ADC saturation removed (REJECT_SATURATED_BINS), as in
    Figc_ACG_ScalingCorrected_v11_SaturationReject.py: the amplifier clips at
    +/-InputRange, read from each file's own .ncs header (1500 or 1000 uV).
    Clipped samples are found on the raw, repaired native-rate trace, before
    any filtering, and every clipped run is widened by SATURATION_MARGIN_S
    (0.2 s) on both sides, because filtering spreads the clip edges. Every
    0.25 s time bin (BINSIZE) that overlaps a widened run is removed (its
    Frequency/Power are set to NaN and the row is dropped with the other NaN
    bins), and the widened-run samples are left out of the relative-power
    denominator (TotalPower). Per-file counts -> Saturation_summary.xlsx.
"""

import os
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm, linregress, ttest_1samp, wilcoxon
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import openpyxl

# ---- Timestamp-anchored .ncs loading ----
# ncs_repair.py corrects the partial-record and sample-loss timing error in the
# raw .ncs files (see its header). It sits in this script's folder; the absolute
# path also finds it when running cell-by-cell from another working directory.
import sys
NCS_REPAIR_DIR = r'C:\Runita\NMR\Mole-Rat_PlaceCell\Codes\Theta\Codes\ncsDropCorr'
if NCS_REPAIR_DIR not in sys.path:
    sys.path.insert(0, NCS_REPAIR_DIR)
from ncs_repair import load_repaired_uv, open_ncs, find_runs
from NcsFileList_Utils import listed_ncs_files, listed_copies, natural_key


# %% ==================== Configuration ===========================================

# Root directory to search recursively for .ncs files. Every .ncs found
# anywhere under this tree (in any subfolder) is processed, paired with the
# single tracking .csv that lives in the same folder.
ROOT_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final'

# Acceleration vs. theta stats (mixed-model plots + summary) are written here.
OUTPUT_DIR = Path(ROOT_DIR) / 'Results' / 'ThetaVsAcc'

# Per-time-bin rows of every analysed file (all speeds; Mobile flags the rows
# used in the fits).
OUTPUT_PARQUET = str(OUTPUT_DIR / 'ThetaVsAcc_out.parquet')

# ---- Acceleration ----
# Acceleration (cm/s^2) = central difference of the per-time-bin (BINSIZE)
# speed, see process_ncs_session. Signed: speeding up > 0, slowing down < 0.
ACC_UNIT = 'cm/s²'
# Acceleration-bin width (cm/s^2) for every analysis (mixed model, per-session
# OLS, individual-file fits, speed-controlled model). Bins are centred on 0:
# bin k = floor(a / w + 0.5) covers [(k - 0.5) w, (k + 0.5) w), centre k * w.
ACC_BIN_WIDTH = 4.0

# ---- Immobility exclusion ----
# Time bins with running speed below MIN_SPEED_CMS are immobility: kept in the
# output and drawn in EXCLUDED_COLOR on the speed-axis figures, but not used
# in any fit (mixed model or per-session OLS).
MIN_SPEED_CMS = 4.0
EXCLUDED_COLOR = '#B0B0B0'

# ---- Copied LFP files ----
# Some recording folders hold bit-identical copies of one .ncs signal (placed
# so every tetrode had a matched LFP file). True: analyse only the first file
# of each set of copies (natural name order, e.g. CSC1 of CSC1 = CSC4), as
# listed in NCS_FILE_LIST_CSV (made by Make_NcsFileList.py for ROOT_DIR).
# False: every .ncs file under ROOT_DIR (directory walk, copies included).
DROP_DUPLICATE_LFP = True
NCS_FILE_LIST_CSV = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final\NcsFileList\Ncs_files_to_analyse.csv'
# Run metadata (all settings, data/script paths) + a copy of this script
# are written here -- see save_run_metadata.
META_DIR = OUTPUT_DIR / 'meta'

# ---- Acquisition (Neuralynx .ncs) ----
NATIVE_FS = 32000  # native .ncs sampling rate (Hz)
# V per ADC count is read from each file's own header (-ADBitVolts, see
# read_ncs_adbitvolts): it follows the recording's InputRange, which differs
# between animals (1500 uV -> 4.58e-8 V, 1000 uV -> 3.05e-8 V), so a single
# hard-coded constant would scale animals' power differently.

# Neuralynx .ncs record format (512 int16 samples per record, 16 kB header skipped)
ncs_dtype = np.dtype([
    ('timestamp'  , '<u8'),
    ('sc_number'  , '<u4'),
    ('cell_number', '<u4'),
    ('params'     , '<u4'),
    ('samples'    , '<i2', (512,)),
])

# ---- ADC saturation (as in Figc_ACG_ScalingCorrected_v11_SaturationReject.py) ----
# The clipping level is each file's own -InputRange header value (uV), never a
# fixed number: 1500 uV for some recordings, 1000 uV for others. Clipped samples
# are stored at ADMaxValue = 32767 counts, so a sample counts as clipped if
# |value| >= InputRange - SATURATION_TOL_STEPS ADC steps (one step = ADBitVolts);
# 1.5 steps flags exactly the +/-32767 rail and is safe against rounding.
REJECT_SATURATED_BINS = True     # True: drop every time bin within SATURATION_MARGIN_S of clipping
SATURATION_TOL_STEPS  = 1.5      # ADC steps below +/-InputRange that still count as clipped
# Each clipped run is widened by this much on both sides before time bins are
# rejected, since filtering spreads the sharp clip edges into neighbouring samples.
SATURATION_MARGIN_S   = 0.2      # s, either side of each clipped run

# ---- 50 Hz line-noise cleaning (European mains) ----
LINE_HARMONICS = [50.0, 100.0, 150.0, 200.0]
NOTCH_Q = 30.0

# ---- Low-pass filtering (restrict the broadband trace to <100 Hz before
# the theta-band filter below, so high-frequency content can't distort peak
# finding) ----
LOWPASS_CUTOFF_HZ = 100.0
LOWPASS_ORDER = 4

# ---- Working sample rate for the cleaned LFP / theta filtering / peak
# finding (the raw .ncs is resampled down to this rate) ----
FS_LFP = 1000

# ---- Theta band-pass (applied after the general LFP cleanup, on top of it) ----
BANDPASS_CUTOFF = (3.0, 7.0)
FILTER_ORDER = 5

# ---- Peak/trough detection for the instantaneous frequency/power estimate
# (see estimate_instantaneous_frequency_power) -- findMinMax.m's hysteresis
# threshold, as a fraction of the band-passed signal's median absolute
# amplitude. Dunn et al.'s own value ("chosen empirically"). ----
PEAK_THRESH_PC = 0.25

# ---- Edge trimming (seconds) -- drops filtfilt edge artifacts from the
# start/end of each cleaned trace ----
TRIM_DUR = 0.25

BINSIZE = 0.25

# Representative value computed for every bin -- both the per-time-bin
# (BINSIZE) aggregation in bin_data and the per-speed-bin aggregation in
# aggregate_by_acc_bin (the values the mixed model is fit on). 'median' or
# 'mean'.
BIN_AGG_METHOD = 'median'

# Acceleration-binning applied before the mixed model (see aggregate_by_acc_bin):
# with BINSIZE=0.25 s time bins, a session contributes thousands of
# Frequency/Power/Acceleration points, which are not independent samples of
# the relationship and inflate the mixed model's apparent significance.
# Grouping by ACC_BIN_WIDTH-wide acceleration bins and taking the median
# Frequency/Power within each (session, bin) group gives one value per
# acceleration bin visited per session; the mixed model's predictor is the
# bin center (..., -4, 0, 4, ... cm/s^2).
# Speed bins (cm/s, anchored at 0: [0,4), [4,8), ...) are used only for the
# speed axis of the speed-controlled model (run_speed_controlled_model).
SPEED_BIN_WIDTH_CMS = 4.0

# ---- Theta-power normalization (see add_normalized_power) ----
# dB baseline: each session's baseline B is the BIN_AGG_METHOD power of its
# time bins with speed in [lo, hi) cm/s. A fixed speed window every session
# covers keeps the baseline from depending on how much fast running a
# session happens to contain (as a session-wide baseline would). The v8
# window (2-5 cm/s) lies mostly below MIN_SPEED_CMS, so it is moved to the
# slowest mobile speeds, keeping its 3 cm/s width. Per-session coverage is
# written to dB_baseline_coverage.xlsx.
DB_BASELINE_SPEED_CMS = (4.0, 7.0)
# Relative power: band whose total power (mean square of the band-passed
# LFP over the whole session) is the denominator -- matches NORM_BAND in
# Fig4a_PSD_v2_Simul.py.
REL_POWER_BAND = (1.0, 100.0)
REL_POWER_FILTER_ORDER = 4

# (column, file-name suffix, y-axis label, title, slope unit) per power measure
POWER_MEASURES = [
    ("Power", "", "Theta Power [µV²]",
     "Theta Power vs. Acceleration (no normalization)", "µV²"),
    ("PowerdB", "_dB",
     f"Theta Power [dB re {DB_BASELINE_SPEED_CMS[0]:g}–{DB_BASELINE_SPEED_CMS[1]:g} cm/s]",
     "Theta Power vs. Acceleration (dB baseline)", "dB"),
    ("PowerRel", "_Relative",
     f"Relative Theta Power [% of {REL_POWER_BAND[0]:g}–{REL_POWER_BAND[1]:g} Hz power]",
     "Theta Power vs. Acceleration (relative power)", "%"),
]

# ---- Tracking position cleaning (matches
# ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean_v3.py's
# _smooth_tracking_position): frame-to-frame jumps implying a speed above
# POS_JUMP_THRESH_CMS are tracking artifacts, linearly interpolated over,
# then the x/y traces are Gaussian-smoothed with sigma POS_SMOOTH_SIGMA_SMP
# (in samples). Tracking .csv column layout: UNIX timestamp in us (col A),
# x in cm (col D), y in cm (col E).
POS_JUMP_THRESH_CMS = 90.0
POS_SMOOTH_SIGMA_SMP = 1.0

# Per-animal colors — first 4 colors of matplotlib's Pastel1 palette.
ANIMAL_COLORS = {
    'Fa8477': '#FBB4AE',  # Pastel1[0] red
    'Fa1059': '#B3CDE3',  # Pastel1[1] blue
    'Fa23BD': '#CCEBC5',  # Pastel1[2] green
    'Fa5834': '#DECBE4',  # Pastel1[3] purple
}


def animal_color(label, default='#888888'):
    """ANIMAL_COLORS entry for an animal label, matched on the ID so that e.g.
    'Fa1059', '1059Test' and 'FA1059' all resolve; `default` if no ID matches."""
    label = str(label).upper()
    for animal_id, color in ANIMAL_COLORS.items():
        if animal_id[2:].upper() in label:
            return color
    return default


def ordered_animals(labels):
    """Unique animal labels in ANIMAL_COLORS order (unknown labels last, sorted)."""
    labels = list(pd.unique(pd.Series(labels).dropna()))
    known = [a for a in ANIMAL_COLORS if a in labels]
    return known + sorted(a for a in labels if a not in known)


# %% ==================== .ncs / tracking data import ==============================

def read_ncs_adbitvolts(fpath):
    """V per ADC count, from the '-ADBitVolts' line of the 16 kB .ncs header."""
    with open(fpath, 'rb') as f:
        header = f.read(16 * 1024).decode('latin-1')
    m = re.search(r'-ADBitVolts\s+(\S+)', header)
    if m is None:
        raise ValueError('No -ADBitVolts entry in the .ncs header.')
    return float(m.group(1))


def read_ncs_input_range(fpath):
    """Amplifier input range (uV) from the '-InputRange' line of the .ncs header,
    i.e. the level at which the recorded signal clips. None if the header has no
    InputRange (simulated files), in which case no saturation check is possible."""
    with open(fpath, 'rb') as f:
        header = f.read(16 * 1024).decode('latin-1')
    m = re.search(r'^-InputRange[ \t]+(\S+)', header, flags=re.MULTILINE)
    return float(m.group(1)) if m else None


def saturated_samples(lfp_uv, input_range_uv, ad_bit_volts, tol_steps=SATURATION_TOL_STEPS):
    """Boolean mask of clipped samples of the raw, native-rate trace (uV): those
    within `tol_steps` ADC steps of +/-`input_range_uv`. All False if the file
    has no InputRange."""
    if input_range_uv is None:
        return np.zeros(len(lfp_uv), dtype=bool)
    rail_uv = input_range_uv - tol_steps * ad_bit_volts * 1e6
    return np.abs(lfp_uv) >= rail_uv


def near_clip_mask(sat_native, n_out, fs_native=NATIVE_FS, fs_out=FS_LFP,
                   margin_s=SATURATION_MARGIN_S):
    """`sat_native` (native-rate clipped-sample mask) mapped onto the first
    `n_out` samples of the fs_out trace that clean_lfp returns (an fs_out sample
    is clipped if any of the native samples it covers is), then every clipped
    run widened by `margin_s` on both sides."""
    step = int(round(fs_native / fs_out))
    n_full = -(-len(sat_native) // step) * step
    padded = np.zeros(n_full, dtype=bool)
    padded[:len(sat_native)] = sat_native
    sat = padded.reshape(-1, step).any(axis=1)
    sat = np.pad(sat, (0, max(0, n_out - len(sat))))[:n_out]

    starts, lengths = find_runs(sat)
    margin = int(round(margin_s * fs_out))
    edges = np.zeros(n_out + 1, dtype=np.int64)
    np.add.at(edges, np.clip(starts - margin, 0, n_out), 1)
    np.add.at(edges, np.clip(starts + lengths + margin, 0, n_out), -1)
    return np.cumsum(edges[:-1]) > 0


def load_ncs(fpath):
    """Read a Neuralynx .ncs file.

    Returns
    -------
    lfp             : ndarray  repaired trace in microvolts (uV), on a uniform
                                NATIVE_FS grid (see ncs_repair.py)
    start_timestamp : int      UNIX timestamp (us) of the first sample -- same
                                clock as the tracking .csv.
    ad_bit_volts    : float    V per ADC count (this file's -ADBitVolts)
    """
    ad_bit_volts = read_ncs_adbitvolts(fpath)
    lfp, start_us = load_repaired_uv(fpath, ad_bit_volts)
    return lfp, start_us, ad_bit_volts


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
    """Remove a polynomial trend (default: linear) from the full LFP trace."""
    return signal.detrend(np.asarray(x, dtype=np.float64), type=dtype)  # type: ignore


def lowpass_filter(x, fs_hz, cutoff_hz, order=4):
    """Zero-phase Butterworth low-pass filter (filtfilt)."""
    nyq = fs_hz / 2.0
    b, a = signal.butter(order, cutoff_hz / nyq, btype='low')
    return signal.filtfilt(b, a, np.asarray(x, dtype=np.float64))


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


def _smooth_tracking_position(x_cm, y_cm, t_us,
                              jump_thresh_cms=POS_JUMP_THRESH_CMS,
                              sigma_samples=POS_SMOOTH_SIGMA_SMP):
    """Clean and smooth the x/y tracking position before any downstream use
    (iterative jump removal + Gaussian smoothing)."""
    n = len(x_cm)
    if n < 2:
        return x_cm.copy(), y_cm.copy()

    bad = np.zeros(n, dtype=bool)
    for _ in range(n):
        good_idx = np.where(~bad)[0]
        if len(good_idx) < 2:
            break
        dt_good = np.diff(t_us[good_idx]) * 1e-6
        with np.errstate(invalid='ignore', divide='ignore'):
            step_speed = np.hypot(np.diff(x_cm[good_idx]), np.diff(y_cm[good_idx])) / dt_good
        step_speed[dt_good <= 0] = 0.0

        newly_bad = step_speed > jump_thresh_cms
        if not newly_bad.any():
            break
        bad[good_idx[1:][newly_bad]] = True

    good_idx = np.where(~bad)[0]
    if len(good_idx) == 0 or len(good_idx) == n:
        x_clean, y_clean = x_cm.copy(), y_cm.copy()
    else:
        x_clean = np.interp(t_us, t_us[good_idx], x_cm[good_idx])
        y_clean = np.interp(t_us, t_us[good_idx], y_cm[good_idx])

    x_smooth = gaussian_filter1d(x_clean, sigma=sigma_samples, mode='nearest')
    y_smooth = gaussian_filter1d(y_clean, sigma=sigma_samples, mode='nearest')
    return x_smooth, y_smooth


def _instantaneous_speed(x_cm, y_cm, t_us):
    """Per-frame running speed (cm/s), padded to len(t_us) (repeats last value)."""
    dt_s = np.diff(t_us) * 1e-6
    with np.errstate(invalid='ignore', divide='ignore'):
        speed = np.hypot(np.diff(x_cm), np.diff(y_cm)) / dt_s
    speed[dt_s <= 0] = np.nan
    return np.append(speed, speed[-1])


def compute_velocity_from_position(csv_path):
    """Compute running speed (cm/s) from a tracking .csv.

    Column layout (positional): UNIX timestamp in us (col A), x in cm
    (col D), y in cm (col E). Position is cleaned first (jump removal +
    Gaussian smoothing, see `_smooth_tracking_position`), then speed is the
    frame-to-frame displacement of the cleaned trace divided by the actual
    elapsed time between samples.

    Returns
    -------
    time_us : ndarray   absolute UNIX timestamp of each tracking sample (us)
    speed   : ndarray   running speed (cm/s), same length as time_us
    """
    df = pd.read_csv(csv_path, usecols=[0, 3, 4])
    df.columns = ['time_us', 'x', 'y']
    df = df.sort_values('time_us').reset_index(drop=True)

    time_us = df['time_us'].to_numpy(dtype=np.float64)
    x_cm = df['x'].to_numpy(dtype=np.float64)
    y_cm = df['y'].to_numpy(dtype=np.float64)

    x_smooth, y_smooth = _smooth_tracking_position(x_cm, y_cm, time_us)
    speed = _instantaneous_speed(x_smooth, y_smooth, time_us)

    return time_us, speed


def clean_lfp(raw_lfp, fs_native=NATIVE_FS, fs_target=FS_LFP):
    """Resample -> notch -> detrend -> low-pass, same pipeline/order as
    ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean_v3.py's
    process_ncs_for_acg."""
    lfp = signal.resample_poly(raw_lfp, int(fs_target), int(fs_native))
    lfp = notch_filter(lfp, fs_target, LINE_HARMONICS, NOTCH_Q)
    lfp = detrend_signal(lfp, dtype='linear')
    lfp = lowpass_filter(lfp, fs_target, LOWPASS_CUTOFF_HZ, LOWPASS_ORDER)
    return lfp


# %% ==================== Theta filtering / peak-based freq & power =================

def filter_signal(lfp, fs, bandpass_cutoff, filter_order):
    # second-order sections, not (b, a): a narrow (3-7 Hz) order-5 band-pass at
    # 1 kHz has poles so close to the unit circle that the (b, a) polynomial
    # form rounds them outside it, and filtfilt then blows up to an all-NaN trace
    sos = signal.butter(filter_order, bandpass_cutoff, btype="band", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, lfp)
    return filtered


def trim_signal(sig, fs, trim_dur):
    trim_samples = int(trim_dur * fs)
    return sig[trim_samples:-trim_samples]


# ============================================================================
# Peak-trough interpolation (Dunn et al. 2022, Nature Communications
# 13:5905, the ferret theta paper) -- instantaneous frequency/power from
# linear interpolation of peak-to-peak and trough-to-trough intervals of the
# band-passed signal, rather than an analytic-signal (Hilbert/Generalized
# Phase) estimate. Ported verbatim from
# calculate_peak_trough_signal_parameters.m / findMinMax.m /
# clean_peaks_and_troughs.m (Toolbox/Signal-processing/,
# https://github.com/slsdunn/theta-paper-code).
# ============================================================================

def _find_min_max(sig, thresh_pc):
    """findMinMax.m: hysteresis-based local peak/trough detector -- a new
    peak is only confirmed once the signal has since fallen at least
    `thresh_pc` * median(|sig|) below it (and symmetrically for troughs), so
    small ripples on the way to the real extremum don't register as spurious
    extrema of their own.

    Returns
    -------
    max_point, min_point : (n, 2) arrays of [sample_index, value], in
        detection order.
    min_change : the amplitude range (thresh_pc * median(|sig|)) used.
    """
    n = len(sig)
    valid = ~np.isnan(sig)
    if not valid.any():
        return np.empty((0, 2)), np.empty((0, 2)), np.nan

    min_change = np.nanmedian(np.abs(sig)) * thresh_pc

    first_valid = np.flatnonzero(valid)[0]
    next_max_y = sig[first_valid] + min_change
    next_min_y = sig[first_valid] - min_change

    looking_for = 1  # 1 = next confirmation will be a trough (tracking a running local max) -- 0 = next confirmation will be a peak
    local_max_y = local_max_yx = None
    local_min_y = local_min_yx = None
    max_point, min_point = [], []

    for x in range(n):
        y = sig[x]
        if np.isnan(y):
            continue

        if y > next_max_y:
            if looking_for == 1:
                if local_min_yx is None:
                    looking_for = 0
                    continue
                min_point.append((local_min_yx, local_min_y))
            next_min_y = y - min_change
            next_max_y = y + min_change
            looking_for = 0
            local_max_y, local_max_yx = y, x

        if local_max_yx is not None and local_max_y <= y:
            local_max_y, local_max_yx = y, x

        if y < next_min_y:
            if looking_for == 0:
                if local_max_yx is None:
                    looking_for = 1
                    continue
                max_point.append((local_max_yx, local_max_y))
            next_max_y = y + min_change
            next_min_y = y - min_change
            looking_for = 1
            local_min_y, local_min_yx = y, x

        if local_min_yx is not None and local_min_y >= y:
            local_min_y, local_min_yx = y, x

    max_arr = np.array(max_point, dtype=np.float64).reshape(-1, 2)
    min_arr = np.array(min_point, dtype=np.float64).reshape(-1, 2)
    return max_arr, min_arr, min_change


def _clean_peaks_troughs(sig, peaks, troughs):
    """clean_peaks_and_troughs.m: merge peaks+troughs into one time-ordered
    extrema list, drop peaks below zero and troughs above zero (detection
    artifacts), then resolve any two consecutive same-type extrema -- either
    a spurious double-detection (the weaker of the pair is dropped) or a
    genuine missed opposite-type extremum in between (inserted, at the
    sample where the signal peaks/troughs within that span).

    Returns
    -------
    extrema : (n, 3) array of [sample_index, value, kind], sorted by sample
        index; kind is 1 for a peak, -1 for a trough.
    """
    kind = np.concatenate([np.ones(len(peaks)), -np.ones(len(troughs))])
    extrema = np.column_stack([
        np.concatenate([peaks[:, 0], troughs[:, 0]]),
        np.concatenate([peaks[:, 1], troughs[:, 1]]),
        kind,
    ])

    remove_negative_peaks = (extrema[:, 1] < 0) & (extrema[:, 2] == 1)
    remove_positive_troughs = (extrema[:, 1] > 0) & (extrema[:, 2] == -1)
    extrema = extrema[~(remove_negative_peaks | remove_positive_troughs)]
    extrema = extrema[np.argsort(extrema[:, 0], kind='stable')]

    while True:
        same_kind = np.flatnonzero(np.diff(extrema[:, 2]) == 0)
        if same_kind.size == 0:
            break
        i = int(same_kind[0])
        i0, i1 = int(extrema[i, 0]), int(extrema[i + 1, 0])
        seg = sig[i0:i1 + 1]

        if extrema[i, 2] == 1:  # two consecutive peaks
            missed_val = np.min(seg)
            missed_idx = i0 + int(np.argmin(seg))
            missed_kind = -1
            if missed_val > 0:  # no real trough between them -- spurious double peak
                drop = i if extrema[i, 1] < extrema[i + 1, 1] else i + 1
                extrema = np.delete(extrema, drop, axis=0)
                continue
        else:  # two consecutive troughs
            missed_val = np.max(seg)
            missed_idx = i0 + int(np.argmax(seg))
            missed_kind = 1
            if missed_val < 0:  # no real peak between them -- spurious double trough
                drop = i if extrema[i, 1] > extrema[i + 1, 1] else i + 1
                extrema = np.delete(extrema, drop, axis=0)
                continue

        extrema = np.insert(extrema, i + 1, [missed_idx, missed_val, missed_kind], axis=0)

    return extrema


def _interp1_extrap_nan(xp, fp, x):
    """MATLAB interp1 default (linear, no extrapolation): NaN outside
    [xp[0], xp[-1]]."""
    if len(xp) < 2:
        return np.full(len(x), np.nan)
    return np.interp(x, xp, fp, left=np.nan, right=np.nan)


def _calc_peak_trough_params(extrema, t, nan_mask):
    """calc_peak_trough_params (nested in
    calculate_peak_trough_signal_parameters.m): instantaneous frequency and
    power from peak-to-peak / trough-to-trough interpolation.

    Each peak-to-peak (and, independently, trough-to-trough) interval gives
    one frequency estimate (1 / interval duration), assigned at the sample
    midway between the two peaks (avoiding the lag a peak-locked estimate
    would otherwise have); linearly interpolating that across every sample
    of `t`, doing the same for troughs, and averaging the two gives the
    instantaneous frequency trace. Power is the squared amplitude envelope,
    where the envelope at each sample is the mean of the (linearly
    interpolated) peak and trough amplitude traces.
    """
    peaks = extrema[extrema[:, 2] == 1]
    troughs = extrema[extrema[:, 2] == -1]

    peak_idx, trough_idx = peaks[:, 0].astype(int), troughs[:, 0].astype(int)
    peak_t, trough_t = t[peak_idx], t[trough_idx]
    peak_f = 1.0 / np.diff(peak_t)
    trough_f = 1.0 / np.diff(trough_t)

    # midpoint sample between consecutive peaks/troughs, used as the time
    # coordinate for that interval's frequency estimate
    peak_mid_idx = np.round(peak_idx[:-1] + 0.5 * np.diff(peak_idx)).astype(int)
    trough_mid_idx = np.round(trough_idx[:-1] + 0.5 * np.diff(trough_idx)).astype(int)

    peak_f_t = _interp1_extrap_nan(t[peak_mid_idx], peak_f, t)
    trough_f_t = _interp1_extrap_nan(t[trough_mid_idx], trough_f, t)
    # samples outside the first/last interval midpoint are NaN in both traces;
    # nanmean correctly returns NaN there but emits a RuntimeWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        instantaneous_freq = np.nanmean(np.column_stack([peak_f_t, trough_f_t]), axis=1)

    peak_amp_t = _interp1_extrap_nan(peak_t, peaks[:, 1], t)
    trough_amp_t = _interp1_extrap_nan(trough_t, troughs[:, 1], t)
    peak_amp_t[nan_mask] = np.nan
    trough_amp_t[nan_mask] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        envelope = np.nanmean(np.column_stack([np.abs(peak_amp_t), np.abs(trough_amp_t)]), axis=1)
    instantaneous_power = envelope ** 2

    instantaneous_freq[nan_mask] = np.nan
    return instantaneous_freq, instantaneous_power


def estimate_instantaneous_frequency_power(theta, t, thresh_pc=PEAK_THRESH_PC):
    """Per-sample instantaneous theta frequency (Hz) and power (uV^2) of the
    band-passed signal `theta`, via the peak-trough interpolation method of
    Dunn et al. 2022 (see module header): local peaks/troughs are detected
    with a hysteresis threshold (`thresh_pc` * median(|theta|)), cleaned so
    they strictly alternate, and the frequency/amplitude of each peak-to-peak
    and trough-to-trough interval is linearly interpolated across every
    sample -- rather than read off an analytic-signal phase.
    """
    nan_mask = np.isnan(theta)
    max_point, min_point, _min_change = _find_min_max(theta, thresh_pc)
    extrema = _clean_peaks_troughs(theta, max_point, min_point)
    return _calc_peak_trough_params(extrema, t, nan_mask)


_BIN_AGG_FUNCS = {'median': np.nanmedian, 'mean': np.nanmean}


def bin_data(data, ts, bins, agg_method=BIN_AGG_METHOD):
    agg_func = _BIN_AGG_FUNCS[agg_method]
    bin_centers = (bins[:-1] + bins[1:])/2
    data_binned = []
    for i in range(len(bins)-1):
        bin_start = bins[i]
        bin_stop = bins[i+1]
        # speed values within the bin
        idx = np.where((ts>=bin_start)&(ts<=bin_stop))[0]
        # nan-aware: instantaneous_freq masks out biologically-impossible
        # (non-positive) samples as NaN -- a bin should fall back to
        # whichever of its samples are still valid rather than being
        # discarded outright over a few masked ones.
        # an all-NaN bin correctly yields NaN, but nanmedian/nanmean warn about it
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            data_binned.append(agg_func(data[idx]) if len(idx) else np.nan)
    return data_binned, bin_centers


def bin_any(mask, ts, bins):
    """Per bin of `bins` (same [start, stop] inclusive edges as bin_data):
    True if any sample of `mask` with time `ts` (sorted) falls in it."""
    cum = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
    lo = np.searchsorted(ts, bins[:-1], side='left')
    hi = np.searchsorted(ts, bins[1:], side='right')
    return (cum[hi] - cum[lo]) > 0


def drop_nan(inst_freq_binned, inst_power_binned, inst_speed_binned, bin_centers):
    idx_nan = np.where(np.isnan(inst_speed_binned))
    inst_freq_binned = np.delete(inst_freq_binned, idx_nan)
    inst_power_binned = np.delete(inst_power_binned, idx_nan)
    inst_speed_binned = np.delete(inst_speed_binned, idx_nan)
    bin_centers = np.delete(bin_centers, idx_nan)

    idx_nan = np.where(np.isnan(inst_power_binned))
    inst_freq_binned = np.delete(inst_freq_binned, idx_nan)
    inst_power_binned = np.delete(inst_power_binned, idx_nan)
    inst_speed_binned = np.delete(inst_speed_binned, idx_nan)
    bin_centers = np.delete(bin_centers, idx_nan)
    assert len(inst_freq_binned) == len(inst_power_binned) == len(inst_speed_binned) == len(bin_centers)
    return inst_freq_binned, inst_power_binned, inst_speed_binned, bin_centers


# %% ==================== Per-file / root-directory processing =====================

def process_ncs_session(fpath):
    """Load one .ncs file + its folder's tracking .csv, clean/preprocess the
    LFP (resample -> notch -> detrend -> low-pass -> theta band-pass), and
    return a DataFrame of binned instantaneous theta frequency/power vs.
    running speed for that session.

    With REJECT_SATURATED_BINS, time bins within SATURATION_MARGIN_S of an ADC
    clipped sample are removed (see module header).

    Returns (per-time-bin DataFrame, saturation summary dict).
    """
    raw_lfp, lfp_start_us, ad_bit_volts = load_ncs(fpath)
    # clipped samples are found on the raw native-rate trace, before any filtering
    input_range_uv = read_ncs_input_range(fpath)
    sat_native = saturated_samples(raw_lfp, input_range_uv, ad_bit_volts)
    lfp = clean_lfp(raw_lfp)
    near_clip = near_clip_mask(sat_native, len(lfp))
    if not REJECT_SATURATED_BINS:
        near_clip = np.zeros(len(lfp), dtype=bool)
    theta = filter_signal(lfp, FS_LFP, BANDPASS_CUTOFF, FILTER_ORDER)
    broadband = filter_signal(lfp, FS_LFP, REL_POWER_BAND, REL_POWER_FILTER_ORDER)

    lfp_ts = lfp_start_us / 1e6 + np.arange(len(theta)) / FS_LFP  # absolute UNIX time (s)

    theta = trim_signal(theta, FS_LFP, TRIM_DUR)
    lfp_ts = trim_signal(lfp_ts, FS_LFP, TRIM_DUR)
    near_clip = trim_signal(near_clip, FS_LFP, TRIM_DUR)
    # session's total REL_POWER_BAND power (uV^2): by Parseval, the mean square
    # of the band-passed trace equals the integral of its PSD over that band.
    # Samples within SATURATION_MARGIN_S of clipping are left out.
    broadband = trim_signal(broadband, FS_LFP, TRIM_DUR)
    total_power = np.mean(broadband[~near_clip] ** 2) if (~near_clip).any() else np.nan

    pos_path = find_position_file(fpath)
    if pos_path is None:
        raise ValueError('No tracking .csv found next to this file.')
    time_us, speed = compute_velocity_from_position(pos_path)
    speed_ts = time_us / 1e6  # absolute UNIX time (s), same clock as lfp_ts

    # restrict speed samples to the (trimmed) LFP epoch's own time window,
    # rather than assuming a fixed tracking frame rate
    in_window = (speed_ts >= lfp_ts.min()) & (speed_ts <= lfp_ts.max())
    inst_speed = speed[in_window]
    speed_ts = speed_ts[in_window]

    inst_freq, inst_power = estimate_instantaneous_frequency_power(theta, lfp_ts, PEAK_THRESH_PC)

    bin_start = lfp_ts.min().round()
    bin_stop = lfp_ts.max().round()
    bins = np.arange(bin_start, bin_stop + BINSIZE, step=BINSIZE)

    inst_speed_binned, _ = bin_data(inst_speed, speed_ts, bins)
    inst_freq_binned, _ = bin_data(inst_freq, lfp_ts, bins)
    inst_power_binned, bin_centers = bin_data(inst_power, lfp_ts, bins)
    inst_freq_binned = np.asarray(inst_freq_binned)
    inst_power_binned = np.asarray(inst_power_binned)
    inst_speed_binned = np.asarray(inst_speed_binned, dtype=np.float64)

    # acceleration (cm/s^2) of each time bin: central difference of the binned
    # speed, (v[i+1] - v[i-1]) / (2 * BINSIZE), one-sided at the two ends. Taken
    # here, on the full gap-free time-bin grid, so neighbours are the true
    # adjacent bins; a NaN neighbour (no tracking samples) gives NaN.
    inst_acc_binned = (np.gradient(inst_speed_binned, BINSIZE)
                       if len(inst_speed_binned) >= 2 else np.full(len(inst_speed_binned), np.nan))

    # remove time bins within SATURATION_MARGIN_S of clipping: NaN here, then
    # dropped by drop_nan with the other NaN bins
    bin_sat = bin_any(near_clip, lfp_ts, bins)
    n_valid_before = int((np.isfinite(inst_power_binned) & np.isfinite(inst_speed_binned)).sum())
    inst_freq_binned[bin_sat] = np.nan
    inst_power_binned[bin_sat] = np.nan

    # drop_nan (shared with the speed script) only knows freq/power/speed; the
    # acceleration is carried through by dropping the same bins (by bin centre)
    acc_by_center = pd.Series(inst_acc_binned, index=np.asarray(bin_centers))
    inst_freq_binned, inst_power_binned, inst_speed_binned, bin_centers = drop_nan(
        inst_freq_binned, inst_power_binned, inst_speed_binned, np.asarray(bin_centers))
    inst_acc_binned = acc_by_center.loc[bin_centers].to_numpy()

    n_clipped = int(sat_native.sum())
    sat_info = {
        "InputRange_uV": input_range_uv,
        "n_clipped_samples": n_clipped,
        "pct_samples_clipped": 100 * n_clipped / len(sat_native) if len(sat_native) else np.nan,
        "n_clipped_runs": len(find_runs(sat_native)[0]),
        "pct_time_near_clip": 100 * near_clip.mean() if len(near_clip) else np.nan,
        "n_bins_before": n_valid_before,
        "n_bins_removed_saturation": n_valid_before - len(bin_centers),
        "n_bins_kept": len(bin_centers),
    }

    out = pd.DataFrame({
        "Power": inst_power_binned,
        "Frequency": inst_freq_binned,
        "Speed": inst_speed_binned,
        "Acceleration": inst_acc_binned,
        "Time": bin_centers,
        "TotalPower": total_power,
    })
    # bins whose acceleration is undefined (a neighbouring bin has no speed)
    n_acc_nan = int(out["Acceleration"].isna().sum())
    sat_info["n_bins_dropped_no_acceleration"] = n_acc_nan
    out = out[out["Acceleration"].notna()].reset_index(drop=True)
    sat_info["n_bins_kept"] = len(out)
    return out, sat_info


def process_root_directory(root_dir):
    """Run process_ncs_session on every .ncs file to analyse under `root_dir`,
    and concatenate the per-bin results, tagging each row with a
    session id derived from the file's path (relative to `root_dir`), plus the
    recording folder it came from (channels in the same folder were recorded
    simultaneously) and the animal (first path component).

    With DROP_DUPLICATE_LFP, the files come from NCS_FILE_LIST_CSV (copies
    left out, see Make_NcsFileList.py); the left-out copies are written to
    OUTPUT_DIR. Without it, every .ncs under `root_dir` is used."""
    if DROP_DUPLICATE_LFP:
        ncs_files = listed_ncs_files(NCS_FILE_LIST_CSV, under=root_dir)
        dups = listed_copies(NCS_FILE_LIST_CSV, under=root_dir)
        dups['Recording'] = dups['abs_path'].map(
            lambda p: os.path.relpath(os.path.dirname(p), root_dir).replace(os.sep, '/'))
        dups[['Recording', 'file_name', 'copy_of']].to_excel(
            OUTPUT_DIR / 'Dropped_duplicate_ncs.xlsx', index=False)
    else:
        ncs_files = []
        for root, _dirs, files in os.walk(root_dir):
            for fname in files:
                if fname.endswith('.ncs'):
                    ncs_files.append(os.path.join(root, fname))
        ncs_files.sort(key=lambda p: (os.path.dirname(p), natural_key(p)))
        print(f'Found {len(ncs_files)} .ncs files under {root_dir}')

    all_results, sat_rows = [], []
    for fpath in ncs_files:
        rel = os.path.relpath(fpath, root_dir)
        ses_id = os.path.splitext(rel)[0].replace(os.sep, '_')
        rel_dir = os.path.dirname(rel)
        try:
            df_session, sat = process_ncs_session(fpath)
            df_session['Session'] = ses_id
            df_session['Recording'] = rel_dir.replace(os.sep, '/')
            df_session['Animal'] = rel_dir.split(os.sep)[0]
            if sat['InputRange_uV'] is None:
                sat_msg = 'no -InputRange in header, saturation check skipped'
            else:
                sat_msg = (f'InputRange {sat["InputRange_uV"]:g} uV, '
                           f'{sat["pct_samples_clipped"]:.3f}% samples clipped, '
                           f'{sat["n_bins_removed_saturation"]}/{sat["n_bins_before"]} bins removed '
                           f'within {SATURATION_MARGIN_S:g} s of clipping')
            print(f'  OK: {rel}  [{len(df_session)} bins; {sat_msg}]')
            all_results.append(df_session)
            sat_rows.append({'Session': ses_id, 'Animal': rel_dir.split(os.sep)[0],
                             'Recording': rel_dir.replace(os.sep, '/'),
                             'file_name': os.path.basename(fpath), **sat})
        except Exception as e:
            print(f'  SKIP: {rel} -- {e}')

    print(f'-> {len(all_results)} files processed\n')
    if not all_results:
        raise ValueError(f'No files processed successfully under {root_dir}')

    sat_df = pd.DataFrame(sat_rows)
    sat_df.to_excel(OUTPUT_DIR / 'Saturation_summary.xlsx', index=False)
    n_before, n_removed = sat_df['n_bins_before'].sum(), sat_df['n_bins_removed_saturation'].sum()
    print(f'ADC saturation: {int((sat_df["n_clipped_samples"] > 0).sum())}/{len(sat_df)} files contain '
          f'clipped samples; {n_removed}/{n_before} time bins ({100 * n_removed / max(n_before, 1):.2f}%) '
          f'removed within {SATURATION_MARGIN_S:g} s of clipping'
          + ('' if REJECT_SATURATED_BINS else ' [REJECT_SATURATED_BINS = False: nothing removed]'))
    return pd.concat(all_results, ignore_index=True)


def add_normalized_power(df, group="Session"):
    """Add the two normalized theta-power columns (see POWER_MEASURES) to the
    per-time-bin rows; `Power` itself is left in uV^2.

    PowerdB  = 10*log10(Power / B), B = the session's BIN_AGG_METHOD Power over
               time bins with speed in DB_BASELINE_SPEED_CMS. Normalizing per
               session turns each recording's multiplicative gain (electrode
               placement/impedance) into 0 dB at the baseline speed, so
               sessions are compared on change rather than absolute power.
               The median of dB values equals the dB of the median, so the
               later speed-bin aggregation is unaffected by the order.
    PowerRel = 100 * (Power / 2) / TotalPower. Power is the squared peak/trough
               amplitude; a sinusoid of amplitude A has mean square A^2/2,
               so Power/2 is the theta mean square that TotalPower (mean
               square of the REL_POWER_BAND trace) is directly comparable to --
               i.e. the theta-band integral of Fig4a_PSD's normalized PSD.

    Sessions without any baseline-speed bins get NaN PowerdB (and are
    dropped from the dB fits only).
    """
    d = df.copy()
    lo, hi = DB_BASELINE_SPEED_CMS
    in_baseline = (d["Speed"] >= lo) & (d["Speed"] < hi)
    baseline = d[in_baseline].groupby(group)["Power"].agg(BIN_AGG_METHOD)
    d["PowerBaseline"] = d[group].map(baseline)
    with np.errstate(divide='ignore', invalid='ignore'):
        d["PowerdB"] = 10 * np.log10(d["Power"] / d["PowerBaseline"])
    d["PowerdB"] = d["PowerdB"].replace([np.inf, -np.inf], np.nan)
    d["PowerRel"] = 100 * (d["Power"] / 2) / d["TotalPower"]

    n_missing = d.loc[d["PowerBaseline"].isna(), group].nunique()
    if n_missing:
        print(f'  {n_missing} sessions have no {lo:g}-{hi:g} cm/s bins -- no dB baseline, '
              f'excluded from the dB fits')
    return d


def save_baseline_coverage(df, group="Session"):
    """Per session: seconds of running in the dB baseline window
    (DB_BASELINE_SPEED_CMS), in the v8 window (2-5 cm/s) for comparison, below
    MIN_SPEED_CMS and in total, plus the baseline power -> dB_baseline_coverage.xlsx."""
    def secs(mask):
        return df[mask].groupby(group).size() * BINSIZE
    lo, hi = DB_BASELINE_SPEED_CMS
    s = df["Speed"]
    cov = pd.DataFrame({
        f"s_{lo:g}_{hi:g}cms (baseline)": secs((s >= lo) & (s < hi)),
        "s_2_5cms (v8 baseline)":          secs((s >= 2) & (s < 5)),
        f"s_below_{MIN_SPEED_CMS:g}cms":   secs(s < MIN_SPEED_CMS),
        "s_total":                         secs(s.notna()),
    }).fillna(0)
    cov["PowerBaseline_uV2"] = df.groupby(group)["PowerBaseline"].first()
    meta = df.drop_duplicates(group).set_index(group)[["Animal", "Recording"]]
    cov = meta.join(cov).reset_index()
    cov.to_excel(OUTPUT_DIR / "dB_baseline_coverage.xlsx", index=False)
    b = cov[f"s_{lo:g}_{hi:g}cms (baseline)"]
    print(f'  dB baseline {lo:g}-{hi:g} cm/s: median {b.median():.0f} s per session '
          f'(min {b.min():.0f} s); v8 2-5 cm/s: median {cov["s_2_5cms (v8 baseline)"].median():.0f} s')
    return cov


# %% ==================== Analysis labels / table output ===========================
#
# Every analysis below runs on the combined data (LAB_ALL, written to OUTPUT_DIR
# as .xlsx) and again on the acceleration-only and deceleration-only time bins
# (SPLIT_LABELS, see run_acc_decel_split; written to OUTPUT_DIR/Acc and
# OUTPUT_DIR/Decel as .csv). A label dict carries the text that differs:
#   name  : quantity in titles ('Acceleration' / 'Deceleration')
#   xname : x-axis quantity ('Acceleration' / '|Deceleration|')
#   short : short model label in the speed-controlled figure
#   note  : extra line under the figures describing the subset ('' = none)
#   fmt   : 'xlsx' (one workbook, a sheet per table) or 'csv' (one file per table)
LAB_ALL = dict(key='All', name='Acceleration', xname='Acceleration', short='Acc', note='', fmt='xlsx')


def save_tables(out_dir, stem, sheets, fmt='xlsx'):
    """Write `sheets` ({sheet name: DataFrame}) as <stem>.xlsx (one sheet each;
    a single table goes to the default sheet) or, with fmt='csv', as
    <stem>.csv (single table) / <stem>_<sheet>.csv (several)."""
    out_dir = Path(out_dir)
    if fmt == 'csv':
        for name, t in sheets.items():
            fname = f'{stem}.csv' if len(sheets) == 1 else f'{stem}_{name}.csv'
            t.to_csv(out_dir / fname, index=False, encoding='utf-8-sig')
    elif len(sheets) == 1:
        next(iter(sheets.values())).to_excel(out_dir / f'{stem}.xlsx', index=False)
    else:
        with pd.ExcelWriter(out_dir / f'{stem}.xlsx') as xw:
            for name, t in sheets.items():
                t.to_excel(xw, sheet_name=name, index=False)


# %% ==================== Acceleration vs. theta statistics (mixed linear model) ==========

def acc_bin_index(acc, width=ACC_BIN_WIDTH):
    """Acceleration-bin index, bins centred on 0: k = floor(a / w + 0.5), i.e.
    bin k covers [(k - 0.5) w, (k + 0.5) w) and has centre k * w."""
    return np.floor(np.asarray(acc, dtype=np.float64) / width + 0.5)


def speed_bin_index(speed, width=SPEED_BIN_WIDTH_CMS):
    """Speed-bin index, bins anchored at 0 as in the speed script: [k w, (k+1) w)."""
    return np.floor(np.asarray(speed, dtype=np.float64) / width)


def aggregate_by_acc_bin(df, acc_bin_width=ACC_BIN_WIDTH, group="Session",
                         agg_method=BIN_AGG_METHOD):
    """Representative (median or mean, per agg_method) Frequency/Power/
    Acceleration/Speed per (session, acceleration bin).

    Reduces the raw per-time-bin rows (one every BINSIZE seconds -- thousands
    per session) to one row per `acc_bin_width` cm/s^2 acceleration bin
    actually visited within each session, so the mixed model is fit on a value
    that summarises a distinct acceleration rather than on many highly
    autocorrelated time samples. `Acceleration` (and `Speed`) in the output
    are the agg_method'd values of the bin's contributing samples;
    `AccBinCenter` is the bin's center (k * w, bins centred on 0).
    """
    d = df.copy()
    d["AccBin"] = acc_bin_index(d["Acceleration"], acc_bin_width)
    # Recording/Animal are constant within a session -- carried through for plotting
    carry = [c for c in ("Recording", "Animal") if c in d.columns]
    values = [c for c in ("Frequency", *(m[0] for m in POWER_MEASURES)) if c in d.columns]
    agg = (d.groupby([group, *carry, "AccBin"])
            .agg(**{c: (c, agg_method) for c in values},
                 Acceleration=("Acceleration", agg_method),
                 Speed=("Speed", agg_method),
                 n=("Acceleration", "size"))
            .reset_index(drop=False))
    agg["AccBinCenter"] = agg["AccBin"] * acc_bin_width
    return agg.drop(columns="AccBin")


def fit_mixed_model(df, response, predictor="Speed", group="Session"):
    """
    Random-intercept linear mixed model: response ~ predictor, grouped by
    session, so repeated bins from the same session don't count as
    independent observations.
    """
    model = smf.mixedlm(f"{response} ~ {predictor}", df, groups=df[group])
    return model.fit()


def _axes_y_below(fig, ax, artist=None, pad=0.03):
    """y (in `ax` axes coords) just below `artist` -- or, if None, below
    everything already drawn on `ax` (tick labels, xlabel) -- so the next
    element placed there can't overlap it."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = artist.get_window_extent(renderer) if artist is not None else ax.get_tightbbox(renderer)
    return ax.transAxes.inverted().transform((0, bb.y0))[1] - pad


def add_details_below(fig, ax, text, legend_handles=None, legend_ncol=3, fontsize=9):
    """Put the legend (optional) and then the stats `text` underneath the
    axes, stacked so neither overlaps the data, the axis labels or each
    other. Save with bbox_inches='tight' so they're kept in the image."""
    anchor = None
    if legend_handles:
        anchor = ax.legend(handles=legend_handles, loc='upper center',
                           bbox_to_anchor=(0.5, _axes_y_below(fig, ax)),
                           ncol=legend_ncol, fontsize=fontsize, frameon=False)
    ax.text(0.5, _axes_y_below(fig, ax, anchor), text, transform=ax.transAxes,
            ha='center', va='top', fontsize=fontsize, linespacing=1.4)


def plot_mixed_model_fit(df, result, response, predictor, ylabel, title, color, out_path,
                         xlabel=f"Acceleration bin center ({ACC_UNIT})", group="Session",
                         df_excluded=None, note=''):
    """Scatter the binned data plus the mixed model's fixed-effect fit and its 95% CI band.
    Points are coloured by animal (ANIMAL_COLORS) when `df` has an Animal
    column, otherwise all drawn in `color`. `df_excluded` (acceleration bins
    of immobile time bins, speed < MIN_SPEED_CMS, not in the fit) is drawn in
    EXCLUDED_COLOR. The fit line covers only the fitted acceleration range.
    Legend and all model statistics are written below the plot."""
    intercept = result.params["Intercept"]
    slope = result.params[predictor]
    pvalue = result.pvalues[predictor]
    slope_se = result.bse[predictor]
    slope_z = result.tvalues[predictor]
    slope_ci_low, slope_ci_high = result.conf_int().loc[predictor]
    cov = result.cov_params()

    x_range = np.linspace(df[predictor].min(), df[predictor].max(), 100)
    y_pred = intercept + slope * x_range
    se = np.sqrt(
        cov.loc["Intercept", "Intercept"]
        + x_range**2 * cov.loc[predictor, predictor]
        + 2 * x_range * cov.loc["Intercept", predictor]
    )
    z = norm.ppf(0.975)
    y_ci_low = y_pred - z * se
    y_ci_high = y_pred + z * se

    fig, ax = plt.subplots(figsize=(7, 5))
    handles = []
    n_excluded = 0 if df_excluded is None else len(df_excluded)
    if n_excluded:
        handles.append(ax.scatter(df_excluded[predictor], df_excluded[response], color=EXCLUDED_COLOR,
                                  edgecolor='#707070', linewidth=0.5, alpha=0.8, s=25,
                                  label=f"speed < {MIN_SPEED_CMS:g} cm/s (not in fit)"))
    ax.axvline(0, color='#707070', linestyle=':', linewidth=1)
    if "Animal" in df.columns:
        for a in ordered_animals(df["Animal"]):
            sub = df[df["Animal"] == a]
            handles.append(ax.scatter(sub[predictor], sub[response], color=animal_color(a, default=color),
                                      edgecolor='k', linewidth=0.5, alpha=0.8, s=25, label=a))
    else:
        handles.append(ax.scatter(df[predictor], df[response], color=color, edgecolor='k',
                                  alpha=0.6, s=25, label="Acceleration bins"))
    handles += ax.plot(x_range, y_pred, color="black", linewidth=2, label="Mixed Linear Fit")
    handles.append(ax.fill_between(x_range, y_ci_low, y_ci_high, color="grey", alpha=0.5, label="95% CI"))
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

    n_animals = df["Animal"].nunique() if "Animal" in df.columns else np.nan
    details = "\n".join([
        f"Model: {response} ~ {predictor} + (1 | {group})   [linear mixed model, REML]",
        f"Slope β = {slope:.4g} ± {slope_se:.3g} SE,  95% CI [{slope_ci_low:.4g}, {slope_ci_high:.4g}],  "
        f"z = {slope_z:.2f},  " + (f"p = {pvalue:.3g}" if pvalue > 0 else "p < 1e-300"),
        f"Intercept = {intercept:.4g};  random-intercept var = {result.cov_re.iloc[0, 0]:.3g};  "
        f"residual var = {result.scale:.3g};  converged = {result.converged}",
        f"n = {result.nobs} acceleration bins from {df[group].nunique()} sessions, {n_animals} animals;  "
        f"bin width = {ACC_BIN_WIDTH:g} {ACC_UNIT} (centred on 0),  value per bin = {BIN_AGG_METHOD}",
        f"Acceleration = central difference of the {BINSIZE:g} s binned speed;  "
        f"only time bins with speed ≥ {MIN_SPEED_CMS:g} cm/s fitted",
        f"Gray: {n_excluded} acceleration bins of immobile time bins (speed < {MIN_SPEED_CMS:g} cm/s), "
        f"shown but excluded from the fit",
        *([note] if note else []),
    ])
    add_details_below(fig, ax, details, legend_handles=handles, legend_ncol=4 if n_excluded else 3)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


def run_acc_vs_theta_stats(df, out_dir=OUTPUT_DIR, lab=LAB_ALL):
    """Fit and plot mixed linear models of Frequency~Acceleration and, for each
    POWER_MEASURES column, Power~Acceleration (random intercept per Session),
    on ACC_BIN_WIDTH-wide acceleration bins: per (session, bin) the
    representative (BIN_AGG_METHOD) Frequency/Power, against the bin's center
    acceleration -- rather than the raw per-time-bin rows (see
    aggregate_by_acc_bin). Only mobile time bins (speed >= MIN_SPEED_CMS) are
    fitted. Outputs go to `out_dir`; `lab` sets the labels / table format (see
    LAB_ALL). Returns {measure: fitted result}."""
    out_dir = Path(out_dir)
    name, xlabel = lab['name'], f"{lab['xname']} bin center ({ACC_UNIT})"
    df_clean = df.dropna(subset=["Frequency", "Power", "Speed", "Acceleration"])
    # mobile and immobile time bins are aggregated separately; only the
    # mobile rows enter the fits, the immobile ones are drawn in gray
    parts = []
    for mobile, d in df_clean.groupby("Mobile"):
        parts.append(aggregate_by_acc_bin(d).assign(InAnalysis=bool(mobile)))
    df_all = pd.concat(parts, ignore_index=True)
    # the model's Acceleration predictor is the bin center; the within-bin
    # representative acceleration/speed are kept for reference only
    df_all = df_all.rename(columns={"Acceleration": "AccWithinBin", "Speed": "SpeedWithinBin",
                                    "AccBinCenter": "Acceleration"})
    df_all = df_all.sort_values(["Session", "InAnalysis", "Acceleration"]).reset_index(drop=True)
    save_tables(out_dir, "ThetaVsAcc_binned_medians", {"binned": df_all}, lab['fmt'])
    df_stats = df_all[df_all["InAnalysis"]]
    df_excl = df_all[~df_all["InAnalysis"]]

    results = {}
    freq_result = fit_mixed_model(df_stats, "Frequency", predictor="Acceleration")
    results["Frequency"] = freq_result
    print(freq_result.summary())
    plot_mixed_model_fit(
        df_stats, freq_result, "Frequency", "Acceleration",
        ylabel="Theta Frequency [Hz]",
        title=f"Theta Frequency vs. {name} (Mixed Model)",
        color="red", out_path=out_dir / "ThetaFrequency_vs_Acc_MixedModel.png",
        xlabel=xlabel, df_excluded=df_excl, note=lab['note'])

    for col, suffix, ylabel, title, _unit in POWER_MEASURES:
        d = df_stats.dropna(subset=[col])
        results[col] = fit_mixed_model(d, col, predictor="Acceleration")
        print(results[col].summary())
        plot_mixed_model_fit(
            d, results[col], col, "Acceleration",
            ylabel=ylabel, title=f"{title.replace('Acceleration', name)}, Mixed Model",
            color="royalblue", out_path=out_dir / f"ThetaPower{suffix}_vs_Acc_MixedModel.png",
            xlabel=xlabel, df_excluded=df_excl.dropna(subset=[col]), note=lab['note'])

    with open(out_dir / "MixedModel_Stats_Summary.txt", "w", encoding="utf-8") as f:
        if lab['note']:
            f.write(lab['note'] + "\n\n")
        f.write(f"=== Theta Frequency vs {name} ===\n")
        f.write(str(freq_result.summary()))
        for col, _suffix, _ylabel, title, _unit in POWER_MEASURES:
            f.write(f"\n\n=== {title.replace('Acceleration', name)} [{col}] ===\n")
            f.write(str(results[col].summary()))

    return results


# %% ==================== Acceleration vs. theta, controlling for speed ==========
#
# Acceleration and speed are not independent (an animal can only decelerate
# from a high speed, and speeding up happens mostly at low speeds), so an
# acceleration effect in the models above could be carried by speed. Here
# both enter one random-intercept model,
#     response ~ Acceleration + Speed + (1 | Session),
# fit on 2-D bins: per session, the BIN_AGG_METHOD value of each
# (SPEED_BIN_WIDTH_CMS speed bin x ACC_BIN_WIDTH acceleration bin) cell that
# the session visited, with the two bin centres as predictors -- the same
# bin-then-fit approach as the acceleration-only model, so thousands of
# autocorrelated time bins don't inflate the significance. Mobile time bins only.

def aggregate_by_speed_acc_bin(df, group="Session", agg_method=BIN_AGG_METHOD):
    """One row per (session, speed bin, acceleration bin) cell: agg_method'd
    Frequency/Power measures, the number of time bins, and the cell's speed
    (anchored at 0) and acceleration (centred on 0) bin centres."""
    d = df.copy()
    d["SpeedBin"] = speed_bin_index(d["Speed"])
    d["AccBin"] = acc_bin_index(d["Acceleration"])
    carry = [c for c in ("Recording", "Animal") if c in d.columns]
    values = [c for c in ("Frequency", *(m[0] for m in POWER_MEASURES)) if c in d.columns]
    agg = (d.groupby([group, *carry, "SpeedBin", "AccBin"])
            .agg(**{c: (c, agg_method) for c in values},
                 SpeedWithinBin=("Speed", agg_method),
                 AccWithinBin=("Acceleration", agg_method),
                 n=("Acceleration", "size"))
            .reset_index(drop=False))
    agg["Speed"] = (agg["SpeedBin"] + 0.5) * SPEED_BIN_WIDTH_CMS
    agg["Acceleration"] = agg["AccBin"] * ACC_BIN_WIDTH
    return agg.drop(columns=["SpeedBin", "AccBin"])


def _coef_row(result, name):
    lo, hi = result.conf_int().loc[name]
    return {"beta": result.params[name], "se": result.bse[name], "ci_low": lo, "ci_high": hi,
            "z": result.tvalues[name], "pvalue": result.pvalues[name]}


def plot_speed_controlled(rows, out_path, lab=LAB_ALL):
    """Per measure: the acceleration slope (95% CI) of the acceleration-only
    model and of the Acceleration + Speed model, side by side."""
    measures = rows["Measure"].tolist()
    short = lab['short']
    fig, axes = plt.subplots(1, len(measures), figsize=(3.6 * len(measures), 3.4), squeeze=False)
    for ax, (_, r) in zip(axes[0], rows.iterrows()):
        models = [(f"{short} only", "acc_only"), (f"{short} + Speed", "acc_ctrl")]
        for y, (label, key) in enumerate(models):
            b, lo, hi, p = (r[f"{key}_beta"], r[f"{key}_ci_low"], r[f"{key}_ci_high"],
                            r[f"{key}_pvalue"])
            color = 'black' if p < 0.05 else '0.55'
            ax.errorbar(b, y, xerr=[[b - lo], [hi - b]], fmt='o', color=color, ms=7, capsize=4, lw=1.6)
            ax.annotate(f"p = {p:.3g}", (b, y), xytext=(0, 9), textcoords='offset points',
                        ha='center', fontsize=8, color=color)
        ax.axvline(0, color='grey', ls='--', lw=1)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([m[0] for m in models], fontsize=9)
        ax.set_ylim(-0.7, len(models) - 0.3)
        ax.set_title(r["Measure"], fontsize=11, weight='bold')
        ax.set_xlabel(f"{lab['xname']} β ({r['Unit']} per {ACC_UNIT})", fontsize=9)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 4), useMathText=True)
        ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    details = (f"{short} only: response ~ {lab['xname']} + (1 | Session) on {ACC_BIN_WIDTH:g} {ACC_UNIT} "
               f"bins.   {short} + Speed: response ~ {lab['xname']} + Speed + (1 | Session) on "
               f"{SPEED_BIN_WIDTH_CMS:g} cm/s × {ACC_BIN_WIDTH:g} {ACC_UNIT} bins per session.\n"
               f"Speed ≥ {MIN_SPEED_CMS:g} cm/s only; 95% CI; black = p < 0.05."
               + (f"\n{lab['note']}" if lab['note'] else ""))
    fig.suptitle(f"Theta vs. {lab['name'].lower()}, with and without controlling for speed",
                 fontsize=12, weight='bold')
    fig.tight_layout()
    fig.text(0.5, -0.02, details, ha='center', va='top', fontsize=8.5, linespacing=1.4)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


def run_speed_controlled_model(df, acc_only_results, out_dir=OUTPUT_DIR, lab=LAB_ALL):
    """Fit response ~ Acceleration + Speed + (1 | Session) on 2-D speed x
    acceleration bins for Frequency and every POWER_MEASURES column; compare
    the acceleration slope with the acceleration-only model
    (`acc_only_results`, from run_acc_vs_theta_stats). Outputs go to `out_dir`."""
    out_dir = Path(out_dir)
    d = df.dropna(subset=["Frequency", "Power", "Speed", "Acceleration"])
    d = d[d["Mobile"]]
    r_time = np.corrcoef(d["Acceleration"], d["Speed"])[0, 1]
    df2 = aggregate_by_speed_acc_bin(d)
    df2 = df2.sort_values(["Session", "Speed", "Acceleration"]).reset_index(drop=True)
    save_tables(out_dir, "SpeedControlled_speed_x_acc_binned_medians", {"binned": df2}, lab['fmt'])
    r_cells = np.corrcoef(df2["Acceleration"], df2["Speed"])[0, 1]

    units = {"Frequency": "Hz", **{m[0]: m[4] for m in POWER_MEASURES}}
    rows, summaries = [], []
    for col, unit in units.items():
        dd = df2.dropna(subset=[col])
        res = smf.mixedlm(f"{col} ~ Acceleration + Speed", dd, groups=dd["Session"]).fit()
        print(res.summary())
        summaries.append(f"\n\n=== {col} ~ Acceleration + Speed + (1 | Session) ===\n{res.summary()}")
        acc, spd = _coef_row(res, "Acceleration"), _coef_row(res, "Speed")
        only = _coef_row(acc_only_results[col], "Acceleration")
        row = {"Measure": col, "Unit": unit,
               "n_cells": int(res.nobs), "n_sessions": dd["Session"].nunique(),
               "converged": res.converged}
        row.update({f"acc_ctrl_{k}": v for k, v in acc.items()})
        row.update({f"speed_ctrl_{k}": v for k, v in spd.items()})
        row.update({f"acc_only_{k}": v for k, v in only.items()})
        rows.append(row)
    rows = pd.DataFrame(rows)

    save_tables(out_dir, "SpeedControlled_MixedModel", {
        "coefficients": rows,
        "acc_speed_correlation": pd.DataFrame([{"r_acc_speed_time_bins": r_time, "n_time_bins": len(d),
                                                "r_acc_speed_bin_cells": r_cells, "n_bin_cells": len(df2)}]),
    }, lab['fmt'])
    plot_speed_controlled(rows, out_dir / "SpeedControlled_AccSlopes.png", lab)

    header = (f"{lab['xname']} vs speed (mobile time bins): Pearson r = {r_time:.3f} over {len(d)} "
              f"{BINSIZE:g} s time bins; r = {r_cells:.3f} over {len(df2)} speed x acceleration cells.\n")
    if lab['note']:
        header = lab['note'] + "\n" + header
    with open(out_dir / "SpeedControlled_MixedModel_Summary.txt", "w", encoding="utf-8") as f:
        f.write(header)
        for _, r in rows.iterrows():
            f.write(f"{r['Measure']}: acc-only beta = {r['acc_only_beta']:.4g} (p = {r['acc_only_pvalue']:.3g}); "
                    f"acc | speed beta = {r['acc_ctrl_beta']:.4g} (p = {r['acc_ctrl_pvalue']:.3g}); "
                    f"speed | acc beta = {r['speed_ctrl_beta']:.4g} (p = {r['speed_ctrl_pvalue']:.3g})"
                    f"  [{r['Unit']} per {ACC_UNIT} / per cm/s]\n")
        f.writelines(summaries)
    print(header)
    return rows


# %% ==================== Speed vs. theta statistics: per-session OLS (Dunn et al. Fig. 3 method) ==========
#
# Dunn et al. 2022 (Nat Commun 13:5905) did NOT use a mixed-effects model
# (lme4/stargazer) for the theta-frequency/power vs. speed relationship shown in
# their Fig. 3c,d,i,j and the "Speed vs freq./pow." columns of Fig. 3f,l. That LMM
# machinery (Stats/figure4_LMM.R, figure5_LMM.R, figure6_LMM.R in
# https://github.com/slsdunn/theta-paper-code) was used for a *different* analysis:
# their autocorrelation-based "peak range" theta-regularity metric vs. movement
# state / atropine / trial epoch (their Figs. 4-6). For the continuous
# speed-vs-frequency/power relationship, the Methods/legend text describes a plain
# per-session (per-channel) ordinary least-squares regression fit to speed-binned
# median frequency/power values ("Regression line fitted to median values ... in
# each speed bin"), summarised across sessions/channels as mean +/- SD of the fitted
# slope (beta_1) -- e.g. "rat: 0.026 +/- 0.007, ferret: 0.010 +/- 0.003" (main text)
# -- with individual channels/sessions called significant via a Bonferroni-corrected
# p-value (their Fig. 3 legend: p < 0.0016 = 0.05/32 channels).
#
# The MATLAB helper that actually performs this fit
# (linear_fit_of_table_or_struct_vars.m) is referenced by their plotting code
# (Paper-figure-plotting/supfigure2_depth_profiles.m) but is not included in the
# public repository -- only the calling/plotting code is. The implementation below
# follows the Methods text and figure legends directly, as a paper-faithful
# alternative to run_acc_vs_theta_stats's mixed model above (that mixed model
# pools all sessions into one fit with session as a random intercept and a single
# shared slope -- a different, not wrong, question from the paper's per-session
# slope distribution).

# Acceleration bins of the same ACC_BIN_WIDTH as the mixed model. Unlike the
# speed script (which, following the paper, dropped speeds above each
# session's 90th percentile), no percentile cut is applied: every mobile
# acceleration bin is kept.
FIG3_ACC_BIN_WIDTH = ACC_BIN_WIDTH


def fig3_trim_and_bin(df, min_bin_s=0.0):
    """The per-session OLS input: mobile time bins (speed >= MIN_SPEED_CMS) of
    each session, and their FIG3_ACC_BIN_WIDTH acceleration-bin medians. The
    OLS predictor is each bin's median acceleration (`Acceleration`), as the
    speed script used each speed bin's median speed.

    min_bin_s    : acceleration bins (per session) with less than this much
                   data (n time bins * BINSIZE seconds) are dropped, together
                   with their time bins; 0 = keep every visited bin.
    Returns (kept time-bin rows, acceleration-bin rows)."""
    df_clean = df.dropna(subset=["Frequency", "Power", "Speed", "Acceleration"]).copy()
    df_clean = df_clean[df_clean["Mobile"]]

    if min_bin_s > 0:
        acc_bin = acc_bin_index(df_clean["Acceleration"], FIG3_ACC_BIN_WIDTH)
        n_in_bin = df_clean.groupby([df_clean["Session"], acc_bin])["Acceleration"].transform("size")
        df_clean = df_clean[n_in_bin * BINSIZE >= min_bin_s]

    df_binned = aggregate_by_acc_bin(df_clean, acc_bin_width=FIG3_ACC_BIN_WIDTH)
    return df_clean, df_binned


def fit_per_session_ols(df, response, predictor="Acceleration", group="Session"):
    """Per-session ordinary least-squares regression of `response` on `predictor`,
    one independent fit per session (no pooling), matching Dunn et al.'s Fig. 3
    approach. Returns one row per session with slope (beta1), intercept, R^2,
    p-value and the number of acceleration bins the fit used.
    """
    rows = []
    for ses_id, g in df.groupby(group):
        g = g.dropna(subset=[predictor, response])
        if len(g) < 3:
            continue
        slope, intercept, r_value, p_value, std_err = linregress(g[predictor], g[response])
        rows.append({
            group: ses_id,
            "beta1": slope,
            "beta1_se": std_err,
            "intercept": intercept,
            "r": r_value,
            "r2": r_value ** 2,
            "pvalue": p_value,
            "n_bins": len(g),
        })
    return pd.DataFrame(rows)


def summarise_slopes_across_recordings(fits, meta, group="Session"):
    """Group-level test of the per-session OLS slopes (two-stage /
    summary-statistics approach): each recording's fitted beta1 is one data
    point, tested against 0 across recordings.

    Channels in the same recording folder share the same running/acceleration trace
    and aren't independent, so their slopes are averaged first to one beta1
    per recording. The recording-level slopes are then tested with a
    one-sample t-test and a Wilcoxon signed-rank test (H0: mean/median slope
    = 0). Per-animal mean slopes are returned too, since recordings from the
    same animal aren't fully independent either.
    """
    fits = fits.merge(meta, on=group, how="left")
    per_rec = (fits.groupby(["Animal", "Recording"], as_index=False)
                   .agg(beta1=("beta1", "mean"), n_channels=("beta1", "size")))
    per_animal = (per_rec.groupby("Animal", as_index=False)
                     .agg(beta1_mean=("beta1", "mean"), beta1_sd=("beta1", "std"),
                          n_recordings=("beta1", "size")))
    per_animal = (per_animal.set_index("Animal")
                     .loc[ordered_animals(per_animal["Animal"])].reset_index())

    slopes = per_rec["beta1"].to_numpy()
    n = len(slopes)
    t_res = ttest_1samp(slopes, 0.0) if n >= 2 else None
    try:
        w_res = wilcoxon(slopes)
    except ValueError:  # n too small or all slopes exactly 0
        w_res = None

    stats = {
        "n_recordings": n,
        "n_animals": per_rec["Animal"].nunique(),
        "beta1_mean": slopes.mean() if n else np.nan,
        "beta1_sem": slopes.std(ddof=1) / np.sqrt(n) if n >= 2 else np.nan,
        "t": t_res.statistic if t_res else np.nan,
        "t_pvalue": t_res.pvalue if t_res else np.nan,
        "wilcoxon_W": w_res.statistic if w_res else np.nan,
        "wilcoxon_pvalue": w_res.pvalue if w_res else np.nan,
        "n_positive": int((slopes > 0).sum()),
    }
    return stats, per_rec, per_animal


def plot_recording_slopes(per_rec, grp, ylabel, title, out_path, note=''):
    """One dot per recording (its OLS slope, channels averaged), grouped and
    coloured by animal (ANIMAL_COLORS), with each animal's mean as a black bar
    and the group-level test result annotated."""
    rng = np.random.default_rng(0)
    animals_here = ordered_animals(per_rec["Animal"])

    fig, ax = plt.subplots(figsize=(1.4 * len(animals_here) + 2, 4.5))
    for i, a in enumerate(animals_here):
        vals = per_rec.loc[per_rec["Animal"] == a, "beta1"].to_numpy()
        x = i + rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(x, vals, s=45, color=animal_color(a), edgecolor='k', linewidth=0.6, zorder=3)
        ax.hlines(vals.mean(), i - 0.25, i + 0.25, color='black', lw=2, zorder=4)
    ax.axhline(0, color='grey', ls='--', lw=1)
    ax.set_xticks(range(len(animals_here)))
    ax.set_xticklabels(animals_here)
    ax.set_xlim(-0.6, len(animals_here) - 0.4)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.set_title(title, fontsize=13, weight='bold')
    # all stats go below the plot, so they can't overlap the title or hide points
    details = "\n".join([
        f"Mean β₁ = {grp['beta1_mean']:.3g} ± {grp['beta1_sem']:.2g} SEM",
        f"One-sample t({grp['n_recordings'] - 1}) = {grp['t']:.2f},  p = {grp['t_pvalue']:.3g}",
        f"Wilcoxon W = {grp['wilcoxon_W']:.4g},  p = {grp['wilcoxon_pvalue']:.3g}",
        f"{grp['n_positive']}/{grp['n_recordings']} recordings with positive slope",
        f"n = {grp['n_recordings']} recordings, {grp['n_animals']} animals",
        "Black bars: per-animal mean",
        *([note] if note else []),
    ])
    add_details_below(fig, ax, details)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


def run_acc_vs_theta_persession_stats(df, n_channels_for_bonferroni=1, out_dir=OUTPUT_DIR, lab=LAB_ALL):
    """Dunn et al.'s Fig. 3 statistics with acceleration as the predictor: bin
    each session's mobile data into FIG3_ACC_BIN_WIDTH-wide acceleration bins
    (median Frequency/Power per bin; no percentile cut), fit one OLS regression
    per session (not a mixed model), and summarise the fitted slopes as mean
    +/- SD across sessions, plus the count of sessions individually
    significant at a Bonferroni-corrected threshold (0.05 /
    n_channels_for_bonferroni).
    """
    out_dir, name = Path(out_dir), lab['name']
    _df_clean, df_binned = fig3_trim_and_bin(df)
    save_tables(out_dir, "Fig3_acc_binned_medians", {"binned": df_binned}, lab['fmt'])

    alpha_bonf = 0.05 / n_channels_for_bonferroni

    meta = df[["Session", "Recording", "Animal"]].drop_duplicates("Session")

    units = {"Frequency": "Hz", **{m[0]: m[4] for m in POWER_MEASURES}}
    summary_lines = [lab['note']] if lab['note'] else []
    for response in units:
        fits = fit_per_session_ols(df_binned, response)
        save_tables(out_dir, f"Fig3_persession_OLS_{response}", {"fits": fits}, lab['fmt'])

        n_sig = int((fits["pvalue"] < alpha_bonf).sum())
        lines = [(f"{response} vs {lab['xname']} (per-session OLS, n={len(fits)} sessions): "
                  f"beta1 = {fits['beta1'].mean():.4g} +/- {fits['beta1'].std():.4g} (mean +/- SD), "
                  f"{n_sig}/{len(fits)} sessions significant at Bonferroni p < {alpha_bonf:.4g}")]

        # group-level result across all recordings
        grp, per_rec, per_animal = summarise_slopes_across_recordings(fits, meta)
        save_tables(out_dir, f"Fig3_group_slopes_{response}", {
            "per_recording": per_rec, "per_animal": per_animal, "group_test": pd.DataFrame([grp])},
            lab['fmt'])
        plot_recording_slopes(
            per_rec, grp, ylabel=f"β₁ ({units[response]} per {ACC_UNIT})",
            title=f"Theta {response} vs. {name} slope",
            out_path=out_dir / f"Fig3_recording_slopes_{response}.png", note=lab['note'])

        lines.append(
            f"  Group level (n={grp['n_recordings']} recordings from {grp['n_animals']} animals, "
            f"channels averaged within recording): beta1 = {grp['beta1_mean']:.4g} +/- "
            f"{grp['beta1_sem']:.4g} (mean +/- SEM); one-sample t({grp['n_recordings'] - 1}) = "
            f"{grp['t']:.3f}, p = {grp['t_pvalue']:.3g}; Wilcoxon W = {grp['wilcoxon_W']:.4g}, "
            f"p = {grp['wilcoxon_pvalue']:.3g}; {grp['n_positive']}/{grp['n_recordings']} "
            f"recordings with positive slope")
        for _, r in per_animal.iterrows():
            lines.append(f"    {r['Animal']}: beta1 = {r['beta1_mean']:.4g} +/- {r['beta1_sd']:.4g} "
                         f"(mean +/- SD, n={r['n_recordings']} recordings)")

        for line in lines:
            print(line)
        summary_lines.extend(lines)

    with open(out_dir / "Fig3_persession_OLS_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    return df_binned


# %% ==================== Individual-file correlations (per-file OLS plots + % positive) ====
#
# Every analysed .ncs file (one Session) is fit on its own with the per-session
# OLS above (fig3_trim_and_bin -> fit_per_session_ols), but with its own
# acceleration-bin inclusion rule (INDIV_MIN_TIME_PER_BIN_S),
# so slopes differ from Fig3_persession_OLS_<measure>.xlsx. Per measure, the p-values
# of all fitted files are Benjamini-Hochberg FDR-corrected, and each file is
# put in one CORREL_CATEGORIES class. The % summary uses every analysed file
# as the denominator: files without a fit (< 3 acceleration bins, or no finite
# value of the measure) are counted as "Not fitted". Power is the relative
# (normalized) power PowerRel, not PowerdB.

INDIV_DIR = OUTPUT_DIR / 'IndivdualCorrel'
# (column, panel title, y-axis label)
INDIV_MEASURES = [
    ("Frequency", "Theta frequency", "Theta Frequency [Hz]"),
    ("PowerRel", "Theta power (relative)", POWER_MEASURES[2][2]),
]
INDIV_FDR_Q = 0.05
# Acceleration-bin inclusion for the individual fits only: no per-file
# percentile cut, and an acceleration bin is kept only with >=
# INDIV_MIN_TIME_PER_BIN_S of data (5 s = 20 time bins of BINSIZE), which
# drops thinly sampled bins (rare extreme accelerations, tracking artifacts).
INDIV_MIN_TIME_PER_BIN_S = 5.0
# (class, colour), in bar-stacking order from the baseline up
CORREL_CATEGORIES = [
    ("Positive, significant", "#c43c3b"),
    ("Positive, n.s.",        "#f2b0ae"),
    ("Negative, n.s.",        "#9ec5f4"),
    ("Negative, significant", "#1c5cab"),
    ("Not fitted",            "#d9d8d4"),
]
_DARK_CATEGORIES = {"Positive, significant", "Negative, significant"}
# Epoch cloud behind the speed-bin medians (individual plots only): every
# 0.25 s time bin that went into the fit, drawn as a small borderless gray dot
# with low opacity, so the gray gets darker where epochs overlap (denser).
INDIV_EPOCH_COLOR = '0.35'      # darkest gray reached where many dots stack
# opacity of a single epoch dot = INDIV_EPOCH_ALPHA_K / sqrt(n epochs), clipped
# to INDIV_EPOCH_ALPHA_RANGE: files with few epochs still show each dot, files
# with thousands don't turn into a solid block
INDIV_EPOCH_ALPHA_K = 4.0
INDIV_EPOCH_ALPHA_RANGE = (0.04, 0.35)
INDIV_EPOCH_SIZE = 9            # marker area (pt^2)
# y-axis limited to this percentile range of the epochs (plus the medians,
# whiskers and fit), so a few extreme epochs can't flatten the plot; the
# number of epochs outside the axis is given in the panel text.
INDIV_EPOCH_YLIM_PCTILE = (1.0, 99.0)


def classify_individual_fits(fits, sessions, q_thresh=INDIV_FDR_Q):
    """`fits` (fit_per_session_ols output) re-indexed onto every session in
    `sessions`, with the BH-FDR q-value (over the fitted files) and the class."""
    out = pd.DataFrame({"Session": list(sessions)}).merge(fits, on="Session", how="left")
    out["qvalue"] = np.nan
    fitted = out["pvalue"].notna()
    if fitted.any():
        out.loc[fitted, "qvalue"] = multipletests(out.loc[fitted, "pvalue"], method="fdr_bh")[1]

    def _cls(r):
        if np.isnan(r["beta1"]):
            return "Not fitted"
        sig = r["qvalue"] < q_thresh
        if r["beta1"] > 0:
            return "Positive, significant" if sig else "Positive, n.s."
        return "Negative, significant" if sig else "Negative, n.s."
    out["Category"] = out.apply(_cls, axis=1)
    return out


def plot_individual_correlation(ses_id, raw, binned, cls_rows, out_path, lab=LAB_ALL):
    """One file: per measure, its FIG3_ACC_BIN_WIDTH acceleration-bin medians
    (whiskers = IQR of the 0.25 s time bins in that bin) against the bin's
    median acceleration, with the file's own OLS line (solid if FDR-significant,
    dashed if not). Behind them, every 0.25 s time bin (epoch) used in the fit
    is drawn at its own acceleration as a faint gray dot; overlapping dots add
    up, so denser regions are darker (INDIV_EPOCH_*)."""
    fig, axes = plt.subplots(1, len(INDIV_MEASURES), figsize=(5.5 * len(INDIV_MEASURES), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (col, panel_title, ylabel) in zip(axes, INDIV_MEASURES):
        b = binned.dropna(subset=[col]).sort_values("Acceleration")
        r = raw.dropna(subset=[col])
        fit = cls_rows[col]
        animal = fit["Animal"]
        y_keep = []  # values the y-axis must show

        if len(r):
            alpha = float(np.clip(INDIV_EPOCH_ALPHA_K / np.sqrt(len(r)), *INDIV_EPOCH_ALPHA_RANGE))
            ax.scatter(r["Acceleration"], r[col], s=INDIV_EPOCH_SIZE, color=INDIV_EPOCH_COLOR,
                       alpha=alpha, linewidths=0, zorder=1, rasterized=True)
            y_keep += list(np.percentile(r[col], INDIV_EPOCH_YLIM_PCTILE))

        if len(b):
            q = (r.assign(AccBin=acc_bin_index(r["Acceleration"], FIG3_ACC_BIN_WIDTH))
                  .groupby("AccBin")[col].quantile([0.25, 0.75]).unstack())
            ab = acc_bin_index(b["Acceleration"], FIG3_ACC_BIN_WIDTH)
            lo = q.reindex(ab)[0.25].to_numpy()
            hi = q.reindex(ab)[0.75].to_numpy()
            y = b[col].to_numpy()
            ax.errorbar(b["Acceleration"], y, yerr=[np.maximum(y - lo, 0), np.maximum(hi - y, 0)],
                        fmt='none', ecolor='0.2', elinewidth=1, capsize=3, zorder=2)
            ax.scatter(b["Acceleration"], y, s=55, color=animal_color(animal), edgecolor='k',
                       linewidth=0.7, zorder=3)
            y_keep += [np.nanmin(lo), np.nanmax(hi), y.min(), y.max()]

        if fit["Category"] != "Not fitted":
            x = np.linspace(b["Acceleration"].min(), b["Acceleration"].max(), 50)
            y_fit = fit["intercept"] + fit["beta1"] * x
            ax.plot(x, y_fit, color='black', lw=2,
                    ls='-' if fit["qvalue"] < INDIV_FDR_Q else '--', zorder=4)
            y_keep += [y_fit.min(), y_fit.max()]
            details = [
                f"β₁ = {fit['beta1']:.4g} ± {fit['beta1_se']:.2g} SE per {ACC_UNIT}",
                f"r = {fit['r']:.3f},  R² = {fit['r2']:.3f}",
                f"p = {fit['pvalue']:.3g},  q (BH-FDR) = {fit['qvalue']:.3g}",
                f"n = {int(fit['n_bins'])} acceleration bins ({len(r)} time bins)",
                f"Class: {fit['Category']}",
            ]
        else:
            why = (f"no finite {col} values" if r.empty and not raw.empty
                   else f"{len(b)} acceleration bins (< 3)")
            details = [f"Not fitted: {why}"]

        y_keep = np.asarray(y_keep, dtype=float)
        y_keep = y_keep[np.isfinite(y_keep)]
        if len(y_keep):
            y0, y1 = y_keep.min(), y_keep.max()
            pad = 0.05 * (y1 - y0) if y1 > y0 else 1.0
            ax.set_ylim(y0 - pad, y1 + pad)
            n_out = int(((r[col] < y0 - pad) | (r[col] > y1 + pad)).sum())
            details.append(f"Gray: {len(r)} epochs ({BINSIZE:g} s), darker = denser"
                           + (f"; {n_out} outside y-axis" if n_out else ""))
        details = "\n".join(details)
        ax.set_title(panel_title, fontsize=12, weight='bold')
        ax.axvline(0, color='0.6', linestyle=':', linewidth=1, zorder=0)
        ax.set_xlabel(f"{lab['xname']} ({ACC_UNIT}); points = median per {FIG3_ACC_BIN_WIDTH:g} {ACC_UNIT} bin",
                      fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        add_details_below(fig, ax, details)
    fig.suptitle(ses_id if lab['key'] == 'All' else f"{ses_id}  [{lab['name']} epochs]", fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


def _stacked_percent_bars(ax, groups):
    """`groups`: list of (x label, Series of class counts). One 100% bar each."""
    cats = [c for c, _ in CORREL_CATEGORIES]
    for i, (_label, counts) in enumerate(groups):
        counts = counts.reindex(cats, fill_value=0)
        total = counts.sum()
        bottom = 0.0
        for cat, color in CORREL_CATEGORIES:
            pct = 100 * counts[cat] / total if total else 0
            if pct > 0:
                ax.bar(i, pct, bottom=bottom, width=0.7, color=color,
                       edgecolor='white', linewidth=1.5)
                if pct >= 4:  # count on a second line only where the segment has room
                    label = f"{pct:.0f}%\n({counts[cat]})" if pct >= 9 else f"{pct:.0f}% ({counts[cat]})"
                    ax.text(i, bottom + pct / 2, label, ha='center',
                            va='center', fontsize=8,
                            color='white' if cat in _DARK_CATEGORIES else '0.15')
                else:     # too thin for text inside: the % just right of the segment
                    ax.text(i + 0.37, bottom + pct / 2, f"{pct:.1f}%", ha='left',
                            va='center', fontsize=6.5, color='0.15')
            bottom += pct
        ax.text(i, 101.5, f"n = {total}", ha='center', va='bottom', fontsize=8, color='0.3')
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, 108)
    ax.spines[['top', 'right']].set_visible(False)


def plot_percent_positive(cls_long, out_path, lab=LAB_ALL):
    """% of all analysed files in each class: one bar per measure (all files),
    then one bar per animal for each measure."""
    animals = ordered_animals(cls_long["Animal"])
    n_panels = 1 + len(INDIV_MEASURES)
    fig = plt.figure(figsize=(3 + 2.2 * len(animals) * len(INDIV_MEASURES), 7.2))
    gs = fig.add_gridspec(2, n_panels, height_ratios=[4, 1.25],
                          width_ratios=[len(INDIV_MEASURES)] + [len(animals)] * len(INDIV_MEASURES))
    ax0 = fig.add_subplot(gs[0, 0])
    _stacked_percent_bars(ax0, [(title, cls_long.loc[cls_long["Measure"] == col, "Category"].value_counts())
                                for col, title, _ in INDIV_MEASURES])
    ax0.set_ylabel("% of LFP files", fontsize=11)
    ax0.set_title("All files", fontsize=12, weight='bold')
    for k, (col, title, _) in enumerate(INDIV_MEASURES, start=1):
        ax = fig.add_subplot(gs[0, k], sharey=ax0)
        sub = cls_long[cls_long["Measure"] == col]
        _stacked_percent_bars(ax, [(a, sub.loc[sub["Animal"] == a, "Category"].value_counts())
                                   for a in animals])
        ax.set_title(f"{title}, by animal", fontsize=12, weight='bold')
        ax.tick_params(labelleft=False)

    lines = []
    for col, title, _ in INDIV_MEASURES:
        c = cls_long.loc[cls_long["Measure"] == col, "Category"].value_counts()
        n = int(c.sum())
        n_pos = int(c.get("Positive, significant", 0) + c.get("Positive, n.s.", 0))
        n_fit = n - int(c.get("Not fitted", 0))
        lines.append(f"{title}: slope > 0 in {n_pos}/{n} files ({100 * n_pos / n:.1f}%); "
                     f"significantly positive in {c.get('Positive, significant', 0)}/{n} "
                     f"({100 * c.get('Positive, significant', 0) / n:.1f}%); "
                     f"significantly negative in {c.get('Negative, significant', 0)}/{n}; "
                     f"not fitted {c.get('Not fitted', 0)}/{n}  (BH-FDR over {n_fit} fitted files)")
    lines.append(f"Per-file OLS of {FIG3_ACC_BIN_WIDTH:g} {ACC_UNIT} acceleration-bin medians (speeds ≥ "
                 f"{MIN_SPEED_CMS:g} cm/s, no percentile cut, acceleration bins with ≥ "
                 f"{INDIV_MIN_TIME_PER_BIN_S:g} s of data); significant = Benjamini-Hochberg q < "
                 f"{INDIV_FDR_Q:g}, per measure.")
    if lab['note']:
        lines.append(lab['note'])
    axl = fig.add_subplot(gs[1, :])
    axl.axis('off')
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, color in CORREL_CATEGORIES]
    axl.legend(handles, [c for c, _ in CORREL_CATEGORIES], loc='upper center',
               ncol=len(CORREL_CATEGORIES), frameon=False, fontsize=9)
    axl.text(0.5, 0.55, "\n".join(lines), transform=axl.transAxes, ha='center', va='top',
             fontsize=9, linespacing=1.5)
    fig.suptitle(f"Theta vs. {lab['name'].lower()}: individual-file correlations", fontsize=13, weight='bold')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


# ---- Files significant for both measures (frequency AND relative power) ----
# Only files whose slope is FDR-significant (CORREL_CATEGORIES "... significant")
# for every measure in INDIV_MEASURES are kept; each is put in one class by the
# signs of its two slopes. Colours: the significant-positive red and
# significant-negative blue of CORREL_CATEGORIES, and their exact 50/50 RGB
# mixture (violet) for opposite signs.
def _mix_hex(c1, c2):
    """Per-channel RGB average of two '#rrggbb' colours."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round((x + y) / 2):02x}' for x, y in zip(a, b))


_CORREL_COLOR = dict(CORREL_CATEGORIES)
BOTH_SIG_CATEGORIES = [
    ("Both positive",  _CORREL_COLOR["Positive, significant"]),
    ("Opposite signs", _mix_hex(_CORREL_COLOR["Positive, significant"],
                                _CORREL_COLOR["Negative, significant"])),
    ("Both negative",  _CORREL_COLOR["Negative, significant"]),
]
_SIG_SIGN = {"Positive, significant": 1, "Negative, significant": -1}


def both_significant_table(cls_long):
    """One row per file: each measure's slope, q-value and class, whether the
    file is significant for all INDIV_MEASURES, and its sign combination
    (BOTH_SIG_CATEGORIES; empty if not significant for both)."""
    cols = [m[0] for m in INDIV_MEASURES]
    meta = cls_long.drop_duplicates("Session").set_index("Session")[["Animal", "Recording"]]
    wide = cls_long.pivot(index="Session", columns="Measure", values=["beta1", "qvalue", "Category"])
    out = meta.copy()
    for col in cols:
        for v in ("beta1", "qvalue", "Category"):
            out[f"{v}_{col}"] = wide[(v, col)]
    signs = pd.DataFrame({col: out[f"Category_{col}"].map(_SIG_SIGN) for col in cols})
    both = signs.notna().all(axis=1)
    out["BothSignificant"] = both
    out["Combination"] = ""
    out.loc[both & (signs > 0).all(axis=1), "Combination"] = "Both positive"
    out.loc[both & (signs < 0).all(axis=1), "Combination"] = "Both negative"
    opp = both & (out["Combination"] == "")
    out.loc[opp, "Combination"] = "Opposite signs"
    out["Direction"] = ""
    out.loc[opp, "Direction"] = signs[opp].apply(
        lambda r: ", ".join(f"{c} {'+' if r[c] > 0 else '−'}" for c in cols), axis=1)
    return out.reset_index()


def plot_both_significant(both_tbl, out_path, lab=LAB_ALL):
    """Number of files significant for both measures, per sign combination:
    all files (left) and by animal (right)."""
    sig = both_tbl[both_tbl["BothSignificant"]]
    cats = [c for c, _ in BOTH_SIG_CATEGORIES]
    animals = ordered_animals(both_tbl["Animal"])
    n_all = int(len(both_tbl))
    counts_all = sig["Combination"].value_counts().reindex(cats, fill_value=0)
    counts_an = (sig.groupby("Animal")["Combination"].value_counts().unstack(fill_value=0)
                    .reindex(index=animals, columns=cats, fill_value=0))
    ymax = max(1, int(counts_all.max()))

    fig = plt.figure(figsize=(4 + 1.6 * len(animals), 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.1], width_ratios=[1.3, len(animals)])
    ax0 = fig.add_subplot(gs[0, 0])
    for i, (cat, color) in enumerate(BOTH_SIG_CATEGORIES):
        ax0.bar(i, counts_all[cat], width=0.7, color=color, edgecolor='k', linewidth=0.6)
        ax0.text(i, counts_all[cat] + 0.02 * ymax, str(counts_all[cat]),
                 ha='center', va='bottom', fontsize=10)
    ax0.set_xticks(range(len(cats)))
    ax0.set_xticklabels(cats, rotation=30, ha='right', fontsize=9)
    ax0.set_ylabel("Number of LFP files", fontsize=11)
    ax0.set_title(f"All files\n({len(sig)} of {n_all} significant for both)", fontsize=11, weight='bold')

    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    w = 0.8 / len(cats)
    for j, (cat, color) in enumerate(BOTH_SIG_CATEGORIES):
        x = np.arange(len(animals)) + (j - (len(cats) - 1) / 2) * w
        ax1.bar(x, counts_an[cat], width=w, color=color, edgecolor='k', linewidth=0.6)
        for xi, n in zip(x, counts_an[cat]):
            ax1.text(xi, n + 0.02 * ymax, str(n), ha='center', va='bottom', fontsize=9)
    n_files_an = both_tbl.groupby("Animal").size().reindex(animals, fill_value=0)
    ax1.set_xticks(range(len(animals)))
    ax1.set_xticklabels([f"{a}\n({int(counts_an.loc[a].sum())} of {n_files_an[a]})" for a in animals],
                        fontsize=9)
    ax1.set_title("By animal", fontsize=11, weight='bold')
    ax1.tick_params(labelleft=False)
    for ax in (ax0, ax1):
        ax.set_ylim(0, ymax * 1.15 + 0.5)
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.spines[['top', 'right']].set_visible(False)

    titles = {col: title for col, title, _ in INDIV_MEASURES}
    opp = sig.loc[sig["Combination"] == "Opposite signs", "Direction"].value_counts()
    opp_txt = "; ".join(f"{d}: {n}" for d, n in opp.items()) if len(opp) else "none"
    lines = [
        f"Kept: files whose slope is significant (BH-FDR q < {INDIV_FDR_Q:g}, per measure) for both "
        + " and ".join(titles.values()) + f" ({len(sig)} of {n_all} analysed files).",
        f"Opposite signs split: {opp_txt}.",
        *([lab['note']] if lab['note'] else []),
    ]
    axl = fig.add_subplot(gs[1, :])
    axl.axis('off')
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor='k', linewidth=0.6)
               for _, c in BOTH_SIG_CATEGORIES]
    axl.legend(handles, cats, loc='upper center', ncol=len(cats), frameon=False, fontsize=9)
    axl.text(0.5, 0.5, "\n".join(lines), transform=axl.transAxes, ha='center', va='top',
             fontsize=9, linespacing=1.5)
    fig.suptitle(f"Theta frequency and power both modulated by {lab['name'].lower()}: direction",
                 fontsize=12, weight='bold')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


def save_both_significant(cls_long, out_dir=OUTPUT_DIR, lab=LAB_ALL):
    """IndividualCorrel_BothSignificant.png + .xlsx (per-file table and counts)."""
    tbl = both_significant_table(cls_long)
    plot_both_significant(tbl, Path(out_dir) / "IndividualCorrel_BothSignificant.png", lab)
    cats = [c for c, _ in BOTH_SIG_CATEGORIES]
    sig = tbl[tbl["BothSignificant"]]
    counts = (sig.groupby("Animal")["Combination"].value_counts().unstack(fill_value=0)
                 .reindex(index=ordered_animals(tbl["Animal"]), columns=cats, fill_value=0))
    counts.loc["All files"] = counts.sum()
    n_files = tbl.groupby("Animal").size()
    counts["n_both_significant"] = counts[cats].sum(axis=1)
    counts["n_files"] = [n_files.get(a, 0) for a in counts.index[:-1]] + [len(tbl)]
    save_tables(out_dir, "IndividualCorrel_BothSignificant", {
        "per_file": tbl, "counts": counts.rename_axis("Animal").reset_index()}, lab['fmt'])
    print(counts.to_string())
    return tbl


def run_individual_correlations(df, out_dir=OUTPUT_DIR, lab=LAB_ALL):
    """Per-file OLS fits, FDR classes, one PNG per file in <out_dir>/IndivdualCorrel,
    the % summary figure and IndividualCorrel_per_file.xlsx."""
    out_dir = Path(out_dir)
    indiv_dir = out_dir / INDIV_DIR.name
    indiv_dir.mkdir(parents=True, exist_ok=True)
    df_clean, df_binned = fig3_trim_and_bin(df, min_bin_s=INDIV_MIN_TIME_PER_BIN_S)
    meta = (df[["Session", "Recording", "Animal"]].drop_duplicates("Session")
              .sort_values("Session").reset_index(drop=True))
    sessions = meta["Session"].tolist()

    per_measure = {}
    for col, _title, _ylabel in INDIV_MEASURES:
        cls = classify_individual_fits(fit_per_session_ols(df_binned, col), sessions)
        per_measure[col] = cls.merge(meta, on="Session", how="left").assign(Measure=col)
    cls_long = pd.concat(per_measure.values(), ignore_index=True)

    raw_by_ses = dict(tuple(df_clean.groupby("Session")))
    bin_by_ses = dict(tuple(df_binned.groupby("Session")))
    empty_raw, empty_bin = df_clean.iloc[:0], df_binned.iloc[:0]
    for i, ses_id in enumerate(sessions, start=1):
        rows = {col: per_measure[col].set_index("Session").loc[ses_id] for col in per_measure}
        fname = re.sub(r'[^\w.-]', '_', ses_id) + '.png'
        plot_individual_correlation(ses_id, raw_by_ses.get(ses_id, empty_raw),
                                    bin_by_ses.get(ses_id, empty_bin), rows, indiv_dir / fname, lab)
        if i % 25 == 0 or i == len(sessions):
            print(f'  individual correlation plots: {i}/{len(sessions)}')

    plot_percent_positive(cls_long, out_dir / "IndividualCorrel_PercentPositive.png", lab)
    save_both_significant(cls_long, out_dir, lab)   # files significant for both measures

    cats = [c for c, _ in CORREL_CATEGORIES]
    overall = (cls_long.groupby("Measure")["Category"].value_counts().unstack(fill_value=0)
                       .reindex(columns=cats, fill_value=0))
    by_animal = (cls_long.groupby(["Measure", "Animal"])["Category"].value_counts()
                         .unstack(fill_value=0).reindex(columns=cats, fill_value=0))
    for t in (overall, by_animal):
        t["n_files"] = t[cats].sum(axis=1)
        t["pct_slope_positive"] = 100 * (t["Positive, significant"] + t["Positive, n.s."]) / t["n_files"]
        t["pct_sig_positive"] = 100 * t["Positive, significant"] / t["n_files"]
        t["pct_sig_negative"] = 100 * t["Negative, significant"] / t["n_files"]
    cols = ["Measure", "Session", "Animal", "Recording", "beta1", "beta1_se", "intercept",
            "r", "r2", "pvalue", "qvalue", "n_bins", "Category"]
    save_tables(out_dir, "IndividualCorrel_per_file", {
        "per_file": cls_long[cols], "counts_all_files": overall.reset_index(),
        "counts_by_animal": by_animal.reset_index()}, lab['fmt'])
    print(overall.to_string())
    return cls_long


# %% ==================== Acceleration vs. deceleration epochs, analysed separately ====
#
# The time bins are split by the sign of their acceleration and every analysis
# above (mixed model, speed-controlled model, per-session OLS, individual-file
# fits) is run on each half on its own:
#   * Acc   : time bins with a >= ACC_BIN_WIDTH / 2 (acceleration bins centred
#             4, 8, 12 ... cm/s^2) -> OUTPUT_DIR/Acc
#   * Decel : time bins with a < -ACC_BIN_WIDTH / 2 (bins centred -4, -8 ...)
#             -> OUTPUT_DIR/Decel
# An epoch is one 0.25 s time bin (BINSIZE); no minimum run length. The 0 bin
# [-ACC_BIN_WIDTH/2, ACC_BIN_WIDTH/2) -- near-constant speed -- is in neither.
# The predictor is the MAGNITUDE |a| in both halves, so the Decel x-axis runs from
# 0 to hard braking and a positive slope means theta rises with stronger
# deceleration (same reading as in Acc). The signed value is kept as
# AccelerationSigned. Mobility rule, binning, bin-inclusion rules and power
# measures are those of the combined analyses. Tables are written as .csv.

SPLIT_LABELS = {
    'Acc': dict(key='Acc', name='Acceleration', xname='Acceleration', short='Acc', fmt='csv',
                note=f'Acceleration epochs only: {BINSIZE:g} s time bins with a ≥ {ACC_BIN_WIDTH / 2:g} '
                     f'{ACC_UNIT} (near-zero bin excluded).'),
    'Decel': dict(key='Decel', name='Deceleration', xname='|Deceleration|', short='|Decel|', fmt='csv',
                  note=f'Deceleration epochs only: {BINSIZE:g} s time bins with a < −{ACC_BIN_WIDTH / 2:g} '
                       f'{ACC_UNIT} (near-zero bin excluded); x = |a|, so slope > 0 = theta rises with '
                       f'harder braking.'),
}


def split_acc_decel(df, key):
    """Time bins of one direction ('Acc' or 'Decel', see SPLIT_LABELS), with
    Acceleration replaced by its magnitude and the signed value kept as
    AccelerationSigned. Uses the same 0-centred bins as everywhere else, so the
    0 bin is excluded from both."""
    k = acc_bin_index(df["Acceleration"])
    d = df[k >= 1 if key == 'Acc' else k <= -1].copy()
    d["AccelerationSigned"] = d["Acceleration"]
    d["Acceleration"] = d["Acceleration"].abs()
    return d


def run_acc_decel_split(df, out_root=OUTPUT_DIR, source=''):
    """Run every acceleration analysis separately on the acceleration and the
    deceleration time bins (split_acc_decel) -> <out_root>/Acc and
    <out_root>/Decel. `source` (e.g. the parquet path) is recorded in each
    folder's Split_settings.csv."""
    import datetime
    k_all = acc_bin_index(df["Acceleration"])
    mobile = df["Mobile"].to_numpy() & np.isfinite(df["Acceleration"].to_numpy())
    for key, lab in SPLIT_LABELS.items():
        out_dir = Path(out_root) / key
        out_dir.mkdir(parents=True, exist_ok=True)
        d = split_acc_decel(df, key)
        n_mob = int(d["Mobile"].sum())
        print(f'\n===== {key}: {len(d)} time bins ({n_mob} mobile = '
              f'{100 * n_mob / max(mobile.sum(), 1):.1f}% of mobile time bins; '
              f'{100 * ((k_all == 0) & mobile).sum() / max(mobile.sum(), 1):.1f}% of mobile bins '
              f'are in the excluded 0 bin) -> {out_dir}')

        settings = {
            'run_datetime': datetime.datetime.now().isoformat(timespec='seconds'),
            'input': source or 'in-memory DataFrame (ThetaVsAcc driver)',
            'subset': lab['note'],
            'epoch': f'one {BINSIZE:g} s time bin, classed by the sign of its acceleration; no minimum run length',
            'predictor': '|a| (magnitude) in cm/s^2; signed value kept as AccelerationSigned',
            'ACC_BIN_WIDTH': ACC_BIN_WIDTH, 'MIN_SPEED_CMS': MIN_SPEED_CMS,
            'SPEED_BIN_WIDTH_CMS': SPEED_BIN_WIDTH_CMS, 'BIN_AGG_METHOD': BIN_AGG_METHOD,
            'INDIV_MIN_TIME_PER_BIN_S': INDIV_MIN_TIME_PER_BIN_S, 'INDIV_FDR_Q': INDIV_FDR_Q,
            'n_time_bins': len(d), 'n_mobile_time_bins': n_mob,
            'n_sessions': d["Session"].nunique(), 'n_animals': d["Animal"].nunique(),
            'abs_acc_mobile_p5_p50_p95': ' / '.join(
                f'{v:.1f}' for v in d.loc[d["Mobile"], "Acceleration"].quantile([0.05, 0.5, 0.95])),
        }
        pd.DataFrame({'parameter': list(settings), 'value': [str(v) for v in settings.values()]}
                     ).to_csv(out_dir / 'Split_settings.csv', index=False, encoding='utf-8-sig')

        res = run_acc_vs_theta_stats(d, out_dir, lab)
        run_speed_controlled_model(d, res, out_dir, lab)
        run_acc_vs_theta_persession_stats(d, out_dir=out_dir, lab=lab)
        run_individual_correlations(d, out_dir, lab)


# %% ==================== Run metadata ===============================================

def _script_path():
    """Absolute path of this script (falls back to the working directory when
    run cell-by-cell/interactively, where __file__ isn't defined)."""
    try:
        return Path(__file__).resolve()
    except NameError:
        return Path(NCS_REPAIR_DIR) / 'ThetaVsAcc.py'


def save_run_metadata(meta_dir=META_DIR):
    """Write every setting that determines this script's output (config block
    + hard-coded values used later in the file), the data directory it was
    run on, and the script's name/directory and run time to a 2-column
    (parameter, value) CSV in `meta_dir`, and save a copy of the script
    itself there, so the results folder records exactly how it was made."""
    import datetime
    import shutil
    script_path = _script_path()
    meta = {
        # ---- provenance ----
        'script_name':              script_path.name,
        'script_dir':               script_path.parent,
        'run_datetime':             datetime.datetime.now().isoformat(timespec='seconds'),
        'data_dir (ROOT_DIR)':      ROOT_DIR,
        'OUTPUT_DIR':               OUTPUT_DIR,
        'OUTPUT_PARQUET':           OUTPUT_PARQUET,
        # ---- acceleration variant ----
        'predictor':                'Acceleration (signed, cm/s^2) = np.gradient(binned speed, BINSIZE): '
                                    'central difference of the BIN_AGG_METHOD speed of adjacent '
                                    'BINSIZE time bins, on the gap-free grid before bins are dropped',
        'ACC_BIN_WIDTH':            f'{ACC_BIN_WIDTH} cm/s^2, bins centred on 0 (k = floor(a/w + 0.5))',
        'speed_controlled_model':   'response ~ Acceleration + Speed + (1 | Session), REML, on per-session '
                                    f'2-D bins ({SPEED_BIN_WIDTH_CMS} cm/s speed anchored at 0 x '
                                    f'{ACC_BIN_WIDTH} cm/s^2 acceleration centred on 0), bin centres as predictors',
        # ---- mobility ----
        'MIN_SPEED_CMS':            f'{MIN_SPEED_CMS} (time bins below are excluded from all fits, '
                                    'drawn gray on the mixed-model figures)',
        'DROP_DUPLICATE_LFP':       DROP_DUPLICATE_LFP,
        'NCS_FILE_LIST_CSV':        NCS_FILE_LIST_CSV,
        'duplicate_signature':      'SHA-1 of ADBitVolts + record timestamp/n_valid/samples, '
                                    'per recording folder; first in natural name order kept '
                                    '(Make_NcsFileList.py)',
        'fig3_order':               'drop speed < MIN_SPEED_CMS; no percentile cut on acceleration',
        'rel_power_denominator':    'whole session (all speeds), samples near clipping excluded',
        # ---- acquisition ----
        'NATIVE_FS':                NATIVE_FS,
        'ADBitVolts':               'read per file from the .ncs header (-ADBitVolts)',
        'ncs_dtype':                str(ncs_dtype),
        'ncs_samples_used':         'valid samples on a timestamp-anchored grid, gaps up to '
                                    '1 s interpolated (ncs_repair.py)',
        # ---- ADC saturation ----
        'InputRange':               'per-file, from .ncs header (-InputRange, uV)',
        'REJECT_SATURATED_BINS':    REJECT_SATURATED_BINS,
        'SATURATION_TOL_STEPS':     SATURATION_TOL_STEPS,
        'SATURATION_MARGIN_S':      SATURATION_MARGIN_S,
        'saturation_detection':     'raw repaired native-rate trace, before filtering; '
                                    '|uV| >= InputRange - SATURATION_TOL_STEPS * ADBitVolts; '
                                    'mapped to FS_LFP, each clipped run widened by '
                                    'SATURATION_MARGIN_S both sides; time bins overlapping a '
                                    'widened run removed; widened-run samples excluded from TotalPower',
        # ---- LFP cleaning (resample -> notch -> detrend -> low-pass) ----
        'FS_LFP':                   FS_LFP,
        'LINE_HARMONICS':           LINE_HARMONICS,
        'NOTCH_Q':                  NOTCH_Q,
        'detrend_type':             'linear',
        'LOWPASS_CUTOFF_HZ':        LOWPASS_CUTOFF_HZ,
        'LOWPASS_ORDER':            LOWPASS_ORDER,
        # ---- theta band-pass / peak-trough frequency & power ----
        'BANDPASS_CUTOFF':          BANDPASS_CUTOFF,
        'FILTER_ORDER':             FILTER_ORDER,
        'PEAK_THRESH_PC':           PEAK_THRESH_PC,
        'TRIM_DUR':                 TRIM_DUR,
        # ---- binning ----
        'BINSIZE':                  BINSIZE,
        'BIN_AGG_METHOD':           BIN_AGG_METHOD,
        'SPEED_BIN_WIDTH_CMS':      f'{SPEED_BIN_WIDTH_CMS} (speed axis of the speed-controlled model only)',
        # ---- theta-power normalization ----
        'DB_BASELINE_SPEED_CMS':    DB_BASELINE_SPEED_CMS,
        'REL_POWER_BAND':           REL_POWER_BAND,
        'REL_POWER_FILTER_ORDER':   REL_POWER_FILTER_ORDER,
        'POWER_MEASURES':           [m[0] for m in POWER_MEASURES],
        # ---- tracking ----
        'tracking_csv_columns':     'A=time_us, D=x_cm, E=y_cm',
        'POS_JUMP_THRESH_CMS':      POS_JUMP_THRESH_CMS,
        'POS_SMOOTH_SIGMA_SMP':     POS_SMOOTH_SIGMA_SMP,
        # ---- statistics ----
        'mixed_model':              'response ~ Acceleration + (1 | Session), REML; '
                                    'Acceleration = acceleration-bin center',
        'FIG3_ACC_BIN_WIDTH':       FIG3_ACC_BIN_WIDTH,
        'fig3_percentile_cut':      'none',
        'n_channels_for_bonferroni': 1,
        'ols_min_acc_bins':         3,
        'ols_predictor':            'median acceleration within each acceleration bin',
        'INDIV_MIN_TIME_PER_BIN_S': f'{INDIV_MIN_TIME_PER_BIN_S} (acceleration bins with less data '
                                    'dropped from the individual fits; no percentile cut)',
        'individual_correl':        f'per-file OLS of {[m[0] for m in INDIV_MEASURES]}, '
                                    f'Benjamini-Hochberg FDR q < {INDIV_FDR_Q} across fitted files '
                                    'per measure; % denominator = all analysed files',
        'acc_decel_split':          'all analyses above repeated on Acc (a >= ACC_BIN_WIDTH/2) and Decel '
                                    '(a < -ACC_BIN_WIDTH/2) time bins separately, predictor |a|, 0 bin in '
                                    'neither -> OUTPUT_DIR/Acc, OUTPUT_DIR/Decel (.csv tables)',
        # ---- plots ----
        'ANIMAL_COLORS':            ANIMAL_COLORS,
        'figure_dpi':               200,
    }
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / 'run_metadata.csv'
    pd.DataFrame({'parameter': list(meta.keys()),
                  'value':     [str(v) for v in meta.values()]}).to_csv(out_path, index=False)
    print(f'Saved run metadata -> {out_path}')

    if script_path.is_file():
        shutil.copy2(script_path, meta_dir)
        print(f'Saved script copy -> {meta_dir / script_path.name}')
    else:
        print(f'  Script not found at {script_path} -- no copy saved to {meta_dir}')


# %% ==================== Driver =====================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_run_metadata(META_DIR)

    df = process_root_directory(ROOT_DIR)
    df["Mobile"] = df["Speed"] >= MIN_SPEED_CMS
    df = add_normalized_power(df)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f'Saved {len(df)} rows -> {OUTPUT_PARQUET}')
    print(f'  {(~df["Mobile"]).mean():.1%} of time bins are < {MIN_SPEED_CMS:g} cm/s '
          f'(excluded from the fits)')
    save_baseline_coverage(df)

    acc = df.loc[df["Mobile"], "Acceleration"]
    print(f'  Acceleration (mobile time bins): 5th/50th/95th percentile = '
          f'{acc.quantile(0.05):.1f} / {acc.median():.1f} / {acc.quantile(0.95):.1f} {ACC_UNIT}')

    if df["Session"].nunique() > 0:
        acc_results = run_acc_vs_theta_stats(df)        # mixed model (session as random intercept)
        run_speed_controlled_model(df, acc_results)      # Acceleration + Speed, 2-D bins
        run_acc_vs_theta_persession_stats(df)            # Dunn et al. Fig. 3's per-session OLS
        run_individual_correlations(df)                  # one OLS plot per file + % positive
        run_acc_decel_split(df)                          # all of the above, Acc / Decel separately
    else:
        print('No sessions processed successfully -- skipping acceleration vs. theta statistics.')
