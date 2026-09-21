# -*- coding: utf-8 -*-
"""
Phase-precession pipeline for real Neuralynx recordings (.ntt spikes, .ncs
LFP, tracking .csv), run using AditiPrecessionUtils.py's own functions --
not a reimplementation of its formulas -- for every step of the analysis:

    getISI          -> ISI / coefficient of variation (QC only)
    getPhase        -> per-spike theta phase, by linear interpolation
                       between consecutive peaks of the theta-filtered LFP
    calcTMI         -> Theta Modulation Index of the spike-phase distribution
    frRatemap       -> 1D firing-rate map (raw + Gaussian-smoothed) vs. position
    fieldDetect     -> place-field boundaries (10% peak-rate threshold) and
                       the field-restricted, field-normalized spike position/phase
    circRegress/lcc -> Kempter (2012)-style circular-linear regression of
                       phase against normalized field position
    kempCorr        -> the circular-circular correlation reported alongside
    plotPrecession, plotSpikeDensity, movingavgPhase, findPhaseValley
                    -> the module's own precession visualizations

Reference for the phase-precession analysis this reproduces:
    https://onlinelibrary.wiley.com/doi/10.1002/hipo.23641

Data layout (identical to ThetaMod_PhasePrecession_v2.py in this folder):
ROOT_FOLDER is searched recursively; every folder that directly contains at
least one .ncs and at least one .ntt file is treated as a session.
    *.ncs   Neuralynx continuous (LFP) file. The first one (natural sort of
            filename) is used as the theta reference channel.
    *.ntt   Neuralynx tetrode spike files, one file per already-isolated
            unit. Every .ntt file in the folder is processed.
    tracking .csv, auto-detected as the first .csv in the folder. Column
            order is fixed: col A = timestamp, col B = x (pixels, unused),
            col C = y (pixels, unused), col D = x (cm), col E = y (cm).
            Column D (x, cm) is used directly as the 1D track position that
            AditiPrecessionUtils' functions expect (they operate on a single
            scalar position variable, not 2D coordinates).

Two adaptations were necessary to run this module's functions -- written
for a constant-velocity, fixed-number-of-passes simulated rat -- on real,
variable-speed tracking, and are called out where they occur below:
  1. frRatemap's occupancy is a single constant (binsize/vel * numpasses,
     identical in every spatial bin -- see its own body), so 'vel' and
     'numpasses' are estimated from the real trajectory (median running
     speed; count of low/high track-end crossings) rather than being a
     given simulation parameter.
  2. fieldDetect's optional spkts branch ("if spkts:") raises ValueError
     when passed a real (>1-element) ndarray, since a non-empty array's
     truth value is ambiguous. This pipeline doesn't need the field-
     restricted spike timestamps that branch would add to its output, so
     it is simply called without spkts (its default None short-circuits
     that branch safely).

No p-value/significance test is available from AditiPrecessionUtils itself
for either calcTMI or circRegress/kempCorr as written (circRegress's own
'psim' field is only ever populated with NaN, in its <3-spike short-circuit
branch -- the main code path never computes it). calcTMI's TMI is still
reported descriptively here, with no significance test.

circRegress's fit (slope, corr), however, IS tested for significance in
this script, via shuffle_precession_significance -- a circular time-shift
shuffle ported from ThetaMod_PhasePrecession_withGenPhsAp_v7.py's function
of the same name (Climer, Newman & Hasselmo 2013-style shuffle test):
spike times are circularly shifted against the tracking/LFP traces many
times (N_PRECESSION_SHUFFLES), apu.spk_pos_x/apu.getPhase/apu.circRegress
are re-run on each shuffle exactly as on the real data (same fixed field
boundaries, same renormalization), and the fraction of shuffle |corr| >=
the observed |corr| gives p_corr_shuffle (p_slope_shuffle is the same test
computed on the slope alone, as a secondary diagnostic). A cell is
classified 'phase_precessing' if p_corr_shuffle < ALPHA and the slope is
negative, 'phase_recessing' if p_corr_shuffle < ALPHA and the slope is
positive, and 'phase_locked' if the fit was not significant (rather than
being discarded) -- stored in is_precessing / is_recessing /
is_phase_locked / PrecessionClass.

Requires: numpy, scipy, pandas, matplotlib, openpyxl, plus
AditiPrecessionUtils.py's own dependencies (scikit-learn, shapely, joblib)
even though this script does not call the functions that need them
(GMMcluster, calcWithinPolygon), since they are imported unconditionally at
the top of that module. scikit-image is NOT required here -- it's only
imported lazily inside isolateContours, which this pipeline never calls.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

import AditiPrecessionUtils as apu

# ============================================================================
# Configuration -- EDIT THESE
# ============================================================================

ROOT_FOLDER = Path(r"C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True")
OUTPUT_EXCEL_NAME = 'AditiPhasePrecession_Analyzed.xlsx'   # written to ROOT_FOLDER

TRACKING_TIME_UNIT = 'us'          # 'us', 'ms', or 's' -- units of the tracking timestamp column
LFP_THETA_BAND = (3.0, 7.0)        # Hz, band-pass applied to the LFP before apu.getPhase's peak-finding

POSBIN_SIZE_CM = 4.0               # spatial bin width for apu.frRatemap's posbins
FIELD_THRESH = 0.1                 # apu.fieldDetect's own 10%-of-peak threshold, passed through explicitly
MIN_SPEED_CM_S = 2.0               # tracking samples slower than this are treated as immobile when
                                    # estimating frRatemap's constant running 'vel' (see module docstring)
TRACK_END_QUANTILE = 0.1           # low/high-end quantile used to count track passes for 'numpasses'

MIN_SPIKES_FOR_TMI = 8             # apu.calcTMI needs a reasonable sample of phases to be meaningful
MIN_SPIKES_INFIELD_FOR_FIT = 20    # apu.circRegress accepts >=3, but a stable fit needs more
PPDIR = 0                          # apu.circRegress's ppdir: 0 = unconstrained slope sign (direction
                                    # reported afterward via its sign), -1 = constrain to negative
                                    # (classic precessing) slopes only, 1 = positive (recessing) only
MAXSLOPE = 'default'               # apu.circRegress's maxslope: 'default' allows <=720 deg of phase
                                    # change across the field; see its own comment

ALPHA = 0.05                       # significance threshold for the circular time-shift shuffle test
N_PRECESSION_SHUFFLES = 500        # circular time-shift shuffles testing the circRegress fit (corr,
                                    # slope) against a null of no position-phase relationship -- see
                                    # shuffle_precession_significance, ported from
                                    # ThetaMod_PhasePrecession_withGenPhsAp_v7.py's function of the
                                    # same name
PRECESSION_MIN_SHIFT_FRAC = 0.1    # minimum circular shift, as a fraction of the tracking/LFP overlap
                                    # window's duration, so no shuffle leaves spikes nearly unshifted
RANDOM_SEED = 0                    # seed for the shuffle test's RNG, for reproducibility

RUN_INTRINSIC_FREQ_ANALYSIS = False   # apu.TempAutocorr / apu.precessionFreq / apu.thetaskipAmp are
                                       # O(n_spikes^2) double loops in this module (unmodified here);
                                       # off by default, gated further by the spike-count cap below
MAX_SPIKES_FOR_AUTOCORR = 3000        # skip the O(n^2) intrinsic-frequency block above this spike count

MOVING_AVG_WINLEN_CM = 5.0            # apu.movingavgPhase's winlen, passed through explicitly

NCS_SAMPLES_PER_RECORD = 512
HEADER_BYTES = 16 * 1024
DEFAULT_ADBITVOLTS = 0.000000195


# ============================================================================
# Neuralynx / tracking file I/O -- identical to ThetaMod_PhasePrecession_v2.py
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
    """Load a Neuralynx .ncs file. Returns (samples_uV, timestamps_s, fs_hz)."""
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
    """Load a Neuralynx .ntt file. Returns dict {cell_number: spike_times_s}."""
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


def _find_tracking_file(folder: Path) -> Path:
    candidates = sorted(folder.glob('*.csv'))
    if not candidates:
        raise FileNotFoundError(f'No .csv tracking file found in {folder}')
    return candidates[0]


def load_tracking(path: Path, time_unit: str = 'us'):
    """Load tracking coordinates already in cm. Returns (pos_ts_s, pos_xy_cm)."""
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


def find_session_folders(root: Path) -> list[Path]:
    """Recursively find folders directly containing both .ncs and .ntt files."""
    ncs_parents = {p.parent for p in root.rglob('*.ncs')}
    return sorted(folder for folder in ncs_parents if any(folder.glob('*.ntt')))


def bandpass_filter(data, low, high, fs, order=3):
    """Zero-phase Butterworth band-pass (SOS form), used only to turn the raw
    LFP into the oscillatory 'sinInput' apu.getPhase expects -- getPhase
    itself does no filtering, it just finds peaks in whatever signal it's given."""
    nyq = fs / 2.0
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 1 - 1e-6)
    if low_n >= high_n:
        raise ValueError(
            f'invalid filter band after clamping to Nyquist: requested ({low:.4g}, {high:.4g}), '
            f'fs={fs:.4g} -> normalized ({low_n:.4g}, {high_n:.4g})')
    sos = signal.butter(order, [low_n, high_n], btype='band', output='sos')
    filtered = signal.sosfiltfilt(sos, data)
    if np.any(np.isnan(filtered)):
        filtered = signal.sosfilt(sos, data)
    return filtered


# ============================================================================
# Real-tracking-only glue: position at spike time, running speed, pass count
# (none of this is in AditiPrecessionUtils -- it has no I/O or position-
# tracking code at all, since it was written around simulated position traces)
# ============================================================================

def spk_pos_x(pos_ts, pos_x, spk_ts):
    """Nearest tracked x position (cm) at each spike time."""
    idx = np.searchsorted(pos_ts, spk_ts)
    idx = np.clip(idx, 1, len(pos_ts) - 1)
    left, right = idx - 1, idx
    use_left = np.abs(spk_ts - pos_ts[left]) <= np.abs(pos_ts[right] - spk_ts)
    idx = np.where(use_left, left, right)
    return pos_x[idx]


def estimate_vel_and_numpasses(pos_ts, pos_x, min_speed_cm_s=MIN_SPEED_CM_S,
                                end_quantile=TRACK_END_QUANTILE):
    """Estimate the two real-data stand-ins apu.frRatemap needs for its
    single occupancy constant (binsize/vel * numpasses, identical in every
    spatial bin -- see frRatemap's own body): 'vel' is the median running
    speed over above-threshold samples (excluding immobility), 'numpasses'
    is the number of times the trajectory crosses between the low and high
    ends of its tracked range (a simple threshold-crossing lap count)."""
    dt = np.diff(pos_ts)
    dx = np.diff(pos_x)
    with np.errstate(invalid='ignore', divide='ignore'):
        speed = np.abs(dx) / dt
    speed = speed[np.isfinite(speed)]
    moving = speed >= min_speed_cm_s
    vel = float(np.median(speed[moving])) if np.any(moving) else float(np.median(speed))

    lo = np.quantile(pos_x, end_quantile)
    hi = np.quantile(pos_x, 1 - end_quantile)
    zone = np.where(pos_x <= lo, -1, np.where(pos_x >= hi, 1, 0))
    nonzero_zone = zone[zone != 0]
    numpasses = int(np.sum(np.diff(nonzero_zone) != 0)) if len(nonzero_zone) > 1 else 0

    return vel, max(numpasses, 1)


def shuffle_precession_significance(spk_ts, pos_ts, pos_x, filtered_theta, lfp_ts, fldstart, fldend,
                                     observed_corr, observed_slope_deg, ppdir, maxslope, rng,
                                     n_shuffles=N_PRECESSION_SHUFFLES,
                                     min_shift_frac=PRECESSION_MIN_SHIFT_FRAC):
    """Null distribution for the apu.circRegress circular-linear fit, via a
    circular time-shift shuffle -- ported from
    ThetaMod_PhasePrecession_withGenPhsAp_v7.py's shuffle_precession_significance
    (Kempter et al. 2012 fit, Climer/Newman/Hasselmo-style shuffle test).

    Spike times are shifted by a random offset -- at least min_shift_frac of
    the tracking/LFP overlap window, so no shuffle leaves the true alignment
    nearly intact -- and wrapped within that window. Position (apu.spk_pos_x)
    and theta phase (apu.getPhase, re-run on the shifted times exactly as it
    runs on the real spike times) are then re-evaluated at the shifted times,
    restricted to the same fixed field boundaries (fldstart/fldend -- these
    describe where the place field is, so are NOT recomputed per shuffle,
    only the spike-timing/theta relationship within it is being tested),
    renormalized exactly as apu.fieldDetect does its own field-restricted
    positions, and refit with apu.circRegress -- so the null also reflects
    the slope/correlation being *estimated*, not assumed known.

    This preserves the spike train's own temporal structure (ISI/bursting)
    and the natural relationship between position and LFP theta phase,
    destroying only the cell-specific link between spike timing and where in
    the field/theta cycle the animal actually was.

    Returns (p_corr, p_slope, shuffle_corrs, shuffle_slopes). p_corr tests
    circRegress's own 'corr' (the combined goodness-of-fit-and-slope
    statistic used for classification below); p_slope tests the fitted
    slope alone, reported as a secondary diagnostic.
    """
    t_lo = max(pos_ts.min(), lfp_ts.min())
    t_hi = min(pos_ts.max(), lfp_ts.max())
    duration = t_hi - t_lo
    min_shift = min_shift_frac * duration

    shuffle_corrs = np.full(n_shuffles, np.nan)
    shuffle_slopes = np.full(n_shuffles, np.nan)
    for i in range(n_shuffles):
        shift = rng.uniform(min_shift, duration - min_shift)
        shifted_ts = t_lo + np.mod(spk_ts - t_lo + shift, duration)

        shifted_pos = spk_pos_x(pos_ts, pos_x, shifted_ts)
        shifted_phase_deg = apu.getPhase(shifted_ts, filtered_theta, lfp_ts)

        infield = (shifted_pos >= fldstart) & (shifted_pos <= fldend) & ~np.isnan(shifted_phase_deg)
        x = shifted_pos[infield]
        if len(x) < MIN_SPIKES_INFIELD_FOR_FIT:
            continue
        x_norm = x - x.min()
        x_span = x_norm.max()
        if x_span <= 0:
            continue
        x_norm = x_norm / x_span
        y_rad = np.mod(np.deg2rad(shifted_phase_deg[infield]), 2 * np.pi)

        try:
            reg_i = apu.circRegress(x_norm, y_rad, None, ppdir, maxslope=maxslope)
        except Exception:
            continue
        shuffle_corrs[i] = reg_i['corr']
        shuffle_slopes[i] = np.rad2deg(reg_i['slope_opt'])

    valid = np.isfinite(shuffle_corrs) & np.isfinite(shuffle_slopes)
    n_valid = int(np.sum(valid))
    if n_valid == 0 or np.isnan(observed_corr) or np.isnan(observed_slope_deg):
        return np.nan, np.nan, shuffle_corrs, shuffle_slopes

    p_corr = float((np.sum(np.abs(shuffle_corrs[valid]) >= np.abs(observed_corr)) + 1) / (n_valid + 1))
    p_slope = float((np.sum(np.abs(shuffle_slopes[valid]) >= np.abs(observed_slope_deg)) + 1) / (n_valid + 1))
    return p_corr, p_slope, shuffle_corrs, shuffle_slopes


# ============================================================================
# Per-unit analysis, calling AditiPrecessionUtils's own functions throughout
# ============================================================================

def analyze_unit(spk_ts, pos_ts, pos_x, filtered_theta, lfp_ts, posbins, vel, numpasses,
                  unit_label, output_dir: Path, rng):
    """Run the full AditiPrecessionUtils battery for one unit. Returns a
    result dict for the summary table; writes one combined summary figure."""
    row = dict(n_spikes_total=len(spk_ts))

    # ---- ISI / CV (apu.getISI) ----
    _isi, cv = apu.getISI(spk_ts)
    row['ISI_CV'] = float(cv)

    # ---- theta phase per spike (apu.getPhase) ----
    if len(spk_ts) > 20000:
        print(f'    {unit_label}: {len(spk_ts)} spikes -- apu.getPhase\'s peak-bracketing '
              f'loop may take a while.')
    spkphase = apu.getPhase(spk_ts, filtered_theta, lfp_ts)
    valid_phase = ~np.isnan(spkphase)
    row['n_spikes_phase'] = int(np.sum(valid_phase))

    # ---- Theta Modulation Index (apu.calcTMI) ----
    row['TMI'] = np.nan
    if row['n_spikes_phase'] >= MIN_SPIKES_FOR_TMI:
        row['TMI'] = float(apu.calcTMI(spkphase[valid_phase]))
        plt.close('all')  # calcTMI calls plt.hist() unconditionally, even with plotit=False

    # ---- place field + phase precession (frRatemap / fieldDetect / circRegress / kempCorr) ----
    row.update(PrecessionTested=False, PrecessionSkippedReason='', FieldStart_cm=np.nan,
               FieldEnd_cm=np.nan, FieldCentre_cm=np.nan, n_spikes_infield=0,
               CircCorr=np.nan, Slope_deg_per_field=np.nan, PhaseOffset_deg=np.nan,
               PhaseValley_deg=np.nan, p_corr_shuffle=np.nan, p_slope_shuffle=np.nan,
               is_precessing=False, is_recessing=False, is_phase_locked=False,
               PrecessionClass=None)

    if row['n_spikes_phase'] < MIN_SPIKES_INFIELD_FOR_FIT:
        row['PrecessionSkippedReason'] = (
            f'only {row["n_spikes_phase"]} spikes with valid theta phase '
            f'(< MIN_SPIKES_INFIELD_FOR_FIT={MIN_SPIKES_INFIELD_FOR_FIT})')
        return row

    spk_x = spk_pos_x(pos_ts, pos_x, spk_ts)
    _ratemap, gaussratemap, _occmap = apu.frRatemap(spk_x, posbins, numpasses, vel)

    try:
        # spkts is deliberately not passed: fieldDetect's own "if spkts:" branch raises
        # ValueError on a real (>1-element) ndarray -- see module docstring above.
        fielddata = apu.fieldDetect(gaussratemap, posbins, spk_x, spkphase, thresh=FIELD_THRESH)
    except Exception as exc:
        row['PrecessionSkippedReason'] = f'fieldDetect ERROR ({exc})'
        return row

    row['FieldStart_cm'] = float(fielddata['fldstart'])
    row['FieldEnd_cm'] = float(fielddata['fldend'])
    row['FieldCentre_cm'] = float(fielddata['fldcentre'])

    normx = fielddata['normspkpos']
    phs_deg = fielddata['spkphs']
    row['n_spikes_infield'] = len(phs_deg)

    if row['n_spikes_infield'] < MIN_SPIKES_INFIELD_FOR_FIT:
        row['PrecessionSkippedReason'] = (
            f'only {row["n_spikes_infield"]} spikes within the detected field '
            f'(< MIN_SPIKES_INFIELD_FOR_FIT={MIN_SPIKES_INFIELD_FOR_FIT})')
        return row

    y_rad = np.mod(np.deg2rad(phs_deg), 2 * np.pi)
    try:
        regData = apu.circRegress(normx, y_rad, None, PPDIR, maxslope=MAXSLOPE)
    except Exception as exc:
        row['PrecessionSkippedReason'] = f'circRegress ERROR ({exc})'
        return row

    slope_rad = regData['slope_opt']
    slope_deg = float(np.rad2deg(slope_rad))
    row['PrecessionTested'] = True
    row['CircCorr'] = float(regData['corr'])
    row['Slope_deg_per_field'] = slope_deg
    row['PhaseOffset_deg'] = float(np.rad2deg(regData['phase_opt']) % 360)
    row['PhaseValley_deg'] = float(apu.findPhaseValley(phs_deg, ncopies=2))

    # ---- significance of the fit via circular time-shift shuffle (see
    # shuffle_precession_significance above) -- gates the precessing/
    # recessing/phase_locked classification instead of the fitted slope's
    # raw sign alone ----
    p_corr_shuffle, p_slope_shuffle, _shuffle_corrs, _shuffle_slopes = shuffle_precession_significance(
        spk_ts, pos_ts, pos_x, filtered_theta, lfp_ts, row['FieldStart_cm'], row['FieldEnd_cm'],
        row['CircCorr'], slope_deg, PPDIR, MAXSLOPE, rng)
    row['p_corr_shuffle'] = p_corr_shuffle
    row['p_slope_shuffle'] = p_slope_shuffle

    is_significant_precession = bool(np.isfinite(p_corr_shuffle) and p_corr_shuffle < ALPHA)
    # Negative slope: theta-phase precessing (spike phase advances to earlier
    # phase over the field). Positive slope: theta-phase recessing (phase
    # moves later). Neither significant: phase locked (theta-modulated
    # firing with no systematic phase drift across the field).
    row['is_precessing'] = bool(is_significant_precession and slope_deg < 0)
    row['is_recessing'] = bool(is_significant_precession and slope_deg > 0)
    row['is_phase_locked'] = bool(np.isfinite(p_corr_shuffle) and not is_significant_precession)
    if not np.isfinite(p_corr_shuffle):
        row['PrecessionClass'] = None
    elif row['is_precessing']:
        row['PrecessionClass'] = 'phase_precessing'
    elif row['is_recessing']:
        row['PrecessionClass'] = 'phase_recessing'
    else:
        row['PrecessionClass'] = 'phase_locked'

    # ---- optional intrinsic-frequency block (apu.TempAutocorr / precessionFreq / thetaskipAmp) ----
    row['IntrinsicFreq_Hz'] = np.nan
    row['ThetaSkipAmp'] = np.nan
    if RUN_INTRINSIC_FREQ_ANALYSIS and len(spk_ts) <= MAX_SPIKES_FOR_AUTOCORR:
        tempcorr = apu.TempAutocorr(spk_ts)
        row['IntrinsicFreq_Hz'] = float(apu.precessionFreq(spk_ts))
        row['ThetaSkipAmp'] = float(apu.thetaskipAmp(tempcorr))

    _plot_unit_summary(phs_deg, normx, spkphase[valid_phase], fielddata, regData,
                        slope_deg, row, unit_label, output_dir)
    return row


def _plot_unit_summary(phs_deg, normx, all_valid_phase_deg, fielddata, regData,
                        slope_deg, row, unit_label, output_dir: Path):
    """One combined figure per unit, drawn by calling apu.calcTMI / apu.plotPrecession /
    apu.plotSpikeDensity / apu.movingavgPhase directly onto its four panels (plt.sca
    redirects the pyplot-state calls those last three make onto the chosen subplot).
    Only called once row['n_spikes_phase'] has already cleared MIN_SPIKES_INFIELD_FOR_FIT
    (>= MIN_SPIKES_FOR_TMI), so calcTMI always has enough spikes here."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(unit_label, fontsize=12, fontweight='bold')

    plt.sca(axes[0, 0])
    apu.calcTMI(all_valid_phase_deg, plotit=axes[0, 0])

    plt.sca(axes[0, 1])
    x_two = np.concatenate([normx, normx])
    y_two = np.concatenate([phs_deg, phs_deg + 360])
    apu.plotPrecession(x_two, y_two, cms=False)
    xg = np.linspace(0, 1, 200)
    phi_deg = np.rad2deg(np.mod(regData['slope_opt'] * xg + regData['phase_opt'], 2 * np.pi))
    axes[0, 1].plot(xg, phi_deg, 'r', lw=2)
    axes[0, 1].plot(xg, phi_deg + 360, 'r', lw=2)
    axes[0, 1].set_xlabel('normalized field position')
    axes[0, 1].set_title(f'rho={row["CircCorr"]:.2f}  p_shuffle={row["p_corr_shuffle"]:.3g}  '
                          f'slope={slope_deg:.1f} deg/field ({row["PrecessionClass"]})', fontsize=10)

    plt.sca(axes[1, 0])
    x_field_two = np.concatenate([fielddata['spkpos'], fielddata['spkpos']])
    apu.plotSpikeDensity(x_field_two, y_two, xedges=np.linspace(fielddata['fldstart'],
                          fielddata['fldend'], 21), yedges=np.arange(0, 730, 10),
                          gaussSigma=1, cm=True, plotit=axes[1, 0])
    axes[1, 0].set_xlabel('position (cm)')

    plt.sca(axes[1, 1])
    apu.movingavgPhase(fielddata['spkpos'], fielddata['spkphs'], winlen=MOVING_AVG_WINLEN_CM,
                        overlap=0.5, method='mean', plotit=True, smSigma=1.5)
    axes[1, 1].set_title('moving-average phase vs. field position', fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / f'{unit_label}_AditiPrecession.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Batch main
# ============================================================================

def process_session(data_folder: Path, rng) -> list[dict]:
    output_dir = data_folder / 'AditiPhasePrecession'
    output_dir.mkdir(parents=True, exist_ok=True)

    ncs_files = sorted(data_folder.glob('*.ncs'), key=_natural_key)
    theta_ncs = ncs_files[0]
    print(f'  Using LFP file: {theta_ncs.name}')
    lfp_sig, lfp_ts, lfp_fs = load_ncs(theta_ncs)
    filtered_theta = bandpass_filter(lfp_sig, LFP_THETA_BAND[0], LFP_THETA_BAND[1], lfp_fs)

    try:
        tracking_path = _find_tracking_file(data_folder)
        print(f'  Using tracking file: {tracking_path.name}')
        pos_ts, pos_xy = load_tracking(tracking_path, TRACKING_TIME_UNIT)
        pos_x = pos_xy[:, 0]  # x (cm) used directly as the 1D track position
    except FileNotFoundError as exc:
        print(f'  {exc} -- skipping this session (AditiPrecessionUtils needs position for '
              f'frRatemap/fieldDetect/circRegress).')
        return []

    vel, numpasses = estimate_vel_and_numpasses(pos_ts, pos_x)
    lo = np.floor(pos_x.min() / POSBIN_SIZE_CM) * POSBIN_SIZE_CM
    hi = np.ceil(pos_x.max() / POSBIN_SIZE_CM) * POSBIN_SIZE_CM
    posbins = np.arange(lo, hi + POSBIN_SIZE_CM, POSBIN_SIZE_CM)
    print(f'  vel={vel:.1f} cm/s, numpasses={numpasses}, posbins={len(posbins)} x '
          f'{POSBIN_SIZE_CM:.1f} cm')

    session_label = '_'.join(data_folder.parts[-3:])
    ntt_files = sorted(data_folder.glob('*.ntt'), key=_natural_key)

    rows = []
    for ntt_path in ntt_files:
        print(f'  Processing: {ntt_path.name}')
        units = load_ntt_spike_times(ntt_path)
        for cell_number, spk_ts in units.items():
            unit_label = f'{ntt_path.stem}_cell{cell_number}' if len(units) > 1 else ntt_path.stem
            row = dict(Session=session_label, FolderPath=str(data_folder), Unit=unit_label,
                       ntt_file=ntt_path.name, cell_number=cell_number)
            try:
                unit_row = analyze_unit(spk_ts, pos_ts, pos_x, filtered_theta, lfp_ts, posbins,
                                         vel, numpasses, unit_label, output_dir, rng)
                row.update(unit_row)
                if unit_row.get('PrecessionTested'):
                    print(f'    {unit_label}: TMI={unit_row["TMI"]:.2f}  '
                          f'corr={unit_row["CircCorr"]:.2f}  p_shuffle={unit_row["p_corr_shuffle"]:.3g}  '
                          f'slope={unit_row["Slope_deg_per_field"]:.1f} deg/field '
                          f'({unit_row["PrecessionClass"]})')
                else:
                    print(f'    {unit_label}: precession not tested '
                          f'({unit_row["PrecessionSkippedReason"]})')
            except Exception as exc:
                row['PrecessionSkippedReason'] = f'ERROR ({exc})'
                print(f'    {unit_label}: ERROR ({exc})')
            rows.append(row)

    return rows


def main():
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    rng = np.random.default_rng(RANDOM_SEED)

    session_folders = find_session_folders(ROOT_FOLDER)
    if not session_folders:
        raise FileNotFoundError(f'No folders with both .ncs and .ntt files found under {ROOT_FOLDER}')

    all_rows = []
    for data_folder in session_folders:
        print(f'\n=== Session: {data_folder} ===')
        try:
            all_rows.extend(process_session(data_folder, rng))
        except Exception as exc:
            print(f'ERROR processing {data_folder}: {exc}')
            continue

    columns = ['Session', 'FolderPath', 'Unit', 'ntt_file', 'cell_number', 'n_spikes_total',
               'n_spikes_phase', 'ISI_CV', 'TMI', 'PrecessionTested', 'FieldStart_cm',
               'FieldEnd_cm', 'FieldCentre_cm', 'n_spikes_infield', 'CircCorr',
               'Slope_deg_per_field', 'p_corr_shuffle', 'p_slope_shuffle', 'is_precessing',
               'is_recessing', 'is_phase_locked', 'PrecessionClass', 'PhaseOffset_deg',
               'PhaseValley_deg', 'IntrinsicFreq_Hz', 'ThetaSkipAmp', 'PrecessionSkippedReason']
    df = pd.DataFrame(all_rows, columns=columns)
    excel_path = ROOT_FOLDER / OUTPUT_EXCEL_NAME
    df.to_excel(excel_path, sheet_name='AditiPhasePrecession', index=False)
    print(f'\nDone. {len(df)} unit(s) processed. Summary saved to {excel_path}')


if __name__ == '__main__':
    main()
