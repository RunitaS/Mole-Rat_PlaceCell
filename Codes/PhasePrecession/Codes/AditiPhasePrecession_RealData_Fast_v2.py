# -*- coding: utf-8 -*-
"""
Phase-precession pipeline for real Neuralynx recordings (.ntt spikes, .ncs
LFP, tracking .csv) -- single-file, optimized version of
AditiPhasePrecession_RealData.py + AditiPrecessionUtils.py.

Reference for the phase-precession analysis this reproduces:
    https://onlinelibrary.wiley.com/doi/10.1002/hipo.23641

WHY THIS FILE EXISTS (relative to the two-file original)
----------------------------------------------------------------------------
The original AditiPhasePrecession_RealData.py imported AditiPrecessionUtils
and called its functions unmodified. It was correct but very slow, for one
dominant reason plus a couple of secondary ones:

1. THE MAIN BOTTLENECK -- apu.getPhase(spkTimes, sinInput, time) called
   scipy.signal.find_peaks() on the ENTIRE filtered LFP trace on every
   single call. shuffle_precession_significance() calls this 500 times per
   unit (N_PRECESSION_SHUFFLES) to build the null distribution for the
   circular-linear fit, plus once more for the real spikes. For a
   multi-minute LFP trace at tens of kHz that's ~500x redundant
   whole-trace peak-finding per unit, repeated for every unit in every
   session -- the peak times never change between shuffles, only the spike
   times being tested against them do.
   FIX: theta_peak_times() is now computed ONCE per session, right after
   band-pass filtering, and reused for every unit and every shuffle via
   assign_theta_phase(spike_times, pk_times) -- the same searchsorted-based
   bracketing/interpolation apu.getPhase used, just decoupled from
   re-finding the peaks.

2. calcTMI() used plt.hist() purely to get histogram counts, creating a
   throwaway matplotlib Figure/Axes every call even when not plotting.
   FIX: calc_tmi() uses np.histogram() for the counts; ax.bar()/ax.plot()
   are only touched when an Axes is actually passed in to plot on.

3. Two correctness bugs found while porting (both in code paths this
   pipeline mostly avoids triggering, but fixed here since the file was
   being rewritten anyway):
     - fieldDetect's "if spkts:" branch used the truthiness of a numpy
       array, which raises ValueError for >1 elements. The original
       pipeline dodges this by never passing spkts. field_detect() here
       uses "if spk_ts is not None:" instead.
     - TempAutocorr's nested loop was `for j in range(1, len(spktimes))`
       with inner `for k in range(1, j+1)`, which never includes spike
       index 0 in any pair -- it's silently dropped from the
       autocorrelogram. temp_autocorr() below is a vectorized rewrite
       (per-spike searchsorted window instead of an O(n^2) double loop)
       that naturally includes every spike. This path is off by default
       (RUN_INTRINSIC_FREQ_ANALYSIS = False) and gated further by a spike
       count cap, so it wasn't the main slowdown, but it was an O(n^2)
       pure-Python double loop that would have been very slow if enabled
       on more than a few thousand spikes, so it's fixed too.

CROSS-CHECK AGAINST github.com/CINPLA/phase-precession
----------------------------------------------------------------------------
That repo's core.py (cl_regression / cl_corr, a direct translation of
Richard Kempter's original MATLAB) implements the same Kempter (2012)
circular-linear regression used here (circ_regress/lcc below) and the same
Jammalamadaka & SenGupta (1988) circular-circular correlation used here
(kemp_corr below, equivalent to their corrcc/corr_cc):

  CINPLA:  model(x,slope,phi0) = 2*pi*slope*x + phi0            [slope in CYCLES/x-unit]
           goodness(x,phase,slope) = |sum(exp(i*(phase - 2*pi*slope*x)))| / n
           fminbound(-goodness, min_slope, max_slope) -> best slope
           phi0 = atan2(sum(sin(resid)), sum(cos(resid)))
           circ_x = mod(2*pi*|slope|*x, 2*pi); corr = circ-circ corr(circ_x, phase)

  here:    alpha(m) = y - m*x                                    [m in RAD/x-unit]
           circ_r(alpha) = |sum(exp(i*alpha))| / n
           fminbound(-circ_r, ...) -> best slope m
           phase  = atan2(sum(sin(resid)), sum(cos(resid)))
           theta = mod(|m|*x, 2*pi); corr = kemp_corr(theta, y)

These are the SAME algorithm under the substitution m = 2*pi*slope_CINPLA
(i.e. this file's slope is already in radians per x-unit, so the 2*pi is
absorbed into m rather than applied separately) -- verified line-by-line
against core.py on 2026-09-13. No change was made to this regression's
math; only the redundant LFP peak-finding described above was removed.

Data layout:
ROOT_FOLDER is searched recursively; every folder that directly contains at
least one .ncs and at least one .ntt file is treated as a session.
    *.ncs   Neuralynx continuous (LFP) file, one per tetrode/channel. Each
            .ntt file is matched to the .ncs file sharing its embedded
            channel number (e.g. TT1.ntt -> CSC1.ncs, TT12.ntt -> CSC12.ncs)
            via match_ncs_to_ntt(), rather than every unit in the session
            being referenced to a single session-wide LFP channel.
    *.ntt   Neuralynx tetrode spike files, one file per already-isolated
            unit. Every .ntt file in the folder is processed.
    tracking .csv, auto-detected as the first .csv in the folder. Column
            order is fixed: col A = timestamp, col B = x (pixels, unused),
            col C = y (pixels, unused), col D = x (cm), col E = y (cm).

Two adaptations for real, variable-speed tracking (unchanged from the
original -- AditiPrecessionUtils' rate-map logic assumed a constant-
velocity, fixed-number-of-passes simulated rat):
  1. frRatemap's occupancy is a single constant (binsize/vel * numpasses,
     identical in every spatial bin -- see fr_ratemap's body below), so
     'vel' and 'numpasses' are estimated from the real trajectory (median
     running speed; count of low/high track-end crossings).
  2. field_detect() takes spk_ts=None here (its optional field-restricted
     timestamp output isn't needed by this pipeline).

Significance of the circ_regress fit (slope, corr) is tested via
shuffle_precession_significance -- a circular time-shift shuffle
(Climer, Newman & Hasselmo 2013-style): spike times are circularly shifted
against the tracking/LFP traces N_PRECESSION_SHUFFLES times, position and
theta phase are re-evaluated at the shifted times (same fixed field
boundaries, same renormalization), and refit with circ_regress. A cell is
'phase_precessing' if p_corr_shuffle < ALPHA and slope < 0, 'phase_recessing'
if p_corr_shuffle < ALPHA and slope > 0, else 'phase_locked'.

Requires: numpy, scipy, pandas, matplotlib, openpyxl. (No sklearn, shapely,
scikit-image or joblib -- those were only needed by AditiPrecessionUtils
functions this pipeline never called: GMMcluster, calcWithinPolygon,
isolateContours, linRegPrecession/linfit.)
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
from scipy.signal import find_peaks
from scipy.optimize import fminbound
from scipy.ndimage import gaussian_filter
from scipy.stats import circmean

# ============================================================================
# Configuration -- EDIT THESE
# ============================================================================

ROOT_FOLDER = Path(r"C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True")
OUTPUT_EXCEL_NAME = 'AditiPhasePrecession_Analyzed.xlsx'   # written to ROOT_FOLDER

TRACKING_TIME_UNIT = 'us'          # 'us', 'ms', or 's' -- units of the tracking timestamp column
LFP_THETA_BAND = (3.0, 7.0)        # Hz, band-pass applied to the LFP before theta-peak finding

POSBIN_SIZE_CM = 4.0               # spatial bin width for fr_ratemap's posbins
FIELD_THRESH = 0.1                 # field_detect's own 10%-of-peak threshold, passed through explicitly
MIN_SPEED_CM_S = 2.0               # tracking samples slower than this are treated as immobile when
                                    # estimating fr_ratemap's constant running 'vel' (see module docstring)
TRACK_END_QUANTILE = 0.1           # low/high-end quantile used to count track passes for 'numpasses'

MIN_SPIKES_FOR_TMI = 8             # calc_tmi needs a reasonable sample of phases to be meaningful
MIN_SPIKES_INFIELD_FOR_FIT = 20    # circ_regress accepts >=3, but a stable fit needs more
PPDIR = 0                          # circ_regress's ppdir: 0 = unconstrained slope sign (direction
                                    # reported afterward via its sign), -1 = constrain to negative
                                    # (classic precessing) slopes only, 1 = positive (recessing) only
MAXSLOPE = 'default'               # circ_regress's maxslope: 'default' allows <=720 deg of phase
                                    # change across the field; see its own comment

ALPHA = 0.05                       # significance threshold for the circular time-shift shuffle test
N_PRECESSION_SHUFFLES = 500        # circular time-shift shuffles testing the circ_regress fit (corr,
                                    # slope) against a null of no position-phase relationship
PRECESSION_MIN_SHIFT_FRAC = 0.1    # minimum circular shift, as a fraction of the tracking/LFP overlap
                                    # window's duration, so no shuffle leaves spikes nearly unshifted
RANDOM_SEED = 0                    # seed for the shuffle test's RNG, for reproducibility

RUN_INTRINSIC_FREQ_ANALYSIS = False   # temp_autocorr / precession_freq / theta_skip_amp are now
                                       # vectorized (searchsorted-windowed, not an O(n^2) double loop),
                                       # but still off by default to match the original pipeline's output
MAX_SPIKES_FOR_AUTOCORR = 3000        # skip the intrinsic-frequency block above this spike count

MOVING_AVG_WINLEN_CM = 5.0            # moving_avg_phase's winlen, passed through explicitly

NCS_SAMPLES_PER_RECORD = 512
HEADER_BYTES = 16 * 1024
DEFAULT_ADBITVOLTS = 0.000000195


# ============================================================================
# Small numerical utilities (formerly in AditiPrecessionUtils.py) -- only the
# functions this pipeline actually calls are kept
# ============================================================================

def remove_nans(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    keep = ~(np.isnan(x) | np.isnan(y))
    return x[keep], y[keep]


def gauss_kernel(n=11, sigma=1.0):
    r = np.arange(-(n // 2), n // 2 + 1)
    return np.exp(-(r.astype(float) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def gauss_smooth(y, n=11, sigma=1.0, return_nan=True):
    """NaN-preserving 1D Gaussian smoothing with edge-padding (apu.gaussSmooth,
    unchanged logic -- padding by len(kernel) on each side with the trailing
    values, convolving 'same', then cropping the padding back off)."""
    y = np.asarray(y, dtype=float)
    g = gauss_kernel(n=n, sigma=sigma)
    nanmask = np.isnan(y)
    if nanmask.any():
        y = np.nan_to_num(y)
    pad = len(g)
    yy = np.pad(y, (pad, pad), mode='edge')
    y2 = np.convolve(yy, g, mode='same')[pad:-pad]
    if return_nan and nanmask.any():
        y2 = y2.copy()
        y2[nanmask] = np.nan
    return y2


def get_isi(spike_ts):
    """ISI (ms) and coefficient of variation (apu.getISI, without the
    plotting option, which this pipeline never used)."""
    if np.any(np.isnan(spike_ts)):
        warnings.warn('CV will not be reliable as there are nans present in the data')
    isi = np.diff(spike_ts)
    if np.any((isi > 0) & (isi < 0.1)):
        isi = isi * 1000  # was in seconds; convert to ms
    cv = np.std(isi) / np.mean(isi)
    return isi, cv


def theta_peak_times(sin_input, time):
    """Times of local maxima ('peaks') of the theta-filtered LFP -- phase
    0/360 deg is defined at each peak (apu.getPhase's convention).
    Compute this ONCE per session and reuse it for every unit and every
    shuffle via assign_theta_phase() -- see the module docstring for why
    that matters."""
    idx, _ = find_peaks(sin_input)
    return time[idx]


def assign_theta_phase(spike_ts, pk_times):
    """Per-spike theta phase (deg), by linear interpolation between the
    bracketing entries of pk_times (must be sorted ascending). Vectorized
    equivalent of apu.getPhase's per-spike loop via searchsorted -- each
    spike is bracketed in O(log n_peaks) instead of rescanning all peaks."""
    spike_ts = np.asarray(spike_ts)
    if len(pk_times) == 0:
        return np.full(spike_ts.shape, np.nan)

    before_idx = np.clip(np.searchsorted(pk_times, spike_ts, side='right') - 1, 0, len(pk_times) - 1)
    after_idx = np.clip(np.searchsorted(pk_times, spike_ts, side='left'), 0, len(pk_times) - 1)
    pk_before = pk_times[before_idx]
    pk_after = pk_times[after_idx]
    interpk = pk_after - pk_before

    with np.errstate(invalid='ignore', divide='ignore'):
        phase = ((spike_ts - pk_before) / interpk) * 360.0

    inrange = (spike_ts >= pk_times[0]) & (spike_ts <= pk_times[-1])
    return np.where(inrange, phase, np.nan)


def calc_tmi(spk_phases, ax=None, findpeaks=False):
    """Theta Modulation Index (apu.calcTMI): 5 wrapped copies of the spike
    phases -> histogram -> Gaussian smooth -> take the centre 2 cycles (to
    avoid edge artifacts from smoothing across the wrap) -> max-normalize.
    TMI = 1 - min(normalized counts). Uses np.histogram instead of
    plt.hist() for the counts (identical counts, no throwaway Figure)."""
    spk_phases = np.asarray(spk_phases)
    numcycles = 5
    binwidth = 36
    phscopies = np.concatenate([spk_phases + j * 360 for j in range(numcycles)])

    edges = np.arange(0, numcycles * 360, binwidth)
    counts, edges = np.histogram(phscopies, bins=edges)
    smcounts = gauss_smooth(counts.astype(float), n=7, sigma=0.5)

    edges2 = edges[:-1]
    sel = (edges2 >= 360) & (edges2 <= 1080)
    edges2 = edges2[sel] - 360
    counts2 = smcounts[sel]
    normcounts = counts2 / np.max(counts2)
    tmi = 1 - np.min(normcounts)
    bincentres = edges2 + binwidth / 2

    if ax is not None:
        ax.clear()
        ax.bar(x=edges2, height=normcounts, width=np.diff(np.append(edges2, np.nan)),
               align='edge', fc='skyblue', ec='black')
        ax.set_title('TMI=' + str(round(tmi, 2)))
        ax.plot(bincentres, normcounts, linewidth=2)
        ax.set_xticks([0, 360, 720])
        ax.set_xlabel('degrees')

    if findpeaks:
        return tmi, edges2, bincentres, normcounts
    return tmi


def fr_ratemap(spk_pos, posbins, numpasses, vel):
    """1D firing-rate map, raw + Gaussian-smoothed (apu.frRatemap). Constant
    occupancy per bin (binsize/vel * numpasses) -- see module docstring for
    why this real-data pipeline estimates vel/numpasses rather than being
    given them as simulation parameters."""
    binsize = posbins[1] - posbins[0]
    occ = (binsize / vel) * numpasses

    spkmap = np.zeros(len(posbins))
    counts, _ = np.histogram(spk_pos, bins=posbins)
    spkmap[1:] = counts  # spkmap[0] stays 0, matching the original's loop starting at index 1

    occmap = np.full(len(posbins), occ, dtype=float)
    ratemap = spkmap / occmap
    gaussratemap = gauss_smooth(ratemap, n=4, sigma=1)
    return ratemap, gaussratemap, occmap


def field_detect(ratemap, posbins, spk_pos, spk_phase, spk_ts=None, thresh=0.1):
    """Place-field boundaries via a 10%-of-peak threshold, and the
    field-restricted, field-normalized spike position/phase (apu.fieldDetect).

    Fix vs. the original: its "if spkts:" branch used the truthiness of a
    numpy array, which raises ValueError for any array with >1 element --
    changed to "if spk_ts is not None:" here."""
    idx = np.where(ratemap >= thresh * np.max(ratemap))[0]
    normposbins = posbins[idx] - np.min(posbins[idx])
    normposbins = normposbins / np.max(normposbins)
    strtfld = posbins[idx.min()]
    endfld = posbins[idx.max()]

    posofpk = posbins[np.argmax(ratemap)]
    infield = (spk_pos >= strtfld) & (spk_pos <= endfld)
    x = spk_pos[infield]
    y = spk_phase[infield]

    x_ = x - np.min(x)
    x_ = x_ / np.max(x_)

    fielddata = {}
    fielddata['spkpos'], fielddata['spkphs'] = remove_nans(x, y)
    fielddata['normspkpos'], _ = remove_nans(x_, y)
    fielddata['fldstart'] = strtfld
    fielddata['fldend'] = endfld
    fielddata['fldcentre'] = posofpk
    fielddata['normfldcentre'] = normposbins[np.argmax(ratemap[idx])]
    fielddata['normfldstart'] = np.min(x_)
    fielddata['normfldend'] = np.max(x_)

    if spk_ts is not None:
        t = spk_ts[infield]
        _, fielddata['spkts'] = remove_nans(x, t)

    return fielddata


# ---- Circular-linear regression (Kempter 2012) -- see module docstring for
# the line-by-line cross-check against github.com/CINPLA/phase-precession ----

def circ_r(alpha):
    """Mean resultant vector length of angles alpha (unweighted; apu.circ_r
    with its default w=None, d=None)."""
    return np.abs(np.sum(np.exp(1j * alpha))) / len(alpha)


def _cl_cost(slope, x, y):
    return -circ_r(y - slope * x)


def kemp_corr(theta, t):
    """Circular-circular correlation, Jammalamadaka & SenGupta (1988) /
    Kempter et al. (2012) eq. 3 (apu.kempCorr) -- equivalent to
    pycircstat.corrcc / CINPLA/phase-precession's cl_corr."""
    t_bar = circmean(t)
    theta_bar = circmean(theta)
    num = np.sum(np.sin(t - t_bar) * np.sin(theta - theta_bar))
    den = np.sqrt(np.sum(np.sin(t - t_bar) ** 2) * np.sum(np.sin(theta - theta_bar) ** 2))
    return num / den


def lcc(x, y, max_slope, con):
    """Bounded 1D optimization of the slope m maximizing circ_r(y - m*x)
    (apu.lcc). See module docstring for the correspondence with
    CINPLA/phase-precession's cl_regression (same algorithm, slope
    convention differs by a factor of 2*pi)."""
    slope, fval, _err, _numfnc = fminbound(_cl_cost, con[0] * max_slope, con[1] * max_slope,
                                            args=(x, y), full_output=True)
    resid = y - slope * x
    S = np.sum(np.sin(resid))
    C = np.sum(np.cos(resid))
    phase = np.arctan2(S, C)

    theta = np.mod(np.abs(slope) * x, 2 * np.pi)
    corr = kemp_corr(theta, y)

    return {'slope_opt': slope, 'fval_opt': fval, 'phase_opt': phase, 'corr': corr}


def circ_regress(x, y, ppdir, maxslope='default'):
    """Circular-linear regression of phase y (rad, in [0, 2*pi)) against
    position x (apu.circRegress). ppdir constrains the fitted slope's sign
    (0 = unconstrained, -1/+1 = negative/positive only); maxslope bounds how
    much phase change is allowed across the full span of x. (apu.circRegress
    took an unused 'k' parameter, always called with k=None -- dropped here.)"""
    assert np.all(y < 2 * np.pi) and np.all(y >= 0), 'Angular values outside the allowed range [0, 2pi).'
    if len(x) < 3:
        return {'slope_opt': np.nan, 'phase_opt': np.nan, 'fval_opt': np.nan, 'corr': np.nan}

    eps = np.finfo(float).eps
    if ppdir == -1:
        con = [-1, eps]
    elif ppdir == 1:
        con = [eps, -1]
    else:
        con = [-1, 1 - eps]

    x_range = np.max(x) - np.min(x)
    if maxslope == 'default':
        max_slope = (4 * np.pi) / x_range      # allows up to 720 deg of phase change across the field
    elif maxslope == 'greater':
        max_slope = (12 * np.pi) / x_range
    elif maxslope == 'lesser':
        max_slope = (2 * np.pi) / x_range
    else:
        raise ValueError(f"unknown maxslope option: {maxslope!r}")

    return lcc(x, y, max_slope, con)


def find_phase_valley(y, ncopies, bins=50):
    """Trough of the (wrapped) spike-phase distribution (apu.findPhaseValley)."""
    y_ = np.concatenate([y + j * 360 for j in range(ncopies)])
    counts, edges = np.histogram(y_, bins=bins)
    return edges[np.argmin(counts)]


# ---- Plotting helpers (apu.plotPrecession / plotSpikeDensity / movingavgPhase),
# rewritten to take an explicit Axes instead of relying on pyplot's global
# "current axes" state (plt.sca) ----

def plot_precession(ax, x, y, cms=False):
    if cms and np.any(x < 2):
        x = x * 100
        ax.set_xlabel('cms')
    else:
        ax.set_xlabel('m')
    ax.scatter(x, y)
    ax.set_ylim([0, 720])
    ax.set_yticks([0, 360, 720])
    ax.set_ylabel('degrees')


def plot_spike_density(ax, x, y, xedges, yedges, gauss_sigma=1, cm=True):
    if cm and np.any(x < 2) and np.any(xedges < 2):
        x = x * 100
        xedges = xedges * 100
    H, _, _ = np.histogram2d(x, y, bins=(xedges, yedges))
    H = gaussian_filter(H, sigma=gauss_sigma)
    xmesh, ymesh = np.meshgrid(xedges, yedges)
    ax.pcolormesh(xmesh, ymesh, H.T, cmap='jet')
    ax.set_xlabel('metres')
    ax.set_ylabel('phase')
    ax.set_yticks([0, 360, 720])
    ax.set_title('Spike Density')
    return H


def moving_avg_phase(ax, x, y, winlen=5, overlap=0.5, sm_sigma=False):
    if np.any(x < 2):
        x = x * 100
    step = winlen * (1 - overlap)
    xbins = np.arange(np.min(x), np.max(x), step)
    movingavg, xxbins = [], []
    for j in xbins:
        idx = (x > j) & (x < j + winlen)
        if np.sum(idx) > 5:
            xxbins.append(np.mean(x[idx]))
            movingavg.append(circmean(y[idx], high=360, nan_policy='omit'))

    ax.scatter(x, y, c='k')
    ax.plot(xxbins, movingavg, c='r', lw=2)
    ax.set_xlabel('cms')
    ax.set_ylabel('degrees')
    if sm_sigma and len(movingavg) > 0:
        smav = gauss_smooth(np.array(movingavg), n=7, sigma=sm_sigma)
        ax.plot(xxbins, smav, '--')
    return xxbins, movingavg


# ---- Optional intrinsic-frequency block (apu.TempAutocorr / precessionFreq /
# thetaskipAmp), off by default -- RUN_INTRINSIC_FREQ_ANALYSIS. Rewritten as a
# per-spike searchsorted-window + np.add.at instead of an O(n^2) double loop;
# see module docstring for the off-by-one bug this also fixes (the original
# loop never included spike index 0 in any pair). ----

def temp_autocorr(spktimes):
    """+-1000 ms, 20 ms-bin autocorrelogram from continuous spike times (s)."""
    spktimes = np.sort(np.asarray(spktimes))
    bins = np.arange(-1000, 1020, 20)
    counts = np.zeros(len(bins))
    window_s = 1.0
    for i in range(len(spktimes)):
        lo = np.searchsorted(spktimes, spktimes[i] - window_s, side='left')
        hi = np.searchsorted(spktimes, spktimes[i] + window_s, side='right')
        diffs_ms = (spktimes[lo:hi] - spktimes[i]) * 1000.0
        idx = np.clip(((diffs_ms + 1000) / 20).astype(int), 0, len(bins) - 1)
        np.add.at(counts, idx, 1)
    smoothcount = gauss_smooth(counts, n=7, sigma=1)
    return {'bins': bins, 'counts': smoothcount, 'isi': np.diff(spktimes)}


def precession_freq(spktimes):
    """Intrinsic firing frequency from the +-500 ms, 2 ms-bin autocorrelogram
    peak (apu.precessionFreq), zero-lag peak removed before peak-finding."""
    spktimes = np.sort(np.asarray(spktimes))
    bins = np.arange(-500, 503, 2)
    counts = np.zeros(len(bins))
    window_s = 0.5
    for i in range(len(spktimes)):
        lo = np.searchsorted(spktimes, spktimes[i] - window_s, side='left')
        hi = np.searchsorted(spktimes, spktimes[i] + window_s, side='right')
        diffs_ms = (spktimes[lo:hi] - spktimes[i]) * 1000.0
        idx = np.clip(((diffs_ms + 500) / 2).astype(int), 0, len(bins) - 1)
        np.add.at(counts, idx, 1)
    counts[(bins >= -50) & (bins <= 50)] = 0  # remove the centre peak +-50ms, as in the original
    smoothcount = gauss_smooth(counts, n=11, sigma=2)
    idx, _ = find_peaks(smoothcount)
    return 1000.0 / np.sort(np.abs(bins[idx]))[0]


def theta_skip_amp(tempcorr):
    """Theta-skipping amplitude: 2nd autocorrelogram peak (~250ms) over the
    1st (~125ms) (apu.thetaskipAmp)."""
    amp = tempcorr['counts']
    bins = tempcorr['bins']
    firstpk = np.max(amp[(bins > 75) & (bins < 150)])
    secondpk = np.max(amp[(bins > 200) & (bins < 300)])
    return secondpk / firstpk


# ============================================================================
# Neuralynx / tracking file I/O -- unchanged from the original
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


def bandpass_filter(data, low, high, fs, order=3):
    """Zero-phase Butterworth band-pass (SOS form), used only to turn the raw
    LFP into the oscillatory signal theta_peak_times finds peaks in."""
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
    """Estimate the two real-data stand-ins fr_ratemap needs for its single
    occupancy constant: 'vel' is the median running speed over
    above-threshold samples, 'numpasses' is the number of low/high
    track-end crossings (a simple threshold-crossing lap count)."""
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


def shuffle_precession_significance(spk_ts, pos_ts, pos_x, pk_times, lfp_t_lo, lfp_t_hi,
                                     fldstart, fldend, observed_corr, observed_slope_deg,
                                     ppdir, maxslope, rng,
                                     n_shuffles=N_PRECESSION_SHUFFLES,
                                     min_shift_frac=PRECESSION_MIN_SHIFT_FRAC):
    """Null distribution for the circ_regress circular-linear fit, via a
    circular time-shift shuffle (Climer/Newman/Hasselmo-style).

    Spike times are shifted by a random offset -- at least min_shift_frac of
    the tracking/LFP overlap window -- and wrapped within that window.
    Position (spk_pos_x) and theta phase (assign_theta_phase, against the
    SAME pk_times computed once for the whole session -- this is the fix
    for the redundant find_peaks() calls the original pipeline made here)
    are re-evaluated at the shifted times, restricted to the same fixed
    field boundaries, renormalized exactly as field_detect does, and refit
    with circ_regress.

    Returns (p_corr, p_slope, shuffle_corrs, shuffle_slopes)."""
    t_lo = max(pos_ts.min(), lfp_t_lo)
    t_hi = min(pos_ts.max(), lfp_t_hi)
    duration = t_hi - t_lo
    min_shift = min_shift_frac * duration

    shuffle_corrs = np.full(n_shuffles, np.nan)
    shuffle_slopes = np.full(n_shuffles, np.nan)
    for i in range(n_shuffles):
        shift = rng.uniform(min_shift, duration - min_shift)
        shifted_ts = t_lo + np.mod(spk_ts - t_lo + shift, duration)

        shifted_pos = spk_pos_x(pos_ts, pos_x, shifted_ts)
        shifted_phase_deg = assign_theta_phase(shifted_ts, pk_times)

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
            reg_i = circ_regress(x_norm, y_rad, ppdir, maxslope=maxslope)
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
# Per-unit analysis
# ============================================================================

def analyze_unit(spk_ts, pos_ts, pos_x, pk_times, lfp_t_lo, lfp_t_hi, posbins, vel, numpasses,
                  unit_label, output_dir: Path, rng):
    """Run the full analysis battery for one unit. Returns a result dict for
    the summary table; writes one combined summary figure."""
    row = dict(n_spikes_total=len(spk_ts))

    # ---- ISI / CV ----
    _isi, cv = get_isi(spk_ts)
    row['ISI_CV'] = float(cv)

    # ---- theta phase per spike (against the session's precomputed peak times) ----
    spkphase = assign_theta_phase(spk_ts, pk_times)
    valid_phase = ~np.isnan(spkphase)
    row['n_spikes_phase'] = int(np.sum(valid_phase))

    # ---- Theta Modulation Index ----
    row['TMI'] = np.nan
    if row['n_spikes_phase'] >= MIN_SPIKES_FOR_TMI:
        row['TMI'] = float(calc_tmi(spkphase[valid_phase]))

    # ---- place field + phase precession ----
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
    _ratemap, gaussratemap, _occmap = fr_ratemap(spk_x, posbins, numpasses, vel)

    try:
        fielddata = field_detect(gaussratemap, posbins, spk_x, spkphase, thresh=FIELD_THRESH)
    except Exception as exc:
        row['PrecessionSkippedReason'] = f'field_detect ERROR ({exc})'
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
        regData = circ_regress(normx, y_rad, PPDIR, maxslope=MAXSLOPE)
    except Exception as exc:
        row['PrecessionSkippedReason'] = f'circ_regress ERROR ({exc})'
        return row

    slope_rad = regData['slope_opt']
    slope_deg = float(np.rad2deg(slope_rad))
    row['PrecessionTested'] = True
    row['CircCorr'] = float(regData['corr'])
    row['Slope_deg_per_field'] = slope_deg
    row['PhaseOffset_deg'] = float(np.rad2deg(regData['phase_opt']) % 360)
    row['PhaseValley_deg'] = float(find_phase_valley(phs_deg, ncopies=2))

    # ---- significance via circular time-shift shuffle ----
    p_corr_shuffle, p_slope_shuffle, _shuffle_corrs, _shuffle_slopes = shuffle_precession_significance(
        spk_ts, pos_ts, pos_x, pk_times, lfp_t_lo, lfp_t_hi, row['FieldStart_cm'], row['FieldEnd_cm'],
        row['CircCorr'], slope_deg, PPDIR, MAXSLOPE, rng)
    row['p_corr_shuffle'] = p_corr_shuffle
    row['p_slope_shuffle'] = p_slope_shuffle

    is_significant_precession = bool(np.isfinite(p_corr_shuffle) and p_corr_shuffle < ALPHA)
    # Negative slope: theta-phase precessing (spike phase advances to earlier
    # phase over the field). Positive slope: theta-phase recessing (phase
    # moves later). Neither significant: phase locked.
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

    # ---- optional intrinsic-frequency block ----
    row['IntrinsicFreq_Hz'] = np.nan
    row['ThetaSkipAmp'] = np.nan
    if RUN_INTRINSIC_FREQ_ANALYSIS and len(spk_ts) <= MAX_SPIKES_FOR_AUTOCORR:
        tempcorr = temp_autocorr(spk_ts)
        row['IntrinsicFreq_Hz'] = float(precession_freq(spk_ts))
        row['ThetaSkipAmp'] = float(theta_skip_amp(tempcorr))

    _plot_unit_summary(phs_deg, normx, spkphase[valid_phase], fielddata, regData,
                        slope_deg, row, unit_label, output_dir)
    return row


def _plot_unit_summary(phs_deg, normx, all_valid_phase_deg, fielddata, regData,
                        slope_deg, row, unit_label, output_dir: Path):
    """One combined figure per unit: TMI histogram, precession scatter with
    fitted line, spike-density heatmap, moving-average phase vs. position.
    Only called once row['n_spikes_phase'] has already cleared
    MIN_SPIKES_INFIELD_FOR_FIT (>= MIN_SPIKES_FOR_TMI), so calc_tmi always
    has enough spikes here."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(unit_label, fontsize=12, fontweight='bold')

    calc_tmi(all_valid_phase_deg, ax=axes[0, 0])

    x_two = np.concatenate([normx, normx])
    y_two = np.concatenate([phs_deg, phs_deg + 360])
    plot_precession(axes[0, 1], x_two, y_two, cms=False)
    xg = np.linspace(0, 1, 200)
    phi_deg = np.rad2deg(np.mod(regData['slope_opt'] * xg + regData['phase_opt'], 2 * np.pi))
    axes[0, 1].plot(xg, phi_deg, 'r', lw=2)
    axes[0, 1].plot(xg, phi_deg + 360, 'r', lw=2)
    axes[0, 1].set_xlabel('normalized field position')
    axes[0, 1].set_title(f'rho={row["CircCorr"]:.2f}  p_shuffle={row["p_corr_shuffle"]:.3g}  '
                          f'slope={slope_deg:.1f} deg/field ({row["PrecessionClass"]})', fontsize=10)

    x_field_two = np.concatenate([fielddata['spkpos'], fielddata['spkpos']])
    plot_spike_density(axes[1, 0], x_field_two, y_two,
                        xedges=np.linspace(fielddata['fldstart'], fielddata['fldend'], 21),
                        yedges=np.arange(0, 730, 10), gauss_sigma=1, cm=True)
    axes[1, 0].set_xlabel('position (cm)')

    moving_avg_phase(axes[1, 1], fielddata['spkpos'], fielddata['spkphs'],
                      winlen=MOVING_AVG_WINLEN_CM, overlap=0.5, sm_sigma=1.5)
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

    try:
        tracking_path = _find_tracking_file(data_folder)
        print(f'  Using tracking file: {tracking_path.name}')
        pos_ts, pos_xy = load_tracking(tracking_path, TRACKING_TIME_UNIT)
        pos_x = pos_xy[:, 0]  # x (cm) used directly as the 1D track position
    except FileNotFoundError as exc:
        print(f'  {exc} -- skipping this session (need position for fr_ratemap/field_detect/circ_regress).')
        return []

    vel, numpasses = estimate_vel_and_numpasses(pos_ts, pos_x)
    lo = np.floor(pos_x.min() / POSBIN_SIZE_CM) * POSBIN_SIZE_CM
    hi = np.ceil(pos_x.max() / POSBIN_SIZE_CM) * POSBIN_SIZE_CM
    posbins = np.arange(lo, hi + POSBIN_SIZE_CM, POSBIN_SIZE_CM)
    print(f'  vel={vel:.1f} cm/s, numpasses={numpasses}, posbins={len(posbins)} x '
          f'{POSBIN_SIZE_CM:.1f} cm')

    session_label = '_'.join(data_folder.parts[-3:])
    ntt_files = sorted(data_folder.glob('*.ntt'), key=_natural_key)

    # Cache per matched .ncs file so tetrodes that happen to share one (or a
    # session with a single .ntt) don't reload/refilter it twice.
    lfp_cache: dict[Path, tuple] = {}

    rows = []
    for ntt_path in ntt_files:
        print(f'  Processing: {ntt_path.name}')

        try:
            theta_ncs = match_ncs_to_ntt(ntt_path, ncs_files)
        except FileNotFoundError as exc:
            print(f'    {exc} -- skipping this tetrode.')
            continue

        if theta_ncs not in lfp_cache:
            print(f'    Using LFP file: {theta_ncs.name}')
            lfp_sig, lfp_ts, lfp_fs = load_ncs(theta_ncs)
            filtered_theta = bandpass_filter(lfp_sig, LFP_THETA_BAND[0], LFP_THETA_BAND[1], lfp_fs)
            # Computed ONCE per matched LFP channel and reused for every unit
            # on this tetrode and every shuffle below -- this is the main
            # performance fix relative to the original pipeline (see module
            # docstring).
            pk_times = theta_peak_times(filtered_theta, lfp_ts)
            lfp_t_lo, lfp_t_hi = float(lfp_ts.min()), float(lfp_ts.max())
            lfp_cache[theta_ncs] = (pk_times, lfp_t_lo, lfp_t_hi)
        pk_times, lfp_t_lo, lfp_t_hi = lfp_cache[theta_ncs]

        units = load_ntt_spike_times(ntt_path)
        for cell_number, spk_ts in units.items():
            unit_label = f'{ntt_path.stem}_cell{cell_number}' if len(units) > 1 else ntt_path.stem
            row = dict(Session=session_label, FolderPath=str(data_folder), Unit=unit_label,
                       ntt_file=ntt_path.name, cell_number=cell_number, lfp_file=theta_ncs.name)
            try:
                unit_row = analyze_unit(spk_ts, pos_ts, pos_x, pk_times, lfp_t_lo, lfp_t_hi, posbins,
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

    columns = ['Session', 'FolderPath', 'Unit', 'ntt_file', 'lfp_file', 'cell_number', 'n_spikes_total',
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
