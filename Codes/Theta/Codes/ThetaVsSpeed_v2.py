"""
Theta frequency/power vs. running speed.

Recursively finds every Neuralynx .ncs LFP file under ROOT_DIR, pairs it with
the tracking .csv in its own folder, cleans the LFP the same way as
ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean_v3.py (resample -> notch ->
detrend -> low-pass), then band-pass filters to theta and estimates
instantaneous frequency/power from the Generalized Phase (GP) of the
band-passed signal (Davis, Muller et al. 2020, Nature 587:432-436 -- the same
corrected analytic-signal phase used in Ref_ThetaSpeed.py, ported here
verbatim), binned against running speed computed from the cleaned tracking
position.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d, label
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import openpyxl


# %% ==================== Configuration ===========================================

# Root directory to search recursively for .ncs files. Every .ncs found
# anywhere under this tree (in any subfolder) is processed, paired with the
# single tracking .csv that lives in the same folder.
ROOT_DIR = r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True'

OUTPUT_XLSX = os.path.join(ROOT_DIR, 'ThetaVsSpeed_out.xlsx')

# Speed vs. theta stats (mixed-model plots + summary) are written here.
OUTPUT_DIR = Path(ROOT_DIR) / 'Output_ThetaVsSpeed'

# ---- Acquisition (Neuralynx .ncs) ----
NATIVE_FS = 32000  # native .ncs sampling rate (Hz)
ADBitVolts = 0.000003051757812500000169  # V per ADC count (Neuralynx header)

# Neuralynx .ncs record format (512 int16 samples per record, 16 kB header skipped)
ncs_dtype = np.dtype([
    ('timestamp'  , '<u8'),
    ('sc_number'  , '<u4'),
    ('cell_number', '<u4'),
    ('params'     , '<u4'),
    ('samples'    , '<i2', (512,)),
])

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
FS_LFP = 250

# ---- Theta band-pass (applied after the general LFP cleanup, on top of it) ----
BANDPASS_CUTOFF = (3.0, 7.0)
FILTER_ORDER = 5

# ---- Edge trimming (seconds) -- drops filtfilt edge artifacts from the
# start/end of each cleaned trace ----
TRIM_DUR = 0.25

BINSIZE = 0.25

# ---- Tracking position cleaning (matches
# ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean_v3.py's
# _smooth_tracking_position): frame-to-frame jumps implying a speed above
# POS_JUMP_THRESH_CMS are tracking artifacts, linearly interpolated over,
# then the x/y traces are Gaussian-smoothed with sigma POS_SMOOTH_SIGMA_SMP
# (in samples). Tracking .csv column layout: UNIX timestamp in us (col A),
# x in cm (col D), y in cm (col E).
POS_JUMP_THRESH_CMS = 80.0
POS_SMOOTH_SIGMA_SMP = 1.0


# %% ==================== .ncs / tracking data import ==============================

def load_ncs(fpath):
    """Read a Neuralynx .ncs file.

    Returns
    -------
    lfp             : ndarray  raw trace in microvolts (uV)
    start_timestamp : int      UNIX timestamp (us) of the first sample -- same
                                clock as the tracking .csv.
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
    b, a = signal.butter(filter_order, bandpass_cutoff, btype="band", fs=fs)
    filtered = signal.filtfilt(b, a, lfp)
    return filtered


def trim_signal(sig, fs, trim_dur):
    trim_samples = int(trim_dur * fs)
    return sig[trim_samples:-trim_samples]


# ============================================================================
# Generalized Phase (Davis, Muller et al. 2020, Nature 587:432-436; Muller et
# al. 2016, eLife 5:e17267) -- corrected replacement for the plain Hilbert-
# transform phase. Ported verbatim from Ref_ThetaSpeed.py's
# generalized_phase_vector (itself ported from generalized_phase_vector.m,
# https://github.com/mullerlab/generalized-phase), which corrects the plain
# analytic-signal phase for epochs where instantaneous frequency collapses or
# reverses sign, instead of the peak-to-peak/trough-to-trough interpolation
# previously used here to assign instantaneous frequency between cycles.
# ============================================================================

def _gp_rewrap(xp):
    """rewrap.m: fold an unwrapped phase trace back into (-pi, pi]."""
    return xp - 2 * np.pi * np.floor((xp - np.pi) / (2 * np.pi)) - 2 * np.pi


def generalized_phase_vector(x, fs, lp, nwin=3):
    """Generalized Phase of a single real-valued time series
    (generalized_phase_vector.m). Drop-in replacement for
    np.angle(signal.hilbert(x)) via np.angle(xgp): corrects the analytic
    signal's phase for epochs where the instantaneous frequency falls below
    `lp`, which otherwise corrupt the plain Hilbert phase with spurious
    reversed-phase assignments.

    Parameters
    ----------
    x  : 1D real array, already bandpass-filtered over the band of interest
         (the same band whose low edge is passed as `lp`).
    fs : sampling rate of x, Hz.
    lp : low-frequency cutoff -- the lower edge of the bandpass filter that
         produced x. Instantaneous frequency below this value marks a
         phase-slip epoch to be corrected.
    nwin : safety-window multiplier extending each detected phase-slip
         epoch (generalized_phase_vector.m default: 3).

    Returns
    -------
    xgp : complex analytic signal with the corrected ("generalized") phase.
    wt  : instantaneous frequency estimate (Hz).
    idx : boolean mask, True where the sample fell inside a phase-slip
          epoch and its phase in `xgp` was therefore *reconstructed*
          (pchip-filled across a gap whose true cumulative cycle count is
          not preserved -- see the note above _detect_phase_slip below)
          rather than measured. Safe to read xgp's *wrapped* phase
          (np.angle(xgp)) at these samples; not safe to use them for
          anything depending on cumulative/unwrapped phase, such as
          instantaneous frequency.
    """
    x = np.asarray(x, dtype=np.float64)
    npts = x.shape[0]
    dt = 1.0 / fs

    def _inst_freq(xo):
        wt = np.zeros(npts)
        wt[:-1] = np.angle(xo[1:] * np.conj(xo[:-1])) / (2 * np.pi * dt)
        return wt

    xo = signal.hilbert(x)
    ph = np.angle(xo)
    md = np.abs(xo)
    wt = _inst_freq(xo)

    # rectify rotation direction so instantaneous frequency is positive
    finite_wt = wt[np.isfinite(wt)]
    sign_if = np.sign(np.mean(finite_wt)) if finite_wt.size else 1.0
    if sign_if == -1:
        xo = md * np.exp(1j * (sign_if * ph))
        ph = np.angle(xo)
        md = np.abs(xo)
        wt = _inst_freq(xo)

    if np.all(np.isnan(ph)):
        return np.full(npts, np.nan, dtype=np.complex128), wt, np.ones(npts, dtype=bool)

    # find negative-/low-frequency ("phase slip") epochs and extend each by
    # nwin x its own width
    idx = wt < lp
    idx[0] = False
    labeled, n_groups = label(idx)
    for kk in range(1, n_groups + 1):
        idxs = np.flatnonzero(labeled == kk)
        start, stop = idxs[0], idxs[-1]
        extended_stop = min(start + (stop - start) * nwin, npts - 1)
        idx[start:extended_stop + 1] = True

    # unwrap only the trustworthy (unflagged) samples, then reconstruct the
    # flagged samples by shape-preserving (pchip) interpolation of that
    # trustworthy unwrapped trend, and rewrap.
    valid = ~idx
    if np.count_nonzero(valid) < 2:
        return np.full(npts, np.nan, dtype=np.complex128), wt, np.ones(npts, dtype=bool)

    valid_positions = np.flatnonzero(valid)
    p_valid_unwrapped = np.unwrap(ph[valid])

    p = np.empty(npts, dtype=np.float64)
    p[valid] = p_valid_unwrapped
    invalid_positions = np.flatnonzero(idx)
    if invalid_positions.size:
        filler = PchipInterpolator(valid_positions, p_valid_unwrapped, extrapolate=True)
        p[invalid_positions] = filler(invalid_positions)

    p = _gp_rewrap(p)

    xgp = md * np.exp(1j * p)
    return xgp, wt, idx


def estimate_instantaneous_frequency_power(theta, fs, lowcut, highcut):
    """Per-sample instantaneous theta frequency (Hz) and power (uV^2) from
    the Generalized Phase of the band-passed signal `theta` -- the corrected
    analytic-signal phase itself, not an interpolation of peak-to-peak /
    trough-to-trough intervals.

    Frequency is the *consecutive-sample* phase advance of the corrected
    signal (the same bounded estimator generalized_phase_vector uses
    internally for its own `wt`, applied here to `xgp`), not the derivative
    of the fully unwrapped phase -- differentiating a globally unwrapped
    phase would compound errors across the whole trace.

    Critically, samples that generalized_phase_vector had to *reconstruct*
    (its `idx` mask: a phase-slip epoch, pchip-filled across a gap whose
    true cumulative cycle count is not preserved -- see its own docstring)
    are excluded here entirely, not merely sign-clipped: xgp's phase there
    was fabricated to give a plausible *wrapped* value, not a plausible
    *rate of change*, so a per-sample frequency computed from it (of either
    sign, and of any magnitude -- including values that land back inside
    the passband by chance) is not physically meaningful. A frequency
    estimate spans two samples (i, i+1), so either endpoint being
    reconstructed invalidates it. As a final sanity net, any surviving
    estimate outside the bandpass filter's own [lowcut, highcut] range is
    also dropped -- even a "trustworthy" sample can't produce a frequency
    outside the band that was filtered into `theta` in the first place; a
    value out there means the analytic-signal estimate itself is
    unreliable (e.g. right at the filter's transition band). Power is the
    squared GP amplitude envelope, defined at every sample of `theta` with
    no interpolation needed."""
    xgp, _wt, idx = generalized_phase_vector(theta, fs, lowcut)
    dt = 1.0 / fs
    instantaneous_freq = np.full(len(xgp), np.nan)
    instantaneous_freq[:-1] = np.angle(xgp[1:] * np.conj(xgp[:-1])) / (2 * np.pi * dt)
    reconstructed = idx[:-1] | idx[1:]
    instantaneous_freq[:-1][reconstructed] = np.nan
    with np.errstate(invalid='ignore'):
        out_of_band = (instantaneous_freq < lowcut) | (instantaneous_freq > highcut)
    instantaneous_freq[out_of_band] = np.nan
    instantaneous_power = np.abs(xgp) ** 2
    return instantaneous_freq, instantaneous_power


def bin_data(data, ts, bins):
    bin_centers = (bins[:-1] + bins[1:])/2
    data_binned = []
    for i in range(len(bins)-1):
        bin_start = bins[i]
        bin_stop = bins[i+1]
        # speed values within the bin
        idx = np.where((ts>=bin_start)&(ts<=bin_stop))[0]
        # nanmedian: instantaneous_freq masks out biologically-impossible
        # (non-positive) samples as NaN -- a bin should fall back to
        # whichever of its samples are still valid rather than being
        # discarded outright over a few masked ones.
        with np.errstate(invalid='ignore'):
            data_binned.append(np.nanmedian(data[idx]) if len(idx) else np.nan)
    return data_binned, bin_centers


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
    """
    raw_lfp, lfp_start_us = load_ncs(fpath)
    lfp = clean_lfp(raw_lfp)
    theta = filter_signal(lfp, FS_LFP, BANDPASS_CUTOFF, FILTER_ORDER)

    lfp_ts = lfp_start_us / 1e6 + np.arange(len(theta)) / FS_LFP  # absolute UNIX time (s)

    theta = trim_signal(theta, FS_LFP, TRIM_DUR)
    lfp_ts = trim_signal(lfp_ts, FS_LFP, TRIM_DUR)

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

    inst_freq, inst_power = estimate_instantaneous_frequency_power(
        theta, FS_LFP, BANDPASS_CUTOFF[0], BANDPASS_CUTOFF[1])

    bin_start = lfp_ts.min().round()
    bin_stop = lfp_ts.max().round()
    bins = np.arange(bin_start, bin_stop + BINSIZE, step=BINSIZE)

    inst_speed_binned, _ = bin_data(inst_speed, speed_ts, bins)
    inst_freq_binned, _ = bin_data(inst_freq, lfp_ts, bins)
    inst_power_binned, bin_centers = bin_data(inst_power, lfp_ts, bins)

    inst_freq_binned, inst_power_binned, inst_speed_binned, bin_centers = drop_nan(
        np.asarray(inst_freq_binned), np.asarray(inst_power_binned),
        np.asarray(inst_speed_binned), np.asarray(bin_centers))

    return pd.DataFrame({
        "Power": inst_power_binned,
        "Frequency": inst_freq_binned,
        "Speed": inst_speed_binned,
        "Time": bin_centers,
    })


def process_root_directory(root_dir):
    """Recursively find every .ncs under `root_dir`, run process_ncs_session
    on each, and concatenate the per-bin results, tagging each row with a
    session id derived from the file's path (relative to `root_dir`)."""
    ncs_files = []
    for root, _dirs, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith('.ncs'):
                ncs_files.append(os.path.join(root, fname))
    print(f'Found {len(ncs_files)} .ncs files under {root_dir}')

    all_results = []
    for fpath in ncs_files:
        rel = os.path.relpath(fpath, root_dir)
        ses_id = os.path.splitext(rel)[0].replace(os.sep, '_')
        try:
            df_session = process_ncs_session(fpath)
            df_session['Session'] = ses_id
            print(f'  OK: {rel}  [{len(df_session)} bins]')
            all_results.append(df_session)
        except Exception as e:
            print(f'  SKIP: {rel} -- {e}')

    print(f'-> {len(all_results)} files processed\n')
    if not all_results:
        raise ValueError(f'No files processed successfully under {root_dir}')
    return pd.concat(all_results, ignore_index=True)


# %% ==================== Speed vs. theta statistics (mixed linear model) ==========

def fit_mixed_model(df, response, predictor="Speed", group="Session"):
    """
    Random-intercept linear mixed model: response ~ predictor, grouped by
    session, so repeated bins from the same session don't count as
    independent observations.
    """
    model = smf.mixedlm(f"{response} ~ {predictor}", df, groups=df[group])
    return model.fit()


def plot_mixed_model_fit(df, result, response, predictor, ylabel, title, color, out_path):
    """Scatter the binned data plus the mixed model's fixed-effect fit and its 95% CI band."""
    intercept = result.params["Intercept"]
    slope = result.params[predictor]
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

    plt.figure(figsize=(7, 5))
    plt.scatter(df[predictor], df[response], color=color, edgecolor='k', alpha=0.6, s=25)
    plt.plot(x_range, y_pred, color="black", linewidth=2, label="Mixed Linear Fit")
    plt.fill_between(x_range, y_ci_low, y_ci_high, color="grey", alpha=0.5, label="95% CI")
    plt.xlabel(f"{predictor} (cm/s)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_speed_vs_theta_stats(df):
    """Fit and plot mixed linear models of Frequency~Speed and Power~Speed (random intercept per Session)."""
    df_stats = df.dropna(subset=["Frequency", "Power", "Speed"])

    freq_result = fit_mixed_model(df_stats, "Frequency")
    print(freq_result.summary())
    plot_mixed_model_fit(
        df_stats, freq_result, "Frequency", "Speed",
        ylabel="Theta Frequency [Hz]",
        title="Theta Frequency vs. Speed (Mixed Model)",
        color="red", out_path=OUTPUT_DIR / "ThetaFrequency_vs_Speed_MixedModel.png")

    power_result = fit_mixed_model(df_stats, "Power")
    print(power_result.summary())
    plot_mixed_model_fit(
        df_stats, power_result, "Power", "Speed",
        ylabel="Theta Power [µV²]",
        title="Theta Power vs. Speed (Mixed Model)",
        color="royalblue", out_path=OUTPUT_DIR / "ThetaPower_vs_Speed_MixedModel.png")

    with open(OUTPUT_DIR / "MixedModel_Stats_Summary.txt", "w") as f:
        f.write("=== Theta Frequency vs Speed ===\n")
        f.write(str(freq_result.summary()))
        f.write("\n\n=== Theta Power vs Speed ===\n")
        f.write(str(power_result.summary()))

    return freq_result, power_result


# %% ==================== Driver =====================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = process_root_directory(ROOT_DIR)
    df.to_excel(OUTPUT_XLSX, index=False)
    print(f'Saved {len(df)} rows -> {OUTPUT_XLSX}')

    if df["Session"].nunique() > 0:
        run_speed_vs_theta_stats(df)
    else:
        print('No sessions processed successfully -- skipping speed vs. theta statistics.')
