
import os
from pathlib import Path
from scipy import signal
from scipy.stats import norm
import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Root directory to search recursively for .ncs files. Every .ncs found
# anywhere under this tree (in any subfolder) is processed.
ROOT_DIR = Path(r"C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True")

# All outputs (plots, stats summary, Excel results) are written here.
OUTPUT_DIR = ROOT_DIR / "Output_ThetaVsSpeed"

# ---- Acquisition ----
NATIVE_FS = 32000  # native Neuralynx .ncs sampling rate (Hz)
ADBitVolts = 0.000003051757812500000169  # V per ADC count (Neuralynx header)

# Neuralynx .ncs record format (512 int16 samples per record, 16 kB header skipped)
ncs_dtype = np.dtype([
    ('timestamp'  , '<u8'),
    ('sc_number'  , '<u4'),
    ('cell_number', '<u4'),
    ('params'     , '<u4'),
    ('samples'    , '<i2', (512,)),
])

# ---- Velocity / running-speed tracking ----
# Every .ncs file's session folder is expected to hold one tracking .csv with
# a UNIX timestamp (us, column A) on the same clock as the .ncs timestamps,
# plus x/y position in either pixels (columns B, C) or cm (columns D, E).
POS_JUMP_THRESH_CMS  = 80.0   # frame-to-frame jumps implying a speed above this (cm/s) are tracking artifacts
POS_SMOOTH_SIGMA_SMP = 1.0    # Gaussian smoothing sigma (in samples) applied to x/y tracking position

# 'pixel' or 'cm' -- set interactively at startup (see __main__ below).
# When 'pixel', x/y (columns B, C) are converted to cm using ARENA_WIDTH_CM;
# when 'cm', x/y (columns D, E) are used directly with no conversion.
COORD_UNITS = 'pixel'
ARENA_WIDTH_CM = 80.0  # physical arena width/length in cm, used only when COORD_UNITS == 'pixel'

FS_SPEED = 30
FS_LFP = 1000  # target downsample rate (Hz)
BANDPASS_CUTOFF = (3, 7)
FILTER_ORDER = 5
TRIM_DUR = 0.25
PEAK_THRESHOLD = 0.5
BINSIZE = 0.25

# ---- LFP cleaning (line noise / slow drift / high-frequency noise) ----
# Applied to the downsampled LFP before theta-band filtering. Matches
# ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean_v3.py.
LINE_HARMONICS = [50.0, 100.0, 150.0, 200.0]  # European mains harmonics, below FS_LFP/2 = 500 Hz
NOTCH_Q = 30.0
LOWPASS_CUTOFF_HZ = 100.0  # Hz -- restrict to <100 Hz before bandpassing to theta
LOWPASS_ORDER = 4          # Butterworth order (zero-phase via filtfilt)

# ---- Artifact rejection (bin-wise peak-to-peak, robust MAD outlier) ----
MAD_THRESH = 5.0


# %% ==================== .ncs / tracking data import ==============================

def load_ncs(fpath):
    """
    Read a Neuralynx .ncs file.

    Returns:
        lfp             : raw trace in microvolts (uV)
        start_timestamp : UNIX timestamp (us) of the first sample -- same
                           clock as the tracking .csv, used to align LFP and
                           speed to a common time base.
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
    return signal.detrend(np.asarray(x, dtype=np.float64), type=dtype) # type: ignore


def lowpass_filter(x, fs_hz, cutoff_hz, order=4):
    """Zero-phase Butterworth low-pass filter (filtfilt, so no phase distortion
    of the theta-band content the ACG matching cares about)."""
    nyq = fs_hz / 2.0
    b, a = signal.butter(order, cutoff_hz / nyq, btype='low')
    return signal.filtfilt(b, a, np.asarray(x, dtype=np.float64))


def _robust_high_outliers(x, thresh, ref_mask=None):
    """Boolean mask of samples that are high outliers by robust (MAD) z-score.

    If `ref_mask` is given, the median/MAD reference statistics are estimated
    from `x[ref_mask]` only, so epochs excluded upstream can't skew the
    robust threshold -- z-scores are still returned for every element of `x`.
    """
    x = np.asarray(x, dtype=np.float64)
    ref = x if ref_mask is None else x[np.asarray(ref_mask, dtype=bool)]
    if ref.size == 0:
        ref = x
    med = np.median(ref)
    mad = np.median(np.abs(ref - med))
    if mad == 0:
        mad = 1e-20
    robust_z = 0.6745 * (x - med) / mad
    return robust_z > thresh


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


def _smooth_tracking_position(
    x_cm:            np.ndarray,
    y_cm:            np.ndarray,
    t_us:            np.ndarray,
    jump_thresh_cms: float = POS_JUMP_THRESH_CMS,
    sigma_samples:   float = POS_SMOOTH_SIGMA_SMP,
) -> tuple:
    """Clean and smooth the x/y tracking position before any downstream use.

    Iterative jump removal + Gaussian smoothing: frame-to-frame steps between
    surviving samples that imply a speed above `jump_thresh_cms` are treated
    as tracking artifacts, filled by linear interpolation, then the cleaned
    x/y traces are Gaussian-smoothed (sigma in samples).
    """
    from scipy.ndimage import gaussian_filter1d

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


def _instantaneous_speed(x_cm: np.ndarray, y_cm: np.ndarray, t_us: np.ndarray) -> np.ndarray:
    """Per-frame running speed (cm/s), padded to len(t_us) (repeats last value)."""
    dt_s = np.diff(t_us) * 1e-6
    with np.errstate(invalid='ignore', divide='ignore'):
        speed = np.hypot(np.diff(x_cm), np.diff(y_cm)) / dt_s
    speed[dt_s <= 0] = np.nan
    return np.append(speed, speed[-1])


def compute_velocity_from_position(csv_path):
    """Compute running speed (cm/s) from a tracking .csv.

    Column layout (positional): UNIX timestamp in us (col A) on the same
    clock as the .ncs LFP timestamps; x/y position in either pixels
    (cols B, C) or cm (cols D, E), selected by COORD_UNITS. Pixel
    coordinates are converted to cm using ARENA_WIDTH_CM; cm coordinates
    are used as-is.

    Returns:
        time_us : absolute UNIX timestamp of each tracking sample (us)
        speed   : running speed (cm/s), same length as time_us
    """
    if COORD_UNITS == 'cm':
        df = pd.read_csv(csv_path, usecols=[0, 3, 4])
    else:
        df = pd.read_csv(csv_path, usecols=[0, 1, 2])
    df.columns = ['time_us', 'x', 'y']
    df = df.sort_values('time_us').reset_index(drop=True)

    time_us = df['time_us'].to_numpy(dtype=np.float64)
    x = df['x'].to_numpy(dtype=np.float64)
    y = df['y'].to_numpy(dtype=np.float64)

    if COORD_UNITS == 'cm':
        x_cm, y_cm = x, y
    else:
        px_per_cm = max(x.max() - x.min(), y.max() - y.min()) / ARENA_WIDTH_CM
        x_cm = x / px_per_cm
        y_cm = y / px_per_cm

    x_smooth, y_smooth = _smooth_tracking_position(x_cm, y_cm, time_us)
    speed = _instantaneous_speed(x_smooth, y_smooth, time_us)

    return time_us, speed


def extract_data(ncs_path, fs_lfp=FS_LFP):
    """
    Extracts instantaneous speed and downsampled LFP directly from a .ncs
    file and its accompanying tracking .csv.

    Arguments:
        ncs_path: path to the .ncs file to load
        fs_lfp: target LFP sampling rate after downsampling (Hz)
    Returns:
        speed: instantaneous speed of the animal (cm/s)
        speed_ts: time points speed was sampled at (s, relative to the .ncs start)
        lfp: 1D array of LFP values (uV), downsampled to fs_lfp and cleaned
             (line-noise notch, detrend, low-pass -- see notch_filter,
             detrend_signal, lowpass_filter)
        lfp_ts: 1D array of corresponding timestamps (s, relative to the .ncs start)
    """
    raw_lfp, start_timestamp = load_ncs(ncs_path)
    lfp = signal.resample_poly(raw_lfp, fs_lfp, NATIVE_FS)
    lfp = notch_filter(lfp, fs_lfp, LINE_HARMONICS, NOTCH_Q)
    lfp = detrend_signal(lfp, dtype='linear')
    lfp = lowpass_filter(lfp, fs_lfp, LOWPASS_CUTOFF_HZ, LOWPASS_ORDER)
    lfp_ts = np.arange(lfp.shape[0]) / fs_lfp

    pos_path = find_position_file(ncs_path)
    if pos_path is None:
        raise FileNotFoundError(f'No tracking .csv found next to {ncs_path}')
    time_us, speed = compute_velocity_from_position(pos_path)
    speed_ts = (time_us - start_timestamp) / 1e6

    return speed, speed_ts, lfp, lfp_ts

def filter_signal(lfp, fs, bandpass_cutoff, filter_order):
    # Narrow-band Butterworth bandpass at this order/fs is numerically unstable
    # in the default transfer-function ('ba') form -- filtfilt silently returns
    # all-NaN. Second-order-sections form (sosfiltfilt) is the stable equivalent.
    sos = signal.butter(filter_order, bandpass_cutoff, btype="band", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, lfp)
    return filtered

def trim_signal(signal, fs, trim_dur):
    trim_samples = int(trim_dur * fs)
    trimmed_signal = signal[trim_samples:-trim_samples]
    return trimmed_signal

def find_peaks_troughs(sig, ts, threshold, max_freq=BANDPASS_CUTOFF[1]):
    # A bare height threshold lets a small secondary bump on the shoulder of
    # a (real, asymmetric) theta cycle get picked up as its own peak, which
    # halves the apparent period -- i.e. doubles the instantaneous frequency.
    # Enforcing a minimum spacing of one cycle at the bandpass's upper edge
    # rules out any peak-to-peak/trough-to-trough interval faster than that.
    fs = 1 / np.median(np.diff(ts))
    min_distance = max(int(fs / max_freq), 1)
    min_height = np.std(sig) * threshold  # Set a minimum height threshold to avoid noise
    peak_indices, _ = signal.find_peaks(sig, height=min_height, distance=min_distance)  # Find indices of peaks
    trough_indices, _ = signal.find_peaks(-sig, height=min_height, distance=min_distance)  # Find indices of troughs
    peak_times = ts[peak_indices]          # Get timestamps of peaks
    trough_times = ts[trough_indices]      # Get timestamps of troughs
    return peak_times, trough_times

def find_missing_peaks(peak_times, trough_times, sig, sig_ts):
    # A gap between two consecutive detections can span more than one missed
    # cycle (e.g. a low-amplitude stretch where several real peaks fall below
    # the height threshold in find_peaks_troughs). A single fill pass only
    # inserts one trough/peak per gap, collapsing a multi-cycle gap into one
    # long interval -- i.e. an instantaneous frequency well below the theta
    # band. Looping until no gap remains subdivides such gaps fully instead.
    peak_times = np.sort(peak_times)
    trough_times = np.sort(trough_times)

    while True:
        changed = False

        new_troughs = []
        for i in range(len(peak_times)-1):        # Iterate over each pair of consecutive peaks
            troughs_between_peaks = trough_times[(trough_times>peak_times[i])&(trough_times<peak_times[i+1])]  # Find troughs between peaks
            if len(troughs_between_peaks) == 0:   # If no trough found between peaks
                time_idx_between_peaks = np.where((sig_ts>peak_times[i])&(sig_ts<peak_times[i+1]))[0]  # Indices between peaks
                if len(time_idx_between_peaks) == 0:
                    continue
                signal_between_peaks = sig[time_idx_between_peaks]  # Signal between peaks
                min_idx = np.argmin(signal_between_peaks)  # Index of minimum (trough)
                min_idx = time_idx_between_peaks[min_idx]  # Convert to global index
                new_troughs.append(sig_ts[min_idx])        # Get timestamp of trough
                changed = True
        if new_troughs:
            trough_times = np.sort(np.append(trough_times, new_troughs))  # Add missing troughs

        new_peaks = []
        for i in range(len(trough_times)-1):           # Iterate over each pair of consecutive troughs
            mask = (peak_times>trough_times[i])&(peak_times<trough_times[i+1])  # Find peaks between troughs
            peak_between_troughs = peak_times[mask]
            if len(peak_between_troughs) == 0:         # If no peak found between troughs
                time_idx_between_troughs = np.where((sig_ts>trough_times[i])&(sig_ts<trough_times[i+1]))[0]  # Indices between troughs
                if len(time_idx_between_troughs) == 0:
                    continue
                signal_between_troughs = sig[time_idx_between_troughs]  # Signal between troughs
                max_idx = np.argmax(signal_between_troughs)  # Index of maximum (peak)
                max_idx = time_idx_between_troughs[max_idx]  # Convert to global index
                new_peaks.append(sig_ts[max_idx])             # Get timestamp of peak
                changed = True
        if new_peaks:
            peak_times = np.sort(np.append(peak_times, new_peaks))  # Add missing peaks

        if not changed:
            break

    if len(peak_times)>len(trough_times):          # If more peaks than troughs
        peak_times = peak_times[:-1]               # Remove last peak
    elif len(trough_times)>len(peak_times):        # If more troughs than peaks
        trough_times = trough_times[:-1]           # Remove last trough
    assert len(trough_times) == len(peak_times)    # Ensure equal number of peaks and troughs

    return peak_times, trough_times

def estimate_instantaneous_frequency(peak_times, trough_times, lfp_ts):
    peak_to_peak_freq = 1 / np.diff(peak_times)    # Compute frequency from peak-to-peak intervals
    trough_to_trough_freq = 1 / np.diff(trough_times)  # Compute frequency from trough-to-trough intervals
    freq_time_peaks = (peak_times[:-1] + peak_times[1:]) / 2  # Midpoints between peaks
    freq_time_troughs = (trough_times[:-1] + trough_times[1:]) / 2  # Midpoints between troughs

    #Averaging of signal from peak and trough
    interp_peak_freq = np.interp(lfp_ts, freq_time_peaks, peak_to_peak_freq)  # Interpolate peak-to-peak freq to LFP timestamps
    interp_trough_freq = np.interp(lfp_ts, freq_time_troughs, trough_to_trough_freq)  # Interpolate trough-to-trough freq
    instantaneous_freq = (interp_peak_freq + interp_trough_freq) / 2          # Average the two frequencies
    return instantaneous_freq

def estimate_instantaneous_power(sig, lfp_ts, peak_times, trough_times):
    #Find timestamps for prak and trouhgs
    peak_indices = np.array([np.abs(lfp_ts - t).argmin() for t in peak_times])
    peak_amplitudes = sig[peak_indices]

    trough_indices = np.array([np.abs(lfp_ts - t).argmin() for t in trough_times])
    trough_amplitudes = sig[trough_indices]

    #Compute power/cycle
    instantaneous_power = (peak_amplitudes**2 + trough_amplitudes**2)/2      # Compute average power per cycle
    cycle_times = (peak_times+trough_times)/2                                # Compute cycle midpoints
    instantaneous_power = np.interp(lfp_ts, cycle_times, instantaneous_power)  # Interpolate power to LFP timestamps
    return instantaneous_power

def reject_artifact_bins(theta, lfp_ts, bins, mad_thresh=MAD_THRESH):
    """Boolean keep-mask, one per BINSIZE bin: peak-to-peak amplitude of the
    cleaned theta signal within each bin, robust (MAD) outlier rejection.
    Mirrors reject_artifact_epochs in
    ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean_v3.py, adapted to this
    script's time-bins (bin_data) instead of fixed-length epochs.
    """
    ptp = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        idx = np.where((lfp_ts >= bins[i]) & (lfp_ts <= bins[i + 1]))[0]
        if idx.size:
            ptp[i] = theta[idx].max() - theta[idx].min()

    valid = ~np.isnan(ptp)
    keep = np.zeros(len(ptp), dtype=bool)
    keep[valid] = ~_robust_high_outliers(ptp[valid], mad_thresh)
    return keep


def bin_data(data, ts, bins):
    bin_centers = (bins[:-1] + bins[1:])/2
    data_binned = []
    for i in range(len(bins)-1):
        bin_start = bins[i]
        bin_stop = bins[i+1]
        # speed values within the bin
        idx = np.where((ts>=bin_start)&(ts<=bin_stop))[0]
        data_binned.append(np.median(data[idx]))
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


_PIXEL_ANSWERS = {'pixel', 'pixels', 'px'}
_CM_ANSWERS    = {'cm', 'cms', 'centimeter', 'centimeters', 'centimetre', 'centimetres'}

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _coord_answer = input("Are the tracking coordinates in pixels or cm? [pixel/cm]: ").strip().lower()
    while _coord_answer not in _PIXEL_ANSWERS | _CM_ANSWERS:
        _coord_answer = input("Please enter 'pixel' or 'cm': ").strip().lower()
    COORD_UNITS = 'pixel' if _coord_answer in _PIXEL_ANSWERS else 'cm'
    print(f"Using '{COORD_UNITS}' tracking coordinates.\n")

    ncs_files = sorted(Path(ROOT_DIR).rglob('*.ncs'))
    print(f'Found {len(ncs_files)} .ncs files under {ROOT_DIR}')

    session_dfs = []

    for ncs_file in ncs_files:
        ses_id = str(ncs_file.relative_to(ROOT_DIR).with_suffix('')).replace(os.sep, '_')
        try:
            inst_speed, speed_ts, lfp, lfp_ts = extract_data(ncs_file)
            theta = filter_signal(lfp, FS_LFP, BANDPASS_CUTOFF, FILTER_ORDER)

            theta = trim_signal(theta, FS_LFP, TRIM_DUR)
            lfp_ts = trim_signal(lfp_ts, FS_LFP, TRIM_DUR)
            inst_speed = trim_signal(inst_speed, FS_SPEED, TRIM_DUR)
            speed_ts = trim_signal(speed_ts, FS_SPEED, TRIM_DUR)

            peak_ts, trough_ts = find_peaks_troughs(theta, lfp_ts, PEAK_THRESHOLD)

            peak_ts_clean, trough_ts_clean = find_missing_peaks(peak_ts, trough_ts, theta, lfp_ts)

            inst_freq = estimate_instantaneous_frequency(peak_ts_clean, trough_ts_clean, lfp_ts)

            inst_power = estimate_instantaneous_power(theta, lfp_ts, peak_ts_clean, trough_ts_clean)

            bin_start = lfp_ts.min().round()
            bin_stop = lfp_ts.max().round()
            bins = np.arange(bin_start, bin_stop+BINSIZE, step=BINSIZE)

            inst_speed_binned, _ = bin_data(inst_speed, speed_ts, bins)
            inst_freq_binned, _ = bin_data(inst_freq, lfp_ts, bins)
            inst_power_binned, bin_centers = bin_data(inst_power, lfp_ts, bins)

            inst_freq_binned = np.asarray(inst_freq_binned, dtype=float)
            inst_power_binned = np.asarray(inst_power_binned, dtype=float)
            inst_speed_binned = np.asarray(inst_speed_binned, dtype=float)
            bin_centers = np.asarray(bin_centers, dtype=float)

            keep_artifact = reject_artifact_bins(theta, lfp_ts, bins)
            inst_freq_binned = inst_freq_binned[keep_artifact]
            inst_power_binned = inst_power_binned[keep_artifact]
            inst_speed_binned = inst_speed_binned[keep_artifact]
            bin_centers = bin_centers[keep_artifact]

            inst_freq_binned, inst_power_binned, inst_speed_binned, bin_centers = drop_nan(inst_freq_binned, inst_power_binned, inst_speed_binned, bin_centers)

            df_session = pd.DataFrame({
                "Power": inst_power_binned,
                "Frequency": inst_freq_binned,
                "Speed": inst_speed_binned,
                "Time": bin_centers,
                "Session": ses_id
            })
            print(df_session)
            session_dfs.append(df_session)
        except Exception as e:
            print(f'  SKIP: {ses_id} -- {e}')

    if session_dfs:
        df = pd.concat(session_dfs, ignore_index=True)
        df[["Power", "Frequency", "Speed", "Time"]] = df[["Power", "Frequency", "Speed", "Time"]].astype(float)
    else:
        df = pd.DataFrame(columns=["Power", "Frequency", "Speed", "Time", "Session"])

    df.to_excel(OUTPUT_DIR / "out.xlsx", index=False)

    if df["Session"].nunique() > 0:
        run_speed_vs_theta_stats(df)
    else:
        print('No sessions processed successfully -- skipping speed vs. theta statistics.')
