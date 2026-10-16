"""
ED_min frequency-bank simulation.

Quantifies how the autocorrelogram peak-range method's ED_min statistic
(Dunn et al. 2022, Nat Commun 13:6997, Supp. Fig. 5 -- see
ACG_theta_continuity_TT_ED_min_Simul.py for the full method translation)
behaves when a synthetic LFP oscillation is matched against reference
sinusoid banks that do or do not overlap its own frequency.

Three surrogate "LFP" traces are synthesised -- pure sinusoids at 1, 5 and
20 Hz with Poisson shot noise added and band-limited to look like a real
broadband LFP recording -- and each is matched (via ED_min) against three
reference sinusoid banks:
    1. 0.1-2 Hz   (delta-range bank; starts at 0.1 Hz, not 0, since a 0 Hz
                   "sinusoid" is flat and has no side peak to match against)
    2. 3-7 Hz     (theta-range bank)
    3. 18-22 Hz   (beta/gamma-range bank)
all at 0.1 Hz resolution, producing a 3x3 grid of ED_min values plus a
best-matching-sinusoid ACG panel for every (surrogate, bank) combination.

Two complementary edge-case probes are included:
  * EDGE_TEST_FREQS / run_edge_tests -- holds a surrogate's true frequency
    fixed and evaluates the ED against a single reference sinusoid just below
    and just above its own bank's edges (e.g. the 1 Hz surrogate, whose own
    bank is 0.1-2 Hz, against reference sinusoids at 0.05 Hz and 2.1 Hz).
  * SURROGATE_EDGE_TEST_FREQS / run_surrogate_edge_tests -- the mirror image:
    holds a bank fixed and synthesises a NEW surrogate LFP whose true
    frequency sits just below/above that bank's edges (e.g. surrogates at
    0.05 Hz and 2.1 Hz against the full 0.1-2 Hz bank), then lets the bank
    search as normal (ED_min over the whole bank, not a single frequency).
Both quantify how sharply the fit degrades right outside the bank that
actually contains the true frequency -- the first by moving the reference,
the second by moving the signal.

A third probe (find_ed_min_unity_freq / run_ed_min_unity_search) turns that
same question into a root-finding problem: for each bank, what PURE (noise-
free) sine frequency -- one just below the bank's lower edge, one just above
its upper edge -- gives ED_min exactly 1.0? That is the effective "capture
edge" of the bank at the fit-quality threshold (ACG_ED_MIN_THRESH = 1.0 in
ACG_theta_continuity_TT_ED_min_Simul.py) used to accept/reject real epochs.

Reuses create_sine_ref_xcorrs / quantify_xcorr_epochs directly from
ACG_theta_continuity_TT_ED_min_Simul.py so the matching algorithm is
identical to the one used on real recordings.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import brentq
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ACG_theta_continuity_TT_ED_min_Simul import (
    create_sine_ref_xcorrs, quantify_xcorr_epochs, ACG_PEAK_PROMINENCE,
)


# %% ==================== Configuration ===========================================

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'ED_min_bank_simulation_results')

FS_ACG = 1000.0        # Hz -- sampling rate of the surrogate traces / ACG analysis
EPOCH_SEC = 45.0        # s  -- long enough that even the lowest frequency
                        #        tested anywhere in this script (the 0.05 Hz
                        #        below-edge test, 20 s period) shows its full
                        #        first side peak, flanking troughs (out to 1.5
                        #        periods = 30 s) AND the zero-crossings that
                        #        bracket them (out to 1.75 periods = 35 s),
                        #        all within the autocorrelogram's lag range,
                        #        with margin

SURROGATE_FREQS_HZ = [1.0, 5.0, 20.0]

# (fmin, fmax) Hz, inclusive -- all stepped at 0.1 Hz resolution
REFERENCE_BANKS = {
    '0.1-2 Hz':  (0.1, 2.0),
    '3-7 Hz':    (3.0, 7.0),
    '18-22 Hz':  (18.0, 22.0),
}
FREQ_RES = 0.1

# Edge-case probe: for each surrogate, its own bank's boundaries plus/minus
# one reference-bank step (0.1 Hz), i.e. just outside where that surrogate's
# own bank would search -- quantifies how sharply ED_min rises the moment the
# reference bank no longer reaches the surrogate's true frequency.
EDGE_TEST_FREQS = {
    1.0:  {'bank': '0.1-2 Hz',  'below': 0.05,  'above': 2.1},
    5.0:  {'bank': '3-7 Hz',    'below': 2.9,   'above': 7.1},
    20.0: {'bank': '18-22 Hz',  'below': 17.9,  'above': 22.1},
}

# Mirror-image edge-case probe: for each bank, synthesise a NEW surrogate LFP
# whose true frequency sits just below/above that bank's own edges (same
# frequencies as EDGE_TEST_FREQS, indexed by bank instead of by surrogate),
# then run the full bank search against it -- quantifies how well a bank can
# still "reach" an oscillation whose true frequency just barely misses it.
SURROGATE_EDGE_TEST_FREQS = {
    '0.1-2 Hz':  {'below': 0.05,  'above': 2.1},
    '3-7 Hz':    {'below': 2.9,   'above': 7.1},
    '18-22 Hz':  {'below': 17.9,  'above': 22.1},
}

# Which SURROGATE_FREQS_HZ entry is each bank's own (in-bank) surrogate in
# run_bank_matching -- used purely to look up that pairing's ED_min as a
# for-comparison baseline in the surrogate edge-of-bank probe.
BANK_OWN_SURROGATE = {
    '0.1-2 Hz':  1.0,
    '3-7 Hz':    5.0,
    '18-22 Hz':  20.0,
}

# ---- Surrogate LFP synthesis ----
AMPLITUDE_UV = 150.0     # peak amplitude of the clean sinusoid (uV-like units)
POISSON_RATE = 100.0     # lambda (events/sample) of the underlying Poisson process
NOISE_SCALE = 0.5        # noise std, as a fraction of AMPLITUDE_UV, added to the sine
NOISE_LOWPASS_HZ = 100.0 # Hz -- band-limits the shot noise to broadband-LFP-like content
NOISE_LOWPASS_ORDER = 4


# %% ==================== Surrogate LFP synthesis ===================================

def make_surrogate_lfp(freq_hz, duration_s=EPOCH_SEC, fs_hz=FS_ACG,
                        amplitude=AMPLITUDE_UV, poisson_rate=POISSON_RATE,
                        noise_scale=NOISE_SCALE, lowpass_hz=NOISE_LOWPASS_HZ,
                        lowpass_order=NOISE_LOWPASS_ORDER, seed=None):
    """Synthesise one surrogate LFP trace: a pure sinusoid at `freq_hz` plus
    band-limited Poisson shot noise, to resemble a real (noisy) LFP recording.

    Shot noise is drawn once as a homogeneous Poisson count process
    (rate `poisson_rate` events/sample), converted to a zero-mean/unit-variance
    sequence, then zero-phase low-pass filtered at `lowpass_hz` so its spectral
    content matches the broadband-but-not-white character of real LFP noise
    (raw sample-wise Poisson counts are white up to Nyquist, which no real LFP
    is). The filtered noise is re-normalised to unit variance (filtering
    removes most of its power) before being scaled to `noise_scale * amplitude`
    and added to the clean sinusoid.

    Returns
    -------
    t   : ndarray, time vector (s)
    lfp : ndarray, surrogate LFP trace (same units as `amplitude`)
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs_hz))
    t = np.arange(n) / fs_hz

    clean = amplitude * np.sin(2 * np.pi * freq_hz * t)

    counts = rng.poisson(lam=poisson_rate, size=n).astype(np.float64)
    shot = (counts - poisson_rate) / np.sqrt(poisson_rate)  # zero-mean, unit-variance

    nyq = fs_hz / 2.0
    b, a = signal.butter(lowpass_order, lowpass_hz / nyq, btype='low')
    shot_filt = signal.filtfilt(b, a, shot)
    shot_filt = shot_filt / np.std(shot_filt)

    lfp = clean + noise_scale * amplitude * shot_filt
    return t, lfp


# %% ==================== ED_min matching across banks ==============================

def run_bank_matching(surrogate_freqs=SURROGATE_FREQS_HZ, banks=REFERENCE_BANKS,
                      freq_res=FREQ_RES, epoch_sec=EPOCH_SEC, fs=FS_ACG,
                      seed_base=0):
    """Synthesise one surrogate trace per frequency in `surrogate_freqs`, then
    run quantify_xcorr_epochs against every bank in `banks`.

    Returns
    -------
    results_df : DataFrame, one row per (surrogate_freq, bank) combination,
                 with the quantify_xcorr_epochs columns (ED_min, freq [the
                 matched reference frequency], peak1, trough1, trough2,
                 peakrange, peakrangenorm, ...) plus 'surrogate_freq' and
                 'bank'.
    traces     : dict {surrogate_freq: (t, lfp)} -- the synthesised epochs,
                 for plotting.
    """
    traces = {}
    rows = []

    for i, sf in enumerate(surrogate_freqs):
        t, lfp = make_surrogate_lfp(sf, duration_s=epoch_sec, fs_hz=fs, seed=seed_base + i)
        traces[sf] = (t, lfp)
        data_epoch = lfp[:, None]  # (winlength, 1) -- quantify_xcorr_epochs layout

        for bank_name, freq_range in banks.items():
            _XC, res = quantify_xcorr_epochs(
                data_epoch, freq_range=freq_range, freq_resolution=freq_res,
                fs=fs, peak_prominence=ACG_PEAK_PROMINENCE, ed_min_thresh=None)
            row = res.iloc[0].to_dict()
            row['surrogate_freq'] = sf
            row['bank'] = bank_name
            row['bank_range'] = f'{freq_range[0]:g}-{freq_range[1]:g} Hz'
            rows.append(row)

    results_df = pd.DataFrame(rows)
    return results_df, traces


# %% ==================== Edge-of-bank ED probe ======================================

def run_edge_tests(traces, edge_test_freqs=EDGE_TEST_FREQS, freq_res=FREQ_RES, fs=FS_ACG):
    """For each surrogate LFP, compute the (normalised) Euclidean distance --
    the same ED_min metric quantify_xcorr_epochs computes over a bank, here
    evaluated against a single reference sinusoid -- at one frequency just
    below and one just above its own matching bank's edges (see
    EDGE_TEST_FREQS). This probes how sharply the fit degrades the instant
    the reference bank no longer reaches the surrogate's true frequency.

    Each test frequency is run through quantify_xcorr_epochs as a
    single-point "bank" (freq_range=(f, f)) so the result uses exactly the
    same computation path (and full column set: ED_min, peak1, trough1/2,
    peakrange, ...) as run_bank_matching, just restricted to one frequency.

    Returns
    -------
    edge_df : DataFrame, one row per (surrogate_freq, edge_position)
              combination, with the quantify_xcorr_epochs columns plus
              'surrogate_freq', 'own_bank', 'edge_position' ('below'/'above'),
              and 'test_freq'.
    """
    rows = []
    for sf, spec in edge_test_freqs.items():
        _t, lfp = traces[sf]
        data_epoch = lfp[:, None]
        for edge_position in ('below', 'above'):
            f = spec[edge_position]
            _XC, res = quantify_xcorr_epochs(
                data_epoch, freq_range=(f, f), freq_resolution=freq_res,
                fs=fs, peak_prominence=ACG_PEAK_PROMINENCE, ed_min_thresh=None)
            row = res.iloc[0].to_dict()
            row['surrogate_freq'] = sf
            row['own_bank'] = spec['bank']
            row['edge_position'] = edge_position
            row['test_freq'] = f
            rows.append(row)

    edge_df = pd.DataFrame(rows)
    return edge_df


def run_surrogate_edge_tests(banks=REFERENCE_BANKS, surrogate_edge_freqs=SURROGATE_EDGE_TEST_FREQS,
                             freq_res=FREQ_RES, epoch_sec=EPOCH_SEC, fs=FS_ACG, seed_base=100):
    """Mirror image of run_edge_tests: for each bank, synthesise a NEW
    surrogate LFP whose true frequency sits just below/above that bank's own
    edges (SURROGATE_EDGE_TEST_FREQS), then run the FULL bank search
    (quantify_xcorr_epochs with that bank's whole freq_range, exactly as in
    run_bank_matching) against it -- i.e. does the bank's nearest edge
    frequency still capture an oscillation whose true frequency just barely
    falls outside the bank.

    Returns
    -------
    results_df : DataFrame, one row per (bank, edge_position) combination,
                 with the quantify_xcorr_epochs columns plus 'bank',
                 'bank_range', 'edge_position' ('below'/'above'), and
                 'surrogate_freq' (the true frequency of the synthesised
                 surrogate for that row).
    traces     : dict {(bank, edge_position): (t, lfp)} -- the synthesised
                 surrogate epochs, for plotting.
    """
    traces = {}
    rows = []

    for i, (bank_name, spec) in enumerate(surrogate_edge_freqs.items()):
        freq_range = banks[bank_name]
        for j, edge_position in enumerate(('below', 'above')):
            f = spec[edge_position]
            t, lfp = make_surrogate_lfp(f, duration_s=epoch_sec, fs_hz=fs,
                                        seed=seed_base + 2 * i + j)
            traces[(bank_name, edge_position)] = (t, lfp)
            data_epoch = lfp[:, None]

            _XC, res = quantify_xcorr_epochs(
                data_epoch, freq_range=freq_range, freq_resolution=freq_res,
                fs=fs, peak_prominence=ACG_PEAK_PROMINENCE, ed_min_thresh=None)
            row = res.iloc[0].to_dict()
            row['bank'] = bank_name
            row['bank_range'] = f'{freq_range[0]:g}-{freq_range[1]:g} Hz'
            row['edge_position'] = edge_position
            row['surrogate_freq'] = f
            rows.append(row)

    results_df = pd.DataFrame(rows)
    return results_df, traces


# %% ==================== ED_min == 1 crossing frequencies ==========================

def pure_sine_epoch(freq_hz, duration_s=EPOCH_SEC, fs_hz=FS_ACG, amplitude=1.0):
    """A single noise-free sinusoid epoch -- used for the ED_min==1 root
    search, where a deterministic (not Poisson-noisy) probe signal is needed
    so ED_min(freq) is a smooth, repeatable function of freq alone."""
    n = int(round(duration_s * fs_hz))
    t = np.arange(n) / fs_hz
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def ed_min_for_pure_sine(freq_hz, freq_range, freq_res=FREQ_RES, fs=FS_ACG,
                         epoch_sec=EPOCH_SEC, peak_prominence=ACG_PEAK_PROMINENCE):
    """ED_min of a pure (noise-free) `freq_hz` sinusoid against the full
    reference bank spanning `freq_range` -- one call = one point on the
    ED_min(freq) curve used by find_ed_min_unity_freq / the unity-crossing plot.
    """
    lfp = pure_sine_epoch(freq_hz, duration_s=epoch_sec, fs_hz=fs)
    data_epoch = lfp[:, None]
    _XC, res = quantify_xcorr_epochs(
        data_epoch, freq_range=freq_range, freq_resolution=freq_res,
        fs=fs, peak_prominence=peak_prominence, ed_min_thresh=None)
    return float(res['ED_min'].iloc[0])


def find_ed_min_unity_freq(freq_range, side, freq_res=FREQ_RES, fs=FS_ACG,
                           epoch_sec=EPOCH_SEC, step=0.02, max_offset=2.0,
                           target=1.0):
    """Root-find the pure-sine frequency just outside `freq_range` (on `side`
    -- 'below' the lower edge or 'above' the upper edge) whose ED_min against
    that bank equals exactly `target` (1.0 by default, i.e. ACG_ED_MIN_THRESH).

    The bank's own edge frequency is assumed to give ED_min well below
    `target` (it's itself inside the bank, so the fit should be near-perfect
    for a pure sine); the search then steps outward from that edge by `step`
    Hz until ED_min crosses `target`, and brentq refines the root within the
    bracket where the sign change was found.

    Returns the crossing frequency (Hz).
    """
    lo_edge, hi_edge = freq_range
    edge = lo_edge if side == 'below' else hi_edge

    def f(x):
        return ed_min_for_pure_sine(x, freq_range, freq_res=freq_res, fs=fs,
                                    epoch_sec=epoch_sec) - target

    f_edge = f(edge)
    offset = step
    while offset <= max_offset:
        probe = edge - offset if side == 'below' else edge + offset
        if probe <= 0:
            offset += step
            continue
        f_probe = f(probe)
        if np.sign(f_probe) != np.sign(f_edge) or f_probe == 0:
            lo, hi = (probe, edge) if side == 'below' else (edge, probe)
            return brentq(f, lo, hi, xtol=1e-5)
        offset += step

    raise RuntimeError(f'No ED_min={target:g} crossing found {side} {freq_range} '
                       f'within {max_offset:g} Hz')


def run_ed_min_unity_search(banks=REFERENCE_BANKS, freq_res=FREQ_RES, fs=FS_ACG,
                            epoch_sec=EPOCH_SEC, target=1.0):
    """For every bank in `banks`, find the pure-sine frequency just below its
    lower edge and just above its upper edge where ED_min crosses `target`
    (see find_ed_min_unity_freq).

    Returns a DataFrame with one row per bank: 'bank', 'bank_range',
    'low_edge_hz', 'high_edge_hz', 'below_unity_freq', 'above_unity_freq'
    (the two crossing frequencies), and the Hz distance of each crossing from
    its edge ('below_margin_hz', 'above_margin_hz').
    """
    rows = []
    for bank_name, freq_range in banks.items():
        f_below = find_ed_min_unity_freq(freq_range, 'below', freq_res=freq_res,
                                         fs=fs, epoch_sec=epoch_sec, target=target)
        f_above = find_ed_min_unity_freq(freq_range, 'above', freq_res=freq_res,
                                         fs=fs, epoch_sec=epoch_sec, target=target)
        rows.append({
            'bank': bank_name,
            'bank_range': f'{freq_range[0]:g}-{freq_range[1]:g} Hz',
            'low_edge_hz': freq_range[0],
            'high_edge_hz': freq_range[1],
            'below_unity_freq': f_below,
            'above_unity_freq': f_above,
            'below_margin_hz': freq_range[0] - f_below,
            'above_margin_hz': f_above - freq_range[1],
        })
    return pd.DataFrame(rows)


# %% ==================== Plotting ===================================================

def plot_surrogate_traces(traces, zoom_sec=2.0, figsize=(9, 6)):
    """One panel per surrogate frequency: a `zoom_sec`-long snippet of the
    synthesised (sinusoid + Poisson shot noise) trace."""
    freqs = sorted(traces.keys())
    fig, axes = plt.subplots(len(freqs), 1, figsize=figsize, sharex=False)
    if len(freqs) == 1:
        axes = [axes]
    for ax, f in zip(axes, freqs):
        t, lfp = traces[f]
        mask = t <= zoom_sec
        ax.plot(t[mask], lfp[mask], color='black', linewidth=0.8)
        ax.set_ylabel('Amplitude (uV)')
        ax.set_title(f'Surrogate LFP: {f:g} Hz + Poisson noise (first {zoom_sec:g} s)')
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    return fig, axes


def _plot_best_fit_panel(ax, lfp, row, fs=FS_ACG):
    """One ACG panel: data autocorrelogram vs. its best-matching reference
    sinusoid's autocorrelogram, with the peak-range measurement marked in
    green (style of plot_example_epoch in ACG_theta_continuity_TT_ED_min_Simul.py).
    """
    winlength = len(lfp)
    xc_raw = signal.correlate(lfp, lfp, mode='full', method='auto')
    xc = xc_raw / np.max(xc_raw)
    lags_s = (np.arange(len(xc)) - (winlength - 1)) / fs

    refXC, *_ = create_sine_ref_xcorrs(np.array([row['freq']]), winlength, fs=fs,
                                       peak_prominence=ACG_PEAK_PROMINENCE)
    ax.plot(lags_s, refXC[:, 0], color='0.7', linewidth=1.5, label='matched sinusoid')
    ax.plot(lags_s, xc, color='black', linewidth=1, label='data')

    peak_lag = lags_s[int(row['peak1_idx'])]
    trough_mean = np.mean([row['trough1'], row['trough2']])
    ax.plot([peak_lag, peak_lag], [trough_mean, row['peak1']],
            color='limegreen', linewidth=2, zorder=5)
    ax.scatter([peak_lag], [row['peak1']], color='limegreen', s=15, zorder=6)

    ax.axhline(0, color='0.85', linewidth=0.5, zorder=0)
    # Zoom to a few periods of the matched frequency so the fit is legible,
    # rather than showing the full +/-EPOCH_SEC lag range -- capped at 5 s so
    # a poorly-matched low reference frequency (e.g. a 20 Hz surrogate best
    # "matched" to 0.1 Hz because nothing in that bank fits) doesn't force an
    # unreadable, near-solid-black multi-cycle window.
    zoom = min(winlength / fs, max(6.0 / row['freq'], 0.5), 5.0)
    ax.set_xlim(-zoom, zoom)
    ax.set_ylim(-1, 1.05)
    ax.set_title(f"matched {row['freq']:.1f} Hz | ED_min={row['ED_min']:.3f}",
                fontsize=9)


def plot_best_fit_grid(results_df, traces, banks=REFERENCE_BANKS,
                       surrogate_freqs=SURROGATE_FREQS_HZ, fs=FS_ACG,
                       figsize=(13, 10)):
    """3 (surrogate freq) x 3 (reference bank) grid of best-matching-sinusoid
    ACG panels, one per (surrogate, bank) combination.

    Rows = surrogate LFP frequency, columns = reference bank. Each panel's
    title reports the matched reference frequency and ED_min for that
    combination.
    """
    bank_names = list(banks.keys())
    fig, axes = plt.subplots(len(surrogate_freqs), len(bank_names),
                             figsize=figsize, squeeze=False)

    for i, sf in enumerate(surrogate_freqs):
        _t, lfp = traces[sf]
        for j, bank_name in enumerate(bank_names):
            ax = axes[i, j]
            row = results_df[(results_df['surrogate_freq'] == sf) &
                            (results_df['bank'] == bank_name)].iloc[0]
            _plot_best_fit_panel(ax, lfp, row, fs=fs)
            if j == 0:
                ax.set_ylabel(f'{sf:g} Hz surrogate\nr')
            if i == 0:
                ax.set_title(f'{bank_name} bank\n' + ax.get_title())
            if i == len(surrogate_freqs) - 1:
                ax.set_xlabel('Lag (s)')

    axes[0, 0].legend(loc='upper right', fontsize=7, frameon=False)
    fig.suptitle('Best-matching reference sinusoid ACG fit, per surrogate x bank', y=1.01)
    fig.tight_layout()
    return fig, axes


def plot_edge_test_grid(edge_df, traces, surrogate_freqs=SURROGATE_FREQS_HZ, fs=FS_ACG,
                        figsize=(9, 10)):
    """3 (surrogate freq) x 2 (just-below-edge, just-above-edge) grid of ACG
    fit panels for the EDGE_TEST_FREQS probe -- same panel style as
    plot_best_fit_grid, so the edge-frequency fits can be compared directly
    to the in-bank fits."""
    positions = ('below', 'above')
    fig, axes = plt.subplots(len(surrogate_freqs), len(positions),
                             figsize=figsize, squeeze=False)

    for i, sf in enumerate(surrogate_freqs):
        _t, lfp = traces[sf]
        for j, position in enumerate(positions):
            ax = axes[i, j]
            row = edge_df[(edge_df['surrogate_freq'] == sf) &
                         (edge_df['edge_position'] == position)].iloc[0]
            _plot_best_fit_panel(ax, lfp, row, fs=fs)
            if j == 0:
                ax.set_ylabel(f"{sf:g} Hz surrogate\n(own bank: {row['own_bank']})\nr")
            if i == 0:
                ax.set_title(f'just {position} own bank edge\n' + ax.get_title())
            if i == len(surrogate_freqs) - 1:
                ax.set_xlabel('Lag (s)')

    axes[0, 0].legend(loc='upper right', fontsize=7, frameon=False)
    fig.suptitle('Edge-of-bank ED probe: fit just outside each surrogate\'s own bank', y=1.01)
    fig.tight_layout()
    return fig, axes


def plot_surrogate_edge_test_grid(surrogate_edge_df, surrogate_edge_traces,
                                  banks=REFERENCE_BANKS, fs=FS_ACG, figsize=(9, 10)):
    """3 (reference bank) x 2 (surrogate just-below-edge, just-above-edge)
    grid of ACG fit panels for the SURROGATE_EDGE_TEST_FREQS probe -- same
    panel style as plot_best_fit_grid/plot_edge_test_grid, for a surrogate
    whose true frequency sits just outside the bank being searched."""
    bank_names = list(banks.keys())
    positions = ('below', 'above')
    fig, axes = plt.subplots(len(bank_names), len(positions),
                             figsize=figsize, squeeze=False)

    for i, bank_name in enumerate(bank_names):
        for j, position in enumerate(positions):
            ax = axes[i, j]
            _t, lfp = surrogate_edge_traces[(bank_name, position)]
            row = surrogate_edge_df[(surrogate_edge_df['bank'] == bank_name) &
                                    (surrogate_edge_df['edge_position'] == position)].iloc[0]
            _plot_best_fit_panel(ax, lfp, row, fs=fs)
            if j == 0:
                ax.set_ylabel(f"{bank_name} bank\nsurrogate {row['surrogate_freq']:g} Hz\nr")
            if i == 0:
                ax.set_title(f'surrogate just {position} bank edge\n' + ax.get_title())
            if i == len(bank_names) - 1:
                ax.set_xlabel('Lag (s)')

    axes[0, 0].legend(loc='upper right', fontsize=7, frameon=False)
    fig.suptitle("Surrogate edge-of-bank probe: bank's fit to a surrogate just outside it",
                y=1.01)
    fig.tight_layout()
    return fig, axes


def plot_ed_min_unity_curves(unity_df, banks=REFERENCE_BANKS, freq_res=FREQ_RES, fs=FS_ACG,
                             epoch_sec=EPOCH_SEC, n_points=61, figsize=(9, 10)):
    """3 (bank) x 2 (below edge, above edge) grid of ED_min(freq) curves for a
    pure sine swept across each bank edge, with a horizontal line at ED_min=1
    and a vertical line at the root found by find_ed_min_unity_freq.

    `unity_df` is the output of run_ed_min_unity_search.
    """
    bank_names = list(banks.keys())
    sides = ('below', 'above')
    fig, axes = plt.subplots(len(bank_names), len(sides), figsize=figsize, squeeze=False)

    for i, bank_name in enumerate(bank_names):
        freq_range = banks[bank_name]
        row = unity_df[unity_df['bank'] == bank_name].iloc[0]
        for j, side in enumerate(sides):
            ax = axes[i, j]
            edge = freq_range[0] if side == 'below' else freq_range[1]
            root = row['below_unity_freq'] if side == 'below' else row['above_unity_freq']
            span = 1.5 * abs(edge - root) + 1e-6
            lo = max(edge - span, 1e-3) if side == 'below' else edge - 0.2 * span
            hi = edge + 0.2 * span if side == 'below' else edge + span

            freqs = np.linspace(lo, hi, n_points)
            ed = [ed_min_for_pure_sine(f, freq_range, freq_res=freq_res, fs=fs,
                                       epoch_sec=epoch_sec) for f in freqs]

            ax.plot(freqs, ed, color='black', linewidth=1.2)
            ax.axhline(1.0, color='0.6', linestyle='--', linewidth=1)
            ax.axvline(edge, color='#4C72B0', linestyle=':', linewidth=1, label='bank edge')
            ax.axvline(root, color='#C44E52', linestyle='--', linewidth=1.2,
                      label=f'ED_min=1 @ {root:.4f} Hz')
            ax.set_ylim(0, max(1.4, max(ed) * 1.05))
            ax.legend(fontsize=6, frameon=False, loc='lower right' if side == 'below' else 'lower left')

            if j == 0:
                ax.set_ylabel(f'{bank_name} bank\nED_min')
            if i == 0:
                ax.set_title(f'just {side} edge')
            if i == len(bank_names) - 1:
                ax.set_xlabel('Pure sine test frequency (Hz)')

    fig.suptitle('ED_min(freq) for a pure sine swept across each bank edge', y=1.01)
    fig.tight_layout()
    return fig, axes


def plot_edmin_heatmap(results_df, banks=REFERENCE_BANKS,
                       surrogate_freqs=SURROGATE_FREQS_HZ, figsize=(6, 5)):
    """Heatmap of ED_min values: rows = surrogate LFP frequency, columns =
    reference bank. Annotated with the numeric ED_min value in each cell."""
    bank_names = list(banks.keys())
    mat = np.array([[results_df[(results_df['surrogate_freq'] == sf) &
                                (results_df['bank'] == bn)]['ED_min'].iloc[0]
                     for bn in bank_names]
                    for sf in surrogate_freqs])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, cmap='viridis_r', aspect='auto')
    ax.set_xticks(range(len(bank_names)))
    ax.set_xticklabels(bank_names)
    ax.set_yticks(range(len(surrogate_freqs)))
    ax.set_yticklabels([f'{f:g} Hz' for f in surrogate_freqs])
    ax.set_xlabel('Reference bank')
    ax.set_ylabel('Surrogate LFP frequency')
    ax.set_title('ED_min: surrogate LFP vs. reference bank')

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center',
                   color='white' if mat[i, j] < mat.max() * 0.6 else 'black', fontsize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('ED_min (normalised distance to best-matching sinusoid)')
    fig.tight_layout()
    return fig, ax


# %% ==================== Driver =====================================================

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_df, traces = run_bank_matching()

    cols = ['surrogate_freq', 'bank', 'bank_range', 'ED_min', 'freq',
           'peakrange', 'peakrangenorm']
    print(results_df[cols].to_string(index=False))

    results_path = os.path.join(OUTPUT_DIR, 'edmin_surrogate_vs_bank.csv')
    results_df.to_csv(results_path, index=False)
    print(f'\nSaved results -> {results_path}')

    fig_traces, _ = plot_surrogate_traces(traces)
    traces_path = os.path.join(OUTPUT_DIR, 'surrogate_lfp_traces.png')
    fig_traces.savefig(traces_path, dpi=200, bbox_inches='tight')
    plt.close(fig_traces)
    print(f'Saved figure -> {traces_path}')

    fig_grid, _ = plot_best_fit_grid(results_df, traces)
    grid_path = os.path.join(OUTPUT_DIR, 'best_fit_acg_grid.png')
    fig_grid.savefig(grid_path, dpi=200, bbox_inches='tight')
    plt.close(fig_grid)
    print(f'Saved figure -> {grid_path}')

    fig_heat, _ = plot_edmin_heatmap(results_df)
    heat_path = os.path.join(OUTPUT_DIR, 'edmin_heatmap.png')
    fig_heat.savefig(heat_path, dpi=200, bbox_inches='tight')
    plt.close(fig_heat)
    print(f'Saved figure -> {heat_path}')

    # ---- Edge-of-bank ED probe ----
    edge_df = run_edge_tests(traces)
    own_bank_edmin = {sf: results_df.loc[(results_df['surrogate_freq'] == sf) &
                                         (results_df['bank'] == spec['bank']),
                                         'ED_min'].iloc[0]
                      for sf, spec in EDGE_TEST_FREQS.items()}
    edge_df['own_bank_edmin'] = edge_df['surrogate_freq'].map(own_bank_edmin)

    edge_cols = ['surrogate_freq', 'own_bank', 'own_bank_edmin',
                'edge_position', 'test_freq', 'ED_min', 'freq']
    print('\nEdge-of-bank ED probe (frequencies just outside each surrogate\'s own bank):')
    print(edge_df[edge_cols].to_string(index=False))

    edge_path = os.path.join(OUTPUT_DIR, 'edmin_edge_of_bank_probe.csv')
    edge_df.to_csv(edge_path, index=False)
    print(f'Saved results -> {edge_path}')

    fig_edge, _ = plot_edge_test_grid(edge_df, traces)
    edge_fig_path = os.path.join(OUTPUT_DIR, 'edge_of_bank_acg_grid.png')
    fig_edge.savefig(edge_fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig_edge)
    print(f'Saved figure -> {edge_fig_path}')

    # ---- Surrogate edge-of-bank probe (mirror image: new surrogates just
    # outside each bank's edges, matched against the full bank) ----
    surrogate_edge_df, surrogate_edge_traces = run_surrogate_edge_tests()
    in_bank_edmin = {bank_name: results_df.loc[
        (results_df['bank'] == bank_name) &
        (results_df['surrogate_freq'] == BANK_OWN_SURROGATE[bank_name]), 'ED_min'].iloc[0]
        for bank_name in REFERENCE_BANKS}
    surrogate_edge_df['in_bank_edmin'] = surrogate_edge_df['bank'].map(in_bank_edmin)

    surrogate_edge_cols = ['bank', 'bank_range', 'in_bank_edmin',
                          'edge_position', 'surrogate_freq', 'ED_min', 'freq']
    print("\nSurrogate edge-of-bank probe (surrogate's true frequency just outside the bank):")
    print(surrogate_edge_df[surrogate_edge_cols].to_string(index=False))

    surrogate_edge_path = os.path.join(OUTPUT_DIR, 'edmin_surrogate_edge_probe.csv')
    surrogate_edge_df.to_csv(surrogate_edge_path, index=False)
    print(f'Saved results -> {surrogate_edge_path}')

    fig_surr_edge, _ = plot_surrogate_edge_test_grid(surrogate_edge_df, surrogate_edge_traces)
    surr_edge_fig_path = os.path.join(OUTPUT_DIR, 'surrogate_edge_of_bank_acg_grid.png')
    fig_surr_edge.savefig(surr_edge_fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig_surr_edge)
    print(f'Saved figure -> {surr_edge_fig_path}')

    # ---- ED_min == 1 crossing frequencies (pure sine, per bank) ----
    unity_df = run_ed_min_unity_search()
    print('\nPure-sine frequencies where ED_min crosses 1.0, per bank:')
    print(unity_df.to_string(index=False))

    unity_path = os.path.join(OUTPUT_DIR, 'edmin_unity_crossing_freqs.csv')
    unity_df.to_csv(unity_path, index=False)
    print(f'Saved results -> {unity_path}')

    fig_unity, _ = plot_ed_min_unity_curves(unity_df)
    unity_fig_path = os.path.join(OUTPUT_DIR, 'edmin_unity_crossing_curves.png')
    fig_unity.savefig(unity_fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig_unity)
    print(f'Saved figure -> {unity_fig_path}')
