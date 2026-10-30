# -*- coding: utf-8 -*-
"""
Validate the .ncs repair in ncs_repair.py.

Test 1 - Synthetic ground truth: known partial records, gaps, timestamp jitter and a gap
         on either side of the 1 s limit. Checks placement, filling and the limit.
Test 2 - Interpolation accuracy: delete known stretches from a real, untouched part of
         an LFP, fill them, and compare with the deleted signal. Shows how much signal
         survives at each gap length.
Test 3 - Spike-LFP alignment in every session with enough same-tetrode spikes: a spike
         leaves a sharp LFP deflection, so the spike-triggered LFP peak lag measures the
         LFP timing error against an independent record. A correct trace has the same
         lag early and late in the session.

Writes interpolation_accuracy.csv and spike_lfp_alignment.csv to OUT_DIR.
"""

import os
import re

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import irfft, next_fast_len, rfft

from ncs_repair import (NATIVE_FS, SAMPLES_PER_RECORD, header_value, load_ncs_repaired, open_ncs,
                        read_header, repair_arrays)

# %% ==================== Parameters ==========================================

ROOT_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Alpha200'
OUT_DIR = os.path.join(ROOT_DIR, 'Results', 'NcsDropCheck')

INJECTION_FILE = r'Fa5834\Open\Day2\1Cntrl\CSC1ch2.ncs'
INJECTION_GAPS_MS = [0.5, 2, 10, 50, 100, 250, 500, 1000]
INJECTION_REPEATS = 200
THETA_BAND = (3, 7)   # Hz

MIN_SPIKES = 300      # same-tetrode spikes needed for a session to be tested
N_SEGMENTS = 3        # spike-triggered peak measured separately in early/mid/late session
LAG_MARGIN = 250      # samples searched on top of the expected timing error
SPIKE_BAND = (150, 450)  # Hz; isolates the sharp spike deflection in the 500 Hz LFP
MIN_PEAK_Z = 8        # a peak below this is reported as unresolved

ntt_dtype = np.dtype([('timestamp', '<u8'), ('acq', '<u4'), ('cell', '<u4'),
                      ('params', '<u4', (8,)), ('waveform', '<i2', (32, 4))])

# %% ==================== Test 1: synthetic ground truth ======================


def test_synthetic():
    """Run the repair on records whose true content and timing are known."""
    fs = NATIVE_FS
    n_true = 512 * 400
    t = np.arange(n_true) / fs
    truth = 3000 * np.sin(2 * np.pi * 6 * t) + 800 * np.sin(2 * np.pi * 90 * t)

    # Records of varying length and spacing: (grid start, valid samples)
    starts, lengths = [], []
    pos, rng = 0, np.random.default_rng(0)
    while pos + 512 < n_true:
        n = 512 if rng.random() > 0.2 else int(rng.integers(1, 512))
        gap = int(rng.integers(1, 80)) if rng.random() < 0.1 else 0
        starts.append(pos)
        lengths.append(n)
        pos += n + gap
    starts, lengths = np.array(starts), np.array(lengths)

    samples = np.full((len(starts), 512), 12345, dtype=np.float64)  # stale filler, must never appear
    for i, (s, n) in enumerate(zip(starts, lengths)):
        samples[i, :n] = truth[s:s + n]
    ts = 1_000_000 + np.round(starts * 1e6 / fs).astype(np.int64)

    out = repair_arrays(ts, lengths, samples, fs=fs)
    n = len(out['lfp'])
    assert np.nanmax(np.abs(out['lfp'])) < 3900, 'stale filler leaked into the trace'
    real = ~out['interpolated']
    assert np.allclose(out['lfp'][real], truth[:n][real]), 'real samples moved off their true grid position'
    assert out['interpolated'].sum() == n - np.unique(np.concatenate(
        [np.arange(s, s + m) for s, m in zip(starts, lengths)])).size, 'wrong number of slots filled'
    max_err = np.abs(out['lfp'] - truth[:n])[out['interpolated']].max()
    assert max_err < 200, f'interpolation error too large: {max_err:.1f}'

    # Overlap: the later record's timestamp is 5 samples early, and it wins those slots
    ts2 = np.array([0, round((512 - 5) * 1e6 / fs)], dtype=np.int64)
    s2 = np.vstack([np.full(512, 1.0), np.full(512, 2.0)])
    o2 = repair_arrays(ts2, [512, 512], s2, fs=fs)
    assert len(o2['lfp']) == 1019 and o2['lfp'][506] == 1 and o2['lfp'][507] == 2

    # The 1 s limit: exactly 1 s is filled, 1 s + 1 sample stays NaN
    for gap, filled in [(fs, True), (fs + 1, False)]:
        ts3 = np.array([0, round((512 + gap) * 1e6 / fs)], dtype=np.int64)
        o3 = repair_arrays(ts3, [512, 512], np.ones((2, 512)), fs=fs)
        assert o3['interpolated'].sum() == (gap if filled else 0)
        assert o3['unfilled'].sum() == (0 if filled else gap)
    print('Test 1 passed: placement, stale slots, overlap, filling and the 1 s limit')


# %% ==================== Test 2: interpolation accuracy ======================


def test_interpolation_accuracy():
    """Delete known stretches from a real LFP; compare the fill with what was deleted."""
    fs = NATIVE_FS
    lfp = load_ncs_repaired(os.path.join(ROOT_DIR, INJECTION_FILE))['lfp']
    lfp = lfp[:int(300 * fs)]
    theta = signal.sosfiltfilt(signal.butter(3, THETA_BAND, 'bandpass', fs=fs, output='sos'), lfp)

    rng = np.random.default_rng(0)
    rows = []
    for gap_ms in INJECTION_GAPS_MS:
        n_gap = max(1, int(round(gap_ms / 1e3 * fs)))
        rmse, rms, r = [], [], []
        for start in rng.integers(fs, len(lfp) - n_gap - fs, INJECTION_REPEATS):
            idx = np.arange(start, start + n_gap)
            filled = np.linspace(lfp[start - 1], lfp[start + n_gap], n_gap + 2)[1:-1]
            rmse.append(np.sqrt(np.mean((filled - lfp[idx]) ** 2)))
            rms.append(np.sqrt(np.mean((lfp[idx] - lfp[idx].mean()) ** 2)))
            if n_gap > 2:
                # Theta recovered: how well the fill tracks the theta actually in the gap
                r.append(np.corrcoef(filled, theta[idx])[0, 1])
        rows.append({'gap_ms': gap_ms, 'rmse_counts': np.median(rmse), 'signal_rms_counts': np.median(rms),
                     'rmse_over_signal_rms': np.median(rmse) / np.median(rms),
                     'corr_fill_vs_theta': np.nanmedian(r) if r else np.nan})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT_DIR, 'interpolation_accuracy.csv'), index=False)
    print('Test 2: fill vs deleted signal\n' + table.round(3).to_string(index=False))


# %% ==================== Test 3: spike-LFP alignment =========================


def ad_channels(fpath):
    """AD channel numbers listed in an .ncs or .ntt header."""
    value = header_value(read_header(fpath), 'ADChannel')
    return {int(v) for v in re.split(r'\s+', value)} if value else set()


def peak_lag(trace, spike_idx, max_lag):
    """Lag and robust z of the largest spike-triggered deflection, via FFT correlation."""
    lo, hi = spike_idx.min() - max_lag, spike_idx.max() + max_lag + 1
    lo, hi = max(lo, 0), min(hi, len(trace))
    x = trace[lo:hi]
    train = np.bincount(spike_idx - lo, minlength=len(x)).astype(np.float64)
    n_fft = next_fast_len(2 * len(x), real=True)
    xc = irfft(rfft(x, n_fft) * np.conj(rfft(train, n_fft)), n_fft)
    lags = np.arange(-max_lag, max_lag + 1)
    sta = xc[lags % n_fft] / len(spike_idx)
    med = np.median(sta)
    mad = 1.4826 * np.median(np.abs(sta - med))
    k = np.argmax(np.abs(sta - med))
    return int(lags[k]), abs(sta[k] - med) / mad


def test_session(folder):
    """Peak lag per session segment, on the raw concatenated trace and the repaired trace."""
    fnames = sorted(os.listdir(folder))
    ncs = [f for f in fnames if f.endswith('.ncs')]
    ntt = [f for f in fnames if f.endswith('.ntt')]
    tetrode_ch = {f: ad_channels(os.path.join(folder, f)) for f in ntt}

    # The LFP channel with the most spikes from its own tetrode
    best = None
    for c in ncs:
        channels = ad_channels(os.path.join(folder, c))
        files = [t for t in ntt if tetrode_ch[t] & channels]
        if not files:
            continue
        spikes = np.concatenate([np.fromfile(os.path.join(folder, t), dtype=ntt_dtype, offset=16 * 1024)['timestamp']
                                 for t in files]).astype(np.int64)
        if best is None or len(spikes) > len(best[1]):
            best = (c, spikes)
    if best is None or len(best[1]) < MIN_SPIKES:
        return []
    channel, spikes = best

    fpath = os.path.join(folder, channel)
    data = open_ncs(fpath)
    ts0 = int(data['timestamp'][0])
    raw = np.asarray(data['samples']).astype(np.float64).ravel()
    repaired = np.nan_to_num(load_ncs_repaired(fpath)['lfp'])

    sos = signal.butter(3, SPIKE_BAND, 'bandpass', fs=NATIVE_FS, output='sos')
    raw = signal.sosfiltfilt(sos, raw)
    repaired = signal.sosfiltfilt(sos, repaired)

    idx = np.sort(np.round((spikes - ts0) * (NATIVE_FS / 1e6)).astype(np.int64))
    idx = idx[(idx >= 0) & (idx < len(repaired))]
    raw_error = abs(len(raw) - len(repaired))

    rows = []
    for seg, s in enumerate(np.array_split(idx, N_SEGMENTS)):
        if len(s) < MIN_SPIKES // N_SEGMENTS:
            continue
        lag_rep, z_rep = peak_lag(repaired, s, LAG_MARGIN)
        lag_raw, z_raw = peak_lag(raw, s[s < len(raw)], raw_error + LAG_MARGIN)
        rows.append({'session': os.path.relpath(folder, ROOT_DIR), 'channel': channel, 'segment': seg,
                     'n_spikes': len(s), 'lag_repaired': lag_rep, 'z_repaired': z_rep,
                     'lag_raw': lag_raw, 'z_raw': z_raw})
    return rows


def test_spike_alignment():
    folders = sorted(root for root, _, fnames in os.walk(ROOT_DIR)
                     if any(f.endswith('.ncs') for f in fnames))
    rows = []
    for i, folder in enumerate(folders, 1):
        rows += test_session(folder)
        print(f'  session {i}/{len(folders)}', flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'spike_lfp_alignment.csv'), index=False)

    table = pd.DataFrame(rows)
    resolved = table[table['z_repaired'] >= MIN_PEAK_Z]
    n_ok = resolved.groupby('session').size()
    spread = resolved[resolved['session'].isin(n_ok[n_ok == N_SEGMENTS].index)].groupby('session').agg(
        rep_spread=('lag_repaired', lambda v: v.max() - v.min()),
        raw_spread=('lag_raw', lambda v: v.max() - v.min()))
    print(f'Test 3: {table["session"].nunique()} sessions tested, {len(spread)} with all segments resolved (z >= {MIN_PEAK_Z})')
    print(f'  Early-to-late lag change, repaired: median {spread["rep_spread"].median():.0f}, max {spread["rep_spread"].max():.0f} samples')
    print(f'  Early-to-late lag change, raw     : median {spread["raw_spread"].median():.0f}, max {spread["raw_spread"].max():.0f} samples')


# %% ==================== Main ================================================

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    test_synthetic()
    test_interpolation_accuracy()
    test_spike_alignment()
