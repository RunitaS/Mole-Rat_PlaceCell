# -*- coding: utf-8 -*-
"""
Timestamp-anchored loading and repair of Neuralynx .ncs files.

What is wrong with `np.concatenate(data['samples'])`
----------------------------------------------------
Record layout is (timestamp, channel, SamplingFreq, NumValidSamples, 512 samples).
The reference loader calls the last two header fields 'cell_number' and 'params',
so NumValidSamples has never been checked. Two problems follow:
  1. In a partial record only the first NumValidSamples are data. The rest of the
     512 slots are stale buffer contents from earlier records, which concatenation
     treats as signal. This stretches the trace, and timing error accumulates.
  2. Samples are also genuinely lost between records: the next record's timestamp
     is later than the previous record's last valid sample. Dropping the stale
     slots alone therefore does not fix the timing.

The fix
-------
Every valid sample goes on a uniform 32 kHz grid at the index given by its OWN
record's timestamp, so timing error cannot accumulate across the session. Empty
grid slots are linearly interpolated between the neighbouring real samples, unless
the run of empty slots is longer than `max_gap_s`, in which case it stays NaN.
Where a record's timestamp overlaps the previous record, the later record wins.

Why timestamps are trusted over sample count: spike-triggered LFP peaks stay at a
constant lag on the timestamp clock, and shift by exactly the accumulated drift on
the sample-count clock. See DebugNcsDrops/NcsRepair_Validate.py.

Raw files are only ever opened read-only.
"""

import os
import re

import numpy as np

NCS_HEADER_BYTES = 16 * 1024
SAMPLES_PER_RECORD = 512
NATIVE_FS = 32000  # native Neuralynx .ncs sampling rate (Hz)
MAX_GAP_S = 1.0    # runs of missing samples longer than this are not interpolated

# Neuralynx .ncs record format
ncs_dtype = np.dtype([
    ('timestamp'  , '<u8'),
    ('channel'    , '<u4'),
    ('fs'         , '<u4'),
    ('n_valid'    , '<u4'),
    ('samples'    , '<i2', (SAMPLES_PER_RECORD,)),
])


def read_header(fpath):
    """Return the 16 kB ASCII header as text."""
    with open(fpath, 'rb') as f:
        return f.read(NCS_HEADER_BYTES).split(b'\x00')[0].decode('latin1')


def header_value(header, key):
    """Value of a '-Key value' header line, or None if the key is absent."""
    m = re.search(rf'^-{key}[ \t]+([^\r\n]+)', header, flags=re.MULTILINE)
    return m.group(1).strip() if m else None


def open_ncs(fpath):
    """Read-only memory map of the records in an .ncs file."""
    return np.memmap(fpath, dtype=ncs_dtype, mode='r', offset=NCS_HEADER_BYTES)


def find_runs(is_true):
    """Start index and length of each run of True values."""
    edges = np.diff(np.r_[0, np.asarray(is_true, dtype=np.int8), 0])
    starts = np.flatnonzero(edges == 1)
    return starts, np.flatnonzero(edges == -1) - starts


def grid_layout(timestamps, n_valid, fs=NATIVE_FS):
    """Grid index of every valid sample, anchored to each record's own timestamp.

    Returns
    -------
    sample_idx : ndarray  grid index of each valid sample, in record order
    n_grid     : int      grid length (first to last valid sample)
    """
    ts = np.asarray(timestamps, dtype=np.int64)
    nv = np.asarray(n_valid, dtype=np.int64)
    start_idx = np.round((ts - ts.min()) * (fs / 1e6)).astype(np.int64)
    n_grid = int((start_idx + nv).max())
    offset_in_rec = np.arange(nv.sum()) - np.repeat(np.r_[0, np.cumsum(nv)[:-1]], nv)
    return np.repeat(start_idx, nv) + offset_in_rec, n_grid


def record_lengths(n_valid):
    """Valid samples per record. Simulated files (Simulate_LFP_FaSimul.py) never set
    NumValidSamples, so a file with all zeros is read as full records."""
    nv = np.clip(np.asarray(n_valid, dtype=np.int64), 0, SAMPLES_PER_RECORD)
    return nv if nv.any() else np.full(nv.shape, SAMPLES_PER_RECORD, dtype=np.int64)


def repair_arrays(timestamps, n_valid, samples, fs=NATIVE_FS, max_gap_s=MAX_GAP_S):
    """Build the repaired trace from raw record arrays. See `load_ncs_repaired`."""
    nv = record_lengths(n_valid)
    ts = np.asarray(timestamps, dtype=np.int64)
    order = np.argsort(ts, kind='stable')  # a no-op unless timestamps run backwards
    valid_mask = np.arange(SAMPLES_PER_RECORD)[None, :] < nv[order, None]
    values = np.asarray(samples)[order][valid_mask].astype(np.float64)

    sample_idx, n_grid = grid_layout(ts[order], nv[order], fs)
    lfp = np.full(n_grid, np.nan)
    lfp[sample_idx] = values  # repeated indices: the later record overwrites

    missing = np.isnan(lfp)
    run_starts, run_lengths = find_runs(missing)
    max_gap_samples = int(round(max_gap_s * fs))
    is_long = run_lengths > max_gap_samples

    interpolated = missing.copy()
    for start, length in zip(run_starts[is_long], run_lengths[is_long]):
        interpolated[start:start + length] = False
    grid = np.flatnonzero(~missing)
    lfp[interpolated] = np.interp(np.flatnonzero(interpolated), grid, lfp[grid])

    return {
        'lfp': lfp,                       # ADC counts on the uniform grid; NaN in long gaps
        'start_timestamp': int(ts.min()), # us; sample i is at start_timestamp + i * 1e6 / fs
        'fs': fs,
        'interpolated': interpolated,     # True where the value was filled in
        'unfilled': missing & ~interpolated,  # True in gaps longer than max_gap_s
        'gap_starts': run_starts,
        'gap_lengths': run_lengths,
    }


def load_ncs_repaired(fpath, max_gap_s=MAX_GAP_S):
    """Load an .ncs file onto a uniform, timestamp-anchored grid.

    Returns
    -------
    dict with the keys of `repair_arrays`, plus
    ad_bit_volts : float  from this file's header; lfp * ad_bit_volts * 1e6 is in uV
    """
    data = open_ncs(fpath)
    header = read_header(fpath)
    fs = int(float(header_value(header, 'SamplingFrequency')))
    out = repair_arrays(data['timestamp'], data['n_valid'], data['samples'],
                        fs=fs, max_gap_s=max_gap_s)
    out['ad_bit_volts'] = float(header_value(header, 'ADBitVolts'))
    return out


def load_repaired_uv(fpath, ad_bit_volts, max_gap_s=MAX_GAP_S):
    """Repaired trace for the analysis scripts' load_ncs.

    `ad_bit_volts` comes from the calling script, which handles simulated files with
    an empty header. A gap longer than `max_gap_s` raises, because its NaNs would
    spread through the resampling and filter steps; the scripts report that file as
    SKIP. Sample i of the returned trace is at start_us + i * 1e6 / NATIVE_FS.

    Returns
    -------
    lfp_uv   : ndarray  repaired trace in microvolts on a uniform NATIVE_FS grid
    start_us : int      UNIX timestamp (us) of the first sample
    """
    data = open_ncs(fpath)
    out = repair_arrays(data['timestamp'], data['n_valid'], data['samples'], max_gap_s=max_gap_s)
    if out['unfilled'].any():
        raise ValueError(f'gap longer than {max_gap_s} s in the record timestamps.')
    return out['lfp'] * ad_bit_volts * 1e6, out['start_timestamp']


def write_repaired_ncs(src_path, dst_path, max_gap_s=MAX_GAP_S):
    """Write a repaired copy with the original header and only full 512-sample records.

    Existing scripts that concatenate `data['samples']` then run unchanged on the copy.
    Trailing samples that do not fill a record (< 16 ms) are dropped. A gap longer than
    `max_gap_s` cannot be stored in int16 records, so it raises instead of being
    silently filled; use `load_ncs_repaired` for such a file.
    """
    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        raise ValueError('Refusing to overwrite the raw file.')
    out = load_ncs_repaired(src_path, max_gap_s=max_gap_s)
    if out['unfilled'].any():
        raise ValueError(f'{src_path}: has a gap longer than {max_gap_s} s.')

    n_records = len(out['lfp']) // SAMPLES_PER_RECORD
    dt_us = SAMPLES_PER_RECORD * 1e6 / out['fs']
    raw = open_ncs(src_path)
    records = np.zeros(n_records, dtype=ncs_dtype)
    records['timestamp'] = out['start_timestamp'] + np.round(np.arange(n_records) * dt_us).astype(np.int64)
    records['channel'] = raw['channel'][0]
    records['fs'] = out['fs']
    records['n_valid'] = SAMPLES_PER_RECORD
    records['samples'] = np.round(
        out['lfp'][:n_records * SAMPLES_PER_RECORD]).astype(np.int16).reshape(n_records, -1)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(src_path, 'rb') as f:
        header_bytes = f.read(NCS_HEADER_BYTES)
    with open(dst_path, 'wb') as f:
        f.write(header_bytes)
        records.tofile(f)
