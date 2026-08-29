# -*- coding: utf-8 -*-
"""
Diagnostic visualization for Step 6 of the Pass Index algorithm (see
Pass_Index_Algo.md): shows how the Hilbert-transform phase (pass_index_trace,
range [-1, 1]) is assigned across individual field traversals.

Walks VIZ_ROOT_DIR recursively for session folders that hold .ntt, .ncs, and
.csv (tracking) files together, then makes one plot per .ntt file using that
file's main cell (the sorted, non-zero cluster with the most spikes -- each
.ntt is expected to contain exactly one real cell; cluster 0/noise is always
skipped). For each unit, reruns Steps 1-6 of compute_pass_index while keeping
the intermediate arrays that function normally discards, so each pass (one
Hilbert-phase cycle) can be plotted individually.

Output: <session_folder>/ThetaMod_PhasePrecession/<unit_label>_PassTraversals.png
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

from ThetaMod_PhasePRecession import (
    ROOT_FOLDER, TRACKING_TIME_UNIT, METHOD, BINSIDE, SMTH_WIDTH, FILTER_BAND,
    MIN_SPIKES_FOR_FIT, find_session_folders, _find_tracking_file, _natural_key,
    load_tracking, load_ntt_spike_times, spk_pos, rate_map, field_index_map,
    field_index_per_position, sample_along_arc, bandpass_filter, auto_filter_band,
)

# ----------------------------------------------------------------------------
# Hard-coded root directory to scan (recursively) for session folders. A
# session folder is any folder that directly contains .ntt, .ncs, and .csv
# (tracking) files together.
# ----------------------------------------------------------------------------
VIZ_ROOT_DIR = Path(r'C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/PlaceCell_True')


def find_tracked_session_folders(root: Path) -> list[Path]:
    """Session folders under root that directly contain .ntt, .ncs, and .csv
    (tracking) files together."""
    sessions = []
    for data_folder in find_session_folders(root):
        try:
            _find_tracking_file(data_folder)
        except FileNotFoundError:
            continue
        sessions.append(data_folder)
    return sessions


def main_cell_spikes(units: dict):
    """Pick the file's main cell: the sorted, non-zero cluster with the most
    spikes (cluster 0/noise is already excluded by load_ntt_spike_times).
    Each .ntt is expected to hold exactly one real cell, but if sorting left
    more than one non-zero cluster behind, the largest one wins."""
    cell_number = max(units, key=lambda c: len(units[c]))
    return cell_number, units[cell_number]


def iter_target_units(root: Path):
    """Yield (data_folder, unit_label, pos_ts, pos_xy, spk_ts) for every
    .ntt file's main cell across every tracked session folder under root."""
    for data_folder in find_tracked_session_folders(root):
        tracking_path = _find_tracking_file(data_folder)
        pos_ts, pos_xy = load_tracking(tracking_path, TRACKING_TIME_UNIT)

        ntt_files = sorted(data_folder.glob('*.ntt'), key=_natural_key)
        for ntt_path in ntt_files:
            units = load_ntt_spike_times(ntt_path)
            if not units:
                print(f'  Skipping {ntt_path.name}: no sorted (non-zero) clusters.')
                continue
            cell_number, spk_ts = main_cell_spikes(units)
            spk_ts = spk_ts[(spk_ts >= pos_ts.min()) & (spk_ts <= pos_ts.max())]
            if len(spk_ts) < MIN_SPIKES_FOR_FIT:
                print(f'  Skipping {ntt_path.name} cell {cell_number}: '
                      f'only {len(spk_ts)} spikes in tracked window '
                      f'(need >= {MIN_SPIKES_FOR_FIT}).')
                continue
            unit_label = f'{ntt_path.stem}_cell{cell_number}'
            yield data_folder, unit_label, pos_ts, pos_xy, spk_ts


def compute_pass_traces(pos_ts, pos_xy, spk_ts, method=METHOD, binside=BINSIDE,
                         smth_width=SMTH_WIDTH, filter_band=FILTER_BAND):
    """Steps 1-6 of compute_pass_index, with intermediates kept for plotting."""
    n_dims = pos_xy.shape[1]
    if binside == 'auto':
        binside = 2.0 * n_dims
    if smth_width == 'auto':
        smth_width = 3.0 * binside

    spk_xy, _ = spk_pos(pos_ts, pos_xy, spk_ts)
    rmap, occupancy, x_edges, y_edges = rate_map(pos_ts, pos_xy, spk_xy, binside, smth_width)
    fi_map = field_index_map(rmap, occupancy, method)
    field_index = field_index_per_position(pos_xy, fi_map, x_edges, y_edges)

    cc, ts2, resampled = sample_along_arc(pos_ts, pos_xy, field_index)

    if filter_band == 'auto':
        filter_band = auto_filter_band(method, rmap, occupancy, binside, n_dims)
    fs_arc = 1.0 / np.mean(np.diff(cc))
    filtered_field_index = bandpass_filter(resampled, filter_band[0], filter_band[1], fs_arc)

    pass_index_trace = np.angle(signal.hilbert(filtered_field_index)) / np.pi
    unwrapped = np.unwrap(pass_index_trace * np.pi)

    return dict(cc=cc, ts2=ts2, resampled=resampled,
                filtered_field_index=filtered_field_index,
                pass_index_trace=pass_index_trace, unwrapped=unwrapped,
                spk_xy=spk_xy)


def segment_passes(unwrapped, min_samples=5):
    """One pass = one contiguous run of samples between successive floor(2*pi)
    boundary crossings of the unwrapped Hilbert phase (Step 6 / Step 9's own
    definition of a traversal: 'one cycle of the filtered field-index signal
    = one field pass'). Returns a list of (start_idx, stop_idx) slices."""
    cycle = np.floor((unwrapped + np.pi) / (2 * np.pi)).astype(np.int64)
    boundaries = np.where(np.diff(cycle) != 0)[0] + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(unwrapped)]))
    return [(s, e) for s, e in zip(starts, stops) if (e - s) >= min_samples]


def plot_pass_traversals(pass_index_trace, passes, unit_label, out_path: Path):
    """Overlay every pass's pass_index_trace against normalized progress
    through that pass, colored by pass order, to show how consistently the
    Hilbert phase sweeps -1 -> +1 (entry -> exit) across separate traversals."""
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap('viridis')
    n_passes = len(passes)

    for i, (s, e) in enumerate(passes):
        trace = pass_index_trace[s:e]
        progress = np.linspace(0, 1, len(trace))
        color = cmap(i / max(n_passes - 1, 1))
        ax.plot(progress, trace, color=color, alpha=0.5, linewidth=1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_passes - 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Pass number (chronological)')

    ax.axhline(-1, color='0.6', linestyle='--', linewidth=0.8)
    ax.axhline(0, color='0.6', linestyle='--', linewidth=0.8)
    ax.axhline(1, color='0.6', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Normalized progress through pass (0 = start, 1 = end)')
    ax.set_ylabel('Pass index (Hilbert phase / pi)')
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f'{unit_label}\nPass index vs. progress through pass, '
                 f'{n_passes} traversals (Step 6)')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def process_unit(data_folder, unit_label, pos_ts, pos_xy, spk_ts):
    print(f'Session: {data_folder}')
    print(f'Unit: {unit_label}  ({len(spk_ts)} spikes)')

    traces = compute_pass_traces(pos_ts, pos_xy, spk_ts)
    passes = segment_passes(traces['unwrapped'])
    print(f'Segmented {len(passes)} passes (Hilbert-phase-cycle definition).')

    output_dir = data_folder / 'ThetaMod_PhasePrecession'
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'{unit_label}_PassTraversals.png'
    plot_pass_traversals(traces['pass_index_trace'], passes, unit_label, out_path)
    print(f'Saved: {out_path}')
    return out_path


def main():
    out_paths = []
    for data_folder, unit_label, pos_ts, pos_xy, spk_ts in iter_target_units(VIZ_ROOT_DIR):
        try:
            out_paths.append(process_unit(data_folder, unit_label, pos_ts, pos_xy, spk_ts))
        except Exception as exc:
            print(f'  FAILED on {unit_label} in {data_folder}: {exc}')
    print(f'\nDone. Generated {len(out_paths)} plot(s).')
    return out_paths


if __name__ == '__main__':
    main()
