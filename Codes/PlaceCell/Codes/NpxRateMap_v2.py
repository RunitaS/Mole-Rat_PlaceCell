# -*- coding: utf-8 -*-
"""
NpxRateMap_v2.py – Neuropixels batch place-cell analysis

Walks ROOT_FOLDER, laid out like the tetrode data:

    <ROOT_FOLDER>/<Animal ID>/<Arena type>/<Day>/<Session type>/
        spike_times.npy, spike_clusters.npy, cluster_group.tsv, params.py
        (Kilosort/phy output) + one '<...>_cm.csv' tracking file

and runs the PlaceCell_Main_v10.py pipeline on every unit of a given phy
quality (default 'good') in every such session folder whose phy depth lies
within that session's hippocampal range in DEPTH_EXCEL, using the tracking
'_cm.csv' in that same folder. All analyses are the v10 functions themselves
(imported, not re-implemented), with the v10 parameters, so every metric,
shuffle, figure and Excel sheet is produced the same way as for tetrode
(.ntt) data:

  * fixed-bin occupancy / spike maps, visit-count valid-bin criterion,
    Gaussian-smoothed ratemap, SIR, sparsity, coherence
  * location-shuffle bootstrap for SIR + coherence   -> shuffling_VisitCrit/
  * split-half stability                              -> ratemaps_VisitCrit/
  * theta modulation (autocorrelogram FFT)
  * speed modulation, binned + time-domain, shuffles  -> speed modulation_VisitCrit/
  * place-field detection (2-D grid or Circle/angular), field-area override
  * ratemap figure (+ field panels for place cells)   -> ratemaps_VisitCrit/
  * speed / dt_s plots next to each tracking file
  * Excel: Full / First_Half / Second_Half / PlaceFields / Speed_Summary,
    either Neuropixels-only or appended to an existing tetrode workbook
    (TETRODE_EXCEL -> COMBINED_EXCEL)
  * <output_excel>_run_metadata.csv

Only the Neuropixels-specific parts differ from v10:
  1. Spikes are read straight from the phy files (spike_times.npy /
     spike_clusters.npy, labels from cluster_group.tsv, sampling rate from
     params.py) and converted to µs so they share the tracking time base,
     instead of from a .ntt file.
  2. Tracking is the same cm-converted '_cm.csv' format as the tetrode data
     (column A = time, D = x (cm), E = y (cm); see
     ManageFiles/pixel_to_cm_conversion_v4_batch.py), except that time is in
     ms. It is converted to µs; everything after that is v10's cm-mode
     _load_tracking cleaning.
  3. Units are named 'cluster_<id>' (the 'unit' column). As for tetrodes,
     'session' is the folder path relative to ROOT_FOLDER, figures go into
     that session folder (ratemaps_VisitCrit/, shuffling_VisitCrit/, ...) and
     Circle vs 2-D arena is decided by v10 from the folder path.
  4. Confirmed place cells: instead of copying .ntt/.nev/.ncs files, the
     tracking CSV and each cell's spike times (s, .npy) are written to
     OUTPUT_PLACE_TRUE/<session path>/.

To change an analysis parameter (bin size, bootstrap count, place-cell
thresholds...), edit it in PlaceCell_Main_v10.py, or override it on `pc`
in the USER SETTINGS block below (e.g. pc.N_BOOTSTRAP = 500).

The notebook port that previously lived in this file is saved as
NpxRateMap_v2_notebookport_backup.py.
"""

#%%############################################################################
# USER SETTINGS ###############################################################
###############################################################################

import os

# Root of the Neuropixels data: Animal ID > Arena > Day > Session type >
# phy output + '_cm.csv'. Every folder below it holding spike_times.npy,
# spike_clusters.npy and exactly one '_cm.csv' is processed.
ROOT_FOLDER = r'X:\NMR_group_data\Runita\Analysis\Thesis\Neuropixel'

# Output locations
OUTPUT_EXCEL      = os.path.join(ROOT_FOLDER, 'All_Npx_PlaceChar_VisitCrit.xlsx')
OUTPUT_PLACE_TRUE = os.path.join(ROOT_FOLDER, 'PC_True')

# Append option. Set TETRODE_EXCEL to an existing PlaceCell_Main_v10 output
# workbook to combine with it: on every sheet the Neuropixels rows are added
# after the last tetrode row, a 'data_source' column ('tetrode' /
# 'neuropixels') is added, Speed_Summary is recomputed over both, and the
# result is saved as COMBINED_EXCEL. The tetrode workbook itself is only read,
# never modified. Leave TETRODE_EXCEL = None to write a Neuropixels-only
# workbook to OUTPUT_EXCEL instead.
TETRODE_EXCEL  = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\All_TT_PlaceChar_VisitCrit.xlsx'
COMBINED_EXCEL = os.path.join(ROOT_FOLDER, 'All_TT_Npx_PlaceChar_VisitCrit.xlsx')

# phy cluster label to analyse ('group' column of cluster_group.tsv)
UNIT_QUALITY = 'good'

# Hippocampus depth range per session. Columns: Recording_Path (relative to
# ROOT_FOLDER, e.g. 'Fa0156\Circle\Day6\1Cntrl'), Top_Depth_um,
# Bottom_Depth_um. Probe depth is 0 at the tip and increases towards the
# brain surface, so Bottom (DG side) < Top (CA1 side). Only UNIT_QUALITY
# units whose phy 'depth' (cluster_info.tsv) lies within
# [Bottom_Depth_um, Top_Depth_um] are analysed; sessions missing from the
# sheet are skipped. Set to None to analyse every UNIT_QUALITY unit.
DEPTH_EXCEL = os.path.join(ROOT_FOLDER, 'SessionDepth.xlsx')

# Tracking '_cm.csv' layout (same as tetrode): column A = time, D = x (cm),
# E = y (cm). Neuropixels tracking time is in ms -> converted to µs.
TRACKING_TIME_TO_US = 1000.0

# Only passed through to v10's function signatures; with cm tracking there is
# no pixel -> cm conversion here (that was done by the conversion script).
ARENA_WIDTH_CM = 80.0


#%%############################################################################
# Imports #####################################################################
###############################################################################

import re
import shutil
import threading
import concurrent.futures
from datetime import datetime

import numpy as np
import pandas as pd

import PlaceCell_Utils as pc

# Workbook this run writes: the combined tetrode + Neuropixels file when
# appending, otherwise the Neuropixels-only file.
RESULT_EXCEL = COMBINED_EXCEL if TETRODE_EXCEL else OUTPUT_EXCEL

# Point the v10 module at this run's outputs and settings. Every v10 function
# reads these module globals at call time.
pc.root_folder      = ROOT_FOLDER
pc.output_excel     = RESULT_EXCEL
pc.Output_PlaceTrue = OUTPUT_PLACE_TRUE
pc.arena_width_cm   = ARENA_WIDTH_CM
pc.COORD_UNITS      = 'cm'


#%%############################################################################
# Neuropixels adapters ########################################################
###############################################################################

# ntt_path (os.path.join(session_dir, 'cluster_<id>')) -> spike times (µs)
_SPIKES_US: dict[str, np.ndarray] = {}

_tracking_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_tracking_lock = threading.Lock()

PHY_REQUIRED_FILES = ('spike_times.npy', 'spike_clusters.npy', 'params.py')


def _read_sample_rate(dp: str) -> float:
    """'sample_rate' from phy's params.py (Hz)."""
    with open(os.path.join(dp, 'params.py')) as f:
        m = re.search(r'^\s*sample_rate\s*=\s*([0-9.eE+-]+)', f.read(), re.MULTILINE)
    if m is None:
        raise ValueError(f"No 'sample_rate' in {os.path.join(dp, 'params.py')}")
    return float(m.group(1))


def _read_unit_ids(dp: str, quality: str) -> list[int]:
    """Cluster ids labelled `quality` in phy (cluster_group.tsv, falling back
    to the 'group' column of cluster_info.tsv)."""
    for fname in ('cluster_group.tsv', 'cluster_info.tsv'):
        path = os.path.join(dp, fname)
        if os.path.isfile(path):
            labels = pd.read_csv(path, sep='\t')
            if 'group' in labels.columns and 'cluster_id' in labels.columns:
                is_q = labels['group'].astype(str).str.strip() == quality
                return sorted(int(c) for c in labels.loc[is_q, 'cluster_id'])
    raise FileNotFoundError(f'No cluster_group.tsv / cluster_info.tsv with a '
                            f"'group' column in {dp}")


def _session_key(rel_path: str) -> str:
    """Normalised session path, for matching folders to DEPTH_EXCEL rows."""
    return os.path.normcase(os.path.normpath(str(rel_path).strip().strip('\\/')))


def _load_depth_ranges(xlsx_path: str) -> dict[str, tuple[float, float]]:
    """{session key: (bottom_um, top_um)} from the DEPTH_EXCEL sheet."""
    df = pd.read_excel(xlsx_path)
    missing = {'Recording_Path', 'Top_Depth_um', 'Bottom_Depth_um'} - set(df.columns)
    if missing:
        raise ValueError(f'{xlsx_path} is missing column(s) {sorted(missing)}')
    df = df.dropna(subset=['Recording_Path', 'Top_Depth_um', 'Bottom_Depth_um'])
    ranges = {}
    for _, row in df.iterrows():
        top, bottom = float(row['Top_Depth_um']), float(row['Bottom_Depth_um'])
        if bottom > top:
            print(f"[WARN] {row['Recording_Path']}: Bottom_Depth_um > Top_Depth_um "
                  f'in {os.path.basename(xlsx_path)}; using the range as min–max')
        ranges[_session_key(row['Recording_Path'])] = (min(top, bottom), max(top, bottom))
    return ranges


def _read_unit_depths(dp: str) -> pd.Series:
    """phy's per-cluster 'depth' (µm from the probe tip), indexed by cluster_id."""
    path = os.path.join(dp, 'cluster_info.tsv')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'No cluster_info.tsv (needed for unit depth) in {dp}')
    info = pd.read_csv(path, sep='\t')
    if 'depth' not in info.columns or 'cluster_id' not in info.columns:
        raise ValueError(f"cluster_info.tsv has no 'cluster_id'/'depth' column in {dp}")
    return info.set_index('cluster_id')['depth'].astype(float)


def _read_spikes_us(dp: str, unit_ids: list[int]) -> dict[int, np.ndarray]:
    """Spike times (µs) of each unit in `unit_ids`, from spike_times.npy
    (samples) and spike_clusters.npy."""
    fs = _read_sample_rate(dp)
    spike_times    = np.load(os.path.join(dp, 'spike_times.npy')).ravel()
    spike_clusters = np.load(os.path.join(dp, 'spike_clusters.npy')).ravel()
    if len(spike_times) != len(spike_clusters):
        raise ValueError(f'spike_times.npy and spike_clusters.npy differ in length in {dp}')
    return {u: spike_times[spike_clusters == u].astype(np.float64) / fs * 1e6
            for u in unit_ids}


def _read_tracking_raw(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read x, y (cm) and t (µs) from a '_cm.csv' tracking file, cached per path."""
    with _tracking_lock:
        if csv_path not in _tracking_cache:
            data = (pd.read_excel(csv_path) if csv_path.lower().endswith('.xlsx')
                    else pd.read_csv(csv_path))
            # Column A = timestamp (ms), column D = x (cm), column E = y (cm)
            t = np.asarray(data.iloc[:, 0], dtype=float) * TRACKING_TIME_TO_US
            x = np.asarray(data.iloc[:, 3], dtype=float)
            y = np.asarray(data.iloc[:, 4], dtype=float)
            _tracking_cache[csv_path] = (x, y, t)
        return _tracking_cache[csv_path]


def _load_tracking_npx(csv_path: str, arena_width_cm: float,
                       half: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Neuropixels version of pc._load_tracking (cm mode). Identical
    cleaning and half-split; only the time unit (ms -> µs) differs.
    """
    x, y, t = _read_tracking_raw(csv_path)

    mask = ~np.isin(x, [1, -1])
    x, y, t = x[mask], y[mask], t[mask]

    # Pad the differential arrays with zeros/ones to prevent dropping the final frame
    dx = np.append(np.diff(x), 0)
    dy = np.append(np.diff(y), 0)
    dt = np.append(np.diff(t), 1)

    dxy = np.hypot(dx, dy)
    valid_dt = dt > 0

    speed = np.zeros_like(dxy)
    speed[valid_dt] = dxy[valid_dt] / dt[valid_dt]

    keep = np.where(valid_dt & (speed < 0.006))[0]
    x, y, t = x[keep], y[keep], t[keep]

    order = np.argsort(t)
    x, y, t = x[order], y[order], t[order]

    if half is not None:
        mid = len(t) // 2
        if half == 'first':
            x, y, t = x[:mid], y[:mid], t[:mid]
        elif half == 'second':
            x, y, t = x[mid:], y[mid:], t[mid:]

    if len(t) == 0:
        return x.astype(np.float64), y.astype(np.float64), t.astype(np.float64)

    # Coordinates are already in cm – just zero the origin for binning.
    x_cm = x - x.min()
    y_cm = y - y.min()

    return x_cm, y_cm, t


def compute_metrics_npx(csv_path: str, ntt_path: str,
                        arena_width_cm: float, target_bin_cm: float,
                        half: str | None = None) -> tuple:
    """pc.compute_metrics with the .ntt memmap replaced by the unit's
    Neuropixels spike times (µs). Every other step is unchanged.
    """
    # ── 1-2. Load, clean, and convert tracking ─────────────────────────────────
    x_cm, y_cm, t = _load_tracking_npx(csv_path, arena_width_cm, half=half)

    if len(t) == 0:
        return ({'n_spikes': 0, 'n_discarded': 0, 'peak_fr': 0.0, 'mean_fr': 0.0, 'sir': 0.0}, {})

    # ── 2b. Smooth tracking position (jump removal + Gaussian smoothing) ──────
    x_cm, y_cm = pc._smooth_tracking_position(x_cm, y_cm, t)

    # ── 3. Bin tracking positions ─────────────────────────────────────────────
    n_bins_x = int(np.ceil(x_cm.max() / target_bin_cm))
    n_bins_y = int(np.ceil(y_cm.max() / target_bin_cm))

    beh_bx = np.clip((x_cm / target_bin_cm).astype(int), 0, n_bins_x - 1)
    beh_by = np.clip((y_cm / target_bin_cm).astype(int), 0, n_bins_y - 1)

    # ── 4. Load spikes & nearest-timestamp assignment (50 ms gate) ────────────
    spike_ts = np.sort(_SPIKES_US[ntt_path])

    if half is not None:
        t_lo = t[0]  - pc.MAX_GAP_US
        t_hi = t[-1] + pc.MAX_GAP_US
        spike_ts = spike_ts[(spike_ts >= t_lo) & (spike_ts <= t_hi)]

    idx   = np.searchsorted(t, spike_ts, side='left')
    idx_l = np.clip(idx - 1, 0, len(t) - 1)
    idx_r = np.clip(idx,     0, len(t) - 1)
    dist_l   = np.abs(spike_ts - t[idx_l])
    dist_r   = np.abs(spike_ts - t[idx_r])
    nearest  = np.where(dist_l <= dist_r, idx_l, idx_r)
    min_dist = np.minimum(dist_l, dist_r)

    valid_spike  = min_dist <= pc.MAX_GAP_US
    spike_frame  = nearest[valid_spike]
    n_spikes     = int(valid_spike.sum())
    n_discarded  = int((~valid_spike).sum())

    sp_bx = beh_bx[spike_frame]
    sp_by = beh_by[spike_frame]

    # ── 5. Build occupancy and spike-count maps ────────────────────────────────
    dt_frames        = np.empty(len(t), dtype=np.float64)
    dt_frames[0]     = 1.0 / pc.fps
    raw_dt           = np.diff(t) * 1e-6

    # Cap dt_frames to avoid artificial occupancy hotspots when tracking drops
    max_frame_s      = 2.0 / pc.fps
    dt_frames[1:]    = np.minimum(raw_dt, max_frame_s)

    occ_map   = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    spike_map = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)

    np.add.at(occ_map,   (beh_bx, beh_by), dt_frames)
    np.add.at(spike_map, (sp_bx,  sp_by),  1.0)

    visit_map  = pc._visits_2d(beh_bx, beh_by, n_bins_x, n_bins_y)
    valid_mask = pc._valid_bin_mask(occ_map, visit_map)

    # ── 6. Non-smoothed firing rate map ──────────────────────────────────────
    fr_raw = np.zeros_like(occ_map)
    fr_raw[valid_mask] = spike_map[valid_mask] / occ_map[valid_mask]

    # ── 7. Smoothed firing rate map ───────────────────────────────────────────
    fr_smooth = pc._gaussian_smooth(fr_raw, valid_mask, target_bin_cm)

    # ── 8. Compute metrics ────────────────────────────────────────────────────
    ctx = dict(spike_ts=spike_ts[valid_spike], spike_frame=spike_frame, t=t,
               x_cm=x_cm, y_cm=y_cm,
               beh_bx=beh_bx, beh_by=beh_by,
               occ_map=occ_map, valid_mask=valid_mask,
               dt_frames=dt_frames,
               n_bins_x=n_bins_x, n_bins_y=n_bins_y,
               fr_raw=fr_raw, fr_smooth=fr_smooth)

    if not valid_mask.any():
        return ({'n_spikes': n_spikes, 'n_discarded': n_discarded,
                 'peak_fr': 0.0, 'mean_fr': 0.0, 'sir': 0.0,
                 'sparsity': 0.0, 'coherence': float('nan')}, ctx)

    total_occ_s = occ_map[valid_mask].sum()
    pi_flat     = occ_map[valid_mask] / total_occ_s
    ri_flat     = fr_smooth[valid_mask]
    ri_flat_raw = fr_raw[valid_mask]
    r_mean      = float(np.sum(pi_flat * ri_flat))
    r_mean_raw  = float(np.sum(pi_flat * ri_flat_raw))

    peak_fr = float(fr_raw[valid_mask].max())
    mean_fr = r_mean

    sir = 0.0
    if r_mean_raw > 0:
        nonzero = ri_flat_raw > 0
        ratio   = ri_flat_raw[nonzero] / r_mean_raw
        sir     = float(np.sum(pi_flat[nonzero] * ratio * np.log2(ratio)))

    # Sparsity = (Σ pi ri)² / Σ pi ri²   (Skaggs et al. 1996)
    spar_num = float(np.sum(pi_flat * ri_flat))
    spar_den = float(np.sum(pi_flat * ri_flat ** 2))
    sparsity = float((spar_num ** 2) / spar_den) if spar_den > 0 else 0.0

    # Spatial coherence: smoothed map vs 8-neighbour mean (Fisher Z)
    coherence = pc._compute_coherence(fr_smooth, valid_mask, n_bins_x, n_bins_y)

    metrics = {
        'n_spikes':    n_spikes,
        'n_discarded': n_discarded,
        'peak_fr':     round(peak_fr,   4),
        'mean_fr':     round(mean_fr,   4),
        'sir':         round(sir,       4),
        'sparsity':    round(sparsity,  4),
        'coherence':   round(coherence, 4) if not np.isnan(coherence) else float('nan'),
    }

    return metrics, ctx


# Route v10's pipeline through the Neuropixels adapters. pc._run_job,
# pc._plot_and_save_speed and pc._plot_and_save_dt_s look these names up in
# the pc module at call time. pc._is_circle_arena is left as is: it checks
# the session folder path for 'Circle', exactly as for tetrode data.
pc._load_tracking    = _load_tracking_npx
pc.compute_metrics   = compute_metrics_npx


#%%############################################################################
# Run metadata ################################################################
###############################################################################

NPX_CONFIG_VARS = [
    'ROOT_FOLDER', 'OUTPUT_EXCEL', 'OUTPUT_PLACE_TRUE',
    'TETRODE_EXCEL', 'COMBINED_EXCEL', 'RESULT_EXCEL',
    'UNIT_QUALITY', 'DEPTH_EXCEL', 'TRACKING_TIME_TO_US', 'ARENA_WIDTH_CM',
]


def _save_run_metadata(output_path: str) -> str:
    """Same as pc._save_run_metadata (every v10 setting in RUN_CONFIG_VARS),
    plus this script's name and the Neuropixels-specific settings."""
    rows = [
        {'parameter': 'script_name', 'value': os.path.basename(__file__)},
        {'parameter': 'analysis_module', 'value': os.path.basename(pc.__file__)},
        {'parameter': 'run_timestamp', 'value': datetime.now().isoformat(timespec='seconds')},
    ]
    rows += [{'parameter': name, 'value': getattr(pc, name)} for name in pc.RUN_CONFIG_VARS]
    rows += [{'parameter': name, 'value': globals()[name]} for name in NPX_CONFIG_VARS]

    meta_path = os.path.splitext(output_path)[0] + '_run_metadata.csv'
    pd.DataFrame(rows).to_csv(meta_path, index=False)
    print(f'[SAVED] Run metadata: {meta_path}')
    return meta_path


#%%############################################################################
# Batch run ###################################################################
###############################################################################

def _find_sessions(root: str) -> list[tuple[str, str]]:
    """(session_dir, tracking_csv) for every folder under `root` holding the
    phy output files and exactly one '_cm.csv' – the Neuropixels counterpart
    of v10's os.walk over .ntt folders. OUTPUT_PLACE_TRUE is skipped."""
    skip = os.path.normcase(os.path.abspath(OUTPUT_PLACE_TRUE))
    sessions = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if os.path.normcase(os.path.abspath(os.path.join(dirpath, d))) != skip)
        if not all(f in filenames for f in PHY_REQUIRED_FILES):
            continue
        tracking_files = [f for f in filenames if f.lower().endswith('_cm.csv')]
        if len(tracking_files) != 1:
            print(f'[SKIP] {os.path.relpath(dirpath, root)}: expected 1 _cm.csv tracking '
                  f'file, found {len(tracking_files)} {tracking_files}')
            continue
        sessions.append((dirpath, os.path.join(dirpath, tracking_files[0])))
    return sessions


DATA_SHEETS = ('Full', 'First_Half', 'Second_Half', 'PlaceFields')


def _check_tetrode_excel() -> None:
    """Fail before any analysis runs if the append settings are unusable."""
    if not TETRODE_EXCEL:
        return
    if not os.path.isfile(TETRODE_EXCEL):
        raise FileNotFoundError(f'TETRODE_EXCEL not found: {TETRODE_EXCEL}')
    same = (os.path.normcase(os.path.abspath(TETRODE_EXCEL))
            == os.path.normcase(os.path.abspath(COMBINED_EXCEL)))
    if same:
        raise ValueError('COMBINED_EXCEL must differ from TETRODE_EXCEL '
                         '(the tetrode workbook is never overwritten).')
    sheet_names = pd.ExcelFile(TETRODE_EXCEL).sheet_names
    if 'Full' not in sheet_names:
        raise ValueError(f"TETRODE_EXCEL has no 'Full' sheet (sheets: {sheet_names}).")


def _speed_summary(df_full: pd.DataFrame) -> pd.DataFrame:
    """v10's Speed_Summary sheet. When df_full has a 'data_source' column,
    per-source counts are added next to the combined 'count'."""
    def counts(df):
        is_speed    = df['speed_cell'] == True    # noqa: E712  ('not tested' compares False)
        is_place    = df['place_cell'] == True    # noqa: E712
        is_nonplace = df['place_cell'] == False   # noqa: E712
        return [len(df),
                int(is_place.sum()),
                int(is_speed.sum()),
                int((is_place & is_speed).sum()),
                int((is_nonplace & is_speed).sum()),
                int((df['speed_cell'] == 'not tested').sum())]

    summary = pd.DataFrame({
        'metric': ['Total units processed',
                   'Place cells',
                   'Speed cells (all, passed binned AND inst. tests)',
                   'Place cells also speed-modulated',
                   'Non-place cells speed-modulated',
                   'Speed not tested'],
        'count':  counts(df_full),
    })
    if 'data_source' in df_full.columns:
        for source in ('tetrode', 'neuropixels'):
            summary[f'count_{source}'] = counts(df_full[df_full['data_source'] == source])
    return summary


def _combine_with_tetrode(npx_sheets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Append the Neuropixels rows after the last tetrode row on each data
    sheet of TETRODE_EXCEL. Columns are matched by name; tetrode sheets
    other than the data sheets and Speed_Summary are carried over unchanged.
    """
    tt_sheets = pd.read_excel(TETRODE_EXCEL, sheet_name=None)

    tt_keys = set(zip(tt_sheets['Full']['session'].astype(str), tt_sheets['Full']['unit'].astype(str)))
    npx_keys = set(zip(npx_sheets['Full']['session'].astype(str), npx_sheets['Full']['unit'].astype(str)))
    overlap = tt_keys & npx_keys
    if overlap:
        print(f'[WARN] {len(overlap)} Neuropixels session/unit pair(s) already exist in '
              f'TETRODE_EXCEL and will appear twice, e.g. {sorted(overlap)[:3]}')

    combined = {}
    for name in DATA_SHEETS:
        tt  = tt_sheets.get(name, pd.DataFrame(columns=npx_sheets[name].columns)).copy()
        npx = npx_sheets[name].copy()
        if 'data_source' not in tt.columns:
            tt['data_source'] = 'tetrode'
        npx['data_source'] = 'neuropixels'
        combined[name] = pd.concat([tt, npx], ignore_index=True)
        print(f'  {name}: {len(tt)} tetrode + {len(npx)} Neuropixels row(s)')

    for name, df in tt_sheets.items():
        if name not in DATA_SHEETS and name != 'Speed_Summary':
            combined[name] = df
    return combined


def main():
    if not os.path.isdir(ROOT_FOLDER):
        raise FileNotFoundError(f'ROOT_FOLDER not found: {ROOT_FOLDER}')
    _check_tetrode_excel()
    depth_ranges = None
    if DEPTH_EXCEL:
        if not os.path.isfile(DEPTH_EXCEL):
            raise FileNotFoundError(f'DEPTH_EXCEL not found: {DEPTH_EXCEL}')
        depth_ranges = _load_depth_ranges(DEPTH_EXCEL)
        print(f'Hippocampal depth ranges for {len(depth_ranges)} session(s) '
              f'from {DEPTH_EXCEL}')
    print(f"Using '{pc.COORD_UNITS}' tracking coordinates.\n")

    _save_run_metadata(RESULT_EXCEL)

    # ── Collect jobs: every UNIT_QUALITY unit of every session folder ──────────
    sessions = _find_sessions(ROOT_FOLDER)
    print(f'Found {len(sessions)} Neuropixels session folder(s) under {ROOT_FOLDER}\n')

    all_jobs   = []
    dir_to_csv = {}
    for dirpath, csv_path in sessions:
        name = os.path.relpath(dirpath, ROOT_FOLDER)
        depth_range = None
        if depth_ranges is not None:
            depth_range = depth_ranges.get(_session_key(name))
            if depth_range is None:
                print(f'[SKIP] {name}: no hippocampal depth range in {os.path.basename(DEPTH_EXCEL)}')
                continue
        try:
            units = _read_unit_ids(dirpath, UNIT_QUALITY)
            n_quality = len(units)
            if depth_range is not None:
                # Keep units whose depth lies within the hippocampus
                # (Bottom = DG side, lower depth; Top = CA1 side, higher depth).
                bottom, top = depth_range
                depths = _read_unit_depths(dirpath)
                no_depth = [u for u in units if u not in depths.index]
                if no_depth:
                    print(f'  [WARN] {name}: no depth for cluster(s) {no_depth}; excluded')
                units = [u for u in units
                         if u in depths.index and bottom <= depths[u] <= top]
            unit_spks = _read_spikes_us(dirpath, units)
        except Exception as e:
            print(f'[SKIP] {name}: could not read phy output ({e})')
            continue
        if not units:
            where = (f' within {depth_range[0]:.0f}–{depth_range[1]:.0f} µm'
                     if depth_range is not None else '')
            print(f"[SKIP] {name}: no '{UNIT_QUALITY}' units{where} "
                  f'({n_quality} {UNIT_QUALITY} unit(s) in total)')
            continue

        dir_to_csv[dirpath] = csv_path

        _, _, t_trk = _load_tracking_npx(csv_path, ARENA_WIDTH_CM)
        in_hpc = (f' in hippocampus ({depth_range[0]:.0f}–{depth_range[1]:.0f} µm, '
                  f'of {n_quality})' if depth_range is not None else '')
        print(f'{name}: {len(units)} {UNIT_QUALITY} unit(s){in_hpc}, '
              f'arena = {"Circle" if pc._is_circle_arena(dirpath) else "2D"}, '
              f'tracking = {os.path.basename(csv_path)}')
        if len(t_trk):
            print(f'  tracking: {t_trk[0] * 1e-6:.1f}–{t_trk[-1] * 1e-6:.1f} s')

        for u in units:
            unit_name = f'cluster_{u}'
            spikes_us = unit_spks[u]
            _SPIKES_US[os.path.join(dirpath, unit_name)] = spikes_us
            all_jobs.append((dirpath, csv_path, unit_name))

            if len(t_trk) and len(spikes_us):
                in_trk = np.mean((spikes_us >= t_trk[0]) & (spikes_us <= t_trk[-1]))
                if in_trk < 0.5:
                    print(f'  [WARN] {unit_name}: only {100 * in_trk:.0f}% of spikes fall within '
                          f'the tracking time range – check the time bases')

    total_units = len(all_jobs)
    print(f'Found {total_units} unit(s) across all sessions.\n')
    if total_units == 0:
        return

    # ── Speed-vs-time plots (one per tracking file) ─────────────────────────────
    print(f'Generating speed and dt_s plots for {len(dir_to_csv)} tracking file(s)...')
    for csv_path in set(dir_to_csv.values()):
        try:
            pc._plot_and_save_speed(csv_path, ARENA_WIDTH_CM)
        except Exception as e:
            print(f'  SPEED PLOT ERROR for {csv_path}: {e}')
        try:
            pc._plot_and_save_dt_s(csv_path, ARENA_WIDTH_CM, pc.fps)
        except Exception as e:
            print(f'  DT_S PLOT ERROR for {csv_path}: {e}')
    print()

    job_args = [
        (idx, total_units, idx - 1, dirpath, csv_path, unit_name)
        for idx, (dirpath, csv_path, unit_name) in enumerate(all_jobs, start=1)
    ]

    # ── Parallel execution ────────────────────────────────────────────────────────

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=pc.MAX_WORKERS) as executor:
        futures = {executor.submit(pc._run_job, args): args for args in job_args}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: r[0]['job_order'])

    # ── Save to Excel (same sheets/columns as PlaceCell_Main_v10) ─────────────────

    column_order = ['session', 'unit', 'n_spikes', 'n_discarded',
                    'peak_fr', 'mean_fr', 'sir', 'sparsity', 'coherence',
                    'stability_score', 'stability_p_value', 'stability_n_bins',
                    'bootstrap_mean', 'bootstrap_p95', 'bootstrap_sig',
                    'coherence_bootstrap_mean', 'coherence_bootstrap_p95', 'coherence_bootstrap_sig',
                    'theta_modulated', 'theta_peak_freq',
                    'speed_score', 'speed_p_value', 'speed_r2', 'speed_beta', 'speed_f0', 'speed_modulated',
                    'speed_shuffle_mean', 'speed_shuffle_lo', 'speed_shuffle_hi',
                    'speed_shuffle_p', 'speed_modulated_shuffle', 'speed_shuffle_ran',
                    'speed_score_td', 'speed_p_value_td', 'speed_r2_td',
                    'speed_shuffle_mean_td', 'speed_shuffle_lo_td', 'speed_shuffle_hi_td',
                    'speed_shuffle_p_td', 'speed_modulated_td', 'speed_shuffle_ran_td',
                    'p_speed', 'n_speed',
                    'final_speed_score', 'speed_cell',
                    'place_cell', 'n_fields_detected']

    field_columns = ['session', 'unit', 'arena_type', 'field_number',
                     'peak_fr_hz', 'peak_fr_hz_raw',
                     'n_bins', 'area_cm2', 'pct_of_occupied_area',
                     'total_occupied_bins', 'min_field_size_bins',
                     'rate_threshold_hz', 'mean_fr_threshold_hz', 'is_centre_field',
                     'field_raw_max_fr_hz',
                     # 2-D grid (Open field / Linear track)
                     'peak_bin_x', 'peak_bin_y', 'field_raw_max_bin_x', 'field_raw_max_bin_y',
                     'com_bin_x', 'com_bin_y', 'com_cm_x', 'com_cm_y',
                     'bbox_x_min', 'bbox_x_max', 'bbox_y_min', 'bbox_y_max',
                     'bbox_cm_x_min', 'bbox_cm_x_max', 'bbox_cm_y_min', 'bbox_cm_y_max',
                     # Angular (Circle track)
                     'peak_bin_theta', 'peak_theta_deg', 'field_raw_max_bin_theta', 'com_theta_deg',
                     'arc_length_cm', 'track_width_cm', 'pct_of_track_circumference',
                     'theta_span_deg', 'bbox_theta_min_deg', 'bbox_theta_max_deg', 'wraps_theta_seam',
                     'bin_coords']

    df_full   = pd.DataFrame([r[0] for r in results], columns=column_order)
    df_first  = pd.DataFrame([r[1] for r in results], columns=column_order)
    df_second = pd.DataFrame([r[2] for r in results], columns=column_order)

    all_field_rows = [f for r in results for f in r[3]]
    df_fields = pd.DataFrame(all_field_rows, columns=field_columns)

    npx_summary = _speed_summary(df_full)
    (_, _, n_speed_cells, n_place_speed_mod,
     n_nonplace_speed_mod, n_speed_not_tested) = npx_summary['count'].tolist()

    sheets = {'Full': df_full, 'First_Half': df_first,
              'Second_Half': df_second, 'PlaceFields': df_fields}
    if TETRODE_EXCEL:
        print(f'\nAppending Neuropixels results to tetrode workbook {TETRODE_EXCEL}')
        sheets = _combine_with_tetrode(sheets)
    sheets['Speed_Summary'] = _speed_summary(sheets['Full'])

    # Data sheets first, in v10's order, then any extra tetrode sheets.
    order = [*DATA_SHEETS, 'Speed_Summary']
    order += [n for n in sheets if n not in order]
    with pd.ExcelWriter(RESULT_EXCEL, engine='openpyxl') as writer:
        for name in order:
            sheets[name].to_excel(writer, sheet_name=name, index=False)

    print(f'\nDone. Results saved to {RESULT_EXCEL}')
    if TETRODE_EXCEL:
        combined_full = sheets['Full']
        print(f'Combined workbook: {len(combined_full)} unit(s), '
              f'{int((combined_full["place_cell"] == True).sum())} place cell(s) '  # noqa: E712
              f'(tetrode + Neuropixels). Neuropixels-only numbers below.')
    print(f'Total units processed              : {len(df_full)}')
    print(f'Place cells found                  : {df_full["place_cell"].sum()}')
    print(f'Speed cells (binned AND inst. tests): {n_speed_cells}  '
          f'(shuffle-confirmed, {pc.SPEED_N_SHUFFLE} shuffles, '
          f'{pc.SPEED_SHUFFLE_MARGIN_S:.0f}s window)')
    print(f'Place cells also speed-modulated    : {n_place_speed_mod}')
    print(f'Non-place cells speed-modulated     : {n_nonplace_speed_mod}')
    print(f'Speed not tested                    : {n_speed_not_tested}')
    n_fields_col = pd.to_numeric(df_full['n_fields_detected'], errors='coerce')
    print(f'Place fields detected                : {len(df_fields)}  '
          f'(threshold method, across {int((n_fields_col > 0).sum())} place cell(s))')

    # ── Save tracking + spike times for confirmed place cells ─────────────────
    # Neuropixels counterpart of v10's .ntt/.nev/.ncs copy: the session's
    # folder path (Animal/Arena/Day/Session) is replicated under
    # OUTPUT_PLACE_TRUE, holding the tracking file plus one spike-time file (s)
    # per place cell.

    place_rows = df_full[df_full['place_cell'] == True]  # noqa: E712
    copied_tracking_dirs = set()

    for _, row in place_rows.iterrows():
        dirpath  = os.path.join(ROOT_FOLDER, row['session'])
        dest_dir = os.path.join(OUTPUT_PLACE_TRUE, row['session'])
        os.makedirs(dest_dir, exist_ok=True)

        spikes_us = _SPIKES_US.get(os.path.join(dirpath, row['unit']))
        if spikes_us is not None:
            np.save(os.path.join(dest_dir, f"{row['unit']}_spike_times_s.npy"), spikes_us * 1e-6)

        if dirpath not in copied_tracking_dirs:
            csv_path = dir_to_csv.get(dirpath)
            if csv_path and os.path.isfile(csv_path):
                shutil.copy2(csv_path, dest_dir)
            copied_tracking_dirs.add(dirpath)

    print(f'Place-cell files saved to  : {OUTPUT_PLACE_TRUE}')


if __name__ == "__main__":
    main()
