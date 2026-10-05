# -*- coding: utf-8 -*-
"""
Re-compute the spatial information rate (SIR) and its location-shuffling
bootstrap on a dataset that PlaceCell_Main_v11.py has already processed, using
the occupancy-weighted smoothed ratemap (USE_OCC_WEIGHTED_SI / SI_MIN_OCC_S in
PlaceCell_Main_v11.py), and update the old results Excel file.

For every unit in the old Excel file (Full, First_Half and Second_Half sheets):
  - sir, bootstrap_mean, bootstrap_p95, bootstrap_sig are replaced with the new
    values; the previous values are kept in *_old columns for comparison.
  - a unit previously classified place_cell = True that FAILS the new bootstrap
    test is changed to place_cell = False (no unit is promoted to True, since
    field detection etc. is not re-run here).
  - Full-sheet units demoted this way have their rows removed from the
    PlaceFields sheet and n_fields_detected cleared; Speed_Summary is recomputed.
All other columns and sheets are copied unchanged.

Output : <old Excel folder>/All_TT_PlaceChar_AdptBin.xlsx
Figures: <unit folder>/shuffling_AdptBin/<unit>_<label>_bootstrapping.png

The real-data SIR and every shuffle use PlaceCell_Main_v11._compute_sir, so the
bootstrap null distribution is built with exactly the same SIR calculation.
Binning/tracking settings (fps, target_bin_cm, arena_width_cm, valid-bin
criteria, COORD_UNITS, ...) are taken from PlaceCell_Main_v11.py and must match
the settings of the original run (checked against its _run_metadata.csv).
"""

import os
import sys
import random
import concurrent.futures
import threading
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import PlaceCell_Main_v11 as pc  # noqa: E402

# ── Parameters to edit ────────────────────────────────────────────────────────
root_folder = r'X:\NMR_group_data\Runita\Analysis\Mean_KDE_Open_PascalOldenburg\SessionType_Sorted\CorrectedData'
old_excel   = r'X:\NMR_group_data\Runita\Analysis\Mean_KDE_Open_PascalOldenburg\SessionType_Sorted\CorrectedData\All_TT_PlaceChar_VisitCrit.xlsx'

OUTPUT_NAME     = 'All_TT_PlaceChar_AdptBin.xlsx'   # saved next to old_excel
SHUFFLE_DIRNAME = 'shuffling_AdptBin'               # bootstrap figures, per unit folder

# SIR on the occupancy-weighted smoothed ratemap, bins with >= 0.5 s occupancy.
pc.USE_OCC_WEIGHTED_SI = True
pc.SI_MIN_OCC_S        = 0.5

MAX_WORKERS = pc.MAX_WORKERS

# (sheet name, half argument for compute_metrics, figure label) -- labels match
# the ones PlaceCell_Main_v11.py used for its bootstrap figures.
HALVES = (('Full',        None,     'full'),
          ('First_Half',  'first',  'first_half'),
          ('Second_Half', 'second', 'second_half'))

SIR_COLS = ('sir', 'bootstrap_mean', 'bootstrap_p95', 'bootstrap_sig')

# Settings that must be identical to the original run for the rows to be
# comparable (checked against the original run's metadata CSV, if present).
MUST_MATCH_VARS = ('fps', 'target_bin_cm', 'arena_width_cm',
                   'use_min_occ_s', 'min_occ_s', 'use_min_visits', 'min_visits',
                   'MAX_GAP_US', 'POS_JUMP_THRESH_CMS', 'POS_SMOOTH_SIGMA_SMP',
                   'COORD_UNITS', 'GAUSSIAN_SIGMA_CM')

_print_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_settings_match(old_excel_path: str) -> None:
    """Warn if the binning/tracking settings differ from the original run."""
    meta_path = os.path.splitext(old_excel_path)[0] + '_run_metadata.csv'
    if not os.path.isfile(meta_path):
        print(f'[WARN] No run metadata found ({meta_path}); cannot verify that the '
              f'settings in PlaceCell_Main_v11.py match the original run.\n')
        return
    old = dict(pd.read_csv(meta_path, dtype=str).values)
    mismatches = [(name, old[name], getattr(pc, name)) for name in MUST_MATCH_VARS
                  if name in old and str(old[name]) != str(getattr(pc, name))]
    for name, old_val, new_val in mismatches:
        print(f'[WARN] {name}: original run = {old_val}, PlaceCell_Main_v11.py now = {new_val}')
    if mismatches:
        print('[WARN] Settings differ from the original run -- new SIR values will not '
              'be computed on the same ratemaps.\n')


def _find_tracking_file(dirpath: str, exclude: set[str]) -> str | None:
    """Same tracking-file selection as PlaceCell_Main_v11.py's batch scan."""
    files = [f for f in os.listdir(dirpath)
             if f.lower().endswith(('.csv', '.xlsx'))
             and f.lower() not in exclude
             and not f.lower().endswith('_run_metadata.csv')]
    if pc.COORD_UNITS == 'cm':
        files = [f for f in files if f.lower().endswith('_cm.csv')]
    else:
        files = [f for f in files if not f.lower().endswith('_cm.csv')]
    return os.path.join(dirpath, files[0]) if len(files) == 1 else None


def _sir_bootstrap(ctx: dict, real_sir: float, bin_cm: float) -> tuple[dict, np.ndarray | None]:
    """SIR-only location-shuffling bootstrap, same shuffling as
    PlaceCell_Main_v11._run_bootstrap: the position series is circularly
    shifted by a random offset (>= 20 s from either end) while spike-to-frame
    assignments stay fixed. Each shuffle's SIR uses pc._compute_sir."""
    spike_frame = ctx['spike_frame']
    n_frames    = len(ctx['t'])
    margin      = int(20 * pc.fps)

    if len(spike_frame) == 0:
        return {'bootstrap_mean': float('nan'), 'bootstrap_p95': float('nan'),
                'bootstrap_sig': False}, None
    if n_frames <= 2 * margin:
        return {'bootstrap_mean': float('nan'), 'bootstrap_p95': float('nan'),
                'bootstrap_sig': None}, None

    beh_bx, beh_by = ctx['beh_bx'], ctx['beh_by']
    occ_map, valid_mask = ctx['occ_map'], ctx['valid_mask']
    shape = (ctx['n_bins_x'], ctx['n_bins_y'])

    sir_i = np.zeros(pc.N_BOOTSTRAP, dtype=np.float64)
    for i in range(pc.N_BOOTSTRAP):
        rnd        = random.randint(margin, n_frames - margin)
        shuf_frame = (spike_frame + rnd) % n_frames

        spike_map = np.zeros(shape, dtype=np.float64)
        np.add.at(spike_map, (beh_bx[shuf_frame], beh_by[shuf_frame]), 1.0)
        fr_raw = np.zeros_like(spike_map)
        np.divide(spike_map, occ_map, out=fr_raw, where=valid_mask)

        sir_i[i] = pc._compute_sir(spike_map, occ_map, fr_raw, valid_mask, bin_cm)

    bootstrap_p95 = float(np.percentile(sir_i, 95))
    return {'bootstrap_mean': round(float(np.mean(sir_i)), 4),
            'bootstrap_p95':  round(bootstrap_p95, 4),
            'bootstrap_sig':  bool(real_sir > bootstrap_p95)}, sir_i


def _plot_sir_bootstrap(sir_i: np.ndarray, real_sir: float, boot: dict,
                        ntt_path: str, label: str) -> None:
    """SIR shuffle histogram, same layout as the SIR panel of _run_bootstrap."""
    fig = Figure(figsize=(5.5, 4.5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    counts = np.asarray(ax.hist(sir_i, 100, color='black')[0])
    max_count = float(counts.max()) if counts.max() > 0 else 1.0

    box_plot = ax.boxplot(sir_i, whis=[5, 95], orientation='horizontal', showfliers=False,  # type: ignore
                          positions=[-max_count / 10], widths=max_count / 15)
    ax.plot([real_sir, real_sir], [0, max_count], 'r-.')

    ci95 = float(box_plot['whiskers'][1].get_xdata()[1])
    ax.set_title(f"Bootstrap mean SIR = {boot['bootstrap_mean']:.3f} bits/spike\n"
                 f"Upper 95% CI = {ci95:.3f} bits/spike\n"
                 f"Cell SIR = {real_sir:.3f} bits/spike "
                 f"({'p < 0.05' if boot['bootstrap_sig'] else 'ns'})",
                 multialignment='center')
    yticks = [0, max_count * 0.25, max_count * 0.5, max_count * 0.75, max_count]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(round(v, 1)) for v in yticks])
    ax.set_ylabel('count')
    ax.set_xlabel(f'spatial information rate (bits/spike)\n'
                  f'occupancy-weighted map, bins >= {pc.SI_MIN_OCC_S:g} s')
    ax.set_ylim(-max_count / 7, max_count * 1.1)
    fig.tight_layout()

    ntt_name = os.path.splitext(os.path.basename(ntt_path))[0]
    save_dir = os.path.join(os.path.dirname(ntt_path), SHUFFLE_DIRNAME)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{ntt_name}_{label}_bootstrapping.png')
    fig.savefig(save_path, dpi=150)
    with _print_lock:
        print(f'  [SAVED] {save_path}')


def _process_unit(args) -> tuple[tuple[str, str], dict]:
    """New SIR + bootstrap for one unit, for the full session and both halves.
    Returns {sheet: {sir, bootstrap_mean, bootstrap_p95, bootstrap_sig} | None};
    None means that sheet's old values are kept (unit could not be processed)."""
    idx, total, session, unit, csv_path = args
    dirpath  = root_folder if session == '.' else os.path.join(root_folder, session)
    ntt_path = os.path.join(dirpath, unit)

    with _print_lock:
        print(f'[{idx}/{total}  {100 * idx / total:.1f}%]  {session}  |  {unit}')

    out: dict = {}
    for sheet, half, label in HALVES:
        try:
            metrics, ctx = pc.compute_metrics(csv_path, ntt_path, pc.arena_width_cm,
                                              pc.target_bin_cm, half=half)
            if not ctx:
                # Same as the main script: no tracking data -> no bootstrap.
                out[sheet] = {'sir': metrics.get('sir'), 'bootstrap_mean': None,
                              'bootstrap_p95': None, 'bootstrap_sig': None}
                continue
            real_sir = float(metrics.get('sir', 0.0))
            boot, sir_i = _sir_bootstrap(ctx, real_sir, pc.target_bin_cm)
            if sir_i is not None:
                _plot_sir_bootstrap(sir_i, real_sir, boot, ntt_path, label)
            out[sheet] = {'sir': round(real_sir, 4), **boot}
        except Exception as e:
            with _print_lock:
                print(f'  ERROR in {unit} [{label}]: {e} -- keeping old values')
            out[sheet] = None
    return (session, unit), out


def _update_sheet(df: pd.DataFrame, sheet: str, results: dict) -> set[tuple[str, str]]:
    """Write new SIR/bootstrap values into `df` in place (old values kept in
    *_old columns) and demote place cells that fail the new bootstrap.
    Returns the (session, unit) keys demoted from place_cell True to False."""
    for col in SIR_COLS + ('place_cell',):
        df[f'{col}_old'] = df[col]
    df['bootstrap_sig'] = df['bootstrap_sig'].astype(object)
    df['place_cell']    = df['place_cell'].astype(object)

    demoted = set()
    for i, row in df.iterrows():
        key = (str(row['session']), str(row['unit']))
        new = results.get(key, {}).get(sheet)
        if new is None:
            continue
        for col in SIR_COLS:
            df.at[i, col] = new[col]
        if row['place_cell'] == True and new['bootstrap_sig'] is not True:  # noqa: E712
            df.at[i, 'place_cell'] = False
            demoted.add(key)
    return demoted


def _speed_summary(df_full: pd.DataFrame) -> pd.DataFrame:
    """Same Speed_Summary sheet as PlaceCell_Main_v11.py, from the updated Full sheet."""
    is_speed    = df_full['speed_cell'] == True    # noqa: E712
    is_place    = df_full['place_cell'] == True    # noqa: E712
    is_nonplace = df_full['place_cell'] == False   # noqa: E712
    return pd.DataFrame({
        'metric': ['Total units processed',
                   'Place cells',
                   'Speed cells (all, passed binned AND inst. tests)',
                   'Place cells also speed-modulated',
                   'Non-place cells speed-modulated',
                   'Speed not tested'],
        'count':  [len(df_full),
                   int(is_place.sum()),
                   int(is_speed.sum()),
                   int((is_place & is_speed).sum()),
                   int((is_nonplace & is_speed).sum()),
                   int((df_full['speed_cell'] == 'not tested').sum())],
    })


def _save_run_metadata(output_path: str) -> None:
    rows = [{'parameter': 'script_name', 'value': os.path.basename(__file__)},
            {'parameter': 'run_timestamp', 'value': datetime.now().isoformat(timespec='seconds')},
            {'parameter': 'root_folder', 'value': root_folder},
            {'parameter': 'old_excel', 'value': old_excel},
            {'parameter': 'output_excel', 'value': output_path}]
    rows += [{'parameter': name, 'value': getattr(pc, name)} for name in pc.RUN_CONFIG_VARS
             if name not in ('root_folder', 'output_excel', 'Output_PlaceTrue')]
    meta_path = os.path.splitext(output_path)[0] + '_run_metadata.csv'
    pd.DataFrame(rows).to_csv(meta_path, index=False)
    print(f'[SAVED] Run metadata: {meta_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    output_excel = os.path.join(os.path.dirname(old_excel), OUTPUT_NAME)
    print(f'SIR: occupancy-weighted smoothed map (sigma = {pc.GAUSSIAN_SIGMA_CM} cm), '
          f'bins >= {pc.SI_MIN_OCC_S} s occupancy; {pc.N_BOOTSTRAP} shuffles.\n')

    _check_settings_match(old_excel)
    _save_run_metadata(output_excel)

    sheets = pd.read_excel(old_excel, sheet_name=None)
    for sheet, _, _ in HALVES:
        if sheet not in sheets:
            raise KeyError(f"Sheet '{sheet}' not found in {old_excel}")
        sheets[sheet]['session'] = sheets[sheet]['session'].astype(str)
        sheets[sheet]['unit']    = sheets[sheet]['unit'].astype(str)

    # Units to re-run: every (session, unit) in the Full sheet.
    exclude = {os.path.basename(old_excel).lower(), OUTPUT_NAME.lower()}
    units   = list(dict.fromkeys(zip(sheets['Full']['session'], sheets['Full']['unit'])))
    jobs, missing = [], []
    for session, unit in units:
        dirpath  = root_folder if session == '.' else os.path.join(root_folder, session)
        csv_path = _find_tracking_file(dirpath, exclude) if os.path.isdir(dirpath) else None
        if csv_path is None or not os.path.isfile(os.path.join(dirpath, unit)):
            missing.append((session, unit))
        else:
            jobs.append((session, unit, csv_path))

    for session, unit in missing:
        print(f'[WARN] {session} | {unit}: .ntt or tracking file not found -- keeping old values')
    print(f'Re-computing SIR for {len(jobs)} of {len(units)} unit(s).\n')

    job_args = [(i, len(jobs), s, u, c) for i, (s, u, c) in enumerate(jobs, start=1)]
    results: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for key, out in executor.map(_process_unit, job_args):
            results[key] = out

    # ── Update sheets ─────────────────────────────────────────────────────────
    demoted_by_sheet = {sheet: _update_sheet(sheets[sheet], sheet, results)
                        for sheet, _, _ in HALVES}

    demoted_full = demoted_by_sheet['Full']
    if demoted_full:
        df_full = sheets['Full']
        is_demoted = [(s, u) in demoted_full for s, u in zip(df_full['session'], df_full['unit'])]
        df_full['n_fields_detected'] = df_full['n_fields_detected'].astype(object)
        df_full.loc[is_demoted, 'n_fields_detected'] = None

        if 'PlaceFields' in sheets:
            df_fields = sheets['PlaceFields']
            keep = [(str(s), str(u)) not in demoted_full
                    for s, u in zip(df_fields['session'], df_fields['unit'])]
            sheets['PlaceFields'] = df_fields[keep]

    if 'Speed_Summary' in sheets and 'speed_cell' in sheets['Full']:
        sheets['Speed_Summary'] = _speed_summary(sheets['Full'])

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\nDone. Results saved to {output_excel}')
    for sheet, _, _ in HALVES:
        df = sheets[sheet]
        n_old = int((df['place_cell_old'] == True).sum())   # noqa: E712
        n_new = int((df['place_cell'] == True).sum())       # noqa: E712
        n_sig = int((df['bootstrap_sig'] == True).sum())    # noqa: E712
        print(f'{sheet:12s}: place cells {n_old} -> {n_new} '
              f'({len(demoted_by_sheet[sheet])} failed new bootstrap); '
              f'{n_sig}/{len(df)} units pass new SIR bootstrap')
    if missing:
        print(f'{len(missing)} unit(s) not found on disk -- old values kept.')
