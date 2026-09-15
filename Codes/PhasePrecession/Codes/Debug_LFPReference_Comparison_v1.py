# -*- coding: utf-8 -*-
"""
Diagnostic (read-only, no production behavior changed): for every unit,
compute Steps 1-3 TWICE, once under each of two different LFP
reference-channel schemes, and report both side by side so the effect of
ThetaMod_v12's per-tetrode LFP matching can be checked against real data
before deciding whether to change it.

  'session_wide' : ONE LFP channel for the whole session -- the first .ncs
      file in natural-sort order -- used for every unit in that session.
      This is what the older pipeline (Debug_GPA_Out_..._v7.py) did.
  'tetrode'       : each unit's own tetrode-matched .ncs channel, via
      ThetaMod_v12.match_ncs_to_ntt(). This is what ThetaMod_v12.py
      currently does for every unit.

Motivation: after switching to per-tetrode matching, far more cells are
landing in the 'phase_locked' bucket than before, and Step 3 (the
~500-shuffle circular-linear fit) is taking much longer to run overall.
One likely explanation is that using the SAME tetrode's own LFP channel to
estimate a unit's theta phase risks spike-waveform bleed-through
contaminating that channel's theta-band signal, spuriously inflating
Step 2's TMI/Rayleigh significance (more cells reach the expensive Step 3)
while the true field-position/phase relationship measured in Step 3 shows
no real precession for many of them (-> phase_locked). A second, unrelated
possibility is simply that not every tetrode sits in a good theta-generating
layer, degrading GP phase quality relative to always using one known-good
channel. This script does not try to decide between these -- it just
surfaces the paired numbers per unit so that can be judged from the data.

This script does NOT modify ThetaMod_v12.py; it imports its data-loading,
Generalized-Phase, TMI, and Pass-Index machinery directly so the exact same
math is used under both reference schemes. It reuses the FULL production
Step 3 shuffle test (compute_pass_index, N_PRECESSION_SHUFFLES shuffles) on
each candidate unit under BOTH schemes, so expect this to take roughly
2x as long per candidate unit as a normal ThetaMod_v12 run over the same
units (only units that are TMI-significant under at least one scheme pay
that cost, same gate as production). Use MAX_SESSIONS / MAX_UNITS_PER_SESSION
below to bound a first exploratory run.

Output: ROOT_FOLDER/lfp_reference_comparison.xlsx ('Comparison' + 'Summary'
sheets) and ROOT_FOLDER/lfp_reference_comparison.png (paired scatter plots).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ThetaMod_v12 as tm

# ============================================================================
# Configuration -- EDIT THESE
# ============================================================================

ROOT_FOLDER = tm.ROOT_FOLDER          # reuse ThetaMod_v12's own data root by default
OUTPUT_XLSX_NAME = 'lfp_reference_comparison.xlsx'
OUTPUT_PLOT_NAME = 'lfp_reference_comparison.png'

MAX_SESSIONS = None              # int to cap sessions processed (None = all)
MAX_UNITS_PER_SESSION = None     # int to cap units processed per session (None = all)


# ============================================================================

def _load_lfp_bundle(ncs_path: Path):
    """Load + GP-filter one .ncs file. Returns a dict matching the fields
    process_session's lfp_cache stores in ThetaMod_v12.py, so downstream
    calls (compute_theta_modulation / compute_pass_index) are identical."""
    lfp_sig, lfp_ts, lfp_fs = tm.load_ncs(ncs_path)
    filtered_lfp = tm.bandpass_filter(lfp_sig, tm.LFP_FILTER_BAND[0], tm.LFP_FILTER_BAND[1], lfp_fs)
    xgp_lfp, _wt_pre, _idx = tm.generalized_phase_vector(filtered_lfp, lfp_fs, tm.LFP_FILTER_BAND[0])
    lfp_phase_unwrapped = np.unwrap(np.angle(xgp_lfp))
    return dict(lfp_sig=lfp_sig, lfp_ts=lfp_ts, lfp_fs=lfp_fs,
                lfp_phase_unwrapped=lfp_phase_unwrapped)


def _theta_and_precession(spk_ts, lfp_bundle, pos_ts, pos_xy, rng):
    """Run Steps 1-3 for one unit against one LFP reference bundle. Returns
    a flat dict of the metrics worth comparing across schemes."""
    metrics, _phase_deg = tm.compute_theta_modulation(
        spk_ts, lfp_bundle['lfp_ts'], lfp_bundle['lfp_phase_unwrapped'], rng)

    out = dict(
        TMI=metrics.get('TMI', np.nan),
        TMI_Significant=metrics.get('TMI_Significant', False),
        Rayleigh_p=metrics.get('Rayleigh_p', np.nan),
        SignificantThetaModulation=metrics.get('SignificantThetaModulation', False),
        MRL=metrics.get('MRL', np.nan),
        PrecessionTested=False, rho=np.nan, slope_deg_per_pass=np.nan,
        p_rho_shuffle=np.nan, PrecessionClass=None,
    )

    if not metrics['TMI_Significant'] or pos_ts is None:
        return out

    t_start = max(pos_ts.min(), lfp_bundle['lfp_ts'].min())
    t_stop = min(pos_ts.max(), lfp_bundle['lfp_ts'].max())
    spk_ts_overlap = spk_ts[(spk_ts >= t_start) & (spk_ts <= t_stop)]
    if len(spk_ts_overlap) < tm.MIN_SPIKES_FOR_FIT:
        return out

    try:
        results = tm.compute_pass_index(
            pos_ts, pos_xy, spk_ts_overlap, lfp_bundle['lfp_ts'], lfp_bundle['lfp_sig'],
            lfp_bundle['lfp_fs'], rng, method=tm.METHOD, binside=tm.BINSIDE,
            smth_width=tm.SMTH_WIDTH, filter_band=tm.FILTER_BAND,
            lfp_filter_band=tm.LFP_FILTER_BAND, slope_bnds=tm.SLOPE_BNDS)
    except Exception as exc:
        print(f'      precession fit error: {exc}')
        return out

    out.update(PrecessionTested=True, rho=results['rho'],
                slope_deg_per_pass=results['slope_deg_per_pass'],
                p_rho_shuffle=results['p_rho_shuffle'],
                PrecessionClass=results['precession_class'])
    return out


def process_session(data_folder: Path, rng, max_units=None) -> list[dict]:
    ncs_files = sorted(data_folder.glob('*.ncs'), key=tm._natural_key)
    if not ncs_files:
        return []
    session_wide_ncs = ncs_files[0]

    pos_ts = pos_xy = None
    try:
        tracking_path = tm._find_tracking_file(data_folder)
        pos_ts, pos_xy = tm.load_tracking(tracking_path, tm.TRACKING_TIME_UNIT)
    except FileNotFoundError:
        pass

    session_label = '_'.join(data_folder.parts[-3:])
    ntt_files = sorted(data_folder.glob('*.ntt'), key=tm._natural_key)
    if max_units is not None:
        ntt_files = ntt_files[:max_units]

    lfp_cache: dict[Path, dict] = {}

    def get_bundle(ncs_path: Path) -> dict:
        if ncs_path not in lfp_cache:
            print(f'    loading/filtering LFP: {ncs_path.name}')
            lfp_cache[ncs_path] = _load_lfp_bundle(ncs_path)
        return lfp_cache[ncs_path]

    session_bundle = get_bundle(session_wide_ncs)

    rows = []
    for ntt_path in ntt_files:
        try:
            tetrode_ncs = tm.match_ncs_to_ntt(ntt_path, ncs_files)
        except FileNotFoundError as exc:
            print(f'  {ntt_path.name}: {exc} -- skipping.')
            continue
        tetrode_bundle = get_bundle(tetrode_ncs)
        same_channel = (tetrode_ncs == session_wide_ncs)

        units = tm.load_ntt_spike_times(ntt_path)
        for cell_number, spk_ts in units.items():
            unit_label = f'{ntt_path.stem}_cell{cell_number}' if len(units) > 1 else ntt_path.stem
            print(f'  {unit_label}: session-wide={session_wide_ncs.name}  '
                  f'tetrode={tetrode_ncs.name}{" (same)" if same_channel else ""}')

            session_metrics = _theta_and_precession(spk_ts, session_bundle, pos_ts, pos_xy, rng)
            tetrode_metrics = _theta_and_precession(spk_ts, tetrode_bundle, pos_ts, pos_xy, rng)

            row = dict(Session=session_label, FolderPath=str(data_folder), Unit=unit_label,
                       ntt_file=ntt_path.name, SessionWideLFP=session_wide_ncs.name,
                       TetrodeLFP=tetrode_ncs.name, SameChannel=same_channel,
                       n_spikes_total=len(spk_ts))
            row.update({f'{k}_session': v for k, v in session_metrics.items()})
            row.update({f'{k}_tetrode': v for k, v in tetrode_metrics.items()})

            both_tested = session_metrics['PrecessionTested'] and tetrode_metrics['PrecessionTested']
            row['ClassificationChanged'] = (
                both_tested and session_metrics['PrecessionClass'] != tetrode_metrics['PrecessionClass'])
            row['TMI_Significant_only_tetrode'] = (
                tetrode_metrics['TMI_Significant'] and not session_metrics['TMI_Significant'])
            row['TMI_Significant_only_session'] = (
                session_metrics['TMI_Significant'] and not tetrode_metrics['TMI_Significant'])

            print(f'      TMI  session={session_metrics["TMI"]:.3f} (sig={session_metrics["TMI_Significant"]})  '
                  f'tetrode={tetrode_metrics["TMI"]:.3f} (sig={tetrode_metrics["TMI_Significant"]})')
            if both_tested:
                print(f'      class session={session_metrics["PrecessionClass"]}  '
                      f'tetrode={tetrode_metrics["PrecessionClass"]}'
                      f'{"  <-- CHANGED" if row["ClassificationChanged"] else ""}')

            rows.append(row)

    return rows


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    n_total = len(df)
    n_tmi_session = int((df['TMI_Significant_session'] == True).sum())      # noqa: E712
    n_tmi_tetrode = int((df['TMI_Significant_tetrode'] == True).sum())      # noqa: E712
    n_tmi_only_tetrode = int(df['TMI_Significant_only_tetrode'].sum())
    n_tmi_only_session = int(df['TMI_Significant_only_session'].sum())
    n_both_tested = int((df['PrecessionTested_session'] & df['PrecessionTested_tetrode']).sum())
    n_changed = int(df['ClassificationChanged'].sum())
    n_same_channel = int(df['SameChannel'].sum())

    session_locked = (df['PrecessionClass_session'] != 'phase_locked') & (df['PrecessionClass_tetrode'] == 'phase_locked')
    n_flip_to_locked = int((session_locked & df['PrecessionTested_session'] & df['PrecessionTested_tetrode']).sum())

    rows = [
        dict(Metric='Units compared', Count=n_total),
        dict(Metric='Units where tetrode-matched channel == session-wide channel', Count=n_same_channel),
        dict(Metric='TMI_Significant under session-wide reference', Count=n_tmi_session),
        dict(Metric='TMI_Significant under tetrode reference', Count=n_tmi_tetrode),
        dict(Metric='TMI_Significant ONLY under tetrode reference (not session-wide)', Count=n_tmi_only_tetrode),
        dict(Metric='TMI_Significant ONLY under session-wide reference (not tetrode)', Count=n_tmi_only_session),
        dict(Metric='Units precession-tested under BOTH schemes', Count=n_both_tested),
        dict(Metric='PrecessionClass differs between schemes', Count=n_changed),
        dict(Metric='  of which: precessing/recessing (session-wide) -> phase_locked (tetrode)',
             Count=n_flip_to_locked),
    ]
    return pd.DataFrame(rows, columns=['Metric', 'Count'])


def plot_comparison(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    colors = np.where(df['SameChannel'], 'tab:red', 'tab:blue')
    ax.scatter(df['TMI_session'], df['TMI_tetrode'], c=colors, s=18, alpha=0.6, edgecolor='none')
    lims = [0, 1]
    ax.plot(lims, lims, 'k--', linewidth=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('TMI (session-wide reference)')
    ax.set_ylabel('TMI (tetrode-matched reference)')
    ax.set_title('Theta Modulation Index by reference scheme')

    both = df[df['PrecessionTested_session'] & df['PrecessionTested_tetrode']].copy()
    ax = axes[1]
    if len(both):
        colors2 = np.where(both['SameChannel'], 'tab:red', 'tab:blue')
        x = both['slope_deg_per_pass_session'].abs()
        y = both['slope_deg_per_pass_tetrode'].abs()
        ax.scatter(x, y, c=colors2, s=18, alpha=0.6, edgecolor='none')
        lim = max(x.max(), y.max(), 1.0) * 1.05
        ax.plot([0, lim], [0, lim], 'k--', linewidth=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('|slope| deg/pass (session-wide reference)')
    ax.set_ylabel('|slope| deg/pass (tetrode-matched reference)')
    ax.set_title('Precession slope magnitude by reference scheme\n(units precession-tested under both)')

    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red', label='same channel'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', label='different channel')]
    fig.legend(handles=legend_elems, loc='lower center', ncol=2, fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    rng = np.random.default_rng(tm.RANDOM_SEED)

    session_folders = tm.find_session_folders(ROOT_FOLDER)
    if MAX_SESSIONS is not None:
        session_folders = session_folders[:MAX_SESSIONS]
    if not session_folders:
        raise FileNotFoundError(f'No folders with both .ncs and .ntt files found under {ROOT_FOLDER}')

    all_rows = []
    for data_folder in session_folders:
        print(f'\n=== Session: {data_folder} ===')
        try:
            all_rows.extend(process_session(data_folder, rng, max_units=MAX_UNITS_PER_SESSION))
        except Exception as exc:
            print(f'ERROR processing {data_folder}: {exc}')
            continue

    if not all_rows:
        print('No units produced comparison rows.')
        return

    df = pd.DataFrame(all_rows)
    summary_df = build_summary(df)

    excel_path = ROOT_FOLDER / OUTPUT_XLSX_NAME
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Comparison', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    plot_path = ROOT_FOLDER / OUTPUT_PLOT_NAME
    plot_comparison(df, plot_path)

    print(f'\nDone. {len(df)} unit(s) compared.')
    print(summary_df.to_string(index=False))
    print(f'\nComparison table: {excel_path}')
    print(f'Comparison plot:  {plot_path}')


if __name__ == '__main__':
    main()
