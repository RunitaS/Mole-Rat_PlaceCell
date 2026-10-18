# -*- coding: utf-8 -*-
"""
Fig1def_QualityGates_AcceptReject.py

Accepted / rejected units at each quality gate of Fig1def_spike_quality_v2.py,
read from its QUALITY_METRICS.xlsx ('Quality Metrics' sheet).

Gates (same thresholds as Fig1def_spike_quality_v2.py):
    ISI      : isi_violation_pct    < ISI_ACCEPT_MAX_PCT   (ISI_GOOD_PCT = 0.1 % in v2)
    SNR      : snr                  > SNR_ACCEPT_MIN       (SNR_LOW_OK  = 2.5)
    TmplCorr : template_correlation > TMPL_CORR_ACCEPT_MIN (0.5)
    Manual   : manual review of multi-trigger units ('manual_review' column);
               applies only to reviewed units, its decision overrides the three
               automatic gates. Units never reviewed (non-MT, or 'not reviewed')
               are not tested by this gate.

Empty .ntt files (n_spikes == 0) are skipped: they are removed before any count
or figure and only listed in the 'Skipped_Empty_Ntt' sheet of the summary Excel.

A NaN metric fails its gate (as in v2), but is kept apart from a failure on the
threshold: NaN means the metric could not be computed (1 spike → ISI and
TmplCorr NaN). These rejections are drawn hatched and marked '*' in the
gate-combination labels.

Final decision = 'accepted' column of the sheet. The script re-derives it from
the gates above and reports any unit where the two disagree.

Rejected at the end (accepted == False) units are split into
    Rejected by all gates        : failed every gate that was applied to the unit
                                   (ISI, SNR, TmplCorr, + Manual if reviewed)
    Rejected by >= 1 gate        : failed at least one applied gate

Outputs (OUTPUT_DIR):
    Fig1_Percent_Accepted_Rejected_PerGate.png/.svg   each gate on all units (independent)
    Fig2_Sequential_Gates.png/.svg                    gates applied in order (funnel)
    Fig3_Rejected_Common_vs_AtLeastOne.png/.svg       final-rejected units per gate,
                                                      common to all gates vs >= 1 gate,
                                                      and every exact combination
    Fig4_Rejected_Metric_Distributions.png/.svg       ISI / SNR / TmplCorr of final-rejected
                                                      units (all gates vs some gates)
    QualityGates_Summary.xlsx                         per-gate, sequential, combinations,
                                                      unit lists and per-unit gate flags
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import Patch

# ── User settings ─────────────────────────────────────────────────────────────

INPUT_EXCEL = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data\QUALITY_METRICS.xlsx'
SHEET       = 'Quality Metrics'
OUTPUT_DIR  = os.path.join(os.path.dirname(INPUT_EXCEL), 'quality_gate_analysis')

# Must match Fig1def_spike_quality_v2.py (is_unit_accepted)
ISI_ACCEPT_MAX_PCT   = 0.1
SNR_ACCEPT_MIN       = 2.5
TMPL_CORR_ACCEPT_MIN = 0.5

# ── Plot style (same scheme as PlaceCell_CompareRuns.py) ──────────────────────

# Gate colours are equidistant on the magma colormap: n gates sit at
# k/(n+1), k = 1..n, so the near-black / near-white ends are never used.
FILE_CMAP      = 'magma'
REJECT_COLOR   = '#e1e0d9'
COMMON_COLOR   = '#52514e'
INK_PRIMARY    = '#0b0b0b'
INK_SECONDARY  = '#52514e'
INK_MUTED      = '#898781'
GRID_COLOR     = '#e1e0d9'
SURFACE        = '#fcfcfb'

ATLEAST1_COLOR = INK_MUTED      # 'rejected by >= 1 gate'
ALLGATE_COLOR  = COMMON_COLOR   # 'rejected by all gates' (common)
NAN_HATCH      = '////'         # rejected because the metric is NaN (< 2 spikes)
NAN_LABEL      = 'metric NaN (< 2 spikes)'

# (key, metric column, long name, short tick label)
GATES = [
    ('isi',    'isi_violation_pct',    f'ISI violations < {ISI_ACCEPT_MAX_PCT:g} %',
     f'ISI viol.\n< {ISI_ACCEPT_MAX_PCT:g} %'),
    ('snr',    'snr',                  f'SNR > {SNR_ACCEPT_MIN:g}',
     f'SNR\n> {SNR_ACCEPT_MIN:g}'),
    ('tc',     'template_correlation', f'Template correlation > {TMPL_CORR_ACCEPT_MIN:g}',
     f'Template corr.\n> {TMPL_CORR_ACCEPT_MIN:g}'),
    ('manual', None,                   'Manual multi-trigger review',
     'Manual MT\nreview'),
]
GATE_KEYS  = [k for k, *_ in GATES]
AUTO_KEYS  = ['isi', 'snr', 'tc']
GATE_COL   = {k: col for k, col, *_ in GATES}
GATE_NAME  = {k: name for k, _, name, _ in GATES}
GATE_SHORT = {k: short for k, _, _, short in GATES}
GATE_ABBR  = {'isi': 'ISI', 'snr': 'SNR', 'tc': 'TmplCorr', 'manual': 'Manual'}
GATE_COLOR = {k: to_hex(matplotlib.colormaps[FILE_CMAP](v))
              for k, v in zip(GATE_KEYS, np.linspace(0, 1, len(GATES) + 2)[1:-1])}


# ── Loading and gate flags ────────────────────────────────────────────────────

def _to_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ('true', '1', 'yes')
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    return bool(v)


def _load(path):
    df = pd.read_excel(path, sheet_name=SHEET, engine='openpyxl')
    df['accepted']      = df['accepted'].apply(_to_bool)
    df['multi_trigger'] = df['multi_trigger'].apply(_to_bool)
    for k in AUTO_KEYS:
        df[GATE_COL[k]] = pd.to_numeric(df[GATE_COL[k]], errors='coerce')
    return df


def _gate_flags(df):
    """Per unit: pass_/tested_/nan_<gate>, the derived decision, and the
    failed-gate summary used for the final-rejected categories."""
    out = pd.DataFrame(index=df.index)
    out['pass_isi'] = df['isi_violation_pct'] < ISI_ACCEPT_MAX_PCT      # NaN -> False
    out['pass_snr'] = df['snr'] > SNR_ACCEPT_MIN
    out['pass_tc']  = df['template_correlation'] > TMPL_CORR_ACCEPT_MIN
    for k in AUTO_KEYS:
        out[f'tested_{k}'] = True
        out[f'nan_{k}']    = df[GATE_COL[k]].isna()

    review = df['manual_review'].fillna('').astype(str).str.strip().str.lower()
    out['tested_manual'] = review.str.startswith('accepted') | (review == 'rejected')
    out['pass_manual']   = review.str.startswith('accepted')
    out['nan_manual']    = False

    out['pass_auto']   = out[[f'pass_{k}' for k in AUTO_KEYS]].all(axis=1)
    # Manual decision overrides the automatic one for reviewed units
    out['derived_accepted'] = np.where(out['tested_manual'], out['pass_manual'], out['pass_auto'])
    out['final_accepted']   = df['accepted'].values

    fail     = pd.DataFrame({k: out[f'tested_{k}'] & ~out[f'pass_{k}'] for k in GATE_KEYS})
    fail_nan = pd.DataFrame({k: fail[k] & out[f'nan_{k}'] for k in GATE_KEYS})
    out['n_gates_tested'] = out[[f'tested_{k}' for k in GATE_KEYS]].sum(axis=1)
    out['n_gates_failed'] = fail.sum(axis=1)
    out['failed_nan_only'] = (out['n_gates_failed'] > 0) & (fail_nan.sum(axis=1) == out['n_gates_failed'])
    out['failed_gates'] = [' + '.join(GATE_ABBR[k] + ('*' if fail_nan.at[i, k] else '')
                                      for k in GATE_KEYS if fail.at[i, k])
                           for i in fail.index]
    rej = ~out['final_accepted']
    out['rejected_all_gates']     = rej & (out['n_gates_failed'] == out['n_gates_tested'])
    out['rejected_atleast1_gate'] = rej & (out['n_gates_failed'] >= 1)
    return out


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, out_dir, name):
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'), dpi=300,
                    bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def _legend_outside(ax, handles):
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK_PRIMARY,
              loc='upper left', bbox_to_anchor=(1.0, 1.0))


def _nan_bar(ax, x, height, bottom, width, color):
    """Hatched bar segment for rejections caused by a NaN metric."""
    hatch_col = INK_MUTED if color == REJECT_COLOR else SURFACE
    ax.bar(x, height, width, bottom=bottom, color=color, edgecolor=hatch_col,
           linewidth=0, hatch=NAN_HATCH)


def _nan_patch(color=REJECT_COLOR, label=f'Rejected, {NAN_LABEL}'):
    return Patch(facecolor=color, edgecolor=INK_MUTED if color == REJECT_COLOR else SURFACE,
                 hatch=NAN_HATCH, label=label)


def _split_bar(ax, x, n_thr, n_nan, color, width, ymax):
    """Bar = threshold failures (solid) + NaN failures (hatched), total labelled on top."""
    ax.bar(x, n_thr, width, color=color, edgecolor=SURFACE, linewidth=1.5)
    if n_nan:
        _nan_bar(ax, x, n_nan, n_thr, width, color)
    lab = f'{n_thr + n_nan}' + (f'\n({n_nan}*)' if n_nan else '')
    ax.text(x, n_thr + n_nan + ymax * 0.01, lab, ha='center', va='bottom', fontsize=8,
            color=INK_SECONDARY, linespacing=1.15)


# ── Figures ───────────────────────────────────────────────────────────────────

def _plot_percent_per_gate(rows, out_dir):
    """Fig 1: stacked 100 % bars per gate (accepted = gate colour, rejected = gray,
    hatched gray = rejected because the metric is NaN). Each gate is evaluated on
    every unit it tests, independently of the others."""
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(9, 1.5 * len(rows)), 5.5), facecolor=SURFACE)
    _style_axes(ax)
    for xi, r in zip(x, rows):
        p_thr = r['pct_rejected'] - r['pct_rejected_nan']
        ax.bar(xi, r['pct_accepted'], 0.6, color=r['color'], edgecolor=SURFACE, linewidth=1.5)
        ax.bar(xi, p_thr, 0.6, bottom=r['pct_accepted'], color=REJECT_COLOR,
               edgecolor=SURFACE, linewidth=1.5)
        if r['pct_rejected_nan']:
            _nan_bar(ax, xi, r['pct_rejected_nan'], r['pct_accepted'] + p_thr, 0.6, REJECT_COLOR)
        txt = f'{r["pct_accepted"]:.1f} % acc.\n{r["pct_rejected"]:.1f} % rej.'
        if r['pct_rejected_nan']:
            txt += f'\n({r["pct_rejected_nan"]:.1f} % NaN)'
        ax.text(xi, 101, txt, ha='center', va='bottom', fontsize=7.5, color=INK_PRIMARY,
                linespacing=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r["short"]}\n(n = {r["n_tested"]})' for r in rows],
                       fontsize=9, color=INK_PRIMARY)
    ax.set_ylabel('% of units tested by the gate', color=INK_PRIMARY)
    ax.set_ylim(0, 120)
    ax.set_yticks(range(0, 101, 20))
    ax.set_title('Accepted vs rejected units at each quality gate (each gate on its own)',
                 color=INK_PRIMARY, fontsize=11, loc='left')
    handles = [Patch(facecolor=GATE_COLOR[k], label=f'{GATE_NAME[k]} (accepted)') for k in GATE_KEYS]
    handles += [Patch(facecolor=COMMON_COLOR, label='All automatic gates / final (accepted)'),
                Patch(facecolor=REJECT_COLOR, label='Rejected, threshold'),
                _nan_patch()]
    _legend_outside(ax, handles)
    _save(fig, out_dir, 'Fig1_Percent_Accepted_Rejected_PerGate')


def _plot_sequential(seq_rows, n_final, n_rescued, out_dir):
    """Fig 2: gates applied in order. Bar height = units entering the gate, split
    into passed (gate colour) and rejected at this gate (gray; hatched = NaN)."""
    x = np.arange(len(seq_rows) + 1)
    ymax = max(r['n_in'] for r in seq_rows) or 1
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    _style_axes(ax)
    for xi, r in zip(x, seq_rows):
        n_thr = r['n_rej'] - r['n_rej_nan']
        ax.bar(xi, r['n_pass'], 0.6, color=GATE_COLOR[r['key']], edgecolor=SURFACE, linewidth=1.5)
        ax.bar(xi, n_thr, 0.6, bottom=r['n_pass'], color=REJECT_COLOR, edgecolor=SURFACE, linewidth=1.5)
        if r['n_rej_nan']:
            _nan_bar(ax, xi, r['n_rej_nan'], r['n_pass'] + n_thr, 0.6, REJECT_COLOR)
        txt = f'in {r["n_in"]}\npass {r["n_pass"]} ({r["pct_pass"]:.1f} %)\nrej. {r["n_rej"]}'
        if r['n_rej_nan']:
            txt += f' ({r["n_rej_nan"]} NaN)'
        ax.text(xi, r['n_in'] + ymax * 0.01, txt, ha='center', va='bottom', fontsize=7.5,
                color=INK_PRIMARY, linespacing=1.2)
    xf = x[-1]
    ax.bar(xf, n_final - n_rescued, 0.6, color=COMMON_COLOR, edgecolor=SURFACE, linewidth=1.5)
    if n_rescued:
        ax.bar(xf, n_rescued, 0.6, bottom=n_final - n_rescued, color=COMMON_COLOR, alpha=0.55,
               edgecolor=SURFACE, linewidth=1.5, hatch='\\\\\\')
    ax.text(xf, n_final + ymax * 0.01, f'{n_final}\n(incl. {n_rescued} rescued\nby manual review)',
            ha='center', va='bottom', fontsize=7.5, color=INK_PRIMARY, linespacing=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([GATE_SHORT[r['key']] for r in seq_rows] + ['Final\naccepted'],
                       fontsize=9, color=INK_PRIMARY)
    ax.set_ylabel('Number of units', color=INK_PRIMARY)
    ax.set_ylim(0, ymax * 1.24)
    ax.set_title('Units passing each gate when the gates are applied in order\n'
                 '(manual review: auto-accepted multi-trigger units that were reviewed)',
                 color=INK_PRIMARY, fontsize=11, loc='left')
    handles = [Patch(facecolor=GATE_COLOR[r['key']], label=f'{GATE_NAME[r["key"]]} (passed)')
               for r in seq_rows]
    handles += [Patch(facecolor=REJECT_COLOR, label='Rejected at this gate, threshold'),
                _nan_patch(label=f'Rejected at this gate, {NAN_LABEL}'),
                Patch(facecolor=COMMON_COLOR, label='Final accepted'),
                Patch(facecolor=COMMON_COLOR, alpha=0.55, edgecolor=SURFACE, hatch='\\\\\\',
                      label='Failed an automatic gate,\nmanually accepted')]
    _legend_outside(ax, handles)
    _save(fig, out_dir, 'Fig2_Sequential_Gates')


def _plot_rejected(n_rej_total, per_gate, per_gate_nan, n_atleast1, n_atleast1_nan,
                   n_all, n_all_nan, combos, out_dir):
    """Fig 3: final-rejected units. Top: per gate, >= 1 gate, all gates (common).
    Bottom: every exact combination of failed gates. Hatched = NaN failures."""
    n_c = max(len(combos), 1)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(max(10, 1.15 * n_c + 1), 10),
                                   gridspec_kw={'hspace': 0.45}, facecolor=SURFACE)
    for ax in (ax0, ax1):
        _style_axes(ax)

    keys   = GATE_KEYS + ['atleast1', 'all']
    labels = [GATE_SHORT[k] for k in GATE_KEYS] + ['Rejected by\n≥ 1 gate', 'Rejected by\nall gates']
    n_tot  = [per_gate[k] for k in GATE_KEYS] + [n_atleast1, n_all]
    n_nan  = [per_gate_nan[k] for k in GATE_KEYS] + [n_atleast1_nan, n_all_nan]
    colors = [GATE_COLOR[k] for k in GATE_KEYS] + [ATLEAST1_COLOR, ALLGATE_COLOR]
    ymax = max(n_tot + [1])
    for xi, (t, nn, c) in enumerate(zip(n_tot, n_nan, colors)):
        _split_bar(ax0, xi, t - nn, nn, c, 0.6, ymax)
    ax0.set_xticks(np.arange(len(keys)))
    ax0.set_xticklabels(labels, fontsize=9, color=INK_PRIMARY)
    ax0.set_ylim(0, ymax * 1.18)
    ax0.set_ylabel('Number of final-rejected units', color=INK_PRIMARY)
    ax0.set_title(f'Final-rejected units (n = {n_rej_total}) failing each gate',
                  color=INK_PRIMARY, fontsize=11, loc='left')
    handles = [Patch(facecolor=GATE_COLOR[k], label=GATE_NAME[k]) for k in GATE_KEYS]
    handles += [Patch(facecolor=ATLEAST1_COLOR, label='Rejected by ≥ 1 gate'),
                Patch(facecolor=ALLGATE_COLOR, label='Rejected by all gates tested'),
                _nan_patch(INK_MUTED, label=f'Hatched / * = {NAN_LABEL}')]
    _legend_outside(ax0, handles)

    if combos:
        c_lab = [c['failed_gates'].replace(' + ', '\n+ ') for c in combos]
        c_val = [c['n_units'] for c in combos]
        ymax1 = max(c_val)
        for xi, c in enumerate(combos):
            col = ALLGATE_COLOR if c['all_gates'] else ATLEAST1_COLOR
            n_nan = c['n_units'] if c['nan_only'] else 0
            _split_bar(ax1, xi, c['n_units'] - n_nan, n_nan, col, 0.6, ymax1)
        ax1.set_xticks(np.arange(len(combos)))
        ax1.set_xticklabels(c_lab, fontsize=8, color=INK_PRIMARY)
        ax1.set_ylim(0, ymax1 * 1.18)
    ax1.set_ylabel('Number of final-rejected units', color=INK_PRIMARY)
    ax1.set_title('Exact combination of failed gates  (* = metric NaN, unit has < 2 spikes)',
                  color=INK_PRIMARY, fontsize=11, loc='left')
    _legend_outside(ax1, [Patch(facecolor=ATLEAST1_COLOR, label='Rejected by ≥ 1 gate (not all)'),
                          Patch(facecolor=ALLGATE_COLOR, label='Rejected by all gates tested'),
                          _nan_patch(INK_MUTED, label=f'All failures: {NAN_LABEL}')])
    _save(fig, out_dir, 'Fig3_Rejected_Common_vs_AtLeastOne')


def _plot_rejected_metrics(df, flags, out_dir, n_bins=30):
    """Fig 4: metric distributions of final-rejected units, stacked:
    rejected by all gates (bottom) + rejected by some but not all gates (top).
    Units whose metric is NaN cannot be placed on the axis and are counted in
    each panel's corner note."""
    rej   = ~flags['final_accepted'].values
    all_g = flags['rejected_all_gates'].values[rej]
    thr   = {'isi': ISI_ACCEPT_MAX_PCT, 'snr': SNR_ACCEPT_MIN, 'tc': TMPL_CORR_ACCEPT_MIN}
    xlab  = {'isi': 'ISI violations  (%)', 'snr': 'SNR  (A_peak / σ_noise)',
             'tc': 'Mean Pearson r  (spike vs template)'}
    fig, axes = plt.subplots(1, len(AUTO_KEYS), figsize=(5.0 * len(AUTO_KEYS), 4.2), facecolor=SURFACE)
    for ax, k in zip(axes, AUTO_KEYS):
        _style_axes(ax)
        vals  = df[GATE_COL[k]].values[rej].astype(float)
        valid = ~np.isnan(vals)
        v, a  = vals[valid], all_g[valid]
        if len(v) == 0:
            ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, ha='center',
                    va='center', fontsize=10, color=INK_MUTED)
        else:
            edges   = np.histogram_bin_edges(v, bins=n_bins)
            widths  = np.diff(edges)
            centers = edges[:-1] + widths / 2
            n_all, _  = np.histogram(v[a],  bins=edges)
            n_some, _ = np.histogram(v[~a], bins=edges)
            ax.bar(centers, n_all, width=widths, color=ALLGATE_COLOR, edgecolor=SURFACE, linewidth=1.0)
            ax.bar(centers, n_some, width=widths, bottom=n_all, color=ATLEAST1_COLOR,
                   edgecolor=SURFACE, linewidth=1.0)
            ax.axvline(thr[k], color=GATE_COLOR[k], linewidth=1.5, linestyle='--', zorder=4)
            ax.set_ylim(0, max(int((n_all + n_some).max()), 1) * 1.45)   # room for the corner note
        n_nan_all  = int((~valid & all_g).sum())
        n_nan_some = int((~valid & ~all_g).sum())
        ax.text(0.98, 0.98, f'threshold {thr[k]:g}\nn plotted = {len(v)}\n'
                            f'NaN, not plotted = {n_nan_all + n_nan_some}\n'
                            f'  (all gates {n_nan_all}, ≥ 1 gate {n_nan_some})',
                transform=ax.transAxes, ha='right', va='top', fontsize=8, color=INK_SECONDARY,
                linespacing=1.25)
        ax.set_title(GATE_NAME[k], color=INK_PRIMARY, fontsize=11, loc='left')
        ax.set_xlabel(xlab[k], color=INK_PRIMARY)
    axes[0].set_ylabel('Number of final-rejected units', color=INK_PRIMARY)
    handles = [Patch(facecolor=ALLGATE_COLOR, label=f'Rejected by all gates (n = {int(all_g.sum())})'),
               Patch(facecolor=ATLEAST1_COLOR,
                     label=f'Rejected by ≥ 1 but not all gates (n = {int((~all_g).sum())})'),
               plt.Line2D([], [], color=INK_MUTED, lw=1.5, ls='--', label='Gate threshold')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=9,
               labelcolor=INK_PRIMARY)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, out_dir, 'Fig4_Rejected_Metric_Distributions')


# ── Main ──────────────────────────────────────────────────────────────────────

def main(input_excel=INPUT_EXCEL, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    df_all = _load(input_excel)
    empty  = pd.to_numeric(df_all['n_spikes'], errors='coerce') == 0
    df_empty = df_all.loc[empty, ['session', 'unit', 'n_spikes', 'accepted']]
    df    = df_all.loc[~empty].reset_index(drop=True)
    flags = _gate_flags(df)
    n_total = len(df)
    print(f'{len(df_all)} units in {os.path.basename(input_excel)} [{SHEET}]; '
          f'{len(df_empty)} empty .ntt files (0 spikes) skipped -> {n_total} units analysed')

    mismatch = flags['derived_accepted'] != flags['final_accepted']
    if mismatch.any():
        print(f'  WARNING: {int(mismatch.sum())} unit(s) where the gates do not reproduce the '
              f"'accepted' column (listed in sheet 'Accepted_Mismatch'); 'accepted' is used.")

    all_true = pd.Series(True, index=df.index)
    auto_nan_only = (~flags['pass_auto']
                     & ~pd.concat([~flags[f'pass_{k}'] & ~flags[f'nan_{k}'] for k in AUTO_KEYS],
                                  axis=1).any(axis=1))

    # ── Per gate, independent ────────────────────────────────────────────────

    gate_rows = []
    for k in GATE_KEYS + ['auto', 'final']:
        if k == 'auto':
            tested, passed, rej_nan = all_true, flags['pass_auto'], auto_nan_only
            name, short, color = 'All automatic gates (ISI & SNR & TmplCorr)', 'All automatic\ngates', COMMON_COLOR
        elif k == 'final':
            tested, passed = all_true, flags['final_accepted']
            rej_nan = ~passed & flags['failed_nan_only']
            name, short, color = 'Final decision (accepted column)', 'Final\ndecision', COMMON_COLOR
        else:
            tested, passed = flags[f'tested_{k}'], flags[f'pass_{k}']
            rej_nan = tested & ~passed & flags[f'nan_{k}']
            name, short, color = GATE_NAME[k], GATE_SHORT[k], GATE_COLOR[k]
        n_t = int(tested.sum())
        n_a = int((tested & passed).sum())
        n_n = int(rej_nan.sum())
        pct = (lambda n: 100.0 * n / n_t if n_t else 0.0)
        gate_rows.append({'gate': name, 'short': short, 'color': color, 'n_tested': n_t,
                          'n_accepted': n_a, 'n_rejected': n_t - n_a,
                          'n_rejected_threshold': n_t - n_a - n_n, 'n_rejected_nan': n_n,
                          'pct_accepted': pct(n_a), 'pct_rejected': pct(n_t - n_a),
                          'pct_rejected_nan': pct(n_n)})
    df_gate = pd.DataFrame(gate_rows).drop(columns=['short', 'color'])
    print('\nPer gate (each gate on the units it tests):')
    print(df_gate.to_string(index=False, float_format='%.2f'))
    _plot_percent_per_gate(gate_rows, output_dir)

    # ── Sequential (ISI -> SNR -> TmplCorr -> manual review) ─────────────────

    seq_rows, alive = [], all_true.copy()
    for k in AUTO_KEYS:
        n_in    = int(alive.sum())
        n_nan   = int((alive & ~flags[f'pass_{k}'] & flags[f'nan_{k}']).sum())
        alive   = alive & flags[f'pass_{k}']
        seq_rows.append({'key': k, 'gate': GATE_NAME[k], 'n_in': n_in,
                         'n_pass': int(alive.sum()), 'n_rej_nan': n_nan})
    n_in = int(alive.sum())
    n_manual_rej = int((alive & flags['tested_manual'] & ~flags['pass_manual']).sum())
    seq_rows.append({'key': 'manual', 'gate': GATE_NAME['manual'], 'n_in': n_in,
                     'n_pass': n_in - n_manual_rej, 'n_rej_nan': 0})
    for r in seq_rows:
        r['n_rej']    = r['n_in'] - r['n_pass']
        r['pct_pass'] = 100.0 * r['n_pass'] / r['n_in'] if r['n_in'] else 0.0
    n_final   = int(flags['final_accepted'].sum())
    n_rescued = int((flags['final_accepted'] & ~flags['pass_auto']).sum())
    df_seq = pd.DataFrame(seq_rows).drop(columns=['key'])
    df_seq = pd.concat([df_seq, pd.DataFrame([{
        'gate': 'Final accepted', 'n_pass': n_final,
        'pct_pass': 100.0 * n_final / n_total if n_total else 0.0}])], ignore_index=True)
    df_seq['rescued_by_manual_review'] = [np.nan] * len(seq_rows) + [n_rescued]
    print('\nSequential gates:')
    print(df_seq.to_string(index=False, float_format='%.2f'))
    _plot_sequential(seq_rows, n_final, n_rescued, output_dir)

    # ── Final-rejected units: common to all gates vs >= 1 gate ───────────────

    rej = ~flags['final_accepted']
    n_rej_total  = int(rej.sum())
    per_gate     = {k: int((rej & flags[f'tested_{k}'] & ~flags[f'pass_{k}']).sum()) for k in GATE_KEYS}
    per_gate_nan = {k: int((rej & ~flags[f'pass_{k}'] & flags[f'nan_{k}']).sum()) for k in GATE_KEYS}
    n_atleast1     = int(flags['rejected_atleast1_gate'].sum())
    n_atleast1_nan = int((flags['rejected_atleast1_gate'] & flags['failed_nan_only']).sum())
    n_all          = int(flags['rejected_all_gates'].sum())
    n_all_nan      = int((flags['rejected_all_gates'] & flags['failed_nan_only']).sum())
    n_none = n_rej_total - n_atleast1
    if n_none:
        print(f'  NOTE: {n_none} final-rejected unit(s) fail none of the gates '
              f'(see Accepted_Mismatch).')

    combos = (flags.loc[rej & (flags['n_gates_failed'] > 0)]
              .groupby('failed_gates')
              .agg(n_units=('failed_gates', 'size'), all_gates=('rejected_all_gates', 'all'),
                   nan_only=('failed_nan_only', 'all'), n_gates_failed=('n_gates_failed', 'first'))
              .reset_index()
              .sort_values(['n_gates_failed', 'n_units'], ascending=[True, False]))
    print(f'\nFinal-rejected units: {n_rej_total}')
    print(f'  rejected by >= 1 gate : {n_atleast1}  ({n_atleast1_nan} only because of NaN metrics)')
    print(f'  rejected by all gates : {n_all}  ({n_all_nan} only because of NaN metrics)')
    print('  per gate              : ' + ', '.join(f'{GATE_ABBR[k]} {per_gate[k]} ({per_gate_nan[k]} NaN)'
                                                   for k in GATE_KEYS))
    print(combos.to_string(index=False))
    _plot_rejected(n_rej_total, per_gate, per_gate_nan, n_atleast1, n_atleast1_nan,
                   n_all, n_all_nan, combos.to_dict('records'), output_dir)
    _plot_rejected_metrics(df, flags, output_dir)

    # ── Excel ────────────────────────────────────────────────────────────────

    unit_cols = ['session', 'unit', 'n_spikes', 'isi_violation_pct', 'snr', 'template_correlation',
                 'peak_amp_uV', 'multi_trigger', 'manual_review', 'accepted']
    flag_cols = ([f'pass_{k}' for k in GATE_KEYS] + [f'nan_{k}' for k in AUTO_KEYS]
                 + ['tested_manual', 'pass_auto', 'derived_accepted',
                    'n_gates_tested', 'n_gates_failed', 'failed_gates', 'failed_nan_only',
                    'rejected_all_gates', 'rejected_atleast1_gate'])
    per_unit = pd.concat([df[[c for c in unit_cols if c in df.columns]], flags[flag_cols]], axis=1)
    per_unit['pass_manual'] = per_unit['pass_manual'].astype(object)
    per_unit.loc[~per_unit['tested_manual'], 'pass_manual'] = np.nan   # not reviewed -> blank
    df_rej_counts = pd.DataFrame(
        [{'category': f'Failed {GATE_NAME[k]}', 'n_units': per_gate[k], 'n_units_nan': per_gate_nan[k]}
         for k in GATE_KEYS]
        + [{'category': 'Rejected by >= 1 gate', 'n_units': n_atleast1, 'n_units_nan': n_atleast1_nan},
           {'category': 'Rejected by all gates tested', 'n_units': n_all, 'n_units_nan': n_all_nan},
           {'category': 'Final rejected (total)', 'n_units': n_rej_total, 'n_units_nan': np.nan}])

    summary_xlsx = os.path.join(output_dir, 'QualityGates_Summary.xlsx')
    with pd.ExcelWriter(summary_xlsx, engine='openpyxl') as writer:
        df_gate.to_excel(writer, sheet_name='Per_Gate', index=False)
        df_seq.to_excel(writer, sheet_name='Sequential', index=False)
        df_rej_counts.to_excel(writer, sheet_name='Rejected_Counts', index=False)
        combos.to_excel(writer, sheet_name='Rejected_Combinations', index=False)
        per_unit[per_unit['rejected_all_gates']].to_excel(
            writer, sheet_name='Rejected_All_Gates', index=False)
        per_unit[per_unit['rejected_atleast1_gate']].to_excel(
            writer, sheet_name='Rejected_AtLeast1_Gate', index=False)
        per_unit[mismatch.values].to_excel(writer, sheet_name='Accepted_Mismatch', index=False)
        per_unit.to_excel(writer, sheet_name='Per_Unit', index=False)
        df_empty.to_excel(writer, sheet_name='Skipped_Empty_Ntt', index=False)

    print(f'\nDone. Figures and tables saved to {output_dir}')


if __name__ == '__main__':
    main()
