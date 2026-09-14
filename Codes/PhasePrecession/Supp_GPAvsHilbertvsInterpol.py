# -*- coding: utf-8 -*-
"""
Thesis-justification analysis for the Generalized Phase (GP) correction:
recursively scans a root data folder for every LFP (.ncs) file, and for
each one computes what fraction of its per-sample instantaneous theta
frequency falls outside the theta passband under three estimation methods,
all applied to the SAME bandpass-filtered trace so the comparison isolates
the estimation algorithm rather than upstream signal conditioning:

  PRE-GP  : raw Hilbert-transform instantaneous frequency, no phase-slip
            correction (generalized_phase_vector's own wt_pre).
  POST-GP : Hilbert + Generalized-Phase correction (gp_instantaneous_
            frequency). Reconstructed-adjacent samples are excluded
            (EXCLUDED below); of the remaining samples, any whose estimate
            still lands outside the passband is flagged out-of-band by
            that function's own mask -- reported here BEFORE that
            function's pchip repair of those samples, so this is the
            "would be unreliable" rate the repair is fixing, not the
            near-zero rate you'd get by re-checking the repaired values.
  INTERP  : ThetaVsSpeed.py's peak-to-peak / trough-to-trough interval
            interpolation method (estimate_instantaneous_frequency), which
            never excludes samples (np.interp fills every sample), so it
            has no EXCLUDED category.

Every file's raw samples are hashed after loading; an exact duplicate of
an already-processed file (e.g. the same physical LFP saved under several
per-tetrode filenames) is logged and skipped so it can't be double-counted
in the thesis-wide average.

Outputs (written to OUTPUT_DIR):
  gpa_out_of_band_per_file.csv   -- one row per unique file, all counts/%
  gpa_out_of_band_per_file.png   -- per-file in-band/out-of-band/excluded
                                     breakdown, one cluster of 3 stacked
                                     bars (PRE-GP, POST-GP, INTERP) per file
  gpa_out_of_band_summary.png    -- across-files mean +/- SD out-of-band %
                                     for the three methods (the headline
                                     "GPA reduces unreliable estimates"
                                     figure)
"""
from __future__ import annotations

import hashlib
import sys
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# Configuration -- EDIT THESE
# ============================================================================

ROOT_DIR = Path(r"C:/Runita/NMR/analysis/AllSort_Results/PlaceCell/Data/Debug")
OUTPUT_DIR = ROOT_DIR / "Output_GPA_OutOfBand_Justification"

PHASEPRECESSION_DIR = Path(r"C:/Runita/NMR/Mole-Rat_PlaceCell/Codes/PhasePrecession/Codes")
THETA_DIR = Path(r"C:/Runita/NMR/Mole-Rat_PlaceCell/Codes/Theta/Codes")

sys.path.insert(0, str(PHASEPRECESSION_DIR))
sys.path.insert(0, str(THETA_DIR))

import Debug_GPA_PhaseValues_ThetaMod_PhasePrec_SpikeLFP_Matched_v10 as gpav10  # noqa: E402
import ThetaVsSpeed as tvs  # noqa: E402

LFP_FILTER_BAND = gpav10.LFP_FILTER_BAND  # (3.0, 7.0) Hz


# ============================================================================
# Per-file analysis
# ============================================================================

def _breakdown(n_total, finite_mask, out_of_band_mask):
    """Sample counts split into in-band / out-of-band / excluded (NaN),
    all as fractions of n_total so every method's bars sum to 100%."""
    n_finite = int(finite_mask.sum())
    n_oob = int(out_of_band_mask.sum())
    n_inband = n_finite - n_oob
    n_excluded = n_total - n_finite
    return dict(
        n_inband=n_inband, n_oob=n_oob, n_excluded=n_excluded,
        pct_inband=100.0 * n_inband / n_total,
        pct_oob=100.0 * n_oob / n_total,
        pct_excluded=100.0 * n_excluded / n_total,
        pct_oob_of_finite=(100.0 * n_oob / n_finite) if n_finite else float('nan'),
    )


def analyze_file(ncs_path: Path) -> dict:
    lfp_sig, lfp_ts, lfp_fs = gpav10.load_ncs(ncs_path)
    file_hash = hashlib.blake2b(lfp_sig.tobytes()).hexdigest()

    filtered = gpav10.bandpass_filter(lfp_sig, LFP_FILTER_BAND[0], LFP_FILTER_BAND[1], lfp_fs)
    n_total = len(filtered)

    # --- PRE-GP: raw Hilbert ---
    xgp, wt_pre, idx = gpav10.generalized_phase_vector(filtered, lfp_fs, LFP_FILTER_BAND[0])
    finite_pre = np.isfinite(wt_pre)
    with np.errstate(invalid='ignore'):
        oob_pre = finite_pre & ((wt_pre < LFP_FILTER_BAND[0]) | (wt_pre > LFP_FILTER_BAND[1]))
    pre = _breakdown(n_total, finite_pre, oob_pre)

    # --- POST-GP: Hilbert + Generalized-Phase correction ---
    wt_post, wt_post_interpolated = gpav10.gp_instantaneous_frequency(
        xgp, idx, lfp_fs, LFP_FILTER_BAND[0], LFP_FILTER_BAND[1])
    finite_post = np.isfinite(wt_post) | wt_post_interpolated  # repaired samples count as "evaluated"
    post = _breakdown(n_total, finite_post, wt_post_interpolated)

    # --- INTERP: ThetaVsSpeed.py peak/trough interpolation, same trace ---
    peak_ts, trough_ts = tvs.find_peaks_troughs(
        filtered, lfp_ts, tvs.PEAK_THRESHOLD, max_freq=LFP_FILTER_BAND[1])
    peak_ts_c, trough_ts_c = tvs.find_missing_peaks(peak_ts, trough_ts, filtered, lfp_ts)
    inst_freq_interp = tvs.estimate_instantaneous_frequency(peak_ts_c, trough_ts_c, lfp_ts)
    finite_interp = np.isfinite(inst_freq_interp)
    with np.errstate(invalid='ignore'):
        oob_interp = finite_interp & ((inst_freq_interp < LFP_FILTER_BAND[0]) |
                                       (inst_freq_interp > LFP_FILTER_BAND[1]))
    interp = _breakdown(n_total, finite_interp, oob_interp)

    return dict(
        path=ncs_path, file_hash=file_hash, n_total=n_total, fs=lfp_fs,
        pre=pre, post=post, interp=interp,
    )


# ============================================================================
# Driver
# ============================================================================

def find_lfp_files(root: Path):
    return sorted(root.rglob('*.ncs'))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ncs_files = find_lfp_files(ROOT_DIR)
    print(f'Found {len(ncs_files)} .ncs files under {ROOT_DIR}')

    seen_hashes: dict[str, Path] = {}
    results = []
    duplicates = []

    for i, ncs_path in enumerate(ncs_files, 1):
        rel = ncs_path.relative_to(ROOT_DIR)
        print(f'[{i}/{len(ncs_files)}] {rel} ...', flush=True)
        try:
            r = analyze_file(ncs_path)
        except Exception as exc:
            print(f'    SKIP (error): {exc}')
            traceback.print_exc()
            continue

        if r['file_hash'] in seen_hashes:
            print(f'    DUPLICATE of {seen_hashes[r["file_hash"]]} -- excluded from averages')
            duplicates.append((ncs_path, seen_hashes[r['file_hash']]))
            continue
        seen_hashes[r['file_hash']] = rel
        r['label'] = str(rel).replace('\\', '/')
        results.append(r)

    if not results:
        print('No unique LFP files were successfully processed -- nothing to plot.')
        return

    write_csv(results, OUTPUT_DIR / 'gpa_out_of_band_per_file.csv')
    plot_per_file(results, OUTPUT_DIR / 'gpa_out_of_band_per_file.png')
    plot_summary(results, OUTPUT_DIR / 'gpa_out_of_band_summary.png')

    print()
    print(f'{len(results)} unique files analyzed, {len(duplicates)} duplicate files excluded.')
    for method in ('pre', 'post', 'interp'):
        vals = np.array([r[method]['pct_oob'] for r in results])
        print(f'{method.upper():8s}: mean out-of-band = {vals.mean():.2f}% '
              f'(SD {vals.std():.2f}%) of all samples, across {len(vals)} files')
    print()
    print(f'Results written to {OUTPUT_DIR}')


def write_csv(results, out_path: Path):
    import csv
    fieldnames = ['file', 'n_total', 'fs']
    for method in ('pre', 'post', 'interp'):
        for key in ('pct_inband', 'pct_oob', 'pct_excluded', 'pct_oob_of_finite',
                    'n_inband', 'n_oob', 'n_excluded'):
            fieldnames.append(f'{method}_{key}')

    with open(out_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {'file': r['label'], 'n_total': r['n_total'], 'fs': r['fs']}
            for method in ('pre', 'post', 'interp'):
                for key, val in r[method].items():
                    row[f'{method}_{key}'] = val
            writer.writerow(row)
    print(f'Wrote {out_path}')


def plot_per_file(results, out_path: Path):
    methods = [('pre', 'PRE-GP'), ('post', 'POST-GP'), ('interp', 'INTERP')]
    colors = {'inband': '#55A868', 'oob': '#C44E52', 'excluded': '#B0B0B0'}

    n_files = len(results)
    n_methods = len(methods)
    bar_w = 0.8 / n_methods
    x = np.arange(n_files)

    fig, ax = plt.subplots(figsize=(max(8, n_files * 0.9), 6))
    for m_idx, (key, mlabel) in enumerate(methods):
        offset = (m_idx - (n_methods - 1) / 2) * bar_w
        inband = np.array([r[key]['pct_inband'] for r in results])
        oob = np.array([r[key]['pct_oob'] for r in results])
        excluded = np.array([r[key]['pct_excluded'] for r in results])

        xpos = x + offset
        ax.bar(xpos, inband, width=bar_w, color=colors['inband'],
               label='in-band' if m_idx == 0 else None)
        ax.bar(xpos, oob, width=bar_w, bottom=inband, color=colors['oob'],
               label='out-of-band' if m_idx == 0 else None)
        ax.bar(xpos, excluded, width=bar_w, bottom=inband + oob, color=colors['excluded'],
               label='excluded (no estimate)' if m_idx == 0 else None)

        for xp in xpos:
            ax.text(xp, -2, mlabel, ha='center', va='top', fontsize=6, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([r['label'] for r in results], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('% of samples')
    ax.set_ylim(-14, 101)
    ax.set_title(f'Instantaneous theta-frequency reliability per LFP file '
                 f'(passband {LFP_FILTER_BAND[0]:g}-{LFP_FILTER_BAND[1]:g} Hz)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')


def plot_summary(results, out_path: Path):
    methods = [('pre', 'PRE-GP\n(raw Hilbert)'), ('post', 'POST-GP\n(Hilbert + GP)'),
               ('interp', 'INTERP\n(peak/trough)')]

    means, sds = [], []
    for key, _ in methods:
        vals = np.array([r[key]['pct_oob'] for r in results])
        means.append(vals.mean())
        sds.append(vals.std())

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=sds, capsize=6,
                   color=['#8C8C8C', '#4C72B0', '#DD8452'])
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in methods])
    ax.set_ylabel('% of samples out-of-band (mean \u00b1 SD across files)')
    ax.set_title(f'Out-of-band instantaneous theta frequency\n'
                 f'({len(results)} unique LFP files, passband '
                 f'{LFP_FILTER_BAND[0]:g}-{LFP_FILTER_BAND[1]:g} Hz)')
    for xi, m in zip(x, means):
        ax.text(xi, m, f'{m:.1f}%', ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
