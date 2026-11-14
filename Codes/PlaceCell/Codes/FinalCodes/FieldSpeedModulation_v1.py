# -*- coding: utf-8 -*-
"""
Speed modulation of the firing INSIDE each place field, one test per field,
and the proportion of place fields that are speed-modulated compared across
the conditions of ArenaSessionType_StatsComparison_v20.py.

Per field (the analysis of PlaceCell_Main_v21._compute_speed_modulation,
restricted to the time the animal spends inside the field):
  1. Each place cell's session is rebuilt with PlaceCell_Main_v21.compute_metrics
     (same tracking cleaning / smoothing, 50 ms spike gate). Speed modulation
     uses the contiguous, pre-immobility-filter track and spikes (ctx['*_all']),
     downsampled by pc.SPEED_MOD_DOWNSAMPLE_FACTOR, exactly as PlaceCell_Main_v21.
  2. Over the WHOLE session, as in PlaceCell_Main_v21: per-frame running speed and
     instantaneous firing rate (all the cell's spikes), each Gaussian-smoothed
     (pc.SPEED_SMOOTH_S).
  3. In-field mask: a frame is in the field if its position falls in one of the
     field's bins (bin_coords of FIELDS_EXCEL 'PlaceFields'). 2-D arenas: the
     compute_metrics bin grid (pc.target_bin_cm). Circle track: the angular bins of
     pc._build_ratemap_circular (ring centre / bin width fitted on the moving
     frames, as in PlaceCell_Main_v21, applied to every frame).
  4. Analysis samples = frames with pc.SPEED_MIN_CMS < smoothed speed <
     pc.SPEED_MAX_CMS, a finite smoothed rate, AND inside the field. On these
     samples, unchanged from PlaceCell_Main_v21:
       - time-domain score: Pearson r of smoothed speed vs smoothed rate
       - binned score: median rate per pc.SPEED_BIN_CMS speed bin, bins with
         < pc.SPEED_MIN_BIN_FRAC of the field's samples dropped, weighted
         (n samples per bin) linear regression; r is the score
       - initial gate: each method's shuffle runs only if its parametric
         p < 0.05
       - circular-shift shuffle (pc.SPEED_N_SHUFFLE, >= pc.SPEED_SHUFFLE_MARGIN_S
         from either end) of the WHOLE-SESSION spike-count trace, re-smoothed,
         then the same in-field mask; two-tailed 2.5 / 97.5 % CI and p
         (pc._shuffle_ci_and_pvalue)
       - final_speed_score = mean of the binned and time-domain scores;
         speed_cell: False if the two scores have opposite signs or either
         shuffle ran and failed; True if both shuffles ran and passed;
         'not tested' otherwise.
     The smoothed rate at in-field samples near the field edge contains
     smoothing bleed (sigma pc.SPEED_SMOOTH_S) from spikes just outside the field.
     All fields of one cell share the same shuffle offsets (one shuffle loop per
     cell); each field's test on its own is unchanged.
  5. Minimum data (fields below either minimum are not tested and are left out
     of every statistic):
       MIN_INFIELD_SPIKES     moving in-field spikes (speed >= pc.IMMOBILITY_SPEED_CMS,
                              the spikes behind the ratemaps; = field_n_spikes of the
                              workbook when the rebuilt grid matches)
       MIN_INFIELD_DWELL_PCT  in-field time at pc.SPEED_MIN_CMS-pc.SPEED_MAX_CMS / whole-
                              session time at that speed, in % (frame durations capped at
                              2 / fps, as the occupancy maps)
  Figure per field (2 x 2, as PlaceCell_Main_v21's speed figure; in-field
  samples only, the time series shows the whole session with the in-field samples
  highlighted): <session>\\FIELD_SPEED_SUBDIR\\<unit>_F<n>_field_speed_modulation.png
  under PLACE_CELL_FOLDER.

Speed class of a field (as v20's add_speed_class): p-type if speed_cell is True
and final_speed_score > 0, n-type if < 0, non-modulated otherwise (speed_cell
False or 'not tested'); fields with no final_speed_score have no class.

Statistics (functions of ArenaSessionType_StatsComparison_v20, imported as asc),
one folder per level in OUTPUT_DIR:
  SessionType\\<arena>\\       level 1, session types within each arena
  ArenaType\\                  level 2, Circle / Linear / Open
  AnimalControlledArenaType\\  level 3, asc.ANIMAL_CONTROLLED_COMPARISONS side by side
  AnimalID\\                   level 4, animals
  AllFields\\                  level 5, every field pooled, descriptives only
Each folder:
  Categorical_field_speed_modulated.png   % of place fields speed-modulated;
                                          chi-square + Holm pairwise chi-square
  Categorical_field_speed_direction.png   p-type / n-type / non-modulated fields;
                                          3-category chi-square + pairwise
  FieldSpeed_<score>.png                  per-field speed scores (final, binned,
                                          time-domain); asc.CONTINUOUS_TEST
                                          (Kruskal-Wallis + Mann-Whitney U, Holm)
  Histograms\\Hist_FieldSpeed_*.png       the same scores as histograms
  FieldVsCell_speed_class.png             per group: each field's class against its
                                          cell's whole-session class (workbook
                                          'Full': speed_cell + final_speed_score);
                                          chi-square of independence per group
Fields of one cell are treated as independent rows (as v20 treats its
per-field multi-peak metrics).

Outputs:
  OUTPUT_DIR\\FieldSpeedModulation.xlsx   PerField (every field), Settings
  OUTPUT_DIR\\FieldSpeed_Stats.xlsx       sheets 'Ses<arena>_*', 'Arena_*',
                                         'AC<A><B>_*', 'Animal_*', 'All_*'
  OUTPUT_DIR\\FieldSpeed_AllStats.csv
  OUTPUT_DIR\\RunSnapshots\\<timestamp>\\  code + RunMetadata.xlsx
"""

import os
import sys
import shutil
import platform
import tempfile
import threading
import importlib.metadata
import concurrent.futures
from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# CellClusteredStats_Utils.py lives one folder up (appended, so this folder's
# PlaceCell_Main_v21.py and ArenaSessionType_StatsComparison_v20.py come first).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import PlaceCell_Main_v21 as pc
import ArenaSessionType_StatsComparison_v20 as asc   # statistics + plot style
import CellClusteredStats_Utils as ccs
import PlaceFieldShape_Utils as pfs

# ── Parameters ──────────────────────────────────────────────────────────────

# Minimum data per field (fields below either are not tested; see docstring).
MIN_INFIELD_SPIKES    = 10      # moving in-field spikes
MIN_INFIELD_DWELL_PCT = 5.0     # % of the session's 1-90 cm/s time spent in the field

# Place cells ('Full') and their fields ('PlaceFields', bin_coords).
PLACE_CELL_FOLDER = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final'
FIELDS_EXCEL      = os.path.join(PLACE_CELL_FOLDER, 'All_TT_PlaceChar_SiSkaggsAdptBin_Final_OnlyPC.xlsx')

OUTPUT_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final\Results\PC_Char\FieldSpeedMod_v1'

# True: run the per-field speed analysis (figures + FIELD_RESULTS_XLSX), then the
# statistics. False: skip it and run the statistics on an existing FIELD_RESULTS_XLSX.
RUN_FIELD_SPEED = True

FIELD_RESULTS_XLSX = os.path.join(OUTPUT_DIR, 'FieldSpeedModulation.xlsx')
STATS_XLSX         = os.path.join(OUTPUT_DIR, 'FieldSpeed_Stats.xlsx')
STATS_CSV          = os.path.join(OUTPUT_DIR, 'FieldSpeed_AllStats.csv')
RUN_SNAPSHOT_SUBDIR = 'RunSnapshots'

# Per-field figures: <session>\FIELD_SPEED_SUBDIR under PLACE_CELL_FOLDER.
FIELD_SPEED_SUBDIR = 'field_speed_modulation'
SAVE_FIELD_FIGURES = True

# A field whose rebuilt PLACE_FIELD_RATEMAP_TYPE peak differs from its workbook
# peak_fr_hz by more than RATEMAP_CHECK_TOL_HZ is flagged (ratemap_check False:
# its bins may not line up with the rebuilt grid). True: leave those fields out
# of the statistics.
RATEMAP_CHECK_TOL_HZ     = 0.002
EXCLUDE_RATEMAP_MISMATCH = False

MAX_WORKERS = pc.MAX_WORKERS
SHUFFLE_SEED = None   # None: unseeded, as PlaceCell_Main_v21; an int makes the shuffles reproducible

# Speed-analysis settings: PlaceCell_Main_v21's (listed here so the run snapshot records them).
SPEED_MIN_CMS          = pc.SPEED_MIN_CMS
SPEED_MAX_CMS          = pc.SPEED_MAX_CMS
SPEED_BIN_CMS          = pc.SPEED_BIN_CMS
SPEED_SMOOTH_S         = pc.SPEED_SMOOTH_S
SPEED_MIN_BIN_FRAC     = pc.SPEED_MIN_BIN_FRAC
SPEED_N_SHUFFLE        = pc.SPEED_N_SHUFFLE
SPEED_SHUFFLE_MARGIN_S = pc.SPEED_SHUFFLE_MARGIN_S
SPEED_MOD_DOWNSAMPLE_FACTOR = pc.SPEED_MOD_DOWNSAMPLE_FACTOR
FPS = pc.fps

# Statistics: comparison folders (level 1, 2 and 3 use asc's folder names).
ANIMAL_ID_DIR  = 'AnimalID'
ALL_FIELDS_DIR = 'AllFields'
ALL_FIELDS_LABEL = 'All fields'

FIELD_SPEED_METRICS = {
    'final_speed_score': 'Field speed score (mean of binned and time-domain r)',
    'speed_score':       'Field speed score, binned regression (r)',
    'speed_score_td':    'Field speed score, time-domain (Pearson r)',
}
MODULATED_LABEL = 'Speed-modulated place fields (% of place fields)'
SPEED_DIR_LABEL = ('Field speed-direction classification '
                   '(p-type / n-type / non-modulated, % of place fields)')
SPEED_CLASSES = ['p', 'n', 'non']
SPEED_CLASS_NAMES = {'p': 'p-type', 'n': 'n-type', 'non': 'non-modulated'}

_print_lock = threading.Lock()


def _log(msg: str):
    with _print_lock:
        print(msg)


# ── Per-field speed modulation ──────────────────────────────────────────────

_SPEED_KEYS = ['speed_score', 'speed_p_value', 'speed_r2', 'speed_beta', 'speed_f0',
               'speed_modulated', 'speed_shuffle_mean', 'speed_shuffle_lo', 'speed_shuffle_hi',
               'speed_shuffle_p', 'speed_modulated_shuffle', 'speed_shuffle_ran',
               'speed_score_td', 'speed_p_value_td', 'speed_r2_td', 'speed_shuffle_mean_td',
               'speed_shuffle_lo_td', 'speed_shuffle_hi_td', 'speed_shuffle_p_td',
               'speed_modulated_td', 'speed_shuffle_ran_td', 'p_speed', 'n_speed']


def _empty_result() -> dict:
    """Same defaults as PlaceCell_Main_v21._compute_speed_modulation."""
    r = {k: float('nan') for k in _SPEED_KEYS}
    r.update(speed_modulated=None, speed_modulated_shuffle=None, speed_modulated_td=None,
             p_speed=None, n_speed=None, speed_shuffle_ran=False, speed_shuffle_ran_td=False)
    return r


def _final_class(r: dict) -> dict:
    """final_speed_score and speed_cell, same rule as PlaceCell_Main_v21._build_row."""
    s_bin, s_td = r['speed_score'], r['speed_score_td']
    both = bool(np.isfinite(s_bin) and np.isfinite(s_td))
    final = round((s_bin + s_td) / 2.0, 4) if both else float('nan')
    m_bin, m_td = r['speed_modulated_shuffle'], r['speed_modulated_td']
    if both and np.sign(s_bin) * np.sign(s_td) < 0:
        cell = False
    elif m_bin is False or m_td is False:
        cell = False
    elif m_bin is True and m_td is True:
        cell = True
    else:
        cell = 'not tested'
    return {'final_speed_score': final, 'speed_cell': cell}


def field_speed_modulation(x_cm, y_cm, t_us, spike_ts_us, pos_fps, field_masks: dict,
                           moving_spikes: dict, rng) -> tuple[dict, dict]:
    """PlaceCell_Main_v21._compute_speed_modulation restricted, per field, to the
    frames inside that field (field_masks: {field_number: bool per frame}).
    moving_spikes: {field_number: moving in-field spike count} for the minimum.
    Returns ({field_number: result dict}, plot data shared by the fields)."""
    out = {fn: dict(_empty_result(), status='', n_samples_infield=0,
                    infield_time_s=np.nan, total_time_s=np.nan, infield_dwell_pct=np.nan,
                    n_spikes_infield_analysis=0)
           for fn in field_masks}
    n = len(t_us)
    if n < 3 or len(spike_ts_us) == 0:
        for r in out.values():
            r['status'] = 'no tracking / spikes'
        return out, {}

    # ── Whole-session speed and rate, exactly as PlaceCell_Main_v21 ─────────
    dt_s = np.diff(t_us) * 1e-6
    with np.errstate(invalid='ignore', divide='ignore'):
        speed = np.hypot(np.diff(x_cm), np.diff(y_cm)) / dt_s
    speed[dt_s <= 0] = np.nan
    speed = np.append(speed, speed[-1])
    sigma_samples = max(SPEED_SMOOTH_S * pos_fps, 1e-6)
    speed_smooth = pc._smooth_1d(speed, np.isfinite(speed), sigma_samples)

    interval_idx = np.clip(np.searchsorted(t_us, spike_ts_us, side='right') - 1, 0, n - 2)
    counts = np.bincount(interval_idx, minlength=n - 1).astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        fr_inst = counts / dt_s
    fr_inst[dt_s <= 0] = np.nan
    fr_inst = np.append(fr_inst, fr_inst[-1])
    finite = np.isfinite(fr_inst)
    fr_smooth = pc._smooth_1d(fr_inst, finite, sigma_samples)

    base_range = ((speed_smooth > SPEED_MIN_CMS) & (speed_smooth < SPEED_MAX_CMS)
                  & np.isfinite(fr_smooth))
    # Frame durations for the dwell minimum, capped as the occupancy maps.
    w = np.append(dt_s, dt_s[-1])
    w = np.where(np.isfinite(w) & (w > 0), np.minimum(w, 2.0 / pos_fps), 0.0)
    total_time = float(w[base_range].sum())
    counts_pad = np.append(counts, 0.0)          # spikes per sample (interval)

    speed_bins  = np.arange(SPEED_MIN_CMS, SPEED_MAX_CMS + SPEED_BIN_CMS, SPEED_BIN_CMS)
    n_bins      = len(speed_bins) - 1
    bin_centres = 0.5 * (speed_bins[:-1] + speed_bins[1:])

    state = {}
    for fn, mask in field_masks.items():
        r = out[fn]
        in_range = base_range & mask
        r['n_samples_infield'] = int(in_range.sum())
        r['infield_time_s'] = round(float(w[in_range].sum()), 3)
        r['total_time_s'] = round(total_time, 3)
        r['infield_dwell_pct'] = (round(100.0 * w[in_range].sum() / total_time, 3)
                                  if total_time > 0 else np.nan)
        r['n_spikes_infield_analysis'] = int(counts_pad[in_range].sum())

        fails = []
        if moving_spikes[fn] < MIN_INFIELD_SPIKES:
            fails.append(f'{moving_spikes[fn]} moving in-field spikes < {MIN_INFIELD_SPIKES}')
        if not (r['infield_dwell_pct'] >= MIN_INFIELD_DWELL_PCT):
            fails.append(f'in-field dwell {r["infield_dwell_pct"]} % < {MIN_INFIELD_DWELL_PCT:g} %')
        if fails:
            r['status'] = 'below minimum: ' + '; '.join(fails)
            continue

        if in_range.sum() < 3:
            r['status'] = '< 3 in-field samples'
            continue
        speed_valid = speed_smooth[in_range]
        rate_valid  = fr_smooth[in_range]
        if np.std(speed_valid) == 0 or np.std(rate_valid) == 0:
            r['status'] = 'constant in-field speed or rate'
            continue

        r_td, p_td = pearsonr(speed_valid, rate_valid)
        r['speed_score_td']   = round(float(r_td), 4)
        r['speed_r2_td']      = round(float(r_td ** 2), 4)
        r['speed_p_value_td'] = round(float(p_td), 4)

        bin_idx = np.clip(np.digitize(speed_valid, speed_bins) - 1, 0, n_bins - 1)
        binned_rate = np.full(n_bins, np.nan)
        n_per_bin = np.zeros(n_bins, dtype=int)
        bin_masks = []
        for b in range(n_bins):
            sel = bin_idx == b
            bin_masks.append(sel)
            n_per_bin[b] = int(sel.sum())
            if n_per_bin[b] > 0:
                binned_rate[b] = float(np.median(rate_valid[sel]))
        total_pts = n_per_bin.sum()
        binned_rate[(n_per_bin / total_pts) < SPEED_MIN_BIN_FRAC] = np.nan
        fit_sel = np.isfinite(binned_rate)
        if fit_sel.sum() < 3:
            r['status'] = '< 3 speed bins'
            continue
        reg = pc._weighted_linregress(bin_centres[fit_sel], binned_rate[fit_sel],
                                      n_per_bin[fit_sel].astype(np.float64))
        r['speed_score']     = round(float(reg.rvalue), 4)
        r['speed_p_value']   = round(float(reg.pvalue), 4)
        r['speed_r2']        = round(float(reg.rvalue ** 2), 4)
        r['speed_beta']      = round(float(reg.slope), 4)
        r['speed_f0']        = round(float(reg.intercept), 4)
        r['speed_modulated'] = bool(reg.pvalue < 0.05)
        r['status'] = 'tested'

        fit_idx = np.where(fit_sel)[0]
        state[fn] = dict(in_range=in_range, speed_valid=speed_valid, rate_valid=rate_valid,
                         r_td=float(r_td), real_score=float(reg.rvalue), reg=reg,
                         bin_sig=r['speed_modulated'], td_sig=bool(p_td < 0.05),
                         fit_sel=fit_sel, binned_rate=binned_rate,
                         n_per_bin_fit=n_per_bin[fit_idx].astype(np.float64),
                         bin_centres_fit=bin_centres[fit_idx],
                         bin_masks_fit=[bin_masks[b] for b in fit_idx],
                         shuff=np.full(SPEED_N_SHUFFLE, np.nan),
                         shuff_td=np.full(SPEED_N_SHUFFLE, np.nan), shuffled=False)

    # ── Circular-shift shuffle of the whole-session spike train ─────────────
    n_intervals = len(counts)
    margin = int(SPEED_SHUFFLE_MARGIN_S * pos_fps)
    gated = {fn: s for fn, s in state.items() if s['bin_sig'] or s['td_sig']}
    if n_intervals > 2 * margin and gated:
        for s in gated.values():
            s['shuffled'] = True
        for i in range(SPEED_N_SHUFFLE):
            rnd = int(rng.integers(margin, n_intervals - margin + 1))   # randint: inclusive
            with np.errstate(invalid='ignore', divide='ignore'):
                fr_inst_s = np.roll(counts, rnd) / dt_s
            fr_inst_s[dt_s <= 0] = np.nan
            fr_inst_s = np.append(fr_inst_s, fr_inst_s[-1])
            fr_smooth_s = pc._smooth_1d(fr_inst_s, finite, sigma_samples)
            for s in gated.values():
                rate_s = fr_smooth_s[s['in_range']]
                if s['td_sig'] and np.std(rate_s) > 0:
                    s['shuff_td'][i] = pearsonr(s['speed_valid'], rate_s)[0]
                if s['bin_sig']:
                    br = np.array([np.median(rate_s[m]) for m in s['bin_masks_fit']])
                    if np.std(br) != 0:
                        s['shuff'][i] = pc._weighted_linregress(
                            s['bin_centres_fit'], br, s['n_per_bin_fit']).rvalue

    for fn, s in state.items():
        r = out[fn]
        if s['shuffled']:
            r['speed_shuffle_ran'] = s['bin_sig']
            r['speed_shuffle_ran_td'] = s['td_sig']
            summ = pc._shuffle_ci_and_pvalue(s['real_score'], s['shuff'])
            if summ is not None:
                r.update(speed_shuffle_mean=round(summ.mean, 4), speed_shuffle_lo=round(summ.lo, 4),
                         speed_shuffle_hi=round(summ.hi, 4), speed_shuffle_p=round(summ.p, 4),
                         speed_modulated_shuffle=summ.modulated)
            summ_td = pc._shuffle_ci_and_pvalue(s['r_td'], s['shuff_td'])
            if summ_td is not None:
                r.update(speed_shuffle_mean_td=round(summ_td.mean, 4),
                         speed_shuffle_lo_td=round(summ_td.lo, 4),
                         speed_shuffle_hi_td=round(summ_td.hi, 4),
                         speed_shuffle_p_td=round(summ_td.p, 4),
                         speed_modulated_td=summ_td.modulated,
                         p_speed=bool(s['r_td'] > summ_td.hi), n_speed=bool(s['r_td'] < summ_td.lo))

    for r in out.values():
        r.update(_final_class(r))
    plot = dict(t_us=t_us, speed_smooth=speed_smooth, fr_smooth=fr_smooth,
                bin_centres=bin_centres, state=state)
    return out, plot


def _plot_field(plot: dict, fn: int, r: dict, title: str, save_path: str):
    """2 x 2 figure of one field, as PlaceCell_Main_v21's speed figure (in-field samples)."""
    s = plot['state'][fn]
    fig = Figure(figsize=(14, 10))
    FigureCanvasAgg(fig)
    ax_bin, ax_td = fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2)
    ax_bin_shuf, ax_td_shuf = fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4)

    # Binned scatter / fit (in-field samples)
    reg, fit_sel, bc = s['reg'], s['fit_sel'], plot['bin_centres']
    ax_bin.scatter(s['speed_valid'], s['rate_valid'], s=3, color='0.75', alpha=0.4,
                   label='in-field samples')
    ax_bin.scatter(bc[fit_sel], s['binned_rate'][fit_sel], s=80, color='black', label='binned median')
    fx = np.array([bc[fit_sel].min(), bc[fit_sel].max()])
    ax_bin.plot(fx, reg.slope * fx + reg.intercept, 'r-', label='fit')
    ax_bin.text(0.02, 0.98,
                f"binned: r = {reg.rvalue:.3f}, slope = {reg.slope:.3f}, "
                f"intercept = {reg.intercept:.3f}, p = {reg.pvalue:.3g}\n"
                f"time-domain: r = {s['r_td']:.3f}, p(shuffle) = {r['speed_shuffle_p_td']}\n"
                f"{r['n_samples_infield']} in-field samples, {r['infield_time_s']} s "
                f"({r['infield_dwell_pct']} % of {SPEED_MIN_CMS:g}-{SPEED_MAX_CMS:g} cm/s time)",
                transform=ax_bin.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_bin.set_xlabel('speed (cm/s)')
    ax_bin.set_ylabel('firing rate (Hz)')
    ax_bin.legend(loc='lower right', fontsize=8)

    # Time series: whole session, in-field analysis samples highlighted
    time_s = (plot['t_us'] - plot['t_us'][0]) * 1e-6
    inr = s['in_range']
    ax_td.plot(time_s, plot['speed_smooth'], color='0.8', linewidth=0.5)
    ax_td.plot(time_s, np.where(inr, plot['speed_smooth'], np.nan), color='tab:blue', linewidth=0.8)
    ax_td.set_xlabel('time (s)')
    ax_td.set_ylabel('speed (cm/s), in-field samples blue', color='tab:blue')
    ax_tw = ax_td.twinx()
    ax_tw.plot(time_s, np.where(inr, plot['fr_smooth'], np.nan), color='tab:orange', linewidth=0.8)
    ax_tw.set_ylabel('firing rate (Hz), in-field samples', color='tab:orange')

    if r['p_speed']:
        lbl, col = 'p-Speed', 'green'
    elif r['n_speed']:
        lbl, col = 'n-Speed', 'red'
    else:
        lbl, col = 'non-speed', 'black'
    ax_td.set_title(f"time-domain: r = {s['r_td']:.3f}" +
                    (f"  {lbl} (shuffle p = {r['speed_shuffle_p_td']})"
                     if r['speed_modulated_td'] is not None else
                     f" (parametric p = {r['speed_p_value_td']})"),
                    color=col if r['speed_modulated_td'] is not None else 'black')

    for ax, sh, real, lo, hi, p, mod, name in (
            (ax_bin_shuf, s['shuff'], s['real_score'], r['speed_shuffle_lo'], r['speed_shuffle_hi'],
             r['speed_shuffle_p'], r['speed_modulated_shuffle'], 'binned-fit'),
            (ax_td_shuf, s['shuff_td'], s['r_td'], r['speed_shuffle_lo_td'], r['speed_shuffle_hi_td'],
             r['speed_shuffle_p_td'], r['speed_modulated_td'], 'time-domain')):
        v = sh[np.isfinite(sh)]
        if mod is None or not len(v):
            ax.text(0.5, 0.5, f'{name} shuffle not run\n(initial fit not significant)',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        cnt = ax.hist(v, 100, color='black')[0]
        ax.plot([real, real], [0, max(cnt.max(), 1)], 'r-.')
        ax.axvline(lo, color='0.5', linestyle=':', linewidth=0.8)
        ax.axvline(hi, color='0.5', linestyle=':', linewidth=0.8)
        ax.set_title(f'{name}: shuffle mean r = {np.mean(v):.3f}\n95% CI = [{lo:.3f}, {hi:.3f}]\n'
                     f'field r = {real:.3f} (p = {p}, {"p < 0.05" if mod else "ns"})')
        ax.set_xlabel(f'{name} speed score (r)')
        ax.set_ylabel('count')

    fig.suptitle(f"{title}\nfinal speed score = {r['final_speed_score']}, "
                 f"speed-modulated = {r['speed_cell']}", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)


# ── One cell: rebuild, field masks, analysis ────────────────────────────────

def _find_tracking_csv(dirpath: str):
    """Same tracking-file selection as PlaceCell_Main_v21's batch scan."""
    tracking = [f for f in os.listdir(dirpath) if f.lower().endswith(('.csv', '.xlsx'))]
    if pc.COORD_UNITS == 'cm':
        tracking = [f for f in tracking if f.lower().endswith('_cm.csv')]
    else:
        tracking = [f for f in tracking if not f.lower().endswith('_cm.csv')]
    return os.path.join(dirpath, tracking[0]) if len(tracking) == 1 else None


def _field_bins(bin_coords: str, is_circle: bool) -> np.ndarray:
    """bin_coords -> (n, 2) array of (bx, by), or (n,) array of angular bins."""
    if is_circle:
        return np.array([int(b) for b in str(bin_coords).split(';')])
    return np.array([[int(v) for v in p.split('-')] for p in str(bin_coords).split(';')])


def _run_cell(args) -> list[dict]:
    idx, total, session, unit, fdf, seed = args
    dirpath  = os.path.join(PLACE_CELL_FOLDER, session)
    ntt_path = os.path.join(dirpath, unit)
    _log(f'[{idx}/{total}]  {session}  |  {unit}  ({len(fdf)} field(s))')

    base = [{'session': session, 'unit': unit, 'field_number': int(f['field_number']),
             'field_geometry': f['arena_type'], 'n_bins': f.get('n_bins'),
             'field_n_spikes_excel': f.get('field_n_spikes'), 'peak_fr_hz': f.get('peak_fr_hz')}
            for _, f in fdf.iterrows()]

    def _fail(msg):
        _log(f'  [SKIP] {session} | {unit}: {msg}')
        return [dict(b, status=msg, **_empty_result(), final_speed_score=np.nan,
                     speed_cell='not tested') for b in base]

    if not os.path.isfile(ntt_path):
        return _fail('.ntt not found')
    csv_path = _find_tracking_csv(dirpath)
    if csv_path is None:
        return _fail('no single tracking file')
    _, ctx = pc.compute_metrics(csv_path, ntt_path, pc.arena_width_cm, pc.target_bin_cm)
    if not ctx or ctx.get('spike_ts_all') is None or not len(ctx['spike_ts_all']):
        return _fail('no tracking / spikes after rebuild')

    ds = SPEED_MOD_DOWNSAMPLE_FACTOR
    x_all, y_all, t_all = ctx['x_cm_all'][::ds], ctx['y_cm_all'][::ds], ctx['t_all'][::ds]
    is_circle = pc._is_circle_arena(dirpath)
    if is_circle != (str(fdf['arena_type'].iloc[0]) == 'Circle'):
        return _fail(f'arena mismatch (folder Circle = {is_circle}, workbook '
                     f'{fdf["arena_type"].iloc[0]})')
    sf = ctx['spike_frame']

    if is_circle:
        cc = pc._build_ratemap_circular(ctx, pc.target_bin_cm)
        theta = np.mod(np.arctan2(y_all - cc['cy'], x_all - cc['cx']), 2.0 * np.pi)
        frame_bin = np.clip((theta / cc['bin_width_rad']).astype(int), 0, cc['n_bins_theta'] - 1)
        spike_bin = cc['beh_bt'][sf]
        rate_map, grid = cc['fr_field'], (cc['n_bins_theta'],)
    else:
        nbx, nby = ctx['n_bins_x'], ctx['n_bins_y']
        bx = np.clip((x_all / pc.target_bin_cm).astype(int), 0, nbx - 1)
        by = np.clip((y_all / pc.target_bin_cm).astype(int), 0, nby - 1)
        frame_bin = bx * nby + by
        spike_bin = ctx['beh_bx'][sf] * nby + ctx['beh_by'][sf]
        rate_map, grid = ctx['fr_field'], (nbx, nby)

    masks, moving_spikes, checks = {}, {}, {}
    for b in base:
        fn = b['field_number']
        bins = _field_bins(fdf.loc[fdf['field_number'] == fn, 'bin_coords'].iloc[0], is_circle)
        if is_circle:
            if bins.min() < 0 or bins.max() >= grid[0]:
                return _fail(f'field {fn} outside the rebuilt {grid[0]}-bin angular map')
            flat = bins
            peak = float(rate_map[bins].max())
        else:
            if bins.min() < 0 or bins[:, 0].max() >= grid[0] or bins[:, 1].max() >= grid[1]:
                return _fail(f'field {fn} outside the rebuilt {grid[0]} x {grid[1]} grid')
            flat = bins[:, 0] * grid[1] + bins[:, 1]
            peak = float(rate_map[bins[:, 0], bins[:, 1]].max())
        in_field = np.zeros(int(np.prod(grid)), dtype=bool)
        in_field[flat] = True
        masks[fn] = in_field[frame_bin]
        moving_spikes[fn] = int(in_field[spike_bin].sum())
        checks[fn] = (round(peak, 4), bool(abs(peak - float(b['peak_fr_hz'])) <= RATEMAP_CHECK_TOL_HZ))

    rng = np.random.default_rng(seed)
    res, plot = field_speed_modulation(x_all, y_all, t_all, ctx['spike_ts_all'], FPS / ds,
                                       masks, moving_spikes, rng)

    rows = []
    ntt_name = os.path.splitext(unit)[0]
    for b in base:
        fn = b['field_number']
        r = res[fn]
        row = dict(b, n_spikes_moving_infield=moving_spikes[fn],
                   rebuilt_peak_fr_hz=checks[fn][0], ratemap_check=checks[fn][1], **r)
        if SAVE_FIELD_FIGURES and plot and fn in plot['state']:
            path = os.path.join(dirpath, FIELD_SPEED_SUBDIR,
                                f'{ntt_name}_F{fn}_field_speed_modulation.png')
            try:
                _plot_field(plot, fn, r, f'{session}  |  {ntt_name}  |  field {fn}', path)
                row['figure'] = path
            except Exception as e:  # noqa: BLE001
                _log(f'  PLOT ERROR {session} | {unit} F{fn}: {e}')
        rows.append(row)
    n_mod = sum(r['speed_cell'] is True for r in res.values())
    _log(f'  done {session} | {unit}: ' + ', '.join(
        f"F{fn} {res[fn]['status'] if res[fn]['status'] != 'tested' else res[fn]['speed_cell']}"
        for fn in res) + f'  ({n_mod} modulated)')
    if not all(c[1] for c in checks.values()):
        _log(f'  [WARN] {session} | {unit}: rebuilt field peak != workbook peak_fr_hz in field(s) '
             f'{[fn for fn, c in checks.items() if not c[1]]}')
    return rows


def run_field_speed(full: pd.DataFrame, fields: pd.DataFrame) -> pd.DataFrame:
    """Per-field speed modulation of every field of every place cell; writes
    FIELD_RESULTS_XLSX and returns its PerField table."""
    is_pc = full['place_cell'].apply(asc._to_bool)
    keys = set(zip(full.loc[is_pc, 'session'], full.loc[is_pc, 'unit']))
    keep = np.array([(s, u) in keys for s, u in zip(fields['session'], fields['unit'])], dtype=bool)
    f = fields[keep & fields['bin_coords'].notna().to_numpy()]
    cells = list(f.groupby(['session', 'unit'], sort=True))
    print(f'\nField speed modulation: {len(keys)} place cell(s) in {FIELDS_EXCEL}; '
          f'{len(cells)} with fields, {len(f)} field(s).\n')

    seeds = (np.random.SeedSequence(SHUFFLE_SEED).spawn(len(cells))
             if SHUFFLE_SEED is not None else [None] * len(cells))
    job_args = [(i, len(cells), s, u, d.sort_values('field_number'), seeds[i - 1])
                for i, ((s, u), d) in enumerate(cells, start=1)]
    rows = []
    with matplotlib.rc_context({'axes.spines.top': True, 'axes.spines.right': True}), \
            concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_run_cell, a): a for a in job_args}
        for fut in concurrent.futures.as_completed(futures):
            a = futures[fut]
            try:
                rows += fut.result()
            except Exception as e:  # noqa: BLE001
                print(f'  ERROR in {a[2]} | {a[3]}: {e}')

    per_field = pd.DataFrame(rows).sort_values(['session', 'unit', 'field_number'], ignore_index=True)
    per_field['tested'] = per_field['status'] == 'tested'
    settings = pd.DataFrame(
        [{'parameter': 'script', 'value': os.path.basename(__file__)},
         {'parameter': 'run_timestamp', 'value': datetime.now().isoformat(timespec='seconds')}]
        + [{'parameter': k, 'value': str(v)} for k, v in _module_settings().items()])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with pd.ExcelWriter(FIELD_RESULTS_XLSX, engine='openpyxl') as w:
        per_field.to_excel(w, sheet_name='PerField', index=False)
        settings.to_excel(w, sheet_name='Settings', index=False)

    print(f'\nPer-field results saved to {FIELD_RESULTS_XLSX}')
    print(per_field['status'].str.split(':').str[0].value_counts().to_string())
    n_bad = int((~per_field['ratemap_check'].fillna(False).astype(bool)).sum())
    if n_bad:
        print(f'[WARN] {n_bad} field(s) failed ratemap_check (EXCLUDE_RATEMAP_MISMATCH = '
              f'{EXCLUDE_RATEMAP_MISMATCH}).')
    spk_diff = (pd.to_numeric(per_field['n_spikes_moving_infield'], errors='coerce')
                != pd.to_numeric(per_field['field_n_spikes_excel'], errors='coerce'))
    if spk_diff.any():
        print(f'[NOTE] {int(spk_diff.sum())} field(s): rebuilt moving in-field spikes != '
              f'workbook field_n_spikes.')
    return per_field


# ── Statistics table ────────────────────────────────────────────────────────

def build_stats_df(per_field: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    """Tested fields, tagged with arena / session type / animal / cell_id (asc._tag_groups),
    with speed_class / speed_cell_final (asc.add_speed_class), the cell's 'sir' (point
    shade) and the cell's whole-session speed class ('cell_speed_class')."""
    df = per_field[per_field['status'] == 'tested'].copy()
    print(f'\nStatistics: {len(df)} of {len(per_field)} field(s) tested.')
    if EXCLUDE_RATEMAP_MISMATCH:
        ok = df['ratemap_check'].fillna(False).astype(bool)
        print(f'EXCLUDE_RATEMAP_MISMATCH: {int((~ok).sum())} field(s) left out.')
        df = df[ok]
    cell_cols = full[['session', 'unit', 'sir', 'final_speed_score', 'speed_cell']].rename(
        columns={'final_speed_score': 'cell_final_speed_score', 'speed_cell': 'cell_speed_cell'})
    df = df.merge(cell_cols.drop_duplicates(['session', 'unit']), on=['session', 'unit'], how='left')
    df['cell_speed_class'] = [asc._speed_class(s, c) for s, c in
                              zip(df['cell_final_speed_score'], df['cell_speed_cell'])]
    df = asc.add_speed_class(asc._tag_groups(df, 'place field(s) (field speed)'))
    df['all_cells'] = ALL_FIELDS_LABEL
    return df


# ── Plots / tests for one comparison ────────────────────────────────────────

def plot_field_speed_direction(df: pd.DataFrame, out_path: str, grp, stats_lines=None,
                               posthoc_df=None):
    """asc.plot_speed_direction for fields (y label / legend in field terms)."""
    present, pct, ns = [], {k: [] for k in SPEED_CLASSES}, []
    for g in grp.order:
        cls = df.loc[df[grp.col] == g, 'speed_class'].dropna()
        if not len(cls):
            continue
        present.append(g)
        ns.append(len(cls))
        for k in SPEED_CLASSES:
            pct[k].append(100.0 * (cls == k).sum() / len(cls))
    if not present:
        return
    fig_w = asc._bar_fig_width(len(present))
    stats_lines = asc._wrap_stats(stats_lines, fig_w)
    fig, ax, ax_info = asc._new_fig((fig_w, 5.5), legend_rows=1, stats_lines=stats_lines)
    pos = np.arange(len(present))
    kw = dict(edgecolor=asc.BAR_EDGE, linewidth=asc.BAR_EDGE_LW, width=asc.BAR_WIDTH, zorder=2)
    for bars in (ax.bar(pos, pct['p'], color=[asc._color(g)['dark'] for g in present],
                        label='p-type field', **kw),
                 ax.bar(pos, pct['n'], bottom=pct['p'], color=[asc._color(g)['light'] for g in present],
                        label='n-type field', **kw),
                 ax.bar(pos, pct['non'], bottom=np.add(pct['p'], pct['n']), color=asc.SIG_GRAY['face'],
                        label='non-modulated', **kw)):
        asc.plu.label_bars(ax, bars, inside=True, fontsize=asc.BAR_PCT_FS)   # stacked: % in each segment
    ax.set_xticks(pos)
    ax.set_xticklabels([f'{asc._label(g)}\n(n={n})' for g, n in zip(present, ns)])
    ax.set_ylabel('% of place fields', fontsize=asc.FS_LABEL, labelpad=8)
    asc._set_percent_ylim(ax, dict(zip(present, pos)), posthoc_df, 100)
    asc._set_title(ax, asc._titled('Field speed-direction classification', grp))
    asc._style_axes(ax)
    h, l = ax.get_legend_handles_labels()
    ax_info.legend(h, l, loc='upper center', fontsize=asc.FS_LEGEND, ncol=3)
    asc._draw_stats(ax_info, stats_lines)
    fig.tight_layout()
    asc._save(fig, out_path)


def field_vs_cell(df: pd.DataFrame, grp, out_path: str) -> pd.DataFrame:
    """Per group: each field's speed class against its cell's whole-session class,
    stacked bars (x = cell class, stack = % of fields per field class), chi-square
    of independence per group. Returns the counts table."""
    sub = df[df['speed_class'].isin(SPEED_CLASSES) & df['cell_speed_class'].isin(SPEED_CLASSES)]
    present = [g for g in grp.order if (sub[grp.col] == g).any()]
    rows, lines = [], []
    for g in present:
        s = sub[sub[grp.col] == g]
        tab = pd.crosstab(s['cell_speed_class'], s['speed_class']).reindex(
            index=SPEED_CLASSES, columns=SPEED_CLASSES, fill_value=0)
        for c in SPEED_CLASSES:
            for f in SPEED_CLASSES:
                n_c = int(tab.loc[c].sum())
                rows.append({grp.col: g, 'cell_class': SPEED_CLASS_NAMES[c],
                             'field_class': SPEED_CLASS_NAMES[f], 'n_fields': int(tab.loc[c, f]),
                             'pct_of_cell_class': 100.0 * tab.loc[c, f] / n_c if n_c else np.nan})
        arr = tab.to_numpy(float)
        arr = arr[arr.sum(axis=1) > 0][:, arr.sum(axis=0) > 0]
        agree = 100.0 * np.trace(tab.to_numpy()) / len(s)
        line = f'{asc._label(g)} (n={len(s)} fields): {agree:.1f} % share their cell\'s class'
        if arr.shape[0] >= 2 and arr.shape[1] >= 2:
            chi2_, p, dof, exp = stats.chi2_contingency(arr)
            line += f'; χ²({dof}) = {chi2_:.2f}, p = {asc._fmt_p(p)} {asc._stars(p)}'
            if (exp < 5).any():
                line += f' ({int((exp < 5).sum())} expected counts < 5)'
            for r in rows[-9:]:          # this group's 3 x 3 rows
                r.update(chi2=chi2_, dof=dof, p_value=p)
        else:
            line += '; no test (< 2 classes)'
        lines.append(line)
    table = pd.DataFrame(rows)
    if not present:
        return table

    lines.append('Chi-square of independence (field class x cell class) within each group; '
                 'fields of one cell treated as independent.')
    n_pan = len(present)
    lines = asc._wrap_stats(lines, max(asc.HIST_PANEL_W * n_pan, 7))
    # Own panels (asc._new_panel_fig shares the x axis, which would share the tick labels).
    info_h = len(lines) * asc.STATS_LINE_IN + 0.4
    fig = plt.figure(figsize=(max(asc.HIST_PANEL_W * n_pan, 7), asc.HIST_PANEL_H + info_h))
    gs = fig.add_gridspec(2, n_pan, height_ratios=[asc.HIST_PANEL_H, info_h])
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_pan)]
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.axis('off')
    for ax, g in zip(axes, present):
        t = table[table[grp.col] == g]
        x = np.arange(3)
        bottom = np.zeros(3)
        c = asc._color(g)
        for f, colr in zip(SPEED_CLASSES, (c['dark'], c['light'], asc.SIG_GRAY['face'])):
            v = [t.loc[(t['cell_class'] == SPEED_CLASS_NAMES[cc]) &
                       (t['field_class'] == SPEED_CLASS_NAMES[f]), 'pct_of_cell_class'].fillna(0).iloc[0]
                 for cc in SPEED_CLASSES]
            bars = ax.bar(x, v, bottom=bottom, color=colr, edgecolor=asc.BAR_EDGE,
                          linewidth=asc.BAR_EDGE_LW, width=asc.BAR_WIDTH, zorder=2,
                          label=f'{SPEED_CLASS_NAMES[f]} field')
            asc.plu.label_bars(ax, bars, inside=True, fontsize=asc.BAR_PCT_FS - 2)   # stacked
            bottom += v
        n_c = [int(t.loc[t['cell_class'] == SPEED_CLASS_NAMES[cc], 'n_fields'].sum())
               for cc in SPEED_CLASSES]
        ax.set_xticks(x)
        ax.set_xticklabels([f'{SPEED_CLASS_NAMES[cc]}\ncell (n={n})' for cc, n in zip(SPEED_CLASSES, n_c)],
                           fontsize=asc.FS_TICK - 1)
        ax.set_ylim(0, 100)
        ax.set_title(asc._label(g), fontsize=asc.FS_LABEL, pad=10)
        asc._style_axes(ax)
    axes[0].set_ylabel('% of the cell class\'s fields', fontsize=asc.FS_LABEL, labelpad=6)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=asc.FS_LEGEND, bbox_to_anchor=(0.5, 1.0))
    asc._draw_stats(ax_info, lines)
    fig.suptitle(asc._titled('Field speed class vs its cell\'s whole-session class', grp),
                 fontsize=asc.FS_TITLE, fontweight='bold', y=1.1)
    fig.tight_layout()
    asc._save(fig, out_path)
    return table


def run_field_comparison(grp, df: pd.DataFrame, plots_dir: str) -> dict:
    """Every field-speed statistic and plot across the groups of `grp`; df already
    restricted to the fields to compare. Returns {table_name: table}."""
    name = grp.title or grp.noun.capitalize()
    print(f'\n{"=" * 70}\n{name} -- {[asc._label(g) for g in grp.order]}\n{"=" * 70}')
    print('Tested place fields per group:')
    print(df[grp.col].value_counts().reindex(grp.order).fillna(0).astype(int).to_string())
    if not len(df):
        print('No fields; skipping.')
        return {}
    os.makedirs(plots_dir, exist_ok=True)
    # Point shade: the cell's SIR rank among this comparison's cells (fields share it).
    grp = replace(grp, shade=asc.cell_shades(df.drop_duplicates(['session', 'unit'])))

    # % speed-modulated fields
    cat_d, cat_o, cat_p = asc.compare_categorical(df, 'speed_cell_final', MODULATED_LABEL, grp)
    asc.plot_categorical(df, 'speed_cell_final', MODULATED_LABEL,
                         os.path.join(plots_dir, 'Categorical_field_speed_modulated.png'), grp,
                         stats_lines=asc._categorical_stats_lines(cat_d, cat_o, cat_p, grp),
                         posthoc_df=cat_p, ylabel='% of place fields')

    # p / n / non
    sd_d, sd_o, sd_p = asc.compare_speed_direction(df, grp)
    for t in (sd_d, sd_p):
        if len(t):
            t['metric'] = SPEED_DIR_LABEL
    sd_o['metric'] = SPEED_DIR_LABEL
    plot_field_speed_direction(df, os.path.join(plots_dir, 'Categorical_field_speed_direction.png'),
                               grp, stats_lines=asc._speed_dir_stats_lines(sd_d, sd_o, sd_p, grp),
                               posthoc_df=sd_p)

    # Per-field speed scores
    desc, omni, post = asc.run_all_comparisons(df, FIELD_SPEED_METRICS, 'FieldSpeed', grp, plots_dir)
    hist_dir = os.path.join(plots_dir, 'Histograms')
    os.makedirs(hist_dir, exist_ok=True)
    for col, label in FIELD_SPEED_METRICS.items():
        asc.plot_histogram_by_group(df, col, label,
                                    os.path.join(hist_dir, f'Hist_FieldSpeed_{col}.png'), grp)
    asc.plot_speed_histograms_by_group(df, 'final_speed_score', FIELD_SPEED_METRICS['final_speed_score'],
                                       os.path.join(hist_dir, 'Hist_FieldSpeed_modulated_fields.png'), grp)

    fvc = field_vs_cell(df, grp, os.path.join(plots_dir, 'FieldVsCell_speed_class.png'))

    if grp.tests:
        asc._print_significant(pd.concat([pd.DataFrame([cat_o, sd_o]), omni], ignore_index=True),
                               name, 70, 35)
    print(f'Plots saved to {plots_dir}')
    return {'CatDesc': cat_d, 'CatOmnibus': pd.DataFrame([cat_o]), 'CatPostHoc': cat_p,
            'SpeedDir_Desc': sd_d, 'SpeedDir_Omnibus': pd.DataFrame([sd_o]), 'SpeedDir_PostHoc': sd_p,
            'Desc': desc, 'Omnibus': omni, 'PostHoc': post, 'FieldVsCell': fvc}


def run_animal_controlled(df: pd.DataFrame, out_dir: str) -> dict:
    """Level 3: each asc.ANIMAL_CONTROLLED_COMPARISONS entry, plots side by side."""
    all_tables = {}
    with tempfile.TemporaryDirectory() as tmp:
        # Same y-range for the two versions of every plot (asc.plu.shared_y_axes).
        with asc.plu.shared_y_axes([os.path.join(tmp, n) for n in asc.ANIMAL_CONTROLLED_COMPARISONS]):
            for name, (animals, arenas) in asc.ANIMAL_CONTROLLED_COMPARISONS.items():
                grp = asc.animal_controlled_grouping(animals, arenas)
                wanted = {a.lower() for a in animals}
                sel = df[df['animal'].astype(str).str.lower().isin(wanted) & df['arena_type'].isin(grp.order)]
                print(f'\n{name}: tested fields per animal and arena:')
                print(pd.crosstab(sel['animal'], sel['arena_type'], margins=True).to_string())
                tables = run_field_comparison(grp, sel, os.path.join(tmp, name))
                tag = 'AC' + ''.join(grp.order)
                for tname, tbl in tables.items():
                    if len(tbl):
                        tbl.insert(0, 'animals', ', '.join(animals))
                    all_tables[f'{tag}_{tname}'] = tbl
        asc._combine_side_by_side([os.path.join(tmp, n) for n in asc.ANIMAL_CONTROLLED_COMPARISONS],
                                  out_dir)
    return all_tables


# ── Run snapshot ────────────────────────────────────────────────────────────

def _module_settings() -> dict:
    return {k: v for k, v in globals().items()
            if k.isupper() and not k.startswith('_')
            and isinstance(v, (str, int, float, bool, list, tuple, dict, type(None)))}


def save_run_snapshot() -> str:
    """This script + the local modules it uses, and RunMetadata.xlsx (run info +
    every setting of this script and PlaceCell_Main_v21), in
    OUTPUT_DIR\\RUN_SNAPSHOT_SUBDIR\\<timestamp>."""
    now = datetime.now()
    snap = os.path.join(OUTPUT_DIR, RUN_SNAPSHOT_SUBDIR, now.strftime('%Y%m%d_%H%M%S'))
    code = os.path.join(snap, 'code')
    os.makedirs(code, exist_ok=True)
    script = os.path.abspath(__file__)
    mods = [pc, asc, ccs, pfs]
    for p in [script] + [m.__file__ for m in mods]:
        shutil.copy2(p, code)
    run = [('run_timestamp', now.isoformat(timespec='seconds')), ('script', script),
           ('git_commit', asc._git('rev-parse', 'HEAD') or 'n/a'),
           ('git_uncommitted_changes', asc._git('status', '--porcelain', '--', script,
                                                *[m.__file__ for m in mods]) or 'none'),
           ('python', sys.version.replace('\n', ' ')), ('platform', platform.platform()),
           ('machine', platform.node())]
    for pkg in ('numpy', 'pandas', 'scipy', 'matplotlib', 'statsmodels'):
        try:
            run.append((f'{pkg}_version', importlib.metadata.version(pkg)))
        except importlib.metadata.PackageNotFoundError:
            pass
    params = ([{'module': __name__, 'parameter': k, 'value': str(v)}
               for k, v in _module_settings().items()]
              + asc._module_parameters(pc))
    with pd.ExcelWriter(os.path.join(snap, 'RunMetadata.xlsx'), engine='openpyxl') as w:
        pd.DataFrame(run, columns=['item', 'value']).to_excel(w, sheet_name='Run', index=False)
        pd.DataFrame(params).to_excel(w, sheet_name='Parameters', index=False)
    print(f'Run snapshot saved to {snap}')
    return snap


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_run_snapshot()
    print(f'Loading {FIELDS_EXCEL} ...')
    full = pd.read_excel(FIELDS_EXCEL, sheet_name='Full')
    fields = pd.read_excel(FIELDS_EXCEL, sheet_name='PlaceFields')

    if RUN_FIELD_SPEED:
        per_field = run_field_speed(full, fields)
    else:
        print(f'Loading PerField of {FIELD_RESULTS_XLSX} ...')
        per_field = pd.read_excel(FIELD_RESULTS_XLSX, sheet_name='PerField')

    df = build_stats_df(per_field, full)
    asc.GROUP_COLORS.setdefault(ALL_FIELDS_LABEL, asc.GROUP_COLORS[asc.ALL_CELLS_LABEL])
    print('\nTested fields by arena and session type:')
    print(pd.crosstab(df['arena_type'], df['session_type'].fillna('unrecognized'), margins=True))

    all_tables = {}
    # 1. Session types within each arena
    for arena in asc.ARENA_TYPES:
        grp = asc.session_grouping(arena)
        sub = df[(df['arena_type'] == arena) & df['session_type'].isin(grp.order)]
        for k, t in run_field_comparison(grp, sub, os.path.join(OUTPUT_DIR, 'SessionType', arena)).items():
            all_tables[f'Ses{arena}_{k}'] = t
    # 2. Arena types
    for k, t in run_field_comparison(asc.ARENA_GROUPING, df, os.path.join(OUTPUT_DIR, 'ArenaType')).items():
        all_tables[f'Arena_{k}'] = t
    # 3. Animal-controlled arena types
    all_tables.update(run_animal_controlled(df, os.path.join(OUTPUT_DIR, asc.ANIMAL_CONTROLLED_DIR)))
    # 4. Animal IDs
    animal_grp = asc.animal_id_grouping(df)
    for k, t in run_field_comparison(animal_grp, df, os.path.join(OUTPUT_DIR, ANIMAL_ID_DIR)).items():
        all_tables[f'Animal_{k}'] = t
    # 5. All fields pooled (no tests)
    pooled = asc.Grouping('all_cells', [ALL_FIELDS_LABEL], 'fields', tests=False)
    for k, t in run_field_comparison(pooled, df, os.path.join(OUTPUT_DIR, ALL_FIELDS_DIR)).items():
        if 'Omnibus' not in k and 'PostHoc' not in k:
            all_tables[f'All_{k}'] = t

    with pd.ExcelWriter(STATS_XLSX, engine='openpyxl') as w:
        for sheet, tbl in all_tables.items():
            tbl.to_excel(w, sheet_name=sheet[:31], index=False)
    parts = []
    for sheet, tbl in all_tables.items():
        if tbl is not None and len(tbl):
            t = tbl.copy()
            t.insert(0, 'table', sheet)
            parts.append(t)
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(STATS_CSV, index=False, encoding='utf-8-sig')
    print(f'\nStatistics saved to {STATS_XLSX} and {STATS_CSV}')
    print(f'Done. Plots under {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
