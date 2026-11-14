# -*- coding: utf-8 -*-
"""
Temporal stability of theta-modulated vs non-theta-modulated place cells,
compared within every condition of ArenaSessionType_StatsComparison_v20.py.

Nothing is recomputed from .ntt / tracking files: every value is read from
workbooks that already exist.

Inputs
  TS_XLSX        TemporalStability.xlsx written by ArenaSessionType_StatsComparison_v20
                 (OUTPUT_DIR of that run), sheets 'PerCell' and 'PerField'.
  THETAMOD_XLSX  ThetaMod_Pure_v3.xlsx, sheet 'Units' (one row per unit and session).

Theta-modulation label (ThetaMod_Pure_v3, 'ThetaModulated' = Rayleigh+ & TMI+)
  TM     ThetaModulated == True
  nonTM  ThetaModulated == False AND the unit was tested (>= 50 theta-epoch spikes,
         i.e. ThetaModClass != 'too few spikes'); with TESTED_ONLY = False the
         untested units are nonTM too.
  Place cells with no row in the Units sheet (e.g. Fa5834, no LFP analysis) and
  untested units are left out. Cells are matched by the session path from the
  animal folder onwards + the .ntt file name.

Metrics (one value per cell, within its place fields)
  occ_wmean_pct_across_bins     bin-wise: mean over the cell's field bins of
                                % visits with >= 1 spike, each bin weighted by its
                                dwell (PerCell, as computed by v20)
  lap_occ_wmean_across_fields   lap-wise: mean over the cell's fields of
                                lap_stability_score (fraction of passes with
                                >= MIN_SPIKES_PER_PASS spikes), each field weighted
                                by its dwell occupancy_s_field_bins (from PerField;
                                fields with no pass are left out)

Conditions (one folder each, same groups as v20)
  SessionType\\<arena>\\        level 1: session types within each arena
  ArenaType\\                   level 2: Circle / Linear / Open
  AnimalControlledArenaType\\   level 3: ANIMAL_CONTROLLED_COMPARISONS side by side
  AnimalID\\                    level 4: per animal
  AllCells\\                    level 5: every cell pooled

Test, within each group of a condition: two-sided Mann-Whitney U, TM vs nonTM,
Holm-Bonferroni over the groups of that plot (per metric). Rows (one unit in one
session) are treated as independent; 'repeats' in the tables says how many cells
appear in several sessions. Effect size: rank-biserial r = 2 U_TM / (n_TM n_nonTM) - 1
(> 0: TM cells higher).

Outputs (OUTPUT_DIR)
  <condition folder>\\TS_ThetaMod_<metric>.png and TS_ThetaMod_Stats.xlsx
  TS_ThetaMod_AllStats.xlsx  (sheets Cells, Desc, Tests, Settings) + .csv of Tests
  code\\                      copy of this script
"""

import os
import shutil
import sys
import tempfile
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import ArenaSessionType_StatsComparison_v20 as asc   # groups, colours, plot style
from ArenaSessionType_StatsComparison_v20 import ccs, plu, plt

# ── Parameters ──────────────────────────────────────────────────────────────

RESULTS_DIR = (r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt'
               r'\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final\Results')
TS_XLSX       = os.path.join(RESULTS_DIR, 'PC_Char', 'v4', 'TemporalStability.xlsx')
THETAMOD_XLSX = os.path.join(RESULTS_DIR, 'ThetaMod', 'ThetaMod_Pure_v3', 'ThetaMod_Pure_v3.xlsx')
OUTPUT_DIR    = os.path.join(RESULTS_DIR, 'TS_ThetaMod', 'v1')

# True: nonTM = tested units only (ThetaModClass != 'too few spikes').
TESTED_ONLY = True
UNTESTED_CLASS = 'too few spikes'

ALPHA = asc.ALPHA

METRICS = {
    'occ_wmean_pct_across_bins':
        'Bin-wise temporal stability: occupancy-weighted mean over field bins '
        '(% visits with >= 1 spike)',
    'lap_occ_wmean_across_fields':
        f'Lap-wise temporal stability: occupancy-weighted mean over fields '
        f'(fraction of passes with >= {asc.MIN_SPIKES_PER_PASS} spikes)',
}

CLASSES = ['TM', 'nonTM']
CLASS_LABEL = {'TM': 'Theta-modulated', 'nonTM': 'Not theta-modulated'}
CLASS_SHORT = {'TM': 'TM', 'nonTM': 'non-TM'}
NON_TM_HATCH = '////'
PAIR_OFFSET = 0.2      # x offset of the TM / nonTM boxes from the group position
BOX_W = 0.34
GROUP_W = 1.5          # figure width per group (inches)

STATS_NAME = 'TS_ThetaMod_Stats.xlsx'


# ── Data ────────────────────────────────────────────────────────────────────

def lap_occ_weighted(per_field: pd.DataFrame) -> pd.DataFrame:
    """Per cell: lap_stability_score of its fields, weighted by field dwell."""
    f = per_field.copy()
    f['lap_stability_score'] = pd.to_numeric(f['lap_stability_score'], errors='coerce')
    f['occupancy_s_field_bins'] = pd.to_numeric(f['occupancy_s_field_bins'], errors='coerce')
    f = f[f['lap_stability_score'].notna() & (f['occupancy_s_field_bins'] > 0)]
    f['_w'] = f['lap_stability_score'] * f['occupancy_s_field_bins']
    g = (f.groupby(['session', 'unit'])
          .agg(_wsum=('_w', 'sum'), _occ=('occupancy_s_field_bins', 'sum'),
               n_fields_lap=('field_number', 'count'), n_passes_fields=('n_passes', 'sum'))
          .reset_index())
    g['lap_occ_wmean_across_fields'] = g['_wsum'] / g['_occ']
    return g.drop(columns=['_wsum', '_occ'])


def load_cells() -> pd.DataFrame:
    """PerCell temporal stability + lap-wise weighted value + theta class, tagged with
    arena / session type / animal / cell_id like v20. Only TM / nonTM rows are kept."""
    print(f'Loading {TS_XLSX}')
    per_cell = pd.read_excel(TS_XLSX, sheet_name='PerCell')
    per_field = pd.read_excel(TS_XLSX, sheet_name='PerField')
    df = per_cell.merge(lap_occ_weighted(per_field), on=['session', 'unit'], how='left')

    print(f'Loading {THETAMOD_XLSX}')
    tm = pd.read_excel(THETAMOD_XLSX, sheet_name='Units')
    tm['_skey'] = tm['FolderPath'].map(asc._session_key)
    tm['_ukey'] = tm['ntt_file'].map(asc._unit_key)
    dup = tm.duplicated(['_skey', '_ukey'])
    if dup.any():
        print(f'WARNING: {int(dup.sum())} duplicate unit row(s) in the Units sheet; first kept.')
        tm = tm[~dup]
    tm = tm[['_skey', '_ukey', 'ThetaModulated', 'ThetaModClass', 'TMI', 'TMI_shuffle_p',
             'Rayleigh_p', 'MRL', 'n_spikes_ACGtheta']]

    df['_skey'] = df['session'].map(asc._session_key)
    df['_ukey'] = df['unit'].map(asc._unit_key)
    df = df.merge(tm, on=['_skey', '_ukey'], how='left', indicator=True)

    tmb = df['ThetaModulated'].map(asc._to_tri_bool)
    untested = df['ThetaModClass'].astype(str).str.strip().str.lower() == UNTESTED_CLASS
    df['theta_class'] = None
    df.loc[tmb == True, 'theta_class'] = 'TM'                               # noqa: E712
    non = (tmb == False)                                                    # noqa: E712
    if TESTED_ONLY:
        non &= ~untested
    df.loc[non, 'theta_class'] = 'nonTM'

    df = asc._tag_groups(df.drop(columns=['arena_type']), 'place cell(s) (temporal stability)')
    no_match = df['_merge'] == 'left_only'
    print(f'\n{len(df)} place cell row(s) in PerCell:')
    print(f'    {int(no_match.sum())} not in the ThetaMod Units sheet, by animal: '
          f'{df.loc[no_match, "animal_id"].value_counts().to_dict()}')
    print(f'    {int((~no_match & untested).sum())} untested ({UNTESTED_CLASS})'
          f'{" -> left out" if TESTED_ONLY else " -> nonTM"}')
    df = df[df['theta_class'].notna()].drop(columns=['_merge', '_skey', '_ukey'])
    print(f'    kept: {int((df["theta_class"] == "TM").sum())} TM, '
          f'{int((df["theta_class"] == "nonTM").sum())} nonTM')
    print(pd.crosstab([df['animal_id'], df['arena_type']], df['theta_class'], margins=True))
    for m in METRICS:
        n_nan = int(df[m].isna().sum())
        if n_nan:
            print(f'    {n_nan} kept row(s) have no {m} (no field / no pass); left out of it.')
    return df


# ── Statistics ──────────────────────────────────────────────────────────────

def _desc(v: np.ndarray) -> dict:
    if not len(v):
        return dict(n=0, mean=np.nan, sd=np.nan, sem=np.nan, median=np.nan, q1=np.nan, q3=np.nan)
    return dict(n=len(v), mean=np.mean(v), sd=np.std(v, ddof=1) if len(v) > 1 else np.nan,
                sem=stats.sem(v) if len(v) > 1 else np.nan, median=np.median(v),
                q1=np.percentile(v, 25), q3=np.percentile(v, 75))


def compare_tm(df: pd.DataFrame, metric: str, grp: asc.Grouping) -> tuple:
    """Per group of grp: TM vs nonTM descriptives and Mann-Whitney U; Holm over the
    groups. Returns (desc_df, tests_df, values {group: {class: array}})."""
    desc, tests, values = [], [], {}
    for g in grp.order:
        sub = df[df[grp.col] == g].copy()
        sub[metric] = pd.to_numeric(sub[metric], errors='coerce')
        sub = sub.dropna(subset=[metric])
        if not len(sub):
            continue
        v = {c: sub.loc[sub['theta_class'] == c, metric].to_numpy(float) for c in CLASSES}
        values[g] = v
        for c in CLASSES:
            desc.append({'metric': metric, grp.col: g, 'theta_class': c,
                         'n_cells': sub.loc[sub['theta_class'] == c, 'cell_id'].nunique(),
                         **_desc(v[c])})
        a, b = v['TM'], v['nonTM']
        row = {'metric': metric, grp.col: g, 'n_TM': len(a), 'n_nonTM': len(b),
               'median_TM': np.median(a) if len(a) else np.nan,
               'median_nonTM': np.median(b) if len(b) else np.nan,
               'repeats': ccs.repetition_summary(sub, 'theta_class')}
        if len(a) >= 2 and len(b) >= 2 and np.ptp(np.r_[a, b]) > 0:
            res = stats.mannwhitneyu(a, b, alternative='two-sided')
            row.update(test='Mann-Whitney U (two-sided)', U=float(res.statistic),
                       p_raw=float(res.pvalue),
                       rank_biserial_r=2.0 * res.statistic / (len(a) * len(b)) - 1.0)
        else:
            row.update(test='insufficient data (< 2 per class or constant)', U=np.nan,
                       p_raw=np.nan, rank_biserial_r=np.nan)
        tests.append(row)
    tests_df = pd.DataFrame(tests)
    if len(tests_df):
        tests_df['p_holm'] = ccs.holm(tests_df['p_raw'].to_numpy(float))
        tests_df['n_tests_holm'] = int(tests_df['p_raw'].notna().sum())
        tests_df['significant'] = tests_df['p_holm'] < ALPHA
    return pd.DataFrame(desc), tests_df, values


# ── Plot ────────────────────────────────────────────────────────────────────

def _stats_lines(tests_df: pd.DataFrame, grp: asc.Grouping) -> list:
    lines = []
    for _, r in tests_df.iterrows():
        head = (f'{asc._label(r[grp.col])}: TM n={r["n_TM"]}, median {r["median_TM"]:.3g} | '
                f'non-TM n={r["n_nonTM"]}, median {r["median_nonTM"]:.3g}')
        if pd.notna(r['p_raw']):
            head += (f' | U = {r["U"]:.1f}, p = {asc._fmt_p(r["p_raw"])}, '
                     f'p(Holm) = {asc._fmt_p(r["p_holm"])} {asc._stars(r["p_holm"])}, '
                     f'r = {r["rank_biserial_r"]:+.2f}')
        else:
            head += ' | no test (< 2 cells per class)'
        lines.append(head)
    n_h = int(tests_df['p_raw'].notna().sum()) if len(tests_df) else 0
    lines.append(f'Mann-Whitney U, two-sided, TM vs non-TM within each group; Holm over the '
                 f'{n_h} test(s) of this plot; rows (unit x session) treated as independent; '
                 f'r = rank-biserial (> 0: TM higher).')
    return lines


def plot_tm(values: dict, tests_df: pd.DataFrame, metric: str, label: str,
            grp: asc.Grouping, out_path: str):
    present = list(values)
    if not present:
        return
    fig_w = len(present) * GROUP_W + asc.AXIS_PAD_W
    lines = asc._wrap_stats(_stats_lines(tests_df, grp), fig_w)
    fig, ax, ax_info = asc._new_fig((fig_w, 5.5), legend_rows=1, stats_lines=lines)

    rng = np.random.default_rng(0)
    tops = {}
    for i, g in enumerate(present):
        col = asc._color(g)
        for c, off in zip(CLASSES, (-PAIR_OFFSET, PAIR_OFFSET)):
            v = values[g][c]
            if not len(v):
                continue
            pos = i + off
            face = col['face'] if c == 'TM' else col['light']
            bp = ax.boxplot([v], positions=[pos], widths=BOX_W, showfliers=False,
                            patch_artist=True, zorder=2,
                            boxprops=dict(linewidth=asc.BOX_LW, color=asc.PAL_BLACK),
                            whiskerprops=dict(linewidth=asc.BOX_LW, color=asc.PAL_BLACK),
                            capprops=dict(linewidth=asc.BOX_LW, color=asc.PAL_BLACK),
                            medianprops=dict(linewidth=asc.MEDIAN_LW, color=asc.PAL_BLACK))
            bp['boxes'][0].set_facecolor(face)
            bp['boxes'][0].set_edgecolor(asc.PAL_BLACK)
            if c == 'nonTM':
                bp['boxes'][0].set_hatch(NON_TM_HATCH)
            ax.scatter(pos + rng.uniform(-0.07, 0.07, len(v)), v, s=16,
                       marker='o' if c == 'TM' else 's', facecolors=face,
                       edgecolors=asc.PAL_BLACK, linewidths=0.4, alpha=0.85, zorder=3)
        both = np.r_[values[g]['TM'], values[g]['nonTM']]
        tops[g] = both.max() if len(both) else np.nan

    # Significance line over each TM / non-TM pair with Holm p < ALPHA.
    y_lo, y_hi = ax.get_ylim()
    span = y_hi - y_lo
    top_needed = y_hi
    if len(tests_df):
        for i, g in enumerate(present):
            r = tests_df[tests_df[grp.col] == g]
            if not len(r) or not (r['p_holm'].iloc[0] < ALPHA):
                continue
            y = tops[g] + 0.06 * span
            ax.plot([i - PAIR_OFFSET, i + PAIR_OFFSET], [y, y], color=asc.PAL_BLACK,
                    linewidth=asc.BOX_LW, clip_on=False, zorder=5)
            ax.text(i, y + 0.03 * span, asc._stars(r['p_holm'].iloc[0]), ha='center',
                    va='center', fontsize=asc.FS_LABEL + 3, fontweight='bold', zorder=5)
            top_needed = max(top_needed, y + 0.1 * span)
    ax.set_ylim(y_lo, top_needed)

    ticks = []
    for g in present:
        ticks.append(f'{asc._label(g)}\nTM {len(values[g]["TM"])} | '
                     f'non {len(values[g]["nonTM"])}')
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(ticks, fontsize=asc.FS_TICK - 1)
    ax.set_xlim(-0.6, len(present) - 0.4)
    ax.set_ylabel(textwrap.fill(label, 32), fontsize=asc.FS_LABEL - 2, labelpad=8)
    asc._set_title(ax, asc._titled(f'{label}: theta-modulated vs not', grp))
    asc._style_axes(ax)

    handles = [Patch(facecolor='#BFBFBF', edgecolor=asc.PAL_BLACK, label=CLASS_LABEL['TM']),
               Patch(facecolor='#E6E6E6', edgecolor=asc.PAL_BLACK, hatch=NON_TM_HATCH,
                     label=CLASS_LABEL['nonTM'])]
    ax_info.legend(handles=handles, loc='upper center', ncol=2, fontsize=asc.FS_LEGEND)
    asc._draw_stats(ax_info, lines)
    fig.tight_layout()
    asc._save(fig, out_path)


# ── One condition ───────────────────────────────────────────────────────────

def run_condition(df: pd.DataFrame, grp: asc.Grouping, out_dir: str, level: str,
                  write_xlsx: bool = True) -> tuple:
    name = grp.title or grp.noun
    print(f'\n{"=" * 70}\n{level} | {name}: {[asc._label(g) for g in grp.order]}\n{"=" * 70}')
    if not len(df):
        print('No cells; skipped.')
        return pd.DataFrame(), pd.DataFrame()
    print(pd.crosstab(df[grp.col], df['theta_class']).reindex(
        [g for g in grp.order if g in set(df[grp.col])]))
    os.makedirs(out_dir, exist_ok=True)
    all_desc, all_tests = [], []
    for metric, label in METRICS.items():
        d, t, values = compare_tm(df, metric, grp)
        plot_tm(values, t, metric, label, grp, os.path.join(out_dir, f'TS_ThetaMod_{metric}.png'))
        for _, r in t.iterrows():
            flag = '*' if r.get('significant') else ' '
            print(f'  {flag} {metric:28s} {asc._label(r[grp.col]):10s} TM {r["n_TM"]:3d} '
                  f'med {r["median_TM"]:.3g} | non {r["n_nonTM"]:3d} med {r["median_nonTM"]:.3g} | '
                  f'p {asc._fmt_p(r["p_raw"])}, Holm {asc._fmt_p(r["p_holm"])}')
        all_desc.append(d)
        all_tests.append(t)
    desc = asc._concat_tables(all_desc).rename(columns={grp.col: 'group'})
    tests = asc._concat_tables(all_tests).rename(columns={grp.col: 'group'})
    for t in (desc, tests):
        if len(t):
            t.insert(0, 'grouping', grp.col)
            t.insert(0, 'condition', f'{level} | {name}')
    if write_xlsx:
        with pd.ExcelWriter(os.path.join(out_dir, STATS_NAME), engine='openpyxl') as w:
            desc.to_excel(w, sheet_name='Desc', index=False)
            tests.to_excel(w, sheet_name='Tests', index=False)
    print(f'Saved to {out_dir}')
    return desc, tests


def main():
    os.makedirs(os.path.join(OUTPUT_DIR, 'code'), exist_ok=True)
    shutil.copy2(os.path.abspath(__file__), os.path.join(OUTPUT_DIR, 'code'))
    df = load_cells()
    descs, tests = [], []

    def _add(res):
        descs.append(res[0])
        tests.append(res[1])

    # 1. Session types within each arena
    for arena in asc.ARENA_TYPES:
        grp = asc.session_grouping(arena)
        sub = df[(df['arena_type'] == arena) & df['session_type'].isin(grp.order)]
        _add(run_condition(sub, grp, os.path.join(OUTPUT_DIR, 'SessionType', arena),
                           'SessionType'))

    # 2. Arena types
    _add(run_condition(df, asc.ARENA_GROUPING, os.path.join(OUTPUT_DIR, 'ArenaType'), 'ArenaType'))

    # 3. Animal-controlled arena types, side by side with a shared y-range
    ac_dir = os.path.join(OUTPUT_DIR, asc.ANIMAL_CONTROLLED_DIR)
    ac_desc, ac_tests = [], []
    with tempfile.TemporaryDirectory() as tmp:
        dirs = [os.path.join(tmp, n) for n in asc.ANIMAL_CONTROLLED_COMPARISONS]
        with plu.shared_y_axes(dirs):
            for (name, (animals, arenas)), d in zip(asc.ANIMAL_CONTROLLED_COMPARISONS.items(), dirs):
                grp = asc.animal_controlled_grouping(animals, arenas)
                wanted = {a.lower() for a in animals}
                sub = df[df['animal'].astype(str).str.lower().isin(wanted)
                         & df['arena_type'].isin(grp.order)]
                missing = [a for a in animals if a.lower() not in
                           set(sub['animal'].astype(str).str.lower())]
                if missing:
                    print(f'NOTE {name}: no theta-tested cells of {missing}; '
                          f'the comparison holds only the other animal(s).')
                    grp = asc.animal_controlled_grouping(
                        [a for a in animals if a not in missing], arenas)
                    grp.title += f' (no theta data: {", ".join(missing)})'
                dd, tt = run_condition(sub, grp, d, 'AnimalControlled', write_xlsx=False)
                ac_desc.append(dd)
                ac_tests.append(tt)
        asc._combine_side_by_side(dirs, ac_dir)
    ac_desc, ac_tests = asc._concat_tables(ac_desc), asc._concat_tables(ac_tests)
    with pd.ExcelWriter(os.path.join(ac_dir, STATS_NAME), engine='openpyxl') as w:
        ac_desc.to_excel(w, sheet_name='Desc', index=False)
        ac_tests.to_excel(w, sheet_name='Tests', index=False)
    _add((ac_desc, ac_tests))

    # 4. Animal IDs
    _add(run_condition(df, asc.animal_id_grouping(df), os.path.join(OUTPUT_DIR, asc.ANIMAL_ID_DIR),
                       'AnimalID'))

    # 5. All cells pooled
    _add(run_condition(df, asc.ALL_CELLS_GROUPING, os.path.join(OUTPUT_DIR, asc.ALL_CELLS_DIR),
                       'AllCells'))

    desc_all, tests_all = asc._concat_tables(descs), asc._concat_tables(tests)
    cell_cols = ['session', 'unit', 'animal_id', 'arena_type', 'session_type', 'cell_id',
                 'theta_class', 'ThetaModClass', 'TMI', 'TMI_shuffle_p', 'Rayleigh_p', 'MRL',
                 'n_spikes_ACGtheta', 'n_fields', 'occ_wmean_pct_across_bins',
                 'lap_occ_wmean_across_fields', 'n_fields_lap', 'n_passes_fields',
                 'lap_stability_score', 'mean_lap_stability_across_fields']
    settings = pd.DataFrame([
        ('run_timestamp', datetime.now().isoformat(timespec='seconds')),
        ('script', os.path.abspath(__file__)), ('TS_XLSX', TS_XLSX),
        ('THETAMOD_XLSX', THETAMOD_XLSX), ('TESTED_ONLY', TESTED_ONLY),
        ('MIN_SPIKES_PER_PASS (lap stability, from v20)', asc.MIN_SPIKES_PER_PASS),
        ('ALPHA', ALPHA), ('LINK_CELLS_ACROSS_ARENAS (cell_id, from v20)',
                           asc.LINK_CELLS_ACROSS_ARENAS)], columns=['item', 'value'])
    xlsx = os.path.join(OUTPUT_DIR, 'TS_ThetaMod_AllStats.xlsx')
    with pd.ExcelWriter(xlsx, engine='openpyxl') as w:
        df[[c for c in cell_cols if c in df.columns]].to_excel(w, sheet_name='Cells', index=False)
        desc_all.to_excel(w, sheet_name='Desc', index=False)
        tests_all.to_excel(w, sheet_name='Tests', index=False)
        settings.to_excel(w, sheet_name='Settings', index=False)
    tests_all.to_csv(os.path.join(OUTPUT_DIR, 'TS_ThetaMod_AllTests.csv'), index=False,
                     encoding='utf-8-sig')
    print(f'\nAll statistics saved to {xlsx}\nDone. Output in {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
