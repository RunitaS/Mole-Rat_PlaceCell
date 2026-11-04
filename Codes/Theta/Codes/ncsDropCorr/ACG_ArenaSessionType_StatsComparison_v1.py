# -*- coding: utf-8 -*-
"""
Statistical comparison of the ACG theta-continuity results
(Figc_ACG_ScalingCorrected_v4_SessionSegregation.py, its epoch table
'acg_theta_continuity_epochs.csv') at the three levels of
ArenaSessionType_StatsComparison_v18.py:

  1. Session types WITHIN each arena type:
         Open   : Cntrl, Rotate, Zero
         Linear : 0, 90, 180, 270
         Circle : Stnd, NoRot, Rot
  2. Arena types as a whole (Circle / Linear / Open, session types pooled).
  3. Animal-controlled arena types (ANIMAL_CONTROLLED_COMPARISONS): each
     comparison uses only the animals recorded in all of its arenas
         Circle vs Open : Fa8477, Fa5834
         Linear vs Open : Fa1059, Fa23BD
     The comparisons are drawn side by side in one figure, each panel with its
     own statistics; they are not tested against each other.

Arena and session type are parsed from each epoch's session folder exactly as
in ArenaSessionType_StatsComparison_v18.py (extract_arena_type /
extract_session_type). Only epochs labelled 'immobile' or 'moving' are in the
epoch table (the ACG script drops the rest).

Metrics:

  1. ED_min rejection: % of epochs whose ED_min (normalised distance to the
     closest reference sinusoid) is above ACG_ED_MIN_THRESH ('skipped_edmin').
     Denominator: every epoch in the epoch table, i.e. epochs that passed the
     delta/theta and MAD artifact rejection.
  2. ED_min rejection in immobility vs mobility:
       - % of immobility epochs rejected and % of mobility epochs rejected
         (each normalised within its own movement state), compared across the
         groups and, within each group, immobility vs mobility;
       - split of the rejected epochs: % that are immobility epochs.
  3. Centroids, immobility vs mobility (accepted epochs only, ~skipped, as in
     compute_group_centroids of the ACG script):
       - matched-sinusoid frequency ('freq', Hz)
       - normalised autocorrelogram peak range ('peakrangenorm')
     One plot per metric. The unit is CENTROID_UNIT: 'recording' = one value
     per recording (.ncs file) and movement state, the mean of its accepted
     epochs (that recording's centroid); 'epoch' = every accepted epoch. Each
     plot shows, per group, the immobility and mobility distributions (boxes),
     every recording centroid (dots, 'recording' only), the centroid of each
     of the 4 animals (large markers) and the group centroid (diamond, mean of
     the plotted values).

Tests (the same as ArenaSessionType_StatsComparison_v18.py):
  - % outcomes, epoch level: chi-square test of independence on the
    groups x (rejected, kept) epoch counts, then pairwise chi-square tests
    (Yates-corrected for 2 x 2 tables), Holm-Bonferroni corrected within each
    measure. Epochs are treated as independent; 'model_note' flags expected
    counts < 5. Immobility vs mobility within a group: 2 x 2 chi-square
    (Yates), Holm-corrected across the groups.
  - continuous values (the per-recording % of metrics 1-2, and the frequency
    and peak range of metric 3): CONTINUOUS_TEST
      'KW'  : Kruskal-Wallis omnibus + pairwise Mann-Whitney U, Holm corrected;
              rows treated as independent.
      'LMM' : linear mixed model on ranks (METRIC_TRANSFORM),
                value ~ group + (1|animal) + (1|animal:day) + (1|channel)
              likelihood-ratio omnibus + pairwise Wald contrasts, Holm
              corrected (CellClusteredStats_Utils.compare_continuous_nested).
    Here a 'cell' of CellClusteredStats_Utils is one .ncs channel within one
    recording day (animal | day | channel file name), so the same channel
    recorded in several sessions of a day is a repeated measure.
    Across groups: within each movement state. Immobility vs mobility within
    a group: the same test with the movement state as the group, Holm across
    the groups.

Output (OUTPUT_DIR = <ROOT_DIR>\\Results\\ACG\\Stats):
    SessionType\\<arena>\\*.png          session types of <arena>
    ArenaType\\*.png                     arena types
    AnimalControlledArenaType\\*.png     the animal-controlled comparisons side by side
      EDmin_Rejected_Overall.png         metric 1
      EDmin_Rejected_ImmobileVsMobile.png  metric 2, per-movement-state rates
      EDmin_Rejected_Split.png           metric 2, split of the rejected epochs
      Centroid_Frequency.png             metric 3, frequency
      Centroid_PeakRange.png             metric 3, peak range
    ACG_SessionArena_Stats.xlsx          sheets '<prefix>_<table>', prefix
                                         'Ses<arena>', 'Arena' or 'AC<A><B>'
                                         (+ 'Recordings', 'Centroids')
    ACG_SessionArena_AllStats.csv        every table in long format
    RunSnapshots\\<YYYYmmdd_HHMMSS>\\     this script + RunMetadata.csv

INPUT_CSV is only read, not modified.
"""

import os
import re
import sys
import shutil
import platform
import subprocess
import textwrap
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# CellClusteredStats_Utils.py (shared with ArenaSessionType_StatsComparison_v18.py)
PLACECELL_CODES_DIR = r'C:\Runita\NMR\Mole-Rat_PlaceCell\Codes\PlaceCell\Codes'
if PLACECELL_CODES_DIR not in sys.path:
    sys.path.append(PLACECELL_CODES_DIR)
import CellClusteredStats_Utils as ccs

# ── Parameters ──────────────────────────────────────────────────────────────

ROOT_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final'

# Output folder of the ACG run (Figc_ACG_ScalingCorrected_v4_SessionSegregation.py)
ACG_RUN_DIR = os.path.join(ROOT_DIR, 'Results', 'ACG', 'ACG_theta_continuity_NoiseRemoval_EdgePad_NcsDropCorr')
INPUT_CSV   = os.path.join(ACG_RUN_DIR, 'acg_theta_continuity_epochs.csv')
ACG_META_CSV = os.path.join(ACG_RUN_DIR, 'meta', 'run_metadata.csv')

OUTPUT_DIR = os.path.join(ROOT_DIR, 'Results', 'ACG', 'Stats')
STATS_XLSX = os.path.join(OUTPUT_DIR, 'ACG_SessionArena_Stats.xlsx')
STATS_CSV  = os.path.join(OUTPUT_DIR, 'ACG_SessionArena_AllStats.csv')
RUN_SNAPSHOT_SUBDIR = 'RunSnapshots'

# ED_min threshold of the ACG run, used for labels only (the rejection itself is
# the 'skipped_edmin' column). Read from ACG_META_CSV when it is there.
ACG_ED_MIN_THRESH = 1.0

ARENA_TYPES = ['Circle', 'Linear', 'Open']
ARENA_SESSION_TYPES = {
    'Open':   ['Cntrl', 'Rotate', 'Zero'],
    'Linear': ['0', '90', '180', '270'],
    'Circle': ['Stnd', 'NoRot', 'Rot'],
}
ALPHA = 0.05

# Animal-controlled arena comparisons (level 3): {name: (animals, arenas)}.
ANIMAL_CONTROLLED_DIR = 'AnimalControlledArenaType'
ANIMAL_CONTROLLED_COMPARISONS = {
    'Circle_vs_Open': (['Fa8477', 'Fa5834'], ['Circle', 'Open']),
    'Linear_vs_Open': (['Fa1059', 'Fa23BD'], ['Linear', 'Open']),
}

# True: the same channel recorded in two arenas on the same day is one 'cell'
# (repeated measure) for the LMM; False: arenas are kept apart.
LINK_CELLS_ACROSS_ARENAS = True

# 'KW' : Kruskal-Wallis + pairwise Mann-Whitney U (Holm), rows independent
# 'LMM': nested linear mixed model (animal / day / channel), see the docstring
CONTINUOUS_TEST = 'KW'
METRIC_TRANSFORM = 'rank'     # LMM only: 'rank' or 'none'

# Unit of the centroid comparison (metric 3):
#   'recording': one value per .ncs file and movement state (mean of its accepted epochs)
#   'epoch'    : every accepted epoch
CENTROID_UNIT = 'recording'
# 'recording' only: recording centroids from fewer accepted epochs are left out (1 = keep all).
MIN_EPOCHS_PER_CENTROID = 1

MOVEMENTS = ['immobile', 'moving']
MOVEMENT_LABELS = {'immobile': 'Immobility', 'moving': 'Mobility'}

CENTROID_METRICS = {
    'freq':          ('Frequency', 'Matched-sinusoid frequency (Hz)', 'Centroid_Frequency'),
    'peakrangenorm': ('Peak range', 'Autocorr. peak range (norm.)', 'Centroid_PeakRange'),
}

# ── Groupings ───────────────────────────────────────────────────────────────


@dataclass
class Grouping:
    """The column holding the group label, the groups in plotting order, a
    plural noun for the stats text and a prefix for plot titles."""
    col: str
    order: list
    noun: str
    title: str = ''


ARENA_GROUPING = Grouping('arena_type', ARENA_TYPES, 'arena types', 'Arena types')
MOVEMENT_GROUPING = Grouping('movement', MOVEMENTS, 'movement states')


def session_grouping(arena: str) -> Grouping:
    return Grouping('session_type', ARENA_SESSION_TYPES[arena], 'session types',
                    f'{arena} arena')


def animal_controlled_grouping(animals: list, arenas: list) -> Grouping:
    return Grouping('arena_type', [a for a in ARENA_TYPES if a in arenas], 'arena types',
                    f'{" + ".join(animals)} only')


# ── Session-path parsing (as in ArenaSessionType_StatsComparison_v18.py) ────

def extract_arena_type(session_path: str):
    parts = re.split(r'[\\/]+', str(session_path))
    for part in parts:
        for arena in ARENA_TYPES:
            if part.strip().lower() == arena.lower():
                return arena
    return None


def extract_session_type(session_path: str, arena: str):
    if arena is None:
        return None
    label = re.split(r'[\\/]+', str(session_path).strip())[-1].strip().lower()
    if arena == 'Open':
        for t in ARENA_SESSION_TYPES['Open']:
            if t.lower() in label:
                return t
    elif arena == 'Linear':
        m = re.search(r'_(270|180|90|0)$', label)
        if m:
            return m.group(1)
    elif arena == 'Circle':
        # 'norot' must be tested before 'rot', which it contains.
        if 'norot' in label:
            return 'NoRot'
        if 'stnd' in label:
            return 'Stnd'
        if 'rot' in label:
            return 'Rot'
    return None


def _label(group) -> str:
    """Display label for plots: Linear session types are angles."""
    return f'{group}°' if str(group).isdigit() else str(group)


def _to_bool(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() == 'true'
    return False if pd.isna(x) else bool(x)


# ── Data ────────────────────────────────────────────────────────────────────

ID_COLS = ['session', 'unit', 'animal', 'rec_day', 'cell_id', 'arena_type', 'session_type']


def read_acg_settings():
    """ACG_ED_MIN_THRESH from the ACG run's metadata, if available."""
    global ACG_ED_MIN_THRESH
    if not os.path.isfile(ACG_META_CSV):
        print(f'No ACG run metadata at {ACG_META_CSV}; using ACG_ED_MIN_THRESH = {ACG_ED_MIN_THRESH:g}')
        return
    meta = pd.read_csv(ACG_META_CSV).set_index('parameter')['value']
    if 'ACG_ED_MIN_THRESH' in meta and meta['ACG_ED_MIN_THRESH'] != 'None':
        ACG_ED_MIN_THRESH = float(meta['ACG_ED_MIN_THRESH'])
    print(f'ACG run: {meta.get("script_name", "?")} ({meta.get("run_datetime", "?")}), '
          f'ED_min threshold {ACG_ED_MIN_THRESH:g}')


def load_epochs() -> pd.DataFrame:
    """Epoch table with arena / session type re-parsed from the session folder
    and the CellClusteredStats_Utils ids (animal, rec_day, cell_id) of each
    recording. Epochs with no recognizable arena are dropped."""
    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(f'{INPUT_CSV} not found -- run '
                                f'Figc_ACG_ScalingCorrected_v4_SessionSegregation.py first.')
    print(f'Loading {INPUT_CSV} ...')
    ep = pd.read_csv(INPUT_CSV, usecols=['file', 'session_folder', 'movement', 'skipped',
                                         'skipped_edmin', 'freq', 'peakrangenorm'])
    ep['skipped'] = ep['skipped'].map(_to_bool)
    ep['skipped_edmin'] = ep['skipped_edmin'].map(_to_bool)
    n_other = int((~ep['movement'].isin(MOVEMENTS)).sum())
    if n_other:
        print(f'NOTE: {n_other} epoch(s) with movement not in {MOVEMENTS}; dropped.')
    ep = ep[ep['movement'].isin(MOVEMENTS)].copy()

    # One row per recording (.ncs file): arena / session type / ids.
    files = ep.drop_duplicates('file')[['file', 'session_folder']].copy()
    files['session'] = files['session_folder'].astype(str)
    files['unit'] = files['file'].map(lambda f: os.path.basename(str(f)))
    files['arena_type'] = files['session'].map(extract_arena_type)
    no_arena = files['arena_type'].isna()
    if no_arena.any():
        print(f'WARNING: {int(no_arena.sum())} recording(s) with no recognizable arena '
              f'type ({ARENA_TYPES}); excluded:')
        for s in sorted(files.loc[no_arena, 'session'].unique()):
            print(f'    {s}')
    files = files[~no_arena].copy()
    files['session_type'] = [extract_session_type(s, a)
                             for s, a in zip(files['session'], files['arena_type'])]
    no_type = files['session_type'].isna()
    if no_type.any():
        print(f'WARNING: {int(no_type.sum())} recording(s) with no recognizable session type; '
              f'kept in the arena comparison, excluded from the session-type comparison:')
        for s in sorted(files.loc[no_type, 'session'].unique()):
            print(f'    {s}')
    files = ccs.add_cell_ids(files, include_arena=not LINK_CELLS_ACROSS_ARENAS)

    ep = ep.merge(files[['file'] + ID_COLS], on='file', how='inner')
    print(f'{len(ep)} epochs from {ep["file"].nunique()} recordings '
          f'({ep["session"].nunique()} sessions, {ep["animal"].nunique()} animals).')
    return ep


def recording_table(ep: pd.DataFrame) -> pd.DataFrame:
    """One row per recording (.ncs file): epoch and ED_min-rejected counts (all /
    immobility / mobility) and the PREVALENCE_MEASURES percentages ('pct_<key>',
    NaN where the denominator is 0)."""
    rej = ep['skipped_edmin']
    imm = ep['movement'].eq('immobile')
    mov = ep['movement'].eq('moving')
    flags = pd.DataFrame({'n_epochs': 1, 'n_edmin_rejected': rej,
                          'n_immobile': imm, 'n_immobile_rejected': rej & imm,
                          'n_moving': mov, 'n_moving_rejected': rej & mov},
                         index=ep.index).astype(int)
    counts = flags.groupby(ep['file']).sum()
    rec = ep.groupby('file')[ID_COLS].first().join(counts).reset_index()
    for key, (_label_, k_col, n_col) in PREVALENCE_MEASURES.items():
        rec[f'pct_{key}'] = 100 * rec[k_col] / rec[n_col].where(rec[n_col] > 0)
    return rec


def centroid_table(ep: pd.DataFrame) -> pd.DataFrame:
    """Accepted epochs (~skipped) as the CENTROID_UNIT rows: one per recording and
    movement state (mean freq / peakrangenorm, 'n_epochs') or one per epoch."""
    acc = ep[~ep['skipped']]
    if CENTROID_UNIT == 'epoch':
        return acc[['file', 'movement'] + list(CENTROID_METRICS) + ID_COLS].assign(n_epochs=1)
    if CENTROID_UNIT != 'recording':
        raise ValueError(f"CENTROID_UNIT must be 'recording' or 'epoch', not {CENTROID_UNIT!r}")
    g = acc.groupby(['file', 'movement'])
    cen = g[list(CENTROID_METRICS)].mean().join(g.size().rename('n_epochs')).reset_index()
    n_all = len(cen)
    cen = cen[cen['n_epochs'] >= MIN_EPOCHS_PER_CENTROID]
    if len(cen) < n_all:
        print(f'NOTE: {n_all - len(cen)} recording centroid(s) from < {MIN_EPOCHS_PER_CENTROID} '
              f'accepted epochs left out.')
    return cen.merge(acc.groupby('file')[ID_COLS].first().reset_index(), on='file')


# ── Statistics (as in ArenaSessionType_StatsComparison_v18.py) ──────────────

def _finish(label: str, groups_str: str, n_groups: int, omni, posthoc_rows):
    """(omnibus dict, posthoc DataFrame) in the format of compare_categorical /
    compare_groups of ArenaSessionType_StatsComparison_v18.py."""
    omnibus = {'metric': label, 'n_groups': n_groups, 'groups': groups_str}
    if omni is None:
        omnibus.update({'test': 'insufficient data', 'statistic': np.nan,
                        'p_value': np.nan, 'dof': np.nan, 'significant': False})
        return omnibus, pd.DataFrame()
    omnibus.update(omni)
    omnibus['significant'] = bool(pd.notna(omni['p_value']) and omni['p_value'] < ALPHA)
    return omnibus, pd.DataFrame([{'metric': label, **r} for r in posthoc_rows])


def _chi_square_tests(table: list, present: list, summary: str):
    """Chi-square test of independence on the groups x categories count table,
    then pairwise chi-square tests (Yates-corrected for 2 x 2 tables),
    Holm-Bonferroni corrected. Categories with no counts are dropped. Returns
    (omnibus_dict, posthoc_rows), or (None, []) if < 2 groups or categories."""
    arr = np.array(table, dtype=float)
    arr = arr[:, arr.sum(axis=0) > 0] if arr.size else arr
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None, []

    chi2, p, dof, expected = stats.chi2_contingency(arr)
    note = f'Chi-square (epochs treated as independent): {summary}'
    n_low = int((expected < 5).sum())
    if n_low:
        note += (f'; {n_low} of {expected.size} expected counts < 5, '
                 f'chi-square approximation may be unreliable')
    omni = {'test': 'Chi-square test of independence', 'stat_name': 'χ²',
            'statistic': chi2, 'dof': dof, 'p_value': p, 'model_note': note}

    rows = []
    for i, j in combinations(range(len(present)), 2):
        sub_t = arr[[i, j]]
        sub_t = sub_t[:, sub_t.sum(axis=0) > 0]
        if sub_t.shape[1] < 2:
            c2, pp, d = np.nan, np.nan, np.nan
        else:
            c2, pp, d, _e = stats.chi2_contingency(sub_t)
        rows.append({'comparison': f'{present[i]} vs {present[j]}',
                     'test': 'Pairwise chi-square' + (' (Yates)' if sub_t.shape == (2, 2) else ''),
                     'stat_name': 'χ²', 'statistic': c2, 'dof': d, 'p_raw': pp})
    return omni, ccs._finish_posthoc(rows)


def compare_groups(df: pd.DataFrame, metric_col: str, metric_label: str, grp: Grouping):
    """Descriptive stats, omnibus test and post-hoc pairwise tests of one
    continuous value across the groups of `grp` (CONTINUOUS_TEST, as
    compare_groups of ArenaSessionType_StatsComparison_v18.py). Returns
    (desc_df, omnibus_dict, posthoc_df)."""
    groups, n_cells = {}, {}
    for g in grp.order:
        sub = df.loc[df[grp.col] == g, [metric_col, 'cell_id']].copy()
        sub[metric_col] = pd.to_numeric(sub[metric_col], errors='coerce')
        sub = sub.dropna(subset=[metric_col])
        if len(sub):
            groups[g] = sub[metric_col].to_numpy(dtype=float)
            n_cells[g] = sub['cell_id'].nunique()

    desc_df = pd.DataFrame([{
        'metric': metric_label, 'group': g, 'n': len(v), 'n_channels': n_cells[g],
        'mean': np.mean(v), 'median': np.median(v),
        'q1': np.percentile(v, 25), 'q3': np.percentile(v, 75),
        'std': np.std(v, ddof=1) if len(v) > 1 else np.nan,
        'sem': stats.sem(v) if len(v) > 1 else np.nan,
        'min': np.min(v), 'max': np.max(v)} for g, v in groups.items()])
    groups_str = ', '.join(f'{g} (n={len(v)})' for g, v in groups.items())

    if CONTINUOUS_TEST == 'KW':
        if len(groups) < 2 or any(len(v) < 2 for v in groups.values()):
            omni, rows = None, []
        else:
            sub = df[df[grp.col].isin(list(groups))].copy()
            sub[metric_col] = pd.to_numeric(sub[metric_col], errors='coerce')
            summary = ccs.repetition_summary(sub.dropna(subset=[metric_col]), grp.col)
            summary = summary.replace(' cells (', ' channel-days (')
            try:
                omni, rows = ccs._classic_continuous(
                    groups, f'Kruskal-Wallis + Mann-Whitney U (rows treated as independent): {summary}')
            except ValueError:  # e.g. all values identical
                omni, rows = None, []
    elif CONTINUOUS_TEST == 'LMM':
        omni, rows = ccs.compare_continuous_nested(df, grp.col, grp.order, metric_col,
                                                   transform=METRIC_TRANSFORM)
    else:
        raise ValueError(f"CONTINUOUS_TEST must be 'LMM' or 'KW', not {CONTINUOUS_TEST!r}")
    omnibus, posthoc_df = _finish(metric_label, groups_str, len(groups), omni, rows)
    return desc_df, omnibus, posthoc_df


# (label, rejected-count column, denominator column)
PREVALENCE_MEASURES = {
    'overall':  (f'% of all epochs with ED_min > {ACG_ED_MIN_THRESH:g}',
                 'n_edmin_rejected', 'n_epochs'),
    'immobile': (f'% of immobility epochs with ED_min > {ACG_ED_MIN_THRESH:g}',
                 'n_immobile_rejected', 'n_immobile'),
    'moving':   (f'% of mobility epochs with ED_min > {ACG_ED_MIN_THRESH:g}',
                 'n_moving_rejected', 'n_moving'),
    'rejected_in_immobility': ('% of ED_min-rejected epochs that are immobility epochs',
                               'n_immobile_rejected', 'n_edmin_rejected'),
}


def _set_prevalence_labels():
    """PREVALENCE_MEASURES labels with the ED_min threshold read from the ACG run."""
    for key, word in (('overall', 'all'), ('immobile', 'immobility'), ('moving', 'mobility')):
        _l, k_col, n_col = PREVALENCE_MEASURES[key]
        PREVALENCE_MEASURES[key] = (f'% of {word} epochs with ED_min > {ACG_ED_MIN_THRESH:g}',
                                    k_col, n_col)


def compare_prevalence(rec: pd.DataFrame, key: str, grp: Grouping) -> dict:
    """One PREVALENCE_MEASURES measure across the groups of `grp`: chi-square
    on the pooled epoch counts ('epoch') and CONTINUOUS_TEST on the
    per-recording % ('recording'). 'desc' has both."""
    label, k_col, n_col = PREVALENCE_MEASURES[key]
    sub = rec[rec[grp.col].isin(grp.order) & (rec[n_col] > 0)]
    tot = sub.groupby(grp.col)[[k_col, n_col]].sum()
    present = [g for g in grp.order if g in tot.index]
    table = [[int(tot.loc[g, k_col]), int(tot.loc[g, n_col] - tot.loc[g, k_col])] for g in present]
    n_rec = sub.groupby(grp.col)['file'].nunique()

    summary = f'{int(sub[n_col].sum())} epochs from {sub["file"].nunique()} recordings'
    omni, rows = _chi_square_tests(table, present, summary) if len(present) >= 2 else (None, [])
    epoch = _finish(label, ', '.join(f'{g} (n={k + f})' for g, (k, f) in zip(present, table)),
                    len(present), omni, rows)

    rec_desc, rec_omni, rec_post = compare_groups(sub, f'pct_{key}', label, grp)
    desc = pd.DataFrame([{'metric': label, 'group': g, 'n_recordings': int(n_rec.get(g, 0)),
                          'n_rejected': k, 'n_epochs': k + f, 'pct_epochs': 100 * k / (k + f)}
                         for g, (k, f) in zip(present, table)])
    if len(desc) and len(rec_desc):
        desc = desc.merge(rec_desc.drop(columns=['metric', 'n']).add_prefix('rec_pct_')
                          .rename(columns={'rec_pct_group': 'group'}), on='group', how='left')
    return {'desc': desc, 'epoch': epoch, 'recording': (rec_omni, rec_post)}


def compare_prevalence_within(rec: pd.DataFrame, grp: Grouping) -> pd.DataFrame:
    """Within each group of `grp`: ED_min rejection rate of immobility vs
    mobility epochs, 2 x 2 chi-square on the epoch counts ('epoch_*') and
    CONTINUOUS_TEST on the per-recording % ('rec_*'); Holm across the groups."""
    label = f'ED_min > {ACG_ED_MIN_THRESH:g} rate, immobility vs mobility'
    rows = []
    for g in grp.order:
        s = rec[rec[grp.col] == g]
        if s.empty:
            continue
        ki, ni = int(s['n_immobile_rejected'].sum()), int(s['n_immobile'].sum())
        km, nm = int(s['n_moving_rejected'].sum()), int(s['n_moving'].sum())
        omni = None
        if ni and nm:
            omni, _ = _chi_square_tests([[ki, ni - ki], [km, nm - km]], MOVEMENTS,
                                        f'{ni + nm} epochs from {s["file"].nunique()} recordings')
        long = pd.concat([s.assign(movement=m, pct=s[f'pct_{m}']) for m in MOVEMENTS],
                         ignore_index=True)
        _d, rec_omni, _p = compare_groups(long, 'pct', label, MOVEMENT_GROUPING)
        omni = omni or {}
        rows.append({'metric': label, 'group': g,
                     'n_immobile_rejected': ki, 'n_immobile': ni,
                     'pct_immobile': 100 * ki / ni if ni else np.nan,
                     'n_moving_rejected': km, 'n_moving': nm,
                     'pct_moving': 100 * km / nm if nm else np.nan,
                     'epoch_test': omni.get('test', 'insufficient data'),
                     'epoch_statistic': omni.get('statistic', np.nan),
                     'epoch_dof': omni.get('dof', np.nan),
                     'epoch_p_raw': omni.get('p_value', np.nan),
                     'epoch_model_note': omni.get('model_note', ''),
                     'rec_test': rec_omni['test'], 'rec_stat_name': rec_omni.get('stat_name', ''),
                     'rec_statistic': rec_omni['statistic'], 'rec_p_raw': rec_omni['p_value'],
                     'rec_model_note': rec_omni.get('model_note', '')})
    out = pd.DataFrame(rows)
    if len(out):
        for lvl in ('epoch', 'rec'):
            out[f'{lvl}_p_holm'] = ccs.holm(out[f'{lvl}_p_raw'].to_numpy(float))
            out[f'{lvl}_significant'] = out[f'{lvl}_p_holm'] < ALPHA
    return out


def compare_centroid(cen: pd.DataFrame, metric: str, grp: Grouping) -> dict:
    """Metric 3 for one centroid metric: across the groups of `grp` within each
    movement state, and immobility vs mobility within each group (Holm across
    the groups). Also the per-animal centroids."""
    name = CENTROID_METRICS[metric][0]
    sub = cen[cen[grp.col].isin(grp.order)]
    across, desc_parts = {}, []
    for m in MOVEMENTS:
        d, o, p = compare_groups(sub[sub['movement'] == m], metric,
                                 f'{name}, {MOVEMENT_LABELS[m].lower()}', grp)
        if len(d):
            n_ep = sub[sub['movement'] == m].groupby(grp.col)['n_epochs'].sum()
            d.insert(2, 'movement', m)
            d.insert(4, 'n_epochs', d['group'].map(n_ep).to_numpy())
            desc_parts.append(d)
        across[m] = (o, p)

    within = []
    for g in grp.order:
        s = sub[sub[grp.col] == g]
        if s.empty:
            continue
        _d, o, _p = compare_groups(s, metric, f'{name}, immobility vs mobility', MOVEMENT_GROUPING)
        within.append({'metric': o['metric'], 'group': g, 'test': o['test'],
                       'stat_name': o.get('stat_name', ''), 'statistic': o['statistic'],
                       'dof': o.get('dof', np.nan), 'p_raw': o['p_value'],
                       'model_note': o.get('model_note', '')})
    within = pd.DataFrame(within)
    if len(within):
        within['p_holm'] = ccs.holm(within['p_raw'].to_numpy(float))
        within['significant'] = within['p_holm'] < ALPHA

    animal = (sub.groupby([grp.col, 'movement', 'animal'])[metric]
                 .agg(['mean', 'median', 'count']).reset_index()
                 .rename(columns={grp.col: 'group', 'count': 'n'}))
    animal.insert(0, 'metric', name)
    return {'data': sub, 'desc': pd.concat(desc_parts, ignore_index=True) if desc_parts
            else pd.DataFrame(), 'across': across, 'within': within, 'animal': animal}


def analyse(grp: Grouping, rec: pd.DataFrame, cen: pd.DataFrame) -> dict:
    """Every test of one comparison (rec / cen already restricted to its rows)."""
    name = grp.title or grp.noun.capitalize()
    print(f'\n{"=" * 70}\n{name} -- comparing {grp.noun}: {[_label(g) for g in grp.order]}\n{"=" * 70}')
    print('Recordings / epochs per group:')
    print(rec.groupby(grp.col)[['file', 'n_epochs']].agg({'file': 'nunique', 'n_epochs': 'sum'})
          .reindex(grp.order).fillna(0).astype(int).to_string())
    return {'grp': grp, 'rec': rec,
            'prev': {key: compare_prevalence(rec, key, grp) for key in PREVALENCE_MEASURES},
            'prev_within': compare_prevalence_within(rec, grp),
            'cent': {m: compare_centroid(cen, m, grp) for m in CENTROID_METRICS}}


def result_tables(res: dict) -> dict:
    """{table name: DataFrame} of one analyse() result."""
    def _rows(omnibus, posthoc, **tags):
        """Omnibus row ('comparison' = 'omnibus', p in 'p_raw') + post-hoc rows."""
        out = [pd.DataFrame([{**tags, 'comparison': 'omnibus', **omnibus,
                              'p_raw': omnibus.get('p_value')}]).drop(columns=['p_value'])]
        if posthoc is not None and len(posthoc):
            out.append(posthoc.assign(**tags))
        return out

    tests = [t for key, pr in res['prev'].items()
             for lvl in ('epoch', 'recording') for t in _rows(*pr[lvl], measure=key, level=lvl)]
    cent_tests = [t for metric, c in res['cent'].items() for m, (o, p) in c['across'].items()
                  for t in _rows(o, p, measure=metric, level=CENTROID_UNIT, movement=m)]
    within_cent = [c['within'].assign(measure=m) for m, c in res['cent'].items() if len(c['within'])]
    return {
        'Prev_Desc':    _concat([pr['desc'].assign(measure=k) for k, pr in res['prev'].items()]),
        'Prev_Tests':   _concat(tests),
        'Prev_Within':  res['prev_within'],
        'Cent_Desc':    _concat([c['desc'] for c in res['cent'].values()]),
        'Cent_Animal':  _concat([c['animal'] for c in res['cent'].values()]),
        'Cent_Tests':   _concat(cent_tests),
        'Cent_Within':  _concat(within_cent),
    }


def _concat(tables: list) -> pd.DataFrame:
    tables = [t for t in tables if t is not None and len(t)]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


# ── Figure style (as ArenaSessionType_StatsComparison_v18.py) ───────────────

PAL_BLACK = '#000000'
FS_LABEL, FS_TITLE, FS_TICK, FS_LEGEND, FS_STATS = 14, 15, 10, 10, 9
STATS_LINE_IN = 0.2    # info-panel height per line of stats text (inches)
LEGEND_ROW_IN = 0.3    # info-panel height per row of legend entries (inches)
STATS_CHARS_PER_IN = 15
PLOT_H = 5.0
FIG_DPI = 500
AXIS_LW, BOX_LW, MEDIAN_LW = 0.8, 1.0, 1.5
BAR_EDGE, BAR_EDGE_LW, BAR_WIDTH = 'white', 0.6, 0.6
GROUP_W, PAIR_GROUP_W, AXIS_PAD_W = 1.1, 1.6, 1.4

plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
    'svg.fonttype':       'none',
    'axes.labelweight':   'bold',
    'axes.titleweight':   'bold',
    'axes.linewidth':     AXIS_LW,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'xtick.major.width':  AXIS_LW,
    'ytick.major.width':  AXIS_LW,
    'axes.facecolor':     'white',
    'figure.facecolor':   'white',
    'savefig.facecolor':  'white',
    'legend.frameon':     False,
})

# Arena / session-type fills of ArenaSessionType_StatsComparison_v18.py.
GROUP_FACE = {
    'Open': '#7FB3E5', 'Linear': '#7FFAFA', 'Circle': '#FA7FFA',
    'Cntrl': '#7FB3E5', 'Rotate': '#0066CC', 'Zero': '#BFBFBF',
    '0': '#008080', '90': '#00B3B3', '180': '#33D6D6', '270': '#7FFAFA',
    'Stnd': '#FDC4FD', 'NoRot': '#FA7FFA', 'Rot': '#C000C0',
}
# Movement states (magma(0.5) immobile, viridis(0.5) moving, the colormaps of the
# ACG script's scatter) and their centroid diamonds (_draw_freq_peakrange_scatter:
# pink immobile, viridis(0.8) moving).
MOVEMENT_FACE = {'immobile': matplotlib.colormaps['magma'](0.5),
                 'moving':   matplotlib.colormaps['viridis'](0.5)}
CENTROID_FACE = {'immobile': '#FF1493', 'moving': matplotlib.colormaps['viridis'](0.8)}
# Per-animal colours of the ACG script (Pastel1) + one marker per animal.
ANIMAL_COLORS = {'Fa8477': '#FBB4AE', 'Fa1059': '#B3CDE3', 'Fa23BD': '#CCEBC5', 'Fa5834': '#DECBE4'}
ANIMAL_MARKERS = {'Fa8477': 'o', 'Fa1059': 's', 'Fa23BD': '^', 'Fa5834': 'v'}


def _animal_key(label):
    label = str(label).upper()
    return next((a for a in ANIMAL_COLORS if a[2:].upper() in label), None)


def _animal_color(label):
    return ANIMAL_COLORS.get(_animal_key(label), '#888888')


def _animal_marker(label):
    return ANIMAL_MARKERS.get(_animal_key(label), 'P')


def _fmt_p(p) -> str:
    if p is None or pd.isna(p):
        return 'n/a'
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def _stars(p) -> str:
    if p is None or pd.isna(p):
        return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < ALPHA else 'n.s.'


def _pair_label(cmp: str) -> str:
    return ' vs '.join(_label(g) for g in str(cmp).split(' vs '))


def _test_lines(omnibus: dict, posthoc_df: pd.DataFrame) -> list:
    """Omnibus test and Holm-corrected post-hoc tests as text lines."""
    if omnibus.get('test') == 'insufficient data':
        return ['Insufficient data for a statistical test']
    dof = omnibus.get('dof')
    dof_str = f', dof = {int(dof)}' if dof is not None and pd.notna(dof) else ''
    p = omnibus.get('p_value')
    lines = [f'{omnibus["test"]}: {omnibus.get("stat_name", "stat")} = {omnibus["statistic"]:.3f}'
             f'{dof_str}, p = {_fmt_p(p)} {_stars(p)}']
    if posthoc_df is not None and len(posthoc_df):
        lines.append(f'Post-hoc ({posthoc_df["test"].iloc[0]}, Holm-Bonferroni corrected):')
        for _, r in posthoc_df.iterrows():
            fmt = '.1f' if r['stat_name'] == 'U' else '.3f'
            lines.append(f'{_pair_label(r["comparison"])}: {r["stat_name"]} = {r["statistic"]:{fmt}}, '
                         f'p = {_fmt_p(r["p_raw"])}, p(Holm) = {_fmt_p(r["p_holm"])} {_stars(r["p_holm"])}')
    return lines


def _note_line(omnibus: dict) -> list:
    note = omnibus.get('model_note')
    return [note] if isinstance(note, str) and note else []


def _wrap(lines, width_in: float) -> list:
    width = max(40, int(width_in * STATS_CHARS_PER_IN))
    return [w for line in lines for w in (textwrap.wrap(line, width) or [''])]


def _titled(text: str, grp: Grouping) -> str:
    return f'{grp.title}: {text}' if grp.title else text


def _draw_brackets(ax, pairs: list, y_base: float, y_step: float, y_text: float):
    """Line + stars over every (x1, x2, p) with p < ALPHA, narrower spans lowest.
    Returns the y of the highest line, or None."""
    sig = sorted((abs(x2 - x1), min(x1, x2), max(x1, x2), _stars(p))
                 for x1, x2, p in pairs if p is not None and pd.notna(p) and p < ALPHA)
    y = y_base
    for _span, x1, x2, stars in sig:
        ax.plot([x1, x2], [y, y], color=PAL_BLACK, linewidth=BOX_LW, clip_on=False, zorder=5)
        ax.text((x1 + x2) / 2, y + y_text, stars, ha='center', va='center',
                fontsize=FS_LABEL, fontweight='bold', clip_on=False, zorder=5)
        y += y_step
    return y - y_step if sig else None


def _posthoc_pairs(posthoc: pd.DataFrame, pos: dict) -> list:
    pairs = []
    if posthoc is None or not len(posthoc):
        return pairs
    for _, r in posthoc.iterrows():
        a, b = [s.strip() for s in str(r['comparison']).split(' vs ')]
        if a in pos and b in pos:
            pairs.append((pos[a], pos[b], r['p_holm']))
    return pairs


def _fit_ylim(ax, y_lo: float, top: float, pairs: list, pct: bool = False):
    """Brackets above `top` and a y-range that holds them."""
    span = max(top - y_lo, 1e-9)
    y_hi = _draw_brackets(ax, pairs, top + 0.06 * span, 0.09 * span, 0.03 * span)
    ax.set_ylim(y_lo, max(top, y_hi if y_hi is not None else top) + 0.08 * span)
    if pct and top >= 100:      # stacked shares: keep the ticks at 0-100
        ax.set_yticks(np.arange(0, 101, 20))


def _rec_dots(ax, rec: pd.DataFrame, grp_col: str, groups: list, xpos, col: str, jitter=0.1):
    """Per-recording values as small dots coloured by animal; returns their max."""
    rng = np.random.default_rng(0)
    top = -np.inf
    for x, g in zip(xpos, groups):
        s = rec.loc[rec[grp_col] == g, [col, 'animal']].dropna(subset=[col])
        if s.empty:
            continue
        ax.scatter(x + rng.uniform(-jitter, jitter, len(s)), s[col], s=9,
                   facecolors=[_animal_color(a) for a in s['animal']], edgecolors='0.25',
                   linewidths=0.4, zorder=3)
        top = max(top, s[col].max())
    return top


def _animal_handles(animals) -> list:
    return [Line2D([0], [0], marker=_animal_marker(a), linestyle='None', markersize=8,
                   markerfacecolor=_animal_color(a), markeredgecolor='0.25', label=a)
            for a in [a for a in ANIMAL_COLORS if a in set(animals)]
            + sorted(set(animals) - set(ANIMAL_COLORS))]


def _style(ax):
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.spines[['top', 'right']].set_visible(False)


# ── Plots (draw functions: (ax, grp, res) -> (legend handles, stats lines)) ──

def _groups_with(df: pd.DataFrame, grp: Grouping) -> list:
    return [g for g in grp.order if (df[grp.col] == g).any()]


def draw_overall(ax, grp, res):
    """Metric 1: % of all epochs rejected (bars: epochs pooled; dots: recordings)."""
    pr = res['prev']['overall']
    desc = pr['desc']
    if desc.empty:
        return [], ['No epochs']
    groups, x = list(desc['group']), np.arange(len(desc))
    pct = desc['pct_epochs'].to_numpy(float)
    ax.bar(x, pct, width=BAR_WIDTH, color=[GROUP_FACE.get(g, '#BFBFBF') for g in groups],
           edgecolor=BAR_EDGE, linewidth=BAR_EDGE_LW, zorder=2)
    top = max(np.nanmax(pct), _rec_dots(ax, res['rec'], grp.col, groups, x, 'pct_overall'))
    _fit_ylim(ax, 0, top if top > 0 else 1, _posthoc_pairs(pr['epoch'][1], dict(zip(groups, x))),
              pct=True)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{_label(g)}\n{p:.1f}%\n({k}/{n})\n{r} rec' for g, p, k, n, r in
                        zip(groups, pct, desc['n_rejected'], desc['n_epochs'], desc['n_recordings'])])
    ax.set_ylabel('% of epochs', fontsize=FS_LABEL)
    ax.set_title(textwrap.fill(_titled(PREVALENCE_MEASURES['overall'][0], grp), 40),
                 fontsize=FS_TITLE - 2)

    lines = ['Epoch level (all epochs of a group pooled):'] + _test_lines(*pr['epoch'])
    lines += _note_line(pr['epoch'][0])
    lines += ['Recording level (one % per recording):'] + _test_lines(*pr['recording'])
    lines += _note_line(pr['recording'][0])
    lines.append('Bars: epochs pooled; dots: one recording each (animal colour); '
                 f'brackets: epoch-level pairwise tests, Holm p < {ALPHA:g}')
    return _animal_handles(res['rec']['animal'].dropna().unique()), lines


def draw_immobile_vs_mobile(ax, grp, res):
    """Metric 2: % of immobility / mobility epochs rejected, side by side per group."""
    within = res['prev_within']
    if within.empty:
        return [], ['No epochs']
    groups, x = list(within['group']), np.arange(len(within))
    off = {'immobile': -0.2, 'moving': 0.2}
    top = 0.0
    for m in MOVEMENTS:
        pct = within[f'pct_{m}'].to_numpy(float)
        ax.bar(x + off[m], np.nan_to_num(pct), width=0.38, color=MOVEMENT_FACE[m],
               edgecolor=BAR_EDGE, linewidth=BAR_EDGE_LW, zorder=2)
        top = max(top, np.nanmax(pct) if np.isfinite(pct).any() else 0,
                  _rec_dots(ax, res['rec'], grp.col, groups, x + off[m], f'pct_{m}', jitter=0.07))
    pairs = [(xi + off['immobile'], xi + off['moving'], p)
             for xi, p in zip(x, within['epoch_p_holm'])]
    _fit_ylim(ax, 0, top if top > 0 else 1, pairs, pct=True)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{_label(r.group)}\nImm {r.pct_immobile:.1f}%\n'
                        f'({r.n_immobile_rejected}/{r.n_immobile})\nMob {r.pct_moving:.1f}%\n'
                        f'({r.n_moving_rejected}/{r.n_moving})' for r in within.itertuples()],
                       fontsize=FS_TICK - 1)
    ax.set_ylabel('% of epochs of that movement state', fontsize=FS_LABEL)
    ax.set_title(textwrap.fill(_titled(f'ED_min > {ACG_ED_MIN_THRESH:g} rejection, '
                                       f'immobility vs mobility', grp), 40), fontsize=FS_TITLE - 2)

    lines = [f'Immobility vs mobility within each {grp.noun.rstrip("s")} '
             f'(Holm across {grp.noun}):']
    for r in within.itertuples():
        lines.append(f'{_label(r.group)}: epochs {r.epoch_test} χ² = {r.epoch_statistic:.2f}, '
                     f'p(Holm) = {_fmt_p(r.epoch_p_holm)} {_stars(r.epoch_p_holm)} | recordings '
                     f'{r.rec_test} p(Holm) = {_fmt_p(r.rec_p_holm)} {_stars(r.rec_p_holm)}')
    for m in MOVEMENTS:
        pr = res['prev'][m]
        lines.append(f'Across {grp.noun}, {MOVEMENT_LABELS[m].lower()} epochs -- epoch level:')
        lines += _test_lines(*pr['epoch'])
        lines.append(f'Across {grp.noun}, {MOVEMENT_LABELS[m].lower()} epochs -- recording level:')
        lines += _test_lines(*pr['recording'])
    lines += _note_line(res['prev']['overall']['recording'][0])
    lines.append('Bars: epochs pooled; dots: one recording each (animal colour); '
                 f'brackets: immobility vs mobility (epoch level), Holm p < {ALPHA:g}')
    handles = [Patch(facecolor=MOVEMENT_FACE[m], label=MOVEMENT_LABELS[m]) for m in MOVEMENTS]
    return handles + _animal_handles(res['rec']['animal'].dropna().unique()), lines


def draw_split(ax, grp, res):
    """Metric 2: split of the ED_min-rejected epochs between immobility and mobility."""
    pr = res['prev']['rejected_in_immobility']
    desc = pr['desc']
    if desc.empty:
        return [], ['No ED_min-rejected epochs']
    groups, x = list(desc['group']), np.arange(len(desc))
    imm = desc['pct_epochs'].to_numpy(float)
    ax.bar(x, imm, width=BAR_WIDTH, color=MOVEMENT_FACE['immobile'], edgecolor=BAR_EDGE,
           linewidth=BAR_EDGE_LW, zorder=2)
    ax.bar(x, 100 - imm, bottom=imm, width=BAR_WIDTH, color=MOVEMENT_FACE['moving'],
           edgecolor=BAR_EDGE, linewidth=BAR_EDGE_LW, zorder=2)
    _fit_ylim(ax, 0, 100, _posthoc_pairs(pr['epoch'][1], dict(zip(groups, x))), pct=True)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{_label(g)}\nImm {p:.1f}%\nMob {100 - p:.1f}%\n(n={n} rejected)'
                        for g, p, n in zip(groups, imm, desc['n_epochs'])])
    ax.set_ylabel('% of ED_min-rejected epochs', fontsize=FS_LABEL)
    ax.set_title(textwrap.fill(_titled(f'ED_min > {ACG_ED_MIN_THRESH:g} rejected epochs: '
                                       f'immobility vs mobility share', grp), 40),
                 fontsize=FS_TITLE - 2)
    lines = ['Epoch level (rejected epochs pooled; groups x (immobility, mobility)):']
    lines += _test_lines(*pr['epoch']) + _note_line(pr['epoch'][0])
    lines += ['Recording level (one % per recording with >= 1 rejected epoch):']
    lines += _test_lines(*pr['recording']) + _note_line(pr['recording'][0])
    handles = [Patch(facecolor=MOVEMENT_FACE[m], label=MOVEMENT_LABELS[m]) for m in MOVEMENTS]
    return handles, lines


def draw_centroid(ax, grp, res, metric):
    """Metric 3: immobility vs mobility distribution of one centroid metric per
    group; recording dots, per-animal centroids and the group centroid."""
    name, ylabel, _stem = CENTROID_METRICS[metric]
    c = res['cent'][metric]
    sub = c['data']
    groups = _groups_with(sub, grp)
    if not groups:
        return [], ['No accepted epochs']
    off = {'immobile': -0.2, 'moving': 0.2}
    rng = np.random.default_rng(0)
    top, bottom = -np.inf, np.inf
    for m in MOVEMENTS:
        data, pos, gs = [], [], []
        for i, g in enumerate(groups):
            v = sub.loc[(sub[grp.col] == g) & (sub['movement'] == m), metric].dropna()
            if len(v):
                data.append(v.to_numpy(float))
                pos.append(i + off[m])
                gs.append(g)
        if not data:
            continue
        bp = ax.boxplot(data, positions=pos, widths=0.34, showfliers=False, patch_artist=True,
                        boxprops=dict(linewidth=BOX_LW, color=PAL_BLACK),
                        whiskerprops=dict(linewidth=BOX_LW, color=PAL_BLACK),
                        capprops=dict(linewidth=BOX_LW, color=PAL_BLACK),
                        medianprops=dict(linewidth=MEDIAN_LW, color=PAL_BLACK), zorder=2)
        for patch in bp['boxes']:
            patch.set_facecolor(matplotlib.colors.to_rgba(MOVEMENT_FACE[m], 0.45))
        for p_, g, v in zip(pos, gs, data):
            s = sub[(sub[grp.col] == g) & (sub['movement'] == m)].dropna(subset=[metric])
            if CENTROID_UNIT == 'recording':
                ax.scatter(p_ + rng.uniform(-0.1, 0.1, len(s)), s[metric], s=8,
                           facecolors=[_animal_color(a) for a in s['animal']],
                           edgecolors='0.3', linewidths=0.3, alpha=0.8, zorder=3)
            for a, va in s.groupby('animal')[metric]:
                ax.scatter(p_, va.mean(), s=70, marker=_animal_marker(a),
                           facecolors=_animal_color(a), edgecolors=PAL_BLACK, linewidths=0.8,
                           zorder=4)
            ax.scatter(p_, np.mean(v), s=110, marker='D', facecolors=CENTROID_FACE[m],
                       edgecolors=PAL_BLACK, linewidths=1.0, zorder=5)
            lo, hi = (np.min(v), np.max(v)) if CENTROID_UNIT == 'recording' else \
                np.percentile(v, [1, 99])
            top, bottom = max(top, hi), min(bottom, lo)

    pairs = []
    for m in MOVEMENTS:
        pairs += _posthoc_pairs(c['across'][m][1], {g: i + off[m] for i, g in enumerate(groups)})
    if len(c['within']):
        pos_g = {g: i for i, g in enumerate(groups)}
        pairs += [(pos_g[r.group] + off['immobile'], pos_g[r.group] + off['moving'], r.p_holm)
                  for r in c['within'].itertuples() if r.group in pos_g]
    span = top - bottom if top > bottom else 1.0
    _fit_ylim(ax, bottom - 0.05 * span, top, pairs)
    ax.set_xticks(np.arange(len(groups)))
    n_rec = sub.groupby(grp.col)['file'].nunique()
    ax.set_xticklabels([f'{_label(g)}\n({n_rec.get(g, 0)} rec)' for g in groups])
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.set_title(textwrap.fill(_titled(f'{name} centroid, immobility vs mobility', grp), 40),
                 fontsize=FS_TITLE - 2)

    unit = ('one value per recording and movement state = mean of its accepted epochs '
            '(the recording centroid)' if CENTROID_UNIT == 'recording'
            else 'one value per accepted epoch')
    lines = [f'Unit: {unit}. Diamond: group centroid (mean); large markers: animal centroids.']
    for r in c['desc'].itertuples():
        sd = f'{r.std:.3g}' if pd.notna(r.std) else 'n/a'
        lines.append(f'{_label(r.group)} {MOVEMENT_LABELS[r.movement].lower()} (n={r.n}, '
                     f'{int(r.n_epochs)} epochs): mean {r.mean:.3g} ± {sd} SD | '
                     f'median {r.median:.3g} [Q1 {r.q1:.3g}, Q3 {r.q3:.3g}]')
    for m in MOVEMENTS:
        lines.append(f'Across {grp.noun}, {MOVEMENT_LABELS[m].lower()}:')
        lines += _test_lines(*c['across'][m])
    if len(c['within']):
        lines.append(f'Immobility vs mobility within each {grp.noun.rstrip("s")} '
                     f'({c["within"]["test"].iloc[0]}, Holm across {grp.noun}): ' +
                     ', '.join(f'{_label(r.group)} p(Holm) = {_fmt_p(r.p_holm)} {_stars(r.p_holm)}'
                               for r in c['within'].itertuples()))
    lines += _note_line(c['across']['immobile'][0])
    lines.append(f'Brackets: Holm p < {ALPHA:g} (across {grp.noun} within a movement state; '
                 f'immobility vs mobility within a {grp.noun.rstrip("s")})')
    handles = [Patch(facecolor=matplotlib.colors.to_rgba(MOVEMENT_FACE[m], 0.45),
                     edgecolor=PAL_BLACK, label=MOVEMENT_LABELS[m]) for m in MOVEMENTS]
    handles += [Line2D([0], [0], marker='D', linestyle='None', markersize=8,
                       markerfacecolor=CENTROID_FACE[m], markeredgecolor=PAL_BLACK,
                       label=f'{MOVEMENT_LABELS[m]} centroid') for m in MOVEMENTS]
    return handles + _animal_handles(sub['animal'].dropna().unique()), lines


PLOTS = [
    ('EDmin_Rejected_Overall',          draw_overall,            GROUP_W),
    ('EDmin_Rejected_ImmobileVsMobile', draw_immobile_vs_mobile, PAIR_GROUP_W),
    ('EDmin_Rejected_Split',            draw_split,              GROUP_W),
] + [(stem, (lambda m: lambda ax, grp, res: draw_centroid(ax, grp, res, m))(metric), PAIR_GROUP_W)
     for metric, (_n, _y, stem) in CENTROID_METRICS.items()]


def save_panels(results: list, draw, group_w: float, out_path: str):
    """One figure with a panel per analyse() result in `results` (side by side),
    each with its own legend and stats text underneath."""
    widths = [max(4.5, len(r['grp'].order) * group_w + AXIS_PAD_W) for r in results]
    # First pass on throw-away axes to get each panel's legend + stats lines.
    probe = plt.figure()
    drawn = [draw(probe.add_subplot(), r['grp'], r) for r in results]
    plt.close(probe)
    lines = [_wrap(ls, w) for (_h, ls), w in zip(drawn, widths)]
    n_leg = [int(np.ceil(len(h) / 3)) for h, _l in drawn]
    info_h = max(len(l) * STATS_LINE_IN + n * LEGEND_ROW_IN for l, n in zip(lines, n_leg)) + 0.3

    fig = plt.figure(figsize=(sum(widths) + 0.3 * (len(results) - 1), PLOT_H + info_h),
                     layout='constrained')
    subfigs = fig.subfigures(1, len(results), width_ratios=widths, wspace=0.03, squeeze=False)[0]
    for sf, r, ls, n in zip(subfigs, results, lines, n_leg):
        ax, ax_info = sf.subplots(2, 1, gridspec_kw={'height_ratios': [PLOT_H, info_h]})
        handles, _l = draw(ax, r['grp'], r)
        _style(ax)
        ax_info.axis('off')
        if handles:
            ax_info.legend(handles=handles, loc='upper center', ncol=3, fontsize=FS_LEGEND)
        ax_info.text(0.5, 1 - (n * LEGEND_ROW_IN + 0.1) / info_h, '\n'.join(ls),
                     transform=ax_info.transAxes, ha='center', va='top', fontsize=FS_STATS,
                     linespacing=1.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure -> {out_path}')


def plot_all(results: list, out_dir: str):
    for stem, draw, group_w in PLOTS:
        try:
            save_panels(results, draw, group_w, os.path.join(out_dir, f'{stem}.png'))
        except Exception as e:  # noqa: BLE001 -- keep the other plots going
            plt.close('all')
            print(f'  SKIP {stem}.png: {type(e).__name__}: {e}')


def print_significant(tables: dict, name: str):
    rows = []
    for tname in ('Prev_Tests', 'Cent_Tests'):
        t = tables.get(tname)
        if t is None or not len(t):
            continue
        p = t['p_holm'].fillna(t['p_raw']) if 'p_holm' in t else t['p_raw']
        for (_i, r), pv in zip(t.iterrows(), p):
            if pd.notna(pv) and pv < ALPHA:
                rows.append(f"  {r['measure']:22s} {r['level']:10s} {str(r.get('movement', '')):9s} "
                            f"{str(r['comparison']):18s} {r['test']:34s} p={pv:.4g}")
    for tname, cols in (('Prev_Within', ['epoch_p_holm', 'rec_p_holm']), ('Cent_Within', ['p_holm'])):
        t = tables.get(tname)
        if t is None or not len(t):
            continue
        for _i, r in t.iterrows():
            for col in cols:
                if pd.notna(r[col]) and r[col] < ALPHA:
                    rows.append(f"  {r.get('measure', 'edmin'):22s} {col:12s} immobile vs moving "
                                f"in {r['group']}: p(Holm)={r[col]:.4g}")
    print(f'\nSignificant differences, {name} (p < {ALPHA}):' if rows
          else f'\nNo significant differences ({name}).')
    for line in rows:
        print(line)


# ── Run snapshot ────────────────────────────────────────────────────────────

def _git(*args) -> str:
    try:
        return subprocess.run(['git', *args], cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def save_run_snapshot() -> str:
    """Copy this script (+ CellClusteredStats_Utils) and write RunMetadata.csv
    (time, git commit, input, every parameter) to
    OUTPUT_DIR\\RUN_SNAPSHOT_SUBDIR\\<timestamp>."""
    now = datetime.now()
    snap_dir = os.path.join(OUTPUT_DIR, RUN_SNAPSHOT_SUBDIR, now.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(snap_dir, exist_ok=True)
    script = os.path.abspath(__file__)
    for path in (script, ccs.__file__):
        shutil.copy2(path, snap_dir)
    rows = [('run_timestamp', now.isoformat(timespec='seconds')), ('script', script),
            ('git_commit', _git('rev-parse', 'HEAD') or 'n/a'),
            ('git_uncommitted_changes', _git('status', '--porcelain', '--', script) or 'none'),
            ('python', sys.version.replace('\n', ' ')), ('machine', platform.node())]
    plain = (str, int, float, bool, list, tuple, dict, type(None))
    rows += [(k, str(v)) for k, v in globals().items()
             if k.isupper() and not k.startswith('_') and isinstance(v, plain)]
    pd.DataFrame(rows, columns=['parameter', 'value']).to_csv(
        os.path.join(snap_dir, 'RunMetadata.csv'), index=False, encoding='utf-8-sig')
    print(f'Run snapshot saved to {snap_dir}')
    return snap_dir


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    read_acg_settings()
    _set_prevalence_labels()
    save_run_snapshot()

    ep = load_epochs()
    rec = recording_table(ep)
    cen = centroid_table(ep)
    del ep
    print('\nRecordings by arena and session type:')
    print(pd.crosstab(rec['arena_type'], rec['session_type'].fillna('unrecognized'), margins=True))

    all_tables = {}

    def _store(prefix, tables, **tags):
        for tname, tbl in tables.items():
            if len(tbl):
                for i, (k, v) in enumerate(tags.items()):
                    tbl.insert(i, k, v)
            all_tables[f'{prefix}_{tname}'] = tbl

    # ── 1. Session types within each arena ──────────────────────────────────
    for arena in ARENA_TYPES:
        grp = session_grouping(arena)

        def _in_arena(d):
            return d[(d['arena_type'] == arena) & d['session_type'].isin(grp.order)]

        if _in_arena(rec).empty:
            print(f'\nNo recordings for {arena}; skipping.')
            continue
        res = analyse(grp, _in_arena(rec), _in_arena(cen))
        plot_all([res], os.path.join(OUTPUT_DIR, 'SessionType', arena))
        tables = result_tables(res)
        print_significant(tables, grp.title)
        _store(f'Ses{arena}', tables, arena_type=arena)

    # ── 2. Arena types (session types pooled) ───────────────────────────────
    res = analyse(ARENA_GROUPING, rec, cen)
    plot_all([res], os.path.join(OUTPUT_DIR, 'ArenaType'))
    tables = result_tables(res)
    print_significant(tables, 'Arena types')
    _store('Arena', tables)

    # ── 3. Arena types, only animals recorded in all compared arenas ────────
    ac_results = []
    for name, (animals, arenas) in ANIMAL_CONTROLLED_COMPARISONS.items():
        grp = animal_controlled_grouping(animals, arenas)
        wanted = {a.lower() for a in animals}

        def _select(d):
            return d[d['animal'].astype(str).str.lower().isin(wanted)
                     & d['arena_type'].isin(grp.order)]

        sel = _select(rec)
        print(f'\n{name}: recordings per animal and arena:')
        print(pd.crosstab(sel['animal'], sel['arena_type'], margins=True))
        for arena in grp.order:
            missing = [a for a in animals if not ((sel['animal'].str.lower() == a.lower())
                                                  & (sel['arena_type'] == arena)).any()]
            if missing:
                print(f'WARNING: no recordings of {missing} in the {arena} arena.')
        if sel.empty:
            continue
        res = analyse(grp, sel, _select(cen))
        ac_results.append(res)
        tables = result_tables(res)
        print_significant(tables, grp.title)
        _store('AC' + ''.join(grp.order), tables, ac_comparison=name, animals=', '.join(animals))
    if ac_results:
        plot_all(ac_results, os.path.join(OUTPUT_DIR, ANIMAL_CONTROLLED_DIR))

    # ── Save ────────────────────────────────────────────────────────────────
    with pd.ExcelWriter(STATS_XLSX, engine='openpyxl') as writer:
        for sheet, tbl in all_tables.items():
            tbl.to_excel(writer, sheet_name=sheet[:31], index=False)
        rec.to_excel(writer, sheet_name='Recordings', index=False)
        if CENTROID_UNIT == 'recording':
            cen.to_excel(writer, sheet_name='Centroids', index=False)
    print(f'\nAll statistics saved to {STATS_XLSX}')

    parts = [t.assign(table=s) for s, t in all_tables.items() if len(t)]
    if parts:
        long = pd.concat(parts, ignore_index=True)
        long.insert(0, 'table', long.pop('table'))
        long.to_csv(STATS_CSV, index=False, encoding='utf-8-sig')
        print(f'All statistics also saved to {STATS_CSV}')

    print(f'\nContinuous test: '
          f'{"Kruskal-Wallis + Mann-Whitney U (Holm), independence assumed" if CONTINUOUS_TEST == "KW" else "nested LMM"}; '
          f'centroid unit: {CENTROID_UNIT}')
    print(f'Done. Plots saved under {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
