# -*- coding: utf-8 -*-
"""
Statistical comparison of the ACG theta-continuity results
(Figc_ACG_ScalingCorrected_v9_FOOOF.py -- or v8_NoiseCleaning, same epoch-table
format -- its epoch table 'acg_theta_continuity_epochs.csv') at the three levels of
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

Copied LFP files (DROP_DUPLICATE_LFP): some session folders hold copies of one
LFP signal under several .ncs names (placed there so that every tetrode had a
matched LFP file for another analysis; e.g. CSC1 = CSC4, CSC2 = CSC3,
CSC5 = CSC6 = CSC7 = CSC8). Within a session, files whose epoch results are
identical (DUP_SIGNATURE_COLS of every epoch) are copies; only the first of
each set (natural file-name order) is analysed. The dropped copies are listed
in the 'LFP_Duplicates_Dropped' sheet.

Arena and session type are parsed from each epoch's session folder exactly as
in ArenaSessionType_StatsComparison_v18.py (extract_arena_type /
extract_session_type). Only epochs labelled 'immobile' or 'moving' are in the
epoch table (the ACG script drops the rest).

v2 vs v1: v1 read the ED_min-threshold runs (v4: 'skipped_edmin', ACG_ED_MIN_THRESH).
v8/v9 have no ED_min threshold: every epoch is matched against the broadband
1-20 Hz reference bank (ACG_REF_BANK) and rejected as NON-THETA when the frequency
of its ED_min sinusoid lies outside ACG_FREQ_RANGE (3-7 Hz) -- the
'skipped_outofband' column. Metrics 1-2 are now the non-theta rejection.

Metrics:

  1. Non-theta rejection: % of epochs whose ED_min (closest-matching) reference
     sinusoid is outside ACG_FREQ_RANGE ('skipped_outofband'). Denominator:
     every epoch in the epoch table, i.e. epochs that passed the record-integrity,
     delta/theta and MAD artifact rejection.
  2. Non-theta rejection in immobility vs mobility:
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
  4. Theta-range epochs in immobility vs mobility: % of immobility epochs and
     % of mobility epochs that are theta epochs (each normalised within its
     own movement state). A theta epoch is an accepted (~skipped) epoch of the
     ACG run: matched-sinusoid frequency ('freq') in ACG_FREQ_RANGE. Denominator:
     every epoch of that movement state in the epoch table (non-theta epochs
     included). Compared across the groups within each movement state, and
     immobility vs mobility within each group, with the same tests as metric 2.
     With no ED_min threshold, a theta epoch is just a non-rejected one, so
     metric 4 = 100 % - metric 2 per movement state and its tests give the
     same statistics and p-values as metric 2's (kept as the theta-prevalence
     view of the same counts).

Tests (the same as ArenaSessionType_StatsComparison_v18.py):
  - % outcomes, epoch level: chi-square test of independence on the
    groups x (rejected, kept) epoch counts, then pairwise chi-square tests
    (Yates-corrected for 2 x 2 tables), Holm-Bonferroni corrected within each
    measure. Epochs are treated as independent; 'model_note' flags expected
    counts < 5.
  - % outcomes, immobility vs mobility within a group (metrics 2 and 4):
    PAIRED. Every PAIRED_UNIT (a recording, or a session with its channels'
    epoch counts summed) has a % for both movement states (units with
    < PAIRED_MIN_EPOCHS epochs in either state are left out); the pairs are
    compared with the Wilcoxon signed-rank test, Holm-corrected across the
    groups. Units are treated as independent (with PAIRED_UNIT = 'recording',
    the channels of one session count as separate pairs).
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
    Across groups: within each movement state.
  - centroids (metric 3), immobility vs mobility within a group: PAIRED, one
    pair per PAIRED_UNIT (its immobility and mobility centroids; with
    CENTROID_UNIT = 'epoch' the pairs are still per-unit means), Wilcoxon
    signed-rank test, Holm across the groups.
  The immobility-vs-mobility plots (metrics 2 and 4) show the paired tests and,
  across the groups, only the epoch-level chi-square tests (as the % outcomes
  of ArenaSessionType_StatsComparison_v19.py); the recording-level across-group
  tests are in the workbook only.

Output (OUTPUT_DIR = <ACG_RUN_DIR>\\Stats):
    SessionType\\<arena>\\*.png          session types of <arena>
    ArenaType\\*.png                     arena types
    AnimalControlledArenaType\\*.png     the animal-controlled comparisons side by side
      NonTheta_Rejected_Overall.png      metric 1
      NonTheta_Rejected_ImmobileVsMobile.png  metric 2, per-movement-state rates
      NonTheta_Rejected_Split.png        metric 2, split of the rejected epochs
      Centroid_Frequency.png             metric 3, frequency
      Centroid_PeakRange.png             metric 3, peak range
      Theta_ImmobileVsMobile.png         metric 4
    ACG_SessionArena_Stats.xlsx          sheets '<prefix>_<table>', prefix
                                         'Ses<arena>', 'Arena' or 'AC<A><B>'
                                         (+ 'Recordings', 'Centroids').
                                         Metric 4: measures 'theta_immobile' /
                                         'theta_moving' in Prev_Desc / Prev_Tests,
                                         immobility vs mobility in Theta_Within
    ACG_SessionArena_AllStats.csv        every table in long format
    RunSnapshots\\<YYYYmmdd_HHMMSS>\\     this script + RunMetadata.csv

INPUT_CSV is only read, not modified.
"""

import os
import re
import sys
import hashlib
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
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# CellClusteredStats_Utils.py (shared with ArenaSessionType_StatsComparison_v18.py)
PLACECELL_CODES_DIR = r'C:\Runita\NMR\Mole-Rat_PlaceCell\Codes\PlaceCell\Codes'
if PLACECELL_CODES_DIR not in sys.path:
    sys.path.append(PLACECELL_CODES_DIR)
import CellClusteredStats_Utils as ccs

# ── Parameters ──────────────────────────────────────────────────────────────

ROOT_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final'

# Output folder of the ACG run: OUTPUT_DIR of Figc_ACG_ScalingCorrected_v9_FOOOF.py.
# (The 5 Nov v8_NoiseCleaning run of the same format was moved to
#  ROOT_DIR\Results\ACG\ACG_theta_continuity_BroadBandFitting_NcsDropCorr_NoiseCleaned.)
ACG_RUN_DIR = os.path.join(ROOT_DIR, 'Results', 'ACG',
                           'ACG_theta_continuity_BP1-20_Demean_SaturationRejection')
INPUT_CSV   = os.path.join(ACG_RUN_DIR, 'acg_theta_continuity_epochs.csv')
ACG_META_CSV = os.path.join(ACG_RUN_DIR, 'meta', 'run_metadata.csv')

# Inside the ACG run folder, so the stats stay with the run they were computed from.
OUTPUT_DIR = os.path.join(ACG_RUN_DIR, 'Stats')
STATS_XLSX = os.path.join(OUTPUT_DIR, 'ACG_SessionArena_Stats.xlsx')
STATS_CSV  = os.path.join(OUTPUT_DIR, 'ACG_SessionArena_AllStats.csv')
RUN_SNAPSHOT_SUBDIR = 'RunSnapshots'

# Theta band of the ACG run (its ACG_FREQ_RANGE, Hz, inclusive): an epoch whose
# ED_min sinusoid lies outside it is rejected as non-theta ('skipped_outofband').
# Used for labels and a consistency check; read from ACG_META_CSV when it is there.
THETA_RANGE = (3.0, 7.0)
ACG_REF_BANK = (1.0, 20.0)   # reference bank of the ACG run (labels only)

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

# Copied LFP files: within a session, .ncs files with identical epoch results
# (these columns, every epoch) are copies of one signal -- only the first file
# (natural name order, e.g. CSC1 of CSC1 = CSC4) is analysed. False: keep all.
DROP_DUPLICATE_LFP = True
DUP_SIGNATURE_COLS = ['epoch_index', 'epoch_start_us', 'movement', 'ED_min', 'freq',
                      'peakrangenorm', 'skipped']

# Immobility vs mobility within a group is a PAIRED comparison (Wilcoxon
# signed-rank) for the % of metrics 2 and 4 and the centroids of metric 3; one
# pair per unit:
#   'recording': one .ncs file (its % of immobility and % of mobility epochs)
#   'session'  : one session folder, the epoch counts of its channels summed
PAIRED_UNIT = 'recording'
# Units with fewer epochs than this in either movement state are left out (1 = keep all).
PAIRED_MIN_EPOCHS = 1

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


def _parse_range(text) -> tuple:
    """'(3.0, 7.0)' -> (3.0, 7.0)."""
    lo, hi = (float(v) for v in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(text))[:2])
    return lo, hi


def read_acg_settings():
    """THETA_RANGE (= ACG_FREQ_RANGE) and ACG_REF_BANK from the ACG run's metadata,
    if available. Refuses ED_min-threshold runs (v4-era), which need v1."""
    global THETA_RANGE, ACG_REF_BANK
    if not os.path.isfile(ACG_META_CSV):
        print(f'No ACG run metadata at {ACG_META_CSV}; using THETA_RANGE = {THETA_RANGE}, '
              f'ACG_REF_BANK = {ACG_REF_BANK}')
        return
    meta = pd.read_csv(ACG_META_CSV).set_index('parameter')['value']
    if 'ACG_ED_MIN_THRESH' in meta and str(meta['ACG_ED_MIN_THRESH']) != 'None':
        raise ValueError(f'{ACG_META_CSV} is from an ED_min-threshold ACG run '
                         f'(ACG_ED_MIN_THRESH = {meta["ACG_ED_MIN_THRESH"]}); '
                         f'use ACG_ArenaSessionType_StatsComparison_v1.py for it.')
    if 'ACG_FREQ_RANGE' in meta:
        THETA_RANGE = _parse_range(meta['ACG_FREQ_RANGE'])
    if 'ACG_REF_BANK' in meta:
        ACG_REF_BANK = _parse_range(meta['ACG_REF_BANK'])
    print(f'ACG run: {meta.get("script_name", "?")} ({meta.get("run_datetime", "?")}), '
          f'theta band {_band()}, reference bank {ACG_REF_BANK[0]:g}-{ACG_REF_BANK[1]:g} Hz, '
          f'REJECT_PARTIAL_RECORD_EPOCHS = {meta.get("REJECT_PARTIAL_RECORD_EPOCHS", "?")}')


def load_epochs() -> pd.DataFrame:
    """Epoch table with arena / session type re-parsed from the session folder
    and the CellClusteredStats_Utils ids (animal, rec_day, cell_id) of each
    recording. Epochs with no recognizable arena are dropped."""
    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(f'{INPUT_CSV} not found -- run '
                                f'Figc_ACG_ScalingCorrected_v9_FOOOF.py first.')
    print(f'Loading {INPUT_CSV} ...')
    cols = pd.read_csv(INPUT_CSV, nrows=0).columns
    if 'skipped_outofband' not in cols:
        raise ValueError(f'{INPUT_CSV} has no skipped_outofband column'
                         + (' (it has skipped_edmin: an ED_min-threshold run -- use '
                            'ACG_ArenaSessionType_StatsComparison_v1.py)'
                            if 'skipped_edmin' in cols else '') + '.')
    ep = pd.read_csv(INPUT_CSV, usecols=['file', 'session_folder', 'movement', 'skipped',
                                         'skipped_outofband', 'freq', 'peakrangenorm',
                                         'ED_min', 'epoch_index', 'epoch_start_us'])
    ep['skipped'] = ep['skipped'].map(_to_bool)
    ep['skipped_outofband'] = ep['skipped_outofband'].map(_to_bool)
    ep, dups = drop_duplicate_recordings(ep)
    n_other = int((~ep['movement'].isin(MOVEMENTS)).sum())
    if n_other:
        print(f'NOTE: {n_other} epoch(s) with movement not in {MOVEMENTS}; dropped.')
    ep = ep[ep['movement'].isin(MOVEMENTS)].copy()

    # Epochs skipped for being too short (skipped, not out of band) have no
    # frequency estimate: neither theta nor non-theta, so they are dropped.
    too_short = ep['skipped'] & ~ep['skipped_outofband']
    if too_short.any():
        print(f'NOTE: {int(too_short.sum())} too-short skipped epoch(s) (no frequency estimate); dropped.')
        ep = ep[~too_short].copy()

    # Metric 4: theta epochs = the ACG run's accepted epochs.
    ep['theta'] = ~ep['skipped']
    # Consistency check: accepted <=> freq in THETA_RANGE (inclusive).
    lo, hi = THETA_RANGE
    in_band = pd.to_numeric(ep['freq'], errors='coerce').between(lo - 1e-9, hi + 1e-9)
    n_diff = int((in_band != ep['theta']).sum())
    print(f'Theta epochs (ED_min sinusoid in {_band()}): {int(ep["theta"].sum())} of {len(ep)}; '
          f'non-theta: {int(ep["skipped_outofband"].sum())}'
          + (f'; WARNING: {n_diff} epoch(s) whose freq disagrees with THETA_RANGE {THETA_RANGE} '
             f'(different ACG_FREQ_RANGE in the run?)' if n_diff else ''))

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
    return ep, dups


def _natural_key(path) -> tuple:
    """'CSC10ch2.ncs' after 'CSC2ch2.ncs' (numbers compared as numbers)."""
    return tuple((0, int(t), '') if t.isdigit() else (1, 0, t.lower())
                 for t in re.split(r'(\d+)', os.path.basename(str(path))) if t)


def drop_duplicate_recordings(ep: pd.DataFrame):
    """Copied LFP files: within a session folder, .ncs files whose epoch results
    (DUP_SIGNATURE_COLS of every epoch) are identical are copies of one signal
    (placed there so that every tetrode had a matched LFP file). Only the first
    file of each set of copies (natural file-name order) is kept. Returns
    (epochs, table of the dropped copies). DROP_DUPLICATE_LFP = False keeps all."""
    if not DROP_DUPLICATE_LFP:
        return ep, pd.DataFrame()
    cols = [c for c in DUP_SIGNATURE_COLS if c in ep.columns]
    rows = []
    for (sess, f), d in ep.groupby(['session_folder', 'file'], sort=False):
        s = d.sort_values(['epoch_start_us', 'epoch_index'])[cols].copy()
        num = s.select_dtypes('number').columns
        s[num] = s[num].round(9)
        sig = hashlib.sha1(pd.util.hash_pandas_object(s, index=False).to_numpy().tobytes())
        rows.append((sess, f, sig.hexdigest(), len(d)))
    files = pd.DataFrame(rows, columns=['session_folder', 'file', 'signature', 'n_epochs'])
    files = files.iloc[sorted(range(len(files)), key=lambda i: (files['session_folder'].iat[i],
                                                                _natural_key(files['file'].iat[i])))]
    files['kept_file'] = files.groupby(['session_folder', 'signature'])['file'].transform('first')
    dups = files[files['file'] != files['kept_file']].drop(columns='signature')
    n_sets = files.groupby(['session_folder', 'signature']).ngroups
    print(f'Duplicate LFP files: {len(files)} files, {n_sets} distinct signals; '
          f'{len(dups)} copies in {dups["session_folder"].nunique()} sessions dropped '
          f'(first file of each set kept).')
    dups = dups.assign(file_name=dups['file'].map(lambda p: os.path.basename(str(p))),
                       copy_of=dups['kept_file'].map(lambda p: os.path.basename(str(p))))
    return ep[~ep['file'].isin(set(dups['file']))].copy(), dups.reset_index(drop=True)


def recording_table(ep: pd.DataFrame) -> pd.DataFrame:
    """One row per recording (.ncs file): epoch, non-theta-rejected and theta-epoch
    counts (all / immobility / mobility) and the PREVALENCE_MEASURES percentages
    ('pct_<key>', NaN where the denominator is 0)."""
    rej = ep['skipped_outofband']
    theta = ep['theta']
    imm = ep['movement'].eq('immobile')
    mov = ep['movement'].eq('moving')
    flags = pd.DataFrame({'n_epochs': 1, 'n_rejected': rej,
                          'n_immobile': imm, 'n_immobile_rejected': rej & imm,
                          'n_moving': mov, 'n_moving_rejected': rej & mov,
                          'n_immobile_theta': theta & imm, 'n_moving_theta': theta & mov},
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
PREVALENCE_MEASURES = {   # labels set in _set_prevalence_labels (theta band of the run)
    'overall':  ('% of all epochs rejected as non-theta', 'n_rejected', 'n_epochs'),
    'immobile': ('% of immobility epochs rejected as non-theta', 'n_immobile_rejected', 'n_immobile'),
    'moving':   ('% of mobility epochs rejected as non-theta', 'n_moving_rejected', 'n_moving'),
    'rejected_in_immobility': ('% of non-theta rejected epochs that are immobility epochs',
                               'n_immobile_rejected', 'n_rejected'),
    # metric 4
    'theta_immobile': ('% of immobility epochs in theta', 'n_immobile_theta', 'n_immobile'),
    'theta_moving':   ('% of mobility epochs in theta', 'n_moving_theta', 'n_moving'),
}

# Immobility-vs-mobility comparisons: {family: (suffix of the count columns
# 'n_<movement>_<suffix>', {movement: PREVALENCE_MEASURES key})}
WITHIN_FAMILIES = {
    'nontheta': ('rejected', {'immobile': 'immobile', 'moving': 'moving'}),
    'theta':    ('theta', {'immobile': 'theta_immobile', 'moving': 'theta_moving'}),
}


def _band() -> str:
    return f'{THETA_RANGE[0]:g}-{THETA_RANGE[1]:g} Hz'


def _nontheta() -> str:
    """Short definition of a non-theta epoch, for labels."""
    return f'non-theta (ED_min sinusoid outside {_band()})'


def _set_prevalence_labels():
    """PREVALENCE_MEASURES labels with the theta band read from the ACG run."""
    for key, word in (('overall', 'all'), ('immobile', 'immobility'), ('moving', 'mobility')):
        _l, k_col, n_col = PREVALENCE_MEASURES[key]
        PREVALENCE_MEASURES[key] = (f'% of {word} epochs rejected as {_nontheta()}', k_col, n_col)
    for key, word in (('theta_immobile', 'immobility'), ('theta_moving', 'mobility')):
        _l, k_col, n_col = PREVALENCE_MEASURES[key]
        PREVALENCE_MEASURES[key] = (f'% of {word} epochs in {_band()} theta', k_col, n_col)


def _within_label(family: str) -> str:
    if family == 'nontheta':
        return 'Non-theta rejection rate, immobility vs mobility'
    return f'{_band()} theta-epoch rate, immobility vs mobility'


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


def _unit_word(plural: bool = True) -> str:
    return PAIRED_UNIT + ('s' if plural else '')


def _paired_wilcoxon(moving, immobile) -> tuple:
    """Wilcoxon signed-rank test of paired values (moving vs immobile):
    (test name, W, p); 'insufficient data' with < 2 pairs or no non-zero difference."""
    a, b = np.asarray(moving, float), np.asarray(immobile, float)
    if len(a) >= 2 and np.any(a != b):
        try:
            w = stats.wilcoxon(a, b)
            return 'Wilcoxon signed-rank (paired)', float(w.statistic), float(w.pvalue)
        except ValueError:
            pass
    return 'insufficient data', np.nan, np.nan


def paired_units(rec: pd.DataFrame, grp: Grouping, family: str) -> pd.DataFrame:
    """One row per PAIRED_UNIT of the groups of `grp` with >= PAIRED_MIN_EPOCHS
    epochs in both movement states: its % of immobility and % of mobility
    epochs that are `family` epochs ('pct_immobile', 'pct_moving') and
    'diff_mob_minus_imm' (percentage points)."""
    suffix, _keys = WITHIN_FAMILIES[family]
    k_i, k_m = f'n_immobile_{suffix}', f'n_moving_{suffix}'
    cols = [k_i, 'n_immobile', k_m, 'n_moving']
    sub = rec[rec[grp.col].isin(grp.order)]
    if PAIRED_UNIT == 'recording':
        u = sub[['file', 'session', 'animal', grp.col] + cols].rename(columns={'file': 'unit_id'})
        u = u.assign(n_channels=1)
    elif PAIRED_UNIT == 'session':
        u = (sub.groupby(['session', grp.col])
                .agg(animal=('animal', 'first'), n_channels=('file', 'nunique'),
                     **{c: (c, 'sum') for c in cols})
                .reset_index())
        u.insert(0, 'unit_id', u['session'])
    else:
        raise ValueError(f"PAIRED_UNIT must be 'recording' or 'session', not {PAIRED_UNIT!r}")
    min_n = max(1, PAIRED_MIN_EPOCHS)
    u = u[(u['n_immobile'] >= min_n) & (u['n_moving'] >= min_n)].copy()
    u['pct_immobile'] = 100 * u[k_i] / u['n_immobile']
    u['pct_moving'] = 100 * u[k_m] / u['n_moving']
    u['diff_mob_minus_imm'] = u['pct_moving'] - u['pct_immobile']
    u.insert(0, 'measure', family)
    return u.rename(columns={grp.col: 'group'}).reset_index(drop=True)


def compare_prevalence_within(rec: pd.DataFrame, grp: Grouping, family: str = 'nontheta'):
    """Within each group of `grp`: rate of the WITHIN_FAMILIES `family`
    ('nontheta': non-theta rejection, 'theta': theta epochs) in immobility vs
    mobility, PAIRED per PAIRED_UNIT (paired_units): Wilcoxon signed-rank test
    of the per-unit % (immobility vs mobility), Holm across the groups. The
    pooled epoch counts / % are kept as descriptives. Returns (one row per
    group, the per-unit pairs)."""
    suffix, _keys = WITHIN_FAMILIES[family]
    label = _within_label(family)
    units = paired_units(rec, grp, family)
    n_rec_units = (rec[rec[grp.col].isin(grp.order)]['file'].nunique() if PAIRED_UNIT == 'recording'
                   else rec[rec[grp.col].isin(grp.order)]['session'].nunique())
    if len(units) < n_rec_units:
        print(f'NOTE ({family}): {n_rec_units - len(units)} {_unit_word()} with < '
              f'{max(1, PAIRED_MIN_EPOCHS)} epoch(s) in a movement state left out of the paired test.')
    rows = []
    for g in grp.order:
        s = rec[rec[grp.col] == g]
        if s.empty:
            continue
        ki, ni = int(s[f'n_immobile_{suffix}'].sum()), int(s['n_immobile'].sum())
        km, nm = int(s[f'n_moving_{suffix}'].sum()), int(s['n_moving'].sum())
        u = units[units['group'] == g]
        d = u['diff_mob_minus_imm'].to_numpy(float)
        test, w_stat, p = _paired_wilcoxon(u['pct_moving'], u['pct_immobile'])
        row = {'measure': family, 'metric': label, 'group': g,
               f'n_immobile_{suffix}': ki, 'n_immobile': ni,
               'pct_immobile_pooled': 100 * ki / ni if ni else np.nan,
               f'n_moving_{suffix}': km, 'n_moving': nm,
               'pct_moving_pooled': 100 * km / nm if nm else np.nan,
               'paired_unit': PAIRED_UNIT, 'n_pairs': len(u),
               'n_animals': u['animal'].nunique(),
               'mean_pct_immobile': u['pct_immobile'].mean(),
               'mean_pct_moving': u['pct_moving'].mean(),
               'median_pct_immobile': u['pct_immobile'].median(),
               'median_pct_moving': u['pct_moving'].median(),
               'mean_diff_mob_minus_imm': d.mean() if len(d) else np.nan,
               'median_diff_mob_minus_imm': np.median(d) if len(d) else np.nan,
               'n_mob_higher': int((d > 0).sum()), 'n_imm_higher': int((d < 0).sum()),
               'n_tied': int((d == 0).sum()),
               'test': test, 'stat_name': 'W', 'statistic': w_stat, 'p_raw': p}
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out['p_holm'] = ccs.holm(out['p_raw'].to_numpy(float))
        out['significant'] = out['p_holm'] < ALPHA
    return out, units


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

    # Immobility vs mobility: paired per PAIRED_UNIT (Wilcoxon signed-rank).
    pairs = paired_centroids(sub, metric, grp)
    within = []
    for g in grp.order:
        if not (sub[grp.col] == g).any():
            continue
        u = pairs[pairs['group'] == g]
        d = u['diff_mob_minus_imm'].to_numpy(float)
        test, w_stat, p = _paired_wilcoxon(u['moving'], u['immobile'])
        within.append({'metric': f'{name}, immobility vs mobility (paired)', 'group': g,
                       'paired_unit': PAIRED_UNIT, 'n_pairs': len(u),
                       'n_animals': u['animal'].nunique(),
                       'median_immobile': u['immobile'].median(),
                       'median_moving': u['moving'].median(),
                       'median_diff_mob_minus_imm': np.median(d) if len(d) else np.nan,
                       'n_mob_higher': int((d > 0).sum()), 'n_imm_higher': int((d < 0).sum()),
                       'test': test, 'stat_name': 'W', 'statistic': w_stat, 'p_raw': p,
                       'model_note': f'Wilcoxon signed-rank on {_unit_word(False)} pairs '
                                     f'(pairs treated as independent)'})
    within = pd.DataFrame(within)
    if len(within):
        within['p_holm'] = ccs.holm(within['p_raw'].to_numpy(float))
        within['significant'] = within['p_holm'] < ALPHA

    animal = (sub.groupby([grp.col, 'movement', 'animal'])[metric]
                 .agg(['mean', 'median', 'count']).reset_index()
                 .rename(columns={grp.col: 'group', 'count': 'n'}))
    animal.insert(0, 'metric', name)
    return {'data': sub, 'desc': pd.concat(desc_parts, ignore_index=True) if desc_parts
            else pd.DataFrame(), 'across': across, 'within': within, 'animal': animal,
            'paired': pairs}


def paired_centroids(sub: pd.DataFrame, metric: str, grp: Grouping) -> pd.DataFrame:
    """One row per PAIRED_UNIT with an accepted-epoch centroid in both movement
    states: 'immobile' / 'moving' = mean `metric` of its accepted epochs (a
    'recording' unit is its recording centroid; a 'session' unit pools the
    epochs of its recordings), 'diff_mob_minus_imm'."""
    key = {'recording': 'file', 'session': 'session'}.get(PAIRED_UNIT)
    if key is None:
        raise ValueError(f"PAIRED_UNIT must be 'recording' or 'session', not {PAIRED_UNIT!r}")
    s = sub.dropna(subset=[metric])
    s = s.assign(_wv=s[metric] * s['n_epochs'])
    agg = s.groupby([key, grp.col, 'movement']).agg(
        animal=('animal', 'first'), n=('n_epochs', 'sum'), wv=('_wv', 'sum'))
    agg['value'] = agg['wv'] / agg['n']
    val = agg['value'].unstack('movement').reindex(columns=MOVEMENTS)
    n = agg['n'].unstack('movement').reindex(columns=MOVEMENTS).add_prefix('n_epochs_')
    out = (val.join(n).join(agg['animal'].groupby(level=[0, 1]).first())
              .dropna(subset=MOVEMENTS).reset_index()
              .rename(columns={key: 'unit_id', grp.col: 'group'}))
    out.columns.name = None
    out['diff_mob_minus_imm'] = out['moving'] - out['immobile']
    out.insert(0, 'metric', metric)
    return out


def analyse(grp: Grouping, rec: pd.DataFrame, cen: pd.DataFrame) -> dict:
    """Every test of one comparison (rec / cen already restricted to its rows)."""
    name = grp.title or grp.noun.capitalize()
    print(f'\n{"=" * 70}\n{name} -- comparing {grp.noun}: {[_label(g) for g in grp.order]}\n{"=" * 70}')
    print('Recordings / epochs per group:')
    print(rec.groupby(grp.col)[['file', 'n_epochs']].agg({'file': 'nunique', 'n_epochs': 'sum'})
          .reindex(grp.order).fillna(0).astype(int).to_string())
    within = {fam: compare_prevalence_within(rec, grp, fam) for fam in WITHIN_FAMILIES}
    return {'grp': grp, 'rec': rec,
            'prev': {key: compare_prevalence(rec, key, grp) for key in PREVALENCE_MEASURES},
            'within': {fam: w for fam, (w, _u) in within.items()},
            'paired': {fam: u for fam, (_w, u) in within.items()},
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
        'Prev_Within':  res['within']['nontheta'],
        'Theta_Within': res['within']['theta'],
        'Theta_PairedUnits': res['paired']['theta'],
        'Cent_Desc':    _concat([c['desc'] for c in res['cent'].values()]),
        'Cent_Animal':  _concat([c['animal'] for c in res['cent'].values()]),
        'Cent_Tests':   _concat(cent_tests),
        'Cent_Within':  _concat(within_cent),
        'Cent_PairedUnits': _concat([c['paired'] for c in res['cent'].values()]),
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
# Fill of the movement-state boxes and bars: MOVEMENT_FACE at 45 % over white
# (opaque, so bars and boxes have the same shade).
MOVEMENT_FILL_ALPHA = 0.45
MOVEMENT_FILL = {m: tuple(MOVEMENT_FILL_ALPHA * np.array(c[:3]) + (1 - MOVEMENT_FILL_ALPHA))
                 for m, c in MOVEMENT_FACE.items()}
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


def draw_immobile_vs_mobile(ax, grp, res, family='nontheta'):
    """Metrics 2 ('nontheta': % of epochs rejected as non-theta) and 4 ('theta': % of
    epochs in theta): immobility and mobility side by side per group, paired per
    PAIRED_UNIT (bars: mean of the per-unit %; dots joined by a line: one unit)."""
    _suffix, keys = WITHIN_FAMILIES[family]
    within, units = res['within'][family], res['paired'][family]
    if within.empty:
        return [], ['No epochs']
    groups, x = list(within['group']), np.arange(len(within))
    off = {'immobile': -0.2, 'moving': 0.2}
    for m in MOVEMENTS:
        ax.bar(x + off[m], np.nan_to_num(within[f'mean_pct_{m}'].to_numpy(float)), width=0.38,
               color=MOVEMENT_FILL[m], edgecolor=BAR_EDGE, linewidth=BAR_EDGE_LW, zorder=2)
    rng = np.random.default_rng(0)
    top = 0.0
    for xi, g in zip(x, groups):
        u = units[units['group'] == g]
        if u.empty:
            continue
        jit = rng.uniform(-0.06, 0.06, len(u))
        xy_i = np.c_[xi + off['immobile'] + jit, u['pct_immobile'].to_numpy(float)]
        xy_m = np.c_[xi + off['moving'] + jit, u['pct_moving'].to_numpy(float)]
        ax.add_collection(LineCollection(np.stack([xy_i, xy_m], axis=1), colors='0.5',
                                         linewidths=0.4, alpha=0.5, zorder=2.5))
        faces = [_animal_color(a) for a in u['animal']]
        for xy in (xy_i, xy_m):
            ax.scatter(xy[:, 0], xy[:, 1], s=9, facecolors=faces, edgecolors='0.25',
                       linewidths=0.4, zorder=3)
        top = max(top, xy_i[:, 1].max(), xy_m[:, 1].max())
    pairs = [(xi + off['immobile'], xi + off['moving'], p)
             for xi, p in zip(x, within['p_holm'])]
    _fit_ylim(ax, 0, top if top > 0 else 1, pairs, pct=True)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{_label(r["group"])}\n({r["n_pairs"]} {_unit_word(r["n_pairs"] != 1)})\n'
                        f'Imm {r["mean_pct_immobile"]:.1f}%\nMob {r["mean_pct_moving"]:.1f}%'
                        for _i, r in within.iterrows()],
                       fontsize=FS_TICK - 1)
    ax.set_ylabel('% of epochs of that movement state', fontsize=FS_LABEL)
    title = (f'Non-theta (outside {_band()}) rejection, immobility vs mobility'
             if family == 'nontheta' else
             f'Epochs in {_band()} theta, immobility vs mobility')
    ax.set_title(textwrap.fill(_titled(title, grp), 40), fontsize=FS_TITLE - 2)

    lines = []
    if family == 'theta':
        lines.append(f'Theta epoch: ED_min sinusoid frequency in {_band()} (reference bank '
                     f'{ACG_REF_BANK[0]:g}-{ACG_REF_BANK[1]:g} Hz). Denominator: every epoch of that '
                     f'movement state (non-theta epochs included), so these % = 100 - the non-theta '
                     f'rejection %, with the same test results.')
    else:
        lines.append(f'Non-theta epoch: ED_min sinusoid frequency outside {_band()} (reference '
                     f'bank {ACG_REF_BANK[0]:g}-{ACG_REF_BANK[1]:g} Hz). Denominator: every epoch '
                     f'of that movement state.')
    unit_def = ('one recording (.ncs file) = one pair' if PAIRED_UNIT == 'recording' else
                'one session = one pair, epoch counts of its channels summed')
    lines.append(f'Immobility vs mobility within each {grp.noun.rstrip("s")}, paired '
                 f'({unit_def}): Wilcoxon signed-rank test, Holm across {grp.noun}:')
    for r in within.itertuples():
        if r.test == 'insufficient data':
            lines.append(f'{_label(r.group)} (n = {r.n_pairs} pairs): insufficient data')
            continue
        lines.append(f'{_label(r.group)} (n = {r.n_pairs} pairs, {r.n_animals} animals): '
                     f'median Δ(mob − imm) = {r.median_diff_mob_minus_imm:+.1f} pp '
                     f'(mob higher in {r.n_mob_higher}, imm higher in {r.n_imm_higher}), '
                     f'W = {r.statistic:.1f}, p = {_fmt_p(r.p_raw)}, '
                     f'p(Holm) = {_fmt_p(r.p_holm)} {_stars(r.p_holm)}')
    if PAIRED_UNIT == 'recording':
        lines.append('Pairs treated as independent; the channels of one session are separate pairs.')
    for m in MOVEMENTS:
        lines.append(f'Across {grp.noun}, {MOVEMENT_LABELS[m].lower()} epochs:')
        lines += _test_lines(*res['prev'][keys[m]]['epoch'])
    lines += _note_line(res['prev'][keys['immobile']]['epoch'][0])
    lines.append(f'Bars: mean of the per-{_unit_word(False)} %; dots: one {_unit_word(False)} each '
                 f'(animal colour), joined across movement states; brackets: paired immobility '
                 f'vs mobility test, Holm p < {ALPHA:g}')
    handles = [Patch(facecolor=MOVEMENT_FILL[m], label=MOVEMENT_LABELS[m]) for m in MOVEMENTS]
    return handles + _animal_handles(res['rec']['animal'].dropna().unique()), lines


def draw_split(ax, grp, res):
    """Metric 2: split of the non-theta rejected epochs between immobility and mobility."""
    pr = res['prev']['rejected_in_immobility']
    desc = pr['desc']
    if desc.empty:
        return [], ['No non-theta rejected epochs']
    groups, x = list(desc['group']), np.arange(len(desc))
    imm = desc['pct_epochs'].to_numpy(float)
    ax.bar(x, imm, width=BAR_WIDTH, color=MOVEMENT_FILL['immobile'], edgecolor=BAR_EDGE,
           linewidth=BAR_EDGE_LW, zorder=2)
    ax.bar(x, 100 - imm, bottom=imm, width=BAR_WIDTH, color=MOVEMENT_FILL['moving'],
           edgecolor=BAR_EDGE, linewidth=BAR_EDGE_LW, zorder=2)
    _fit_ylim(ax, 0, 100, _posthoc_pairs(pr['epoch'][1], dict(zip(groups, x))), pct=True)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{_label(g)}\nImm {p:.1f}%\nMob {100 - p:.1f}%\n(n={n} rejected)'
                        for g, p, n in zip(groups, imm, desc['n_epochs'])])
    ax.set_ylabel('% of non-theta rejected epochs', fontsize=FS_LABEL)
    ax.set_title(textwrap.fill(_titled(f'Non-theta (outside {_band()}) rejected epochs: '
                                       f'immobility vs mobility share', grp), 40),
                 fontsize=FS_TITLE - 2)
    lines = ['Epoch level (rejected epochs pooled; groups x (immobility, mobility)):']
    lines += _test_lines(*pr['epoch']) + _note_line(pr['epoch'][0])
    lines += ['Recording level (one % per recording with >= 1 rejected epoch):']
    lines += _test_lines(*pr['recording']) + _note_line(pr['recording'][0])
    handles = [Patch(facecolor=MOVEMENT_FILL[m], label=MOVEMENT_LABELS[m]) for m in MOVEMENTS]
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
    # One jitter per recording, so its immobility and mobility dots line up for the pair lines.
    files = sorted(sub['file'].unique(), key=str)
    jit = dict(zip(files, np.random.default_rng(0).uniform(-0.1, 0.1, len(files))))
    join = CENTROID_UNIT == 'recording' and PAIRED_UNIT == 'recording'
    if join:
        pos_g = {g: i for i, g in enumerate(groups)}
        pr = c['paired'][c['paired']['group'].isin(pos_g)]
        x0 = pr['group'].map(pos_g).to_numpy(float) + pr['unit_id'].map(jit).to_numpy(float)
        segs = np.stack([np.c_[x0 + off['immobile'], pr['immobile'].to_numpy(float)],
                         np.c_[x0 + off['moving'], pr['moving'].to_numpy(float)]], axis=1)
        ax.add_collection(LineCollection(segs, colors='0.5', linewidths=0.4, alpha=0.5,
                                         zorder=2.5))
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
            patch.set_facecolor(MOVEMENT_FILL[m])
        for p_, g, v in zip(pos, gs, data):
            s = sub[(sub[grp.col] == g) & (sub['movement'] == m)].dropna(subset=[metric])
            if CENTROID_UNIT == 'recording':
                ax.scatter(p_ + s['file'].map(jit).to_numpy(float), s[metric], s=8,
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
        unit_def = ('one recording = one pair (its immobility and mobility centroids)'
                    if PAIRED_UNIT == 'recording' else
                    'one session = one pair, accepted epochs of its recordings pooled')
        lines.append(f'Immobility vs mobility within each {grp.noun.rstrip("s")}, paired '
                     f'({unit_def}): Wilcoxon signed-rank test, Holm across {grp.noun}:')
        for r in c['within'].itertuples():
            if r.test == 'insufficient data':
                lines.append(f'{_label(r.group)} (n = {r.n_pairs} pairs): insufficient data')
                continue
            lines.append(f'{_label(r.group)} (n = {r.n_pairs} pairs, {r.n_animals} animals): '
                         f'median Δ(mob − imm) = {r.median_diff_mob_minus_imm:+.3g} '
                         f'(mob higher in {r.n_mob_higher}, imm higher in {r.n_imm_higher}), '
                         f'W = {r.statistic:.1f}, p = {_fmt_p(r.p_raw)}, '
                         f'p(Holm) = {_fmt_p(r.p_holm)} {_stars(r.p_holm)}')
    lines += _note_line(c['across']['immobile'][0])
    lines.append(f'Brackets: Holm p < {ALPHA:g} (across {grp.noun} within a movement state; '
                 f'paired immobility vs mobility within a {grp.noun.rstrip("s")})'
                 + ('; grey lines join the two centroids of one recording' if join else ''))
    handles = [Patch(facecolor=MOVEMENT_FILL[m], edgecolor=PAL_BLACK, label=MOVEMENT_LABELS[m])
               for m in MOVEMENTS]
    handles += [Line2D([0], [0], marker='D', linestyle='None', markersize=8,
                       markerfacecolor=CENTROID_FACE[m], markeredgecolor=PAL_BLACK,
                       label=f'{MOVEMENT_LABELS[m]} centroid') for m in MOVEMENTS]
    return handles + _animal_handles(sub['animal'].dropna().unique()), lines


PLOTS = [
    ('NonTheta_Rejected_Overall',          draw_overall,            GROUP_W),
    ('NonTheta_Rejected_ImmobileVsMobile', draw_immobile_vs_mobile, PAIR_GROUP_W),
    ('NonTheta_Rejected_Split',            draw_split,              GROUP_W),
] + [(stem, (lambda m: lambda ax, grp, res: draw_centroid(ax, grp, res, m))(metric), PAIR_GROUP_W)
     for metric, (_n, _y, stem) in CENTROID_METRICS.items()] + [
    ('Theta_ImmobileVsMobile',
     lambda ax, grp, res: draw_immobile_vs_mobile(ax, grp, res, 'theta'), PAIR_GROUP_W),
]


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
    for tname, cols in (('Prev_Within', ['p_holm']),
                        ('Theta_Within', ['p_holm']),
                        ('Cent_Within', ['p_holm'])):
        t = tables.get(tname)
        if t is None or not len(t):
            continue
        for _i, r in t.iterrows():
            for col in cols:
                if pd.notna(r[col]) and r[col] < ALPHA:
                    rows.append(f"  {r.get('measure', 'nontheta'):22s} {col:12s} immobile vs moving "
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

    ep, dups = load_epochs()
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
        if len(dups):
            dups.to_excel(writer, sheet_name='LFP_Duplicates_Dropped', index=False)
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
