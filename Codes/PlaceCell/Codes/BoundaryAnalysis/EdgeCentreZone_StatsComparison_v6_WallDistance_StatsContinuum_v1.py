# -*- coding: utf-8 -*-
"""
Place-cell metrics as a function of DISTANCE FROM THE NEAREST WALL: every place cell is placed at
the wall distance of the PEAK BIN OF ITS RAW RATE MAP, and every place-cell characterization
metric is tested for a trend with that distance. Unlike EdgeCentreZone_StatsComparison_v6 (two
zones per arena), the distance is a continuum measured in the same units (2 x 2 cm bins) in every
arena, so arenas of different size are on one scale.

WALL DISTANCE (integer 'wall_dist', in bins; the nearest wall has distance 0):
  Open   circular open field, dia 60 cm -> radius 30 cm = 15 radial rings of BIN_CM (2 cm).
           wall_dist = floor((30 cm - radial distance of the bin centre) / 2 cm)
           0 = outermost ring (bin centre 28-30 cm from the centre) ... 14 = bins around the centre
  Linear 80 x 8 cm track, 4 bins across the width
           wall_dist = bins to the nearest wall: the 2 outer rows are 0, the 2 middle rows 1; the
           first / last column (track ends) are 0 too if LINEAR_ENDS_ARE_WALLS
  Circle annular track, 4 cm wide = 2 radial rows, each against a wall (outer / inner)
           wall_dist = 0 for every bin -> a single level, no trend test within the arena
Peak bin, tracking, speed filter and valid bins exactly as in v6 (its functions are imported, not
copied). v6's dwell criterion (both zones >= MIN_ZONE_DWELL_PCT) is NOT applied: there are no two
zones to balance. Instead the dwell at every distance is reported per session (DistSessions) and
compared with where the peaks are (PeakDistance_vs_Dwell.png, sheet PeakDistOccupancy).

TESTS -- for every metric, in each subset of cells (as in v6: 1. each arena, 2. each session type
of an arena, 3. the animal-controlled arena comparisons):
  Trend within each group (arena / session type), groups with >= 2 distance levels only:
    continuous  TREND_TEST 'Spearman': Spearman rank correlation of the metric with wall_dist
                (rows treated as independent, like v6's 'KW')
                TREND_TEST 'LMM': rank(metric) ~ wall_dist + (1|animal) + (1|animal:day) +
                (1|cell), likelihood-ratio test of the slope (ML fits); slope in SD of the ranks
                per bin (REML fit)
    yes/no      logistic regression of the outcome on wall_dist (TREND_TEST 'LMM': GEE clustered
                by cell), Wald test; odds ratio per bin of distance
  Pooled over the groups (>= 2 groups with >= 2 distance levels): the same models with the group
    as a covariate (rank(metric) ~ wall_dist + group; ranks over all pooled rows), i.e. one
    distance effect estimated within the groups; and the wall_dist x group interaction (do the
    slopes differ between groups?), likelihood-ratio / Wald test.
  Matched distance: at every wall_dist held by >= 2 groups, the groups are compared with v10's
    tests (Kruskal-Wallis / Mann-Whitney or LMM; chi-square), e.g. Circle vs Linear vs Open
    cells at distance 0.
  Trend p-values are also Holm-corrected across the metrics of a group (p_holm_metrics).
Speed direction is tested as two yes/no outcomes (p-type, n-type speed cell).

Metrics (v6's metric_specs: v10's continuous metrics, stability score per stability type,
proportions, and the temporal-stability metrics of TEMPORAL_STABILITY_EXCEL).

PLOTS (OUTPUT_DIR\\Arena, \\SessionType\\<arena>, \\AnimalControlledArenaType\\<name>): one per
metric, x = wall_dist; every group's cells as points (shaded by significance as in v10) and their
median + IQR per distance (continuous), or % yes with Wilson 95 % CI per distance (yes/no); the
groups side by side at each distance, the trend / pooled / matched-distance tests underneath.
OUTPUT_DIR\\WallDistanceSchematic_<arena>.png: the bins of each arena shaded by wall_dist.

Outputs (OUTPUT_DIR):
  WallDistance_Stats.xlsx   DistCounts        place-cell rows / cells per group and distance
                            DistSessions      % dwell at every distance, every session
                            DistCells         peak bin + wall_dist (+ temporal stability) of
                                              every unit in PLACE_CELL_DIR
                            PeakDistOccupancy observed peaks vs dwell-expected peaks per distance
                            Trend             within-group trend test of every metric
                            Pooled            group-adjusted distance effect + interaction
                            Matched_Omnibus / Matched_PostHoc  groups at the same distance
                            Desc              descriptives per group and distance
  WallDistance_AllStats.csv all of the above statistics tables in one long-format CSV

INPUT_EXCEL, PLACE_CELL_DIR and TEMPORAL_STABILITY_EXCEL are only read.
"""

import os
import sys
import textwrap
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
from scipy import stats

import statsmodels.api as sm
import statsmodels.formula.api as smf

# EdgeCentreZone_StatsComparison_v6_intmZoneOpen.py (geometry, tracking, raw-map peak bin, cell
# matching, temporal stability) lives in this folder, or in its OldVer\ folder once superseded;
# it puts v10 and CellClusteredStats_Utils on the path.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, 'OldVer')]
import EdgeCentreZone_StatsComparison_v6_intmZoneOpen as ez   # noqa: E402
asc, ccs = ez.asc, ez.ccs
import matplotlib.pyplot as plt                              # noqa: E402

# ── Parameters ──────────────────────────────────────────────────────────────

INPUT_EXCEL    = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\PC_SmoothCompare\CrossCheck_DoubleRun\All_TT_PlaceChar_SiSkaggsAdptBin_Alpha200.xlsx'
PLACE_CELL_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Alpha200'
OUTPUT_DIR     = os.path.join(PLACE_CELL_DIR, 'Results', 'WallDistanceContinuum')

# Output of TemporalStabilityMap_v6_MapPassStability.py on PLACE_CELL_DIR.
TEMPORAL_STABILITY_EXCEL = os.path.join(PLACE_CELL_DIR, 'Results', 'TemporalStability.xlsx')

# 'Spearman': Spearman correlation per group, linear model on ranks for the pooled tests,
#             logistic regression -- rows treated as independent (like v6's 'KW').
# 'LMM'     : rank LMM with random intercepts animal / day / cell, logistic GEE clustered by cell
#             (slower: one mixed model per metric and group).
TREND_TEST = 'Spearman'

# Models are fitted to 'rank' (ranks of the metric) or 'none' (raw values; the slope is then in
# the metric's units per bin).
TREND_TRANSFORM = 'rank'

# Linear track: the track ends count as walls (first / last column of bins at distance 0).
LINEAR_ENDS_ARE_WALLS = True

# Plots: x-axis width per distance bin (inches) and the largest x offset between the groups at
# one distance (distance bins).
DIST_W       = 0.6
GROUP_OFFSET = 0.22

BIN_CM = ez.BIN_CM
DIST_LABEL = f'Distance from the nearest wall (bins of {BIN_CM:g} cm)'


# ── Wall distance of every bin ──────────────────────────────────────────────

def wall_dist_units(geom, arena: str) -> np.ndarray:
    """Integer wall distance (bins) of every flat bin of `geom`; -1 outside the arena."""
    bx, by = np.divmod(np.arange(geom.n_bins), geom.ny)
    if arena == 'Open':
        radius = ez.OPEN_DIAMETER_CM / 2.0
        n_rings = int(round(radius / BIN_CM))
        r = np.hypot(geom.bin_x - geom.cx, geom.bin_y - geom.cy)
        units = np.clip(np.floor((radius - r) / BIN_CM), 0, n_rings - 1).astype(int)
    elif arena == 'Linear':
        units = np.minimum(by, geom.ny - 1 - by)
        if LINEAR_ENDS_ARE_WALLS:
            units = np.minimum(units, np.minimum(bx, geom.nx - 1 - bx))
    else:
        units = np.minimum(by, geom.ny - 1 - by)
    return np.where(geom.geom_valid, units, -1)


def distance_dwell_pct(units: np.ndarray, sess: dict) -> np.ndarray:
    """% of the dwell in valid bins spent at each wall distance 0 .. units.max()."""
    occ, valid = sess['occ_map'], sess['valid']
    dwell = np.bincount(units[valid], weights=occ[valid], minlength=units.max() + 1)
    total = dwell.sum()
    return 100.0 * dwell / total if total > 0 else np.full(len(dwell), np.nan)


# ── Scan PLACE_CELL_DIR: dwell per distance and every unit's peak distance ──

def scan_place_cell_dir(geoms: dict, units: dict) -> tuple:
    """Dwell per wall distance of every session folder in PLACE_CELL_DIR and the raw-map peak bin
    / wall distance of every .ntt unit in it. Returns (sessions_df, cells_df)."""
    out_dir = os.path.normcase(os.path.abspath(OUTPUT_DIR))
    folders = []
    for dirpath, dirnames, filenames in os.walk(PLACE_CELL_DIR):
        dirnames[:] = [d for d in dirnames
                       if os.path.normcase(os.path.abspath(os.path.join(dirpath, d))) != out_dir]
        ntt_files = sorted(f for f in filenames if f.lower().endswith('.ntt'))
        if ntt_files:
            folders.append((dirpath, ez._tracking_files(filenames), ntt_files))

    session_rows, cell_rows = [], []
    for i, (dirpath, tracking, ntt_files) in enumerate(sorted(folders), start=1):
        rel = os.path.relpath(dirpath, PLACE_CELL_DIR)
        arena = asc.extract_arena_type(rel)
        srow = dict(session=rel, _skey=ez._session_key(rel), arena_type=arena,
                    session_type=asc.extract_session_type(rel, arena),
                    animal=ccs._path_token(rel, ccs.ANIMAL_PATTERN), n_units=len(ntt_files),
                    total_dwell_s=np.nan, note='')
        print(f'[{i}/{len(folders)}] {rel}  ({arena}, {len(ntt_files)} unit(s))')

        sess, dwell = None, None
        if arena is None:
            srow['note'] = 'no recognizable arena in path'
        elif len(tracking) != 1:
            srow['note'] = f'{len(tracking)} tracking files (need exactly 1)'
        else:
            try:
                sess = ez.session_occupancy(os.path.join(dirpath, tracking[0]), geoms[arena], arena)
            except Exception as e:
                srow['note'] = f'tracking error: {e}'
            if sess is None and not srow['note']:
                srow['note'] = 'unusable tracking'
        if sess is not None:
            dwell = distance_dwell_pct(units[arena], sess)
            srow['total_dwell_s'] = round(float(sess['occ_map'][sess['valid']].sum()), 2)
            srow.update({f'dwell_pct_d{k}': round(float(v), 3) for k, v in enumerate(dwell)})
        if srow['note']:
            print(f'    SKIPPED: {srow["note"]}')
        session_rows.append(srow)

        for ntt in ntt_files:
            crow = dict(session=rel, unit=ntt, _skey=srow['_skey'], _ukey=ez._unit_key(ntt),
                        arena_type=arena, session_type=srow['session_type'],
                        animal=srow['animal'], wall_dist=np.nan, peak_dist_dwell_pct=np.nan,
                        excluded_because=srow['note'])
            if sess is not None:
                try:
                    peak = ez.raw_map_peak(sess, ez._load_spike_ts(os.path.join(dirpath, ntt)),
                                           geoms[arena])
                except Exception as e:
                    peak = {}
                    crow['excluded_because'] = f'spike/rate map error: {e}'
                crow.update(peak)
                if peak.get('peak_bin') is not None:
                    d = int(units[arena][peak['peak_bin']])
                    crow.update(wall_dist=d, peak_dist_dwell_pct=round(float(dwell[d]), 3))
                elif not crow['excluded_because']:
                    crow['excluded_because'] = 'no spikes in valid bins'
            cell_rows.append(crow)

    cells_df = pd.DataFrame(cell_rows).drop(columns=['peak_bin', 'peak_in_wall_zone'],
                                            errors='ignore')
    return pd.DataFrame(session_rows), cells_df


def attach_distance(df: pd.DataFrame, cells_df: pd.DataFrame, what: str) -> pd.DataFrame:
    """Add 'wall_dist' (+ peak-bin) columns of cells_df to the Excel rows in df; rows with no
    distance (unit not in PLACE_CELL_DIR, no spikes, unusable tracking) are dropped, with a
    report."""
    df = df.copy()
    df['_skey'] = df['session'].map(ez._session_key)
    df['_ukey'] = df['unit'].map(ez._unit_key)
    zcols = ['_skey', '_ukey', 'arena_type', 'wall_dist', 'peak_dist_dwell_pct',
             'peak_rate_raw_hz', 'peak_bin_x_cm', 'peak_bin_y_cm', 'peak_dist_from_wall_cm',
             'excluded_because']
    z = cells_df[[c for c in zcols if c in cells_df.columns]].rename(
        columns={'arena_type': '_scan_arena'})
    dup = z.duplicated(['_skey', '_ukey'], keep='first')
    if dup.any():
        print(f'WARNING: {int(dup.sum())} unit(s) appear twice in PLACE_CELL_DIR under the same '
              f'animal-relative path; the first is used.')
        z = z[~dup]
    merged = df.merge(z, on=['_skey', '_ukey'], how='left', indicator=True)

    not_found = merged['_merge'] == 'left_only'
    no_dist = ~not_found & merged['wall_dist'].isna()
    print(f'\n{what}: {len(merged)} place-cell row(s) in the Excel file; '
          f'{int(merged["wall_dist"].notna().sum())} with a wall distance, '
          f'{int(not_found.sum())} not found in PLACE_CELL_DIR, {int(no_dist.sum())} excluded:')
    for reason, n in merged.loc[no_dist, 'excluded_because'].fillna('').value_counts().items():
        print(f'    {n:4d}  {reason or "unknown"}')
    if not_found.any():
        print('  Not found in PLACE_CELL_DIR (first 20):')
        for s, u in merged.loc[not_found, ['session', 'unit']].head(20).itertuples(index=False):
            print(f'    {s} | {u}')

    out = merged[merged['wall_dist'].notna()].drop(columns=['_merge'])
    bad = out['arena_type'] != out['_scan_arena']
    if bad.any():
        print(f'WARNING: {int(bad.sum())} row(s) whose Excel arena does not match their folder; dropped.')
        out = out[~bad]
    out['wall_dist'] = out['wall_dist'].astype(int)
    return out.drop(columns=['_scan_arena'])


def add_speed_direction_flags(df: pd.DataFrame) -> pd.DataFrame:
    """'is_p_speed' / 'is_n_speed' (True/False/None) from v10's speed_class."""
    df = df.copy()
    for col, k in (('is_p_speed', 'p'), ('is_n_speed', 'n')):
        df[col] = df['speed_class'].map(lambda c, k=k: None if c is None or pd.isna(c) else c == k)
    return df


def metric_specs() -> list:
    """v6's metrics; the 3-way speed direction becomes two yes/no outcomes."""
    specs = []
    for m in ez.metric_specs():
        if m.kind != 'speed_dir':
            specs.append(m)
            continue
        specs += [ez.Metric('Categorical_p_speed', 'Cat{}', 'cat', 'full', 'is_p_speed',
                            'p-type speed cells (% of place cells)'),
                  ez.Metric('Categorical_n_speed', 'Cat{}', 'cat', 'full', 'is_n_speed',
                            'n-type speed cells (% of place cells)')]
    return specs


# ── Trend tests ─────────────────────────────────────────────────────────────

@dataclass
class DistGroup:
    """One group of cells of a comparison (an arena or a session type): its label in tables and
    plots and its key in v10's GROUP_COLORS."""
    label: str
    key: str
    full_df: pd.DataFrame
    field_df: pd.DataFrame

    def values(self, m) -> pd.DataFrame:
        """Cells with a value of m.col, in '_y' (numeric; 1.0 / 0.0 for yes/no metrics), and the
        group label in 'grp'."""
        sub = self.field_df if m.source == 'field' else self.full_df
        if m.stab_type:
            sub = sub[sub['stability_type'] == m.stab_type]
        sub = sub.copy()
        if m.kind == 'box':
            sub['_y'] = pd.to_numeric(sub[m.col], errors='coerce')
        else:
            sub['_y'] = sub[m.col].apply(asc._to_tri_bool).map(
                lambda v: np.nan if v is None else float(v))
        sub['grp'] = self.label
        return sub.dropna(subset=['_y'])


def _levels_str(d) -> str:
    lv = np.unique(d)
    return f'{lv[0]}' if len(lv) == 1 else f'{lv[0]}-{lv[-1]}'


def _p_str(p) -> str:
    return f'p = {asc._fmt_p(p)} {asc._stars(p)}'


def _model_data(sub: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Model frame: y, dist, grp and the nesting ids (as in CellClusteredStats_Utils; rows whose
    animal / day could not be parsed are their own unit)."""
    sess = 'unparsed:' + sub['session'].astype(str)
    animal = sub['animal'].astype(object).where(sub['animal'].notna(), sess).astype(str)
    day = np.where(sub['rec_day'].notna(),
                   animal + '|' + sub['rec_day'].astype(str).str.lower(), sess)
    return pd.DataFrame({'y': y, 'dist': sub['wall_dist'].to_numpy(float),
                         'grp': sub['grp'].astype(str).to_numpy(), 'animal': animal.to_numpy(),
                         'day': day, 'cell': sub['cell_id'].astype(str).to_numpy()})


def _random_levels(data: pd.DataFrame) -> list:
    if TREND_TEST != 'LMM':
        return []
    n_anim, n_day = data['animal'].nunique(), data['day'].nunique()
    n_rep = int((data.groupby('cell').size() > 1).sum())
    return ((['animal'] if n_anim > 1 else []) + (['day'] if n_day > n_anim else [])
            + (['cell'] if n_rep else []))


def rank_model_test(sub: pd.DataFrame, full: str, reduced: str) -> dict:
    """Likelihood-ratio test of `full` vs `reduced` on the ranks (TREND_TRANSFORM) of sub['_y'],
    standardized; mixed model with nested random intercepts if TREND_TEST == 'LMM' (the
    outermost level is dropped until the fit works), else a linear model. 'slope' = the 'dist'
    coefficient of the full model (REML fit), in SD of the ranks per bin."""
    y = sub['_y'].to_numpy(float)
    if TREND_TRANSFORM == 'rank':
        y = stats.rankdata(y)
    if not y.std() > 0:
        return dict(test='n/a', text='metric is constant')
    data = _model_data(sub, (y - y.mean()) / y.std())
    re_full = _random_levels(data)

    final, err = None, None
    for re_levels in ([re_full[i:] for i in range(len(re_full))] or [[]]):
        try:
            ml_full = ccs._fit_lmm(data, full, re_levels, reml=False)
            ml_red  = ccs._fit_lmm(data, reduced, re_levels, reml=False)
            final   = ccs._fit_lmm(data, full, re_levels, reml=True)
            break
        except Exception as e:  # noqa: BLE001
            final, err = None, e
    if final is None:
        return dict(test='n/a', text=f'model failed ({err})', model_note=f'model failed ({err})')

    def _n_fe(res):
        return len(res.fe_params) if hasattr(res, 'fe_params') else len(res.params)

    lrt = max(2.0 * (ml_full.llf - ml_red.llf), 0.0)
    dof = _n_fe(ml_full) - _n_fe(ml_red)
    p = float(stats.chi2.sf(lrt, dof))
    what = 'ranks' if TREND_TRANSFORM == 'rank' else 'raw values'
    if re_levels:
        note = (f'LMM on {what}, random intercepts: '
                f'{" / ".join(ccs._RE_LABELS[l] for l in re_levels)}')
        try:
            note += ccs._variance_note(final, re_levels)
        except Exception:  # noqa: BLE001
            pass
        test = 'LMM likelihood-ratio test'
    else:
        note, test = f'Linear model on {what} (rows treated as independent)', 'Linear model LRT'
    slope = float(final.params['dist']) if 'dist' in final.params.index else np.nan
    se = float(final.bse['dist']) if 'dist' in final.bse.index else np.nan
    # The slope is the tested effect only when the reduced model has no distance term.
    text = f'slope = {slope:+.3f} SD/bin, ' if pd.notna(slope) and 'dist' not in reduced else ''
    return dict(test=test, statistic=lrt, stat_name='χ²', dof=dof, p_value=p, slope_sd_per_bin=slope,
                slope_se=se, model_note=note, text=f'{text}χ²({dof}) = {lrt:.2f}, {_p_str(p)}')


def logit_test(sub: pd.DataFrame, formula: str, interaction: bool = False) -> dict:
    """Wald test of the 'dist' coefficient (or, interaction=True, of every 'dist:group'
    coefficient) of a logistic model of sub['_y']; GEE clustered by cell if TREND_TEST == 'LMM'
    and some cell repeats, else ordinary logistic regression."""
    data = _model_data(sub, sub['_y'].to_numpy(float))
    clustered = TREND_TEST == 'LMM' and bool((data.groupby('cell').size() > 1).any())
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if clustered:
                res = smf.gee(formula, 'cell', data, family=sm.families.Binomial(),
                              cov_struct=sm.cov_struct.Exchangeable()).fit()
                test = 'Logistic GEE Wald test (clustered by cell)'
            else:
                res = smf.glm(formula, data, family=sm.families.Binomial()).fit()
                test = 'Logistic regression Wald test (rows independent)'
        names = ([n for n in res.params.index if ':' in n] if interaction else ['dist'])
        b = res.params[names].to_numpy(float)
        V = res.cov_params().loc[names, names].to_numpy(float)
        if not (np.all(np.isfinite(b)) and np.all(np.isfinite(V))):
            raise ValueError('non-finite estimates')
        wald = float(b @ np.linalg.solve(V, b))
    except Exception as e:  # noqa: BLE001
        return dict(test='n/a', text=f'logistic model failed ({e})',
                    model_note=f'logistic model failed ({e})')
    dof = len(names)
    p = float(stats.chi2.sf(wald, dof))
    out = dict(test=test, statistic=wald, stat_name='χ²', dof=dof, p_value=p, model_note=test)
    text = f'χ²({dof}) = {wald:.2f}, {_p_str(p)}'
    if not interaction:
        with np.errstate(over='ignore'):   # near-separated fits: CI overflows to inf
            out.update(odds_ratio_per_bin=float(np.exp(b[0])),
                       or_ci_low=float(np.exp(b[0] - 1.96 * np.sqrt(V[0, 0]))),
                       or_ci_high=float(np.exp(b[0] + 1.96 * np.sqrt(V[0, 0]))))
        text = f'OR = {out["odds_ratio_per_bin"]:.3f} per bin, ' + text
    out['text'] = text
    return out


def group_trend(sub: pd.DataFrame, m) -> dict:
    """Trend of one metric with wall_dist within one group."""
    d, y = sub['wall_dist'].to_numpy(float), sub['_y'].to_numpy(float)
    row = dict(n=len(sub), n_cells=sub['cell_id'].nunique(), n_dist_levels=len(np.unique(d)),
               dist_levels=_levels_str(d.astype(int)) if len(d) else '')
    if row['n_dist_levels'] < 2:
        return dict(row, test='n/a', text='one distance level, no trend test')
    if len(sub) < 3 or np.ptp(y) == 0:
        return dict(row, test='n/a', text='too few values / constant')
    if m.kind != 'box':
        return dict(row, **logit_test(sub, 'y ~ dist'))

    rho, p_s = stats.spearmanr(d, y)
    row.update(spearman_rho=float(rho), spearman_p=float(p_s))
    if TREND_TEST == 'LMM':
        return dict(row, **rank_model_test(sub, 'y ~ dist', 'y ~ 1'))
    return dict(row, test='Spearman rank correlation', statistic=float(rho), stat_name='ρ',
                p_value=float(p_s), model_note='Spearman (rows treated as independent)',
                text=f'Spearman ρ = {rho:+.3f}, {_p_str(p_s)}')


def pooled_trend(subs: list, m) -> tuple:
    """(group-adjusted distance effect, distance x group interaction) over the groups in `subs`
    with >= 2 distance levels, or (None, None) if fewer than 2 such groups."""
    subs = [s for s in subs if s['wall_dist'].nunique() >= 2 and np.ptp(s['_y'].to_numpy()) > 0]
    if len(subs) < 2:
        return None, None
    sub = pd.concat(subs, ignore_index=True)
    info = dict(groups=', '.join(s['grp'].iloc[0] for s in subs), n=len(sub),
                n_cells=sub['cell_id'].nunique())
    if m.kind == 'box':
        main = rank_model_test(sub, 'y ~ dist + C(grp)', 'y ~ C(grp)')
        inter = rank_model_test(sub, 'y ~ dist * C(grp)', 'y ~ dist + C(grp)')
    else:
        main = logit_test(sub, 'y ~ dist + C(grp)')
        inter = logit_test(sub, 'y ~ dist * C(grp)', interaction=True)
    return dict(info, **main), dict(info, **inter)


def matched_distance(subs: list, m, labels: list) -> tuple:
    """v10's group comparison at every distance held by >= 2 groups. Returns (omnibus rows,
    posthoc frames)."""
    if len(subs) < 2:
        return [], []
    sub = pd.concat(subs, ignore_index=True)
    omni_rows, posthoc = [], []
    for d in sorted(sub['wall_dist'].unique()):
        at_d = sub[sub['wall_dist'] == d]
        present = [g for g in labels if (at_d['grp'] == g).any()]
        if len(present) < 2:
            continue
        grp = asc.Grouping('grp', present, 'groups')
        if m.kind == 'box':
            _, omni, ph = asc.compare_groups(at_d, '_y', m.label, grp)
        else:
            _, omni, ph = asc.compare_categorical(at_d, '_y', m.label, grp)
        omni_rows.append(dict(wall_dist=int(d), **omni))
        if len(ph):
            ph.insert(0, 'wall_dist', int(d))
            posthoc.append(ph)
    return omni_rows, posthoc


def describe(sub: pd.DataFrame, m) -> list:
    """Descriptives of one group's metric at every distance."""
    rows = []
    for d, at_d in sub.groupby('wall_dist'):
        y = at_d['_y'].to_numpy(float)
        r = dict(wall_dist=int(d), n=len(y), n_cells=at_d['cell_id'].nunique())
        if m.kind == 'box':
            r.update(mean=np.mean(y), median=np.median(y), q1=np.percentile(y, 25),
                     q3=np.percentile(y, 75), std=np.std(y, ddof=1) if len(y) > 1 else np.nan,
                     min=np.min(y), max=np.max(y))
        else:
            r.update(n_true=int(y.sum()), n_false=int(len(y) - y.sum()),
                     pct_true=100.0 * y.mean())
        rows.append(r)
    return rows


# ── Plots ───────────────────────────────────────────────────────────────────

def _offsets(n: int) -> np.ndarray:
    step = min(GROUP_OFFSET, 0.75 / max(n, 1))
    return (np.arange(n) - (n - 1) / 2.0) * step


def _wilson(k: float, n: int, z: float = 1.96) -> tuple:
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def _stats_lines(res: dict, noun: str) -> list:
    lines = [f'{label} (n={r["n"]}, distance {r["dist_levels"]}): {r["text"]}'
             for label, r in res['trend'].items()]
    if res['pooled'] is not None:
        lines.append(f'Pooled ({res["pooled"]["groups"]}), {noun}-adjusted distance effect: '
                     f'{res["pooled"]["text"]}')
        lines.append(f'Distance × {noun} interaction (do the slopes differ?): '
                     f'{res["interaction"]["text"]}')
    matched = [f'd={o["wall_dist"]}: {_p_str(o["p_value"])}' for o in res['matched']
               if pd.notna(o.get('p_value'))]
    if matched:
        test = res['matched'][0].get('test', '')
        lines.append(f'{noun.capitalize()}s at matched distance ({test}): ' + '; '.join(matched))
    return lines


def _trend_fig(data: list, lines: list, max_d: int):
    fig_w = max(6.0, (max_d + 1) * DIST_W + asc.AXIS_PAD_W)
    lines = asc._wrap_stats(lines, fig_w)
    legend_rows = int(np.ceil(len(data) / 3))
    fig, ax, ax_info = asc._new_fig((fig_w, 5.0), legend_rows=legend_rows, stats_lines=lines)
    ez._draw_stats_top(ax_info, lines, legend_rows)
    return fig, ax, ax_info


def _legend_handle(g: DistGroup, n: int, r: dict) -> Line2D:
    c = asc._color(g.key)
    stars = asc._stars(r.get('p_value')) if r.get('test') != 'n/a' else ''
    return Line2D([0], [0], marker='D', color=c['dark'], markerfacecolor=c['dark'],
                  markeredgecolor=asc.PAL_BLACK, markeredgewidth=0.6, linewidth=1.5,
                  label=f'{g.label} (n={n}) {stars}'.strip())


def _finish_trend(fig, ax, ax_info, handles, max_d, ylabel, title, out_path):
    ax_info.legend(handles=handles, loc='upper center', ncol=min(len(handles), 3),
                   fontsize=asc.FS_LEGEND)
    ax.set_xticks(range(max_d + 1))
    ax.set_xlim(-0.6, max_d + 0.6)
    ax.set_xlabel(DIST_LABEL, fontsize=asc.FS_LABEL, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=asc.FS_LABEL, labelpad=8)
    asc._set_title(ax, title)
    asc._style_axes(ax)
    fig.tight_layout()
    asc._save(fig, out_path)


def plot_continuous_trend(groups: list, m, res: dict, noun: str, out_path: str, title: str):
    """Every cell as a point at its wall distance, median + IQR of each group at each distance."""
    data = [(g, s) for g, s in ((g, g.values(m)) for g in groups) if len(s)]
    if not data:
        return
    max_d = max(int(s['wall_dist'].max()) for _, s in data)
    lines = _stats_lines(res, noun)
    if m.col in asc.SIGNIFICANCE_METRICS:
        lines.append('Gray points: failed their significance / shuffle test (as in v10).')
    fig, ax, ax_info = _trend_fig(data, lines, max_d)

    rng = np.random.default_rng(0)
    handles = []
    for (g, sub), off in zip(data, _offsets(len(data))):
        c = asc._color(g.key)
        d, y = sub['wall_dist'].to_numpy(float), sub['_y'].to_numpy(float)
        colors = asc._point_colors(m.col, sub, g.key)
        ax.scatter(d + off + rng.uniform(-0.05, 0.05, len(d)), y, s=14, alpha=0.75,
                   facecolors=[fc for fc, _ec in colors], edgecolors=[ec for _fc, ec in colors],
                   linewidths=0.4, zorder=2)
        lv = np.unique(d)
        med = np.array([np.median(y[d == l]) for l in lv])
        q1 = np.array([np.percentile(y[d == l], 25) for l in lv])
        q3 = np.array([np.percentile(y[d == l], 75) for l in lv])
        ax.errorbar(lv + off, med, yerr=[med - q1, q3 - med], fmt='-D', color=c['dark'],
                    markersize=6, markeredgecolor=asc.PAL_BLACK, markeredgewidth=0.6,
                    linewidth=1.5, elinewidth=1.2, capsize=3, zorder=4)
        handles.append(_legend_handle(g, len(sub), res['trend'][g.label]))
    _finish_trend(fig, ax, ax_info, handles, max_d, textwrap.fill(m.label, 30),
                  f'{title}: {m.label}', out_path)


def plot_binary_trend(groups: list, m, res: dict, noun: str, out_path: str, title: str):
    """% yes of each group at each wall distance, Wilson 95 % CI."""
    data = [(g, s) for g, s in ((g, g.values(m)) for g in groups) if len(s)]
    if not data:
        return
    max_d = max(int(s['wall_dist'].max()) for _, s in data)
    fig, ax, ax_info = _trend_fig(data, _stats_lines(res, noun) + ['Error bars: Wilson 95 % CI.'],
                                  max_d)
    handles = []
    for (g, sub), off in zip(data, _offsets(len(data))):
        c = asc._color(g.key)
        lv, pct, lo, hi = [], [], [], []
        for d, at_d in sub.groupby('wall_dist'):
            k, n = at_d['_y'].sum(), len(at_d)
            a, b = _wilson(k, n)
            lv.append(d)
            pct.append(100.0 * k / n)
            lo.append(100.0 * a)
            hi.append(100.0 * b)
        lv, pct = np.array(lv, float), np.array(pct)
        yerr = np.clip([pct - np.array(lo), np.array(hi) - pct], 0, None)   # rounding at 0 / 100 %
        ax.errorbar(lv + off, pct, yerr=yerr, fmt='-D',
                    color=c['dark'], markerfacecolor=c['face'], markersize=6,
                    markeredgecolor=asc.PAL_BLACK, markeredgewidth=0.6, linewidth=1.5,
                    elinewidth=1.2, capsize=3, zorder=4)
        handles.append(_legend_handle(g, len(sub), res['trend'][g.label]))
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    _finish_trend(fig, ax, ax_info, handles, max_d, '% of cells', f'{title}: {m.label}', out_path)


def plot_peak_distribution(full_df: pd.DataFrame, sessions_df: pd.DataFrame, geoms: dict,
                           units: dict, out_path: str) -> pd.DataFrame:
    """Per arena: % of place-cell rows whose peak is at each wall distance, vs the % expected
    from where the animals were (sum over rows of their session's dwell fraction at each
    distance) and the % of the arena's bins at that distance; chi-square goodness of fit of the
    observed against the dwell-expected counts. Returns the per-distance table."""
    sess = sessions_df.drop_duplicates('_skey').set_index('_skey')
    arenas = [a for a in asc.ARENA_TYPES if (full_df['arena_type'] == a).any()]
    if not arenas:
        return pd.DataFrame()
    widths = [max(units[a].max() + 1, 4) for a in arenas]
    fig, axes = plt.subplots(1, len(arenas), figsize=(max(7.0, 0.5 * sum(widths) + 2.5), 4.6),
                             gridspec_kw={'width_ratios': widths}, squeeze=False)
    rows = []
    for arena, ax in zip(arenas, axes[0]):
        sub = full_df[full_df['arena_type'] == arena]
        u = units[arena]
        levels = np.arange(u.max() + 1)
        obs = np.bincount(sub['wall_dist'], minlength=len(levels)).astype(float)
        dcols = [f'dwell_pct_d{k}' for k in levels]
        frac = sess.reindex(sub['_skey'])[dcols].to_numpy(float) / 100.0
        exp = np.nansum(frac, axis=0)
        exp *= obs.sum() / exp.sum() if exp.sum() > 0 else np.nan
        area = 100.0 * np.bincount(u[u >= 0], minlength=len(levels)) / (u >= 0).sum()

        ok = exp > 0
        gof = ''
        if ok.sum() >= 2:
            chi2, p = stats.chisquare(obs[ok], exp[ok] * obs[ok].sum() / exp[ok].sum())
            n_low = int((exp[ok] < 5).sum())
            gof = f'\nvs dwell: χ²({ok.sum() - 1}) = {chi2:.1f}\n{_p_str(p)}'
            if n_low:
                gof += f' ({n_low} exp. < 5)'
        else:
            chi2 = p = np.nan
        for k in levels:
            rows.append(dict(arena_type=arena, wall_dist=int(k), n_rows_observed=int(obs[k]),
                             pct_observed=100.0 * obs[k] / obs.sum(), n_rows_expected_dwell=exp[k],
                             pct_expected_dwell=100.0 * exp[k] / obs.sum(), pct_arena_bins=area[k],
                             gof_chi2=chi2, gof_p=p))

        c = asc._color(arena)
        ax.bar(levels, 100.0 * obs / obs.sum(), color=c['face'], edgecolor=asc.BAR_EDGE,
               linewidth=asc.BAR_EDGE_LW, width=0.8, zorder=2, label='Place-cell peaks')
        ax.plot(levels, 100.0 * exp / obs.sum(), '-o', color=c['dark'], markersize=5,
                markeredgecolor=asc.PAL_BLACK, markeredgewidth=0.5, linewidth=1.5, zorder=3,
                label='Expected from dwell')
        ax.plot(levels, area, linestyle=(0, (3, 2)), marker='s', color=asc.PAL_GRAY,
                markersize=4, linewidth=1.2, zorder=3, label='% of arena bins')
        ax.set_xticks(levels)
        ax.set_xlim(-0.6, max(levels[-1], 1) + 0.6)
        ax.set_title(f'{arena} (n={int(obs.sum())}){gof}', fontsize=asc.FS_LABEL - 1, pad=8)
        asc._style_axes(ax)
    axes[0][0].set_ylabel('% of place-cell rows', fontsize=asc.FS_LABEL)
    axes[0][-1].legend(loc='upper right', fontsize=asc.FS_LEGEND - 1)
    fig.supxlabel(DIST_LABEL, fontsize=asc.FS_LABEL, fontweight='bold')
    fig.tight_layout()
    asc._save(fig, out_path)
    print(f'Peak-distance distribution saved to {out_path}')
    return pd.DataFrame(rows)


def plot_distance_schematics(geoms: dict, units: dict, out_dir: str):
    """OUTPUT_DIR\\WallDistanceSchematic_<arena>.png: every bin shaded by its wall distance (dark
    = at the wall, light = far from it), numbered (linear track: every bin; open field: along
    one radius)."""
    for arena, geom in geoms.items():
        u = units[arena]
        patches, idx = ez._bin_patches(geom, arena)
        base = asc.ARENA_COLORS[arena]
        cmap = LinearSegmentedColormap.from_list(f'dist_{arena}', [base['dark'], base['light']])
        norm = Normalize(0, max(1, u.max()))
        fig, ax = plt.subplots(figsize=(9.0, 2.4) if arena == 'Linear' else (6.2, 5.5))
        ax.add_collection(PatchCollection(patches, facecolors=cmap(norm(u[idx])),
                                          edgecolors='white', linewidths=0.4, zorder=2))
        wall_kw = dict(fill=False, edgecolor=asc.PAL_BLACK, linewidth=asc.BOX_LW, zorder=3)
        if arena == 'Linear':
            ax.add_patch(Rectangle((0, 0), geom.length, geom.width, **wall_kw))
            w, h = geom.length, geom.width
            numbered = idx
            ax.set_xlabel('Along the track (cm)', fontsize=asc.FS_LABEL)
            ax.set_ylabel('Width\n(cm)', fontsize=asc.FS_LABEL, rotation=0, ha='right', va='center')
        else:
            radii = ((ez.OPEN_DIAMETER_CM / 2.0,) if arena == 'Open' else (geom.outer_r, geom.inner_r))
            for r in radii:
                ax.add_patch(Circle((geom.cx, geom.cy), r, **wall_kw))
            w = h = 2.0 * radii[0]
            numbered = (idx[(np.divmod(idx, geom.ny)[1] == geom.ny // 2) & (geom.bin_x[idx] > geom.cx)]
                        if arena == 'Open' else [])
            ax.set_xlabel('x (cm)', fontsize=asc.FS_LABEL)
            ax.set_ylabel('y (cm)', fontsize=asc.FS_LABEL)
        for b in numbered:
            x, y = ((geom.bin_x[b], geom.bin_y[b]) if arena != 'Linear' else
                    ((b // geom.ny + 0.5) * geom.bin_cm_x, (b % geom.ny + 0.5) * geom.bin_cm_y))
            ax.text(x, y, str(u[b]), ha='center', va='center', fontsize=5, zorder=4,
                    color=asc.PAL_BLACK if norm(u[b]) > 0.5 else 'white')

        pad = 2.0
        ax.set_xlim(-pad, w + pad)
        ax.set_ylim(-pad, h + pad)
        ax.set_aspect('equal')
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8, pad=0.03,
                          ticks=range(u.max() + 1))
        cb.set_label('Wall distance (bins)', fontsize=asc.FS_LABEL - 2)
        ax.set_title(f'{arena} arena: distance from the nearest wall\n'
                     f'({ez.ARENA_SIZE_NOTES[arena]}; {BIN_CM:g} x {BIN_CM:g} cm bins; '
                     f'levels 0-{u.max()})', fontsize=asc.FS_TITLE - 2, pad=12)
        asc._style_axes(ax)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f'WallDistanceSchematic_{arena}.png')
        asc._save(fig, out_path)
        print(f'Wall-distance schematic saved to {out_path}')


# ── One comparison ──────────────────────────────────────────────────────────

def run_comparison(info: dict, groups: list, metrics: list, noun: str, title: str,
                   plots_dir: str, tables: dict):
    """Trend, pooled and matched-distance tests of every metric over `groups`; table rows are
    appended to `tables` with `info` in the first columns, plots go to `plots_dir`."""
    groups = [g for g in groups if len(g.full_df)]
    print(f'\n{"=" * 70}\n{title}\n{"=" * 70}')
    if not groups:
        print('No place cells; skipping.')
        return
    for g in groups:
        counts = g.full_df.groupby('wall_dist').agg(n_rows=('cell_id', 'size'),
                                                    n_cells=('cell_id', 'nunique')).reset_index()
        print(f'  {g.label}: ' + ', '.join(f'd{r.wall_dist}={r.n_rows}' for r in counts.itertuples()))
        tables['DistCounts'] += [dict(info, group=g.label, **r) for r in counts.to_dict('records')]

    labels = [g.label for g in groups]
    results, trend_rows = {}, []
    for m in metrics:
        subs = {g.label: g.values(m) for g in groups}
        subs = {k: s for k, s in subs.items() if len(s)}
        res = dict(trend={}, pooled=None, interaction=None, matched=[])
        for label, sub in subs.items():
            res['trend'][label] = group_trend(sub, m)
            trend_rows.append(dict(info, group=label, metric=m.label, metric_col=m.col,
                                   stab_type=m.stab_type or '', kind=m.kind,
                                   **res['trend'][label]))
            tables['Desc'] += [dict(info, group=label, metric=m.label, **r)
                               for r in describe(sub, m)]
        res['pooled'], res['interaction'] = pooled_trend(list(subs.values()), m)
        for what, r in (('distance (group-adjusted)', res['pooled']),
                        ('distance x group interaction', res['interaction'])):
            if r is not None:
                tables['Pooled'].append(dict(info, metric=m.label, metric_col=m.col,
                                             effect=what, **r))
        res['matched'], posthoc = matched_distance(list(subs.values()), m, labels)
        tables['Matched_Omnibus'] += [dict(info, **o) for o in res['matched']]
        tables['Matched_PostHoc'] += [dict(info, **r) for ph in posthoc
                                      for r in ph.to_dict('records')]
        results[m.stem] = res

    # Holm across the metrics of each group.
    trend_df = pd.DataFrame(trend_rows)
    if 'p_value' not in trend_df.columns:
        trend_df['p_value'] = np.nan
    trend_df['p_holm_metrics'] = trend_df.groupby('group')['p_value'].transform(
        lambda p: ccs.holm(p.to_numpy(float)))
    trend_df['significant'] = trend_df['p_value'] < asc.ALPHA
    tables['Trend'] += trend_df.drop(columns=['text']).to_dict('records')

    sig = trend_df[trend_df['significant']]
    print(f'  Significant trends with wall distance (p < {asc.ALPHA}): {len(sig)}')
    for r in sig.itertuples():
        print(f'    {r.group:12s} {r.metric[:60]:60s} p = {asc._fmt_p(r.p_value)} '
              f'(Holm over metrics {asc._fmt_p(r.p_holm_metrics)})')

    if not any(g.full_df['wall_dist'].nunique() >= 2 for g in groups):
        print('  Every group has a single distance level; no plots.')
        return
    os.makedirs(plots_dir, exist_ok=True)
    for m in metrics:
        plot = plot_continuous_trend if m.kind == 'box' else plot_binary_trend
        plot(groups, m, results[m.stem], noun, os.path.join(plots_dir, f'{m.stem}.png'), title)
    print(f'  Plots saved to {plots_dir}')


def main():
    asc.CONTINUOUS_TEST = 'LMM' if TREND_TEST == 'LMM' else 'KW'   # matched-distance tests
    asc.METRIC_TRANSFORM = TREND_TRANSFORM
    metrics = metric_specs()

    geoms = ez.make_geometries()
    units = {a: wall_dist_units(g, a) for a, g in geoms.items()}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_distance_schematics(geoms, units, OUTPUT_DIR)

    print(f'Scanning {PLACE_CELL_DIR} ...')
    sessions_df, cells_df = scan_place_cell_dir(geoms, units)
    print(f'\n{len(sessions_df)} session folder(s), '
          f'{int(sessions_df["total_dwell_s"].notna().sum())} with usable tracking.')

    print(f'\nLoading {INPUT_EXCEL} ...')
    full = pd.read_excel(INPUT_EXCEL, sheet_name='Full')
    fields = pd.read_excel(INPUT_EXCEL, sheet_name='PlaceFields')
    full_df = add_speed_direction_flags(
        attach_distance(asc.build_full_metrics_df(full), cells_df, 'Full sheet'))
    field_df = attach_distance(asc.build_field_metrics_df(full, fields), cells_df, 'PlaceFields sheet')

    print(f'\nLoading {TEMPORAL_STABILITY_EXCEL} ...')
    ts_df = ez.load_temporal_stability(TEMPORAL_STABILITY_EXCEL)
    cells_df = cells_df.merge(ts_df, on=['_skey', '_ukey'], how='left')
    full_df = full_df.merge(ts_df, on=['_skey', '_ukey'], how='left')

    print('\nPlace cells by arena and wall distance (Full sheet):')
    print(pd.crosstab(full_df['arena_type'], full_df['wall_dist'], margins=True).to_string())

    tables = {k: [] for k in ('DistCounts', 'Trend', 'Pooled', 'Matched_Omnibus',
                              'Matched_PostHoc', 'Desc')}

    def _sel(d, arena, extra=None):
        mask = d['arena_type'] == arena
        return d[mask & extra(d)] if extra else d[mask]

    # ── 1. Arenas, all cells ────────────────────────────────────────────────
    groups = [DistGroup(a, a, _sel(full_df, a), _sel(field_df, a)) for a in asc.ARENA_TYPES]
    run_comparison(dict(level='Arena', comparison='Arena', animals='all'), groups, metrics,
                   'arena', 'Wall distance, all cells', os.path.join(OUTPUT_DIR, 'Arena'), tables)
    occ_df = plot_peak_distribution(full_df, sessions_df, geoms, units,
                                    os.path.join(OUTPUT_DIR, 'PeakDistance_vs_Dwell.png'))

    # ── 2. Session types within each arena ──────────────────────────────────
    for arena in asc.ARENA_TYPES:
        groups = []
        for stype in asc.ARENA_SESSION_TYPES[arena]:
            def _is(d, stype=stype):
                return d['session_type'] == stype
            groups.append(DistGroup(asc._label(stype), stype, _sel(full_df, arena, _is),
                                    _sel(field_df, arena, _is)))
        run_comparison(dict(level='SessionType', comparison=f'{arena} session types',
                            animals='all'), groups, metrics, 'session type',
                       f'Wall distance, {arena} arena by session type',
                       os.path.join(OUTPUT_DIR, 'SessionType', arena), tables)

    # ── 3. Animal-controlled: only the animals recorded in all compared arenas ──
    for name, (animals, arenas) in asc.ANIMAL_CONTROLLED_COMPARISONS.items():
        wanted = {a.lower() for a in animals}

        def _of(d):
            return d['animal'].astype(str).str.lower().isin(wanted)
        groups = [DistGroup(a, a, _sel(full_df, a, _of), _sel(field_df, a, _of))
                  for a in asc.ARENA_TYPES if a in arenas]
        run_comparison(dict(level='AnimalControlled', comparison=name, animals=', '.join(animals)),
                       groups, metrics, 'arena', f'Wall distance, {" + ".join(animals)} only',
                       os.path.join(OUTPUT_DIR, asc.ANIMAL_CONTROLLED_DIR, name), tables)

    # ── Save ────────────────────────────────────────────────────────────────
    stat_tables = {k: pd.DataFrame(v) for k, v in tables.items()}
    sheets = {'DistCounts': stat_tables.pop('DistCounts'),
              'DistSessions': sessions_df.drop(columns=['_skey']),
              'DistCells': cells_df.drop(columns=['_skey', '_ukey']),
              'PeakDistOccupancy': occ_df}
    sheets.update(stat_tables)
    stats_xlsx = os.path.join(OUTPUT_DIR, 'WallDistance_Stats.xlsx')
    ez._write_workbook(stats_xlsx, sheets)
    print(f'\nWall distances and all statistics saved to {stats_xlsx}')

    parts = []
    for name, tbl in {'PeakDistOccupancy': occ_df, **stat_tables}.items():
        if len(tbl):
            tbl = tbl.copy()
            tbl.insert(0, 'table', name)
            parts.append(tbl)
    if parts:
        stats_csv = os.path.join(OUTPUT_DIR, 'WallDistance_AllStats.csv')
        pd.concat(parts, ignore_index=True).to_csv(stats_csv, index=False, encoding='utf-8-sig')
        print(f'All statistics also saved to {stats_csv}')
    print(f'\nDone. Plots saved under {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
