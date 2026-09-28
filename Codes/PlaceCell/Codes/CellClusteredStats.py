# -*- coding: utf-8 -*-
"""
Repeated-measures-aware statistics shared by ArenaType_StatsComparison_v6.py
and SessionType_StatsComparison_v5.py.

Every row of the characterization workbook is one unit in one session. Within
a recording day the same unit is recorded in several sessions (spike sorting
spans the day's sessions, so the unit label, e.g. 'TT6_SS_03_SS_01.ntt', names
the same cell in every session of that day). Rows of the same cell are
therefore NOT independent, which Kruskal-Wallis / Mann-Whitney / chi-square /
Fisher all assume. Some cells appear once, some several times, so the data are
partially paired: Friedman (needs every cell in every group) and Kruskal-Wallis
(needs independence) are both wrong. Instead:

  Continuous metrics -- rank-based linear mixed model (LMM):
      rank(metric) ~ group + (1 | cell)
    The metric is rank-transformed over all rows being compared (average ranks
    for ties), so -- like Kruskal-Wallis -- it compares groups on ranks and is
    robust to skew/outliers; with a single fixed factor this is exactly the
    Aligned Rank Transform. The random intercept per cell absorbs the
    correlation between repeated rows of the same cell; cells seen once still
    contribute. Omnibus: likelihood-ratio test (ML fits, full vs. no-group
    model), chi-square with k-1 dof. Post hoc: pairwise Wald z-contrasts of the
    group coefficients (REML fit), Holm-Bonferroni corrected.

  Yes/no outcomes (percentages) -- logistic GEE, clustered by cell:
      logit P(outcome) ~ group, exchangeable working correlation within cell,
      cluster-robust (sandwich) standard errors.
    Omnibus: Wald chi-square (k-1 dof). Post hoc: pairwise Wald z-tests on the
    log-odds difference (reported as an odds ratio), Holm-Bonferroni corrected.

  3-category outcome (p-/n-/non-speed) -- multinomial (baseline-logit) GEE,
    clustered by cell, independence working correlation + sandwich SEs.
    Omnibus: Wald chi-square, 2(k-1) dof. Post hoc: pairwise 2-dof Wald tests,
    Holm-Bonferroni corrected.

Fallbacks (recorded in each result's 'model_note' / test name):
  - no cell appears more than once: the rows are independent, so the classic
    tests are valid and used (Kruskal-Wallis + Mann-Whitney U; chi-square +
    Fisher's exact / pairwise chi-square).
  - a group at 0 % or 100 % (logistic/multinomial models cannot be estimated,
    'separation') or a model that fails to fit: classic tests are used and the
    note says independence was ASSUMED, so interpret those p-values cautiously.
"""

import re
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import NominalGEE
from statsmodels.stats.multitest import multipletests

ALPHA = 0.05

# Tokens of the session path (split on \ / _) that name the animal and the
# recording day, e.g. 'Fa1059\\Linear\\Day10\\1_0' or
# 'Cntrl\\Open\\Fa1059_Day11_3Cntrl' -> animal 'Fa1059', day 'Day10'/'Day11'.
ANIMAL_PATTERN = r'^Fa[0-9A-Za-z]+$'
DAY_PATTERN    = r'^(?:Exp)?Day\d+$'


# ── Cell identity ────────────────────────────────────────────────────────────

def _path_token(session_path: str, pattern: str):
    for tok in re.split(r'[\\/_]+', str(session_path)):
        if re.match(pattern, tok.strip(), flags=re.IGNORECASE):
            return tok.strip()
    return None


def add_cell_ids(df: pd.DataFrame, include_arena: bool = True) -> pd.DataFrame:
    """Add 'animal', 'rec_day' and 'cell_id' columns. cell_id =
    animal | [arena_type |] day | unit, so rows with the same cell_id are the
    same unit recorded in different sessions of one day. With
    include_arena=False, a unit recorded in two arenas on the same day (same
    label) is also treated as one cell. Rows whose animal/day can't be parsed
    get a cell_id unique to that row (treated as independent)."""
    df = df.copy()
    df['animal']  = df['session'].apply(lambda s: _path_token(s, ANIMAL_PATTERN))
    df['rec_day'] = df['session'].apply(lambda s: _path_token(s, DAY_PATTERN))

    ok = df['animal'].notna() & df['rec_day'].notna()
    if (~ok).any():
        print(f'WARNING: animal/day not recognized in {int((~ok).sum())} row(s) '
              f'(patterns {ANIMAL_PATTERN!r}, {DAY_PATTERN!r}); those rows are '
              f'treated as independent cells:')
        for s in sorted(df.loc[~ok, 'session'].astype(str).unique()):
            print(f'    {s}')

    unit = df['unit'].astype(str).str.strip().str.lower()
    parts = [df['animal'].astype(str)]
    if include_arena:
        parts.append(df['arena_type'].astype(str))
    parts += [df['rec_day'].astype(str).str.lower(), unit]
    cid = parts[0]
    for p in parts[1:]:
        cid = cid + '|' + p
    df['cell_id'] = np.where(ok, cid, df['session'].astype(str) + '|' + unit)
    return df


def repetition_summary(df: pd.DataFrame, group_col: str) -> str:
    """One-line summary: rows, unique cells, repeated cells, and how many of
    those span more than one group."""
    counts = df.groupby('cell_id').size()
    n_rep = int((counts > 1).sum())
    n_span = int((df.groupby('cell_id')[group_col].nunique() > 1).sum())
    return (f'{len(df)} observations from {len(counts)} cells ({n_rep} recorded in >1 '
            f'session, {n_span} in >1 {group_col.replace("_", " ")})')


# ── Helpers ──────────────────────────────────────────────────────────────────

def holm(p_values):
    """Holm-Bonferroni adjusted p-values (NaNs are left as NaN)."""
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method='holm')[1]
    return out


def _finish_posthoc(rows: list) -> list:
    adj = holm([r['p_raw'] for r in rows])
    for r, pa in zip(rows, adj):
        r['p_holm'] = pa
        r['significant'] = bool(pd.notna(pa) and pa < ALPHA)
    return rows


def _dummies(labels: np.ndarray, levels: list) -> np.ndarray:
    """Intercept + one treatment dummy per non-reference level."""
    cols = [np.ones(len(labels))]
    cols += [(labels == lvl).astype(float) for lvl in levels[1:]]
    return np.column_stack(cols)


def _contrast(levels, a, b, width):
    """Contrast vector (length `width`) for coef(a) - coef(b); the reference
    level (levels[0]) has coefficient 0."""
    c = np.zeros(width)
    ia, ib = levels.index(a), levels.index(b)
    if ia > 0:
        c[ia] += 1.0
    if ib > 0:
        c[ib] -= 1.0
    return c


def _clean(df, group_col, value_col, levels, to_numeric=True):
    sub = df[[group_col, 'cell_id', value_col]].copy()
    if to_numeric:
        sub[value_col] = pd.to_numeric(sub[value_col], errors='coerce')
    sub = sub.dropna(subset=[value_col])
    sub = sub[sub[group_col].isin(levels)]
    present = [l for l in levels if (sub[group_col] == l).any()]
    return sub, present


def _n_repeated(sub):
    return int((sub.groupby('cell_id').size() > 1).sum())


# ── Continuous metrics ───────────────────────────────────────────────────────

def _classic_continuous(groups: dict, note: str):
    stat, p = stats.kruskal(*groups.values())
    omni = {'test': 'Kruskal-Wallis', 'stat_name': 'H', 'statistic': stat,
            'dof': len(groups) - 1, 'p_value': p, 'model_note': note}
    rows = []
    for g1, g2 in combinations(groups.keys(), 2):
        u, pp = stats.mannwhitneyu(groups[g1], groups[g2], alternative='two-sided')
        rows.append({'comparison': f'{g1} vs {g2}', 'test': 'Mann-Whitney U',
                     'stat_name': 'U', 'statistic': u, 'p_raw': pp})
    return omni, _finish_posthoc(rows)


def compare_continuous(df: pd.DataFrame, group_col: str, levels: list, metric_col: str):
    """Omnibus + Holm-corrected pairwise tests of one continuous metric across
    the groups of `group_col` (in `levels` order). Returns (omnibus_dict,
    posthoc_rows) or (None, []) if fewer than 2 groups with >= 2 values."""
    sub, present = _clean(df, group_col, metric_col, levels)
    groups = {l: sub.loc[sub[group_col] == l, metric_col].to_numpy(float) for l in present}
    if len(groups) < 2 or any(len(v) < 2 for v in groups.values()):
        return None, []

    summary = repetition_summary(sub, group_col)
    if _n_repeated(sub) == 0:
        return _classic_continuous(groups, f'{summary}; no repeated cells, so '
                                           f'rows are independent')

    y = stats.rankdata(sub[metric_col].to_numpy(float))
    y = (y - y.mean()) / y.std()   # scale only, for numerical stability
    labels = sub[group_col].to_numpy()
    X_full = _dummies(labels, present)
    X_red = X_full[:, :1]
    clusters = sub['cell_id'].to_numpy()
    k = len(present)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ml_full = sm.MixedLM(y, X_full, groups=clusters).fit(reml=False)
            ml_red  = sm.MixedLM(y, X_red,  groups=clusters).fit(reml=False)
            reml    = sm.MixedLM(y, X_full, groups=clusters).fit(reml=True)
        lrt = max(2.0 * (ml_full.llf - ml_red.llf), 0.0)
        p = stats.chi2.sf(lrt, k - 1)
        beta = np.asarray(reml.fe_params)
        V = np.asarray(reml.cov_params())[:k, :k]
        if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(V)):
            raise ValueError('non-finite estimates')
    except Exception as e:  # noqa: BLE001
        return _classic_continuous(groups, f'{summary}; mixed model failed ({e}), '
                                           f'independence ASSUMED')

    omni = {'test': 'Rank LMM likelihood-ratio test', 'stat_name': 'χ²',
            'statistic': lrt, 'dof': k - 1, 'p_value': p,
            'model_note': f'Rank-transformed LMM, random intercept per cell: {summary}'}
    rows = []
    for a, b in combinations(present, 2):
        c = _contrast(present, a, b, k)
        est = float(c @ beta)
        se = float(np.sqrt(c @ V @ c))
        z = est / se if se > 0 else np.nan
        rows.append({'comparison': f'{a} vs {b}', 'test': 'Rank LMM Wald contrast',
                     'stat_name': 'z', 'statistic': z,
                     'p_raw': 2 * stats.norm.sf(abs(z)) if pd.notna(z) else np.nan})
    return omni, _finish_posthoc(rows)


# ── Yes/no outcomes ──────────────────────────────────────────────────────────

def _classic_binary(table: list, present: list, note: str):
    arr = np.array(table)
    chi2, p, dof, _ = stats.chi2_contingency(arr)
    omni = {'test': 'Chi-square test of independence', 'stat_name': 'χ²',
            'statistic': chi2, 'dof': dof, 'p_value': p, 'model_note': note}
    rows = []
    for i, j in combinations(range(len(present)), 2):
        odds_ratio, pp = stats.fisher_exact(np.array([table[i], table[j]]))
        rows.append({'comparison': f'{present[i]} vs {present[j]}', 'test': "Fisher's exact",
                     'stat_name': 'OR', 'statistic': odds_ratio, 'odds_ratio': odds_ratio,
                     'p_raw': pp})
    return omni, _finish_posthoc(rows)


def compare_binary(df: pd.DataFrame, group_col: str, levels: list, col: str, to_bool):
    """Omnibus + Holm-corrected pairwise tests of one True/False outcome across
    groups. `to_bool` maps a cell value to True/False/None. Returns
    (omnibus_dict, posthoc_rows, table) where table = [[n_true, n_false], ...]
    per present group; omnibus_dict is None if < 2 groups."""
    sub = df[[group_col, 'cell_id', col]].copy()
    sub['_y'] = sub[col].apply(to_bool)
    sub = sub.dropna(subset=['_y'])
    sub = sub[sub[group_col].isin(levels)]
    present = [l for l in levels if (sub[group_col] == l).any()]
    table = []
    for l in present:
        yl = sub.loc[sub[group_col] == l, '_y'].astype(bool)
        table.append([int(yl.sum()), int((~yl).sum())])
    if len(present) < 2:
        return None, [], table

    summary = repetition_summary(sub, group_col)
    if _n_repeated(sub) == 0:
        return (*_classic_binary(table, present, f'{summary}; no repeated cells, '
                                                 f'so rows are independent'), table)
    if any(t == 0 or f == 0 for t, f in table):
        return (*_classic_binary(table, present, f'{summary}; a group is at 0 % or 100 %, '
                                                 f'so the GEE is not estimable -- '
                                                 f'independence ASSUMED'), table)

    y = sub['_y'].astype(float).to_numpy()
    X = _dummies(sub[group_col].to_numpy(), present)
    k = len(present)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = sm.GEE(y, X, groups=sub['cell_id'].to_numpy(),
                         family=sm.families.Binomial(),
                         cov_struct=sm.cov_struct.Exchangeable()).fit()
        beta = np.asarray(res.params)
        V = np.asarray(res.cov_params())
        if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(V)):
            raise ValueError('non-finite estimates')
        b, Vb = beta[1:], V[1:, 1:]
        wald = float(b @ np.linalg.solve(Vb, b))
    except Exception as e:  # noqa: BLE001
        return (*_classic_binary(table, present, f'{summary}; GEE failed ({e}), '
                                                 f'independence ASSUMED'), table)

    omni = {'test': 'Logistic GEE Wald test', 'stat_name': 'χ²', 'statistic': wald,
            'dof': k - 1, 'p_value': stats.chi2.sf(wald, k - 1),
            'model_note': f'Logistic GEE clustered by cell, robust SEs: {summary}'}
    rows = []
    for a, bb in combinations(present, 2):
        c = _contrast(present, a, bb, k)
        est = float(c @ beta)
        se = float(np.sqrt(c @ V @ c))
        z = est / se if se > 0 else np.nan
        rows.append({'comparison': f'{a} vs {bb}', 'test': 'GEE Wald contrast',
                     'stat_name': 'OR', 'statistic': np.exp(est), 'odds_ratio': np.exp(est),
                     'z': z, 'p_raw': 2 * stats.norm.sf(abs(z)) if pd.notna(z) else np.nan})
    return omni, _finish_posthoc(rows), table


# ── 3-category outcome ───────────────────────────────────────────────────────

def _classic_nominal(table: list, present: list, note: str):
    chi2, p, dof, _ = stats.chi2_contingency(np.array(table))
    omni = {'test': 'Chi-square test of independence', 'stat_name': 'χ²',
            'statistic': chi2, 'dof': dof, 'p_value': p, 'model_note': note}
    rows = []
    for i, j in combinations(range(len(present)), 2):
        sub_t = np.array([table[i], table[j]])
        sub_t = sub_t[:, sub_t.sum(axis=0) > 0]   # drop all-zero categories
        if sub_t.shape[1] < 2:
            c2, pp = np.nan, np.nan
        else:
            c2, pp, _d, _e = stats.chi2_contingency(sub_t)
        rows.append({'comparison': f'{present[i]} vs {present[j]}',
                     'test': 'Chi-square test of independence', 'stat_name': 'χ²',
                     'statistic': c2, 'p_raw': pp})
    return omni, _finish_posthoc(rows)


def compare_nominal(df: pd.DataFrame, group_col: str, levels: list, col: str,
                    categories: list):
    """Omnibus + Holm-corrected pairwise tests of a multi-category outcome
    (values in `categories`) across groups. Returns (omnibus_dict,
    posthoc_rows, table) with table = per-group counts in `categories` order."""
    sub = df[[group_col, 'cell_id', col]].dropna(subset=[col])
    sub = sub[sub[group_col].isin(levels) & sub[col].isin(categories)]
    present = [l for l in levels if (sub[group_col] == l).any()]
    table = [[int(((sub[group_col] == l) & (sub[col] == c)).sum()) for c in categories]
             for l in present]
    if len(present) < 2:
        return None, [], table

    summary = repetition_summary(sub, group_col)
    if _n_repeated(sub) == 0:
        return (*_classic_nominal(table, present, f'{summary}; no repeated cells, '
                                                  f'so rows are independent'), table)
    if any(n == 0 for row in table for n in row):
        return (*_classic_nominal(table, present, f'{summary}; a group has 0 cells in a '
                                                  f'category, so the GEE is not estimable '
                                                  f'-- independence ASSUMED'), table)

    k, m = len(present), len(categories)
    y = sub[col].map({c: i for i, c in enumerate(categories)}).to_numpy()
    X = _dummies(sub[group_col].to_numpy(), present)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = NominalGEE(y, X, groups=sub['cell_id'].to_numpy(),
                             cov_struct=sm.cov_struct.Independence()).fit()
        # Params come in one block of k coefficients per non-reference category
        # (the last category is the reference).
        beta = np.asarray(res.params)
        V = np.asarray(res.cov_params())
        if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(V)):
            raise ValueError('non-finite estimates')

        def _wald(C):
            d = C @ beta
            return float(d @ np.linalg.solve(C @ V @ C.T, d))

        eye = np.eye(k)
        C_omni = np.vstack([np.kron(np.eye(m - 1)[j], eye[i])
                            for j in range(m - 1) for i in range(1, k)])
        wald = _wald(C_omni)
    except Exception as e:  # noqa: BLE001
        return (*_classic_nominal(table, present, f'{summary}; GEE failed ({e}), '
                                                  f'independence ASSUMED'), table)

    dof = (k - 1) * (m - 1)
    omni = {'test': 'Multinomial GEE Wald test', 'stat_name': 'χ²', 'statistic': wald,
            'dof': dof, 'p_value': stats.chi2.sf(wald, dof),
            'model_note': f'Multinomial GEE clustered by cell, robust SEs: {summary}'}
    rows = []
    for a, b in combinations(present, 2):
        c = _contrast(present, a, b, k)
        C = np.vstack([np.kron(np.eye(m - 1)[j], c) for j in range(m - 1)])
        w = _wald(C)
        rows.append({'comparison': f'{a} vs {b}', 'test': 'Multinomial GEE Wald contrast',
                     'stat_name': 'χ²', 'statistic': w, 'dof': m - 1,
                     'p_raw': stats.chi2.sf(w, m - 1)})
    return omni, _finish_posthoc(rows), table
