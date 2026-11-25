"""
SurgeryVitals_Plot_v1.py
========================
Variability of vitals across surgery time.

Input  : Vitals.xlsx (Sheet1), columns
         A Date | B Animal | C Species | D Exclude? | E SurvivedSurgery? | F Time
         G timesincestart | H Timemin | I breathrate | J Bodytemp | K Isoflurane
Output : figures (PNG + PDF/SVG) and CSV tables in OUT_DIR.

Surgery definition
------------------
A new surgery starts at every row with a value in column A (Date). Some animals
had two surgeries (e.g. FM51605, FA167DCE89, FAC3A3, #0156, 1680378B, FA0420),
so surgeries, not animals, are the unit. Time since start (column G, h:mm:ss)
restarts at 0:00:00 for each surgery. Rows inside a surgery are sorted by
time because a few were entered out of order; exact duplicate rows are dropped.

Breathing-rate cleaning (column I)
----------------------------------
- Ranges such as '25-30'           -> midpoint (27.5)
- Any other text ('<20', 'removed from Isofluran') -> missing
- 0 breaths/min                    -> missing (one entry at the end of a surgery)

Isoflurane cleaning (column K)
------------------------------
0 % readings at the end of a surgery (after the last non-zero reading) mean the
animal was taken off anaesthesia; they are removed. Earlier 0 % readings stay.

Groups (one column of panels each)
----------------------------------
All surgeries | Exclude? = No | Exclude? = Yes

What is plotted per vital
-------------------------
- thin grey line: one surgery
- blue line + band: mean +/- SD across surgeries in BIN_MIN-minute bins of
  time since start. Each surgery first contributes its own mean within a bin,
  so densely sampled surgeries do not weigh more. A bin is shown only when at
  least MIN_SURG_PER_BIN surgeries contribute.
Y limits are shared across groups. All panels of one figure share one time
axis, from 0 to the last reading of any vital in any group of that figure,
rounded up to X_ROUND_H hours.
"""

import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
INPUT_XLSX = r'C:\Runita\NMR\Thesis\Figs\Surgery\Vitals.xlsx'
SHEET = 'Sheet1'
OUT_DIR = r'C:\Runita\NMR\Thesis\Figs\Surgery\VitalsPlots_v1'

BIN_MIN = 15             # width of time bins for the group mean +/- SD (min)
MIN_SURG_PER_BIN = 3     # bins with fewer contributing surgeries are not drawn
X_ROUND_H = 0.5          # per-figure time axis max is rounded up to this (h)
FIG_FORMATS = ('png', 'pdf', 'svg')

# Column positions (0-based) as in the sheet: A=0 ... K=10
COL = dict(date=0, animal=1, species=2, exclude=3, survived=4,
           clock=5, tsince=6, breath=8, temp=9, iso=10)

VITALS = [  # (column key, axis label)
    ('breath', 'Breathing rate (breaths/min)'),
    ('temp',   'Body temperature (°C)'),
    ('iso',    'Isoflurane (%)'),
]

GROUPS = [  # (name, title, filter on per-surgery Exclude value)
    ('All',        'All surgeries',  None),
    ('ExcludeNo',  'Exclude? = No',  'No'),
    ('ExcludeYes', 'Exclude? = Yes', 'Yes'),
]

# Colours / ink
C_TRACE = '#b5b3ad'      # individual surgeries (light, recessive)
TRACE_ALPHA = 0.45
TRACE_LW = 0.6
C_MEAN = '#104281'       # group mean line (dark blue)
C_BAND = '#1c5cab'       # +/- SD band
BAND_ALPHA = 0.30
C_INK = '#0b0b0b'
C_INK2 = '#52514e'
C_GRID = '#e1e0d9'
C_AXIS = '#c3c2b7'


# ----------------------------------------------------------------------------
# Loading and cleaning
# ----------------------------------------------------------------------------
_RANGE_RE = re.compile(r'^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$')


def clean_breath(v):
    """Numeric breath rate; ranges -> midpoint; other text and 0 -> NaN."""
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        v = float(v)
    else:
        s = str(v).strip()
        m = _RANGE_RE.match(s)
        if m:
            v = (float(m.group(1)) + float(m.group(2))) / 2.0
        else:
            try:
                v = float(s)
            except ValueError:
                return np.nan
    return np.nan if v <= 0 else v


def parse_tsince_min(v):
    """'h:mm:ss' string / time / timedelta -> minutes since surgery start."""
    if pd.isna(v):
        return np.nan
    if isinstance(v, pd.Timedelta):
        return v.total_seconds() / 60.0
    if hasattr(v, 'hour') and hasattr(v, 'minute'):          # datetime.time
        return v.hour * 60 + v.minute + getattr(v, 'second', 0) / 60.0
    if isinstance(v, (int, float, np.integer, np.floating)):  # Excel day fraction
        return float(v) * 24 * 60
    td = pd.to_timedelta(str(v).strip(), errors='coerce')
    return np.nan if pd.isna(td) else td.total_seconds() / 60.0


def load_vitals(path=INPUT_XLSX, sheet=SHEET):
    raw = pd.read_excel(path, sheet_name=sheet)
    c = raw.columns
    df = pd.DataFrame({
        'Date':     pd.to_datetime(raw[c[COL['date']]], errors='coerce'),
        'Animal':   raw[c[COL['animal']]].astype(str).str.strip(),
        'Species':  raw[c[COL['species']]],
        'Exclude':  raw[c[COL['exclude']]].astype(str).str.strip(),
        'Survived': raw[c[COL['survived']]].astype(str).str.strip(),
        'Clock':    raw[c[COL['clock']]],
        't_min':    raw[c[COL['tsince']]].map(parse_tsince_min),
        'breath_raw': raw[c[COL['breath']]],
        'breath':   raw[c[COL['breath']]].map(clean_breath),
        'temp':     pd.to_numeric(raw[c[COL['temp']]], errors='coerce'),
        'iso':      pd.to_numeric(raw[c[COL['iso']]], errors='coerce'),
    })
    df = df[df['Animal'].ne('nan') & df['t_min'].notna()].copy()

    # New surgery at each row with a Date (or when the animal name changes).
    new_surg = df['Date'].notna() | df['Animal'].ne(df['Animal'].shift())
    df['SurgeryIdx'] = new_surg.cumsum()
    df['Date'] = df.groupby('SurgeryIdx')['Date'].transform('first')

    # Per-surgery Exclude / Survived (must be constant inside a surgery)
    for col in ('Exclude', 'Survived'):
        n_vals = df.groupby('SurgeryIdx')[col].nunique()
        bad = n_vals[n_vals > 1].index.tolist()
        if bad:
            print(f'[warn] {col} not constant within surgeries {bad}; using first value')
        df[col] = df.groupby('SurgeryIdx')[col].transform('first')

    df['Surgery'] = (df['Animal'] + ' ' + df['Date'].dt.strftime('%Y-%m-%d')
                     + ' (#' + df['SurgeryIdx'].astype(str) + ')')

    # Text breath entries that were changed
    txt = df[df['breath_raw'].notna() & df['breath'].isna()]
    rng = df[df['breath_raw'].map(lambda v: isinstance(v, str) and bool(_RANGE_RE.match(v)))]
    if len(txt):
        print(f'[info] breath rate set to missing for {len(txt)} rows: '
              f'{sorted(txt["breath_raw"].astype(str).unique())}')
    if len(rng):
        print(f'[info] breath-rate ranges -> midpoint for {len(rng)} rows: '
              f'{sorted(rng["breath_raw"].unique())}')

    # Order rows in time inside each surgery; drop exact duplicates
    n0 = len(df)
    df = df.sort_values(['SurgeryIdx', 't_min'], kind='stable')
    df = df.drop_duplicates(subset=['SurgeryIdx', 't_min', 'breath', 'temp', 'iso'])
    if len(df) < n0:
        print(f'[info] dropped {n0 - len(df)} duplicate row(s)')

    # Isoflurane 0 % at the end of a surgery = animal taken off anaesthesia.
    # Remove the trailing run of 0 % readings (iso only; other vitals kept).
    # 0 % readings followed by a non-zero reading later in the surgery are kept.
    def _trailing_zero_mask(iso):
        nz = iso.notna() & iso.ne(0)
        after_last_nz = ~nz[::-1].cummax()[::-1] if nz.any() else pd.Series(True, index=iso.index)
        return after_last_nz & iso.eq(0)

    drop_iso = df.groupby('SurgeryIdx', group_keys=False)['iso'].apply(_trailing_zero_mask)
    drop_iso = drop_iso.reindex(df.index, fill_value=False)
    if drop_iso.any():
        lst = ', '.join(f'{s} @ {t:.0f} min'
                        for s, t in df.loc[drop_iso, ['Surgery', 't_min']].itertuples(index=False))
        print(f'[info] isoflurane 0 % at end of surgery removed ({int(drop_iso.sum())} rows): {lst}')
    df.loc[drop_iso, 'iso'] = np.nan
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------
def group_subset(df, excl_value):
    return df if excl_value is None else df[df['Exclude'] == excl_value]


def binned_summary(df, vital):
    """Mean +/- SD across surgeries in BIN_MIN bins (surgery means first)."""
    d = df[['SurgeryIdx', 't_min', vital]].dropna()
    if d.empty:
        return pd.DataFrame(columns=['bin_start_min', 'bin_center_min', 'mean', 'sd', 'n_surgeries'])
    d = d.assign(bin=(d['t_min'] // BIN_MIN).astype(int))
    per_surg = d.groupby(['bin', 'SurgeryIdx'])[vital].mean()
    g = per_surg.groupby(level='bin')
    out = pd.DataFrame({'mean': g.mean(), 'sd': g.std(ddof=1), 'n_surgeries': g.size()})
    out.index.name = 'bin'
    out = out.reset_index()
    out['bin_start_min'] = out['bin'] * BIN_MIN
    out['bin_center_min'] = out['bin_start_min'] + BIN_MIN / 2.0
    return out[['bin_start_min', 'bin_center_min', 'mean', 'sd', 'n_surgeries']]


def per_surgery_summary(df):
    rows = []
    for sid, d in df.groupby('SurgeryIdx'):
        r = dict(SurgeryIdx=sid, Surgery=d['Surgery'].iat[0], Animal=d['Animal'].iat[0],
                 Date=d['Date'].iat[0].date() if pd.notna(d['Date'].iat[0]) else None,
                 Species=d['Species'].iat[0], Exclude=d['Exclude'].iat[0],
                 Survived=d['Survived'].iat[0], duration_min=d['t_min'].max())
        for v, _ in VITALS:
            x = d[v].dropna()
            r[f'{v}_n'] = len(x)
            r[f'{v}_mean'] = x.mean() if len(x) else np.nan
            r[f'{v}_sd'] = x.std(ddof=1) if len(x) > 1 else np.nan
            r[f'{v}_cv'] = (r[f'{v}_sd'] / r[f'{v}_mean']
                            if len(x) > 1 and r[f'{v}_mean'] else np.nan)
            r[f'{v}_min'] = x.min() if len(x) else np.nan
            r[f'{v}_max'] = x.max() if len(x) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def style_axis(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(C_AXIS)
    ax.tick_params(colors=C_INK2, labelsize=9)
    ax.grid(axis='y', color=C_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_vital(ax, df, vital, summary):
    for _, d in df.groupby('SurgeryIdx'):
        d = d[['t_min', vital]].dropna()
        if len(d) == 0:
            continue
        ax.plot(d['t_min'] / 60.0, d[vital], color=C_TRACE, lw=TRACE_LW,
                alpha=TRACE_ALPHA, marker='o', ms=1.5, mew=0, zorder=1)
    s = summary[summary['n_surgeries'] >= MIN_SURG_PER_BIN]
    if len(s):
        x = s['bin_center_min'] / 60.0
        sd = s['sd'].fillna(0)
        ax.fill_between(x, s['mean'] - sd, s['mean'] + sd, color=C_BAND,
                        alpha=BAND_ALPHA, lw=0, zorder=2)
        ax.plot(x, s['mean'], color=C_MEAN, lw=2.2, zorder=3)
    style_axis(ax)


def figure_xmax(df, groups):
    """Common time axis for one figure: last reading of any vital in any of its
    groups (h), rounded up to X_ROUND_H."""
    vit = [v for v, _ in VITALS]
    t_last = 0.0
    for _, _, excl in groups:
        dg = group_subset(df, excl)
        t = dg.loc[dg[vit].notna().any(axis=1), 't_min']
        if len(t):
            t_last = max(t_last, t.max() / 60.0)
    return max(np.ceil(t_last / X_ROUND_H) * X_ROUND_H, X_ROUND_H)


def draw_grid(df, groups, summaries, ylims, title):
    nrow, ncol = len(VITALS), len(groups)
    xmax = figure_xmax(df, groups)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol + 0.8, 2.75 * nrow + 0.9),
                             sharex=False, squeeze=False)
    for j, (gname, gtitle, excl) in enumerate(groups):
        dg = group_subset(df, excl)
        n_surg = dg['SurgeryIdx'].nunique()
        n_anim = dg['Animal'].nunique()
        for i, (v, vlabel) in enumerate(VITALS):
            ax = axes[i, j]
            plot_vital(ax, dg, v, summaries[(gname, v)])
            ax.set_ylim(*ylims[v])
            ax.set_xlim(0, xmax)
            ax.set_xticks(np.arange(0, xmax + 1e-9, 1.0))
            if j == 0:
                ax.set_ylabel(vlabel, color=C_INK, fontsize=10)
            if i == 0:
                ax.set_title(f'{gtitle}\n{n_surg} surgeries, {n_anim} animals',
                             color=C_INK, fontsize=10)
            ax.set_xlabel('Time since start (h)', color=C_INK, fontsize=9)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=C_TRACE, lw=TRACE_LW, alpha=TRACE_ALPHA, marker='o',
                      ms=1.5, label='one surgery'),
               Line2D([], [], color=C_MEAN, lw=2.2, label=f'mean ({BIN_MIN}-min bins)'),
               Patch(facecolor=C_BAND, alpha=BAND_ALPHA, label='± SD across surgeries')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, color=C_INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def save_fig(fig, stem, n_tries=5):
    """Save in all formats; retry briefly if the file is locked (sync / viewer)."""
    import time
    for ext in FIG_FORMATS:
        path = os.path.join(OUT_DIR, f'{stem}.{ext}')
        for k in range(n_tries):
            try:
                fig.savefig(path, dpi=300, bbox_inches='tight')
                break
            except OSError as e:
                if k == n_tries - 1:
                    print(f'[warn] could not save {path}: {e}')
                else:
                    time.sleep(1.0)
    plt.close(fig)


def padded_limits(series, pad_frac=0.05):
    lo, hi = np.nanmin(series), np.nanmax(series)
    pad = (hi - lo) * pad_frac if hi > lo else 1.0
    return lo - pad, hi + pad


# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_vitals()

    print(f'[info] {df["SurgeryIdx"].nunique()} surgeries, {df["Animal"].nunique()} animals, '
          f'{len(df)} rows')
    for gname, gtitle, excl in GROUPS:
        dg = group_subset(df, excl)
        print(f'       {gtitle:15s}: {dg["SurgeryIdx"].nunique():2d} surgeries, '
              f'{dg["Animal"].nunique():2d} animals')

    # Summaries per group x vital
    summaries, rows = {}, []
    for gname, gtitle, excl in GROUPS:
        dg = group_subset(df, excl)
        for v, _ in VITALS:
            s = binned_summary(dg, v)
            summaries[(gname, v)] = s
            rows.append(s.assign(group=gname, vital=v))
    pd.concat(rows, ignore_index=True)[
        ['group', 'vital', 'bin_start_min', 'bin_center_min', 'mean', 'sd', 'n_surgeries']
    ].to_csv(os.path.join(OUT_DIR, 'Vitals_BinnedMeanSD.csv'), index=False)

    per_surgery_summary(df).to_csv(os.path.join(OUT_DIR, 'Vitals_PerSurgerySummary.csv'),
                                   index=False)
    df.drop(columns=['breath_raw']).to_csv(os.path.join(OUT_DIR, 'Vitals_Clean_Long.csv'),
                                          index=False)

    # Y limits shared across groups; one time axis per figure (see figure_xmax)
    ylims = {v: padded_limits(df[v]) for v, _ in VITALS}

    fig = draw_grid(df, GROUPS, summaries, ylims, 'Surgery vitals over time')
    print(f'[info] time axis Vitals_AllGroups: 0-{figure_xmax(df, GROUPS):g} h')
    save_fig(fig, 'Vitals_AllGroups')
    for g in GROUPS:
        fig = draw_grid(df, [g], summaries, ylims, f'Surgery vitals over time - {g[1]}')
        print(f'[info] time axis Vitals_{g[0]}: 0-{figure_xmax(df, [g]):g} h')
        save_fig(fig, f'Vitals_{g[0]}')

    print(f'[done] outputs in {OUT_DIR}')


if __name__ == '__main__':
    main()
