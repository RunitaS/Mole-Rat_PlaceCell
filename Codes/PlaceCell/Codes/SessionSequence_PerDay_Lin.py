"""
Session sequence per recording day, one schematic per arena (Linear track,
Open field).

Goes through every input workbook of an arena, reads the session paths (sheet
'Full', column 'session', e.g. 'Fa1059\\Linear\\Day10\\2_270' or
'Fa23BD\\Open\\Day3\\1Rotate'), and extracts for each row the animal
('Fa1059'), recording day ('Day10'), session order ('2') and session type
('270' / 'Rotate'). Rows from other arenas are ignored.

Plot: a single schematic with one row per recording day. Each row is a
sequence of equal-size horizontal bars, one per session in the order it was
recorded (1, 2, 3, ...), coloured by session type (the colours of the
SessionType statistics plots; see the key). Rows are stacked top to bottom by
animal (animals ordered by their first recording day) and by day number
within an animal; a dashed horizontal line separates the animals. Linear
track: Test sessions (geomagnetic field, then zero field) have the 2nd half
of the bar in gray; Control sessions are the session colour throughout.

Output per arena: PNG / PDF / SVG in its plots_dir, and the per-day sequence
table in an Excel workbook next to them.
"""

import os
import re

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch

# ── Parameters ──────────────────────────────────────────────────────────────

LIN_ROOT  = r'X:\NMR_group_data\Runita\Analysis\Mean_KDE_Open_PascalOldenburg\LinearTrack_GeoMagVsZero'
OPEN_ROOT = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt'

SHEET = 'Full'
TEST_H2_FACE = '#A6A6A6'   # zero-field 2nd half of Linear Test sessions (FldVsZero gray)

# Per arena:
#   excels         condition -> workbook (conditions listed in
#                  zero_half_conds get the gray zero-field 2nd half)
#   session_types  session types in key order
#   colors         session type -> bar colour (as in the statistics plots)
#   labels         session type -> key label
ARENAS = {
    'Linear': dict(
        excels={
            'Cntrl': os.path.join(LIN_ROOT, 'Data_Control', 'All_TT_PlaceChar_AdptBin.xlsx'),
            'Test':  os.path.join(LIN_ROOT, 'Data', 'All_TT_PlaceChar_AdptBin.xlsx'),
        },
        zero_half_conds={'Test'},
        session_types=['0', '90', '180', '270'],
        # the 'face' tones of SessionType_StatsComparison_v6_Lin_*.py
        colors={'0': '#008080', '90': '#00B3B3', '180': '#33D6D6', '270': '#7FFAFA'},
        labels={'0': '0°', '90': '90°', '180': '180°', '270': '270°'},
        plots_dir=os.path.join(LIN_ROOT, 'SessionSequence_Lin'),
        fig_name='SessionSequence_PerDay_Lin',
    ),
    'Open': dict(
        excels={'All': os.path.join(OPEN_ROOT, 'All_TT_PlaceChar_AdptBin.xlsx')},
        zero_half_conds=set(),
        session_types=['Cntrl', 'Rotate', 'Zero'],
        # the 'face' tones of SessionType_StatsComparison_v6_Open_Halves.py
        colors={'Cntrl': '#7FB3E5', 'Rotate': '#0066CC', 'Zero': '#BFBFBF'},
        labels={'Cntrl': 'Cntrl', 'Rotate': 'Rotate', 'Zero': 'Zero'},
        plots_dir=os.path.join(OPEN_ROOT, 'SessionSequence_Open'),
        fig_name='SessionSequence_PerDay_Open',
    ),
}
RUN_ARENAS = ['Linear', 'Open']

ANIMAL_PATTERN = r'^Fa[0-9A-Za-z]+$'
DAY_PATTERN    = r'^(?:Exp)?Day(\d+)$'

# ── Figure style (as in the SessionType_StatsComparison_v6 scripts) ─────────

PAL_BLACK = '#000000'
PAL_GRAY  = '#7F7F7F'
FIG_FONT  = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
FS_LABEL, FS_TITLE, FS_LEGEND = 14, 15, 11
FIG_DPI = 500

AXIS_LW     = 0.8
BAR_EDGE_LW = 0.6
BAR_LEN     = 1.0        # x length of one session bar
BAR_GAP     = 0.08       # x gap between the bars of one day
BAR_THICK   = 0.7        # y thickness of a day's row (rows 1 apart)
ANIMAL_GAP  = 3.0        # extra y gap between animals, in rows (separator line in it)
BAR_W_IN    = 0.9        # figure width per session bar (inches)
ROW_H_IN    = 0.076      # figure height per day row (inches); bar = BAR_THICK of it
PAD_W_IN    = 3.2        # extra figure width for animal labels + legend
PAD_H_IN    = 1.2        # extra figure height for the title

plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    FIG_FONT,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
    'svg.fonttype':       'none',
    'text.color':         PAL_BLACK,
    'axes.labelcolor':    PAL_BLACK,
    'axes.labelweight':   'bold',
    'axes.titleweight':   'bold',
    'axes.edgecolor':     PAL_BLACK,
    'axes.linewidth':     AXIS_LW,
    'axes.facecolor':     'white',
    'figure.facecolor':   'white',
    'savefig.facecolor':  'white',
    'axes.grid':          False,
    'legend.frameon':     False,
})

# ── Parsing ─────────────────────────────────────────────────────────────────

def _session_regex(session_types: list):
    """'<order>[_]<type>' of the last folder, e.g. '3_180', '2Rotate'."""
    alts = '|'.join(re.escape(t) for t in sorted(session_types, key=len, reverse=True))
    return re.compile(rf'^(\d+)_?({alts})$', re.IGNORECASE)


def parse_session(session_path: str, arena: str, session_re, session_types: list):
    """'Fa1059\\Linear\\Day10\\2_270' ->
    dict(animal='Fa1059', day='Day10', day_num=10, order=2, session_type='270');
    None if the path is not in `arena` or its last folder doesn't match."""
    folders = [p.strip() for p in re.split(r'[\\/]+', str(session_path).strip()) if p.strip()]
    if not folders or arena.lower() not in (f.lower() for f in folders[:-1]):
        return None
    m = session_re.match(folders[-1])
    if not m:
        return None
    canon = {t.lower(): t for t in session_types}
    # animal / day tokens may sit inside a folder name ('ExpDay2_OpenArena_28Oct25')
    tokens = [t.strip() for f in folders[:-1] for t in f.split('_')]
    animal = next((t for t in tokens if re.match(ANIMAL_PATTERN, t, re.IGNORECASE)), None)
    day, day_num = None, None
    for t in tokens:
        d = re.match(DAY_PATTERN, t, re.IGNORECASE)
        if d:
            day, day_num = t, int(d.group(1))
            break
    return dict(animal=animal, day=day, day_num=day_num,
                order=int(m.group(1)), session_type=canon[m.group(2).lower()])


def load_sequences(arena: str, cfg: dict) -> pd.DataFrame:
    """One row per (condition, animal, day, session) of `arena` over its workbooks."""
    session_re = _session_regex(cfg['session_types'])
    rows = []
    for cond, path in cfg['excels'].items():
        print(f'Loading {arena} [{cond}]: {path}')
        sessions = pd.read_excel(path, sheet_name=SHEET, usecols=['session'])['session']
        for s in sessions.dropna().astype(str).unique():
            folders = re.split(r'[\\/]+', s)
            if arena.lower() not in (f.strip().lower() for f in folders):
                continue                                  # another arena
            info = parse_session(s, arena, session_re, cfg['session_types'])
            if info is None or info['animal'] is None or info['day'] is None:
                print(f'WARNING [{arena} {cond}]: session path not recognized; skipped: {s}')
                continue
            rows.append(dict(condition=cond, session=s, **info))

    df = pd.DataFrame(rows)
    if df.empty:
        print(f'No {arena} sessions found in the input workbooks.')
        return df

    dup = df.duplicated(['animal', 'day_num', 'order'], keep=False)
    if dup.any():
        print(f'WARNING [{arena}]: the same animal/day/session order appears more than once:')
        print(df.loc[dup].sort_values(['animal', 'day_num', 'order']).to_string(index=False))
    return df.sort_values(['animal', 'day_num', 'order']).reset_index(drop=True)


def animal_order(df: pd.DataFrame) -> list:
    """Animals ordered by their first recording day (e.g. Fa23BD: Day1 first)."""
    first = df.groupby('animal')['day_num'].min().reset_index()
    return list(first.sort_values(['day_num', 'animal'])['animal'])


def sequence_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per animal/day: condition and the session type of session 1, 2, ..."""
    wide = (df.pivot_table(index=['animal', 'day_num', 'day', 'condition'],
                           columns='order', values='session_type', aggfunc='first')
              .rename(columns=lambda o: f'session_{o}')
              .reset_index())
    wide['_a'] = wide['animal'].map({a: i for i, a in enumerate(animal_order(df))})
    wide = wide.sort_values(['_a', 'day_num']).drop(columns='_a')
    wide.columns.name = None
    order_cols = [c for c in wide.columns if c.startswith('session_')]
    wide['sequence'] = wide[order_cols].apply(
        lambda r: ' -> '.join(str(v) for v in r if pd.notna(v)), axis=1)
    return wide.drop(columns='day_num')

# ── Plot ────────────────────────────────────────────────────────────────────

def plot_sequences(df: pd.DataFrame, arena: str, cfg: dict):
    colors = cfg['colors']
    animals = animal_order(df)
    n_rows = df.groupby(['animal', 'day_num']).ngroups
    n_sess = int(df['order'].max())

    n_row_units = n_rows + ANIMAL_GAP * (len(animals) - 1) + 1.2
    fig, ax = plt.subplots(figsize=(PAD_W_IN + BAR_W_IN * n_sess,
                                    PAD_H_IN + ROW_H_IN * n_row_units))
    # x in axes units, y in data units (for the animal labels left of the bars)
    yx = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

    y = 0.0                       # rows go downwards: y = 0, -1, -2, ...
    for a_i, animal in enumerate(animals):
        a_df = df[df['animal'] == animal]
        a_top = y
        for day_num in sorted(a_df['day_num'].unique()):
            d_df = a_df[a_df['day_num'] == day_num].sort_values('order')
            cond = d_df['condition'].iloc[0]
            for _, r in d_df.iterrows():
                x = (r['order'] - 1) * (BAR_LEN + BAR_GAP)
                color = colors.get(r['session_type'], PAL_GRAY)
                if cond in cfg['zero_half_conds']:
                    # 1st half geomagnetic field, 2nd half zero field (gray)
                    ax.barh(y, BAR_LEN / 2, left=x, height=BAR_THICK,
                            color=color, linewidth=0)
                    ax.barh(y, BAR_LEN / 2, left=x + BAR_LEN / 2, height=BAR_THICK,
                            color=TEST_H2_FACE, linewidth=0)
                    color = 'none'
                ax.barh(y, BAR_LEN, left=x, height=BAR_THICK, color=color,
                        edgecolor=PAL_BLACK, linewidth=BAR_EDGE_LW)
            y -= 1
        a_bottom = y + 1
        # animal label (with a bracket) beside its sequences
        ax.plot([-0.03, -0.03], [a_top + BAR_THICK / 2, a_bottom - BAR_THICK / 2],
                transform=yx, color=PAL_BLACK, linewidth=AXIS_LW, clip_on=False)
        ax.text(-0.05, (a_top + a_bottom) / 2, animal, transform=yx,
                ha='right', va='center', fontsize=FS_LABEL, fontweight='bold')
        if a_i < len(animals) - 1:
            y -= ANIMAL_GAP
            sep = a_bottom - (1 + ANIMAL_GAP) / 2
            ax.axhline(sep, xmin=-0.25, xmax=1.0, color=PAL_BLACK, linestyle='--',
                       linewidth=1.0, clip_on=False)

    ax.set_xticks([])
    ax.set_xlim(-BAR_GAP, n_sess * (BAR_LEN + BAR_GAP))
    ax.set_yticks([])
    ax.set_ylim(y + 1 - 0.6, 0.6)
    for side in ('left', 'bottom', 'top', 'right'):
        ax.spines[side].set_visible(False)

    handles = [Patch(facecolor=colors[t], edgecolor=PAL_BLACK, linewidth=BAR_EDGE_LW,
                     label=cfg['labels'].get(t, t))
               for t in cfg['session_types'] if t in set(df['session_type'])]
    if set(df['condition']) & cfg['zero_half_conds']:
        handles.append(Patch(facecolor=TEST_H2_FACE, edgecolor=PAL_BLACK,
                             linewidth=BAR_EDGE_LW, label='Zero field'))
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1.0),
              fontsize=FS_LEGEND)
    ax.set_title(f'Session sequence per recording day ({arena})',
                 fontsize=FS_TITLE, pad=10)

    fig.tight_layout()
    os.makedirs(cfg['plots_dir'], exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        out = os.path.join(cfg['plots_dir'], f'{cfg["fig_name"]}.{ext}')
        fig.savefig(out, dpi=FIG_DPI, bbox_inches='tight')
        print(f'Saved: {out}')
    plt.close(fig)


def run_arena(arena: str, cfg: dict):
    print(f'\n{"=" * 70}\n{arena}\n{"=" * 70}')
    df = load_sequences(arena, cfg)
    if df.empty:
        return
    table = sequence_table(df)
    print(f'\nSession sequence per recording day ({arena}):')
    print(table.to_string(index=False))

    os.makedirs(cfg['plots_dir'], exist_ok=True)
    out_xlsx = os.path.join(cfg['plots_dir'], f'{cfg["fig_name"]}.xlsx')
    with pd.ExcelWriter(out_xlsx) as xw:
        table.to_excel(xw, sheet_name='SequencePerDay', index=False)
        df.to_excel(xw, sheet_name='Sessions', index=False)
    print(f'Saved: {out_xlsx}')

    plot_sequences(df, arena, cfg)


def main():
    for arena in RUN_ARENAS:
        run_arena(arena, ARENAS[arena])


if __name__ == '__main__':
    main()
