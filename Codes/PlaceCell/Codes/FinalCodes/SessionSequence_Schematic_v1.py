"""Schematic of the recording sessions and their sequence, one plot per arena type
(Circular track, Linear track, Open field).

Each row is one recording day; each thick line in it is one session, coloured by
session type and placed at its position in that day's sequence. Rows are grouped by
animal in implantation order (ANIMAL_ORDER) and, within an animal, run from the first
recording day (top) to the last (bottom). The day numbers only set this order and are
not shown. The pastel block to the left of each group is the animal's colour. The key
below each plot gives the session-type colours. Animals not recorded in an arena are
left out of its plot.

The sequence is read from the folder names only (no data files are opened):
    DATA_ROOT\\<animal>\\<Circle|Linear|Open>\\Day<d>\\<n><session type>
    e.g. Fa1059\\Linear\\Day10\\2_270  ->  2nd session of the day, 270 deg
A position whose folder is missing is left empty in its row, so the later sessions
keep their true position.

Colours, session types, their parsing and the figure style are imported from
ArenaSessionType_StatsComparison_v21 (the import takes about 15 s).

Outputs, in OUTPUT_DIR:
    SessionSequence_<arena>.png / .pdf
    SessionSequence_Sessions.csv     the sessions drawn (with day numbers, for checking)
"""
import os
import re
import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ArenaSessionType_StatsComparison_v21 as v  # noqa: E402

# ── Configuration ───────────────────────────────────────────────────────────
# Corr_Data holds every sorted session. The quality-filtered tree
#   ...\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final
# has the same layout but lacks the sessions dropped by the filtering; use it here
# to draw only the sessions of the final place-cell dataset.
DATA_ROOT  = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data'
OUTPUT_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final\Results\SessionSequence'

ANIMAL_ORDER = ['Fa8477', 'Fa5834', 'Fa1059', 'Fa23BD']   # order of implantation
ARENA_TITLES = {'Circle': 'Circular track', 'Linear': 'Linear track', 'Open': 'Open field'}

# Layout (inches unless noted); the same in all three plots
ROW_H        = 0.30    # height of one row (recording day)
LINE_H       = 0.20    # thickness of a session line
SESSION_W    = 0.85    # width of one session position
SESSION_GAP  = 0.06    # gap between consecutive sessions
ANIMAL_GAP   = 0.16    # extra gap between animals
ANIMAL_W     = 0.85    # width of the animal block
ANIMAL_PAD   = 0.12    # gap between the animal block and the rasters
MARGIN_L, MARGIN_R = 0.15, 0.15
MARGIN_TOP   = 0.50    # title
AXIS_H       = 0.62    # x ticks and x label
KEY_H        = 0.40    # key


# ── Sessions from the folder names ──────────────────────────────────────────
def _subdirs(path):
    return sorted(e.name for e in os.scandir(path) if e.is_dir())


def collect_sessions(root: str) -> pd.DataFrame:
    """One row per session folder: animal, arena, day, position (in the day's
    sequence), session_type, folder. Folders that can't be parsed are reported."""
    rows, skipped = [], []
    for animal_dir in _subdirs(root):
        animal = v.canonical_animal_id(animal_dir)
        if animal not in ANIMAL_ORDER:
            continue
        for arena_dir in _subdirs(os.path.join(root, animal_dir)):
            arena = v.extract_arena_type(arena_dir)
            if arena is None:
                skipped.append(os.path.join(animal_dir, arena_dir))
                continue
            for day_dir in _subdirs(os.path.join(root, animal_dir, arena_dir)):
                day = re.fullmatch(r'day[\s_]*(\d+(?:\.\d+)?)', day_dir.strip(), re.I)
                day_rel = os.path.join(animal_dir, arena_dir, day_dir)
                if day is None:
                    skipped.append(day_rel)
                    continue
                for ses_dir in _subdirs(os.path.join(root, day_rel)):
                    rel = os.path.join(day_rel, ses_dir)
                    # '1_0' -> 1; '2NoRot' -> 2 (the number before the session type)
                    pos = re.match(r'(\d+)(?=[_\s]|[A-Za-z])', ses_dir.strip())
                    ses = v.extract_session_type(rel, arena)
                    if pos is None or ses is None:
                        skipped.append(rel)
                        continue
                    rows.append({'animal': animal, 'arena': arena,
                                 'day': float(day.group(1)), 'position': int(pos.group(1)),
                                 'session_type': ses, 'folder': rel})
    for rel in skipped:
        print(f'  [skipped] not a recognised arena / day / session folder: {rel}')
    if not rows:
        raise RuntimeError(f'No session folders found in {root}')

    df = pd.DataFrame(rows)
    df['animal'] = pd.Categorical(df['animal'], ANIMAL_ORDER, ordered=True)
    df['arena']  = pd.Categorical(df['arena'], v.ARENA_TYPES, ordered=True)
    df = df.sort_values(['arena', 'animal', 'day', 'position']).reset_index(drop=True)

    key = ['arena', 'animal', 'day', 'position']
    dup = df[df.duplicated(key, keep=False)]
    if len(dup):
        raise ValueError('Two sessions with the same number on one day:\n'
                         + '\n'.join(dup['folder']))
    for (arena, animal, day), g in df.groupby(key[:3], observed=True):
        missing = sorted(set(range(1, g['position'].max() + 1)) - set(g['position']))
        if missing:
            print(f'  [gap] {animal} {arena} Day{day:g}: no folder for session '
                  f'{", ".join(map(str, missing))} (left empty)')
    return df


# ── Plot ────────────────────────────────────────────────────────────────────
def plot_arena(df: pd.DataFrame, arena: str):
    """Session rasters of one arena. Axes units are inches, y pointing down."""
    animals = [a for a in ANIMAL_ORDER if (df['animal'] == a).any()]
    n_pos   = int(df['position'].max())

    # Row centres
    row_y, spans, y = {}, {}, 0.0
    for i, animal in enumerate(animals):
        if i:
            y += ANIMAL_GAP
        top = y
        for day in sorted(df.loc[df['animal'] == animal, 'day'].unique()):
            row_y[(animal, day)] = y + ROW_H / 2
            y += ROW_H
        spans[animal] = (top, y)
    ras_h, ras_w = y, n_pos * SESSION_W

    x0    = MARGIN_L + ANIMAL_W + ANIMAL_PAD
    fig_w = x0 + ras_w + MARGIN_R
    fig_h = MARGIN_TOP + ras_h + AXIS_H + KEY_H
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax  = fig.add_axes([x0 / fig_w, (AXIS_H + KEY_H) / fig_h, ras_w / fig_w, ras_h / fig_h])
    ax.set_xlim(0, ras_w)
    ax.set_ylim(ras_h, 0)

    pad = (ROW_H - LINE_H) / 2
    for animal, (top, bottom) in spans.items():
        ax.add_patch(Rectangle((-ANIMAL_PAD - ANIMAL_W, top + pad), ANIMAL_W,
                               bottom - top - 2 * pad, facecolor=v.ANIMAL_COLORS[animal],
                               edgecolor='none', clip_on=False))
        ax.text(-ANIMAL_PAD - ANIMAL_W / 2, (top + bottom) / 2, animal, ha='center',
                va='center_baseline', fontsize=v.FS_TICK, fontweight='bold', clip_on=False)

    for r in df.itertuples():
        x = (r.position - 1) * SESSION_W + SESSION_GAP / 2
        ax.add_patch(Rectangle((x, row_y[(r.animal, r.day)] - LINE_H / 2),
                               SESSION_W - SESSION_GAP, LINE_H, edgecolor='none',
                               facecolor=v.SESSION_COLORS[r.session_type]['face']))

    ax.set_xticks([(i + 0.5) * SESSION_W for i in range(n_pos)])
    ax.set_xticklabels(range(1, n_pos + 1), fontsize=v.FS_TICK)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 4))
    ax.set_xlabel('Session sequence', fontsize=v.FS_LABEL)

    centre = (MARGIN_L + (x0 + ras_w)) / 2 / fig_w   # middle of animal blocks + rasters
    fig.text(centre, 1 - 0.12 / fig_h, ARENA_TITLES[arena], ha='center', va='top',
             fontsize=v.FS_TITLE, fontweight='bold')

    # Key: the session types drawn, as lines of the raster's thickness
    present = set(df['session_type'])
    types   = [t for t in v.ARENA_SESSION_TYPES[arena] if t in present]
    handles = [Line2D([], [], color=v.SESSION_COLORS[t]['face'], lw=LINE_H * 72 * 0.8,
                      solid_capstyle='butt') for t in types]
    fig.legend(handles, [v._label(t) for t in types], loc='center', ncol=len(types),
               bbox_to_anchor=(centre, KEY_H / 2 / fig_h), fontsize=v.FS_LEGEND,
               frameon=False, handlelength=1.8, handletextpad=0.5, columnspacing=1.3)
    return fig


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'Reading session folders in {DATA_ROOT}')
    df = collect_sessions(DATA_ROOT)
    df.to_csv(os.path.join(OUTPUT_DIR, 'SessionSequence_Sessions.csv'), index=False)

    for arena in v.ARENA_TYPES:
        sub = df[df['arena'] == arena]
        if sub.empty:
            print(f'{arena}: no sessions, no plot')
            continue
        fig = plot_arena(sub, arena)
        out = os.path.join(OUTPUT_DIR, f'SessionSequence_{arena}')
        fig.savefig(out + '.pdf', bbox_inches='tight')
        v._save(fig, out + '.png')
        n_days = sub.groupby('animal', observed=True)['day'].nunique()
        print(f'{arena}: {len(sub)} sessions; days: '
              + ', '.join(f'{a} {n}' for a, n in n_days.items()) + f'  -> {out}.png / .pdf')


if __name__ == '__main__':
    main()
