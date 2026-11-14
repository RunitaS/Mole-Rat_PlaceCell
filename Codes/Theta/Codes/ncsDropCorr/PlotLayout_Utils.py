"""Plot helpers shared by the arena / session-type comparison scripts.

Identical copies live in
    PlaceCell\\Codes\\FinalCodes
    Theta\\Codes\\ncsDropCorr
    PhasePrecession\\Codes
(each script imports whichever copy is first on its sys.path) -- keep them in sync.

match_ylims(axes)
    One y-range for several axes: the lowest lower limit and the highest upper
    limit of them all.
shared_y_axes(src_dirs)
    Context manager for the animal-controlled comparisons (Fa8477 + Fa5834
    Circle vs Open | Fa1059 + Fa23BD Linear vs Open), whose plots are saved to
    one folder per comparison and then shown side by side. While it is active,
    a figure saved under one of `src_dirs` is held back until the figure with
    the same relative path has been saved under every folder (or the block
    ends); the matching axes of those figures then get one y-range
    (match_figure_ylims) and all of them are written. The plotting code itself
    is unchanged.
label_bars(ax, bars)
    The % value above each bar (or, inside=True, in the middle of each stacked
    segment).
"""
import os
import threading
from contextlib import contextmanager

import numpy as np
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure


# ── Shared y-range ──────────────────────────────────────────────────────────

def _y_matchable(ax) -> bool:
    """Data axes whose y-range can be shared: visible rectilinear axes with an
    automatic aspect, no image (maps) and not a colour bar. Hidden text /
    legend panels (axis('off')) are left alone."""
    return (ax.get_visible() and ax.axison and ax.name == 'rectilinear'
            and ax.get_aspect() == 'auto' and not ax.images
            and not hasattr(ax, '_colorbar') and ax.get_label() != '<colorbar>')


def match_ylims(axes) -> None:
    """Give every axis in `axes` the y-range from the lowest lower limit to the
    highest upper limit among them (an inverted axis stays inverted)."""
    axes = [ax for ax in axes if ax is not None]
    if len(axes) < 2:
        return
    lims = [ax.get_ylim() for ax in axes]
    lo = min(min(l) for l in lims)
    hi = max(max(l) for l in lims)
    for ax, (a, b) in zip(axes, lims):
        ax.set_ylim((hi, lo) if a > b else (lo, hi))


def match_figure_ylims(figs, name: str = '') -> None:
    """Share the y-range between the corresponding data axes (same position in
    fig.axes, _y_matchable only) of several figures of the same plot. Figures
    with a different number of data axes are left unchanged (a note is
    printed), since their panels cannot be paired."""
    per_fig = [[ax for ax in f.axes if _y_matchable(ax)] for f in figs]
    per_fig = [a for a in per_fig if a]
    if len(per_fig) < 2:
        return
    if len({len(a) for a in per_fig}) != 1:
        print(f'  NOTE {name}: the side-by-side figures have different panels; '
              f'y-axes not matched.')
        return
    for axes in zip(*per_fig):
        match_ylims(list(axes))


@contextmanager
def shared_y_axes(src_dirs, match=None):
    """While active, figures saved (Figure.savefig, so also plt.savefig) under
    one of `src_dirs` -- one folder per comparison, in the order they are run --
    are held back. Once the figure with the same path relative to its folder
    has been saved under the last folder (or when the block ends), the held
    figures of that path get matched y-ranges (match_figure_ylims) and are all
    written. `match(rel_path) -> bool` limits this to some of the files (the
    rest are saved at once). Saving to anything else is not affected."""
    roots = [os.path.normcase(os.path.abspath(os.fspath(d))) for d in src_dirs]
    held = {}                      # rel path -> {folder index: (fig, fname, args, kwargs)}
    lock = threading.RLock()
    orig = Figure.savefig

    def _locate(fname):
        if not isinstance(fname, (str, os.PathLike)):
            return None
        p = os.path.normcase(os.path.abspath(os.fspath(fname)))
        for i, root in enumerate(roots):
            if p.startswith(root + os.sep):
                rel = os.path.relpath(p, root)
                if match is None or match(rel):
                    return i, rel
        return None

    def _flush(rel):
        entries = held.pop(rel)
        if len(entries) > 1:
            match_figure_ylims([e[0] for e in entries.values()], rel)
        for fig, fname, args, kwargs in entries.values():
            orig(fig, fname, *args, **kwargs)

    def savefig(self, fname, *args, **kwargs):
        loc = _locate(fname)
        if loc is None:
            return orig(self, fname, *args, **kwargs)
        i, rel = loc
        with lock:
            held.setdefault(rel, {})[i] = (self, fname, args, kwargs)
            if i == len(roots) - 1:          # every earlier comparison has run
                _flush(rel)
        return None

    Figure.savefig = savefig
    try:
        yield
    finally:
        Figure.savefig = orig
        with lock:
            for rel in list(held):
                _flush(rel)


# ── % labels on bars ────────────────────────────────────────────────────────

def _is_dark(color) -> bool:
    r, g, b = to_rgb(color)
    return 0.299 * r + 0.587 * g + 0.114 * b < 0.5


def label_bars(ax, bars, values=None, fmt='{:.1f}%', inside=False, fontsize=None,
               extra=None, pad_pt=2.0):
    """Write each bar's value (`values`, default its height) formatted with `fmt`
    above the bar, or with inside=True (stacked segments) in its middle -- white
    on dark fills -- skipping segments of height 0. NaN values get no label.
    `extra`: optional per-bar text added on a second line (e.g. 'n=12')."""
    bars = list(bars)
    if values is None:
        values = [b.get_height() for b in bars]
    extra = list(extra) if extra is not None else [None] * len(bars)
    fs = fontsize if fontsize is not None else 8
    for b, v, ex in zip(bars, values, extra):
        if v is None or not np.isfinite(v):
            continue
        text = fmt.format(v) + (f'\n{ex}' if ex not in (None, '') else '')
        x = b.get_x() + b.get_width() / 2
        if inside:
            if b.get_height() <= 0:
                continue
            ax.text(x, b.get_y() + b.get_height() / 2, text, ha='center', va='center',
                    fontsize=fs, zorder=6,
                    color='white' if _is_dark(b.get_facecolor()) else 'black')
        else:
            top = b.get_y() + max(b.get_height(), 0.0)
            ax.annotate(text, (x, top), xytext=(0, pad_pt), textcoords='offset points',
                        ha='center', va='bottom', fontsize=fs, zorder=6, annotation_clip=False,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
