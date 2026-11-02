# -*- coding: utf-8 -*-
"""
Check the pixel->cm conversion of every session's tracking file against the
real arena geometry, and list the points that fall outside the arena. Tracking
files are only read, never changed.

Real geometry (cm):
    Open    disc, outer diameter 60
    Circle  annulus, outer diameter 80, inner diameter 72
    Linear  rectangle, 80 x 8, horizontal or vertical

Sessions: every folder <root>\\<animal>\\<arena>\\<day>\\<session> holding one
*_cm.csv file, from TRACKING_ROOTS; a session in several roots is taken from the
first. Columns are read by position: time, x_px, y_px, x_cm, y_cm.

Why two scales: pixel_to_cm_conversion_v4_batch.py sets each file's scale from
arena size / its pixel range, so a wrong range puts every frame of that file on a
wrong scale, and a wrongly scaled session would look as if its frames left the
arena. So the conversion is checked first:
    cm_per_px       the file's own scale, fitted from its pixel and cm columns
    cm_per_px_ref   median cm_per_px of the same animal + arena (same camera setup)
    cm_per_px_used  cm_per_px_ref if the file differs from it by > SCALE_FRAC, else
                    the file's own: small differences are normal (e.g. a narrow ring
                    fixes its own scale better than the median does)
    Positions used below = the file's cm x cm_per_px_used / cm_per_px.

For each session, at cm_per_px_used:
    1. Boundary. The nominal shape, at real size, is slid over the frames to the
       position holding the most frames, i.e. onto the concentration of points;
       stray points cannot move it. A Linear rectangle is also tried horizontal and
       vertical, each with a tilt up to +-MAX_TILT_SEARCH_DEG.
    2. Distance of every frame beyond that boundary (0 if inside; into the hole
       counts on the Circle track). Frames further than TOL_CM are out of bounds.
       Tested twice: on the cm columns (what the analyses use; the conversion had
       already interpolated its jumps), and on the raw pixel columns mapped to the
       same coordinates (stray points there can set the file's scale).
    3. Size of the footprint (frames within the boundary + TOL_CM), as percentile
       ranges (SIZE_PCTILE) so a few frames do not set it:
           Open / Circle  width over all directions (mean, min, max)
                          Circle: inner diameter (INNER_PCTILE of the radial distance)
           Linear         length along and width across the fitted rectangle
       Multiplied by cm_per_px / cm_per_px_used, this is the size in the file.

Flags:
    SCALE           the file's cm_per_px differs from cm_per_px_ref by > SCALE_FRAC
    SIZE            a footprint dimension IN THE FILE differs from nominal by > TOL_CM
    OUT_OF_BOUNDS   >= FLAG_MIN_OUT_FRAMES frames of the cm columns out of bounds
    PX_OUT          >= FLAG_MIN_OUT_FRAMES frames of the pixel columns out of bounds
    PARTIAL         footprint below PARTIAL_FRAC of nominal at cm_per_px_used: the
                    animal did not reach part of the arena (not an error by itself, but
                    it gives too large a scale, since the range was stretched to the arena)
    SHAPE           Open / Circle footprint not round at cm_per_px_used
                    (min / max width < MIN_ROUNDNESS)
n_out_after_cleaning: out-of-bounds frames left after PlaceCell_Main_v21's jump
removal and smoothing, i.e. those that reach the analyses. n_speed_jumps: frame-to-
frame steps above its artifact speed (at cm_per_px_used).

Output (OUTPUT_DIR):
    TrackingGeometry_Check.xlsx       'PerSession' (every session), 'Flagged'
    OutOfBounds_Frames.csv            every frame out of bounds in either the cm or the
                                      pixel columns, of every session
    Figures\\<session>.png             density map at cm_per_px_used, boundary (solid),
                                      boundary + TOL_CM (dashed), out-of-bounds frames
                                      (red: cm columns; orange: pixel columns)
    ContactSheet_<arena>.png          all sessions of an arena on one page
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe for batch runs
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Polygon
from scipy.signal import fftconvolve

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'PlaceCell', 'Codes', 'FinalCodes'))
import PlaceCell_Main_v21 as pc

# ── USER INPUT ──────────────────────────────────────────────────────────────
_THESIS = r'X:\NMR_group_data\Runita\Analysis\Thesis'

# In order of preference: a session in several roots is taken from the first.
TRACKING_ROOTS = [
    os.path.join(_THESIS, r'Corr_Data_SpkQltyFilt\SpikeQualityFilt\QualFiltData_AllCells'),
    os.path.join(_THESIS, 'Corr_Data'),
]
OUTPUT_DIR = os.path.join(_THESIS, r'Corr_Data_SpkQltyFilt\SpikeQualityFilt\TrackingGeometryCheck')

OPEN_DIAMETER_CM   = 60.0
CIRCLE_OUTER_CM    = 80.0
CIRCLE_INNER_CM    = 72.0
LINEAR_LENGTH_CM   = 80.0
LINEAR_WIDTH_CM    = 8.0
NOMINAL_SIZE_CM    = {'Open': OPEN_DIAMETER_CM, 'Circle': CIRCLE_OUTER_CM, 'Linear': LINEAR_LENGTH_CM}

TOL_CM        = 4.0    # allowed leeway around the nominal shape
TOL_SMALL_CM  = 2.0    # also counted, for information

SIZE_PCTILE   = 0.1    # footprint widths: percentile range (this, 100 - this)
INNER_PCTILE  = 1.0    # Circle inner diameter: this percentile of the radial distance

SCALE_FRAC           = 0.065 # SCALE flag: fraction from the reference (= TOL_CM on the 60 cm arena)
FLAG_MIN_OUT_FRAMES  = 5     # OUT_OF_BOUNDS / PX_OUT need at least this many frames
PARTIAL_FRAC         = 0.9   # PARTIAL flag: footprint below this fraction of nominal
MIN_ROUNDNESS        = 0.9   # SHAPE flag, Open / Circle
MAX_TILT_SEARCH_DEG  = 6     # Linear: tilts tried, in 1-degree steps
ARENAS = ('Open', 'Linear', 'Circle')
# ─────────────────────────────────────────────────────────────────────────────


def find_sessions():
    """(relative session folder, arena, csv path, root), the first root winning."""
    sessions = {}
    for root in TRACKING_ROOTS:
        for dirpath in glob.glob(os.path.join(glob.escape(root), '*', '*', '*', '*')):
            if not os.path.isdir(dirpath):
                continue
            rel = os.path.relpath(dirpath, root)
            arena = next((a for a in ARENAS if rel.split(os.sep)[1].lower() == a.lower()), None)
            if arena is None or rel.lower() in sessions:
                continue
            tracking = [f for f in os.listdir(dirpath) if f.lower().endswith('_cm.csv')]
            if len(tracking) == 1:
                sessions[rel.lower()] = (rel, arena, os.path.join(dirpath, tracking[0]), root)
            elif len(tracking) > 1:
                print(f'WARNING: {len(tracking)} *_cm.csv files in {dirpath}; skipped.')
    return sorted(sessions.values())


def cm_per_px(px, cm):
    """Slope of cm on px: least squares, refitted on the frames within 1 cm of the
    first fit (frames interpolated during conversion do not match their pixels).
    Returns (slope, fraction of frames further than 1 cm from the final line)."""
    slope, icpt = np.polyfit(px, cm, 1)
    near = np.abs(cm - (slope * px + icpt)) < 1.0
    if near.sum() > 2:
        slope, icpt = np.polyfit(px[near], cm[near], 1)
    return slope, float(np.mean(np.abs(cm - (slope * px + icpt)) >= 1.0))


def load_session(rel, arena, csv_path, root):
    data = pd.read_csv(csv_path)
    t  = data.iloc[:, 0].to_numpy(float)
    px = data.iloc[:, 1].to_numpy(float)
    py = data.iloc[:, 2].to_numpy(float)
    x  = data.iloc[:, 3].to_numpy(float)
    y  = data.iloc[:, 4].to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(t) & ~np.isin(x, [1, -1])
    t, px, py, x, y = t[valid], px[valid], py[valid], x[valid], y[valid]
    axis_px, axis_cm = (px, x) if np.ptp(px) >= np.ptp(py) else (py, y)
    scale, frac_mismatch = cm_per_px(axis_px, axis_cm)
    return dict(session=rel, animal=rel.split(os.sep)[0], arena=arena, tracking_root=os.path.basename(root),
                tracking_file=csv_path, corrected_file='was_interpolated' in data.columns,
                n_frames=len(x), n_invalid=int((~valid).sum()), cm_per_px=scale,
                pct_px_cm_mismatch=100.0 * frac_mismatch,
                frame=np.flatnonzero(valid), t=t, px=px, py=py, x=x, y=y)


def rotate(x, y, angle):
    """Coordinates along (u) and across (v) the direction `angle`."""
    return x * np.cos(angle) + y * np.sin(angle), -x * np.sin(angle) + y * np.cos(angle)


def shape_mask(arena, half):
    """Nominal shape on a 1-cm grid of (2 half + 1)^2 cells."""
    uu, vv = np.meshgrid(np.arange(-half, half + 1), np.arange(-half, half + 1), indexing='ij')
    if arena == 'Open':
        return np.hypot(uu, vv) <= OPEN_DIAMETER_CM / 2
    if arena == 'Circle':
        r = np.hypot(uu, vv)
        return (r >= CIRCLE_INNER_CM / 2) & (r <= CIRCLE_OUTER_CM / 2)
    return (np.abs(uu) <= LINEAR_LENGTH_CM / 2) & (np.abs(vv) <= LINEAR_WIDTH_CM / 2)


def place_shape(u, v, arena):
    """(centre u, centre v, frames held) of the position at which the nominal shape
    holds the most frames. Positions within 0.5 % of the best are averaged, so an
    arena the animal only partly covered gets the middle of the equally good ones."""
    half = int(np.ceil(NOMINAL_SIZE_CM[arena] / 2)) + 2
    u_edges = np.arange(np.floor(u.min()) - half, np.ceil(u.max()) + half + 1)
    v_edges = np.arange(np.floor(v.min()) - half, np.ceil(v.max()) + half + 1)
    counts, _, _ = np.histogram2d(u, v, bins=[u_edges, v_edges])
    score = fftconvolve(counts, shape_mask(arena, half).astype(float), mode='same')
    best = np.argwhere(score >= 0.995 * score.max())
    return u_edges[0] + 0.5 + best[:, 0].mean(), v_edges[0] + 0.5 + best[:, 1].mean(), score.max()


def fit_boundary(x, y, arena):
    """(angle, centre u, centre v). Open / Circle: no rotation. Linear: the best of
    horizontal / vertical with a small tilt; the least tilted wins a tie."""
    if arena != 'Linear':
        return (0.0, *place_shape(x, y, arena)[:2])
    tilts = sorted(range(-MAX_TILT_SEARCH_DEG, MAX_TILT_SEARCH_DEG + 1), key=abs)
    best = None
    for base in (0, 90):
        for tilt in tilts:
            angle = np.radians(base + tilt)
            cu, cv, held = place_shape(*rotate(x, y, angle), arena)
            if best is None or held > best[0] * 1.001:
                best = (held, angle, cu, cv)
    return best[1:]


def distance_outside(u, v, arena):
    """Distance (cm) of each frame beyond the nominal shape centred at (0, 0)."""
    if arena == 'Open':
        return np.maximum(np.hypot(u, v) - OPEN_DIAMETER_CM / 2, 0.0)
    if arena == 'Circle':
        r = np.hypot(u, v)
        return np.maximum(np.maximum(r - CIRCLE_OUTER_CM / 2, CIRCLE_INNER_CM / 2 - r), 0.0)
    du = np.maximum(np.abs(u) - LINEAR_LENGTH_CM / 2, 0.0)
    dv = np.maximum(np.abs(v) - LINEAR_WIDTH_CM / 2, 0.0)
    return np.hypot(du, dv)


def percentile_width(values):
    lo, hi = np.percentile(values, [SIZE_PCTILE, 100 - SIZE_PCTILE], axis=0)
    return hi - lo


def footprint_size(u, v, arena):
    """Footprint dimensions (cm), from frames in coordinates centred on the shape."""
    if arena == 'Linear':
        visited = np.unique(np.floor((u + LINEAR_LENGTH_CM / 2) / 2.0))
        return dict(length_cm=percentile_width(u), width_cm=percentile_width(v),
                    pct_length_visited=100.0 * np.isin(np.arange(LINEAR_LENGTH_CM / 2.0), visited).mean())
    dirs = np.radians(np.arange(0, 180, 5))
    widths = percentile_width(np.outer(u, np.cos(dirs)) + np.outer(v, np.sin(dirs)))
    size = dict(diameter_cm=widths.mean(), diameter_min_cm=widths.min(), diameter_max_cm=widths.max())
    if arena == 'Circle':
        size['inner_diameter_cm'] = 2 * np.percentile(np.hypot(u, v), INNER_PCTILE)
    return size


def size_errors(size, arena):
    """{dimension: measured - nominal}."""
    if arena == 'Open':
        return {'diameter': size['diameter_cm'] - OPEN_DIAMETER_CM}
    if arena == 'Circle':
        return {'outer diameter': size['diameter_cm'] - CIRCLE_OUTER_CM,
                'inner diameter': size['inner_diameter_cm'] - CIRCLE_INNER_CM}
    return {'length': size['length_cm'] - LINEAR_LENGTH_CM, 'width': size['width_cm'] - LINEAR_WIDTH_CM}


def n_runs(mask):
    return int(np.sum(np.diff(np.concatenate(([0], mask.astype(int)))) == 1))


def n_out_after_cleaning(s, ratio, x_ref, y_ref, to_shape):
    """Out-of-bounds frames left after PlaceCell_Main_v21's cleaning, at the
    reference scale. Its output has its origin moved, so it is shifted back using
    a frame kept by both."""
    xc, yc, tc = pc._load_tracking(s['tracking_file'], pc.arena_width_cm)
    if len(tc) < 2:
        return 0
    xs, ys = pc._smooth_tracking_position(xc * ratio, yc * ratio, tc)
    first = int(np.argmin(np.abs(s['t'] - tc[0])))
    u, v = to_shape(xs - xc[0] * ratio + x_ref[first], ys - yc[0] * ratio + y_ref[first])
    return int(np.sum(distance_outside(u, v, s['arena']) > TOL_CM))


def analyse_session(s, cm_per_px_ref):
    arena = s['arena']
    off_scale = abs(s['cm_per_px'] / cm_per_px_ref - 1) > SCALE_FRAC
    cm_per_px_used = cm_per_px_ref if off_scale else s['cm_per_px']
    ratio = cm_per_px_used / s['cm_per_px']   # file cm -> cm at cm_per_px_used
    x, y = s['x'] * ratio, s['y'] * ratio
    angle, cu, cv = fit_boundary(x, y, arena)

    def to_shape(xx, yy):
        uu, vv = rotate(xx, yy, angle)
        return uu - cu, vv - cv

    u, v = to_shape(x, y)
    dist = distance_outside(u, v, arena)
    out = dist > TOL_CM
    in_tol = ~out

    # Pixel columns in the same coordinates: same slope, offset from the typical frame.
    x_px = cm_per_px_used * s['px'] + np.median(x - cm_per_px_used * s['px'])
    y_px = cm_per_px_used * s['py'] + np.median(y - cm_per_px_used * s['py'])
    dist_px = distance_outside(*to_shape(x_px, y_px), arena)
    out_px = dist_px > TOL_CM

    size_used = footprint_size(u[in_tol], v[in_tol], arena)
    size_file = {k: (val / ratio if k.endswith('_cm') else val) for k, val in size_used.items()}

    dt = np.diff(s['t']) * 1e-6
    with np.errstate(divide='ignore', invalid='ignore'):
        step_speed = np.hypot(np.diff(x), np.diff(y)) / dt
    tilt = (np.degrees(angle) + 45) % 90 - 45

    row = {k: s[k] for k in ('session', 'animal', 'arena', 'tracking_root', 'tracking_file', 'corrected_file',
                             'n_frames', 'n_invalid', 'cm_per_px', 'pct_px_cm_mismatch')}
    row.update(cm_per_px_ref=cm_per_px_ref, cm_per_px_ratio=s['cm_per_px'] / cm_per_px_ref,
               cm_per_px_used=cm_per_px_used, px_range=max(np.ptp(s['px']), np.ptp(s['py'])),
               **size_file, **{f'used_{k}': val for k, val in size_used.items() if k.endswith('_cm')},
               tilt_deg=tilt if arena == 'Linear' else np.nan,
               n_beyond_2cm=int((dist > TOL_SMALL_CM).sum()),
               n_out=int(out.sum()), pct_out=100.0 * out.mean(), n_out_bouts=n_runs(out),
               max_dist_cm=float(dist.max()),
               n_out_after_cleaning=n_out_after_cleaning(s, ratio, x, y, to_shape),
               n_px_out=int(out_px.sum()), pct_px_out=100.0 * out_px.mean(), n_px_out_bouts=n_runs(out_px),
               max_px_dist_cm=float(dist_px.max()),
               n_speed_jumps=int(np.sum(step_speed > pc.POS_JUMP_THRESH_CMS)))
    keep = out | out_px
    frames = pd.DataFrame(dict(session=s['session'], frame_index=s['frame'][keep], time=s['t'][keep],
                               x_px=s['px'][keep], y_px=s['py'][keep], x_cm=s['x'][keep], y_cm=s['y'][keep],
                               dist_beyond_boundary_cm=dist[keep], dist_beyond_boundary_px_cols_cm=dist_px[keep]))
    centre = (cu * np.cos(angle) - cv * np.sin(angle), cu * np.sin(angle) + cv * np.cos(angle))
    plot_data = dict(x=x, y=y, out=out, x_px=x_px[out_px], y_px=y_px[out_px], arena=arena, centre=centre, angle=angle)
    return row, frames, plot_data


def add_flags(df):
    flags = []
    for _, row in df.iterrows():
        arena = row['arena']
        file_size = {k: row[k] for k in ('length_cm', 'width_cm', 'diameter_cm', 'inner_diameter_cm') if pd.notna(row.get(k))}
        used_size = {k: row[f'used_{k}'] for k in file_size}
        f = []
        if abs(row['cm_per_px_ratio'] - 1) > SCALE_FRAC:
            f.append('SCALE')
        if any(abs(e) > TOL_CM for e in size_errors(file_size, arena).values()):
            f.append('SIZE')
        if row['n_out'] >= FLAG_MIN_OUT_FRAMES:
            f.append('OUT_OF_BOUNDS')
        if row['n_px_out'] >= FLAG_MIN_OUT_FRAMES:
            f.append('PX_OUT')
        main_dim = 'length_cm' if arena == 'Linear' else 'diameter_cm'
        if used_size[main_dim] < PARTIAL_FRAC * NOMINAL_SIZE_CM[arena]:
            f.append('PARTIAL')
        if arena != 'Linear' and row['diameter_min_cm'] / row['diameter_max_cm'] < MIN_ROUNDNESS:
            f.append('SHAPE')
        flags.append(' + '.join(f))
    df['flags'] = flags
    return df


# ── Figures ─────────────────────────────────────────────────────────────────

def draw_boundary(ax, arena, centre, angle, grow, **style):
    cx, cy = centre
    if arena == 'Linear':
        hu, hv = LINEAR_LENGTH_CM / 2 + grow, LINEAR_WIDTH_CM / 2 + grow
        corners = np.array([[-hu, -hv], [hu, -hv], [hu, hv], [-hu, hv]])
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        ax.add_patch(Polygon(corners @ rot.T + [cx, cy], closed=True, fill=False, **style))
        return
    radii = ([OPEN_DIAMETER_CM / 2 + grow] if arena == 'Open'
             else [CIRCLE_OUTER_CM / 2 + grow, max(CIRCLE_INNER_CM / 2 - grow, 0.1)])
    for r in radii:
        ax.add_patch(Circle((cx, cy), r, fill=False, **style))


def draw_session(ax, plot_data, title, fontsize=8):
    x, y, out = plot_data['x'], plot_data['y'], plot_data['out']
    x_edges = np.arange(x.min() - 1, x.max() + 2, 1.0)
    y_edges = np.arange(y.min() - 1, y.max() + 2, 1.0)
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    ax.imshow(np.ma.masked_equal(counts, 0).T, origin='lower', cmap='viridis', norm=LogNorm(vmin=1),
              extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], interpolation='nearest')
    draw_boundary(ax, plot_data['arena'], plot_data['centre'], plot_data['angle'], 0.0, edgecolor='black', linewidth=1.0)
    draw_boundary(ax, plot_data['arena'], plot_data['centre'], plot_data['angle'], TOL_CM, edgecolor='black',
                  linewidth=0.7, linestyle='--')
    x_px, y_px = plot_data['x_px'], plot_data['y_px']
    if len(x_px):
        ax.scatter(x_px, y_px, s=6, marker='x', color='darkorange', linewidths=0.5, zorder=3)
    if out.any():
        ax.scatter(x[out], y[out], s=6, facecolors='none', edgecolors='red', linewidths=0.5, zorder=4)
    half = NOMINAL_SIZE_CM[plot_data['arena']] / 2 + TOL_CM + 2
    cx, cy = plot_data['centre']
    all_x, all_y = np.concatenate((x, x_px)), np.concatenate((y, y_px))
    ax.set_xlim(min(cx - half, all_x.min() - 2), max(cx + half, all_x.max() + 2))
    ax.set_ylim(min(cy - half, all_y.min() - 2), max(cy + half, all_y.max() + 2))
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 2)


def size_text(row, prefix=''):
    if row['arena'] == 'Linear':
        return f'{row[prefix + "length_cm"]:.0f} x {row[prefix + "width_cm"]:.1f}'
    if row['arena'] == 'Circle':
        return f'{row[prefix + "diameter_cm"]:.0f} / {row[prefix + "inner_diameter_cm"]:.0f}'
    return f'{row[prefix + "diameter_cm"]:.0f}'


def panel_title(row):
    flags = f' [{row["flags"]}]' if row['flags'] else ''
    return (f'{row["session"]}{flags}\nfile {size_text(row)} cm at {row["cm_per_px"]:.3f} cm/px; '
            f'used {size_text(row, "used_")} cm at {row["cm_per_px_used"]:.3f}\n'
            f'out: cm {row["n_out"]} (max {row["max_dist_cm"]:.0f} cm), '
            f'px {row["n_px_out"]} (max {row["max_px_dist_cm"]:.0f} cm)')


def save_figures(df, plots):
    fig_dir = os.path.join(OUTPUT_DIR, 'Figures')
    os.makedirs(fig_dir, exist_ok=True)
    for _, row in df.iterrows():
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        draw_session(ax, plots[row['session']], panel_title(row), fontsize=10)
        ax.set_xlabel('x (cm, at cm_per_px_used)')
        ax.set_ylabel('y (cm, at cm_per_px_used)')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, row['session'].replace(os.sep, '_') + '.png'), dpi=150)
        plt.close(fig)

    for arena in ARENAS:
        sub = df[df['arena'] == arena]
        n_col = 6
        n_row = int(np.ceil(len(sub) / n_col))
        fig, axes = plt.subplots(n_row, n_col, figsize=(3.3 * n_col, 3.8 * n_row), squeeze=False)
        for ax in axes.ravel():
            ax.axis('off')
        for ax, (_, row) in zip(axes.ravel(), sub.iterrows()):
            ax.axis('on')
            draw_session(ax, plots[row['session']], panel_title(row), fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f'ContactSheet_{arena}.png'), dpi=170)
        plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sessions = find_sessions()
    print(f'Found {len(sessions)} session(s) with tracking.')
    loaded = []
    for i, args in enumerate(sessions, start=1):
        print(f'  [{i}/{len(sessions)}] loading {args[0]}')
        loaded.append(load_session(*args))

    scales = pd.DataFrame([{k: s[k] for k in ('session', 'animal', 'arena', 'cm_per_px')} for s in loaded])
    refs = scales.groupby(['animal', 'arena'])['cm_per_px'].median()

    rows, frames, plots = [], [], {}
    for s in loaded:
        row, frm, plot_data = analyse_session(s, refs[(s['animal'], s['arena'])])
        rows.append(row)
        frames.append(frm)
        plots[s['session']] = plot_data

    df = add_flags(pd.DataFrame(rows))
    flagged = df[df['flags'] != '']
    with pd.ExcelWriter(os.path.join(OUTPUT_DIR, 'TrackingGeometry_Check.xlsx'), engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='PerSession', index=False)
        flagged.to_excel(writer, sheet_name='Flagged', index=False)
    pd.concat(frames, ignore_index=True).to_csv(os.path.join(OUTPUT_DIR, 'OutOfBounds_Frames.csv'), index=False)
    save_figures(df, plots)

    print(f'\n{len(flagged)} of {len(df)} session(s) flagged:')
    for _, row in flagged.iterrows():
        print(f'  {row["session"]}: {row["flags"]} | file {size_text(row)} cm at {row["cm_per_px"]:.3f} cm/px '
              f'(ratio {row["cm_per_px_ratio"]:.2f}); used {size_text(row, "used_")} cm; '
              f'cm out {row["n_out"]} (max {row["max_dist_cm"]:.1f} cm, {row["n_out_after_cleaning"]} after cleaning); '
              f'px out {row["n_px_out"]} (max {row["max_px_dist_cm"]:.1f} cm); {row["n_speed_jumps"]} speed jumps')
    print(f'\nDone. Results saved under {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
