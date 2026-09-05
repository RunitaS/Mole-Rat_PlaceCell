# -*- coding: utf-8 -*-
"""
Quadrant-fold matching diagnostics for MeanRateMap_QuadrantAnalysis_v2.py.

Builds the exact same handler objects (pure geometry -- no tracking/spike data
needed) and visualizes, step by step, how the whole-arena bin grid is split
into 4 quadrants and which bin in each quadrant is registered onto which bin
of the single reference quadrant (`handler._quad_idx_flat` / `n_quad_bins`,
the arrays that `_fold_mean_map` / `_fold_peak_bin` actually use).

For each arena this produces one figure with 4 panels:
  1. Raw quadrant membership (which of the 4 copies each bin belongs to).
  2. Fold colour-code: every bin is coloured by the *local* reference-quadrant
     coordinate it is registered to; identical colour = matched bin. Any
     asymmetry, transpose, or mis-registration shows up as a colour pattern
     that looks rotated/mirrored between quadrants instead of identical.
  3. A handful of labelled landmark bins (near-wall, near-center, corner,
     mid-wall) traced across all 4 quadrants with matching marker+colour, to
     make "wall a maps to wall a'" concrete.
  4. The folded reference quadrant alone -- the actual output of the fold.

Figures are saved as PNGs in the repository root.
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MeanRateMap_QuadrantAnalysis_v2 import (  # noqa: E402
    ARENA_CONFIGS, target_bin_cm, make_handler,
)

REPO_ROOT = r'c:\Runita\NMR\Mole-Rat_PlaceCell'


def _savefig_retry(fig, save_path: str, dpi: int, attempts: int = 5, delay_s: float = 0.5):
    """Windows occasionally holds a transient lock on a just-created/just-viewed PNG
    (indexing/AV scan); retry a few times before giving up."""
    for attempt in range(attempts):
        try:
            fig.savefig(save_path, dpi=dpi)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay_s)

LANDMARK_COLORS  = ['#e6194B', '#3cb44b', '#4363d8', '#f58231']
LANDMARK_MARKERS = ['o', 's', '^', 'D']


# ============================================================================
# Reflect-fold arenas (open_field, linear_track)
# ============================================================================

def _build_reflect_quadrant_id(nx: int, ny: int) -> np.ndarray:
    """Mirrors `_build_reflect_quadrant_fold`'s own transform list, but tags
    each bin with which of the 4 raw copies (1=reference as-is, 2=x-mirrored,
    3=y-mirrored, 4=both-mirrored) it belongs to, instead of the local index."""
    half_x, half_y = nx // 2, ny // 2
    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    transforms = [
        (I,              J,              1),
        (nx - 1 - I,     J,              2),
        (I,              ny - 1 - J,     3),
        (nx - 1 - I,     ny - 1 - J,     4),
    ]
    quad_id = np.zeros((nx, ny), dtype=int)
    for Ti, Tj, qid in transforms:
        sub_i = Ti[:half_x, :half_y]
        sub_j = Tj[:half_x, :half_y]
        quad_id[sub_i, sub_j] = qid
    return quad_id


def plot_reflect_fold_diagnostics(handler, arena_name: str, save_path: str,
                                   circle_diameter_cm: float = None):
    nx, ny = handler.nx, handler.ny
    bin_x = getattr(handler, 'bin_cm_x', getattr(handler, 'bin_cm', 1.0))
    bin_y = getattr(handler, 'bin_cm_y', getattr(handler, 'bin_cm', 1.0))
    half_x, half_y = handler.quad_shape
    quad_idx = handler._quad_idx_flat.reshape(nx, ny)
    quad_id = _build_reflect_quadrant_id(nx, ny)

    local_i = np.where(quad_idx >= 0, quad_idx // half_y, -1)
    local_j = np.where(quad_idx >= 0, quad_idx % half_y, -1)
    m = quad_idx >= 0
    rgb = np.full((nx, ny, 3), 0.85)
    rgb[..., 0][m] = local_i[m] / max(half_x - 1, 1)
    rgb[..., 1][m] = local_j[m] / max(half_y - 1, 1)
    rgb[..., 2][m] = 0.55

    landmarks = [
        ('near own wall (li=0,lj=0)',        0,                0),
        ('long-wall midpoint (wall a)',      half_x // 2,      0),
        ('short-wall midpoint (wall b)',     0,                half_y // 2),
        ('near-center corner (far wall)',    half_x - 1,       half_y - 1),
    ]

    x_edges = np.arange(nx + 1) * bin_x
    y_edges = np.arange(ny + 1) * bin_y
    extent = [0, nx * bin_x, 0, ny * bin_y]

    fig, axes = plt.subplots(1, 4, figsize=(21, 7.5))

    # --- Panel 1: raw quadrant membership -----------------------------------
    ax = axes[0]
    cmap4 = matplotlib.colors.ListedColormap(['#dddddd', '#f4a6a6', '#a6c8f4', '#a6f4b8', '#f4e2a6'])
    ax.pcolormesh(x_edges, y_edges, quad_id.T, cmap=cmap4, vmin=0, vmax=4, shading='auto')
    ax.axvline(half_x * bin_x, color='k', lw=1.5)
    ax.axhline(half_y * bin_y, color='k', lw=1.5)
    if circle_diameter_cm is not None:
        ax.add_patch(Circle((circle_diameter_cm / 2, circle_diameter_cm / 2),
                             circle_diameter_cm / 2, fill=False, ec='k', lw=1.2, ls=':'))
    ax.set_aspect('equal')
    ax.set_title('Step 1: raw arena split\ninto 4 quadrants')
    leg_quad = [Line2D([0], [0], marker='s', color='w', markerfacecolor=c, markersize=13, label=l)
                for c, l in zip(['#f4a6a6', '#a6c8f4', '#a6f4b8', '#f4e2a6'],
                                 ['Q1 reference (identity)', 'Q2 (x-mirrored)',
                                  'Q3 (y-mirrored)', 'Q4 (x+y mirrored)'])]

    # --- Panel 2: fold colour-code ------------------------------------------
    ax = axes[1]
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower', extent=extent)
    ax.axvline(half_x * bin_x, color='k', lw=1, ls='--')
    ax.axhline(half_y * bin_y, color='k', lw=1, ls='--')
    if circle_diameter_cm is not None:
        ax.add_patch(Circle((circle_diameter_cm / 2, circle_diameter_cm / 2),
                             circle_diameter_cm / 2, fill=False, ec='k', lw=1.2, ls=':'))
    ax.set_aspect('equal')
    ax.set_title('Step 2: fold colour-code\n(identical colour = matched bin)')

    # --- Panel 3: landmark bins traced across all 4 quadrants ---------------
    ax = axes[2]
    ax.imshow(np.ones((ny, nx, 3)), origin='lower', extent=extent)
    ax.axvline(half_x * bin_x, color='k', lw=1, ls='--')
    ax.axhline(half_y * bin_y, color='k', lw=1, ls='--')
    if circle_diameter_cm is not None:
        ax.add_patch(Circle((circle_diameter_cm / 2, circle_diameter_cm / 2),
                             circle_diameter_cm / 2, fill=False, ec='k', lw=1.2, ls=':'))
    leg_land = []
    for k, (label, li, lj) in enumerate(landmarks):
        target_local = li * half_y + lj
        hits = np.argwhere(quad_idx == target_local)
        color, marker = LANDMARK_COLORS[k % 4], LANDMARK_MARKERS[k % 4]
        for (i, j) in hits:
            ax.scatter((i + 0.5) * bin_x, (j + 0.5) * bin_y, s=130, color=color,
                       marker=marker, edgecolor='k', linewidth=0.9, zorder=3)
        leg_land.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor=color,
                                markeredgecolor='k', markersize=10, label=label))
    ax.set_aspect('equal')
    ax.set_title('Step 3: example bins matched\nacross all 4 quadrants')

    # --- Panel 4: folded reference quadrant (actual fold output) ------------
    ax = axes[3]
    ax.imshow(np.transpose(rgb[:half_x, :half_y], (1, 0, 2)), origin='lower',
              extent=[0, half_x * bin_x, 0, half_y * bin_y])
    ax.set_aspect('equal')
    ax.set_title(f'Step 4: folded reference quadrant\n(final output, {half_x}x{half_y} bins)')

    fig.suptitle(f'{arena_name}: quadrant fold matching diagnostic '
                 f'(full grid {nx}x{ny} bins, bin size {bin_x:.1f}x{bin_y:.1f} cm)', y=0.98)
    fig.legend(handles=leg_quad, loc='lower center', bbox_to_anchor=(0.27, 0.0),
               ncol=2, fontsize=8, frameon=False)
    fig.legend(handles=leg_land, loc='lower center', bbox_to_anchor=(0.63, 0.0),
               ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.13, 1, 0.92])
    _savefig_retry(fig, save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


# ============================================================================
# Roll-fold arena (circular_track)
# ============================================================================

def plot_ring_fold_diagnostics(handler, save_path: str):
    n_bins = handler.n_bins
    qn = handler.n_quad_bins
    quad_idx = handler._quad_idx_flat
    bin_width_deg = handler.bin_width_deg
    mean_r = (handler.inner_r + handler.outer_r) / 2.0

    theta_edges = np.deg2rad(np.arange(n_bins + 1) * bin_width_deg)
    r_edges = np.array([handler.inner_r, handler.outer_r])
    quad_id = (np.arange(n_bins) // qn + 1).astype(float)

    fig = plt.figure(figsize=(21, 7.0))

    # --- Panel 1: raw quadrant membership (4 arcs) ---------------------------
    ax = fig.add_subplot(1, 4, 1, projection='polar')
    ax.set_theta_zero_location('E'); ax.set_theta_direction(1)
    cmap4 = matplotlib.colors.ListedColormap(['#f4a6a6', '#a6c8f4', '#a6f4b8', '#f4e2a6'])
    ax.pcolormesh(theta_edges, r_edges, quad_id[None, :], cmap=cmap4, vmin=1, vmax=4, shading='auto')
    ax.set_ylim(0, handler.outer_r + 5); ax.set_yticklabels([]); ax.grid(False)
    ax.set_title('Step 1: ring split into\n4 equal 90 deg arcs')

    # --- Panel 2: fold colour-code (within-arc position) --------------------
    ax = fig.add_subplot(1, 4, 2, projection='polar')
    ax.set_theta_zero_location('E'); ax.set_theta_direction(1)
    cmap = matplotlib.colormaps['twilight'].copy() if hasattr(matplotlib.colormaps['twilight'], 'copy') \
        else plt.get_cmap('twilight')
    cmap.set_bad('white')
    vals = np.where(quad_idx >= 0, quad_idx, np.nan).astype(float)
    pcm = ax.pcolormesh(theta_edges, r_edges, vals[None, :], cmap=cmap, vmin=0, vmax=qn - 1, shading='auto')
    ax.set_ylim(0, handler.outer_r + 5); ax.set_yticklabels([]); ax.grid(False)
    ax.set_title('Step 2: fold colour-code\n(identical colour = matched bin)')

    # --- Panel 3: landmark bins traced across all 4 arcs ---------------------
    ax = fig.add_subplot(1, 4, 3, projection='polar')
    ax.set_theta_zero_location('E'); ax.set_theta_direction(1)
    ax.set_ylim(0, handler.outer_r + 5); ax.set_yticklabels([]); ax.grid(False)
    landmarks = [('arc start', 0), ('arc 1/4', qn // 4), ('arc mid', qn // 2), ('arc end', qn - 1)]
    leg = []
    for k, (label, local) in enumerate(landmarks):
        hits = np.where(quad_idx == local)[0]
        color, marker = LANDMARK_COLORS[k % 4], LANDMARK_MARKERS[k % 4]
        for b in hits:
            theta = np.deg2rad((b + 0.5) * bin_width_deg)
            ax.scatter(theta, mean_r, s=130, color=color, marker=marker,
                      edgecolor='k', linewidth=0.9, zorder=3)
        leg.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor=color,
                          markeredgecolor='k', markersize=10, label=label))
    ax.set_title('Step 3: example bins matched\nacross all 4 arcs')

    # --- Panel 4: folded reference arc (actual fold output) ------------------
    ax = fig.add_subplot(1, 4, 4, projection='polar')
    ax.set_theta_zero_location('E'); ax.set_theta_direction(1)
    ref_theta_edges = np.linspace(0, np.pi / 2, qn + 1)
    ax.pcolormesh(ref_theta_edges, r_edges, np.arange(qn, dtype=float)[None, :],
                 cmap=cmap, vmin=0, vmax=qn - 1, shading='auto')
    ax.set_thetamin(0); ax.set_thetamax(90)
    ax.set_ylim(0, handler.outer_r + 5); ax.set_yticklabels([]); ax.grid(False)
    ax.set_title(f'Step 4: folded reference arc\n(final output, {qn} bins)')

    fig.suptitle(f'Circular Track: quadrant fold matching diagnostic '
                 f'(full ring {n_bins} bins, {bin_width_deg:.2f} deg/bin)', y=0.98)
    fig.legend(handles=leg, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.1, 1, 0.92])
    _savefig_retry(fig, save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def _raw_quadrant_block(content: np.ndarray, quad_id: np.ndarray, k: int) -> np.ndarray:
    """The quadrant-k sub-array exactly as physically cut from the whole grid, in its
    own raw (un-mirrored) room orientation -- no transform applied."""
    rows = np.where(np.any(quad_id == k, axis=1))[0]
    cols = np.where(np.any(quad_id == k, axis=0))[0]
    return content[np.ix_(rows, cols)]


def _registered_quadrant_block(content: np.ndarray, quad_idx: np.ndarray, quad_id: np.ndarray,
                                k: int, half_x: int, half_y: int) -> np.ndarray:
    """The quadrant-k sub-array after applying the SAME registration the analysis code
    uses (gathered via `quad_idx`/`quad_id`, not a hand-derived flip) -- i.e. what that
    quadrant looks like once expressed in the reference quadrant's local coordinates."""
    mask = quad_id == k
    ii, jj = np.where(mask)
    local = quad_idx[ii, jj]
    li, lj = local // half_y, local % half_y
    shape = (half_x, half_y, content.shape[-1]) if content.ndim == 3 else (half_x, half_y)
    out = np.zeros(shape, dtype=content.dtype)
    out[li, lj] = content[ii, jj]
    return out


_QUAD_COLORS_REFLECT = {1: '#f4a6a6', 2: '#a6c8f4', 3: '#a6f4b8', 4: '#f4e2a6'}
_QUAD_LABELS_REFLECT = {1: 'Q1 (identity)', 2: 'Q2 (x-mirrored)',
                         3: 'Q3 (y-mirrored)', 4: 'Q4 (x+y mirrored)'}


def plot_full_protocol_reflect(handler, arena_name: str, save_path: str,
                                circle_diameter_cm: float = None):
    """Full cut -> register -> average protocol for a mirror-fold arena (open field /
    linear track): each of the 4 quadrants shown individually as physically cut, then
    individually after registration (should now look identical to one another), then
    averaged into the final folded reference quadrant."""
    nx, ny = handler.nx, handler.ny
    bin_x = getattr(handler, 'bin_cm_x', getattr(handler, 'bin_cm', 1.0))
    bin_y = getattr(handler, 'bin_cm_y', getattr(handler, 'bin_cm', 1.0))
    half_x, half_y = handler.quad_shape
    quad_idx = handler._quad_idx_flat.reshape(nx, ny)
    quad_id = _build_reflect_quadrant_id(nx, ny)

    local_i = np.where(quad_idx >= 0, quad_idx // half_y, -1)
    local_j = np.where(quad_idx >= 0, quad_idx % half_y, -1)
    m = quad_idx >= 0
    rgb = np.full((nx, ny, 3), 0.85)
    rgb[..., 0][m] = local_i[m] / max(half_x - 1, 1)
    rgb[..., 1][m] = local_j[m] / max(half_y - 1, 1)
    rgb[..., 2][m] = 0.55

    fig = plt.figure(figsize=(22, 15))
    gs = fig.add_gridspec(3, 20, height_ratios=[1, 0.85, 0.85], hspace=0.65, wspace=1.4)

    # --- Row 1: overview -----------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0:5])
    cmap5 = matplotlib.colors.ListedColormap(['#dddddd'] + [_QUAD_COLORS_REFLECT[k] for k in (1, 2, 3, 4)])
    ax1.pcolormesh(np.arange(nx + 1) * bin_x, np.arange(ny + 1) * bin_y, quad_id.T,
                   cmap=cmap5, vmin=0, vmax=4, shading='auto')
    ax1.axvline(half_x * bin_x, color='k', lw=1.5)
    ax1.axhline(half_y * bin_y, color='k', lw=1.5)
    if circle_diameter_cm is not None:
        ax1.add_patch(Circle((circle_diameter_cm / 2, circle_diameter_cm / 2),
                             circle_diameter_cm / 2, fill=False, ec='k', lw=1.2, ls=':'))
    ax1.set_aspect('equal')
    ax1.set_title('1. Whole arena ->\n4 raw quadrants')

    ax2 = fig.add_subplot(gs[0, 5:10])
    ax2.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower', extent=[0, nx * bin_x, 0, ny * bin_y])
    ax2.axvline(half_x * bin_x, color='k', lw=1, ls='--')
    ax2.axhline(half_y * bin_y, color='k', lw=1, ls='--')
    if circle_diameter_cm is not None:
        ax2.add_patch(Circle((circle_diameter_cm / 2, circle_diameter_cm / 2),
                             circle_diameter_cm / 2, fill=False, ec='k', lw=1.2, ls=':'))
    ax2.set_aspect('equal')
    ax2.set_title('2. Fold colour-code\n(identical colour = matched bin)')

    ax3 = fig.add_subplot(gs[0, 10:15])
    ax3.imshow(np.transpose(rgb[:half_x, :half_y], (1, 0, 2)), origin='lower',
              extent=[0, half_x * bin_x, 0, half_y * bin_y])
    ax3.set_aspect('equal')
    ax3.set_title('3. Folded reference quadrant\n(final averaged output)')

    ax4 = fig.add_subplot(gs[0, 15:20]); ax4.axis('off')
    leg0 = [Line2D([0], [0], marker='s', color='w', markerfacecolor=_QUAD_COLORS_REFLECT[k],
                   markersize=14, label=_QUAD_LABELS_REFLECT[k]) for k in (1, 2, 3, 4)]
    ax4.legend(handles=leg0, loc='center', fontsize=10, frameon=False, title='Quadrant key')

    # --- Row 2: each quadrant as physically cut (no transform) ---------------
    for idx, k in enumerate((1, 2, 3, 4)):
        ax = fig.add_subplot(gs[1, idx * 5:idx * 5 + 5])
        block = _raw_quadrant_block(rgb, quad_id, k)
        ax.imshow(np.transpose(block, (1, 0, 2)), origin='lower', aspect='equal')
        for spine in ax.spines.values():
            spine.set_edgecolor(_QUAD_COLORS_REFLECT[k]); spine.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{_QUAD_LABELS_REFLECT[k]}\nas physically cut (raw)', fontsize=9)

    # --- Row 3: each quadrant after registration, + their average ------------
    registered = []
    for idx, k in enumerate((1, 2, 3, 4)):
        ax = fig.add_subplot(gs[2, idx * 4:idx * 4 + 4])
        block = _registered_quadrant_block(rgb, quad_idx, quad_id, k, half_x, half_y)
        registered.append(block)
        ax.imshow(np.transpose(block, (1, 0, 2)), origin='lower', aspect='equal')
        for spine in ax.spines.values():
            spine.set_edgecolor(_QUAD_COLORS_REFLECT[k]); spine.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{_QUAD_LABELS_REFLECT[k]}\nregistered to reference frame', fontsize=9)

    ax_avg = fig.add_subplot(gs[2, 16:20])
    avg = np.mean(np.stack(registered, axis=0), axis=0)
    ax_avg.imshow(np.transpose(avg, (1, 0, 2)), origin='lower', aspect='equal')
    ax_avg.set_xticks([]); ax_avg.set_yticks([])
    ax_avg.set_title('Average of the 4\nregistered quadrants\n(= final fold output)', fontsize=9)

    fig.suptitle(f'{arena_name}: full quadrant-matching protocol '
                f'(grid {nx}x{ny} bins -> reference {half_x}x{half_y} bins)', fontsize=13, y=0.985)
    _savefig_retry(fig, save_path, dpi=180)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


def plot_full_protocol_ring(handler, save_path: str):
    """Full cut -> register -> average protocol for the circular track's roll fold:
    each of the 4 arcs shown individually at its true physical angular position, then
    individually rolled onto the reference 0-90 deg arc, then averaged."""
    n_bins = handler.n_bins
    qn = handler.n_quad_bins
    quad_idx = handler._quad_idx_flat
    bin_width_deg = handler.bin_width_deg

    quad_id = (np.arange(n_bins) // qn + 1).astype(int)
    quad_colors = {1: '#f4a6a6', 2: '#a6c8f4', 3: '#a6f4b8', 4: '#f4e2a6'}
    quad_labels = {1: 'Arc 1 (0-90 deg)', 2: 'Arc 2 (90-180 deg)',
                   3: 'Arc 3 (180-270 deg)', 4: 'Arc 4 (270-360 deg)'}

    cmap = matplotlib.colormaps['twilight'].copy() if hasattr(matplotlib.colormaps['twilight'], 'copy') \
        else plt.get_cmap('twilight')
    within_arc = quad_idx.astype(float)

    r_edges = np.array([handler.inner_r, handler.outer_r])
    theta_edges_full = np.deg2rad(np.arange(n_bins + 1) * bin_width_deg)
    theta_edges_ref = np.linspace(0, np.pi / 2, qn + 1)

    fig = plt.figure(figsize=(22, 15))
    gs = fig.add_gridspec(3, 20, height_ratios=[1, 0.85, 0.85], hspace=0.7, wspace=1.4)

    # --- Row 1: overview -------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0:5], projection='polar')
    ax1.set_theta_zero_location('E'); ax1.set_theta_direction(1)
    cmap4 = matplotlib.colors.ListedColormap([quad_colors[k] for k in (1, 2, 3, 4)])
    ax1.pcolormesh(theta_edges_full, r_edges, quad_id[None, :].astype(float),
                   cmap=cmap4, vmin=1, vmax=4, shading='auto')
    ax1.set_ylim(0, handler.outer_r + 5); ax1.set_yticklabels([]); ax1.grid(False)
    ax1.set_title('1. Whole ring ->\n4 raw arcs')

    ax2 = fig.add_subplot(gs[0, 5:10], projection='polar')
    ax2.set_theta_zero_location('E'); ax2.set_theta_direction(1)
    ax2.pcolormesh(theta_edges_full, r_edges, within_arc[None, :], cmap=cmap, vmin=0, vmax=qn - 1, shading='auto')
    ax2.set_ylim(0, handler.outer_r + 5); ax2.set_yticklabels([]); ax2.grid(False)
    ax2.set_title('2. Fold colour-code\n(identical colour = matched bin)')

    ax3 = fig.add_subplot(gs[0, 10:15], projection='polar')
    ax3.set_theta_zero_location('E'); ax3.set_theta_direction(1)
    ax3.pcolormesh(theta_edges_ref, r_edges, np.arange(qn, dtype=float)[None, :],
                  cmap=cmap, vmin=0, vmax=qn - 1, shading='auto')
    ax3.set_thetamin(0); ax3.set_thetamax(90)
    ax3.set_ylim(0, handler.outer_r + 5); ax3.set_yticklabels([]); ax3.grid(False)
    ax3.set_title('3. Folded reference arc\n(final averaged output)')

    ax4 = fig.add_subplot(gs[0, 15:20]); ax4.axis('off')
    leg0 = [Line2D([0], [0], marker='s', color='w', markerfacecolor=quad_colors[k],
                   markersize=14, label=quad_labels[k]) for k in (1, 2, 3, 4)]
    ax4.legend(handles=leg0, loc='center', fontsize=9, frameon=False, title='Arc key')

    # --- Row 2: each arc as physically cut (true room angle) -------------------
    for idx, k in enumerate((1, 2, 3, 4)):
        ax = fig.add_subplot(gs[1, idx * 5:idx * 5 + 5], projection='polar')
        seg = np.arange((k - 1) * qn, k * qn)
        seg_theta_edges = theta_edges_full[(k - 1) * qn: k * qn + 1]
        ax.set_theta_zero_location('E'); ax.set_theta_direction(1)
        ax.pcolormesh(seg_theta_edges, r_edges, within_arc[seg][None, :],
                     cmap=cmap, vmin=0, vmax=qn - 1, shading='auto')
        ax.set_thetamin(np.degrees(seg_theta_edges[0])); ax.set_thetamax(np.degrees(seg_theta_edges[-1]))
        ax.set_ylim(0, handler.outer_r + 5); ax.set_yticklabels([]); ax.grid(False)
        for spine in ax.spines.values():
            spine.set_edgecolor(quad_colors[k]); spine.set_linewidth(3)
        ax.set_title(f'{quad_labels[k]}\nas physically cut', fontsize=9)

    # --- Row 3: each arc rolled onto the reference arc, + their average --------
    registered = []
    for idx, k in enumerate((1, 2, 3, 4)):
        ax = fig.add_subplot(gs[2, idx * 4:idx * 4 + 4], projection='polar')
        seg = np.arange((k - 1) * qn, k * qn)
        vals = within_arc[seg]
        registered.append(vals)
        ax.set_theta_zero_location('E'); ax.set_theta_direction(1)
        ax.pcolormesh(theta_edges_ref, r_edges, vals[None, :], cmap=cmap, vmin=0, vmax=qn - 1, shading='auto')
        ax.set_thetamin(0); ax.set_thetamax(90)
        ax.set_ylim(0, handler.outer_r + 5); ax.set_yticklabels([]); ax.grid(False)
        for spine in ax.spines.values():
            spine.set_edgecolor(quad_colors[k]); spine.set_linewidth(3)
        ax.set_title(f'{quad_labels[k]}\nregistered (rolled to 0-90 deg)', fontsize=9)

    ax_avg = fig.add_subplot(gs[2, 16:20], projection='polar')
    avg = np.mean(np.stack(registered, axis=0), axis=0)
    ax_avg.set_theta_zero_location('E'); ax_avg.set_theta_direction(1)
    ax_avg.pcolormesh(theta_edges_ref, r_edges, avg[None, :], cmap=cmap, vmin=0, vmax=qn - 1, shading='auto')
    ax_avg.set_thetamin(0); ax_avg.set_thetamax(90)
    ax_avg.set_ylim(0, handler.outer_r + 5); ax_avg.set_yticklabels([]); ax_avg.grid(False)
    ax_avg.set_title('Average of the 4\nregistered arcs\n(= final fold output)', fontsize=9)

    fig.suptitle(f'Circular Track: full quadrant-matching protocol '
                f'(ring {n_bins} bins -> reference arc {qn} bins)', fontsize=13, y=0.985)
    _savefig_retry(fig, save_path, dpi=180)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


# ============================================================================
if __name__ == '__main__':
    handler_of = make_handler(ARENA_CONFIGS['open_field'])
    plot_reflect_fold_diagnostics(
        handler_of, 'Open Field',
        os.path.join(REPO_ROOT, 'QuadrantFold_Diagnostic_OpenField.png'),
        circle_diameter_cm=ARENA_CONFIGS['open_field']['diameter_cm'],
    )
    plot_full_protocol_reflect(
        handler_of, 'Open Field',
        os.path.join(REPO_ROOT, 'QuadrantMatching_FullProtocol_OpenField.png'),
        circle_diameter_cm=ARENA_CONFIGS['open_field']['diameter_cm'],
    )

    handler_lt = make_handler(ARENA_CONFIGS['linear_track'])
    plot_reflect_fold_diagnostics(
        handler_lt, 'Linear Track',
        os.path.join(REPO_ROOT, 'QuadrantFold_Diagnostic_LinearTrack.png'),
    )
    plot_full_protocol_reflect(
        handler_lt, 'Linear Track',
        os.path.join(REPO_ROOT, 'QuadrantMatching_FullProtocol_LinearTrack.png'),
    )

    handler_ct = make_handler(ARENA_CONFIGS['circular_track'])
    plot_ring_fold_diagnostics(
        handler_ct,
        os.path.join(REPO_ROOT, 'QuadrantFold_Diagnostic_CircularTrack.png'),
    )
    plot_full_protocol_ring(
        handler_ct,
        os.path.join(REPO_ROOT, 'QuadrantMatching_FullProtocol_CircularTrack.png'),
    )

    print('\nDone.')
