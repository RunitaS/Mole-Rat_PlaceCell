# -*- coding: utf-8 -*-
"""
Local significant differences between the real-data KDEs and their observed-null KDEs
(Duong 2013, "Local significant differences from nonparametric two-sample tests",
J. Nonparametric Statistics 25:3, 635-645).

Inputs (data only -- nothing is imported from the pipelines that wrote them):
  KDE_<kind>.npz       real pooled-map KDEs        (MeanRateMap_QuadrantAnalysis_v24.py)
  Null_KDE_<kind>.npz  observed-null pooled KDEs   (ObservedNull_TiledPlaceFields.py)
  For each arena present in both files: '<arena>_density' (edge-corrected KDE, cm^-2),
  '<arena>_density_raw' (uncorrected), '<arena>_cov' (kernel bandwidth matrix H, cm^2),
  '<arena>_neff' (Kish effective sample size of the weighted KDE), '<arena>_domain' (bins the KDE
  was evaluated / edge-corrected over).

Method (Duong 2013, Sec. 2), applied per map kind x arena:
  1. Evaluation points x_j = bin centres of the COMMON domain (real domain & null domain). With
     RENORMALISE_ON_COMMON_DOMAIN both densities are rescaled to integrate to 1 over that common
     domain, so the comparison is of where the mass sits, not of how much of it fell outside.
  2. Local statistic U(x) = [f1(x) - f2(x)]^2, f1 = real KDE, f2 = observed-null KDE. Under the
     local null H0(x): f1(x) = f2(x),  X^2(x) = U(x) / sigma_U^2(x)  ->  chi^2 with 1 df, where
         sigma_U^2(x) = R(K) [ n1^-1 |H1|^-1/2 f1(x) + n2^-1 |H2|^-1/2 f2(x) ],
     R(K) = integral K^2 = 1 / (4 pi) for the 2D Gaussian kernel.
     Boundary term: the KDEs here are edge-corrected (raw KDE divided by the kernel mass m_H(x)
     inside the domain). For the Gaussian kernel K_H^2 = R(K) |H|^-1/2 K_{H/2}, so the variance of
     the corrected estimate is the paper's term multiplied by m_{H/2}(x) / m_H(x)^2, which is 1 in
     the interior and > 1 at a wall (fewer data points feed a wall bin's estimate). For an
     uncorrected KDE the factor reduces to m_{H/2}(x). Both masses are computed here over each
     KDE's own domain; the m_H computed here is checked against the correction the upstream KDE
     actually applied (density / density_raw) as a geometry sanity check.
  3. p_j = P(chi^2_1 >= X^2_j); Hochberg (1988) step-up over all m evaluation points of the
     (kind, arena) family: j* = max{ j : p_(j) <= alpha / (m - j + 1) }, reject p_(1..j*).
  4. Rejected points with f1 > f2: real density significantly ABOVE the observed null;
     f1 < f2: significantly BELOW. Each sign is split into 8-connected clusters of bins
     (wrapping around the circular track).

Sample sizes n1, n2 (SAMPLE_SIZE_MODE):
  'neff'  -- Kish effective sample size of each weighted KDE (the sample size the weighted
             estimator's variance actually scales with). For the peak-proportion maps this is ~ the
             number of (simulated) cells, i.e. a genuine sample of peak locations. For the overall
             and field-only maps the weighted points are BINS of a smoothed map, which are spatially
             correlated, so neff overstates the independent information there -> the test is liberal
             for those two map kinds; check with 'cells'.
  'cells' -- n1 = number of real place cells pooled (AllArenas_Summary.xlsx), n2 = number of
             simulated cells pooled (ObservedNull_Summary.xlsx). Conservative for the bin-weighted
             maps.

Outputs (OUTPUT_DIR, all prefixed DuongTest_):
  DuongTest_<kind>.png   rows = arenas; columns = real KDE, observed-null KDE, difference, signed z.
                         Fluorescent GREEN outline = cluster where real is significantly ABOVE the
                         observed null; fluorescent MAGENTA outline = significantly BELOW.
  DuongTest_Clusters.xlsx  'Tests' (one row per kind x arena) and 'Clusters' (one row per cluster)
  DuongTest_Results.npz    per-bin densities, variances, z, p, rejection, cluster labels
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT_DIRECTORY = r'X:\NMR_group_data\Runita\Analysis\Mean_KDE_Open_PascalOldenburg\Open_KDE\CorrectedData\Data\SessionTypeSorted_PC\Open\Cntrl'
KDE_DIR    = os.path.join(ROOT_DIRECTORY, 'MeanRM_Quad_v23', 'AllArenas')
OUTPUT_DIR = KDE_DIR

# map kind -> (real KDE npz, observed-null KDE npz, title)
MAP_KINDS = {
    'overall':    ('KDE_FigS1H_MeanFieldIndex.npz', 'Null_KDE_FigS1H_MeanFieldIndex.npz',
                   'Overall mean field-index map'),
    'field_only': ('KDE_FieldOnly_MeanFieldIndex.npz', 'Null_KDE_FieldOnly_MeanFieldIndex.npz',
                   'Field-only mean field-index map'),
    'peak':       ('KDE_PeakProportion_Map.npz', 'Null_KDE_PeakProportion_Map.npz',
                   'Peak proportion map'),
}

REAL_SUMMARY_XLSX = 'AllArenas_Summary.xlsx'      # used only with SAMPLE_SIZE_MODE = 'cells'
NULL_SUMMARY_XLSX = 'ObservedNull_Summary.xlsx'   # used only with SAMPLE_SIZE_MODE = 'cells'

ALPHA = 0.05                        # family-wise level per (map kind, arena), Hochberg step-up
SAMPLE_SIZE_MODE = 'neff'           # 'neff' or 'cells' (see docstring)
RENORMALISE_ON_COMMON_DOMAIN = True
MIN_CLUSTER_BINS = 1                # clusters smaller than this are not outlined / reported

GREEN_ABOVE   = '#39FF14'           # fluorescent green: real significantly ABOVE observed null
MAGENTA_BELOW = '#FF00FF'           # fluorescent magenta: real significantly BELOW observed null
OUTLINE_LW    = 2.2

R_K_GAUSS_2D = 1.0 / (4.0 * np.pi)  # R(K) = integral of K^2 for the standard bivariate normal

# Arena geometry -- must give the same bin grid as the pipelines that wrote the KDEs
# (2 x 2 cm bins; flat index = ix * ny + iy). Checked against the npz array lengths and against
# ObservedNull_Maps.npz '<arena>_nx_ny' when that file is present.
BIN_CM = 2.0
ARENAS = {
    'open_field':     dict(shape='disc', diameter_cm=60.0, title='Open Field'),
    'circular_track': dict(shape='ring', outer_diameter_cm=80.0, inner_diameter_cm=72.0,
                           title='Circular Track'),
    'linear_track':   dict(shape='rect', length_cm=80.0, width_cm=8.0, title='Linear Track'),
}
ARENA_ORDER = ['open_field', 'circular_track', 'linear_track']
ROW_HEIGHT  = {'open_field': 1.0, 'circular_track': 1.0, 'linear_track': 0.45}

EDGE_PTS = 6    # points per bin side, so ring bins follow the arcs


# ============================================================================
# Bin grids in room coordinates
# ============================================================================

class ArenaGrid:
    """Bin grid of one arena. Bin (i, j) spans grid coordinates u in [i, i+1], v in [j, j+1];
    edge_to_xy maps grid coordinates to room cm (for the ring: u = angular, v = radial)."""

    def __init__(self, key: str):
        cfg = ARENAS[key]
        self.key, self.title, self.shape = key, cfg['title'], cfg['shape']
        if self.shape == 'disc':
            self.radius = cfg['diameter_cm'] / 2.0
            self.nx = self.ny = int(np.ceil(cfg['diameter_cm'] / BIN_CM))
            self.dx = self.dy = BIN_CM
            self.cx = self.cy = self.radius
            self.wrap_x = False
        elif self.shape == 'ring':
            self.r_out = cfg['outer_diameter_cm'] / 2.0
            self.r_in = cfg['inner_diameter_cm'] / 2.0
            self.cx = self.cy = self.r_out
            circumference = np.pi * (self.r_out + self.r_in)
            self.nx = max(8, 4 * int(round(circumference / BIN_CM / 4.0)))
            self.ny = max(2, int(round((self.r_out - self.r_in) / BIN_CM)))
            self.dth = 2.0 * np.pi / self.nx
            self.dr = (self.r_out - self.r_in) / self.ny
            self.wrap_x = True
        else:
            self.length, self.width = cfg['length_cm'], cfg['width_cm']
            self.nx = max(4, int(round(self.length / BIN_CM)))
            self.ny = max(2, int(round(self.width / BIN_CM)))
            self.dx, self.dy = self.length / self.nx, self.width / self.ny
            self.wrap_x = False
        self.n_bins = self.nx * self.ny
        self.ix, self.iy = np.divmod(np.arange(self.n_bins), self.ny)
        self.xy = np.vstack(self.edge_to_xy(self.ix + 0.5, self.iy + 0.5))      # (2, n_bins)

        if self.shape == 'ring':
            r_lo = self.r_in + self.iy * self.dr
            self.area = 0.5 * self.dth * ((r_lo + self.dr) ** 2 - r_lo ** 2)
            r = np.hypot(self.xy[0] - self.cx, self.xy[1] - self.cy)
            self.dist_to_wall = np.minimum(r - self.r_in, self.r_out - r)
        else:
            self.area = np.full(self.n_bins, self.dx * self.dy)
            if self.shape == 'disc':
                r = np.hypot(self.xy[0] - self.cx, self.xy[1] - self.cy)
                self.dist_to_wall = np.clip(self.radius - r, 0.0, None)
            else:
                x, y = self.xy
                self.dist_to_wall = np.minimum.reduce([x, self.length - x, y, self.width - y])

    def edge_to_xy(self, u, v):
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        if self.shape == 'ring':
            th, r = u * self.dth, self.r_in + v * self.dr
            return self.cx + r * np.cos(th), self.cy + r * np.sin(th)
        return u * self.dx, v * self.dy

    def draw_outline(self, ax):
        kw = dict(fill=False, edgecolor='0.25', lw=0.9, zorder=3)
        if self.shape == 'disc':
            ax.add_patch(matplotlib.patches.Circle((self.cx, self.cy), self.radius, **kw))
        elif self.shape == 'ring':
            for rr in (self.r_in, self.r_out):
                ax.add_patch(matplotlib.patches.Circle((self.cx, self.cy), rr, **kw))
        else:
            ax.add_patch(matplotlib.patches.Rectangle((0, 0), self.length, self.width, **kw))

    def in_grid(self, i, j):
        if self.wrap_x:
            i %= self.nx
        return (0 <= i < self.nx) and (0 <= j < self.ny), i


# ============================================================================
# Kernel masses and the variance of each KDE (Duong 2013, Theorem 2.1 + boundary term)
# ============================================================================

def gaussian_mass_in_domain(eval_xy: np.ndarray, src_xy: np.ndarray, src_area: np.ndarray,
                            H: np.ndarray) -> np.ndarray:
    """m(x) = sum over domain bins y of N(x - y; 0, H) * area(y): the share of a Gaussian kernel
    of covariance H centred at x that falls inside the domain (Riemann sum over bins)."""
    H_inv = np.linalg.inv(H)
    coef = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(H)))
    d = eval_xy[:, :, None] - src_xy[:, None, :]                 # (2, n_eval, n_src)
    maha = np.einsum('aes,ab,bes->es', d, H_inv, d)
    return coef * np.exp(-0.5 * maha) @ src_area


def kde_side(z, arena: str, grid: ArenaGrid, common: np.ndarray, n: float) -> dict:
    """Density and its asymptotic variance on the common evaluation points for one KDE."""
    dens = z[f'{arena}_density']
    raw  = z[f'{arena}_density_raw']
    H    = np.asarray(z[f'{arena}_cov'], dtype=float)
    own  = z[f'{arena}_domain'].astype(bool)

    ev, src = grid.xy[:, common], grid.xy[:, own]
    m_H  = gaussian_mass_in_domain(ev, src, grid.area[own], H)
    m_H2 = gaussian_mass_in_domain(ev, src, grid.area[own], H / 2.0)

    f, f_raw = dens[common], raw[common]
    c = f / f_raw                                    # multiplier the upstream KDE applied to raw
    corrected = not np.allclose(c, 1.0, rtol=1e-6)
    edge_check = float(np.max(np.abs(c * m_H - 1.0))) if corrected else 0.0

    f_true = f_raw / m_H                             # boundary-unbiased estimate of f(x)
    var = c ** 2 * R_K_GAUSS_2D / (n * np.sqrt(np.linalg.det(H))) * f_true * m_H2

    scale = 1.0
    if RENORMALISE_ON_COMMON_DOMAIN:
        scale = 1.0 / float(np.sum(f * grid.area[common]))
    return dict(f=f * scale, var=var * scale ** 2, scale=scale, H=H, n=n,
                corrected=corrected, edge_check=edge_check,
                mass_outside_common=float(1.0 - np.sum(dens[common] * grid.area[common])
                                          / np.nansum(dens[own] * grid.area[own])))


def hochberg_reject(p: np.ndarray, alpha: float) -> tuple:
    """Hochberg (1988) step-up. Returns (reject mask, p cut-off or nan, j*)."""
    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    ok = p_sorted <= alpha / (m - np.arange(1, m + 1) + 1)
    reject = np.zeros(m, dtype=bool)
    if not ok.any():
        return reject, float('nan'), 0
    j_star = int(np.flatnonzero(ok).max()) + 1
    reject[order[:j_star]] = True
    return reject, float(p_sorted[j_star - 1]), j_star


# ============================================================================
# Clusters
# ============================================================================

def label_clusters(grid: ArenaGrid, mask_flat: np.ndarray) -> tuple:
    """8-connected components of mask_flat (wrapping in u on the ring). (labels, n), -1 = none."""
    m2 = mask_flat.reshape(grid.nx, grid.ny)
    lab = np.full(grid.n_bins, -1, dtype=int)
    lab2 = lab.reshape(grid.nx, grid.ny)
    n = 0
    for i0, j0 in np.argwhere(m2):
        if lab2[i0, j0] >= 0:
            continue
        lab2[i0, j0] = n
        stack = [(i0, j0)]
        while stack:
            i, j = stack.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ok, ii = grid.in_grid(i + di, j + dj)
                    if ok and m2[ii, j + dj] and lab2[ii, j + dj] < 0:
                        lab2[ii, j + dj] = n
                        stack.append((ii, j + dj))
        n += 1
    return lab, n


def outline_segments(grid: ArenaGrid, mask_flat: np.ndarray) -> list:
    """Every bin side separating a bin of mask_flat from a bin outside it, in room cm."""
    m2 = mask_flat.reshape(grid.nx, grid.ny)
    s = np.linspace(0.0, 1.0, EDGE_PTS)
    one = np.ones_like(s)

    def inside(i, j):
        ok, ii = grid.in_grid(i, j)
        return ok and m2[ii, j]

    segs = []
    for i, j in np.argwhere(m2):
        for (di, dj), (u, v) in (((-1, 0), (i * one, j + s)), ((1, 0), ((i + 1) * one, j + s)),
                                 ((0, -1), (i + s, j * one)), ((0, 1), (i + s, (j + 1) * one))):
            if not inside(i + di, j + dj):
                segs.append(np.column_stack(grid.edge_to_xy(u, v)))
    return segs


def cluster_table(kind: str, grid: ArenaGrid, common: np.ndarray, res: dict) -> list:
    rows = []
    idx = np.flatnonzero(common)
    for direction, labels, n in (('real > null', res['lab_above'], res['n_above']),
                                 ('real < null', res['lab_below'], res['n_below'])):
        for k in range(n):
            sel = labels[idx] == k                      # positions within the common domain
            b = idx[sel]
            a = grid.area[b]
            row = dict(map_kind=kind, arena=grid.key, direction=direction, cluster=k + 1,
                       n_bins=int(len(b)), area_cm2=round(float(a.sum()), 2),
                       centroid_x_cm=round(float(np.sum(grid.xy[0, b] * a) / a.sum()), 2),
                       centroid_y_cm=round(float(np.sum(grid.xy[1, b] * a) / a.sum()), 2),
                       mean_dist_to_wall_cm=round(float(np.sum(grid.dist_to_wall[b] * a) / a.sum()), 2),
                       min_dist_to_wall_cm=round(float(grid.dist_to_wall[b].min()), 2),
                       max_dist_to_wall_cm=round(float(grid.dist_to_wall[b].max()), 2),
                       mean_real_density=float(np.mean(res['f_real'][sel])),
                       mean_null_density=float(np.mean(res['f_null'][sel])),
                       mean_ratio_real_over_null=round(float(np.mean(res['f_real'][sel] / res['f_null'][sel])), 4),
                       excess_mass=round(float(np.sum((res['f_real'][sel] - res['f_null'][sel]) * a)), 5),
                       peak_abs_z=round(float(np.max(np.abs(res['z'][sel]))), 3),
                       min_p=float(np.min(res['p'][sel])))
            if grid.shape == 'ring':
                ang = np.arctan2(np.sum(np.sin(grid.ix[b] * grid.dth + grid.dth / 2) * a),
                                 np.sum(np.cos(grid.ix[b] * grid.dth + grid.dth / 2) * a))
                row['centroid_angle_deg'] = round(float(np.degrees(ang) % 360.0), 1)
            rows.append(row)
    return rows


# ============================================================================
# One (map kind, arena) comparison
# ============================================================================

def sample_sizes(arena: str, real_z, null_z) -> tuple:
    if SAMPLE_SIZE_MODE == 'neff':
        return float(real_z[f'{arena}_neff']), float(null_z[f'{arena}_neff'])
    if SAMPLE_SIZE_MODE == 'cells':
        real_df = pd.read_excel(os.path.join(KDE_DIR, REAL_SUMMARY_XLSX))
        null_df = pd.read_excel(os.path.join(KDE_DIR, NULL_SUMMARY_XLSX), sheet_name='Arenas')
        n_real = int((real_df['arena'] == arena).sum())
        n_null = int(null_df.loc[null_df['arena'] == arena, 'n_cells_pooled'].iloc[0])
        return float(n_real), float(n_null)
    raise ValueError(f'Unknown SAMPLE_SIZE_MODE {SAMPLE_SIZE_MODE!r}')


def compare_arena(arena: str, grid: ArenaGrid, real_z, null_z) -> dict:
    common = (real_z[f'{arena}_domain'].astype(bool) & null_z[f'{arena}_domain'].astype(bool)
              & np.isfinite(real_z[f'{arena}_density']) & np.isfinite(null_z[f'{arena}_density'])
              & (real_z[f'{arena}_density_raw'] > 0) & (null_z[f'{arena}_density_raw'] > 0))
    n_real, n_null = sample_sizes(arena, real_z, null_z)
    real = kde_side(real_z, arena, grid, common, n_real)
    null = kde_side(null_z, arena, grid, common, n_null)

    diff = real['f'] - null['f']
    sd = np.sqrt(real['var'] + null['var'])
    z = diff / sd
    x2 = z ** 2
    p = chi2.sf(x2, df=1)
    reject, p_cut, j_star = hochberg_reject(p, ALPHA)

    above_flat = np.zeros(grid.n_bins, dtype=bool)
    below_flat = np.zeros(grid.n_bins, dtype=bool)
    above_flat[np.flatnonzero(common)[reject & (diff > 0)]] = True
    below_flat[np.flatnonzero(common)[reject & (diff < 0)]] = True

    out = dict(common=common, real=real, null=null, f_real=real['f'], f_null=null['f'],
               diff=diff, z=z, x2=x2, p=p, reject=reject, p_cut=p_cut, j_star=j_star)
    for tag, mask in (('above', above_flat), ('below', below_flat)):
        lab, n = label_clusters(grid, mask)
        sizes = np.bincount(lab[lab >= 0], minlength=n)
        keep = np.flatnonzero(sizes >= MIN_CLUSTER_BINS)
        remap = np.full(max(n, 1), -1)
        remap[keep] = np.arange(len(keep))
        lab = np.where(lab >= 0, remap[np.clip(lab, 0, None)], -1)
        out[f'lab_{tag}'], out[f'n_{tag}'] = lab, len(keep)
    return out


# ============================================================================
# Plotting (true arena geometry)
# ============================================================================

def bin_polygons(grid: ArenaGrid, bins: np.ndarray) -> np.ndarray:
    s = np.linspace(0.0, 1.0, EDGE_PTS)
    du = np.concatenate([s, np.ones_like(s), s[::-1], np.zeros_like(s)])
    dv = np.concatenate([np.zeros_like(s), s, np.ones_like(s), s[::-1]])
    x, y = grid.edge_to_xy(grid.ix[bins][:, None] + du, grid.iy[bins][:, None] + dv)
    return np.stack([x, y], axis=-1)


def draw_map(ax, grid: ArenaGrid, common: np.ndarray, values: np.ndarray, cmap, norm):
    bins = np.flatnonzero(common)
    pc = PolyCollection(bin_polygons(grid, bins), array=values, cmap=cmap, norm=norm,
                        edgecolors='face', linewidths=0.3, zorder=1)
    ax.add_collection(pc)
    grid.draw_outline(ax)
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.axis('off')
    return pc


def draw_cluster_outlines(ax, grid: ArenaGrid, res: dict):
    halo = [pe.Stroke(linewidth=OUTLINE_LW + 1.8, foreground='black'), pe.Normal()]
    for tag, colour in (('above', GREEN_ABOVE), ('below', MAGENTA_BELOW)):
        mask = res[f'lab_{tag}'] >= 0
        if mask.any():
            lc = LineCollection(outline_segments(grid, mask), colors=colour, linewidths=OUTLINE_LW,
                                capstyle='round', joinstyle='round', zorder=6)
            lc.set_path_effects(halo)
            ax.add_collection(lc)


def plot_kind(kind: str, title: str, grids: dict, results: dict, save_path: str):
    arenas = [a for a in ARENA_ORDER if a in results]
    ratios = [ROW_HEIGHT[a] for a in arenas]
    fig = plt.figure(figsize=(23, 1.2 + 5.6 * sum(ratios)))
    gs = fig.add_gridspec(len(arenas), 4, height_ratios=ratios)
    dens_cmap = plt.get_cmap('Greys')
    div_cmap = plt.get_cmap('RdBu_r')

    for r, arena in enumerate(arenas):
        grid, res = grids[arena], results[arena]
        common = res['common']
        both = np.concatenate([res['f_real'], res['f_null']])
        dens_norm = Normalize(vmin=float(both.min()), vmax=float(both.max()))
        dmax = float(np.max(np.abs(res['diff']))) or 1e-12
        zmax = float(np.max(np.abs(res['z']))) or 1.0
        z_star = float(norm.isf(res['p_cut'] / 2.0)) if np.isfinite(res['p_cut']) else float('nan')

        panels = [
            (res['f_real'], dens_cmap, dens_norm, 'density (cm$^{-2}$)',
             f"{grid.title} -- REAL KDE\n"
             f"n1 = {res['real']['n']:.1f}, H1 sd = {np.sqrt(res['real']['H'][0, 0]):.1f} x "
             f"{np.sqrt(res['real']['H'][1, 1]):.1f} cm"),
            (res['f_null'], dens_cmap, dens_norm, 'density (cm$^{-2}$)',
             f"OBSERVED-NULL KDE\n"
             f"n2 = {res['null']['n']:.1f}, H2 sd = {np.sqrt(res['null']['H'][0, 0]):.1f} x "
             f"{np.sqrt(res['null']['H'][1, 1]):.1f} cm"),
            (res['diff'], div_cmap, TwoSlopeNorm(0.0, -dmax, dmax), 'real - null (cm$^{-2}$)',
             'Difference  f$_{real}$ - f$_{null}$'),
            (res['z'], div_cmap, TwoSlopeNorm(0.0, -zmax, zmax), 'signed z = (f$_1$ - f$_2$) / $\\sigma_U$',
             f"Duong local test, Hochberg alpha = {ALPHA:g}\n"
             f"m = {common.sum()} points, {res['reject'].sum()} rejected "
             + (f"(|z| >= {z_star:.2f})" if np.isfinite(z_star) else '(none)')
             + f"; clusters: {res['n_above']} above, {res['n_below']} below"),
        ]
        for c, (vals, cmap, nrm, cbar_label, ttl) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            pc = draw_map(ax, grid, common, vals, cmap, nrm)
            draw_cluster_outlines(ax, grid, res)
            ax.set_title(ttl, fontsize=9)
            fig.colorbar(pc, ax=ax, orientation='horizontal', fraction=0.05, pad=0.03,
                         label=cbar_label)

    handles = [Line2D([], [], color=GREEN_ABOVE, lw=OUTLINE_LW,
                      path_effects=[pe.Stroke(linewidth=OUTLINE_LW + 1.8, foreground='black'), pe.Normal()],
                      label='Real significantly ABOVE observed null'),
               Line2D([], [], color=MAGENTA_BELOW, lw=OUTLINE_LW,
                      path_effects=[pe.Stroke(linewidth=OUTLINE_LW + 1.8, foreground='black'), pe.Normal()],
                      label='Real significantly BELOW observed null')]
    fig.legend(handles=handles, loc='upper right', ncol=2, frameon=False, fontsize=10)
    renorm = 'renormalised on the common domain' if RENORMALISE_ON_COMMON_DOMAIN else 'as estimated'
    fig.suptitle(f'{title}: real vs observed-null KDE -- Duong (2013) local significant differences\n'
                 f'(densities {renorm}; n = {SAMPLE_SIZE_MODE}; blank = outside the common domain)',
                 x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97 - 0.15 / (1.2 + 5.6 * sum(ratios))))
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {save_path}')


# ============================================================================
# Main
# ============================================================================

def arenas_in(z) -> set:
    return {k[:-len('_density')] for k in z.files if k.endswith('_density')}


def check_grid(arena: str, grid: ArenaGrid, z):
    n = len(z[f'{arena}_density'])
    if n != grid.n_bins:
        raise ValueError(f'{arena}: KDE has {n} bins, geometry here gives {grid.n_bins} '
                         f'({grid.nx} x {grid.ny}) -- check ARENAS / BIN_CM')
    maps_path = os.path.join(KDE_DIR, 'ObservedNull_Maps.npz')
    if os.path.exists(maps_path):
        with np.load(maps_path) as mz:
            if f'{arena}_nx_ny' in mz.files and tuple(mz[f'{arena}_nx_ny']) != (grid.nx, grid.ny):
                raise ValueError(f'{arena}: null grid {tuple(mz[f"{arena}_nx_ny"])} != '
                                 f'({grid.nx}, {grid.ny})')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    grids = {a: ArenaGrid(a) for a in ARENA_ORDER}
    test_rows, cluster_rows, saved = [], [], {}

    for kind, (real_file, null_file, title) in MAP_KINDS.items():
        real_path, null_path = os.path.join(KDE_DIR, real_file), os.path.join(KDE_DIR, null_file)
        if not (os.path.exists(real_path) and os.path.exists(null_path)):
            print(f'[SKIP {kind}] missing {real_file if not os.path.exists(real_path) else null_file}')
            continue
        real_z, null_z = np.load(real_path), np.load(null_path)
        results = {}
        for arena in ARENA_ORDER:
            if arena not in arenas_in(real_z) or arena not in arenas_in(null_z):
                continue
            grid = grids[arena]
            check_grid(arena, grid, real_z)
            check_grid(arena, grid, null_z)
            res = compare_arena(arena, grid, real_z, null_z)
            results[arena] = res

            for side in ('real', 'null'):
                if res[side]['corrected'] and res[side]['edge_check'] > 0.02:
                    print(f'  [WARN] {kind}/{arena}/{side}: edge-correction mass here differs from '
                          f'the upstream KDE by up to {100 * res[side]["edge_check"]:.1f} %')
            print(f'[{kind}] {arena}: m = {res["common"].sum()}, rejected = {res["reject"].sum()} '
                  f'(p cut-off {res["p_cut"]:.3g}), clusters above = {res["n_above"]}, '
                  f'below = {res["n_below"]}')

            test_rows.append(dict(
                map_kind=kind, arena=arena, alpha=ALPHA, sample_size_mode=SAMPLE_SIZE_MODE,
                n1_real=round(res['real']['n'], 2), n2_null=round(res['null']['n'], 2),
                H1=np.round(res['real']['H'], 3).tolist(), H2=np.round(res['null']['H'], 3).tolist(),
                m_points=int(res['common'].sum()),
                real_domain_bins=int(real_z[f'{arena}_domain'].sum()),
                null_domain_bins=int(null_z[f'{arena}_domain'].sum()),
                real_mass_outside_common=round(res['real']['mass_outside_common'], 4),
                null_mass_outside_common=round(res['null']['mass_outside_common'], 4),
                hochberg_j_star=res['j_star'], hochberg_p_cutoff=res['p_cut'],
                n_rejected=int(res['reject'].sum()),
                n_bins_real_above=int(np.sum(res['reject'] & (res['diff'] > 0))),
                n_bins_real_below=int(np.sum(res['reject'] & (res['diff'] < 0))),
                n_clusters_above=res['n_above'], n_clusters_below=res['n_below'],
                min_p=float(res['p'].min()), max_abs_z=round(float(np.abs(res['z']).max()), 3),
                edge_check_real=round(res['real']['edge_check'], 5),
                edge_check_null=round(res['null']['edge_check'], 5)))
            cluster_rows += cluster_table(kind, grid, res['common'], res)

            pre = f'{kind}_{arena}_'
            for name, vals in (('f_real', res['f_real']), ('f_null', res['f_null']),
                               ('var_real', res['real']['var']), ('var_null', res['null']['var']),
                               ('z', res['z']), ('p', res['p'])):
                arr = np.full(grid.n_bins, np.nan)
                arr[res['common']] = vals
                saved[pre + name] = arr
            rej = np.zeros(grid.n_bins, dtype=bool)
            rej[res['common']] = res['reject']
            saved[pre + 'reject'] = rej
            saved[pre + 'common_domain'] = res['common']
            saved[pre + 'cluster_above'] = res['lab_above']
            saved[pre + 'cluster_below'] = res['lab_below']

        if results:
            plot_kind(kind, title, grids, results, os.path.join(OUTPUT_DIR, f'DuongTest_{kind}.png'))

    if test_rows:
        xlsx = os.path.join(OUTPUT_DIR, 'DuongTest_Clusters.xlsx')
        cluster_cols = ['map_kind', 'arena', 'direction', 'cluster', 'n_bins', 'area_cm2',
                        'centroid_x_cm', 'centroid_y_cm', 'centroid_angle_deg',
                        'mean_dist_to_wall_cm', 'min_dist_to_wall_cm', 'max_dist_to_wall_cm',
                        'mean_real_density', 'mean_null_density', 'mean_ratio_real_over_null',
                        'excess_mass', 'peak_abs_z', 'min_p']
        clusters = pd.DataFrame(cluster_rows).reindex(columns=cluster_cols)
        with pd.ExcelWriter(xlsx) as xw:
            pd.DataFrame(test_rows).to_excel(xw, sheet_name='Tests', index=False)
            clusters.to_excel(xw, sheet_name='Clusters', index=False)
        print(f'[SAVED] {xlsx}')
        npz = os.path.join(OUTPUT_DIR, 'DuongTest_Results.npz')
        np.savez_compressed(npz, **saved)
        print(f'[SAVED] {npz}')


if __name__ == '__main__':
    main()
    print('\nDone.')
