# -*- coding: utf-8 -*-
"""
Observed NULL mean rate maps from a UNIFORM, NON-OVERLAPPING TILING of perfect place fields,
read out along the animals' real (concatenated) trajectory, and the Duong (2013) local test of
the real KDEs against this null.

Null question: if the arena were covered uniformly by place fields that never share a bin, and
the animals had moved exactly as they did (their tracking), what would the pooled overall /
field-only / peak-proportion maps -- and their KDEs -- look like? A real-vs-null difference is
then field placement that the trajectory + the analysis pipeline do not explain on their own.

Step 1  Concatenated occupancy (concatenated_occupancy): every session folder of the arena under
        TRACKING_ROOT_DIRECTORY holding exactly one tracking file is loaded through the REAL
        pipeline's tracking path (base._session_positions -> base.compute_cell_ratemap with no
        spikes), so centring, position smoothing, the speed filter and the 2 x 2 cm binning are
        the ones behind the real rate maps. Concatenating the trajectories = summing the
        sessions' dwell-time maps. Bins with >= base.min_occ_s of concatenated dwell are sampled.

Step 2  Tiling (tile_shape / make_tiles): the arena's bin grid is cut into non-overlapping
        sx x sy tiles, one simulated place cell each. The field starts at FIELD_SIZE_BINS x
        FIELD_SIZE_BINS (3 x 3 = 9 bins); per axis it grows by 1..MAX_FIELD_SIZE_EXTRA bins until
        it divides that axis evenly, and an axis narrower than the field is used whole (then the
        other side grows until the field has >= FIELD_SIZE_BINS^2 bins). With the current grids:
        open field 30 x 30 bins -> 3 x 3, circular track 120 x 2 -> 5 x 2, linear track
        40 x 4 -> 4 x 4.
        Rate profile (pass-index style): 1 at the peak bin, falling linearly with distance from
        the tile centre to 0 one bin beyond the tile's farthest bin, and 0 outside the tile --
        every tile bin is > 20 % of peak and no two fields share a bin. Tiles are cut by the
        arena wall (open-field circle, linear-track ends) as a real field would be.
        TILE_ALL_PHASES: the tiling is repeated at every offset of the tile lattice (sx * sy
        tilings, each non-overlapping on its own) so every bin is equally likely to hold a field
        peak; False = one tiling anchored at bin (0, 0), whose peak map is a lattice.

Step 3  Read-out along the trajectory (simulate_arena): a tile cell's spike count in bin b is
        Poisson(PEAK_RATE_HZ * rate_b * concatenated dwell_b) -- the sum over every frame the
        animal spent in b -- and its raw rate map = spikes / dwell (NOISE = 'poisson'; 'none'
        uses the expected rates, one draw). Every simulated cell then goes through the REAL
        per-cell pipeline (handler.smooth, base.field_index_map, base.extract_place_field_mask
        -- >= FIELD_PEAK_FRAC (20 %) of the smoothed peak AND > the cell's mean rate, >=
        base.MIN_FIELD_BINS contiguous bins -- and the raw peak bin) and the real pooling:
          overall    : mean field index per bin over all cells
          field_only : field index inside each cell's field, 0 elsewhere, mean over all cells
          peak       : % of cells whose raw peak is in the bin
        over N_ITER Poisson draws. A cell with no spikes in any sampled bin is dropped. A peak
        tie (equal raw rates, e.g. the plateau of an even-sized field without noise) splits
        that cell's peak count equally between the tied bins.

Step 4  KDE of the three null maps (base.kde_density_map: same Scott bandwidth and edge
        correction as the real maps).

Step 5  Duong local test, real KDE vs null KDE (base.duong_local_test, Hochberg-adjusted,
        alpha = base.DUONG_ALPHA; n null = pooled simulated cells), plotted by
        base.plot_duong_comparison (magenta outline: real higher, cyan outline: null higher).
        Real place cells: NullMeanRateMap_SimulatedFields_v2.load_real_sessions (the real
        pipeline's session selection and rate maps, without the SIR bootstrap).

Outputs (OUTPUT_DIR), TAG = TrajectoryTiling_<N_ITER>x:
  Null_<TAG>_Tiling.png      : concatenated occupancy per arena with the tile layout (phase 0)
  Null_<TAG>_PooledMaps.png  : the three null maps
  Null_<TAG>_KDE.png         : their KDEs
  Null_<TAG>.npz             : maps, occupancy and KDE outputs (key names read by
                               base.compare_kde_to_null)
  Null_<TAG>_Summary.xlsx    : 'Arenas' (field size, tiles, cells, KDE bandwidths), 'Sessions'
  Duong_Real_vs_<TAG>.png / .npz / .xlsx
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import MeanRateMap_QuadrantAnalysis_v21 as base          # noqa: E402
import NullMeanRateMap_SimulatedFields_v2 as reloc      # noqa: E402  (real cells, grid helpers)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Folder searched (recursively) for the tracking files of Step 1. Every sub-folder whose path
# contains an 'Open' / 'Linear' / 'Circle' component and holds exactly one *_cm.csv is used.
# The REAL place cells of Step 5 still come from base.ROOT_DIRECTORY (v21).
TRACKING_ROOT_DIRECTORY = r'X:\NMR_group_data\Runita\Analysis\Mean_KDE_Open_PascalOldenburg\Open_KDE\CorrectedData\Data\SessionTypeSorted_PC\Open\Cntrl'

FIELD_SIZE_BINS     = 3        # starting field side (bins): 3 x 3 = 9 bins
MAX_FIELD_SIZE_EXTRA = 3        # a side may grow by up to this many bins to tile the grid evenly
TILE_ALL_PHASES      = True     # average over every offset of the tile lattice (see Step 2)
PEAK_RATE_HZ         = 1.0      # field peak rate (Hz); sets the Poisson noise level only
NOISE                = 'poisson'   # 'poisson' (spikes drawn along the trajectory) or 'none'
N_ITER               = 20       # Poisson draws (forced to 1 when NOISE = 'none')

OUTPUT_DIR = os.path.join(base.OUTPUT_DIR, 'AllArenas')
TAG = f'TrajectoryTiling_{N_ITER if NOISE == "poisson" else 1}x'

_MAP_KINDS = {  # kind -> (values_are_mass for kde_density_map, title, colorbar label)
    'overall':    (False, 'Overall mean field index', 'Field index (a.u.)'),
    'field_only': (False, 'Field-only mean field index', 'Field index, background = 0 (a.u.)'),
    'peak':       (True,  'Peak proportion', '% of cells with peak in bin'),
}


# ============================================================================
# Step 1: concatenated occupancy from every tracking file
# ============================================================================

def find_tracking_files(arena_key: str) -> list:
    """(session_name, tracking_path) for every folder of this arena holding exactly one tracking
    file (same file rule as the real pipeline; no .ntt needed, no coverage criteria)."""
    out_dir = os.path.normcase(os.path.abspath(base.OUTPUT_DIR))
    sessions = []
    for dirpath, dirnames, filenames in os.walk(TRACKING_ROOT_DIRECTORY):
        dirnames[:] = [d for d in dirnames
                       if not os.path.normcase(os.path.abspath(os.path.join(dirpath, d))).startswith(out_dir)]
        if base._detect_arena_key(dirpath) != arena_key:
            continue
        tracking_all = [f for f in filenames if f.lower().endswith(('.csv', '.xlsx'))]
        if base.COORD_UNITS == 'cm':
            tracking = [f for f in tracking_all if f.lower().endswith('_cm.csv')]
        else:
            tracking = [f for f in tracking_all if not f.lower().endswith('_cm.csv')]
        if len(tracking) == 1:
            sessions.append((os.path.relpath(dirpath, TRACKING_ROOT_DIRECTORY),
                             os.path.join(dirpath, tracking[0])))
        elif len(tracking) > 1:
            print(f'  [SKIP] {dirpath}: {len(tracking)} tracking files, expected 1')
    return sorted(sessions)


def concatenated_occupancy(arena_key: str, handler) -> tuple:
    """(dwell s per bin summed over sessions, sampled-bin mask, per-session rows)."""
    geom = reloc._geom_valid(handler)
    occ = np.zeros(handler.n_bins)
    rows = []
    for session_name, path in find_tracking_files(arena_key):
        try:
            x_cm, y_cm, t = base._session_positions(path, handler)
            cell = (base.compute_cell_ratemap(x_cm, y_cm, t, np.empty(0), handler)
                    if len(t) >= 2 else None)
        except Exception as e:
            print(f'  ERROR [{arena_key}] {session_name}: {e}')
            continue
        if cell is None or cell['occ_map'].sum() <= 0:
            print(f'  [SKIP no tracking] [{arena_key}] {session_name}')
            continue
        s_occ = np.where(geom, cell['occ_map'], 0.0)
        occ += s_occ
        rows.append(dict(arena=arena_key, session=session_name,
                         tracking_file=os.path.basename(path),
                         dwell_s=round(float(s_occ.sum()), 2),
                         bins_sampled_in_session=int((s_occ >= base.min_occ_s).sum())))
    valid = (occ >= base.min_occ_s) & geom
    print(f'[{arena_key}] {len(rows)} tracking sessions, {occ.sum() / 60.0:.1f} min concatenated, '
          f'{int(valid.sum())}/{int(geom.sum())} bins sampled')
    return occ, valid, rows


# ============================================================================
# Step 2: uniform non-overlapping tiling
# ============================================================================

def _axis_field_size(n_axis: int, start: int) -> int:
    """Smallest side in [start, FIELD_SIZE_BINS + MAX_FIELD_SIZE_EXTRA] dividing n_axis evenly;
    the whole axis if it is narrower than FIELD_SIZE_BINS."""
    if n_axis <= FIELD_SIZE_BINS:
        return n_axis
    for s in range(start, FIELD_SIZE_BINS + MAX_FIELD_SIZE_EXTRA + 1):
        if n_axis % s == 0:
            return s
    print(f'  [WARN] no field side in {start}..{FIELD_SIZE_BINS + MAX_FIELD_SIZE_EXTRA} divides '
          f'{n_axis} bins -- using {start}, the last tile along this axis is partial')
    return start


def tile_shape(handler) -> tuple:
    sx = _axis_field_size(handler.nx, FIELD_SIZE_BINS)
    sy = _axis_field_size(handler.ny, FIELD_SIZE_BINS)
    target = FIELD_SIZE_BINS ** 2
    # a thin axis (the ring track is 2 bins wide): lengthen the other side to keep >= target bins
    for _ in range(MAX_FIELD_SIZE_EXTRA):
        if sx * sy >= target:
            break
        if sy < sx or sy == handler.ny:
            sx = _axis_field_size(handler.nx, sx + 1)
        else:
            sy = _axis_field_size(handler.ny, sy + 1)
    return sx, sy


def make_tiles(handler, sx: int, sy: int, px: int, py: int) -> list:
    """One non-overlapping tiling with its lattice offset by (px, py) bins: a list of
    (flat bin indices, rate profile) per tile, tiles cut to the arena geometry."""
    wrap = reloc._wraps(handler)
    geom = reloc._geom_valid(handler)
    ii, jj = np.meshgrid(np.arange(sx), np.arange(sy), indexing='ij')
    d = np.hypot(ii - (sx - 1) / 2.0, jj - (sy - 1) / 2.0)
    prof = 1.0 - d / (d.max() + 1.0)          # 0 one bin beyond the farthest tile bin
    prof = (prof / prof.max()).ravel()          # peak bin = 1
    ii, jj = ii.ravel(), jj.ravel()

    if wrap:
        x_starts = np.arange(px, px + handler.nx, sx)
    else:
        x_starts = np.arange(px - sx if px else 0, handler.nx, sx)
    y_starts = np.arange(py - sy if py else 0, handler.ny, sy)

    tiles = []
    for x0 in x_starts:
        for y0 in y_starts:
            bx, by = x0 + ii, y0 + jj
            if wrap:
                bx = bx % handler.nx
            keep = (bx >= 0) & (bx < handler.nx) & (by >= 0) & (by < handler.ny)
            flat = bx[keep] * handler.ny + by[keep]
            inside = geom[flat]
            if inside.any():
                tiles.append((flat[inside], prof[keep][inside]))
    return tiles


def make_tilings(handler) -> tuple:
    sx, sy = tile_shape(handler)
    pxs = range(sx) if (TILE_ALL_PHASES and sx < handler.nx) else [0]
    pys = range(sy) if (TILE_ALL_PHASES and sy < handler.ny) else [0]
    tilings = [make_tiles(handler, sx, sy, px, py) for px in pxs for py in pys]
    return sx, sy, tilings


# ============================================================================
# Step 3: read-out along the trajectory, real per-cell pipeline, pooling
# ============================================================================

def run_cell(handler, occ, valid, fr_raw):
    """Real per-cell pipeline on one simulated raw rate map -> (fi_map, field_mask, peak bins)."""
    if not (fr_raw[valid] > 0).any():
        return None
    fr_smooth = handler.smooth(fr_raw, valid)
    fi_map = base.field_index_map(fr_smooth, valid)
    pi = occ[valid] / occ[valid].sum()
    mean_fr = round(float(np.sum(pi * fr_smooth[valid])), 4)   # as compute_cell_ratemap
    field_mask = base.extract_place_field_mask(
        dict(valid=valid, fr_smooth=fr_smooth, mean_fr=mean_fr), handler)
    peak_bins = np.flatnonzero(valid & (fr_raw == fr_raw[valid].max()))
    return fi_map, field_mask, peak_bins


class TilingPool:
    """Running-sum equivalents of pool_fine_map / pool_field_only_map / pool_peak_proportion_map
    (every simulated cell shares the concatenated sampled bins)."""

    def __init__(self, valid: np.ndarray):
        n = len(valid)
        self.valid       = valid
        self.fi_sum      = np.zeros(n)
        self.fo_sum      = np.zeros(n)
        self.n_field     = np.zeros(n, dtype=np.int64)
        self.peak_counts = np.zeros(n)
        self.n_cells     = 0

    def add(self, fi_map, field_mask, peak_bins):
        v = self.valid
        self.fi_sum[v] += fi_map[v]
        m = field_mask & v
        self.fo_sum[m] += fi_map[m]
        self.n_field[m] += 1
        self.peak_counts[peak_bins] += 1.0 / len(peak_bins)
        self.n_cells += 1

    def maps(self) -> dict:
        """kind -> (values, shown-bins mask), matching the base pooling functions."""
        n, v = len(self.valid), self.valid
        overall, field_only, peak = (np.full(n, np.nan) for _ in range(3))
        f = (self.n_field > 0) & v
        if self.n_cells > 0:
            overall[v] = self.fi_sum[v] / self.n_cells
            field_only[f] = self.fo_sum[f] / self.n_cells
            peak[v] = self.peak_counts[v] / self.n_cells * 100.0
        return {'overall': (overall, v), 'field_only': (field_only, f), 'peak': (peak, v)}


def simulate_arena(handler, occ, valid, tilings, arena_key) -> tuple:
    pool = TilingPool(valid)
    n_dropped = 0
    n_iter = N_ITER if NOISE == 'poisson' else 1
    occ_v = occ[valid]
    for it in range(n_iter):
        rng = np.random.default_rng(base._stable_seed('null_tiling', arena_key, it))
        for tiles in tilings:
            for flat, prof in tiles:
                lam = np.zeros(handler.n_bins)
                lam[flat] = PEAK_RATE_HZ * prof
                fr_raw = np.zeros(handler.n_bins)
                if NOISE == 'poisson':
                    fr_raw[valid] = rng.poisson(lam[valid] * occ_v) / occ_v
                else:
                    fr_raw[valid] = lam[valid]
                cell = run_cell(handler, occ, valid, fr_raw)
                if cell is None:
                    n_dropped += 1
                    continue
                pool.add(*cell)
        print(f'  [{arena_key} tiling] draw {it + 1}/{n_iter}', end='\r')
    print(f'\n  [{arena_key} tiling] {pool.n_cells} simulated cells pooled, '
          f'{n_dropped} dropped (no spikes in sampled bins)')
    return pool, n_dropped


# ============================================================================
# Plots
# ============================================================================

def plot_tiling(handlers, arena_occ, arena_tilings, arena_shape, out_dir):
    keys = [k for k in base._ARENA_ORDER if k in arena_occ]
    fig, axes = plt.subplots(len(keys), 1, figsize=(9, 4.5 * len(keys)), squeeze=False)
    for ax, key in zip(axes[:, 0], keys):
        h = handlers[key]
        occ, valid = arena_occ[key]
        cmap, norm = base.make_cmap_norm(occ[valid])
        pcm = h.plot_bins_2d(ax, occ, valid, cmap, norm)
        fig.colorbar(pcm, ax=ax, label='Concatenated dwell time (s)')
        tiles = arena_tilings[key][0]
        for flat, _ in tiles:
            m = np.zeros(h.n_bins, dtype=bool)
            m[flat] = True
            base._outline_bin_clusters(ax, h, m, 'k', lw=0.6)
        sx, sy = arena_shape[key]
        ax.set_title(f'{base._ARENA_TITLES[key]} -- field {sx} x {sy} bins, {len(tiles)} tiles '
                     f'(phase 0 of {len(arena_tilings[key])})', fontsize=9)
    fig.suptitle('Concatenated occupancy (all tracking files) and the uniform field tiling')
    fig.tight_layout()
    p = os.path.join(out_dir, f'Null_{TAG}_Tiling.png')
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {p}')


def plot_null(handlers, pooled, kdes, n_cells, out_dir):
    keys = [k for k in base._ARENA_ORDER if k in pooled]
    noise = f'Poisson, {N_ITER} draws' if NOISE == 'poisson' else 'no noise'
    sim_desc = f'uniform non-overlapping field tiling read out along the concatenated trajectory ({noise})'

    fig = plt.figure(figsize=(5 * len(keys), 13))
    for row, (kind, (_, title, label)) in enumerate(_MAP_KINDS.items()):
        axes = reloc._arena_axes(fig, 3, keys, row, polar_for_ring=True)
        for key in keys:
            values, shown = pooled[key][kind]
            cmap, norm = base.make_cmap_norm(values[shown])
            pcm = handlers[key].plot_fine(axes[key], values, shown, cmap, norm)
            axes[key].set_title(f'{base._ARENA_TITLES[key]} -- {title}\n'
                                f'({n_cells[key]} sim. cells)', fontsize=9)
            fig.colorbar(pcm, ax=axes[key], shrink=0.7, label=label)
    fig.suptitle(f'NULL pooled maps -- {sim_desc}')
    fig.tight_layout()
    p = os.path.join(out_dir, f'Null_{TAG}_PooledMaps.png')
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {p}')

    fig = plt.figure(figsize=(6 * len(keys), 13))
    for row, (kind, (_, title, _)) in enumerate(_MAP_KINDS.items()):
        axes = reloc._arena_axes(fig, 3, keys, row, polar_for_ring=False)
        for key in keys:
            ax = axes[key]
            kde_out = kdes[key].get(kind)
            if kde_out is None:
                ax.set_title(f'{base._ARENA_TITLES[key]} -- {title}\n(no KDE: too few data)')
                ax.axis('off')
                continue
            domain = pooled[key]['peak'][1]
            density = kde_out['density']
            cmap, norm = base.make_cmap_norm(density[domain])
            pcm = handlers[key].plot_bins_2d(ax, density, domain & np.isfinite(density), cmap, norm)
            sd = np.sqrt(np.diag(kde_out['cov']))
            ax.set_title(f'{base._ARENA_TITLES[key]} -- {title}\n'
                         f'Scott factor={kde_out["factor"]:.3f}, neff={kde_out["neff"]:.0f}, '
                         f'BW sd={sd[0]:.1f} x {sd[1]:.1f} cm', fontsize=9)
            fig.colorbar(pcm, ax=ax, label='KDE density (cm$^{-2}$)')
    corr = 'edge-corrected' if base.KDE_EDGE_CORRECTION else 'no edge correction'
    fig.suptitle(f'NULL KDE (Scott\'s rule, {corr}) -- {sim_desc}')
    fig.tight_layout()
    p = os.path.join(out_dir, f'Null_{TAG}_KDE.png')
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {p}')


# ============================================================================
# Step 5: Duong test, real KDE vs tiling-null KDE
# ============================================================================

def duong_real_vs_null(handlers, real_results, null_kdes, null_domains, null_n, out_dir):
    tests, saved, summary_rows, bin_rows = {}, {}, [], []
    for key in base._ARENA_ORDER:
        results = real_results.get(key) or []
        if not results or key not in null_kdes:
            continue
        handler = handlers[key]
        domain_r = base._union_valid(handler, results)
        xy = handler.bin_centres_xy()
        edge = getattr(handler, 'edge_zone_flat', None)
        for kind, (pool_fn, _, _) in base._KDE_MAP_KINDS.items():
            values, values_are_mass = pool_fn(handler, results)
            kde_r = base.kde_density_map(handler, values, domain_r, values_are_mass) if domain_r.any() else None
            kde_n = null_kdes[key].get(kind)
            if kde_r is None or kde_n is None:
                tests[(key, kind)] = None
                continue
            n_r = len(results)
            test_bins = (domain_r & null_domains[key]
                         & np.isfinite(kde_r['density']) & np.isfinite(kde_n['density']))
            res = base.duong_local_test(kde_r, n_r, kde_n, null_n[key], test_bins)
            res.update(n_real=n_r, n_null=null_n[key])
            tests[(key, kind)] = res

            pre = f'{key}_{kind}'
            for k in ('diff', 'z', 'p', 'sig', 'test_bins'):
                saved[f'{pre}_{k}'] = res[k]
            saved[f'{pre}_real_density'] = kde_r['density']
            saved[f'{pre}_null_density'] = kde_n['density']

            sig, diff, t = res['sig'], res['diff'], res['test_bins']
            summary_rows.append(dict(
                arena=key, map_kind=kind, n_real=n_r, n_null=null_n[key],
                n_bins_tested=int(t.sum()), n_sig=int(sig.sum()),
                n_sig_real_gt_null=int((sig & (diff > 0)).sum()),
                n_sig_real_lt_null=int((sig & (diff < 0)).sum()),
                pct_bins_sig=100.0 * sig.sum() / t.sum() if t.any() else np.nan,
                hochberg_p_threshold=res['p_threshold'],
                min_p=float(np.nanmin(res['p'])) if t.any() else np.nan,
                max_abs_z=float(np.nanmax(np.abs(res['z']))) if t.any() else np.nan))
            print(f'  [Duong {TAG}] {key} {kind}: {int(sig.sum())}/{int(t.sum())} bins significant '
                  f'(real>null {summary_rows[-1]["n_sig_real_gt_null"]}, '
                  f'real<null {summary_rows[-1]["n_sig_real_lt_null"]})')

            for b in np.flatnonzero(sig):
                bx, by = divmod(int(b), handler.ny)
                row = dict(arena=key, map_kind=kind, bin=int(b), bx=bx, by=by,
                           x_cm=float(xy[0, b]), y_cm=float(xy[1, b]),
                           real_density=float(kde_r['density'][b]),
                           null_density=float(kde_n['density'][b]),
                           diff=float(diff[b]), z=float(res['z'][b]), p=float(res['p'][b]),
                           direction='real > null' if diff[b] > 0 else 'real < null')
                if edge is not None:
                    row['zone'] = 'edge' if edge[b] else 'centre'
                bin_rows.append(row)

    if not summary_rows:
        print('[SKIP Duong test] no real / null KDE pairs')
        return
    base.plot_duong_comparison(handlers, tests, TAG,
                               os.path.join(out_dir, f'Duong_Real_vs_{TAG}.png'))
    npz_path = os.path.join(out_dir, f'Duong_Real_vs_{TAG}.npz')
    np.savez_compressed(npz_path, alpha=base.DUONG_ALPHA, **saved)
    print(f'[SAVED] {npz_path}')
    xlsx = os.path.join(out_dir, f'Duong_Real_vs_{TAG}.xlsx')
    with pd.ExcelWriter(xlsx) as xw:
        pd.DataFrame(summary_rows).to_excel(xw, sheet_name='Summary', index=False)
        pd.DataFrame(bin_rows, columns=['arena', 'map_kind', 'bin', 'bx', 'by', 'x_cm', 'y_cm',
                                        'real_density', 'null_density', 'diff', 'z', 'p',
                                        'direction', 'zone']
                     ).to_excel(xw, sheet_name='SignificantBins', index=False)
    print(f'[SAVED] {xlsx}')


# ============================================================================
# Run
# ============================================================================

def run(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    handlers, arena_occ, arena_tilings, arena_shape = {}, {}, {}, {}
    pooled, kdes, n_cells, domains = {}, {}, {}, {}
    saved, arena_rows, session_rows = {}, [], []

    for key in base._ARENA_ORDER:
        handler = base.make_handler(base.ARENA_CONFIGS[key])
        occ, valid, rows = concatenated_occupancy(key, handler)       # Step 1
        session_rows.extend(rows)
        if not valid.any():
            continue
        sx, sy, tilings = make_tilings(handler)                        # Step 2
        n_tiles = [len(t) for t in tilings]
        print(f'  [{key}] field {sx} x {sy} bins, {len(tilings)} tiling phase(s), '
              f'{min(n_tiles)}-{max(n_tiles)} tiles each')
        pool, n_dropped = simulate_arena(handler, occ, valid, tilings, key)   # Step 3
        if pool.n_cells == 0:
            continue

        handlers[key], arena_occ[key] = handler, (occ, valid)
        arena_tilings[key], arena_shape[key] = tilings, (sx, sy)
        maps = pool.maps()
        pooled[key], kdes[key], n_cells[key], domains[key] = maps, {}, pool.n_cells, valid
        saved[f'{key}_domain'] = valid
        saved[f'{key}_n_cells'] = pool.n_cells
        saved[f'{key}_occupancy_s'] = occ

        row = dict(arena=key, n_tracking_sessions=len(rows),
                   concatenated_min=round(float(occ.sum()) / 60.0, 1),
                   n_bins_sampled=int(valid.sum()), field_bins_x=sx, field_bins_y=sy,
                   n_tiling_phases=len(tilings), n_tiles_total=int(sum(n_tiles)),
                   noise=NOISE, peak_rate_hz=PEAK_RATE_HZ,
                   n_iter=N_ITER if NOISE == 'poisson' else 1,
                   n_sim_cells=pool.n_cells, n_dropped=n_dropped)
        for kind, (values_are_mass, _, _) in _MAP_KINDS.items():       # Step 4
            values, _ = maps[kind]
            saved[f'{key}_{kind}_map'] = values
            kde_out = base.kde_density_map(handler, values, valid, values_are_mass)
            kdes[key][kind] = kde_out
            if kde_out is None:
                continue
            for k in ('density', 'density_raw', 'scaled', 'cov'):
                saved[f'{key}_{kind}_{k}'] = kde_out[k]
            saved[f'{key}_{kind}_factor'] = kde_out['factor']
            saved[f'{key}_{kind}_neff'] = kde_out['neff']
            sd = np.sqrt(np.diag(kde_out['cov']))
            row.update({f'{kind}_scott_factor': kde_out['factor'], f'{kind}_neff': kde_out['neff'],
                        f'{kind}_bw_sd_x_cm': sd[0], f'{kind}_bw_sd_y_cm': sd[1]})
        arena_rows.append(row)

    if not pooled:
        print(f'No tracking sessions found under {TRACKING_ROOT_DIRECTORY} -- nothing to simulate.')
        return

    plot_tiling(handlers, arena_occ, arena_tilings, arena_shape, out_dir)
    plot_null(handlers, pooled, kdes, n_cells, out_dir)
    npz_path = os.path.join(out_dir, f'Null_{TAG}.npz')
    np.savez_compressed(npz_path, mode='trajectory_tiling', field_size_bins=FIELD_SIZE_BINS,
                        noise=NOISE, peak_rate_hz=PEAK_RATE_HZ, **saved)
    print(f'[SAVED] {npz_path}')
    xlsx = os.path.join(out_dir, f'Null_{TAG}_Summary.xlsx')
    with pd.ExcelWriter(xlsx) as xw:
        pd.DataFrame(arena_rows).to_excel(xw, sheet_name='Arenas', index=False)
        pd.DataFrame(session_rows).to_excel(xw, sheet_name='Sessions', index=False)
    print(f'[SAVED] {xlsx}')

    real_results = {}                                                   # Step 5
    for key in pooled:
        _, _, real_results[key] = reloc.load_real_sessions(key)
    duong_real_vs_null(handlers, real_results, kdes, domains, n_cells, out_dir)


if __name__ == '__main__':
    print(f'Tracking files from : {TRACKING_ROOT_DIRECTORY}')
    print(f'Real place cells from: {base.ROOT_DIRECTORY}')
    run(OUTPUT_DIR)
    print('\nDone.')
