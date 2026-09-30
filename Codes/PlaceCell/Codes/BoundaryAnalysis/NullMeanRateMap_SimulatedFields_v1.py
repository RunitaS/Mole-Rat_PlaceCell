# -*- coding: utf-8 -*-
"""
Observed NULL mean rate maps from simulated, uniformly tiling Gaussian place fields.

Question: if place fields tiled the whole arena uniformly (one simulated cell centred on
every spatial bin), each field a clean 2D Gaussian, and the animal moved through them with
exactly the occupancy it really had in each recorded session -- what would the pooled
overall / field-only / peak-proportion maps, and their KDE fits, look like? Any structure
in these null maps (e.g. an edge-vs-centre gradient) is produced by sampling + the analysis
pipeline alone, not by where fields really are, so the real maps should be read against them.

Everything except the spikes is taken from MeanRateMap_QuadrantAnalysis_v21.py (imported as
`base`), so the null goes through the SAME procedure as the real data:
  * sessions        : same ROOT_DIRECTORY walk, tracking-file choice, arena detection, and
                      session-level coverage criteria (zone / bin coverage, if enabled);
  * occupancy       : base._session_positions (load, jump removal, smoothing, open-field
                      centring) -> handler.orient -> base._speed_mask -> handler.to_bins,
                      speed-filtered occupancy, min_occ_s + geom_valid masking -- exactly the
                      occupancy compute_cell_ratemap builds for every real unit;
  * per-cell maps   : handler.smooth (valid-masked Gaussian), base.field_index_map,
                      base.extract_place_field_mask (mean-rate + 20 %-of-peak + >= 9 bins),
                      raw-map peak bin;
  * pooling         : NullPool reproduces pool_fine_map / pool_field_only_map /
                      pool_peak_proportion_map exactly, but as running sums (tens of thousands
                      of simulated cells would not fit as per-cell arrays);
  * kernel fit      : base.kde_density_map (weighted gaussian_kde, Scott's rule, edge
                      correction per base.KDE_EDGE_CORRECTION), domain = union of valid bins,
                      identical to plot_kde_maps.

Simulated fields (field_template): a 25 x 25 bin square (FIELD_SIZES_BINS; bins are the
pipeline's ~2 x 2 cm bins in every arena, so ~50 x 50 cm) centred on the peak bin -- the only
field size used for the observed null / occupancy-control KDE maps. Rate is a 2D Gaussian,
1.0 at the peak bin, falling to FIELD_EDGE_RATE (0.2) at the field's corner bins, and 0
outside the square. On the circular track the field wraps around the arc-length axis; on the
narrow tracks it is truncated by the track width. One simulated cell per field centre, centres
on every geometrically valid bin (every FIELD_CENTRE_STRIDE_BINS-th bin), for every session --
so every session contributes one complete uniform tiling.

Spikes (SIM_MODES):
  'expected' : noiseless -- the raw rate in each sampled bin is exactly the field template.
               Occupancy then shapes the null only through WHICH bins are sampled (min_occ_s,
               arena coverage) and the valid-masked smoothing.
  'poisson'  : spikes in each bin ~ Poisson(POISSON_PEAK_RATE_HZ * template * occupancy_s),
               i.e. what the animal's actual passes would yield; rarely visited bins get noisy
               rate estimates, which mainly affects the raw peak bin (peak-proportion map).
A simulated cell whose field lies entirely in unsampled bins, or that fires no spikes, is
dropped (a real unit like that would never have been recorded as a place cell).

Outputs (OUTPUT_DIR, one set per mode x field size):
  Null_<mode>_<k>x<k>_PooledMaps.png  : overall / field-only / peak-proportion null maps
  Null_<mode>_<k>x<k>_KDE.png         : their KDE densities (same plot as plot_kde_maps)
  Null_<mode>_<k>x<k>.npz             : maps + KDE outputs; KDE keys follow plot_kde_maps'
                                        npz naming with the map kind inserted, e.g.
                                        open_field_overall_density, _scaled, _cov, _factor
  Null_Occupancy.png                  : mean normalized occupancy per arena (context)
  Null_Summary.xlsx                   : cells simulated / dropped and KDE bandwidth per run
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import MeanRateMap_QuadrantAnalysis_v21 as base  # noqa: E402

# ============================================================================
# CONFIGURATION
# ============================================================================

FIELD_SIZES_BINS         = (25,)     # side length (bins) of the square simulated field --
                                     # 25 x 25 is the sole observed null (occupancy control)
FIELD_EDGE_RATE          = 0.2       # rate at the field's corner bins (peak bin = 1.0)
FIELD_CENTRE_STRIDE_BINS = 1         # 1 = a field centred on every bin; 2 = every other bin, ...
SIM_MODES                = ('expected', 'poisson')
POISSON_PEAK_RATE_HZ     = 10.0      # peak rate used by the 'poisson' mode

# same folder as the real results (base run_full_pipeline writes to OUTPUT_DIR/AllArenas);
# every null file is prefixed 'Null_', so nothing overwrites a real result
OUTPUT_DIR = os.path.join(base.OUTPUT_DIR, 'AllArenas')

_MAP_KINDS = {  # kind -> (values_are_mass for kde_density_map, title, colorbar label)
    'overall':    (False, 'Overall mean field index', 'Field index (a.u.)'),
    'field_only': (False, 'Field-only mean field index', 'Field index, background = 0 (a.u.)'),
    'peak':       (True,  'Peak proportion', '% of cells with peak in bin'),
}


# ============================================================================
# Sessions + occupancy (same selection / occupancy as the real pipeline)
# ============================================================================

def find_sessions(arena_key: str) -> list:
    """Sessions of this arena exactly as collect_arena_results selects them: one tracking
    file + at least one .ntt, passing the zone-coverage screen if it is enabled."""
    handler = base.make_handler(base.ARENA_CONFIGS[arena_key])
    sessions = []
    for dirpath, _, filenames in os.walk(base.ROOT_DIRECTORY):
        if base._detect_arena_key(dirpath) != arena_key:
            continue
        tracking_files_all = [f for f in filenames if f.lower().endswith(('.csv', '.xlsx'))]
        if base.COORD_UNITS == 'cm':
            tracking_files = [f for f in tracking_files_all if f.lower().endswith('_cm.csv')]
        else:
            tracking_files = [f for f in tracking_files_all if not f.lower().endswith('_cm.csv')]
        ntt_files = [f for f in filenames if f.lower().endswith('.ntt')]
        if len(tracking_files) != 1 or not ntt_files:
            continue
        csv_path = os.path.join(dirpath, tracking_files[0])
        session_name = os.path.relpath(dirpath, base.ROOT_DIRECTORY)

        if base.USE_ZONE_COVERAGE_CRITERION:
            zone_cov = base.session_zone_coverage(csv_path, handler)
            if zone_cov is None or min(zone_cov) < base.MIN_ZONE_COVERAGE_PCT:
                print(f'  [SKIP zone coverage] [{arena_key}] {session_name}')
                continue
        sessions.append((session_name, csv_path))
    return sessions


def session_occupancy(csv_path: str, handler) -> tuple | None:
    """(occ_map, valid) for one session -- the occupancy half of compute_cell_ratemap,
    step for step, so every simulated cell sees the occupancy a real unit of that session
    saw."""
    x_cm, y_cm, t = base._session_positions(csv_path, handler)
    if len(t) < 2:
        return None
    x_cm, y_cm = handler.orient(x_cm, y_cm)
    moving = base._speed_mask(x_cm, y_cm, t)
    bin_idx, sample_valid = handler.to_bins(x_cm, y_cm)
    t = t[sample_valid]
    bin_idx = bin_idx[sample_valid]
    moving = moving[sample_valid]
    if len(t) < 2:
        return None

    n = len(t)
    dt_frames = np.empty(n, dtype=np.float64)
    dt_frames[0] = 1.0 / base.fps
    dt_frames[1:] = np.minimum(np.diff(t) * 1e-6, 2.0 / base.fps)

    if base.SPEED_FILTER_OCCUPANCY:
        t, bin_idx, dt_frames = t[moving], bin_idx[moving], dt_frames[moving]
        if len(t) < 2:
            return None

    occ_map = np.zeros(handler.n_bins, dtype=np.float64)
    np.add.at(occ_map, bin_idx, dt_frames)
    geom_valid = getattr(handler, 'geom_valid', np.ones(handler.n_bins, dtype=bool))
    valid = (occ_map >= base.min_occ_s) & geom_valid
    return occ_map, valid


# ============================================================================
# Simulated fields
# ============================================================================

def field_centres(handler) -> np.ndarray:
    """Flat indices of the field centres: every geometrically valid bin (on the stride)."""
    bx, by = np.divmod(np.arange(handler.n_bins), handler.ny)
    geom_valid = getattr(handler, 'geom_valid', np.ones(handler.n_bins, dtype=bool))
    s = FIELD_CENTRE_STRIDE_BINS
    return np.flatnonzero(geom_valid & (bx % s == 0) & (by % s == 0))


def field_template(handler, centre_flat: int, size_bins: int) -> np.ndarray:
    """Flat-bin rate template of one simulated field: 2D Gaussian over a size_bins x
    size_bins square centred on centre_flat, 1.0 at the centre and FIELD_EDGE_RATE at the
    corner bins, 0 outside the square. Wraps along arc-length on the circular track."""
    bx, by = np.divmod(np.arange(handler.n_bins), handler.ny)
    cx, cy = divmod(int(centre_flat), handler.ny)
    dx = bx - cx
    if isinstance(handler, base.CircularTrackHandler):
        dx = (dx + handler.nx // 2) % handler.nx - handler.nx // 2
    dy = by - cy
    h = (size_bins - 1) / 2.0
    inside = (np.abs(dx) <= h) & (np.abs(dy) <= h)
    if h == 0:
        return inside.astype(np.float64)
    # corner bin (h, h): exp(-2h^2 / (2 sigma^2)) = FIELD_EDGE_RATE
    sigma2 = h * h / np.log(1.0 / FIELD_EDGE_RATE)
    return np.where(inside, np.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma2)), 0.0)


def simulate_cell(handler, occ_map: np.ndarray, valid: np.ndarray, template: np.ndarray,
                  mode: str, rng) -> tuple | None:
    """One simulated cell through the real per-cell pipeline. Returns
    (fi_map, field_mask, peak_bin_raw), or None if it has no firing in any sampled bin."""
    fr_raw = np.zeros(handler.n_bins, dtype=np.float64)
    if mode == 'expected':
        fr_raw[valid] = template[valid]
    elif mode == 'poisson':
        spikes = rng.poisson(POISSON_PEAK_RATE_HZ * template * occ_map)
        fr_raw[valid] = spikes[valid] / occ_map[valid]
    else:
        raise ValueError(f'Unknown SIM_MODE: {mode}')
    if not (fr_raw[valid] > 0).any():
        return None

    fr_smooth = handler.smooth(fr_raw, valid)
    fi_map = base.field_index_map(fr_smooth, valid)
    pi = occ_map[valid] / occ_map[valid].sum()
    mean_fr = round(float(np.sum(pi * fr_smooth[valid])), 4)   # as compute_cell_ratemap
    field_mask = base.extract_place_field_mask(
        dict(valid=valid, fr_smooth=fr_smooth, mean_fr=mean_fr), handler)
    peak_bin_raw = int(np.argmax(np.where(valid, fr_raw, -np.inf)))
    return fi_map, field_mask, peak_bin_raw


# ============================================================================
# Pooling (running-sum equivalents of pool_fine_map / pool_field_only_map /
# pool_peak_proportion_map)
# ============================================================================

class NullPool:
    def __init__(self, n_bins: int):
        self.fi_sum      = np.zeros(n_bins)
        self.n_sampled   = np.zeros(n_bins)
        self.fo_sum      = np.zeros(n_bins)
        self.n_field     = np.zeros(n_bins, dtype=np.int64)
        self.peak_counts = np.zeros(n_bins)
        self.visited     = np.zeros(n_bins, dtype=bool)
        self.n_cells     = 0

    def add(self, valid, fi_map, field_mask, peak_bin_raw):
        self.fi_sum[valid] += fi_map[valid]
        self.n_sampled[valid] += 1.0
        m = field_mask & valid
        self.fo_sum[m] += fi_map[m]
        self.n_field[m] += 1
        self.peak_counts[peak_bin_raw] += 1.0
        self.visited |= valid
        self.n_cells += 1

    def maps(self) -> dict:
        """kind -> (values, shown-bins mask), matching the base pooling functions."""
        n = len(self.fi_sum)
        overall = np.full(n, np.nan)
        s = self.n_sampled > 0
        overall[s] = self.fi_sum[s] / self.n_sampled[s]

        field_only = np.full(n, np.nan)
        f = (self.n_field > 0) & s
        field_only[f] = self.fo_sum[f] / self.n_sampled[f]

        peak = np.full(n, np.nan)
        if self.n_cells > 0:
            peak[self.visited] = self.peak_counts[self.visited] / self.n_cells * 100.0
        return {'overall': (overall, s), 'field_only': (field_only, f),
                'peak': (peak, self.visited)}


# ============================================================================
# Run
# ============================================================================

def load_arena_occupancy(arena_key: str) -> tuple:
    handler = base.make_handler(base.ARENA_CONFIGS[arena_key])
    occs = []
    for session_name, csv_path in find_sessions(arena_key):
        try:
            out = session_occupancy(csv_path, handler)
        except Exception as e:
            print(f'  ERROR [{arena_key}] {session_name}: {e}')
            continue
        if out is None:
            print(f'  [SKIP no tracking] [{arena_key}] {session_name}')
            continue
        occ_map, valid = out
        if base.USE_BIN_COVERAGE_CRITERION and valid.sum() < handler.coverage_threshold_bins:
            print(f'  [SKIP low coverage] [{arena_key}] {session_name}')
            continue
        occs.append((session_name, occ_map, valid))
    print(f'[{arena_key}] {len(occs)} sessions with usable occupancy')
    return handler, occs


def simulate_arena(handler, occs: list, mode: str, size_bins: int, arena_key: str) -> tuple:
    pool = NullPool(handler.n_bins)
    centres = field_centres(handler)
    templates = [field_template(handler, c, size_bins) for c in centres]
    n_dropped = 0
    for i, (session_name, occ_map, valid) in enumerate(occs, start=1):
        rng = np.random.default_rng(base._stable_seed('null', mode, size_bins, session_name))
        for template in templates:
            cell = simulate_cell(handler, occ_map, valid, template, mode, rng)
            if cell is None:
                n_dropped += 1
                continue
            pool.add(valid, *cell)
        print(f'  [{arena_key} {mode} {size_bins}x{size_bins}] session {i}/{len(occs)}', end='\r')
    print(f'\n  [{arena_key} {mode} {size_bins}x{size_bins}] {pool.n_cells} simulated cells '
          f'pooled, {n_dropped} dropped (no firing in sampled bins)')
    return pool, n_dropped


def _arena_axes(fig, n_rows, keys, row, polar_for_ring):
    axes = {}
    for j, key in enumerate(keys):
        proj = 'polar' if (polar_for_ring and key == 'circular_track') else None
        axes[key] = fig.add_subplot(n_rows, len(keys), row * len(keys) + j + 1, projection=proj)
    return axes


def plot_run(handlers, pooled, kdes, n_cells, mode, size_bins, out_dir):
    keys = [k for k in base._ARENA_ORDER if k in pooled]
    tag = f'{mode}_{size_bins}x{size_bins}'
    sim_desc = (f'{size_bins}x{size_bins}-bin Gaussian fields (1.0 -> {FIELD_EDGE_RATE} at corners), '
                f'{"noiseless rates" if mode == "expected" else f"Poisson spikes, peak {POISSON_PEAK_RATE_HZ:g} Hz"}')

    # pooled null maps
    fig = plt.figure(figsize=(5 * len(keys), 13))
    for row, (kind, (_, title, label)) in enumerate(_MAP_KINDS.items()):
        axes = _arena_axes(fig, 3, keys, row, polar_for_ring=True)
        for key in keys:
            values, shown = pooled[key][kind]
            cmap, norm = base.make_cmap_norm(values[shown])
            pcm = handlers[key].plot_fine(axes[key], values, shown, cmap, norm)
            axes[key].set_title(f'{base._ARENA_TITLES[key]} -- {title}\n(n={n_cells[key]} sim. cells)',
                                fontsize=9)
            fig.colorbar(pcm, ax=axes[key], shrink=0.7, label=label)
    fig.suptitle(f'NULL pooled maps -- {sim_desc}')
    fig.tight_layout()
    p = os.path.join(out_dir, f'Null_{tag}_PooledMaps.png')
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {p}')

    # KDE fits (same plot as plot_kde_maps: density per cm^2 on the flat bin grid)
    fig = plt.figure(figsize=(6 * len(keys), 13))
    for row, (kind, (_, title, _)) in enumerate(_MAP_KINDS.items()):
        axes = _arena_axes(fig, 3, keys, row, polar_for_ring=False)
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
    p = os.path.join(out_dir, f'Null_{tag}_KDE.png')
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {p}')


def plot_occupancy(handlers, arena_occs, out_dir):
    keys = [k for k in base._ARENA_ORDER if arena_occs.get(k)]
    if not keys:
        return
    fig = plt.figure(figsize=(5 * len(keys), 5))
    for j, key in enumerate(keys):
        handler = handlers[key]
        occ = np.mean([o / o[v].sum() * np.where(v, 1.0, 0.0) for _, o, v in arena_occs[key]], axis=0)
        shown = np.any([v for _, _, v in arena_occs[key]], axis=0)
        proj = 'polar' if key == 'circular_track' else None
        ax = fig.add_subplot(1, len(keys), j + 1, projection=proj)
        cmap, norm = base.make_cmap_norm(occ[shown])
        pcm = handler.plot_fine(ax, occ, shown, cmap, norm)
        ax.set_title(f'{base._ARENA_TITLES[key]}\n(n={len(arena_occs[key])} sessions)')
        fig.colorbar(pcm, ax=ax, shrink=0.7, label='Fraction of session time')
    fig.suptitle('Mean normalized occupancy (valid bins) used for the null simulation')
    fig.tight_layout()
    p = os.path.join(out_dir, 'Null_Occupancy.png')
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {p}')


def run_null(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    handlers, arena_occs = {}, {}
    for key in base._ARENA_ORDER:
        handlers[key], arena_occs[key] = load_arena_occupancy(key)
    plot_occupancy(handlers, arena_occs, out_dir)

    summary = []
    for mode in SIM_MODES:
        for size_bins in FIELD_SIZES_BINS:
            pooled, kdes, n_cells, saved = {}, {}, {}, {}
            for key in base._ARENA_ORDER:
                if not arena_occs[key]:
                    continue
                handler = handlers[key]
                pool, n_dropped = simulate_arena(handler, arena_occs[key], mode, size_bins, key)
                maps = pool.maps()
                domain = pool.visited
                pooled[key], kdes[key], n_cells[key] = maps, {}, pool.n_cells
                saved[f'{key}_domain'] = domain
                saved[f'{key}_n_cells'] = pool.n_cells   # n for the Duong test in base

                row = dict(arena=key, mode=mode, field_size_bins=size_bins,
                           n_sessions=len(arena_occs[key]), n_sim_cells=pool.n_cells,
                           n_dropped=n_dropped)
                for kind, (values_are_mass, _, _) in _MAP_KINDS.items():
                    values, _ = maps[kind]
                    saved[f'{key}_{kind}_map'] = values
                    kde_out = (base.kde_density_map(handler, values, domain, values_are_mass)
                               if domain.any() else None)
                    kdes[key][kind] = kde_out
                    if kde_out is None:
                        continue
                    for k in ('density', 'density_raw', 'scaled', 'cov'):
                        saved[f'{key}_{kind}_{k}'] = kde_out[k]
                    saved[f'{key}_{kind}_factor'] = kde_out['factor']
                    saved[f'{key}_{kind}_neff'] = kde_out['neff']
                    sd = np.sqrt(np.diag(kde_out['cov']))
                    row.update({f'{kind}_scott_factor': kde_out['factor'],
                                f'{kind}_neff': kde_out['neff'],
                                f'{kind}_bw_sd_x_cm': sd[0], f'{kind}_bw_sd_y_cm': sd[1]})
                summary.append(row)

            if not pooled:
                print('No sessions found under ROOT_DIRECTORY -- nothing to simulate.')
                return
            plot_run(handlers, pooled, kdes, n_cells, mode, size_bins, out_dir)
            npz_path = os.path.join(out_dir, f'Null_{mode}_{size_bins}x{size_bins}.npz')
            np.savez_compressed(npz_path, mode=mode, field_size_bins=size_bins,
                                field_edge_rate=FIELD_EDGE_RATE,
                                poisson_peak_rate_hz=POISSON_PEAK_RATE_HZ, **saved)
            print(f'[SAVED] {npz_path}')

    xlsx = os.path.join(out_dir, 'Null_Summary.xlsx')
    pd.DataFrame(summary).to_excel(xlsx, index=False)
    print(f'[SAVED] {xlsx}')


if __name__ == '__main__':
    print(f'ROOT_DIRECTORY: {base.ROOT_DIRECTORY}')
    run_null(OUTPUT_DIR)
    print('\nDone.')
