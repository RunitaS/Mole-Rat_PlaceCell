# -*- coding: utf-8 -*-
"""
Observed NULL mean rate maps by RELOCATING each real place field, and the Duong (2013) local
test of the real KDE against the null KDE.

Null question: if every recorded place cell kept its own field (shape, size, rates, noise) but
the field sat at a uniformly random location within the bins its session sampled, what would the
pooled overall / field-only / peak-proportion maps and their KDEs look like? Bins where the real
KDE differs from this null KDE are where field locations deviate from uniform beyond what the
animal's sampling + the analysis pipeline produce by themselves.

Why relocation (vs v1's 25 x 25 Gaussian tiling): v1 centred fields on every geometrically valid
bin, including bins a session never sampled. field_index_map rescales each cell to its highest
SAMPLED bin, so those fields' peaks were displaced onto the rims of unsampled holes, and the
~50 cm fields in a 60 cm arena added a pure-geometry centre dome. Here centres are drawn only
from sampled bins, each session contributes exactly its number of real cells, and fields are the
real ones.

Relocation, per real cell and draw (relocated_raw_map):
  * field region : the cell's extracted field mask (base.extract_place_field_mask; if empty,
                   its bins >= FIELD_PEAK_FRAC of the smoothed peak), dilated by
                   RELOCATE_MARGIN_BINS so the field's flanks move with it;
  * source rates : the cell's RAW (unsmoothed) rate map over that region -- so after relocation
                   the map is smoothed exactly ONCE, by the same valid-masked handler.smooth as
                   a real cell, and the smoothing's wall / hole renormalisation is the one at the
                   NEW location. Region bins the source session did not sample have no raw rate;
                   they get the valid-masked smoothed extrapolation of the raw map (fill rate);
  * background   : the cell's occupancy-weighted mean raw rate outside the region
                   (= out-of-region spikes / out-of-region time);
  * placement    : the smoothed-peak bin of the field lands on a centre drawn uniformly from the
                   bins the session sampled (open field: random 90-degree rotation / reflection
                   when RELOCATE_RANDOM_ORIENT); region bins falling off the arena are cut (wall
                   truncation); the circular track wraps along arc length;
  * noise        : every bin gets exactly one sampling-noise realisation -- sampled in-field
                   bins keep the cell's real raw rates (its real noise); background bins and
                   filled bins get Poisson(rate x new-bin occupancy) / occupancy.
  Caveat: the in-field noise is the source location's (its occupancy), not the new one's.

Each simulated cell then goes through the real per-cell pipeline (handler.smooth,
base.field_index_map, base.extract_place_field_mask, raw peak bin) and the real pooling
(NullPool == pool_fine_map / pool_field_only_map / pool_peak_proportion_map), N_ITER times.
A simulated cell with no spikes in any sampled bin is dropped.

Duong test (base.duong_local_test, Hochberg-adjusted, alpha = base.DUONG_ALPHA): real KDE (real
cells, n = real cells) vs null KDE (n = pooled simulated cells = N_ITER x real cells, so the null
KDE's own variance is negligible and the test is effectively real vs the null expectation).
Bins tested = sampled in both, finite in both KDEs.

Real cells use the real pipeline's session selection and base.compute_cell_ratemap, without the
SIR bootstrap (not needed here), so their pooled maps / KDEs are the real pipeline's.

Outputs (OUTPUT_DIR):
  Null_Relocation_<N>x_PooledMaps.png / _KDE.png : null mean maps and their KDEs
  Null_Relocation_<N>x.npz   : null maps, KDE outputs (v1 key names) and per-bin 2.5 / 97.5 %
                               bands across draws (<arena>_<kind>_lo / _hi); its name matches
                               the 'Null_*x*.npz' glob of compare_kde_to_null in v21 / v22
  Duong_Real_vs_Relocation_<N>x.png  : real - null KDE, significant bins solid
  Duong_Real_vs_Relocation_<N>x.npz  : diff / z / p / sig / test_bins per arena and map kind
  Duong_Real_vs_Relocation_<N>x.xlsx : 'Summary' (per arena x map kind) and 'SignificantBins'
                                       (one row per significant bin: position, densities, z, p)
  Null_Relocation_Summary.xlsx       : cells / drops / KDE bandwidths per arena
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import MeanRateMap_QuadrantAnalysis_v21 as base   # noqa: E402

# ============================================================================
# CONFIGURATION
# ============================================================================

N_ITER                 = 200         # random re-draws of every session's field centres
BAND_PCT               = (2.5, 97.5) # per-bin null band across draws
RELOCATE_MARGIN_BINS   = 3           # field-mask dilation (bins) carried along with the field
RELOCATE_RANDOM_ORIENT = True        # open field: random rotation / reflection of each moved field
FILL_MIN_WEIGHT        = 1e-3        # extrapolation weight below which a fill bin gets background

OUTPUT_DIR = os.path.join(base.OUTPUT_DIR, 'AllArenas')
TAG = f'Relocation_{N_ITER}x'

_MAP_KINDS = {  # kind -> (values_are_mass for kde_density_map, title, colorbar label)
    'overall':    (False, 'Overall mean field index', 'Field index (a.u.)'),
    'field_only': (False, 'Field-only mean field index', 'Field index, background = 0 (a.u.)'),
    'peak':       (True,  'Peak proportion', '% of cells with peak in bin'),
}


def _geom_valid(handler) -> np.ndarray:
    return getattr(handler, 'geom_valid', np.ones(handler.n_bins, dtype=bool))


def _wraps(handler) -> bool:
    return isinstance(handler, base.CircularTrackHandler)


# ============================================================================
# Real place cells per session (same selection as the real pipeline, no bootstrap)
# ============================================================================

def find_sessions(arena_key: str) -> list:
    """(session_name, dirpath, csv_path) exactly as collect_arena_results selects them."""
    handler = base.make_handler(base.ARENA_CONFIGS[arena_key])
    sessions = []
    for dirpath, _, filenames in os.walk(base.ROOT_DIRECTORY):
        if base._detect_arena_key(dirpath) != arena_key:
            continue
        tracking_all = [f for f in filenames if f.lower().endswith(('.csv', '.xlsx'))]
        if base.COORD_UNITS == 'cm':
            tracking = [f for f in tracking_all if f.lower().endswith('_cm.csv')]
        else:
            tracking = [f for f in tracking_all if not f.lower().endswith('_cm.csv')]
        if len(tracking) != 1 or not any(f.lower().endswith('.ntt') for f in filenames):
            continue
        csv_path = os.path.join(dirpath, tracking[0])
        session_name = os.path.relpath(dirpath, base.ROOT_DIRECTORY)
        if base.USE_ZONE_COVERAGE_CRITERION:
            zone_cov = base.session_zone_coverage(csv_path, handler)
            if zone_cov is None or min(zone_cov) < base.MIN_ZONE_COVERAGE_PCT:
                print(f'  [SKIP zone coverage] [{arena_key}] {session_name}')
                continue
        sessions.append((session_name, dirpath, csv_path))
    return sessions


def load_real_sessions(arena_key: str) -> tuple:
    """handler, sessions = [dict(name, occ_map, valid, cells)], real_results (for base pooling).
    Each cell: fr_raw, fr_smooth, fi_map, valid, field_mask, peak_bin_raw, peak_fr."""
    handler = base.make_handler(base.ARENA_CONFIGS[arena_key])
    sessions, real_results = [], []
    for session_name, dirpath, csv_path in find_sessions(arena_key):
        try:
            x_cm, y_cm, t = base._session_positions(csv_path, handler)
        except Exception as e:
            print(f'  ERROR [{arena_key}] {session_name}: {e}')
            continue
        if len(t) < 2:
            continue
        cells = []
        for ntt_file in sorted(f for f in os.listdir(dirpath) if f.lower().endswith('.ntt')):
            try:
                spk = np.memmap(os.path.join(dirpath, ntt_file), dtype=base.ntt_dtype, mode='r',
                                offset=16 * 1024)
                spk = spk[spk['cell_number'] != 0]
                spike_ts = np.sort(spk['timestamp'].astype(np.float64))
                if len(spike_ts) == 0:
                    continue
                cell = base.compute_cell_ratemap(x_cm, y_cm, t, spike_ts, handler)
            except Exception as e:
                print(f'  ERROR [{arena_key}] {session_name}/{ntt_file}: {e}')
                continue
            if cell is None or not cell['valid'].any():
                continue
            valid = cell['valid']
            cells.append(dict(
                fr_raw=cell['fr_raw'], fr_smooth=cell['fr_smooth'], fi_map=cell['fi_map'],
                valid=valid, occ_map=cell['occ_map'],
                field_mask=base.extract_place_field_mask(cell, handler),
                peak_bin_raw=int(np.argmax(np.where(valid, cell['fr_raw'], -np.inf))),
                peak_fr=float(cell['fr_smooth'][valid].max())))
        if not cells:
            continue
        occ_map, valid = cells[0]['occ_map'], cells[0]['valid']
        if base.USE_BIN_COVERAGE_CRITERION and valid.sum() < handler.coverage_threshold_bins:
            print(f'  [SKIP low coverage] [{arena_key}] {session_name}')
            continue
        sessions.append(dict(name=session_name, occ_map=occ_map, valid=valid, cells=cells))
        real_results.extend(cells)
    print(f'[{arena_key}] {len(sessions)} sessions, {len(real_results)} real place cells')
    return handler, sessions, real_results


# ============================================================================
# Relocation of a real field (raw rates, smoothed once afterwards)
# ============================================================================

def _dilate(handler, mask: np.ndarray, n_iter: int) -> np.ndarray:
    """8-neighbour binary dilation on the (nx, ny) grid; wraps along x on the circular track."""
    m = mask.reshape(handler.nx, handler.ny).copy()
    for _ in range(n_iter):
        out = m.copy()
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                if sx == 0 and sy == 0:
                    continue
                r = np.roll(np.roll(m, sx, axis=0), sy, axis=1)
                if not _wraps(handler):
                    if sx == 1:
                        r[0, :] = False
                    elif sx == -1:
                        r[-1, :] = False
                if sy == 1:
                    r[:, 0] = False
                elif sy == -1:
                    r[:, -1] = False
                out |= r
        m = out
    return m.ravel()


def _extrapolated_smooth(handler, fr_raw: np.ndarray, valid: np.ndarray) -> tuple:
    """Valid-masked Gaussian smoothing of the raw map (same sigma as handler.smooth) WITHOUT
    zeroing the unsampled bins: (rate, weight) everywhere, used to fill field bins the source
    session did not sample."""
    mode = ('wrap' if _wraps(handler) else 'constant', 'constant')
    shape = (handler.nx, handler.ny)
    num = gaussian_filter(np.where(valid, fr_raw, 0.0).reshape(shape),
                          sigma=base.RATEMAP_SMOOTH_SIGMA_BINS, mode=mode, cval=0.0)
    w = gaussian_filter(valid.astype(np.float64).reshape(shape),
                        sigma=base.RATEMAP_SMOOTH_SIGMA_BINS, mode=mode, cval=0.0)
    rate = np.where(w > 0, num / np.where(w > 0, w, 1.0), 0.0)
    return rate.ravel(), w.ravel()


def _offsets(handler, centre_flat: int, idx: np.ndarray) -> tuple:
    bx, by = np.divmod(idx, handler.ny)
    cx, cy = divmod(int(centre_flat), handler.ny)
    dx = bx - cx
    if _wraps(handler):
        dx = (dx + handler.nx // 2) % handler.nx - handler.nx // 2
    return dx, by - cy


def relocation_source(handler, cell: dict) -> dict:
    """Static (draw-independent) part of one real cell's relocation: region offsets relative
    to the field's smoothed-peak bin, their source rates, and which of them are real samples."""
    fr_raw, fr_smooth, valid, occ = cell['fr_raw'], cell['fr_smooth'], cell['valid'], cell['occ_map']
    fmask = cell['field_mask']
    if not fmask.any():
        fmask = valid & (fr_smooth >= base.FIELD_PEAK_FRAC * cell['peak_fr'])
    region = _dilate(handler, fmask, RELOCATE_MARGIN_BINS) & _geom_valid(handler)

    out = valid & ~region
    bg = float(np.sum(fr_raw[out] * occ[out]) / occ[out].sum()) if out.any() else 0.0

    fill, w = _extrapolated_smooth(handler, fr_raw, valid)
    idx = np.flatnonzero(region)
    sampled = valid[idx]
    rate = np.where(sampled, fr_raw[idx], np.where(w[idx] >= FILL_MIN_WEIGHT, fill[idx], bg))

    anchor = int(np.argmax(np.where(fmask, fr_smooth, -np.inf)))
    dx, dy = _offsets(handler, anchor, idx)
    return dict(dx=dx, dy=dy, rate=rate, sampled=sampled, bg=bg)


def relocated_raw_map(handler, src: dict, centre_flat: int, occ_map: np.ndarray,
                      valid: np.ndarray, rng) -> np.ndarray:
    """One relocated raw rate map (Hz) on the target session's sampled bins, 0 elsewhere."""
    dx, dy = src['dx'], src['dy']
    if RELOCATE_RANDOM_ORIENT and isinstance(handler, base.OpenFieldHandler):
        k = rng.integers(8)
        if k & 1:
            dx, dy = dy, dx
        if k & 2:
            dx = -dx
        if k & 4:
            dy = -dy
    cx, cy = divmod(int(centre_flat), handler.ny)
    nbx, nby = cx + dx, cy + dy
    if _wraps(handler):
        nbx = nbx % handler.nx
    keep = (nbx >= 0) & (nbx < handler.nx) & (nby >= 0) & (nby < handler.ny)
    tgt = nbx[keep] * handler.ny + nby[keep]
    keep_geom = _geom_valid(handler)[tgt]
    tgt = tgt[keep_geom]
    src_rate = src['rate'][keep][keep_geom]
    src_sampled = src['sampled'][keep][keep_geom]

    expected = np.full(handler.n_bins, src['bg'])   # rates still to be Poisson-sampled
    real_raw = np.zeros(handler.n_bins, dtype=bool)
    fr_raw = np.zeros(handler.n_bins)
    expected[tgt] = src_rate
    real_raw[tgt] = src_sampled
    fr_raw[tgt[src_sampled]] = src_rate[src_sampled]   # real raw rates keep their own noise

    pois = valid & ~real_raw
    fr_raw[pois] = rng.poisson(expected[pois] * occ_map[pois]) / occ_map[pois]
    fr_raw[~valid] = 0.0
    return fr_raw


# ============================================================================
# One simulated cell through the real per-cell pipeline, and pooling
# ============================================================================

def run_cell(handler, occ_map, valid, fr_raw):
    if not (fr_raw[valid] > 0).any():
        return None
    fr_smooth = handler.smooth(fr_raw, valid)          # the ONE smoothing step
    fi_map = base.field_index_map(fr_smooth, valid)
    pi = occ_map[valid] / occ_map[valid].sum()
    mean_fr = round(float(np.sum(pi * fr_smooth[valid])), 4)   # as compute_cell_ratemap
    field_mask = base.extract_place_field_mask(
        dict(valid=valid, fr_smooth=fr_smooth, mean_fr=mean_fr), handler)
    peak_bin_raw = int(np.argmax(np.where(valid, fr_raw, -np.inf)))
    return fi_map, field_mask, peak_bin_raw


class NullPool:
    """Running-sum equivalents of pool_fine_map / pool_field_only_map / pool_peak_proportion_map."""

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
        s = self.n_sampled > 0
        overall = np.full(n, np.nan)
        overall[s] = self.fi_sum[s] / self.n_sampled[s]
        f = (self.n_field > 0) & s
        field_only = np.full(n, np.nan)
        field_only[f] = self.fo_sum[f] / self.n_sampled[f]
        peak = np.full(n, np.nan)
        if self.n_cells > 0:
            peak[self.visited] = self.peak_counts[self.visited] / self.n_cells * 100.0
        return {'overall': (overall, s), 'field_only': (field_only, f), 'peak': (peak, self.visited)}

    def band_maps(self) -> dict:
        """As maps(), but field-only is 0 (not NaN) in sampled bins with no field, so
        percentiles across draws are well defined."""
        maps = self.maps()
        s = self.n_sampled > 0
        fo = np.full(len(s), np.nan)
        fo[s] = self.fo_sum[s] / self.n_sampled[s]
        return {'overall': maps['overall'][0], 'field_only': fo, 'peak': maps['peak'][0]}


def simulate_arena(handler, sessions, arena_key):
    pool = NullPool(handler.n_bins)
    iter_maps = {kind: [] for kind in _MAP_KINDS}
    n_dropped = 0
    prep = [(s, np.flatnonzero(s['valid']), [relocation_source(handler, c) for c in s['cells']])
            for s in sessions]

    for it in range(N_ITER):
        it_pool = NullPool(handler.n_bins)
        for s, centres, srcs in prep:
            rng = np.random.default_rng(base._stable_seed('null_relocation', arena_key, s['name'], it))
            for src, centre in zip(srcs, rng.choice(centres, size=len(srcs), replace=True)):
                fr_raw = relocated_raw_map(handler, src, centre, s['occ_map'], s['valid'], rng)
                cell = run_cell(handler, s['occ_map'], s['valid'], fr_raw)
                if cell is None:
                    n_dropped += 1
                    continue
                pool.add(s['valid'], *cell)
                it_pool.add(s['valid'], *cell)
        for kind, m in it_pool.band_maps().items():
            iter_maps[kind].append(m)
        print(f'  [{arena_key} relocation] draw {it + 1}/{N_ITER}', end='\r')
    print(f'\n  [{arena_key} relocation] {pool.n_cells} simulated cells pooled '
          f'({N_ITER} draws), {n_dropped} dropped (no spikes in sampled bins)')

    bands = {}
    for kind, lst in iter_maps.items():
        with warnings.catch_warnings():   # never-sampled bins are all-NaN -> NaN band
            warnings.simplefilter('ignore', RuntimeWarning)
            stack = np.array(lst)
            bands[kind] = (np.nanpercentile(stack, BAND_PCT[0], axis=0),
                           np.nanpercentile(stack, BAND_PCT[1], axis=0))
    return pool, n_dropped, bands


# ============================================================================
# Plots
# ============================================================================

def _arena_axes(fig, n_rows, keys, row, polar_for_ring):
    axes = {}
    for j, key in enumerate(keys):
        proj = 'polar' if (polar_for_ring and key == 'circular_track') else None
        axes[key] = fig.add_subplot(n_rows, len(keys), row * len(keys) + j + 1, projection=proj)
    return axes


def plot_null(handlers, pooled, kdes, n_cells, n_real, out_dir):
    keys = [k for k in base._ARENA_ORDER if k in pooled]
    sim_desc = f'real fields (raw rates) relocated to uniform sampled bins, smoothed once, {N_ITER} draws'

    fig = plt.figure(figsize=(5 * len(keys), 13))
    for row, (kind, (_, title, label)) in enumerate(_MAP_KINDS.items()):
        axes = _arena_axes(fig, 3, keys, row, polar_for_ring=True)
        for key in keys:
            values, shown = pooled[key][kind]
            cmap, norm = base.make_cmap_norm(values[shown])
            pcm = handlers[key].plot_fine(axes[key], values, shown, cmap, norm)
            axes[key].set_title(f'{base._ARENA_TITLES[key]} -- {title}\n'
                                f'({n_real[key]} real cells x {N_ITER} draws = {n_cells[key]} sim. cells)',
                                fontsize=9)
            fig.colorbar(pcm, ax=axes[key], shrink=0.7, label=label)
    fig.suptitle(f'NULL pooled maps -- {sim_desc}')
    fig.tight_layout()
    p = os.path.join(out_dir, f'Null_{TAG}_PooledMaps.png')
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {p}')

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
    p = os.path.join(out_dir, f'Null_{TAG}_KDE.png')
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f'[SAVED] {p}')


# ============================================================================
# Duong test: real KDE vs relocation-null KDE
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
    handlers, arena_sessions, real_results = {}, {}, {}
    for key in base._ARENA_ORDER:
        handlers[key], arena_sessions[key], real_results[key] = load_real_sessions(key)

    pooled, kdes, n_cells, n_real, domains, saved, summary = {}, {}, {}, {}, {}, {}, []
    for key in base._ARENA_ORDER:
        sessions = arena_sessions[key]
        if not sessions:
            continue
        handler = handlers[key]
        pool, n_dropped, bands = simulate_arena(handler, sessions, key)
        maps = pool.maps()
        domain = pool.visited
        pooled[key], kdes[key], n_cells[key], domains[key] = maps, {}, pool.n_cells, domain
        n_real[key] = len(real_results[key])
        saved[f'{key}_domain'] = domain
        saved[f'{key}_n_cells'] = pool.n_cells
        saved[f'{key}_n_real_cells'] = n_real[key]

        row = dict(arena=key, n_iter=N_ITER, n_sessions=len(sessions), n_real_cells=n_real[key],
                   n_sim_cells=pool.n_cells, n_dropped=n_dropped)
        for kind, (values_are_mass, _, _) in _MAP_KINDS.items():
            values, _ = maps[kind]
            saved[f'{key}_{kind}_map'] = values
            saved[f'{key}_{kind}_lo'], saved[f'{key}_{kind}_hi'] = bands[kind]
            kde_out = base.kde_density_map(handler, values, domain, values_are_mass) if domain.any() else None
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
        summary.append(row)

    if not pooled:
        print('No sessions found under ROOT_DIRECTORY -- nothing to simulate.')
        return
    plot_null(handlers, pooled, kdes, n_cells, n_real, out_dir)
    npz_path = os.path.join(out_dir, f'Null_{TAG}.npz')
    np.savez_compressed(npz_path, method='relocation_raw', n_iter=N_ITER, band_pct=np.array(BAND_PCT),
                        relocate_margin_bins=RELOCATE_MARGIN_BINS, **saved)
    print(f'[SAVED] {npz_path}')
    xlsx = os.path.join(out_dir, 'Null_Relocation_Summary.xlsx')
    pd.DataFrame(summary).to_excel(xlsx, index=False)
    print(f'[SAVED] {xlsx}')

    duong_real_vs_null(handlers, real_results, kdes, domains, n_cells, out_dir)


if __name__ == '__main__':
    print(f'ROOT_DIRECTORY: {base.ROOT_DIRECTORY}')
    run(OUTPUT_DIR)
    print('\nDone.')
