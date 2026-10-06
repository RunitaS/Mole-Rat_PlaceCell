# MeanRateMap_QuadrantAnalysis_v27_shuffle.py — step-by-step algorithm

This document describes every computation in the script, in the order they run. Function names are
given in `code font` so each step can be found in the commented source.

Notation: *t* is time (µs in the files, seconds after conversion), (*x*, *y*) is position in cm, *i*
indexes spatial bins, *p_i* is occupancy probability, *r_i* is firing rate in bin *i*.

---

## 0. Overview of the pipeline (`run_full_pipeline`)

```
for each arena in [open_field, circular_track, linear_track]:
    build geometry handler (bin grid, areas, edge zone)
    find session folders of that arena
    for each unit (.ntt file) of each session (in parallel):
        load + clean + smooth tracking, speed-filter frames
        load spikes, match each spike to its nearest frame
        occupancy map, spike-count map -> raw rate -> smoothed rate -> field-index map
        metrics (peak, mean, spatial information, sparsity), circular-shift bootstrap
        peak bins and place-field mask
plot 3 pooled maps per arena (overall mean FI, field-only mean FI, peak proportion)
for each of the 3 pooled maps: weighted, edge-corrected 2-D Gaussian KDE per arena
Duong (2013) local test of real KDE vs null KDE   [skipped: no null KDEs are built]
write per-cell summary workbook
```

Output folder: `OUTPUT_DIR/AllArenas`.

---

## 1. Configuration (constants at the top of the file)

| Constant | Value | Meaning |
|---|---|---|
| `ROOT_DIRECTORY` | `...\SessionTypeSorted_PC\Open\Cntrl` | Folder walked to find sessions |
| `fps` | 30 | Camera frame rate |
| `target_bin_cm` | 2.0 | Bin size (cm) |
| `min_occ_s` | 1 | Minimum occupancy (s) for a bin to count as validly sampled |
| `MAX_GAP_US` | 50 000 | Max spike-to-frame time gap (50 ms) |
| `POS_JUMP_THRESH_CMS` | 90 | Speed (cm/s) that marks a tracking jump |
| `POS_SMOOTH_SIGMA_SMP` | 5 | Trajectory smoothing SD (samples) |
| `MIN_SPEED_CMS`, `MAX_SPEED_CMS` | 0.5, 90 | Speed window for spikes and occupancy |
| `RATEMAP_SMOOTH_SIGMA_BINS` | 3 | Rate-map smoothing SD (bins, = 6 cm) |
| `FIELD_PEAK_FRAC`, `MIN_FIELD_BINS` | 0.20, 9 | Place-field threshold and minimum size |
| `N_BOOTSTRAP` | 1000 | Number of circular shifts |
| `KDE_BW_METHOD`, `KDE_EDGE_CORRECTION` | 'scott', True | KDE bandwidth rule, boundary correction |
| `DUONG_ALPHA` | 0.05 | Family-wise level (Hochberg) |
| `DUONG_SAMPLE_SIZE_MODE` | 'neff' | Sample sizes used in the Duong variance |
| `USE_BIN_COVERAGE_CRITERION`, `USE_ZONE_COVERAGE_CRITERION` | False, False | Optional inclusion criteria (both off) |

---

## 2. Arena geometry (`OpenFieldHandler`, `CircularTrackHandler`, `LinearTrackHandler`, `make_handler`)

Every arena is a 2-D grid of `nx × ny` bins stored as a flat index `flat = i·ny + j`.

### 2.1 Open field (circle, diameter 60 cm)
1. `nx = ny = ceil(60 / 2) = 30` → 900 bins; centre (cx, cy) = (30, 30) cm.
2. Bin centres: x = (i + 0.5)·2, y = (j + 0.5)·2.
3. `geom_valid`: bin centre within radius 30 cm of the centre.
4. `dist_to_wall = max(0, 30 − r)`, r = distance of the bin centre from the arena centre.
5. Edge zone: `dist_to_wall ≤ 9.25 cm`.
6. Bin area = 4 cm².
7. `to_bins`: i = floor(x/2), j = floor(y/2), both clipped to the grid; a sample is kept if it is within
   radius + 2 cm (32 cm) of the centre.

### 2.2 Circular track (annulus, inner radius 36 cm, outer radius 40 cm)
1. Mean radius 38 cm; circumference 2π·38 ≈ 238.8 cm.
2. Angular bins `nx = max(8, 4·round(238.8/2/4)) = 120` (3° each); radial bins `ny = max(2, round(4/2)) = 2`
   (2 cm each) → 240 bins; centre (40, 40) cm.
3. All bins are geometrically valid. `dist_to_wall` = distance from the radial bin centre to the nearer wall
   (1 cm for both radial bins).
4. Edge zone = outer radial half (j ≥ ny/2), i.e. the bins next to the outer wall.
5. Bin area = annular sector: ½·Δθ·(r_hi² − r_lo²).
6. `to_bins`: θ = atan2(y − cy, x − cx) in degrees, mod 360; i = floor(θ/3); radial offset
   r − 36 clipped to [0, 4], j = floor(offset/2). Kept if 32 ≤ r ≤ 44 cm (±4 cm tolerance).
7. The angular axis is periodic: smoothing and connected components wrap around.

### 2.3 Linear track (80 × 8 cm)
1. `nx = round(80/2) = 40`, `ny = round(8/2) = 4` → 160 bins of 2 × 2 cm.
2. `dist_to_wall` = min(distance to the nearer end wall, distance to the nearer long wall).
3. Edge zone: bins whose centre is ≥ 2 cm away from the long midline (the rows next to the long
   walls), **or** within 2 cm of an end wall.
4. `orient`: if the trajectory's y-extent exceeds its x-extent, rotate 90° CCW: (x, y) → (−y, x), then
   shift both axes to start at 0.
5. `to_bins`: clip x to [0, 80] and y to [0, 8], then floor-divide by the bin size. Every sample is kept.

---

## 3. Session discovery (`find_arena_sessions`, `_detect_arena_key`)

1. Walk every folder under `ROOT_DIRECTORY` (the output folder is pruned from the walk).
2. Classify the folder: split its path into folder names and return the arena of the first name equal
   (case-insensitive) to `open`, `linear`, `circle` or `circular`. Folders of a different arena are skipped.
3. Tracking files: `.csv`/`.xlsx`; in `'cm'` mode only files ending `_cm.csv` are tracking files.
4. A folder is a session only if it has **exactly one** tracking file and **at least one** `.ntt` file.
5. (Off by default) zone-coverage criterion: compute the session occupancy map (Section 4), then the %
   of valid occupancy in the edge zone and centre zone (`_zone_occupancy_pct`); reject the session if either
   is below 0.1 %.
6. Return `(relative path, folder, tracking path, sorted .ntt list)` for every session, sorted.

---

## 4. Tracking preprocessing (one session)

### 4.1 Load (`_load_tracking`)
1. Read the table (Excel or CSV). In `'cm'` mode: t = column 0 (µs), x = column 3, y = column 4.
2. Drop samples with x = 1 or x = −1 (the tracker's "lost" values).
3. Forward differences: dx_k = x_{k+1} − x_k, dy_k, dt_k (the last sample gets dx = dy = 0, dt = 1) <FIX: {x_:1 = x_:-1, y_:1 = y_:-1, dt = 33.33 msec}.
4. Speed_k = √(dx² + dy²) / dt for dt > 0. Keep samples with dt > 0 **and** speed < 0.006 cm/µs <FIX: 0.00009 cm/µs>
   (= 6000 cm/s, so this removes only gross glitches).
5. Sort by time.
6. Shift so min(x) = min(y) = 0 (cm mode). In pixel mode, also divide by <FIX: Use center as midpoint instead, or skip this step. Centering procedure repeated in step 4.3>
   px_per_cm = max(x-extent, y-extent) / arena width.

### 4.2 Jump removal and smoothing (`_smooth_tracking_position`)
1. Repeat until nothing changes: over the samples still marked good, compute step speed between
   consecutive good samples, v = distance / (Δt·10⁻⁶) cm/s; mark the **later** sample of each step with
   v > 90 cm/s as bad.
2. Replace bad samples by linear interpolation in time between good samples (`np.interp`).
3. Smooth x and y separately with a 1-D Gaussian, σ = 5 samples (edges padded with the end value).

### 4.3 Centring (`_centre_open_field_tracking`), open field only
x ← x + (cx − (min x + max x)/2), y ← y + (cy − (min y + max y)/2): the midpoint of the trajectory's bounding box
is moved to the arena centre (30, 30). <FIX: Apply to all arenas>

### 4.4 Frame table (`_session_frames`) 
<FIX: Apply 0.5 min speed requirement to all arenas, not just linear track>
1. Orient (linear track only, Section 2.3).
2. Speed filter (`_speed_mask`): v_k = |p_k − p_{k−1}| / Δt (cm/s); frame 0 takes v_1.
   `moving_k = 0.5 ≤ v_k ≤ 90` (a NaN speed fails).
3. Bin every frame (`to_bins`) and discard frames that fall outside the arena tolerance.
4. Dwell time per frame: dt_0 = 1/30 s; dt_k = min(t_k − t_{k−1}, 2/30 s), which caps tracking gaps at two frames.
5. Frames counted in occupancy: `occ_frame = moving` (because `SPEED_FILTER_OCCUPANCY = True`).

---

## 5. Per-unit processing (`process_unit`, run in a 4-thread pool by `collect_arena_results`)

### 5.1 Spikes
1. Memory-map the `.ntt` file, skipping the 16 kB header. Each record holds a uint64 timestamp (µs), the
   channel number, the cell (cluster) number, 8 feature words and a 32 × 4 int16 waveform.
2. Keep records with `cell_number ≠ 0` (cluster 0 = unassigned spikes). Sort the timestamps.

### 5.2 Spike-to-frame matching and speed filter (`compute_cell_ratemap`)
1. For each spike, find the frames immediately before and after it (`searchsorted`) and take the nearer
   one (ties go to the earlier frame).
2. Matched if |t_spike − t_frame| ≤ 50 ms. Used if matched **and** the *frame is moving*.
   `n_spikes` = used spikes; `n_spikes_speed_excluded` = matched spikes on non-moving frames.
3. Remove non-moving frames from occupancy: `new_index = cumsum(moving) − 1` maps each original frame to its
   position among moving frames; the spike frames are re-indexed with it; t, bin and dt keep only the moving frames.

### 5.3 Maps (`compute_cell_ratemap` → `rate_maps_from_counts`)
1. Occupancy: occ_i = Σ dt_k over moving frames in bin i (s).
2. Spike counts: s_i = number of used spikes whose frame lies in bin i.
3. Valid bins: occ_i ≥ 1 s **and** *`geom_valid`* <FIX: Bin centering criteria only applied to open>.
4. Raw rate: r_i^raw = s_i / occ_i (Hz) in valid bins, 0 elsewhere.
5. Smoothed rate (`_gaussian_smooth_2d`), a normalised convolution with σ = 3 bins: <FIX: Apply 2D Gaussian smoothin edge correction>
   r^smooth = G * (r^raw·M) / G * M, where M = valid mask and * is 2-D Gaussian filtering (zero padding; the angular
   axis wraps on the circular track). Invalid bins are set to 0.
6. Field-index map (`field_index_map`): FI_i = (r_i − min r) / (max r − min r) over valid bins (0 if the map is flat,
   NaN in invalid bins). <FIX: Make sure the Field-index transformation is applied to smoothed rate map>

### 5.4 Metrics (`rate_maps_from_counts`), using the smoothed rates over valid bins
<FIX: Confirm if these metrics are calculated on raw rate maps>
- **p_i = occ_i / Σ occ** WhoKnew! probability of occupancy in a bin is basically proportion of time spent in that bin.
- Mean rate **r̄ = Σ p_i r_i**; **peak rate = max r_i**
- Skaggs spatial information (bits/spike): **SI = Σ p_i (r_i/r̄) log₂(r_i/r̄), summed over bins with r_i > 0**
- **Sparsity = (Σ p_i r_i)² / Σ p_i r_i²**
- All values are rounded to 4 decimals.

### 5.5 Peaks and place field (`_cell_peaks_and_field`, `extract_place_field_mask`)
<FIX: Extract fields from smoothed rate maps, but detect peak bin from raw rate maps>
1. `peak_bin` = argmax of the smoothed rate over valid bins; `peak_bin_raw` = argmax of the raw rate.
2. Candidate field bins: valid **and** r^smooth > r̄ **and** r^smooth > 0.2 · peak.
3. Group the candidates into 8-connected components (`_connected_components_2d_flat`: an iterative depth-first
   search over the 3 × 3 neighbourhood, wrapping the angular axis on the ring). Components with ≥ 9 bins form the
   place-field mask.

### 5.6 Circular-shift bootstrap (`run_bootstrap_generic`): diagnostic only
1. Seed = 64-bit BLAKE2b hash of "session|unit" XOR 0 (`_stable_seed`), so every cell's result is reproducible
   regardless of thread order.
2. If there are no spikes → not significant; if the session has ≤ 1200 moving frames (2 × 20 s) → sig = None.
3. Repeat 1000 times: draw a shift k uniformly in [600, N − 600] frames; shifted spike frames = (f + k) mod N;
   rebuild the spike map with occupancy and valid bins unchanged; smooth; compute SI. <MAJOR FIX: bootstrapping is applied to smoothed rate map>
4. Significant if the real SI > 95th percentile of the 1000 shuffled SIs. The flag is reported, but it does not
   exclude any cells.

### 5.7 Inclusion
Every unit that produces a rate map is pooled. The optional bin-coverage criterion (valid bins ≥ 1 % of
arena bins) is off. Any exception inside a unit's processing drops that unit silently.

---

## 6. Pooling across cells (per arena)

Let C be the set of pooled cells and V_c the valid bins of cell c.

1. **Overall mean field-index map** (`pool_fine_map`): M_i = mean of FI_c,i over cells with i ∈ V_c
   (`nanmean` of a cells × bins matrix). Defined where at least one cell sampled the bin.
2. **Field-only mean map** (`pool_field_only_map`): M_i = Σ_{c: i ∈ field_c} FI_c,i / #{c: i ∈ V_c}. Cells that
   sampled bin i but have no field there count as 0. Defined only in bins that lie inside at least one cell's field.
3. **Peak-proportion map** (`pool_peak_proportion_map`): P_i = 100 · #{c: peak_bin_raw_c = i} / |C|, defined on
   the union of valid bins.
4. Domain for the KDE (`_union_valid`): the union of all V_c.

Stage-1 figures (`plot_fig_S1H`, `plot_field_only_mean_maps`, `plot_peak_proportion_maps`): one panel per arena
(polar axes for the ring), jet colour map scaled to the finite min–max, and white for unsampled bins.

---

## 7. KDE of each pooled map (`arena_kde` → `kde_density_map`)

For each map kind (overall, field_only, peak) and each arena:
1. Data points = bin centres (cm) (`bin_centres_xy`).
2. Values outside the domain, or non-finite, are set to 0; negative values are clipped to 0.
3. Weights (mass): for intensity maps (overall, field-only) w_i = value_i · area_i; for the peak map
   w_i = value_i (it is already a proportion).
4. Fit `scipy.stats.gaussian_kde` to the bins with w > 0 (at least 3 are needed). The weights are normalised internally;
   kernel covariance H = (Scott factor)² · weighted covariance of the points, with Scott factor = n_eff^(−1/6) and
   n_eff = 1 / Σ ŵ_i² (Kish effective sample size).
5. Evaluate the density f̂_raw(x) at every domain bin centre.
6. Edge correction (`_kernel_mass_in_domain`): m_H(x) = Σ_{y ∈ domain} N(x − y; 0, H) · area(y), the share of the kernel
   that falls inside the sampled domain (Riemann sum). Corrected density f̂ = f̂_raw / m_H.
7. `scaled` = f̂ · Σw (· area_i for the peak map): the density converted back into map units.
8. Plot (`plot_kde_maps`): open field on the left; ring (unrolled to arc length × radial position) top right; linear
   track bottom right. Titles report the Scott factor, n_eff and kernel SDs √diag(H). Also saved: `KDE_<stem>.npz`
   (density, raw density, scaled map, covariance, factor, n_eff, domain).

---

## 8. Duong (2013) local test (`run_duong_tests` → `duong_compare_arena`)

**Status: skipped in this script.** `kdes['null']` is empty, so the function prints "skipped" and writes nothing.
Once null KDEs are supplied, it runs as follows for every (map kind, arena):

1. **Sample sizes**: n1 = n_eff(real) and n2 = n_eff(null) in `'neff'` mode; or the numbers of real and null cells in `'cells'` mode.
2. **Common domain**: bins in both domains where both corrected densities are finite and both raw densities are > 0.
3. **Per side** (`_duong_kde_side`), at the common points:
   - m_H and m_{H/2} = kernel mass inside the KDE's own domain for bandwidth H and H/2.
   - c = f̂ / f̂_raw (the correction the KDE actually applied); sanity check max|c·m_H − 1| (a warning is printed if it exceeds 2 %).
   - f = f̂_raw / m_H.
   - Variance: Var = c² · R(K) / (n · |H|^½) · f · m_{H/2}, with R(K) = 1/(4π). In the interior this is
     Duong's R(K) n⁻¹|H|^−½ f(x); at a wall it is larger.
   - Renormalisation: scale = 1 / Σ_common f̂·area; f ← f̂·scale, Var ← Var·scale².
   - Mass outside the common domain = 1 − Σ_common f̂·area / Σ_own f̂·area.
4. **Statistic**: z = (f1 − f2) / √(Var1 + Var2); X² = z²; p = P(χ²₁ ≥ X²).
5. **Hochberg step-up** (`hochberg_reject`): sort p ascending; j* = max{ j : p_(j) ≤ α / (m − j + 1) }; reject the j*
   smallest p-values. The p cut-off is p_(j*).
6. **Clusters**: rejected points with f1 > f2 ("above") and with f1 < f2 ("below") are each grouped into 8-connected
   clusters (`_duong_cluster_labels`, minimum 1 bin).
7. **Cluster table** (`duong_cluster_rows`): number of bins; area; area-weighted centroid (x, y); area-weighted
   mean, min and max distance to the wall; mean real and null density; mean real/null ratio; excess mass
   Σ(f1 − f2)·area; peak |z|; min p. On the ring it also gives the area-weighted circular mean angle
   atan2(Σa·sinθ, Σa·cosθ).
8. **Figure** (`plot_duong_kind`): one row per arena, five columns: real density, null density (shared grey scale),
   difference and z (diverging scales centred on 0), and the cluster map. Green outlines/fill mark real > null and magenta
   mark real < null. The |z| threshold shown is z* = Φ⁻¹(1 − p_cut/2). Bins are drawn as true-geometry polygons
   (`_bin_polygons`, ring bins follow the arcs), and outlines are the bin edges between differently labelled bins
   (`_field_boundary_segments`).
9. **Files**: `DuongTest_<kind>.png`, `DuongTest_Clusters.xlsx` (sheets "Tests" and "Clusters"), `DuongTest_Results.npz`.

---

## 9. Export (`export_excel`)

`AllArenas_Summary.xlsx` has one row per pooled cell: arena, session, unit, n_spikes, n_spikes_speed_excluded,
peak_fr, mean_fr, SI, sparsity and bootstrap_sig.

---

## 10. Output files (in `OUTPUT_DIR/AllArenas`)

| File | Content |
|---|---|
| `WholeRM_MeanFieldIndex.png` | Overall mean field-index map per arena |
| `FieldOnly_MeanFieldIndex.png` | Field-only mean map per arena |
| `PeakProportion_Map.png` | % of cells peaking in each bin |
| `KDE_FigS1H_MeanFieldIndex.png/.npz`, `KDE_FieldOnly_MeanFieldIndex.png/.npz`, `KDE_PeakProportion_Map.png/.npz` | KDE of each pooled map |
| `DuongTest_*` | Only when null KDEs are supplied |
| `AllArenas_Summary.xlsx` | Per-cell diagnostics |
