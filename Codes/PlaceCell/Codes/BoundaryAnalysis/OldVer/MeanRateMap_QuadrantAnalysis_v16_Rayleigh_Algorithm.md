# Algorithm: `MeanRateMap_QuadrantAnalysis_v16_Rayleigh.py`

This document explains, in plain language, everything the script does, in the order it does it.
Numbers quoted as "current" come from the settings at the top of the script as it is saved now.

---

## 0. The big picture (one paragraph)

You have many place cells recorded in three kinds of arena (an open circular field, a circular
track and a linear track). The script asks one question: **where in the arena do place fields
sit, and is that pattern non-random?** In particular, is there a preference for the walls
(boundaries), and (in the open field) is there a *direction* the fields lean towards?

To answer it, the script:

1. Finds all recording sessions and throws out sessions where the animal did not explore enough.
2. Builds a **firing-rate map** for every cell (how fast it fires in each small square of floor).
3. **Pools** all the cells' maps into group maps (three flavours: *overall*, *peak*, *field-only*).
4. **Folds** the arena into one quadrant (the "Muessig et al. Fig 1" trick) so the four quarters can be averaged.
5. Compares the pooled maps against **null models** ("what would we see if fields were spread evenly?") using bootstrap confidence intervals.
6. In the open field, runs a **Rayleigh test** to see if the pattern is biased toward a particular compass direction.
7. Saves figures (.png) and tables (.xlsx).

```
 tracking + spikes ──► session screening ──► per-cell rate maps ──► pooled maps
                                                                        │
              ┌─────────────────────────────────────────────────────────┤
              ▼                    ▼                     ▼              ▼
     Fig S1H / field-only    Fig 1B/D quadrant     Null models    Boundary (1D) KDE
        mean maps               fold maps        (uniform-field,   Quadrant KDE
                                                   occupancy)      Fine-map KDE
                                                                   Rayleigh (open field)
                                                                        │
                                                          Null comparison + Excel/PNG outputs
```

**Important:** the script does **not** decide whether a cell is a place cell. Every `.ntt` file
under the root folder is *assumed* to be an already-accepted place cell (that screening was done
by an earlier pipeline). The only inclusion rules here are about how well the *animal* covered the
arena in that session.

---

## 1. Settings

| Setting | Current value | Meaning |
|---|---|---|
| `ROOT_DIRECTORY` | `...\SessionType_Sorted\Open\Zero` | Folder tree to search for sessions |
| `OUTPUT_DIR` | `<root>\MeanRM_Quad_Rayleigh` | Where results go (a sub-folder `AllArenas` is created) |
| `fps` | 30 Hz | Tracking frame rate |
| `target_bin_cm` | 2 cm | Size of each square spatial bin |
| `min_occ_s` | 1 s | A bin needs at least this much time spent in it to count |
| `MAX_GAP_US` | 50 000 µs (50 ms) | A spike is only used if a tracking sample exists within 50 ms |
| `POS_JUMP_THRESH_CMS` | 90 cm/s | Frame-to-frame speeds above this are treated as tracking glitches |
| `POS_SMOOTH_SIGMA_SMP` | 1 sample | Smoothing of the position trace |
| `RATEMAP_SMOOTH_SIGMA_BINS` | 3 bin | Smoothing of rate maps |
| `FIELD_PEAK_FRAC` / `MIN_FIELD_BINS` | 0.20 / 9 | Place-field definition (see step 6) |
| `N_BOOTSTRAP` | 1000 | Shuffles per cell for the spatial-information test |
| `N_KDE_BOOTSTRAP` | 1000 | Cell-resamples for the confidence intervals |
| `KDE_ALPHA` | 0.05 | Two-sided 95 % interval (2.5th–97.5th percentile) |
| `COVERAGE_FRACTION` | 0.01 (debug value; comment says it was 0.80) | Fraction of arena bins that must be covered |
| `MIN_ZONE_COVERAGE_PCT` | 0.1 % | Minimum time in edge zone and in centre zone |
| `CENTRE_OPEN_FIELD_TRACKING` | True | Re-centre open-field tracking on the real arena centre |
| `ROTATE_SESSION_CCW_DEG` | 0.0 (the 120° value is commented out) | Rotation applied to "Rotate" sessions (currently none) |
| `NULL_MODE` | `'uniform_field'` | Which null is the *primary* one |
| `FIELD_FOOTPRINT_BINS` | linear 7, circular 9, open 12 | Assumed size of a place field in the uniform-field null |
| `PEAK_WITHIN_FIELD` | `'occupancy'` | How the null picks a "peak" bin inside a field |
| `RAYLEIGH_BIN_DEG` / `RAYLEIGH_RADIAL_POWER` | 30° / 1.0 | Rayleigh angular bins and wall-weighting |
| `QUADRANT_COLOUR_SCALE` | `'shared'` | Whether the 3 arenas in a figure row share one colour scale |
| `BOOTSTRAP_SEED` | 0 | Makes the shuffle test reproducible |

---

## 2. Step 1 – Find the sessions and decide which arena each belongs to

For each of the three arenas (`open_field`, `circular_track`, `linear_track`, always in that order):

1. Walk through every sub-folder of `ROOT_DIRECTORY`.
2. **Arena detection from the folder path.** If any folder name in the path is (case-insensitive)
   `Open` → open field; `Linear` → linear track; `Circle` or `Circular` → circular track.
   (Example: `.../Fa8477/Open/Day8/1Cntrl` is an open-field session.) Folders that belong to a different arena are skipped for this arena.
3. A folder counts as a **session** only if it has **exactly one** tracking file ending in `_cm.csv`
   (because `COORD_UNITS = 'cm'`) **and** at least one `.ntt` spike file.
4. The session name is the folder path relative to the root.

> Note: with the current `ROOT_DIRECTORY` (…`\Open\Zero`), only open-field sessions exist under it,
> so circular and linear arenas will find no sessions.

---

## 3. Step 2 – Load and clean the tracking (done once per session)

Function chain: `_session_positions` → `_load_tracking` → `_smooth_tracking_position` →
`_centre_open_field_tracking` → `_apply_session_rotation`.

### 3a. Load
1. Read the CSV. Column 0 = time (µs, same clock as the spikes), column 3 = x (cm), column 4 = y (cm).
2. **Drop "lost tracking" samples** where x is exactly 1 or −1 (the tracker's error code).
3. For each sample, compute the step to the next sample and its speed = distance ÷ time step.
   Keep only samples with a positive time step and speed below `0.006` (in file units). With microsecond
   time stamps this is a very generous limit, so in practice it only removes extreme jumps and duplicate/backwards time stamps.
4. Sort by time.
5. **Shift the coordinates** so the smallest x and smallest y are 0 (`x − x.min()`, `y − y.min()`).
   So each session's frame is anchored on its own bounding box.

### 3b. Remove jumps and smooth
1. Compute the speed between consecutive "good" samples in cm/s.
2. Any sample reached by a jump faster than **80 cm/s** is marked bad. Repeat (because removing one
   sample changes the neighbours' speeds) until no more jumps are found.
3. Replace bad samples by **linear interpolation** over time.
4. Apply a **Gaussian smoothing** (σ = 1 sample) to x and y.

### 3c. Open field only: re-centre the arena
Problem: because coordinates were anchored to the bounding box, the assumed arena centre (30, 30) can be
up to ~3 cm off if the animal never touched one side of the wall.

Algorithm (`_estimate_disc_centre`):
1. Start from the midpoint of the tracking's bounding box.
2. Cut the surroundings into 72 angular sectors; in each sector keep the sample **farthest** from the start point (the "outer envelope" = where the animal came closest to the wall).
3. Discard sectors whose farthest sample is not at least 90 % of the arena radius (the animal never went near the wall there, so it tells us nothing about where the wall is).
4. Refuse to fit (and leave the data unchanged, with a printed warning) if fewer than 12 sectors survive,
   or the surviving sectors leave an angular gap wider than 180° (all on one side), or the fitted centre moves by more than 25 % of the radius.
5. Otherwise fit a circle of the known radius (30 cm) to the envelope points by iterating
   `centre = mean(points) − radius × mean(unit vectors from centre to points)` until it stops changing (max 100 iterations).
6. **Translate** all tracking so this fitted centre lands on (30, 30).

### 3d. Open field only: undo rotation of "Rotate" sessions
If the session folder name contains "rotate", the positions are rotated counter-clockwise about the arena centre by
`ROTATE_SESSION_CCW_DEG` so all sessions share one frame. **Currently set to 0°, so nothing changes.**
(Rotation preserves distance from the centre, so it does not alter the edge/centre split.)

---

## 4. Step 3 – Turn positions into spatial bins (arena "handlers")

Each arena has a "handler" that converts an (x, y) position to a **single bin number** (0 … n_bins−1). This lets all later
code be written once for all three arenas. Current geometry with 2 cm bins:

| | Open field | Circular track | Linear track |
|---|---|---|---|
| Physical size | disc, 60 cm diameter | ring, outer Ø 80 cm, inner Ø 72 cm (4 cm wide) | 80 × 8 cm |
| Grid | 30 × 30 (x, y) | 120 arc bins × 2 radial bins | 40 length bins × 4 width bins |
| Bins that are real floor | those whose centre is within 30 cm of the centre (`geom_valid`) | all 240 | all 160 |
| Bin coordinates | `x/2`, `y/2` | arc bin = angle ÷ 3°; radial bin = (r − inner radius) ÷ 2 cm | `x/2`, `y/2` |
| Sample kept if… | within radius + 1 bin of centre | radius within 4 cm of the track | always |
| Special handling | none | arc axis **wraps around** (bin 119 touches bin 0) | vertical tracks are **rotated 90°** so the long axis is always x |

### Edge vs centre zones (used only for the session screen, step 5)
* **Open field:** "edge" = bin centre within **9.25 cm** of the wall.
* **Circular track:** "edge" = the outer radial bin; "centre" = the inner radial bin.
* **Linear track:** "edge" = within 2 cm of either end wall, **or** at least 2 cm away from the mid-line across the width; the rest is "centre".

### Extra distances precomputed for later analyses
* Distance to the nearest wall (open field: from the circular wall; circular track: to the nearer of inner/outer rim; linear: nearest of the 4 walls).
* Boundary-analysis x-axis (see Step 10): distance from **arena centre** (open), distance from the **track mid-point** (linear), or **position along the ring in cm** (circular).

---

## 5. Step 4 – Session screening (two "did the animal cover enough arena?" tests)

Done from tracking alone, before any spikes are loaded.

**Test A – Edge/centre balance** (`USE_ZONE_COVERAGE_CRITERION`), once per session:
1. Load positions (step 3), bin them, add up time spent in each bin.
   Time per sample = gap to the previous sample, **capped at 2/fps (≈ 67 ms)**, so pauses/dropouts don't inflate occupancy (first sample gets 1/fps).
2. Keep bins with ≥ `min_occ_s` (1 s) and inside the arena.
3. Compute the % of total valid time spent in edge bins and in centre bins.
4. If either percentage is below `MIN_ZONE_COVERAGE_PCT` (0.1 %), **skip the whole session** (all its cells).
   A session whose tracking can't be read is also skipped.

**Test B – Whole-arena bin coverage** (`USE_BIN_COVERAGE_CRITERION`), applied inside the per-unit step:
* Count the valid bins (occupancy ≥ 1 s, inside the arena). If fewer than `ceil(COVERAGE_FRACTION × total arena bins)` the unit is skipped.
  Because it depends only on tracking, it removes *all* units from that session together.

---

## 6. Step 5 – Build one cell's rate map and statistics (`process_unit`)

Units are processed in parallel by 4 worker threads. For every `.ntt` file:

### 6a. Read the spikes
1. Memory-map the file, skipping the 16 KB header.
2. Drop spikes whose `cell_number` is 0 (unclustered noise).
3. Convert time stamps to floats and sort. If there are none, skip the unit.

### 6b. Assign spikes to positions
For each spike time, find the **nearest tracking sample** (looking to the left and right). If that sample is
more than 50 ms away, the spike is discarded. Otherwise the spike is credited to that sample's spatial bin.

### 6c. Occupancy, spike counts and rate maps
1. **Occupancy map** = total time (s) in each bin (rule from step 4).
2. **Spike map** = number of accepted spikes in each bin.
3. **Valid bins** = occupancy ≥ 1 s **and** inside the arena.
4. **Raw rate** = spikes ÷ occupancy (Hz) in each valid bin (0 elsewhere).
5. **Smoothed rate**: 2D Gaussian (σ = 1 bin), done the "NaN-safe" way — smooth the rates and smooth a 0/1 validity map separately, then divide.
   This means invalid/unvisited bins don't drag values towards zero. On the circular track the arc axis wraps around; the radial axis does not.
6. **Field-index map** (0–1): `(rate − min) / (max − min)` over valid bins of the smoothed map. Every cell's own peak becomes 1 and its lowest bin 0. If the map is completely flat, all values are 0.
   This is what gets pooled later, so a high-rate cell does not count more than a low-rate cell.

### 6d. Per-cell statistics (diagnostics, they do **not** exclude cells)
Let `pᵢ` = share of valid occupancy in bin i, `rᵢ` = smoothed rate in bin i.
* **Mean rate** `r̄ = Σ pᵢ rᵢ` (occupancy-weighted).
* **Peak rate** = maximum smoothed rate.
* **Spatial information (`sir`)** = `Σ pᵢ (rᵢ/r̄) log₂(rᵢ/r̄)` over bins with rᵢ > 0 (bits per spike — how much a spike tells you about location).
* **Sparsity** = `(Σ pᵢ rᵢ)² / Σ pᵢ rᵢ²` (small = firing concentrated in few bins).

### 6e. Shuffle test for spatial information (diagnostic only)
Repeated 1000 times:
1. Pick a random shift between 20 s and (session length − 20 s) (in frames).
2. Shift every spike's frame index by that amount, wrapping around the end of the session (circular shift). Positions stay unchanged, so the animal's path and the cell's firing statistics are preserved but their relationship is scrambled.
3. Rebuild spike map → rate → smooth → recompute spatial information.

Then: `bootstrap_mean`, `bootstrap_p95` (95th percentile of the shuffled values), and `bootstrap_sig = (real SIR > p95)`.
Special cases: no spikes → NaN/False; session shorter than 40 s worth of frames → `None`.
The random generator is seeded from the unit's identity (`_stable_seed`), so results are reproducible and independent of thread timing.
**This result is written to Excel but does not filter anything.**

### 6f. Peak bin
The valid bin with the highest smoothed rate = the cell's single "peak location".

### 6g. Place-field mask
1. A bin **qualifies** if its *raw* rate is above the cell's mean rate **and** above 20 % of the cell's raw peak.
2. Group qualifying bins into connected blobs (8-neighbour connectivity, i.e. diagonals count; wrap-around for the circular track), using a flood-fill.
3. Keep only blobs of **≥ 9 bins**. Their bins are the cell's place-field mask.

The unit is now stored with: rate maps, field-index map, valid mask, occupancy map, peak bin, field mask, and the statistics.

---

## 7. Step 6 – Pool cells into group maps

There are three pooled maps, used everywhere below. Call them **overall**, **peak**, and **field**.

### 7a. "Overall" mean map (Fig S1H) – `pool_fine_map`
For each bin, take the field-index values of all cells that had this bin valid and **average** them (ignoring cells where the bin was not visited). Bins that no cell visited are invalid.

### 7b. "Field-only" mean map – `pool_field_only_map`
Uses only bins inside each cell's own place field, so a cell contributes nothing where it doesn't fire.
1. For each bin, `fi_sum` = sum of the field-index values of all cells whose field covers this bin; `n_fields` = how many fields cover it.
2. `total` = sum of `n_fields` over the whole map.
3. Bin value = `fi_sum / total`.
So a bin scores high if **many fields overlap it** and/or those fields have **high field index** (near their own peaks).

### 7c. "Peak" map – `pool_peak_proportion_fine`
Count how many cells have their peak bin in each bin, and convert to **percent of all peaks**.

---

## 8. Step 7 – Quadrant folding (Muessig et al. Fig 1A)

**Idea:** the walls of the arena are equivalent, so slice the arena into 4 quadrants, flip each onto one reference quadrant so that
"wall a lands on wall a′" and "wall b on wall b′", and average the four copies. The result is one small map that shows firing versus
distance from the walls, with 4× the data.

How the flip is built (`_build_reflect_quadrant_fold`): mirror each axis about its own mid-line independently
(not a 90° rotation, which would swap axes and match the wrong walls together).
* **Open field** (30 × 30): 4 mirror combinations → one 15 × 15 quadrant (225 folded bins).
* **Linear track** (40 × 4): → 20 × 2 folded bins (40).
* **Circular track**: the ring is cut into four 90° arcs (30 arc bins each) using the two symmetry axes;
  arcs 1 and 3 have their arc order reversed so that distance from the nearest axis lines up. The radial direction is kept unchanged (reflecting about a diameter does not change a point's radius), giving 30 arc × 2 radial = 60 folded bins.

Two operations are then used:
* **Fold a mean map** (`_fold_mean_map`): each folded bin = average of the (valid) values of its partner bins (up to 4).
* **Fold a peak** (`fold_peak_bin`): a peak bin is simply re-labelled with its folded bin number, and counted.

Figure results:
* **Fig 1B**: folded % of peaks per bin.
* **Fig 1D**: the overall mean map (7a) folded.
* Also folded: the field-only map (7b), used in the quadrant KDE analysis.

---

## 9. Step 8 – Null models (what "no preference" looks like)

A significance test needs a comparison. The script builds two nulls for every analysis and marks one as primary (`NULL_MODE`).

### 9a. Occupancy null (the old one)
The pooled **dwell time** map (`_pooled_occupancy`): for each bin, sum the occupancy over all cells (optionally restricted to each cell's own field bins to pair with the field map).
Idea: "if the cells fired wherever the animal spent time, the map would look like the time map."

### 9b. Uniform-field null (primary) – `build_uniform_field_null`
Idea: "if place fields were scattered **uniformly at random**, what maps would the observed animals' sessions produce?" This corrects for the fact that fields of finite size can't be centred right at the wall, so wall bins can be covered by fewer possible fields.

1. **Field footprint** — a fixed shape per arena:
   * Open field: 12 bins, the most compact shape (a 4 × 4 block with corners cut).
   * Tracks: full-width columns along the track, last column partial and centred (linear: 7 bins; circular: 9 bins).
   * Mirror-image versions of track footprints are all used and given equal total weight.
2. **All allowed placements**: slide the footprint to every position where it fits entirely inside the arena (wrapping around for the circular track). Each placement is equally likely.
3. **Per session** (each session is counted once per unit that survived, as in the real pooling):
   * *Observed part of the field* = footprint ∩ bins the animal actually visited (valid bins).
   * Drop placements whose observed part has fewer than 7 bins (they would not have been detected as a field).
   * Re-normalise the weights of the remaining placements.
   * `cov` = probability that each bin lies in the observed field.
   * For the peak: the peak inside a field is drawn with probability proportional to the session's **dwell-time share** in each field bin (`PEAK_WITHIN_FIELD='occupancy'`).
4. **Add up** over sessions (weighted by number of units) to get:
   * **overall null** = expected chance a valid bin is in a field (compares with the *overall* map),
   * **field null** = share of total field coverage (compares with the *field-only* map),
   * **peak null** = expected % of peaks per bin.
   * plus `field_weight` = dwell time inside the synthetic fields (used as weights in the 1D analysis).

Nulls and observed maps are in different units, so before every comparison **both are rescaled to sum to 100 %** (`_normalize_pct`).

---

## 10. Step 9 – Figures of the group maps (no statistics)

* `FigS1H_MeanFieldIndex.png` – the overall mean map, one panel per arena (circular track drawn on a polar plot).
* `FieldOnly_MeanFieldIndex.png` – the field-only mean map.
* `Fig1BD_QuadrantFold.png` – top row folded peak proportions (Fig 1B); bottom row folded mean field index (Fig 1D). Colours follow `QUADRANT_COLOUR_SCALE`.
* `AllArenas_Summary.xlsx` – one row per unit: arena, session, unit, n_spikes, peak rate, mean rate, SIR, sparsity, `bootstrap_sig`, and the folded peak bin.
* `UniformFieldNull_ExpectedMaps.png` – the expected (null) overall/field/peak maps from 9b.

---

## 11. Step 10 – Boundary-preference analysis (1D curves) → `WallDistance_KDE.png`, `WallDistance_KDE_Peaks.xlsx`

**Question:** does firing (or the density of peaks/fields) depend on distance from the boundary more than chance predicts?

Run for each arena × each map type (overall, peak, field). Circular track is run twice (inner ring, outer ring).

1. **Choose the x-axis**, in 2 cm position bins:
   * Open field: distance from arena **centre** (0–30 cm, 15 bins).
   * Linear track: distance from the track **mid-point** (0–40 cm, 20 bins; both ends fold together, width ignored).
   * Circular track: **position around the ring** in cm (120 bins, wraps around), separately for the inner bin row and the outer bin row (a 2-bin-wide track has no continuous radial axis).
2. **Build the observed curve** (`_build_observed_curve`):
   * *overall*: average the pooled map's bins within each position bin (each bin equal weight).
   * *field*: weighted average, weights = pooled dwell time inside field bins.
   * *peak*: histogram of where the cells' peaks fall (counts per position bin).
   * Then smooth with a 1D Gaussian (σ = 1 position bin; wraps for the ring; NaN-safe) and rescale to sum to 100 %.
3. **Build both null curves** (uniform-field and occupancy) using the *same* steps (average or sum → smooth → normalise). Mean-type maps get mean-type nulls; peak counts get sum-type nulls.

<BS way of analyzing significance! DO NOT USE THIS!!!>
4. **Bootstrap** (1000 times): resample the place cells **with replacement**, rebuild the curve each time.
   Take the 2.5th, 50th and 97.5th percentiles at each position → `real_lo`, `real_med`, `real_hi`.
5. **Significance:** a position is significant if **`real_lo` > null** (the whole 95 % interval is above the null).
   Runs of ≥ 3 consecutive significant positions are reported as "boundary peaks" (start, end, position of highest median, its %, and the null %).
6. Plots: null(s), observed median ± interval band, shaded significant ranges. Excel lists the peaks.

---

## 12. Step 11 – Quadrant-fold KDE analysis → `QuadrantFold_KDE.png`, `QuadrantFold_KDE_Clusters.xlsx`

A 2D version of step 10 on the small folded grid (step 8). For each arena × map type:

1. **Observed**: pool the maps → fold → smooth with a 2D Gaussian (σ = 1 bin, no wrap) → rescale to 100 %.
2. **Nulls** (uniform-field and occupancy): fold the null map, smooth, rescale the same way.
3. **Bootstrap** the cells 1000 times, repeating step 1 for each resample → per-bin 2.5th/50th/97.5th percentiles.
4. **Data mask** (`quad_valid`): only folded bins that really have data in the observed map *and* both nulls are tested.
   (Smoothing zero-fills empty bins, so without this mask activity would "bleed" into places outside the arena — e.g. corners of the open-field grid — and produce fake significance.)
5. For each null:
   * **Significant** (above null) bin: `real_lo > null`.
   * **Depleted** (below null) bin: `real_hi < null`.
   * **Clusters**: connected groups of significant bins; each reported with size, peak location and peak %.
6. Figure: 3 × 3 grid (map type × arena) showing the median map with black outlines around significant bins, and a thin panel beneath each row showing **observed − null** on a blue-white-red scale centred on zero.

---

## 13. Step 12 – Fine-map (whole arena) KDE analysis → `FineMap_KDE.png`, `FineMap_KDE_Clusters.xlsx`

Exactly the same procedure as Step 11 (quadrant KDE), but on the **unfolded** whole-arena grid, so you see where in the actual arena (not a folded quadrant) the pooled map is significantly above the null.
Differences: the 2D smoothing uses each handler's own smoothing (so the circular track wraps around), and clusters use each handler's own connected-component rule (also wrapping for the ring). Mask = `fine_valid`.

---

## 14. Step 13 – Rayleigh vector analysis (open field only) → `Rayleigh_OpenField*.png`, `Rayleigh_OpenField.xlsx`

**Question:** is the pooled map "lopsided" toward a compass direction (e.g. more firing towards the East wall)?

Run on three versions of each map type: **`pooled`** (the raw pooled maps of step 7), **`kde`** (bootstrap-median smoothed map of the fine-map KDE analysis, Step 12, restricted to sampled bins), and **`kde_sig`** (as `kde` but only the significantly elevated bins).

1. **Find the true centre** (`_circle_centre_from_circumference`): take the outline bins of the circle (bins inside the arena that touch an outside/off-grid neighbour), and use the midpoint of their extent in x and y. It depends only on the arena outline, not on which bins hold data. With an even grid the centre sits on a corner shared by four bins, so no bin is exactly at the centre.
2. **Angle of each bin**: for every valid bin compute its angle around that centre (degrees anticlockwise from +x, like the plots). Each bin goes into exactly one of **12 angular bins of 30°** (half-open intervals so none is lost or double-counted).
3. **Radial weight** = `(distance from centre ÷ arena radius)^1`, so wall-side bins count more than centre-side bins.
4. **Magnitude of each angular bin** = the sum (`RAYLEIGH_MAGNITUDE='sum'`) of `map value × radial weight` over the bins in it. (The unweighted sum is also stored.)
5. **Weighted Rayleigh test** on the 12 angular bins, using each bin's centre angle and magnitude as the weight:
   * `C = Σ w cos θ`, `S = Σ w sin θ`, `R = √(C²+S²) / Σ w` (0 = perfectly even, 1 = all in one direction).
   * Correct R for grouped angles (Zar): `R × (d/2) / sin(d/2)` with d = 30° in radians (capped at 1).
   * Mean direction = `atan2(S, C)`.
   * `z = n R²` and Zar's (1999) approximation `p = exp(√(1 + 4n + 4(n² − (nR)²)) − (1 + 2n))`.
   * The sample size `n` is: the **number of place-cell peaks** for the `pooled` peak map; otherwise the **number of valid spatial bins** (for `kde_sig`, the number of significant bins).
6. **Significant** if `p < 0.05`.
7. Figure per map type: the map with the centre marked (+) and a polar histogram of the 12 magnitudes with an arrow showing the resultant vector (length ∝ R, red if significant).
   Excel: sheet `Rayleigh` (one row per kind × map type: centre, R, mean direction, z, p, significant) and sheet `AngularBins` (magnitudes of each 30° bin).

---

## 15. Step 14 – Compare the two nulls → `*_NullComparison.png`, `NullComparison_Significance.xlsx`

* `QuadrantFold_KDE_NullComparison.png` and `FineMap_KDE_NullComparison.png`: for each bin, colour-coded as significant against **both** nulls, **only** the uniform-field null, **only** the old occupancy null, or neither.
* `NullComparison_Significance.xlsx`: one row per analysis (`boundary_1D`, `quadrant_fold`, `fine_map`) × arena × map type (× side) giving the number of significant bins/positions vs each null, the overlap, how many are *depleted* (significantly below), number of clusters, the correlation between the two null curves and their total-variation difference (in percentage points).
  The table is also printed to the console.

---

## 16. Order in which `run_full_pipeline` runs everything

1. Create `OUTPUT_DIR/AllArenas`.
2. For each arena: `collect_arena_results` (steps 1–6) → handler + list of unit results.
3. `plot_fig_S1H`, `plot_field_only_mean_maps`, `plot_fig_1BD`, `export_excel`, `plot_uniform_field_null_maps` (steps 7–10).
4. `run_boundary_firing_analysis` (Step 10).
5. `run_quadrant_kde_analysis` (Step 11).
6. `run_fine_kde_analysis` (Step 12).
7. `run_rayleigh_analysis` (Step 13).
8. `plot_null_comparison` ×2 and `export_null_comparison` (Step 14).

Functions defined but **not** called by `__main__`: `debug_plot_circular_track_raw` (raw occupancy/rate plots for circular-track debugging) and `run_circular_track_pipeline_test` (runs the pipeline for the circular track alone).

---

## 17. Things worth knowing when you read the results

* **Settings currently in debug/test state:** `COVERAGE_FRACTION = 0.01` (its comments refer to 80 %) and `MIN_ZONE_COVERAGE_PCT = 0.1`, so the screens are very lenient; `ROTATE_SESSION_CCW_DEG = 0` (rotation off).
* **Only open-field data are reachable** with the current `ROOT_DIRECTORY`; the circular and linear arenas will be empty (I did not run the script, so I can't say how the plotting steps behave with empty arenas).
* **`min_occ_s` is 1 s** although a comment beside it says 0.5 s; the code value (1 s) is what is used.
* **Tracking speed filter** (`speed < 0.006`) is expressed in file units per time-stamp unit; the real jump removal is the 80 cm/s step in 3b.
* **Cell "qualification" values** (spikes, SIR, sparsity, shuffle) are reported but **not** used to exclude cells.
* **Field-only map weighting:** the 1D "field" curve uses pooled dwell time inside field bins as weights (the docstring calls this "field-density" weighting).
* **Colour scales:** with `'shared'`, colours in a figure row depend on all three arenas' data; a change in one arena can repaint the others even if their numbers are identical.
* **Rayleigh `n`:** for the map-based tests `n` counts spatial bins, which are spatially smoothed and therefore not independent, so p-values should be read with caution.
* **Randomness:** the per-cell shuffle is seeded per unit; the bootstrap CIs use `numpy.random.default_rng(0)`, so re-runs give identical results.
