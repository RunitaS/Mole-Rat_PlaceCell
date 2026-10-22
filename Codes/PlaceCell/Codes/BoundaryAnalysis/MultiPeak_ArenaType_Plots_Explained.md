# What the ArenaType figures show

**Where they come from.** The figures are made by `run_stats()` in
[`MultiPeak_StatsComparison.py`](../MultiPeak_StatsComparison.py). It is called
at the end of the multi-peak KDE script in this folder
(`MeanRateMap_QuadrantAnalysis_v27_5_M2_BinShuffle_AdptBin_Session_Rayleigh.py`)
when `RUN_STATS = True`. It borrows its tests and plotting functions from
`ArenaSessionType_StatsComparison_v8`. The only copy of that module found was
[`OldVers/ArenaSessionType_StatsComparison_v8.py`](../OldVers/ArenaSessionType_StatsComparison_v8.py),
so that is the version described here.

The figures are saved in `<output_dir>\Stats\ArenaType\` and its `Histograms\`
subfolder.

**What "multi-peak" means here.** It means one isolated place field has two or
more peaks inside it. It does not mean a cell has several separate fields; the
number of fields per cell is analysed in the other stats script. Every metric
below describes the internal shape of a field.

---

## What goes into every figure

**1. Fields that are dropped first** (`load_tables` in `MultiPeak_StatsComparison.py`):

- **No KDE was fitted** (`kde_note` is not empty). The field had fewer than
  `MIN_KDE_POINTS` = 3 bins with non-zero rate, or all such bins were at one spot.
- **`ratemap_check` is False.** The rebuilt ratemap did not reproduce the Excel
  `peak_fr_hz`.

The per-cell numbers are then recomputed from the fields that remain. A cell
with 3 fields, one of them dropped, counts as a 2-field cell.

**2. Groups.** The arena (Circle, Linear or Open) is read from the session
path. All session types within an arena are pooled. Colours: magenta for
Circle, cyan for Linear, blue for Open.

**3. How the field metrics are computed** in the main script:

- Each field gets a weighted Gaussian KDE. It is fitted on the field's bins,
  weighted by the rate in the `KDE_RATEMAP_TYPE` map (Skaggs adaptive-binned).
- The kernel width is `KDE_BW_METHOD` × that field's own weighted std.
  Smoothing therefore scales with field size.
- Local maxima of the KDE are found inside the field.
- A maximum counts as a peak if both hold:
  - its height is ≥ 75 % of the field's highest peak (`PEAK_REL_HEIGHT_FRAC`);
  - the KDE dips at all on the way to any higher peak (`MIN_TROUGH_DIP_FRAC` = 0).
- With ≥ 2 counted peaks, the field is split into sub-fields along the valley
  floors (a watershed split).
- **Circle fields** are analysed in 1-D along the track angle. **Open and
  Linear fields** are analysed on the 2-D grid.

**4. Text under each plot.** For each arena: n, mean ± SD, and median [Q1, Q3].
Then the omnibus test and the pairwise tests, with Holm-corrected p-values. A
last line notes how many rows come from repeated cells.

**5. Brackets with stars** connect pairs of arenas whose Holm-corrected p is
< 0.05: `*` < 0.05, `**` < 0.01, `***` < 0.001.

**6. The tests:**

- **Box plots and histograms:** Kruskal-Wallis across the three arenas, then
  pairwise Mann-Whitney U tests. This is `CONTINUOUS_TEST = 'KW'`.
- **Bar plots:** a chi-square test on an arena × (yes/no) table, then pairwise
  2×2 chi-square tests with Yates correction. Both are Holm-corrected.

**7. Box-plot layout.** The box spans Q1 to Q3, and the black line inside it is
the median. Whiskers reach 1.5 × the box height. Every data point is drawn as a
jittered dot, so outliers are shown as dots rather than hidden.

---

## Per-cell figures (one dot = one place cell in one session)

### `Cell_n_multi_peak_fields.png`: multi-peak fields per cell

- **Calculation:** the number of the cell's kept fields with `multi_peak == True`.
- **Meaning:** how many fields with inner sub-peaks a cell has.
- **Caveats:**
  - This depends directly on how many fields the cell has. An arena where cells
    have more fields will score higher even if each field is no more likely to
    be multi-peak.
  - Most values are 0 or 1, so the boxes are often squashed to a line. The
    histogram or `Cell_pct_…` is easier to read.

### `Cell_pct_multi_peak_fields.png`: % of each cell's fields that are multi-peak

- **Calculation:** 100 × `n_multi_peak_fields` / `n_fields`.
- **Meaning:** the same question as above, corrected for the number of fields.
  It answers "does this cell tend to have structured fields?"
- **Caveat:** values are discrete (0, 33, 50, 100 …). A one-field cell can only
  score 0 or 100.

### `Cell_max_n_peaks.png`: most peaks in any one field of the cell

- **Calculation:** the largest `n_peaks` among the cell's fields.
- **Meaning:** 1 means all the cell's fields have a single peak. 2 or more means
  at least one field splits, and a 3 means one field has three sub-peaks. It
  measures the cell's most fragmented field.

### `Categorical_has_multi_peak_field.png`: % of place cells with ≥ 1 multi-peak field (bar plot)

- **Calculation:** for each arena, the share of cells with
  `n_multi_peak_fields > 0`.
- **Meaning:** how common within-field substructure is across the place-cell
  population of each arena.

---

## Per-field figures (one dot = one place field)

### `Field_n_peaks.png`: number of peaks per field

- **Calculation:** the number of counted KDE peaks in the field, as defined in
  point 3 of the first section.
- **Meaning:** 1 is a single peak. 2 or more means the field is made of
  sub-fields: one firing region containing several local hotspots.
- **Caveat:** a cell's fields are not independent of each other, but
  Kruskal-Wallis treats every row as independent.

### `Categorical_multi_peak.png`: % of place fields that are multi-peak (bar plot)

- **Calculation:** for each arena, the share of fields with `n_peaks ≥ 2`.
- **Meaning:** the main prevalence measure, counted per field.
- ⚠ **The y-axis label is wrong.** It reads "% of cells" because
  `plot_categorical` hard-codes that label (`ArenaSessionType_StatsComparison_v8.py`,
  line 964). For this figure it should say "% of place fields".

---

## Multi-peak fields only (one dot = one field with ≥ 2 peaks)

These three describe what the substructure looks like. Their sample sizes are
much smaller, so an arena with few multi-peak fields can end up with
"insufficient data".

### `MultiField_shallowest_trough_rel.png`: trough height ÷ the lower peak

- **Calculation:**
  1. Take each pair of neighbouring sub-fields, a and b.
  2. Find the saddle: the highest KDE value at which you can cross from a into b
     along their shared border.
  3. Divide the saddle by the lower of the two peaks:
     `trough_rel = saddle / min(peak_a, peak_b)`.
  4. The field's value is the **largest** of these ratios, meaning its weakest
     split.
- **Meaning:** how deep the valley between peaks is. This tells you how
  convincingly the field splits.
  - Near **0**: the rate drops almost to zero between peaks. The two peaks
    behave like two separate fields that happen to touch.
  - Near **1**: only a slight dip. The second peak is a shoulder on one field,
    which could be real or could be noise.
- **Caveat:** with `MIN_TROUGH_DIP_FRAC` = 0, any dip, however small, is enough.
  Values close to 1 are therefore allowed and are the least convincing splits.

### `MultiField_lowest_counted_peak_rel.png`: lowest peak ÷ highest peak

- **Calculation:** min(counted peak heights) / max(counted peak heights), using
  the KDE heights in Hz.
- **Meaning:** how balanced the peaks are.
  - Near **1**: peaks of equal strength (two co-dominant hotspots).
  - Near **0.75**: one dominant peak plus a weaker satellite.
- ⚠ **The range is fixed by the threshold.** A peak only counts if it reaches
  75 % of the highest, so this value can never fall below 0.75. Every arena's
  values sit between 0.75 and 1, which squeezes any differences. Changing
  `PEAK_REL_HEIGHT_FRAC` moves the floor.

### `MultiField_peak_separation_cm.png`: distance between neighbouring peaks

- **Calculation:**
  - For each pair of neighbouring sub-fields, measure the distance between
    their KDE peak bins and multiply by the bin size (`target_bin_cm`, 2 cm).
    - Open and Linear: straight-line (Euclidean) distance on the grid.
    - Circle: distance along the track.
  - The field's value is the **mean** over its neighbouring pairs.
- **Meaning:** the spatial scale of the substructure, i.e. how far apart the
  hotspots inside a field are.
- **Caveats:**
  - The smallest possible value is about 4 cm, because two maxima cannot sit in
    adjacent bins.
  - The largest possible value is limited by field size. Arenas with bigger
    fields will show bigger separations for that reason alone.

---

## `Histograms/` (7 figures)

`Hist_Cell_n_multi_peak_fields`, `Hist_Cell_pct_multi_peak_fields`,
`Hist_Cell_max_n_peaks`, `Hist_Field_n_peaks`,
`Hist_MultiField_shallowest_trough_rel`,
`Hist_MultiField_lowest_counted_peak_rel`, `Hist_MultiField_peak_separation_cm`

These show the same data as the 7 box plots above, drawn as distributions:

- One panel per arena.
- All panels share the same 15 bins, set over the pooled range, so they can be
  compared directly.
- A dashed line marks each arena's median, labelled `med=`.
- The test text underneath is the same as on the box plots.

They are most useful for the integer metrics (`n_peaks`, `max_n_peaks`,
`n_multi_peak_fields`), where the box plots are uninformative. With 15 bins over
a range like 1–3, you get thin bars at the integers and empty gaps between them.

---

## Caveats when comparing arenas

1. **Repeated cells.** Each row is one unit in one session, and the same cell
   recorded in several sessions appears several times. With `'KW'`, every row
   is treated as independent, so the p-values are anti-conservative. The note
   under each plot gives the amount of repetition. Setting
   `CONTINUOUS_TEST = 'LMM'` in v8 fits a mixed model with animal, day and cell
   terms, which accounts for this. The chi-square tests have the same
   independence assumption.
2. **Circle uses a different method.** Circle fields are analysed in 1-D and
   Open/Linear fields in 2-D. Part of any Circle-vs-other difference in how
   often fields are multi-peak can come from this alone.
3. **Field size and sampling.** The KDE width scales with each field, but other
   steps work in fixed cm:
   - the 2 cm bins;
   - the Skaggs adaptive discs used for the KDE weights, which grow larger,
     i.e. smooth more, where spikes or occupancy are sparse.

   Arenas with larger fields or sparser coverage can differ in peak counts for
   those reasons. A useful control is to compare `n_peaks` against `area_cm2`
   for each arena.
4. **Bandwidth sensitivity.** A smaller `KDE_BW_METHOD` gives more peaks (e.g.
   the `Smooth025` vs `Smooth05` runs). An arena difference that shows up in
   both runs is more trustworthy than one that appears in only one.
