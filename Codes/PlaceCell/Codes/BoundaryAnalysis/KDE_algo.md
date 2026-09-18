# KDE analysis for boundary preference quantification

One important thing to flag up front: despite being named "KDE" throughout (function names, file names, constants like N_KDE_BOOTSTRAP), the current implementation is not a literal kernel density estimate anymore. The code's own comment at MeanRateMap_QuadrantAnalysis_v10.py:1161-1166 says it was rewritten from "the old weighted-KDE-of-raw-samples design" into a bin-average + Gaussian-smooth design, but the "KDE" name stuck around. I'll explain what it actually does now.

The big picture
For each arena, the code asks: "Does firing rate depend on distance from the boundary/center, more than you'd expect just from how much time the animal spent at each distance?" It builds two curves along a 1D position axis and compares them:

Observed curve — average firing vs. distance
Null curve — average occupancy (time spent) vs. distance, dressed up to look like the observed curve
If firing is genuinely biased toward certain positions (not just because the animal happened to linger there), the observed curve should rise above the null curve at those positions.

Step 1 — Collapse 2D space to a 1D "distance" axis
Each arena has its own geometric axis, computed in _arena_position_axis (MeanRateMap_QuadrantAnalysis_v10.py:1276-1288):

Open field: distance from the arena's center (0–30 cm)
Linear track: distance from the track's midpoint (folds both ends onto one axis)
Circular track: arc-length position around the ring (wraps around, since it's a loop), split into inner-ring/outer-ring
Every spatial bin in the original 2D rate map gets assigned one of these 1D distance values.

Step 2 — Turn a 2D map into a 1D curve (bin-averaging)
This is _distance_bin_average (MeanRateMap_QuadrantAnalysis_v10.py:1199-1220). All 2D spatial bins that share roughly the same distance value get grouped together (e.g., "all bins 8–10 cm from center"), and their values are averaged (weighted mean if a weight is supplied, plain mean otherwise). This gives one number per distance-bin — essentially a 1D histogram-of-averages, not a smooth kernel density yet.

Step 3 — Gaussian smoothing (this is the only real "kernel" left)
_gaussian_smooth_1d_nan (MeanRateMap_QuadrantAnalysis_v10.py:1175-1188) smooths that 1D curve with a Gaussian kernel. It uses the classic NaN-safe trick:

Smooth value × validity-mask with a Gaussian filter
Separately smooth just the validity-mask
Divide the two
This prevents empty/invalid bins from silently dragging the curve toward zero — the smoothing only pulls in real neighboring data, appropriately weighted by how much valid data is nearby. For the circular track, wrap=True makes the smoothing treat the axis as a closed loop (so position near 0° blends with position near 360°).

Step 4 — Normalize to % of total
_normalize_pct (MeanRateMap_QuadrantAnalysis_v10.py:1191-1196) rescales the curve so it sums to 100. This is necessary because firing-rate curves and occupancy (seconds) curves are in completely different units — you can't compare "field-index" to "seconds" directly, but you can compare their shapes once both are expressed as "% of the curve's own total."

The null occupancy curve — the part you asked about
This is _build_null_curve (MeanRateMap_QuadrantAnalysis_v10.py:1317-1336), and it mirrors _build_observed_curve step-for-step but swaps in occupancy (dwell time) instead of firing.

The intuition: if the animal spends way more time near the wall than in the center, then even a cell with no real boundary preference will look like it "fires more near the wall" just from sample-size effects (more data points near the wall = more spikes counted there, and noisier/possibly-inflated rate estimates in undersampled bins). The null curve captures exactly that sampling bias, so it can be subtracted out.

Concretely, for each of the three map types the pairing is:

Observed (_build_observed_curve)	Null (_build_null_curve)
'overall': mean field-index per position (unweighted mean across cells' whole maps)	pooled occupancy per position, same unweighted-mean aggregation
'field': mean field-index restricted to cells' own place-field bins, occupancy-weighted	pooled occupancy restricted to those same field bins, same aggregation
'peak': count of place-cell peak bins falling at each position (like a histogram)	pooled occupancy summed (not averaged) per position — a "mass," matching the count-like nature of peaks
The raw occupancy numbers come from _pooled_occupancy (MeanRateMap_QuadrantAnalysis_v10.py:1251-1262), which just sums each place cell's occ_map (seconds of dwell time per spatial bin, computed way upstream in process_unit at MeanRateMap_QuadrantAnalysis_v10.py:858-860 — literally np.add.at(occ_map, bin_idx, dt_frames), i.e. how many seconds of tracking data landed in each bin) across all place cells, optionally restricted to only their field bins.

That pooled occupancy map then goes through the exact same pipeline as the observed curve: bin-average onto the 1D distance axis → Gaussian-smooth → normalize to %. So the null curve is "what the distance-tuning curve would look like if firing rate were flat everywhere and all you were measuring was how unevenly the animal sampled space."

Step 5 — Deciding significance
Bootstrap the observed curve: resample place cells with replacement 1000 times (N_KDE_BOOTSTRAP), rebuild the observed curve each time → gives a distribution of curves → take the 2.5th/50th/97.5th percentiles at each position (MeanRateMap_QuadrantAnalysis_v10.py:1390-1398). The null curve itself is not bootstrapped — it's built once from the full real population.
Flag positions: a position counts as significant if even the worst-case (2.5th percentile / lower bound) of the observed bootstrap distribution still exceeds the null curve (MeanRateMap_QuadrantAnalysis_v10.py:1343: sig = lo > null_curve).
Group into peaks: contiguous runs of significant positions (at least 3 bins long) are reported as boundary-preference peaks, with the local maximum as the reported peak location (MeanRateMap_QuadrantAnalysis_v10.py:1339-1358).
So in plain terms: the null curve isn't a statistical kernel density estimate of anything — it's "the occupancy map put through the identical averaging-and-smoothing recipe as the firing curve," so that any leftover elevation of the real curve above it reflects a genuine firing bias, not just where the animal happened to walk more.