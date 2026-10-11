# DEBUG PC ERRORS

## Speed modulation test skipping

Before a cell gets a speed_cell label of True or False, it passes through seven gates. Most of your "not tested" cells are almost certainly explained by gate 5 together with the labelling rule in gate 7.

Gates before the speed analysis runs (_run_job)
Gate 1: compute_metrics returns real data. If tracking is empty (L1902), every frame is immobile (L1932), or an exception is raised (L2717), the context comes back empty and the cell keeps 'not tested'.
Gate 2: Full session only (half is None). The first-half and second-half rows are never tested, so they are always 'not tested'. If you count across all rows including the half splits, that alone inflates the number.
Gate 3: At least one spike in spike_ts_all. That means a sorted spike (cluster ≠ 0) within MAX_GAP_US (50 ms) of a tracking frame (L1969).

Gates inside _compute_speed_modulation (tracking downsampled to 15 fps)

Gate 4: Basic data checks (each returns early with everything NaN/None):
    at least 3 frames and at least 1 spike (L861);
    at least 3 samples with 1 < smoothed speed < 90 cm/s and a finite rate (L921);
    speed and rate both vary, i.e. std > 0 (L926);
    at least 3 speed bins left after dropping bins with < 0.2% of samples (L1017).

Gate 5: Initial parametric significance screen (L1048–1049). This is the main one. The 1000-shuffle test runs only for a method whose parametric p is below 0.05:
    binned: the weighted-regression reg.pvalue;
    time-domain: the Pearson p_td.
The two methods are screened separately. A method that fails its screen never gets a shuffle, and its speed_modulated_shuffle or speed_modulated_td stays None.

Gate 6: Recording length (L1066). This requires n_intervals > 2 × 20 s × 15 fps, so more than 600 frames (about 40 s). It rarely matters.

Gate 7: At least 100 finite shuffle scores for the CI and p-value (L758). Otherwise the result stays None.

Final labelling (L2869–2876)
False if the binned and time-domain scores have opposite signs, or if either shuffle result is False.
True if both shuffle results are True.
'not tested' in every other case.
So "not tested" here does not mean the cell was skipped for lack of data. Two common cases end up there:

Both methods fail the p < 0.05 screen. Both flags are None, so the label is 'not tested', not False. These are really cells that failed the speed test, but they are labelled as if they were never tested.
Only one method passes its screen and its shuffle comes out significant. The other flag is None rather than False, so the label is again 'not tested' (unless the two scores have opposite signs).
The code itself notes that the parametric p is anti-conservative (L951). A cell that can't clear even that screen is very likely not speed-modulated.

To see the split in your output, cross-tabulate the speed_shuffle_ran and speed_shuffle_ran_td columns against speed_cell. A "not tested" cell with both columns False failed the parametric screen. One with both columns empty never got that far (gates 1–4).

If you'd like, I can change the labels so that failing the screen counts as False, keeping 'not tested' for cells with too little data.

## Occupancy weighted smoothing

With your current settings (GAUSSIAN_SIGMA_CM = 3, target_bin_cm = 4.0), the Gaussian kernel has a sigma of 0.75 bins. I computed the kernel the code builds, and its weights are:


0.048  0.116  0.048
0.116  0.283  0.116      ← the centre bin keeps 28% of the weight
0.048  0.116  0.048
94% of the total weight sits in the centre bin and its 8 immediate neighbours (a 12 × 12 cm patch).
Bins 2–3 away share the remaining ~6%. The kernel is cut off at 3 sigma, which makes it 7 × 7 bins.
<FIX: use 8x8 kernel instead of 7x7 kernel>
The full width at half maximum is about 7 cm (~1.8 bins).
In practice, each bin is blended with its immediate neighbours and almost nothing further away.

What the method does, in simple terms
The method smooths spike counts and time spent separately, then divides one by the other. It does not smooth the firing-rate map directly.


rate = smooth(spike counts) / smooth(time spent)
Function: _occ_weighted_smooth

The whole pipeline, step by step
Step A: Clean and smooth the trajectory in time (before any binning)
_smooth_tracking_position

Jump removal: a frame-to-frame step faster than 90 cm/s (POS_JUMP_THRESH_CMS) is treated as a tracker glitch. The check is repeated until no new glitches are found.
Glitch frames are refilled by linear interpolation between the good frames on either side.
x and y are each smoothed over time with a 1-D Gaussian of sigma = 5 frames (POS_SMOOTH_SIGMA_SMP). At 30 fps that is about 167 ms. This removes video jitter from the path.
Step B: Remove immobility and match spikes to positions
compute_metrics, steps 2c–4

Frames where speed is below 0.5 cm/s are dropped.
Each spike goes to the nearest video frame, but only if that frame is within 50 ms. Spikes that land on an immobile frame are dropped too.
Each frame counts as time spent, capped at 2 frames (66.7 ms). This stops tracking gaps from inflating occupancy.
Step C: Build two raw maps on a 4 cm grid
Lines 1982–1989

Spike map: the number of spikes fired in each bin.
Occupancy map: the seconds spent in each bin.
Valid mask: a bin is kept only if it was entered at least 2 separate times (min_visits = 2). The minimum-time filter use_min_occ_s is off.
Spikes are never smoothed in time. They are only counted per bin and then smoothed across space in Step D.

Step D: Occupancy-weighted spatial smoothing
Invalid bins are set to 0 in both maps, so they give nothing to their neighbours.
The same 7 × 7 Gaussian kernel is applied to the spike map and to the occupancy map. In each bin you get:
smoothed spikes = 0.283 × own spikes + 0.116 × each side neighbour's spikes + 0.048 × each diagonal neighbour's spikes + …
smoothed time = the same weighted sum, using seconds instead of spikes.
Rate = smoothed spikes ÷ smoothed time, computed only in valid bins with time > 0. Every other bin is 0.
The smoothed occupancy map is also returned and later used as the pi (time-fraction) weights in the SIR and sparsity formulas.
Edge handling: outside the arena is treated as zeros. Near a wall, the numerator and the denominator lose the same neighbours, so the ratio corrects itself automatically. No separate edge correction is needed.

Step E: Where this map is used
Because SI_RATEMAP_TYPE = 'occ_weighted_smooth', this map is used only for SIR and sparsity, and for their bootstrap shuffles. The other metrics use different maps:

Peak and mean firing rate: the raw, unsmoothed map.
Coherence, split-half stability and place-field detection: the ordinary Gaussian-smoothed rate map (_gaussian_smooth). It uses the same sigma but smooths the rates directly.
Not every bin enters the SIR and sparsity sums: a bin must also have at least 0.5 s of raw occupancy (SI_MIN_OCC_S).

Why weight by occupancy? A worked example
Take two neighbouring bins:

Bin A: the animal passed through for 0.1 s and the cell fired 1 spike. Raw rate = 10 Hz, a noisy estimate.
Bin B: the animal spent 5 s there and the cell fired 5 spikes. Raw rate = 1 Hz, a reliable estimate.
Here is the smoothed rate at bin A, using the centre weight 0.283 and the neighbour weight 0.116:

Method	Calculation	Result
Smoothing the rate map	(0.283×10 + 0.116×1) / (0.283 + 0.116)	7.4 Hz
Occupancy-weighted	(0.283×1 + 0.116×5) / (0.283×0.1 + 0.116×5)	1.4 Hz
When rates are smoothed directly, the brief visit to A still dominates and a "hot spot" appears from a single spike. With occupancy weighting, each bin counts in proportion to how much time the animal spent there. The well-sampled bin B therefore pulls A toward a believable value. This is the purpose described in the docstring: low-occupancy bins cannot dominate.

Parameter summary
Parameter	Value	What it controls
GAUSSIAN_SIGMA_CM	3 cm (0.75 bins)	Width of the spatial smoothing <Fix: change to 6>
target_bin_cm	4 cm	Bin size
Kernel cut-off	3 sigma → 7 × 7 bins	How far the kernel reaches <Fix: change to 4>
POS_SMOOTH_SIGMA_SMP	5 frames (~167 ms)	Smoothing of the path over time
POS_JUMP_THRESH_CMS	90 cm/s	Threshold for tracker glitches
IMMOBILITY_SPEED_CMS	0.5 cm/s	Speed below which frames are dropped
MAX_GAP_US	50 ms	Maximum spike-to-frame distance
min_visits	2	Visits needed for a bin to be valid
SI_MIN_OCC_S	0.5 s	Raw time needed for a bin to enter SIR and sparsity
Sigma is set in cm, not bins: the comment at line 302 cites Hockeimer et al., whose sigma is 1.5 bins × 2.1 cm = 3.15 cm. Your 3 cm matches that physical width. Measured in bins, though, your sigma is half theirs (0.75 vs 1.5) because your bins are about twice as large. That is why each bin keeps 28% of its own weight, which makes the smoothing light.

### Occupancy weighted smoothing Vs Adaptive binning

They are different methods. Your occupancy-weighted smoothing is what Grieves calls the histogram method with smoothing before division (his Fig 2). Adaptive binning (Yartsev & Ulanovsky) is a separate method, although the two are closely related.

How your three options map onto the Grieves paper
Your SI_RATEMAP_TYPE	Grieves's name
'raw'	Histogram, no smoothing
'occ_weighted_smooth'	Histogram, Gaussian-smoothed spike and dwell maps, then divided (Fig 2)
'gaussian_smooth'	Histogram, smoothing applied after division, ignoring empty bins (S1 Fig, nanconv)
none	Adaptive binning
What the two methods share
Both compute rate = spikes counted in a neighbourhood ÷ time spent in that neighbourhood. That is a ratio of sums, not an average of per-bin rates, so in both methods a briefly visited bin cannot dominate. Both are also computed by convolving the spike and dwell histograms, with zero padding at the edges.

Where they differ
Your method	Adaptive binning
Neighbourhood size	Fixed: Gaussian with sigma = 3 cm everywhere	Different for every bin: a circle grows until it holds at least t seconds of tracking (Yartsev used 1 s)
Weighting inside the neighbourhood	Gaussian: the centre bin counts 28%, neighbours less	Flat disk: every bin inside the circle counts equally, nothing outside counts
Well-sampled bins	Still blended with their neighbours	Circle stays at one bin, so no smoothing
Poorly sampled bins	Smoothed by the same small amount	Circle grows until there is enough data, so heavy smoothing
Rule for when to stop	None	Minimum time inside the circle, t
In short, adaptive binning is your method with the Gaussian replaced by a flat disk whose radius is chosen separately for each bin from local occupancy. It smooths only where data is sparse, so it adds the most smoothing where sampling noise is worst.

What the Grieves paper says about the two
Smoothing before vs after division: the two give similar accuracy, except that smoothing before division is less accurate at very small bin sizes (S1 Fig). So your occ_weighted_smooth and gaussian_smooth should produce similar rate maps. The SI difference in my earlier simulation came mostly from which occupancy weights (pi) each option uses.
Adaptive binning had among the lowest map errors and the best place-field detection (Fig 10A, C). It was also the hardest to break with bad parameter choices: any bin under 50 mm combined with t > 0.5 s worked well (Fig 7F). The paper also warns that it does not natively produce a dwell map. For Skaggs SI you would still need occupancy weights from the plain histogram.
Your parameters vs his Table 1: the histogram minimum-error equation is smoothing = 16.2 + 0.119r − 0.218d (mm), where r is field radius and d is session length in minutes. I don't know your field sizes or session lengths, so I assumed a field radius of about 150 mm and a 20-minute session. That gives sigma ≈ 30 mm, which matches your GAUSSIAN_SIGMA_CM = 3. Your 40 mm bins are somewhat larger than the 20–30 mm he and most published studies use.
Adaptive binning could be added as a fourth SI_RATEMAP_TYPE option in your script, then compared against the current three in the same ground-truth simulation.

## Effect of smoothing procedure on accurancy of spatial information score 

The raw map is the least accurate of the three. To check this, I ran a simulation using your exact kernel: a 3 cm sigma, 4 cm bins, an 80 cm arena and a 20-minute session. Spikes were Poisson, and occupancy was deliberately uneven across bins. Because the true firing-rate map was known, I could compute the true Skaggs SI and compare each method against it. Each condition was repeated 300 times.

Result 1: cells with no spatial tuning (true SI = 0)
Cell	Raw	Occupancy-weighted	Gaussian-smoothed rate
0.2 Hz (~240 spikes)	0.98 ± 0.08	0.18 ± 0.03	0.32 ± 0.12
0.5 Hz (~600 spikes)	0.44 ± 0.04	0.07 ± 0.01	0.15 ± 0.06
2.0 Hz (~2400 spikes)	0.11 ± 0.01	0.02 ± 0.003	0.05 ± 0.04
Result 2: real place fields (peak 8 Hz)
Field width	True SI	Raw	Occupancy-weighted	Gaussian-smoothed rate
sigma 4 cm	1.71	2.40 (+40%)	1.54 (−10%)	1.63 (−5%)
sigma 8 cm	1.73	2.01 (+16%)	1.62 (−6%)	1.69 (−2%)
sigma 12 cm	1.14	1.30 (+14%)	1.06 (−7%)	1.13 (−1%)
What this means
The raw map is always inflated, and the inflation depends on spike count. A completely untuned 0.2 Hz cell scored 0.98 bits/spike, about twice your PLACE_CELL_MIN_SIR = 0.5 cutoff. The reason is that each bin's rate is estimated from very few spikes. That random bin-to-bin noise looks like spatial contrast, and the Skaggs formula rewards contrast. The false SI is roughly (number of bins) ÷ (2 × ln 2 × number of spikes), so low-firing cells get the most. This is why Skaggs et al. did not use raw bins either. The paper you attached says they built rate maps "using an adaptive binning technique designed to minimize error".

Occupancy-weighted smoothing has the lowest noise floor and the lowest variance. For untuned cells it is 4–5× less inflated than the raw map and about 2× less than the Gaussian-smoothed rate map. It also varies far less from run to run. For real place cells it underestimates by about 6–10%, because smoothing slightly blurs the field peak.

Gaussian rate smoothing came closest for real fields, but partly by luck. It also blurs the peak, which pushes SI down. Separately, bins the animal visited only briefly still carry full weight, which pushes SI up. Here the two errors happen to cancel. The second error also makes it 4× more variable than occupancy-weighted for untuned cells (±0.12 vs ±0.03). In addition, it pairs a smoothed rate map with raw occupancy weights, so the Skaggs mean rate no longer equals the true mean rate. The occupancy-weighted method keeps that consistent.

How this fits with Zhang et al.
Their Fig. S2 shows that smoothing lowers SI. The simulation suggests why: most of the extra SI in the unsmoothed maps is sampling noise rather than real information.
Their Fig. S3 shows that permutation testing is robust to bin size and smoothing. This also applies to you: your bootstrap shuffles use the same SI method as the real data (line 1270), so the shuffles cancel much of the noise inflation. The method choice therefore matters most for two things:
the fixed 0.5 SI threshold, and
comparing SI values between cells, animals or conditions that differ in firing rate or sampling.
Recommendation
Keep SI_RATEMAP_TYPE = 'occ_weighted_smooth'. It trades a small, predictable underestimate (~5–10%) for a much lower and more stable noise floor. That makes your fixed threshold meaningful and keeps low-firing cells from scoring high just because they fired few spikes.

Simulation caveats: firing was Poisson and occupancy was independent from bin to bin. Real spike trains are burstier and real trajectories are more uneven, and both would likely increase the raw map's inflation further. The script is si_bias_sim.py.

For an answer specific to your mole-rat data, I can rerun it on your real trajectories and occupancy maps, adding synthetic cells with known fields.

### Grieves smoothing method comparision

The paper doesn't answer this directly. Grieves never computes spatial information. He scores each method by how closely its rate map matches the true firing map (MISE on maps normalised to sum to 1), plus place-field detection, computation time and empty bins. So any ranking for SI is an inference from those results, not something he measured.

That inference is fairly reasonable, though. Skaggs SI depends only on the shape of the rate map and the occupancy weights: scaling a rate map up or down doesn't change it. Normalised MISE also measures shape, so a method with lower MISE should generally give an SI closer to the true value.

Each method, judged for SI
Histogram with Gaussian smoothing (what you have now)

It produces the spike, dwell and rate maps directly. The SI weights (pi) and the rate map therefore come from the same data and the same smoothing.
At its minimum-error settings it was among the most accurate methods of all (Fig 10A). It was also nearly the fastest.
The paper explicitly calls it "very well suited to shuffle-based analyses". That matters for your 1000 shuffles per cell.
The paper's overall recommendation is this method "in the majority of cases".
Adaptive binning

With balanced settings it had the lowest error and the best place-field detection (Fig 10A, C). It was also the most forgiving of parameter choices (Fig 7F), and it handles sparse sampling best.
Drawback stated in the paper: it does not natively produce a dwell map, so the SI weights would have to come from a separate histogram. That breaks the consistency between the rate map and pi.
Advantage for shuffles (my inference, not from the paper): each bin's radius depends only on the trajectory, not on spikes. A circular-shift shuffle leaves the trajectory unchanged, so the radii can be computed once per session and reused for every shuffle.
Adaptive smoothing (Skaggs & McNaughton)

This is historically the closest to what Skaggs used. The 1993 paper says its maps came from "an adaptive binning technique designed to minimize error (Skaggs and McNaughton, submitted)", which is this method.
Its accuracy was good: similar to adaptive binning and better than the histogram at balanced settings (Fig 10A).
Problem for SI (my reasoning, not the paper's): the size of each neighbourhood depends on the spike count. In-field bins and high-rate cells get less smoothing; out-of-field bins and low-rate cells get more. The amount of smoothing therefore depends on the very signal SI is measuring. That makes SI harder to compare between cells or conditions that differ in firing rate, which is what you would compare across animals. The neighbourhoods also change in every shuffle, and it has the same missing-dwell-map drawback as adaptive binning.
ASH (averaged shifted histogram)

It was the fastest, with medium accuracy and a few more empty bins (Fig 10A, D).
It offers no real advantage over a smoothed histogram for a single summary number like SI. Its main benefit is higher-resolution maps.
Bottom line
Method	Map accuracy (paper)	Fit for SI and shuffles
Smoothed histogram	High at minimum-error settings	Best overall: consistent pi, fast, the paper's default
Adaptive binning	Highest at balanced settings, most robust	Strong alternative, especially for sparse sampling; needs a separate dwell map
Adaptive smoothing	Good	Spike-dependent smoothing makes SI comparisons across firing rates less clean
ASH	Medium	No advantage for SI
Recommendation: keep the smoothed histogram (occ_weighted_smooth) as your main method. Your sigma of 3 cm already sits near the paper's minimum-error value. The one change worth considering is a smaller bin size: 2–3 cm instead of 4 cm, which is where Grieves's minimum-error solutions and most published studies sit.

If you worry that low spike counts or uneven sampling in the mole rats are hurting SI, adaptive binning is the one to test. It could be added to the simulation script and scored against true SI, so the choice rests on measured bias instead of map MISE.

### Optimizing alpha value in Skagg's Adaptive Bin approach

with 4 cm bins, an alpha between 0.0003 and 0.001 is the best compromise in my simulation. The current 0.0001 smooths too much: it underestimates the SI of real place fields by 13–19%. Switching to 2 cm bins doesn't help. Once alpha is rescaled for the smaller bins, the results are essentially the same as at 4 cm, and the run is about 6× slower.

What I ran
I used your _adaptive_binned_ratemap code on simulated data where the true SI is known. Each of 30 repeats was a random-walk trajectory: 20 minutes at 30 fps in an 80 cm arena, about 8 cm/s. On each I simulated four cells: two untuned cells (0.3 and 1 Hz, true SI = 0) and two place cells (6 Hz peak, field sd 6 cm and 10 cm). I scored each setting by bias and RMSE against the true SI. Script: alpha_sweep.py.

4 cm bins: bias (estimated SI − true SI)

alpha	Untuned 0.3 Hz	Untuned 1 Hz	Field sd 6 cm (true 1.62)	Field sd 10 cm (true 1.33)
0.00003	+0.02	+0.01	−0.48	−0.31
0.0001 (current)	+0.04	+0.02	−0.30	−0.17
0.0003	+0.07	+0.03	−0.21	−0.10
0.001	+0.11	+0.04	−0.11	−0.05
0.003	+0.16	+0.05	0.00	0.00
0.01	+0.18	+0.05	+0.03	+0.01
Your current method (occupancy-weighted)	+0.12	+0.04	−0.09	−0.07
No single alpha is unbiased for every cell. Lower alpha means more smoothing. That lowers the false SI of untuned cells but also flattens real fields; higher alpha does the reverse. Alpha only chooses where on that trade-off you sit. Around 0.001 the adaptive method behaves almost exactly like your current occupancy-weighted method.

For classification, the choice barely matters. Untuned cells score at most about 0.18 at any alpha, well under your 0.5 threshold, and real fields stay well above it. Alpha matters when you compare SI values, for example between animals or conditions. In that case, fix one alpha in advance and use it for every dataset.

The Grieves paper agrees roughly. Converted to your units, his minimum-error formula gives alpha ≈ 0.005 for 4 cm bins. That matches the simulation, where field accuracy is best around 0.003. His score measures map shape only and ignores the false SI of untuned cells, so it leans toward less smoothing.

Will 2 cm bins help?
No, not for SI. The disc radius is counted in bins, so the same alpha means something different at 2 cm. The equivalent alpha is half the 4 cm value. With that adjustment the results match: 2 cm at alpha 0.0003 gives +0.11, +0.04, −0.11 and −0.04, almost identical to 4 cm at 0.001. At 2 cm with alpha left unchanged, the false SI of untuned cells rises sharply (up to +0.56).

If you switch to 2 cm anyway:

Halve ADAPTIVE_ALPHA_BASE, and set ADAPTIVE_MAX_RSQ = 800 so the disc can still reach the same 56 cm.
target_bin_cm affects the whole pipeline: coherence, MIN_FIELD_SIZE_BINS, the visit criterion and split-half stability. Per-bin occupancy also drops about 4×, so more bins fail the 0.5 s SI_MIN_OCC_S cut. In the simulation that had little effect, but slow mole-rat trajectories could differ.
Each map takes about 6× longer: roughly 35 s per cell for 1000 shuffles.
How to choose alpha for your own data
The simulation used made-up trajectories, firing rates and field sizes. The best alpha depends on all three (Grieves Table 1), so tune it on your own data:

Simulation on your real trajectories (best for SI). Replace walk() in the script with your actual tracking files, set field sizes and rates to match your cells, and pick the alpha with the lowest error. You can weight untuned and tuned cells by how much you care about each.
Cross-validation on real data (no ground truth needed). Split each session into alternating 1-minute blocks. Build the adaptive map from the odd blocks, and score how well it predicts the even blocks' spike counts. Sum that score over cells and pick the alpha that predicts best. This tunes map accuracy rather than SI bias directly, but it uses only your recordings.

### Skagg's adaptive bin Vs Yartsev's adaptive bin approach

Both methods work the same way up to the last step. They differ only in the rule that decides how big each bin's circle gets.

What they share (lines 517–546): for every bin centre, the code tries a set of growing radii k, from binsize up to maxdist (640 mm by default) in 32 steps. For each radius it counts:

n: position samples inside the circle (occupancy)
s: spikes inside the circle
It then picks the smallest radius that meets a criterion and sets the rate to s / (n/srate) inside that circle. If no radius meets it, it falls back to the largest radius, k(end). radmap stores the radius it chose.

Skaggs ('adaptive', line 552): the circle grows until

r ≥ a / (n · √s)

r is the radius in bin units (k ./ binsize).
n is the occupancy as a raw sample count, not seconds.
s is the spike count.
a is a scaling constant. The code reuses rmset.ssigma for it (default 80).
The rule needs both occupancy and spikes:

Strong firing (large s) meets the rule at a small radius, so you get little smoothing and a sharp field.
Weak firing or sparse sampling pushes the radius larger, so you get more smoothing.
Silent areas: when s = 0, the right-hand side is infinite and the rule can never pass. The circle keeps growing until it reaches spikes, often from a nearby field, or until maxdist. So silent bins near a field get a low but non-zero rate from a large circle.
Units: n is in samples and r is in bins, so the effective strength of a changes with srate and binsize.
Yartsev ('yadaptive', line 572): the circle grows until it holds at least 1 second of occupancy (n >= 1 after converting to seconds).

Spikes play no part in choosing the radius, which depends only on where the animal went.
Well-visited areas get small radii; rarely visited areas get large ones.
The radius map is the same for every cell recorded in that session.
Silent areas stay at exactly 0 Hz with a small radius, so the method doesn't borrow spikes from neighbours.
The 1 s threshold is hard-coded. It doesn't use rmset.mindwell.
Side by side:

Skaggs 'adaptive'	Yartsev 'yadaptive'
Radius depends on	occupancy and spikes	occupancy only
Rule	r ≥ a/(n·√s)	occupancy ≥ 1 s
Same radius map for every cell?	No	Yes
Bins with no spikes	grow until they reach spikes or maxdist	stay small, rate = 0
Tuning parameter	a (reuses ssigma)	none (1 s hard-coded)
Why this matters for your shuffle/null analyses: with Yartsev, the radius map stays fixed when you shuffle spikes, because it depends only on the trajectory. Any difference between real and shuffled maps then comes from the spikes alone. With Skaggs, shuffling the spikes also changes the smoothing radius for every bin. That makes the null distribution partly reflect changes in smoothing rather than spike positions alone. The kernel-accelerated versions ('kadaptive' / 'kyadaptive', line 590) apply these same two rules on a fine pre-binned grid.
