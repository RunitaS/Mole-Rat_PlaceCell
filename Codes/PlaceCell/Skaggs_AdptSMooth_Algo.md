# Skaggs adaptive binning Algo
The code is in four functions: _adaptive_disc_geometry, _adaptive_disc_sums, _skaggs_adaptive_core and _adaptive_rate_at.

**The idea**
A fixed Gaussian kernel smooths every part of the map by the same amount. Adaptive binning gives each bin its own circular "disc". The disc keeps growing until it holds enough data for a reliable rate estimate. Skaggs & McNaughton (1998) defined "enough data" as:

$$r ;\ge; \frac{\alpha}{n\sqrt{s}} \quad\Longleftrightarrow\quad r \cdot n \cdot \sqrt{s} ;\ge; \alpha$$

<r is the disc radius.
<n is the occupancy inside the disc.
<s is the number of spikes inside the disc.
The rate for the bin is then spikes in the disc / occupancy in the disc. This is pooled counts, not an average of the per-bin rates.

So discs stay small where there is plenty of occupancy and plenty of spikes (in-field, well sampled) and grow large where data are sparse (out-of-field, rarely visited).


0. Build the inputs (lines 2264–2271)

Positions go into target_bin_cm = 2 cm bins.
occ_map holds the seconds spent per bin. Each frame adds its dt, capped at 2/fps so tracking gaps don't inflate occupancy.
spike_map holds the spike count per bin.
Immobile frames (< 0.5 cm/s) and their spikes have already been removed.
valid_mask is the set of bins with occupancy > 0 and at least min_visits = 2 visits. use_min_occ_s is currently off.

1. Mask invalid bins. Invalid bins have their spikes and occupancy set to 0. They neither add to any disc nor get a disc of their own. The map is zero-padded so discs can extend past the edge.

2. List candidate discs. All offsets (dx, dy) with dx² + dy² ≤ ADAPTIVE_MAX_RSQ − 1 = 199 are sorted by distance. Candidate discs are r² = 0, 1, 2, … 199 in bin units. An r² that is not a sum of two squares (3, 6, 7 …) gives the same disc as the r² below it. disc_r_bins stores the true radius, which is the square root of the largest enclosed d².

3. Sum every disc for every bin at once. A cumulative sum over the sorted offsets gives the spike and occupancy totals of every candidate disc around every bin. This vectorised version gives the same result as Grieves' MATLAB loop, which grows r² one step at a time.

4. Convert the radius to cm (line 672)
r_cm = (disc_r_bins + 0.5) * bin_cm. The extra half bin follows Grieves' 'kadaptive', so even the single-bin disc has a nonzero radius. With 2 cm bins:

r²	bins in disc	r_cm
0	1	1.0
1	5	3.0
2	9	3.83
4	13	5.0
5	21	5.47
8	25	6.66
197 (max)	~620	~29

5. Apply the criterion (line 673)
r_cm * occ_disc * sqrt(spk_disc) >= SKAGGS_ALPHA. The first (smallest) disc that passes is chosen. If none passes, the largest disc is used.

6. Compute the rate (_adaptive_rate_at)

rate = spk_disc / occ_disc at the chosen disc.
If the chosen disc holds less than ADAPTIVE_MIN_OCC_S = 0.5 s, the bin gets rate 0 and SI weight 0.

7. Compute SI and sparsity

The weights pᵢ are the raw occupancy, not the disc occupancy. Discs overlap, so disc occupancy would over-weight sparsely sampled bins.
Only bins with raw occupancy ≥ SI_MIN_OCC_S = 0.5 s enter the sums.
Every bootstrap shuffle goes through the same steps, so the shuffled spike trains get their own adaptive discs.
Worked example (2 cm bins, 0.5 s occupancy per bin, α = 50)
In-field bin at 10 Hz. At r² = 1 the disc holds 2.5 s and 25 spikes: 3 · 2.5 · 5 = 37.5, which fails. At r² = 2 it holds 4.5 s and 45 spikes: 3.83 · 4.5 · 6.7 = 116, which passes. Radius ≈ 3.8 cm.
Out-of-field bin at 0.5 Hz. At r² = 4 the disc holds 6.5 s and 3.25 spikes: 5 · 6.5 · 1.8 = 58.6, which passes. Radius ≈ 5 cm.
Silent bin. With s = 0 the product is 0, so the criterion can never pass. The disc grows until it takes in spikes from nearby bins, or until it reaches the maximum size. Silent regions therefore pick up some rate from neighbouring fields.

## Parameters that affects the smoothing
Parameter	Value	Effect
<SKAGGS_ALPHA	50 cm·s·√spike>	Main control. Higher α gives larger discs and more smoothing.
<target_bin_cm	2>	α is in physical units, so it carries over between bin sizes. The bin size still sets how finely the radius can step and the smallest radius (half a bin).
<ADAPTIVE_MAX_RSQ	200>	Caps the radius at about 29 cm. This only matters for bins that never pass the criterion (very sparse or silent regions, low-spike cells).
<ADAPTIVE_MIN_OCC_S	0.5 s>	Doesn't change the radius. It decides which bins get a rate at all.

use_min_visits / min_visits, use_min_occ_s / min_occ_s	True / 2, False / 0.5	Invalid bins are holes that contribute nothing, so discs near holes and walls grow larger.
fps, the dt cap (2/fps), IMMOBILITY_SPEED_CMS, MAX_GAP_US, position smoothing and jump filtering		These change n and s, and so indirectly the radius.
The data (session length, firing rate, how evenly the arena was sampled)		Discs shrink as the data grow. A short session or the half-session maps get more smoothing at the same α.
These do not affect the adaptive map: the Gaussian sigma (it is used for coherence, stability and fields), SI_MIN_OCC_S (it only selects bins for the SI sum) and SAVE_SI_RATEMAP.

## How the radius scales
In a uniformly sampled 2-D patch with occupancy density ρ (s/cm²) and rate f (Hz), the disc holds n ≈ πρr² and s ≈ fπρr². Substituting gives

$$r^* \approx \left(\frac{\alpha}{(\pi\rho)^{1.5}\sqrt{f}}\right)^{1/4}$$

The radius goes as α^¼. Doubling α makes discs only about 19% larger. Your calibration agrees: α = 50 → 100 took the in-field radius from 3.0 to 3.8 cm. The choice of α is therefore forgiving, and it should be searched in steps of 2× or more.
The radius goes as f^(−⅛). In-field and out-of-field discs differ only modestly.
The radius goes as ρ^(−⅜). Occupancy and session length matter more than firing rate.
Narrow tracks. On the linear and circular tracks the disc's area grows roughly in proportion to r rather than r², so there r ∝ α^0.4. The radius is a bit more sensitive to α there.
How to choose α
There is no single correct α. It is a bias–variance trade-off:

Too small: rate estimates are noisy, and SI is biased upward by limited sampling. This bias is strongest for low-rate cells and short sessions.
Too large: fields blur and merge, and SI is biased downward.
Ways to choose it, from most to least principled:

Cross-validated likelihood (best). Split each session into interleaved chunks, for example alternate 1-minute blocks or odd and even laps. Build the adaptive map from the training chunks for each α in a grid such as {10, 25, 50, 100, 200, 400}. Score the held-out spikes with a Poisson log-likelihood, using predicted count = rate × occupancy in each bin or frame. Sum across cells (per arena if you like) and take the α with the highest score. Use k-fold with large training fractions (e.g. 4/5). A half-split under-represents the data, because the Skaggs criterion's dependence on data amount doesn't exactly match the statistically optimal bandwidth scaling.

Simulations with known ground truth. Use your real trajectories to generate Poisson cells with known fields (a range of sizes and peak rates), plus non-spatial cells. For each α, measure the rate-map error and the SI error (estimated minus true). Choose an α that keeps both small, which is what Grieves' "balanced" solution does. Also check the false-positive rate of the shuffle test on the non-spatial cells.

A geometric limit (what your current calibration uses). The median in-field disc radius should stay below the smallest field radius you want to resolve. Your comments record this: at α = 50 the in-field median is 3.0 cm, against a 10th-percentile field radius of about 3.7 cm, and no bin reaches the maximum radius. That gives an upper bound on α, not an optimum.

A literature value converted to your units. Use SKAGGS_ALPHA = alpha_Grieves × target_bin_cm / fps. Grieves' 2247 converts to about 55, which supports 50.

Sensitivity check (do this in any case). Rerun at about 0.5×, 1× and 2× your chosen α and report how many cells are classed as place cells and how the SI distributions compare. Because r ∝ α^¼ the results should barely change. If they do change, those cells' classification depends on the smoothing.

## Diagnostics to check at any α:

the median disc radius in-field and out-of-field (already shown in the ratemap figure title);
the fraction of bins at the maximum radius, which should be about 0;
the fraction of bins zeroed by ADAPTIVE_MIN_OCC_S.

Caveats:
Smoothing depends on the data, so groups with different session lengths or occupancy coverage (arenas, animals) get different effective smoothing at the same α. The criterion partly compensates, but keep this in mind when comparing SI across groups.
The shuffles are smoothed with the same rule. Shuffled spikes are spread out, so their discs tend to be larger and their SI lower. This is fair to the significance test, but it means α changes the null distribution as well as the real SI.