**Place cell characterization**
# to-do:

- [ ] ***`Major fix! Exclude cluster 0 spike in this and all other codes!`*** 

- [ ] **Add Mehta skewness score calc to individual place fields.**

- [ ] **Check the shuffling analysis. SOmetime it gets 102/102 cells, other time it gets 100/102 cells as true place cells.**

- [ ] **Check the part when triangular kernel smoothing is applied. It sould be after peak and SIR estimation.**

- [ ] **Plot place fields in python code** *Compare trajectory maps with rate maps. Pierre's code had some discrepancies.*


# Speed modulation algo:

_compute_speed_modulation

Inputs: position track (x_cm, y_cm, t_us), spike timestamps (spike_ts_us), the position tracking frame rate (pos_sample_rate_hz, called with fps=30 Hz at the call site line 853), and five tunable parameters that default to module-level constants (lines 132-136):

SPEED_MIN_CMS = 2.0, SPEED_MAX_CMS = 90.0 — usable speed window
SPEED_BIN_CMS = 2.0 — speed-bin width
SPEED_SMOOTH_S = 0.08 — Gaussian smoothing sigma (seconds)
SPEED_MIN_BIN_FRAC = 0.002 — minimum occupancy fraction to keep a bin (converted to sec?)
Step 0 — Early exit. If fewer than 3 position samples or zero spikes exist, return a result dict of all-NaN / None (311-317).

Step 1 — Per-frame running speed (324-330).
For each of the n-1 inter-frame gaps:

dt_s = diff(t_us) * 1e-6 — true elapsed time per gap (not an assumed fixed frame period).
speed = hypot(diff(x_cm), diff(y_cm)) / dt_s — Euclidean displacement / elapsed time.
Any gap with dt_s <= 0 (duplicate/out-of-order timestamp) is set to NaN.
The speed array is padded by repeating the last value, so it has length n (one speed value per position sample).
Step 2 — Instantaneous firing rate on the same frame base (341-352).

Each spike is assigned to the inter-frame interval it falls into via searchsorted(t_us, spike_ts_us, side='right') - 1, clipped to valid range.
counts = bincount(interval_idx) gives spike counts per interval.
fr_inst = counts / dt_s — raw (unsmoothed) instantaneous rate per interval, in Hz.
Padded to length n the same way as speed.
Step 3 — Gaussian smoothing of the firing rate only (367-371).

sigma_samples = smooth_window_s * pos_sample_rate_hz (e.g. 0.08 s × 30 Hz ≈ 2.4 samples).
NaNs are zero-filled before filtering (so they don't smear through the kernel) and restored afterward via the finite mask.
gaussian_filter1d is applied with mode='nearest'. Note: speed itself is not smoothed, only firing rate is.
Step 4 — Restrict to a usable speed range (379-386).

Keep only samples where min_speed_cms < speed < max_speed_cms and fr_smooth is finite.
Abort (return NaN result) if fewer than 3 samples survive, or if either speed or rate has zero variance.
Step 5 — Bin firing rate by speed (391-420).

Build bin edges from min_speed_cms to max_speed_cms in steps of speed_bin_cms (e.g. 44 bins for the 2–90 cm/s default range).
Assign every valid sample to a bin via digitize.
For each bin, compute the mean smoothed firing rate and the sample count.
Bins whose sample count is < min_bin_frac of the total (e.g. < 0.2%) are dropped (set to NaN) — this mostly prunes the sparsely-sampled high-speed tail.
Step 6 — Linear regression on the binned means (422-436).

Require ≥ 3 surviving bins.
scipy.stats.linregress(bin_centres, mean_rate) fits rate = speed_beta * speed + speed_f0.
Outputs: speed_score = Pearson r (reg.rvalue), speed_p_value = reg.pvalue (two-sided t-test on whether the slope differs from 0), speed_beta = slope, speed_f0 = intercept, speed_modulated = p < 0.05.
Important: the regression is fit across n_bins binned points (e.g. ~44), not across the underlying thousands of frames. This is the effective sample size that drives the p-value/degrees of freedom.
Step 7 — Optional diagnostic plot (439-472).
If ntt_path is given, saves a scatter of raw (speed, rate) samples, the binned means, and the fitted line with r/slope/intercept/p annotated, into a speed modulation subfolder next to the .ntt file.

Return: dict with speed_score, speed_p_value, speed_beta, speed_f0, speed_modulated.




# Speed mod: Parameters affecting p-value

**A. Parameters already exposed in this function: Parameter and Effect on p-value**

[speed_bin_cms] (currently 2)	Directly sets the degrees of freedom of the regression: wider bins → fewer points → less statistical power but less per-bin noise; narrower bins → more points/df but noisier, more NaN-dropped bins at the tails. This is a genuine bias/variance trade-off, not just cosmetic.	1–3 cm/s is typical in the literature. For a species with a narrow speed range (see below), too-wide bins can collapse the whole range into <10 points, making the p-value unstable.
min_speed_cms / max_speed_cms (2–90)	Sets which frames enter the analysis at all. 

[low_cutoff] is the most consequential: near-zero speed samples often show elevated/depressed firing from different phenomena (SWR-associated bursts, theta-related resting activity), and including them creates spurious slope/p-values unrelated to genuine speed tuning.	Typical literature range: exclude <2–5 cm/s. The upper bound matters less for significance but affects which points anchor the high end of the fit; if the animal rarely reaches 90 cm/s, that part of the range is just empty bins.

[smooth_window_s] (0.08 s / σ)	Because only the rate is smoothed and not speed, heavier smoothing reduces the residual scatter of fr_smooth around the fit without a matching reduction in speed noise — this can artificially tighten the fit and inflate significance if the smoothing window bleeds firing across a speed transition (e.g., rate from a fast bout leaking into slow-bout frames). It also introduces **autocorrelation between adjacent frames**, which the classic linregress p-value doesn't account for (it assumes independent residuals) — this generally makes the p-value anti-conservative (falsely low).	Published speed-tuning studies use 100–300 ms smoothing (boxcar or Gaussian). 80 ms is on the short/conservative end, which is good for minimizing this bias but yields noisier per-bin rates.

[min_bin_frac] (0.002)	Controls how aggressively sparse (mostly high-speed) bins are excluded. Too permissive → few-sample bins with noisy means get equal regression weight to well-sampled bins (the fit is unweighted OLS, so a bin averaged from 2 samples counts the same as one averaged from 2000) → can produce spurious slopes/low p-values driven by 1-2 outlier bins. Too strict → throws away the high-speed range entirely, shrinking n_bins and power.	Consider either raising this threshold, or (better) switching to a variance-weighted regression (np.polyfit(..., w=sqrt(n_per_bin)) or statsmodels.WLS) so bin means are weighted by their sampling density.


**B. Structural choices not exposed as parameters (but that matter just as much)**

* *Binned-mean regression vs. per-frame or per-shuffle test.* Fitting on ~40 binned means (rather than the raw thousands of frames) is a deliberate noise-reduction step, but it also means the parametric p-value from linregress is really testing "do these ~40 points show a nonzero slope," which assumes Gaussian, **homoscedastic residuals** — an assumption that's **shaky for firing-rate data (Poisson-like, variance scales with the mean, and bins have unequal sample counts)**. This is the single biggest hidden factor: the reported p-value is likely optimistic relative to a permutation/shuffle-based p-value.

*No shuffle-based null distribution for speed modulation* — unlike the SIR bootstrap elsewhere in this file (_run_bootstrap, circular shift), speed modulation here relies purely on the parametric linregress p-value. A more robust, and commonly used, approach is to circularly shift the spike train relative to the speed trace many times (same trick already used for spatial information), recompute r each time, and take the p-value as the fraction of shuffles exceeding the real r. This breaks the temporal-autocorrelation problem in (A) above and doesn't assume any particular residual distribution.

*Pearson r (linear) vs. Spearman/monotonic fit.* Some cells show saturating or non-monotonic speed tuning; a linear fit will underestimate modulation (raising p) in those cells. Spearman correlation or a rank-based test is more robust to this and to bin-mean outliers.

*Recording/session duration and locomotion coverage.* If the animal doesn't traverse the full speed range with enough dwell time, high-speed bins are undersampled, shrink n_bins after the min_bin_frac cut, and a couple of noisy tail bins gain outsized leverage on the slope — this can flip significance session to session even for the same "true" tuning.

*Multiple-comparisons correction across cells.* With a fixed α = 0.05 applied cell-by-cell across a whole batch, ~5% of non-modulated cells will be flagged "significant" by chance alone. If the downstream question is "what fraction of the population is speed-modulated," an FDR (Benjamini-Hochberg) correction across all tested units is the standard fix — this file doesn't do that.

*Species-appropriate speed range.* This matters specifically for mole-rats: they are fossorial and move much more slowly than mice/rats. If actual running speeds rarely exceed, say, 15–20 cm/s, then a fixed 90 cm/s upper bound and 2 cm/s bins mean most of the 44 bins are empty/dropped, and the "real" data occupies only a handful of bins near the low end — collapsing the effective degrees of freedom far below what the code implies. It's worth checking the empirical speed distribution (e.g., 95th/99th percentile of speed) per session and setting max_speed_cms (and possibly a finer speed_bin_cms, e.g. 1 cm/s) accordingly, rather than reusing a rat/mouse-calibrated default.


**Recommended values / checks**

Set min_speed_cms based on when immobility-related bursting (SWRs) stops, not an arbitrary constant — inspect the autocorrelogram/theta modulation near threshold.
Set max_speed_cms from the session's own speed distribution (e.g. 99th percentile) rather than a fixed 90 cm/s, especially for slow-moving species.
Keep speed_bin_cms small enough to give ≥ 8-10 surviving bins after the min_bin_frac cut, but not so small that most bins get pruned as under-sampled — this is worth checking empirically per session rather than assuming 2 cm/s always works.
Symmetric smoothing: either also lightly smooth speed with the same kernel, or reduce smooth_window_s further, to avoid the asymmetric-smoothing bias described above.
Replace/augment the parametric linregress p-value with a shuffle-based p-value (circular time-shift of spikes vs. speed, analogous to the existing SIR bootstrap) for a more defensible significance claim.
Weight the regression by n_per_bin (or use Spearman) rather than unweighted OLS on bin means.
Apply FDR correction across the batch if reporting "fraction of cells speed-modulated."


* Instantaneous firing rate

Position samples define n-1 inter-frame intervals; dt_s is the actual duration of each (from timestamps, not an assumed fixed rate).
Each spike is assigned to the interval it falls in via searchsorted(..., side='right') - 1, counts per interval are tallied with bincount, and fr_inst = counts / dt_s gives one raw Hz value per interval — this is the "instantaneous" rate, essentially a per-frame PSTH.
NaN intervals (bad dt) are zero-filled before Gaussian smoothing (so they don't smear NaN across the kernel), then restored afterward via the finite mask.
Window size for firing-rate smoothing

* Controlled by smooth_window_s = SPEED_SMOOTH_S = 0.08 s (80 ms), defined at PlaceCellCharacterization_SI_Spar_Cohr_PeakFR_MeanFR_Shuffling_TwoHalvesCopmare_ThetaMod_Batch_GPU_Final.py:135.
This value is the Gaussian kernel's standard deviation (sigma), converted to samples as sigma_samples = smooth_window_s * pos_sample_rate_hz. With the tracking rate fps = 30 Hz used elsewhere in the file, that's sigma_samples ≈ 2.4 frames.
scipy.gaussian_filter1d truncates the kernel at 4·sigma by default, so the effective smoothing window (full kernel support) is roughly ±4×80 ms ≈ ±320 ms, not just the 80 ms sigma itself.
Speed binning

* Bin edges are built with speed_bin_cms-wide bins (default 2 cm/s) spanning [min_speed_cms, max_speed_cms] (default 2–90 cm/s).
Each valid (speed, smoothed-rate) sample is assigned to a bin via np.digitize, and the smoothed rate values in each bin are averaged to get one mean rate per bin.
Bins holding fewer than min_bin_frac (default 0.2%) of all samples are dropped as unreliable — mainly affects the sparsely-sampled high-speed tail.
The surviving bin centres/means are fit with linregress to get speed_beta (slope), speed_f0 (intercept), speed_score (r), and speed_p_value.

## Speed mod to-do:

- [ ] **For all shuffling analysis throughout the code, use the same shuffled spike trains instead of creating new shufflis everytime. Computationally more efficient this way.** *store it in memap and use whenever a shuffling criteria is to be applied.*

- [ ] **Use GLM to predict factor causing the cell to be either n or p type speed modulated.** *PTP model has already quantified the different parameters affecting the differential speed modulation. Cross check in your data if their prediction fits the cause of n and p type speed mod. spuriosu correlations.*

- [x] ** Add stats singificance step for r2 value before both speed domain and time domain analysis. Use cell for further analysis only if significance criteria is fulfilled.** 

- [x] ** combine the binning analysis and time domain analysis and save all images in the same folder labelled speedVsfiring.**

- [ ] **Instantaneous firing rates for my plots are too low. What is the difference between intrinsic and instantaneous firing rates? Is intrinsic firing rate calculated/sec?**
*Use this paper as ref.:https://www.nature.com/articles/s41598-020-58194-1*

- [ ] **Add n and p speed types to binning methods doendent on slope of lin. reg.**

- [ ] **Replicate the methods mentioned in https://www.nature.com/articles/s41598-020-58194-1 . What TF is p-speed and n-speed? They have characterized the correct parameters for this analysis.** *p-speed is +ve speed correlation and n-speed is negative speed correlation.*

- [ ] **Look for methods to control the scatter and outliers in lin. reg.**

- [ ] **Apply speed modulation only to isolated place fields.**

- [x] ** Try plotting the speed in time domain for every trial. Also, plot the dt_s, and Euclidean dist in time domain ** *dt_s is constant '33.33 or 33.34 msec'.

- [x] ** Try merging two bins. This means that speed will be smoothed 2x but the spike bins will be wider That might reduce the scatter. check the bin size used for instantaneous firing rate in other papers. Maybe 33.33 msec is too low.**

- [x] ** Unequal samples per speed bin problem: That's about how the regression itself is fit, not about correcting many tests. Right now (_compute_speed_modulation, PlaceCellCharacterization_SpeedModv3.py:519) linregress is fit on bin_centres vs mean_rate — each surviving speed bin contributes one point, equally weighted, regardless of whether it was built from 5000 samples or 50 samples (only SPEED_MIN_BIN_FRAC filters out the very sparsest bins, it doesn't equalize the rest). A sparsely-sampled bin has a noisier mean but the same leverage in the fit as a densely-sampled bin — that's a bias/variance issue in the regression's weighting, unrelated to multiple comparisons. The fix for that is either: weighted least squares, weighting each bin by n_per_bin (or 1/variance), instead of scipy.stats.linregress, or**

- [ ] **Quantify the time shifts to check the range of offsets**

- [x] ** Resolution in Mcnaughton 1983 is pretty low. Replicate their resolution and check for speed modulation.**  
*The animals's position was continuously sampled by the computer at a rate of 10 Hz. The resolution in the position measure was estimated at about 0.5 cm. Since instantaneous velocity was calculated from the distanced moved between sampling points its resolution was therefore about 5 cm/sec.*

- [ ] **Change n,n+1 to n,n-1. Pad i=1 speed.** ***Retrospective speed coding and prostpecive speed coding is a thing!*** *n,n+1 will give retrospective speed, while n,n-1 will give prospective speed*

- [x] ** Gaussian smoothing of speed is missing ** * Is it pre smoothed in compute_metrics? * *There was no speed or position smoothing anywhere in the code. Added position smoothing. Does the speed need further smoothing?*

<NOTE>:One thing worth flagging: since the plot uses the already-smoothed position (jump-removed + Gaussian-smoothed from your earlier request), the speed trace will look cleaner than raw tracking speed — let me know if you instead wanted the raw, pre-smoothing speed plotted for QC/comparison purposes.

- [x] ** Change upper limit to 80cm/s **

- [x] ** Use median instead of mean after time domain to speed domain transformation.**

- [x] ** Switch to stats to Linear Mixed Model instead ** *Not required. Data from same cell in same session doesn't require mixed model.*

- [x] ** Cross check shuffling analysis. Distribution of random shuffles is not Gaussian. It's a weird convex distribuiton ** *This is due to the fact that r values take either +ve or -ve values in range -1 to 1. You will get two peak at roughly -0.5 and 0.5.

- [x] **Place cells are modulated by speed but aren't speed cells. Check difference between Adriano Tort (2018) and MEC speed cells analysis, Kropff (2015). (e.g., McNaughton et al., 1983; Wiener et al., 1989; Czurko´et al., 1999; Hirase et al., 1999; Ekstrom et al., 2001; Maurer et al., 2005) have analyzed modulation. Place cells don't change firing at micro seconds level. Only god knows how the two differ! **

- [x] ** Change speed bin len: default is 2cm/s **

- [x] ** Change smooting factor **

- [x] ** Two methods possible for speed analysis. 1. Kropff/Tort: Correlate in time domain. 2. Bin the data and correlate speed bins. each wll yield different results **
*Currently my data not yielding any positive results (post shufling) for even the binning methods which should have had speed modulation.*

- [x] ** Add smoothing before binning.**

- [x] ** Smooth the speed as much as the spikes **

- [x] ** Change range to 2-60cm/s **

- [x] ** Add shuffling procedure to confirm results **

- [x] ** Chech distribution for normality. If not normal, use spearman **

- [x] ** Correct for multiple comparisons **

- [x] ** Check if speed modulated cells are also theta modulated ** *They are not!*
        ** WHY TF ARE SPEED MOD CELLS NOT THETA MOD?** *Read Buzsaki's PTP model paper*