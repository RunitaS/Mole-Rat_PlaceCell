# Speed Modulation algo in PlaceCellCharacterization_SpeedModv2.py:

To speed modulate or not to speed modulate? That is the question.....

Initially demonstrated by Mcnaughton, Barnes, O'Keefe in their 1983 paper. Place cells/ Complex Spiking cells were found to be modulated by speed, albeit not as strongly as Internurons/Theta cells.

Buzsaki believes that place cells aren't speed modulate: https://www.pnas.org/doi/epdf/10.1073/pnas.1912792116 
Only a proportion of cells show +ve and -ve corrlation which are artifacts of theta phase precession? Isn't that the definition of speed modulation though?

This analysis is anyway testing for speed modulation (uses binning approach transforming signal from time domain to speed domain). Speed cells are a different story altogether. They show a correlation of firing rate with speed at instantaneous level. They have significant correlation in time domain (Kropff MEC 2015, and Tort 2018).
*Speed modulated cells should have had the same correlation though right? How is data binning positive for speed effect? Does this mean that the speed modulation is very weak and become statistically significant only when you pool data into same speed bins?*


Everything starts in compute_metrics() (PlaceCellCharacterization_SpeedModv2.py:700), called once per unit by _run_job().

Position data (:704-737):

Load tracking CSV/XLSX → raw t (timestamp, µs), x, y.
Drop dropout frames coded as ±1.
Compute frame-to-frame speed and drop implausibly fast jumps (**speed < 0.006 in raw units** *adjust this value to cm. 0.006 is proabbly based on pixels*) — a tracking-artifact filter.
Sort by t.
Convert to cm (x_cm, y_cm) — either already-cm columns, or pixel→cm via arena width.
Spike data (:769-793):

Memory-map the .ntt spike file, drop cluster 0 (unsorted/noise).
Sort spike timestamps (spike_ts, µs, same clock as position t).
For each spike, searchsorted finds the nearest position frame; spikes farther than MAX_GAP_US (50 ms) from any tracked frame are discarded as valid_spike = False — this is the first spike↔position correlation point, used to build the spatial rate map (SIR/sparsity/coherence), not yet speed.
This x_cm, y_cm, t (position) and spike_ts[valid_spike] (spike) are packed into ctx and passed to _compute_speed_modulation(ctx['x_cm'], ctx['y_cm'], ctx['t'], ctx['spike_ts'], fps, ...) at :963-966.

_compute_speed_modulation step by step
0. Guard. If fewer than 3 position samples or zero spikes, return all-NaN result immediately.

1. Per-frame speed from position data alone (:337-345)

dt_s = diff(t_us) * 1e-6 — real inter-frame interval (not assumed fixed 30 Hz), one value per of the n-1 gaps.
speed = hypot(diff(x_cm), diff(y_cm)) / dt_s —> Euclidean distance / time.
**Try plotting the speed in time domain for every trial. Also, plot the dt_s, and Euclidean dist in time domain**
Zero/negative dt_s → NaN (guards duplicate/out-of-order timestamps).
Pad to length n by repeating the last value, so speed[i] aligns with interval [t[i], t[i+1]).
At this point only position data has been touched — spikes haven't entered yet.
**Change n,n+1 to n,n-1. Pad i=1 speed.**

2. Spike, Pos/speed Ts matching - Instantaneous firing rate, put on the same frame base as speed (:356-378) — this is where spike data and position data are correlated:

interval_idx = searchsorted(t_us, spike_ts_us, side='right') - 1, **clipped to [0, n-2]** *why is it clipped?*: assigns every spike to the position interval it falls inside.
50 ms gate (MAX_GAP_US, same convention as the earlier place-map gating): a spike is only kept if it's within 50 ms of the nearer bounding position timestamp of its interval — drops spikes recorded during tracking dropouts. *flag if poition drops are above 10 frames*
counts = bincount(interval_idx) → spike count per interval.
fr_inst = counts / dt_s → raw instantaneous rate (Hz) per interval, padded to length n the same way as speed. **what is the bin width? is it 33.33msec?**
Gaussian-smooth fr_inst (gaussian_filter1d, σ = smooth_window_s * pos_sample_rate_hz samples, default 0.3 s × 30 Hz ≈ 9 samples) → fr_smooth. NaNs are zero-filled before filtering then restored after, so they don't bleed across the kernel.
**Gaussian smoothing of speed is missing** *check is speed is pres-smootherd in compute_metrics step.*
Now speed[i] and fr_smooth[i] are two parallel arrays, one value per position frame — the spike train has been fully converted into the position/time base.

***Gaussian smooth the spikes and speed either before or after Ts matching?***

3. Restrict to usable range (:405-412): keep only frames with SPEED_MIN_CMS (2 cm/s) < speed < SPEED_MAX_CMS (60 cm/s) **Change upper limit to 80cm/s** and finite fr_smooth (excludes near-stationary/resting periods and tracking-jump outliers) *How does smoothing exclude immobility periods of spiking?*. Need ≥3 valid samples with nonzero variance in both speed and rate, else abort.

4. Bin rate by speed and fit (:417-462):

Build SPEED_BIN_CMS-wide bins (default 4 cm/s) spanning [2, 60] cm/s. *Change to 5cm/s bins. It will yield 12 bins in total which can be tested statistically.*
Assign each valid sample to a bin (digitize), average fr_smooth within each bin → mean_rate per bin.
Drop bins holding < SPEED_MIN_BIN_FRAC (0.2%) of total samples (protects the sparse high-speed tail).
Linear regression (linregress) of mean_rate vs. bin_centres (speed) over the remaining bins (need ≥3): **Switch to Generalized Linear Mixed Model instead**
speed_score = r (reg.rvalue)
speed_p_value = p (reg.pvalue)
speed_beta = slope, speed_f0 = intercept
speed_modulated = p < 0.05
5. Circular-shift shuffle confirmation (:472-514), only if there are enough intervals (n_intervals > 2 * MARGIN_FRAMES, margin = 20 s of frames): *What is the meaning of interval here? Invertvals between what?*

1000 times (SPEED_N_SHUFFLE): circularly roll the per-interval spike counts by a random offset ≥20 s from either end (np.roll), recompute fr_inst_s → fr_smooth_s, re-bin using the same bin_idx/fit_idx (these depend only on speed, not firing, so they're fixed), refit linregress, store reg_s.rvalue. **Quantify the time shifts to check the rsnge of offsets** *Go through this shuffling procedure and verify if this is correct.* 
**Weird convex dist. instead of a typical Gaussian dist.!**
This decorrelates firing from speed while preserving each trace's autocorrelation structure — same logic as the SIR spatial bootstrap.
From the ≥100 valid shuffle scores: speed_shuffle_mean, 2.5th/97.5th percentiles (speed_shuffle_lo/hi), two-tailed speed_shuffle_p = (#|shuffle r| ≥ |real r| + 1) / (N + 1).
speed_modulated_shuffle = real r falls outside the [lo, hi] 95% shuffle interval — the more robust, final speed-modulation verdict.
Optionally saves a shuffle-histogram PNG if ntt_path given.
6. Diagnostic plot (:546-579): if ntt_path given, scatter raw speed/rate samples + binned means + fit line, annotate with r/slope/intercept/p, save PNG under <ntt_dir>/speed modulation/.

Returns a dict with all the above fields (speed_score, speed_p_value, speed_beta, speed_f0, speed_modulated, speed_shuffle_mean/lo/hi/p, speed_modulated_shuffle).

Summary of the spike↔position correlation points
Upstream in compute_metrics: spikes matched to nearest position frame (50 ms gate) to build the spatial ratemap/SIR — produces ctx['spike_ts'], already filtered to spikes near valid tracking.
Inside _compute_speed_modulation, step 2: that same spike train is re-binned onto per-frame intervals (not point-frames) via searchsorted, with its own independent 50 ms gate, and converted to a smoothed instantaneous rate — this is the actual join between the spike-derived rate trace and the position-derived speed trace, both indexed 0..n-1 on the position frame base.
Step 4 regression: rate (from spikes) vs. speed (from position) binned and correlated via Pearson r / linear regression — the core "speed score."
Step 5 shuffle: re-tests that correlation by shifting the spike-derived trace relative to the fixed speed-derived binning, to confirm the correlation isn't a statistical artifact.