**Place cell characterization**
# to-do

- [ ] **Check if speed modulated cells are also theta modulated**

- [ ] **Check the shuffling analysis. SOmetime it gets 102/102 cells, other time it gets 100/102 cells as true place cells.**

- [ ] **Speed modulation plots look weird. Chec inst. firingrate procedure. Change firing rate and speed bin sizes.**


# Speed modulation 

Instantaneous firing rate

Position samples define n-1 inter-frame intervals; dt_s is the actual duration of each (from timestamps, not an assumed fixed rate).
Each spike is assigned to the interval it falls in via searchsorted(..., side='right') - 1, counts per interval are tallied with bincount, and fr_inst = counts / dt_s gives one raw Hz value per interval — this is the "instantaneous" rate, essentially a per-frame PSTH.
NaN intervals (bad dt) are zero-filled before Gaussian smoothing (so they don't smear NaN across the kernel), then restored afterward via the finite mask.
Window size for firing-rate smoothing

Controlled by smooth_window_s = SPEED_SMOOTH_S = 0.08 s (80 ms), defined at PlaceCellCharacterization_SI_Spar_Cohr_PeakFR_MeanFR_Shuffling_TwoHalvesCopmare_ThetaMod_Batch_GPU_Final.py:135.
This value is the Gaussian kernel's standard deviation (sigma), converted to samples as sigma_samples = smooth_window_s * pos_sample_rate_hz. With the tracking rate fps = 30 Hz used elsewhere in the file, that's sigma_samples ≈ 2.4 frames.
scipy.gaussian_filter1d truncates the kernel at 4·sigma by default, so the effective smoothing window (full kernel support) is roughly ±4×80 ms ≈ ±320 ms, not just the 80 ms sigma itself.
Speed binning

Bin edges are built with speed_bin_cms-wide bins (default 2 cm/s) spanning [min_speed_cms, max_speed_cms] (default 2–90 cm/s).
Each valid (speed, smoothed-rate) sample is assigned to a bin via np.digitize, and the smoothed rate values in each bin are averaged to get one mean rate per bin.
Bins holding fewer than min_bin_frac (default 0.2%) of all samples are dropped as unreliable — mainly affects the sparsely-sampled high-speed tail.
The surviving bin centres/means are fit with linregress to get speed_beta (slope), speed_f0 (intercept), speed_score (r), and speed_p_value.