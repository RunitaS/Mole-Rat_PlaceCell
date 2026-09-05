Synthesized spectral decomposition theory with methodological validation

This is a two-stage pipeline, and the two stages rest on the same core idea from a different angle. 
I'll walk through the theory first (because the code only makes sense once the 1/f logic is clear), 
then go through Part 1 (PSD → FOOOF) and Part 2 (fBOSC → episode/IEI/speed) step by step, and finish 
with a few things in the code worth watching and the citations.

The central idea: periodic vs. aperiodic

Every electrophysiological power spectrum is a mixture of two things. 
There is an aperiodic (1/f) component — broadband power that falls off with frequency, 
appearing as a straight line (or a line with a bend) in log-log space — 
and there are periodic components, the narrowband "bumps" that sit on top of that background 
and correspond to genuine oscillations (theta, alpha, etc.).

The aperiodic part is not noise to be discarded. Its steepness (the "exponent" or slope) 
and vertical position ("offset") carry physiological meaning — the slope is thought to track 
excitation/inhibition balance, aggregate spiking, and arousal state. 
The problem is that if you measure "theta power" naively by summing power in 3–7 Hz, 
you are measuring the oscillation plus whatever the 1/f background happens to be at those frequencies. 
If the slope tilts between conditions, your band power changes even when the oscillation itself did not. 
Both FOOOF and BOSC/fBOSC exist to separate these two contributions properly — FOOOF to parametrise them, 
fBOSC to detect discrete oscillatory bursts above the aperiodic floor. 
This pipeline uses FOOOF to characterise the average spectrum and then uses fBOSC 
(which has FOOOF inside it) to find, second by second, when theta is actually present.

FOOOF / specparam — the theory

FOOOF ("fitting oscillations and one over f", now often called specparam) models the log power spectrum 
as an aperiodic component plus a sum of Gaussian peaks. In the knee formulation this pipeline uses, 
the aperiodic term is

L(F) = b − log₁₀(k + F^χ)

where b is the offset, k is the knee parameter, and χ is the exponent (the slope). 
The knee is what lets the fit bend — real neural spectra often flatten below some frequency 
(roughly 0.5–10 Hz) and steepen above it, so a single straight line systematically misfits 
the low-frequency end. That is exactly why this script sets aperiodic_mode='knee': it fits from 1–40 Hz, 
a wide range that spans the knee region, and a linear fit there would be biased.

The fitting is iterative. FOOOF first fits the aperiodic component, subtracts it, then finds the largest 
remaining bump, models it as a Gaussian (centre frequency, power, bandwidth), subtracts it, 
and repeats until peaks fall below the height/threshold criteria. Then it refits the aperiodic component 
on the peak-removed spectrum, so the final 1/f estimate is not contaminated by the oscillatory 
peaks sitting on top of it. This is the key robustness property: the background fit does not get 
dragged upward by a strong theta peak. The output is aperiodic_params_ (offset, knee, exponent) 
and peak_params_ (one row per peak: CF, power, bandwidth), plus goodness-of-fit r_squared_ and error_.

In the code this maps directly onto:

FOOOF_SETTINGS: peak_width_limits=[1,8] (peaks narrower than 1 Hz or wider than 8 Hz are rejected — 
a guard against the fit absorbing aperiodic curvature as a fake wide "peak"), max_n_peaks=6, 
min_peak_height=0.1, peak_threshold=2.0 (a peak must exceed 2 SD of the flattened spectrum's noise), 
aperiodic_mode='knee'.
extract_theta_peak() pulls the strongest peak whose centre frequency lands in THETA_BAND=(3,7), 
and theta_range_from_peak() converts a peak's CF and bandwidth into a [cf − bw/2, cf + bw/2] band. 
This is the crucial move that makes the theta band data-driven per recording rather than a fixed 3–7 Hz 
window applied to everyone.

#%%
BOSC and fBOSC — the theory

BOSC ("Better OSCillation") answers a different question: not "what does the average spectrum look like" 
but "at each moment, is there a real oscillation here or just background?" It has two steps.

First, a time-frequency decomposition (Morlet wavelets) gives power at every frequency and every time 
point. Under the null hypothesis that only aperiodic background is present, wavelet power at a given 
frequency is theoretically χ²-distributed with 2 degrees of freedom, scaled by the expected background 
power at that frequency. So if you know the background, you know the whole null distribution of 
"power you'd see with no oscillation," and you can set a power threshold at, say, the 95th or 
99th percentile of that χ² distribution. Anything exceeding it is unlikely to be background alone.

Second, a duration threshold: a supra-threshold excursion only counts as an oscillation if it persists 
for a minimum number of cycles (classically ~3). A single spurious sample crossing the threshold is not 
a burst; sustained rhythmicity is. Together, the power-and-duration criteria define discrete oscillatory 
episodes with an onset, offset, mean frequency, and mean power.

Everything hinges on estimating that background correctly, and this is where the original BOSC and eBOSC 
are fragile. Original BOSC fits the 1/f with ordinary least squares on the log spectrum; eBOSC uses a 
robust fit and lets you manually exclude peak frequencies. Both fit a straight line, so where the 
spectrum has a knee, the 1/f fits from BOSC and eBOSC fail to model the non-linear spectrum, and the 
resulting power thresholds are too high for frequencies below the knee — meaning real low-frequency 
bursts get missed. And if you don't manually exclude peaks, a strong oscillation inflates the background 
estimate right where you're trying to detect it. 
biorxiv

fBOSC (Seymour, Alexander & Maguire, 2022) fixes both problems by swapping in FOOOF for the background 
fit. As the authors put it, the FOOOF algorithm performs an initial 1/f fit and iteratively models 
oscillatory peaks above this background as Gaussians; these peaks are removed from the spectra and 
the 1/f fit is performed again, so the fit is not influenced by oscillatory peaks and no manual selection 
of peak frequencies is required. Because FOOOF can include a knee, fBOSC is especially beneficial where 
power spectra contain a knee below ~0.5–10 Hz, which is typical in neural data, and unlike other methods 
it was unaffected by oscillatory peaks in the spectrum. Once fBOSC has this robust background, 
the χ²-percentile power threshold and the cycle-duration threshold proceed exactly as in classic BOSC. 
That's why the header describes fBOSC as fitting "its own background spectrum internally" — it re-derives 
the 1/f fit and thresholds from the LFP trace itself, and does not read Part 1's saved FOOOF results. 
bioRxiv

Part 1 — the PSD → FOOOF pipeline, step by step

1. Read the raw LFP. load_ncs() memory-maps the Neuralynx .ncs file (512 int16 samples per record, 
16 kB header skipped), concatenates the samples, and scales by ADBitVolts to get microvolts. It also 
returns the first-sample UNIX timestamp (µs), which is essential later because the tracking CSV lives 
on the same clock.

2. Downsample. resample_poly(lfp, 500, 32000) takes the trace from 32 kHz to 500 Hz. That sets Nyquist 
at 250 Hz and makes 2 s epochs cheap. 
nperseg = 2*500 = 1000 samples → frequency resolution df = fs/nperseg = 0.5 Hz.

3. Clean the time series. notch_filter() applies zero-phase IIR notches at 50/100/150/200 Hz 
(European mains + harmonics under Nyquist). Then detrend_signal() removes a linear trend to kill 
slow drift that a notch can't touch.

4. Speed-gate the epochs. find_position_file() grabs the tracking CSV in the same folder; 
compute_velocity_from_position() computes frame-to-frame displacement divided by the actual elapsed time 
from the column-A timestamps (so dropped frames don't bias speed), then Savitzky-Golay smooths it. 
compute_epoch_speed_keep() marks each 2 s epoch keep/reject by whether the animal's median smoothed 
speed during it falls in [SPEED_MIN_CMS, SPEED_MAX_CMS] = 1–90 cm/s. Aligning epochs uses the .ncs 
start timestamp against the tracking clock — no assumed shared t=0. This restricts the theta analysis 
to periods of active locomotion, when hippocampal theta is expected, and excludes immobility and grooming 
artifacts.

5. Reject bad epochs in a defined order (compute_psd_clean_epochs): (i) velocity gate from step 4; then 
within the survivors, (ii) a delta/theta filter — reject an epoch if its 1–3 Hz (delta) power exceeds its 
3–7 Hz (theta) power, removing epochs dominated by large-amplitude delta/LIA rather than theta; then (iii) 
dual MAD outlier rejection — reject if the epoch is a high outlier (robust z > 5) on either broadband 
peak-to-peak amplitude or delta power. Critically, the median/MAD reference statistics are computed only 
from epochs that already passed (i) and (ii), so artifacts can't inflate the threshold that's supposed to 
catch them. Welch's PSD is then averaged over the surviving epochs.

6. Remove residual line noise spectrally. clean_line_noise_psd() uses FOOOF's interpolate_spectrum to 
interpolate the PSD across ±2 Hz windows around each harmonic — a second line-noise defence on top of 
the time-domain notch.

7. Normalise. Each PSD is divided by its total power in NORM_BAND=(1,100) Hz, giving relative power. 
This removes overall amplitude differences (electrode impedance, distance from source) so spectra are 
comparable across files/animals — you're comparing shape, not absolute scale.

8. FOOOF everything. process_animal() returns one normalised PSD per file plus the cleaned full-length 
trace (lfp_store, which Part 2 needs). build_fooof_results() fits a FOOOFGroup to every file's PSD over 
1–40 Hz, saves an individual model-fit figure per file, and records offset/knee/exponent, peak params, 
R², and error. fooof_results_to_df() extracts the theta peak per file and flattens everything into one 
row per recording. export_low_quality_fits() flags files with R² < 0.98 or error > 0.4 for QC.

9. Plots. Mean ± SEM spectra per animal, per-file FOOOF fits, theta-property histograms (CF, power, 
bandwidth, exponent, offset), fit-quality box plots, and the composite plot_master_summary figure. 
All processed PSDs, the cleaned traces, and the full FOOOF output are pickled to disk.

Part 2 — fBOSC, IEI, and speed, step by step

Part 2 is gated behind interactive y/n prompts and reuses the cleaned full traces from Part 1 
(lfp_store inside processed_psds.pkl). Note an important conceptual point: fBOSC runs on the entire 
continuous notched/detrended trace, not on the speed-gated, delta-rejected epochs used for the PSD. 
So fBOSC sees all of time (including immobility and delta-heavy periods), detects episodes there, 
and speed is then correlated post hoc. The two parts treat the same cleaned trace differently on purpose.

1. Run fBOSC per file. For each file, timestamps are built in microseconds, and fBOSCpy_wrapper_v2(...) 
is called with F_array = 1–40 Hz in 1 Hz steps and Fs = 500. It returns raw and FWHM-post-processed 
episode tables. Each episode carries Onset/Offset (µs), DurationS, DurationC (cycles), FrequencyMean, 
PowerMean. Per-file rows are tagged with animal/day/session/tetrode/channel and session_duration_sec, 
and combined into fBOSC_episodesTable (cached to a pickle).

2. Filter and bin. filtered_episodes = episodes with FrequencyMean < 20 Hz. Histograms of episode 
frequency follow, and the theta band is taken as 3–7 Hz for the main theta-time calculation.

3. Compute theta occupancy ("p-episode"). calculate_total_theta_time() takes all theta-band episodes in 
a session and computes the union of their [Onset, Offset] intervals (merging overlaps so overlapping 
episodes aren't double-counted). Dividing that union duration by session duration gives the proportion 
of time the animal spent in a theta episode — the p-episode measure. pepisode_df assembles total episode 
counts, theta episode counts, session duration, total theta time, and proportion per session.

4. Frequency-distribution plots. Several cells compute, per session, the fraction of episodes falling 
in each frequency bin, then average across sessions (mean ± SEM), overall and per animal. These describe 
where in frequency the oscillatory episodes concentrate.

5. Episode-property and validity plots. plot_episode_properties shows distributions of frequency, 
duration (cycles and seconds), and power. The most theoretically important check is 
plot_duration_frequency_relationship: DurationS (seconds) should slope down with frequency 
(a fixed number of cycles takes less wall-clock time at higher frequency), while DurationC (cycles) 
should be roughly flat across frequency. That flatness is a sanity check that the wavelet width and 
thresholding aren't manufacturing a spurious frequency-dependence in burst length — a standard BOSC 
diagnostic.

6. IEI / theta-continuity analysis. Inter-episode interval is the gap between one episode's offset and 
the next one's onset. plot_theta_continuity looks at duration and IEI distributions and CDFs, and whether 
long episodes are followed by longer gaps. Then two refinements:

merge_split_episodes(min_iei_s=0.167) merges episodes separated by less than one cycle at 6 Hz, on the 
logic that a real burst briefly dipping under threshold shouldn't be counted as two separate episodes. 
It recomputes duration, sums cycles, and takes duration-weighted means of frequency/power.
test_frequency_continuity_across_iei() is the clever diagnostic: if theta is genuinely continuous but 
occasionally dips below threshold, the episodes flanking a short "suspicious" gap (0.8–1.5 s) should have 
nearly identical mean frequency; if theta truly stops and restarts, flanking frequencies should differ 
more. It compares |Δfrequency| across suspicious gaps vs. genuine long gaps (>3 s) with a Mann-Whitney 
test. This directly tests the "transient bursts vs. sustained rhythm" question that motivated the whole 
burst-detection literature.

7. Speed–theta correlation. Tracking is loaded, speed interpolators built, and attach_speed_to_episodes() 
samples running speed during each episode and during its preceding IEI (carefully converting the µs Onset 
to tracking-clock seconds). The plots then test movement-gating of theta: speed during theta vs. during 
IEI, episode duration vs. speed, IEI length vs. speed, P(theta | speed bin) (the classic movement-gating 
curve), speed peri-aligned to episode onset, and session-level p-episode vs. mean speed. This is the 
ultimate payoff — linking discrete theta bursts to behaviour.

A few things in the code worth flagging

These don't break the analysis but are worth your attention:

Theta band is inconsistent across cells. The main theta-time calc uses 3–7 Hz (cell 73), but cell 76 uses 
3–6 Hz and its own comment says "6–10 Hz," and cell 77/78 rebin differently. Decide on one theta 
definition and apply it everywhere, or you'll get subtly different theta counts in different figures.
Part 2's speed section (cells 102–106) will not run as written. It calls file_manager (a FileManager 
object) and helper functions (load_tracking_and_compute_speed is defined, but file_manager.
list_indexed_metadata / .get_files are not), which don't exist in this script — the header 
block warns about exactly this. Part 1's own compute_velocity_from_position already reads tracking; 
you'd need to bridge the two.
Minor docstring drift. compute_psd_clean_epochs says "4 s epochs" but nperseg is 2 s; the nperseg 
comment says "1000 samples at 1000 Hz" but it's 1000 samples at 500 Hz (the math — 0.5 Hz resolution — 
is correct). Cosmetic only.
fBOSC vs. PSD scope mismatch (by design, but be aware). As noted, fBOSC runs on the full trace while the 
PSD/FOOOF characterisation is speed- and delta-gated. Your p-episode and speed correlations therefore 
include immobility periods; that's fine and standard, but keep it in mind when comparing Part 1 and 
Part 2 numbers.

Citations

If you use this pipeline, cite:

fBOSC: Seymour, R. A., Alexander, N. A., & Maguire, E. A. (2022). Robust estimation of 1/f activity 
improves oscillatory burst detection. European Journal of Neuroscience, 56(10), 5836–5852. 
https://doi.org/10.1111/ejn.15829
BOSC / single-trial rhythm characterisation (eBOSC lineage): Kosciessa, J. Q., Grandy, T. H., Garrett, 
D. D., & Werkle-Bergner, M. (2020). Single-trial characterization of neural rhythms: Potential and 
challenges. NeuroImage, 206, 116331.
FOOOF / specparam: Donoghue, T., Haller, M., Peterson, E. J., Varma, P., Sebastian, P., Gao, R., … & 
Voytek, B. (2020). Parameterizing neural power spectra into periodic and aperiodic components. 
Nature Neuroscience, 23(12), 1655–1665.



Every fBOSC setting that influences theta-episode detection
These live in cfg_fBOSC inside fBOSCpy_wrapper_v2.py, which is where the actual fBOSC algorithm is configured. 
Only F_array and postproc are currently exposed as top-level knobs in nmr4019_fBOSC.py 
(FBOSC_F_ARRAY, FBOSC_POSTPROC) — everything else below is hardcoded inside the wrapper.

Setting	Current value	What it controls
F (FBOSC_F_ARRAY)	1–20 Hz, 1 Hz steps	Frequencies scanned for oscillations; also sets the range of the 
background/aperiodic (FOOOF) fit used to derive the power threshold
wavenumber	6	Morlet wavelet cycles — trades time resolution vs. frequency resolution in the wavelet 
transform; higher = better frequency precision but worse temporal precision (blurs episode onset/offset)
fsample	= LFP sampling rate (500 Hz, from fs_down)	Sampling rate fed to the wavelet transform; not really 
"tunable" but scales all the padding/duration-in-samples math below
pad.tfr_s	1 s	Bilateral padding trimmed after the wavelet transform, to discard edge artifacts
pad.detection_s	0.5 s	Bilateral "shoulder" padding trimmed after rhythm detection, needed so 
duration-threshold checks aren't biased at trace edges
pad.background_s	1 s	Padding trimmed from the segment used to estimate the background power spectrum
fooof.peak_width_limits	[2, 12] Hz	Constrains how narrow/wide a periodic peak FOOOF is allowed to fit when 
modeling the background spectrum — affects how much of a true theta peak gets "absorbed" into the aperiodic 
fit vs. treated as background
fooof.max_n_peaks	inf	Max number of periodic peaks FOOOF can fit while modeling the background
fooof.min_peak_height	0.1	Minimum peak height for FOOOF to count something as a peak
fooof.peak_threshold	2.0	SD-based threshold for peak detection in the FOOOF fit
fooof.aperiodic_mode	'knee'	Aperiodic (1/f) model shape — 'knee' fits a bend in the spectrum, 
'fixed' assumes pure power-law; wrong choice biases the background fit and thus the power threshold 
everywhere, including in the theta band
threshold.percentile	0.95	Percentile of the χ² distribution over the background fit used to set 
the power threshold (pt) — an epoch's wavelet power must exceed this to count as "oscillatory"; raising 
it → stricter, fewer/shorter episodes
threshold.duration	3 cycles at every frequency	Minimum number of cycles an oscillation must sustain to 
count as an episode (converted to samples via fsample/F, so at lower theta frequencies this is a longer 
absolute duration)
postproc.use (FBOSC_POSTPROC)	True	Whether onset/offset boundaries get refined at all after initial 
detection
postproc.method	'FWHM'	Boundary-refinement algorithm — Full-Width-Half-Max vs. 'MaxBias'; changes exactly 
where an episode is judged to start/end, which affects DurationS/FrequencyMean and therefore which episodes 
fall inside your 3–7 Hz theta filter
postproc.edgeOnly	'yes'	Whether the boundary-deconvolution is applied only at episode edges (recommended) 
or throughout
postproc.effSignal	'PT'	Which signal (power-threshold-referenced) is used for the post-processing 
correction
Downstream of fBOSC itself, two more settings shape what you actually call "theta" in the output, though 
they don't affect detection:

THETA_BAND (line 167) = (3, 7) Hz — used by the FOOOF/PSD side (Part 1) to pull out the theta peak, 
independent of fBOSC.
The FrequencyMean 3–7 Hz filter applied to fBOSC's output episode table (Part 2, cells 73/76) — this is a 
post-hoc selection of already-detected episodes by their mean frequency, now consistent everywhere after 
today's edit.