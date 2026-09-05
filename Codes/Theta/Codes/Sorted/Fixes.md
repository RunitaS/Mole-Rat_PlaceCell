#%% fBOSC fixes

- [ ] **Remove delta/theta flter from pipeline completely. Analysis should be run on minimal filtered data**

FOOOF WARNING: Lower-bound peak width limit is < or ~= the frequency resolution: 1.00 <= 1.00
        Lower bounds below frequency-resolution have no effect (effective lower bound is the frequency resolution).
        Too low a limit may lead to overfitting noise as small bandwidth peaks.
        We recommend a lower bound of approximately 2x the frequency resolution.

1. The background/threshold fit is running on a very sparse frequency grid (1–20 Hz, 1 Hz steps = ~20 points).
This is likely why changing aperiodic_mode and threshold.percentile barely helped — both only change how the aperiodic curve bends through the data, but with only ~20 points FOOOF's fit is poorly constrained regardless of mode, and can land close to (or through) real theta power no matter what shape you allow it. FOOOF's own guidance assumes a reasonably dense spectrum; 1 Hz resolution over a 20 Hz span is coarse for separating a periodic peak from the aperiodic component. Try running the background fit at finer resolution (e.g. 0.25–0.5 Hz steps, or a longer/denser F_array) — this changes the input data to the fit rather than its shape, which is the piece you haven't touched yet.

2. wavenumber = 6 — the classic eBOSC time/frequency tradeoff knob, and per the original paper, one of the main levers controlling single-trial sensitivity.
A higher wavenumber gives better frequency precision but a longer, smoother wavelet in time — meaning power take longer to build up and decay slower at bout onset/offset, and instantaneous power estimates are more smeared/averaged. If your theta bouts are short or their amplitude fluctuates quickly, a high wavenumber can make the power trace dip below pt more often than the underlying signal actually does. Try lowering it (e.g. 3–4) as a test.

3. postproc.use = True (FWHM) — check whether disabling postproc entirely changes things, not just switching FWHM↔MaxBias.
Notably, the reference eBOSC example script (eBOSC_example_empirical.m in jkosciessa/eBOSC) ships with postproc.use = 'no' by default — postproc is optional refinement, not a required step. FWHM/MaxBias both shrink raw detections down to a "core," and this shrinkage is exactly what produces your orange "candidate (dropped)" category (pre-postproc detected, post-postproc gone). Run once with FBOSC_POSTPROC = False to see how much of your drop is attributable purely to this trimming step, independent of the power/duration thresholds. Also worth knowing: Kosciessa et al. found FWHM gave lower specificity (i.e., is the more permissive of the two methods) in their simulations, and used MaxBias for their main analyses — so if anything, FWHM should already be your more inclusive choice; the "no postproc at all" test is the more informative one.

4. threshold.duration (3 cycles) interacting with real amplitude modulation.
Rodent theta amplitude isn't constant — it modulates within a bout (often on the ~0.5–1 s scale). If your bouts' power genuinely dips below pt faster than 3 cycles can bridge, no percentile change will fix it, only a shorter min_ncycles/min_duration_s will. Worth a quick test (e.g. 2 cycles).

5. Confirm FOOOF is actually flagging theta as a peak in the background fit, not just fitting through it.
Your _fbosc_fit.png figures already annotate "Theta range: X–Y Hz" when a peak is found within theta_band. Pull up that plot for a file where you're seeing lots of missed theta — if that annotation is absent, FOOOF isn't identifying a periodic component there at all (regardless of aperiodic_mode), so its full power is being counted as "background," inflating pt. That would point at peak_width_limits/peak_threshold/min_peak_height in cfg_fBOSC["fooof"], not the aperiodic mode — which fits with why changing aperiodic_mode alone didn't help.

One caveat from the literature itself: Kosciessa et al.'s own simulations report that eBOSC/BOSC-family detection has "strongly impaired sensitivity when rhythmic power is low" relative to background — i.e., precision/specificity is prioritized over recall by design. If your rodent LFP theta SNR (relative to a noisy, movement-heavy broadband background) sits in that low-SNR regime, you may be running into a real ceiling of this method rather than a single wrong setting — worth keeping in mind if #1–#5 only produce partial improvement.

I'd start with #1 (frequency resolution of the background fit) and #3 (postproc off) — those are the two you haven't touched yet and each is testable on a single file in a couple minutes via fBOSCpy_wrapper_v2 directly, without rerunning the whole batch.



My strategy:

Try out MaxBias method instead of FWHM, if that doesn't work out, switch off post-processing altogether.
Too many episode dropper. Try switching off post_proc to see if that solves dropped episodes problem.

Turning off post proc increased number of episode.

Changing resolution from 1Hz to 0.1Hz also increased the number of episodes.

Note: Lower the wave number, cleaner the theta detection. Power threshold probbly too high. Higher time resoltuion. Increasing the number of waveforms from 6 to 8 and then 12 also increased the number of episodes, however, 12 might be a little too high as it is also inflating the episodes in alpha/beta range. Leave it as 8 for now. 
3 had cleanest theta detection but lowest episode rate.

_____________________________________________________________________________________________________________________
# Bycycle fixes:

Fixes for Bycycle:

## To-Do List for Bycycle Code Cleaning Procedure

- [ ] ** What cleaning procedure is applied to the FOOOF pipeline in Bycycle code? ** Notch > Detrend > Hampel

- [ ] ** Remove outliers. ** added hampel

- [ ] **Replicate Cole 2018 hippocampal theta method. Generate cde from scratch. Current code isn't different enough. It has only modified the preexisting pipeline.**

- [ ] **Apply Bycycle only to epochs when the animal is moving.**

- [ ] ** Fix ADBitsVolts conversion. Unit is not in  microV **

- [ ] ** Plot % change for each parameter at a range of different values.** 
        * Check FIR kernel overlapping window. Maybe the cause for inversion of theta cycle increase with increase in low pass freq (ideally it should have decreased due to 50 Hz noise and fast gamma) *
        * The cause for inversion was low monotonicity. When too low it forgives the line noise and fast gamma overriding the theta. For soem reason this was increasing the theta when low pass freq was increased. *

- [ ] ** Plot the trace moving window. **

- [ ] ** Make sure your signal is detrended. Zero passing line will be wrong if the signal is not detrended. **

- [ ] ** How is FOOOF output used in Bycycle ** 
        * It wasn't used before. Applied the changes now. The signal only uses LFP race devoid of aperiodic noise.
        * Use TRUE/FALSE flags on BYCYCLE_USE_FOOOF_THETA, BYCYCLE_GATE_ON_FOOOF_QUALITY to control this.

- [ ] ** Why does the bandpass filtered signal gain sinusoidality? ** Because Nonsinusoidality is caused by freq + it's octave. Bandpasing the signal removes the harmonics making a non sinusoidal signal sinusoidal. Hilbert vs interpolation sinusoidality may be different in peak and trough periods!

- [ ] **Why TF is my theta so sinusoidal? It should have had sinusoidality. Look at your sawtooth code again. What method were you using there?**
- [ ] **Compare sinusoidality with mouse hippocampus data. Could be due to epilectpic brain.**

- [ ] **compare theta asymmetries between animals.(volt_amp,      period,  time_rdsym,  time_ptsym)** 

- [ ] **Apply to neuropixel data to check for layerwise asymmetry in theta waveform**


# Theta sinusoidality:

Theta shape for different species will be different? Depends on input patterns.

What makes hippocampal theta non-sinusoidal in the first place

A sinusoid is the single "purest" oscillation — one frequency, smooth, symmetric rise and fall. Hippocampal theta departs from that in two well-documented ways: it's rise-decay asymmetric (often described as sawtooth-like, with one flank steeper than the other) and it can be peak-trough asymmetric (sharper troughs than peaks, or vice versa, depending on the recording site and reference). The waveform genuinely isn't a clean sine wave.

The reasons are biophysical, and they're the interesting part:

It's the sum of spatially distributed synaptic currents with different timing. The theta-band LFP at any electrode is a mixture of 
transmembrane currents from multiple pathways that are active at systematically different phases. Entorhinal input (via the perforant path) 
and the CA3 Schaffer-collateral input arrive at different dendritic layers and at different theta phases; each contributes a current 
sink/source with its own time course. When you sum inputs that rise and fall on different schedules, the composite waveform is not 
sinusoidal — the flanks become unequal. The steep flank corresponds to the moment when the dominant synaptic current is being driven 
hard; the shallow flank to when it's passively decaying. So the asymmetry is essentially a readout of the sequence of synaptic events 
within each theta cycle.

It rides on top of intrinsically non-sinusoidal cellular events. Theta is temporally organized around spiking and around fast dendritic 
events, and the LFP contains contributions that are not smooth: the currents underlying a sink build and collapse asymmetrically. 
The membrane's own filtering and the geometry of the pyramidal-cell layer shape the extracellular field into something with sharper 
transients than a sine.

Volume conduction and layer geometry. The waveform recorded at any site is a weighted, distance-attenuated sum of all these dipoles. 
The exact shape — and even the direction of the asymmetry — changes with electrode depth relative to the pyramidal layer, which is 
why the asymmetry "flips" as you move the electrode through the stratum. This is also why your three animals showed asymmetries in 
different directions earlier: plausibly just electrode placement relative to the layer.

The important conceptual point: this non-sinusoidal shape is information. The asymmetry reflects the balance and timing of the 
underlying inputs, and it changes with behavior and pathology. That's precisely why you'd want to measure it rather than filter it away.

Why a bandpass filter erases that shape

Now the first question, which follows directly. Here's the key fact about how a non-sinusoidal periodic waveform is represented in 
the frequency domain:

Any non-sinusoidal but periodic waveform is, by Fourier's theorem, a sum of sinusoids: a fundamental at the base frequency plus 
harmonics 
at integer multiples. A 6 Hz sawtooth is not "6 Hz energy shaped like a sawtooth" — it is genuinely 6 Hz + 12 Hz + 18 Hz + ... 
sinusoidal components, in specific amplitude and phase relationships. Those harmonics are the sharp flanks and asymmetry. The 
steepness of the fast flank, the sharpness of the trough — all of it is encoded in the higher harmonics and, crucially, in their 
phase alignment with the fundamental.

A bandpass filter set to, say, 3–7 Hz does two destructive things:

It removes the harmonics. The 12 Hz, 18 Hz components that encode the asymmetry lie outside the theta passband, so the filter deletes 
them. What survives is only the fundamental — and a single sinusoidal component is, by definition, a sinusoid. You have literally 
thrown away the parts of the signal that made it non-sinusoidal. The output must look like a sine wave because you've kept only the 
sinusoidal fundamental.

It scrambles what little it keeps. Filters impose phase shifts that vary with frequency. Even a small amount of leaked harmonic content 
is phase-distorted relative to the fundamental, so the specific phase alignment that produced the original shape is destroyed. Waveform 
shape lives in cross-frequency phase relationships, and that is exactly what a filter disturbs.

So narrowband filtering doesn't "measure theta cleanly" — it manufactures a sinusoid from whatever was there. If you then ask "is the 
filtered signal sinusoidal?", the answer is trivially yes, but it's a property of the filter, not of the brain. This is the circularity 
I flagged earlier and the core motivation for bycycle: by working on the broadband (only lightly low-passed) signal and measuring extrema 
and flanks in the time domain, it keeps the harmonics in place so the real shape survives to be quantified.

Two clarifications worth having:

The distinction between "true non-sinusoidal shape" and "harmonics" is somewhat semantic — they're the same phenomenon viewed in two 
domains. Sharp waveform = harmonic content, always. Where it gets subtle is that harmonics in a power spectrum can also arise from a 
separate, independent oscillator that happens to sit at a multiple of theta; distinguishing genuine waveform-shape harmonics 
(phase-locked to the fundamental) from independent rhythms is exactly what phase-coupling measures like bicoherence are for. So if you 
want to confirm that your theta asymmetry is real shape rather than a coincidental second rhythm, bicoherence between the fundamental 
and its harmonics is the complementary check.

And this is why bycycle uses its bandpass only to find the extrema, then measures features on the low-passed-but-not-narrowband 
signal — a narrow band is fine for the modest job of locating where peaks and troughs are, but you never let it touch the signal 
whose shape you actually care about.

# Bycycle parameters:

Signal conditioning / extrema localization

BYCYCLE_F_THETA = (3.0, 7.0) — This is the bandpass applied only to localize peaks and troughs: bycycle filters the signal in this band, 
finds zero-crossings, then locates the actual extremum in the low-passed signal between each pair of crossings. It defines the timescale 
of a "cycle" and does not filter the signal that shape features are measured on. Two effects matter for theta. First, a narrow band 
forces the localization signal toward a sinusoid, which can mislocate the extrema of genuinely asymmetric (sawtooth) theta and bias your 
rdsym/ptsym back toward 0.5 — possibly part of why your symmetry values looked so flat. Second, and more consequential for the burst rate: 
3–7 Hz may simply be the wrong band. Rodent locomotor theta frequently runs 7–10 Hz; if the animal's dominant theta sits above 7 Hz, this 
band attenuates it, the zero-crossings get ragged, and you find few clean consistent cycles — which reads out as a low burst fraction. 
If your species/state has fast running theta, widening to ~(4,10) or (5,10) is worth testing.

BYCYCLE_LOWPASS_HZ = 20.0 — A low-pass applied before extrema detection so that faster content (gamma, spikes, sharp-wave ripples, EMG) 
doesn't create spurious local extrema or make each cycle jagged. This directly interacts with the monotonicity gate below. Raising the 
cutoff lets more high-frequency content through → rougher cycles → lower monotonicity → fewer bursts. Lowering it smooths cycles → higher 
monotonicity → more bursts, but at the cost of erasing theta's higher harmonics, which are precisely what make the waveform non-sinusoidal.
At 20 Hz you keep roughly up to the 3rd harmonic of a 6 Hz rhythm (~18 Hz), so you preserve some shape while removing gamma. Your note 
about dropping it from 25 to 20 would slightly smooth the signal — expect marginally higher monotonicity/burst rate and marginally less 
measured asymmetry.

BYCYCLE_FILTER_SECONDS = 0.5 — The length (in seconds) of that low-pass FIR kernel. Longer kernels give a sharper cutoff (cleaner 
separation of theta from faster content) but more temporal smearing and edge effects; shorter kernels have a broader transition band 
and leak more high frequency through. At 0.5 s the transition band is roughly a couple of Hz, which is a sensible sharpness for a 20 Hz 
cutoff. This mostly tunes how cleanly the low-pass does its job; it's not usually a make-or-break parameter unless it's set very short.

BYCYCLE_CENTER_EXTREMA = 'trough' — This sets whether cycles are centered on peaks (segmented trough-to-trough) or troughs (segmented 
peak-to-peak). Your inline comment says "trough… hippocampal theta convention," but the value is 'peak' — worth reconciling, because it 
changes the reference point for your shape features and thus how you interpret time_rdsym/time_ptsym. The effect on which cycles exist 
and on the burst fraction is modest (cycles are just offset by half a period), but it shifts the flank grouping, so specific cycles' 
consistency/monotonicity values move a little. The bigger issue is interpretive: if you're comparing your asymmetry direction to 
published hippocampal theta (usually trough-centered), a peak-centered analysis can look flipped. Pick one, match it to the literature 
you're comparing against, and keep the comment and the code consistent.

Burst-detection thresholds

A cycle is flagged is_burst only if it, together with enough neighbors, clears all four gates. Since amp_fraction is off (below), your 
1.2% is being driven entirely by consistency + monotonicity + the run-length requirement.

amp_fraction_threshold = 0.0 — Each cycle's amplitude is expressed as its fractional rank within the whole recording's amplitude 
distribution; a cycle passes if that fraction ≥ threshold. At 0.0 this gate is disabled — even the tiniest, noise-level cycles are 
amplitude-eligible. This is important for reading your result: the low burst rate is not because low-amplitude junk is being excluded. 
It also means you have no protection against calling low-amplitude non-oscillatory wiggles "theta" once they happen to be consistent. 
For theta you often want a modest value here (say 0.2–0.4) so that flat, low-power stretches are excluded on amplitude grounds; leaving 
it at 0 shifts all the selectivity onto the other three gates.

amp_consistency_threshold = 0.6 — Measures how similar the rise/decay amplitudes are between a cycle and its neighbors, as a ratio 
bounded 0–1 (1 = identical amplitudes cycle-to-cycle). 0.6 requires adjacent cycles' amplitudes to be within ~60% of each other. 
Raising it demands a steadier envelope (fewer, cleaner bursts); lowering it tolerates waxing/waning amplitude. 0.6 is moderate and 
usually not the main bottleneck for theta, which tends to have a reasonably stable envelope during a bout.

period_consistency_threshold = 0.75 — Ratio of adjacent cycle periods (shorter/longer), so 0.75 means consecutive periods must be within 
75% of each other — i.e., the rhythm must stay fairly regular from cycle to cycle. Real theta is quite periodic, so during genuine bouts 
this should pass. But if your f_range is mismatched (previous point) and the "cycles" are partly noise-driven, periods jitter and this 
gate rejects them. Raising toward 0.8–0.9 enforces metronomic regularity; lowering tolerates frequency drift 
(e.g., accelerating/decelerating theta during movement transitions).

monotonicity_threshold = 0.8 — The fraction of samples within a cycle where the signal moves in the "correct" direction 
(rising throughout the rise flank, falling throughout the decay). 1.0 is a perfectly smooth cycle; 0.8 requires 80% clean motion. 
This is usually the gate that most punishes residual high-frequency content, and given amp_fraction = 0, it's a prime suspect for 
your 1.2%. Any beta/low-gamma surviving the 20 Hz low-pass adds little bumps to each cycle, drops monotonicity below 0.8, and the 
cycle is rejected even if it's genuinely rhythmic. Loosening to 0.6–0.7, or lowering the low-pass cutoff, both raise the burst 
count — but loosening too far starts admitting non-oscillatory noise. This is the classic tension: monotonicity and the low-pass 
together trade sensitivity against false positives.

min_n_cycles = 3 — At least 3 consecutive cycles must clear all the above gates before any of them is labeled a burst; brief 
1
–2 cycle blips are discarded. This is the standard, sensible value. Raising it (4–5) demands longer sustained rhythmicity and will 
lower your burst fraction further — relevant if your theta is truly intermittent, since short real bouts get thrown out. Lowering to 
2 admits briefer events.

Putting it against your 1.2%

With amplitude gating off, three things most plausibly explain such a low burst rate, in rough priority order: the f_range may miss 
faster theta (check whether the animal's theta is really ≤7 Hz, ideally against the LFP power spectrum); the monotonicity 0.8 gate is 
rejecting cycles roughened by content still present below 20 Hz; and it may partly be genuine if the recording is dominated by non-theta 
states (quiet immobility, non-REM sleep, anesthesia). The way to disambiguate is quick: overlay is_burst on a few raw segments where you 
know theta is present (e.g., during running, if you have position/velocity), and check the periods in Hz. If clearly-theta segments 
aren't being flagged, relax monotonicity and/or widen f_range and re-run; if they are flagged and the rest of the recording genuinely 
isn't theta, then 1.2% is real and simply reflects the behavioral composition of your data.

