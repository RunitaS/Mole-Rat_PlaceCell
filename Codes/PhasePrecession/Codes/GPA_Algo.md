# Generalized Phase Approach Algo

## Main Algo

You filter the raw LFP into a frequency band of interest, turn it into a rotating-vector ("analytic") representation using the Hilbert transform so you can read off an instantaneous angle at every timestep, then patch the handful of moments where that angle briefly spins the wrong way (because the waveform's envelope collapsed toward zero) by smoothly interpolating through those bad moments instead of trusting the raw arctangent there.

Think of it like a clock hand that's supposed to sweep steadily forward. Most of the time it does. But every so often — when the signal amplitude dips almost to zero — the "hand" gets confused about which way it's pointing and briefly jitters or spins backward. GPA is a rule for detecting those brief confused moments and replacing them with a sensible, smoothly-interpolated sweep, without ever deleting the timestamp. <Check if phase reversals are caused by phase reset>

0. Start with the raw signal
You begin with a raw, wideband local field potential (LFP) trace, x_raw(t), sampled at some rate fs (e.g., ~32 kHz raw, often downsampled for LFP work). This raw trace contains all frequencies mixed together — theta, delta, gamma, noise, everything.

1. Bandpass filter into the band of interest
Before anything else, you isolate the frequency band you actually care about (e.g., 3–7 Hz for theta) with a zero-phase bandpass filter:

In this codebase: a 4th-order Butterworth filter in second-order-sections (SOS) form, applied forward-and-backward with sosfiltfilt (bandpass_filter).
Why zero-phase (filtfilt) matters: a normal (causal) filter shifts the phase of the signal. Since the entire point of this pipeline is to measure phase accurately, you cannot afford any filter-induced phase shift — so the signal is filtered once forward and once backward, which cancels all phase distortion (at the cost of doubling the effective filter order).
Call the result x(t) — this is the real-valued, band-limited signal that everything downstream operates on.
Why filter first, not just Hilbert-transform the raw wideband signal? The Hilbert transform's "instantaneous phase" only means something physically sensible for a signal that's (locally) close to a single oscillation. A wideband signal has many frequencies superimposed, so its "phase" would be a meaningless mush. Restricting to one band (e.g., theta) makes the phase interpretable as "where in the theta cycle are we right now."

2. Build the analytic signal with the Hilbert transform
This is the classical part, common to any instantaneous-phase estimate (not unique to GPA).

Intuition: A real oscillation like x(t) = cos(2πft) doesn't tell you unambiguously "where in the cycle" you are — cosine gives the same value on the way up and on the way down. To resolve that ambiguity you need a second signal that's 90° (a quarter cycle) out of phase, e.g. sin(2πft). Together, (cos, sin) act like (x, y) coordinates of a point going around a circle — from that pair you can read off the angle unambiguously. The Hilbert transform is the mathematical machine that manufactures that "90°-shifted twin" signal automatically from any real signal, so you don't need to already know its frequency.

Concretely (how scipy.signal.hilbert / the MATLAB original does it — the FFT method of Marple 1999):

Take the FFT of x(t), giving its frequency spectrum.

Zero out all the negative-frequency components.

Double the positive-frequency components (to conserve total energy after removing half the spectrum).

Keep the DC (0 Hz) and Nyquist components unscaled.

Take the inverse FFT. The result is a complex-valued signal:

<x_a(t) = x(t) + i·H[x(t)]>

where <H[x(t)]> (the actual "Hilbert transform" of x) is exactly that 90°-phase-shifted twin signal, and i is the imaginary unit combining them into one complex number per timestep.

This x_a(t) is called the analytic signal. At every instant it's a single point in the complex plane, and as time moves forward that point traces a path — ideally winding steadily around the origin once per cycle of the oscillation.

From the analytic signal, you get three quantities for free, at every single timestamp:

Instantaneous phase: — the angle of that point from the origin, wrapped into (−π, π]. This is "where in the theta cycle" you are right now (e.g., trough, rising, peak, falling).
φ(t) = <angle(x_a(t)) = atan2(Im(x_a(t)), Re(x_a(t)))>
Instantaneous amplitude / envelope: — the distance of that point from the origin, i.e., how "big" the oscillation is right now.
<A(t) = |x_a(t)| = sqrt(Re² + Im²)>
Instantaneous frequency: how fast the angle is sweeping forward, i.e., the time-derivative of the (unwrapped) phase divided by 2π (details in Step 5).
<f(t) = dt{φ(t)}/2π>
So far this is just plain Hilbert-transform analysis — no GPA correction yet.

3. Why plain Hilbert phase breaks down (the problem GPA fixes)
The clean picture above assumes the point traced by x_a(t) circles the origin at a roughly steady rate, like a clock hand. In real, noisy, non-perfectly-sinusoidal biological signals (a filtered theta LFP is not a perfect sine wave — it has asymmetric rise/fall shape, harmonics, notches, brief double-peaks), the traced path can occasionally pass very close to the origin (the envelope A(t) collapses near zero) or even loop the wrong way for an instant.

When the path passes near the origin, the angle φ(t) becomes poorly defined — a tiny wiggle in x(t) or H[x(t)] near that point causes a huge swing in the arctangent angle, sometimes making the phase appear to spin backward for a few samples. This is called a phase slip.

Because instantaneous frequency is just "how fast the phase is changing," a phase slip shows up as a wild, physically impossible instantaneous frequency spike — including negative frequencies (phase going backward) or absurdly large ones — even though the signal was filtered to (say) 3–7 Hz and can't truly contain anything outside that band. (Your repo's GPA_Algo.md notes an observed raw range of −1891 to +1609 Hz for a theta-band signal — obviously an artifact of a handful of near-zero-envelope samples, not real theta activity.)

This is exactly the artifact GPA was designed to correct.

4. The GPA correction algorithm, step by step
This is generalized_phase_vector. Given the filtered signal <x(t), sampling rate fs, the filter's low cutoff lp> (e.g. 3 Hz), and a <safety-margin multiplier nwin (default 3>, from the original MATLAB generalized_phase_vector.m):

    1. Compute the raw analytic signal and raw instantaneous frequency.

    xo = hilbert(x), φ_raw(t) = angle(xo), A(t) = |xo|.
    Raw instantaneous frequency wt_raw(t) = phase advance per sample, converted to Hz (see Step 5's formula) — this is the "PRE-GP" signal, the one full of spikes/negative values.
    2. Rectify the rotation direction.

    Check the average sign of the finite raw instantaneous frequencies. If it's negative overall (the analytic signal happens to be winding clockwise rather than counter-clockwise — a convention/sign ambiguity, not an error), flip the sign of the phase and rebuild xo, φ, A, wt_raw from the flipped version. This just guarantees "forward in time = phase increasing," a bookkeeping convention, before any artifact-detection happens.
    3. Flag "phase-slip" epochs.

    Mark every sample where wt_raw(t) < lp (the filter's own low-frequency edge) as untrustworthy. Note this single threshold catches both the physically-impossible negative frequencies and any positive-but-too-low frequency — anything below the band you filtered to isolate cannot be real.
    4. Extend each flagged run by a safety margin.

    For each contiguous run of flagged samples (found by connected-component labeling), extend it forward to nwin (=3) times its own original width.
    Why: the artifact doesn't cleanly start and stop exactly where the frequency crosses the threshold — the distortion "bleeds" into neighboring samples that still look superficially fine but are contaminated by the same collapsed-envelope event. Padding generously avoids leaving corrupted samples just outside the flagged zone.
    5. Unwrap the phase using only the trustworthy (unflagged) samples.

    np.unwrap is applied only to the valid subset of φ_raw, producing a continuously increasing (not wrapped to ±π) phase trend for the good stretches.
    Critically, the flagged samples are not included in this unwrap step — the code's own comment explains that unwrapping straight through a phase-slip epoch bakes a spurious fractional-cycle drift into the trend (because the raw phase genuinely wobbles there), which then corrupts the reconstruction. Skipping the bad samples entirely and reconstructing them afterward (next step) was empirically found to give correct results, while unwrapping through them did not.
    6. Reconstruct the flagged (bad) samples by shape-preserving interpolation.

    Using PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) — a shape-preserving interpolant that won't overshoot or oscillate the way a plain cubic spline might — fit a curve through the valid unwrapped-phase points, then evaluate that curve at the flagged (invalid) sample positions.
    This effectively draws a smooth, monotonically increasing phase ramp bridging the gap, using the trend on either side, instead of trusting the raw (and wrong) arctangent inside the gap.
    No samples are ever deleted: every timestamp — flagged or not — ends up with a phase value.
    7. Rewrap the reconstructed phase back into (−π, π].

    Standard modulo-2π wrapping, _gp_rewrap, converting the unwrapped (Step 5–6) trace back to a normal angle.
    8. Rebuild the corrected analytic signal.

    xgp(t) = A(t) · exp(i · φ_corrected(t)) — same envelope A(t) as before (amplitude was never in question — only phase was corrupted), now paired with the corrected phase.
    The function returns:

    xgp — the corrected analytic signal (same length as input, no gaps),
    wt_raw — the raw pre-correction instantaneous frequency (for diagnostics/plotting),
    idx — a boolean mask marking which samples were reconstructed (interpolated) rather than measured.
    That corrected phase, angle(xgp), is the "Generalized Phase." This is the number you actually use for everything downstream — spike-phase histograms, Theta Modulation Index, phase-precession regression, etc.

5. Getting instantaneous phase, frequency, and power from the (corrected) signal
Once you have xgp(t) = A(t)·e^{iφ(t)}:

Instantaneous phase


φ(t) = angle(xgp(t))   # wrapped to (-π, π]
This is what you read off directly at each spike time to ask "what theta phase did this spike fire at?"

Instantaneous frequency
Rather than a naive finite-difference of the wrapped phase (which would break every time phase wraps from +π to −π), the standard trick is to compute the phase advance between two consecutive samples directly as a complex-number operation, which handles wraparound automatically:


Δφ(t) = angle( xgp(t+1) · conj(xgp(t)) )      # phase advance between adjacent samples
f(t)  = Δφ(t) / (2π · dt)                     # convert radians/sample to Hz, dt = 1/fs
(conj = complex conjugate; multiplying by the conjugate of the previous sample and taking the angle of the product gives exactly the phase difference, correctly wrapped.)

For quality control, this codebase then goes a step further and blanks out (NaNs) any frequency estimate that:

spans a reconstructed sample (i.e., either of the two adjacent samples used in the difference was interpolated in Step 6) — because a pchip fill gives a believable angle but not a believable rate of change, so any frequency computed across that boundary is discarded rather than reported;
falls outside the original filter's own passband (e.g., outside 3–7 Hz) even among "trustworthy" samples — since a signal that was bandlimited to 3–7 Hz cannot truthfully have an instantaneous frequency of, say, 10 Hz; a value like that means the estimate itself (not necessarily the underlying data) is unreliable, often right at the filter's transition band.
This masked frequency is only a debugging/QC signal ("did GPA produce physically sensible frequencies?") — it is not the same thing as the corrected phase itself, which remains defined and usable at every sample, reconstructed or not.

Instantaneous power
Power is simply the squared envelope of the analytic signal:


P(t) = A(t)² = |xgp(t)|²
Since the envelope A(t) was never touched by the GPA phase correction (only the angle was reconstructed in flagged epochs — the magnitude was carried through unchanged from the original Hilbert transform), instantaneous power is computed exactly the same way whether or not you've applied the GP correction.

### Summary flow

raw wideband LFP
      │  bandpass filter (zero-phase Butterworth, e.g. 3–7 Hz)
      ▼
filtered signal x(t)
      │  Hilbert transform  →  analytic signal xo(t) = A(t)·e^{iφ_raw(t)}
      ▼
raw phase φ_raw, raw instantaneous frequency wt_raw
      │  flag samples where wt_raw < filter's low cutoff
      │  extend flagged runs ×3
      │  unwrap valid samples only → PCHIP-interpolate across flagged gaps
      │  rewrap
      ▼
corrected phase φ_gp  →  xgp(t) = A(t)·e^{iφ_gp(t)}     ("Generalized Phase")
      │
      ├── instantaneous phase:      angle(xgp)                     — used at every timestamp/spike
      ├── instantaneous frequency:  angle(xgp[t+1]·conj(xgp[t]))/(2π·dt),
      │                             then QC-masked (drop reconstructed-adjacent
      │                             and out-of-band estimates) for diagnostics
      └── instantaneous power:      |xgp(t)|² = A(t)²
The key conceptual point worth remembering: GPA never removes data. It only recognizes the brief moments where the plain Hilbert phase is untrustworthy (because the signal's envelope collapsed) and replaces the angle there with a smooth, sensible interpolation — while power/amplitude and everything outside those brief windows is untouched.


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## GPA unwrap correction
Two different things are happening — correction of the phase, and masking of the frequency estimate used for QC
The plots you're looking at are not showing what GPA does to the signal; they're showing what happens when you take the derivative of GPA's corrected phase and then deliberately hide the parts of that derivative that can't be trusted. Those are two separate steps in Debug_GPA_PhaseValues_ThetaMod_PhasePrec_SpikeLFP_Matched_v9.py, and conflating them is easy to do from the histograms alone.

1. Why raw Hilbert phase produces those wild negative frequencies (PRE-GP panel)
np.angle(hilbert(x)) computes phase as a plain four-quadrant arctangent every sample. Instantaneous frequency is just the sample-to-sample phase difference. When the filtered waveform has a brief local double-peak or notch, the analytic signal's envelope passes close to zero and the phase can momentarily spin backward — a "phase slip." That's exactly why your PRE-GP histogram has a "true range" of [-1891, +1609] Hz for a signal that's supposed to be theta (3–7 Hz): a handful of near-zero-envelope samples produce huge, meaningless instantaneous-frequency spikes in either direction.

2. How generalized_phase_vector (Davis, Muller et al. 2020) corrects the phase itself
This is generalized_phase_vector, lines 476–551:

Take the normal Hilbert analytic signal, get its raw phase and instantaneous frequency (wt_raw) — this is what's plotted as "PRE-GP."
Flag every sample where wt_raw < lp (line 506) — lp is the lower edge of the bandpass filter (3 Hz here). Any frequency below that — including negative ones — counts as corrupted, not just literally-negative ones.
Extend each flagged run by nwin=3× its own width (lines 508–513) — a safety margin, because the artifact bleeds into neighboring "clean-looking" samples too.
Unwrap the phase using only the untouched/valid samples, then reconstruct the unwrapped phase across each flagged gap by shape-preserving (pchip) interpolation between the valid phase trend on either side (lines 538–546). This effectively draws a smooth, monotonically-increasing phase ramp through the bad epoch instead of trusting the arctangent there.
Rewrap back into (−π, π] and rebuild the corrected analytic signal xgp.
Critically: no samples are deleted here. xgp is defined at every single time point, same length as the input. The signal/phase trace used for everything downstream (spike-phase assignment, TMI, Pass Index phase precession) still has full temporal coverage — the bad stretch's phase is estimated rather than measured, but it's still there and still used.

3. Why the POST-GP histogram nonetheless looks like it lost samples
That's gp_instantaneous_frequency, a completely separate, debugging-only computation. It recomputes frequency from xgp, then actively throws out (sets to NaN) any frequency estimate that:

spans a reconstructed sample (idx[i] or idx[i+1] True, line 580) — because a pchip-filled phase gives a plausible value but not a plausible rate of change; a frequency estimate needs two adjacent samples, so if either one was fabricated, the resulting "frequency" is meaningless and is discarded, not just softened.
falls outside the filter's own passband (lines 582–584) even among the "trustworthy" samples — the logic being that a signal bandpassed to 3–7 Hz simply cannot have a real instantaneous frequency of, say, 10 Hz; if the estimate says otherwise, the estimate itself is unreliable (e.g., filter transition-band edge effects), so it's excluded rather than plotted as if real.
Then the plotting function (plot_gp_instantaneous_frequency) does finite_wt = wt[np.isfinite(wt)] before histogramming — so every one of those NaN'd estimates simply doesn't appear in the POST-GP bar count. That's why the POST-GP peak count (~720k) is visibly lower than PRE-GP (~800k): it isn't that the underlying LFP epoch was excised from the recording, it's that this particular diagnostic refuses to report a frequency number for samples it can't vouch for.

### Summary
GPA's actual output — the corrected phase xgp — never drops time samples. It patches the corrupted stretches with an interpolated, monotonically sensible phase so downstream phase-based analyses (spike-phase histograms, TMI, phase precession) have a continuous, artifact-free phase to read at every timestamp.
What you're seeing "disappear" in the graph is only the instantaneous-frequency QC metric, which is intentionally more conservative than the phase itself: it hides any frequency number derived even partly from reconstructed phase, and additionally clips anything outside the filter band, because — as the code's own comment puts it — you can read a reconstructed sample's wrapped phase safely, but you should never trust its rate of change.
The result (tight histogram sitting inside [3, 7] Hz) is the intended success signal: it shows that wherever GPA does report a frequency, that frequency is physically consistent with the passband — not that data was thrown away.