# Matching Sinusoid Frequencies with Autocorrelograms: The Theory

How to check whether two sinusoids share a frequency by computing autocorrelograms of 1-s epochs and taking the Euclidean distance between them.

## 1. Why an autocorrelogram reveals frequency, and nothing else

Take a sinusoid x(t) = A·sin(2πft + φ). Its autocorrelation at lag τ is

R(τ) = (A²/2)·cos(2πfτ)

The phase φ drops out completely, and the amplitude A only scales the curve. If you normalise so that R(0) = 1, you get

ρ(τ) = cos(2πfτ)

which depends on frequency alone. Two sinusoids at the same frequency produce the same normalised autocorrelogram, however different their phases or amplitudes. That is why comparing autocorrelograms is a good frequency test: it ignores everything you don't care about.

This is the Wiener–Khinchin theorem at work. The autocorrelation is the Fourier transform of the power spectrum, so comparing autocorrelograms is a phase-blind way of comparing power spectra.

## 2. Why chunk into 1-s epochs and average

You only ever have a finite-length estimate of R(τ), and a single estimate is noisy. Cutting the signal into epochs, computing an autocorrelogram for each, and averaging them reduces the variance of the estimate. This is the same idea as Bartlett's or Welch's averaging for spectra. It also makes the method tolerant of slow drifts or non-stationarity, since each epoch is treated on its own.

The epoch length T sets two limits:

- **Maximum lag.** You can only estimate lags up to T, and lags near T rest on very few samples, so in practice you use lags up to some L < T (for example half the epoch).
- **Frequency resolution.** Two frequencies are distinguishable only once their cosines have drifted noticeably out of step within the lag window. That needs roughly Δf ≳ 1/L. With 1-s epochs you can comfortably separate frequencies about 1 Hz apart, but differences of a few tenths of a Hz give only small distances.

You also need several cycles per epoch, so a 1-s window works well for, say, 5 Hz and up, and poorly near 1 Hz.

The usual "biased" estimator multiplies the autocorrelogram by a triangular taper (1 − |τ|/T). Because both signals are epoched identically, the taper is the same on both sides and doesn't create a false difference.

## 3. Noise, and why you drop lag 0

Real signals are sinusoid plus noise. For a sinusoid with power S plus white noise with power N, the normalised autocorrelation becomes

ρ(τ) = [N/(S+N)]·δ(τ) + [S/(S+N)]·cos(2πfτ)

The noise contributes only a spike at lag 0. Beyond lag 0, the oscillation is still cos(2πfτ), just shrunk by the signal-to-noise ratio. So two signals with the same frequency but different SNRs would show a nonzero Euclidean distance purely because of amplitude.

The fix is to discard lag 0 and then scale each autocorrelogram to unit length before comparing them. What's left is the shape of the curve, which encodes frequency.

## 4. What the Euclidean distance actually measures

After that normalisation, the squared Euclidean distance between two vectors a and b is

D² = 2·(1 − r)

where r is their correlation (cosine similarity). So Euclidean distance on unit-normalised autocorrelograms is equivalent to correlating them.

For two pure cosines at frequencies f₁ and f₂, compared over lags 0 to L, you can work out the distance analytically:

D² ≈ 2·[1 − sinc(2π·Δf·L)], where sinc(x) = sin(x)/x and Δf = f₁ − f₂

This tells you:

- **D = 0 only when Δf = 0.** Matching frequencies give zero distance, apart from estimation noise.
- **For small mismatches, D grows linearly:** D ≈ 2π·Δf·L/√3. Doubling the lag window doubles your sensitivity.
- **D saturates near √2** once the two cosines are uncorrelated over the window, at roughly Δf ≈ 1/L.
- **D is not monotonic in Δf.** The sinc oscillates, so D wobbles around √2 for larger mismatches. Treat it as a same-or-different measure, not a ruler for how far apart the frequencies are.

As a check, a simulation with lags up to 0.5 s gave the following. The formula predicts D ≈ 0.53 for a 0.3 Hz mismatch and ≈ 1.41 for 1 Hz. The simulation measured 0.46 and 1.22, slightly lower because the taper down-weights the long lags.

## 5. Deciding whether the frequencies "match"

A distance alone doesn't tell you whether it is small enough, because estimation noise always makes it a bit above zero. You need a reference for how big D gets by chance:

- **Within-signal floor.** Split one signal's epochs into two random halves and compute D between them, many times. That gives the distance you'd expect from noise alone when the frequency is identical.
- **Permutation test.** Pool the epochs from both signals, shuffle which signal each epoch is assigned to, and recompute D. If the observed D is larger than nearly all shuffled values, the two sets of autocorrelograms differ.
- **Bootstrap.** Resample epochs with replacement to put a confidence interval on D.

The permutation test has a catch. Its null hypothesis is that the epochs are interchangeable, not that the frequencies match. Differences in SNR or noise colour can make it reject even when the frequencies are identical, especially with many epochs. In the simulation, a 10 Hz signal compared with another 10 Hz signal at lower SNR gave a tiny distance (0.09) but was still "significant," just like the 10.3 Hz signal (0.46). So report the size of D alongside any p-value, and judge it against the within-signal floor or a tolerance you decide in advance, for example the D the sinc formula predicts for the smallest Δf you care about.

## 6. Limits worth keeping in mind

- **Multiple components.** If a signal has harmonics or several oscillations, its autocorrelogram is a sum of cosines weighted by their power. The distance then compares whole spectra, so a shared fundamental with different harmonic content will still give a nonzero distance.
- **It's the spectrum in disguise.** Since the autocorrelogram and the power spectrum carry the same information, comparing spectral peaks directly is often more interpretable. The autocorrelogram approach is useful when you want a single number summarising overall rhythmic similarity.
