# Methods: LFP, running speed and spectrogram figures

Script: `LFP_Speed_Spectrogram_ScalingCorrected.py`

## 1. Overview

For each LFP channel recorded in each session, a four-panel figure was made:

| Panel | Content |
|---|---|
| A | Running speed of the animal over the whole recording |
| B | A 5 s stretch of the LFP, with the theta rhythm (3–7 Hz) drawn on top and coloured by theta phase |
| C | Spectrogram of the whole recording, 1–100 Hz |
| D | The same spectrogram, zoomed in to 1–20 Hz |

No part of the recording was removed. Instead, noise was reduced by filtering the whole trace (Section 4). The steps were adapted from a MATLAB pipeline (`SpeedSpectrogram_v2.m`), with the changes listed below.

## 2. Data

- **Animals:** Fa1059, Fa23BD, Fa5834 and Fa8477.
- **LFP:** Neuralynx `.ncs` files, sampled at 32,000 samples per second (32 kHz). One figure was made for every `.ncs` file in each animal's folder.
- **Position:** the tracking `.csv` file in the same folder as the `.ncs` file (time stamp, x and y in cm). If the folder held more than one `.csv`, the first one in alphabetical order was used.
- **Files left out** (logged as "SKIP" with the reason):
  - no tracking file in the folder,
  - a gap of more than 1 s in the LFP record time stamps,
  - an LFP trace shorter than 5 s,
  - tracking that does not overlap the LFP in time.

## 3. Loading the LFP

### 3.1 Placing every sample at its correct time

A `.ncs` file stores the signal in blocks ("records") of up to 512 samples. Each record has a time stamp and a count of how many of its 512 samples are real. Two timing errors occur if all records are simply joined end to end:

1. A partly filled record contains old, leftover values after its real samples.
2. Some samples are lost between records, so the next record starts later than the previous one ended.

To avoid both, each record's real samples were placed on an evenly spaced 32 kHz time grid, at the position given by that record's own time stamp:

```
position of a sample on the grid = (record time stamp − first time stamp) × 32,000 samples per second
                                   + position of the sample inside its record
```

- Leftover values in partly filled records were discarded.
- Empty places on the grid (lost samples) were filled by drawing a straight line between the real samples on either side (linear interpolation).
- If two records overlapped in time, the later record was kept.
- If a gap was longer than 1 s, the file was not used.

### 3.2 Conversion to microvolts

The file stores whole numbers (ADC counts). These were converted to microvolts with the scale factor written in the file header (`ADBitVolts`, volts per count):

```
LFP (µV) = stored value × ADBitVolts × 1,000,000
```

### 3.3 Marking saturated samples

The amplifier can only record up to a fixed voltage (its input range; 1,500 µV for Fa1059 and Fa23BD, 1,000 µV for Fa5834 and Fa8477). A signal larger than this is cut flat ("saturated" or "clipped"). Before any filtering, a raw sample was marked as saturated when:

```
|LFP sample| ≥ 0.99 × largest value the recorder can store
largest value the recorder can store = 32,767 × ADBitVolts × 1,000,000 µV
```

That is, any sample within 1 % of the limit. These marks were used only to choose the 5 s example (Section 7).

## 4. Noise reduction of the LFP

The steps below were applied, in this order, to the whole recording.

### 4.1 Lowering the sampling rate to 1 kHz

The trace was reduced from 32,000 to 1,000 samples per second by keeping one sample in every 32, after first removing all frequencies above 500 Hz with a low-pass filter (so that they could not fold back into lower frequencies). This was done with SciPy's `resample_poly` (a Kaiser-window FIR low-pass filter, built in).

The MATLAB pipeline used 250 Hz. 1 kHz was used here because, at 250 Hz, the wavelet step (Section 4.3) removed most of the 60–90 Hz power and created a false dark line at 75 Hz in the spectrogram. At 1 kHz neither happened.

### 4.2 Removing mains (electrical) noise

Mains electricity adds noise at 50 Hz and at its multiples. A narrow "notch" filter was applied at 50, 100, 150 and 200 Hz (all multiples below 500 Hz). Each notch has a quality factor Q = 30, which sets how narrow it is:

```
width of the notch = notch frequency ÷ Q
```

So the notch is about 1.7 Hz wide at 50 Hz, 3.3 Hz at 100 Hz, 5 Hz at 150 Hz and 6.7 Hz at 200 Hz. Each filter was run forwards and then backwards over the trace, so the signal is not shifted in time.

### 4.3 Wavelet denoising

The trace was split into frequency layers with a wavelet transform (wavelet "sym4", 4 levels). At 1 kHz the layers are:

| Layer | Frequencies (approx.) | Treated? |
|---|---|---|
| Detail 1 | 250–500 Hz | yes |
| Detail 2 | 125–250 Hz | yes |
| Detail 3 | 62.5–125 Hz | yes |
| Detail 4 | 31.25–62.5 Hz | yes |
| Approximation | 0–31.25 Hz (includes theta) | **no, kept unchanged** |

In each "detail" layer, small values (treated as noise) were removed and the rest were shrunk by the same amount. The cut-off was chosen by the BayesShrink rule:

```
noise level   = median of |values in Detail 1| ÷ 0.6745
signal level  = square root of ( average of (value²) in this layer − noise level² )
cut-off       = noise level² ÷ signal level
```

Each value in the layer was then changed as follows:

```
if |value| ≤ cut-off:   new value = 0
otherwise:              new value = value made smaller (towards 0) by the cut-off
```

If a layer had no signal above the noise level, the whole layer was set to 0. The layers were then put back together into one trace.

*Note:* MATLAB's `wdenoise` (used in the original pipeline) chooses its cut-off with a different rule (empirical Bayes). The same wavelet and number of levels were used, but the results are not identical.

### 4.4 Removing slow drift

A straight line was fitted to the whole trace and subtracted from it (linear detrend).

### 4.5 Replacing outliers

Single large spikes in the trace were found and replaced, following MATLAB's `filloutliers(x, 'linear', 'movmedian', …)`. For every sample, a 0.4 s window around it (400 samples) was used:

```
local median  = median of the 400 samples around this one
spread (MAD)  = 1.4826 × median of | each sample in the window − local median |
outlier if    | this sample − local median | > 3 × spread
```

(The factor 1.4826 makes the MAD comparable to a standard deviation for normally distributed data.) Each outlier was replaced by a straight line drawn between the nearest non-outlier samples on either side. The fraction of samples replaced was recorded for every file.

### 4.6 Smoothing (not applied)

The MATLAB pipeline also applied a median filter and a Savitzky–Golay smoothing filter. These were switched off here: on three test sessions they removed about 99 % of 30–60 Hz power and almost all 60–90 Hz power (while keeping about 98 % of theta power), which blanks the upper half of the 1–100 Hz spectrogram.

## 5. Running speed

### 5.1 Speed from position

For each pair of consecutive video frames:

```
distance moved = square root of ( (change in x)² + (change in y)² )      [cm]
time passed    = (time stamp of this frame − time stamp of previous frame) ÷ 1,000,000   [s]
speed          = distance moved ÷ time passed                             [cm/s]
```

The first frame was given a speed of 0, as was any frame whose time stamp did not increase. The speed was then smoothed with a Savitzky–Golay filter (a moving window of 5 frames, fitting a 3rd-order polynomial in each window), and any negative values produced by the smoothing were set to 0.

### 5.2 Lining up speed with the LFP

The tracking and the LFP use the same clock. Each frame's time was expressed in seconds from the first LFP sample:

```
frame time (s) = (frame time stamp − time stamp of the first LFP sample) ÷ 1,000,000
```

Frames before the start or after the end of the LFP were dropped.

### 5.3 Speed shown in panel A

For display only, the speed was smoothed further with a Gaussian (bell-shaped) window with a standard deviation of 0.25 s. The snippet choice (Section 7) and the median speed written above panel B use the speed from 5.1, without this extra smoothing.

## 6. Spectrogram

A spectrogram shows how much power the LFP has at each frequency, at each moment of the session. It was computed from the noise-reduced 1 kHz trace with SciPy's `signal.spectrogram`, using the same window settings as the power spectra (PSD) elsewhere in this analysis.

### 6.1 Cutting the trace into overlapping windows

- **Window length:** 2 s (2,000 samples).
- **Step between windows:** 0.5 s (500 samples). Neighbouring windows therefore overlap by 1.5 s (75 %).
- **Edges:** so that the first window is centred at the very start of the recording and the last at its end, 1 s of mirror-image signal was added at each end of the trace before cutting (reflect padding). The first and last second of each spectrogram are therefore partly based on mirrored data.
- **Time of a window:** the time of its centre, in seconds from the start of the LFP.

### 6.2 Power in each window

For each window:

1. **Remove the average.** The mean of the 2,000 samples was subtracted, so a constant offset does not appear as power at 0 Hz.
2. **Taper the edges.** Each sample was multiplied by a Hann window, a bell-shaped weight that is 0 at both ends and 1 in the middle:

   ```
   weight of sample i = 0.5 − 0.5 × cos(2π × i ÷ 2000),   i = 0, 1, …, 1999
   ```

   This stops the abrupt cut at the window edges from creating false power at other frequencies.
3. **Split into frequencies.** A Fourier transform (FFT) of the tapered window gives one value for each frequency.
4. **Turn into power per Hz** (power spectral density):

   ```
   power at frequency f = |Fourier value at f|² ÷ (1000 × sum of all weight²)      [µV²/Hz]
   ```

   Power at each frequency between 0 and 500 Hz was then doubled, to include the matching negative-frequency half of the transform (not done for 0 Hz and 500 Hz themselves).

The spacing between frequencies is:

```
frequency step = sampling rate ÷ window length in samples = 1000 ÷ 2000 = 0.5 Hz
```

So the frequencies are 0, 0.5, 1.0, 1.5, … 500 Hz, and the theta band 3–7 Hz contains 9 frequency values (3.0, 3.5, … 7.0 Hz).

### 6.3 Filling in the mains-noise frequencies

Any mains noise left after the notch filter, as well as the dip the notch itself leaves, was removed from each window's spectrum. For each of 50, 100, 150 and 200 Hz, the power from 2 Hz below to 2 Hz above (for example 48.0–52.0 Hz, 9 values) was replaced by a straight line joining the power just below and just above that range. This used FOOOF's `interpolate_spectrum`:

```
lower anchor = median of the 3 values just below the range      (e.g. 46.5, 47.0, 47.5 Hz)
upper anchor = median of the 3 values at the top of the range   (e.g. 52.0, 52.5, 53.0 Hz)
```

The line was drawn on log scales (log of frequency against log of power), which suits the way LFP power falls as frequency rises. This was done the same way in the power spectra (`Fig4a_PSD_v2_Simul_ScalingCorrected.py`).

### 6.4 Display (panels C and D)

- **Colour = power on a log scale:** `colour value = log10(power in µV²/Hz)`. One unit of colour value means 10 times more power.
- **Colour map:** viridis (dark blue = low power, yellow = high power).
- **Colour limits:** set separately for each panel, from the frequencies shown in that panel:

  ```
  darkest colour  = 1st percentile of all colour values in the panel
  brightest colour = 99th percentile of all colour values in the panel
  ```

  The brightest and darkest 1 % of values are shown at the end colours, so that a few extreme values do not wash out the rest. Because the limits are set per panel and per file, colours show where power is high or low *within* a panel; they cannot be compared between panels or files.
- **Panel C:** 1–100 Hz. **Panel D:** 1–20 Hz, from the same spectrogram.
- Each coloured tile is centred on its window's time and its frequency value.
- Two white dashed lines mark the start and end of the 5 s snippet shown in panel B.

## 7. Choosing the 5 s snippet (panel B)

### 7.1 Theta share

For one spectrogram window, the theta share is the fraction of its 1–100 Hz power that lies in the theta band:

```
theta share = (sum of power at 3–7 Hz) ÷ (sum of power at 1–100 Hz)
```

A value of 0.30 means that 30 % of the 1–100 Hz power is theta.

### 7.2 Score of a snippet

Candidate snippets were 5 s long and started every 0.5 s, from the start of the recording to 5 s before its end. Only the spectrogram windows that lie entirely inside the snippet were used: 7 windows, centred 1.0, 1.5, … 4.0 s after the snippet start.

```
score of a snippet = the lowest theta share among its 7 windows
```

Using the lowest value, not the average, means theta has to be present for the whole 5 s; a short, large burst of theta cannot win.

### 7.3 Which snippets could be chosen

In order of preference:

1. snippets with no saturated sample (Section 3.3) **and** a median speed of at least 4 cm/s (the moving threshold used elsewhere in the analysis);
2. if none: snippets with no saturated sample, regardless of speed;
3. if none: all snippets.

The snippet with the highest score in the first group that was not empty was shown. Which group it came from was written to the run log, so every file has a figure. (A start time can also be set by hand for a file; this was not done.)

### 7.4 Numbers shown above panel B

- **Median speed:** median of the speed (Section 5.1) over the frames inside the snippet.
- **Theta share:** theta power ÷ 1–100 Hz power of the *average* spectrum of the 7 windows:

  ```
  theta share shown = (sum of 3–7 Hz power over the 7 windows) ÷ (sum of 1–100 Hz power over the 7 windows)
  ```

  This is usually higher than the score, which is the lowest of the 7 window values.
- "SATURATED" is added if the snippet contains a saturated sample.

## 8. Theta rhythm and theta phase (panel B)

### 8.1 Theta-filtered trace

The noise-reduced trace was band-pass filtered at 3–7 Hz (5th-order Butterworth filter), run forwards and then backwards so that the theta wave is not shifted in time. The filter was applied to the whole recording, and the 5 s snippet was cut out afterwards, so the snippet is not affected by filter start-up effects at its edges.

### 8.2 Theta phase

The phase tells where in the theta cycle each moment lies. It was obtained with the Hilbert transform, which turns the theta wave into a rotating arrow; the phase is the angle of that arrow:

```
phase = angle of the Hilbert (analytic) signal of the theta trace        [radians, −π to +π]
```

- 0 = the peak of the theta wave
- ±π = the trough
- −π/2 = the rising part halfway between trough and peak; +π/2 = the falling part halfway between peak and trough

### 8.3 Display

The noise-reduced LFP is drawn in the animal's colour. The theta trace is drawn on top, half transparent, with its colour changing with phase. The colour map (hsv) is circular, so −π and +π have the same colour.

## 9. Figure layout

- **A:** speed over the whole recording (time 0 to the end of the LFP), with the 5 s snippet shaded in grey.
- **B:** the 5 s LFP snippet and its theta trace, with a phase colour bar.
- **C, D:** spectrograms of the whole recording, with colour bars in log10 µV²/Hz.

Animal colours (matplotlib "Pastel1"): Fa8477 red, Fa1059 blue, Fa23BD green, Fa5834 purple. Figures were saved as PNG (300 dpi) and SVG in a `LFP_Speed_Spectrogram` folder next to each `.ncs` file.

## 10. Record keeping

Each run writes `run_metadata.xlsx` to `LFP_Speed_Spectrogram_RunInfo` under the data folder, with three sheets:

- **code_provenance:** script name, SHA-256 fingerprint of the script, git commit and whether the script differs from it, run time, computer, and Python and package versions. A copy of the script is saved in `code_snapshot`.
- **settings:** every analysis setting.
- **processed_files:** every `.ncs` file, with its tracking file, duration, fraction of samples replaced as outliers, total saturated time, snippet start, which snippet criteria were met, theta share and score, or the reason it was skipped.

## 11. Software

Python 3.13 with NumPy 2.4, SciPy 1.18, pandas 2.3, matplotlib 3.11, PyWavelets 1.8 and FOOOF 1.1.1 (versions in the environment at the time of writing; the versions of a given run are in its `code_provenance` sheet).

## 12. Parameters

| Setting | Value |
|---|---|
| Raw sampling rate | 32,000 Hz |
| Sampling rate after downsampling | 1,000 Hz |
| Longest timestamp gap that is filled | 1 s |
| Saturation threshold | 99 % of the recorder's limit |
| Mains notch frequencies | 50, 100, 150, 200 Hz |
| Notch quality factor (Q) | 30 |
| Wavelet, levels | sym4, 4 (BayesShrink, soft cut-off) |
| Detrend | straight line, whole trace |
| Outlier window | 0.4 s (400 samples) |
| Outlier threshold | 3 × scaled MAD |
| Median / Savitzky–Golay smoothing of the LFP | off |
| Speed smoothing | Savitzky–Golay, 5 frames, 3rd order |
| Extra smoothing of plotted speed | Gaussian, SD 0.25 s |
| Spectrogram window | 2 s Hann (2,000 samples) |
| Spectrogram step | 0.5 s (75 % overlap) |
| Frequency step | 0.5 Hz |
| Mean removal per window | yes |
| Power units | µV²/Hz |
| Mains interpolation | ±2 Hz around 50, 100, 150, 200 Hz |
| Colour limits | 1st–99th percentile of log10 power, per panel |
| Theta band | 3–7 Hz |
| Band for theta share | 1–100 Hz |
| Theta filter | 5th-order Butterworth band-pass, forwards and backwards |
| Snippet length | 5 s |
| Snippet start spacing | 0.5 s |
| Moving threshold for snippet | median speed ≥ 4 cm/s |
