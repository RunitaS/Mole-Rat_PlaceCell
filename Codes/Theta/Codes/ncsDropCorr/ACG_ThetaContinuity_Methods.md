# Theta continuity analysis (autocorrelogram peak range): plain-language description

This document explains, step by step, what the script
`Figc_ACG_ScalingCorrected_v11_SaturationReject.py` does. Formulas are written as plain
text (no special maths formatting) and every symbol is explained in words.

---

## 0. The big picture

**The question.** Is theta rhythm present all the time, or only while the animal is
moving? To answer this, we cut each LFP recording into short 1-second pieces
("epochs"), and for every epoch we ask two things:

1. **Which frequency is the main rhythm in this epoch?** If it lies between 3 and 7 Hz,
   the epoch counts as a *theta epoch*; otherwise it is a *non-theta epoch*.
2. **How regular (clean, sine-like) is that rhythm?** This is a number called the
   *normalised peak range*. It is close to 1 for a clean, regular oscillation and
   close to 0 when there is no regular oscillation.

We then compare these results between epochs where the animal was moving and epochs
where it was still, and between arenas and session types.

**Where the method comes from.** The approach is taken from Dunn et al. (2022), who used
it in ferrets. We translated their MATLAB code (`quantify_xcorr_epochs.m` and
`create_sine_ref_xcorrs.m`) into Python.

**The whole pipeline at a glance** (each step is explained in its own section below):

```
Raw .ncs file (32 kHz)
  │
  ├─ 1. Read the file and fix its timing
  ├─ 2. Mark samples where the amplifier clipped
  ├─ 3. Clean the signal (downsample to 1 kHz, remove 50 Hz mains noise,
  │      remove slow drift, replace spikes of noise, remove >100 Hz)
  ├─ 4. Cut into 1-second epochs
  ├─ 5. Throw away bad epochs (broken records, clipping,
  │      slow-wave dominated, abnormally large)
  ├─ 6. Label each epoch "moving" or "immobile" from the tracking data
  │
  ├─ 7. Compute each epoch's autocorrelogram
  ├─ 8. Find the sine wave whose autocorrelogram looks most like it
  │      → frequency estimate of the epoch
  ├─ 9. Theta epoch if that frequency is 3–7 Hz, otherwise non-theta
  ├─ 10. Measure how regular the rhythm is (normalised peak range)
  │
  ├─ 11. Summary numbers (% non-theta epochs, averages)
  └─ 12. Statistics (arena types, session types, moving vs immobile)
```

### A few terms used throughout

| Term | Meaning |
|---|---|
| LFP | Local field potential: the slow electrical signal recorded on each channel. |
| Sample | One measured value of the signal. At 1 kHz there are 1,000 samples per second. |
| Epoch | A 1-second piece of the recording (1,000 samples after downsampling). |
| Sinusoid / sine wave | A perfectly regular wave of one frequency. |
| Autocorrelogram (ACG) | A curve that shows how similar a signal is to a time-shifted copy of itself (explained in Section 7). |
| Lag | The amount of time shift used in the autocorrelogram. |
| Zero phase filter | A filter run once forwards and once backwards, so it does not shift the signal in time. |
| Median | The middle value when numbers are sorted. Unlike the mean, it is not pulled around by a few extreme values. |
| MAD | Median absolute deviation: the median of how far each value is from the median. A "robust" version of the standard deviation. |

---

## 1. Reading the raw data and fixing its timing

**What the raw file looks like.** Neuralynx stores the LFP (sampled at 32,000 samples per
second, i.e. 32 kHz) in "records". Each record has room for 512 samples, a timestamp
(the time of its first sample, in microseconds) and a number saying how many of the 512
slots actually contain data ("NumValidSamples").

**The problem.** Normally all 512 slots are filled. But during short hiccups in
acquisition, the system writes "short records" where only a few slots (e.g. 1–10) contain
real data and the rest contain leftover junk. If you simply join all 512 slots of every
record end to end, the junk gets inserted into the signal and everything after it is
shifted in time, so the LFP no longer lines up with the tracking data.

**What we do.**

- From each record we keep only the valid samples.
- Each valid sample is put at its correct position in time, using the record's own
  timestamp. In words: "position of the sample = how much time has passed since the start
  of the recording, converted to a number of 32 kHz samples, plus its position inside the
  record".

  ```
  position = round( (record timestamp − first timestamp) × 32,000 / 1,000,000 )  +  j
  ```

  where `j` is 0 for the first valid sample of the record, 1 for the second, and so on.
  (Timestamps are in microseconds, which is why we divide by 1,000,000.)
- Some positions on this regular time grid end up empty (no sample was recorded there).
  These are filled in by drawing a straight line between the recorded samples on either
  side (linear interpolation).
- If a gap is longer than 1 second, the file is not analysed (a gap that long cannot be
  sensibly filled in).
- The raw values are in ADC "counts" (the integer the digitiser produces). They are
  turned into microvolts using the conversion factor stored in each file's header
  ("ADBitVolts", volts per count):

  ```
  value in µV = count × ADBitVolts × 1,000,000
  ```

  This factor is read from every file separately, because it differs between animals
  (it depends on the amplifier's input range).
- Later (Section 5), any 1-s epoch that contains even one short record is thrown away,
  to be safe.

---

## 2. Finding amplifier clipping (saturation)

**The problem.** The amplifier can only record voltages within a fixed range, written in
the file header as "InputRange" (1,000 µV in some recordings, 1,500 µV in others). If the
real signal goes beyond this range, the recording gets stuck at the limit and the wave
appears with a flat, cut-off top or bottom. Such pieces of signal are distorted and should
not be analysed.

**What we do.**

- This is checked on the raw 32 kHz signal, *before* any filtering (filtering would smooth
  the flat tops and make them harder to see).
- A sample counts as clipped if its size (ignoring its sign) is at the limit, allowing a
  tiny tolerance of 1.5 ADC steps (one step = the smallest voltage change the digitiser
  can record):

  ```
  sample is clipped if   |value| ≥ InputRange − 1.5 × (one ADC step)
  ```

- Every stretch of clipped samples is then widened by 0.2 s on each side, because
  the later filtering spreads the effect of the sharp clip edges into the neighbouring
  signal.
- Any 1-s epoch that touches one of these widened stretches is thrown away (Section 5).

---

## 3. Cleaning the signal

These steps are applied to the whole recording at once, in this order.

1. **Downsampling to 1 kHz.** 32,000 samples per second is far more than needed for
   rhythms below 20 Hz. The signal is reduced to 1,000 samples per second. Before
   dropping samples, the method removes frequencies above 500 Hz, which would
   otherwise fold back and appear as false low frequencies ("aliasing").

2. **Removing mains noise.** The electricity supply (50 Hz in Europe) adds a hum at 50 Hz
   and its multiples. Very narrow "notch" filters remove 50, 100, 150 and 200 Hz.
   Each notch removes only a narrow band around its frequency (about 1.7 Hz wide at
   50 Hz; set by the "quality factor" Q = 30). They are zero phase (see table above).

3. **Wavelet denoising: currently switched off** (`APPLY_WAVELET_DENOISE = False`).
   This step would only change frequencies above about 31 Hz, so it has no effect on the
   1–20 Hz range this analysis uses. It was tested on 4 sessions and kept 100% of the
   1–12 Hz power, so it was switched off. (If it were switched on: the signal is split
   into frequency layers with the "sym4" wavelet, the small, noise-like parts of the
   high-frequency layers are shrunk towards zero using the BayesShrink rule of Chang et
   al. 2000, and the signal is put back together.)

4. **Removing slow drift (detrending).** If the whole recording slowly creeps up or down,
   a straight line is fitted through the whole trace and subtracted.

5. **Replacing noise spikes (outlier replacement).** For each sample, we look at the
   0.4 s of signal around it (200 samples before, 199 after; shorter at the very start and
   end of the recording) and work out the median and the MAD of that window. A sample is
   an outlier if it is unusually far from that local median:

   ```
   outlier if   |sample − local median| > 3 × 1.4826 × local MAD
   ```

   (Multiplying the MAD by 1.4826 makes it comparable to a standard deviation, so this
   means roughly "more than 3 standard deviations away from its neighbours".)
   Outliers are replaced by a straight line between the nearest normal samples.
   This copies MATLAB's `filloutliers(x, 'linear', 'movmedian', ...)`.

6. **Removing fast activity (low-pass filter).** Everything above 100 Hz is removed with a
   4th-order Butterworth low-pass filter, applied zero phase. (A Butterworth filter is a
   standard filter that is flat in the range it keeps; "4th order" describes how steeply
   it cuts off above 100 Hz.)

The result is called the **cleaned trace**. It is used for the epoch rejection steps in
Section 5.

---

## 4. Cutting the recording into 1-second epochs

The cleaned trace is cut into consecutive, non-overlapping 1-s pieces of 1,000 samples:
epoch 0 is samples 0–999, epoch 1 is samples 1,000–1,999, and so on. Any leftover piece
shorter than 1 s at the end is dropped.

Each epoch's start and end time (in the same clock as the tracking camera) is taken from
the corrected 32 kHz timing of Section 1. This matters for Section 6, where we look up how
fast the animal was moving during each epoch.

---

## 5. Throwing away bad epochs

Four checks are applied, in this order. An epoch must pass all four.

1. **Broken records.** The epoch is rejected if it contains any short record
   (Section 1).

2. **Clipping.** The epoch is rejected if it overlaps a clipped stretch, including the
   0.2 s margin (Section 2).

3. **Slow-wave dominated (delta vs theta power).** We compute the power spectrum of the
   epoch (how much signal there is at each frequency). This uses a single 1-s window with a
   Hann taper (a bell-shaped weighting that fades the window's edges to zero, to avoid
   edge effects), after subtracting the epoch's mean. With 1-s windows, the spectrum has one
   value per whole Hz (1 Hz, 2 Hz, 3 Hz ...). Then:

   ```
   delta power = sum of power at 1, 2 and 3 Hz
   theta power = sum of power at 3, 4, 5, 6 and 7 Hz
   reject the epoch if   delta power > theta power
   ```

   (The 3 Hz value is included in both sums.) The reason: noise often shows up as large,
   slow waves. These are not always large enough to be caught by the amplitude check
   below, and would otherwise be mistaken for slow theta.

   **Note for interpretation:** because of this check, epochs dominated by 1–3 Hz activity
   never reach the theta / non-theta decision in Section 9. A "non-theta" epoch in this
   analysis is therefore an epoch that was *not* dominated by delta power, but whose main
   rhythm still fell outside 3–7 Hz.

4. **Abnormally large epochs (amplitude).** For each epoch we take its peak-to-peak
   amplitude (highest value minus lowest value). We then ask how unusual each epoch's
   amplitude is compared with the other epochs of the same recording, using a robust
   z-score:

   ```
   robust z = 0.6745 × (amplitude − median amplitude) / MAD of amplitudes
   reject the epoch if   robust z > 5
   ```

   The median and MAD are calculated only from the epochs of that recording that
   passed checks 1–3, so epochs that were already thrown out cannot distort the
   threshold. (Multiplying by 0.6745 makes this z-score comparable to an ordinary
   z-score, so "z > 5" means "much larger than the typical epoch".)

---

## 6. Was the animal moving or still?

Every session folder contains a tracking file with the animal's position (x and y, in cm)
for each camera frame, with timestamps on the same clock as the LFP.

1. **Removing tracking errors.** Sometimes the tracker briefly jumps to a wrong position.
   For each pair of neighbouring frames, we calculate the speed the animal would have
   needed to get from one to the other (distance divided by time). If that speed is above
   80 cm/s, the later frame is marked as a tracking error. This check is then repeated on
   the remaining good frames until no new errors are found, so that errors lasting
   several frames in a row are also caught. Bad frames are replaced by a straight line
   (in time) between the nearest good frames.

2. **Smoothing.** The x and y traces are smoothed slightly with a Gaussian (bell-shaped)
   smoothing window with a width (sigma) of 1 frame.

3. **Speed.** The speed at each frame is the distance moved to the next frame divided by
   the time between the two frames:

   ```
   speed = √( (x_next − x)² + (y_next − y)² ) / (t_next − t)        [cm/s]
   ```

4. **Labelling each epoch.** Each epoch gets the median speed of all the camera frames
   that fall inside it (or the speed of the nearest frame if none fall inside it).

   ```
   median speed > 4 cm/s   →  "moving"
   median speed ≤ 4 cm/s   →  "immobile"
   ```

   Epochs whose speed cannot be worked out are dropped.

---

## 7. The autocorrelogram (ACG)

### What an autocorrelogram is

Take the 1-s epoch and a copy of it. Slide the copy along in time by some amount (the
"lag") and measure how well the two line up (multiply them point by point and add the
results up). Repeat for every possible shift. The resulting curve is the autocorrelogram.

- At zero shift, the signal lines up perfectly with itself. The curve is scaled so that
  this value is exactly 1.
- If the signal is a regular rhythm, shifting it by one full cycle lines it up with itself
  again, so the curve has another **peak** at a lag of one cycle. Shifting it by half a
  cycle puts peaks against troughs, so the curve has a **trough** (negative values) there.
- If the signal has no rhythm, the curve quickly drops to around zero and stays flat.

Example for a clean 6 Hz rhythm (one cycle = 1/6 s ≈ 167 ms):

```
 ACG value
   1 ┤█  ← zero lag (always 1)
     │ █
     │  █                 ┌── first side peak (lag ≈ 167 ms = one cycle)
     │   █              ▄█▄
   0 ┼────█──────────▄█────█▄──────────▄▄──── lag
     │     █        █        █       ▄█
     │      █▄   ▄█           █▄  ▄█
  −1 ┤        ▀█▀               ▀▀
             ↑                    ↑
       first trough          second trough
       (≈ 83 ms)              (≈ 250 ms)
```

Because the epoch is only 1 s long, the further you slide the copy, the less of the
two overlaps. So even for a perfect sine wave, the peaks get smaller as the lag grows.
This matters in Section 10.

### What we do

- The autocorrelogram is calculated from a **separate copy** of the cleaned trace that
  has been band-pass filtered to keep only 1–20 Hz (Butterworth filter, zero phase).
  This copy is cut into exactly the same 1-s epochs. The rejection checks of Section 5
  still use the cleaned trace without this band-pass. Only epochs that passed every
  check in Sections 5 and 6 are analysed.
- Each epoch's mean value is subtracted first. Why: if the epoch sits slightly above or
  below zero overall, that offset adds a broad hump to the autocorrelogram, which makes
  the epoch look like it contains a slow rhythm.
- The autocorrelogram is computed for every possible lag from −999 ms to +999 ms (1,999
  values) and divided by its value at zero lag, so it starts at 1.

  ```
  R(lag)   = sum over n of  x(n) × x(n + lag)
  ACG(lag) = R(lag) / R(0)
  ```

  where `x(n)` is the n-th sample of the epoch (after subtracting the mean).

---

## 8. Estimating the frequency of each epoch (matching to reference sine waves)

1. **A library of reference sine waves.** We create 191 perfect sine waves, one for each
   frequency from 1.0 to 20.0 Hz in steps of 0.1 Hz. Each is 1 s long at 1 kHz, the same
   length as an epoch. We compute the autocorrelogram of each, in the same way as for the
   data.

2. **Comparing the epoch with every reference.** For each reference, we measure how
   different its autocorrelogram is from the epoch's autocorrelogram: at every lag, take
   the difference between the two curves, square it, add these up over all lags, and take
   the square root. This is the ordinary (Euclidean) distance between the two curves.
   It is then divided by the size of the reference curve. "Size" means the same
   square-and-sum calculation done on the reference curve alone, i.e. its distance from a
   curve that is zero at every lag. Dunn et al. (2022, Supplementary Fig. 5d) call this
   the reference autocorrelogram's "Euclidean distance from the origin":

   ```
   D(f) = √( sum over lags of (ACG_epoch − ACG_ref,f)² )
          ─────────────────────────────────────────────
          √( sum over lags of (ACG_ref,f)² )
   ```

   A small D(f) means the epoch's autocorrelogram looks very much like that of a sine
   wave at frequency f.

   **What the division does.** It turns the distance into a relative distance: how large
   the mismatch is compared with the reference curve itself. This makes the value easy to
   read. The scale below holds for D(f) at any reference frequency f, and therefore also
   for ED_min (step 3), which is simply the smallest of the epoch's D(f) values:

   - D = 0: the epoch's autocorrelogram is identical to the reference.
   - D ≈ 1: the epoch has no rhythm. Its autocorrelogram is flat (1 at zero lag, about 0
     everywhere else), so it fits the reference about as well as a curve that is zero
     everywhere.
   - D ≈ 1.4: the epoch has a strong rhythm at a clearly different frequency (for example
     a pure 10 or 15 Hz rhythm compared with a 6 Hz reference).

   These values follow from expanding the squared distance (ordinary algebra, no extra
   assumptions). Writing "overlap" for the sum over lags of ACG_epoch × ACG_ref,f:

   ```
   D(f)² = 1 + ( (size of ACG_epoch)² − 2 × overlap ) / (size of ACG_ref,f)²
   ```

   - Same rhythm: the overlap is about equal to both squared sizes, so D ≈ 0.
   - No rhythm: the epoch's ACG is 1 only at zero lag, so its squared size and the overlap
     are both about 1, tiny next to the reference's squared size (about 335). So D ≈ 1.
   - Different strong rhythm: the two curves rise and fall out of step, so the overlap is
     about 0, while the epoch's squared size is about as large as the reference's. So
     D² ≈ 1 + 1 = 2 and D ≈ √2 ≈ 1.41.

   D is above 1 exactly when the overlap is less than half the epoch's squared ACG size.
   D > 1 on its own does not decide whether an epoch is theta. That is decided only by
   which reference frequency gives the smallest D (step 3 and Section 9).

   This step follows the original code of Dunn et al. (`create_sine_ref_xcorrs.m` computes
   the size of each reference curve, and `quantify_xcorr_epochs.m` divides by it).
   It hardly changes which frequency wins. All reference curves start at 1 at zero lag
   and have almost the same size: about 18.3 for every reference from 3 to 20 Hz. The
   size grows only for the slowest references (18.7 at 2 Hz, 19.9 at 1 Hz), because so
   few cycles fit in 1 s. Its main effect is to put ED_min on the readable 0-to-1 scale
   described above.

3. **The best match.** The reference frequency with the smallest distance is the epoch's
   **frequency estimate**. That smallest distance is called **ED_min**. It tells you how
   good the best match was (smaller = more sine-like).

---

## 9. Theta or non-theta?

```
frequency estimate between 3.0 and 7.0 Hz (inclusive)  →  theta epoch
anything else (1.0–2.9 Hz or 7.1–20.0 Hz)              →  non-theta epoch
```

- No cut-off is applied to ED_min. A theta epoch is decided by its frequency alone.
- Why the library covers 1–20 Hz and not just 3–7 Hz: if the library only contained
  3–7 Hz sine waves, an epoch with, say, a 10 Hz rhythm would be forced to match the
  closest available reference (7 Hz) and be wrongly called theta. With a wide library, it
  matches 10 Hz and is correctly called non-theta.

---

## 10. How regular is the rhythm? (peak range)

The idea: in a clean rhythm, the autocorrelogram has a tall first side peak and deep
troughs on either side of it. The bigger the difference between the peak and the
troughs, the more regular the rhythm.

### Step A: where to look for the peak and troughs

We use the matched reference sine wave (from Section 8) as a guide to where the peak and
troughs of the epoch's autocorrelogram should be. On the reference's autocorrelogram we
find:

- its first peak after zero lag, and the trough just before and just after it
  (only peaks and troughs that stand out by at least 0.05 are counted), and
- the points where the reference curve crosses zero. (Where the curve changes sign
  between two lags, the exact crossing point is found by drawing a straight line between
  those two values.)

This gives three search windows, each lying between two zero crossings:

| Window | What it covers | Example for 6 Hz |
|---|---|---|
| Peak window | the positive hump containing the first side peak | about 125–208 ms |
| Trough 1 window | the negative dip before that peak | about 42–125 ms |
| Trough 2 window | the negative dip after that peak | about 208–292 ms |

### Step B: measure the epoch

In the **epoch's own** autocorrelogram we take:

```
P  = highest value inside the peak window
T1 = lowest value inside the trough 1 window
T2 = lowest value inside the trough 2 window

peak range  PR = P − (T1 + T2) / 2
```

In words: the height of the peak above the average of the two troughs.

### Step C: normalise by the reference

The same measurement is made on the matched reference sine wave itself (PR_ref), and the
epoch's value is divided by it:

```
normalised peak range = PR / PR_ref
```

**Why this division is needed.** As explained in Section 7, because the epoch is only 1 s
long, even a perfect sine wave's autocorrelogram peaks get smaller the further they are
from zero lag. A slow rhythm has its first peak far out (e.g. at 333 ms for 3 Hz), so it
is shrunk more than a fast rhythm's (e.g. 143 ms for 7 Hz). For a perfect sine wave the
peak range is roughly:

```
PR_ref ≈ 2 × (1 − 1 / frequency)          (frequency in Hz, epoch length 1 s)

3 Hz → about 1.33        7 Hz → about 1.71
```

So without the division, slower rhythms would always look less regular, even if they were
perfect. After dividing, a perfect sine wave of any frequency scores about 1, and an epoch
without a regular rhythm (flat autocorrelogram) scores about 0.

### Very slow matches

The second trough window ends at a lag of about 7 / (4 × frequency) seconds. This only
fits inside a 1-s epoch when the frequency is above 1.75 Hz. Epochs matched to a
reference at or below 1.7 Hz therefore get no peak range. These are all non-theta epochs
anyway, so they are still counted in the theta / non-theta percentages.

---

## 11. Summary numbers

For any group of epochs (e.g. one animal, one arena, one session type):

| Measure | Calculation | Meaning |
|---|---|---|
| % non-theta, all epochs | non-theta epochs / all epochs × 100 | how often theta was absent overall |
| % non-theta, immobile | non-theta immobile epochs / all immobile epochs × 100 | how often theta was absent while still |
| % non-theta, moving | non-theta moving epochs / all moving epochs × 100 | how often theta was absent while moving |
| % of non-theta epochs that were immobile | non-theta immobile epochs / all non-theta epochs × 100 | of all the epochs without theta, what share happened while the animal was still |

The immobile and moving percentages are each divided by their *own* total, so they can be
compared even when an animal spent much more time moving than still (or the other way
round).

Other summaries:

- **Centroids:** for the theta epochs of each movement state, the average frequency
  estimate and the average normalised peak range. These are the diamond markers on the
  scatter plots.
- **Median and interquartile range** (the middle 50% of values) of the normalised peak
  range, per animal, channel and movement state.
- **ED_min vs normalised peak range:** across all epochs that have both values (theta and
  non-theta), we test whether a poorer sine-wave match (higher ED_min) goes with a less
  regular rhythm (lower peak range), using Pearson's correlation (straight-line
  relationship) and Spearman's correlation (the same, but on ranks, so it only asks
  whether one goes up when the other goes down).

---

## 12. Statistics

We compare how often epochs are non-theta (the four % measures of Section 11):

- across **arena types**: circular track, linear track, open arena;
- across **session types within each arena**: open arena: control, rotated, zero;
  linear track: 0°, 90°, 180°, 270°; circular track: standard, no rotation, rotation.

Each comparison is tested in two ways.

### 12a. Epoch level (every epoch counted separately)

For each measure, we make a table of counts: one row per group (e.g. one per arena), and
two columns: number of non-theta epochs and number of theta epochs.

- **All groups together: chi-square test.** It asks: "Is the proportion of non-theta
  epochs the same in every group?" It compares the observed counts with the counts you
  would expect if all groups had the same proportion:

  ```
  expected count of a cell = (row total × column total) / grand total
  chi-square = sum over all cells of (observed − expected)² / expected
  ```

  A large chi-square (small p-value) means the groups differ.

- **Pairs of groups.** Each pair (e.g. linear vs open) is tested on its own 2 × 2 table
  with the chi-square test with Yates' correction (a small adjustment that makes the
  test less likely to give false positives when tables are 2 × 2). If any expected count
  is below 5, the chi-square approximation is unreliable, so Fisher's exact test is used
  instead. Fisher's test calculates the probability exactly. Its effect size is the odds
  ratio.

- **Moving vs immobile within each group.** Within each arena or session type, we ask
  whether the non-theta proportion differs between immobile and moving epochs, using the
  same 2 × 2 test (rows: immobile, moving; columns: non-theta, theta).

### 12b. Session level (one number per recording session)

**Why this is needed.** Epochs from the same recording are not independent: neighbouring
epochs are similar, and all channels of one session record the same brain state.
Counting every epoch separately can therefore make differences look more significant than
they are. As a more conservative check, each recording session is reduced to a single
percentage (all its channels' epochs pooled):

```
session % = non-theta epochs in the session / all epochs in the session × 100
```

- **All groups together: Kruskal–Wallis test.** All session percentages are ranked
  from smallest to largest, and the test asks whether some groups tend to have higher
  ranks than others. It is the rank-based version of a one-way ANOVA and does not assume
  the data are normally distributed.
- **Pairs of groups: Mann–Whitney U test** (two-sided). The rank-based version of a
  t-test between two groups.

### 12c. Correcting for many comparisons (Holm–Bonferroni)

When several pairs are tested, some will look significant by chance. The pairwise
p-values are corrected with the Holm–Bonferroni method:

1. Sort the m p-values from smallest to largest.
2. Multiply the smallest by m, the second smallest by m − 1, the third by m − 2, and so on.
3. If a corrected value is smaller than the one before it, raise it to match the one
   before it (so the order is kept), and cap all values at 1.

Example with 3 pairs: p = 0.01, 0.02, 0.04 → corrected 0.03, 0.04, 0.04.

The correction is applied separately to each "family" of tests: the pairwise tests of
one measure at one level, or the moving-vs-immobile tests across groups. The
all-groups-together tests (chi-square, Kruskal–Wallis) are not corrected. Significance
level: 0.05.

### 12d. Controlling for animal identity

Not every animal was recorded in every arena. So a difference between arenas could
really be a difference between animals. The arena comparisons are therefore repeated
using only animals that were recorded in both arenas being compared:

- circular track vs open arena: Fa8477 and Fa5834 only;
- linear track vs open arena: Fa1059 and Fa23BD only.

### Software

Python, with NumPy, SciPy, pandas and PyWavelets.

---

## Parameter summary

| Step | Setting | Value |
|---|---|---|
| Recording | Original sampling rate | 32 kHz |
| Timing repair | Longest gap filled in | 1 s (files with longer gaps are not analysed) |
| Clipping | Limit; tolerance; margin | ± InputRange from each file's header; 1.5 ADC steps; 0.2 s each side |
| Downsampling | Analysis sampling rate | 1 kHz |
| Mains noise | Notch frequencies; Q | 50, 100, 150, 200 Hz; 30 |
| Wavelet denoising | On/off | off (only affects > 31 Hz) |
| Detrending | Type | straight line over the whole recording |
| Outlier replacement | Window; threshold | 0.4 s moving median; 3 scaled MADs |
| Low-pass | Cut-off; order | 100 Hz; 4th-order Butterworth, zero phase |
| Epochs | Length | 1 s (1,000 samples), not overlapping |
| Delta vs theta check | Bands | reject if power at 1–3 Hz > power at 3–7 Hz |
| Amplitude check | Robust z cut-off | 5 (peak-to-peak amplitude, within each recording) |
| Tracking | Jump cut-off; smoothing | 80 cm/s; Gaussian, sigma = 1 frame |
| Movement | Speed cut-off | > 4 cm/s moving, ≤ 4 cm/s immobile |
| ACG input | Band-pass; mean removal | 1–20 Hz, Butterworth order 4, zero phase; mean removed per epoch |
| Reference sine waves | Range; step | 1–20 Hz; 0.1 Hz (191 references) |
| Peak / trough detection | Minimum prominence | 0.05 |
| Theta epoch | Frequency estimate | 3–7 Hz (inclusive) |
| Statistics | Significance level; correction | 0.05; Holm–Bonferroni |

---

## References

- Chang, S. G., Yu, B., & Vetterli, M. (2000). Adaptive wavelet thresholding for image
  denoising and compression. *IEEE Transactions on Image Processing*, 9(9), 1532–1546.
- Dunn, S. L. S., Town, S. M., Bizley, J. K., & Bendor, D. (2022). Behaviourally modulated
  hippocampal theta oscillations in the ferret persist during both locomotion and
  immobility. *Nature Communications*, 13. https://doi.org/10.1038/s41467-022-33507-2
- Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics*, 6(2), 65–70.
