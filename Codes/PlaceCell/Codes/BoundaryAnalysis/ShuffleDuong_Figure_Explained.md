# Method 2 shuffle + Duong test: what the analysis does and how to read the figure

Script: `MeanRateMap_QuadrantAnalysis_v27_5_M2_BinShuffle_AdptBin_Session_Rayleigh.py`
Figure: `ShuffleDuong_<kind>.png` (example: Linear_0, field-only mean field index map)

## The question being asked

After pooling, you have one map per session type, e.g. "where on the Linear_0 track do place fields tend to sit?" The analysis asks:

> **Is the pooled map's spatial pattern different from what you'd get if the same values were scattered at random along the track?**

Here is how it answers that.

1. **Real KDE:** smooth the real pooled map and turn it into a probability density, so the whole track sums to 1. This shows where the field mass sits.
2. **1000 shuffled KDEs:** take the same pooled values, put them in random bins, and smooth them the same way. Do this 1000 times. This shows what "no spatial pattern" looks like.
3. **Compare real against shuffles in two separate ways.** These are the two analyses that run after the KDEs:
   - **Analysis A, the Duong test (strict).** For each of the 1000 shuffles, it tests every bin: is the real density significantly higher, lower, or not different from this shuffle? It then counts how often each bin came out significant. **This is what feeds the arm-preference index.**
   - **Analysis B, the pointwise envelope (simple).** For each bin, it ranks the real density among the 1000 shuffled densities. Is the real value above almost all of them, or below almost all of them?

## The two analyses in plain words

**Duong test (A).** Imagine two smoothed maps, real and shuffle #1. At bin 37 the real one is 0.0020 and the shuffle is 0.0016. Is that gap real, or could it be noise?

The Duong test estimates how noisy each smoothed map is. The noise estimate depends on:
- how many data points went into the map (`neff`),
- how wide the smoothing kernel is,
- how high the density is.

From that it gets a z-score per bin: the gap divided by the noise. Because it tests 160 bins at once, it applies the Hochberg correction so that roughly 1 bin in 20 doesn't light up by chance. The result for each bin is **above**, **below** or **not different**. That is repeated against all 1000 shuffles.

**Envelope (B).** Take bin 37 and line up its 1000 shuffled densities from lowest to highest.
- If the real value is higher than the top 2.5% of them, the bin is green.
- If it is lower than the bottom 2.5%, the bin is magenta.

There is no noise formula and no correction for testing 160 bins.

## The six panels, left to right

| # | Panel | What it shows | What it says here (Linear_0, field-only) |
|---|---|---|---|
| 1 | **REAL KDE** | Smoothed real pooled map as a density in cm⁻² (darker = more field mass). Green/magenta outlines would mark Duong-consistent bins. | Mass rises steadily from **left (light) to right (dark)**. Place fields pile up toward the right end. There are no outlines because Duong found nothing. |
| 2 | **One shuffled pooled map** | One example of the 1000 random rearrangements, *before* smoothing (jet colours, raw pooled values). It shows what "no structure" looks like. | Salt-and-pepper noise with no gradient. Values are 0–0.1 because a field-only bin value = sum of field index ÷ number of cells that sampled it. |
| 3 | **Mean SHUFFLED KDE** | Average of the 1000 smoothed shuffles, on the **same grey scale as panel 1**, so the two can be compared directly. | Uniform grey everywhere, about 0.00156 cm⁻² = 1 / (80 × 8 cm²), i.e. perfectly flat. Against this, the real map is about 0.65× flat at the left end and about 1.4× at the right. |
| 4 | **Fraction significant: above − below** | For each bin, (fraction of shuffles where Duong said "real higher") − (fraction where it said "real lower"). Red = consistently higher, blue = consistently lower, white = never significant. | **All white (0).** In none of the 1000 comparisons did Duong call any bin significant. |
| 5 | **Consistent clusters** | Bins significant in ≥ 95% of shuffles, painted green (higher) or magenta (lower) and grouped into clusters. **These are the bins the arm-preference index counts.** | **Empty (0 / 0)**, which is why every arm-preference index is NaN ("no sig. bins"). |
| 6 | **Pointwise envelope** | Analysis B: green = real above the 97.5th percentile of the shuffles, magenta = below the 2.5th. Not corrected for 160 bins. | 27 green bins on the **right**, 33 magenta on the **left**. By chance you'd expect about 4 each side, so this gradient is clearly not random by this criterion. |

**Title terms:**
- "Renormalised on the common domain": both maps are rescaled to sum to 1 over the same bins, so you compare *where* the mass is, not how much.
- "n = neff": the sample size plugged into Duong's noise formula.
- "Blank": bins outside that shared area. Here there are none.

## Why the two analyses disagree

They measure "noise" differently.

- **The envelope only asks how much the shuffles vary among themselves.** Every shuffled KDE is nearly flat and nearly identical, so that spread is tiny. The real map only has to tilt a little to fall outside it. The envelope also treats the real map as exact.
- **Duong also asks how uncertain the real map itself is**, i.e. how much it would change with a different set of cells. On an 8 cm-wide track the smoothing kernel covers a small area, and a kernel estimate over a small area is noisy. Add the 160-bin correction, and a ±40% tilt doesn't clear the bar: the largest |z| here is about 1.6, and about 3.6 is needed.

Because all 1000 shuffled KDEs are nearly identical and flat, the Duong test gives nearly the same answer against every one of them. The 1000 repetitions therefore add little: in practice it is one comparison of the real map against a flat map, repeated.

## What this figure tells you

For Linear_0, field-only place-field mass is **tilted toward the right end of the track and away from the left**.
- **Envelope (loose, descriptive):** the tilt is clearly larger than random rearrangements produce.
- **Duong (strict, accounts for uncertainty in the real map and for 160 tests):** no single bin is significantly different.

So the honest summary is: there is a consistent left-to-right gradient, but it isn't strong enough to localise significantly at the bin level with this test on this track. Which of the two criteria should drive the linear track arm-preference index is still an open decision.
