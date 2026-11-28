# Statistical analysis

**Comparison groups.** In all analyses, we compared arena types restricted to the animals recorded in both arenas being compared (Circle vs Open: two animals; Linear vs Open: two other animals), so that arena effects are not confounded with animal identity. The two comparisons were not tested against each other. We also report results for all data pooled. Some recording folders held identical copies of one LFP signal under several file names; only the first file of each set of copies was analysed.

## Place-cell properties

Except for place-cell yield, which uses all recorded units, these analyses include only units classified as place cells. Each data point is one unit in one session. Field-shape metrics use one value per place field. For all place cells pooled, we report descriptive statistics without tests.

**Continuous metrics.** These were peak, mean and within-field firing rate, spatial information, sparsity, coherence, split-half stability score, speed score, field number, field area, in/out-of-field spike ratio, field shape, temporal and lap stability, and the multi-peak field measures. For each metric we tested for an overall group difference with a Kruskal–Wallis test. Pairs of groups were then compared with two-sided Mann–Whitney U tests. We also compared the stability score separately within each stability class (significant positive, significant negative, non-significant). Data are reported as mean ± SD and median [Q1, Q3].

**Proportions.** These were place-cell yield, theta modulation, coherence shuffle significance, split-half stability significance (overall, and separately for positive and negative scores), speed cells, and multi-peak cells and fields. Each was compared across groups with a chi-square test of independence, followed by pairwise chi-square tests. Speed-cell direction (positive, negative, non-speed) was tested the same way as a 2 × 3 table. Proportions are shown with Wilson 95% confidence intervals.

**Wall distance.** Each place cell was assigned the distance from the nearest wall of the peak bin of its unsmoothed rate map, in 2-cm steps (Open: 0–14; Linear: 0–1). All Circle-track bins are against a wall, so Circle cells have a single distance level and were not tested for distance effects. Within each group, we did three things:

- We compared metrics across distance levels using the tests above.
- We tested for a trend with distance: Spearman's rank correlation for continuous metrics, and logistic regression with a Wald test (odds ratio per bin) for proportions. Trend p-values were Holm-corrected across metrics.
- At each distance level shared by two or more groups, we compared the groups using the tests above.

Across groups, we tested the distance effect with group as a covariate using a linear model on ranks (likelihood-ratio test), or logistic regression for proportions. We tested whether slopes differed between groups with a distance × group interaction term.

**Control analyses.** Firing rates in rarely visited bins can be unreliable. We therefore repeated the peak-firing-rate comparison on cells whose peak bin had low dwell time, defined as at or below a percentile threshold within each group. We repeated the field width-ratio comparison separately on fields of cells whose peak bin was next to the wall and on all other fields.

## Theta continuity (LFP autocorrelogram)

LFP epochs recorded during immobility or mobility were classified as theta when the frequency of their best-matching reference sinusoid (from a 1–20 Hz bank) lay within 3–7 Hz, and as non-theta otherwise.

**Measures.** We measured:
- the percentage of epochs that were non-theta, over all epochs and separately within immobility and within mobility;
- the share of non-theta epochs that occurred during immobility;
- the percentage of theta epochs within each movement state, which is 100% minus the non-theta percentage;
- the centroids of the accepted theta epochs: the matched-sinusoid frequency and the normalised autocorrelogram peak range, each averaged per recording and movement state.

**Across groups.** We compared epoch proportions with a chi-square test of independence on pooled epoch counts, followed by pairwise chi-square tests. We compared per-recording percentages and centroids with a Kruskal–Wallis test followed by pairwise Mann–Whitney U tests, separately for each movement state.

**Immobility vs mobility.** Within each group, and for all recordings pooled, we compared immobility and mobility with a paired Wilcoxon signed-rank test, using one pair per recording (its two percentages, or its two centroids). These p-values were Holm-corrected across groups.

## Theta frequency and power vs running speed

Instantaneous theta frequency and power were estimated in 0.25-s bins and matched to running speed. Bins with speed below 4 cm/s (immobility) were excluded from all fits. Power was analysed in three forms: raw (µV²), in dB relative to each recording's power at 4–7 cm/s, and as a percentage of the recording's total 1–100 Hz power.

**Pooled relationship.** For each recording, we took the median frequency and power in each 4-cm/s speed bin. We fitted a linear mixed model to these medians (value ~ speed + a random intercept per recording, REML), with speed set to the bin centre. The slope was tested with a Wald z-test.

**Per-recording slopes.** Following Dunn et al. (2022), we discarded speeds above each recording's 90th percentile and fitted an ordinary least-squares regression to that recording's speed-bin medians (≥3 bins). Channels recorded together share one speed trace, so their slopes were averaged within each session. These slopes were tested against zero with a one-sample t-test and a Wilcoxon signed-rank test.

**Individual-recording modulation.** For theta frequency and relative power, we fitted a separate regression to each LFP file, without the percentile cut-off, keeping only speed bins with at least 5 s of data. Slope p-values were corrected with the Benjamini–Hochberg false discovery rate (FDR) across all files, separately for each measure. Each file was then classed as significantly positive, significantly negative or non-significant (q < 0.05), or as not fitted. We also counted the files that were significant for both measures, grouped by the signs of their two slopes.

**Group comparisons.** We fitted the mixed model separately within each group. To test whether slopes differed between groups, we compared a model with a speed × group interaction against a common-slope model (maximum likelihood, likelihood-ratio test), followed by pairwise Wald contrasts of the group slopes. Per-session slopes were compared between groups with a Kruskal–Wallis test followed by pairwise Mann–Whitney U tests, and tested against zero within each group. We compared the proportions of speed-modulated files (significant in either direction, positive, negative, or significant for both measures) with a chi-square test of independence and pairwise chi-square tests. Each file kept the class it was given by the FDR correction across all files, and the denominator was all analysed files in the group.

## Multiple comparisons, significance and software

Pairwise p-values were Holm–Bonferroni corrected within each measure. For wall distance this was within each group or distance level, and for LFP analyses within each movement state or measure. Yates' continuity correction was applied to all 2 × 2 chi-square tests, and tables with expected counts below 5 were flagged. Significance was set at α = 0.05, two-sided. Analyses were run in Python using SciPy and statsmodels.

**Limitations.** Several tests treated their values as independent even though they are not:
- A unit recorded in several sessions of the same day contributes one value per session.
- LFP epochs, recordings and simultaneously recorded channels from the same session were each treated as separate observations.
- Sessions from the same animal were treated as independent.

The mixed models account only for repeated speed bins within a recording.
